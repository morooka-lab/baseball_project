"""
bbox_label_web.py で作成した点(point)ラベルから、torchvision の
KeypointRCNN に渡せる Dataset を構築する。

utils/detection_dataset.py (SSDlite用、点を独立した合成bboxとして扱う)とは
別の並行パイプライン。既存のSSDlite学習には一切影響しない。

1クラス=複数点(例: home_plateは5頂点)という構造を、KeypointRCNNの
「1インスタンス=順序付きキーポイント集合」という表現にマッピングする。
アノテーションツール自体には点の順序/識別情報が無い(クリック順のまま
保存されるだけ)ため、読み込み時に _canonical_order() で順序を正規化する。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from utils.detection_dataset import collate_fn  # 純粋なtuple(zip(*batch))、タスク非依存なので再利用
from utils.label_store import TaskConfig, frame_image_path, iter_label_files, split_by_video
from utils.video_io import open_video, read_frame_bgr

_REF_ANGLE = 0.0  # 画像の右方向(+x)を基準にする。


def _canonical_order_by_angle(points: Sequence[Tuple[float, float]]) -> List[int]:
    """centroidからの角度で正規化する(pitcher_plate=2点, home_plate=5点向け)。

    単純にcentroidからの角度でソートする案は、atan2の分岐(±π)付近に点が
    来る配置(例: pitcher_plateの2点はcentroidに対しほぼ真反対にあり、
    常にどちらかが分岐の近くに来る)でサブピクセル誤差だけで順序が反転する
    (シミュレーションで約49%の確率で反転することを確認済み)。
    「一番広い角度の隙間を境界にする」改良も、点配置がほぼ正多角形の場合に
    同様に破綻する(検証: 約74%反転)。

    採用した方式: 固定の基準方向(画像の+x方向)に最も近い点を開始点に
    固定し、そこから反時計回りに並べる。基準方向を典型的な配置
    (pitcher_plateはほぼ水平)に対して非対称に選んでいるため安定する。
    pitcher_plate(2点)・home_plate(5点、五角形で非対称な尖った頂点を持つ)
    では実データ(41ファイル)で全て安定した順序になることを確認済み。
    """
    n = len(points)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    angles = [math.atan2(p[1] - cy, p[0] - cx) for p in points]

    def ang_dist(a: float) -> float:
        d = (a - _REF_ANGLE) % (2 * math.pi)
        return min(d, 2 * math.pi - d)

    start = min(range(n), key=lambda i: (round(ang_dist(angles[i]), 9), i))
    return sorted(
        range(n),
        key=lambda i: (round((angles[i] - angles[start]) % (2 * math.pi), 9), i),
    )


def _canonical_order_quad(points: Sequence[Tuple[float, float]]) -> List[int]:
    """x座標で左右2点ずつに分け、各グループ内をy座標でソートする
    (batter_box_right/left=4点、正方形に近い四角形向け)。

    _canonical_order_by_angle は batter_box の4隅(ほぼ正方形)に適用すると
    不安定になることを実データで確認した: 正方形に近い配置では、基準角度に
    最も近い点が動画(カメラアングル)によって隣り合う2つの角の間で入れ替わり、
    41ファイル中で安定しなかった。

    x座標(box幅方向の変位)は正方形の対角線的な曖昧さの影響を受けにくく、
    実データで41ファイル全てにおいて「左上・左下・右上・右下」の順で完全に
    安定することを確認済み。また、この方式はbatter_box_right/leftの正規化を
    ミラー対称にする副次効果もある(左右反転してラベルを入れ替える
    augmentation使用時に、両クラスのスロット意味が一致するために重要)。
    """
    idx = sorted(range(len(points)), key=lambda i: points[i][0])
    half = len(points) // 2
    left_group = sorted(idx[:half], key=lambda i: points[i][1])
    right_group = sorted(idx[half:], key=lambda i: points[i][1])
    return left_group + right_group


def _canonical_order(points: Sequence[Tuple[float, float]]) -> List[int]:
    """points: 1クラス・1(代表)フレームの点群(cx/cyは正規化・ピクセルどちらでも
    良い、比率のみ使うのでスケール非依存)。
    戻り値: 正規化した順序に並べ替えるための元インデックス列。

    点数によって方式を切り替える(4点=正方形に近い四角形はangle方式が
    不安定なため専用ロジックを使う。詳細は各関数のdocstring参照)。
    完全に対称な配置は理論上どんな幾何学的ルールでも曖昧さが残るため、
    このモジュールを使う前に少数の代表フレームを目視確認すること
    (train_keypoint.py使用前の検証手順を参照)。
    """
    if len(points) == 4:
        return _canonical_order_quad(points)
    return _canonical_order_by_angle(points)


class KeypointSample:
    __slots__ = ("video", "frame", "instances", "image_path")

    def __init__(self, video: str, frame: int, instances: list, image_path: Optional[Path] = None):
        self.video = video
        self.frame = frame
        # instances: [{"class_id": int(0-indexed, global順), "points": [(cx,cy), ...]}]
        # points は既に正規順序(_canonical_order)・検証済み(0点かpoint_counts[class]点)
        self.instances = instances
        self.image_path = image_path


def _load_keypoint_samples(task: TaskConfig) -> List[KeypointSample]:
    """ラベルJSON群を読み込み、KeypointSample一覧を返す。

    annotation_type=="point"のタスク専用。ファイルは書き換えず、メモリ上でのみ
    点順序を正規化する。期待点数(task.point_counts)と一致しないクラス+フレーム
    の組み合わせは警告を出してスキップする(既知の不正ラベルファイル対策)。
    """
    samples = []
    name_to_global = {name: i for i, name in enumerate(task.classes)}
    point_counts = {name: task.point_counts.get(name, 1) for name in task.classes}

    for path in iter_label_files(task.labels_dir):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        video = data["video"]
        file_classes = data.get("classes", task.classes)
        frames_raw = data.get("frames", {})

        # クラスごと・フレームごとに元の並び順のまま(cx,cy)を集める
        per_class_frame_points: Dict[str, Dict[int, List[Tuple[float, float]]]] = {
            name: {} for name in task.classes
        }
        for frame_str, boxes in frames_raw.items():
            frame_num = int(frame_str)
            grouped: Dict[str, List[Tuple[float, float]]] = {}
            for b in boxes:
                cid = b["class_id"]
                name = file_classes[cid] if cid < len(file_classes) else None
                if name is None or name not in name_to_global:
                    continue
                grouped.setdefault(name, []).append((b["cx"], b["cy"]))
            for name, pts in grouped.items():
                per_class_frame_points[name][frame_num] = pts

        # (file, class)ごとに正規順序を1回だけ計算する
        # (期待点数と一致する最初のフレームを代表フレームとして使う)
        canonical_perm: Dict[str, Optional[List[int]]] = {}
        for name, expected_n in point_counts.items():
            frames_for_class = per_class_frame_points.get(name, {})
            ref_frame = next(
                (fn for fn in sorted(frames_for_class) if len(frames_for_class[fn]) == expected_n),
                None,
            )
            if ref_frame is None:
                if frames_for_class:
                    print(f"[WARN] {path.name}: class={name} は期待点数({expected_n})に"
                          f"一致するフレームが無いため、このファイルでは全フレームで"
                          f"このクラスをスキップします(既知の不正ラベルファイル対策)。")
                canonical_perm[name] = None
                continue
            canonical_perm[name] = _canonical_order(frames_for_class[ref_frame])

        for frame_str in frames_raw:
            frame_num = int(frame_str)
            instances = []
            for name, expected_n in point_counts.items():
                pts = per_class_frame_points.get(name, {}).get(frame_num)
                if not pts:
                    continue  # このフレームはそのクラス未アノテーション(0点、正常)
                if len(pts) != expected_n:
                    print(f"[WARN] {path.name} frame={frame_num}: class={name} の点数が"
                          f"{len(pts)}(期待値{expected_n})のためスキップします。")
                    continue
                perm = canonical_perm.get(name)
                if perm is None:
                    continue
                ordered = [pts[i] for i in perm]
                instances.append({"class_id": name_to_global[name], "points": ordered})
            image_path = frame_image_path(video, task.videos_dir, task.frames_dir, frame_num)
            samples.append(KeypointSample(video, frame_num, instances, image_path))
    return samples


def build_splits(task: TaskConfig, val_ratio: float = 0.15, seed: int = 42):
    """動画単位でtrain/valに分割する(detection_dataset.build_splitsと同じ規約:
    同じseed/val_ratioなら同じ動画集合がvalになる)。"""
    samples = _load_keypoint_samples(task)
    return split_by_video(samples, get_video=lambda s: s.video, val_ratio=val_ratio, seed=seed)


class RandomHorizontalFlipWithKeypoints:
    """boxes + keypointsの両方を手動で反転する。

    plain tensorのkeypointsをv2.RandomIoUCrop/RandomZoomOutに通すと、boxesは
    正しくクロップされるがkeypoints座標は追従せず古い座標のまま残ることを
    実機検証済み(サイレントに壊れる)。そのため本パイプラインでは幾何変換系
    augmentationはこのFlipのみに限定し、v2.Composeの自動dispatchには頼らない。

    batter_box_right/batter_box_leftのような左右非対称なクラスについては、
    detection_dataset.RandomHorizontalFlipWithClassSwapと同様にラベルも
    入れ替える。
    """

    def __init__(self, p: float = 0.5, swap_pairs: Optional[dict] = None):
        self.p = p
        self.swap_pairs = swap_pairs or {}

    def __call__(self, img, target):
        if torch.rand(1).item() >= self.p:
            return img, target
        w = img.shape[-1]
        img = F.horizontal_flip(img)
        target = dict(target)
        target["boxes"] = F.horizontal_flip(target["boxes"])
        kp = target["keypoints"].clone()
        kp[..., 0] = (w - 1) - kp[..., 0]
        target["keypoints"] = kp
        labels = target["labels"]
        if self.swap_pairs and labels.numel() > 0:
            new_labels = labels.clone()
            for a, b in self.swap_pairs.items():
                new_labels[labels == a] = b
                new_labels[labels == b] = a
            target["labels"] = new_labels
        return img, target


def get_train_transforms(classes: Optional[List[str]] = None):
    """classes: task.classes(classes.yaml順)。batter_box_right/leftが両方
    含まれる場合のみ、水平反転時にラベルを入れ替える。

    幾何変換はflipのみ(RandomIoUCrop/RandomZoomOutはkeypoint座標が追従しない
    ため今回は見送り。utils/detection_dataset.pyのSSDlineパイプラインより
    augmentationの種類は少ない)。photometric distortは画像のみに作用するため
    そのままv2から使う。
    """
    swap_pairs = {}
    if classes and "batter_box_right" in classes and "batter_box_left" in classes:
        r = classes.index("batter_box_right") + 1  # 背景分+1
        l = classes.index("batter_box_left") + 1
        swap_pairs = {r: l}

    photometric = v2.RandomPhotometricDistort(p=0.5)
    flip = RandomHorizontalFlipWithKeypoints(p=0.5, swap_pairs=swap_pairs)
    to_dtype = v2.ToDtype(torch.float32, scale=True)

    def _transform(img, target):
        img = photometric(img)  # 画像のみに適用、targetは渡さない(座標は不変)
        img, target = flip(img, target)
        img = to_dtype(img)
        return img, target

    return _transform


def get_eval_transforms():
    to_dtype = v2.ToDtype(torch.float32, scale=True)

    def _transform(img, target):
        return to_dtype(img), target

    return _transform


class KeypointDetectionDataset(Dataset):
    """torchvisionのKeypointRCNN用データセット。

    1インスタンス(=1つの物理構造物、例: 1つのhome_plate)ごとに、
    実点のtight extentをtask.point_box_size_for(class_name)の分だけ拡張した
    囲みboxと、max(point_counts)個にパディングしたkeypointsを持つ。
    パディングスロットはvisibility=0(損失計算から除外される)。
    """

    def __init__(self, samples: List[KeypointSample], task: TaskConfig, transforms=None):
        self.samples = samples
        self.transforms = transforms
        self.point_counts = {name: task.point_counts.get(name, 1) for name in task.classes}
        self.k_max = max(self.point_counts.values())
        self.classes = task.classes
        self.point_box_size_for = task.point_box_size_for

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame = None
        if sample.image_path is not None and sample.image_path.exists():
            frame = cv2.imread(str(sample.image_path))
        if frame is None:
            container = open_video(sample.video)
            try:
                frame = read_frame_bgr(container, sample.frame)
            finally:
                container.close()
        if frame is None:
            raise RuntimeError(f"フレーム読み出し失敗: {sample.video} frame={sample.frame}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        img = tv_tensors.Image(torch.from_numpy(frame).permute(2, 0, 1).contiguous())

        boxes_xyxy, labels, keypoints = [], [], []
        for inst in sample.instances:
            name = self.classes[inst["class_id"]]
            pts_px = [(px * w, py * h) for (px, py) in inst["points"]]
            xs = [p[0] for p in pts_px]
            ys = [p[1] for p in pts_px]
            margin = self.point_box_size_for(name)
            x0 = max(0.0, min(xs) - margin * w)
            y0 = max(0.0, min(ys) - margin * h)
            x1 = min(float(w), max(xs) + margin * w)
            y1 = min(float(h), max(ys) + margin * h)
            if x1 <= x0 or y1 <= y0:
                continue
            boxes_xyxy.append([x0, y0, x1, y1])
            labels.append(inst["class_id"] + 1)  # 背景=0を避けて+1

            kp = [[px, py, 1.0] for (px, py) in pts_px]
            n_real = len(pts_px)
            if n_real < self.k_max:
                cx_pad, cy_pad = pts_px[0]  # 実点1個目を複製(paddingスロットの座標)
                kp += [[cx_pad, cy_pad, 0.0]] * (self.k_max - n_real)
            keypoints.append(kp)

        boxes_t = torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.tensor(labels, dtype=torch.int64)
        keypoints_t = torch.tensor(keypoints, dtype=torch.float32).reshape(-1, self.k_max, 3)
        boxes = tv_tensors.BoundingBoxes(boxes_t, format="XYXY", canvas_size=(h, w))

        target = {
            "boxes": boxes,
            "labels": labels_t,
            "keypoints": keypoints_t,
            "image_id": torch.tensor([idx]),
            "video": sample.video,
            "frame": sample.frame,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
