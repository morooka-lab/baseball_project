"""
bbox_label_web.py で作成したラベル(動画+フレーム番号+正規化bbox)から、
torchvision の物体検出モデルに渡せる Dataset を構築する。

各フレームの画像は、tools/extract_frames.py で書き出し済みならその画像
(cv2.imread)を使い、未抽出なら動画からその場で読み出す(フォールバック)。
毎回動画をシーク+デコードするより大幅に高速なため、学習前に
extract_frames.py を実行しておくことを推奨する。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from utils.label_store import TaskConfig, frame_image_path, iter_label_files
from utils.video_io import open_video, read_frame_bgr
import json


class Sample:
    __slots__ = ("video", "frame", "boxes", "image_path")

    def __init__(self, video: str, frame: int, boxes: list, image_path: Optional[Path] = None):
        self.video = video
        self.frame = frame
        self.boxes = boxes  # list of {class_id(name-mapped済み), cx, cy, w, h}
        self.image_path = image_path  # extract_frames.py 書き出し画像(あれば優先して読む)


def _load_samples(task: TaskConfig) -> List[Sample]:
    """ラベルJSON群を読み込み、Sample一覧を返す。

    各ラベルファイルは保存時点の classes 順序を保持しているため、
    global_classes (classes.yaml) の順序とズレていても名前で正しく
    マッピングし直す。

    annotation_type == "point" の場合、box(w/h)を持たないため、各点を
    中心に task.point_box_size_for(class_name)(正規化サイズ、クラスごとに
    上書き可能)の合成bboxを作り、既存の矩形検出パイプライン(SSDlite)を
    そのまま流用できるようにする。
    """
    samples = []
    name_to_global = {name: i for i, name in enumerate(task.classes)}
    is_point = task.annotation_type == "point"

    for path in iter_label_files(task.labels_dir):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        video = data["video"]
        file_classes = data.get("classes", task.classes)

        for frame_str, boxes in data.get("frames", {}).items():
            remapped = []
            for b in boxes:
                cid = b["class_id"]
                name = file_classes[cid] if cid < len(file_classes) else None
                if name is None or name not in name_to_global:
                    continue  # 未知クラスは無視(classes.yaml変更後の古いラベル対策)
                if is_point:
                    size = task.point_box_size_for(name)
                    w, h = size, size
                else:
                    w, h = b["w"], b["h"]
                remapped.append({
                    "class_id": name_to_global[name],
                    "cx": b["cx"], "cy": b["cy"], "w": w, "h": h,
                })
            frame_num = int(frame_str)
            image_path = frame_image_path(video, task.videos_dir, task.frames_dir, frame_num)
            samples.append(Sample(video, frame_num, remapped, image_path))
    return samples


def build_splits(task: TaskConfig, val_ratio: float = 0.15, seed: int = 42):
    """動画単位でtrain/valに分割する(同一クリップ内フレームのリーク防止)。"""
    samples = _load_samples(task)
    if not samples:
        return [], []

    videos = sorted({s.video for s in samples})
    rng = random.Random(seed)
    rng.shuffle(videos)

    if len(videos) >= 4:
        n_val = max(1, round(len(videos) * val_ratio))
        val_videos = set(videos[:n_val])
        train = [s for s in samples if s.video not in val_videos]
        val = [s for s in samples if s.video in val_videos]
        if train and val:
            return train, val

    # 動画本数が少なすぎる場合はサンプル単位でランダム分割
    rng.shuffle(samples)
    n_val = max(1, round(len(samples) * val_ratio)) if len(samples) > 1 else 0
    return samples[n_val:], samples[:n_val]


class RandomHorizontalFlipWithClassSwap:
    """左右反転 + batter_box_right/left のラベル入れ替えを行うカスタム変換。

    v2.RandomHorizontalFlipは画像とbox座標を反転するだけで、クラスラベルは
    入れ替えない。batter_box_right/batter_box_leftのような左右非対称なクラスに
    そのまま適用すると、反転後は物理的に左右が逆になっているのにラベルは元の
    ままになり、矛盾した教師信号になる。画像全体を一様に反転するため、
    フレーム内の全てのright/leftラベルを一律で入れ替えれば正しくなる。

    swap_pairs: {label_a: label_b}。ラベルは背景+1後の値で指定する。
    対象クラスが無ければ {} でよく、その場合は通常のflipと同じ挙動になる。
    """

    def __init__(self, p: float = 0.5, swap_pairs: Optional[dict] = None):
        self.p = p
        self.swap_pairs = swap_pairs or {}

    def __call__(self, img, target):
        if torch.rand(1).item() >= self.p:
            return img, target
        img = F.horizontal_flip(img)
        target = dict(target)
        target["boxes"] = F.horizontal_flip(target["boxes"])
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
    含まれる場合のみ、水平反転時にラベルを入れ替えるカスタムFlipを使う
    (右/左クラスの既知バグの修正)。"""
    swap_pairs = {}
    if classes and "batter_box_right" in classes and "batter_box_left" in classes:
        r = classes.index("batter_box_right") + 1  # 背景分+1
        l = classes.index("batter_box_left") + 1
        swap_pairs = {r: l}

    return v2.Compose([
        v2.RandomPhotometricDistort(p=0.5),
        v2.RandomZoomOut(fill=[123, 117, 104], side_range=(1.0, 2.5), p=0.3),
        v2.RandomIoUCrop(),
        RandomHorizontalFlipWithClassSwap(p=0.5, swap_pairs=swap_pairs),
        v2.SanitizeBoundingBoxes(),
        v2.ToDtype(torch.float32, scale=True),
    ])


def get_eval_transforms():
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
    ])


class DetectionDataset(Dataset):
    """torchvisionの物体検出モデル用データセット。

    ラベルは class_id を 0-indexed(グローバルclasses順)で保持しているが、
    torchvisionの検出モデルは背景=0を予約するため、labelsは +1 して返す。
    """

    def __init__(self, samples: List[Sample], transforms=None):
        self.samples = samples
        self.transforms = transforms

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

        xyxy = []
        labels = []
        for b in sample.boxes:
            cx, cy, bw, bh = b["cx"] * w, b["cy"] * h, b["w"] * w, b["h"] * h
            x0, y0 = cx - bw / 2, cy - bh / 2
            x1, y1 = cx + bw / 2, cy + bh / 2
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(float(w), x1), min(float(h), y1)
            if x1 <= x0 or y1 <= y0:
                continue
            xyxy.append([x0, y0, x1, y1])
            labels.append(b["class_id"] + 1)  # 背景=0を避けて+1

        boxes_t = torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.tensor(labels, dtype=torch.int64)
        boxes = tv_tensors.BoundingBoxes(boxes_t, format="XYXY", canvas_size=(h, w))

        target = {
            "boxes": boxes,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
            "video": sample.video,
            "frame": sample.frame,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))
