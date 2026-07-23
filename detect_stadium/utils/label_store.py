"""
物体検出タスク（球場情報・選手）で共通して使うラベルI/O。

ラベルは動画1本につき1つのJSONファイルとして保存する（videosディレクトリと
同じ相対パス構造をlabelsディレクトリ配下に再現する）。
中身は「アノテーション済みフレームのみ」を持つ疎な構造で、フレームごとに
複数クラス・複数個のbbox(またはpoint)を保持できる。

タスクごとに annotation_type ("bbox" | "point", classes.yamlで指定) を持ち、
bboxタスクは {class_id, cx, cy, w, h}、pointタスクは {class_id, cx, cy} を
box辞書として保存する。cx/cy/w/hはいずれも画像サイズに対する正規化値。

{
  "video": "<動画への絶対パス>",
  "classes": ["pitcher_plate", "home_plate", "batter_box_right", "batter_box_left"],
  "frames": {
    "123": [{"class_id": 0, "cx": 0.51, "cy": 0.62}, ...],   # point の例
    "456": []   # レビュー済みだが対象物なし
  }
}
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import yaml


@dataclass
class TaskConfig:
    name: str
    description: str
    videos_dir: str
    labels_dir: str
    classes: List[str]
    annotation_type: str = "bbox"  # "bbox"(矩形) | "point"(座標点)
    point_box_size: float = 0.03  # pointタスクを学習させる際、各点を中心にこのサイズ(w/h比)の合成bboxを作る(デフォルト)
    point_box_size_overrides: Dict[str, float] = field(default_factory=dict)  # クラス名ごとにpoint_box_sizeを上書き(例: home_plateのように点が密集するクラスを小さくする)
    frames_dir: str = ""  # extract_frames.pyの書き出し先(省略時はlabels_dirの隣の"frames")
    point_counts: Dict[str, int] = field(default_factory=dict)  # クラス名 -> 実体を構成する点の数(例: home_plate=5)。未指定クラスは1点扱い

    def point_box_size_for(self, class_name: str) -> float:
        return self.point_box_size_overrides.get(class_name, self.point_box_size)


def load_tasks(classes_yaml: str) -> Dict[str, TaskConfig]:
    with open(classes_yaml, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    tasks = {}
    for name, cfg in raw.items():
        labels_dir = cfg["labels_dir"]
        tasks[name] = TaskConfig(
            name=name,
            description=cfg.get("description", ""),
            videos_dir=cfg["videos_dir"],
            labels_dir=labels_dir,
            classes=list(cfg["classes"]),
            annotation_type=cfg.get("annotation_type", "bbox"),
            point_box_size=cfg.get("point_box_size", 0.03),
            point_box_size_overrides=cfg.get("point_box_size_overrides", {}),
            frames_dir=cfg.get("frames_dir", str(Path(labels_dir).parent / "frames")),
            point_counts=cfg.get("point_counts", {}),
        )
    return tasks


def load_classes(weights_path: str, task_classes: List[str]) -> List[str]:
    """重みファイルと同じディレクトリのclasses.jsonがあればそちらを優先する
    (学習時のクラス順とclasses.yaml現在値がズレていても推論側が壊れないようにする)。"""
    classes_json = Path(weights_path).parent / "classes.json"
    if classes_json.exists():
        with open(classes_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return task_classes


def list_videos(videos_dir: str) -> List[Path]:
    root = Path(videos_dir)
    if not root.exists():
        return []
    return sorted(root.rglob("*.mp4"))


def label_path_for(video_path: str, videos_dir: str, labels_dir: str) -> Path:
    p = Path(video_path)
    try:
        rel = p.relative_to(videos_dir)
    except ValueError:
        rel = Path(p.name)
    return Path(labels_dir) / rel.with_suffix(".json")


def has_label(video_path: str, videos_dir: str, labels_dir: str) -> bool:
    """動画に対応するラベルJSONが存在するか(=アノテーション済み、学習に使用済みか)。"""
    return label_path_for(video_path, videos_dir, labels_dir).exists()


def frame_image_path(video_path: str, videos_dir: str, frames_dir: str, frame_num: int) -> Path:
    """extract_frames.py が書き出す画像パス(動画ごとにディレクトリを分け、frame_numをファイル名にする)。"""
    p = Path(video_path)
    try:
        rel = p.relative_to(videos_dir)
    except ValueError:
        rel = Path(p.name)
    return Path(frames_dir) / rel.with_suffix("") / f"{frame_num:06d}.jpg"


def load_video_labels(video_path: str, videos_dir: str, labels_dir: str) -> Dict[int, List[dict]]:
    """frame_num(int) -> list of box dict"""
    path = label_path_for(video_path, videos_dir, labels_dir)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.get("frames", {}).items()}


def save_video_labels(
    video_path: str,
    videos_dir: str,
    labels_dir: str,
    classes: List[str],
    frames: Dict[int, List[dict]],
) -> Path:
    path = label_path_for(video_path, videos_dir, labels_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "video": str(video_path),
        "classes": classes,
        "frames": {str(k): v for k, v in sorted(frames.items())},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def iter_label_files(labels_dir: str):
    root = Path(labels_dir)
    if not root.exists():
        return []
    return sorted(root.rglob("*.json"))


def count_annotated_frames(labels_dir: str) -> int:
    """ラベル済みフレーム数の合計（box数が0でもレビュー済みならカウントする）"""
    total = 0
    for path in iter_label_files(labels_dir):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        total += len(data.get("frames", {}))
    return total


def split_by_video(items: list, get_video: Callable[[object], str],
                    val_ratio: float = 0.15, seed: int = 42) -> Tuple[list, list]:
    """items を get_video(item) で得られる動画IDでグルーピングし、動画単位で
    train/valに分割する(同一クリップ内フレームのリーク防止)。

    utils/detection_dataset.py の build_splits() と utils/keypoint_dataset.py の
    build_splits() の両方から使える汎用ヘルパー(サンプルの型に依存しない)。
    動画本数が少なすぎる場合はサンプル単位のランダム分割にフォールバックする。
    """
    if not items:
        return [], []

    videos = sorted({get_video(it) for it in items})
    rng = random.Random(seed)
    rng.shuffle(videos)

    if len(videos) >= 4:
        n_val = max(1, round(len(videos) * val_ratio))
        val_videos = set(videos[:n_val])
        train = [it for it in items if get_video(it) not in val_videos]
        val = [it for it in items if get_video(it) in val_videos]
        if train and val:
            return train, val

    items = list(items)
    rng.shuffle(items)
    n_val = max(1, round(len(items) * val_ratio)) if len(items) > 1 else 0
    return items[n_val:], items[:n_val]
