"""
ラベルJSONが参照しているフレームだけを動画から画像として書き出す。

DetectionDataset は本来、学習の都度動画をシークしてフレームをデコードするが、
これは同じフレームでもエポックごとに毎回シーク+デコードし直すため遅い。
事前に画像化しておけば、学習時は cv2.imread で読むだけになり大幅に高速化できる。

フレーム番号は bbox_label_web.py の /stream_frames と同じ「シークなしの
逐次デコード順」を基準にしているため、動画を先頭から1回だけ順に読み進めながら
対象フレーム番号に達した時点で保存する(動画ごとに複数回シークするより高速かつ、
VFR素材でのシーク誤差の影響も受けない)。

使い方:
    python tools/extract_frames.py --task stadium
    python tools/extract_frames.py --task player --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.label_store import frame_image_path, iter_label_files, load_tasks
from utils.video_io import open_video


def collect_video_frames(labels_dir: str) -> dict:
    """video(絶対パス) -> 対象フレーム番号のset"""
    video_frames: dict = {}
    for path in iter_label_files(labels_dir):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        video = data["video"]
        frames = {int(k) for k in data.get("frames", {}).keys()}
        if frames:
            video_frames.setdefault(video, set()).update(frames)
    return video_frames


def extract_video_frames(
    video_path: str,
    videos_dir: str,
    frames_dir: str,
    frame_nums: set,
    overwrite: bool,
    jpeg_quality: int,
) -> int:
    targets = {}
    for fn in frame_nums:
        out_path = frame_image_path(video_path, videos_dir, frames_dir, fn)
        if out_path.exists() and not overwrite:
            continue
        targets[fn] = out_path
    if not targets:
        return 0

    container = open_video(video_path)
    saved = 0
    remaining = dict(targets)
    try:
        stream = container.streams.video[0]
        idx = 0
        for av_frame in container.decode(stream):
            if idx in remaining:
                out_path = remaining.pop(idx)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                frame = av_frame.to_ndarray(format="bgr24")
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                saved += 1
                if not remaining:
                    break
            idx += 1
    finally:
        container.close()

    if remaining:
        print(f"警告: {Path(video_path).name} で{len(remaining)}フレーム抽出できませんでした "
              f"(動画長を超えるフレーム番号の可能性): {sorted(remaining.keys())}")
    return saved


def parse_args():
    p = argparse.ArgumentParser(description="ラベル済みフレームを動画から画像として抽出する")
    p.add_argument("--task", required=True, choices=["stadium", "player"])
    p.add_argument("--classes-yaml", default=str(ROOT / "data" / "classes.yaml"))
    p.add_argument("--videos-dir", default=None, help="省略時はclasses.yamlの値")
    p.add_argument("--labels-dir", default=None, help="省略時はclasses.yamlの値")
    p.add_argument("--frames-dir", default=None, help="省略時はclasses.yamlの値(未指定ならlabels_dirの隣のframes)")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--overwrite", action="store_true", help="既存の画像も再抽出する")
    return p.parse_args()


def main():
    args = parse_args()
    tasks = load_tasks(args.classes_yaml)
    task = tasks[args.task]
    videos_dir = args.videos_dir or task.videos_dir
    labels_dir = args.labels_dir or task.labels_dir
    frames_dir = args.frames_dir or task.frames_dir

    video_frames = collect_video_frames(labels_dir)
    total_frames = sum(len(v) for v in video_frames.values())
    print(f"対象動画数: {len(video_frames)}  総フレーム数: {total_frames}  出力先: {frames_dir}")
    if not video_frames:
        print("ラベル済みフレームがありません。")
        return

    total_saved = 0
    for video, frame_nums in tqdm(video_frames.items(), desc="抽出"):
        total_saved += extract_video_frames(
            video, videos_dir, frames_dir, frame_nums, args.overwrite, args.jpeg_quality
        )

    skipped = total_frames - total_saved
    print(f"完了: {total_saved} 枚を新規保存 ({skipped} 枚は既存のためスキップ) -> {frames_dir}")


if __name__ == "__main__":
    main()
