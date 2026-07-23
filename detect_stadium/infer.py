"""
学習済みモデルで未ラベル動画に推論する。player(投手・捕手・打者)専用。
stadiumはinfer_keypoint.pyを使う(keypoint版の方が精度が高いため、stadiumの
bbox推論には対応していない)。

フレームごとの検出結果 (bbox + confidence) をJSONに保存し、
必要に応じてbboxを描画したオーバーレイ動画も出力する。
この結果は後段の「低確信度フレーム抽出(能動学習)」や
「精度検証」「最終結果の保存」で共通して使う。

--source配下には学習時に使ったラベル付き動画も混在し得るため、
デフォルトではラベル(--labels-dir配下の対応JSON)が存在する動画は
推論をスキップする(--include-labeledで無効化可能)。

使い方:
    python infer.py --task player --weights runs/train/player/exp/best.pt \
        --source /data2/baseball_data/videos --out-dir runs/detect/player/exp

    python infer.py --task player --weights runs/train/player/exp/best.pt \
        --source path/to/single_clip.mp4 --save-video
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.detector import load_checkpoint
from utils.label_store import has_label, load_classes, load_tasks
from utils.video_io import open_video, stream_meta


def find_videos(source: str):
    p = Path(source)
    if p.is_file():
        return [p]
    return sorted(p.rglob("*.mp4"))


@torch.inference_mode()
def run_on_video(model, video_path: Path, classes, device, conf_thres, frame_stride, save_video_path=None):
    container = open_video(str(video_path))
    stream = container.streams.video[0]
    meta = stream_meta(stream)
    fps, w, h = meta["fps"], meta["width"], meta["height"]

    writer = None
    if save_video_path is not None:
        save_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_video_path), fourcc, fps, (w, h))

    frames_result = {}
    idx = 0
    for av_frame in container.decode(stream):
        frame = av_frame.to_ndarray(format="bgr24")

        if idx % frame_stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)
            pred = model([img])[0]

            dets = []
            for box, label, score in zip(pred["boxes"].tolist(), pred["labels"].tolist(), pred["scores"].tolist()):
                if score < conf_thres:
                    continue
                class_id = label - 1  # 学習時に背景分+1していたので戻す
                if class_id < 0 or class_id >= len(classes):
                    continue
                dets.append({
                    "class_id": class_id,
                    "class_name": classes[class_id],
                    "score": round(score, 4),
                    "box": [round(v, 1) for v in box],
                })
            frames_result[idx] = dets

            if writer is not None:
                for d in dets:
                    x0, y0, x1, y1 = [int(v) for v in d["box"]]
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
                    text = f'{d["class_name"]} {d["score"]:.2f}'
                    cv2.putText(frame, text, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if writer is not None:
            writer.write(frame)

        idx += 1

    container.close()
    if writer is not None:
        writer.release()

    return {
        "video": str(video_path),
        "classes": classes,
        "fps": fps,
        "width": w,
        "height": h,
        "total_frames": idx,
        "frame_stride": frame_stride,
        "frames": frames_result,
    }


def parse_args():
    p = argparse.ArgumentParser(description="学習済みモデルによる推論")
    p.add_argument("--task", default="player", choices=["player"])
    p.add_argument("--classes-yaml", default=str(ROOT / "data" / "classes.yaml"))
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="動画ファイル or 動画を含むディレクトリ")
    p.add_argument("--videos-dir", default=None,
                    help="ラベル有無判定の基準となる動画ルート (default: classes.yamlのvideos_dir)")
    p.add_argument("--labels-dir", default=None,
                    help="ラベル有無判定に使うラベルディレクトリ (default: classes.yamlのlabels_dir)")
    p.add_argument("--include-labeled", action="store_true",
                    help="ラベル付き(学習使用済み)動画もスキップせず推論する")
    p.add_argument("--out-dir", default=None, help="JSON出力先 (default: runs/detect/<task>/<weightsのrun名>)")
    p.add_argument("--conf-thres", type=float, default=0.3)
    p.add_argument("--frame-stride", type=int, default=1, help="Nフレームに1回推論する")
    p.add_argument("--save-video", action="store_true", help="bbox描画済み動画も保存する")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    tasks = load_tasks(args.classes_yaml)
    task = tasks[args.task]
    classes = load_classes(args.weights, task.classes)
    videos_dir = args.videos_dir or task.videos_dir
    labels_dir = args.labels_dir or task.labels_dir

    model = load_checkpoint(args.weights, len(classes) + 1, device=args.device)

    source_path = Path(args.source)
    source_root = source_path if source_path.is_dir() else source_path.parent
    videos = find_videos(args.source)
    if not videos:
        print(f"動画が見つかりません: {args.source}")
        return

    if not args.include_labeled:
        n_before = len(videos)
        videos = [v for v in videos if not has_label(str(v), videos_dir, labels_dir)]
        n_skipped = n_before - len(videos)
        if n_skipped:
            print(f"ラベル付き(学習使用済み)動画を{n_skipped}件スキップしました。")
        if not videos:
            print("未ラベルの動画がありません。")
            return

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "runs" / "detect" / args.task / Path(args.weights).parent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for video in tqdm(videos, desc="推論"):
        rel = video.relative_to(source_root)
        save_video_path = (out_dir / "videos" / rel) if args.save_video else None
        result = run_on_video(model, video, classes, args.device, args.conf_thres, args.frame_stride, save_video_path)
        result["is_labeled"] = has_label(str(video), videos_dir, labels_dir)

        out_json = out_dir / "predictions" / rel.with_suffix(".json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"推論結果を {out_dir} に保存しました。")


if __name__ == "__main__":
    main()
