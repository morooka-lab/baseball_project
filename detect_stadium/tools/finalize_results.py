"""
最終成果物の作成 (ワークフローのステップ5)。

人手アノテーション(正解)がある フレームはそれを優先し、
アノテーションがないフレームはモデル推論結果で埋めて、
動画1本ごとに「最終的な検出結果」をJSON+オーバーレイ動画として保存する。

事前に infer.py(player)またはinfer_keypoint.py(stadium)で推論を実行しておくこと。

使い方:
    python infer_keypoint.py --task stadium --weights runs/train_keypoint/stadium/exp/best.pt \
        --source /data2/baseball_data/detect_dataset/videos --out-dir runs/detect_keypoint/stadium/exp

    python tools/finalize_results.py --task stadium \
        --predictions-dir runs/detect_keypoint/stadium/exp/predictions \
        --out-dir runs/final/stadium
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

from utils.label_store import load_tasks, load_video_labels
from utils.video_io import open_video

COLOR_HUMAN = (0, 220, 0)     # 人手アノテーション(緑)
COLOR_MODEL = (0, 0, 255)     # モデル推論(赤)


def box_from_norm(b, w, h):
    cx, cy, bw, bh = b["cx"] * w, b["cy"] * h, b["w"] * w, b["h"] * h
    return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]


def merge_video(task, pred_data, out_dir, save_video):
    video = pred_data["video"]
    classes = pred_data["classes"]
    w, h = pred_data["width"], pred_data["height"]
    human_labels = load_video_labels(video, task.videos_dir, task.labels_dir)

    merged_frames = {}
    for frame_str, dets in pred_data["frames"].items():
        frame_num = int(frame_str)
        if frame_num in human_labels:
            boxes = human_labels[frame_num]
            merged_frames[frame_num] = {
                "source": "human",
                "detections": [
                    {
                        "class_id": b["class_id"],
                        "class_name": classes[b["class_id"]],
                        "score": 1.0,
                        "box": box_from_norm(b, w, h),
                    }
                    for b in boxes
                ],
            }
        else:
            merged_frames[frame_num] = {"source": "model", "detections": dets}

    result = {
        "video": video,
        "classes": classes,
        "fps": pred_data["fps"],
        "width": w,
        "height": h,
        "frames": merged_frames,
    }

    rel_name = Path(video).stem
    out_json = out_dir / "results" / f"{rel_name}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if save_video:
        container = open_video(video)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        out_video_path = out_dir / "videos" / f"{rel_name}.mp4"
        out_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        idx = 0
        for av_frame in container.decode(stream):
            frame = av_frame.to_ndarray(format="bgr24")
            entry = merged_frames.get(idx)
            if entry:
                color = COLOR_HUMAN if entry["source"] == "human" else COLOR_MODEL
                for d in entry["detections"]:
                    x0, y0, x1, y1 = [int(v) for v in d["box"]]
                    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                    cv2.putText(frame, f'{d["class_name"]} {d["score"]:.2f}', (x0, max(0, y0 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            writer.write(frame)
            idx += 1
        container.close()
        writer.release()

    return out_json


def parse_args():
    p = argparse.ArgumentParser(description="人手アノテーション+モデル推論のマージによる最終結果の保存")
    p.add_argument("--task", required=True, choices=["stadium", "player"])
    p.add_argument("--classes-yaml", default=str(ROOT / "data" / "classes.yaml"))
    p.add_argument("--predictions-dir", required=True, help="infer.pyの出力(predictions/*.json)ディレクトリ")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--save-video", action="store_true", help="緑=人手/赤=モデル で描画した動画も保存する")
    return p.parse_args()


def main():
    args = parse_args()
    tasks = load_tasks(args.classes_yaml)
    task = tasks[args.task]

    pred_files = sorted(Path(args.predictions_dir).rglob("*.json"))
    if not pred_files:
        print(f"推論結果が見つかりません: {args.predictions_dir}")
        return

    out_dir = Path(args.out_dir)
    for pf in tqdm(pred_files, desc="最終結果の作成"):
        with open(pf, "r", encoding="utf-8") as f:
            pred_data = json.load(f)
        merge_video(task, pred_data, out_dir, args.save_video)

    print(f"最終結果を {out_dir} に保存しました。(緑=人手アノテーション優先, 赤=モデル推論で補完)")


if __name__ == "__main__":
    main()
