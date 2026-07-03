"""
infer.py の推論結果から「モデルが自信を持てていないフレーム」を抽出し、
追加アノテーション候補としてリストアップする(能動学習ループ用)。

優先度の高い順に並べる基準:
  1. 検出0件のフレーム (見逃し=偽陰性の可能性、最優先)
  2. 検出はあるが最大confidenceが低いフレーム

既にアノテーション済みのフレームは除外する。

使い方:
    # 事前に低いconf-thresで推論しておく (境界線上の検出も拾うため)
    python infer.py --task stadium --weights runs/train/stadium/exp/best.pt \
        --source /data2/baseball_data/videos --conf-thres 0.05 \
        --out-dir runs/detect/stadium/exp

    python tools/select_uncertain_frames.py --task stadium \
        --predictions-dir runs/detect/stadium/exp/predictions --top-k 50
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.label_store import load_tasks, load_video_labels


def parse_args():
    p = argparse.ArgumentParser(description="低確信度フレーム抽出(能動学習用)")
    p.add_argument("--task", required=True, choices=["stadium", "player"])
    p.add_argument("--classes-yaml", default=str(ROOT / "data" / "classes.yaml"))
    p.add_argument("--predictions-dir", required=True, help="infer.pyの出力(predictions/*.json)ディレクトリ")
    p.add_argument("--conf-thres", type=float, default=0.5, help="この値未満のmax scoreを「不確か」とみなす")
    p.add_argument("--top-k", type=int, default=50, help="候補として出力する最大フレーム数")
    p.add_argument("--max-per-video", type=int, default=8, help="1動画あたりの候補上限(偏り防止)")
    p.add_argument("--out", default=None, help="出力JSONパス (default: <predictions-dir>/../uncertain_frames.json)")
    return p.parse_args()


def main():
    args = parse_args()
    tasks = load_tasks(args.classes_yaml)
    task = tasks[args.task]

    pred_dir = Path(args.predictions_dir)
    pred_files = sorted(pred_dir.rglob("*.json"))
    if not pred_files:
        print(f"推論結果が見つかりません: {pred_dir}")
        return

    candidates = []
    for pf in pred_files:
        with open(pf, "r", encoding="utf-8") as f:
            data = json.load(f)
        video = data["video"]
        already = load_video_labels(video, task.videos_dir, task.labels_dir)

        for frame_str, dets in data["frames"].items():
            frame_num = int(frame_str)
            if frame_num in already:
                continue  # 既にアノテーション済み

            if not dets:
                candidates.append({
                    "video": video, "frame": frame_num, "reason": "no_detection",
                    "max_score": 0.0, "num_detections": 0,
                    "classes_detected": [],
                })
            else:
                max_score = max(d["score"] for d in dets)
                if max_score < args.conf_thres:
                    candidates.append({
                        "video": video, "frame": frame_num, "reason": "low_confidence",
                        "max_score": max_score, "num_detections": len(dets),
                        "classes_detected": sorted({d["class_name"] for d in dets}),
                    })

    # 不確かな順(検出0件を最優先、次にmax_scoreの低い順)にソート
    candidates.sort(key=lambda c: (c["reason"] != "no_detection", c["max_score"]))

    # 1動画あたりの上限を適用しつつtop-kまで採用
    per_video_count = {}
    selected = []
    for c in candidates:
        cnt = per_video_count.get(c["video"], 0)
        if cnt >= args.max_per_video:
            continue
        selected.append(c)
        per_video_count[c["video"]] = cnt + 1
        if len(selected) >= args.top_k:
            break

    out_path = Path(args.out) if args.out else pred_dir.parent / "uncertain_frames.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    print(f"候補 {len(selected)} フレーム (全候補 {len(candidates)} 件中) を {out_path} に保存しました。")
    print(f"内訳: no_detection={sum(1 for c in selected if c['reason']=='no_detection')}  "
          f"low_confidence={sum(1 for c in selected if c['reason']=='low_confidence')}")
    print("\nbbox_label_web.py を起動し、各動画のフレーム番号にジャンプしてアノテーションしてください。")
    for c in selected[:10]:
        print(f"  {Path(c['video']).name}  frame={c['frame']}  reason={c['reason']}  max_score={c['max_score']:.3f}")
    if len(selected) > 10:
        print(f"  ... 他 {len(selected) - 10} 件")


if __name__ == "__main__":
    main()
