"""
学習済みモデルの精度をvalidationセットでmAP評価する。
train.py と同じ (val-ratio, seed) で分割すれば同一のvalidationセットを再現できる。

使い方:
    python val.py --task stadium --weights runs/train/stadium/exp/best.pt
    python val.py --task stadium --weights runs/train/stadium/exp/best.pt --iou-thres 0.5 --conf-thres 0.3
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from infer import load_classes
from models.detector import load_checkpoint
from utils.detection_dataset import DetectionDataset, build_splits, collate_fn, get_eval_transforms
from utils.label_store import load_tasks
from utils.map_eval import evaluate


def parse_args():
    p = argparse.ArgumentParser(description="mAP精度検証")
    p.add_argument("--task", required=True, choices=["stadium", "player"])
    p.add_argument("--classes-yaml", default=str(ROOT / "data" / "classes.yaml"))
    p.add_argument("--videos-dir", default=None)
    p.add_argument("--labels-dir", default=None)
    p.add_argument("--weights", required=True)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--iou-thres", type=float, default=0.5)
    p.add_argument("--conf-thres", type=float, default=0.3, help="評価に使う検出のスコア下限")
    p.add_argument("--out", default=None, help="結果JSONの保存先 (default: weightsと同じディレクトリ)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    tasks = load_tasks(args.classes_yaml)
    task = tasks[args.task]
    if args.videos_dir:
        task.videos_dir = args.videos_dir
    if args.labels_dir:
        task.labels_dir = args.labels_dir

    classes = load_classes(args.weights, task.classes)
    num_classes = len(classes) + 1

    _, val_samples = build_splits(task, val_ratio=args.val_ratio, seed=args.seed)
    if not val_samples:
        print("validationサンプルがありません(データが少なすぎる可能性があります)。")
        return
    print(f"validation samples: {len(val_samples)}")

    val_ds = DetectionDataset(val_samples, transforms=get_eval_transforms())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = load_checkpoint(args.weights, num_classes, device=args.device)

    gts = {c: {} for c in range(len(classes))}
    preds = {c: [] for c in range(len(classes))}

    for image_id, (images, targets) in enumerate(tqdm(val_loader, desc="評価")):
        images = [img.to(args.device) for img in images]
        outputs = model(images)

        target = targets[0]
        for box, label in zip(target["boxes"].tolist(), target["labels"].tolist()):
            c = label - 1
            gts[c].setdefault(image_id, []).append(box)

        out = outputs[0]
        for box, label, score in zip(out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist()):
            if score < args.conf_thres:
                continue
            c = label - 1
            if 0 <= c < len(classes):
                preds[c].append((image_id, score, box))

    per_class, mean_ap = evaluate(gts, preds, len(classes), iou_thres=args.iou_thres)

    print(f"\n=== mAP@{args.iou_thres} (conf>={args.conf_thres}) ===")
    for c, name in enumerate(classes):
        info = per_class[c]
        print(f"  {name:20s} AP={info['ap']:.4f}  (GT数={info['num_gt']})")
    print(f"  {'mAP':20s} = {mean_ap:.4f}")

    result = {
        "weights": args.weights,
        "iou_thres": args.iou_thres,
        "conf_thres": args.conf_thres,
        "num_val_samples": len(val_samples),
        "per_class": {classes[c]: v for c, v in per_class.items()},
        "mAP": mean_ap,
    }
    out_path = Path(args.out) if args.out else Path(args.weights).parent / "val_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n結果を {out_path} に保存しました。")


if __name__ == "__main__":
    main()
