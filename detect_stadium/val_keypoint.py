"""
学習済みキーポイント検出モデルの精度を検証する。train_keypoint.py と同じ
(val-ratio, seed) で分割すれば同一のvalidationセットを再現できる。

2種類の指標を出す:
  (a) box mAP: 各インスタンスの囲みboxをそのまま使い、既存の utils/map_eval.py を
      再利用する(train.py/val.pyのSSDlineベースラインとの粗い比較用)。
  (b) キーポイントPCK的指標: 正規化ユークリッド距離が閾値以内に収まった
      キーポイントの割合。1フレーム・1クラスあたり最大1インスタンスという
      前提のもと、GTと最高スコアの予測を1対1で対応付けるだけの単純な実装。
      この指標がこのプロジェクトの本来の目的(点の位置精度・過剰分割の解消)
      により直結する。

使い方:
    python val_keypoint.py --task stadium --weights runs/train_keypoint/stadium/exp/best.pt
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

from models.keypoint_detector import load_checkpoint
from utils.keypoint_dataset import KeypointDetectionDataset, build_splits, collate_fn, get_eval_transforms
from utils.label_store import load_classes, load_tasks
from utils.map_eval import evaluate


def load_keypoint_meta(weights_path: str, task) -> dict:
    meta_json = Path(weights_path).parent / "keypoint_meta.json"
    if meta_json.exists():
        with open(meta_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"num_keypoints": max(task.point_counts.get(c, 1) for c in task.classes),
            "point_counts": task.point_counts}


def keypoint_pck(gts_by_frame, preds_by_frame, sizes_by_frame, classes, num_classes, px_thresh_norm=0.03):
    """gts_by_frame:   {image_id: {class_id: [(x,y), ...]}}   # 実点のみ(パディング除く)
       preds_by_frame: {image_id: {class_id: (score, [(x,y),...])}}  # クラスごとに最高スコア1件
       sizes_by_frame: {image_id: (w,h)}
       px_thresh_norm: 正規化ユークリッド距離((dx/w)^2+(dy/h)^2の平方根)の許容閾値。
           point_box_sizeと同じ0.03スケールを踏襲し、新しいマジックナンバーを増やさない。
       戻り値: {class_id: {"n_gt","n_matched_inst","n_kpts","n_kpts_ok","pck","recall"}}
    """
    stats = {c: {"n_gt": 0, "n_matched_inst": 0, "n_kpts": 0, "n_kpts_ok": 0} for c in range(num_classes)}
    for image_id, gt_classes in gts_by_frame.items():
        w, h = sizes_by_frame[image_id]
        preds = preds_by_frame.get(image_id, {})
        for c, gt_pts in gt_classes.items():
            stats[c]["n_gt"] += 1
            if c not in preds:
                continue  # 検出漏れ(recallに反映、PCK分母には数えない)
            _, pred_pts = preds[c]
            stats[c]["n_matched_inst"] += 1
            for (gx, gy), (px, py) in zip(gt_pts, pred_pts):
                stats[c]["n_kpts"] += 1
                d = ((gx - px) / w) ** 2 + ((gy - py) / h) ** 2
                if d ** 0.5 <= px_thresh_norm:
                    stats[c]["n_kpts_ok"] += 1
    for c, s in stats.items():
        s["pck"] = s["n_kpts_ok"] / s["n_kpts"] if s["n_kpts"] else 0.0
        s["recall"] = s["n_matched_inst"] / s["n_gt"] if s["n_gt"] else 0.0
    return stats


def parse_args():
    p = argparse.ArgumentParser(description="キーポイント検出モデルの精度検証")
    p.add_argument("--task", default="stadium", choices=["stadium"])
    p.add_argument("--classes-yaml", default=str(ROOT / "data" / "classes.yaml"))
    p.add_argument("--videos-dir", default=None)
    p.add_argument("--labels-dir", default=None)
    p.add_argument("--weights", required=True)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--iou-thres", type=float, default=0.5, help="box mAP用のIoU閾値")
    p.add_argument("--conf-thres", type=float, default=0.3, help="評価に使う検出のスコア下限")
    p.add_argument("--kp-thres", type=float, default=0.03, help="PCK指標の正規化距離閾値")
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
    meta = load_keypoint_meta(args.weights, task)
    num_keypoints = meta["num_keypoints"]
    point_counts = meta["point_counts"]
    num_classes = len(classes) + 1

    _, val_samples = build_splits(task, val_ratio=args.val_ratio, seed=args.seed)
    if not val_samples:
        print("validationサンプルがありません(データが少なすぎる可能性があります)。")
        return
    print(f"validation samples: {len(val_samples)}")

    val_ds = KeypointDetectionDataset(val_samples, task, transforms=get_eval_transforms())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = load_checkpoint(args.weights, num_classes, num_keypoints, device=args.device)

    gts = {c: {} for c in range(len(classes))}
    preds_box = {c: [] for c in range(len(classes))}
    gts_by_frame = {}
    preds_by_frame = {}
    sizes_by_frame = {}

    for image_id, (images, targets) in enumerate(tqdm(val_loader, desc="評価")):
        h, w = images[0].shape[-2:]
        sizes_by_frame[image_id] = (w, h)
        images_dev = [img.to(args.device) for img in images]
        outputs = model(images_dev)

        target = targets[0]
        gt_classes = {}
        for box, label, kp in zip(target["boxes"].tolist(), target["labels"].tolist(),
                                    target["keypoints"].tolist()):
            c = label - 1
            gts[c].setdefault(image_id, []).append(box)
            name = classes[c]
            n_real = point_counts.get(name, 1)
            gt_classes[c] = [(pt[0], pt[1]) for pt in kp[:n_real]]
        gts_by_frame[image_id] = gt_classes

        out = outputs[0]
        best_per_class = {}
        for box, label, score, kp in zip(out["boxes"].tolist(), out["labels"].tolist(),
                                           out["scores"].tolist(), out["keypoints"].tolist()):
            if score < args.conf_thres:
                continue
            c = label - 1
            if not (0 <= c < len(classes)):
                continue
            preds_box[c].append((image_id, score, box))
            name = classes[c]
            n_real = point_counts.get(name, 1)
            if c not in best_per_class or score > best_per_class[c][0]:
                best_per_class[c] = (score, [(pt[0], pt[1]) for pt in kp[:n_real]])
        preds_by_frame[image_id] = best_per_class

    per_class, mean_ap = evaluate(gts, preds_box, len(classes), iou_thres=args.iou_thres)
    pck_stats = keypoint_pck(gts_by_frame, preds_by_frame, sizes_by_frame, classes, len(classes),
                              px_thresh_norm=args.kp_thres)

    print(f"\n=== box mAP@{args.iou_thres} (conf>={args.conf_thres}) ===")
    for c, name in enumerate(classes):
        info = per_class[c]
        print(f"  {name:20s} AP={info['ap']:.4f}  (GT数={info['num_gt']})")
    print(f"  {'mAP':20s} = {mean_ap:.4f}")

    print(f"\n=== keypoint PCK@{args.kp_thres} (conf>={args.conf_thres}) ===")
    for c, name in enumerate(classes):
        s = pck_stats[c]
        print(f"  {name:20s} PCK={s['pck']:.4f}  recall={s['recall']:.4f}  "
              f"(GTインスタンス数={s['n_gt']}, 検出キーポイント数={s['n_kpts']})")

    result = {
        "weights": args.weights,
        "iou_thres": args.iou_thres,
        "conf_thres": args.conf_thres,
        "kp_thres": args.kp_thres,
        "num_val_samples": len(val_samples),
        "box_map": {
            "per_class": {classes[c]: v for c, v in per_class.items()},
            "mAP": mean_ap,
        },
        "keypoint_pck": {classes[c]: v for c, v in pck_stats.items()},
    }
    out_path = Path(args.out) if args.out else Path(args.weights).parent / "val_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n結果を {out_path} に保存しました。")


if __name__ == "__main__":
    main()
