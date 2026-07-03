"""VOCスタイルのmAP計算 (pycocotools不要の軽量実装)。"""

from __future__ import annotations

import numpy as np


def box_iou(box, gt_box):
    x0 = max(box[0], gt_box[0])
    y0 = max(box[1], gt_box[1])
    x1 = min(box[2], gt_box[2])
    y1 = min(box[3], gt_box[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    return inter / (area1 + area2 - inter)


def voc_ap(recalls, precisions):
    """all-point interpolation (VOC2012方式)"""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def average_precision_for_class(gts_by_image: dict, preds: list, iou_thres: float):
    """gts_by_image: {image_id: [box, ...]}, preds: [(image_id, score, box), ...]"""
    npos = sum(len(v) for v in gts_by_image.values())
    if not preds:
        return 0.0, npos
    preds = sorted(preds, key=lambda p: -p[1])
    matched = {img_id: [False] * len(boxes) for img_id, boxes in gts_by_image.items()}

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for i, (image_id, _score, box) in enumerate(preds):
        gt_boxes = gts_by_image.get(image_id, [])
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(gt_boxes):
            if matched[image_id][j]:
                continue
            iou = box_iou(box, gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thres and best_j >= 0:
            tp[i] = 1
            matched[image_id][best_j] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / npos if npos > 0 else np.zeros_like(cum_tp)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    ap = voc_ap(recalls, precisions) if npos > 0 else 0.0
    return ap, npos


def evaluate(gts, preds, num_classes, iou_thres=0.5):
    """
    gts:   {class_id: {image_id: [box, ...]}}
    preds: {class_id: [(image_id, score, box), ...]}
    戻り値: {class_id: {"ap": float, "num_gt": int}}, mAP
    """
    per_class = {}
    for c in range(num_classes):
        ap, npos = average_precision_for_class(gts.get(c, {}), preds.get(c, []), iou_thres)
        per_class[c] = {"ap": ap, "num_gt": npos}
    with_gt = [v["ap"] for v in per_class.values() if v["num_gt"] > 0]
    mean_ap = float(np.mean(with_gt)) if with_gt else 0.0
    return per_class, mean_ap
