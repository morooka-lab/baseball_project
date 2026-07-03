"""キーポイント検出モデル用のシンプルな動画レベル集約。

utils/stabilize.py(SSDlite用)は、モデルが点の識別を持たないためK-meansで
point_counts個に強制クラスタリングし、過剰分割を抑えるための追加のマージ処理
(_merge_close_clusters)まで必要だった。キーポイント検出モデルは各スロットの
識別を自身の出力構造(keypoints配列のインデックス)として既に持っているため、
クラスタリングは一切不要で、クラス×スロットごとに座標の中央値を取るだけでよい。
"""

import numpy as np


def aggregate_video_keypoints(frames: dict, classes: list, conf_thres: float = 0.5) -> list:
    """frames: infer_keypoint.run_on_video()が返す
        {frame_idx: [{"class_id":.., "class_name":.., "score":.., "keypoints":[[x,y],...]}, ...]}
       (keypointsは既にinfer_keypoint.py側でpoint_counts[class]個にトリム済み)
    戻り値: [{"class_id":.., "class_name":.., "point_index":.., "xy":[x,y], "n_frames":..}, ...]
    """
    per_slot = {}  # (class_id, slot) -> list[(x,y)]
    for dets in frames.values():
        for d in dets:
            if d["score"] < conf_thres:
                continue
            for slot, (x, y) in enumerate(d["keypoints"]):
                per_slot.setdefault((d["class_id"], slot), []).append((x, y))

    out = []
    for (cid, slot), coords in sorted(per_slot.items()):
        arr = np.array(coords, dtype=float)
        out.append({
            "class_id": cid,
            "class_name": classes[cid],
            "point_index": slot,
            "xy": [round(float(np.median(arr[:, 0])), 1), round(float(np.median(arr[:, 1])), 1)],
            "n_frames": len(coords),
        })
    return out
