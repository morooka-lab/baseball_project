"""検出結果を動画単位で統計的に集約し、遮蔽・誤検出に強い代表座標を求める。

球場の基準点(投手板・本塁・バッターボックス)は同一動画内(カメラがほぼ固定の
短いクリップ)では位置がほとんど変わらない、という前提のもと、フレームごとの
検出座標をクラスごとにまとめて動画全体で代表座標を算出する。

注意: 1クラスは実際には複数の点で構成される(例: home_plateは五角形の5頂点、
pitcher_plateは2点、batter_boxは各4点)。そのため単純にクラスごと最高scoreの
検出1個を選ぶと点が失われる。task.point_counts(クラス名->点数)を使い、
クラスごとにK-meansで座標をクラスタリングして点を分離してから、
クラスタ(=同一の点)ごとに中央値を取って代表座標とする。

K-meansはpoint_counts個のクラスタに常に分割しようとするため、検出のブレや
重複検出によって実際には1つしかない点が2つのクラスタに割れてしまうことが
ある。これを抑えるため、クラスタ中心同士が近すぎる場合は統合する
(_merge_close_clusters)。
"""

import numpy as np


def _kmeans(points: np.ndarray, k: int, n_iter: int = 50, seed: int = 0) -> np.ndarray:
    """points: (N,2)。戻り値はクラスタ番号の配列(N,)。"""
    n = len(points)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    centers = points[rng.choice(n, size=k, replace=False)].copy()
    labels = None
    for _ in range(n_iter):
        dists = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            mask = labels == i
            if mask.any():
                centers[i] = points[mask].mean(axis=0)
    return labels


def _merge_close_clusters(centers: np.ndarray, labels: np.ndarray, min_dist: float) -> np.ndarray:
    """クラスタ中心同士の距離がmin_dist未満なら同一の点とみなして統合する。

    point_counts個に無理やり分割した結果、本来1点しかない場所が検出のブレで
    2クラスタに割れることがある(例: home_plateの尖った頂点が2つ検出される)。
    そのクラスタの中心同士がお互いの検出boxサイズ(min_dist)より近い場合は
    統合し、過剰分割を防ぐ。
    """
    if min_dist <= 0:
        return labels

    parent = {c: c for c in sorted(set(labels.tolist()))}

    def find(c):
        while parent[c] != c:
            c = parent[c]
        return c

    def cluster_center(root):
        mask = np.array([find(l) == root for l in labels])
        return centers[mask].mean(axis=0)

    changed = True
    while changed:
        changed = False
        roots = sorted({find(c) for c in parent})
        for i in range(len(roots)):
            for j in range(i + 1, len(roots)):
                a, b = roots[i], roots[j]
                if np.linalg.norm(cluster_center(a) - cluster_center(b)) < min_dist:
                    parent[b] = a
                    changed = True
                    break
            if changed:
                break

    return np.array([find(l) for l in labels])


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    """統合でラベル番号が飛び飛びになるため、0始まりの連番に振り直す。"""
    remap = {old: new for new, old in enumerate(sorted(set(labels.tolist())))}
    return np.array([remap[l] for l in labels])


def aggregate_video_detections(frames: dict, classes: list, point_counts: dict = None,
                                conf_thres: float = 0.3, merge_factor: float = 1.0) -> list:
    """frames: infer.run_on_video()が返す {frame_idx: [detection, ...]} 形式。
    point_counts: クラス名 -> そのクラスを構成する点の数(未指定なら1点として扱う)。
    merge_factor: クラスタ中心同士の距離がこの倍率×(そのクラスの検出box中央値サイズ)
        未満なら統合する。0にすると統合を無効化(従来の挙動)。
    戻り値: [{class_id, class_name, point_index, box, n_frames}, ...]
    """
    point_counts = point_counts or {}
    per_class_boxes = {i: [] for i in range(len(classes))}
    for dets in frames.values():
        for d in dets:
            if d["score"] >= conf_thres:
                per_class_boxes[d["class_id"]].append(d["box"])

    stabilized = []
    for cid, boxes in per_class_boxes.items():
        if not boxes:
            continue
        name = classes[cid]
        k = point_counts.get(name, 1)
        arr = np.array(boxes, dtype=float)  # (N,4)
        centers = np.stack([(arr[:, 0] + arr[:, 2]) / 2, (arr[:, 1] + arr[:, 3]) / 2], axis=1)
        labels = _kmeans(centers, k)

        if k > 1 and merge_factor > 0:
            box_sizes = np.concatenate([arr[:, 2] - arr[:, 0], arr[:, 3] - arr[:, 1]])
            min_dist = float(np.median(box_sizes)) * merge_factor
            labels = _merge_close_clusters(centers, labels, min_dist)
        labels = _compact_labels(labels)

        for cluster_id in range(labels.max() + 1):
            mask = labels == cluster_id
            if not mask.any():
                continue
            cluster_boxes = arr[mask]
            box = [
                round(float(np.median(cluster_boxes[:, 0])), 1),
                round(float(np.median(cluster_boxes[:, 1])), 1),
                round(float(np.median(cluster_boxes[:, 2])), 1),
                round(float(np.median(cluster_boxes[:, 3])), 1),
            ]
            stabilized.append({
                "class_id": cid,
                "class_name": name,
                "point_index": int(cluster_id),
                "box": box,
                "n_frames": int(mask.sum()),
            })
    return stabilized
