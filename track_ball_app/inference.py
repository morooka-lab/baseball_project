"""
TrackNet 推論 + 軌跡オーバーレイ描画

学習済み TrackNet (train.py で学習した重み) を使って動画中のボール座標を
1フレームずつ推定し、直近の検出点をつないだ軌跡を映像に重ねて出力する。
"""

from collections import deque

import cv2
import torch
import torchvision.transforms as transforms

from models.tracknet import TrackNet

SQ = 3  # TrackNet は連続する3フレームをまとめて1サンプルとして推論する

TRAIL_LEN = 18          # 軌跡としてつなぐ直近の検出点数
TRAIL_GAP_RESET = 6     # これ以上フレームが空いたら別の投球とみなし軌跡をリセット
TRAJ_COLOR = (246, 130, 59)   # BGR (Tailwind blue-500 相当)
BALL_COLOR = (153, 211, 52)   # BGR (Tailwind emerald-400 相当)


def get_shuttle_position(img):
    """2値化されたヒートマップ (uint8, 2次元) からボール中心座標を求める。"""
    if img.max() <= 0:
        # (visible, cx, cy)
        return (0, 0, 0)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in contours]

    max_idx = 0
    max_area = rects[0][2] * rects[0][3]
    for i, r in enumerate(rects):
        area = r[2] * r[3]
        if area > max_area:
            max_idx = i
            max_area = area

    x, y, bw, bh = rects[max_idx]
    cx, cy = int(x + bw / 2), int(y + bh / 2)

    # (visible, cx, cy)
    return (1, cx, cy)


def load_model(weights_path, device="cpu"):
    model = TrackNet().to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _open_writer(output_path, fps, size):
    for fourcc_str in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*fourcc_str), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("VideoWriter の初期化に失敗しました")


def _draw_trail(frame, trail):
    if len(trail) < 2:
        return frame

    overlay = frame.copy()
    n = len(trail)
    for i in range(1, n):
        weight = i / n
        thickness = max(2, int(2 + 4 * weight))
        color = tuple(int(c * (0.35 + 0.65 * weight)) for c in TRAJ_COLOR)
        cv2.line(overlay, trail[i - 1], trail[i], color, thickness, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)


def process_video(input_path, output_path, model, device, imgsz, progress_cb=None):
    """動画にボールの軌跡を重ねて出力する。

    progress_cb(processed_frames, total_frames) は数フレームごとに呼ばれる。
    """
    vid_cap = cv2.VideoCapture(str(input_path))
    if not vid_cap.isOpened():
        raise RuntimeError("動画を開けませんでした")

    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid_cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = _open_writer(output_path, fps, (w, h))

    trail = deque(maxlen=TRAIL_LEN)
    frames_since_seen = TRAIL_GAP_RESET + 1
    count = 0
    detected_count = 0

    try:
        with torch.inference_mode():
            while True:
                imgs = []
                for _ in range(SQ):
                    ret, img = vid_cap.read()
                    if not ret:
                        break
                    imgs.append(img)

                if len(imgs) < SQ:
                    # 端数フレームは推論せずそのまま書き出し、動画長を保つ
                    for img in imgs:
                        writer.write(img)
                        count += 1
                    break

                imgs_torch = []
                for img in imgs:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    t = transforms.ToTensor()(rgb).to(device)
                    t = transforms.functional.resize(t, imgsz, antialias=True)
                    imgs_torch.append(t)
                batch = torch.cat(imgs_torch, dim=0).unsqueeze(0)

                preds = model(batch)[0].detach().cpu().numpy()
                y_preds = (preds > 0.5).astype("uint8") * 255

                for i in range(SQ):
                    visible, cx_pred, cy_pred = get_shuttle_position(y_preds[i])
                    frame = imgs[i]

                    if visible:
                        cx = int(cx_pred * w / imgsz[1])
                        cy = int(cy_pred * h / imgsz[0])
                        if frames_since_seen > TRAIL_GAP_RESET:
                            trail.clear()
                        trail.append((cx, cy))
                        frames_since_seen = 0
                        detected_count += 1
                    else:
                        frames_since_seen += 1

                    frame = _draw_trail(frame, trail)
                    if trail:
                        cv2.circle(frame, trail[-1], 7, BALL_COLOR, -1, cv2.LINE_AA)
                        cv2.circle(frame, trail[-1], 7, (255, 255, 255), 1, cv2.LINE_AA)

                    writer.write(frame)
                    count += 1

                if progress_cb:
                    progress_cb(count, total_frames)
    finally:
        writer.release()
        vid_cap.release()

    if progress_cb:
        progress_cb(count, total_frames if total_frames > 0 else count)

    return {
        "total_frames": count,
        "detected_frames": detected_count,
        "detection_rate": round(detected_count / count, 4) if count else 0.0,
    }
