import os

import cv2


def create_mp4_clips(video_path, output_dir, clip_timestamps):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(output_dir, exist_ok=True)

    for clip_idx, (start_sec, end_sec) in enumerate(clip_timestamps):
        output_path = os.path.join(output_dir, f"clip_{clip_idx:03d}.mp4")
        # mp4vコーデックで書き出し
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1

        out.release()
        print(f"作成完了: {output_path}")

    cap.release()


if __name__ == "__main__":
    VIDEO_FILE = r"C:\Users\ytake\TrackNet_Baseball\custom_dataset\match\videos\senga_pitch.mp4"  # 元の動画
    OUTPUT_DIR = r"C:\Users\ytake\TrackNet_Baseball\custom_dataset\match\videos\clips"  # クリップの保存先
    TIMESTAMPS = [
        (7.74, 8.48),
        (15.22, 16.08),
        (24.69, 25.16),
        (35.30, 36.04),
        (42.08, 42.78),
        (50.45, 51.22),
        (59.46, 60.29),
        (67.33, 68.20),
        (74.97, 75.74),
        (80.91, 81.82),
        (87.29, 87.99),
        (96.20, 97.00),
        (101.53, 102.54),
        (108.91, 109.64),
        (117.78, 118.52),
        (126.06, 127.09),
        (132.60, 133.47),
        (140.17, 140.81),
        (149.58, 150.28),
        (156.16, 157.02),
        (164.10, 165.03),
        (167.93, 168.60),
        (176.31, 177.18),
        (183.95, 184.75),
        (189.19, 189.86),
        (193.39, 194.19),
        (199.13, 199.93),
        (205.17, 205.91),
        (211.24, 212.18),
        (216.35, 217.25),
        (220.99, 221.79),
        (225.86, 226.69),
        (234.33, 235.30),
        (237.67, 238.37),
        (242.78, 243.44),
        (245.71, 246.51),
    ]
    create_mp4_clips(VIDEO_FILE, OUTPUT_DIR, TIMESTAMPS)
