import os

import cv2


def extract_pitching_clips(video_path, output_base_dir, clip_timestamps):
    """
    動画から指定された複数区間の連続フレームを切り出して保存する

    :param video_path: 元のmp4動画のパス
    :param output_base_dir: 画像を保存する親フォルダ（例: custom_dataset/match/images）
    :param clip_timestamps: 切り出す区間のリスト。[(開始秒, 終了秒), ...] の形式
    """

    # 動画を読み込む
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイル {video_path} を開けませんでした。")
        return

    # 動画のFPSを取得（29.97等の正確な値を取得します）
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"動画のFPS: {fps:.2f}")

    # 各クリップ区間ごとに処理
    for clip_idx, (start_sec, end_sec) in enumerate(clip_timestamps):
        # クリップごとの保存先フォルダを作成 (例: clip_001, clip_002...)
        clip_dir = os.path.join(output_base_dir, f"clip_{clip_idx:03d}")
        os.makedirs(clip_dir, exist_ok=True)
        print(
            f"\n[{clip_idx:03d}] {start_sec}秒 〜 {end_sec}秒 の切り出しを開始します..."
        )
        print(f"保存先: {clip_dir}")

        # 秒数をフレーム番号に変換
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        # 読み込み開始位置をセット
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = 0
        current_frame = start_frame

        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                print("フレームの読み込みが終了したか、エラーが発生しました。")
                break

            # 画像を保存 (TrackNetの仕様に合わせて 0.jpg, 1.jpg, 2.jpg... と連番にする)
            save_path = os.path.join(clip_dir, f"{frame_count}.jpg")
            cv2.imwrite(save_path, frame)

            frame_count += 1
            current_frame += 1

        print(f"完了: {frame_count}枚の画像を保存しました。")

    cap.release()
    print("\nすべてのクリップの切り出しが完了しました！")


# ==========================================
# 実行部分（ここをご自身の環境に合わせて変更してください）
# ==========================================
if __name__ == "__main__":
    # 1. 処理したい動画ファイルのパス (videosフォルダに入れたファイル名に合わせる)
    VIDEO_FILE = "C:\\Users\\ytake\\TrackNet_Baseball\\custom_dataset\\match\\videos\\senga_pitch.mp4"

    # 2. 保存先の親フォルダパス (先ほど作ったimagesフォルダ)
    OUTPUT_DIR = "C:\\Users\\ytake\\TrackNet_Baseball\\custom_dataset\\match\\images"

    # 3. メモしたタイムスタンプ（開始秒, 終了秒）
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

    extract_pitching_clips(VIDEO_FILE, OUTPUT_DIR, TIMESTAMPS)
