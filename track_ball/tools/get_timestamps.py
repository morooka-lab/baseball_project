import cv2


def record_timestamps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("エラー: 動画を読み込めませんでした。パスを確認してください。")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamps = []
    current_start = None
    paused = False

    # マウス操作のコールバック関数
    def mouse_event(event, x, y, flags, param):
        nonlocal current_start, timestamps

        # 現在のフレーム位置から秒数を計算
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = current_frame / fps

        # 【右クリック】: 開始時間を記録
        if event == cv2.EVENT_RBUTTONDOWN:
            current_start = current_time
            print(f"▶ 開始時間をセット: {current_time:.2f}秒")

        # 【左クリック】: 終了時間を記録
        elif event == cv2.EVENT_LBUTTONDOWN:
            if current_start is not None:
                # 開始時刻より前の時間がクリックされた場合のフェイルセーフ
                if current_time <= current_start:
                    print("⚠️ エラー: 終了時間は開始時間より後にしてください。")
                    return
                print(f"⏹ 終了時間をセット: {current_time:.2f}秒")
                timestamps.append((current_start, current_time))
                print(f"  => 記録完了: ({current_start:.2f}, {current_time:.2f})")
                current_start = None
            else:
                print("⚠️ 先に【右クリック】で開始時間をセットしてください。")

    cv2.namedWindow("Video Player")
    cv2.setMouseCallback("Video Player", mouse_event)

    print("=========================================")
    print("🎬 動画アノテーションツール（タイムスタンプ取得）")
    print("=========================================")
    print("【操作方法】")
    print("🖱️ 右クリック : 投球シーンの開始時間を記録")
    print("🖱️ 左クリック : 投球シーンの終了時間を記録")
    print("⌨️ スペースキー: 再生 / 一時停止")
    print("⌨️ D キー     : 1フレーム進む (一時停止中のみ)")
    print("⌨️ A キー     : 1フレーム戻る (一時停止中のみ)")
    print("⌨️ Q キー     : ツールを終了して結果を出力")
    print("=========================================\n")

    ret, frame = cap.read()

    while True:
        if not ret:
            print("動画の終端に達しました。")
            break

        display_frame = frame.copy()
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = current_frame / fps

        # 画面上に現在のタイムスタンプとステータスを表示
        cv2.putText(
            display_frame,
            f"Time: {current_time:.2f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        if current_start is not None:
            cv2.putText(
                display_frame,
                f"Start: {current_start:.2f}s - Left click to END",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
            )

        cv2.imshow("Video Player", display_frame)

        # キー入力制御 (約30FPSに合わせて30ms待機)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
            status = "一時停止" if paused else "再生中"
            print(f"[{status}]")
        elif key == ord("d") and paused:
            # 1フレーム進む
            ret, frame = cap.read()
        elif key == ord("a") and paused:
            # 1フレーム戻る（現在の位置から2つ戻って1つ読み込む）
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 2))
            ret, frame = cap.read()

        if not paused:
            ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    # 最終結果をコンソールに出力
    print("\n\n" + "=" * 50)
    print(
        "✅ 以下のコードをコピーして、extract_clips.py の TIMESTAMPS に貼り付けてください。"
    )
    print("=" * 50)
    print("TIMESTAMPS = [")
    for start, end in timestamps:
        print(f"    ({start:.2f}, {end:.2f}),")
    print("]")
    print("=" * 50)


# ==========================================
# 実行部分
# ==========================================
if __name__ == "__main__":
    # 解析したい動画のパスを指定してください
    VIDEO_FILE = "C:\\Users\\ytake\\TrackNet_Baseball\\custom_dataset\\match\\videos\\senga_pitch.mp4"
    record_timestamps(VIDEO_FILE)
