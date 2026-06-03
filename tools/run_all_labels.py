import glob
import os
import subprocess
import sys


def main():
    # 1. パスの設定
    clips_dir = r"C:\Users\ytake\TrackNet_Baseball\custom_dataset\match\videos\clips"
    labels_dir = r"C:\Users\ytake\TrackNet_Baseball\custom_dataset\match\labels"

    # 2. label_tool.py の正確なパスを自動取得する
    # (この run_all_labels.py と同じフォルダにある label_tool.py を指定)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    label_tool_path = os.path.join(script_dir, "label_tool.py")

    # ファイルの存在チェック
    if not os.path.exists(label_tool_path):
        print("❌ エラー: label_tool.py が見つかりません。")
        print(f"探した場所: {label_tool_path}")
        return

    os.makedirs(labels_dir, exist_ok=True)
    video_files = sorted(glob.glob(os.path.join(clips_dir, "*.mp4")))

    if not video_files:
        print(f"❌ エラー: {clips_dir} に動画ファイルが見つかりません。")
        return

    print(f"合計 {len(video_files)} 個のクリップが見つかりました。\n")

    for video_path in video_files:
        file_name = os.path.basename(video_path)
        csv_name = file_name.replace(".mp4", ".csv")
        csv_path = os.path.join(labels_dir, csv_name)

        if os.path.exists(csv_path):
            print(f"⏩ {file_name} は既にラベルデータが存在するためスキップします。")
            continue

        print("=" * 50)
        print(f"▶ 現在処理中: {file_name}")
        print("=" * 50)

        # 3. 実行コマンドの強化
        # sys.executable を使うことで、現在動いているのと同じPython（仮想環境）を確実に使う
        command = [sys.executable, label_tool_path, video_path, "--csv_dir", labels_dir]

        try:
            # check=True を入れることで、label_tool.py がエラー終了した場合に処理を止めて原因を表示する
            subprocess.run(command, check=True)
            print(f"✅ {file_name} の作業が完了しました。\n")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {file_name} の処理中にエラーが発生したため中断しました。")
            print(f"エラーコード: {e.returncode}")
            break  # エラーが出たら次のループに行かず、安全のために全体をストップ

    print("🎉 すべての処理が終了しました！")


if __name__ == "__main__":
    main()
