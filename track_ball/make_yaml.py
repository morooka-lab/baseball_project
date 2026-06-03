import os


def generate_yaml():
    # 画像フォルダのパス
    images_dir = r"C:\Users\ytake\TrackNet_Baseball\custom_dataset\match\images"
    yaml_path = r"C:\Users\ytake\TrackNet_Baseball\data\baseball.yaml"

    # clip_ から始まるフォルダの一覧を取得して並び替え
    clip_folders = sorted(
        [
            f
            for f in os.listdir(images_dir)
            if os.path.isdir(os.path.join(images_dir, f)) and f.startswith("clip_")
        ]
    )

    if not clip_folders:
        print("エラー: images フォルダの中に clip_XXX フォルダが見つかりません。")
        return

    # YAMLファイルの中身を作成
    yaml_content = "path: ./custom_dataset/match\n\n"

    # 学習用（train）に全クリップを追加
    yaml_content += "train:\n"
    for folder in clip_folders:
        yaml_content += f"  - images/{folder}\n"

    # 検証用（val）にも全クリップを追加
    yaml_content += "\nval:\n"
    for folder in clip_folders:
        yaml_content += f"  - images/{folder}\n"

    # クラス設定
    yaml_content += "\nnc: 1\n"
    yaml_content += "names: ['ball']\n"

    # ファイルに書き込み
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(
        f"✅ 成功！合計 {len(clip_folders)} 個のクリップを登録した baseball.yaml を作成しました！"
    )


if __name__ == "__main__":
    generate_yaml()
