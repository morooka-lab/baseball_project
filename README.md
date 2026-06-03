# TrackNet Baseball Project ⚾

このプロジェクトは、バドミントン/テニス用の飛翔体追跡ネットワークである [TrackNetV2](https://github.com/aodn/TrackNetV2-pytorch) をベースに、**野球の投球軌跡（ボール）を自動で追跡・描画できるようにカスタマイズしたリポジトリ**です。

> **⚠️ 注意事項 (著作権保護の観点から)**
> 本リポジトリには、RKB等の著作権保護対象となる動画データ、画像データ、およびそれらの推論結果は一切含めていません。コードファイルのみの公開としています。
> データセットを手元で構築する手順については後述の「Dataset Preparation」を参照してください。

---

## 🛠 Environment Setup (環境構築)

学習と推論に必要なライブラリをインストールします。
ベースとなる PyTorch 環境に加えて、以下のパッケージが必要です。

```shell
# 仮想環境のアクティベート (例: conda環境)
conda activate tracknet_env

# 不足している必須パッケージのインストール
pip install tqdm tensorboardX torchsummary

# ⚠️ albumentations は仕様変更によるエラーを防ぐため、古い安定版を指定
pip uninstall -y albumentations
pip install albumentations==1.3.1
```

---

## 📁 Dataset Preparation (データ準備)

本プロジェクトでは、短いクリップ動画（`.mp4`）から画像フレームとCSVラベルを生成し、学習に使用します。

### 1. フォルダ構成
ローカル（あなたのPC上）で、以下のようなフォルダ構成になるようにデータを配置してください。（※ `custom_dataset/` や `test_videos/` は `.gitignore` でアップロード除外にしています）

```text
TrackNet_Baseball/
  ├── custom_dataset/
  │   └── match/
  │       ├── images/       (clip_000, clip_001 ... 各フレーム画像)
  │       ├── labels/       (clip_000.csv, clip_001.csv ... 座標データ)
  │       └── videos/clips/ (アノテーション用の短いmp4クリップ)
  ├── data/                 (baseball.yaml などの設定ファイル)
  ├── test_videos/          (テスト用の新しい投球動画)
  └── runs/                 (学習結果や推論結果の出力先)
```

### 2. アノテーション（ラベル付け）
野球のボール追跡に特化したラベル付けを行います。

* **使用ツール:** `tools/label_tool.py` (自動巡回スクリプト `run_all_labels.py` を使用すると便利です)
* **操作方法:**
  * `左クリック`: ボールが見える（VISIBLE） - ボールの先端をクリック
  * `右ダブルクリック`: ボールが見えない（HIDDEN） - ミットに収まった後や投球前
  * `n` キー: 次のフレームへ進む
  * `q` キー: 保存して終了

### 3. YAML設定ファイルの自動生成
手作業による記載ミスを防ぐため、用意した画像フォルダ群から自動で `data/baseball.yaml` を生成します。
プロジェクトのルートディレクトリで以下のスクリプトを実行してください。

```shell
python make_yaml.py
```

---

## 🚀 Training (学習)

準備したデータセットとYAMLファイルを使用して、TrackNetに野球ボールの軌跡を学習させます。

```shell
# 学習の実行 (バッチサイズやエポック数はマシンスペックに応じて調整してください)
python train.py --data data/baseball.yaml --epochs 30 --batch-size 4
```

学習が完了すると、`runs/train/best.pt` に最も精度の高かったAIの重みデータ（モデル）が保存されます。

---

## 🎯 Inference (推論・テスト)

学習したモデル (`best.pt`) を使用して、未知の投球動画にボールの軌跡を描画します。

> **Note:**
> サーバー環境やWindows等での OpenCV ウィンドウ表示エラー（`cvWaitKey` エラー等）を回避するため、`detect.py` の GUI（`cv2.imshow` 等）は無効化・修正されています。実行はバックグラウンドで行われ、完了後に動画が保存されます。

```shell
# テスト動画への軌跡描画
python detect.py --source test_videos/okamoto.mp4 --weights runs/train/best.pt
```

推論が完了すると、`runs/detect/` フォルダ内に、赤い点でボールの軌跡が描画されたテスト結果の `.mp4` 動画が出力されます。

---

## 📚 References
* TrackNetV2 Paper: [TrackNetV2: Efficient Shuttlecock Tracking Network](https://arxiv.org/abs/2012.05996)
* Base Repository: [TrackNetV2-pytorch](https://github.com/aodn/TrackNetV2-pytorch)