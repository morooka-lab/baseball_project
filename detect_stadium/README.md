# ３次元軌道の計算
球場の情報を用いて、2次元のボール追跡結果から３次元のボール軌道を計算する。
３次元の軌道を描くために必要な情報を以下にまとめる。

|検出する情報|具体的な情報|
|---|---|
|球場情報|ピッチャープレート・ホームプレート・バッターボックス|
|人|投手・捕手・バッター|

これらの情報をもとにして、カメラパラメータを計算する。

## 球場情報の検出
映像から各情報を取得するプログラムを取得する。可能なら1つのモデルですべて検出したい。
中継映像からアノテーションをする。

### データセットの構築
||データセットのパス|
|---|---|
|root|/data2/baseball_data/detect_dataset/stadium|
|映像|/data2/baseball_data/detect_dataset/videos|
|ラベル|/data2/baseball_data/detect_dataset/stadium/labels|

## 投手・捕手・バッターの検出
映像から各人物を検出するプログラムを実装する。1つのモデルで3人を検出したい。
中継映像からアノテーションをする。

### データセットの構築
||データセットのパス|
|---|---|
|root|/data2/baseball_data/detect_dataset/player|
|映像|/data2/baseball_data/detect_dataset/videos|
|ラベル|/data2/baseball_data/detect_dataset/player/labels|

---

## 実装方針

track_ball/app のWebアノテーションツールを参考に、球場情報検出・選手検出のどちらも
「矩形(bbox)を複数クラスでアノテーション → 学習 → 未ラベル動画へ推論 →
自信度の低いフレームを追加アノテーション → 精度検証 → 最終結果を保存」
という共通のパイプラインで実装している。

### モデル・ライセンスについて
商用利用を前提に、検出モデルは **torchvision の SSDlite320-MobileNetV3-Large** を使用する。

- 検出ヘッド(分類・回帰)は完全にスクラッチ学習する（既存の学習済み検出モデルは一切流用しない）
- バックボーン(MobileNetV3)のみ、ImageNetの事前学習重み（torchvision公式配布、**BSD-3ライセンス**）を再利用する
- torchvision / PyTorch 本体も BSD-3 ライセンスであり、Ultralytics YOLO(AGPL-3.0)のような
  商用クローズドソース利用時のライセンス制約を受けない

### ディレクトリ構成
```
detect_stadium/
  .venv/                  仮想環境 (Blackwell GPU向けにcu128でtorch/torchvisionを導入済み)
  data/
    classes.yaml           タスクごとのクラス定義・データパス設定
  app/
    bbox_label_web.py      矩形アノテーションWebツール
    templates/label.html
    static/css/label.css
    static/js/label.js
  models/
    detector.py             SSDliteモデルの構築・読み込み
  utils/
    label_store.py          ラベルJSON(動画+フレーム+bbox)のI/O
    video_io.py              動画フレームの堅牢な読み出し(VFR対策)
    detection_dataset.py     torchvision検出モデル用Dataset
    map_eval.py               mAP計算(VOCスタイル、pycocotools不要)
  tools/
    select_uncertain_frames.py  低確信度フレーム抽出(能動学習)
    finalize_results.py          人手アノテーション優先で最終結果をマージ・保存
  train.py                  学習
  val.py                    mAP精度検証
  infer.py                  未ラベル動画への推論
```

### クラス定義 (`data/classes.yaml`)
```yaml
stadium:
  classes: [pitcher_plate, home_plate, batter_box_right, batter_box_left]
player:
  classes: [pitcher, catcher, batter]
```
アノテーションツール・学習・推論・評価すべてがこのファイルを共通で参照する。

---

## 使い方 (ワークフロー)

### 0. 環境構築
```bash
cd detect_stadium
python3 -m venv .venv
source .venv/bin/activate
# Blackwell(RTX PRO 6000等, sm_120)のGPUを使う場合はcu128インデックスを使う
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install -r requirements.txt
```

動画は `/data2/baseball_data/detect_dataset/videos` 以下にある既存のクリップ群を
stadium・playerの両タスクで共通して使う（`data/classes.yaml` の `videos_dir`）。

### 1. アノテーション (bbox_label_web.py)
```bash
python app/bbox_label_web.py --task stadium               # http://localhost:5010
python app/bbox_label_web.py --task player  --port 5011   # 選手検出は別ポートで同時起動可
```
- ドラッグで矩形を新規作成、クリックで選択、ハンドルをドラッグしてリサイズ、ボディをドラッグして移動
- `1`-`9` キーでクラス切替（box選択中はそのboxのクラスを変更）
- `Delete` で選択box削除、`対象物なし [E]` で「レビュー済みだが対象物なし」を記録（ハードネガティブとして学習に使われる）
- 間隔(秒)を指定して `<<` / `>>` でサンプリング移動、`フレーム#` 欄で直接ジャンプも可能（能動学習の候補フレームへ飛ぶ時に便利）
- サイドバーの「アノテーション済み N / 50 フレーム」で最初の目標(50件)への進捗を確認できる

ラベルは動画ごとに `labels_dir` 配下へJSON保存される（`videos_dir` と同じ相対構造）。

### 2. 学習
```bash
python train.py --task stadium --epochs 80 --batch-size 4
```
`runs/train/stadium/exp/{best.pt,last.pt,classes.json}` に保存される。
50件前後の少数データを想定し、`RandomPhotometricDistort` / `RandomZoomOut` / `RandomIoUCrop` /
`RandomHorizontalFlip` によるaugmentationを標準で有効にしている。

### 3. 未ラベルデータへの推論
```bash
python infer.py --task stadium --weights runs/train/stadium/exp/best.pt \
    --source /data2/baseball_data/videos --conf-thres 0.05 \
    --out-dir runs/detect/stadium/exp
```
`runs/detect/stadium/exp/predictions/*.json` にフレームごとのbbox+confidenceが保存される
（`--save-video` でbbox描画済みmp4も出力可能）。

### 4. 低確信度フレームの追加アノテーション（能動学習ループ）
```bash
python tools/select_uncertain_frames.py --task stadium \
    --predictions-dir runs/detect/stadium/exp/predictions --top-k 50
```
検出0件のフレーム→confidenceが低いフレームの順に候補を抽出する
（既にアノテーション済みのフレームは自動的に除外）。
出力された `video` / `frame` を使い、`bbox_label_web.py` の「フレーム#」欄でジャンプして
アノテーションを追加し、`train.py` を再実行して精度を上げる。このループを繰り返す。

### 5. 精度検証・最終結果の保存
```bash
python val.py --task stadium --weights runs/train/stadium/exp/best.pt
```
validationセット(学習時と同じ分割)でクラスごとのAP・mAPを算出する。
十分な精度が出たら、人手アノテーションを優先しつつモデル推論で残りを埋めた
「最終結果」を動画と紐づけて保存する。
```bash
python tools/finalize_results.py --task stadium \
    --predictions-dir runs/detect/stadium/exp/predictions \
    --out-dir runs/final/stadium --save-video
```
`runs/final/stadium/results/*.json`(フレームごとのbbox、`source`が`human`/`model`のどちらかを示す)と、
`runs/final/stadium/videos/*.mp4`(緑=人手アノテーション、赤=モデル推論で描画)が出力される。
