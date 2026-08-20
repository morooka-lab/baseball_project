train.py で学習した TrackNet の重みファイル（例: `best.pt`）をこのディレクトリに置いてください。

- ファイル名を指定しない場合、`app.py` はデフォルトで `weight/best.pt` を探します。
- `best.pt` が無くても、このディレクトリに `.pt` ファイルが1つだけ置かれていればそれを自動的に使用します。
- 別の場所や別名の重みを使う場合は `python app.py --weights /path/to/your.pt` のように指定してください。
