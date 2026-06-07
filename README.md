# track_ball 環境構築 (pyenv)

pyenv を使って Python 環境を構築し、track_ball を動かせる状態にするまでの手順です。

---

## 1. 必要なシステムライブラリのインストール

Python のビルドに必要な依存パッケージを apt でインストールします。

```bash
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils \
  tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

---

## 2. pyenv のインストール

公式の自動インストールスクリプトを実行します。

```bash
curl https://pyenv.run | bash
```

次に、シェルの設定ファイル（`~/.bashrc` または `~/.zshrc`）に以下を追記します。

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```

設定を反映させます。

```bash
source ~/.bashrc
```

インストールを確認します。

```bash
pyenv --version
```

---

## 3. Python のインストール

```bash
# Python 3.11.8 をインストール
pyenv install 3.11.8

# インストール済みのバージョン一覧を確認
pyenv versions
```

プロジェクトディレクトリに移動し、使用するバージョンを固定します（`.python-version` ファイルが作成されます）。

```bash
cd /path/to/baseball_project
pyenv local 3.11.8
```

バージョンが切り替わっていることを確認します。

```bash
python --version
# → Python 3.11.8
```

---

## 4. 仮想環境の作成と有効化

```bash
python -m venv .venv
source .venv/bin/activate
```

プロンプトの先頭に `(.venv)` が表示されれば有効化されています。

---

## 5. 依存パッケージのインストール

```bash
pip install -r track_ball_requirements.txt
```

> **注意:** `requirements.txt` には G25対応の PyTorch（`torch==2.12.0+cu130`）が含まれています。
> G25以外の環境では、別途 CPU 版 PyTorch に差し替えてください。

インストール後、動作確認します。

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

## 6. 仮想環境の無効化

作業が終わったら以下で仮想環境を抜けます。

```bash
deactivate
```
