"""
ウェブアプリ ルーター

各ツールへのリクエストを DispatcherMiddleware でルーティングする。

起動:
    python main.py [--port PORT]

ブラウザ:
    http://localhost:5000

ルーティング:
    /              - トップページ（ツール一覧）
    /animate/      - 3D ボール軌道アニメーション
    /label/        - ボールラベリングツール
    /timestamps/   - タイムスタンプ取得ツール
    /predict/      - ボール検出 API（別途 init_model() が必要）
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template_string
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

from animate_ball_track_app import app as animate_app
from label_web import app as label_app
from get_timestamps_web import app as timestamps_app
from app import app as predict_app

# ── トップページ ───────────────────────────────────────────────────────────────

_LANDING_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>野球ボール解析ツール</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #090b14;
    color: #ccd;
    font-family: 'Segoe UI', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
  }
  h1 { font-size: 1.7em; color: #aac; letter-spacing: 0.04em; }
  .subtitle { color: #334; font-size: 0.88em; margin-bottom: 28px; }
  .tools {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
    max-width: 820px;
    width: 100%;
    padding: 0 20px;
  }
  .card {
    background: rgba(20, 30, 60, 0.55);
    border: 1px solid #1e2a40;
    border-radius: 12px;
    padding: 26px 24px;
    text-decoration: none;
    color: inherit;
    display: flex;
    flex-direction: column;
    gap: 8px;
    transition: border-color .18s, background .18s;
  }
  .card:hover { border-color: #2a9; background: rgba(30, 55, 85, 0.65); }
  .icon  { font-size: 2em; }
  .name  { font-size: 1.05em; color: #dde; font-weight: bold; }
  .desc  { font-size: 0.81em; color: #556; line-height: 1.65; }
  .route { font-size: 0.73em; color: #2a7a5a; font-family: monospace; margin-top: 4px; }
</style>
</head>
<body>
  <h1>野球ボール解析ツール</h1>
  <p class="subtitle">使用するツールを選択してください</p>
  <div class="tools">
    <a href="/animate/" class="card">
      <div class="icon">🎯</div>
      <div class="name">3D 軌道アニメーション</div>
      <div class="desc">DLT カメラキャリブレーションで 3D 軌道を復元し、WebGL でインタラクティブに可視化する</div>
      <div class="route">/animate/</div>
    </a>
    <a href="/label/" class="card">
      <div class="icon">🏷️</div>
      <div class="name">ボールラベリング</div>
      <div class="desc">フレームをブラウザ上でクリックしてボール座標をラベリングし、CSV に保存する</div>
      <div class="route">/label/</div>
    </a>
    <a href="/timestamps/" class="card">
      <div class="icon">✂️</div>
      <div class="name">タイムスタンプ取得</div>
      <div class="desc">動画を再生しながら投球シーンの開始・終了を記録し、ffmpeg でクリップを自動生成する</div>
      <div class="route">/timestamps/</div>
    </a>
    <a href="/predict/" class="card">
      <div class="icon">🤖</div>
      <div class="name">ボール検出 API</div>
      <div class="desc">TrackNet モデルでボール座標を検出する REST API。起動時に --weights でモデルを指定する</div>
      <div class="route">/predict/</div>
    </a>
  </div>
</body>
</html>"""

root_app = Flask(__name__)


@root_app.route("/")
def index():
    return render_template_string(_LANDING_HTML)


# ── DispatcherMiddleware でルーティング ────────────────────────────────────────

application = DispatcherMiddleware(root_app, {
    "/animate":    animate_app,
    "/label":      label_app,
    "/timestamps": timestamps_app,
    "/predict":    predict_app,
})


# ── エントリポイント ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="野球ボール解析ツール ルーター",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例:\n"
            "  python main.py\n"
            "  python main.py --port 8080\n"
            "  python main.py --weights /path/to/best.pt  # 検出APIも有効にする場合"
        ),
    )
    p.add_argument("--port", type=int, default=5000, help="ポート番号 (default: 5000)")
    p.add_argument("--weights", type=str, default="", help="TrackNet モデルの重みファイルパス（省略時は検出APIが無効）")
    p.add_argument("--device", type=str, default="cuda", help="PyTorch デバイス (default: cuda)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.weights:
        from app import init_model
        init_model(args.weights, device=args.device)

    import socket
    hostname = socket.gethostname()
    try:
        server_ip = socket.gethostbyname(hostname)
    except Exception:
        server_ip = "（IP取得失敗）"
    print(f"起動中:")
    print(f"  このサーバー上 : http://localhost:{args.port}")
    print(f"  他ユーザー用  : http://{server_ip}:{args.port}")
    print("  /animate/    - 3D 軌道アニメーション")
    print("  /label/      - ボールラベリング")
    print("  /timestamps/ - タイムスタンプ取得")
    print("  /predict/    - ボール検出 API")
    print("終了: Ctrl+C\n")

    run_simple("0.0.0.0", args.port, application, use_reloader=False, use_debugger=False, threaded=True)
