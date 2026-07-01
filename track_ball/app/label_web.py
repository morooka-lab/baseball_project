"""
Web版ボールラベリングツール (run_all_labels の Web化)

ブラウザからフレームごとにボールをクリックしてラベリングするツール。
videosと同じディレクトリ構造でCSVラベルを保存します。

使い方:
    python label_web.py
    python label_web.py --videos-dir /data2/baseball_data/videos --labels-dir /data2/baseball_data/labels
    python label_web.py --port 5002
"""

import argparse
import base64
import json as _json
import os
import threading
from pathlib import Path

import cv2
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

# ── Config ──────────────────────────────────────────────────────────────────

app = Flask(__name__)

VIDEOS_DIR = "/data2/baseball_data/videos"
LABELS_DIR = "/data2/baseball_data/labels"

# ── VideoCapture pool ────────────────────────────────────────────────────────

_caps: dict = {}
_cap_locks: dict = {}
_pool_lock = threading.Lock()


def _cap_and_lock(path: str):
    with _pool_lock:
        if path not in _caps:
            _caps[path] = cv2.VideoCapture(path)
            _cap_locks[path] = threading.Lock()
        return _caps[path], _cap_locks[path]


def read_frame_jpeg(video_path: str, frame_num: int):
    cap, lock = _cap_and_lock(video_path)
    with lock:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
    if not ret:
        return None
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def get_video_meta(video_path: str) -> dict:
    cap, lock = _cap_and_lock(video_path)
    with lock:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return {"frames": total, "fps": fps, "width": width, "height": height}


# ── Label path helpers ───────────────────────────────────────────────────────

def label_path_for(video_path: str) -> Path:
    """動画パスに対応するラベルCSVパスを返す (videosと同じ相対構造)。"""
    p = Path(video_path)
    try:
        rel = p.relative_to(VIDEOS_DIR)
    except ValueError:
        rel = Path(p.name)
    return Path(LABELS_DIR) / rel.with_suffix(".csv")


def load_labels(video_path: str, total_frames: int) -> list:
    # 全フレーム分のデフォルト配列を作成し、CSVの既存ラベルで上書きする
    result = [{"frame_num": i, "visible": 0, "x": 0.0, "y": 0.0} for i in range(total_frames)]
    csv_path = label_path_for(video_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            idx = int(row["frame_num"])
            if 0 <= idx < total_frames:
                result[idx] = {
                    "frame_num": idx,
                    "visible": int(row["visible"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                }
    return result


def save_labels_csv(video_path: str, data: list) -> Path:
    csv_path = label_path_for(video_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data).sort_values("frame_num", ignore_index=True)
    df.to_csv(csv_path, index=False)
    return csv_path


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("label.html")


@app.route("/files")
def list_files():
    """videos_dir 以下の全mp4を再帰的に列挙し、ラベル済み状態を返す。"""
    root = Path(VIDEOS_DIR)
    if not root.exists():
        return jsonify([])
    files = []
    for f in sorted(root.rglob("*.mp4")):
        csv = label_path_for(str(f))
        files.append({
            "path": str(f),
            "rel": str(f.relative_to(root)),
            "labeled": csv.exists(),
        })
    return jsonify(files)


@app.route("/meta")
def meta():
    path = request.args.get("video", "")
    if not path or not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    return jsonify(get_video_meta(path))


@app.route("/frame")
def frame():
    """フレームNをJPEGで返す。"""
    path = request.args.get("video", "")
    n = int(request.args.get("n", 0))
    if not path or not os.path.exists(path):
        return "not found", 404
    jpg = read_frame_jpeg(path, n)
    if jpg is None:
        return "frame not found", 404
    return Response(jpg, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.route("/labels")
def get_labels():
    path = request.args.get("video", "")
    if not path or not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    m = get_video_meta(path)
    data = load_labels(path, m["frames"])
    return jsonify({"data": data, "csv_path": str(label_path_for(path))})


@app.route("/stream_frames")
def stream_frames():
    """全フレームをJPEG+base64でSSEストリーミング（シーク不要の順次読み出し）。"""
    path = request.args.get("video", "")
    if not path or not os.path.exists(path):
        return "not found", 404

    def generate():
        cap = cv2.VideoCapture(path)
        try:
            i = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                yield f"data: {_json.dumps({'i': i, 'd': b64})}\n\n"
                i += 1
        finally:
            cap.release()
        yield f'data: {{"done":true,"actual":{i}}}\n\n'

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/save", methods=["POST"])
def save():
    body = request.get_json()
    video_path = body.get("video_path", "")
    data = body.get("data", [])
    if not video_path:
        return jsonify({"error": "no video_path"}), 400
    csv_path = save_labels_csv(video_path, data)
    print(f"保存: {csv_path}")
    return jsonify({"csv_path": str(csv_path)})



# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Web版ボールラベリングツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例:\n"
            "  python label_web.py\n"
            "  python label_web.py --videos-dir /data2/baseball_data/videos "
            "--labels-dir /data2/baseball_data/labels --port 5002"
        ),
    )
    p.add_argument("--videos-dir", default=VIDEOS_DIR,
                   help=f"動画ルートディレクトリ (default: {VIDEOS_DIR})")
    p.add_argument("--labels-dir", default=LABELS_DIR,
                   help=f"ラベル保存ルートディレクトリ (default: {LABELS_DIR})")
    p.add_argument("--port", type=int, default=5002,
                   help="ポート番号 (default: 5002)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    VIDEOS_DIR = args.videos_dir
    LABELS_DIR = args.labels_dir

    print(f"動画ディレクトリ : {VIDEOS_DIR}")
    print(f"ラベルディレクトリ: {LABELS_DIR}")
    print(f"ブラウザで http://localhost:{args.port} を開いてください")
    print("(VS Code Remote は自動でポートをフォワードします)")
    print("終了: Ctrl+C\n")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
