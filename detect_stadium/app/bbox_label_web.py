"""
矩形(bbox)・座標点(point)アノテーションWebツール。

track_ball/app/label_web.py (1点クリックのボールラベリング)を参考に、
複数クラス・複数個のbbox/pointをフレームごとに描画してラベリングできるようにしたもの。
球場情報検出(stadium)・選手検出(player)のどちらのタスクでも
--task で切り替えて共通のツールとして使う。アノテーション形式(矩形 or 座標点)は
classes.yamlのタスクごとの annotation_type で決まる(現状: stadium=point, player=bbox)。

使い方:
    python bbox_label_web.py --task stadium
    python bbox_label_web.py --task player --port 5011
    python bbox_label_web.py --task stadium --videos-dir /data2/baseball_data/videos
"""

import argparse
import base64
import json
import sys
import threading
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.label_store import (
    count_annotated_frames,
    label_path_for,
    list_videos,
    load_tasks,
    load_video_labels,
    save_video_labels,
)
from utils.video_io import open_video, read_frame_bgr, stream_meta

app = Flask(__name__)

TASK = None          # utils.label_store.TaskConfig
VIDEOS_DIR = ""
LABELS_DIR = ""
TARGET_FRAMES = 50

# ── PyAVコンテナpool ─────────────────────────────────────────────────────────
# opencv-python内蔵FFmpegはインターレース素材のYUV->BGR変換に失敗し全フレームが
# 真っ黒になるため、フレーム読み出しはPyAV(av)経由で行う(utils/video_io.py参照)。

_caps: dict = {}
_cap_locks: dict = {}
_pool_lock = threading.Lock()


def _cap_and_lock(path: str):
    with _pool_lock:
        if path not in _caps:
            _caps[path] = open_video(path)
            _cap_locks[path] = threading.Lock()
        return _caps[path], _cap_locks[path]


def read_frame_jpeg(video_path: str, frame_num: int):
    container, lock = _cap_and_lock(video_path)
    with lock:
        frame = read_frame_bgr(container, frame_num)
    if frame is None:
        return None
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def get_video_meta(video_path: str) -> dict:
    container, lock = _cap_and_lock(video_path)
    with lock:
        return stream_meta(container.streams.video[0])


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "label.html",
        task_name=TASK.name,
        task_description=TASK.description,
        classes=TASK.classes,
        target_frames=TARGET_FRAMES,
        annotation_type=TASK.annotation_type,
    )


@app.route("/files")
def list_files():
    root = Path(VIDEOS_DIR)
    files = []
    for f in list_videos(VIDEOS_DIR):
        labels = load_video_labels(str(f), VIDEOS_DIR, LABELS_DIR)
        files.append({
            "path": str(f),
            "rel": str(f.relative_to(root)),
            "annotated_frames": len(labels),
        })
    return jsonify(files)


@app.route("/stats")
def stats():
    return jsonify({
        "total_annotated_frames": count_annotated_frames(LABELS_DIR),
        "target": TARGET_FRAMES,
    })


@app.route("/meta")
def meta():
    path = request.args.get("video", "")
    if not path or not Path(path).exists():
        return jsonify({"error": "not found"}), 404
    return jsonify(get_video_meta(path))


@app.route("/frame")
def frame():
    path = request.args.get("video", "")
    n = int(request.args.get("n", 0))
    if not path or not Path(path).exists():
        return "not found", 404
    jpg = read_frame_jpeg(path, n)
    if jpg is None:
        return "frame not found", 404
    return Response(jpg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.route("/stream_frames")
def stream_frames():
    """全フレームをJPEG+base64でSSEストリーミングする(シーク不要の順次読み出し)。

    フロント側でこれを受けてフレームをキャッシュしておくことで、動画選択後の
    フレーム間移動をシーク待ちなしで高速に表示できるようにする
    (track_ball/app/label_web.py のstream_framesと同じ方式)。
    """
    path = request.args.get("video", "")
    if not path or not Path(path).exists():
        return "not found", 404

    def generate():
        container = open_video(path)
        i = 0
        try:
            stream = container.streams.video[0]
            for av_frame in container.decode(stream):
                bgr = av_frame.to_ndarray(format="bgr24")
                _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                yield f"data: {json.dumps({'i': i, 'd': b64})}\n\n"
                i += 1
        finally:
            container.close()
        yield f'data: {json.dumps({"done": True, "actual": i})}\n\n'

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/labels")
def get_labels():
    path = request.args.get("video", "")
    if not path or not Path(path).exists():
        return jsonify({"error": "not found"}), 404
    frames = load_video_labels(path, VIDEOS_DIR, LABELS_DIR)
    return jsonify({
        "frames": frames,
        "classes": TASK.classes,
        "label_path": str(label_path_for(path, VIDEOS_DIR, LABELS_DIR)),
    })


@app.route("/save", methods=["POST"])
def save():
    body = request.get_json()
    video_path = body.get("video_path", "")
    frames = body.get("frames", {})
    if not video_path:
        return jsonify({"error": "no video_path"}), 400
    frames = {int(k): v for k, v in frames.items()}
    path = save_video_labels(video_path, VIDEOS_DIR, LABELS_DIR, TASK.classes, frames)
    print(f"保存: {path} ({len(frames)} フレーム)")
    return jsonify({"label_path": str(path), "annotated_frames": len(frames)})


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="矩形(bbox)アノテーションWebツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例:\n"
            "  python bbox_label_web.py --task stadium\n"
            "  python bbox_label_web.py --task player --port 5011\n"
        ),
    )
    p.add_argument("--task", required=True, choices=["stadium", "player"],
                    help="アノテーション対象タスク")
    p.add_argument("--classes-yaml", default=str(ROOT / "data" / "classes.yaml"),
                    help="タスク定義YAML (default: data/classes.yaml)")
    p.add_argument("--videos-dir", default=None, help="動画ルートディレクトリ (省略時はclasses.yamlの値)")
    p.add_argument("--labels-dir", default=None, help="ラベル保存ルートディレクトリ (省略時はclasses.yamlの値)")
    p.add_argument("--target", type=int, default=50, help="最初の目標アノテーション数 (default: 50)")
    p.add_argument("--port", type=int, default=5010, help="ポート番号 (default: 5010)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tasks = load_tasks(args.classes_yaml)
    TASK = tasks[args.task]

    VIDEOS_DIR = args.videos_dir or TASK.videos_dir
    LABELS_DIR = args.labels_dir or TASK.labels_dir
    TARGET_FRAMES = args.target

    print(f"タスク            : {TASK.name} ({TASK.description})")
    print(f"アノテーション形式: {TASK.annotation_type}")
    print(f"クラス            : {TASK.classes}")
    print(f"動画ディレクトリ  : {VIDEOS_DIR}")
    print(f"ラベルディレクトリ: {LABELS_DIR}")
    print(f"ブラウザで http://localhost:{args.port} を開いてください")
    print("終了: Ctrl+C\n")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
