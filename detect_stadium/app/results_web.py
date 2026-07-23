"""
指定した重みファイルで、videos_dir配下の動画をブラウザ上で選んでその場で
推論・確認するためのビューア(閲覧専用、編集不可)。player(投手・捕手・打者)専用。
stadiumはapp/results_web_keypoint.pyを使う(keypoint版の方が精度が高いため、
stadiumのbbox推論結果閲覧には対応していない)。

bbox_label_web.py と同じ理由(opencv-python内蔵FFmpegのインターレース対応不可)
に加え、cv2.VideoWriterで書き出すオーバーレイ動画のコーデック(mpeg4/mp4v)は
ブラウザの<video>タグでは再生できないため、<video>は使わずに元動画のフレームを
SSEでJPEG配信し、検出結果(bbox)はブラウザ側でcanvasに重ね描きする
(bbox_label_web.pyの/stream_framesと同じ方式)。

推論結果は動画ごとに初回選択時にその場で計算し、--out-dir 配下にJSONとして
キャッシュする(infer.pyと同じ出力形式なので、val.py等とも互換)。2回目以降の
選択はキャッシュを読むだけなので高速。

使い方:
    python app/results_web.py --task player --weights runs/train/player/exp/best.pt
    python app/results_web.py --task player --weights runs/train/player/exp/best.pt \
        --videos-dir /data2/baseball_data/detect_dataset/videos --conf-thres 0.05 --port 5021
"""

import argparse
import base64
import json
import sys
import threading
from pathlib import Path

import cv2
import torch
from flask import Flask, Response, jsonify, render_template, request

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from infer import run_on_video
from models.detector import load_checkpoint
from utils.label_store import has_label, list_videos, load_classes, load_tasks
from utils.stabilize import aggregate_video_detections
from utils.video_io import open_video, read_frame_bgr, stream_meta

app = Flask(__name__)

TASK = None              # utils.label_store.TaskConfig
VIDEOS_DIR = ""
LABELS_DIR = ""
OUT_DIR = None            # Path: 推論結果JSONのキャッシュ先
CLASSES: list = []
MODEL = None
DEVICE = "cpu"
CONF_THRES = 0.05
STABILIZE_CONF_THRES = 0.3
_infer_lock = threading.Lock()  # 単一モデルを複数リクエストから同時に叩かないためのロック

# ── PyAVコンテナpool(bbox_label_web.pyと同じ方式) ───────────────────────────
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


def cache_path_for(video_path: str) -> Path:
    p = Path(video_path)
    try:
        rel = p.relative_to(VIDEOS_DIR)
    except ValueError:
        rel = Path(p.name)
    return OUT_DIR / "predictions" / rel.with_suffix(".json")


def get_or_run_inference(video_path: str) -> dict:
    cpath = cache_path_for(video_path)
    if cpath.exists():
        with open(cpath, "r", encoding="utf-8") as f:
            return json.load(f)

    with _infer_lock:
        # ロック待ちの間に他リクエストが計算済みかもしれないので再チェック
        if cpath.exists():
            with open(cpath, "r", encoding="utf-8") as f:
                return json.load(f)
        result = run_on_video(MODEL, Path(video_path), CLASSES, DEVICE, CONF_THRES, frame_stride=1)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "results.html",
        task_name=TASK.name,
        task_description=TASK.description,
        classes=CLASSES,
        predictions_dir=str(OUT_DIR),
    )


@app.route("/files")
def list_files():
    root = Path(VIDEOS_DIR)
    files = []
    for f in list_videos(VIDEOS_DIR):
        files.append({
            "path": str(f),
            "rel": str(f.relative_to(root)),
            "cached": cache_path_for(str(f)).exists(),
            "labeled": has_label(str(f), VIDEOS_DIR, LABELS_DIR),
        })
    return jsonify(files)


@app.route("/detections")
def detections():
    path = request.args.get("video", "")
    if not path or not Path(path).exists():
        return jsonify({"error": "not found"}), 404
    result = get_or_run_inference(path)
    result["stabilized"] = aggregate_video_detections(
        result["frames"], result["classes"], TASK.point_counts, STABILIZE_CONF_THRES
    )
    return jsonify(result)


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
    """全フレームをJPEG+base64でSSEストリーミングする(bbox_label_web.pyと同じ方式)。"""
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


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="重みファイルを使い、動画を選んでその場で推論・確認するビューア(閲覧専用)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例:\n"
            "  python app/results_web.py --task player --weights runs/train/player/exp/best.pt\n"
            "  python app/results_web.py --task player --weights runs/train/player/exp/best.pt "
            "--conf-thres 0.05 --port 5021\n"
        ),
    )
    p.add_argument("--task", default="player", choices=["player"])
    p.add_argument("--classes-yaml", default=str(ROOT / "data" / "classes.yaml"))
    p.add_argument("--weights", required=True, help="学習済み重み(.pt)")
    p.add_argument("--videos-dir", default=None, help="省略時はclasses.yamlの値")
    p.add_argument("--labels-dir", default=None,
                    help="学習使用済み(ラベルあり)動画の判定に使うラベルディレクトリ (default: classes.yamlの値)")
    p.add_argument("--out-dir", default=None,
                    help="推論結果JSONのキャッシュ先 (default: runs/detect/<task>/<weightsのrun名>)")
    p.add_argument("--conf-thres", type=float, default=0.05,
                    help="推論時に保持する最小confidence(表示側のしきい値スライダーはこれ以上のみ調整可能)")
    p.add_argument("--stabilize-conf-thres", type=float, default=0.3,
                    help="動画単位の統計的平滑化(中央値)に使う検出の最小confidence")
    p.add_argument("--port", type=int, default=5020)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tasks = load_tasks(args.classes_yaml)
    TASK = tasks[args.task]

    VIDEOS_DIR = args.videos_dir or TASK.videos_dir
    LABELS_DIR = args.labels_dir or TASK.labels_dir
    OUT_DIR = Path(args.out_dir) if args.out_dir else ROOT / "runs" / "detect" / args.task / Path(args.weights).parent.name
    CONF_THRES = args.conf_thres
    STABILIZE_CONF_THRES = args.stabilize_conf_thres
    DEVICE = args.device

    CLASSES = load_classes(args.weights, TASK.classes)
    MODEL = load_checkpoint(args.weights, len(CLASSES) + 1, device=DEVICE)

    print(f"タスク              : {TASK.name} ({TASK.description})")
    print(f"重み                : {args.weights}")
    print(f"クラス              : {CLASSES}")
    print(f"動画ディレクトリ    : {VIDEOS_DIR}")
    print(f"ラベルディレクトリ  : {LABELS_DIR}")
    print(f"推論結果キャッシュ先: {OUT_DIR}")
    print(f"device              : {DEVICE}")
    print(f"ブラウザで http://localhost:{args.port} を開いてください")
    print("終了: Ctrl+C\n")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
