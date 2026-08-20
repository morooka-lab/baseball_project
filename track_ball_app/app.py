"""
PitchVision AI - ボール軌跡解析ウェブアプリ

train.py で学習した TrackNet の重みを使い、アップロードされた投球動画から
ボールの軌跡を検出して映像に重ねた動画を生成する。
このディレクトリ (track_ball_app) 内だけで完結しており、track_ball 側のコードには依存しない。

起動:
    weight/ ディレクトリに train.py が出力した重み (例: best.pt) を配置してから、
    python app.py --device cuda

ブラウザ:
    http://localhost:5050
"""

import threading
import uuid
from argparse import ArgumentParser
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from inference import load_model, process_video

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB

APP_DIR = Path(__file__).resolve().parent
WEIGHT_DIR = APP_DIR / "weight"
UPLOAD_DIR = APP_DIR / "runs" / "uploads"
OUTPUT_DIR = APP_DIR / "runs" / "outputs"
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".mp4", ".mov", ".m4v"}

# 起動時に init_model() で設定される
_model = None
_device = "cpu"
_imgsz = (288, 512)

_jobs = {}
_jobs_lock = threading.Lock()


def init_model(weights_path, device="cpu", imgsz=(288, 512)):
    global _model, _device, _imgsz
    _device = device
    _imgsz = tuple(imgsz)
    _model = load_model(weights_path, device=device)


def _set_job(job_id, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _run_job(job_id, input_path, output_path):
    def progress_cb(done, total):
        pct = int(min(99, done / total * 100)) if total else 0
        _set_job(job_id, progress=pct)

    try:
        _set_job(job_id, status="processing", progress=0)
        stats = process_video(input_path, output_path, _model, _device, _imgsz, progress_cb=progress_cb)
        _set_job(job_id, status="done", progress=100, stats=stats)
    except Exception as e:
        _set_job(job_id, status="error", message=str(e))
    finally:
        Path(input_path).unlink(missing_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if _model is None:
        return jsonify({"error": "モデルが初期化されていません。--weights を指定してサーバーを起動してください。"}), 503

    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "ファイルが選択されていません"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"対応していない形式です: {ext or '(不明)'}"}), 400

    job_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}{ext}"
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    file.save(input_path)

    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "progress": 0, "output_path": str(output_path)}

    thread = threading.Thread(
        target=_run_job, args=(job_id, str(input_path), str(output_path)), daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        abort(404)
    return jsonify({k: v for k, v in job.items() if k != "output_path"})


@app.route("/api/result/<job_id>")
def result(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None or job.get("status") != "done":
        abort(404)
    return send_file(job["output_path"], mimetype="video/mp4", conditional=True)


def resolve_weights_path(weights_arg: str) -> Path:
    """指定パスが無い場合、weight/ 直下の *.pt が1つだけならそれを使う。"""
    path = Path(weights_arg)
    if path.exists():
        return path

    candidates = sorted(WEIGHT_DIR.glob("*.pt"))
    if len(candidates) == 1:
        return candidates[0]

    return path


def parse_opt():
    parser = ArgumentParser(description="PitchVision AI ウェブアプリ")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(WEIGHT_DIR / "best.pt"),
        help="Path to trained TrackNet weights (train.py の出力)。省略時は weight/best.pt を使用。",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda または cpu")
    parser.add_argument(
        "--imgsz", "--img", "--img-size", nargs="+", type=int, default=[288, 512],
        help="モデル入力サイズ h,w（学習時と合わせる）",
    )
    parser.add_argument("--port", type=int, default=5050)
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()

    weights_path = resolve_weights_path(opt.weights)
    if not weights_path.exists():
        print(f"エラー: 重みファイルが見つかりません: {weights_path}")
        print(f"train.py で学習したモデルの重み (.pt) を {WEIGHT_DIR} に配置してから、再度起動してください。")
        raise SystemExit(1)

    import torch

    device = opt.device
    if device != "cpu" and not torch.cuda.is_available():
        print("CUDA が利用できないため CPU で実行します")
        device = "cpu"

    init_model(str(weights_path), device=device, imgsz=opt.imgsz)
    print(f"モデルロード完了: {weights_path} (device={device})")
    print(f"起動中: http://localhost:{opt.port}")

    app.run(host="0.0.0.0", port=opt.port, threaded=True)
