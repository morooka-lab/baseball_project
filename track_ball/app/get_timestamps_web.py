"""
Web版 get_timestamps.py
動画をブラウザで再生しながらタイムスタンプを記録し、クリップを作成するツール。

使い方:
    python get_timestamps_web.py
    python get_timestamps_web.py <動画ファイルパス>
    python get_timestamps_web.py --port 5001

VS Code Remote 使用時はポートが自動フォワードされます。
ブラウザで http://localhost:5001 を開いてください。
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)
VIDEO_PATH = ""
DATA_DIR = "/data2/baseball_data/RKB"




@app.route("/")
def index():
    return render_template("timestamps.html")


@app.route("/files")
def list_files():
    """データディレクトリのmp4ファイル一覧を返す。"""
    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        return jsonify([])
    files = sorted(data_dir.glob("*.mp4"))
    return jsonify([str(f) for f in files])


@app.route("/video")
def video():
    """HTTP Range requestに対応した動画ストリーミング（シーク対応）。"""
    path = request.args.get("video", "") or VIDEO_PATH
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return "動画ファイルが見つかりません", 404

    file_size = os.path.getsize(path)
    range_header = request.headers.get("Range")

    if range_header:
        match = re.search(r"bytes=(\d+)-(\d*)", range_header)
        byte_start = int(match.group(1))
        byte_end = (
            int(match.group(2))
            if match.group(2)
            else min(byte_start + 1024 * 1024, file_size - 1)
        )
        length = byte_end - byte_start + 1

        with open(path, "rb") as f:
            f.seek(byte_start)
            data = f.read(length)

        return Response(
            data,
            206,
            mimetype="video/mp4",
            headers={
                "Content-Range": f"bytes {byte_start}-{byte_end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
            },
        )

    return Response(
        open(path, "rb"),
        mimetype="video/mp4",
        headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
    )


@app.route("/save", methods=["POST"])
def save():
    """タイムスタンプをファイルに保存する。"""
    data = request.get_json()
    ts_list = data.get("timestamps", [])
    video_path = data.get("video_path", "") or VIDEO_PATH
    video_path = os.path.abspath(video_path)
    output_dir = data.get("output_dir", "").strip()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir) / "timestamps.py"
    else:
        output_path = Path(video_path).parent / "timestamps.py"

    lines = ["TIMESTAMPS = ["]
    for start, end in ts_list:
        lines.append(f"    ({start:.2f}, {end:.2f}),")
    lines.append("]")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n保存完了: {output_path}")
    return jsonify({"path": str(output_path)})


@app.route("/clip", methods=["POST"])
def clip():
    """タイムスタンプに従ってffmpegで動画クリップを作成する。"""
    data = request.get_json()
    ts_list = data.get("timestamps", [])
    video_path = data.get("video_path", "") or VIDEO_PATH
    video_path = os.path.abspath(video_path)
    output_dir = data.get("output_dir", "").strip()

    if not output_dir:
        output_dir = str(Path(video_path).parent)

    os.makedirs(output_dir, exist_ok=True)

    stem = Path(video_path).stem
    results = []

    for i, (start, end) in enumerate(ts_list):
        duration = round(end - start, 3)
        output_name = f"{stem}_clip_{i+1:03d}_{start:.2f}s_{end:.2f}s.mp4"
        output_path = os.path.join(output_dir, output_name)

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c", "copy",
            output_path,
        ]

        print(f"[{i+1}/{len(ts_list)}] {output_name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            results.append({"path": output_path, "status": "ok"})
            print(f"  → 完了")
        else:
            results.append({"path": output_path, "status": "error", "error": result.stderr[-200:]})
            print(f"  → エラー: {result.stderr[-100:]}")

    return jsonify({"clips": results, "output_dir": output_dir})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Web版タイムスタンプ取得ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例: python get_timestamps_web.py\n     python get_timestamps_web.py /path/to/game.mp4 --port 5001",
    )
    parser.add_argument("video_path", nargs="?", default="", help="動画ファイルのパス（省略可：UIで選択可能）")
    parser.add_argument("--port", type=int, default=5001, help="ポート番号 (default: 5001)")
    parser.add_argument("--data-dir", default=DATA_DIR, help=f"動画ファイルのディレクトリ (default: {DATA_DIR})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DATA_DIR = args.data_dir

    if args.video_path:
        VIDEO_PATH = os.path.abspath(args.video_path)
        if not os.path.exists(VIDEO_PATH):
            print(f"エラー: 動画ファイルが見つかりません: {VIDEO_PATH}")
            sys.exit(1)
        print(f"動画: {VIDEO_PATH}")

    print(f"データディレクトリ: {DATA_DIR}")
    print(f"ブラウザで http://localhost:{args.port} を開いてください")
    print("(VS Code Remote は自動でポートをフォワードします)")
    print("終了: Ctrl+C\n")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
