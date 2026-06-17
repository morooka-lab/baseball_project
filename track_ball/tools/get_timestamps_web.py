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

from flask import Flask, Response, jsonify, render_template_string, request

app = Flask(__name__)
VIDEO_PATH = ""
DATA_DIR = "/data2/baseball_data/RKB"

HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>タイムスタンプ取得ツール</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 12px; height: 100vh; display: flex; flex-direction: column; gap: 8px; }
  h1 { font-size: 1.1em; color: #aaa; margin: 0; flex-shrink: 0; }
  .top-bar { display: flex; gap: 12px; align-items: flex-end; flex-shrink: 0; background: #222; border-radius: 6px; padding: 10px 14px; }
  .top-bar label { font-size: 0.8em; color: #888; display: block; margin-bottom: 4px; }
  .top-bar select, .top-bar input[type=text] {
    background: #333; color: #eee; border: 1px solid #444; border-radius: 4px;
    padding: 6px 8px; font-size: 0.9em;
  }
  .top-bar select { width: 340px; }
  .top-bar input[type=text] { width: 280px; }
  .container { display: flex; gap: 16px; align-items: stretch; flex: 1; overflow: hidden; }
  .left { flex: 1; min-width: 0; display: flex; align-items: center; justify-content: center; background: #000; border-radius: 6px; overflow: hidden; }
  .right { width: 320px; flex-shrink: 0; display: flex; flex-direction: column; gap: 0; overflow-y: auto; }
  video { width: 100%; height: 100%; object-fit: contain; display: block; }
  .time-display { font-size: 1.6em; font-family: monospace; margin: 0 0 6px; color: #4f4; }
  .status { font-size: 0.9em; margin: 0 0 8px; min-height: 1.3em; padding: 6px 10px; border-radius: 4px; }
  .status.waiting  { background: #2a2a2a; color: #aaa; }
  .status.started  { background: #3a2a00; color: #fc0; }
  .status.recorded { background: #0a2a0a; color: #4f4; }
  .status.error    { background: #2a0a0a; color: #f66; }
  .buttons { display: flex; gap: 8px; margin: 0 0 8px; }
  button { padding: 10px 16px; font-size: 0.95em; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; }
  .btn-start { background: #c60; color: #fff; flex: 1; }
  .btn-end   { background: #080; color: #fff; flex: 1; }
  .btn-undo  { background: #444; color: #ccc; }
  .btn-start:hover { background: #e80; }
  .btn-end:hover   { background: #0a0; }
  .btn-undo:hover  { background: #666; }
  .shortcuts { font-size: 0.75em; color: #666; line-height: 1.8; }
  .shortcuts b { color: #888; }
  h2 { font-size: 0.9em; color: #888; margin: 0 0 6px; text-transform: uppercase; letter-spacing: 0.05em; }
  .ts-list { list-style: none; padding: 0; margin: 0 0 8px; max-height: 200px; overflow-y: auto; flex-shrink: 0; }
  .ts-list li {
    background: #252525; border-radius: 4px; padding: 7px 10px; margin-bottom: 4px;
    font-family: monospace; font-size: 0.85em; display: flex; justify-content: space-between; align-items: center;
  }
  .ts-list li.empty { color: #555; font-style: italic; font-family: sans-serif; }
  .del { cursor: pointer; color: #833; padding: 0 4px; }
  .del:hover { color: #f66; }
  .action-buttons { display: flex; gap: 8px; flex-shrink: 0; }
  .btn-save { background: #17a; color: #fff; flex: 1; padding: 11px; font-size: 0.95em; border-radius: 6px; border: none; cursor: pointer; font-weight: bold; }
  .btn-save:hover { background: #19c; }
  .btn-clip { background: #7a2; color: #fff; flex: 1; padding: 11px; font-size: 0.95em; border-radius: 6px; border: none; cursor: pointer; font-weight: bold; }
  .btn-clip:hover { background: #9c3; }
  .btn-clip:disabled { background: #444; color: #666; cursor: not-allowed; }
  .saved-msg { font-size: 0.82em; color: #4f4; margin-top: 6px; min-height: 1.2em; word-break: break-all; }
  .clip-status { font-size: 0.82em; margin-top: 6px; min-height: 1.2em; word-break: break-all; }
  .clip-status.ok    { color: #4f4; }
  .clip-status.error { color: #f66; }
  .clip-status.running { color: #fc0; }
  .sep { border: none; border-top: 1px solid #333; margin: 10px 0; flex-shrink: 0; }
  .output-box { background: #111; border-radius: 4px; padding: 10px; font-family: monospace; font-size: 0.8em; white-space: pre; color: #8f8; max-height: 150px; overflow-y: auto; }
</style>
</head>
<body>
<h1>🎬 タイムスタンプ取得ツール（Web版）</h1>

<div class="top-bar">
  <div>
    <label>動画ファイル</label>
    <select id="fileSelect" onchange="selectFile(this.value)">
      <option value="">-- ファイルを選択 --</option>
    </select>
  </div>
  <div>
    <label>出力先ディレクトリ</label>
    <input type="text" id="outputDir" placeholder="例: /data2/baseball_data/RKB/videos" />
  </div>
</div>

<div class="container">
  <div class="left">
    <video id="video" controls></video>
  </div>

  <div class="right">
    <div class="time-display" id="timeDisplay">00:00.000</div>
    <div class="status waiting" id="status">▶ ファイルを選択して再生してください</div>
    <div class="buttons">
      <button class="btn-start" onclick="markStart()">⏺ 開始 [S]</button>
      <button class="btn-end"   onclick="markEnd()">⏹ 終了 [E]</button>
      <button class="btn-undo"  onclick="undoLast()">↩ [U]</button>
    </div>
    <div class="shortcuts">
      <b>スペース</b> 再生/停止 &nbsp;|&nbsp;
      <b>←/→</b> ±5秒 &nbsp;|&nbsp;
      <b>A/D</b> 1フレーム &nbsp;|&nbsp;
      <b>S</b> 開始 &nbsp;|&nbsp;
      <b>E</b> 終了 &nbsp;|&nbsp;
      <b>U</b> 取り消し
    </div>
    <hr class="sep">
    <h2>記録済み (<span id="count">0</span>件)</h2>
    <ul class="ts-list" id="tsList">
      <li class="empty">まだ記録がありません</li>
    </ul>
    <div class="action-buttons">
      <button class="btn-save" onclick="saveToFile()">💾 保存</button>
      <button class="btn-clip" onclick="createClips()">✂ クリップ作成</button>
    </div>
    <div class="saved-msg" id="savedMsg"></div>
    <div class="clip-status" id="clipStatus"></div>
    <hr class="sep">
    <h2>出力プレビュー</h2>
    <div class="output-box" id="outputBox">TIMESTAMPS = []</div>
  </div>
</div>

<script>
const video = document.getElementById('video');
let currentVideoPath = '';
let startTime = null;
let timestamps = [];
let fps = 30;

// ファイル一覧を取得してセレクタに反映
async function loadFileList() {
  try {
    const res = await fetch('/files');
    const files = await res.json();
    const sel = document.getElementById('fileSelect');
    files.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f;
      opt.textContent = f.split('/').pop();
      sel.appendChild(opt);
    });
    // URL パラメータで動画が指定されていれば選択状態にする
    const params = new URLSearchParams(window.location.search);
    const initialVideo = params.get('video') || '';
    if (initialVideo) {
      sel.value = initialVideo;
      selectFile(initialVideo);
    }
  } catch (e) {
    console.error('ファイル一覧の取得に失敗:', e);
  }
}

function selectFile(path) {
  if (!path) return;
  currentVideoPath = path;
  video.src = '/video?video=' + encodeURIComponent(path);
  video.load();
  // 出力先デフォルト: 動画と同じディレクトリ
  const dir = path.substring(0, path.lastIndexOf('/'));
  const outputDir = document.getElementById('outputDir');
  if (!outputDir.value) outputDir.value = dir;
  document.getElementById('status').textContent = '▶ 再生して投球シーンを探してください';
  document.getElementById('status').className = 'status waiting';
}

function formatTime(t) {
  const m = Math.floor(t / 60).toString().padStart(2, '0');
  const s = (t % 60).toFixed(3).padStart(6, '0');
  return `${m}:${s}`;
}

video.addEventListener('timeupdate', () => {
  document.getElementById('timeDisplay').textContent = formatTime(video.currentTime);
});

function setStatus(msg, cls) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = 'status ' + cls;
}

function markStart() {
  startTime = video.currentTime;
  setStatus(`⏺ 開始: ${formatTime(startTime)} — 終了時刻を記録してください [E]`, 'started');
}

function markEnd() {
  if (startTime === null) {
    setStatus('⚠ 先に開始時刻を記録してください [S]', 'error');
    return;
  }
  const endTime = video.currentTime;
  if (endTime <= startTime) {
    setStatus('⚠ 終了時刻は開始時刻より後にしてください', 'error');
    return;
  }
  const s = parseFloat(startTime.toFixed(2));
  const e = parseFloat(endTime.toFixed(2));
  timestamps.push([s, e]);
  setStatus(`✅ 記録 #${timestamps.length}: (${s.toFixed(2)}, ${e.toFixed(2)})`, 'recorded');
  startTime = null;
  updateUI();
}

function undoLast() {
  if (timestamps.length === 0 && startTime === null) return;
  if (startTime !== null) {
    startTime = null;
    setStatus('↩ 開始時刻をキャンセルしました', 'waiting');
  } else {
    const removed = timestamps.pop();
    setStatus(`↩ #${timestamps.length + 1} (${removed[0].toFixed(2)}, ${removed[1].toFixed(2)}) を取り消しました`, 'waiting');
  }
  updateUI();
}

function deleteAt(idx) {
  timestamps.splice(idx, 1);
  setStatus(`↩ #${idx + 1} を削除しました`, 'waiting');
  updateUI();
}

function updateUI() {
  document.getElementById('count').textContent = timestamps.length;
  const list = document.getElementById('tsList');
  if (timestamps.length === 0) {
    list.innerHTML = '<li class="empty">まだ記録がありません</li>';
  } else {
    list.innerHTML = timestamps.map((ts, i) =>
      `<li><span>#${i+1} &nbsp;${ts[0].toFixed(2)}s → ${ts[1].toFixed(2)}s &nbsp;(${(ts[1]-ts[0]).toFixed(2)}s)</span>
       <span class="del" onclick="deleteAt(${i})">✕</span></li>`
    ).join('');
  }
  const lines = ['TIMESTAMPS = ['];
  timestamps.forEach(ts => lines.push(`    (${ts[0].toFixed(2)}, ${ts[1].toFixed(2)}),`));
  lines.push(']');
  document.getElementById('outputBox').textContent = lines.join('\n');
}

async function saveToFile() {
  if (timestamps.length === 0) {
    setStatus('⚠ 保存するタイムスタンプがありません', 'error');
    return;
  }
  const outputDir = document.getElementById('outputDir').value.trim();
  const res = await fetch('/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ timestamps, video_path: currentVideoPath, output_dir: outputDir })
  });
  const data = await res.json();
  document.getElementById('savedMsg').textContent = `保存先: ${data.path}`;
}

async function createClips() {
  if (timestamps.length === 0) {
    setStatus('⚠ クリップを作成するタイムスタンプがありません', 'error');
    return;
  }
  if (!currentVideoPath) {
    setStatus('⚠ 動画ファイルを選択してください', 'error');
    return;
  }
  const outputDir = document.getElementById('outputDir').value.trim();
  const clipBtn = document.querySelector('.btn-clip');
  clipBtn.disabled = true;
  const clipStatus = document.getElementById('clipStatus');
  clipStatus.className = 'clip-status running';
  clipStatus.textContent = `✂ ${timestamps.length}件のクリップを作成中...`;

  try {
    const res = await fetch('/clip', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ timestamps, video_path: currentVideoPath, output_dir: outputDir })
    });
    const data = await res.json();
    const ok = data.clips.filter(c => c.status === 'ok').length;
    const err = data.clips.filter(c => c.status === 'error').length;
    if (err === 0) {
      clipStatus.className = 'clip-status ok';
      clipStatus.textContent = `✅ ${ok}件のクリップを作成しました → ${data.output_dir}`;
    } else {
      clipStatus.className = 'clip-status error';
      clipStatus.textContent = `⚠ ${ok}件成功 / ${err}件失敗 → ${data.output_dir}`;
    }
  } catch (e) {
    clipStatus.className = 'clip-status error';
    clipStatus.textContent = `エラー: ${e.message}`;
  } finally {
    clipBtn.disabled = false;
  }
}

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  switch (e.key.toLowerCase()) {
    case 's': markStart(); break;
    case 'e': markEnd(); break;
    case 'u': undoLast(); break;
    case ' ':
      e.preventDefault();
      video.paused ? video.play() : video.pause();
      break;
    case 'arrowleft':
      e.preventDefault();
      video.currentTime = Math.max(0, video.currentTime - 5);
      break;
    case 'arrowright':
      e.preventDefault();
      video.currentTime = Math.min(video.duration, video.currentTime + 5);
      break;
    case 'a':
      video.currentTime = Math.max(0, video.currentTime - 1 / fps);
      break;
    case 'd':
      video.currentTime = Math.min(video.duration, video.currentTime + 1 / fps);
      break;
  }
});

loadFileList();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


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
