"""
Web版 get_timestamps.py
動画をブラウザで再生しながらタイムスタンプを記録するツール。

使い方:
    python get_timestamps_web.py <動画ファイルパス>
    python get_timestamps_web.py <動画ファイルパス> --port 5001

VS Code Remote 使用時はポートが自動フォワードされます。
ブラウザで http://localhost:5001 を開いてください。

削除方法:
    このファイル (get_timestamps_web.py) を削除するだけです。
"""

import argparse
import os
import re
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, render_template_string, request

app = Flask(__name__)
VIDEO_PATH = ""

HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>タイムスタンプ取得ツール</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 20px; }
  h1 { font-size: 1.1em; color: #aaa; margin: 0 0 16px; }
  .container { display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }
  .left { flex: 1; min-width: 400px; }
  .right { width: 300px; }
  video { width: 100%; border-radius: 6px; background: #000; display: block; }
  .time-display { font-size: 1.6em; font-family: monospace; margin: 8px 0 4px; color: #4f4; }
  .status { font-size: 0.95em; margin: 6px 0; min-height: 1.3em; padding: 6px 10px; border-radius: 4px; }
  .status.waiting  { background: #2a2a2a; color: #aaa; }
  .status.started  { background: #3a2a00; color: #fc0; }
  .status.recorded { background: #0a2a0a; color: #4f4; }
  .status.error    { background: #2a0a0a; color: #f66; }
  .buttons { display: flex; gap: 8px; margin: 10px 0; }
  button { padding: 10px 16px; font-size: 0.95em; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; }
  .btn-start { background: #c60; color: #fff; flex: 1; }
  .btn-end   { background: #080; color: #fff; flex: 1; }
  .btn-undo  { background: #444; color: #ccc; }
  .btn-start:hover { background: #e80; }
  .btn-end:hover   { background: #0a0; }
  .btn-undo:hover  { background: #666; }
  .shortcuts { font-size: 0.78em; color: #666; line-height: 1.8; }
  .shortcuts b { color: #888; }
  h2 { font-size: 0.9em; color: #888; margin: 0 0 8px; text-transform: uppercase; letter-spacing: 0.05em; }
  .ts-list { list-style: none; padding: 0; margin: 0 0 12px; max-height: 320px; overflow-y: auto; }
  .ts-list li {
    background: #252525; border-radius: 4px; padding: 7px 10px; margin-bottom: 4px;
    font-family: monospace; font-size: 0.85em; display: flex; justify-content: space-between; align-items: center;
  }
  .ts-list li.empty { color: #555; font-style: italic; font-family: sans-serif; }
  .del { cursor: pointer; color: #833; padding: 0 4px; }
  .del:hover { color: #f66; }
  .btn-save { background: #17a; color: #fff; width: 100%; padding: 11px; font-size: 0.95em; border-radius: 6px; border: none; cursor: pointer; font-weight: bold; }
  .btn-save:hover { background: #19c; }
  .saved-msg { font-size: 0.82em; color: #4f4; margin-top: 6px; min-height: 1.2em; word-break: break-all; }
  .sep { border: none; border-top: 1px solid #333; margin: 14px 0; }
  .output-box { background: #111; border-radius: 4px; padding: 10px; font-family: monospace; font-size: 0.8em; white-space: pre; color: #8f8; max-height: 180px; overflow-y: auto; }
</style>
</head>
<body>
<h1>🎬 タイムスタンプ取得ツール（Web版）</h1>
<div class="container">
  <div class="left">
    <video id="video" controls>
      <source src="/video" type="video/mp4">
    </video>
    <div class="time-display" id="timeDisplay">00:00.000</div>
    <div class="status waiting" id="status">▶ 再生して投球シーンを探してください</div>
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
  </div>

  <div class="right">
    <h2>記録済み (<span id="count">0</span>件)</h2>
    <ul class="ts-list" id="tsList">
      <li class="empty">まだ記録がありません</li>
    </ul>
    <button class="btn-save" onclick="saveToFile()">💾 ファイルに保存</button>
    <div class="saved-msg" id="savedMsg"></div>
    <hr class="sep">
    <h2>出力プレビュー</h2>
    <div class="output-box" id="outputBox">TIMESTAMPS = []</div>
  </div>
</div>

<script>
const video = document.getElementById('video');
let startTime = null;
let timestamps = [];

// FPS estimation for 1-frame step (updated after metadata loads)
let fps = 30;
video.addEventListener('loadedmetadata', () => {
  // HTML5 video doesn't expose FPS directly; default 30 is fine for step
});

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
      `<li><span>#${i+1} &nbsp;${ts[0].toFixed(2)}s → ${ts[1].toFixed(2)}s</span>
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
  const res = await fetch('/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ timestamps })
  });
  const data = await res.json();
  document.getElementById('savedMsg').textContent = `保存先: ${data.path}`;
}

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') return;
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
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video")
def video():
    """HTTP Range requestに対応した動画ストリーミング（シーク対応）。"""
    file_size = os.path.getsize(VIDEO_PATH)
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

        with open(VIDEO_PATH, "rb") as f:
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
        open(VIDEO_PATH, "rb"),
        mimetype="video/mp4",
        headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
    )


@app.route("/save", methods=["POST"])
def save():
    """タイムスタンプを動画と同じディレクトリに timestamps.py として保存する。"""
    data = request.get_json()
    ts_list = data.get("timestamps", [])

    output_path = Path(VIDEO_PATH).parent / "timestamps.py"
    lines = ["TIMESTAMPS = ["]
    for start, end in ts_list:
        lines.append(f"    ({start:.2f}, {end:.2f}),")
    lines.append("]")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n保存完了: {output_path}")
    return jsonify({"path": str(output_path)})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Web版タイムスタンプ取得ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例: python get_timestamps_web.py /path/to/game.mp4 --port 5001",
    )
    parser.add_argument("video_path", help="動画ファイルのパス")
    parser.add_argument("--port", type=int, default=5001, help="ポート番号 (default: 5001)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    VIDEO_PATH = os.path.abspath(args.video_path)

    if not os.path.exists(VIDEO_PATH):
        print(f"エラー: 動画ファイルが見つかりません: {VIDEO_PATH}")
        sys.exit(1)

    print(f"動画: {VIDEO_PATH}")
    print(f"ブラウザで http://localhost:{args.port} を開いてください")
    print("(VS Code Remote は自動でポートをフォワードします)")
    print("終了: Ctrl+C\n")

    app.run(host="0.0.0.0", port=args.port, debug=False)
