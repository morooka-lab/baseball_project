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
from flask import Flask, Response, jsonify, render_template_string, request

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
    return render_template_string(HTML)


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


# ── HTML ─────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>ボールラベリングツール</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: sans-serif;
    background: #1a1a1a;
    color: #eee;
    margin: 0;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .header {
    background: #222;
    border-bottom: 1px solid #333;
    padding: 8px 16px;
    display: flex;
    align-items: center;
    gap: 20px;
    flex-shrink: 0;
  }
  .header h1 { margin: 0; font-size: 1em; color: #ddd; }
  .shortcuts {
    font-size: 0.75em;
    color: #555;
    line-height: 1.6;
    margin-left: auto;
  }
  .shortcuts b { color: #888; }
  .main {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  /* ── Sidebar ── */
  .sidebar {
    width: 260px;
    background: #1e1e1e;
    border-right: 1px solid #333;
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    overflow: hidden;
  }
  .sidebar-header {
    padding: 10px 12px;
    font-size: 0.78em;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid #333;
    flex-shrink: 0;
  }
  .file-list {
    overflow-y: auto;
    flex: 1;
  }
  .group-label {
    padding: 10px 12px 4px;
    font-size: 0.73em;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .file-item {
    padding: 7px 12px;
    cursor: pointer;
    font-size: 0.82em;
    display: flex;
    align-items: center;
    gap: 8px;
    overflow: hidden;
  }
  .file-item:hover { background: #2a2a2a; }
  .file-item.active { background: #1e3a55; color: #6cf; }
  .file-name {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex: 1;
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .dot.done { background: #4c4; }
  .dot.todo { background: #444; border: 1px solid #666; }

  /* ── Canvas area ── */
  .canvas-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .canvas-wrap {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #000;
    overflow: hidden;
    position: relative;
  }
  canvas {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    cursor: crosshair;
    display: block;
  }
  .no-video {
    color: #555;
    font-size: 1.1em;
  }

  /* ── Controls bar ── */
  .controls {
    background: #222;
    border-top: 1px solid #333;
    padding: 10px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
    flex-wrap: wrap;
  }
  .frame-info {
    font-family: monospace;
    font-size: 0.88em;
    color: #aaa;
    min-width: 120px;
  }
  .state-badge {
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: bold;
    min-width: 80px;
    text-align: center;
  }
  .state-visible { background: #0a2a0a; color: #4d4; }
  .state-hidden  { background: #2a2a2a; color: #777; }
  .progress-info { font-size: 0.78em; color: #666; }
  .nav-buttons { display: flex; gap: 4px; }
  button {
    background: #333;
    color: #ccc;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 5px 10px;
    cursor: pointer;
    font-size: 0.82em;
  }
  button:hover { background: #444; color: #fff; }
  button:disabled { opacity: 0.3; cursor: not-allowed; }
  .btn-save {
    background: #17a;
    color: #fff;
    border-color: #19c;
    font-weight: bold;
    padding: 5px 16px;
    margin-left: auto;
  }
  .btn-save:hover { background: #19c; }
  .saved-msg {
    font-size: 0.78em;
    color: #4c4;
    min-width: 60px;
  }
  .dirty-badge {
    font-size: 0.75em;
    color: #fa0;
    min-width: 60px;
  }
  #cacheProgress {
    font-size: 0.82em;
    color: #fa0;
    font-family: monospace;
    min-width: 180px;
  }
  .nav-disabled {
    opacity: 0.3;
    pointer-events: none;
  }
  .csv-path {
    font-size: 0.72em;
    color: #555;
    font-family: monospace;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 300px;
  }
</style>
</head>
<body>

<div class="header">
  <h1>ボールラベリングツール</h1>
  <span class="shortcuts">
    <b>クリック</b> ボール位置 &nbsp;|&nbsp;
    <b>右クリック</b> 非表示(HIDDEN) &nbsp;|&nbsp;
    <b>← / →</b> 次/前フレーム &nbsp;|&nbsp;
    <b>N / P</b> 次/前フレーム &nbsp;|&nbsp;
    <b>F / B</b> ±36フレーム &nbsp;|&nbsp;
    <b>Z / X</b> 最初/最後 &nbsp;|&nbsp;
    <b>+ / -</b> サークルサイズ &nbsp;|&nbsp;
    <b>S</b> 保存
  </span>
</div>

<div class="main">
  <div class="sidebar">
    <div class="sidebar-header">動画一覧 (<span id="fileCount">0</span> 本)</div>
    <div class="file-list" id="fileList"></div>
  </div>

  <div class="canvas-area">
    <div class="canvas-wrap" id="canvasWrap">
      <div class="no-video" id="noVideo">← 動画を選択してください</div>
      <canvas id="canvas" style="display:none"></canvas>
    </div>
    <div class="controls">
      <div class="frame-info" id="frameInfo">-- / --</div>
      <div class="state-badge state-hidden" id="stateBadge">--</div>
      <div class="progress-info" id="progressInfo"></div>
      <div id="cacheProgress"></div>
      <div class="nav-buttons" id="navButtons">
        <button onclick="goFirst()" title="最初のフレーム (Z)">|&lt;</button>
        <button onclick="navigate(-36)" title="36フレーム戻る (B)">&lt;&lt;36</button>
        <button onclick="navigate(-1)" title="前のフレーム (P)">&lt;</button>
        <button onclick="navigate(1)"  title="次のフレーム (N)">&gt;</button>
        <button onclick="navigate(36)" title="36フレーム進む (F)">36&gt;&gt;</button>
        <button onclick="goLast()" title="最後のフレーム (X)">&gt;|</button>
      </div>
      <button class="btn-save" onclick="saveLabels()">💾 保存 [S]</button>
      <div class="saved-msg" id="savedMsg"></div>
      <div class="dirty-badge" id="dirtyBadge"></div>
      <div class="csv-path" id="csvPath"></div>
    </div>
  </div>
</div>

<script>
let currentVideo = '';
let labels = [];
let currentFrame = 0;
let totalFrames = 0;
let circleSize = 8;
let dirty = false;

// フレームキャッシュ: フレーム番号 -> HTMLImageElement
const frameCache = new Map();
let loadingFor = '';         // 読み込み中の動画パス（'' = 非読み込み中）
let activeEventSource = null;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// ─── File list ────────────────────────────────────────────────────────────

async function loadFileList() {
  const res = await fetch('/files');
  const files = await res.json();
  document.getElementById('fileCount').textContent = files.length;

  // サブディレクトリごとにグループ化
  const groups = {};
  for (const f of files) {
    const parts = f.rel.split('/');
    const dir = parts.length > 1 ? parts.slice(0, -1).join('/') : '';
    if (!groups[dir]) groups[dir] = [];
    groups[dir].push(f);
  }

  const container = document.getElementById('fileList');
  container.innerHTML = '';

  for (const [dir, items] of Object.entries(groups)) {
    if (dir) {
      const doneCount = items.filter(f => f.labeled).length;
      const gl = document.createElement('div');
      gl.className = 'group-label';
      gl.textContent = `${dir}  (${doneCount}/${items.length})`;
      container.appendChild(gl);
    }
    for (const f of items) {
      const el = document.createElement('div');
      el.className = 'file-item';
      el.dataset.path = f.path;
      el.innerHTML = `
        <div class="dot ${f.labeled ? 'done' : 'todo'}" id="dot_${CSS.escape(f.path)}"></div>
        <span class="file-name" title="${f.rel}">${f.rel.split('/').pop()}</span>`;
      el.onclick = () => selectVideo(f.path, el);
      container.appendChild(el);
    }
  }
}

// ─── Video selection ──────────────────────────────────────────────────────

async function selectVideo(path, el) {
  if (dirty) {
    const ok = confirm('未保存のラベルがあります。保存してから切り替えますか？\n[OK] 保存して切り替え  [キャンセル] 破棄して切り替え');
    if (ok) await saveLabels();
  }

  document.querySelectorAll('.file-item').forEach(e => e.classList.remove('active'));
  if (el) el.classList.add('active');

  currentVideo = path;
  currentFrame = 0;
  labels = [];
  frameCache.clear();

  // メタデータ取得
  const metaRes = await fetch(`/meta?video=${encodeURIComponent(path)}`);
  const meta = await metaRes.json();
  totalFrames = meta.frames;
  canvas.width  = meta.width;
  canvas.height = meta.height;

  // ラベル取得
  const labelsRes = await fetch(`/labels?video=${encodeURIComponent(path)}`);
  const labelsData = await labelsRes.json();
  labels = labelsData.data;
  document.getElementById('csvPath').textContent = labelsData.csv_path;

  document.getElementById('noVideo').style.display = 'none';
  canvas.style.display = 'block';
  dirty = false;
  setDirtyBadge(false);

  await renderFrame(0);
  updateControls();

  // 全フレームをSSEストリームで受信し、キャッシュ完了まで待機
  await preloadAll(path);
}

// ─── Frame cache & SSE preloading ────────────────────────────────────────

function setLoadingUI(isLoading) {
  document.getElementById('navButtons').classList.toggle('nav-disabled', isLoading);
}

function preloadAll(videoPath) {
  // 前回の読み込みを中断
  if (activeEventSource) { activeEventSource.close(); activeEventSource = null; }

  loadingFor = videoPath;
  setLoadingUI(true);
  const prog = document.getElementById('cacheProgress');
  prog.textContent = `読み込み中... 0 / ${totalFrames}`;

  return new Promise((resolve) => {
    const es = new EventSource(`/stream_frames?video=${encodeURIComponent(videoPath)}`);
    activeEventSource = es;
    const pending = [];

    es.onmessage = (e) => {
      // 別の動画が選択された場合は中断
      if (loadingFor !== videoPath) {
        es.close(); activeEventSource = null;
        resolve(); return;
      }

      const msg = JSON.parse(e.data);

      if (msg.done) {
        es.close(); activeEventSource = null;
        // 全 Image.onload の完了を待ってから解放
        Promise.all(pending).then(() => {
          if (loadingFor === videoPath) {
            // CAP_PROP_FRAME_COUNT の誤差を実際の枚数で補正
            if (msg.actual !== undefined && msg.actual !== totalFrames) {
              totalFrames = msg.actual;
              labels = labels.slice(0, totalFrames);
              updateControls();
            }
            loadingFor = '';
            prog.textContent = '';
            setLoadingUI(false);
          }
          resolve();
        });
        return;
      }

      const p = new Promise((res) => {
        const img = new Image();
        img.onload = () => { frameCache.set(msg.i, img); res(); };
        img.onerror = res;
        img.src = 'data:image/jpeg;base64,' + msg.d;
      });
      pending.push(p);
      prog.textContent = `読み込み中... ${msg.i + 1} / ${totalFrames}`;
    };

    es.onerror = () => {
      es.close(); activeEventSource = null;
      if (loadingFor === videoPath) {
        loadingFor = '';
        prog.textContent = '';
        setLoadingUI(false);
      }
      resolve();
    };
  });
}

// ─── Frame rendering ──────────────────────────────────────────────────────

function renderFrame(n) {
  const cached = frameCache.get(n);
  if (cached?.complete && cached.naturalWidth > 0) {
    ctx.drawImage(cached, 0, 0, canvas.width, canvas.height);
    drawOverlay(n);
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    const img = new Image();
    frameCache.set(n, img);
    img.onload = () => {
      // 連続移動でフレームが追い越された場合は描画しない
      if (currentFrame === n) {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        drawOverlay(n);
      }
      resolve();
    };
    img.onerror = resolve;
    img.src = `/frame?video=${encodeURIComponent(currentVideo)}&n=${n}`;
  });
}

function drawOverlay(n) {
  const label = labels[n];
  if (!label) return;

  const fh = Math.round(canvas.height * 0.045);
  ctx.font = `bold ${fh}px sans-serif`;

  // 状態テキスト
  ctx.fillStyle = label.visible ? '#00ff00' : '#888888';
  ctx.fillText(label.visible ? 'VISIBLE' : 'HIDDEN', 20, fh + 10);

  // フレーム番号
  ctx.fillStyle = '#ff4444';
  ctx.font = `${Math.round(fh * 0.85)}px sans-serif`;
  ctx.fillText(`Frame: ${n + 1} / ${totalFrames}`, 20, fh * 2 + 14);

  // ボール円
  if (label.visible) {
    const px = label.x * canvas.width;
    const py = label.y * canvas.height;
    ctx.beginPath();
    ctx.arc(px, py, circleSize, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(255, 0, 0, 0.75)';
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

// ─── Navigation ───────────────────────────────────────────────────────────

async function navigate(delta) {
  if (!currentVideo || loadingFor) return;
  const next = Math.max(0, Math.min(totalFrames - 1, currentFrame + delta));
  if (next === currentFrame) return;
  currentFrame = next;
  updateControls();          // UI を即座に更新
  await renderFrame(next);   // キャッシュがあれば即描画
}

async function goFirst() {
  if (!currentVideo || loadingFor) return;
  currentFrame = 0;
  updateControls();
  await renderFrame(0);
}

async function goLast() {
  if (!currentVideo || loadingFor) return;
  currentFrame = totalFrames - 1;
  updateControls();
  await renderFrame(currentFrame);
}

// ─── Annotation ───────────────────────────────────────────────────────────

canvas.addEventListener('click', async (e) => {
  if (!currentVideo || loadingFor) return;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  const y = (e.clientY - rect.top)  / rect.height;
  labels[currentFrame] = { frame_num: currentFrame, visible: 1, x, y };
  dirty = true;
  setDirtyBadge(true);
  // キャッシュを無効化して再描画（アノテーション変更を反映）
  frameCache.delete(currentFrame);
  await renderFrame(currentFrame);
  updateControls();
});

canvas.addEventListener('contextmenu', async (e) => {
  e.preventDefault();
  if (!currentVideo || loadingFor) return;
  labels[currentFrame] = { frame_num: currentFrame, visible: 0, x: 0, y: 0 };
  dirty = true;
  setDirtyBadge(true);
  frameCache.delete(currentFrame);
  await renderFrame(currentFrame);
  updateControls();
});

// ─── Save ─────────────────────────────────────────────────────────────────

async function saveLabels() {
  if (!currentVideo) return;

  // スパース配列や未定義エントリをデフォルト値で補完してから送信
  const data = Array.from({ length: totalFrames }, (_, i) =>
    labels[i] ?? { frame_num: i, visible: 0, x: 0.0, y: 0.0 }
  );

  const res = await fetch('/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_path: currentVideo, data }),
  });

  const msg = document.getElementById('savedMsg');
  if (!res.ok) {
    msg.style.color = '#f66';
    msg.textContent = `保存失敗 (${res.status})`;
    setTimeout(() => { msg.textContent = ''; msg.style.color = ''; }, 3000);
    return;
  }

  dirty = false;
  setDirtyBadge(false);
  msg.style.color = '';
  msg.textContent = '保存完了';
  setTimeout(() => { msg.textContent = ''; }, 2000);

  const dot = document.getElementById(`dot_${CSS.escape(currentVideo)}`);
  if (dot) dot.className = 'dot done';

  updateControls();
}

// ─── Controls ─────────────────────────────────────────────────────────────

function setDirtyBadge(isDirty) {
  document.getElementById('dirtyBadge').textContent = isDirty ? '未保存' : '';
}

function updateControls() {
  document.getElementById('frameInfo').textContent =
    `Frame: ${currentFrame + 1} / ${totalFrames}`;

  const label = labels[currentFrame];
  const badge = document.getElementById('stateBadge');
  if (label && label.visible) {
    badge.textContent = 'VISIBLE';
    badge.className = 'state-badge state-visible';
  } else {
    badge.textContent = 'HIDDEN';
    badge.className = 'state-badge state-hidden';
  }

  const visibleCount = labels.filter(l => l.visible).length;
  document.getElementById('progressInfo').textContent =
    `ボール: ${visibleCount} / ${totalFrames} フレーム`;
}

// ─── Keyboard shortcuts ───────────────────────────────────────────────────

document.addEventListener('keydown', async (e) => {
  if (e.target.tagName === 'INPUT') return;
  switch (e.key) {
    case 'ArrowRight': e.preventDefault(); await navigate(1);   break;
    case 'ArrowLeft':  e.preventDefault(); await navigate(-1);  break;
    case 'n': case 'N': await navigate(1);   break;
    case 'p': case 'P': await navigate(-1);  break;
    case 'f': case 'F': await navigate(36);  break;
    case 'b': case 'B': await navigate(-36); break;
    case 'z': case 'Z': await goFirst();     break;
    case 'x': case 'X': await goLast();      break;
    case 's': case 'S': await saveLabels();  break;
    case '=': case '+':
      circleSize = Math.min(30, circleSize + 1);
      frameCache.delete(currentFrame);
      await renderFrame(currentFrame);
      break;
    case '-':
      circleSize = Math.max(2, circleSize - 1);
      frameCache.delete(currentFrame);
      await renderFrame(currentFrame);
      break;
  }
});

// ─── Init ─────────────────────────────────────────────────────────────────

loadFileList();
</script>
</body>
</html>
"""


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
