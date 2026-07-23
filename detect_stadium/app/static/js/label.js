// ─── State ──────────────────────────────────────────────────────────────────

let currentVideo = '';
let meta = { frames: 0, fps: 30, width: 0, height: 0 };
let frameLabels = {};      // { frameNum(number): [ {class_id, cx, cy, w, h}, ... ] }
let currentFrame = 0;
let step = 60;             // フレーム単位のサンプリング間隔 (秒指定から算出)
let dirty = false;

let currentClass = 0;      // 新規描画時に使うクラス
let selectedBox = -1;      // 現在フレーム内で選択中のboxインデックス (-1: 未選択)

let dragMode = 'none';     // 'none' | 'draw' | 'move' | 'resize'
let dragStart = null;      // {x, y} キャンバス内部座標
let dragOrigBox = null;    // move/resize開始時点のboxコピー
let resizeCorner = null;   // 'nw','ne','sw','se'

// フレーム画像のキャッシュ(高速表示用)。動画選択時にSSEで全フレームを
// 先読みし、以降のフレーム間移動はキャッシュから即座に描画する
// (track_ball/app/label_web.py と同じ方式)。
const frameCache = new Map();
let loadingFor = '';
let activeEventSource = null;

const HANDLE_R = 9;
let pointRadius = 4;     // 点モードでの描画半径(px, キャンバス内部座標)。UIで可変

// ヒットテスト半径は描画サイズに比例させる。点を小さくして精密に打ちたい
// 場面(ホームベースなど)でも、選択のしやすさが極端に犠牲にならないよう
// 描画半径の3倍程度を確保しつつ、下限も設けている。
function pointHitR() {
  return Math.max(pointRadius * 3, 14);
}

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

function classColor(i) {
  const hue = Math.round((i * 360) / Math.max(1, CLASSES.length));
  return `hsl(${hue}, 75%, 58%)`;
}

// ─── File list & stats ──────────────────────────────────────────────────────

async function loadFileList() {
  const res = await fetch(BASE + '/files');
  const files = await res.json();
  document.getElementById('fileCount').textContent = files.length;

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
      const gl = document.createElement('div');
      gl.className = 'group-label';
      gl.textContent = dir;
      container.appendChild(gl);
    }
    for (const f of items) {
      const el = document.createElement('div');
      el.className = 'file-item';
      el.dataset.path = f.path;
      const dotClass = f.annotated_frames > 0 ? 'done' : 'todo';
      el.innerHTML = `
        <div class="dot ${dotClass}" id="dot_${CSS.escape(f.path)}"></div>
        <span class="file-name" title="${f.rel}">${f.rel.split('/').pop()}</span>
        <span class="frame-count" id="count_${CSS.escape(f.path)}">${f.annotated_frames}</span>`;
      el.onclick = () => selectVideo(f.path, el);
      container.appendChild(el);
    }
  }
}

async function refreshStats() {
  const res = await fetch(BASE + '/stats');
  const s = await res.json();
  document.getElementById('totalAnnotated').textContent = s.total_annotated_frames;
  document.getElementById('targetFrames').textContent = s.target;
}

// ─── Class bar ──────────────────────────────────────────────────────────────

function renderClassBar() {
  const bar = document.getElementById('classBar');
  bar.innerHTML = '';
  CLASSES.forEach((name, i) => {
    const btn = document.createElement('button');
    btn.className = 'class-btn' + (i === currentClass ? ' active' : '');
    btn.style.borderColor = classColor(i);
    if (i === currentClass) btn.style.background = classColor(i);
    btn.textContent = `${i + 1}: ${name}`;
    btn.onclick = () => selectClass(i);
    bar.appendChild(btn);
  });
}

async function selectClass(i) {
  if (i < 0 || i >= CLASSES.length) return;
  if (selectedBox >= 0) {
    const boxes = frameLabels[currentFrame] || [];
    if (boxes[selectedBox]) {
      boxes[selectedBox].class_id = i;
      dirty = true;
      setDirtyBadge(true);
    }
  } else {
    currentClass = i;
  }
  renderClassBar();
  await renderFrame(currentFrame);
}

// ─── Video selection ────────────────────────────────────────────────────────

async function selectVideo(path, el) {
  if (dirty) {
    const ok = confirm('未保存の変更があります。保存してから切り替えますか？\n[OK] 保存して切り替え  [キャンセル] 破棄して切り替え');
    if (ok) await saveLabels();
  }

  document.querySelectorAll('.file-item').forEach(e => e.classList.remove('active'));
  if (el) el.classList.add('active');

  currentVideo = path;
  currentFrame = 0;
  selectedBox = -1;
  dirty = false;
  frameCache.clear();

  const metaRes = await fetch(BASE + `/meta?video=${encodeURIComponent(path)}`);
  meta = await metaRes.json();
  canvas.width = meta.width;
  canvas.height = meta.height;
  updateStep();

  const labelsRes = await fetch(BASE + `/labels?video=${encodeURIComponent(path)}`);
  const data = await labelsRes.json();
  frameLabels = {};
  for (const [k, v] of Object.entries(data.frames || {})) frameLabels[parseInt(k, 10)] = v;
  document.getElementById('labelPath').textContent = data.label_path;

  document.getElementById('noVideo').style.display = 'none';
  canvas.style.display = 'block';
  setDirtyBadge(false);

  await renderFrame(0);
  updateControls();
  await preloadAll(path);
}

function updateStep() {
  const secs = parseFloat(document.getElementById('stepSeconds').value) || 2;
  step = Math.max(1, Math.round(secs * (meta.fps || 30)));
}

// ─── Point size (point mode) ───────────────────────────────────────────────

async function updatePointSize() {
  const input = document.getElementById('pointSize');
  if (!input) return;
  const v = parseInt(input.value, 10);
  if (!Number.isNaN(v)) pointRadius = clamp(v, 1, 30);
  input.value = pointRadius;
  if (currentVideo) await renderFrame(currentFrame);
}

async function adjustPointSize(delta) {
  pointRadius = clamp(pointRadius + delta, 1, 30);
  const input = document.getElementById('pointSize');
  if (input) input.value = pointRadius;
  if (currentVideo) await renderFrame(currentFrame);
}

// ─── Frame cache & SSE preloading ──────────────────────────────────────────

function setLoadingUI(isLoading) {
  document.getElementById('navButtons').classList.toggle('nav-disabled', isLoading);
}

function preloadAll(videoPath) {
  if (activeEventSource) { activeEventSource.close(); activeEventSource = null; }

  loadingFor = videoPath;
  setLoadingUI(true);
  const prog = document.getElementById('cacheProgress');
  prog.textContent = `読み込み中... 0 / ${meta.frames}`;

  return new Promise((resolve) => {
    const es = new EventSource(BASE + `/stream_frames?video=${encodeURIComponent(videoPath)}`);
    activeEventSource = es;
    const pending = [];

    es.onmessage = (e) => {
      if (loadingFor !== videoPath) {
        es.close(); activeEventSource = null;
        resolve(); return;
      }

      const msg = JSON.parse(e.data);

      if (msg.done) {
        es.close(); activeEventSource = null;
        Promise.all(pending).then(() => {
          if (loadingFor === videoPath) {
            if (msg.actual !== undefined && msg.actual !== meta.frames) {
              meta.frames = msg.actual;
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
      prog.textContent = `読み込み中... ${msg.i + 1} / ${meta.frames}`;
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

// ─── Frame rendering ────────────────────────────────────────────────────────

function renderFrame(n) {
  const cached = frameCache.get(n);
  if (cached?.complete && cached.naturalWidth > 0) {
    ctx.drawImage(cached, 0, 0, canvas.width, canvas.height);
    drawBoxes(n);
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      if (currentFrame !== n) { resolve(); return; }
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      drawBoxes(n);
      resolve();
    };
    img.onerror = resolve;
    img.src = BASE + `/frame?video=${encodeURIComponent(currentVideo)}&n=${n}`;
  });
}

function boxToPixelRect(box) {
  const w = box.w * canvas.width;
  const h = box.h * canvas.height;
  const x = box.cx * canvas.width - w / 2;
  const y = box.cy * canvas.height - h / 2;
  return { x, y, w, h };
}

function drawBoxes(n) {
  const boxes = frameLabels[n] || [];
  if (ANNOTATION_TYPE === 'point') {
    drawPoints(boxes);
  } else {
    drawBboxes(boxes);
  }
}

function drawBboxes(boxes) {
  boxes.forEach((box, i) => {
    const r = boxToPixelRect(box);
    const color = classColor(box.class_id);
    const selected = i === selectedBox;
    ctx.lineWidth = selected ? 3 : 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(r.x, r.y, r.w, r.h);

    if (selected) {
      ctx.fillStyle = '#fff';
      const corners = [[r.x, r.y], [r.x + r.w, r.y], [r.x, r.y + r.h], [r.x + r.w, r.y + r.h]];
      for (const [cx, cy] of corners) {
        ctx.fillRect(cx - 5, cy - 5, 10, 10);
        ctx.strokeStyle = color;
        ctx.strokeRect(cx - 5, cy - 5, 10, 10);
      }
    }
  });
}

function drawPoints(boxes) {
  boxes.forEach((pt, i) => {
    const x = pt.cx * canvas.width;
    const y = pt.cy * canvas.height;
    const color = classColor(pt.class_id);
    const selected = i === selectedBox;
    const r = selected ? pointRadius + 3 : pointRadius;

    ctx.beginPath();
    ctx.moveTo(x - r - 5, y);
    ctx.lineTo(x + r + 5, y);
    ctx.moveTo(x, y - r - 5);
    ctx.lineTo(x, y + r + 5);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = selected ? 3 : 2;
    ctx.strokeStyle = '#fff';
    ctx.stroke();
  });
}

// ─── Navigation ─────────────────────────────────────────────────────────────

async function navigate(delta) {
  if (!currentVideo || loadingFor) return;
  const next = Math.max(0, Math.min(meta.frames - 1, currentFrame + delta));
  if (next === currentFrame) return;
  currentFrame = next;
  selectedBox = -1;
  maybeCopyFromPrevFrame(next);
  updateControls();
  await renderFrame(next);
}

async function jumpToFrame() {
  if (!currentVideo || loadingFor) return;
  const input = document.getElementById('jumpFrame');
  const n = parseInt(input.value, 10);
  if (Number.isNaN(n)) return;
  currentFrame = Math.max(0, Math.min(meta.frames - 1, n));
  selectedBox = -1;
  maybeCopyFromPrevFrame(currentFrame);
  updateControls();
  await renderFrame(currentFrame);
}

// ─── Copy from previous labeled frame (前フレームをコピーして編集開始点にする) ──
// playerのように対象物がフレームごとに動く場合、位置がずれた分だけドラッグで
// 微調整すればよいように、未レビューのフレームへ移動した際は現在選択中の
// クラス(currentClass)についてのみ、直近でそのクラスがラベル付けされた
// フレームのbox/pointをコピーして開始点にする。他クラスの既存アノテーション
// には触れない(クラスごとに別々のペースで前方へ進めていく想定のため)。
// 「対象物なし」で確定した(空配列の)フレームはコピー元にもコピー先にもしない
// (誤って「対象物なし」を後続フレームへ広げてしまわないため)。

function autoCopyEnabled() {
  const el = document.getElementById('autoCopyPrev');
  return el ? el.checked : true;
}

function findPrevFrameWithClass(n, classId) {
  for (let k = n - 1; k >= 0; k--) {
    const boxes = frameLabels[k];
    if (boxes && boxes.some(b => b.class_id === classId)) return k;
  }
  return -1;
}

function maybeCopyFromPrevFrame(n) {
  if (!autoCopyEnabled()) return;
  const existing = frameLabels[n];
  if (existing !== undefined) {
    if (existing.length === 0) return;  // 「対象物なし」確定フレームには追加しない
    if (existing.some(b => b.class_id === currentClass)) return;  // このクラスは入力済み
  }

  const prev = findPrevFrameWithClass(n, currentClass);
  if (prev === -1) return;
  const prevBox = frameLabels[prev].find(b => b.class_id === currentClass);
  if (!prevBox) return;

  if (!frameLabels[n]) frameLabels[n] = [];
  frameLabels[n].push({ ...prevBox });
  dirty = true;
  setDirtyBadge(true);
}

// ─── Empty marking ──────────────────────────────────────────────────────────

async function markEmpty() {
  if (!currentVideo) return;
  const boxes = frameLabels[currentFrame];
  if (boxes && boxes.length > 0) {
    flashMsg('先にアノテーションを削除してください', true);
    return;
  }
  frameLabels[currentFrame] = [];
  dirty = true;
  setDirtyBadge(true);
  updateControls();
}

// ─── Propagate to following frames ─────────────────────────────────────────
// 球場のマーカーなどは位置がほとんど変わらないため、現在フレームのアノテー
// ションをそのまま以降のフレームへコピーしてレビュー作業を省略できるように
// する。現在選択中のクラス(currentClass)のbox/pointのみが対象で、対象フレーム
// の他クラスの既存アノテーションはそのまま残す(そのクラスの既存分のみ上書き)。

async function propagateToEnd() {
  if (!currentVideo) return;
  const boxes = frameLabels[currentFrame] || [];
  const classBoxes = boxes.filter(b => b.class_id === currentClass);
  if (classBoxes.length === 0) {
    flashMsg('現在選択中のクラスのアノテーションが現在フレームにありません', true);
    return;
  }

  const endInput = document.getElementById('propagateEnd');
  let end = parseInt(endInput.value, 10);
  if (Number.isNaN(end)) end = meta.frames - 1;
  end = clamp(end, currentFrame, meta.frames - 1);

  const n = end - currentFrame;
  if (n <= 0) {
    flashMsg('終了フレームは現在フレームより後を指定してください', true);
    return;
  }

  const className = CLASSES[currentClass] || `class${currentClass}`;
  const ok = confirm(
    `現在フレーム(${currentFrame})の「${className}」のアノテーションを ` +
    `${currentFrame + 1} 〜 ${end} (${n} フレーム) に適用します。\n` +
    `対象フレームの「${className}」の既存アノテーションは上書きされます(他クラスはそのまま)。よろしいですか？`
  );
  if (!ok) return;

  for (let f = currentFrame + 1; f <= end; f++) {
    const target = (frameLabels[f] || []).filter(b => b.class_id !== currentClass);
    for (const b of classBoxes) target.push({ ...b });
    frameLabels[f] = target;
  }
  dirty = true;
  setDirtyBadge(true);
  updateControls();
  flashMsg(`${n} フレームに適用しました`, false);
}

// ─── Mouse interaction ──────────────────────────────────────────────────────

function toCanvasCoords(e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) / rect.width * canvas.width,
    y: (e.clientY - rect.top) / rect.height * canvas.height,
  };
}

function hitHandle(box, pt) {
  const r = boxToPixelRect(box);
  const corners = {
    nw: [r.x, r.y], ne: [r.x + r.w, r.y],
    sw: [r.x, r.y + r.h], se: [r.x + r.w, r.y + r.h],
  };
  for (const [name, [cx, cy]] of Object.entries(corners)) {
    if (Math.abs(pt.x - cx) <= HANDLE_R && Math.abs(pt.y - cy) <= HANDLE_R) return name;
  }
  return null;
}

function hitBody(box, pt) {
  const r = boxToPixelRect(box);
  return pt.x >= r.x && pt.x <= r.x + r.w && pt.y >= r.y && pt.y <= r.y + r.h;
}

function hitPoint(pt, p) {
  const x = pt.cx * canvas.width, y = pt.cy * canvas.height;
  return Math.hypot(p.x - x, p.y - y) <= pointHitR();
}

canvas.addEventListener('mousedown', (e) => {
  if (!currentVideo || loadingFor) return;
  e.preventDefault();
  const pt = toCanvasCoords(e);
  const boxes = frameLabels[currentFrame] || [];

  if (ANNOTATION_TYPE === 'point') {
    // 既存の点を上から順にヒットテスト
    for (let i = boxes.length - 1; i >= 0; i--) {
      if (hitPoint(boxes[i], pt)) {
        selectedBox = i;
        dragMode = 'move';
        dragOrigBox = { ...boxes[i] };
        dragStart = pt;
        renderCurrent();
        return;
      }
    }
    // 新規点を配置(そのままドラッグして微調整できる)
    const point = {
      class_id: currentClass,
      cx: clamp(pt.x / canvas.width, 0, 1),
      cy: clamp(pt.y / canvas.height, 0, 1),
    };
    if (!frameLabels[currentFrame]) frameLabels[currentFrame] = [];
    frameLabels[currentFrame].push(point);
    selectedBox = frameLabels[currentFrame].length - 1;
    dragMode = 'move';
    dragOrigBox = { ...point };
    dragStart = pt;
    dirty = true;
    setDirtyBadge(true);
    renderCurrent();
    return;
  }

  if (selectedBox >= 0 && boxes[selectedBox]) {
    const handle = hitHandle(boxes[selectedBox], pt);
    if (handle) {
      dragMode = 'resize';
      resizeCorner = handle;
      dragOrigBox = { ...boxes[selectedBox] };
      dragStart = pt;
      return;
    }
    if (hitBody(boxes[selectedBox], pt)) {
      dragMode = 'move';
      dragOrigBox = { ...boxes[selectedBox] };
      dragStart = pt;
      return;
    }
  }

  // 他のboxを上から順にヒットテスト
  for (let i = boxes.length - 1; i >= 0; i--) {
    if (hitBody(boxes[i], pt)) {
      selectedBox = i;
      dragMode = 'move';
      dragOrigBox = { ...boxes[i] };
      dragStart = pt;
      renderCurrent();
      return;
    }
  }

  // 新規box描画開始
  selectedBox = -1;
  dragMode = 'draw';
  dragStart = pt;
});

function renderCurrent() {
  renderFrame(currentFrame);
}

canvas.addEventListener('mousemove', async (e) => {
  if (!currentVideo || dragMode === 'none') return;
  const pt = toCanvasCoords(e);
  const boxes = frameLabels[currentFrame] || [];

  if (ANNOTATION_TYPE === 'point') {
    if (dragMode === 'move' && boxes[selectedBox]) {
      const b = boxes[selectedBox];
      b.cx = clamp(pt.x / canvas.width, 0, 1);
      b.cy = clamp(pt.y / canvas.height, 0, 1);
      dirty = true;
      setDirtyBadge(true);
      await renderFrame(currentFrame);
    }
    return;
  }

  if (dragMode === 'draw') {
    await renderFrame(currentFrame);
    const x = Math.min(dragStart.x, pt.x), y = Math.min(dragStart.y, pt.y);
    const w = Math.abs(pt.x - dragStart.x), h = Math.abs(pt.y - dragStart.y);
    ctx.lineWidth = 2;
    ctx.strokeStyle = classColor(currentClass);
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    return;
  }

  if (dragMode === 'move' && boxes[selectedBox]) {
    const dx = (pt.x - dragStart.x) / canvas.width;
    const dy = (pt.y - dragStart.y) / canvas.height;
    const b = boxes[selectedBox];
    const halfW = dragOrigBox.w / 2, halfH = dragOrigBox.h / 2;
    b.cx = clamp(dragOrigBox.cx + dx, halfW, 1 - halfW);
    b.cy = clamp(dragOrigBox.cy + dy, halfH, 1 - halfH);
    dirty = true;
    setDirtyBadge(true);
    await renderFrame(currentFrame);
    return;
  }

  if (dragMode === 'resize' && boxes[selectedBox]) {
    const orig = dragOrigBox;
    let x0 = (orig.cx - orig.w / 2) * canvas.width;
    let y0 = (orig.cy - orig.h / 2) * canvas.height;
    let x1 = (orig.cx + orig.w / 2) * canvas.width;
    let y1 = (orig.cy + orig.h / 2) * canvas.height;

    if (resizeCorner.includes('w')) x0 = pt.x;
    if (resizeCorner.includes('e')) x1 = pt.x;
    if (resizeCorner.includes('n')) y0 = pt.y;
    if (resizeCorner.includes('s')) y1 = pt.y;

    const minPx = 6;
    if (x1 - x0 < minPx || y1 - y0 < minPx) return;

    const b = boxes[selectedBox];
    b.cx = clamp(((x0 + x1) / 2) / canvas.width, 0, 1);
    b.cy = clamp(((y0 + y1) / 2) / canvas.height, 0, 1);
    b.w = (x1 - x0) / canvas.width;
    b.h = (y1 - y0) / canvas.height;
    dirty = true;
    setDirtyBadge(true);
    await renderFrame(currentFrame);
  }
});

window.addEventListener('mouseup', async (e) => {
  if (!currentVideo || dragMode === 'none') return;
  const pt = toCanvasCoords(e);

  if (ANNOTATION_TYPE === 'point') {
    dragMode = 'none';
    dragStart = null;
    dragOrigBox = null;
    resizeCorner = null;
    updateControls();
    return;
  }

  if (dragMode === 'draw') {
    const x = Math.min(dragStart.x, pt.x), y = Math.min(dragStart.y, pt.y);
    const w = Math.abs(pt.x - dragStart.x), h = Math.abs(pt.y - dragStart.y);
    if (w >= 6 && h >= 6) {
      const box = {
        class_id: currentClass,
        cx: clamp((x + w / 2) / canvas.width, 0, 1),
        cy: clamp((y + h / 2) / canvas.height, 0, 1),
        w: w / canvas.width,
        h: h / canvas.height,
      };
      if (!frameLabels[currentFrame]) frameLabels[currentFrame] = [];
      frameLabels[currentFrame].push(box);
      selectedBox = frameLabels[currentFrame].length - 1;
      dirty = true;
      setDirtyBadge(true);
    }
    await renderFrame(currentFrame);
  }

  dragMode = 'none';
  dragStart = null;
  dragOrigBox = null;
  resizeCorner = null;
  updateControls();
});

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

// ─── Save ───────────────────────────────────────────────────────────────────

async function saveLabels() {
  if (!currentVideo) return;

  const res = await fetch(BASE + '/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_path: currentVideo, frames: frameLabels }),
  });

  const msg = document.getElementById('savedMsg');
  if (!res.ok) {
    flashMsg(`保存失敗 (${res.status})`, true);
    return;
  }
  const data = await res.json();

  dirty = false;
  setDirtyBadge(false);
  flashMsg('保存完了', false);

  const dot = document.getElementById(`dot_${CSS.escape(currentVideo)}`);
  if (dot) dot.className = `dot ${data.annotated_frames > 0 ? 'done' : 'todo'}`;
  const cnt = document.getElementById(`count_${CSS.escape(currentVideo)}`);
  if (cnt) cnt.textContent = data.annotated_frames;

  await refreshStats();
  updateControls();
}

function flashMsg(text, isError) {
  const msg = document.getElementById('savedMsg');
  msg.style.color = isError ? '#f66' : '';
  msg.textContent = text;
  setTimeout(() => { msg.textContent = ''; msg.style.color = ''; }, 2500);
}

// ─── Controls ───────────────────────────────────────────────────────────────

function setDirtyBadge(isDirty) {
  document.getElementById('dirtyBadge').textContent = isDirty ? '未保存' : '';
}

function updateControls() {
  document.getElementById('frameInfo').textContent =
    `Frame: ${currentFrame + 1} / ${meta.frames}`;

  const boxes = frameLabels[currentFrame];
  const badge = document.getElementById('stateBadge');
  if (boxes && boxes.length > 0) {
    badge.textContent = ANNOTATION_TYPE === 'point' ? `${boxes.length} 点` : `${boxes.length} box`;
    badge.className = 'state-badge state-annotated';
  } else if (boxes) {
    badge.textContent = '対象物なし';
    badge.className = 'state-badge state-empty';
  } else {
    badge.textContent = '未レビュー';
    badge.className = 'state-badge state-hidden';
  }

  const annotatedCount = Object.keys(frameLabels).length;
  document.getElementById('progressInfo').textContent =
    `この動画: ${annotatedCount} フレームレビュー済み`;
}

// ─── Keyboard shortcuts ─────────────────────────────────────────────────────

document.addEventListener('keydown', async (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (!currentVideo) return;

  if (e.key >= '1' && e.key <= '9') {
    await selectClass(parseInt(e.key, 10) - 1);
    return;
  }

  switch (e.key) {
    case 'ArrowRight': e.preventDefault(); await navigate(1); break;
    case 'ArrowLeft':  e.preventDefault(); await navigate(-1); break;
    case 'f': case 'F': await navigate(step); break;
    case 'b': case 'B': await navigate(-step); break;
    case 's': case 'S': await saveLabels(); break;
    case 'e': case 'E': await markEmpty(); break;
    case 'p': case 'P': await propagateToEnd(); break;
    case '=': case '+':
      if (ANNOTATION_TYPE === 'point') await adjustPointSize(1);
      break;
    case '-':
      if (ANNOTATION_TYPE === 'point') await adjustPointSize(-1);
      break;
    case 'Delete': case 'Backspace': {
      const boxes = frameLabels[currentFrame];
      if (boxes && selectedBox >= 0 && selectedBox < boxes.length) {
        boxes.splice(selectedBox, 1);
        selectedBox = -1;
        dirty = true;
        setDirtyBadge(true);
        await renderFrame(currentFrame);
        updateControls();
      }
      break;
    }
    case 'Escape':
      selectedBox = -1;
      await renderFrame(currentFrame);
      break;
  }
});

// ─── Init ───────────────────────────────────────────────────────────────────

renderClassBar();
loadFileList();
refreshStats();
