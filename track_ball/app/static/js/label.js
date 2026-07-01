let currentVideo = '';
let labels = [];
let currentFrame = 0;
let totalFrames = 0;
let circleSize = 8;
let dirty = false;

const frameCache = new Map();
let loadingFor = '';
let activeEventSource = null;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// ─── File list ────────────────────────────────────────────────────────────

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

  const metaRes = await fetch(BASE + `/meta?video=${encodeURIComponent(path)}`);
  const meta = await metaRes.json();
  totalFrames = meta.frames;
  canvas.width  = meta.width;
  canvas.height = meta.height;

  const labelsRes = await fetch(BASE + `/labels?video=${encodeURIComponent(path)}`);
  const labelsData = await labelsRes.json();
  labels = labelsData.data;
  document.getElementById('csvPath').textContent = labelsData.csv_path;

  document.getElementById('noVideo').style.display = 'none';
  canvas.style.display = 'block';
  dirty = false;
  setDirtyBadge(false);

  await renderFrame(0);
  updateControls();
  await preloadAll(path);
}

// ─── Frame cache & SSE preloading ────────────────────────────────────────

function setLoadingUI(isLoading) {
  document.getElementById('navButtons').classList.toggle('nav-disabled', isLoading);
}

function preloadAll(videoPath) {
  if (activeEventSource) { activeEventSource.close(); activeEventSource = null; }

  loadingFor = videoPath;
  setLoadingUI(true);
  const prog = document.getElementById('cacheProgress');
  prog.textContent = `読み込み中... 0 / ${totalFrames}`;

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
      if (currentFrame === n) {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        drawOverlay(n);
      }
      resolve();
    };
    img.onerror = resolve;
    img.src = BASE + `/frame?video=${encodeURIComponent(currentVideo)}&n=${n}`;
  });
}

function drawOverlay(n) {
  const label = labels[n];
  if (!label) return;

  const fh = Math.round(canvas.height * 0.045);
  ctx.font = `bold ${fh}px sans-serif`;

  ctx.fillStyle = label.visible ? '#00ff00' : '#888888';
  ctx.fillText(label.visible ? 'VISIBLE' : 'HIDDEN', 20, fh + 10);

  ctx.fillStyle = '#ff4444';
  ctx.font = `${Math.round(fh * 0.85)}px sans-serif`;
  ctx.fillText(`Frame: ${n + 1} / ${totalFrames}`, 20, fh * 2 + 14);

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
  updateControls();
  await renderFrame(next);
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

  const data = Array.from({ length: totalFrames }, (_, i) =>
    labels[i] ?? { frame_num: i, visible: 0, x: 0.0, y: 0.0 }
  );

  const res = await fetch(BASE + '/save', {
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
