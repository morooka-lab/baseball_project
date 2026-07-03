// ─── State ──────────────────────────────────────────────────────────────────

let currentVideo = '';
let meta = { frames: 0, fps: 30, width: 0, height: 0 };
let frameDetections = {};   // { frameNum(number): {source, detections:[{class_id,class_name,score,box}]} }
let stabilizedDetections = []; // [{class_id,class_name,point_index,box,n_frames}, ...] (動画全体の統計的代表座標。1クラスに複数点あり得る)
let useStabilized = false;
let currentFrame = 0;
let confThres = 0.3;

let playing = false;
let playTimer = null;
let playbackFps = 15;

// bbox_label_web.pyと同じ、動画選択時にSSEで全フレームを先読みするキャッシュ方式。
const frameCache = new Map();
let loadingFor = '';
let activeEventSource = null;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

function classColor(i) {
  const hue = Math.round((i * 360) / Math.max(1, CLASSES.length));
  return `hsl(${hue}, 75%, 58%)`;
}

// ─── File list ──────────────────────────────────────────────────────────────

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
      el.className = 'file-item' + (f.labeled ? ' labeled' : '');
      el.dataset.path = f.path;
      el.innerHTML = `
        <span class="dot ${f.cached ? 'done' : 'todo'}" title="${f.cached ? '推論済み(キャッシュあり)' : '未推論'}"></span>
        <span class="file-name" title="${f.rel}">${f.rel.split('/').pop()}</span>
        ${f.labeled ? '<span class="badge-labeled" title="ラベル付き(学習に使用済み)">学習済</span>' : ''}`;
      el.onclick = () => selectVideo(f.path, el);
      container.appendChild(el);
    }
  }
}

// ─── Legend ─────────────────────────────────────────────────────────────────

function renderLegend() {
  const el = document.getElementById('legend');
  el.innerHTML = CLASSES.map((name, i) =>
    `<span class="legend-item"><span class="swatch" style="background:${classColor(i)}"></span>${name}</span>`
  ).join('') + '<span class="legend-item legend-human"><span class="swatch-dashed"></span>人手アノテーション(白縁取り)</span>';
}

// ─── Video selection ────────────────────────────────────────────────────────

async function selectVideo(path, el) {
  stopPlay();

  document.querySelectorAll('.file-item').forEach(e => e.classList.remove('active'));
  if (el) el.classList.add('active');

  currentVideo = path;
  currentFrame = 0;
  frameCache.clear();

  document.getElementById('noVideo').style.display = 'none';
  canvas.style.display = 'none';
  document.getElementById('playBtn').disabled = true;
  document.getElementById('frameSlider').disabled = true;
  const prog = document.getElementById('cacheProgress');
  prog.textContent = '推論中... (初回のみ、動画長により数秒〜数十秒かかります)';

  let data;
  try {
    const detRes = await fetch(BASE + `/detections?video=${encodeURIComponent(path)}`);
    data = await detRes.json();
    if (data.error) throw new Error(data.error);
  } catch (err) {
    prog.textContent = `推論に失敗しました: ${err}`;
    return;
  }

  meta = { frames: data.total_frames, fps: data.fps, width: data.width, height: data.height };
  canvas.width = meta.width;
  canvas.height = meta.height;

  frameDetections = {};
  for (const [k, v] of Object.entries(data.frames || {})) {
    frameDetections[parseInt(k, 10)] = { source: 'model', detections: v };
  }
  stabilizedDetections = data.stabilized || [];

  canvas.style.display = 'block';
  prog.textContent = '';

  const slider = document.getElementById('frameSlider');
  slider.max = Math.max(0, meta.frames - 1);
  slider.value = 0;
  slider.disabled = false;
  document.getElementById('playBtn').disabled = false;

  if (el) {
    const dot = el.querySelector('.dot');
    if (dot) { dot.className = 'dot done'; dot.title = '推論済み(キャッシュあり)'; }
  }

  await renderFrame(0);
  updateFrameInfo();
  await preloadAll(path);
}

// ─── Frame cache & SSE preloading (bbox_label_web.pyと同じ方式) ────────────

function preloadAll(videoPath) {
  if (activeEventSource) { activeEventSource.close(); activeEventSource = null; }

  loadingFor = videoPath;
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
              document.getElementById('frameSlider').max = Math.max(0, meta.frames - 1);
              updateFrameInfo();
            }
            loadingFor = '';
            prog.textContent = '';
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
      if (loadingFor === videoPath) { loadingFor = ''; prog.textContent = ''; }
      resolve();
    };
  });
}

// ─── Frame rendering ────────────────────────────────────────────────────────

function renderFrame(n) {
  const cached = frameCache.get(n);
  if (cached?.complete && cached.naturalWidth > 0) {
    ctx.drawImage(cached, 0, 0, canvas.width, canvas.height);
    drawDetections(n);
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      if (currentFrame !== n) { resolve(); return; }
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      drawDetections(n);
      resolve();
    };
    img.onerror = resolve;
    img.src = BASE + `/frame?video=${encodeURIComponent(currentVideo)}&n=${n}`;
  });
}

function drawDetections(n) {
  if (useStabilized) {
    for (const d of stabilizedDetections) {
      drawOnePoint(d.box, d.class_id, false);
    }
    return;
  }

  const entry = frameDetections[n];
  if (!entry) return;
  const isHuman = entry.source === 'human';

  for (const d of entry.detections) {
    if (!isHuman && d.score < confThres) continue;
    drawOnePoint(d.box, d.class_id, isHuman);
  }
}

function drawOnePoint(box, classId, outline) {
  const [x0, y0, x1, y1] = box;
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  const color = classColor(classId);

  ctx.beginPath();
  ctx.arc(cx, cy, 6, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  if (outline) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#fff';
    ctx.stroke();
  }
}

function onStabilizeToggle(checked) {
  useStabilized = checked;
  if (currentVideo) renderFrame(currentFrame);
}

// ─── Navigation ─────────────────────────────────────────────────────────────

async function navigate(delta) {
  if (!currentVideo) return;
  const next = Math.max(0, Math.min(meta.frames - 1, currentFrame + delta));
  if (next === currentFrame) return;
  currentFrame = next;
  document.getElementById('frameSlider').value = next;
  updateFrameInfo();
  await renderFrame(next);
}

async function onSliderInput(v) {
  if (!currentVideo) return;
  stopPlay();
  currentFrame = Math.max(0, Math.min(meta.frames - 1, parseInt(v, 10)));
  updateFrameInfo();
  await renderFrame(currentFrame);
}

function updateFrameInfo() {
  document.getElementById('frameInfo').textContent = `Frame: ${currentFrame + 1} / ${meta.frames}`;
}

function onConfChange(v) {
  confThres = parseFloat(v);
  document.getElementById('confValue').textContent = confThres.toFixed(2);
  if (currentVideo) renderFrame(currentFrame);
}

// ─── Playback ───────────────────────────────────────────────────────────────

function updatePlaybackFps() {
  playbackFps = parseInt(document.getElementById('playbackFps').value, 10);
  if (playing) { stopPlay(); startPlay(); }
}

function startPlay() {
  if (!currentVideo || playing) return;
  playing = true;
  document.getElementById('playBtn').textContent = '⏸ 一時停止';
  playTimer = setInterval(async () => {
    if (currentFrame >= meta.frames - 1) { stopPlay(); return; }
    currentFrame += 1;
    document.getElementById('frameSlider').value = currentFrame;
    updateFrameInfo();
    await renderFrame(currentFrame);
  }, 1000 / playbackFps);
}

function stopPlay() {
  playing = false;
  document.getElementById('playBtn').textContent = '▶ 再生';
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
}

function togglePlay() {
  if (playing) stopPlay(); else startPlay();
}

// ─── Keyboard shortcuts ─────────────────────────────────────────────────────

document.addEventListener('keydown', async (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (!currentVideo) return;

  switch (e.key) {
    case ' ': e.preventDefault(); togglePlay(); break;
    case 'ArrowRight': e.preventDefault(); stopPlay(); await navigate(1); break;
    case 'ArrowLeft':  e.preventDefault(); stopPlay(); await navigate(-1); break;
    case 'Home': e.preventDefault(); stopPlay(); await navigate(-meta.frames); break;
    case 'End':  e.preventDefault(); stopPlay(); await navigate(meta.frames); break;
  }
});

// ─── Init ───────────────────────────────────────────────────────────────────

renderLegend();
loadFileList();
