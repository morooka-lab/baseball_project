const video = document.getElementById('video');
let currentVideoPath = '';
let startTime = null;
let timestamps = [];
let fps = 30;

async function loadFileList() {
  try {
    const res = await fetch(BASE + '/files');
    const files = await res.json();
    const sel = document.getElementById('fileSelect');
    files.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f;
      opt.textContent = f.split('/').pop();
      sel.appendChild(opt);
    });
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
  video.src = BASE + '/video?video=' + encodeURIComponent(path);
  video.load();
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
  const res = await fetch(BASE + '/save', {
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
    const res = await fetch(BASE + '/clip', {
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
