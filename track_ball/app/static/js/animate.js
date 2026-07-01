import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// サーバーが埋め込んだ JSON データを読み取る
const { detected: DETECTED, interp: INTERP, sz_bottom: SZ_BOTTOM, sz_top: SZ_TOP } =
    JSON.parse(document.getElementById('traj-data').textContent);

// ──────────────────────────────
// レンダラー
// ──────────────────────────────
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.insertBefore(renderer.domElement, document.getElementById('legend'));

// ──────────────────────────────
// シーン
// ──────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x090b14);

// ──────────────────────────────
// カメラ（Z 上方向）
// ──────────────────────────────
const camera = new THREE.PerspectiveCamera(
  55, window.innerWidth / window.innerHeight, 0.05, 200
);
camera.up.set(0, 0, 1);
camera.position.set(10, 2, 5);

// ──────────────────────────────
// OrbitControls
// ──────────────────────────────
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 8, 1.5);
controls.enableDamping  = true;
controls.dampingFactor  = 0.08;
controls.minDistance    = 1;
controls.maxDistance    = 60;
controls.update();

// ──────────────────────────────
// ライティング
// ──────────────────────────────
scene.add(new THREE.AmbientLight(0x7080a0, 1.0));
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(8, -6, 12);
scene.add(sun);

// ──────────────────────────────
// グリッド（Z = 0 の地面）
// ──────────────────────────────
const grid = new THREE.GridHelper(40, 40, 0x1a2540, 0x111828);
grid.rotation.x = Math.PI / 2;
grid.position.set(0, 10, 0);
scene.add(grid);

// ──────────────────────────────
// ホームベース（白い五角形）
// ──────────────────────────────
{
  const pts = [
    [0.0,   0.0,   0.005], [0.216, 0.216, 0.005], [0.216, 0.432, 0.005],
    [-0.216,0.432, 0.005], [-0.216,0.216, 0.005], [0.0,   0.0,   0.005],
  ].map(([x,y,z]) => new THREE.Vector3(x, y, z));
  scene.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: 0xffffff })
  ));
}

// ──────────────────────────────
// ストライクゾーン（黄色の矩形枠, Y = 0 面）
// ──────────────────────────────
{
  const pts = [
    [-0.216, 0, SZ_BOTTOM], [ 0.216, 0, SZ_BOTTOM],
    [ 0.216, 0, SZ_TOP   ], [-0.216, 0, SZ_TOP   ],
    [-0.216, 0, SZ_BOTTOM],
  ].map(([x,y,z]) => new THREE.Vector3(x, y, z));
  scene.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: 0xffee44 })
  ));
}

// ──────────────────────────────
// 投球軌跡チューブ（オレンジ→赤 グラデーション）
// ──────────────────────────────
const TUBE_SEG = 120;
const TUBE_RAD = 8;

const curve = new THREE.CatmullRomCurve3(
  INTERP.map(p => new THREE.Vector3(p[0], p[1], p[2]))
);
const tubeGeom = new THREE.TubeGeometry(curve, TUBE_SEG, 0.04, TUBE_RAD, false);

const nVert  = tubeGeom.attributes.position.count;
const colBuf = new Float32Array(nVert * 3);
for (let i = 0; i < nVert; i++) {
  const t = Math.min(1, Math.floor(i / (TUBE_RAD + 1)) / TUBE_SEG);
  colBuf[i*3]   = 1.0;
  colBuf[i*3+1] = 0.55 * (1.0 - t);
  colBuf[i*3+2] = 0.0;
}
tubeGeom.setAttribute('color', new THREE.BufferAttribute(colBuf, 3));

scene.add(new THREE.Mesh(
  tubeGeom,
  new THREE.MeshPhongMaterial({
    vertexColors: true,
    shininess: 80,
    emissive: new THREE.Color(0.12, 0.04, 0),
  })
));

// ──────────────────────────────
// 検出点（黄色の小球）
// ──────────────────────────────
const detectedMat = new THREE.MeshPhongMaterial({ color: 0xffdd00, emissive: 0x554400 });
DETECTED.forEach(p => {
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.07, 10, 10), detectedMat);
  m.position.set(p[0], p[1], p[2]);
  scene.add(m);
});

// ──────────────────────────────
// リリース点（青）・到達点（赤）マーカー
// ──────────────────────────────
[
  [INTERP[0],              0x44aaff],
  [INTERP[INTERP.length-1], 0xff4444],
].forEach(([p, col]) => {
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(0.12, 14, 14),
    new THREE.MeshPhongMaterial({
      color: col,
      emissive: new THREE.Color(col).multiplyScalar(0.3),
    })
  );
  m.position.set(p[0], p[1], p[2]);
  scene.add(m);
});

// ──────────────────────────────
// アニメーション球（白）
// ──────────────────────────────
const ball = new THREE.Mesh(
  new THREE.SphereGeometry(0.15, 20, 20),
  new THREE.MeshPhongMaterial({ color: 0xffffff, emissive: 0x555555, shininess: 120 })
);
scene.add(ball);

// ──────────────────────────────
// リサイズ対応
// ──────────────────────────────
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ──────────────────────────────
// UI
// ──────────────────────────────
const speedEl = document.getElementById('speed');
const spvEl   = document.getElementById('spv');
const btnPlay = document.getElementById('btnPlay');

speedEl.addEventListener('input', () => { spvEl.textContent = speedEl.value; });

let tParam = 0, playing = true;
btnPlay.addEventListener('click', () => {
  playing = !playing;
  btnPlay.textContent = playing ? '⏸ 一時停止' : '▶ 再生';
});

// ──────────────────────────────
// レンダリングループ
// ──────────────────────────────
function loop() {
  requestAnimationFrame(loop);
  if (playing) {
    tParam = (tParam + 0.004 * parseInt(speedEl.value)) % 1.0;
    ball.position.copy(curve.getPoint(tParam));
  }
  controls.update();
  renderer.render(scene, camera);
}
loop();
