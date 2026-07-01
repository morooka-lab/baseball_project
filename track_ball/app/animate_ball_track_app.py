"""
野球ボール 3D 軌道アニメーション
Flask (軌道計算 → JSON) + Three.js (WebGL リアルタイム 3D)

起動: python animate_ball_track_app.py
ブラウザ: http://localhost:5001
  ドラッグ: 回転  /  ホイール: ズーム  /  右ドラッグ: パン
"""

import json
import numpy as np
from scipy.interpolate import CubicSpline
from flask import Flask, render_template, jsonify


# ===================================================================
# 1.  3D 軌道計算（ball3drcn.py のロジック）
# ===================================================================

def compute_traj_3d() -> np.ndarray:
    """DLT でカメラ行列を推定し、2D 検出座標から 3D 軌道を復元する"""
    pts_3d = np.array([
        [ 0.0,    0.0,   0.0  ],
        [ 0.216,  0.432, 0.0  ],
        [-0.216,  0.432, 0.0  ],
        [ 0.216,  0.216, 0.0  ],
        [-0.216,  0.216, 0.0  ],
        [ 0.305,  18.44, 0.254],
        [-0.305,  18.44, 0.254],
        [ 0.0,    18.0,  0.254],
        [ 0.0,    18.0,  2.104],
        [ 0.0,    -0.6,  0.0  ],
        [ 0.0,    -0.6,  1.0  ],
    ])
    pts_2d = np.array([
        [0.49933978, 0.51409393],
        [0.47359136, 0.52700497],
        [0.52376776, 0.52700497],
        [0.47293115, 0.51878885],
        [0.52508820, 0.51878885],
        [0.46170748, 0.94485300],
        [0.53763230, 0.94602673],
        [0.49801935, 0.94133181],
        [0.50066021, 0.47770829],
        [0.47557201, 0.47536083],
        [0.46698920, 0.28521649],
        
    ])

    def norm2d(pts):
        m = pts.mean(0)
        d = np.linalg.norm(pts - m, axis=1).mean()
        s = np.sqrt(2) / d
        T = np.array([[s, 0, -s*m[0]], [0, s, -s*m[1]], [0, 0, 1]])
        return (T @ np.c_[pts, np.ones(len(pts))].T).T[:, :2], T

    def norm3d(pts):
        m = pts.mean(0)
        d = np.linalg.norm(pts - m, axis=1).mean()
        s = np.sqrt(3) / d
        T = np.array([[s,0,0,-s*m[0]], [0,s,0,-s*m[1]],
                      [0,0,s,-s*m[2]], [0,0,0,1]])
        return (T @ np.c_[pts, np.ones(len(pts))].T).T[:, :3], T

    p2n, T2 = norm2d(pts_2d)
    p3n, T3 = norm3d(pts_3d)

    A = []
    for (X, Y, Z), (u, v) in zip(p3n, p2n):
        A += [[-X,-Y,-Z,-1, 0, 0, 0, 0, u*X,u*Y,u*Z,u],
              [ 0, 0, 0, 0,-X,-Y,-Z,-1, v*X,v*Y,v*Z,v]]
    _, _, V = np.linalg.svd(np.array(A))
    P = np.linalg.inv(T2) @ V[-1].reshape(3, 4) @ T3
    P /= P[-1, -1]

    traj_2d = np.array([
        [0.56800222, 0.52113631], [0.55941942, 0.49414051],
        [0.54951618, 0.47066590], [0.54357424, 0.45892860],
        [0.53697208, 0.44836503], [0.53169035, 0.43897518],
        [0.52376776, 0.43310653], [0.52178712, 0.42958534],
        [0.51914625, 0.42489042], [0.51386453, 0.42723788],
        [0.50990323, 0.42958534], [0.50858280, 0.43662772],
        [0.50726237, 0.43897518],
    ])

    # traj_2d = np.array([
    #     [0.561017, 0.506403],
    #     [0.548305, 0.488324],
    #     [0.535593, 0.468738],
    #     [0.524576, 0.450659],
    #     [0.516949, 0.434087],
    #     [0.507627, 0.419021],
    #     [0.500847, 0.406968],
    #     [0.488983, 0.394915],
    #     [0.482203, 0.387382],
    #     [0.477119, 0.378343],
    #     [0.472034, 0.376836],
    #     [0.467797, 0.373823],
    #     [0.463559, 0.372316],
    #     [0.461864, 0.373823]
    # ])
    N = len(traj_2d)
    Y_s, Y_e = 16.0, 0.0
    traj = []
    for i, (u, v) in enumerate(traj_2d):
        Y = Y_s - (Y_s - Y_e) * i / (N - 1)
        M = np.array([[P[0,0]-u*P[2,0], P[0,2]-u*P[2,2]],
                      [P[1,0]-v*P[2,0], P[1,2]-v*P[2,2]]])
        B = np.array([u*(P[2,1]*Y+P[2,3])-(P[0,1]*Y+P[0,3]),
                      v*(P[2,1]*Y+P[2,3])-(P[1,1]*Y+P[1,3])])
        X, Z = np.linalg.solve(M, B)
        traj.append([X, Y, Z])
    return P, np.array(traj)


# ===================================================================
# 2.  スプライン補間（検出 13 点 → なめらかな 80 点）
# ===================================================================

def interpolate_trajectory(traj_3d: np.ndarray, n_frames: int = 80) -> np.ndarray:
    t  = np.linspace(0, 1, len(traj_3d))
    tf = np.linspace(0, 1, n_frames)
    return np.column_stack([CubicSpline(t, traj_3d[:, k])(tf) for k in range(3)])


# ===================================================================
# 3.  ストライクゾーンの復元
#     打者の体の特徴点ピクセル座標（正規化）→ 3D 高さ Z を推定
# ===================================================================

def compute_strike_zone(
    P: np.ndarray,
    u_knee: float, v_knee: float,   # ひざのピクセル座標
    u_top:  float, v_top:  float,   # わきの下〜ズボン上端中点のピクセル座標
    Y: float = 0.0,                 # 打者の奥行き位置（ホームベース基準）
) -> tuple:
    """画像上の打者特徴点から ストライクゾーンの Z 範囲 [m] を返す"""
    def solve_xz(u: float, v: float):
        M = np.array([[P[0,0]-u*P[2,0], P[0,2]-u*P[2,2]],
                      [P[1,0]-v*P[2,0], P[1,2]-v*P[2,2]]])
        B = np.array([u*(P[2,1]*Y+P[2,3]) - (P[0,1]*Y+P[0,3]),
                      v*(P[2,1]*Y+P[2,3]) - (P[1,1]*Y+P[1,3])])
        _, Z = np.linalg.solve(M, B)
        return float(Z)

    return solve_xz(u_knee, v_knee), solve_xz(u_top, v_top)


# ===================================================================
# 3.  起動時の初期化
# ===================================================================

# 打者のひざ・上端の正規化ピクセル座標（u=横, v=縦, 左上原点, 0〜1）
_U_KNEE, _V_KNEE = 0.5888252148997135, 0.4337790512575613
_U_TOP,  _V_TOP  = 0.5873925501432665, 0.30897803247373445

print("3D 軌道を計算中 ...")
_P, _traj_3d = compute_traj_3d()
_traj_interp = interpolate_trajectory(_traj_3d, n_frames=80)
_SZ_BOTTOM, _SZ_TOP = compute_strike_zone(
    _P, _U_KNEE, _V_KNEE, _U_TOP, _V_TOP, Y=0.0
)
print(f"完了: 検出 {len(_traj_3d)} 点 / 補間 {len(_traj_interp)} 点")
print(f"ストライクゾーン: Z = {_SZ_BOTTOM:.3f}m (下端) 〜 {_SZ_TOP:.3f}m (上端)")


# ===================================================================
# 4.  Flask + Three.js（WebGL インタラクティブ 3D）
# ===================================================================



app = Flask(__name__)


@app.route("/")
def index():
    return render_template(
        "animate.html",
        detected_json=json.dumps(_traj_3d.tolist()),
        interp_json=json.dumps(_traj_interp.tolist()),
        sz_bottom_json=json.dumps(round(_SZ_BOTTOM, 4)),
        sz_top_json=json.dumps(round(_SZ_TOP, 4)),
        sz_bottom_str=f"{_SZ_BOTTOM:.2f}",
        sz_top_str=f"{_SZ_TOP:.2f}",
    )


@app.route("/api/trajectory")
def api_trajectory():
    """検出点・補間点の 3D 座標を JSON で返す"""
    return jsonify(
        detected=_traj_3d.tolist(),
        interpolated=_traj_interp.tolist(),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
