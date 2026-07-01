import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. データ定義
# X軸: +が1塁側(左)、-が3塁側(右)
# ---------------------------------------------------------
pts_3d = np.array([
    [0.0, 0.0, 0.0],              
    [0.216, 0.432, 0.0],          
    [-0.216, 0.432, 0.0],         
    [0.216, 0.216, 0.0],          
    [-0.216, 0.216, 0.0],         
    [0.305, 18.44, 0.254],        
    [-0.305, 18.44, 0.254],       
    [0.0, 18.0, 0.254],           
    [0.0, 18.0, 2.104],           
    [0.0, -0.6, 0.0],             
    [0.0, -0.6, 1.0]              
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
    [0.46698920, 0.28521649] 
])

# ---------------------------------------------------------
# 2. DLT法の安定化（Hartleyの正規化アルゴリズム）
# ---------------------------------------------------------
def normalize_2d(pts):
    mean = np.mean(pts, axis=0)
    dist = np.mean(np.linalg.norm(pts - mean, axis=1))
    scale = np.sqrt(2) / dist
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])
    pts_hom = np.column_stack((pts, np.ones(len(pts))))
    return (T @ pts_hom.T).T[:, :2], T

def normalize_3d(pts):
    mean = np.mean(pts, axis=0)
    dist = np.mean(np.linalg.norm(pts - mean, axis=1))
    scale = np.sqrt(3) / dist
    T = np.array([
        [scale, 0, 0, -scale * mean[0]],
        [0, scale, 0, -scale * mean[1]],
        [0, 0, scale, -scale * mean[2]],
        [0, 0, 0, 1]
    ])
    pts_hom = np.column_stack((pts, np.ones(len(pts))))
    return (T @ pts_hom.T).T[:, :3], T

pts_2d_norm, T_2d = normalize_2d(pts_2d)
pts_3d_norm, T_3d = normalize_3d(pts_3d)

A = []
for i in range(len(pts_3d_norm)):
    X, Y, Z = pts_3d_norm[i]
    u, v = pts_2d_norm[i]
    A.append([-X, -Y, -Z, -1, 0, 0, 0, 0, u*X, u*Y, u*Z, u])
    A.append([0, 0, 0, 0, -X, -Y, -Z, -1, v*X, v*Y, v*Z, v])

A = np.array(A)
_, _, V = np.linalg.svd(A)
P_norm = V[-1].reshape(3, 4)

P = np.linalg.inv(T_2d) @ P_norm @ T_3d
P = P / P[-1, -1]

# ---------------------------------------------------------
# 3. ストライクゾーンを打者の特徴点ピクセル座標から復元する
# ---------------------------------------------------------
# 打者のひざ・上端の正規化ピクセル座標（u=横, v=縦, 左上原点, 0〜1）
_u_knee, _v_knee = 0.5888252148997135, 0.4337790512575613
_u_top,  _v_top  = 0.5873925501432665, 0.30897803247373445

def _solve_sz_z(u, v, Y=0.0):
    """(u, v) と既知の Y から Z（高さ）を復元する"""
    M = np.array([
        [P[0,0] - u*P[2,0], P[0,2] - u*P[2,2]],
        [P[1,0] - v*P[2,0], P[1,2] - v*P[2,2]]
    ])
    B = np.array([
        u*(P[2,1]*Y + P[2,3]) - (P[0,1]*Y + P[0,3]),
        v*(P[2,1]*Y + P[2,3]) - (P[1,1]*Y + P[1,3])
    ])
    _, Z = np.linalg.solve(M, B)
    return float(Z)

sz_bottom = _solve_sz_z(_u_knee, _v_knee)
sz_top    = _solve_sz_z(_u_top,  _v_top)
print(f"ストライクゾーン: Z = {sz_bottom:.3f}m (下端) 〜 {sz_top:.3f}m (上端)")

# ---------------------------------------------------------
# 4. ボールの軌道を3D空間に復元する
# ---------------------------------------------------------
traj_2d = np.array([
    [0.56800222, 0.52113631], [0.55941942, 0.49414051],
    [0.54951618, 0.47066590], [0.54357424, 0.45892860],
    [0.53697208, 0.44836503], [0.53169035, 0.43897518],
    [0.52376776, 0.43310653], [0.52178712, 0.42958534],
    [0.51914625, 0.42489042], [0.51386453, 0.42723788],
    [0.50990323, 0.42958534], [0.50858280, 0.43662772],
    [0.50726237, 0.43897518]
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

traj_3d = []
N = len(traj_2d)
Y_start = 16.
Y_end = 0.0

for i in range(N):
    Y = Y_start - (Y_start - Y_end) * (i / (N - 1))
    u, v = traj_2d[i] 
    
    M = np.array([
        [P[0,0] - u*P[2,0], P[0,2] - u*P[2,2]],
        [P[1,0] - v*P[2,0], P[1,2] - v*P[2,2]]
    ])
    B = np.array([
        u*(P[2,1]*Y + P[2,3]) - (P[0,1]*Y + P[0,3]),
        v*(P[2,1]*Y + P[2,3]) - (P[1,1]*Y + P[1,3])
    ])
    X, Z = np.linalg.solve(M, B)
    traj_3d.append([X, Y, Z])
    
traj_3d = np.array(traj_3d)

# ---------------------------------------------------------
# 5. プロット（鳥瞰図 ＆ 側面図の2画面）
# ---------------------------------------------------------
# 1行2列のグラフを作成 (横長のキャンバス)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# ==========================================
# 左画面: 鳥瞰図 (Bird's-Eye View) - X vs Y
# ==========================================
ax1.plot(traj_3d[:, 0], traj_3d[:, 1], marker='o', color='red', label='Pitch Trajectory')

hp_x = [0, 0.216,  0.216, -0.216, -0.216, 0]
hp_y = [0, 0.216, 0.432,  0.432,  0.216, 0]
ax1.plot(hp_x, hp_y, color='black', label='Home Plate')

pp_x = [0.305, -0.305]
pp_y = [18.44, 18.44]
ax1.plot(pp_x, pp_y, color='black', linewidth=4, label='Pitcher Plate')

ax1.set_xlim(1.5, -1.5)  
ax1.set_ylim(20.0, -2.0) 
ax1.set_xlabel("← 1st Base Side  |  3rd Base Side →")
ax1.set_ylabel("Y (meters) [Distance]")
ax1.set_title("Bird's-Eye View (X vs Y)")
ax1.legend()
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')


# ==========================================
# 右画面: 側面図 (Side View) - Y vs Z
# ==========================================
# 投球軌道の高さを描画
ax2.plot(traj_3d[:, 1], traj_3d[:, 2], marker='o', color='blue', label='Pitch Height')

# 地面 (Z = 0) を緑の破線で描画
ax2.axhline(0, color='green', linestyle='--', label='Ground Level')

# マウンドの高さ (Z = 0.254m) とプレート位置を描画
ax2.plot([18.44, 18.44], [0, 0.254], color='black', linewidth=4, label='Mound')

# ホームベースの範囲を描画
ax2.plot([0, 0.432], [0, 0], color='black', linewidth=4, label='Home Plate')

# ストライクゾーン（打者の特徴点から復元した高さ）を描画
ax2.plot([0, 0], [sz_bottom, sz_top], color='orange', linewidth=2,
         label=f'Strike Zone ({sz_bottom:.2f}–{sz_top:.2f}m)')

# 軸の設定（右側をピッチャー、左側をホームベースにする）
ax2.set_xlim(20.0, -2.0)
# 高さは見やすいように-0.5m 〜 2.5m程度に設定
ax2.set_ylim(-0.5, 2.5) 

ax2.set_xlabel("Y (meters) [Pitcher → Home Plate]")
ax2.set_ylabel("Z (meters) [Height]")
ax2.set_title("Side View (Y vs Z)")
ax2.legend()
ax2.grid(True)
# 側面図はZ軸のスケールを少し強調するため、aspect='equal'は設定しません

# レイアウトを整えて表示
plt.tight_layout()
plt.savefig("track_ball/runs/pitch_trajectory.png", dpi=300)
plt.show()