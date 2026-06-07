import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


COLORS = [
    "#f72585", "#4cc9f0", "#7bf1a8", "#ffd166", "#a855f7",
    "#ff9f1c", "#2ec4b6", "#e71d36", "#80ed99", "#b5838d",
]


def load_detections(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    df_vis = df[df["visible"] == 1].dropna(subset=["x", "y"]).copy()
    df_vis["x"] = df_vis["x"].astype(float)
    df_vis["y"] = df_vis["y"].astype(float)
    return df, df_vis


def detect_pitches(df_vis: pd.DataFrame, max_gap: int, min_points: int) -> list[pd.DataFrame]:
    """フレームギャップでセグメント分割し、min_points 以上のものを投球として返す"""
    if df_vis.empty:
        return []

    df_vis = df_vis.copy()
    df_vis["frame_gap"] = df_vis["frame_num"].diff().fillna(0).astype(int)
    df_vis["seg_id"] = (df_vis["frame_gap"] > max_gap).cumsum()

    pitches = []
    for _, seg in df_vis.groupby("seg_id"):
        if len(seg) >= min_points:
            pitches.append(seg.reset_index(drop=True))
    return pitches


def _ax_style(ax, facecolor="#0d0d1a"):
    ax.set_facecolor(facecolor)
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


def plot_overview(pitches: list[pd.DataFrame], width: int, height: int, out_path: Path):
    """全投球を重ね合わせたオーバービュー"""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0d0d1a")
    _ax_style(ax)

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_title(f"All pitches overlay  (n={len(pitches)})")

    for i, pitch in enumerate(pitches):
        color = COLORS[i % len(COLORS)]
        ax.plot(pitch["x"], pitch["y"], color=color, linewidth=1.5, alpha=0.7)
        ax.scatter(pitch["x"], pitch["y"], color=color, s=15, zorder=5)
        # 開始点に投球番号を表示
        ax.annotate(
            f"#{i+1}",
            (pitch["x"].iloc[0], pitch["y"].iloc[0]),
            color=color,
            fontsize=8,
            fontweight="bold",
        )

    fig.suptitle(out_path.stem, color="white", fontsize=12)
    save_path = out_path.parent / (out_path.stem + "_overview.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {save_path}")


def plot_pitch_grid(pitches: list[pd.DataFrame], width: int, height: int, out_path: Path):
    """投球ごとの個別グリッド（軌跡 + x/y 時系列）"""
    n = len(pitches)
    if n == 0:
        print("No pitches to plot.")
        return

    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 6, rows * 5))
    fig.patch.set_facecolor("#0d0d1a")

    for i, pitch in enumerate(pitches):
        ax = fig.add_subplot(rows, cols, i + 1)
        _ax_style(ax)
        color = COLORS[i % len(COLORS)]

        # 軌跡
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect("equal")
        ax.plot(pitch["x"], pitch["y"], color=color, linewidth=1.5, alpha=0.8)
        ax.scatter(pitch["x"], pitch["y"], color=color, s=20, zorder=5)

        # 開始・終了マーカー
        ax.scatter(pitch["x"].iloc[0], pitch["y"].iloc[0], color="white", s=60, marker="o", zorder=6)
        ax.scatter(pitch["x"].iloc[-1], pitch["y"].iloc[-1], color="white", s=60, marker="x", zorder=6, linewidths=2)

        t_start = pitch["timestamp_ms"].iloc[0]
        t_end = pitch["timestamp_ms"].iloc[-1]
        duration = t_end - t_start
        ax.set_title(
            f"Pitch #{i+1}  ({t_start:.0f}~{t_end:.0f} ms, {duration:.0f} ms, {len(pitch)} pts)",
            fontsize=9,
        )
        ax.set_xlabel("x [px]", fontsize=8)
        ax.set_ylabel("y [px]", fontsize=8)

    fig.suptitle(f"{out_path.stem}  —  pitch grid", color="white", fontsize=12, y=1.01)
    plt.tight_layout()

    save_path = out_path.parent / (out_path.stem + "_pitches.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {save_path}")


def plot_timeseries_all(pitches: list[pd.DataFrame], out_path: Path):
    """全投球の x(t), y(t) を時系列で重ね合わせ（正規化時刻）"""
    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    fig.patch.set_facecolor("#0d0d1a")
    _ax_style(ax_x)
    _ax_style(ax_y)

    for i, pitch in enumerate(pitches):
        color = COLORS[i % len(COLORS)]
        # 投球開始を 0 ms に正規化
        t = pitch["timestamp_ms"] - pitch["timestamp_ms"].iloc[0]
        ax_x.plot(t, pitch["x"], color=color, linewidth=1.5, alpha=0.8, label=f"#{i+1}")
        ax_y.plot(t, pitch["y"], color=color, linewidth=1.5, alpha=0.8, label=f"#{i+1}")

    ax_x.set_ylabel("x [px]")
    ax_x.set_title("x(t) — all pitches (normalized to pitch start)")
    ax_x.legend(loc="upper right", fontsize=7, ncol=3,
                facecolor="#1a1a2e", labelcolor="white", edgecolor="#444")
    ax_x.grid(True, alpha=0.3)

    ax_y.set_xlabel("time from pitch start [ms]")
    ax_y.set_ylabel("y [px]")
    ax_y.set_title("y(t) — all pitches")
    ax_y.invert_yaxis()
    ax_y.legend(loc="upper right", fontsize=7, ncol=3,
                facecolor="#1a1a2e", labelcolor="white", edgecolor="#444")
    ax_y.grid(True, alpha=0.3)

    fig.suptitle(f"{out_path.stem}  —  timeseries", color="white", fontsize=12)
    plt.tight_layout()

    save_path = out_path.parent / (out_path.stem + "_timeseries.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {save_path}")


def export_pitch_csv(pitches: list[pd.DataFrame], out_path: Path):
    """投球番号列を付加して CSV に保存"""
    rows = []
    for i, pitch in enumerate(pitches):
        p = pitch.copy()
        p.insert(0, "pitch_id", i + 1)
        rows.append(p)
    if rows:
        out = pd.concat(rows, ignore_index=True)
        save_path = out_path.parent / (out_path.stem + "_pitches.csv")
        out[["pitch_id", "frame_num", "timestamp_ms", "visible", "x", "y"]].to_csv(save_path, index=False)
        print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize ball trajectory from detect.py CSV output")
    parser.add_argument("--csv", type=str, required=True, help="Path to detection CSV")
    parser.add_argument("--width", type=int, default=1920, help="Video width in pixels")
    parser.add_argument("--height", type=int, default=1080, help="Video height in pixels")
    parser.add_argument("--gap", type=int, default=10, help="Max frame gap to split pitch segments")
    parser.add_argument("--min-points", type=int, default=6, help="Minimum detections to count as a pitch")
    parser.add_argument("--out", type=str, default=None, help="Output base path (default: same dir as CSV)")
    opt = parser.parse_args()

    csv_path = Path(opt.csv)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    out_base = Path(opt.out) if opt.out else csv_path.with_suffix("")

    df, df_vis = load_detections(str(csv_path))
    pitches = detect_pitches(df_vis, max_gap=opt.gap, min_points=opt.min_points)

    print(f"Total frames   : {len(df)}")
    print(f"Detected frames: {len(df_vis)}")
    print(f"Pitches found  : {len(pitches)}")
    for i, p in enumerate(pitches):
        t0, t1 = p['timestamp_ms'].iloc[0], p['timestamp_ms'].iloc[-1]
        print(f"  #{i+1:02d}  frames {p['frame_num'].iloc[0]:4d}~{p['frame_num'].iloc[-1]:4d}"
              f"  {t0:.0f}~{t1:.0f} ms  ({len(p)} pts)")

    if not pitches:
        print("No pitches detected. Try lowering --min-points or --gap.")
        sys.exit(0)

    plot_overview(pitches, opt.width, opt.height, out_base)
    plot_pitch_grid(pitches, opt.width, opt.height, out_base)
    plot_timeseries_all(pitches, out_base)
    export_pitch_csv(pitches, out_base)


if __name__ == "__main__":
    main()
