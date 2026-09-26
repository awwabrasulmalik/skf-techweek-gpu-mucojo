"""Plot learning curves from a tee'd SB3 stdout log + save mp4 pan.

Usage: .venv/bin/python prototype/stair_climb/plot_curves.py --log train_rung0.log --out videos/curves_rung0.mp4
Expects the run to have been piped: train.py ... 2>&1 | tee train_rung0.log
"""
import argparse
import pathlib
import re
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
ap.add_argument("--out", default="videos/curves.mp4")
args = ap.parse_args()

rows = []
cur = {}
for line in pathlib.Path(args.log).read_text().splitlines():
    m = re.search(r"\|\s*(ep_rew_mean|ep_len_mean|total_timesteps|explained_variance)\s*\|\s*([-\d.]+)", line)
    if m:
        cur[m.group(1)] = float(m.group(2))
        if set(cur) >= {"ep_rew_mean", "ep_len_mean", "total_timesteps"}:
            rows.append((cur["total_timesteps"], cur["ep_rew_mean"], cur["ep_len_mean"]))
            cur = {}
assert rows, f"no SB3 metric rows parsed from {args.log} — was stdout tee'd to it?"
steps = np.array([r[0] for r in rows])

frames = []
fig = plt.figure(figsize=(10, 4))
for k in range(1, len(rows) + 1):
    fig.clf()
    ax1 = fig.add_subplot(121)
    ax1.plot(steps[:k], [r[1] for r in rows[:k]], "b-")
    ax1.set_title("ep_rew_mean")
    ax1.set_xlabel("timesteps")
    ax2 = fig.add_subplot(122)
    ax2.plot(steps[:k], [r[2] for r in rows[:k]], "g-")
    ax2.set_title("ep_len_mean")
    ax2.set_xlabel("timesteps")
    fig.tight_layout()
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
plt.close(fig)
pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
imageio.mimsave(args.out, frames, fps=5)
print(f"SAVED {args.out} ({len(frames)} frames from {len(rows)} log rows)")
