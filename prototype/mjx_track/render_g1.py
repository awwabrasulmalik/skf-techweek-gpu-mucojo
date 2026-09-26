"""Stick-figure mp4 from an eval trajectory npz (no OpenGL needed).

Same visual language as stair_climb/eval_render.py: blue skeleton, red joints,
floor grid, green scenery boxes when present.

Usage:
  .venv-mjx/bin/python prototype/mjx_track/render_g1.py --traj videos_mjx/x.npz --out videos_mjx/x.mp4
"""

import argparse
import pathlib
import sys

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--out", default="videos_mjx/eval.mp4")
ap.add_argument("--max-steps", type=int, default=600)
ap.add_argument("--stride", type=int, default=2)
args = ap.parse_args()

d = np.load(args.traj)
X_all, edges = d["xpos"], d["edges"]
boxes = d["boxes"] if "boxes" in d else np.zeros((0, 2, 3))
print(f"traj: {X_all.shape[0]} frames, {X_all.shape[1]} bodies, {len(boxes)} boxes")


def draw_box(ax, c, s, color="green"):
  x0, x1 = c[0] - s[0], c[0] + s[0]
  y0, y1 = c[1] - s[1], c[1] + s[1]
  z0, z1 = c[2] - s[2], c[2] + s[2]
  pts = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
  for a in range(8):
    for b in range(a + 1, 8):
      if sum(abs(pts[a][k] - pts[b][k]) > 1e-9 for k in range(3)) == 1:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                [pts[a][2], pts[b][2]], color=color, lw=1)


frames = []
fig = plt.figure(figsize=(8, 6))
for t in range(0, min(args.max_steps, X_all.shape[0]), args.stride):
  X = X_all[t]
  ax = fig.add_subplot(111, projection="3d")
  for a, b in edges:
    ax.plot([X[a, 0], X[b, 0]], [X[a, 1], X[b, 1]], [X[a, 2], X[b, 2]], "b-", lw=2)
  ax.scatter(X[1:, 0], X[1:, 1], X[1:, 2], s=12, c="red")
  for c, s in boxes:
    draw_box(ax, c, s)
  ax.set_xlim(-2, 5 if len(boxes) else 4)
  ax.set_ylim(-2, 2)
  ax.set_zlim(0, 2)
  ax.grid(True)
  ax.set_title(f"t={t}")
  ax.view_init(elev=15, azim=-60)
  fig.canvas.draw()
  frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
  fig.clf()
plt.close(fig)
pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
imageio.mimsave(args.out, frames, fps=30)
print(f"SAVED {args.out} ({len(frames)} frames)")
sys.exit(0)
