"""Render a checkpoint as a stick-figure mp4 (no OpenGL needed).

Usage: .venv/bin/python prototype/stair_climb/eval_render.py --ckpt checkpoints/rung0.zip --out videos/rung0.mp4
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from stable_baselines3 import PPO

from stair_env import StairEnv

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", default="videos/eval.mp4")
ap.add_argument("--max-steps", type=int, default=600)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rung", type=int, default=None, help="rebuild this rung scene first (else loads scene file as-is)")
args = ap.parse_args()

env = StairEnv(rung=args.rung, build=args.rung is not None)
model = PPO.load(args.ckpt, env=env)
obs, _ = env.reset(seed=args.seed)
# Static scenery: named box geoms (steps, obstacles, landing). Floor drawn as grid.
mujoco.mj_forward(env.model, env.data)
boxes = []
for i in range(env.model.ngeom):
    nm = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    if nm and int(env.model.geom_type[i]) == mujoco.mjtGeom.mjGEOM_BOX:
        boxes.append((env.data.geom_xpos[i].copy(), env.model.geom_size[i].copy()))
print(f"scenery: {len(boxes)} boxes")


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
mujoco.mj_forward(env.model, env.data)
nbodies = env.model.nbody
body_names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(nbodies)]

edges = [(int(env.model.body_parentid[i]), i) for i in range(1, nbodies)
         if int(env.model.body_parentid[i]) != 0]  # skip world anchor
fig = plt.figure(figsize=(8, 6))
for t in range(args.max_steps):
    act, _ = model.predict(obs, deterministic=True)
    obs, rew, term, trunc, _ = env.step(act)
    X = env.data.xpos[:nbodies].copy()  # world-frame body positions
    ax = fig.add_subplot(111, projection="3d")
    for a, b in edges:  # skeleton: connect each body to its parent
        ax.plot([X[a, 0], X[b, 0]], [X[a, 1], X[b, 1]], [X[a, 2], X[b, 2]], "b-", lw=2)
    ax.scatter(X[1:, 0], X[1:, 1], X[1:, 2], s=12, c="red")
    for c, s in boxes:
        draw_box(ax, c, s)
    ax.set_xlim(-2, 4); ax.set_ylim(-2, 2); ax.set_zlim(0, 2)
    ax.grid(True)
    ax.set_title(f"t={t}")
    ax.view_init(elev=15, azim=-60)
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    fig.clf()
    if term or trunc:
        break
plt.close(fig)
pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
imageio.mimsave(args.out, frames, fps=30)
print(f"SAVED {args.out} ({len(frames)} frames)")
