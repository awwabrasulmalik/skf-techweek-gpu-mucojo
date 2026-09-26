"""Gate + held-out evaluation. Prints success tables for pitch/results.md.

Usage: .venv/bin/python prototype/stair_climb/eval.py --ckpt checkpoints/rung0.zip --rung 0
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from stable_baselines3 import PPO

from stair_env import StairEnv

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--rung", type=int, default=0)
ap.add_argument("--episodes", type=int, default=10)
args = ap.parse_args()

model = PPO.load(args.ckpt)
env = StairEnv(rung=args.rung, build=False)
s, lengths, xs, lifts = 0, [], [], []
for i in range(args.episodes):
    obs, _ = env.reset()
    done, n, maxlift = False, 0, 0.0
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(act)
        done = term or trunc
        n += 1
        maxlift = max(maxlift, abs(float(env.data.xpos[env.ankles[0]][2] - env.data.xpos[env.ankles[1]][2])))
    s += info["success"]
    lengths.append(n)
    xs.append(info["pelvis_x"])
    lifts.append(maxlift)
print(f"rung={args.rung} ckpt={args.ckpt}")
print(f"GATE: success {s}/{args.episodes} (need 7/10 to promote)")
print(f"mean ep len {sum(lengths) / len(lengths):.0f}/1000, mean final x {sum(xs) / len(xs):+.2f} m")
print(f"mean max foot-height-split {sum(lifts) / len(lifts):.3f} m (marching needs >0.10)")
