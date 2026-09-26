"""Maintained check: every curriculum rung builds and the env runs on it.

Run: .venv/bin/python prototype/stair_climb/test_rungs.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

from stair_env import StairEnv

for rung in (0, 1, 2, 3):
    env = StairEnv(rung=rung)
    obs, _ = env.reset(seed=0)
    assert np.all(np.isfinite(obs)), f"rung {rung}: non-finite reset obs"
    assert env.observation_space.contains(obs), f"rung {rung}: obs outside space"
    total = 0.0
    for _ in range(200):
        obs, rew, term, trunc, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)) and np.isfinite(rew), f"rung {rung}: non-finite step"
        total += rew
        if term or trunc:
            break
    print(f"rung {rung}: 200-sample rollout finite, reward {total:+.2f}")

# Shaping guard: reward weights must keep their intended signs. A sign flip
# (e.g. -W_LIFT) would silently train the opposite behavior for hours.
import stair_env as se
assert se.W_TRACK > 0 and se.W_LIFT > 0 and se.W_ALT > 0, "bonus weights must be positive"
assert se.W_YCENTER > 0 and se.W_TILT > 0 and se.W_SLIP > 0, "penalty weights must be positive"
assert se.LIFT_BASE < se.LIFT_CAP <= 0.20, "lift band must be sane"
print("shaping signs OK")

# Bypass guard: outside the corridor (|y|>0.6) the episode must terminate
# WITHOUT success, so walking around the stairs can never pass a gate.
env = StairEnv(rung=1, build=False)
env.reset(seed=0)
env.data.qpos[1] = 0.8  # free-joint lateral position
_, _, term, _, info = env.step(env.action_space.sample())
assert term and not info["success"], "corridor breach must terminate without success"
print("corridor guard OK")
print("TEST_RUNGS OK")
