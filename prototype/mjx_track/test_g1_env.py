"""Maintained check: playground G1 env loads, resets, steps finite (GPU track).

Covers the default config plus the stand-phase zero-command override.
Requires CUDA (warp backend). Run:
  .venv-mjx/bin/python prototype/mjx_track/test_g1_env.py
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jp

print("devices:", jax.devices())

from mujoco_playground import locomotion  # noqa: E402


def check_finite(tree, ctx):
  leaves = jax.tree_util.tree_leaves(tree)
  assert leaves, f"{ctx}: empty pytree"
  for leaf in leaves:
    assert jp.all(jp.isfinite(leaf)), f"{ctx}: non-finite leaf"


# 1. Default config: reset + 200 zero-action steps stay finite.
env_cfg = locomotion.get_default_config("G1JoystickFlatTerrain")
env = locomotion.load("G1JoystickFlatTerrain", config=env_cfg)
rng = jax.random.PRNGKey(0)
rng, reset_rng = jax.random.split(rng)
state = jax.jit(env.reset)(reset_rng)
check_finite(state.obs, "reset obs")
jit_step = jax.jit(env.step)
total = 0.0
for _ in range(200):
  rng, act_rng = jax.random.split(rng)
  ctrl = jp.zeros(env.action_size)
  state = jit_step(state, ctrl)
  check_finite(state.obs, "step obs")
  assert jp.isfinite(state.reward), "non-finite reward"
  total += float(state.reward)
print(f"default cfg: 200-step rollout finite, reward {total:+.2f}")

# 2. Stand-phase override: sampled commands must be exactly zero.
env_cfg.lin_vel_x = [0.0, 0.0]
env_cfg.lin_vel_y = [0.0, 0.0]
env_cfg.ang_vel_yaw = [0.0, 0.0]
stand_env = locomotion.load("G1JoystickFlatTerrain", config=env_cfg)
for seed in range(4):
  s = jax.jit(stand_env.reset)(jax.random.PRNGKey(seed))
  cmd = s.info["command"]
  assert cmd.shape == (3,), f"stand cmd shape {cmd.shape}"
  assert bool(jp.all(cmd == 0)), f"stand cmd nonzero: {cmd}"
print("stand override: commands all zero on 4 seeds")
print("TEST_G1_ENV OK")
