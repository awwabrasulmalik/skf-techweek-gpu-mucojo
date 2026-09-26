"""Maintained check: stairs env loads, spawns face stairs, steps finite.

Run: .venv-mjx/bin/python prototype/mjx_track/test_stairs_env.py (needs CUDA)
"""
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jp
import mujoco
import numpy as np

print("devices:", jax.devices())

import stairs_env
from mujoco_playground import locomotion

env_cfg = locomotion.get_default_config("G1JoystickFlatTerrain")
env_cfg = stairs_env.apply_stairs_config(env_cfg)
env = stairs_env.StairsJoystick(config=env_cfg)

# 1. Scene: 5 steps + landing as box geoms.
n_box = sum(
    1 for i in range(env.mj_model.ngeom)
    if env.mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
)
assert n_box >= 6, f"expected 6+ box geoms, got {n_box}"
print(f"scene OK: {n_box} box geoms")

# 2. Spawn narrow + centered (approach the 2 m steps straight on).
for seed in range(8):
  s = jax.jit(env.reset)(jax.random.PRNGKey(seed))
  q = np.asarray(s.data.qpos[3:7])
  assert abs(q[0]) > 0.99, f"spawn yaw too wide: {q}"
  x = float(np.asarray(s.data.qpos[0]))
  y = float(np.asarray(s.data.qpos[1]))
  assert x < 1.0, f"spawn x too close to stairs: {x}"
  assert abs(y) < 0.15, f"spawn y off-center: {y}"
  cmd = np.asarray(jax.block_until_ready(s.info["command"]))
  assert cmd[1] == 0.0, f"lateral command not zero: {cmd}"
print("spawn OK: yaw narrow, centered, zero lateral cmd on 8 seeds")

# 3. 150-step zero-action rollout stays finite; success key present.
rng = jax.random.PRNGKey(0)
state = jax.jit(env.reset)(rng)
jit_step = jax.jit(env.step)
for _ in range(150):
  rng, _ = jax.random.split(rng)
  state = jit_step(state, jp.zeros(env.action_size))
  for leaf in jax.tree_util.tree_leaves(state.obs):
    assert jp.all(jp.isfinite(leaf)), "non-finite obs"
  assert jp.isfinite(state.reward), "non-finite reward"
assert "success" in state.info, "success missing from info"
assert "climb_x" in state.metrics and "climb_z" in state.metrics, (
    "climb shaping metrics missing (reset/step key mismatch?)"
)
assert "crouch" in state.metrics, "crouch metric missing"
assert "plant" in state.metrics, "plant metric missing"
print("rollout OK: 150 steps finite, success + climb + crouch + plant keys")

# 5. Plant bonus fires iff tall + foot planted on a step: teleport
#    upright onto step1, require plant > 0 within 30 steps.
from mujoco import mjx  # noqa: E402  (also imported in §4; kept local)
found = False
for seed in (11, 12, 13):
  st = jax.jit(env.reset)(jax.random.PRNGKey(seed))
  qp = st.data.qpos.at[0].set(1.64)  # step1 center
  qp = qp.at[2].set(0.875)  # home height + one riser
  st = st.replace(data=mjx.forward(env.mjx_model, st.data.replace(qpos=qp)))
  r = jax.random.PRNGKey(seed)
  for _ in range(30):
    r, _ = jax.random.split(r)
    st = jit_step(st, jp.zeros(env.action_size))
    if float(np.asarray(jax.block_until_ready(st.metrics["plant"]))) > 0:
      found = True
      break
  if found:
    break
assert found, "plant bonus never fired while tall on step1"
print("plant OK: foot-on-step bonus fires when tall + planted")

# 4. Body collides with steps: drop the robot onto step1, require >=1
#    active foot/step AND >=1 non-foot body/step contact during the
#    fall. Regression: missing <pair> entries let the robot walk (then
#    fall) through the stairs on the floor plane.
from mujoco import mjx
rng = jax.random.PRNGKey(7)
state = jax.jit(env.reset)(rng)
qpos = state.data.qpos.at[0].set(1.64)  # step1 center
qpos = qpos.at[2].set(1.2)  # drop from above
data = mjx.forward(env.mjx_model, state.data.replace(qpos=qpos))
state = state.replace(data=data)
gid = lambda n: mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, n)
step_ids = [gid(n) for n in
            ("step1", "step2", "step3", "step4", "step5", "landing")]
foot_ids = [gid("left_foot"), gid("right_foot")]
hit_foot = hit_body = False
for i in range(100):
  rng, _ = jax.random.split(rng)
  state = jit_step(state, jp.zeros(env.action_size))
  if i % 5 == 0:
    dist = np.asarray(jax.block_until_ready(state.data._impl.contact__dist))
    geoms = np.asarray(
        jax.block_until_ready(state.data._impl.contact__geom)
    ).astype(int)
    active = dist < 0
    step_hit = np.isin(geoms[:, 0], step_ids) | np.isin(
        geoms[:, 1], step_ids
    )
    foot_hit = np.isin(geoms[:, 0], foot_ids) | np.isin(
        geoms[:, 1], foot_ids
    )
    hit_foot |= bool(np.any(active & step_hit & foot_hit))
    hit_body |= bool(np.any(active & step_hit & ~foot_hit))
assert hit_foot, "no foot/step contacts (missing feet <pair>?)"
assert hit_body, "no body/step contacts (missing thigh/shin <pair>?)"
print("collision OK: foot/step + body/step contacts during drop")
print("TEST_STAIRS_ENV OK")
