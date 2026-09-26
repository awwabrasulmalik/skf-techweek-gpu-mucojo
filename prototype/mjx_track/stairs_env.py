"""Stairs rung env: playground G1 Joystick on a stairs scene.

Reuses everything from mujoco_playground (pinned 4057c14): rewards,
termination, PD actuation, domain randomization, obs/action spaces (so
walk checkpoints resume directly). Adds: stairs scene, narrow spawn
yaw (stairs are straight ahead), success bonus + termination on
reaching the landing.

Success criteria (reward + gate share these):
  root x > GOAL_X (3.3: past the last step at 2.90 + margin) AND
  root z > GOAL_Z (0.95: above mid-stairs, i.e. on the 0.45 landing).
  Success pays +10 once and terminates the episode (like the CPU track).
"""
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import pathlib

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground._src.locomotion.g1.joystick import Joystick

GOAL_X = 3.3
GOAL_Z = 0.95
SUCCESS_BONUS = 10.0
SPAWN_X = 0.25  # narrow: the stairs are 2 m wide, approach centered.
SPAWN_Y = 0.10  # wide spawn + yaw drift walked robots off the stair edge.
SPAWN_YAW = 0.10  # stairs are straight ahead; wide yaw wastes envs.
# Dense climb shaping (the +10 landing bonus alone is never seen, so it
# gives zero gradient). Paid per step, on top of the joystick tracking.
CLIMB_X0 = 1.0  # ramp starts at the stair base.
CLIMB_X_SCALE = 0.05  # +0.125/step max at the landing.
CLIMB_Z0 = 0.80  # pay height-on-stairs only past the approach...
CLIMB_Z_MIN_X = 1.2  # ...so flat walking never earns it.
CLIMB_Z_SCALE = 0.5  # +0.10/step max (0.2 m above flat root height).
# Anti-crouch: run 1508 survived by cowering (h 0.44 vs 0.78 flat).
# One-sided cost below threshold; never punishes standing tall/climbing.
CROUCH_Z = 0.70
CROUCH_SCALE = 1.0  # full crouch (0.44) costs -0.26/step.
# Plant bonus: either foot in stance contact with a step while tall.
# Kills the knees-rest/feet-dangle exploit (feet float -> no bonus).
# Smaller than climb shaping; swing foot airborne is fine (either/or).
PLANT_SCALE = 0.03
STEP_GEOMS = ("step1", "step2", "step3", "step4", "step5", "landing")
FOOT_GEOMS = ("left_foot", "right_foot")

SCENE = str(pathlib.Path(__file__).resolve().parent / "stairs_scene.xml")

# (center, size) boxes mirror stairs_scene.xml (for stick-figure rendering).
BOXES = [
    ((1.64, 0, 0.045), (0.14, 1.0, 0.045)),
    ((1.92, 0, 0.09), (0.14, 1.0, 0.09)),
    ((2.20, 0, 0.135), (0.14, 1.0, 0.135)),
    ((2.48, 0, 0.18), (0.14, 1.0, 0.18)),
    ((2.76, 0, 0.225), (0.14, 1.0, 0.225)),
    ((3.9, 0, 0.225), (1.0, 1.0, 0.225)),
]


def apply_stairs_config(env_cfg: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Stairs overrides on top of the flat G1 config (train AND eval)."""
  env_cfg.swing_height = 0.22  # bigger steps: clear 0.09 risers + foot.
  env_cfg.lin_vel_x = [0.3, 0.7]  # forward only (stairs are +x).
  env_cfg.lin_vel_y = [0.0, 0.0]  # no lateral drift: stay on the 2 m steps.
  env_cfg.ang_vel_yaw = [-0.1, 0.1]
  # Lift the feet (cost: penalizes |foot_z - max| while moving).
  env_cfg.reward_config.max_foot_height = 0.20  # clear 0.09 risers + foot.
  env_cfg.reward_config.scales.feet_clearance = -2.0
  return env_cfg


class StairsJoystick(Joystick):
  """Joystick on the stairs scene with landing success."""

  def __init__(
      self,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    g1_base.G1Env.__init__(
        self, xml_path=SCENE, config=config, config_overrides=config_overrides
    )
    self._post_init()
    gid = lambda n: mujoco.mj_name2id(
        self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, n
    )
    self._step_ids = jp.array([gid(n) for n in STEP_GEOMS])
    self._foot_ids = jp.array([gid(n) for n in FOOT_GEOMS])

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # Mirror of Joystick.reset (playground 4057c14) with ONE change:
    # spawn yaw narrowed to +-SPAWN_YAW (stairs straight ahead).
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-SPAWN_X, SPAWN_X), y=+U(-SPAWN_Y, SPAWN_Y),
    # yaw=+U(-SPAWN_YAW, SPAWN_YAW). Narrow: approach the steps centered.
    rng, key = jax.random.split(rng)
    dx = jax.random.uniform(key, (1,), minval=-SPAWN_X, maxval=SPAWN_X)
    rng, key = jax.random.split(rng)
    dy = jax.random.uniform(key, (1,), minval=-SPAWN_Y, maxval=SPAWN_Y)
    qpos = qpos.at[0].set(qpos[0] + dx[0])
    qpos = qpos.at[1].set(qpos[1] + dy[0])
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-SPAWN_YAW, maxval=SPAWN_YAW)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # qpos[7:]=*U(0.5, 1.5)
    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:].set(
        qpos[7:] * jax.random.uniform(key, (29,), minval=0.5, maxval=1.5)
    )

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # Phase, freq=U(1.0, 1.5)
    rng, key = jax.random.split(rng)
    gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    phase = jp.array([0, jp.pi])

    rng, cmd_rng = jax.random.split(rng)
    cmd = self.sample_command(cmd_rng)

    # Sample push interval.
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        # Phase related.
        "phase_dt": phase_dt,
        "phase": phase,
        # Push related.
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
        "success": jp.zeros((), dtype=bool),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())
    metrics["success"] = jp.zeros(())
    metrics["climb_x"] = jp.zeros(())
    metrics["climb_z"] = jp.zeros(())
    metrics["crouch"] = jp.zeros(())
    metrics["plant"] = jp.zeros(())

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_floor_found_sensor
    ])
    obs = self._get_obs(data, info, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state = super().step(state, action)
    root = state.data.qpos[0:3]
    success = (root[0] > GOAL_X) & (root[2] > GOAL_Z)
    success_f = success.astype(jp.float32)
    # Dense shaping: pay +x progress past the stair base and height
    # gained on the steps, so every episode sees a climb gradient.
    climb_x = CLIMB_X_SCALE * jp.clip(root[0] - CLIMB_X0, 0.0, 2.5)
    climb_z = (
        CLIMB_Z_SCALE
        * jp.clip(root[2] - CLIMB_Z0, 0.0, 0.2)
        * (root[0] > CLIMB_Z_MIN_X).astype(jp.float32)
    )
    crouch = -CROUCH_SCALE * jp.clip(CROUCH_Z - root[2], 0.0, 1.0)
    # Plant bonus: any active foot<->step contact while standing tall.
    con = state.data._impl.contact__dist < 0
    g = state.data._impl.contact__geom
    foot_hit = jp.isin(g[:, 0], self._foot_ids) | jp.isin(
        g[:, 1], self._foot_ids
    )
    step_hit = jp.isin(g[:, 0], self._step_ids) | jp.isin(
        g[:, 1], self._step_ids
    )
    planted = jp.any(con & foot_hit & step_hit).astype(jp.float32)
    tall = (root[2] > CROUCH_Z).astype(jp.float32)
    plant = PLANT_SCALE * planted * tall
    metrics = dict(state.metrics)
    metrics["success"] = success_f
    metrics["climb_x"] = climb_x
    metrics["climb_z"] = climb_z
    metrics["crouch"] = crouch
    metrics["plant"] = plant
    state.info["success"] = success
    return state.replace(
        reward=state.reward
        + SUCCESS_BONUS * success_f
        + climb_x
        + climb_z
        + crouch
        + plant,
        done=jp.maximum(state.done, success_f),  # done is float32 here.
        metrics=metrics,
    )
