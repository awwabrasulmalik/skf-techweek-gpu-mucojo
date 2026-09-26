"""Full-3D MuJoCo video from a trained policy (mesh render, not stick figure).

Restores an orbax ckpt (same wrap stack as eval_g1.py), rolls out the
best of N episodes in MJX, then replays qpos through CPU MuJoCo with
mujoco.Renderer (EGL headless) and writes an mp4.

Usage:
  MUJOCO_GL=egl .venv-mjx/bin/python prototype/mjx_track/render_policy_mujoco.py \\
      --ckpt checkpoints_mjx/g1_walk_0926_1235/100270080 --phase walk \\
      --command 0.5,0,0 --episodes 5 --out videos_mjx/walk_3d.mp4
"""

import argparse
import functools
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "glfw")  # needs X (this box has :0); egl/osmesa lack drivers here.

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402

import jax_compat  # noqa: E402,F401  (shim must load before brax)
import mujoco  # noqa: E402
import stairs_env  # noqa: E402
import numpy as np  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from etils import epath  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

from mujoco_playground import locomotion, wrapper  # noqa: E402
from mujoco_playground.config import locomotion_params  # noqa: E402

REPO = epath.Path(__file__).resolve().parent.parent.parent


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--env", default="G1JoystickFlatTerrain")
  ap.add_argument("--phase", choices=["stand", "walk"], default="walk")
  ap.add_argument("--task", choices=["flat", "stairs"], default="flat")
  ap.add_argument("--command", default="0.5,0,0")
  ap.add_argument("--episodes", type=int, default=5)
  ap.add_argument("--max-frames", type=int, default=500)
  ap.add_argument("--width", type=int, default=960)
  ap.add_argument("--height", type=int, default=640)
  ap.add_argument("--out", default="videos_mjx/policy_3d.mp4")
  args = ap.parse_args()

  cache_dir = REPO / ".jax_cache"
  cache_dir.mkdir(parents=True, exist_ok=True)
  jax.config.update("jax_compilation_cache_dir", str(cache_dir))

  cmd = jp.array([float(x) for x in args.command.split(",")])

  env_cfg = locomotion.get_default_config(args.env)
  if args.phase == "stand":
    env_cfg.lin_vel_x = [0.0, 0.0]
    env_cfg.lin_vel_y = [0.0, 0.0]
    env_cfg.ang_vel_yaw = [0.0, 0.0]
  if args.task == "stairs":
    env_cfg = stairs_env.apply_stairs_config(env_cfg)
    env = stairs_env.StairsJoystick(config=env_cfg)
  else:
    env = locomotion.load(args.env, config=env_cfg)
  ppo_params = locomotion_params.brax_ppo_config(args.env)
  randomizer = locomotion.get_domain_randomizer(args.env)
  pelvis = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
  env.sample_command = lambda rng: cmd

  print(f"restoring {args.ckpt} ...", flush=True)
  make_inference_fn, params, _ = ppo.train(
      num_timesteps=0,
      episode_length=env_cfg.episode_length,
      normalize_observations=True,
      restore_checkpoint_path=epath.Path(args.ckpt).resolve(),
      network_factory=functools.partial(
          ppo_networks.make_ppo_networks, **ppo_params.network_factory
      ),
      num_envs=2,
      environment=env,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      randomization_fn=randomizer,
  )
  jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)

  best, best_key = None, None
  for ep in range(args.episodes):
    rng = jax.random.PRNGKey(ep)
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)
    state.info["command"] = cmd
    traj, root_xy = [], []
    ok, max_x = False, -1e9
    for _ in range(env_cfg.episode_length):
      rng, act_rng = jax.random.split(rng)
      ctrl, _ = jit_inference_fn(state.obs, act_rng)
      state = jit_step(state, ctrl)
      state.info["command"] = cmd
      traj.append(np.asarray(state.data.qpos))
      root_xy.append(np.asarray(state.data.qpos[:2]))
      max_x = max(max_x, float(np.asarray(state.data.qpos[0])))
      ok = ok or bool(state.info.get("success", False))
      if bool(state.done):
        break
    traj = np.stack(traj)
    root_xy = np.stack(root_xy)
    path = float(np.sum(np.linalg.norm(np.diff(root_xy, axis=0), axis=1)))
    key = (ok, max_x) if args.task == "stairs" else (len(traj), path)
    print(f"ep {ep}: success={ok} len={len(traj)} max_x={max_x:.2f}m",
          flush=True)
    if best_key is None or key > best_key:
      best_key, best, best_pelvis = key, traj, pelvis
  print(f"rendering best ep: {len(best)} frames (pelvis body {best_pelvis})")

  # CPU replay with full meshes.
  model = env.mj_model
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, height=args.height, width=args.width)
  cam = mujoco.MjvCamera()
  cam.distance = 4.2
  cam.azimuth = 90  # side view
  cam.elevation = -12
  opt = mujoco.MjvOption()
  frames = best[: args.max_frames]
  fps = round(1.0 / (env.dt * 2))  # render every 2nd step
  out = epath.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8)
  for t in range(0, len(frames), 2):
    data.qpos[:] = frames[t]
    mujoco.mj_forward(model, data)
    cam.lookat[:] = [float(frames[t][0]), 0.0, 0.75]  # track the robot
    renderer.update_scene(data, camera=cam, scene_option=opt)
    writer.append_data(renderer.render())
  writer.close()
  print(f"SAVED {out} ({len(range(0, len(frames), 2))} frames @ {fps}fps)")


if __name__ == "__main__":
  sys.exit(main())
