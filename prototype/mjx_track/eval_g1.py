"""Gate eval for G1 GPU-track policies. Restores an orbax ckpt, fixed commands.

Mirrors stair_climb/eval.py: prints a GATE table for results.md and can save
a body-trajectory npz for render_g1.py.

Usage:
  .venv-mjx/bin/python prototype/mjx_track/eval_g1.py \\
      --ckpt checkpoints_mjx/g1_stand_TAG/<step> --phase stand --episodes 10
  .venv-mjx/bin/python prototype/mjx_track/eval_g1.py \\
      --ckpt checkpoints_mjx/g1_walk_TAG/<step> --phase walk \\
      --command 0.5,0,0 --save-traj videos_mjx/g1_walk_TAG.npz
"""

import argparse
import functools
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402

import jax_compat  # noqa: E402,F401  (shim must load before brax)
import mujoco  # noqa: E402
import stairs_env  # noqa: E402
import numpy as np  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from etils import epath  # noqa: E402

from mujoco_playground import locomotion, wrapper  # noqa: E402
from mujoco_playground.config import locomotion_params  # noqa: E402

REPO = epath.Path(__file__).resolve().parent.parent.parent


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--env", default="G1JoystickFlatTerrain")
  ap.add_argument("--phase", choices=["stand", "walk"], default="stand")
  ap.add_argument("--episodes", type=int, default=10)
  ap.add_argument("--command", default=None, help="vx,vy,vyaw (default: phase stand -> 0,0,0)")
  ap.add_argument("--save-traj", default=None)
  ap.add_argument("--task", choices=["flat", "stairs"], default="flat")
  args = ap.parse_args()

  cache_dir = REPO / ".jax_cache"
  cache_dir.mkdir(parents=True, exist_ok=True)
  jax.config.update("jax_compilation_cache_dir", str(cache_dir))

  if args.command is None:
    cmd = jp.zeros(3) if args.phase == "stand" else jp.array([0.5, 0.0, 0.0])
  else:
    cmd = jp.array([float(x) for x in args.command.split(",")])
  stand = bool(jp.linalg.norm(cmd) < 0.01)

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
  # Fixed-command eval: the env resamples a random command every 500 steps;
  # pin it so the whole episode follows the requested command.
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

  results, best, best_key = [], None, None
  for ep in range(args.episodes):
    rng = jax.random.PRNGKey(ep)
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)
    state.info["command"] = cmd
    traj, survived, max_x, ok = [], 0, -1e9, False
    for _ in range(env_cfg.episode_length):
      rng, act_rng = jax.random.split(rng)
      ctrl, _ = jit_inference_fn(state.obs, act_rng)
      state = jit_step(state, ctrl)
      state.info["command"] = cmd
      traj.append(np.asarray(state.data.xpos))
      max_x = max(max_x, float(np.asarray(state.data.qpos[0])))
      ok = ok or bool(state.info.get("success", False))
      survived += 1
      if bool(state.done):
        break
    traj = np.stack(traj)
    assert traj.shape[1] == env.mj_model.nbody, (
        f"xpos bodies {traj.shape[1]} != mj nbody {env.mj_model.nbody}")
    root = traj[:, pelvis, :]
    drift = float(np.max(np.linalg.norm(root[:, :2] - root[0, :2], axis=1)))
    height = float(np.mean(root[:, 2]))
    min_h = float(np.min(root[:, 2]))
    # Odometry: commands are BODY-frame (get_local_linvel), so world-frame
    # progress is meaningless with random spawn yaw. Measure path + straight.
    path = float(np.sum(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1)))
    net = float(np.linalg.norm(root[-1, :2] - root[0, :2]))
    straight = net / max(path, 1e-6)
    full = survived >= env_cfg.episode_length
    # Height floor (pelvis stand ~0.74): kneeling/crawling must fail the gate.
    if args.task == "stairs":
      success = ok
      key = (ok, max_x)
    elif stand:
      success = full and drift < 0.5 and min_h > 0.55
      key = (survived, -drift)
    else:
      success = full and path > 4.0 and straight > 0.6 and height > 0.55
      key = (survived, path)
    results.append((success, survived, drift, height, min_h, path, straight, max_x))
    if best_key is None or key > best_key:
      best_key, best = key, traj
    print(f"ep {ep}: success={success} len={survived} drift={drift:.2f}m "
          f"path={path:.1f}m str={straight:.2f} h={height:.2f}m max_x={max_x:.2f}m",
          flush=True)

  n_ok = sum(1 for r in results if r[0])
  print(f"ckpt={args.ckpt}")
  need = "need 10/10 stand" if stand else ("need 7/10 stairs" if args.task == "stairs" else "need 7/10 walk")
  print(f"GATE: success {n_ok}/{args.episodes} ({need})")
  print(f"mean len {np.mean([r[1] for r in results]):.0f}/{env_cfg.episode_length}, "
        f"mean drift {np.mean([r[2] for r in results]):+.2f} m, "
        f"mean height {np.mean([r[3] for r in results]):.2f} m")
  if not stand:
    print(f"mean path {np.mean([r[5] for r in results]):.1f} m, "
          f"mean straight {np.mean([r[6] for r in results]):.2f}")
  if args.task == "stairs":
    print(f"mean max_x {np.mean([r[7] for r in results]):.2f} m (goal x>3.3, z>0.95)")

  if args.save_traj:
    m = env.mj_model
    edges = [(int(m.body_parentid[i]), i) for i in range(1, m.nbody)
             if int(m.body_parentid[i]) != 0]
    out = epath.Path(args.save_traj)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.task == "stairs":
      np.savez(out, xpos=best, edges=np.array(edges), dt=env.dt,
               boxes=np.array(stairs_env.BOXES))
    else:
      np.savez(out, xpos=best, edges=np.array(edges), dt=env.dt)
    print(f"SAVED {out} ({best.shape[0]} frames, {best.shape[1]} bodies)")


if __name__ == "__main__":
  sys.exit(main())
