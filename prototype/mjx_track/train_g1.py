"""GPU training entrypoint: G1 stand -> walk via mujoco_playground + Brax PPO.

Phase 0 (stand): velocity commands forced to zero -> pure balance task.
Phase 1 (walk): default command ranges, finetune from a phase-0 checkpoint.

Mirrors the CPU track's CLI/logging style: unique --out per run, periodic
orbax checkpoints (one per eval), final params.pkl + config, text log.

Usage (from repo root):
  .venv-mjx/bin/python prototype/mjx_track/train_g1.py --phase stand \\
      --timesteps 30000000 --num-envs 1024 --out checkpoints_mjx/g1_stand_TAG
  .venv-mjx/bin/python prototype/mjx_track/train_g1.py --phase walk \\
      --timesteps 100000000 --num-envs 1024 --resume checkpoints_mjx/g1_stand_TAG/<step> \\
      --out checkpoints_mjx/g1_walk_TAG
"""

import argparse
import functools
import json
import os
import pickle
import sys
import time
from datetime import datetime

# XLA/GPU flags MUST be set before importing jax (playground notebook pattern).
xla_flags = os.environ.get("XLA_FLAGS", "")
if "xla_gpu_triton_gemm_any" not in xla_flags:
  xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

import jax_compat  # noqa: E402,F401  (shim must load before brax)
import stairs_env  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from etils import epath  # noqa: E402
from flax.training import orbax_utils  # noqa: E402
from orbax import checkpoint as ocp  # noqa: E402
from tensorboardX import SummaryWriter  # noqa: E402

from mujoco_playground import locomotion, wrapper  # noqa: E402
from mujoco_playground.config import locomotion_params  # noqa: E402

REPO = epath.Path(__file__).resolve().parent.parent.parent


def parse_args():
  ap = argparse.ArgumentParser()
  ap.add_argument("--phase", choices=["stand", "walk"], default="stand")
  ap.add_argument("--env", default="G1JoystickFlatTerrain")
  ap.add_argument("--task", choices=["flat", "stairs"], default="flat")
  ap.add_argument("--timesteps", type=int, default=30_000_000)
  ap.add_argument("--num-envs", type=int, default=1024)
  ap.add_argument("--num-evals", type=int, default=10)
  ap.add_argument("--out", default=None, help="ckpt dir, e.g. checkpoints_mjx/g1_stand_TAG")
  ap.add_argument("--log", default=None, help="text log, e.g. runs_mjx/g1_stand_TAG.log")
  ap.add_argument("--seed", type=int, default=1)
  ap.add_argument("--pushes", choices=["on", "off"], default="on",
                  help="push disturbances during training (off = easier reshape)")
  ap.add_argument("--resume", default=None, help="orbax ckpt dir to finetune from")
  return ap.parse_args()


def apply_phase(env_cfg, phase, pushes="on"):
  if phase == "stand":
    # Zero command ranges -> sample_command always returns [0,0,0].
    env_cfg.lin_vel_x = [0.0, 0.0]
    env_cfg.lin_vel_y = [0.0, 0.0]
    env_cfg.ang_vel_yaw = [0.0, 0.0]
    # Kill the hop-in-place exploit: feet_phase is already masked to 0 at
    # zero command, but feet_air_time would still pay for air. Stand = still.
    env_cfg.reward_config.scales.feet_air_time = 0.0
  env_cfg.push_config.enable = (pushes == "on")
  return env_cfg


def main():
  args = parse_args()
  tag = datetime.now().strftime("%m%d_%H%M")
  out = epath.Path(args.out or REPO / f"checkpoints_mjx/g1_{args.phase}_{tag}").resolve()
  out.mkdir(parents=True, exist_ok=True)
  log_path = epath.Path(args.log or REPO / f"runs_mjx/g1_{args.phase}_{tag}.log").resolve()
  log_path.parent.mkdir(parents=True, exist_ok=True)

  # Persistent compile cache inside the repo: survives /tmp wipes (WSL crash).
  cache_dir = REPO / ".jax_cache"
  cache_dir.mkdir(parents=True, exist_ok=True)
  jax.config.update("jax_compilation_cache_dir", str(cache_dir))
  jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
  jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
  print(f"devices: {jax.devices()}", flush=True)

  env_cfg = locomotion.get_default_config(args.env)
  env_cfg = apply_phase(env_cfg, args.phase, args.pushes)
  # Contact buffer scales with env count (default 8*8192 assumes 8192 envs).
  env_cfg.naconmax = 8 * args.num_envs
  if args.task == "stairs":
    env_cfg = stairs_env.apply_stairs_config(env_cfg)
  randomizer = locomotion.get_domain_randomizer(args.env)
  ppo_params = locomotion_params.brax_ppo_config(args.env)

  with open(out / "env_config.json", "w") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=2)

  t0 = time.time()
  log_fp = open(log_path, "a")
  tb = SummaryWriter(log_dir=str(out / "tb"))

  def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_fp.write(line + "\n")
    log_fp.flush()

  log(f"phase={args.phase} task={args.task} env={args.env} timesteps={args.timesteps} "
      f"num_envs={args.num_envs} seed={args.seed} pushes={args.pushes} out={out}")
  if args.resume:
    log(f"resuming from {args.resume}")

  def progress(num_steps, metrics):
    dt = time.time() - t0
    rew = metrics.get("eval/episode_reward", float("nan"))
    std = metrics.get("eval/episode_reward_std", float("nan"))
    log(f"step={num_steps} eval_rew={rew:.2f}+/-{std:.2f} "
        f"elapsed={dt:.0f}s steps_per_sec={num_steps / max(dt, 1e-6):.0f}")
    for k, v in metrics.items():
      try:
        tb.add_scalar(k, float(v), num_steps)
      except (TypeError, ValueError):
        pass
    tb.add_scalar("perf/steps_per_sec", num_steps / max(dt, 1e-6), num_steps)
    tb.flush()

  def policy_params_fn(current_step, make_policy, params):
    del make_policy
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    checkpointer.save(out / f"{current_step}", params, force=True, save_args=save_args)
    log(f"checkpoint saved: {out / str(current_step)}")

  training_params = dict(ppo_params)
  del training_params["network_factory"]
  training_params.update(
      num_timesteps=args.timesteps,
      num_evals=args.num_evals,
      num_envs=args.num_envs,
      seed=args.seed,
  )
  train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory=functools.partial(
          ppo_networks.make_ppo_networks, **ppo_params.network_factory
      ),
      restore_checkpoint_path=epath.Path(args.resume).resolve() if args.resume else None,
      progress_fn=progress,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      policy_params_fn=policy_params_fn,
      randomization_fn=randomizer,
  )

  if args.task == "stairs":
    env = stairs_env.StairsJoystick(config=env_cfg)
    eval_env = stairs_env.StairsJoystick(config=env_cfg)
  else:
    env = locomotion.load(args.env, config=env_cfg)
    eval_env = locomotion.load(args.env, config=env_cfg)
  make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env)

  normalizer_params, policy_params, value_params = params
  with open(out / "params.pkl", "wb") as f:
    pickle.dump(
        {
            "normalizer_params": normalizer_params,
            "policy_params": policy_params,
            "value_params": value_params,
        },
        f,
    )
  with open(out / "meta.json", "w") as fp:
    json.dump(
        {
            "phase": args.phase,
            "task": args.task,
            "env": args.env,
            "timesteps": args.timesteps,
            "num_envs": args.num_envs,
            "num_evals": args.num_evals,
            "seed": args.seed,
            "resume": args.resume,
            "wall_time_s": time.time() - t0,
        },
        fp,
        indent=2,
    )
  log(f"SAVED {out} (params.pkl + {args.num_evals} orbax ckpts)")
  log_fp.close()
  tb.close()


if __name__ == "__main__":
  sys.exit(main())
