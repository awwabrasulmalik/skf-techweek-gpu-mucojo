"""PPO training entrypoint (overnight step 3). Short-proof: --timesteps 20000."""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

import build_scene
from stair_env import StairEnv

_RUNG = 0


def make_env():
    return StairEnv(rung=_RUNG, build=False)


def main():
    global _RUNG
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--out", type=str, default="checkpoints/rung0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=8)
    # MLP policy updates are faster on CPU (SB3 #1245); GPU pays off only for
    # CNN policies or very large batches. Override with --device cuda to compare.
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--resume", type=str, default=None, help="checkpoint .zip to continue from")
    ap.add_argument("--save-freq", type=int, default=100_000,
                    help="timesteps between periodic checkpoints (interrupt-safe)")
    ap.add_argument("--tb", type=str, default=None,
                    help="tensorboard logdir, e.g. runs/ (live curves in browser)")
    args = ap.parse_args()

    _RUNG = args.rung
    build_scene.main(["--rung", str(args.rung)])  # parent builds once; workers only load
    # VecMonitor is NOT added automatically: without it there is no rollout/
    # section (no ep_rew_mean/ep_len_mean) in logs. PPO learns fine either
    # way; this is purely observability.
    env = VecMonitor(SubprocVecEnv([make_env for _ in range(args.n_envs)]))
    if args.resume:
        model = PPO.load(args.resume, env=env, device=args.device)
        model.tensorboard_log = args.tb
        print(f"RESUMED from {args.resume}")
    else:
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, device=args.device,
                    tensorboard_log=args.tb)
    ckpt = CheckpointCallback(save_freq=args.save_freq,
                              save_path=str(pathlib.Path(args.out).parent),
                              name_prefix=pathlib.Path(args.out).name + "_ckpt")
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not args.resume,
                callback=ckpt)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"SAVED {args.out}.zip")


if __name__ == "__main__":
    main()
