# SKF TechWeek — GPU humanoid stairs (Unitree G1, MJX + PPO)

Hackathon project: teach a Unitree G1 humanoid to climb stairs
with reinforcement learning — 2048 parallel robots on one
RTX 4060, gated stand → walk → stairs curriculum, every run
checkpointed, gated on video, and documented.

- Start here for the story: [`pitch/pipeline.md`](pitch/pipeline.md)
  (full pipeline: phases, rewards, reshapes, bugs, run ledger).
- 3-minute pitch: [`pitch/slides.md`](pitch/slides.md) ·
  re-run commands: [`pitch/reproducibility.md`](pitch/reproducibility.md).
- Results table: [`docs/results.md`](docs/results.md) ·
  living training log: [`MJX.md`](MJX.md).

## Layout

| Path | What |
|---|---|
| `prototype/mjx_track/` | GPU training code: `train_g1.py`, `eval_g1.py`, `render_g1.py`, `stairs_env.py`, `stairs_scene.xml`, maintained tests |
| `prototype/stair_climb/` | Earlier CPU-track prototype (reference) |
| `checkpoints_release/` | Two small release ckpts (walk 8/10, stairs best) + eval commands |
| `videos_mjx/`, `videos/` | Gate eval videos |
| `docs/`, `pitch/`, `colab/` | Results, pitch materials, Colab notebook |
| `hackathon-docs/` | Event challenge briefs |

## Quickstart (eval, needs CUDA + the venv)

```bash
python -m venv .venv-mjx && . .venv-mjx/bin/activate
pip install -r prototype/requirements-mjx.txt   # playground pin + mujoco + brax
.venv-mjx/bin/python prototype/mjx_track/test_stairs_env.py   # expect TEST_STAIRS_ENV OK
```

Then run the eval commands in `checkpoints_release/README.md`.
Full training commands: `pitch/reproducibility.md`.

## Status at release

- Walk (flat): **10/10 + 8/10 PASS** — robust joystick tracking.
- Stairs: 0/10 success, full episodes survived post-reshape —
  centered approach, climb shaping, anti-crouch + plant bonus
  (see `pitch/pipeline.md` §5 for the iteration history).

Built in one day: ~650M+ env steps at ~45k steps/s on a laptop
RTX 4060. Sim only, no sim-to-real transfer (yet).
