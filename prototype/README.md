# Prototype skeleton

Layout (only what serves the 05 assumption / 06 must-demonstrate list):

- `prototype/stair_climb/` — MuJoCo scene + rollout scripts (skeleton tonight,
  trained policy tomorrow).
- `docs/rungs/` — one note per curriculum rung (works/why or not/why).
- `docs/01-understand.md … docs/04-create.md` — decision trail.
- `docs/pipeline.md` — the pipeline this prototype evidences.

Run (from repo root, after `uv` venv exists):

```
source .venv/bin/activate
uv pip install -r prototype/requirements.txt
python prototype/stair_climb/smoke.py   # loads scene, steps N frames, prints base height
```

Tonight's bar: scene loads, sim steps, no crash. Training is Saturday's job.

## GPU track (this fork's direction — see MJX.md)

`prototype/stair_climb_mjx/` is superseded by `prototype/mjx_track/`, a thin
layer over `mujoco_playground` (pinned in `requirements-mjx.txt`, env
`.venv-mjx`). Same conventions as the CPU track: unique run names, periodic
checkpoints, gate evals, stick-figure videos.

- `train_g1.py` — Brax PPO on `G1JoystickFlatTerrain` (`--phase stand|walk`).
- `eval_g1.py` — gate eval from an orbax ckpt + trajectory npz.
- `render_g1.py` — stick-figure mp4 from a trajectory npz.
- `test_g1_env.py` — maintained env check (needs CUDA).
