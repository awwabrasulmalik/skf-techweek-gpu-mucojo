# Reproducibility scaffold — every deck claim traces to a command

Hardware: RTX 4060 Laptop 8 GB, WSL2. Seeds: 1 unless noted. All commands
from repo root. Two venvs: `.venv` (CPU track, frozen snapshot) and
`.venv-mjx` (GPU track, active; see prototype/requirements-mjx.txt).

## GPU track (active — slides 6–7)

Stand/walk results: config = train cmd + `checkpoints_mjx/<tag>/env_config.json`;
policy = numbered orbax ckpt; curves = `<tag>/tb` (TensorBoard).

- Re-train (stand): `.venv-mjx/bin/python prototype/mjx_track/train_g1.py
  --phase stand --timesteps 50000000 --num-envs 2048 [--resume <ckpt>]
  --out checkpoints_mjx/<tag> --log runs_mjx/<tag>.log`
- Re-train (walk): same with `--phase walk --timesteps 100000000`.
- Re-gate (stand): `.venv-mjx/bin/python prototype/mjx_track/eval_g1.py
  --ckpt <ckpt> --phase stand --episodes 10` → `GATE: success N/10`.
- Re-gate (walk): same with `--phase walk --command 0.5,0,0`.
- Re-render: `.../eval_g1.py ... --save-traj videos_mjx/<t>.npz` then
  `.../render_g1.py --traj videos_mjx/<t>.npz --out videos_mjx/<t>.mp4`.
- Curves: `.venv-mjx/bin/tensorboard --logdir checkpoints_mjx --port 6006`.
- Env check: `.venv-mjx/bin/python prototype/mjx_track/test_g1_env.py`
  (needs CUDA).

Checkpoint → artifact map (REFRESH as runs land; source: SUPERVISOR_STATE.md):

| Run | Ckpt | Gate | Video | TB |
|---|---|---|---|---|
| g1_standc_0926_0943 | 50135040 | 3/10 len 868 | videos_mjx/g1_standc_0926_0943_eval.mp4 | checkpoints_mjx/g1_standc_0926_0943/tb |
| g1_stande_0926_1029 | 50135040 | 3/10 len 799 | videos_mjx/g1_stande_0926_1029_eval.mp4 | .../tb |
| g1_walk_0926_1216 | (running) | TBD | TBD | .../tb |

## CPU track (frozen snapshot in this fork — backup/detail claims)

- Rebuild clean scene (generated file may carry the MJX capsule patch):
  `.venv/bin/python prototype/stair_climb/build_scene.py --rung N`
- Rung-0 gate: `.venv/bin/python prototype/stair_climb/eval.py
  --ckpt checkpoints/backup/rung0.zip --rung 0 --episodes 10`
- Render: `.venv/bin/python prototype/stair_climb/eval_render.py
  --ckpt <zip> --rung N --out videos/<name>.mp4`
- Rung checks: `.venv/bin/python prototype/stair_climb/test_rungs.py`
- Full overnight history + march run live in the OTHER folder
  (gbg-techweek-hackathon-Placeholder); this fork's copy is a snapshot.

## Deck-freeze spot check (~5 min, run before 16:00 prep)

1. Re-run the gate for the row shown on slide 7 → numbers match.
2. Play the slide-6 mp4 offline; re-export poster (assets/README.md).
3. Open the TB link; confirm the curve screenshot is current.
