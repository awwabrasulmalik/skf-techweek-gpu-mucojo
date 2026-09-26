# MORNING.md (living file — updated per step)

> FORK NOTE (this folder only): CPU track below is frozen history. Active
> work is the GPU track — see `MJX.md`. Other folder untouched.

## Status after steps 0–2 (sprint-builder, branch sprint/env)
- [x] Step 0 deps: torch 2.14.0+cu130 CUDA True (first try, no fallback
  needed) · SB3 2.9.0 · mujoco 3.14 · gymnasium 1.3 · matplotlib · pyyaml.
  `prototype/requirements.txt` pinned.
- [x] Step 1 env: `prototype/stair_climb/stair_env.py` runs. 1000-step random
  rollout finite, mean +0.472; zero-action hold stands 1000/1000, +0.939.
- [x] Step 2 curriculum: `rungs.yaml` (rung0..3, gate 7/10) + `--rung` in
  `build_scene.py`/`smoke.py`. All rungs SMOKE OK (h=0.777).
- [x] Step 3 rung0: 1 M steps (gate 0/10, len 220) + continued 2 M (gate 0/10,
  len 869). Found stand-still exploit → reshaped reward (velocity tracking,
  alive 0.3, residual 40%). Reshaped continuation RUNNING (~20 min).
- [x] Step 4 tooling: `eval.py` (gate table), `eval_render.py` (seeded
  skeleton mp4), `plot_curves.py` (log→curves mp4). A/B: CPU 1963 fps beats
  CUDA 1077 fps → MLP stays on CPU.
- [x] Colab handoff: `colab/G1_Stairs_Train.ipynb` + `COLAB_HOWTO.md`
  (needs `git push` before use). `docs/results.md` started.
- [x] Rung1 (2 M from rung0 brain): gate 0/10, reaches stairs (x +0.34) but
  shuffles — feet never lift. Fix: foot-clearance reward (W_LIFT 8.0, capped,
  speed-gated). Backups in `checkpoints/backup/`.
- [x] Rung1 + lift reward (2 M): gate 1/10 — FIRST CLIMBS. Len 470, x +0.46.
  Video `videos/rung1_lift.mp4`. Winner backed up. Continuing, not promoted.
- [x] Rung1 lift cont. (0237): KILLED — was training stair bypass (y→-1.7,
  knees frozen, no lift). Fix: corridor termination + VecMonitor logging.
- [x] Rung1 + corridor (2 M): gate 0/10 — transition, not stall. Policy dies
  AT the corridor wall (old diagonal habit); ep_rew 444→504 rising. Video
  `videos/rung1_corridor.mp4`. Winner backed up.
- [x] Rung1 corridor cont. (2 M): gate 0/10 — wall-dying persists (y=-0.60,
  upright). Flat twice → y-centering reshape (-|y|/0.6 dense). Backed up.
- [x] Rung1 y-center (2 M): gate 0/10 — centering WORKED (y≈0, no wall
  deaths), now trips at obstacle1 (x=-0.4). Lift is the binding constraint.
  Video `videos/rung1_ycenter.mp4`. Backed up.
- [x] Rung1 lift-test (2 M): gate 0/10 — knees 0.11 (frozen), one seed
  degenerated to 1000-step standing. Verdict: lift never learned; need march
  rung (user call). Video `videos/rung1_lifttest.mp4`.
- [ ] RUNNING: march rung, 2 M flat from clean `backup/rung0.zip`
  (out `checkpoints/march_0926_0458`, log `train_march_0926_0458_loop.log`,
  TB `runs/`). New: alternation bonus + lift cap 0.15 + eval lift stats.
  Next: gate 7/10 + split >0.10 → retry rung 1 from march brain.

## Demo commands (<10 min)
```
source .venv/bin/activate
python prototype/stair_climb/stair_env.py        # ROLLOUT OK
python prototype/stair_climb/smoke.py --rung 2   # SMOKE OK
python -c "import torch; print(torch.cuda.is_available())"  # True
```

## Top 3 next tasks
1. Per-rung success terms in stair_env (rung0/1 hook) + SB3 PPO rung0 train.
2. Eval + matplotlib stick-figure videos (no GL on this box).
3. Colab handoff (notebook + checkpoints + COLAB_HOWTO.md).
