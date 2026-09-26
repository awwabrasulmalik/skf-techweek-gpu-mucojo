# Release checkpoints (2 MB each)

Two orbax checkpoints + their env configs, so the gates are
reproducible without retraining. Needs the repo venv + CUDA
(see `pitch/reproducibility.md` for setup).

## Walk — 8/10 PASS (flat, 0.5 m/s command)

```
.venv-mjx/bin/python prototype/mjx_track/eval_g1.py \
  --ckpt checkpoints_release/walk_8of10 \
  --phase walk --task flat --command 0.5,0,0 --episodes 10 \
  --save-traj /tmp/walk_eval.npz
```

From run `g1_walkc_0926_1314` (+50.1M on top of the 100.3M walk
seed). Gate: 8/10, mean len 864/1000, path ~7.9 m.

## Stairs — 0/10, len 862/1000 (best effort at release time)

```
.venv-mjx/bin/python prototype/mjx_track/eval_g1.py \
  --ckpt checkpoints_release/stairs_0of10_len862 \
  --phase walk --task stairs --command 0.5,0,0 --episodes 10 \
  --save-traj /tmp/stairs_eval.npz
.venv-mjx/bin/python prototype/mjx_track/render_g1.py \
  --traj /tmp/stairs_eval.npz --out /tmp/stairs_eval.mp4
```

From run `g1_stairs_0926_1508` (reshaped env: centered spawn,
dense climb shaping, anti-crouch). Gate: 0/10 success, but full
episodes survived (len 862 vs 258 pre-reshape). Stairs success =
root x > 3.3 m AND z > 0.95 m (up 5×0.09 m steps + landing).

## Stairs FINAL — 1/10 with a landing (the headline ckpt)

```
.venv-mjx/bin/python prototype/mjx_track/eval_g1.py \
  --ckpt checkpoints_release/stairs_final_1of10 \
  --phase walk --task stairs --command 0.5,0,0 --episodes 10 \
  --save-traj /tmp/stairs_final_eval.npz
```

From run `g1_stairs_0926_1535` (+81.1M, anti-crouch + plant
bonus). Gate: 1/10, mean max_x 2.89 m — episode 5 climbs all
5 steps onto the landing. Video: `videos_mjx/stairs_final_3d.mp4`.

Full run ledger + reward history: `docs/results.md`, `pitch/pipeline.md`.
