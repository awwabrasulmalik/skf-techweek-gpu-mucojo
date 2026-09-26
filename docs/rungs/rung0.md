# Rung 0 — flat approach (first attempt, 2026-09-25 evening)

- Config: flat, friction ~0.9, no obstacles, no turn. PPO defaults
  (lr 3e-4, n_steps 2048, batch 64), 8 CPU envs, 1,015,808 steps in 629 s
  (~1612 fps). Seed 0.
- Final train metrics: explained_variance 0.918, clip_fraction 0.499,
  approx_kl 0.065, value_loss 8.1.
- Gate eval (10 episodes, deterministic): success 0/10, mean length 220/1000.
  Gate (7/10) NOT passed — no promotion.
- Video: `videos/rung0_post1M.mp4` (falls ~frame 200). Checkpoint:
  `checkpoints/rung0.zip`.

Not/why (5 lines): the critic learned (explained variance 0.92) but the
actor only survives ~4 s — 1 M steps is early for a 29-DoF humanoid from
scratch, and clip_fraction 0.5 suggests updates are still large. No reward
bug is evident (video shows stand-then-fall, not instant collapse). Next:
continue 2 M steps from this checkpoint and re-evaluate; if ep_len stalls
below ~500, reshape (bigger alive bonus / upright bonus) before rung 1.

## Update — 3 M steps total (continued 2 M from checkpoint)
- Gate: 0/10, mean ep len 869/1000. Stands ~17 s, never walks 1.2 m to end_x.
- Diagnosis: stand-still exploit — ALIVE 1.0/step beat walking rewards.
- Reshape (committed): velocity-tracking reward at 0.5 m/s replaces linear
  W_FWD; ALIVE 1.0→0.3; residual authority 20%→40%. Continuing from checkpoint.

## Update — ~5 M steps, reshaped reward (PROMOTED)
- Gate re-eval first showed 0/10 with mean x +2.32 m — paradox traced to an
  env bug: success thresholds read rung-2 module defaults (end_x 1.4, top_z
  0.9) instead of the rung config, making rung-0 success impossible. Fixed:
  per-rung geometry from `rungs.yaml`. All prior gate numbers used the wrong
  thresholds; the policies were better than measured.
- True gate: **8/10 → PROMOTED to rung 1.** Works/why: velocity tracking
  pays for steady 0.5 m/s walking, low alive bonus removes the statue
  optimum, 40% residual escapes the PD standing well.
