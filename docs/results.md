# Results (living — deck evidence page)

## Rung 0 — flat, walk 1.2 m
| Steps | Gate (need 7/10) | Mean len | Video | Checkpoint |
|---|---|---|---|---|
| 1 M | 0/10 | 220 | `videos/rung0_post1M.mp4` | superseded (wrong gate thresholds) |
| 3 M | 0/10 | 869 | — | superseded (wrong gate thresholds) |
| ~5 M | **8/10 PROMOTED** | 173* | render next | `checkpoints/rung0.zip` |

*Short length is correct: successes terminate on crossing x>0 upright.

## Rung 0b (march) — flat + lift + alternation
| Steps | Gate | Lift split | Video |
|---|---|---|---|
| +2 M from clean rung0 | running | — | `march_0926_0458` |

Finding: stand-still exploit at 3 M (alive bonus > walk payoff). Fix:
velocity-tracking reward + lower alive + wider residual. See
`docs/rungs/rung0.md`.

## Rung 1 — low stairs 0.09 + obstacles
| Steps | Gate (need 7/10) | Mean len | Mean x | Video |
|---|---|---|---|---|
| ~6.6 M | 0/10 | 660 | +0.34 | `videos/rung1_post4M.mp4` (pre-lift shuffle) |
| +2 M lift | 1/10* | 470 | +0.46 | bypass walk-around, not climbs |
| +2 M corridor | 0/10 | 524 | -0.12 | `videos/rung1_corridor.mp4` (unlearning bypass) |
| +2 M corridor cont. | 0/10 | 477 | -0.10 | `videos/rung1_corridor2.mp4` (dies at wall) |
| +2 M y-center | 0/10 | 498 | -0.57 | `videos/rung1_ycenter.mp4` (centered, trips at obstacle) |
| +2 M lift-test | running | — | — | `rung1_0926_0426` |

*1/10 was a walk-around (y→-1.7); corridor made gates honest.

## Held-out (unseen friction/geometry) — runs after first promotion
| Variant | Success |
|---|---|
| (pending) | — |

## A/B: CPU vs CUDA PPO updates (32,768 steps, seed 123, 8 envs)
CPU 1963 fps vs CUDA 1077 fps → CPU 1.8× faster. MLP stays on CPU (SB3 #1245).

## GPU track (this fork, playground + warp, 1024 G1s, RTX 4060)

| Phase | Steps | Gate | Mean len | Video |
|---|---|---|---|---|
| 0 stand (run 0803) | 30.9 M | 0/10 | 212/1000 | `videos_mjx/g1_stand_0926_0803_eval.mp4` |
| 0 stand (run 0822) | +50.1 M | 0/10 | 377/1000 | `videos_mjx/g1_stand2_0926_0822_eval.mp4` |
| 0 stand (run 0843) | +50.1 M | 2/10 | 580/1000 | `videos_mjx/g1_standc_0926_0843_eval.mp4` |
| 0 stand (run 0903) | +50.1 M | 3/10 | 720/1000 | `videos_mjx/g1_standc_0926_0903_eval.mp4` |
| 0 stand (run 0923) | +50.1 M | 2/10 | 868/1000 | `videos_mjx/g1_standc_0926_0923_eval.mp4` |
| 0 stand (run 0943) | +50.1 M | 3/10 | 868/1000 | `videos_mjx/g1_standc_0926_0943_eval.mp4` |
| 0 stand (run 1005) | +50.1 M | 2/10 | 828/1000 | `videos_mjx/g1_standc_0926_1005_eval.mp4` |
| 0 stand (run 1029, no pushes) | +50.1 M | 3/10 | 799/1000 | `videos_mjx/g1_stande_0926_1029_eval.mp4` |
| 1 walk (run 1235) | 100.3 M | 10/10 PASS | 1000/1000 | `videos_mjx/g1_walk_0926_1235_eval.mp4` |
| 1 walk (run 1314) | +50.1 M | 8/10 PASS | 864/1000 | `videos_mjx/g1_walkc_0926_1314_eval.mp4` |
| 2 stairs (run 1350, no-collision bug) | 30.9 M | 0/10 VOID | 1000/1000 | `videos_mjx/g1_stairs_0926_1350_eval.mp4` |
| 2 stairs baseline (fixed env, same ckpt) | — | 0/10, max_x 2.11 m | ~200/1000 | `videos_mjx/g1_stairs_0926_1350_eval_fixed.mp4` |
| 2 stairs baseline (full collision) | — | 0/10, max_x 2.09 m | 206/1000 | `videos_mjx/g1_stairs_0926_1350_eval_fullcol.mp4` |
| 2 stairs (run 1426, full collision) | +30.9 M | 0/10, max_x 1.98 m | 257/1000 | `videos_mjx/g1_stairs_0926_1426_eval.mp4` |
| 2 stairs (run 1444, full collision) | +30.9 M | 0/10, max_x 1.99 m | 258/1000 | `videos_mjx/g1_stairs_0926_1444_eval.mp4` |
| 2 stairs (run 1508, reshaped+centered) | +30.9 M | 0/10, max_x 2.06 m | 862/1000 | `videos_mjx/g1_stairs_0926_1508_eval.mp4` |

Throughput 25–42k steps/s (30× the CPU track). Reward audit + fixes in
`MJX.md`. Continuing +50 M from last ckpt at 2048 envs.
