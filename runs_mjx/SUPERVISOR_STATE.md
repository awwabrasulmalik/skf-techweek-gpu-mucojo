# Supervisor state (GPU track) — machine log, agent merges into MJX.md/results.md

Policy: gate pass (stand 10/10, walk 7/10) → promote; len improving >20% →
continue same phase (+50M, 2048 envs); flat twice → stand: reshape pushes-off
once, else BLOCKED. Crash without SAVED → relaunch same cmd once, then BLOCKED.
No new launches from 13:00 (packaging). Kill: `touch runs_mjx/SUPERVISOR_STOP`.

- [0926 08:15] history: `g1_stand_0926_0803` 30.9M → gate 0/10, len 212/1000.
- [0926 08:22] adopted `g1_stand2_0926_0822` (stand, +50M, 2048 envs, resume 30965760).
- [0926 08:41] gate `g1_stand2_0926_0822/50135040`: 0/10, len 377/1000 (prev 212, +78% improving), drift 0.75m, video `videos_mjx/g1_stand2_0926_0822_eval.mp4`. Note: failures now drift+falls, no position anchor in rewards (tracking only penalizes velocity).
- [0926 08:43] launched `g1_standc_0926_0843` (stand, +50M, 2048 envs, resume 50135040). Step-0 eval -0.51 (restored ✓).
- [0926 09:01] gate `g1_standc_0926_0843/50135040`: 2/10 FIRST SUCCESSES, len 580/1000 (prev 377, +54% improving), video `videos_mjx/g1_standc_0926_0843_eval.mp4`.
- [0926 09:03] launched `g1_standc_0926_0903` (stand, +50M, 2048 envs, resume 50135040). Step-0 eval 0.72 (restored ✓).
- [0926 09:21] gate `g1_standc_0926_0903/50135040`: 3/10, len 720/1000 (prev 580, +24% improving), video `videos_mjx/g1_standc_0926_0903_eval.mp4`.
- [0926 09:23] launched `g1_standc_0926_0923` (stand, +50M, 2048 envs, resume 50135040). Step-0 eval 1.89 (restored ✓).
- [0926 09:41] gate `g1_standc_0926_0923/50135040`: 2/10, len 868/1000 (prev 720, +21% improving, barely), drift now main failure (5/10 survive full len). Video `videos_mjx/g1_standc_0926_0923_eval.mp4`.
- [0926 09:43] launched `g1_standc_0926_0943` (stand, +50M, 2048 envs, resume 50135040). Step-0 eval 2.74 (restored ✓).
- [0926 10:01] gate `g1_standc_0926_0943/50135040`: 3/10, len 868/1000 (prev 868, +0% FLAT 1x). Drift still main failure. Video `videos_mjx/g1_standc_0926_0943_eval.mp4`.
- [0926 10:05] launched `g1_standc_0926_1005` (stand, +50M, 2048 envs, resume 50135040, flat-1x continue). Step-0 eval 3.52 (restored ✓).
- [0926 10:26] gate `g1_standc_0926_1005/50135040`: 2/10, len 828/1000 (prev 868, FLAT 2x). Drift still main failure. Video `videos_mjx/g1_standc_0926_1005_eval.mp4`.
- [0926 10:29] RESHAPE launched `g1_stande_0926_1029` (stand, +50M, 2048 envs, pushes OFF, resume 50135040). Step-0 eval 11.23 (no-push eval env, easier ✓).
- [0926 10:43] user order: stop loop after next delivery. Job 60440956 replaced by 2eede7c0 (final-delivery mode: gate+video+docs, no launches, then halt via LOOP_STOPPED + self-delete).
- [0926 10:50] FINAL gate `g1_stande_0926_1029/50135040` (pushes-off reshape): 3/10, len 799/1000 (prev 828 — reshape did not help; drift 0.98m worse). Video `videos_mjx/g1_stande_0926_1029_eval.mp4`. Note: eval runs WITH pushes (eval has no pushes flag) while this run trained WITHOUT — gate tests push-robustness.
- [0926 10:50] FINAL delivery done. Loop halted per user order (LOOP_STOPPED). No follow-up launched.
- [0926 12:16] user: stand accepted (3/10, rebalances fine). PROMOTED to walk. Split: 1h walk / 3h stairs. Launched `g1_walk_0926_1216` (walk, 100M, 2048 envs, resume stand 0943/50135040 push-robust ckpt).
- [0926 12:16] packaging freeze moved 13:00 -> 15:00 (4h training budget needs afternoon room; pitch 17:00 unchanged).
- [0926 12:31] GPU LOST (dxgvmb_send_create_process failed; JAX fell back to CPU). Walk run g1_walk_0926_1216 was training on CPU — KILLED as useless for timeline. Loop job deleted.
- RESUME PLAN (after user restarts WSL): verify `nvidia-smi` + JAX CudaDevice, then relaunch walk (100M, 2048 envs, resume stand 0943/50135040) + re-arm loop. Partial g1_walk_0926_1216 dir (ckpt 0 only) should be ignored, not resumed.
- [0926 12:35] user restarted WSL, GPU back (CudaDevice ✓). Relaunched `g1_walk_0926_1235` (walk, 100M, 2048 envs, resume stand 0943/50135040), verified on CUDA. Loop re-armed with GPU-gate.
- [0926 13:01] user policy for STAIRS phase: short runs (~30M each), gate + adjust/reshape between runs. No single long run (too much of a gamble). Applies when user orders the stairs promotion.
- [0926 13:10] FIRST WALK GATE `g1_walk_0926_1235/100270080`: 1/10, len 991/1000, drift 6.99m (survives full eps but off-direction/circling; progress<1m along +x). Video `videos_mjx/g1_walk_0926_1235_eval.mp4`. No baseline → continue once.
- [0926 13:14] launched `g1_walkc_0926_1314` (walk, +50M, 2048 envs, resume 100270080, CUDA verified). Step-0 eval 9.40 (restored ✓).
- [0926 13:25] CORRECTED WALK GATE `g1_walk_0926_1235/100270080`: 10/10 PASS (len 1000, path ~7.8m, straight ~0.89). Old 1/10 void (frame + resample bugs). Video re-rendered (longest walk).
- [0926 13:35] walk continuation `g1_walkc_0926_1314/50135040`: 8/10 PASS (len 864, path 7.9m, straight 0.90). Video `videos_mjx/g1_walkc_0926_1314_eval.mp4`. PROMOTE-TO-STAIRS (user promotes manually, no launch).
- [0926 13:48] user: GO STAIRS. Built stairs_env (5x0.09x0.28 + landing, narrow yaw, success x>3.3+z>0.95 +10 bonus, swing 0.22, fwd-only cmds). TEST_STAIRS_ENV OK.
- [0926 13:48] `g1_stairs_0926_1348` died on start: scan pytree mismatch (metrics/info gained "success" in step but not reset). Fixed (keys in both).
- [0926 13:50] launched `g1_stairs_0926_1350` (walk+stairs, 30M, 2048 envs, resume walkc/50135040). Step-0 eval 17.72. ETA ~12 min.
- [0926 14:01] `g1_stairs_0926_1350` DONE 30M (eval +21.18). Gate on broken env 0/10, max_x 9.41m — VOID.
- [0926 14:10] user: robot walks THROUGH stairs, no collision. Root cause: feetonly conaffinity=0, feet collide only with <pair>-named geoms; steps had no pairs (probe: 0 step contacts). Fix: 12 foot<->step/landing pairs in stairs_scene.xml + drop-test in test_stairs_env.py (fails before/passes after). TEST_STAIRS_ENV OK.
- [0926 14:07] `g1_stairs_0926_1407` (broken-env continuation) KILLED as useless.
- [0926 14:15] honest baseline 1350/30965760 on fixed env: 0/10, max_x 2.11m, falls ~200 steps. Video `videos_mjx/g1_stairs_0926_1350_eval_fixed.mp4`.
- [0926 14:18] launched `g1_stairs_0926_1418` (walk+stairs, 30M, 2048 envs, resume 1350/30965760, CUDA verified). First REAL stairs run. ETA ~12 min.
- [0926 14:21] user: body still clips through stairs (feet-only pairs). Fix: all 8 robot collision geoms x 6 boxes = 48 pairs in stairs_scene.xml. Test now requires foot AND non-foot body contacts. TEST_STAIRS_ENV OK.
- [0926 14:23] `g1_stairs_0926_1418` (feet-only env) KILLED. Baseline 1350/30965760 on full-collision env: 0/10, max_x 2.09m. Video `videos_mjx/g1_stairs_0926_1350_eval_fullcol.mp4`.
- [0926 14:26] launched `g1_stairs_0926_1426` (walk+stairs, 30M, 2048 envs, resume 1350/30965760, CUDA verified). First full-collision stairs run. ETA ~12 min.
- [0926 14:37] `g1_stairs_0926_1426` DONE 30M (eval 3.59->5.77). GATE 0/10, max_x 1.98m, len 257 (vs fullcol baseline 0/10, 2.09m: FLAT x1). Video `videos_mjx/g1_stairs_0926_1426_eval.mp4`.
- [0926 14:44] launched `g1_stairs_0926_1444` (walk+stairs, 30M, 2048 envs, resume 1426/30965760, CUDA verified). Second full-collision chunk; flat again -> BLOCKED. ETA ~12 min.
- [0926 14:54] `g1_stairs_0926_1444` DONE 30M (eval 6.25->5.81). GATE 0/10, max_x 1.99m, len 258 (vs 1426 0/10, 1.98m: FLAT x2). Video `videos_mjx/g1_stairs_0926_1444_eval.mp4`.
- [0926 14:58] BLOCKED: stairs flat x2 (max_x 2.09->1.98->1.99, len ~206->257->258). No launches. Awaiting agent/user reshape judgment.
- [0926 15:05] user: 15:00 freeze REMOVED (1h left); commit loop job d3943f8f KILLED; supervisor loop recreated as 2a1c58dd (launches OK until 16:00). Plan: reshape 30M now, then one big final gamble run.
- [0926 15:05] CENTERING BUG confirmed from 1444 traj: spawn y+-0.5 + yaw drift walks robot to y~+1.2 (stair edge y=1.0), falls off the side at x~2.2.
- [0926 15:05] RESHAPE (stairs_env.py): spawn x+-0.25/y+-0.10/yaw+-0.10, lin_vel_y=[0,0], yaw cmd +-0.1, feet_clearance -2.0 w/ max_foot_height 0.20, dense climb_x (+0.125/step max) + climb_z (+0.10/step max), pushes off. TEST_STAIRS_ENV OK (centered spawn + climb keys).
- [0926 15:08] launched `g1_stairs_0926_1508` (walk+stairs RESHAPED, 30M, 2048, pushes off, resume 1444/30965760, CUDA verified). 14:58 BLOCKED lifted. ETA ~12 min.
- [0926 15:18] `g1_stairs_0926_1508` DONE 30M reshaped (eval 8.12->11.19). GATE 0/10, max_x 2.06m, len 862, h 0.44m (vs 1444 0/10, 1.99m, len 258, h 0.74m: position FLAT, survival way up via crouch). Video `videos_mjx/g1_stairs_0926_1508_eval.mp4`.
- [0926 15:24] FLAT-on-position -> reported to user, no auto-launch (final gamble is user's call).
- [0926 15:28] user: option 2, 30-min final run, supervisor loop DISABLED (2a1c58dd deleted). No more auto-gates; user prompts to finalize.
- [0926 15:28] ANTI-CROUCH reshape: one-sided cost below root 0.70 (-1.0 scale; full crouch -0.26/step) + metric. Rejected playground base_height (two-sided, would punish landing height). TEST_STAIRS_ENV OK.
- [0926 15:29] launched FINAL `g1_stairs_0926_1529` (walk+stairs, 80M ~30min, 2048, pushes off, resume 1508/30965760, CUDA verified). ETA ~16:00.
- [0926 15:33] user: reward feet planted on stairs (crouch rests on knees, feet float). Added plant bonus +0.03/step: any active foot<->step contact while root>0.70 (either foot, tall-only). Functional test (§5) proves it fires. TEST_STAIRS_ENV OK.
- [0926 15:34] `g1_stairs_0926_1529` (no plant bonus, reached 9M) KILLED for relaunch.
- [0926 15:35] launched FINAL `g1_stairs_0926_1535` (walk+stairs, 80M ~30min, 2048, pushes off, resume 1508/30965760, CUDA verified). ETA ~16:05. All loops dead; user prompts to finalize.
- [0926 16:02] Final `g1_stairs_0926_1535` DONE 81.1M (eval ~2->20.80). OFFICIAL GATE (reshaped env) 1/10: ep5 SUCCESS (max_x 3.31, h 0.93, len 319); mean max_x 2.89m (+40% vs 1508), len 602, h 0.79 (crouch gone). Video `videos_mjx/g1_stairs_0926_1535_eval.mp4` = the successful climb. NOTE: first gate attempt ran on stale pre-reshape worktree (github-branch side effect, since restored) — voided; numbers above are the valid re-gate.
