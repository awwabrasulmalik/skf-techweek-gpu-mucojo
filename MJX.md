# MJX.md — GPU track (THIS fork's direction; living file)

## High-level plan: stand first, then expand

Engine: `mujoco_playground` (pinned `4057c14`) on the **MuJoCo Warp GPU backend**,
trained with Brax PPO. ~1024 G1 robots in parallel on the RTX 4060 8GB
(playground default 8192 would OOM; scale up only with measured headroom).

- **Phase 0 STAND** (running): `G1JoystickFlatTerrain`, velocity commands forced
  to zero. Pure balance task. Gate: 10/10 episodes survive 1000 steps, drift <0.5 m.
- **Phase 1 WALK**: same env, default command ranges, finetune from phase-0 ckpt.
  Gate: 7/10 survive + track 0.5 m/s, displacement >1 m.
- **Phase 2 STAIRS** (build next): custom warp/MJX stairs env from our
  `rungs.yaml` geometry (low stairs 0.09 + obstacles first), finetune from
  phase-1 ckpt. Gate: 7/10 climbs, same bar as the CPU track.

Conventions mirror the CPU track: unique `--out` per run, one orbax checkpoint
per eval (`checkpoints_mjx/<tag>/<step>/`), `params.pkl` + `env_config.json` +
`meta.json` at the end, text log in `runs_mjx/`, stick-figure mp4s in
`videos_mjx/`, gate tables in `docs/results.md`.

## Why this reverses the old "no MJX" decision

`docs/05-overnight-plan.md` rejected MJX/JAX for CUDA-plugin pain. That is stale:
`jax[cuda12]` + `mujoco-warp` installed first try on Python 3.14 (jaxlib ships
cp314 wheels), and playground wraps the whole stack (env + DR + tuned PPO) so no
custom RL code is needed. Separate `.venv-mjx` keeps the CPU env pristine.

## Commands (from repo root)

```
.venv-mjx/bin/python prototype/mjx_track/test_g1_env.py          # TEST_G1_ENV OK (needs CUDA)
.venv-mjx/bin/python prototype/mjx_track/train_g1.py --phase stand --timesteps 30000000 --num-envs 1024 --out checkpoints_mjx/g1_stand_TAG
.venv-mjx/bin/python prototype/mjx_track/eval_g1.py --ckpt checkpoints_mjx/g1_stand_TAG/<step> --phase stand --episodes 10
.venv-mjx/bin/python prototype/mjx_track/render_g1.py --traj videos_mjx/<tag>.npz --out videos_mjx/<tag>.mp4
nvidia-smi dmon -s mu -c 120   # VRAM/gpu watch (peak decides 1024 vs 2048 envs)
```

## Status log

- 2026-09-26 ~07:40: WSL crash wiped /tmp; `.venv-mjx` survived. GPU back
  (JAX sees cuda:0). Other folder's march run had SAVED before dying (observed
  read-only; untouched).
- 07:45: `playground==0.2.0 @4057c14` + `mujoco-warp==3.14.0` installed, no
  conflicts. G1 env defaults to warp backend (`impl="warp"`).
- 07:50: `test_g1_env.py` → TEST_G1_ENV OK (200-step finite, stand cmds zero).
- 07:51: launched `checkpoints_mjx/g1_stand_0926_0751` (stand, 30M steps,
  1024 envs, 10 evals). First ckpt validates the loop.
- 07:51–08:03: three launch failures, all fixed: (1) brax 0.14.2 calls
  removed `jax.device_put_replicated` → `jax_compat.py` shim (replicate =
  leading axis + single-device put); (2) orbax needs absolute ckpt paths →
  `.resolve()` in train_g1.py. Dead runs: `g1_stand_0926_0751/0757/0800`.
- 08:03: `g1_stand_0926_0803` HEALTHY: GPU 100%, 2395 MiB VRAM, 25–37k
  steps/s. eval_rew -9.7 → -3.0. ETA ~15 min for 30M.
  Next run: try 2048 envs (5.8 GB headroom).
- 08:09: eval+render loop validated on ckpt 10321920: GATE 0/10 (mean len
  48/1000 — early, expected), `videos_mjx/g1_stand_0926_0803_eval.mp4`
  renders. Eval fix: mirror training wrap stack (`wrap_for_brax_training`,
  not stale `BraxEnvWrapper`) + absolute ckpt path.
- 08:10–08:15: REWARD AUDIT (user call). Safe: knees one-way −5°..165°
  (physics) + soft-limit/pose costs; shuffling + hopping killed in walk by
  feet_phase (0.15 m swing on π-offset gait, masked unless moving/commanded).
  Fixed: stand phase sets feet_air_time=0 (was paying for hop-in-place at
  zero command); eval gates gain pelvis-height floor 0.55 m (anti-kneel).
  TB logging added (tensorboardX → `<out>/tb`); run 0803 backfilled.
- 08:15: `g1_stand_0926_0803` DONE 30.9M steps, 41.7k steps/s. Final gate
  0/10, len 212/1000 (was 48 at 10M — learning, too slow). Cause: 30M short
  for 29-DoF + pushes/noise at full defaults. Next: resume +50M, 2048 envs.
- 08:18: `g1_stand2_0926_0818` died instantly: relative `--resume` path
  rejected by orbax → `.resolve()` fix in train_g1.py (same bug class as ckpt).
- 08:22: `g1_stand2_0926_0822` HEALTHY (resume 30.9M ckpt, 2048 envs, +50M,
  air_time=0, TB live): step-0 eval -2.44 (restored level ✓), 5.5M eval
  -1.89, GPU 100%, VRAM 2.4 GB, ~32k steps/s. ETA ~25 min.
- 08:39: `g1_stand2_0926_0822` DONE 50.1M, 48k steps/s, eval -0.46. Gate 0/10,
  len 377 (prev 212, improving) — failures now drift (0.75 m) + late falls.
  Video `videos_mjx/g1_stand2_0926_0822_eval.mp4`.
- 08:43: launched `g1_standc_0926_0843` (stand, +50M, 2048 envs, resume
  50135040). Step-0 eval -0.51 ✓. ETA ~20 min.
- 09:00: `g1_standc_0926_0843` DONE 50.1M, 49.7k steps/s, eval +1.38 (first
  positive run). Gate 2/10 FIRST SUCCESSES, len 580 (prev 377, improving).
  Video `videos_mjx/g1_standc_0926_0843_eval.mp4`.
- 09:03: launched `g1_standc_0926_0903` (stand, +50M, 2048 envs, resume
  50135040). Step-0 eval 0.72 ✓. ETA ~20 min.
- 09:20: `g1_standc_0926_0903` DONE 50.1M, 49.5k steps/s, eval +2.45. Gate
  3/10, len 720 (prev 580, +24% improving). Video
  `videos_mjx/g1_standc_0926_0903_eval.mp4`.
- 09:23: launched `g1_standc_0926_0923` (stand, +50M, 2048 envs, resume
  50135040). Step-0 eval 1.89 ✓. ETA ~20 min.
- 09:40: `g1_standc_0926_0923` DONE 50.1M, 49.3k steps/s, eval +3.39. Gate
  2/10, len 868 (prev 720, +21% improving, barely). Drift now the failure
  mode (5/10 full-length survivals). Video
  `videos_mjx/g1_standc_0926_0923_eval.mp4`.
- 09:43: launched `g1_standc_0926_0943` (stand, +50M, 2048 envs, resume
  50135040). Step-0 eval 2.74 ✓. ETA ~20 min.
- 10:00: `g1_standc_0926_0943` DONE 50.1M, 49.4k steps/s, eval +3.76. Gate
  3/10, len 868 (prev 868, +0% FLAT 1x). Drift still the failure mode.
  Video `videos_mjx/g1_standc_0926_0943_eval.mp4`.
- 10:05: launched `g1_standc_0926_1005` (stand, +50M, 2048 envs, resume
  50135040, flat-1x continue). Step-0 eval 3.52 ✓. ETA ~20 min. If flat
  again → reshape (pushes off).
- 10:22: `g1_standc_0926_1005` DONE 50.1M, 49.3k steps/s, eval +4.19. Gate
  2/10, len 828 (prev 868, FLAT 2x). Video
  `videos_mjx/g1_standc_0926_1005_eval.mp4`.
- 10:29: RESHAPE `g1_stande_0926_1029` (stand, +50M, 2048 envs, pushes OFF,
  resume 50135040). Step-0 eval 11.23 (easier eval env ✓). ETA ~20 min.
  If still flat → BLOCKED for agent judgment.
- 10:47: `g1_stande_0926_1029` DONE 50.1M, 47.6k steps/s, eval +25.2
  (no-push scale). Gate 3/10, len 799 (prev 828 — reshape did NOT help;
  drift 0.98 m worse). Caveat: eval runs WITH pushes, train without.
  Video `videos_mjx/g1_stande_0926_1029_eval.mp4`.
- 10:50: FINAL delivery done. Loop HALTED per user order (LOOP_STOPPED,
  job deleted). Awaiting user adjustment. Stand at ~3/10 after 281 M
  total steps; drift is the binding failure mode.
- 12:16: user accepted stand (rebalances fine, works). PROMOTED to walk.
  Split 1h walk / 3h stairs. Walk rewards verified good as pure defaults
  (phase gait kills shuffle+hop, air-time steps, stop-on-zero-cmd).
  Launched `g1_walk_0926_1216` (100M, 2048 envs, resume push-robust stand
  ckpt 0943/50135040). Loop re-armed (job 7035c49e, walk policy).
  Packaging freeze 13:00 → 15:00 (4h budget).
- Standing user policy for STAIRS: ~30M runs with gate + adjust/reshape
  between each. Never one long stairs run.
- 13:50: STAIRS GO. Env: playground Joystick subclass, 5x0.09x0.28 scene
  + landing, yaw +-0.15, forward cmds 0.3-0.7, swing 0.22. Success:
  x>3.3+z>0.95, +10 bonus, terminates. TEST_STAIRS_ENV OK (incl. scan
  pytree fix). Launched `g1_stairs_0926_1350` (30M, 2048, resume
  walkc/50135040). Step-0 eval 17.72. Loop re-armed (stairs policy).
- 13:07: `g1_walk_0926_1235` DONE 100.3M, 51.7k steps/s, eval +8.85. FIRST
  WALK GATE 1/10, len 991 — survives but off-direction (drift 7 m,
  progress <1 m along +x; circling?). Video
  `videos_mjx/g1_walk_0926_1235_eval.mp4`.
- 13:25: corrected gate on same ckpt with fixed eval: **10/10 PASS**
  (len 1000, path ~7.8 m, straight ~0.89). Old 1/10 void. Walk phase
  effectively done; PROMOTE-TO-STAIRS awaiting user word.
- 13:32: `g1_walkc_0926_1314` DONE +50.1M, 46.5k steps/s, eval +14.68.
  Gate 8/10 PASS (len 864, path 7.9 m, straight 0.90). Video
  `videos_mjx/g1_walkc_0926_1314_eval.mp4`. PROMOTE-TO-STAIRS (user
  promotes; no auto-launch).
- 13:20: EVAL BUGS FOUND (user: "movement looks good, judge direction").
  (1) Commands are BODY-frame (tracking uses get_local_linvel) but the
  gate measured WORLD-+x progress with random spawn yaw — good episodes
  failed. Analyzed traj walked 6 m at +95° (correct local tracking!).
  (2) Env resamples a RANDOM command every 500 steps; eval override was
  overwritten mid-episode. Fixed: pin sample_command + odometry gate
  (path>4 m, straight>0.6, survive, height). Old 1/10 is VOID.
  Stairs note: bump swing_height 0.15→~0.22 (strides small for stairs).
- 13:14: launched `g1_walkc_0926_1314` (walk, +50M, 2048 envs, resume
  100270080, CUDA verified). Step-0 eval 9.40 ✓. ETA ~18 min.
- 12:31: GPU LOST (host dxg channel errors). Walk run was on CPU → killed.
  Loop deleted. PARKED for user WSL restart. Resume: verify CUDA, relaunch
  walk 100M from stand 0943/50135040, re-arm loop.
- 08:30: supervisor = built-in cron loop (job 60440956, every 5 min, skips
  while active). Policy: healthy → heartbeat; finished → gate+video →
  promote/continue/one reshape (`--pushes off`, validated)/BLOCKED. No new
  launches from 13:00. Timeline in `runs_mjx/SUPERVISOR_STATE.md`.
  `supervise.sh` kept as unlaunched fallback — do NOT run both.
- 10:43: user order: stop after next delivery. Job 60440956 → 2eede7c0
  (final-delivery mode: gate+video+docs, zero launches, then halt via
  LOOP_STOPPED + self-delete). User adjusts and recontinues manually.
- 14:01: `g1_stairs_0926_1350` DONE 30M, 46.7k steps/s, eval +21.18.
  Gate on BROKEN env: 0/10, max_x 9.41 m — robot walked THROUGH the
  steps (user spotted it on video). Root cause: G1 feetonly sets robot
  geoms conaffinity=0, feet collide only with `<pair>`-named geoms
  (floor); steps had no pairs. Fix: 12 foot↔step/landing pairs
  (condim=3, friction 0.6) in `stairs_scene.xml` + drop-test regression
  in `test_stairs_env.py` (fails before, passes after).
- 14:15: honest baseline on FIXED env, same ckpt: 0/10, max_x 2.11 m,
  falls ~200 steps at step1/2. Video
  `videos_mjx/g1_stairs_0926_1350_eval_fixed.mp4`. Killed the
  broken-env continuation (1407); launched `g1_stairs_0926_1418` (30M,
  2048, resume 1350/30965760, CUDA verified) — first REAL stairs run.
- 14:21: user: body still clips through steps (feet-only pairs).
  Extended to all 8 robot collision geoms (feet+thighs+shins+hands) x
  6 boxes = 48 pairs; test now requires foot AND body contacts.
  TEST_STAIRS_ENV OK. Killed 1418; baseline on full-collision env
  0/10, max_x 2.09 m (`..._eval_fullcol.mp4`); launched
  `g1_stairs_0926_1426` (30M, 2048, resume 1350/30965760). Torso has
  no collision geom in feetonly (same as floor today) — accepted.
- 14:37: `g1_stairs_0926_1426` DONE 30M, eval 3.59→5.77. GATE 0/10,
  max_x 1.98 m, len 257 — FLAT vs fullcol baseline (0/10, 2.09 m),
  first flat reading. Video `videos_mjx/g1_stairs_0926_1426_eval.mp4`.
  Launched `g1_stairs_0926_1444` (+30M, resume 1426/30965760); flat
  again → BLOCKED for reshape judgment.
- 14:54: `g1_stairs_0926_1444` DONE 30M, eval 6.25→5.81. GATE 0/10,
  max_x 1.99 m, len 258 — FLAT x2 (2.09→1.98→1.99). **BLOCKED**, no
  launches. Robot walks ~2 m and falls at step1/2; 60M full-collision
  training moved nothing. Reshape candidates: feet_clearance reward
  on, pushes off, taller swing, denser climb reward, slower commands.
- 15:05: user: 15:00 freeze REMOVED; commit loop killed; plan = reshape
  30M now, then one big final gamble run. CENTERING BUG (user-spotted,
  traj-confirmed): spawn y±0.5 + drift → y≈+1.2 at the stair edge
  (y=1.0), falls off the side at x≈2.2. RESHAPE: spawn x±0.25/y±0.10/
  yaw±0.10, lin_vel_y=[0,0], yaw cmd ±0.1, feet_clearance −2.0 with
  max_foot_height 0.20, dense climb_x/climb_z shaping, pushes off.
  TEST_STAIRS_ENV OK. Launched `g1_stairs_0926_1508` (30M, 2048,
  resume 1444/30965760, CUDA verified). BLOCKED lifted.
- 15:18: `g1_stairs_0926_1508` DONE 30M reshaped, eval 8.12→11.19.
  GATE 0/10, max_x 2.06 m — position FLAT vs 1444 (1.99 m), but len
  862 vs 258: survives by CROUCHING (h 0.44 vs 0.74 m, straight down
  to ~0.6). Local optimum: cower at the steps, don't climb. Video
  `videos_mjx/g1_stairs_0926_1508_eval.mp4`. No auto-launch; final
  gamble is user's call. Anti-crouch lever ready: base_height scale
  (currently 0.0) punishes root-height deviation.
- 15:28: user: option 2, 30-min final run, supervisor loop disabled.
  Anti-crouch shipped as one-sided cost below root 0.70 (−1.0; full
  crouch −0.26/step), NOT playground base_height (two-sided, would
  punish landing height). TEST_STAIRS_ENV OK.
- 15:33: user: reward feet planted on steps (crouch = knees rest,
  feet float). Plant bonus +0.03/step: any foot↔step contact while
  tall (either foot OK for swing). Functional test proves it fires.
  Killed 1529 (9M, no plant); launched FINAL `g1_stairs_0926_1535`
  (80M ~30 min, 2048, pushes off, resume 1508/30965760). ETA ~16:05.

## Watch items

- VRAM peak at 1024 envs (dmon in /tmp/gpumon.log): if <6GB, try 2048 next run.
- `naconmax = 8 * num_envs` is a guess from the default formula; too small
  errors early and loud — adjust if the run dies at first rollout.
- Never `git push` (no git in this fork anyway). Never touch `~/unitree_rl_gym`
  or the other folder.
