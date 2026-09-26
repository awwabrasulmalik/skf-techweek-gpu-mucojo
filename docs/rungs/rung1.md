# Rung 1 — low stairs 0.09 + obstacles (first attempt)

- Config: 5 steps rise 0.09/run 0.28, friction 0.6–1.0, obstacles 6 cm at
  x=-0.8/-0.4. Resumed from promoted rung0 brain, +2 M steps (~2 h).
  Periodic checkpoints landed (rung1_ckpt_*_steps.zip) — interrupt-safety
  proven in production.
- Gate: 0/10, mean len 712/1000, mean final x -0.65 m (spawn -1.2).
- Video: `videos/rung1_post2M.mp4` (seed 0 survives full 600; stairs and
  obstacles now drawn — 7 green boxes).

Not/why (yet): approaches the stairs (+0.55 m mean) but hasn't engaged the
first step. Not zero progress (survival + forward motion both alive), so per
the rung-freeze rule: CONTINUE 2 M more from this checkpoint before any
reshape. Watch mean final x crossing 0 as the climb-start signal.

## Update — ~6.6 M steps (second 2 M block, gate 0/10, len 660, x +0.34)
- Reaches the stairs (x=0 crossed, +1.54 m from spawn) but slides around
  them: feet never lift, so steps are unclimbable. User-spotted, confirmed
  on video. Classic shuffle exploit (tracking + slip penalty favor gliding).
- Reshape (committed, NO approach change): foot-clearance reward —
  W_LIFT 8.0 × capped clearance above 0.06 m, gated by forward speed
  (ramps 0→1 at 0.5 m/s). Task-space (ankle height), NOT knee-joint angle:
  joint rewards invite crouch exploits; clearance pays only for real lift
  while moving. Max ~1.0, half of tracking.
- Continuing from this checkpoint with the lift reward. Backups in
  `checkpoints/backup/` (rung0 + pre-lift rung1) — a bad run can't destroy
  the previous model.

## Update — +2 M lift reward (gate 1/10, len 470, x +0.46)
- First rung-1 SUCCESSES (1/10): feet lift, stairs engaged, mean x past the
  staircase start. Len dropped (660→470) because successes terminate early
  and failures fall challenging steps — expected shape during climb learning.
- Works/why (partial): clearance reward broke the shuffle; tracking + lift
  compose into step-up attempts. Not promoted — continuing same rung from
  `rung1_0926_0209.zip`. Video: `videos/rung1_lift.mp4`.

## Update — bypass exploit found (user-spotted, trace-confirmed)
- Joint trace (seed 1): pelvis y drifts -0.20 → -1.71 while x advances:
  the policy walks AROUND the stairs (span y ±0.6), never engaging them.
  Knees frozen ≈-0.10, ankles at ground 0.03–0.05: zero lift. The 1/10
  "success" was a walk-around, not a climb — x-metric alone can't prove
  climbing.
- Fix (committed): corridor termination |y|>0.6 (no success), so bypassing
  ends the episode unpaid. Lift reward kept: with the corridor, lift is the
  only way forward. Killed the bypass-training run (was entrenching
  walk-around); relaunched from `rung1_0926_0209.zip` on the fixed env.
- Also fixed: VecMonitor wrap — SB3 never added it, so logs had no rollout/
  section (PPO learns fine without it; purely observability).

## Update — +2 M corridor (gate 0/10, len 524, x -0.12)
- Bypass impossible now; trace shows the transition cost: the policy kept
  its learned diagonal habit (y -0.31 → -0.60) and dies AT the corridor wall
  (step 279). Knees still frozen — lift untested until approach straightens.
- Not/why (transition, not stall): ep_rew rose 444→504 across the run; the
  policy must unlearn walk-around before climb learning starts. Continuing
  same rung from `rung1_0926_0247.zip`. If the next gate still dies at the
  wall, reshape candidate: y-centering reward or narrower spawn noise.
- Video: `videos/rung1_corridor.mp4`.

## Update — +2 M corridor cont. (gate 0/10, len 477, x -0.10)
- Still dies AT the wall (seeds 3+7: y=-0.60 exactly, upright, unfallen).
  Two corridor runs flat → protocol reshape trigger. Diagnosis: termination
  is a sparse don't-die-there signal; the veer starts 100s of steps earlier.
- Reshape (committed): y-centering reward -|y|/0.6 per step (max -1.0 at the
  wall) — dense gradient toward lane center from step 1. Relaunched from
  `rung1_0926_0323.zip` (backed up). Video: `videos/rung1_corridor2.mp4`.

## Update — +2 M y-center (gate 0/10, len 498, x -0.57)
- Y-centering WORKED: traces show y -0.09/+0.15/+0.01 (centered lane),
  zero wall deaths. New failure mode: falls AT obstacle1 (x=-0.4, 6 cm),
  h 0.35-0.37 — trips instead of stepping over. Knees/lift now the binding
  constraint, exactly where the curriculum wants it.
- Improving (deeper failure mode, centered approach) → CONTINUE unchanged
  from `rung1_0926_0355.zip`. The lift reward finally gets a fair test.
  If the next gate still trips with no lift attempt, boost W_LIFT.
- Video: `videos/rung1_ycenter.mp4`.
