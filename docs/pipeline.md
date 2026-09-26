# Learning pipeline — stair climbing (skeleton, fills in during build)

## Stages
1. **Collect** — environment + reference motion. MuJoCo scene (stairs,
   procedural variants); Unitree G1 model; optional G1 `rl_control_routine`
   clips as imitation seed. All assets pinned by URL + hash in
   `prototype/README.md`.
2. **Train** — curriculum ladder, one rung at a time, each rung a config +
   seed + command. Rungs: flat → obstacles → low friction → straight stairs
   → stairs with turn. Policy: RL residual over a classical gait baseline.
   Promotion gate per rung (success rate over N episodes).
3. **Test** — held-out variants never seen in training (unseen friction,
   unseen stair dimensions/turn), scored with the same gates. Sim videos
   recorded per rung for the deck.
4. **Report** — per-rung "if it works, why? / if not, why?" note in
   `docs/rungs/`. This section is the jury deliverable as much as the sim.

## Collect + Train — env design (mirrors 05-overnight-plan step 1)

Gymnasium env (`stair_env.py`) around the generated MuJoCo scene, 29-DoF G1:

- **Observation:** full qpos/qvel + pelvis height + torso up-vector +
  distance-to-first-step + stair-top-relative z.
- **Action (29):** residual torque added to the PD hold (smoke.py gains,
  kp 400/20 legs) — the jury can see baseline vs learned correction.
- **Reward:** +forward/up pelvis progress, +alive bonus per step, −torque²,
  −tilt, −foot-slip proxy (horizontal foot velocity while in contact,
  approximated via ankle body velocity).
- **Termination:** pelvis height < 0.4 (fall); pelvis past stair end + above
  top step (success); 1000 steps (timeout).
- **Validation:** random-action rollout over 1000 steps must stay finite;
  record the mean reward as the first pipeline number. Commit point: env
  runs, reward signs sane.

Trainer: SB3 PPO (lr 3e-4, n_steps 2048, batch 64, γ 0.99, λ 0.95), one
hyperparameter change at a time, rung-gated (promote on ≥7/10 eval).

## Workshop-context axes (kept in mind, per providers)
Obstacle density/height · floor friction range (dry → slippery) · stair
rise/run/turn · lighting/eldom (vision only if time) · payload (carried part).

## Success measures (proposal for one SKF use case)
- Task: G1 climbs a 5-step industrial stair (rise 17–19 cm) 10/10, no falls,
  no handrail, within 2x nominal time.
- Robustness: same policy holds success on ±20% friction and unseen turn.
- Reproducibility: rerun script + seed reproduces the video numbers.
- Business KPIs (to sharpen Sat): fewer manual-programming hours per new
  layout, fewer escorted trials, downtime avoided per fall prevented.

## Collect + Train: env (step 1, done)
`prototype/stair_climb/stair_env.py` — `gymnasium.Env` around the generated
scene. Obs (77,): qpos 36 + qvel 35 + pelvis height + torso up-vector (3) +
distance-to-first-step + stair-top-relative z. Action (29,): residual torque
in [-1, 1] scaled to ±20% of each actuator's ctrl half-range, added to the PD
hold (smoke.py gains: legs kp 400/kd 20, arms kp 40/kd 2; PD recomputed every
2 ms substep, residual held at 50 Hz). Reward: +1.0·vx +3.0·max(0,vz) pelvis
progress, +1.0 alive/step, −1e-5·Σtorque², −1.0·(1−up_z) tilt, −0.5 ankle-slip
proxy (horizontal ankle speed while ankle h < 0.12). Terminate: pelvis h<0.4
(fall); pelvis past stair end + 0.2 above top step (success). Truncate 1000
steps. Hold target = reset pose + 0.03 ankle dorsiflex (pure-zero hold tips
forward, falls ~frame 1200; +0.03 stands 10000+ frames, h=0.791, no drift);
reset noise ±0.01 (±0.02 topples 3/5 seeds).
First numbers: 1000-step random rollout finite, mean reward +0.472 (11
episodes); zero-action hold stands 1000/1000 steps, mean +0.939.
Run: `cd prototype/stair_climb && ../../.venv/bin/python stair_env.py`.

## Curriculum (step 2, done)
`prototype/stair_climb/rungs.yaml` + rung knobs in `build_scene.py`
(n_steps/rise/run, friction midpoint, obstacles, turn landing).
Promotion gate: ≥7/10 eval episodes, never train rung N+1 on untrained N.

| rung | scene | friction range* | obstacles | turn |
|---|---|---|---|---|
| 0 | flat approach (0 steps) | 0.8–1.0 | — | — |
| 1 | 5 low steps, rise 0.09 | 0.6–1.0 | 2 blocks h=0.06 | — |
| 2 | 5 full steps, rise 0.18 | 0.4–1.0 | 2 blocks h=0.08 | — |
| 3 | full stairs + top landing | 0.4–1.0 | 2 blocks h=0.08 | landing 1.0 m |

\* built scene uses range midpoint; trainer samples per-episode (step 3).
Smoke (`smoke.py --rung N`, PD stand 1000 frames, bar h>0.5): rung0
h=0.777 OK · rung1 h=0.777 OK · rung2 h=0.777 OK · rung3 h=0.777 OK.
Known step-3 hook: `stair_env` success thresholds assume full stairs —
rung0/rung1 need per-rung success terms before training on them.
