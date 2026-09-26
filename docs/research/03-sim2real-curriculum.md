# Sim-to-real curriculum — DR, schedules, residuals, rewards

## Domain randomization (ranges from prior results)

- Friction: [0.1, 1.25] + base mass −1.0/+3.0 kg + max push 1.5 m/s
  (uleroboticsgroup/g1_waiter); [0.5, 1.2] + mass −1/+3 (niuverse
  mjlab-workflow); [0.5, 1.25] + CoM ±5 cm (TienKung G1 vision doc); static
  0.3–1.6 / dynamic 0.3–1.2, encoder bias ±0.01–0.015 rad, base CoM ±0.05 m
  (unitree_jump); static 0.3–1.6 / dynamic 0.3–1.2, restitution 0–0.5, pushes
  every 1–3 s (beyondmimic); static U(0.6, 1.0) (RuN).
- Prototype gap: `build_scene.py` fixes FRICTION=1.0, N_STEPS=5, RISE=0.18,
  RUN=0.28 — no DR implemented.

## Curriculum schedules

- 10 terrain levels, riser/tread scaled per level, promote/demote per Rudin et
  al. 2022 rules (firefighting-stair: stage 1 height 0→12 cm, stage 2 2→12 cm,
  width 2.0→1.4 m).
- StairMaster: flat-0° start, height up / tread-depth+width down to 55° incline.
- Bipedal stair RL paper: stair height 0.05→0.30 m.
- Blind-stair two-stage: blind stabilizer pretrain → perceptual fine-tune
  (layered beats one-stage on G1).
- FastStair multi-stage planner-guided: DCM foothold planner pretrains a safety
  base policy, then RL; MuJoCo-validated on 15 cm steps.
- Prototype gap: no curriculum configs or promote/demote rules; `docs/rungs/`
  absent.

## Residual-RL configs

- Paradigm C: frozen MPC/ZMP base + RL residual corrections.
- RuN: pretrained Conditional Motion Generator + lightweight RL residual +
  asymmetric actor-critic.
- GaitSpan: GaitWave rhythm + H-SLIP stride shaping + residual policy, zero-shot
  unseen terrain.
- FastStair: frozen blind policy + vision-net corrective residuals.
- robo_residual/SONIC: frozen SONIC base + residual MLP for new objectives
  (e.g. contact-force penalty).
- Prototype gap: no residual-policy code; smoke.py is a PD-hold (1000 steps,
  h>0.5 assert), no policy/training/video.

## Rewards (weights from prior results)

- Booster Gym: vel-track x/y 1.0, yaw 0.5, vz −2.0, base height −20.0,
  orientation −5.0 + early termination on low base height / bad orientation.
- Humanoid-Gym: orientation tracking 1.0, base-height tracking (target 0.7 m)
  0.5; weighted sum r=Σμᵢrᵢ.
- Omnidirectional-stairs: unsafe-stepping penalty (weighted edge-distance terms).
- FastStair: DCM-foothold tracking reward biasing exploration to feasible contacts.
- Standard aux terms: foot slip, torque/power, action rate, joint accel,
  self-collision, feet air-time/contact pattern.
- Prototype gap: none of these terms are implemented.

## Stair geometry anchors

HRP-4 industrial stairs 5 steps × 24 cm tread × 18.5 cm rise; Ascento blind
15 cm steps (boolean stair-mode switch + asymmetric critic); AGILITY X2 mixed
15/25/20 cm adaptive stairs.

## Limitations / unresolved

- Provenance is snippet-level: arxiv/github bodies could not be fetched
  directly (DNS failures); promote/demote distance thresholds and full
  per-term reward tables rest on search snippets + secondary mirrors, not
  inspected paper PDFs.
- Slippery-fluid friction low end (0.1–0.3) unsourced.
- Exact MuJoCo friction numbers (static/dynamic split, restitution mapping)
  still to pin down before DR configs are written.

## Sources

- Prior result 3 (research-2): DR/curriculum/residual/reward/geometry bundle.
- Prior result 5 (critic): prototype-gap list, snippet-provenance caveat.
- Canonical links (carried over, live-unverified in this pass):
  - https://github.com/unitreerobotics/unitree_rl_gym
  - https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1
  - https://github.com/unitreerobotics/unitree_mujoco
