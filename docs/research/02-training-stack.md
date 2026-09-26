# Training stack — trainer, models, sim-to-sim

## Decision (proposed, not yet encoded in prototype)

- Trainer: `unitree_rl_gym` (Isaac-Gym PPO on legged_gym + rsl_rl), `--task=g1`.
  Successors: `unitree_rl_lab` (IsaacLab; tasks Unitree-G1-29dof-Velocity,
  G1-29dof-Mimic incl. Dance) and `unitree_rl_mjlab` (MuJoCo-native).
- Validate: `deploy/deploy_mujoco` sim-to-sim (g1.yaml) — Isaac-train →
  MuJoCo-validate → real.
- No-GPU fallback: `leggedgym-ex` ports the same envs to Genesis and confirms
  motion.pt compatibility.
- Prototype today is MuJoCo-only (`mujoco>=3.3,<4`, numpy, imageio) with no
  torch/rsl_rl/isaac dep and no training loop — so the trainer, transfer path,
  and GPU plan (Google Cloud compute-optimized per brief, unconfirmed) must
  still be named in pipeline.md before the build can claim a training path.

## Components

- `unitree_rl_gym`: official Isaac-Gym PPO stack for Go2/H1/G1; ships
  `deploy/pre_train/g1/motion.pt` flat-ground checkpoint (drop-in
  MuJoCo-compatible per leggedgym-ex fork). License: BSD-3-Clause (per fork
  LICENSE statements; org page lists it as the locomotion RL example).
- `unitree_rl_lab`: IsaacLab train/play scripts, incl. Isaac-Velocity-Rough-G1-v0
  rough-terrain task for stair-relevant curricula.
- G1 models: MuJoCo Menagerie (BSD-3-Clause, 29-DoF, scene.xml) = canonical
  MJCF; official URDF `robots/g1_description/g1_29dof_rev_1_0.xml` (23/29-DoF
  variants); sim bridge `unitree_mujoco` (vendored copy already in repo under
  `prototype/assets/unitree_mujoco` with g1_23dof.xml, g1_29dof.xml,
  g1_stairs.xml, scene files; vendored LICENSE: BSD-3-Clause, Unitree
  Robotics 2016–2024).
- `rl_control_routine` (Unitree G1 + hands data): SKF-provided input, no public
  URL, no fallback in evidence.

## Limitations / unresolved

- `unitree_rl_gym` LICENSE file itself was not fetched (fork-statement level
  only). HF dataset licenses unchecked.
- motion.pt is flat-ground, not stairs — no stair-specific pretrained G1
  checkpoint in evidence.
- MuJoCo-vs-Isaac decision is still open in pipeline.md; requirements.txt
  cannot run any RL trainer as-is.
- GPU plan (Google Cloud compute-optimized) unconfirmed.

## Sources

- https://github.com/unitreerobotics/unitree_rl_gym
- https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1
- https://github.com/unitreerobotics/unitree_ros
- https://github.com/unitreerobotics/unitree_mujoco
- https://github.com/josetabuyo/leggedgym-ex
- Prior result 2 (research-3): stack/models/sim-to-sim evidence bundle.
- Prior result 5 (critic): MuJoCo-vs-Isaac gap, license/checkpoint caveats.
