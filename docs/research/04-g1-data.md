# G1 data — models, imitation seeds, teleop

## Models (see also 02-training-stack.md)

- MuJoCo Menagerie G1: canonical MJCF, 29-DoF, scene.xml, BSD-3-Clause.
- Official URDF: `robots/g1_description/g1_29dof_rev_1_0.xml`, 23/29-DoF
  variants (unitree_ros).
- Sim bridge `unitree_mujoco`: fetch-verified live by prior result; vendored
  copy already in repo (`prototype/assets/unitree_mujoco`), LICENSE
  BSD-3-Clause, Unitree Robotics 2016–2024.
- `rl_control_routine` (G1 + hands): SKF-provided input, no public URL, no
  fallback — imitation fallback (04-create concept 3) is blocked until SKF
  hands over motion data.

## Imitation / teleop datasets

- LAFAN1-Retargeted (H1/H1_2/G1): motion-capture clips retargeted to G1 —
  candidate imitation seed.
- AMASS-Retargeted (from official unitree_ros model): mocap retargeted to G1.
- AMASS-Retargeted IsaacLab-AMP-ready: AMP motion loader = direct imitation
  seed for the `unitree_rl_lab` Mimic task.
- HIW-500 (BitRobot × HuggingFace × Unitree): 500 h / 10 TB real-home G1
  teleop dataset — largest real-data option, scale vs. relevance TBD.
- VideoMimic (CoRL25): DeepMimic stair ascent/descent on real 23-DoF G1.
- GRAIL: 20k+ generated sequences, 90% real-G1 stair success.

## Limitations / unresolved

- HF dataset licenses (LAFAN1-retarget, AMASS-retarget, HIW-500) unchecked.
- No datasheets / old-simulation-results inputs from SKF contacts yet
  (Meneka Karehalli, Magnus Wahlgard — per brief, virtual support day 1).
- Relevance of 500 h home-teleop (HIW-500) to industrial stairs unassessed;
  second locomotion variant "change of step length", manipulation tasks
  (pick-and-place, wipe/whiteboard), and business KPIs have no data entry.

## Sources

- https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1
- https://github.com/unitreerobotics/unitree_ros
- https://github.com/unitreerobotics/unitree_mujoco
- https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
- https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics
- https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1
- Prior result 2 (research-3): models/imitation/sim-to-sim bundle.
- Prior result 5 (critic): license/data-availability gaps.
