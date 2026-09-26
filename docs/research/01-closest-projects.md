# Closest projects — G1 / humanoid stair climbing

What we borrow: two-stage blind-then-perceptive training, planner-guided
residuals, asymmetric critic, boolean stair-mode switch, adaptive multi-rise
stairs for testing.

| System | Result | Pattern we reuse |
|---|---|---|
| Layered blind-stair (G1) | Blind stabilizer pretrain, then perceptual fine-tune; layered beats one-stage on G1 | Our stage split: residual base first, vision corrections later |
| FastStair | DCM foothold planner pretrains a safety base policy, then RL; validated in MuJoCo on 15 cm steps | Frozen base + RL residual; DCM-foothold tracking reward |
| Firefighting-stair (Rudin et al. 2022 rules) | 10 terrain levels, promote/demote rules; stage 1 height 0→12 cm, stage 2 2→12 cm, width 2.0→1.4 m | Our promotion-gate ladder |
| StairMaster | Flat 0° start, then height up / tread-depth+width down to 55° incline | Curriculum endpoint anchor |
| Ascento blind | Climbs 15 cm steps via boolean stair-mode switch + asymmetric actor-critic | Stair-mode switch + asymmetric critic |
| RuN | Pretrained Conditional Motion Generator + lightweight RL residual + asymmetric actor-critic | Residual-config template |
| GaitSpan | GaitWave rhythm + H-SLIP stride shaping + residual policy, zero-shot unseen terrain | Rhythm base + residual head |
| Bipedal stair RL paper | Stair-height curriculum 0.05→0.30 m | Height schedule anchor |
| VideoMimic (CoRL25) | DeepMimic stair ascent/descent on real 23-DoF G1 | Imitation fallback evidence |
| GRAIL | 20k+ generated sequences, 90% real-G1 stair success | Synthetic-data fallback |

Stair geometry anchors: HRP-4 industrial stairs 5 steps × 24 cm tread × 18.5 cm
rise; Ascento blind 15 cm steps; AGILITY X2 mixed 15/25/20 cm adaptive stairs.
Our pipeline target (pipeline.md): 5-step, rise 17–19 cm, 10/10, no falls, no
handrail, within 2× nominal time.

## Limitations / unresolved

- All entries are prior-result summaries (prior result 3); no paper PDFs or
  project pages were inspected in this pass (DNS failures blocked direct
  fetches). Distances/thresholds for promote/demote rules are snippet-level.
- No stair-specific pretrained G1 checkpoint: the only named checkpoint
  (motion.pt) is flat-ground.
- VLA/foundation, human-feedback, and manual-programming approaches are not
  covered here — see 00-index.md gap note.

## Sources

- Prior result 3 (research-2): curriculum/DR/residual/reward evidence bundle.
- Prior result 2 (research-3): sim-to-sim and model evidence bundle.
- Prior result 5 (critic): gap analysis (no stair checkpoint, snippet-only provenance).
- Canonical links (unverified live in this pass, carry over from prior results):
  - https://github.com/unitreerobotics/unitree_rl_gym
  - https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1
  - https://github.com/unitreerobotics/unitree_mujoco
  - https://github.com/unitreerobotics/unitree_ros
  - https://github.com/josetabuyo/leggedgym-ex
  - https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
  - https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics
  - https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1
