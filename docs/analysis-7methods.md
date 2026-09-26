# 7-method comparison — learning methods for industrial humanoids

Scope: SKF brief mandatory analysis. 7 methods × 8 axes (data effort, training
time, safety, repeatability, adaptability, compute, explainability, deployment
maturity). Split: locomotion vs manipulation vs complete industrial workflows.
Ratings use Low/Med/High (effort, time, compute) and Poor/Fair/Good (safety,
repeatability, adaptability, explainability, maturity). Our evidence base is
`docs/research/*.md` + the overnight plan; snippet-level provenance caveats in
the research index apply.

## Summary table (locomotion focus — our mandatory task)

| Method | Data effort | Train time | Safety | Repeatability | Adaptability | Compute | Explainability | Maturity |
|---|---|---|---|---|---|---|---|---|
| 1. Manual programming | High | High | Good | Good | Poor | Low | Good | Good |
| 2. LfD / teleoperation | High | Med | Fair | Fair | Fair | Low | Fair | Fair |
| 3. Imitation learning | Med | Med | Fair | Fair | Fair | Med | Fair | Fair |
| 4. Reinforcement learning | Low (self-gen) | High | Poor | Fair | Good | High | Poor | Fair |
| 5. Sim-to-real transfer | Low (sim) | Med | Fair | Good | Good | High | Fair | Fair |
| 6. Human feedback (pref/corrective) | Med | Med | Good | Fair | Good | Med | Good | Poor |
| 7. VLA / foundation models | High (pretrain) / Low (finetune) | High / Low | Poor | Poor | Good | High | Poor | Poor |

## Locomotion (stairs / step length — mandatory)

1. **Manual programming** (trajectory/ZMP scripting): explainable and safe on
   known stairs, but every new rise/run/turn is re-engineering. Fails the
   workshop context (slippery floors, turns) on adaptability.
2. **LfD / teleoperation**: a G1 teleop stair demo is expensive to collect and
   risky on real stairs; good for bootstrapping, not for covering friction ×
   geometry variation.
3. **Imitation learning** (AMP/DeepMimic-style): strong seed if stair mocap
   exists — but our evidence has no stair-specific G1 checkpoint (motion.pt is
   flat-ground; see research/04). Blocked until SKF data arrives.
4. **RL**: the only method that discovers slip recovery and turn behaviour
   without demos, but unsafe and sample-hungry on hardware — must train in sim.
5. **Sim-to-real (DR + curriculum)**: the locomotion workhorse. Friction
   [0.4–1.0], obstacle blocks, step jitter, turn platform as DR axes turn the
   providers' workshop context into training signal. Needs the reality gap
   closed by calibration, not assumed away.
6. **Human feedback**: corrective torque / preference ranking can fix one
   failure mode fast (e.g. unsafe foot placement), but does not scale to full
   gait training overnight.
7. **VLA / foundation**: zero credible overnight path to a stair-climbing
   locomotion policy on our hardware; parked as supervisor / next step.

## Manipulation (pick-and-place, wipe / clean whiteboard)

Same 7 methods, different ranking: manual programming and LfD/teleop score
relatively better (constrained workspace, grasp poses scriptable, demos cheap
and safe on a bench), while RL scores worse (contact-rich, sparse reward,
long training). Imitation + human corrective feedback is the practical
manipulation stack; VLA shines brightest here (language-conditioned task
choice) but remains immature for deployment.

## Complete industrial workflows (locomotion + manipulation + task logic)

No single method covers the full workflow. Manual programming gives the task
graph and safety interlocks (Good explainability/maturity); learned skills
fill in locomotion and grasp robustness; human feedback provides the
correction channel operators trust; VLA is a future task-switching layer.
The workflow verdict: scripted skeleton + learned skills + human override.

## Hybrid verdict — tied to our choice

We selected **gated curriculum + domain randomization carrying an RL-residual
policy over a classical PD/gait baseline** (docs/04-create.md concepts 1+2)
because the comparison leaves no single-method winner for stair locomotion:

- Residual structure = manual-programming maturity + explainability (baseline
  behaviour is inspectable; the jury sees what the residual adds) while RL
  supplies the adaptability manual code lacks.
- Curriculum + DR = sim-to-real transfer made explicit and documentable
  stage-by-stage (rung0 flat → rung3 stairs+turn, promotion gate 7/10),
  encoding slippery floors, obstacles and turns as axes from rung 0.
- Imitation bootstrap (concept 3) stays a conditional upgrade: if SKF's
  `rl_control_routine` / G1 motion data arrives, it seeds the residual instead
  of random init — same pipeline, faster rung0.
- Human feedback enters at Test time: operator-marked failure clips become the
  next rung's held-out variants (cheap, trusted correction loop).
- VLA stays parked as the workflow next step (language task switching over
  frozen locomotion/manipulation skills).

In one line: **program the skeleton, randomize the world, learn the residual,
keep the human in the grading loop, and let language command skills — not
replace them.**

## Limitations / verify in the morning

- Ratings are qualitative (Low/Med/High, Poor/Fair/Good), not benchmarked —
  say so on the slide.
- No stair-specific G1 checkpoint in evidence; imitation claims conditional.
- DR ranges and curriculum gates rest on snippet-level provenance (see
  research/00-index.md); numbers to confirm against training logs.
- Business KPIs (programming hours saved, escorted trials avoided) are
  proposed, not measured — see pipeline.md.
