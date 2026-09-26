# Slides — 3:00 total, 6 slides. Criteria tags: [G]eneral / [S]KF.

## Slide 1 — Title (0:00–0:20) [G: clarity]
**Teaching humanoids the stairs, the reproducible way.**
SKF track · Gothenburg Tech Week × Chalmers 2026
One-liner: program the skeleton, randomize the world, learn the residual,
keep the human grading.
Visual: title + best robot still (poster frame, see assets/).
Notes: land the one-liner; it is the talk's spine. No agenda slide.

## Slide 2 — Problem + evidence (0:20–0:50) [G: relevance] [S: reasoning]
SKF engineers struggle to deploy humanoids because locomotion training
for stairs/obstacles/slips is slow, brittle, undocumented — pilots die
before production. Providers told us: a documented, reproducible
pipeline beats raw performance. Our 7-method × 8-axis comparison agrees:
no single method wins locomotion (manual: rigid; RL: unsafe; imitation:
no stair data; VLA: immature) — hybrids win.
Visual: gap arrow (pilot → production) + mini verdict strip.
Notes: one breath for problem, one for evidence. Full table in backup.

## Slide 3 — Choice + pipeline scaffold (0:50–1:25) [G: innovation,
feasibility, technical quality] [S: testable sim]
Choice: gated stand→walk→stairs curriculum carrying RL-residual policies
over classical baselines; friction/obstacles/turns as randomization axes
from rung 0. Isaac rejected on evidence (setup risk); VLA parked as next.
Scaffold: Collect → Train → Test → Report, every stage re-runnable
(commands in reproducibility.md). Gates: 10/10 stand, 7/10 walk, 7/10
stairs. 2048 robots, ~50k steps/s on one RTX 4060.
Visual: 4-stage pipeline + curriculum ladder with gate badges.
Notes: the providers' must-have slide. Do not rush it.

## Slide 4 — DEMO video (1:25–2:05) [G: technical quality] [S: visible effort]
VIDEO PLACEHOLDER → best full-episode mp4 (see assets/README.md).
REFRESH: current best `videos_mjx/g1_stande_0926_1029_eval.mp4` (stand).
Narrate live: baseline vs learned (recovery steps, push handling).
Fallback if training stalls: PD-stand + representative fall + curve.
Notes: offline file (USB + laptop). Talk over it; never dead air.

## Slide 5 — Results, honestly (2:05–2:35) [G: feasibility] [S: reasoning]
Gate table (REFRESH from docs/results.md) + TensorBoard curve.
Works/why (top 1) · Not/why (top 2, e.g. drift binds stand; pushes-off
reshape failed). Every claim → checkpoint + command (reproducibility.md).
Visual: gate table + curve + one "failure → fix" callout.
Notes: failures-with-fixes score reasoning. Never hide them.

## Slide 6 — SKF value + next + ask (2:35–3:00) [G: user value,
scalability]
Use-case pipeline for one SKF stair: measure → train (this curriculum,
seeded) → test on held-out stairs → 10/10, no falls, ±20% friction.
KPIs (proposed): programming-hours ↓, escorted trials ↓, falls avoided ↑.
Next: calibrate sim, Colab handoff, operator-marked failures as new
rungs, VLA task layer. Ask: stair measurements + motion data → pilot.
Visual: use-case strip + KPI mini-table + one-sentence ask.
Notes: end on the ask: "a pipeline SKF can re-run, trust, and extend."

## Backup A — full 7×8 comparison [G: relevance]
Verbatim docs/analysis-7methods.md table + locomotion/manipulation/
workflow split. Use if: "why not teleop / VLA?"

## Backup B — reward audit [G: technical quality]
Knees one-way by physics; phase-gait kills shuffle + hop; stand air-time
exploit found and removed; height floors in gates.
Use if: "how do you know it really walks?"

## Backup C — sim-to-real honesty [S: testable sim]
Gap list + what closes each, in order. Use if: "real G1 when?"
