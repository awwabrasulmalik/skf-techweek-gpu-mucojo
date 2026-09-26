# Research index — SKF track, Route A (depth, stair-climbing)

Selected concept (from [04-create](../04-create.md)): curriculum ladder carrying an
RL-residual policy. Route A depth. Imitation bootstrap is the fallback if G1
motion data arrives; VLA is parked as next step.

- [01-closest-projects](01-closest-projects.md) — prior stair-climbing systems and what we borrow from each.
- [02-training-stack](02-training-stack.md) — trainers, models, sim-to-sim path, MuJoCo-vs-Isaac decision.
- [03-sim2real-curriculum](03-sim2real-curriculum.md) — DR numbers, curriculum schedules, residual patterns, rewards, stair geometry.
- [04-g1-data](04-g1-data.md) — G1 models, imitation/teleop datasets, data seeds.

Pipeline skeleton: [pipeline.md](../pipeline.md). Prototype: `prototype/stair_climb/`
(build_scene.py fixed 5-step 0.18/0.28 m scene, smoke.py PD-hold only).

## Limitations / unresolved

- research-0/research-1 empty: the SKF-mandated 7-method x 8-axis comparison
  (manual programming; LfD/teleop; imitation; RL; sim-to-real; human feedback;
  VLA/foundation, split locomotion vs manipulation vs full workflows + hybrid
  verdict) has no evidence and is NOT covered by these notes.
- No stair-specific pretrained G1 checkpoint exists in evidence (motion.pt is
  flat-ground). `rl_control_routine` has no public URL (SKF-provided input).
- Curriculum promote/demote thresholds and full per-term reward tables rest on
  search snippets + secondary mirrors, not inspected paper PDFs (DNS failures
  blocked arxiv/github fetches). See per-file sections.
- Prototype implements none of the researched items yet (fixed friction 1.0, no
  curriculum configs, no residual code, no reward terms, no video, no
  docs/rungs/). See 03-sim2real-curriculum.md gap list.

## Sources

Exact URLs live in each file's Sources section. No new external fetching was
done for this synthesis; provenance refs are `prior result 2` (stack/models),
`prior result 3` (curriculum/DR/residual/rewards), `prior result 5` (critic gaps).
