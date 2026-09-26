# Pitch notes — SKF track (Route A, stair climbing)

## 60-second answers

1. **What problem matters?** SKF workshop engineers struggle to deploy
   humanoids for shop-floor work because training locomotion that survives
   stairs, obstacles and slippery floors is slow, brittle and poorly
   documented, resulting in pilots that never reach real production.
2. **What evidence supports it?** Provider Q&A (Fri evening) put a
   documented, reproducible training-and-testing pipeline for stair climbing —
   judged with obstacles, slippery floors and stair turns in mind — above raw
   task performance. Trust and reproducibility outrank a one-off climb.
3. **What did you build?** A gated curriculum ladder (flat → low stairs →
   full 5-step stairs → stairs with turn) carrying an RL-residual policy over
   a classical PD baseline in MuJoCo on a 29-DoF G1, with domain randomization
   on friction, obstacles and step geometry from rung 0 — every rung with a
   works/why-or-not/why note, simulation videos, and a Colab handoff.
4. **What did you learn?** [UPDATE IN MORNING from MORNING.md + docs/rungs/:
   success rate per rung, held-out results, one-line why.] Backup line if
   training stalled: the env + curriculum + baseline videos still prove the
   pipeline is real, and the rung notes turn the failure into specified next
   experiments.
5. **What happens next?** Calibrate the sim against SKF stair measurements,
   continue training on the Colab GPU handoff, add operator-feedback failure
   clips as new held-out variants; then a VLA task-switching layer over the
   frozen stair skill. Credible SKF follow-ups: mentorship, thesis, pilot.

## 5-minute pitch outline (parallel-track slot, 17:00–17:50)

| Time | Slot | Content |
|---|---|---|
| 0:00–0:45 | Problem | Workshop stairs/obstacles/slips kill pilots; missing piece is a trusted pipeline, not an algorithm. |
| 0:45–1:30 | Evidence | Provider steer: docs + reasoning first; Route A depth commitment; 7-method comparison → no single-method winner. |
| 1:30–2:00 | Choice | Hybrid verdict in one line: program the skeleton, randomize the world, learn the residual, keep the human grading. Rejected Isaac (feasibility evidence). |
| 2:00–3:30 | DEMO 1 — sim video | Best-episode stair climb mp4 (`videos/stairs_climb.mp4`); narrate baseline vs residual contribution. Fallback: PD-stand + representative fall + learning curve. |
| 3:30–4:15 | DEMO 2 — rung notes | One rung note on screen (works/why or not/why); learning-curves mp4; held-out table (unseen friction 0.5, rise 0.20, turn). |
| 4:15–5:00 | Next + ask | Colab handoff exists; SKF use-case pipeline (Collect→Train→Test→Report) + success measures; ask: stair measurements + motion data → pilot. |

## Demo checklist (morning)

- [ ] `videos/stairs_climb.mp4` plays offline (copy to laptop + USB stick).
- [ ] Fallback mp4s verified if training stalled (baseline stand / fall).
- [ ] One `docs/rungs/rungN.md` screenshottable for the deck.
- [ ] MORNING.md numbers copied into section 4 above.
- [ ] Be in pitch room before 17:00 (Sat schedule).
