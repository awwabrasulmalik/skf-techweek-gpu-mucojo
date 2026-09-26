# Pitch folder — SKF track, 3-minute slot (17:00–17:50)

## What lives here

- `slides.md` — slide-by-slide content: title, trigger lines, visual,
  speaker notes, timing. This is the deck source; copy into slides.
- `reproducibility.md` — the pipeline scaffold: exact commands, seeds,
  checkpoints behind EVERY number and video in the deck.
- `assets/README.md` — video/poster asset list, placeholders, refresh commands.

## Rules (from the brief + schedule docs, user-corrected to 3:00)

- 3 minutes per team. Be in the pitch room before 17:00.
- Format: slide deck + documentation + simulation results with videos.
- General judging: relevance and problem understanding · innovation ·
  feasibility · user value and impact · technical quality / prototype ·
  scalability and next steps · presentation clarity.
- SKF success criteria: good reasoning and concise conclusions · visible
  effort to implement · solid working simulation testable in real life ·
  business KPIs and values.
- Every slide in `slides.md` is tagged with the criteria it serves; the
  coverage checklist below must stay 7/7 + 4/4.

## Coverage checklist (verify before 16:00 prep)

General: [ ] relevance [ ] innovation [ ] feasibility [ ] user value
[ ] technical quality [ ] scalability [ ] clarity
SKF: [ ] reasoning [ ] visible effort [ ] testable sim [ ] KPIs

## Refresh workflow (numbers go stale as training runs)

1. Train loop updates `runs_mjx/SUPERVISOR_STATE.md` + `docs/results.md`.
2. Before deck freeze (~15:00): copy the LATEST gate row + best video path
   into `slides.md` slides 4–5 (marked `REFRESH`).
3. Re-export poster frames: see `assets/README.md`.
4. Run the `reproducibility.md` spot-check (one eval + one render) so the
   deck survives "show me it running".

## 60-second answers (judge-ready, keep in sync with docs/pitch-notes.md)

1. Problem: SKF engineers can't get humanoid locomotion from pilot to
   production — training for stairs/obstacles/slips is slow, brittle,
   undocumented.
2. Evidence: providers rank a documented, reproducible pipeline above raw
   performance; our 7-method comparison finds no single-method winner.
3. Built: gated stand→walk→stairs curriculum, RL over classical baselines,
   2048 robots in parallel on one GPU, every rung gated and documented.
4. Learned: [REFRESH — current: stand 3/10, drift is the binding failure;
   curriculum + reward audits in rung notes].
5. Next: calibrate sim to SKF stairs, operator-feedback loop, VLA task
   layer; ask: stair measurements + motion data → pilot.
