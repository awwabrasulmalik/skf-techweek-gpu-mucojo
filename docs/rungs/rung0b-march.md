# Rung 0b (march) — flat walking WITH lift (inserted rung)

Why: rung 1 exposed that rung 0's graduate shuffles (knees 0.11, ankles at
stance height) and then degenerated (one seed stood still 1000 steps to farm
alive+centering). Lifting was never learned because flat walking never
required it. This rung teaches stepping before stairs are attempted again.

Setup: rung-0 flat scene, from clean `backup/rung0.zip` (pre-shuffle walker),
2 M steps. New shaping (global, stairs benefit too):
- Alternation bonus W_ALT 4.0 × min(|zl-zr|, 0.15): ~0 when standing/hopping,
  large when feet take turns — stepping without a phase clock.
- LIFT_CAP 0.12 → 0.15 (clear 0.09 steps + foot, with margin).
- Eval now reports foot-height-split; marching needs >0.10 m.

Gate: 7/10 crossings AND visible lift (split >0.10). Promote → retry rung 1
from the march brain.
