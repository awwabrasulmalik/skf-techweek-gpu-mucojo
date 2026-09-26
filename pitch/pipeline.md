# Training Pipeline — G1 Stand → Walk → Stairs (GPU track)

How we taught a Unitree G1 humanoid to walk up stairs in one hackathon
day on a single laptop GPU. Companion to `slides.md` (the 3-minute
pitch) and `reproducibility.md` (commands to re-run everything).

Code: `prototype/mjx_track/` · Checkpoints: `checkpoints_mjx/`
Logs: `runs_mjx/` · Videos: `videos_mjx/` · Results table: `docs/results.md`

---

## 1. The stack (what, and where it came from)

| Layer | Choice | Source |
|---|---|---|
| Robot | Unitree G1, `g1_mjx_feetonly` variant | Google DeepMind `mujoco_playground` repo (bundled MuJoCo Menagerie model), pin `4057c14` |
| Simulator | MJX (MuJoCo-XLA) on the Warp GPU backend | PyPI: `mujoco` / `mujoco-mjx` / `mujoco-warp` 3.14.0 |
| Env + rewards | Playground `G1JoystickFlatTerrain` locomotion env | `mujoco_playground` 0.2.0 (our `stairs_env.py` subclasses it) |
| Algorithm | PPO | `brax` 0.14.2 (`brax.training.agents.ppo`, JAX-native) |
| Hardware | 1× NVIDIA RTX 4060 Laptop GPU (8 GiB), WSL | — |
| Scale | 2048 parallel envs, ~35–52k env-steps/sec | ~30× the CPU track's throughput |

Why this stack: MJX compiles the whole physics + PPO update with
JAX and runs thousands of robots in one GPU batch. A 30M-step run
takes ~12 minutes wall-clock, so we could iterate through a full
curriculum (stand → walk → stairs, ~650M+ steps total) between
morning and a 17:00 pitch.

The `feetonly` robot variant matters more than it sounds: only 8
geoms collide (2 foot boxes, 2 thigh + 2 shin + 2 hand capsules).
Everything else is visual mesh with `conaffinity=0`, and feetonly
additionally sets the collision geoms' affinity to 0 — **they
collide only with geoms named in explicit `<pair>` entries.** The
stock file pairs feet with the floor and nothing else. This one
fact caused both collision bugs in §5.

---

## 2. The curriculum idea

One policy, three phases, each resumed from the previous phase's
best checkpoint (same obs/action spaces throughout, so weights
transfer directly):

```
stand (balance, no motion) → walk (flat joystick tracking) → stairs (climb)
```

Each phase narrows what the policy must learn: balance first
without the distraction of motion, then locomotion on flat ground,
then the same locomotion against step obstacles. Gates (§6) decide
promotion; nothing advances on vibes.

---

## 3. Phase 0 — Stand: learn balance, nothing else

**Purpose.** A policy that can't stand can't walk. Stand isolates
the balance problem: hold an upright pose against gravity, joint
noise, and random shoves, with zero locomotion incentive.

**Config** (`train_g1.py --phase stand` → `apply_phase`):
- All command ranges forced to `[0, 0]` — `sample_command` always
  returns `[0,0,0]`, so tracking rewards pay only for stillness.
- `feet_air_time` scale forced to 0 — kills the hop-in-place
  exploit (air-time would otherwise pay for bouncing; feet_phase
  is already masked at zero command, air-time was the hole).
- Pushes ON (default): random shoves every 5–10 s, magnitude
  0.1–2.0 — the policy must *recover*, not just pose.
- 1024 envs for the first run, 2048 after VRAM headroom was
  confirmed (~2.7 GB used of 8 GB).

**Rewards active (playground defaults, per step):**
tracking_lin_vel (+1.0, pays zero velocity), tracking_ang_vel
(+0.75), orientation (−2.0, torso tilt cost), ang_vel_xy (−0.15),
feet_slip (−0.25), feet_phase (+1.0, masked at zero cmd),
stand_still (−1.0 when commanded to move — inert here),
termination (−100 on fall), collision (−0.1). Alive/energy/torque
terms are 0.

**History.** 8 runs, ~382M steps: one 30M seed run + seven +50M
continuations (one with pushes off as a control). Gate (10 eps,
zero command, stay up 1000 steps): climbed 0/10 → 3/10, mean
episode length 212 → ~870/1000. Accepted at 3/10: perfect
stand-still is not the product — push-robust balance that walk
can build on is. The push-robust ckpt `standc_0926_0943/50135040`
became the walk seed.

---

## 4. Phase 1 — Walk: flat joystick tracking

**Purpose.** Turn balance into locomotion: track commanded planar
velocity with a stable gait, no shuffling, no hopping, stop when
commanded zero.

**Config** (`--phase walk`, resumed from the stand ckpt):
- Restored default command ranges (forward/lateral/yaw velocity).
- Full default reward set, pushes ON.
- 100M seed run + one +50M continuation, 2048 envs (~150M total).

**Why the default rewards are enough here:** the gait-phase
machinery (`feet_phase` + `feet_air_time` + `feet_slip`) pays
alternating stance/swing with clean touchdowns, which kills the
shuffle and the two-foot hop; `stand_still` punishes motion at
zero command, which kills drift. Verified by audit, not assumed.

**History.** Seed run `g1_walk_0926_1235` (100.3M): first gate
read 1/10 — then we found the bug was in *our eval*, not the
policy (see §7: wrong frame + resampled commands). Corrected
gate: **10/10 PASS** (1000/1000 steps, ~7.8 m paths). Continuation
`g1_walkc_0926_1314` (+50.1M): **8/10 PASS**. Walk declared done;
its ckpt `50135040` became the stairs seed.

---

## 5. Phase 2 — Stairs: the climb (and everything that broke)

**Purpose.** The pitch demo: walk up 5 steps onto a landing
without falling, veering off, or cheating.

**Scene** (`stairs_scene.xml`): flat approach (spawn x≈0) →
5 steps, 0.09 m rise × 0.28 m run, first step face at x=1.5 →
top landing (1 m deep, top at z=0.45). All 6 boxes are 2 m wide
(y ∈ [−1, 1]) sitting on an infinite floor plane.

**Success criteria** (shared by training and the gate):
root x > 3.3 (past the last step at 2.90 + margin) AND root
z > 0.95 (above mid-stairs — i.e. standing on the 0.45 landing).
Pays **+10 once** and terminates the episode.

**Base stairs config** (`stairs_env.apply_stairs_config`):
forward-only commands (lin_vel_x [0.3, 0.7]), tight yaw,
`swing_height` 0.22 (up from default — clear 0.09 risers + foot).

**Run policy.** User rule: stairs runs are ~30M each (~12 min),
gated + adjusted between runs — never one long gamble. (The
final run is the deliberate exception: §9.)

### 5a. Bug 1 — ghost stairs (no collision at all)

First stairs run looked great on video: 9.4 m max_x. Too great —
the robot walked *through* the steps. Cause: §1's `<pair>` rule.
Our step boxes had `contype/conaffinity=1`, but the robot's
geoms only collide with pair-named geoms, and the pairs named
only the floor. A contact probe confirmed **zero** step contacts;
the infinite floor plane carried the robot straight through.
Fix: explicit foot↔step/landing pairs. The 30M run's gate was
voided (it had trained flat walking, on haunted stairs).

### 5b. Bug 2 — feet-only collision (body clips through)

With feet paired, the robot hit the steps and fell — and its
torso/thighs passed through the boxes on the way down. Fix: pair
**all 8 robot collision geoms × 6 boxes = 48 pairs**. Accepted
gap: the torso has no collision geom in feetonly (same as with
the floor), so only the pelvis mesh can still graze an edge.
Honest baseline on the fixed env: 0/10, max_x 2.09 m, falls at
step 1/2 around step ~200.

### 5c. Bug 3 — walking off the edge (centering)

Two full-collision 30M chunks went nowhere (max_x 2.09 → 1.98 →
1.99, "flat twice" → BLOCKED). Trajectory analysis showed why:
spawn y ± 0.5 plus yaw drift walked the robot to y ≈ +1.2 —
*off the 2 m-wide steps sideways* — where it fell at x ≈ 2.2.
It never attempted the steps head-on. Fix: spawn x ± 0.25 /
y ± 0.10 / yaw ± 0.10, lateral command forced to 0, yaw cmd
±0.1. Centered approach, every env, every episode.

### 5d. Reshape 1 — dense climb shaping (the missing gradient)

The +10 landing bonus was unreachable — no episode ever saw it,
so there was **zero gradient toward climbing**. Added per-step
shaping in `StairsJoystick.step`:
- `climb_x`: +0.05 × clip(x − 1.0, 0, 2.5) — pays +x progress
  past the stair base, up to +0.125/step at the landing.
- `climb_z`: +0.5 × clip(z − 0.80, 0, 0.2) gated on x > 1.2 —
  pays height gained *on the steps*, up to +0.10/step. The x-gate
  means flat walking can never earn it.
- `feet_clearance` −2.0 with `max_foot_height` 0.20 — the
  playground cost penalizing |foot_z − 0.20| during motion, i.e.
  pay the policy to lift its feet over the 0.09 risers.
- Pushes OFF — no random knock-overs while learning precise
  stepping (robustness was already banked in stand/walk).

Result: episodes survived 3× longer (len 258 → 862)… by
**crouching** (height 0.74 → 0.44 m). A cower-at-the-steps local
optimum. Position still flat (max_x 2.06 m).

### 5e. Reshape 2 — anti-crouch + plant bonus (final run)

Two targeted terms against the observed exploit (rest on knees,
feet dangling):
- `crouch`: −1.0 × clip(0.70 − root_z, 0, 1) — one-sided cost
  below 0.70 m; full crouch costs −0.26/step. Deliberately *not*
  the playground's `base_height` term: that one is two-sided
  around a target and would punish standing tall on the landing.
- `plant`: +0.03/step when **either foot** has an active contact
  with any step/landing **while tall** (root > 0.70). Either-foot
  keeps the swing leg legal; tall-only keeps the crouch
  unrewarded; feet-only (not knees/shins) kills the knees-rest
  exploit. Smaller than climb shaping, as designed.

All shaping terms ship as metrics (`climb_x`, `climb_z`,
`crouch`, `plant`) so evals can show *why* a score moved.

---

## 6. Gating: how a run earns its next run

Every finished run is gated before anything continues it:

1. Restore the latest orbax ckpt (highest numbered dir).
2. Roll 10 episodes, fixed command (stand: 0,0,0; stairs:
   0.5,0,0), commanded through the *training* wrap stack.
3. Report `GATE: success N/M`, mean length, drift, path,
   straightness, height, max_x — and save the trajectory +
   render an mp4 (`eval_g1.py` → `render_g1.py`).

Bars: stairs needs 7/10. Stairs decisions: PASS → ask user for
the next rung; IMPROVING (more successes, or same 0 with max_x
up >20%) → continue +30M; FLAT twice (same count, max_x within
20%) → BLOCKED, reshape, no launches. Walk's bar was met at
10/10 and 8/10; stand was accepted at 3/10 with long episodes.

---

## 7. Infrastructure that saved the day

- **Checkpoints.** One orbax ckpt per eval (10/run) + final
  `params.pkl` + `env_config.json`. Absolute paths (orbax
  demands them). Resume-by-ckpt is what makes the curriculum
  possible at all.
- **Eval fidelity.** The walk 1/10 scare was two eval bugs: wrong
  observation frame and resampled (non-pinned) commands. Rule
  since: eval must mirror the training wrap stack exactly, or
  the gate is void.
- **Maintained tests.** `test_g1_env.py` (finite rollout) and
  `test_stairs_env.py`: scene geoms, centered spawn + zero
  lateral cmd, finite rollout with all metric keys, drop-test
  proving foot *and* body contacts (fails without the pairs),
  functional test proving the plant bonus fires. Every bug in §5
  now has a regression test that fails without its fix.
- **Supervisor loop.** A 5-minute cron drove training all day:
  heartbeat when healthy, gate+video+docs on finish, launch per
  the decision policy, GPU-gate before every launch (CUDA verify
  + kill-on-CPU-fallback), one run at a time, time-boxed freeze
  before the pitch. Retired (by user order) for the hand-driven
  final run.
- **Throughput.** 2048 envs × ~45k steps/s on one RTX 4060:
  ~650M+ env steps in a day, ~12 min per 30M iteration.
  Compile cache kept in-repo (survived a WSL crash that wiped
  /tmp mid-day, plus one GPU-driver loss that cost a CPU-poisoned
  run — killed, restarted, relaunched).

---

## 8. Run ledger (GPU track, 2026-09-26)

| Run | Phase | Steps | Result |
|---|---|---|---|
| 0803 | stand | 30.9M | 0/10, len 212 — loop shakeout |
| 0822–1005 (6×) | stand | +50.1M ea | 0→3/10, len → 868 |
| 1029 (no pushes) | stand | +50.1M | 3/10 control |
| 1235 | walk | 100.3M | **10/10 PASS** (after eval fix) |
| 1314 | walk | +50.1M | **8/10 PASS** |
| 1350 | stairs | 30.9M | VOID — ghost stairs (§5a) |
| 1426 | stairs | +30.9M | 0/10, 1.98 m — flat ×1 |
| 1444 | stairs | +30.9M | 0/10, 1.99 m — flat ×2, BLOCKED |
| 1508 | stairs | +30.9M | 0/10, 2.06 m, len 862 — reshape: survives, crouches |
| 1529 | stairs | 9M (killed) | superseded by plant bonus |
| 1535 (final) | stairs | +81.1M | **1/10, max_x 2.89 m** — first landing ever (ep 5, h 0.93) |

---

## 9. Final status + what's next

Walk is a solved rung (10/10, 8/10). Stairs finished **1/10
with a real landing**: the final 81.1M run (centered spawn,
climb shaping, anti-crouch, plant bonus) took mean max_x 2.06 →
2.89 m (+40%) and episode 5 climbed all 5 steps onto the
landing (h 0.93 m). Below the 7/10 bar — one success in ten,
not a solved rung — but the reshape chain converted a
face-plant policy into a mounting policy in a day. Known
limits: torso has no collision geom; landing bonus still the
only success signal; sim-only (no sim-to-real transfer yet).

Next rungs if stairs passes: taller steps, obstacle contacts,
friction randomization, turning on stairs, push-robustness back
on — then the sim-to-real conversation.

*Rule of the day: every surprise became a test, and every test
earned its place by failing first.*

