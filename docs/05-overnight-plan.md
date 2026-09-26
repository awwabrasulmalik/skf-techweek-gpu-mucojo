# Overnight plan — MuJoCo stair-climbing pipeline (Route A, auto-fallback Route B)

Executor: unattended overnight agent. It reads ONLY this repo. Environment:
WSL Linux, Python 3.14 project `.venv`, MuJoCo 3.14 installed, RTX 4060
8 GB fully available for training (torch CUDA attempt first, CPU fallback),
PyPI degraded (allow 60 s+ resolves), no sudo, no local GL rendering (EGL
errors, OSMesa absent — verified 2026-09-25; final rendered video moves to
the Colab handoff, whose GPU does EGL).

## Goal

By morning, either Route A evidence (a trained stair-climbing policy with
videos, metrics and rung notes) or — automatically, no human needed — a
Route B package (working env + curriculum configs, baseline videos, drafted
7-method analysis, argued pipeline). Never end the night empty-handed.

## Success criteria

- A-bar: PPO residual policy climbs the 5-step staircase (rise 0.18 m,
  run 0.28 m) ≥5/10 eval episodes, no fall; `videos/stairs_climb.mp4` and
  `videos/learning_curves.mp4` exist and play; `docs/rungs/*.md` explain
  each rung (works/why or not/why); `MORNING.md` states metrics + demo.
- B-bar (fallback): env + all curriculum rungs build/load/pass smoke;
  baseline videos (PD stand, typical fall) exist; `docs/analysis-7methods.md`
  draft covers the brief's 7 methods × 8 axes; `docs/pipeline.md` extended
  with argued choices; `MORNING.md` states what trained, what did not, why.
- Handoff-bar (always): `colab/` notebook + `checkpoints/` + `COLAB_HOWTO.md`
  exist so the user can continue/finalize training on Colab GPU in the morning
  (see Colab handoff section).

## Key decisions (approved 2026-09-25; alternatives rejected)

1. MuJoCo 3.14 + Unitree G1 29-DoF, assets pinned at commit `1eb6642`
   (`prototype/stair_climb/fetch_assets.sh`). Isaac Gym/Lab REJECTED:
   legacy CUDA + gated downloads + the user's own failed setup + too risky
   unattended. Record this rejection in the pitch — it is feasibility
   evidence, not a gap.
2. Trainer: stable-baselines3 PPO (pip) over a hand-rolled PPO — less custom
   RL code to debug at 3am. torch CUDA build first (use the full 4060);
   CPU channel (verified fast) second; MJX/JAX REJECTED (CUDA plugin pain,
   GBs of downloads).
3. Action space: RL residual over the proven PD baseline (`smoke.py`,
   kp 400/20 legs). Sample-efficient and explainable: the jury can see what
   the baseline does and what the residual adds. Novelty 1.
4. Gated curriculum (Novelty 2): rung0 flat approach → rung1 low stairs
   (rise 0.09) → rung2 full stairs (0.18) → rung3 stairs + turn platform.
   Promote on ≥7/10 eval success; never train rung N+1 on an untrained rung N.
5. Workshop context encoded as domain-randomization axes from the start:
   floor friction {0.4 … 1.0} (slippery floors), obstacle blocks on approach,
   step-dimension jitter, rung3 turn. Even partial training evidences the
   providers' context. Novelty 3 (context-as-axes, not afterthought).
6. Video WITHOUT OpenGL: matplotlib stick-figure animation from logged qpos
   (FK via `mj_forward`, CPU — works) + learning-curve plots, muxed with
   imageio-ffmpeg (already installed). One EGL retry is allowed
   (`MUJOCO_GL=egl` probe); on failure proceed, do not debug GL all night.
7. Imitation bootstrap is OPTIONAL: try public retargeted clips
   (docs/research/04-g1-data.md) only if the download works first try
   (<30 min). `rl_control_routine` is deployment docs, NOT data — do not
   wait for it. SKF motion data, if it arrives in the morning, is a
   stretch goal, not a dependency.
8. GPU, two phases. LOCAL (tonight): torch CUDA build so PPO updates use
   the full RTX 4060; env stepping stays on CPU (MuJoCo has no consumer-GPU
   physics path worth the install pain). If the CUDA wheel fails twice,
   CPU torch — training continues, note it in MORNING.md. COLAB (morning):
   user runs the handoff notebook on a Colab GPU for continued training
   and the final EGL-rendered video.
9. Documentation-during-work rule (presentation backup): NO step is done
   until its doc artifact exists (see Documentation-during-work section).
   The repo must be pitch-usable if the night is interrupted at ANY point.

## Steps (in order; timeboxes are hard)

### 0. Dependencies (≤60 min, else escalate to B-fallback install note)
```
source .venv/bin/activate
uv pip install torch  # CUDA build first: use the full RTX 4060
uv pip install stable-baselines3 matplotlib
uv pip install -r prototype/requirements.txt  # re-pin actual versions
```
Validate: `.venv/bin/python -c "import torch,stable_baselines3,matplotlib,mujoco; print(torch.__version__, torch.cuda.is_available())"` prints
versions + CUDA True. If the CUDA wheel fails twice, install CPU torch
(`uv pip install --index-url https://download.pytorch.org/whl/cpu/ torch`)
and continue — record which build in MORNING.md. If torch fails twice more,
SKIP training (jump to step 6) and record why.

### 1. Gymnasium env wrapper (≤90 min)
New file `prototype/stair_climb/stair_env.py`: `gymnasium.Env` around the
generated scene. Observation: full qpos/qvel + pelvis height + torso up-vector
+ distance-to-first-step + stair-top-relative z. Action (29): residual torque
added to the PD hold (reuse smoke.py gains). Reward: +forward/up progress of
pelvis, +alive bonus/step, −torque², −tilt, −foot-slip proxy (horizontal foot
velocity while in contact — approximate via ankle body velocity).
Terminate: pelvis h<0.4 (fall), pelvis past stair end + above top step
(success), 1000 steps (timeout).
Validate: random-action rollout 1000 steps stays finite; print mean reward.
Commit point: env runs, rewards sane sign.

### 2. Curriculum configs (≤45 min)
Extend `build_scene.py` to take rung args (rise, friction range, obstacles
on/off, turn platform on/off); add `prototype/stair_climb/rungs.yaml`
(rung0..3 + promotion gate 7/10). Validate: every rung builds, loads, and
passes the PD-stand smoke (add `--rung N` to smoke.py).

### 3. Train, rung by rung (≤6 h total, GPU policy updates, 8 parallel CPU envs)
SB3 PPO defaults first (lr 3e-4, n_steps 2048, batch 64, γ 0.99, λ 0.95);
one hyperparameter change at a time, logged. Start rung0; promote only on
gate. Save checkpoint + eval video per rung. If a rung shows zero progress
after ~1.5 h (flat learning curve + 0/10 eval twice in a row): freeze it,
write its `docs/rungs/rungN.md` (not/why), and CONTINUE to the next cheaper
experiment (lower stairs, denser progress reward) — do not burn the night on
one wall.

### 4. Eval + video (≤60 min)
`eval.py`: 10 episodes on trained rungs + held-out variants (unseen friction
0.5, rise 0.20, turn) → prints success table. `make_video.py`: matplotlib
stick-figure mp4 of best episode + learning-curves mp4. Validate: both mp4s
exist, >100 kB, open with imageio read (frame count > 0).

### 5. Rung notes (mandatory, even on failure)
`docs/rungs/rung0.md … rung3.md`: what was trained, hyperparams, metric
table, "works/why" or "not/why" in 5 lines each. This is jury deliverable.

### 6. Auto-fallback packaging (runs ALWAYS; runs INSTEAD of 3–5 if training blocked)
- `docs/analysis-7methods.md` draft from `docs/research/` + brief: 7 methods
  × 8 axes table, locomotion/manipulation/workflow split, hybrid verdict.
- Extend `docs/pipeline.md` with argued choices (this file's decisions 1–9).
- Baseline videos: PD stand + one representative fall, matplotlib-rendered.
- `MORNING.md`: table of what ran (commands + metrics), what failed + why,
  exact demo commands for the pitch, top 3 morning tasks.

## Documentation-during-work (explicit backup rule)

Each step below is DONE only when its doc artifact is committed alongside
the code. If the night is interrupted after any step, the repo still
presents a coherent pipeline story:

- After step 1: `docs/pipeline.md` gains a Collect+Train section describing
  the env (obs/action/reward/termination) in words + the random-rollout
  reward mean as the first number.
- After step 2: `docs/pipeline.md` gains the curriculum table (rungs,
  gates, DR axes) mirroring `rungs.yaml`; smoke results per rung appended.
- After EACH rung trained in step 3: its `docs/rungs/rungN.md` is written
  immediately (hyperparams, metric table, works/why or not/why) — never
  batched to the end of the night.
- After step 4: `videos/` + a `docs/results.md` (success table, held-out
  table, links to mp4s). This file is the deck's evidence page.
- `MORNING.md` is a living file: update it at every step completion, not
  once at dawn.
- Slide-deck hook: keep one `docs/pitch-notes.md` with the 60-second
  answers (problem / evidence / built / learned / next) refreshed whenever
  results change.

## Colab handoff package (built during the night, run by the user in the morning)

Contents (all committed):
- `colab/G1_Stairs_Train.ipynb`: cells — 1) `pip install mujoco
  stable-baselines3 imageio imageio-ffmpeg matplotlib` (Colab ships CUDA
  torch preinstalled — no big download); 2) clone this repo at the recorded
  commit + run `fetch_assets.sh`; 3) install `prototype/requirements.txt`;
  4) continue training from `checkpoints/` (uploaded by user, see below) OR
  retrain a rung from scratch with the same seed; 5) eval + EGL-rendered
  mp4 (`MUJOCO_GL=egl` works on Colab GPUs) saved to Drive/files for download.
- `checkpoints/`: best SB3 `.zip` per completed rung + `README` with rung,
  timesteps, eval score. (Committed only if each <25 MB; else listed in
  `COLAB_HOWTO.md` with Drive-upload instructions.)
- `COLAB_HOWTO.md` (user instructions): 1) open Colab, GPU runtime
  (Runtime → Change runtime type → T4/A100); 2) upload this repo folder
  (or `git clone <url>`) + drag `checkpoints/*.zip` into Colab files;
  3) Runtime → Run all; 4) training continues from the checkpoint and the
  final EGL video appears in files for download; 5) typical runtime: rung
  continuation ~1–3 h on a free T4 — start it first thing in the morning.
Validate: notebook JSON is valid; every cell's command mirrors a command
already run locally that night (no untested Colab-only magic).

## Validation plan (exact checks)
- `sh prototype/stair_climb/fetch_assets.sh` → g1 dir present.
- `.venv/bin/python prototype/stair_climb/smoke.py --rung N` → SMOKE OK all rungs.
- Random rollout script → finite, reward mean printed.
- `eval.py` → success table printed; mp4s exist and decode.
- Morning human opens `MORNING.md` and can demo in <10 min.

## Risks
- PyPI/torch download stalls → CUDA wheel first, CPU channel second, skip
  training third (step 6 covers the night).
- PPO non-convergence → residual + curriculum + shaped rewards mitigate;
  rung-freeze rule caps losses; notes turn failure into jury evidence.
- No GL → matplotlib video path is primary, not a fallback.
- CPU throughput / host RAM: 8 envs of 29-DoF G1 is heavy but should fit;
  drop to 4 envs if the machine swaps. (The 8 GB figure is GPU VRAM, unused
  for CPU physics.)

## Non-goals
No Isaac, no real-robot deploy, no VLA, no second task (step length),
no step-01–04 rework (decisions locked above).

## Timetable sketch (relative)
T+0:00 deps (CUDA torch) · T+1:00 env · T+2:30 curriculum · T+3:00–9:00 train
(gated, GPU updates) · last 90 min: eval + video + notes + Colab package +
MORNING.md. If T+1:00 has no torch at all, training is cancelled at once and
the night becomes Route B packaging + Colab-from-scratch notebook.

## Sprint execution addendum (30-min multi-agent run, 2026-09-25 evening)

- Baseline commit on `main`; workers on git worktrees: `../sprint-builder`
  (branch `sprint/env`, code+env) and `../sprint-docs` (branch `sprint/docs`,
  markdown only). Assets (gitignored) copied into builder worktree; uv cache
  shared so venv rebuilds are fast.
- Monitor agent is read-only: inspects both worktrees + runs builder smoke
  every ~8 min, reports ON-TRACK/DRIFTING/BLOCKED with one steering action;
  orchestrator relays steering via worker messages.
- Workers commit after each working step; orchestrator merges both branches
  to `main` at the end (disjoint files — clean merge), re-runs smoke, reports.
- Sprint scope = plan steps 0–2 + analysis/pitch docs + pipeline.md words.
  Training (step 3+) belongs to the overnight run at the user's home.

## Sources (inspected 2026-09-25 unless noted)
- https://pypi.org/pypi/stable-baselines3/json (v2.9.0, py>=3.10,
  torch>=2.8, numpy<3 — fits .venv numpy 2.5.3)
- https://pypi.org/pypi/torch/json (v2.14 cp314 x86_64 wheel 555 MB —
  CUDA-first attempt needs a long timeout; CPU channel is the fallback)
- https://download.pytorch.org/whl/cpu/ (HTTP 200, 0.45 s via curl)
- https://support.unitree.com/home/en/G1_developer/rl_control_routine
  (HTTP 200 but JS-rendered, ~empty fetch — deployment docs per search;
  treat data claims as conditional)
- Repo facts (verified, no URL needed): MuJoCo 3.14 + G1 stairs smoke OK;
  EGL error + OSMesa absent; assets pinned 1eb6642.
- Secondary (prior workflow, not re-inspected): docs/research/*.md.
