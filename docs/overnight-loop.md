# Overnight loop protocol (read this at every check-in)

Repo: /home/awwab/projects/nvidia-hackathon/gbg-techweek-hackathon-Placeholder
All commands run from repo root. Training runs DETACHED (survives between
turns); this loop only supervises.

## State (how to see what's happening)
- Training alive? Count PARENT processes only (workers match naive pgrep).
  Match ALL rungs (march runs use `--rung 0` — a `[12]`-only pattern misses
  them and risks a duplicate launch):
  `pgrep -af 'train\.py --rung [0-9]' | grep -v -- '-c import' | grep -v pgrep | grep -cv 'bash -c'`
  → 0 = none, 1 = one run (plus its bash wrapper line, ignore it).
  NEVER start a second training run while one is alive. If count >1, keep
  the oldest parent, `kill` the rest, record the incident, and re-verify.
- Every launch uses UNIQUE out/log names (`--out checkpoints/rung<N>_<tag>`,
  log `train_<rung>_<tag>_loop.log`, tag = date+time) so two runs can never
  share/overwrite checkpoints or logs. Incident 01:55 2026-09-26: a
  predecessor run and the loop run shared `--out checkpoints/rung1` and one
  log; adopted the survivor (PID 21895), no data lost (periodic ckpts).
- Progress: `tail -25 train_<rung>_loop.log` (SB3 table: ep_rew_mean,
  ep_len_mean, total_timesteps).
- Checkpoints: `ls -la checkpoints/` (periodic `*_ckpt_*` = interrupt-safe).

## Check-in procedure (every ~25 min)
1. If a run is alive AND healthy (fps >200 in log, log updated <10 min ago):
   just report one line (rung, timesteps, ep_rew_mean trend) and end. No commit needed.
2. If a run is alive but STALLED (no log update >15 min, or fps collapsed):
   kill it (`pkill -f "train.py --rung"`), record in MORNING.md, treat the
   latest periodic checkpoint as the run result, continue at step 3.
3. If NO run is alive (finished or killed):
   a. Gate: `.venv/bin/python prototype/stair_climb/eval.py --ckpt <latest>.zip --rung <N> --episodes 10`
   b. Video: `eval_render.py --ckpt ... --rung <N> --out videos/rung<N>_<steps>.mp4`
   c. Write/update `docs/rungs/rung<N>.md` (works/why or not/why) + `docs/results.md`.
   d. Decide (in order):
      - gate ≥7/10 → promote: next run on rung N+1 resumed from this checkpoint
        (rung 3 done = curriculum complete → go to Morning package).
      - gate <7/10 but improving (len/x up vs last eval) → continue same rung
        from latest checkpoint, 2 M steps.
      - gate <7/10 and flat/zero progress twice in a row → ONE reshape
        (reward/authority/curriculum), document it, continue; if still flat
        after that, freeze the rung, write the not/why note, move morning
        focus to Route B packaging (never leave the repo unpitchable).
   e. Launch next run detached with UNIQUE names (tag = `date +%m%d_%H%M`):
      `setsid -f bash -c "cd <repo> && .venv/bin/python prototype/stair_climb/train.py --rung <N> --timesteps 2000000 --resume <ckpt> --out checkpoints/rung<N>_<tag> --tb runs/ >> train_<rung>_<tag>_loop.log 2>&1" </dev/null`
      then verify with the parent-count command + `head -5 <log>`. After a
      promoted gate, copy the winning `.zip` to the canonical
      `checkpoints/rung<N>.zip` name so later resumes/Colab keep working.
   f. Update MORNING.md (living status + demo commands). Commit everything:
      `git add -A && git commit -m "<rung>: <result> (<metric>)"`.

## Stop conditions
- NO time-based stop. Keep iterating (train → eval → promote/reshape) until
  the user sends an explicit stop prompt. Hard context: track pitch Sat
  17:00, pitch prep from 16:00, pitch workshop 14:00.
- From 13:00 on: prioritize pitch packaging over new training — finish the
  in-flight run, then videos + results + pitch notes first; start a new run
  only if packaging is fully current.
- Curriculum complete (rung 3 gate passed) → keep improving (held-out
  robustness, turn quality, second seed) until stop or 13:00 packaging rule.
- Packaging bar (maintained continuously, not once): best `eval.py` per rung,
  best videos rendered, `docs/results.md` + `MORNING.md` current,
  `docs/pitch-notes.md` refreshed with real numbers, everything committed.
  Every report ends with: what trained, best video paths, gate table, top 3
  pitch-prep tasks.

## Hard rules
- One training run at a time. Ever.
- Every finished run ends committed (code + docs + video + checkpoint).
- Checkpoints <25 MB are committed (Colab handoff); videos too.
- Never `git push` (user's keystroke). Never touch `~/unitree_rl_gym`.
