# Colab handoff — continue training on a free GPU

## Before opening Colab (in this repo, 1 min)
1. `git push origin main` — Colab clones from GitHub; 20+ commits are local-only until you push.
2. Note your latest checkpoint: `ls -la checkpoints/` (small, ~300 KB each).

## In Colab (5 min setup, then hours unattended)
1. Go to colab.research.google.com → Upload `colab/G1_Stairs_Train.ipynb`.
2. Runtime → Change runtime type → GPU (T4 is fine, free tier).
3. Runtime → Run all. When the upload cell runs, drag your `checkpoints/*.zip`
   into the Colab Files pane, then continue.
4. Training continues from your checkpoint with EGL rendering available.
   Rung continuation ≈ 1–3 h on a free T4 — start it first thing in the morning.
5. When done: download `videos/colab_eval.mp4` (EGL-rendered robot, not stick
   figure) + `videos/colab_curves.mp4` + the new `.zip` from the Files pane.

## Gotchas
- Free Colab disconnects after ~idle/reconnect windows: the notebook saves the
  checkpoint only at the end. For long runs, lower `--timesteps` per run and
  re-run the train cell to chain (each run saves its own `.zip`).
- If `git clone` fails (private repo), upload the repo as a zip instead and
  unzip it in the clone cell.
- `MUJOCO_GL=egl` works on Colab GPUs — if it ever errors, fall back to the
  matplotlib videos, which need no GL at all.
