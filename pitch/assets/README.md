# Assets — videos and stills for the deck

## Video placeholders (slide 6 + backups)

| Slot | Wants | Current best | Refresh when |
|---|---|---|---|
| MAIN demo | best full-episode mp4 | `videos_mjx/g1_stande_0926_1029_eval.mp4` (stand, 300 fr) | every new gate video |
| Walk demo | 0.5 m/s tracking clip | TBD (walk running) | walk gate passes |
| Stairs demo | climb clip | TBD (phase 2) | stairs gate passes |
| Fallback | PD-stand + fall + curve | CPU `videos/rung0_preview.mp4` etc. | only if GPU stalls |

Selection rule: latest gate video with the highest gate score; ties go to
the longest full-length survival. Never show a clip whose gate row isn't
in docs/results.md.

## Poster frames (slide 1 + video poster)

```bash
ffmpeg -y -ss 3 -i videos_mjx/<best>.mp4 -vframes 1 pitch/assets/poster.png
ffmpeg -y -ss 8 -i videos_mjx/<best>.mp4 -vframes 1 pitch/assets/title-still.png
```

## Learning curves (slide 7)

Screenshot from TensorBoard (`checkpoints_mjx`, tag filter to the runs on
the gate table). Export at 1600px; keep axes + run names legible.

## Deck-freeze copies (before 16:00 prep)

Copy the frozen mp4 + poster + curve png to laptop AND usb stick AND
`pitch/assets/frozen/`. Play the frozen mp4 once from the stick.
