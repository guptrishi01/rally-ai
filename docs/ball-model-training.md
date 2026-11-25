# Ball model (Track B): training, evaluation, and when to retrain

This covers how the independent tracker's accuracy is measured, how to build a
training set for the learned ball detector (Track B) from your own matches, how to
train it on a cloud GPU, how to pin the result, and how the "retraining suggested"
note decides when your footage is unlike anything the detector was tuned on.

**Nothing here trains on this machine.** It is CPU-only and short of memory
(spec section 12, decision 4). Training runs in a separate cloud-GPU session. Only
inference runs here. Weights never go in the repository.

---

## 1. Measuring accuracy: `scripts/eval_tracker.py`

```
python scripts/eval_tracker.py <pending.json> --reference export.xlsx --clock-offset 1.99 \
    [--start 355 --end 600] [--json results.json]
python scripts/eval_tracker.py <pending.json>        # confirmations only
```

It scores a match staged by `scripts/track_match.py` (a whole match or a
`--start/--end` stretch) against two references (`tracking/evaluate.py`):

* **A SwingVision export of the same match.** This is a *weak* reference.
  SwingVision has its own errors and recording gaps, so results are reported as
  agreement, never as accuracy. The script reports:
  * shot recall and precision, with recall split by court end;
  * hitter end;
  * stroke family, with the tracker's abstentions ("Unclear") counted separately;
  * shot role;
  * serve recall;
  * server end;
  * whether the server won each point;
  * bounce distance error (median, p90, mean), overall and by landing half;
  * speed difference (signed and absolute);
  * a **confidence calibration table**: for each band of hit confidence, how many
    of the tracker's shots paired with a reference shot.

  Every figure is printed as `hits/total (pct)`, and an empty measure prints
  `0/0 (n/a)`. Shots pair by time, closest first and one-to-one, within
  `--tolerance` (0.15 s; serves use 0.3 s). Without `--clock-offset`, the offset is
  estimated by pairing as many shots as possible, and the output says it was
  estimated.
* **The reviewer's own confirmations (the correction loop).** When you confirm a
  tracked point in the review modal, the confirmed outcome is a human judgement.
  The tracker's original guess can still be rebuilt from the shot rows staged with
  the match, so every confirmed point becomes a free labelled example of "who won"
  and "how it ended". This needs no new storage and no schema change.

---

## 2. The "retraining suggested" note

`pipeline.retraining_advice` asks two separate questions. Both are measured against
`src/tracking/models/ball_conditions_reference.json`.

### The reference (measured 2026-11-25, not guessed)

| | value |
|---|---|
| footage | `full_match_video (1).mp4`, SHA-256 `1ee15140b480...dbbd91`, 5741.9 s, 1920x1080 |
| sampling | 288 single frames, one every 20 s from 0 to 5740 s. One seek per frame (about 0.5 s each), so roughly 2.5 minutes in all. Only the court surface (the near half, through the calibration) is measured. |
| court brightness | min 99.3, p5 105.8, median 114.6, p95 119.7, max 123.5, std 4.4 |
| court colour | mean RGB (82.5, 118.7, 169.4); chromaticity distance of single frames from the mean: median 0.017, max 0.062, std 0.0125 |
| ball tracking | rally coverage 0.878, mean confidence 0.435 (Phase 6 chunks over 355-502 s, 7 SwingVision points, 3164 rally frames) |

The old hard-coded values were brightness 120 and RGB (70, 110, 140). Both were
wrong: the real court is about RGB (83, 119, 169).

### Question 1: does the footage look unlike the reference?

A run counts as "outside the reference" in either of these cases:

* Its court brightness lies **beyond the whole match's range** (99-124) by more than
  `max(3 x std, 25% of the median)`. That works out to below **70** or above
  **152**.
* Its **chromaticity** (each channel's share of R+G+B) is further from the
  reference mean than any reference frame was, by more than
  `max(3 x std, 0.03)`. That is a distance above **0.0995**.

How each part was derived:

* The 3-sigma margin is measured. It is the normal variation within one match
  (clouds, shadows, players crossing the court).
* The floors (25%, 0.03) are **a judgement**. No footage in other conditions exists
  yet, so nothing shows where the detector actually starts to fail.
* Chromaticity is used so that a darker picture of the same court registers as a
  brightness difference only, never as a colour difference too.

What these limits would catch:

* The dim limit, 70, is 61% of the reference's median brightness. Footage darker
  than that (for example dim floodlights or dusk) is flagged; brighter night play
  passes.
* A clay court (chromaticity about (0.45, 0.33, 0.22)) is flagged at a distance of
  about 0.33.
* A green court is flagged at about 0.26.

### Question 2: was the ball tracked worse than on the reference?

Ball tracking counts as weak if either of these holds:

* Rally coverage is below 0.6 x 0.878 = **0.53**.
* Mean confidence is below 0.6 x 0.435 = **0.26**.

Two further rules carry over from earlier phases unchanged: more than half the
points weak, or more than 60% of shots with no result. The 0.6 factor is a judgement.

If the reference had no tracking baseline, the old absolute floors would apply and
the note would say they are provisional.

### Where the note appears

The note is informational and never blocks anything. It appears in:

* the import notes;
* the review modal's "Ball model" panel;
* the CLI's `Retraining suggested (informational): ...` line.

It names the reason in plain words, for example: "footage much darker than the
conditions the ball detector was tuned on: court brightness 45 against 99-124 on the
reference recording (39% of its median) - night, dusk or heavy shade".

### Updating the reference after a retrain

Run these over every recording in the new training set:

```
python scripts/measure_conditions.py sample <video> samples.jsonl --calibration cal.json \
    --start 0 --end 1500 --every 20        # repeat in short ranges
python scripts/measure_conditions.py build samples.jsonl <video> \
    --detector ball-heatmap-v1 --tracking-baseline baseline.json
```

`--detector` names the registry entry the reference describes, so the note always
describes the model actually in use.

---

## 3. Camera setup score

`calibrate.setup_advice` turns the calibration's numbers into a rating and plain
consequences. It appears as an import note, as the "Camera setup" panel in review,
and in the CLI output.

On the reference recording:

* The setup score is 66/100, rated "fair" (bands: 75 or more is good, 50-75 fair,
  below 50 poor).
* One pixel at the far baseline covers 0.28 m of depth, about 20 times the near
  end's 0.014 m.
* The far baseline is hidden behind the net tape.

So the advice is that filming from higher up, further back and centred would help
most.

---

## 4. Robustness: what was measured

### Camera drift

The court was re-calibrated at 9 moments across the match. Each calibration was
compared with the run's first calibration, which was made at 355 s. As a noise
floor, a second calibration was made 20 s after each one.

* **Noise floor:** the paired calibrations differed by **0.7-3.4 px** (median
  1.9 px).
* **Real movement:** the camera crept away from its first position by
  **4.9-6.5 px** after about 1600 s, consistently.

Each calibration was then laid over an independent median frame 40-50 s later, and
the mean distance from its lines to the painted lines (`line_alignment_px`) was
measured:

* the first calibration: **0.66 px** on average (t >= 1200 s, 8 moments);
* the fresh calibration: **0.35 px**;
* the fresh one was better at 7 of the 8 moments (all except t = 4400 s).

Per-chunk re-calibration therefore measurably helps, and it is cheap because the
drift check already re-calibrates every chunk.

`PipelineConfig.recalibrate` (default on) adopts a fresh calibration only when both
of these hold:

* the court has moved more than `recalibrate_px = 4` px (above the noise floor);
* the fresh fit lies closer to the painted lines on 5 frames from the middle of the
  chunk.

An unjudgeable or worse fit is never adopted, and the note says which happened.

### Frame drops (VFR)

Gaps longer than the ball track bridges are listed in the staged match, along with
the points they fall inside. A gap counts when it is longer than 0.1 s and also
longer than twice the usual frame interval.

### Damaged or cut-short files

* Frames the timestamp index lists but decoding never produced are reported as
  holes, with times and counts. A chunk's shortfall is measured, never assumed.
* A file whose frames stop more than 2 s before its header's duration is reported
  as cut short.
* A file ffmpeg cannot open at all is refused with ffmpeg's reason (from Phase 2).

### Dead-time pre-pass (`--skip-dead-time`, off by default)

The pre-pass listens to the whole recording's audio (5.7 s for 95 minutes) and skips
stretches of 10 s or more with no burst of ball-strike sounds.

Results on the reference match:

* It skips **893 s of 5742 s (15.6%)** in 37 stretches.
* **0 of 683** SwingVision shots and **0 of 176** points fall in a skipped stretch.
* The closest shot to a skipped stretch is **0.41 s** from its edge.

Caveats:

* With 4 s of padding instead of 5, one shot would be lost.
* The settings were chosen on this same recording.
* The audio here is noisy: neighbouring courts are audible. That keeps many quiet
  stretches active, which is why the saving is modest.
* Skipped frames are still decoded (only tracking is skipped). Tracking is about
  5 s of CPU per video second against about 0.4 s to decode, so the expected saving
  is roughly 14% of tracking time. This was estimated, not timed: a whole-match run
  was not affordable on this machine.

Two alternatives were tried and rejected:

* **Keyframe-only motion:** the recording has a keyframe only about every 59 s.
* **Low-resolution motion over the court:** warm-up and walking between points
  overlap play too much to separate.

---

## 5. Building a training set

```
python scripts/export_ball_dataset.py <pending.json> D:/ball-data/match-2026-08-21 \
    --split train [--min-confidence 0.4] [--frames labelled|all|none] [--confirmed-only]
```

The export:

* reads the run's checkpoints, which are kept by default, to get the ball's
  per-frame track;
* writes one clip per tracked point;
* writes `labels.csv` and `manifest.json`, with the source video's SHA-256, the
  detector, the split and the counts by label source;
* writes JPEG frames.

Keep the output **outside the repository**: it is your own footage.

`labels.csv` columns: `clip, frame, pts, file, visibility, x, y, label_source,
confidence, point_number, confirmed_point, split`.

* `visibility` is 1 (position given), 0 (a person said the ball is not in the frame)
  or -1 (unknown).
* `label_source` is one of:
  * `human`;
  * `tracker-measured`: a pseudo-label from the classic detector, at or above
    `--min-confidence`;
  * `tracker-bridged`: excluded from training by default;
  * `unlabelled`: **never** to be treated as "no ball".

This is the same idea as TrackNet's per-frame label CSV (visibility, x, y), and it is
easy to convert.

### Corrections

1. Open `labels.csv` in a labelling tool (CVAT, Label Studio, or a spreadsheet next
   to the frames).
2. Fix or confirm positions, and set `label_source=human` on every row you checked.
3. Re-export with `--corrections fixed.csv`. Human rows replace tracker rows, and
   frames you added are kept.

Start with points you confirmed in review (`--confirmed-only`, or
`confirmed_point = 1`).

### Splits

Use one split per **match**, never per frame: adjacent frames leak between splits.
Keep at least one whole match, ideally at another venue or time of day, as `test`.

---

## 6. Training on a cloud GPU (Colab or Kaggle)

1. Upload the dataset folders (labels, manifest, frames) to the notebook's storage.
2. Train a TrackNet-family model. Its input is three consecutive frames, resized
   (for example to 512x288). Its target is a Gaussian heatmap at (x, y) for
   `visibility == 1` rows. Rows with `visibility == -1` get no loss.
   * The code may be AGPL or another open licence (spec section 12, decision 1).
   * Public non-commercial tennis datasets may be mixed in to bootstrap (decision 3),
     but evaluate on **your own** held-out match.
3. Evaluate on the `test` match:
   * localisation error (median and p90, in pixels);
   * detection recall and precision at a 4 px tolerance;
   * results by visibility.

   Then run `scripts/eval_tracker.py` on a match tracked with the new model and
   compare with the classic detector's numbers.
4. **Export for CPU inference as TorchScript**, with preprocessing inside the module.
   The module must take a `(3, H, W, 3)` uint8 tensor (the frames as the tracker
   holds them) and return an `(H, W)` float heatmap in 0..1 at the input resolution.
   This keeps resizing and normalisation pinned to the weights:

   ```python
   class Exported(torch.nn.Module):
       def __init__(self, net): super().__init__(); self.net = net
       def forward(self, frames):                      # (3, H, W, 3) uint8
           x = frames.permute(0, 3, 1, 2).float() / 255.0
           h, w = x.shape[-2:]
           x = torch.nn.functional.interpolate(x, size=(288, 512), mode="bilinear")
           heat = self.net(x.reshape(1, 9, 288, 512))[0, 0].sigmoid()
           return torch.nn.functional.interpolate(heat[None, None], size=(h, w))[0, 0]
   torch.jit.script(Exported(model).eval()).save("ball-heatmap-v1.pt")
   ```
5. Download the `.pt` file. Do not commit it.

## 7. Pinning the weights

1. Copy the file into the weights folder, outside the repo
   (`%LOCALAPPDATA%/rally-ai/models` on Windows; see `models.default_weights_dir`).
2. Hash it:
   `python -c "from tracking import models; print(models.weights_checksum('<path>'))"`.
3. Add or replace the registry entry in `src/tracking/models/__init__.py`:

   ```python
   BALL_HEATMAP = ModelSpec(
       name="ball-heatmap-v1", task="ball_heatmap", backend="torch",
       weights="ball-heatmap-v1.pt", release="first-party 2026-12-01 (Colab, 3 matches)",
       licence="first-party", sha256="<the hex digest>",
   )
   ```

   Then set `ModelConfig.ball_model` to its name.
4. `models.load_ball_heatmap(config)` checks that the file exists and that its
   SHA-256 matches. It refuses an unpinned or different file, loads it with
   `torch.jit.load(map_location="cpu")`, and returns a `ball.HeatmapModel` whose
   provenance source is `name@release`. Give it to the pipeline:
   `TrackerStages(ball_detector=lambda cal, fps: ball.HeatmapBallDetector(loaded.model, config.ball, loaded.source))`.
5. Re-measure the conditions reference over the new training recordings, with
   `--detector ball-heatmap-v1` (section 2), and record the new tracking baseline.

## 8. How the retraining flag ties in

* The reference file records which detector it describes and which footage it was
  measured on.
* When a tracked match's lighting or court colour falls outside that footage's range,
  or its ball coverage falls well below that footage's, the note says so and names
  the reason.
* Those matches are the ones to export (section 5), label, and add to the next
  training run, as a held-out `test` match first.
* After the retrain, update the reference (section 2), so the next note is judged
  against the conditions the new model has actually seen.
