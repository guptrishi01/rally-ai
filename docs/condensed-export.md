# The condensed export, measured

Measured 2026-11-14 against a real match: `08-21-2026-Filtered-Rallies.mp4`
(20:03, 2.53 GB, 1920x1080 HEVC) and its own `match2.xlsx` (683 shots, 225
points, `Video Time` to 5685s = 94.8 min).

This revises the claim in CLAUDE.md that the condensed export "cannot be
used for footage checks". The conclusion survives; the reasons are now
measurements rather than inference, and the failure is narrower and better
understood than it was.

## Every video this project has been given is the condensed one

| File | Duration | Sheet's last `Video Time` |
|---|---|---|
| `08-21-2026-Filtered-Rallies.mp4` | 20:03 | 94.8 min |
| `part01__match1_p1.mp4` (2026-08-18) | 7:28 | 36.5 min |
| `part02__match1_p2.mp4` (2026-08-18) | 10:13 | - |

The two 2026-08-18 files were treated as full recordings. They are not.
**The full-length recording has never reached this project**, which means
`check_alignment()` has been correctly refusing to run the footage check on
real data the whole time, rather than silently guessing.

SwingVision's "save video" share action produces this filtered video, not
the original. It carries the annotations, which is why it reads as the
full recording at a glance.

## What the overlay does and does not carry

The overlay sits **top-right**: a court minimap, and a shot label reading
e.g. `Flat Forehand / 57 MPH`. The top-left is empty sky.

There is **no burned-in score and no rally number**. That was worth checking
carefully, because a rally number would have been a direct index into the
sheet's `Point` counter and would have solved alignment outright.

The label's colour is **not** a player flag and not a clean result flag.
Green, orange and yellow all appear; across confirmed matches green turned
up on an opponent `In` and an opponent `Out`, yellow on a tracked-player
`Out`, orange on an opponent `In`. **Nothing should be built on the colour.**

## Alignment by arithmetic: about +/-5 points

The condensed video is the rallies concatenated, so a point's position is
the running sum of earlier rally spans plus a constant padding. Solving the
padding so the total equals the real file length gives ~2.2s (or ~1.4s with
a 4s floor for the 30 zero-span points - a serve straight out still takes
real seconds on screen).

Nine probes were placed by that model and the overlay read at each:

| offset from predicted point | +3 | -2 | +5 | +1 | 0 | -4 | -5 | -4 | 0 |
|---|---|---|---|---|---|---|---|---|---|

Mean absolute error ~2.7 points, worst 5. At ~5s per point that is **+/-25s**
of video - a usable *search window*, but nowhere near enough to grab a
rally directly. Grabbing the wrong rally is worse than not checking: it
produces a confident verdict about a different point.

Within such a window the `(spin, stroke, speed)` triple is close to unique
(`Slice Backhand 26` is globally unique in this match), so a two-stage lock
- predict, then confirm by reading the label - is the obvious repair.

## What the export actually contains: the long rallies only

"Filtered Rallies" keeps all *rallies*, which is not the same as all points.
A point that ends on the serve or the return was never a rally, and those
are absent.

The boundary detector that settles it is free: every point opens with a
serve, and the overlay spells the stroke out. "Serve" renders 61px wide
where "Forehand"/"Backhand" render 100-107px, so counting serves is a
scalar comparison on a fixed band of pixels. The width histogram is cleanly
bimodal - 8.4% of frames in the serve band, 70.5% in the groundstroke band,
17.9% blank - so the detector is not the weak link.

| measured in the video | |
|---|---|
| serve displays | **71** |
| segments (scene cuts / overlay presence) | ~92 |
| median gap between serves | **16.0s** |

Against the sheet's 174 serves and 176 points with shots. Fitting the
concatenation model to each candidate subset identifies which one is
present:

| subset | points | fitted padding | median clip | vs observed 16.0s |
|---|---|---|---|---|
| >= 1 shot | 176 | 2.2s | 5.4s | far too short |
| >= 2 shots | 146 | 2.6s | 6.8s | too short |
| >= 3 shots | 107 | 4.0s | 9.8s | too short |
| **>= 4 shots** | **80** | **6.4s** | **13.9s** | **closest** |
| >= 5 shots | 55 | 11.3s | 20.5s | too long |

So the export holds roughly **the 80 longest points - about 45% of the 176
points with shots, and 36% of all 225.** It is a biased sample, not a
shortened one, and the bias is exactly backwards for verification: **every
ace, double fault, service winner and return error is missing**, because
none of them is a rally.

Order-based matching ("the k-th clip is the k-th point") is therefore
unsafe, and no amount of alignment accuracy fixes it - the points are not
in the file.

## User-confirmed ground truth (2026-11-23)

SwingVision offers only two video exports: a single rally, or "all rallies
of the match." The second produced the file above. For the 2026-08-21
match, the "all rallies" count is **123**. This replaces the ~80 estimate
from the serve-width detector, which under-counted because the overlay
label can persist across short gaps between adjacent rallies.

**No full-match export is possible** from SwingVision - the rallies-only
cut is the only video they hand back. Clearing the rally filter is not
an option the app offers.

The authoritative numbers for this match:

| | |
|---|---|
| sheet points with shots | 176 |
| rallies in the condensed video | **123** |
| coverage ceiling | **~70%** |
| structurally absent | 53 points (~30%) - the serve/return-decided ones |

The 70% ceiling is a property of SwingVision's export format, not of this
code. It cannot be lifted.

## Where this leaves it

The condensed export is the only source available, and the right path is
content-addressed retrieval - **built in 5.6.0 as `source="condensed_rallies"`**
on `FootageConfig`, with `locate_point_in_condensed()` doing the retrieval
and the pipeline skipping `check_alignment()` only when the caller has
opted in explicitly. A point whose rally the retrieval call does not
locate is marked `"not_in_condensed_footage"` and stays `needs_review=True`
- never judged on frames from the wrong rally. The summary reports
"X of Y points played verified; Z not in this footage", so the number has
a denominator every time it is shown.

## 10-point sample on the real file

Ran on 2026-11-23, 10 flagged points spread across the match, on
`claude-opus-5` with retrieval_window=25s, retrieval_frames=6.

| | |
|---|---|
| total spend | **$0.86** |
| per-call cost | $0.07-$0.11 |
| wall time per call | 3-26s (median ~8s) |
| `found=True` | 7 of 10 |
| `found=False` | 3 of 10 (two for honest reasons: wrong server, no net approach) |
| verdicts at `high` confidence | **0** |
| verdicts at `medium` confidence | **0** |
| verdicts at `low` confidence | 10 |

The retrieval discriminates well. Correct rejection reasons the model
produced include *"no approach to the net that would produce the
described backhand volley"* (pt25, which the sheet says ended on a
`Backhand Volley`) and *"the described rally begins with an OPPONENT
serve"* vs. a video showing the tracked player serving (pt93). The
shot fingerprints are being used.

But **no verdict reaches a confidence the auto-confirm rule accepts.**
Stills cannot verify stroke type or speed - the model says this out
loud - and `compare()` only auto-confirms on `high` confidence, so
under the current pipeline no point is ever cleared by retrieval on
condensed footage. That is the honest outcome: the pipeline marks
every retrieval result `"not_in_condensed_footage"` and leaves the
point for a human, with the model's reasoning attached as context.

### What this means for the full sweep

Projecting to all 174 flagged points at the sample's per-call cost:

| | |
|---|---|
| full-match retrieval sweep | **~$15** |
| additional judge calls it would trigger | **$0** (no verdict is confidence-high enough) |
| auto-confirmations at the current bar | **0** |

So the current condensed path is worth running for **the attached
reasoning** - which is richer than "needs review" alone and does help
the reviewer - but not for reducing the review slog. That is the
measurement, not a preference.

### What would change this

Both options are future work and need their own evidence pass:

- **Short video clips instead of stills.** The model's own stated
  reason for low confidence is that stroke type and speed cannot be
  read from a frame. A ~3s H.264 clip per sampled offset would carry
  motion. New prompt shape, different cost model, bigger engineering
  lift.
- **Confidence-aware auto-confirm, condensed-mode-only.** The current
  compare rule is written for full-length footage, where a single
  `high` verdict is believable because the judge saw the actual
  moment. In condensed mode, a `medium` verdict that comes with the
  right server, the right player at the right end, the right shot
  count, and a stated reason could be sufficient. But this is a
  loosening of the honesty rule and should only follow a run where the
  model actually produces medium verdicts worth trusting.

## Identifying the tracked player

Unrelated to alignment, and fixed: the vision prompt identified the tracked
player by court end (`NEAR`/`FAR`). Ends swap at changeovers - visible in
this very footage, where the near player is in a grey shirt at 0:05 and
shirtless at 0:45. `MatchRecord.player_appearance` (intake field "What you
were wearing") now describes them instead, and a positive sighting at the
end the sheet did not expect marks the point **inconclusive**, the same way
a server mismatch does. Appearance says which player to look at, never what
happened to them, so it cannot leak the outcome - asserted structurally by
`test_the_judge_is_never_given_the_recorded_outcome`.
