# Footage-first verification — design for review

Status: **proposal, nothing built.** Written 2026-11-23 for sign-off before
any code or API spend. Implements the "footage is truth, the sheet is
supplementary" rule: when the two disagree, look harder at the footage
instead of handing the point to a human.

## 0. Measured after this was written (2026-11-23) - read this first

Three free, no-API measurements on the 2026-08-21 match overtake parts of
the proposal below. The throwaway scripts are kept as text in `docs/experiments/` (not
linted, not part of the codebase).

**The sheet's "point" is a serve attempt, not a tennis point.** Every one
of the 174 sheet points contains exactly one serve. All 25 second-serve
points sit 5-20 seconds after a one-shot first-serve point by the same
server. `reconstruct.py` scores those first-serve faults as points won or
lost - 24 phantom points. Merging each fault into its second serve moves
the reconstructed score from 7-5, 6-4, 2-1 (three sets) to 7-6, 6-3,
against SwingVision's own 7-6, 6-2. This is a bug in the reconstruction,
and it comes before any footage work.

**The footage on disk has no overlay.** It is 1080x1080 raw rally footage.
Section 1's "the overlay already prints the fingerprint" is true only of
the overlay export, which is a different file. The 5.6.0 retrieval prompt
matched stroke/spin/speed fingerprints, which cannot be read off footage
with no labels - the likely reason every sample verdict came back `low`.

**Stages 0 and 1 work, for free.** A frame-difference scan finds 123 cuts
with a clean gap in the scores (0.106 then 0.089), so 124 clips against
the stated 123 rallies. Aligning those clips to the sheet by order and
duration matches all 124, median error 0.23s, 117 within 1 second. With
the clip order shuffled the same method gets 51-57 within 1 second, so the
fit is real, though loose matches still need a per-clip margin. Three
clips checked by eye had the server at the end the sheet says.

What is in the footage: 23 of the 28 one-shot points (first-serve faults)
are absent, about half the two-shot points are absent, and nearly every
point of three or more shots is present.

Consequence: stage 2 (the paid rally-walker) is not needed for locating a
rally. The cost table in section 4 drops by the walker's share, and the
first paid step becomes judging clips that are already located.

## 0b. Measured 2026-11-24 - the overlay export and the full recording

Two more files arrived for the same match: the rallies video **with** the
shot overlay (19:43, 1080x1080) and the full recording (1:35:42, 1920x1080).
Everything below is free and deterministic (ffmpeg plus pixel matching, no
model call). Scripts: `docs/experiments/2026-11-24_*.py.txt`.

**The full recording matches the sheet's clock.** It runs 5741.9s against
the sheet's last shot at 5685.7s, and its creation time equals the sheet's
start time. Frames pulled at the sheet's serve times at the start, middle
and end (14.25s, 2617.65s, 5682.30s) each show the server the sheet names,
mid-serve. Three frames, checked by eye - not a full audit.

**The overlay agrees with the sheet shot for shot.** The overlay prints
spin and stroke per shot ("Topspin / Forehand") and adds a dot per bounce.
Reading both off every frame gives 579 shots in 124 clips. Aligned to the
sheet by shot sequence alone, all 124 clips match a point, **106 exactly**,
and 95.6% of the 573 sheet shots in those points agree (45-53% with the
clip order shuffled). The 18 inexact clips are mostly one shot off; the
reader cannot see an "Overhead" label and counts repeats by bounce dots, so
those are as likely the reader's limit as a real disagreement - not checked
one by one.

**Two independent alignments agree.** The shot-sequence alignment above
(overlay video) and the duration alignment from section 0 (raw video) were
computed from different files with different signals. They pick the same
sheet point for 120 clips, 117 of them with clip lengths equal within 0.5s.
The two exports are not identical: each has 124 clips but four differ
(overlay-only points 81, 148, 152, 175; raw-only 2, 60, 73, 149).

**What the overlay does not carry.** No player, no score, no result.
Label colour is the stroke (forehand green, backhand and volley orange);
serves come in orange, blue or cyan, cyan being a second serve 11 times of
12, and none of the three tracks who served. So the overlay confirms the
sheet's *shot list*, which is the same tracker's output shown twice - it is
not a second opinion on who won the point.

**A gap is not a missed point - one checked.** Sheet point 3 has no shots.
The full recording between point 2 and point 4 shows the server walking
from the deuce side to the ad side and nothing played.

**The serve labels are less reliable than section 0 assumed.** Points 1
and 2 are both `first_serve` from the deuce side by the same server 10s
apart, and point 4 is served from the ad side - so point 1 was a fault and
point 2 the second serve, though the sheet calls neither a second serve.
The merge rule "a `second_serve` after a one-shot `first_serve`" would miss
this pair. Serving side (same server, same side, twice running) is the
better signal; not yet measured across the match.

## 1. Two facts that change the plan

**The API does not take video.** The earlier plan said "short video clips,
not stills". Checked against the live vision docs on 2026-11-23: the
Messages API accepts JPEG/PNG/GIF/WebP images only (up to 600 per request;
animated GIFs are read as their first frame). There is no mp4 content
block. So "motion" has to be carried by a **dense, time-labelled frame
sequence**, not a clip. ffmpeg still does the cutting; what changes is that
we send 4 frames a second instead of 6 frames across 50 seconds.

That is a real difference from what was measured. The 10-point sample that
came back all-`low` used ~1 frame per 8 seconds. Whether dense frames reach
`medium`/`high` is **not known** and is the first thing the pilot measures.

**The overlay already prints the fingerprint.** Each shot shows a label such
as `Flat Forehand / 57 MPH` top-right, and the sheet records the same
stroke, spin and speed per shot. Locating a rally is therefore mostly
reading text off frames and matching it to the sheet — a much easier task
than judging a rally's outcome, and one a model can do from stills.

## 2. Pipeline

Runs only for points where the first-pass footage read and the sheet
disagree (or the first pass was `low` confidence). Points that already
agree are untouched.

| Stage | What | Model call? | Sees the sheet's outcome? |
|---|---|---|---|
| 0. Segment | Cut the condensed video into rallies | No (ffmpeg + overlay width) | n/a |
| 1. Align | Match rally clips to sheet points by order and duration | No (dynamic programming) | No |
| 2. Rally-walker | Confirm the clip by reading overlay labels | Yes, low-res | No — shot fingerprints only |
| 3. Shot-reader | Read every shot of the clip from dense frames | Yes, mid-res | No — nothing from the sheet |
| 4. End-judge | Judge how the point ended, from dense end frames | Yes, full-res | No |
| 5. Decide | Compare stages 3 and 4, then apply | No (code) | Yes — only here |

### Stage 0 — segment (free)

Find rally boundaries with ffmpeg scene-cut detection plus the existing
"Serve is 61px wide" overlay check. **`footage_rally_count` is the test:**
the segmenter must produce exactly that many clips (123 for 2026-08-21).
The earlier detector found 71–92, so this is not solved; if it cannot be
made to hit the count, stage 1 is skipped and stage 2 does the locating
alone, inside today's ±25s window.

### Stage 1 — align (free)

Clips and points are both in play order, and both have a known length
(clip duration; last-shot minus first-shot `Video Time`). A standard
sequence alignment that allows points to be skipped (the serve/return-
decided ones) gives each clip its most likely point and a margin over the
runner-up. Low-margin clips are marked ambiguous rather than guessed.

### Stage 2 — rally-walker

- **Input:** ~1 frame/s at 640x360 (299 tokens each) across the candidate
  clip and one neighbour each side; the point's shot fingerprints
  (who / stroke / spin / mph); who served; what the tracked player wore.
- **Output:** `{found, clip_start_s, clip_end_s, labels_read: [...],
  labels_matched: n, labels_expected: n, confidence, reasoning}`.
- **Pass rule (code, not the model):** at least 2 labels matched and
  ≥ 60% of the expected labels, in order. Otherwise
  `not_in_condensed_footage`, as today.
- Same parameter allowlist as `locate_point_in_condensed` — it never sees
  `point_end_type` or `point_won`.

### Stage 3 — shot-reader

- **Input:** the located clip at 4 frames/s, 960x540 (700 tokens each),
  each frame labelled with its timestamp; what the tracked player wore.
  **Nothing from the sheet** — not the shot list, not the count.
- **Output:** `{shots: [{t, hitter, stroke, landed: in|out|net|unclear}],
  server, winner, confidence, reasoning}`.

### Stage 4 — end-judge

Today's `judge_point`, unchanged in prompt and allowlist, but fed 12 frames
across the last 3 seconds of the *located* clip instead of a time-seeked
guess. Output: today's `FootageVerdict`.

### Stage 5 — decide (the `compare()` rewrite)

`compare()` keeps its current behaviour for full-length footage. A new
`decide_footage_first(walker, reader, verdict, point)` returns one of:

| Result | When | Effect |
|---|---|---|
| `agrees` | Stages 3 and 4 agree with each other and with the sheet | Flag cleared, `confirmed_by="footage"` |
| `footage_overrides` | Stages 3 and 4 agree with each other, both ≥ `medium`, server and player-end checks pass — and they differ from the sheet | Point rewritten from the footage, flag cleared, `confirmed_by="footage_override"`, an `import_note` records what the sheet said |
| `inconclusive` | Stages 3 and 4 disagree with each other, either is `low`, or an identity check fails | Stays flagged for you, both readings attached |
| `not_in_footage` | Stage 2 did not locate the rally | Stays flagged, as today |

The override is recorded per point (`footage_check.sheet_said`) so it can
be audited and undone.

## 3. The rule this changes — needs your explicit OK

CLAUDE.md says a flag clears **only when two independent sources agree**,
and today those sources are the sheet and the footage. Under footage-first
the sheet no longer gets a vote on a disagreement, so the two agreeing
sources become **two separate reads of the footage** (stage 3 sees the
whole rally and no sheet; stage 4 sees only the ending). They use different
frames and different prompts, but the same model and the same video, so
they are less independent than sheet-vs-footage. A shared blind spot — a
ball too small to see at the far baseline — would fool both.

That is the trade you asked for. It is stated here so it is a decision
rather than a side effect.

## 4. Cost (estimates, list prices, not yet measured)

Per escalated point, for a ~14s rally:

| Stage | Frames | Input tokens | 
|---|---|---|
| 2. Walker | ~40 at 640x360 | ~12k |
| 3. Shot-reader | ~56 at 960x540 | ~39k |
| 4. End-judge | 12 at 1280x720 | ~14k |
| **Total** | | **~65k in, ~6k out** |

| Model | Per point | All 123 rallies (worst case) |
|---|---|---|
| `claude-opus-5-5` ($4 / $20) | ~$0.38 | ~$47 |
| `claude-opus-5` ($5 / $25, today's default) | ~$0.48 | ~$58 |

The worst case assumes every rally escalates. The real figure depends on
how often the first pass and the sheet disagree, which has never been
measured (the sample produced no usable verdicts).

Recommendation: move the default to `claude-opus-5-5` — newer and 20%
cheaper — but only as part of the pilot, since a model change can change
verdicts.

## 5. Pilot before anything else

1. **Free:** build stage 0 and run it on the 2026-08-21 file. Does it find
   123 clips? This decides whether stage 1 exists.
2. **~$4:** run stages 2–4 on the same 10 points as the last sample.
   Report per stage: located or not, confidence, and whether stages 3 and 4
   agreed with each other.
3. **Stop conditions.** If dense frames still return mostly `low`, the
   approach does not work on this footage and the full run is not worth
   buying. If stages 3 and 4 agree with each other on fewer than half the
   points, the override rule is not safe to switch on.

Only if the pilot clears those does the full match run.

## 6. Open questions for you

1. Is the rule change in section 3 what you want?
2. Should a footage override be applied automatically, or queued as a
   one-click "footage says X, sheet said Y — accept?" in the review modal
   for the first match, until the override has a track record?
3. OK to move the footage model to `claude-opus-5-5`?
4. Pilot budget: is ~$4 approved once the API key is confirmed working?
