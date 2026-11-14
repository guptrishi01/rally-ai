# Phone companion — architecture

Status: **design agreed, not yet built.** Written 2026-11-14. Tests come
next, then code.

## The problem

Getting a match into Rally AI currently takes four manual steps: export the
spreadsheet and the recording from SwingVision, push both to Google Drive,
go home, download them, then fill in the intake form and upload. The
information that matters most — how the match actually felt — is entered
hours later, from memory.

The cause is that **capture and transfer are the same step today**. Separate
them and the Drive round-trip disappears.

## Shape of the solution

A small phone app captures the match at the court and queues it. A laptop
that is asleep in another building is no longer in the critical path.

```
  At the court                 At home, same Wi-Fi
  ------------                 -------------------
  fill the form                open the PWA
  submit  ──────────┐          attach xlsx + video
                    │          push  ──────────────┐
            queued locally                         │
            (metadata only)                        ▼
                    └────── metadata syncs ──► Rally AI (Flask)
                            when reachable         │
                                                   ▼
                                       ingest → footage verify
                                       → review → finalize
```

The dashboard stops being an intake form and becomes a progress view.

## Decisions, and why

| Decision | Choice | Why |
|---|---|---|
| App type | **PWA**, served by Flask at `/mobile` | iPhone + free rules out everything else: native iOS needs a Mac (we're on Windows), and a free Apple account expires apps every 7 days. A PWA costs nothing, installs to the home screen, works offline, and ships from the server that already exists. |
| What is queued at the court | **Metadata only** | iOS Safari can evict site storage. Match notes are irreplaceable; the video is still sitting in Photos. Queueing ~400 bytes cannot fail the way queueing 11 GB can. |
| When files are attached | **At home, over Wi-Fi** | They were never going to upload over cellular anyway. This removes Drive without pretending an 11 GB cellular upload is reasonable. |
| Handoff | Phone pushes; dashboard displays | A phone finding a laptop on a home network is far more reliable than the reverse, which needs the phone to run a server at a stable address. |
| Upload | **Chunked and resumable** | A full recording is ~11 GB. A single POST that dies at 80% is unacceptable, and iOS Safari will not reliably hold one connection that long. |
| Compression | **Laptop-side**, after upload | ffmpeg is already a dependency (footage verification). Doing it on-device would need WebCodecs, which is awkward on iOS, and the laptop has to touch the file regardless. |
| Review | Stays on the laptop | Shot tables and footage verdicts need a real screen, and a phone-sized review invites exactly the skim-past-it behaviour the gate exists to prevent. |
| Streak | Weeks containing a match, keyed on the entered match date | Playing is what counts; review chores must not be able to break it. |
| What counts as a match | Only a match with a SwingVision export | "A match doesn't count unless it can be logged." Unrecorded play is practice or hitting with friends, which belongs in notes rather than the record. |
| Score | **Per-set, typed at the court, used as ground truth** | The reconstruction gets the scoreline wrong (3 set-completions on a 2-set match) and `quality_check` can currently only compare it against SwingVision's own Sets sheet, which is not independent. A typed score is the one external reference the pipeline has never had. |
| Progress UI | Both, with the phone as the daily surface | Duolingo works because it is on the phone you already check. A streak you only see at a laptop is not doing much. |
| Notifications | One weekly nudge, best-effort | Fires only when the week is nearly over with no match logged. Needs the laptop awake to send, so a missed nudge is not a bug. |

## Submission lifecycle

A submission is the unit the phone owns until the server accepts it.

| State | Where | Meaning |
|---|---|---|
| `draft` | phone | Being filled in. |
| `queued` | phone | Submitted at the court. Metadata saved locally; nothing sent. |
| `awaiting_files` | server | Metadata reached the laptop. Visible on the dashboard. |
| `uploading` | both | Chunks in flight; resumable. |
| `staged` | server | Files complete, `ingest()` ran, a pending JSON exists. |
| `failed` | both | Something went wrong; the reason is kept and it can be retried. |

**Metadata syncs as soon as the laptop is reachable, ahead of the files.**
That is what lets the dashboard say *"Match vs Alex on 14 Nov — awaiting
files"* without the laptop having to interrogate the phone. The files then
attach to a record the server already knows about.

## API surface

New routes, all requiring a paired token. Existing routes are untouched.

| Route | Purpose |
|---|---|
| `POST /api/pair` | Exchange a short-lived pairing code for a device token. |
| `POST /api/submissions` | Create from metadata. Carries a client-generated idempotency key. |
| `GET /api/submissions` | List; drives both the dashboard banner and phone reconciliation. |
| `PUT /api/submissions/<id>/files/<slot>` | Upload a chunk (`Content-Range`). Re-sending a chunk is a no-op. |
| `POST /api/submissions/<id>/complete` | All chunks present: assemble, transcode, call `ingest()`/`ingest_multi_part()`. |
| `DELETE /api/submissions/<id>` | Abandon one. |

`/complete` is the only route that touches the pipeline, and it calls the
same functions the web form does. **No parallel ingest logic.**

Files land in `data/submissions/<id>/` (gitignored) while incomplete, and
move to `data/uploads/` and `data/media/` on completion, so the existing
archiving and pairing rules apply unchanged.

## Security

Today the app binds `127.0.0.1` and CLAUDE.md says it is "never meant to be
exposed beyond localhost". The phone needs it reachable, so:

- LAN binding is **opt-in**: `python webapp/app.py --lan`. Default unchanged.
- Pairing: the laptop displays a QR containing the LAN URL and a one-time
  code; the phone exchanges it for a long-lived device token.
- Every `/api/submissions*` call carries the token. Without `--lan`, these
  routes still work on localhost for tests.
- The token is per-device and revocable by deleting it on the laptop.

This is a real loosening of the current stance and should stay behind the
flag, documented as such.

## Idempotency

The pending-JSON filename is `date_opponent`, and today a re-submit
**silently overwrites** it. With retries in the picture that is a data-loss
path, so:

- The phone mints a UUID per submission and sends it on every attempt.
- The server records accepted keys; a repeat returns the existing
  submission rather than creating a second one.
- Chunk uploads are idempotent by `(submission, slot, range)`.

## Footage handling

**Crop the overlay for outcome judging.** SwingVision burns a court minimap
and a shot label (*"Topspin Backhand / 43 MPH"*) into the top-right, and
draws the ball trajectory over the court. The label restates the xlsx's own
`Stroke` and `Speed (MPH)` columns, so a model reading it is re-reading an
input rather than watching the rally — and it would make the `net_approach`
check circular, since that is inferred purely from a `Volley`/`Overhead`
stroke. `frames_for_point` crops that block before sending frames.

Worth stating plainly: the footage read is independent of **our
reconstruction heuristic**, which is the comparison that decides whether a
flag clears. It is *not* independent of SwingVision's tracking, since the
ball trail is also their interpretation. Two sources agreeing still means
something; it is not two unrelated observers.

**Transcode after upload** to 720p30 at ~2 Mbps — roughly 1.35 GB per match
instead of 11 GB. The verifier downscales to 1280×720 anyway, so nothing is
lost. The threshold where compression starts smearing a distant ball needs
testing on real footage before going lower.

## The new Overview

Intake moves to the phone, so the dashboard becomes the progress view:

- **Streak** — consecutive weeks containing at least one match
- **This week** — played, or days remaining
- **Form** — last five results
- **Personal bests** — first serve %, points won, aces
- **Trends** — the existing charts
- **Awaiting files** — submissions the phone has queued

One subtlety: a submitted match is a pending JSON, not a database row, so
the streak must count **submissions and finalized matches together**, keyed
on date + opponent so a match that is both does not count twice.

## Open questions

1. **Does a score appear in the overlay between points?** Unknown — the
   condensed export has no between-point footage to check. If it does, it
   is a candidate source for the known scoring discrepancy (the simulation
   produces 3 set-completions where the real match had 2). Check on the
   first full recording.
2. ~~How low can the transcode go?~~ **Measured 2026-11-14** on real
   footage. A 20s segment re-encoded to 720p30 extrapolates to:

   | Encode | 90 min | vs original |
   |---|---|---|
   | original 1080p59 | 11.12 GB | - |
   | **720p30 @ 2.0 Mbps** | **1.25 GB** | 8.9x smaller |
   | 720p30 @ 1.2 Mbps | 0.75 GB | 14.8x smaller |

   The ball stays clearly legible at both; the degradation lands on the
   foliage, not on anything we judge from - tennis footage compresses
   kindly because the ball, players and court lines are all high-contrast
   and large while the trees eating the bitrate are irrelevant. **2 Mbps is
   the default** (visually near-indistinguishable). 1.2 Mbps looks viable
   but was checked on one frame in good light, with the ball against dark
   trees and SwingVision's trail drawn on it - re-check on a full match
   before dropping to it.
3. **Does the phone-saved video carry the same `Video Time` origin** as the
   xlsx? Must be confirmed before trusting `check_alignment` on it — this
   is the assumption the whole footage check rests on. Cheap first check:
   compare the saved video's duration against the largest `Video Time` in
   the sheet (2190s for part 1 of the 2026-08-18 match).
4. **When the typed score and the reconstruction disagree, which goes into
   the `set` table?** The typed score is the better source, but the `point`
   rows are reconstructed, so storing a different scoreline leaves the two
   internally inconsistent. Surfacing the disagreement as an import note is
   clearly right; silently overriding either is clearly wrong. Decide at
   implementation time, with a real discrepancy in hand.

## Follow-on: practice notes

Separate from matches, and deliberately not part of the record. Unrecorded
hitting sessions should be capturable as free text on the phone — a date,
who with, and whatever is worth remembering — without producing a match row,
touching stats, or counting toward the streak. The existing Journal is
per-match (`match.pros/cons/notes`) and cannot hold a session that has no
match, so this is new. Build after Phase 1's tests land.

## Build order

Each phase is useful on its own and testable before the next begins.

1. **Submissions API + pairing** — server-side only, driven by tests. Ends
   with: a match can be created and completed over HTTP with a token.
2. **The PWA** — form, offline queue, sync, attach, chunked upload.
3. **Transcode + overlay crop** — the footage path, measured.
4. **The progress dashboard** — streak, form, bests, awaiting-files.

### What gets tested

- **Unit**: streak computation across week boundaries and gaps; idempotency
  key handling; chunk assembly including out-of-order and repeated chunks;
  metadata validation.
- **API**: pairing and rejection of unpaired requests; the full lifecycle;
  duplicate submission; an upload interrupted mid-chunk and resumed.
- **Behaviour** (`scripts/e2e_check.py`): the PWA at phone width — fill,
  submit offline, show queued, reconcile when reachable.
- **Footage**: that the cropped frame excludes the label region, asserted on
  a real frame rather than by construction.
