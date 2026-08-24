# Court IQ — AI Tennis Coach 🎾

An AI-powered tennis coaching tool that analyzes your match data to help you improve. Log point-by-point statistics from recorded match footage, reflect on what went right and wrong, and let generative AI turn that data into actionable coaching — strategies to implement, weaknesses to address, and exercises to improve your game.

## Overview

The goal of Court IQ is simple: **use AI to become a better tennis player.**

After each match, you review your recorded footage, log detailed point-by-point data, and write down your pros (what went well) and cons (what needs work). Court IQ feeds all of this — your raw statistics and your own self-assessment — into a generative AI coach that provides:

- **Strategic recommendations** — What tactical adjustments to make based on patterns in your data (e.g. "your first serve win % drops in third sets — conserve energy on second serve points early")
- **Improvement plans** — Specific areas to focus on in practice, prioritized by impact on your results
- **Drills and exercises** — Targeted practice routines and physical conditioning tied directly to your identified weaknesses

The AI coaching gets smarter over time as you log more matches — it can spot trends across your history that you might not notice yourself.

### How It Works

1. **Record** your match using [SwingVision](https://swing.vision/) on a phone fence mount — get verbal consent from the other player first, since the camera captures both sides of the court. A Pro subscription isn't required: shot-level AI tracking is free-tier, and Court IQ reconstructs point-by-point data from it when SwingVision's own point/game/set rollup (Pro-only) isn't available
2. **Stage the match** through the web app's Input tab (`webapp/`, or the `scripts/import_match.py` CLI) — upload the `.xlsx` export and any match video, fill in date/opponent/result, energy/mental ratings, and pros/cons. Submitting doesn't navigate away or block — you can switch to the Results tab immediately. If the recording was interrupted and split into multiple exports, they're merged into one continuous match rather than staged as separate, incorrectly-scored pieces. The match is staged as JSON, never written straight to the database
3. **Review** the staged match: confirm or correct every auto-scored point outcome (winner/unforced-error/forced-error calls, and — for reconstructed matches — every point, since it's a heuristic Court IQ derived from raw shot data, not SwingVision's own classification) and tag anything SwingVision doesn't capture at all, like net approaches. Optional Claude-assisted suggestions can help (never auto-resolving), structured-data quality checks flag things worth a second look (a reconstructed score that doesn't match SwingVision's own summary, a serve-order mismatch, recording gaps), and you can write your own plain-language notes per point for Claude to translate into the right fields — still a separate, explicit step before anything is actually applied
4. **Finalize** — the match is only written to SQL once every point is confirmed (unresolved points block the whole match, not just themselves); it then appears in the web app's Results tab for the first time (not before — a match can never be auto-finalized, so submitting alone never puts it there)
5. **Get coached** — Court IQ generates a self-contained HTML report combining your derived stats with AI-generated strategy, drills, and fitness recommendations — viewable by just opening the file, or directly from the Results tab's match list

## Features

- **AI Coach** — Three parallel Claude Sonnet 5 calls (strategy, drills, fitness) analyze your derived statistics and self-reported pros/cons to deliver recommendations grounded in that match's actual numbers, not generic advice — optionally enriched with rally-length patterns (e.g. win rate on short vs. long rallies) when SwingVision's shot-level data is available
- **SwingVision Import** — Matches are recorded via SwingVision (Pro not required — free-tier shot data is reconstructed into points when the Pro-only rollup isn't there, and a recording split across multiple export files gets merged into one continuous match) and loaded through a staged review pipeline that catches unreliable AI classifications before anything reaches the database, with structured-data quality checks (score cross-checks, serve-order validation) flagging things worth a second look
- **Review-Answer Parsing** — Write your own plain-language explanation for a flagged point and Claude translates it into the correct structured fields — still a separate, explicit confirmation step before it's actually applied, never silently auto-resolved
- **Web App** — `webapp/`, one page with two tabs: **Input** stages a match (upload the SwingVision export and video, fill in match setup, optionally kick off Claude-assisted review suggestions) without touching the CLI or hand-editing JSON, submitted without navigating away so you can switch tabs immediately; **Results** is a live dashboard of every finalized match — cross-match trend charts plus a list linking to each match's full individual report. A freshly-submitted match only appears in Results once it's actually been reviewed and finalized, same as everywhere else in the app — nothing here works around that gate
- **Derived Statistics** — All stats (first serve %, break point conversion, winner/UE ratio, hold %, etc.) are computed from raw point data via `src/stats/`, including game-score reconstruction for break/deuce points
- **Match Reports** — A self-contained HTML report per match (stat breakdown + AI coaching), plus a cross-match trend report (serve %, W/UE ratio, break points, win/loss record) — hand-rolled, interactive SVG charts, viewable standalone (no server needed) or live inside the web app's Results tab
- **Match History Trends** — The cross-match report visualizes patterns across your logged matches, not just the most recent one

## Tech Stack

- **Database:** SQLite (`data/schema.sql`) — three normalized (3NF) tables, no redundant storage of derived stats
- **Backend:** Python — SwingVision import pipeline (`src/swingvision_import/`, including shot-level point reconstruction and structured-data quality checks), derived-stat aggregation (`src/stats/`), and an AI coaching engine (`ai/`, using Claude Sonnet 5 via the `anthropic` SDK)
- **Web UI:** `webapp/` — a local Flask app, one page with an Input tab (staging a match) and a Results tab (a live dashboard of finalized matches, reusing `reports/`'s chart-building functions directly); binds to localhost only
- **Reports:** `reports/` — Jinja2-rendered HTML with hand-rolled inline SVG+CSS+JS charts (no charting library, no matplotlib), self-contained enough to stand alone as a downloadable file or be embedded live inside the web app
- **Scripts:** `scripts/` — CLI entry points (`import_match.py`, `resolve_reviews.py`, `generate_report.py`) and the one place a real Anthropic API client is constructed (everywhere else takes one injected, so tests never spend real money)
- **Testing:** pytest, 100% statement coverage across the four core Python packages (`swingvision_import`, `stats`, `ai`, `reports`); ruff (`E, F, I, W`)

## Database Schema

The database uses three normalized (3NF) tables:

- **`match`** — Match metadata (date, opponent, result, pros/cons, energy/mental ratings)
- **`set`** — Set-level data linked to a match (set number, score)
- **`point`** — Individual point data linked to a set (serve data, point outcome, net approaches). A `CHECK` constraint enforces `point_end_type` and `point_won` never disagree (e.g. an `ace` always means the point was won); a net approach's own success is always `net_approach AND point_won` at query time, never a separately stored column

All aggregate statistics are derived from the `point` table through queries rather than stored redundantly. Data-quality findings from the import pipeline (`import_notes`, shot-pattern summaries for the AI coach) live in the pre-SQL staging JSON only, never as database columns. See [`data/schema.sql`](data/schema.sql) for the full schema with example queries.

## Project Structure

```
court-iq/
├── README.md
├── CLAUDE.md
├── .gitignore
├── .env.example                 # Template for ANTHROPIC_API_KEY (.env itself is gitignored)
├── requirements.txt / requirements-dev.txt
├── pyproject.toml               # pytest, coverage, ruff config
├── logging_config.py            # configure_logging() - called once by each entry point
├── data/
│   └── schema.sql               # SQL table definitions and example queries
├── docs/
│   └── stat-definitions.md      # What each stat means and how it's calculated
├── src/
│   ├── swingvision_import/      # SwingVision export -> staged JSON -> SQL pipeline,
│   │                             # incl. shot-level reconstruction + quality checks
│   └── stats/                   # Derived-stat aggregation, reads data/schema.sql
├── ai/                          # AI coaching engine — context building, 3 parallel
│                                 # Claude Sonnet 5 calls (strategy/drills/fitness)
├── reports/                     # Static HTML report generation + hand-rolled SVG charts
├── webapp/                      # Local Flask intake UI (upload + match setup form)
├── scripts/                     # CLI entry points; real Anthropic client construction
└── tests/                       # pytest suite, mirrors each package 1:1
    ├── swingvision_import/
    ├── stats/
    ├── ai/
    ├── reports/
    └── webapp/
```

## Roadmap

- [x] Define trackable statistics
- [x] Design database schema
- [x] Choose tech stack (SQLite + Python; static HTML reports as the initial "frontend" — see Tech Stack)
- [x] Build SwingVision import pipeline (staged JSON review gate before anything reaches SQL)
- [x] Build derived-stat aggregation
- [x] Integrate AI coaching engine (strategy, drills, fitness from stats + pros/cons)
- [x] Generate static HTML match reports (stats + AI coaching, hand-rolled charts)
- [x] AI trend analysis across match history (cross-match trend report)
- [x] Build a match intake UI (`webapp/` — upload + setup form; staging a match no longer requires the CLI or hand-editing JSON)
- [x] Build a results dashboard (`webapp/`'s Results tab — cross-match trends + per-match reports live in the app, for finalized matches)
- [ ] Build a review UI for resolving `needs_review` flags in-browser (still a hand-edit-the-JSON step, or the CLI)
- [ ] Link AI coaching feedback to specific rally footage (SwingVision's raw export has per-shot video timestamps, not yet parsed or threaded through)
- [ ] Deploy / distribute generated reports

## Version History

- **3.0.0** (2026-08-20) — `webapp/` becomes a real two-tab app instead of an intake-only form: **Input** (unchanged fields, now submitted via `fetch()` instead of a full-page-navigation form POST, so the page never blocks or navigates away — `/import`/`/suggest` return an HTML fragment injected into a status area) and **Results** (a live dashboard of every *finalized* match — cross-match trend charts embedded via `<iframe srcdoc>`, reusing `reports/render.py`'s existing chart-building functions as-is rather than a template refactor, plus a match list linking to `GET /report/<match_id>` for each one's full individual report). The finalized-only boundary is deliberate and unchanged from every other part of this app: a match can never be auto-finalized, so a freshly-submitted match's own results aren't available until it's separately reviewed — confirmed by hand that a newly-submitted match does not appear in Results. Viewing a report never spends API money (reads a cached AI coaching report if one exists, never constructs a client). Added `stats.queries.all_match_ids()`, promoted from an ad-hoc copy in `scripts/generate_report.py --history` to a shared, tested function. Major bump: this is the first release where `webapp/` is a genuinely different kind of surface, not an incremental addition to the intake form. 228 tests, 100% statement coverage across the four core packages.
- **2.3.0** (2026-08-20) — `scripts/generate_report.py`: the CLI entry point for the last previously-code-only pipeline step. `python scripts/generate_report.py <match_id>` renders a finalized match's self-contained `report.html`, generating (or loading a cached) AI coaching report along the way; `--no-ai` skips the API call for a stats-only render, `--history` renders the cross-match trend report instead. Verified end to end against a seeded match, including the missing-match error path.
- **2.2.0** (2026-08-20) — Human-review-answer parsing: `pipeline.resolve()`/`review_resolve.py` translate the reviewer's own plain-language notes (`PointRecord.review_answer`, e.g. "she was out of position, clean winner") into structured `point_end_type`/`point_won`/`net_approach` fields — never auto-applied; a separate, explicit `pipeline.apply_resolutions()` (CLI: `scripts/resolve_reviews.py --resolve` / `--apply`) is what actually clears `needs_review`. Found and fixed a third live-API-only bug in the process: responses sometimes arrive wrapped in a markdown code fence despite explicit prompt instructions not to — `ai.client.strip_markdown_fence()` fixes this everywhere JSON gets parsed from a response. Also found and fixed a schema-consistency gap: the live API once returned `point_end_type="ace"` paired with `point_won=false`, an invalid combination under `data/schema.sql`'s `CHECK` constraint — now validated and rejected at parse time instead of surfacing as a confusing error deep inside `finalize()`. 221 tests, 100% statement coverage across the four core packages.
- **2.1.0** (2026-08-20) — Multi-part match merging: `pipeline.ingest_multi_part()` (and the CLI's multi-file `xlsx_paths` + repeatable `--first-server SET:WHO`) merges an interrupted recording split across multiple SwingVision exports into one continuous reconstruction (`reconstruct.merge_shots`), instead of two independently-scored, incorrect partial matches. `suggest()` re-merges the same way via a new `MatchRecord.source_files` field. 197 tests, 100% statement coverage across the four core packages.
- **2.0.0** (2026-08-20) — Multi-agent enrichment + real-client wiring: a real `anthropic.Anthropic()` client and CLI entry point (`scripts/`, fixing two live-API-only bugs no synthetic test caught: `temperature` rejected by `claude-sonnet-5`, and `response.content[0]` unreliably being a `ThinkingBlock`); a non-match-shot filter fixing a real bug in shot-based reconstruction (fed balls between points were silently counted as real points); structured-data quality checks (`import_notes` — score cross-checks against SwingVision's own summary, serve-order and identity validation); a new local Flask intake web UI (`webapp/`) as the primary way to stage a match; optional shot-pattern (rally-length) enrichment for the AI coach; and a 3NF pass on `data/schema.sql` (dropped the fully-redundant `net_point_won` column, added a `CHECK` constraint tying `point_end_type` to `point_won`). Major bump: the schema change and the new intake surface are significant enough to warrant it, even though no real match had been finalized into SQL yet. 188 tests, 100% statement coverage across the four core packages.
- **1.0.1** (2026-08-13) — Hardening pass: uncaught AI-specialist API failures (auth/rate-limit/network) now degrade gracefully instead of crashing report generation, `reports/` gained a `ReportConfig` default output location, and `logging_config.py` gives every module's logger somewhere to actually go. 115 tests, 100% statement coverage.
- **1.0.0** (2026-08-12) — Initial architecture: SwingVision import pipeline (staged-JSON review gate), derived-stat aggregation, AI coaching engine (Claude Sonnet 5, 3 parallel specialists), and static HTML report generation with hand-rolled SVG charts. 105 tests, 100% statement coverage.

## License

GPL-3.0
