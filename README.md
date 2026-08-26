# RallyAI — AI Tennis Coach 🎾

An AI-powered tennis coaching tool that analyzes your match data to help you improve. Log point-by-point statistics from recorded match footage, reflect on what went right and wrong, and let generative AI turn that data into actionable coaching — strategies to implement, weaknesses to address, and exercises to improve your game.

## Overview

The goal of RallyAI is simple: **use AI to become a better tennis player.**

After each match, you review your recorded footage, log detailed point-by-point data, and write down your pros (what went well) and cons (what needs work). RallyAI feeds all of this — your raw statistics and your own self-assessment — into a generative AI coach that provides:

- **Strategic recommendations** — What tactical adjustments to make based on patterns in your data (e.g. "your first serve win % drops in third sets — conserve energy on second serve points early")
- **Improvement plans** — Specific areas to focus on in practice, prioritized by impact on your results
- **Drills and exercises** — Targeted practice routines and physical conditioning tied directly to your identified weaknesses

The AI coaching gets smarter over time as you log more matches — it can spot trends across your history that you might not notice yourself.

### How It Works

1. **Record** your match using [SwingVision](https://swing.vision/) on a phone fence mount — get verbal consent from the other player first, since the camera captures both sides of the court. A Pro subscription isn't required: shot-level AI tracking is free-tier, and RallyAI reconstructs point-by-point data from it when SwingVision's own point/game/set rollup (Pro-only) isn't available
2. **Stage the match** on the dashboard's **Overview** tab (or the `scripts/import_match.py` CLI) — upload the `.xlsx` export and any match video, fill in date/opponent/result, energy/mental ratings, and pros/cons. If the recording was interrupted and split into multiple exports, they're merged into one continuous match rather than staged as separate, incorrectly-scored pieces. The match is staged as JSON, never written straight to the database
3. **Review** every flagged point right there in the browser: pick the correct outcome (winner/unforced-error/forced-error/ace/etc.) and tag anything SwingVision doesn't capture at all, like net approaches — for reconstructed matches, that's every point, since it's a heuristic RallyAI derived from raw shot data, not SwingVision's own classification. Optional Claude-assisted suggestions can help (never auto-resolving); structured-data quality checks surface things worth a second look (a reconstructed score that doesn't match SwingVision's own summary, a serve-order mismatch, recording gaps)
4. **Load Data** — once every flagged point is confirmed, finalizing writes the match to SQL in one step (unresolved points still block the whole match, not just themselves) and the real match ID appears immediately — a match can never be auto-finalized, so staging alone never adds it to your career stats
5. **Journal** it — add a quick "what went right / what went wrong" entry on the **Journal** tab and get a Mouratoglou-style coaching reply grounded in that match's actual numbers, or generate the fuller self-contained HTML report (strategy/drills/fitness from three parallel Claude specialists) — both viewable any time, and the report works standalone with no server running

## Features

- **RallyAI Dashboard** — a React SPA (`frontend/`) with four tabs, served locally by Flask (`python webapp/app.py`):
  - **Overview** — career stat tiles and highlights at a glance, plus the Load Data flow: upload → an in-browser review modal for every flagged point (with an optional Claude-suggestion assist) → finalize, with a tennis-ball progress animation while it loads
  - **Journal** — one sticky note per match (pros/cons/notes) with a Mouratoglou-style AI coach reply grounded in that match's stats and what you wrote
  - **Statistics** — interactive Recharts trend lines and per-match breakdowns across serving, receiving, point outcomes, net play, and clutch stats
  - **Film Review** — a minimal gallery of the footage you've uploaded per match
- **AI Coach** — Three parallel Claude Sonnet 5 calls (strategy, drills, fitness) analyze your derived statistics and self-reported pros/cons to deliver recommendations grounded in that match's actual numbers, not generic advice — optionally enriched with rally-length patterns (e.g. win rate on short vs. long rallies) when SwingVision's shot-level data is available
- **Journal Coach** — A single Mouratoglou-style voice responds directly to your journal entry, grounded in the same derived stats — a second, distinct AI surface from the 3-specialist match report
- **SwingVision Import** — Matches are recorded via SwingVision (Pro not required — free-tier shot data is reconstructed into points when the Pro-only rollup isn't there, and a recording split across multiple export files gets merged into one continuous match) and loaded through a staged review pipeline that catches unreliable AI classifications before anything reaches the database, with structured-data quality checks (score cross-checks, serve-order validation) flagging things worth a second look
- **In-Browser Review** — Confirm every flagged point's real outcome directly in the Overview review modal, with an optional Claude-suggestion assist — no hand-editing JSON or CLI step required (though `scripts/resolve_reviews.py` still works for the same pipeline)
- **Review-Answer Parsing** — The CLI path also supports writing your own plain-language explanation for a flagged point, which Claude translates into the correct structured fields — still a separate, explicit confirmation step before it's applied, never silently auto-resolved
- **Derived Statistics** — All stats (first serve %, break point conversion, winner/UE ratio, hold %, etc.) are computed from raw point data via `src/stats/`, including game-score reconstruction for break/deuce points, fetched concurrently across matches so the dashboard stays fast as your match history grows
- **Match Reports** — A self-contained HTML report per match (stat breakdown + AI coaching), plus a cross-match trend report (serve %, W/UE ratio, break points, win/loss record) — hand-rolled, interactive SVG charts, viewable standalone with no server needed
- **Match History Trends** — Both the Statistics tab and the cross-match report visualize patterns across your logged matches, not just the most recent one

## Tech Stack

- **Database:** SQLite (`data/schema.sql`) — three normalized (3NF) tables, no redundant storage of derived stats
- **Backend:** Python — SwingVision import pipeline (`src/swingvision_import/`, including shot-level point reconstruction and structured-data quality checks), derived-stat aggregation (`src/stats/`), and two AI surfaces (`ai/`: the 3-specialist match coach and the single-voice journal coach, both via Claude Sonnet 5 and the `anthropic` SDK)
- **Web API:** `webapp/` — a local Flask app; `webapp/api.py` is a `/api/*` JSON Blueprint over every pipeline (import/review/finalize, stats, journal), `webapp/app.py` also serves the built frontend and the standalone `/report/<id>` page; binds to localhost only
- **Frontend:** `frontend/` — Vite + React + TypeScript + Tailwind CSS, Recharts for interactive charts, `motion` for animation (the tennis-ball progress bar, tab transitions); built once (`npm run build`) into static assets Flask serves
- **Reports:** `reports/` — Jinja2-rendered HTML with hand-rolled inline SVG+CSS+JS charts (no charting library, no matplotlib) for the standalone downloadable report — a deliberately separate, simpler rendering path from the dashboard's Recharts-based Statistics tab
- **Scripts:** `scripts/` — CLI entry points (`import_match.py`, `resolve_reviews.py`, `generate_report.py`) and the one place a real Anthropic API client is constructed (everywhere else takes one injected, so tests never spend real money)
- **Testing:** pytest, 100% statement coverage across the four core Python packages (`swingvision_import`, `stats`, `ai`, `reports`); ruff (`E, F, I, W`); the frontend is type-checked (`tsc`) and verified by hand (no automated UI test suite yet)

## Database Schema

The database uses three normalized (3NF) tables:

- **`match`** — Match metadata (date, opponent, result, pros/cons, energy/mental ratings)
- **`set`** — Set-level data linked to a match (set number, score)
- **`point`** — Individual point data linked to a set (serve data, point outcome, net approaches). A `CHECK` constraint enforces `point_end_type` and `point_won` never disagree (e.g. an `ace` always means the point was won); a net approach's own success is always `net_approach AND point_won` at query time, never a separately stored column

All aggregate statistics are derived from the `point` table through queries rather than stored redundantly. Data-quality findings from the import pipeline (`import_notes`, shot-pattern summaries for the AI coach) live in the pre-SQL staging JSON only, never as database columns. See [`data/schema.sql`](data/schema.sql) for the full schema with example queries.

## Project Structure

```
rallyai/
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
├── ai/                          # Two AI surfaces: 3 parallel Claude Sonnet 5 calls
│                                 # (strategy/drills/fitness) + a single journal coach voice
├── reports/                     # Standalone HTML report generation + hand-rolled SVG charts
├── webapp/                      # Flask: /api/* JSON Blueprint (api.py) + static/report serving (app.py)
├── frontend/                    # Vite + React + TS + Tailwind dashboard (Overview/Journal/
│                                 # Statistics/Film Review); npm run build -> frontend/dist/
├── scripts/                     # CLI entry points; real Anthropic client construction
└── tests/                       # pytest suite, mirrors each backend package 1:1
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
- [x] Build a match intake UI (staging a match no longer requires the CLI or hand-editing JSON)
- [x] Build a results dashboard (cross-match trends + per-match reports live in the app, for finalized matches)
- [x] Build a review UI for resolving `needs_review` flags in-browser (Overview's review modal — confirm every flagged point directly, no hand-edited JSON or CLI step required)
- [x] Rebuild the dashboard as a modern 4-tab app (Overview/Journal/Statistics/Film Review) with a dedicated journal AI coach and interactive charts
- [ ] Link AI coaching feedback to specific rally footage (SwingVision's raw export has per-shot video timestamps, not yet parsed or threaded through)
- [ ] Deploy / distribute generated reports

## Version History

- **4.0.0** (2026-08-26) — The dashboard becomes a real React SPA (`frontend/`: Vite + TypeScript + Tailwind + Recharts + `motion`), replacing the Jinja two-tab app with four: **Overview** (career stats/highlights + the Load Data flow), **Journal** (one sticky note per match + a new single-voice Mouratoglou-style AI coach, `ai/journal.py`), **Statistics** (interactive Recharts trends/breakdowns), and **Film Review** (a minimal per-match video gallery). `webapp/app.py` is trimmed to serving the built frontend plus the unchanged standalone `/report/<id>` page; all business logic moved to a new `webapp/api.py` JSON Blueprint, a thin layer over the existing pipelines — no parallel logic. Closes the gap CLAUDE.md had flagged as TBD: flagged points are now confirmed directly in a browser review modal (`swingvision_import.review.confirm_point`, backed by the same `VALID_END_TYPES`/`WINNING_END_TYPES`/`LOSING_END_TYPES` consistency check the Claude-parsing path already used, now promoted to one shared source of truth) — the safety gate itself is unchanged, `finalize()` still refuses any match with an unresolved flag. `stats.queries.all_match_stats` fetches every match's stats concurrently via `ThreadPoolExecutor` (a second sanctioned use alongside `ai/generate.py`'s), keeping Overview/Statistics load time close to linear as match count grows; `career_stats_from_matches` is the pure aggregation step over the result. Major bump: a different tech stack for the whole UI, a new AI surface, and a previously-documented gap closed, all in one release. Verified end to end by hand with a live Flask + built-frontend session (Playwright-driven): upload → confirm every flagged point → finalize → Journal → Statistics, no console errors or failed requests. 260 tests, 100% statement coverage across the four core packages.
- **3.0.0** (2026-08-20) — `webapp/` becomes a real two-tab app instead of an intake-only form: **Input** (unchanged fields, now submitted via `fetch()` instead of a full-page-navigation form POST, so the page never blocks or navigates away — `/import`/`/suggest` return an HTML fragment injected into a status area) and **Results** (a live dashboard of every *finalized* match — cross-match trend charts embedded via `<iframe srcdoc>`, reusing `reports/render.py`'s existing chart-building functions as-is rather than a template refactor, plus a match list linking to `GET /report/<match_id>` for each one's full individual report). The finalized-only boundary is deliberate and unchanged from every other part of this app: a match can never be auto-finalized, so a freshly-submitted match's own results aren't available until it's separately reviewed — confirmed by hand that a newly-submitted match does not appear in Results. Viewing a report never spends API money (reads a cached AI coaching report if one exists, never constructs a client). Added `stats.queries.all_match_ids()`, promoted from an ad-hoc copy in `scripts/generate_report.py --history` to a shared, tested function. Major bump: this is the first release where `webapp/` is a genuinely different kind of surface, not an incremental addition to the intake form. 228 tests, 100% statement coverage across the four core packages.
- **2.3.0** (2026-08-20) — `scripts/generate_report.py`: the CLI entry point for the last previously-code-only pipeline step. `python scripts/generate_report.py <match_id>` renders a finalized match's self-contained `report.html`, generating (or loading a cached) AI coaching report along the way; `--no-ai` skips the API call for a stats-only render, `--history` renders the cross-match trend report instead. Verified end to end against a seeded match, including the missing-match error path.
- **2.2.0** (2026-08-20) — Human-review-answer parsing: `pipeline.resolve()`/`review_resolve.py` translate the reviewer's own plain-language notes (`PointRecord.review_answer`, e.g. "she was out of position, clean winner") into structured `point_end_type`/`point_won`/`net_approach` fields — never auto-applied; a separate, explicit `pipeline.apply_resolutions()` (CLI: `scripts/resolve_reviews.py --resolve` / `--apply`) is what actually clears `needs_review`. Found and fixed a third live-API-only bug in the process: responses sometimes arrive wrapped in a markdown code fence despite explicit prompt instructions not to — `ai.client.strip_markdown_fence()` fixes this everywhere JSON gets parsed from a response. Also found and fixed a schema-consistency gap: the live API once returned `point_end_type="ace"` paired with `point_won=false`, an invalid combination under `data/schema.sql`'s `CHECK` constraint — now validated and rejected at parse time instead of surfacing as a confusing error deep inside `finalize()`. 221 tests, 100% statement coverage across the four core packages.
- **2.1.0** (2026-08-20) — Multi-part match merging: `pipeline.ingest_multi_part()` (and the CLI's multi-file `xlsx_paths` + repeatable `--first-server SET:WHO`) merges an interrupted recording split across multiple SwingVision exports into one continuous reconstruction (`reconstruct.merge_shots`), instead of two independently-scored, incorrect partial matches. `suggest()` re-merges the same way via a new `MatchRecord.source_files` field. 197 tests, 100% statement coverage across the four core packages.
- **2.0.0** (2026-08-20) — Multi-agent enrichment + real-client wiring: a real `anthropic.Anthropic()` client and CLI entry point (`scripts/`, fixing two live-API-only bugs no synthetic test caught: `temperature` rejected by `claude-sonnet-5`, and `response.content[0]` unreliably being a `ThinkingBlock`); a non-match-shot filter fixing a real bug in shot-based reconstruction (fed balls between points were silently counted as real points); structured-data quality checks (`import_notes` — score cross-checks against SwingVision's own summary, serve-order and identity validation); a new local Flask intake web UI (`webapp/`) as the primary way to stage a match; optional shot-pattern (rally-length) enrichment for the AI coach; and a 3NF pass on `data/schema.sql` (dropped the fully-redundant `net_point_won` column, added a `CHECK` constraint tying `point_end_type` to `point_won`). Major bump: the schema change and the new intake surface are significant enough to warrant it, even though no real match had been finalized into SQL yet. 188 tests, 100% statement coverage across the four core packages.
- **1.0.1** (2026-08-13) — Hardening pass: uncaught AI-specialist API failures (auth/rate-limit/network) now degrade gracefully instead of crashing report generation, `reports/` gained a `ReportConfig` default output location, and `logging_config.py` gives every module's logger somewhere to actually go. 115 tests, 100% statement coverage.
- **1.0.0** (2026-08-12) — Initial architecture: SwingVision import pipeline (staged-JSON review gate), derived-stat aggregation, AI coaching engine (Claude Sonnet 5, 3 parallel specialists), and static HTML report generation with hand-rolled SVG charts. 105 tests, 100% statement coverage.

## License

GPL-3.0
