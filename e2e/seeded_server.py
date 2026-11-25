#!/usr/bin/env python3
"""Runs the dashboard on synthetic, deterministic data for the behaviour checks.

The behaviour suite used to drive whatever happened to be in the user's own
`data/` - which, on a fresh checkout, is nothing: the Statistics tab then
shows only its empty state, and a chart that only renders with data can
never be checked. This builds a small, fully synthetic dataset in
`e2e/output/seed/` (gitignored, rebuilt on every start) and serves it:

- two finalized matches with bounce positions - one imported from an
  export-shaped staged file, one from the tracker with provenance - so the
  placement heatmap has both kinds to draw;
- one tracked match staged and still flagged, so the review modal shows
  tracker provenance and "Unclear" strokes;
- one tracking job stopped part-way, so the progress panel and its Resume
  button render.

No real tracking can start from here: the launcher is a no-op stand-in,
so even pressing Resume launches nothing. The user's own data is never
read or written.

Usage:
    python e2e/seeded_server.py [--port 5055]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from swingvision_import.config import ImportConfig  # noqa: E402
from swingvision_import.db import get_connection  # noqa: E402
from swingvision_import.load import finalize_and_load  # noqa: E402
from swingvision_import.raw import RawShotRow  # noqa: E402
from swingvision_import.records import MatchRecord, PointRecord, SetRecord  # noqa: E402
from swingvision_import.review import save_pending  # noqa: E402
from webapp.app import create_app  # noqa: E402
from webapp.config import WebAppConfig  # noqa: E402

SEED_DIR = _REPO_ROOT / "e2e" / "output" / "seed"
HOST = "Test Player"
GUEST = "Test Opponent"
PROVENANCES = ("measured", "measured", "measured", "interpolated", "inferred")


class _NoProcess:
    """A process that has already exited, cleanly."""

    pid = 0

    def poll(self) -> int:
        """Always finished."""
        return 0


class NoLaunch:
    """A launcher that starts nothing - this server must never track."""

    def launch(self, argv, log_path):
        """Starts nothing."""
        return _NoProcess()

    def is_alive(self, pid):
        """Nothing it 'started' is ever alive."""
        return False


def _shots(rng: random.Random, points: int) -> list[RawShotRow]:
    """Plausible rallies: serves into the boxes, rallies deep, some out."""
    shots: list[RawShotRow] = []
    for point in range(1, points + 1):
        server = HOST if (point // 4) % 2 == 0 else GUEST
        server_far = (point // 8) % 2 == 1
        length = rng.randint(1, 6)
        for number in range(1, length + 1):
            player = server if number % 2 == 1 else (GUEST if server == HOST else HOST)
            far = server_far if number % 2 == 1 else not server_far
            last = number == length
            serve = number == 1
            if serve:
                x = rng.uniform(0.2, 4.0) * (1 if point % 2 else -1)
                y = rng.uniform(13.5, 18.6)
            else:
                x = rng.uniform(-4.6, 4.6)
                y = rng.uniform(15.0, 25.2) if last else rng.uniform(16.0, 23.0)
            result = "In"
            if last and length > 1:
                result = "In" if abs(x) < 4.115 and y < 23.77 else "Out"
                if rng.random() < 0.2:
                    result, y = "Net", rng.uniform(9.0, 11.5)
            if far:  # recorded in the camera's frame; drawn from the hitter's end
                x, y = -x, 23.77 - y
            seen = rng.random() > 0.15  # some bounces are never seen at all
            shots.append(
                RawShotRow(
                    point_number=point,
                    shot_number=number,
                    player=player,
                    shot_type=("first_serve" if serve else "in_play"),
                    stroke=("Serve" if serve else rng.choice(["Forehand", "Backhand", "Unclear"])),
                    result=result,
                    video_time=point * 30.0 + number,
                    hit_side="far" if far else "near",
                    bounce_x=round(x, 3) if seen else None,
                    bounce_y=round(y, 3) if seen else None,
                )
            )
    return shots


def _trace(shot: RawShotRow, rng: random.Random) -> dict:
    """A provenance trace shaped like tracking.assemble.ShotTrace.to_dict()."""
    guesses = ["result"] if shot.result != "In" else []
    fields = {
        "player": {"value": shot.player, "provenance": "measured", "confidence": 0.9,
                   "source": "players"},
        "hit_side": {"value": shot.hit_side, "provenance": "measured", "confidence": 0.9,
                     "source": None},
        "shot_type": {"value": shot.shot_type, "provenance": "inferred", "confidence": 0.6,
                      "source": "segment-v1"},
        "stroke": {"value": shot.stroke,
                   "provenance": "inferred" if shot.stroke == "Unclear" else "measured",
                   "confidence": None if shot.stroke == "Unclear" else 0.7, "source": "pose"},
        "result": {"value": shot.result, "provenance": "inferred" if guesses else "measured",
                   "confidence": 0.3 if guesses else 0.8, "source": "assemble-v1"},
    }
    if shot.bounce_x is not None:
        fields["bounce_xy"] = {"value": [shot.bounce_x, shot.bounce_y],
                               "provenance": rng.choice(PROVENANCES), "confidence": 0.6,
                               "source": "ball"}
    if shot.stroke == "Unclear":
        guesses.append("stroke")
    return {
        "point_number": shot.point_number, "shot_number": shot.shot_number, "event_index": 0,
        "pts": shot.video_time, "time_uncertainty": 0.04, "confidence": 0.7,
        "tracked": shot.stroke != "Unclear", "fields": fields, "guesses": guesses,
        "notes": ["stroke too low-confidence to name: shown as 'Unclear'"]
        if shot.stroke == "Unclear" else [],
    }


def _tracked_record(date: str, opponent: str, flagged: bool, seed: int) -> MatchRecord:
    """A tracker-staged match with a provenance trace on every shot."""
    rng = random.Random(seed)
    shots = _shots(rng, 24)
    attempts = {}
    for shot in shots:
        attempt = attempts.setdefault(shot.point_number, {
            "number": shot.point_number, "kind": "served", "start": 0.0, "end": 1.0,
            "server_side": shot.hit_side, "server": shot.player, "serve_zone": "deuce",
            "zone_source": "feet", "second_serve": False, "fault": False,
            "confidence": round(rng.uniform(0.4, 0.9), 2),
            "reasons": ["serve seen", "rally of plausible length"], "shots": [],
        })
        attempt["shots"].append(_trace(shot, rng))
    points = []
    for number in sorted(attempts):
        won = rng.random() < 0.55
        points.append(PointRecord(
            game_number=(number - 1) // 4 + 1, point_number=(number - 1) % 4 + 1,
            is_serving=attempts[number]["server"] == HOST, point_won=won,
            point_end_type="winner" if won else "unforced_error",
            needs_review=flagged, source_point_number=number,
            confirmed_by=None if flagged else "human",
            tracking=attempts[number],
        ))
    tracked = sum(1 for a in attempts.values() for t in a["shots"] if t["tracked"])
    return MatchRecord(
        date=date, opponent=opponent, result="W", source_file=f"{date}_match.mp4",
        sets=[SetRecord(set_number=1, games_won=4, games_lost=2, points=points)],
        import_notes=[
            "Staged by the independent tracker from match.mp4, not from a SwingVision export: "
            "every shot is a guess from pixels, and every point needs review.",
        ],
        tracking={
            "source": "assemble-v1",
            "summary": {"shots": len(shots), "tracked_shots": tracked,
                        "guesses": {"result": 9, "stroke": 7, "serve_row": 0}},
            "notes": [f"Tracked {tracked} of {len(shots)} shots with confidence (a hit seen or "
                      "bridged, not inferred, at confidence 0.40 or more)."],
            "attempts": list(attempts.values()),
            "settings": {"host_name": HOST, "guest_name": GUEST},
            "shots": [asdict(shot) for shot in shots],
            "models": {"ball": "classic-ball-v1", "person": "motion"},
            "video": {"name": "match.mp4"},
            "calibration": {"quality": {
                "points": 11, "inliers": 11, "residual_px_median": 0.6, "residual_px_max": 2.1,
                "residual_m_median": 0.01, "residual_m_max": 0.15, "near_depth_m_per_px": 0.014,
                "far_depth_m_per_px": 0.28, "far_lateral_m_per_px": 0.023, "lines_found": None,
                "far_baseline_found": False, "support": None, "setup_score": 66.5,
                "warnings": [],
            }},
            "retraining": {
                "recommended": True, "weak_tracking": True, "conditions": [],
                "note": "Ball tracking was weak on this match (a ball position in 45% of rally "
                        "frames), although the lighting and court colour are within the range "
                        "the ball detector was tuned on. This is informational: nothing is "
                        "blocked, and every point is reviewed as usual.",
            },
        },
    )


def _export_record(seed: int, folder: Path) -> MatchRecord:
    """A match imported from a SwingVision-shaped .xlsx with bounce columns.

    Args:
        seed: Seeds the synthetic shots.
        folder: Where the .xlsx is written.

    Returns:
        The staged record, pointing at that export like a real import.
    """
    from openpyxl import Workbook

    record = _tracked_record("2026-08-14", "Alex", flagged=False, seed=seed)
    shots = [RawShotRow(**row) for row in record.tracking["shots"]]
    workbook = Workbook()
    sets = workbook.active
    sets.title = "Sets"
    sets.append(["Set", "Host Score", "Guest Score", "Set Winner"])
    sets.append([1, 4, 2, "host"])
    workbook.create_sheet("Games").append(["Game", "Set", "Server", "Game Winner"])
    workbook.create_sheet("Points").append(
        ["Point", "Game", "Set", "Match Server", "Point Winner", "Detail"]
    )
    settings = workbook.create_sheet("Settings")
    settings.append(["Host Team", "Guest Team"])
    settings.append([HOST, GUEST])
    sheet = workbook.create_sheet("Shots")
    sheet.append(["Point", "Shot", "Player", "Type", "Stroke", "Result", "Video Time",
                  "Hit Side", "Bounce (x)", "Bounce (y)"])
    for shot in shots:
        stroke = "Forehand" if shot.stroke == "Unclear" else shot.stroke
        sheet.append([shot.point_number, shot.shot_number, shot.player, shot.shot_type, stroke,
                      shot.result, shot.video_time, shot.hit_side, shot.bounce_x, shot.bounce_y])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "2026-08-14_Alex_export.xlsx"
    workbook.save(path)
    record.tracking = None
    for set_record in record.sets:
        for point in set_record.points:
            point.tracking = None
    record.source_file = str(path)
    record.import_notes = []
    return record


def build(seed_dir: Path) -> tuple[ImportConfig, WebAppConfig]:
    """Writes the synthetic dataset afresh.

    Args:
        seed_dir: Where to write it; emptied first.

    Returns:
        The configs that point the app at it.
    """
    shutil.rmtree(seed_dir, ignore_errors=True)
    imports = ImportConfig(pending_dir=seed_dir / "pending", db_path=seed_dir / "rallyai.db")
    web = WebAppConfig(
        uploads_dir=seed_dir / "uploads",
        media_dir=seed_dir / "media",
        submissions_dir=seed_dir / "submissions",
    )
    connection = get_connection(imports.db_path, imports.schema_path)
    try:
        finalized = (
            _export_record(3, web.uploads_dir),
            _tracked_record("2026-08-28", "Jordan", False, 5),
        )
        for record in finalized:
            save_pending(record, imports.pending_dir)
            finalize_and_load(connection, record)
    finally:
        connection.close()
    save_pending(_tracked_record("2026-09-02", "Sam", True, 11), imports.pending_dir)

    work = imports.pending_dir / "tracking_work"
    work.mkdir(parents=True, exist_ok=True)
    job = {"json_filename": "2026-09-05_Casey.json", "argv": ["match.mp4"],
           "date": "2026-09-05", "opponent": "Casey", "video": "casey_match.mp4",
           "launches": 1, "pid": 0, "started_at": 1_790_000_000}
    (work / "2026-09-05_Casey.job.json").write_text(json.dumps(job), "utf-8")
    progress = {"stage": "tracking", "chunks_total": 20, "chunks_done": 7, "chunks_resumed": 2,
                "video_seconds": 5400.0, "video_seconds_done": 1890.0, "elapsed_seconds": 9000.0,
                "eta_seconds": 16740.0, "seconds_per_video_minute": 318.0,
                "message": "Chunk 7/20 tracked."}
    (work / "2026-09-05_Casey.progress.json").write_text(json.dumps(progress), "utf-8")
    return imports, web


def main(argv: list[str] | None = None) -> int:
    """Builds the dataset and serves it until stopped.

    Args:
        argv: Command-line arguments.

    Returns:
        Exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=5055)
    args = parser.parse_args(argv)
    imports, web = build(SEED_DIR)
    create_app(imports, web, tracking_launcher=NoLaunch()).run(
        host="127.0.0.1", port=args.port, debug=False
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
