"""Where each shot landed, on a real court, for the Statistics tab's heatmap.

Both producers of shot rows carry a bounce position in metres in the same
court frame: the SwingVision `.xlsx` parser (its "Bounce (x)/(y)" columns)
and the independent tracker (a homography). The origin is the centre mark
of the near baseline (the camera's end), +x is the camera's right and +y
runs away from the camera: near baseline y=0, net y=11.885, far baseline
y=23.77.

Those coordinates live only in the staged JSON and its sources, never in
`data/schema.sql` (CLAUDE.md forbids schema creep for staging-only data),
so a finalized match is traced back to its staged file by date and
opponent - the same lookup `ai/pipeline.py` and `shot_map.embeddings` use -
and its shots re-read through `swingvision_import.sources`.

**Every bounce is drawn from the hitter's end.** Players change ends, so
raw positions would mix shots aimed at both halves. A shot hit from the
far end is turned through 180 degrees (x -> -x, y -> 23.77 - y): the
hitter is always at the bottom and the target half at the top, and the
hitter's own left and right are kept.

Nothing here is a guess dressed as a measurement: a match whose shots have
no bounce positions says so (`available=False`, with the reason), and a
tracked match reports each bounce's provenance (measured, interpolated,
inferred) so the page can state how many were actually seen.
"""

from __future__ import annotations

import math
import sqlite3
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from swingvision_import import sources
from swingvision_import.config import ImportConfig
from swingvision_import.parse import SwingVisionParser
from swingvision_import.raw import RawShotRow
from swingvision_import.records import MatchRecord
from swingvision_import.review import find_pending_path, load_pending

COURT_LENGTH = 23.77
NET_Y = COURT_LENGTH / 2

_SERVE_TYPES = ("first_serve", "second_serve")
# Not match play: a ball fed across between points, or a hit the producer
# itself classed as nothing (see reconstruct.py's excluded points).
_NOT_PLAY_TYPES = ("none",)
_NOT_PLAY_STROKES = ("Feed",)


@dataclass(frozen=True)
class Bounce:
    """One shot's landing spot, seen from the hitter's end.

    Attributes:
        x: Metres across the court, hitter's right positive.
        y: Metres from the hitter's baseline; the net is at 11.885.
        by: "me" when the tracked player hit it, else "opponent".
        kind: "serve" or "rally".
        result: "In", "Out" or "Net", as the producer recorded it.
        stroke: The stroke, as recorded ("Unclear" from the tracker when
            it could not name one).
        point_number: The shot's point number in its source.
        shot_number: Its position within the point.
        rotated: Whether it was hit from the far end and turned to face
            the same way as the rest.
        provenance: For a tracked match, how the bounce was obtained
            ("measured", "interpolated", "inferred"); None for an export,
            which does not say.
    """

    x: float
    y: float
    by: str
    kind: str
    result: str
    stroke: str
    point_number: int
    shot_number: int
    rotated: bool
    provenance: str | None = None


@dataclass(frozen=True)
class Placement:
    """A match's bounces, with the coverage they represent.

    Attributes:
        match_id: The finalized match.
        available: Whether there is anything to draw.
        reason: When not available, one sentence saying why.
        source: "export" or "tracker"; None when the staged file is gone.
        shots: Match-play shots found (the denominator).
        with_bounce: How many of them carry a bounce position.
        provenance: For a tracked match, bounces per provenance tag;
            empty for an export.
        bounces: The bounces themselves.
    """

    match_id: int
    available: bool
    reason: str | None = None
    source: str | None = None
    shots: int = 0
    with_bounce: int = 0
    provenance: dict[str, int] = field(default_factory=dict)
    bounces: list[Bounce] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialises for the JSON API.

        Returns:
            Every field, bounces as plain dicts.
        """
        return asdict(self)


def _is_play(shot: RawShotRow) -> bool:
    """Whether a shot is match play rather than a feed or a non-shot."""
    return shot.shot_type not in _NOT_PLAY_TYPES and shot.stroke not in _NOT_PLAY_STROKES


def _hit_from_far(shot: RawShotRow) -> bool:
    """Whether the hitter stood at the far end.

    The producer's own `hit_side` first; then the hit position; and only
    for a row with neither, the bounce's half - assuming the ball crossed,
    which a ball into the net did not, so that last rule is a fallback.
    """
    if shot.hit_side in ("near", "far"):
        return shot.hit_side == "far"
    if shot.hit_y is not None:
        return shot.hit_y > NET_Y
    return shot.bounce_y < NET_Y


def _tracker_bounce_provenance(record: MatchRecord) -> dict[tuple[int, int], str]:
    """Bounce provenance per (point, shot) from a tracked match's traces."""
    found: dict[tuple[int, int], str] = {}
    pending = list((record.tracking or {}).get("attempts") or [])
    while pending:
        attempt = pending.pop()
        if attempt.get("folded_fault"):
            pending.append(attempt["folded_fault"])
        for trace in attempt.get("shots") or []:
            bounce = (trace.get("fields") or {}).get("bounce_xy")
            if bounce:
                key = (int(trace["point_number"]), int(trace["shot_number"]))
                found[key] = bounce["provenance"]
    return found


def placement_from_record(
    match_id: int, record: MatchRecord, parser: SwingVisionParser
) -> Placement:
    """Reads a staged record's bounces, turned to face the hitter's way.

    Args:
        match_id: The finalized match the record belongs to.
        record: The staged match record.
        parser: Re-reads the record's export(s), when it has any.

    Returns:
        The placement; `available` is False, with a reason, when no shot
        carries a bounce position.
    """
    source = "tracker" if record.tracking else "export"
    try:
        exports = sources.parse_sources(record, parser)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return Placement(
            match_id,
            False,
            "This match's original export could not be re-read, so its bounce positions "
            "are not available.",
            source,
        )
    if not exports:
        return Placement(
            match_id,
            False,
            "This match kept no per-shot data, so there are no bounce positions to draw.",
            source,
        )
    host = next((e.settings.host_name for e in exports if e.settings), "")
    traced = _tracker_bounce_provenance(record) if source == "tracker" else {}
    shots = [shot for export in exports for shot in export.shots if _is_play(shot)]
    bounces: list[Bounce] = []
    for shot in shots:
        if shot.bounce_x is None or shot.bounce_y is None:
            continue
        x, y = float(shot.bounce_x), float(shot.bounce_y)
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        far = _hit_from_far(shot)
        if far:
            x, y = -x, COURT_LENGTH - y
        bounces.append(
            Bounce(
                x=round(x, 3),
                y=round(y, 3),
                by="me" if host and shot.player == host else "opponent",
                kind="serve" if shot.shot_type in _SERVE_TYPES else "rally",
                result=shot.result,
                stroke=shot.stroke,
                point_number=shot.point_number,
                shot_number=shot.shot_number,
                rotated=far,
                provenance=traced.get((shot.point_number, shot.shot_number)),
            )
        )
    counts = (
        dict(Counter(b.provenance or "unknown" for b in bounces)) if source == "tracker" else {}
    )
    reason = None
    if not bounces:
        reason = (
            f"None of this match's {len(shots)} shots has a bounce position"
            + (
                " - the tracker saw no bounces it could place on the court."
                if source == "tracker"
                else " - the export carried no Bounce (x)/(y) values."
            )
        )
    return Placement(
        match_id=match_id,
        available=bool(bounces),
        reason=reason,
        source=source,
        shots=len(shots),
        with_bounce=len(bounces),
        provenance=counts,
        bounces=bounces,
    )


def match_placement(
    db_path: Path,
    pending_dir: Path,
    match_id: int,
    *,
    import_config: ImportConfig | None = None,
) -> Placement | None:
    """Finds a finalized match's staged file and reads its bounces.

    Args:
        db_path: The SQLite database, for the match's date and opponent.
        pending_dir: Where staged JSON files live.
        match_id: The finalized match.
        import_config: Settings for re-parsing exports; ImportConfig()
            when omitted.

    Returns:
        The placement, or None when no such match exists. A match whose
        staged file is gone (or unreadable) comes back unavailable, with
        that as the reason - not an error.
    """
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT date, opponent FROM match WHERE match_id = ?", (match_id,)
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    path = find_pending_path(row[0], row[1], pending_dir)
    try:
        record = load_pending(path) if path is not None else None
    except (OSError, ValueError):
        record = None
    if record is None:
        return Placement(
            match_id,
            False,
            "This match's staging file is no longer on disk, and bounce positions are kept "
            "only there - never in the database.",
        )
    parser = SwingVisionParser(import_config or ImportConfig())
    return placement_from_record(match_id, record, parser)
