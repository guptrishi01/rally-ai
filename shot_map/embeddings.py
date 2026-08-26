"""Projects every tracked-player shot across one or more matches into a 3D
point cloud, for the dashboard's Statistics tab "Shot Map".

This is a hand-rolled PCA (numpy SVD) over a small hand-built feature
vector per shot - deliberately not a learned/trained embedding. A single
player's match history is hundreds of shots, nowhere near enough to train
a representation that would beat a fixed linear projection of interpretable
features - the same reasoning CLAUDE.md's "Sparse matrix handling... not
adopted" note already applies elsewhere in this codebase. No explanation
or attribution layer here on purpose: this only produces coordinates plus
raw labels (shot_type/stroke/result/point_won) for a scatter plot - the
viewer draws their own conclusions from the shape of the cloud.

Shot provenance (which raw export a match came from, and which shot each
finalized point traces back to) lives only in the pending-review JSON
(swingvision_import.review), not in data/schema.sql - so a match whose
pending file has been deleted, or that came from a direct Points-sheet
parse (no per-shot provenance at all), silently contributes nothing here
rather than raising. That mirrors ai/pipeline.py's shot_pattern_summary
lookup: additive-only, never required.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from swingvision_import import reconstruct
from swingvision_import.config import ImportConfig
from swingvision_import.parse import SwingVisionParser
from swingvision_import.raw import RawShotRow
from swingvision_import.records import MatchRecord
from swingvision_import.review import find_pending_path, load_pending

# Real confirmed value vocabularies - see raw.py's RawShotRow docstring.
_SHOT_TYPES = (
    "first_serve",
    "second_serve",
    "first_return",
    "second_return",
    "serve_plus_one",
    "return_plus_one",
    "in_play",
    "none",
)
_STROKES = (
    "Serve",
    "Forehand",
    "Backhand",
    "Forehand Volley",
    "Backhand Volley",
    "Overhead",
    "Feed",
)
_RESULTS = ("In", "Out", "Net")
_MAX_RALLY_POSITION = 10
_FEATURE_DIM = len(_SHOT_TYPES) + len(_STROKES) + len(_RESULTS) + 2


@dataclass(frozen=True)
class ShotPoint:
    """One tracked-player shot, projected into 3D for the Shot Map.

    Attributes:
        match_id: Which match this shot belongs to.
        point_won: Whether the point this shot belonged to was won by the
            tracked player - the only outcome signal surfaced, no further
            explanation of why.
        shot_type: SwingVision's shot-role label, for the tooltip/legend.
        stroke: The stroke used, for the tooltip/legend.
        result: Whether the shot landed in/out/net, for the tooltip/legend.
        x: Projected coordinate on the first principal axis.
        y: Projected coordinate on the second principal axis.
        z: Projected coordinate on the third principal axis.
    """

    match_id: int
    point_won: bool
    shot_type: str
    stroke: str
    result: str
    x: float
    y: float
    z: float


def _feature_vector(shot: RawShotRow, *, is_serving: bool, rally_position: int) -> list[float]:
    """Builds one shot's fixed-length feature vector.

    Every categorical field is one-hot encoded against its known real
    vocabulary (see the module-level tuples above) rather than hashed or
    learned - keeps the projection's axes traceable back to interpretable
    inputs, appropriate for a "simple visualization," not a trained model.

    Args:
        shot: The raw shot row.
        is_serving: Whether the tracked player served the point this shot
            belongs to (constant across every shot in the point).
        rally_position: This shot's 0-based order among the tracked
            player's own shots within the point.

    Returns:
        A fixed-length numeric vector, in the order: shot_type one-hot,
        stroke one-hot, result one-hot, is_serving, normalized rally
        position.
    """
    vector = [1.0 if shot.shot_type == t else 0.0 for t in _SHOT_TYPES]
    vector += [1.0 if shot.stroke == s else 0.0 for s in _STROKES]
    vector += [1.0 if shot.result == r else 0.0 for r in _RESULTS]
    vector.append(1.0 if is_serving else 0.0)
    vector.append(min(rally_position, _MAX_RALLY_POSITION) / _MAX_RALLY_POSITION)
    return vector


def _pca_3d(features: np.ndarray) -> np.ndarray:
    """Projects a feature matrix onto its top 3 principal components.

    Args:
        features: (n_samples, n_features) array, not yet centered.

    Returns:
        (n_samples, 3) projected coordinates. Degenerate cases (no
        samples, or fewer usable dimensions than 3 - e.g. only 1-2 shots
        logged so far) are padded with zeros on the missing axes rather
        than raising, so a Shot Map with too little data yet still
        renders (a mostly-empty scene), instead of erroring the whole
        Statistics tab.
    """
    n_samples = features.shape[0]
    if n_samples == 0:
        return np.zeros((0, 3))

    centered = features - features.mean(axis=0, keepdims=True)
    n_components = min(3, n_samples, features.shape[1])
    if n_components == 0:
        return np.zeros((n_samples, 3))

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:n_components].T
    if n_components < 3:
        projected = np.hstack([projected, np.zeros((n_samples, 3 - n_components))])
    return projected


def _match_shots(
    record: MatchRecord, parser: SwingVisionParser
) -> tuple[str, dict[int, list[RawShotRow]]]:
    """Re-derives (host_name, shots grouped by point) for a staged match.

    Re-parses the match's original export(s) exactly the way
    SwingVisionImportPipeline.shots_by_point does - duplicated here
    (rather than reusing that pipeline method) only because this also
    needs the raw Settings host_name, which shots_by_point's return shape
    doesn't carry.

    Args:
        record: The staged match record (from a still-present pending
            JSON file).
        parser: A SwingVisionParser to re-read the original export(s) with.

    Returns:
        (host_name, shots_by_point). host_name is "" if there's no
        source file to re-parse, or no Settings sheet was found -
        callers treat that as "nothing usable here," not an error.
    """
    if record.source_files:
        raw_exports = [parser.parse(Path(p)) for p in record.source_files]
        merged = reconstruct.merge_shots([raw.shots for raw in raw_exports])
        host_name = raw_exports[0].settings.host_name if raw_exports[0].settings else ""
        return host_name, reconstruct.group_shots_by_point(merged)
    if record.source_file:
        raw = parser.parse(Path(record.source_file))
        host_name = raw.settings.host_name if raw.settings else ""
        return host_name, reconstruct.group_shots_by_point(raw.shots)
    return "", {}


def build_shot_embeddings(
    db_path: Path,
    pending_dir: Path,
    match_ids: list[int],
    *,
    import_config: ImportConfig | None = None,
) -> list[ShotPoint]:
    """Builds the 3D Shot Map across one or more finalized matches.

    Args:
        db_path: Path to the SQLite database (to look up each match's
            date/opponent, which is how its pending JSON is found).
        pending_dir: Directory pending-review JSON files are staged into.
        match_ids: The finalized matches to include.
        import_config: Settings for re-parsing exports. Defaults to
            ImportConfig() if not given.

    Returns:
        One ShotPoint per tracked-player shot whose point could be traced
        back to a win/loss outcome, across every given match. A match
        with no still-present pending JSON, no source file(s), no
        Settings sheet, or that came entirely from a direct Points-sheet
        parse (no per-shot provenance) contributes nothing - silently,
        not an error, since this is a best-effort visualization over
        however much shot data happens to still be traceable.
    """
    import_config = import_config or ImportConfig()
    parser = SwingVisionParser(import_config)

    feature_rows: list[list[float]] = []
    labels: list[dict[str, object]] = []

    connection = sqlite3.connect(db_path)
    try:
        for match_id in match_ids:
            row = connection.execute(
                "SELECT date, opponent FROM match WHERE match_id = ?", (match_id,)
            ).fetchone()
            if row is None:
                continue
            date, opponent = row
            pending_path = find_pending_path(date, opponent, pending_dir)
            if pending_path is None:
                continue

            record = load_pending(pending_path)
            host_name, shots_by_point = _match_shots(record, parser)
            if not host_name:
                continue

            point_meta = {
                point.source_point_number: (point.point_won, point.is_serving)
                for set_record in record.sets
                for point in set_record.points
                if point.source_point_number is not None
            }

            for point_number, shots in shots_by_point.items():
                meta = point_meta.get(point_number)
                if meta is None:
                    continue
                point_won, is_serving = meta
                tracked_shots = [s for s in shots if s.player == host_name]
                for rally_position, shot in enumerate(tracked_shots):
                    feature_rows.append(
                        _feature_vector(
                            shot, is_serving=is_serving, rally_position=rally_position
                        )
                    )
                    labels.append(
                        {
                            "match_id": match_id,
                            "point_won": point_won,
                            "shot_type": shot.shot_type,
                            "stroke": shot.stroke,
                            "result": shot.result,
                        }
                    )
    finally:
        connection.close()

    matrix = np.array(feature_rows, dtype=float) if feature_rows else np.zeros((0, _FEATURE_DIM))
    coords = _pca_3d(matrix)

    return [
        ShotPoint(
            match_id=label["match_id"],
            point_won=label["point_won"],
            shot_type=label["shot_type"],
            stroke=label["stroke"],
            result=label["result"],
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
            z=float(coords[i, 2]),
        )
        for i, label in enumerate(labels)
    ]
