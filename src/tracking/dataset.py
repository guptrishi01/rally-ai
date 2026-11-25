"""Exporting labelled frames for training the learned ball model (Track B).

The learned ball detector is trained elsewhere - on a cloud GPU, never on
this machine (spec section 12, decision 4) - from frames of the user's own
recordings. This module turns a tracked match into that training set's
**labels**, honestly marked by where each label came from:

* ``tracker-measured`` - the classic detector saw the ball here, at or
  above `min_confidence`. A *pseudo-label*: the detector that is being
  replaced made it, so it carries that detector's mistakes. Use it to
  bootstrap, and have a person check a sample before trusting it.
* ``tracker-bridged`` - the ball's position was bridged across a short
  gap. Kept for completeness; excluded from training by default.
* ``unlabelled`` - no position. **Not** a "no ball here" label: the ball
  may well be in the frame, unseen. Never trained on as a negative.
* ``human`` - a person marked (or confirmed) the position in a labelling
  tool and the corrected file was read back (`apply_corrections`). These
  outrank everything else.

Frames are grouped into **clips**, one per tracked point (its rally, with a
little padding), and every row says whether a person confirmed that point
in review - the correction loop's link: points a reviewer has looked at
are where pseudo-labels are worth checking first.

The format is one CSV (`LABEL_COLUMNS`) plus a manifest naming the source
video by SHA-256, the detector, the split and the counts. Frame images are
written by `scripts/export_ball_dataset.py`, which decodes only the clips.
See docs/ball-model-training.md for the whole loop.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .stitch import BRIDGED_STATE, MEASURED_STATE, CompactBallTrack

DATASET_FORMAT = 1
MEASURED = "tracker-measured"
BRIDGED = "tracker-bridged"
UNLABELLED = "unlabelled"
HUMAN = "human"
SOURCES = (HUMAN, MEASURED, BRIDGED, UNLABELLED)
LABEL_COLUMNS = (
    "clip",
    "frame",
    "pts",
    "file",
    "visibility",
    "x",
    "y",
    "label_source",
    "confidence",
    "point_number",
    "confirmed_point",
    "split",
)


@dataclass(frozen=True)
class LabelRow:
    """One frame's label.

    Attributes:
        clip: The clip (point) it belongs to, e.g. "p0012".
        frame: Its frame number in the whole recording.
        pts: Its timestamp, seconds on the video's clock.
        file: Its image's path relative to the dataset folder.
        visibility: 1 when a position is given, 0 when a person said the
            ball is not in the frame, -1 when nothing is known.
        x: The ball's centre, pixels from the left; None without one.
        y: Pixels from the top; None without one.
        label_source: One of `SOURCES`.
        confidence: The detector's confidence for a tracker label; None
            for a person's.
        point_number: The tracked point the clip is.
        confirmed_point: Whether a person confirmed that point in review.
        split: "train", "val" or "test" - one per match, never per frame.
    """

    clip: str
    frame: int
    pts: float
    file: str
    visibility: int
    x: float | None
    y: float | None
    label_source: str
    confidence: float | None
    point_number: int
    confirmed_point: bool
    split: str

    def to_row(self) -> list[str]:
        """The row as CSV cells.

        Returns:
            One string per `LABEL_COLUMNS` entry; empty for None.
        """

        def cell(value):
            """One value as CSV text."""
            if value is None:
                return ""
            if isinstance(value, bool):
                return "1" if value else "0"
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(value)

        return [cell(getattr(self, name)) for name in LABEL_COLUMNS]

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> LabelRow:
        """Reads a row back from CSV cells.

        Args:
            row: Column name to cell.

        Returns:
            The label.

        Raises:
            ValueError: If a cell is malformed or the source unknown.
        """

        def number(name):
            """A numeric cell, or None when empty."""
            text = (row.get(name) or "").strip()
            return float(text) if text else None

        source = (row.get("label_source") or "").strip()
        if source not in SOURCES:
            raise ValueError(f"Unknown label_source {source!r} (expected one of {SOURCES}).")
        visibility = int(row["visibility"])
        if visibility not in (-1, 0, 1):
            raise ValueError(f"visibility must be -1, 0 or 1, got {visibility}.")
        x, y = number("x"), number("y")
        if visibility == 1 and (x is None or y is None):
            raise ValueError(f"Frame {row['frame']} is marked visible but has no position.")
        return cls(
            clip=row["clip"],
            frame=int(row["frame"]),
            pts=float(row["pts"]),
            file=row["file"],
            visibility=visibility,
            x=x,
            y=y,
            label_source=source,
            confidence=number("confidence"),
            point_number=int(row["point_number"]),
            confirmed_point=row.get("confirmed_point", "0").strip() == "1",
            split=row["split"],
        )


@dataclass(frozen=True)
class Clip:
    """One tracked point's stretch of video.

    Attributes:
        name: "p" and the point number, zero-padded.
        start: First moment, seconds on the video's clock.
        end: Last moment.
        point_number: The point.
        confirmed: Whether a person confirmed it in review.
    """

    name: str
    start: float
    end: float
    point_number: int
    confirmed: bool


def clips_from_record(record, pad_before: float = 1.0, pad_after: float = 1.5) -> list[Clip]:
    """One clip per tracked point that was play (not warm-up or a fed ball).

    Args:
        record: A staged `MatchRecord` made by the tracker.
        pad_before: Seconds before the point's first hit.
        pad_after: Seconds after its last hit.

    Returns:
        The clips, in time order.

    Raises:
        ValueError: If the match was not staged by the tracker.
    """
    tracking = record.tracking
    if not tracking or "attempts" not in tracking:
        raise ValueError("This match was not staged by the tracker: no tracked points to export.")
    confirmed = {
        point.source_point_number
        for set_record in record.sets
        for point in set_record.points
        if point.confirmed_by == "human"
    }
    clips = []
    for attempt in tracking["attempts"]:
        if attempt.get("kind") == "excluded":
            continue
        number = int(attempt["number"])
        clips.append(
            Clip(
                name=f"p{number:04d}",
                start=float(attempt["start"]) - pad_before,
                end=float(attempt["end"]) + pad_after,
                point_number=number,
                confirmed=number in confirmed,
            )
        )
    return sorted(clips, key=lambda clip: clip.start)


def pseudo_labels(
    ball: CompactBallTrack,
    clips: Sequence[Clip],
    *,
    min_confidence: float = 0.4,
    split: str = "train",
) -> list[LabelRow]:
    """Labels every frame of every clip from the tracker's ball track.

    Args:
        ball: The whole run's ball state (from the run's checkpoints).
        clips: The clips.
        min_confidence: A measured position below this is `unlabelled`.
        split: The match's split.

    Returns:
        One row per tracked frame inside a clip, in time order.
    """
    rows = []
    for clip in clips:
        inside = np.flatnonzero((ball.frame_pts >= clip.start) & (ball.frame_pts <= clip.end))
        for k in inside:
            state = int(ball.state[k])
            confidence = float(ball.confidence[k])
            frame = int(ball.frame_numbers[k])
            x, y = (float(v) for v in ball.image_xy[k])
            if state == MEASURED_STATE and confidence >= min_confidence:
                source, visibility = MEASURED, 1
            elif state == BRIDGED_STATE:
                source, visibility = BRIDGED, 1
            else:
                source, visibility = UNLABELLED, -1
            known = visibility == 1
            rows.append(
                LabelRow(
                    clip=clip.name,
                    frame=frame,
                    pts=float(ball.frame_pts[k]),
                    file=f"frames/{clip.name}/{frame:07d}.jpg",
                    visibility=visibility,
                    x=x if known else None,
                    y=y if known else None,
                    label_source=source,
                    confidence=confidence if state != 0 else None,
                    point_number=clip.point_number,
                    confirmed_point=clip.confirmed,
                    split=split,
                )
            )
    return rows


def apply_corrections(rows: Sequence[LabelRow], corrections: Iterable[LabelRow]) -> list[LabelRow]:
    """Lets a person's labels replace the tracker's.

    Only rows whose `label_source` is "human" are taken from
    `corrections` - a corrected file can be the exported file edited in
    place, with untouched rows still saying where they came from.

    Args:
        rows: The exported labels.
        corrections: Labels read back from a person's corrected file.

    Returns:
        The labels, each frame a person marked replaced by their row;
        frames a person marked that were not exported are added.
    """
    human = {(c.clip, c.frame): c for c in corrections if c.label_source == HUMAN}
    out = [human.pop((row.clip, row.frame), row) for row in rows]
    out.extend(human.values())
    return sorted(out, key=lambda row: (row.pts, row.clip))


def write_labels(path: Path | str, rows: Iterable[LabelRow]) -> Path:
    """Writes labels as CSV.

    Args:
        path: The file.
        rows: The labels.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(LABEL_COLUMNS)
        for row in rows:
            writer.writerow(row.to_row())
    return path


def read_labels(path: Path | str) -> list[LabelRow]:
    """Reads labels from CSV.

    Args:
        path: The file.

    Returns:
        The labels.

    Raises:
        ValueError: If a column is missing or a row is malformed.
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in LABEL_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{Path(path).name} is missing column(s): {', '.join(missing)}.")
        return [LabelRow.from_row(row) for row in reader]


def manifest(
    rows: Sequence[LabelRow],
    *,
    video: Mapping,
    detector: str,
    split: str,
    min_confidence: float,
    calibration: Mapping | None = None,
    created: str,
) -> dict:
    """Describes an export, for the training run and for a later audit.

    Args:
        rows: The labels exported.
        video: The source video's identity (name, sha256, width, height).
        detector: The ball detector the pseudo-labels came from.
        split: The match's split.
        min_confidence: The pseudo-label threshold used.
        calibration: The run's calibration, when known.
        created: ISO date.

    Returns:
        JSON-ready description with counts by label source.
    """
    counts = Counter(row.label_source for row in rows)
    return {
        "format": DATASET_FORMAT,
        "created": created,
        "video": dict(video),
        "detector": detector,
        "split": split,
        "split_rule": "one split per match (never per frame): adjacent frames leak.",
        "min_confidence": min_confidence,
        "clips": len({row.clip for row in rows}),
        "frames": len(rows),
        "by_source": {source: counts.get(source, 0) for source in SOURCES},
        "confirmed_point_frames": sum(row.confirmed_point for row in rows),
        "trainable_frames": sum(row.label_source in (HUMAN, MEASURED) for row in rows),
        "calibration": None if calibration is None else dict(calibration),
        "columns": list(LABEL_COLUMNS),
    }


def write_manifest(path: Path | str, data: Mapping) -> Path:
    """Writes the manifest as JSON.

    Args:
        path: The file.
        data: What `manifest` returned.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2) + "\n", encoding="utf-8")
    return path
