"""Tests for exporting ball labels for training the learned ball model."""

from __future__ import annotations

import json

import numpy as np
import pytest

from swingvision_import.records import MatchRecord, PointRecord, SetRecord
from tracking import dataset, stitch

from .conftest import SIZE


def _ball():
    """Ten frames at 10fps: measured, weak, bridged and unknown."""
    states = [1, 1, 1, 2, 0, 1, 1, 0, 1, 1]
    confidence = [0.9, 0.2, 0.8, 0.5, 0.0, 0.7, 0.6, 0.0, 0.9, 0.9]
    xy = np.array([[10.0 + k, 20.0 + k] for k in range(10)])
    xy[np.array(states) == 0] = np.nan
    return stitch.CompactBallTrack(
        frame_numbers=np.arange(100, 110),
        frame_pts=np.arange(10) / 10.0 + 5.0,
        state=np.array(states, dtype=np.int8),
        confidence=np.array(confidence),
        image_xy=xy,
        segment=np.where(np.array(states) > 0, 0, -1),
        image_size=SIZE,
        source="motion-ball",
    )


def _record():
    points = [
        PointRecord(
            game_number=1,
            point_number=1,
            is_serving=True,
            point_won=True,
            point_end_type="winner",
            source_point_number=7,
            confirmed_by="human",
        ),
        PointRecord(
            game_number=1,
            point_number=2,
            is_serving=True,
            point_won=False,
            point_end_type="unforced_error",
            source_point_number=8,
        ),
    ]
    record = MatchRecord(
        date="2026-08-21",
        opponent="Opp",
        result="W",
        sets=[SetRecord(set_number=1, games_won=1, games_lost=0, points=points)],
    )
    record.tracking = {
        "attempts": [
            {"number": 8, "kind": "served", "start": 20.0, "end": 24.0},
            {"number": 7, "kind": "served", "start": 6.0, "end": 8.0},
            {"number": 9, "kind": "excluded", "start": 30.0, "end": 31.0},
        ]
    }
    return record


def test_clips_are_the_points_that_were_play_with_their_review_state():
    clips = dataset.clips_from_record(_record())
    assert [(c.name, c.start, c.end, c.confirmed) for c in clips] == [
        ("p0007", 5.0, 9.5, True),
        ("p0008", 19.0, 25.5, False),
    ]
    with pytest.raises(ValueError, match="not staged by the tracker"):
        dataset.clips_from_record(MatchRecord(date="2026-08-21", opponent="O", result="W"))


def test_pseudo_labels_say_where_each_label_came_from():
    clip = dataset.Clip("p0007", 5.0, 5.95, 7, True)
    rows = dataset.pseudo_labels(_ball(), [clip], min_confidence=0.4, split="val")
    sources = [row.label_source for row in rows]
    assert sources == [
        "tracker-measured",
        "unlabelled",
        "tracker-measured",
        "tracker-bridged",
        "unlabelled",
        "tracker-measured",
        "tracker-measured",
        "unlabelled",
        "tracker-measured",
        "tracker-measured",
    ]
    weak, unknown = rows[1], rows[4]
    assert weak.visibility == -1 and weak.x is None and weak.confidence == 0.2
    assert unknown.confidence is None and unknown.x is None
    first = rows[0]
    assert (first.frame, first.x, first.y, first.split) == (100, 10.0, 20.0, "val")
    assert first.file == "frames/p0007/0000100.jpg" and first.confirmed_point


def test_a_persons_labels_replace_the_trackers_and_can_add_frames():
    clip = dataset.Clip("p0007", 5.0, 5.15, 7, False)
    rows = dataset.pseudo_labels(_ball(), [clip])
    fixed = dataset.LabelRow(
        "p0007", 101, 5.1, rows[1].file, 1, 55.0, 66.0, "human", None, 7, False, "train"
    )
    added = dataset.LabelRow(
        "p0007",
        140,
        9.0,
        "frames/p0007/0000140.jpg",
        0,
        None,
        None,
        "human",
        None,
        7,
        False,
        "train",
    )
    untouched = rows[0]
    merged = dataset.apply_corrections(rows, [fixed, added, untouched])
    assert merged[1] == fixed and merged[0] == untouched and merged[-1] == added


def test_labels_round_trip_through_csv_and_malformed_rows_are_refused(tmp_path):
    clip = dataset.Clip("p0007", 5.0, 5.95, 7, True)
    rows = dataset.pseudo_labels(_ball(), [clip])
    path = dataset.write_labels(tmp_path / "out" / "labels.csv", rows)
    back = dataset.read_labels(path)
    assert [(r.frame, r.label_source, r.visibility) for r in back] == [
        (r.frame, r.label_source, r.visibility) for r in rows
    ]
    assert back[0].x == pytest.approx(rows[0].x) and back[0].confirmed_point is True

    bad = tmp_path / "bad.csv"
    bad.write_text("clip,frame\np1,3\n")
    with pytest.raises(ValueError, match="missing column"):
        dataset.read_labels(bad)
    header = ",".join(dataset.LABEL_COLUMNS)
    for line, message in [
        ("p,1,0.1,f.jpg,1,,,human,,7,0,train", "visible but has no position"),
        ("p,1,0.1,f.jpg,2,1,1,human,,7,0,train", "visibility must be"),
        ("p,1,0.1,f.jpg,1,1,1,robot,,7,0,train", "Unknown label_source"),
    ]:
        bad.write_text(header + "\n" + line + "\n")
        with pytest.raises(ValueError, match=message):
            dataset.read_labels(bad)


def test_the_manifest_counts_by_source_and_names_the_video(tmp_path):
    clips = dataset.clips_from_record(_record())
    rows = dataset.pseudo_labels(_ball(), clips)
    data = dataset.manifest(
        rows,
        video={"name": "match.mp4", "sha256": "a" * 64},
        detector="motion-ball",
        split="train",
        min_confidence=0.4,
        calibration={"homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        created="2026-11-25",
    )
    assert data["frames"] == 10 and data["clips"] == 1
    assert data["by_source"] == {
        "human": 0,
        "tracker-measured": 6,
        "tracker-bridged": 1,
        "unlabelled": 3,
    }
    assert data["trainable_frames"] == 6 and data["confirmed_point_frames"] == 10
    assert "never per frame" in data["split_rule"]
    path = dataset.write_manifest(tmp_path / "m" / "manifest.json", data)
    assert json.loads(path.read_text())["video"]["sha256"] == "a" * 64
    assert (
        dataset.manifest([], video={}, detector="d", split="test", min_confidence=0.4, created="x")[
            "calibration"
        ]
        is None
    )
