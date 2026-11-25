"""Tests for the export_ball_dataset CLI. No real video is decoded."""

from __future__ import annotations

import csv
import json

import numpy as np

from scripts import export_ball_dataset as export
from swingvision_import import review
from swingvision_import.records import MatchRecord, PointRecord, SetRecord
from tracking import dataset, stitch
from tracking.pipeline import CheckpointStore
from tracking.video import VideoFrame


def _ball(start=5.0, count=10):
    states = np.array([1, 1, 0, 1, 2, 1, 1, 1, 0, 1], dtype=np.int8)[:count]
    xy = np.column_stack([np.arange(count) + 10.0, np.arange(count) + 20.0])
    return stitch.CompactBallTrack(
        frame_numbers=np.arange(count) + int(start * 10),
        frame_pts=np.arange(count) / 10.0 + start,
        state=states,
        confidence=np.where(states > 0, 0.8, 0.0),
        image_xy=xy,
        segment=np.where(states > 0, 0, -1),
        image_size=(64, 36),
        source="motion-ball",
    )


def _chunk(index, ball, last=False):
    window = stitch.ChunkWindow(index, 0.0, 1.0, 0.0, 1.0, last)
    return stitch.ChunkResult(
        window, (), (), (), ball, (), (), (), 0, (), {}, None, 0, None, False, 0.0, b"", "x"
    )


def _stage(tmp_path, work):
    point = PointRecord(1, 1, True, True, "winner", source_point_number=3, confirmed_by="human")
    record = MatchRecord(
        date="2026-08-21",
        opponent="Opp",
        result="W",
        sets=[SetRecord(set_number=1, games_won=1, games_lost=0, points=[point])],
    )
    record.tracking = {
        "attempts": [
            {"number": 3, "kind": "served", "start": 5.5, "end": 5.6},
            {"number": 4, "kind": "served", "start": 50.0, "end": 51.0},
        ],
        "run": {"work_folder": str(work)},
        "video": {"path": str(tmp_path / "match.mp4"), "name": "match.mp4", "checksum": "d" * 64},
        "models": {"ball": "motion-ball"},
        "calibration": {"image_size": [64, 36]},
    }
    record.pending_name = "2026-08-21_Opp.json"
    return review.save_pending(record, tmp_path)


def test_a_runs_checkpoints_give_the_whole_ball_track(tmp_path):
    store = CheckpointStore(tmp_path / "work")
    store.save_chunk(_chunk(0, _ball(5.0)))
    store.save_chunk(_chunk(1, _ball(50.0), last=True))
    (tmp_path / "work" / "chunk_0002.pkl").write_bytes(b"not a checkpoint")
    ball = export.load_ball_track(tmp_path / "work")
    assert len(ball.frame_pts) == 20


def test_no_checkpoints_is_a_clear_error(tmp_path, capsys):
    staged = _stage(tmp_path, tmp_path / "missing")
    assert export.main([str(staged), str(tmp_path / "out")]) == 2
    assert "No chunk checkpoints" in capsys.readouterr().err


def test_labels_manifest_and_labelled_frames_are_written(tmp_path, capsys):
    store = CheckpointStore(tmp_path / "work")
    store.save_chunk(_chunk(0, _ball(5.0), last=True))
    staged = _stage(tmp_path, tmp_path / "work")
    asked = []

    def reader(video, start, end):
        asked.append((video.name, round(start, 2), round(end, 2)))
        for k in range(10):
            pts = 5.0 + k / 10.0
            if start <= pts <= end:
                yield VideoFrame(pts, np.full((36, 64, 3), 40 * (k % 6), dtype=np.uint8))

    out = tmp_path / "out"
    assert export.main([str(staged), str(out), "--split", "val"], reader=reader) == 0
    printed = capsys.readouterr().out
    assert "pseudo-labels" in printed
    rows = dataset.read_labels(out / "labels.csv")
    assert {r.clip for r in rows} == {"p0003"} and all(r.split == "val" for r in rows)
    images = sorted(p.name for p in (out / "frames" / "p0003").iterdir())
    assert images == [f"{r.frame:07d}.jpg" for r in rows if r.visibility == 1]
    assert asked == [("match.mp4", 4.95, 5.95)]
    data = json.loads((out / "manifest.json").read_text())
    assert data["video"]["sha256"] == "d" * 64 and data["detector"] == "motion-ball"


def test_corrections_and_review_filters_reach_the_export(tmp_path):
    store = CheckpointStore(tmp_path / "work")
    store.save_chunk(_chunk(0, _ball(5.0), last=True))
    staged = _stage(tmp_path, tmp_path / "work")
    out = tmp_path / "out"
    assert export.main([str(staged), str(out), "--frames", "none"]) == 0
    labels = out / "labels.csv"
    with labels.open(newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    table[0].update(label_source="human", x="33.000", y="44.000", visibility="1")
    fixed = tmp_path / "fixed.csv"
    with fixed.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=dataset.LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(table)
    args = [str(staged), str(out), "--frames", "none", "--corrections", str(fixed)]
    assert export.main([*args, "--confirmed-only"]) == 0
    rows = dataset.read_labels(labels)
    assert rows[0].label_source == "human" and rows[0].x == 33.0
    data = json.loads((out / "manifest.json").read_text())
    assert data["by_source"]["human"] == 1


def test_all_frames_can_be_written_or_none(tmp_path):
    rows = dataset.pseudo_labels(_ball(5.0), [dataset.Clip("p0003", 5.0, 5.95, 3, False)])
    assert export.write_frames(tmp_path / "v.mp4", rows, tmp_path, "none") == 0

    def reader(_video, start, end):
        for k in range(10):
            yield VideoFrame(5.0 + k / 10.0, np.zeros((4, 4, 3), dtype=np.uint8))
        yield VideoFrame(99.0, np.zeros((4, 4, 3), dtype=np.uint8))

    assert export.write_frames(tmp_path / "v.mp4", rows, tmp_path, "all", reader) == 10
