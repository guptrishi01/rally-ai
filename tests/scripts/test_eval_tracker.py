"""Tests for the eval_tracker CLI: a staged tracked match scored against references."""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

from scripts import eval_tracker
from swingvision_import import reconstruct, review
from swingvision_import.raw import RawShotRow
from swingvision_import.records import MatchRecord


def _row(point, shot, t, player, side, stroke, shot_type, result="In"):
    return RawShotRow(point, shot, player, shot_type, stroke, result, t, side)


ROWS = [
    _row(1, 1, 10.0, "Me", "near", "Serve", "first_serve"),
    _row(1, 2, 11.4, "Them", "far", "Forehand", "first_return", "Net"),
    _row(2, 1, 40.0, "Me", "near", "Serve", "first_serve"),
    _row(2, 2, 41.3, "Them", "far", "Backhand", "first_return"),
]


def _stage(tmp_path, tracking=True, confirm=False):
    result = reconstruct.reconstruct_all(ROWS, "Me")
    record = MatchRecord(date="2026-08-21", opponent="Them", result="W", sets=result.sets)
    if tracking:
        record.tracking = {
            "shots": [asdict(r) for r in ROWS],
            "settings": {"host_name": "Me"},
            "range": [0.0, 60.0],
            "attempts": [],
        }
    if confirm:
        record.sets[0].points[0].confirmed_by = "human"
    record.pending_name = "2026-08-21_Them.json"
    return review.save_pending(record, tmp_path)


class FakeParser:
    """Stands in for SwingVisionParser: the same shots, 2s behind the video."""

    def __init__(self, _config):
        pass

    def parse(self, _path):
        shots = [
            RawShotRow(**{**asdict(r), "video_time": r.video_time - 2.0, "player": "Ref"})
            for r in ROWS
        ]
        return SimpleNamespace(shots=shots, settings=SimpleNamespace(host_name="Ref"))


def test_a_staged_match_is_scored_against_an_export_and_written_as_json(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(eval_tracker, "SwingVisionParser", FakeParser)
    staged = _stage(tmp_path, confirm=True)
    out = tmp_path / "results.json"
    code = eval_tracker.main(
        [str(staged), "--reference", "x.xlsx", "--clock-offset", "2.0", "--json", str(out)]
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "weak reference, not ground truth" in printed
    assert "shot recall      4/4 (100%)" in printed
    assert "Against the reviewer's confirmations" in printed
    data = json.loads(out.read_text())
    assert data["reference"]["window"] == [0.0, 60.0]
    assert data["reference"]["serve_recall"]["hits"] == 2
    assert data["confirmed"]["confirmed_points"] == 1


def test_an_explicit_window_overrides_the_tracked_range(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(eval_tracker, "SwingVisionParser", FakeParser)
    staged = _stage(tmp_path)
    assert eval_tracker.main([str(staged), "--reference", "x.xlsx", "--start", "30"]) == 0
    assert "over 30-60s" in capsys.readouterr().out


def test_without_an_export_only_the_confirmations_are_scored(tmp_path, capsys):
    staged = _stage(tmp_path)
    assert eval_tracker.main([str(staged)]) == 0
    printed = capsys.readouterr().out
    assert "SwingVision" not in printed and "0 point(s) confirmed by a person" in printed


def test_a_match_not_staged_by_the_tracker_is_refused(tmp_path, capsys):
    staged = _stage(tmp_path, tracking=False)
    assert eval_tracker.main([str(staged)]) == 2
    assert "was not staged by the tracker" in capsys.readouterr().err
