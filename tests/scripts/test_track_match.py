"""Tests for the track_match CLI's argument plumbing. Nothing is tracked here."""

from __future__ import annotations

import json
import os

import pytest

from scripts import track_match
from swingvision_import import review
from swingvision_import.records import MatchRecord, PointRecord, SetRecord
from tracking.pipeline import (
    CalibrationFailedError,
    ManualKeypoints,
    TrackerPipeline,
    TrackingProgress,
)

BASE = ["--host", "Rishi", "--guest", "Alex", "--host-starts", "far", "--result", "W"]


class FakePipeline:
    """Records what `ingest` was asked, stages a tiny record, reports progress."""

    def __init__(self, pending_dir, error=None):
        self.pending_dir, self.error = pending_dir, error
        self.calls = []

    def ingest(self, video, setup, **kwargs):
        self.calls.append((video, setup, kwargs))
        if self.error is not None:
            raise self.error
        kwargs["progress"](TrackingProgress("tracking", 4, 1, 0, 600.0, 150.0, 300.0, 900.0, 120.0,
                                            "Chunk 1/4 tracked."))
        point = PointRecord(1, 1, True, True, "ace", needs_review=True)
        record = MatchRecord(
            date=kwargs["date"],
            opponent=kwargs["opponent"],
            result=kwargs["result"],
            sets=[SetRecord(1, 0, 0, points=[point])],
            import_notes=["Staged by the independent tracker from match.mp4."],
            tracking={"source": "assemble-v1", "summary": {}, "attempts": [],
                      "notes": ["Tracked 3 of 4 shots with confidence."]},
        )
        return review.save_pending(record, self.pending_dir)


@pytest.fixture
def video(tmp_path):
    """A file standing in for the recording."""
    path = tmp_path / "match.mp4"
    path.write_bytes(b"not really a video")
    os.utime(path, (1_755_734_400, 1_755_734_400))  # 2025-08-21 (UTC)
    return path


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Replaces the real pipeline with a recorder."""
    pipeline = FakePipeline(tmp_path / "pending")
    monkeypatch.setattr(track_match, "build_pipeline", lambda args: pipeline)
    return pipeline


def test_the_arguments_reach_ingest(video, fake, capsys, tmp_path):
    keypoints = tmp_path / "court.json"
    keypoints.write_text(json.dumps({"near_baseline:left_singles": [1, 2]}))
    status = track_match.main(
        [str(video), *BASE, "--host-hand", "left", "--date", "2026-08-21", "--opponent",
         "Alex B", "--start", "355", "--end", "600", "--force", "--keypoints", str(keypoints),
         "--tracked-identity", "Rishi", "--player-appearance", "grey shirt",
         "--first-server", "1=me", "--first-server", "2=opponent", "--energy-rating", "4",
         "--mental-rating", "2", "--location", "Club", "--reviewer-note", "cut at 3-2"]
    )
    assert status == 0
    (called_video, setup, kwargs) = fake.calls[0]
    assert called_video == video
    assert (setup.host_name, setup.guest_name, setup.host_player) == ("Rishi", "Alex", "B")
    assert setup.host_hand == "left" and setup.guest_hand == "right"
    assert kwargs["date"] == "2026-08-21" and kwargs["opponent"] == "Alex B"
    assert (kwargs["start"], kwargs["end"], kwargs["force"]) == (355.0, 600.0, True)
    assert kwargs["keypoints"] == ManualKeypoints({"near_baseline:left_singles": (1.0, 2.0)})
    assert kwargs["tracked_identity"] == "Rishi"
    assert kwargs["player_appearance"] == "grey shirt"
    assert kwargs["first_server_by_set"] == {1: "me", 2: "opponent"}
    assert (kwargs["energy_rating"], kwargs["mental_rating"]) == (4, 2)
    assert (kwargs["location"], kwargs["reviewer_note"]) == ("Club", "cut at 3-2")
    out = capsys.readouterr().out
    assert "[tracking] 1/4 chunks (2.5 of 10.0 min of video), 120s per video minute" in out
    assert "ETA 0:15:00" in out
    assert "Coverage: Tracked 3 of 4 shots with confidence." in out
    assert "1 point(s) need review" in out
    assert "NOTE: Staged by the independent tracker" in out
    assert "scripts/verify_footage.py" in out


def test_without_a_date_the_files_date_is_used_and_said(video, fake, capsys):
    assert track_match.main([str(video), *BASE]) == 0
    (_, _, kwargs) = fake.calls[0]
    assert kwargs["date"].startswith("2025-08-2")
    assert kwargs["opponent"] == "Alex" and kwargs["force"] is False
    assert kwargs["keypoints"] is None and "player_appearance" not in kwargs
    assert kwargs["first_server_by_set"] is None and "energy_rating" not in kwargs
    assert "modification date" in capsys.readouterr().out


def test_a_problem_the_user_can_fix_exits_with_its_message(video, tmp_path, monkeypatch, capsys):
    failing = FakePipeline(tmp_path, CalibrationFailedError("Re-aim the camera."))
    monkeypatch.setattr(track_match, "build_pipeline", lambda args: failing)
    with pytest.raises(SystemExit) as caught:
        track_match.main([str(video), *BASE, "--date", "2026-01-01"])
    assert caught.value.code == 2
    assert "error: Re-aim the camera." in capsys.readouterr().err


def test_a_missing_video_is_refused(tmp_path, capsys):
    with pytest.raises(SystemExit) as caught:
        track_match.main([str(tmp_path / "nope.mp4"), *BASE])
    assert caught.value.code == 2
    assert "no such video" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra",
    [["--resume", "--force"], ["--start", "-3"], ["--end", "soon"], ["--host-starts", "middle"],
     ["--energy-rating", "6"], ["--mental-rating", "x"], ["--first-server", "1=you"],
     ["--first-server", "one=me"]],
)
def test_bad_arguments_are_refused_by_the_parser(video, extra):
    with pytest.raises(SystemExit) as caught:
        track_match.main([str(video), *BASE, *extra])
    assert caught.value.code == 2


def test_the_pipeline_is_built_from_the_options(tmp_path):
    args = track_match.build_parser().parse_args(
        ["v.mp4", *BASE, "--pending-dir", str(tmp_path / "p"), "--work-dir", str(tmp_path / "w"),
         "--chunk-seconds", "60", "--overlap-seconds", "5", "--no-pose"]
    )
    pipeline = track_match.build_pipeline(args)
    assert isinstance(pipeline, TrackerPipeline)
    assert pipeline.import_config.pending_dir == tmp_path / "p"
    assert pipeline.config.pipeline.work_dir == tmp_path / "w"
    assert pipeline.config.pipeline.chunk_seconds == 60.0
    assert pipeline.config.pipeline.overlap_seconds == 5.0
    assert pipeline.stages.use_pose is False
    default = track_match.build_pipeline(track_match.build_parser().parse_args(["v.mp4", *BASE]))
    assert default.stages.use_pose is True and default.config.pipeline.work_dir is None


def test_progress_lines_before_anything_was_timed():
    line = track_match.format_progress(TrackingProgress("opening", message="Reading."))
    assert line == "[opening] Reading."
    assert track_match._clock(None) == "?"
