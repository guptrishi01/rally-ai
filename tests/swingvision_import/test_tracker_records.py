"""Tests for staged records the independent tracker produced.

The tracker's records differ from an import's in two additive fields
(`MatchRecord.tracking`, `PointRecord.tracking`) and in where their shots
live: in the record itself, not in an export to re-parse.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from swingvision_import import review, sources
from swingvision_import.config import ImportConfig
from swingvision_import.parse import SwingVisionParser
from swingvision_import.raw import RawShotRow
from swingvision_import.records import MatchRecord, PointRecord, SetRecord

ROW = RawShotRow(
    point_number=1,
    shot_number=1,
    player="Host",
    shot_type="first_serve",
    stroke="Serve",
    result="In",
    video_time=12.5,
    hit_side="near",
    hit_zone="deuce",
    bounce_x=-2.0,
    bounce_y=16.5,
)


def _tracked(settings=True):
    """A minimal tracker-staged record."""
    block = {"producer": "independent-tracker", "shots": [asdict(ROW)]}
    if settings:
        block["settings"] = {"host_name": "Host", "guest_name": "Guest"}
    point = PointRecord(1, 1, True, True, "ace", needs_review=True, tracking={"number": 1})
    return MatchRecord(
        date="2026-08-21",
        opponent="Guest",
        result="W",
        source_file="C:/videos/match.mp4",
        sets=[SetRecord(set_number=1, games_won=0, games_lost=0, points=[point])],
        tracking=block,
    )


def test_a_tracked_records_own_rows_stand_in_for_the_export():
    (export,) = sources.parse_sources(_tracked(), SwingVisionParser(ImportConfig()))
    assert export.shots == [ROW]
    assert export.settings.host_name == "Host" and export.settings.guest_name == "Guest"


def test_without_names_the_rows_still_come_back():
    (export,) = sources.parse_sources(_tracked(settings=False), SwingVisionParser(ImportConfig()))
    assert export.settings is None and export.shots == [ROW]


def test_an_imported_record_is_not_taken_for_a_tracked_one():
    record = MatchRecord(date="2026-08-21", opponent="Guest", result="W")
    assert sources.tracker_export(record) is None
    record.tracking = {"producer": "independent-tracker"}  # no rows kept
    assert sources.tracker_export(record) is None
    assert sources.parse_sources(record, SwingVisionParser(ImportConfig())) == []


def test_tracking_blocks_survive_the_staged_json(tmp_path):
    path = review.save_pending(_tracked(), tmp_path)
    loaded = review.load_pending(path)
    assert loaded.tracking["shots"][0]["bounce_y"] == 16.5
    assert loaded.sets[0].points[0].tracking == {"number": 1}


def test_files_staged_before_tracking_existed_still_load(tmp_path):
    old = asdict(_tracked())
    del old["tracking"]
    for point in old["sets"][0]["points"]:
        del point["tracking"]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    loaded = review.load_pending(path)
    assert loaded.tracking is None and loaded.sets[0].points[0].tracking is None
