"""Tests for the court-real placement heatmap's data."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import pytest
from openpyxl import Workbook

from shot_map import placement
from swingvision_import.config import ImportConfig
from swingvision_import.db import get_connection
from swingvision_import.load import finalize_and_load
from swingvision_import.parse import SwingVisionParser
from swingvision_import.raw import RawMatchExport, RawSettings, RawShotRow
from swingvision_import.records import MatchRecord, PointRecord, SetRecord
from swingvision_import.review import save_pending


def _shot(point, number, player, shot_type, stroke, result, **where):
    return RawShotRow(point, number, player, shot_type, stroke, result, **where)


def _record(**extra) -> MatchRecord:
    return MatchRecord(
        date="2026-08-21",
        opponent="Alex",
        result="W",
        sets=[
            SetRecord(
                set_number=1,
                games_won=6,
                games_lost=4,
                points=[PointRecord(1, 1, True, True, "ace")],
            )
        ],
        **extra,
    )


def _tracked_record(shots: list[RawShotRow], attempts: list[dict]) -> MatchRecord:
    return _record(
        tracking={
            "source": "assemble-v1",
            "summary": {},
            "notes": [],
            "attempts": attempts,
            "settings": {"host_name": "Rishi", "guest_name": "Alex"},
            "shots": [asdict(s) for s in shots],
        }
    )


def _trace(point, shot, provenance):
    fields = {"player": {"value": "Rishi", "provenance": "measured"}}
    if provenance:
        fields["bounce_xy"] = {"value": [0.0, 0.0], "provenance": provenance}
    return {"point_number": point, "shot_number": shot, "fields": fields}


@pytest.fixture
def parser() -> SwingVisionParser:
    return SwingVisionParser(ImportConfig())


def test_a_tracked_match_faces_every_bounce_the_hitters_way(parser):
    shots = [
        # Near-end serve into the far deuce box: drawn as recorded.
        _shot(1, 1, "Rishi", "first_serve", "Serve", "In", hit_side="near",
              bounce_x=1.5, bounce_y=17.0),
        # Far-end return: turned through 180 degrees.
        _shot(1, 2, "Alex", "first_return", "Backhand", "In", hit_side="far",
              bounce_x=2.0, bounce_y=4.0),
        # No bounce seen: counted in the denominator, not drawn.
        _shot(1, 3, "Rishi", "in_play", "Unclear", "Out", hit_side="near"),
        # A feed between points is not match play at all.
        _shot(2, 1, "Rishi", "none", "Feed", "In", bounce_x=0.0, bounce_y=20.0),
    ]
    folded = {"shots": [_trace(1, 2, "interpolated")]}
    attempts = [{"shots": [_trace(1, 1, "measured"), _trace(1, 3, None)],
                 "folded_fault": folded}]
    result = placement.placement_from_record(9, _tracked_record(shots, attempts), parser)

    assert result.available and result.reason is None
    assert result.source == "tracker"
    assert (result.shots, result.with_bounce) == (3, 2)
    assert result.provenance == {"measured": 1, "interpolated": 1}
    serve, ret = result.bounces
    assert (serve.x, serve.y, serve.by, serve.kind, serve.rotated) == (
        1.5, 17.0, "me", "serve", False
    )
    assert serve.provenance == "measured"
    assert (ret.x, ret.y, ret.by, ret.kind, ret.rotated) == (-2.0, 19.77, "opponent", "rally", True)
    assert ret.provenance == "interpolated"


def test_a_tracked_match_with_no_bounces_says_why(parser):
    shots = [_shot(1, 1, "Rishi", "first_serve", "Serve", "In", hit_side="near")]
    result = placement.placement_from_record(9, _tracked_record(shots, []), parser)
    assert not result.available
    assert result.reason.startswith("None of this match's 1 shots has a bounce position")
    assert "tracker saw no bounces" in result.reason
    assert result.provenance == {}


def test_the_hitters_end_falls_back_to_the_hit_then_the_bounce(parser, monkeypatch):
    shots = [
        _shot(1, 1, "Test Player", "in_play", "Forehand", "In", hit_y=22.0,
              bounce_x=1.0, bounce_y=5.0),
        _shot(1, 2, "Test Player", "in_play", "Forehand", "In", hit_y=1.0,
              bounce_x=1.0, bounce_y=18.0),
        _shot(1, 3, "Test Opponent", "in_play", "Backhand", "In",
              bounce_x=-1.0, bounce_y=3.0),
        _shot(1, 4, "Test Opponent", "in_play", "Backhand", "Out",
              bounce_x=math.nan, bounce_y=3.0),
    ]
    export = RawMatchExport(settings=RawSettings("Test Player", "Test Opponent"), shots=shots)
    monkeypatch.setattr(placement.sources, "parse_sources", lambda record, p: [export])
    result = placement.placement_from_record(3, _record(source_file="x.xlsx"), parser)
    assert result.source == "export"
    assert result.provenance == {}
    assert [(b.rotated, b.by) for b in result.bounces] == [
        (True, "me"),
        (False, "me"),
        (True, "opponent"),
    ]
    assert result.bounces[2].provenance is None
    assert result.shots == 4 and result.with_bounce == 3


def test_an_export_without_settings_has_no_me(parser, monkeypatch):
    shots = [_shot(1, 1, "Someone", "first_serve", "Serve", "In", hit_side="near",
                   bounce_x=0.5, bounce_y=16.0)]
    monkeypatch.setattr(
        placement.sources, "parse_sources", lambda record, p: [RawMatchExport(shots=shots)]
    )
    result = placement.placement_from_record(3, _record(source_file="x.xlsx"), parser)
    assert result.bounces[0].by == "opponent"


def test_an_export_with_no_coordinates_says_so(parser, monkeypatch):
    shots = [_shot(1, 1, "Test Player", "first_serve", "Serve", "In")]
    export = RawMatchExport(settings=RawSettings("Test Player", "Opp"), shots=shots)
    monkeypatch.setattr(placement.sources, "parse_sources", lambda record, p: [export])
    result = placement.placement_from_record(3, _record(source_file="x.xlsx"), parser)
    assert not result.available
    assert "export carried no Bounce (x)/(y) values" in result.reason


def test_a_record_with_no_sources_has_nothing_to_draw(parser):
    result = placement.placement_from_record(3, _record(), parser)
    assert not result.available
    assert "kept no per-shot data" in result.reason


def test_an_unreadable_export_is_reported_not_raised(parser):
    result = placement.placement_from_record(3, _record(source_file="missing.xlsx"), parser)
    assert not result.available
    assert "could not be re-read" in result.reason


def _xlsx_with_bounces(path: Path) -> Path:
    workbook = Workbook()
    sets = workbook.active
    sets.title = "Sets"
    sets.append(["Set", "Host Score", "Guest Score", "Set Winner"])
    sets.append([1, 6, 4, "host"])
    workbook.create_sheet("Games").append(["Game", "Set", "Server", "Game Winner"])
    workbook.create_sheet("Points").append(
        ["Point", "Game", "Set", "Match Server", "Point Winner", "Detail"]
    )
    settings = workbook.create_sheet("Settings")
    settings.append(["Host Team", "Guest Team"])
    settings.append(["Test Player", "Test Opponent"])
    shots = workbook.create_sheet("Shots")
    shots.append(["Point", "Shot", "Player", "Type", "Stroke", "Result", "Hit Side",
                  "Bounce (x)", "Bounce (y)"])
    shots.append([1, 1, "Test Player", "first_serve", "Serve", "In", "far", -1.4, 6.2])
    shots.append([1, 2, "Test Opponent", "first_return", "Forehand", "Net", "near", 0.2, 9.0])
    workbook.save(path)
    return path


def _finalize(import_config: ImportConfig, record: MatchRecord) -> int:
    connection = get_connection(import_config.db_path, import_config.schema_path)
    try:
        return finalize_and_load(connection, record)
    finally:
        connection.close()


def test_a_finalized_export_is_traced_back_through_its_staged_file(tmp_path, import_config):
    xlsx = _xlsx_with_bounces(tmp_path / "export.xlsx")
    record = _record(source_file=str(xlsx))
    save_pending(record, import_config.pending_dir)
    match_id = _finalize(import_config, record)

    result = placement.match_placement(
        import_config.db_path, import_config.pending_dir, match_id, import_config=import_config
    )
    assert result.available and result.source == "export"
    serve, ret = result.bounces
    assert (serve.x, serve.y, serve.rotated, serve.by) == (1.4, 17.57, True, "me")
    assert (ret.result, ret.rotated, ret.by) == ("Net", False, "opponent")
    assert result.to_dict()["bounces"][0]["kind"] == "serve"


def test_an_unknown_match_is_none(import_config):
    _finalize(import_config, _record())
    assert placement.match_placement(import_config.db_path, import_config.pending_dir, 99) is None


def test_a_match_whose_staged_file_is_gone_is_unavailable(import_config):
    match_id = _finalize(import_config, _record())
    result = placement.match_placement(import_config.db_path, import_config.pending_dir, match_id)
    assert not result.available
    assert "no longer on disk" in result.reason
    assert result.source is None


def test_a_corrupt_staged_file_is_treated_as_gone(import_config):
    match_id = _finalize(import_config, _record())
    import_config.pending_dir.mkdir(parents=True)
    (import_config.pending_dir / "2026-08-21_Alex.json").write_text("{not json", "utf-8")
    result = placement.match_placement(import_config.db_path, import_config.pending_dir, match_id)
    assert not result.available
