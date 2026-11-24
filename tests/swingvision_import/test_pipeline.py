from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import zipfile

import pytest
from openpyxl import Workbook

from swingvision_import import footage
from swingvision_import.load import UnresolvedReviewError
from swingvision_import.pipeline import SwingVisionImportPipeline
from swingvision_import.records import MatchRecord
from swingvision_import.review import ConfirmationError, load_pending, save_pending
from tests.ai.conftest import FakeMessage, FakeTextBlock
from tests.swingvision_import.conftest import add_settings_and_shots_sheets


def _clear_review_flags(pipeline, json_path):
    record = load_pending(json_path)
    for set_record in record.sets:
        for point in set_record.points:
            point.needs_review = False
    save_pending(record, pipeline.config.pending_dir)


def test_ingest_writes_pending_json_without_touching_sql(synthetic_xlsx, import_config):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(synthetic_xlsx, date="2026-08-06", opponent="Alex", result="W")

    assert json_path.exists()
    assert not import_config.db_path.exists()


def test_finalize_rejects_until_flags_resolved(synthetic_xlsx, import_config):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(synthetic_xlsx, date="2026-08-06", opponent="Alex", result="W")

    with pytest.raises(UnresolvedReviewError):
        pipeline.finalize(json_path)


def test_finalize_loads_once_flags_are_cleared(synthetic_xlsx, import_config):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(synthetic_xlsx, date="2026-08-06", opponent="Alex", result="W")
    _clear_review_flags(pipeline, json_path)

    match_id = pipeline.finalize(json_path)
    assert match_id == 1


def test_rerunning_finalize_on_same_match_is_rejected_not_duplicated(synthetic_xlsx, import_config):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(synthetic_xlsx, date="2026-08-06", opponent="Alex", result="W")
    _clear_review_flags(pipeline, json_path)

    pipeline.finalize(json_path)
    with pytest.raises(ValueError):
        pipeline.finalize(json_path)


def test_match_overrides_survive_the_full_ingest_then_finalize_round_trip(
    synthetic_xlsx, import_config
):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(
        synthetic_xlsx,
        date="2026-08-06",
        opponent="Alex",
        result="W",
        energy_rating=5,
        pros="Aggressive on return games",
    )
    _clear_review_flags(pipeline, json_path)
    pipeline.finalize(json_path)

    connection = sqlite3.connect(import_config.db_path)
    row = connection.execute("SELECT energy_rating, pros FROM match").fetchone()
    connection.close()
    assert row == (5, "Aggressive on return games")


def test_reingesting_the_same_export_overwrites_the_pending_file_at_the_same_path(
    synthetic_xlsx, import_config
):
    """The same export staged twice is one match, so it must not duplicate.

    CLAUDE.md's "never reprocess or duplicate by default" rule: correcting
    the intake fields and re-staging should land back on the same file.
    """
    pipeline = SwingVisionImportPipeline(import_config)
    first_path = pipeline.ingest(synthetic_xlsx, date="2026-08-06", opponent="Alex", result="W")
    second_path = pipeline.ingest(
        synthetic_xlsx, date="2026-08-06", opponent="Alex", result="W", pros="Revised notes"
    )

    assert first_path == second_path
    assert list(import_config.pending_dir.glob("*.json")) == [first_path]
    assert load_pending(first_path).pros == "Revised notes"


def test_a_second_match_the_same_day_is_staged_beside_the_first_not_over_it(
    synthetic_xlsx, tmp_path, import_config
):
    """Two different matches vs the same opponent on one day both survive.

    This is the data-loss path the date_opponent filename used to have: the
    second ingest silently destroyed the first. Distinguished from a
    re-ingest by the export it came from, not by the name being taken.
    """
    other_xlsx = tmp_path / "second_match.xlsx"
    other_xlsx.write_bytes(synthetic_xlsx.read_bytes())

    pipeline = SwingVisionImportPipeline(import_config)
    first_path = pipeline.ingest(synthetic_xlsx, date="2026-08-06", opponent="Alex", result="W")
    second_path = pipeline.ingest(other_xlsx, date="2026-08-06", opponent="Alex", result="L")

    assert first_path != second_path
    assert second_path.name == "2026-08-06_Alex_2.json"
    assert load_pending(first_path).result == "W"
    assert load_pending(second_path).result == "L"


def test_ingest_falls_back_to_shot_reconstruction_when_points_sheet_is_empty(
    synthetic_non_pro_xlsx, import_config
):
    pipeline = SwingVisionImportPipeline(import_config)

    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx, date="2026-08-06", opponent="Alex", result="W"
    )
    record = load_pending(json_path)

    all_points = [p for s in record.sets for p in s.points]
    assert len(all_points) == 2
    assert all(p.needs_review for p in all_points)
    assert all(p.source_point_number is not None for p in all_points)
    # Point 1 was a single In serve shot by the host - an ace.
    assert any(p.point_end_type == "ace" for p in all_points)


def test_ingest_populates_import_notes_with_quality_check_and_gap_findings(
    synthetic_non_pro_xlsx, import_config
):
    pipeline = SwingVisionImportPipeline(import_config)

    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx,
        date="2026-08-06",
        opponent="Alex",
        result="W",
        first_server_by_set={1: "opponent"},  # deliberately wrong ground truth
        tracked_identity="Someone Else",  # deliberately wrong identity
    )
    record = load_pending(json_path)

    notes = " | ".join(record.import_notes)
    assert "had no shot data" in notes  # point 2 is a gap
    assert "reversed" in notes  # serve-order mismatch
    assert "Someone Else" in notes  # identity mismatch
    assert "does not match" in notes  # reconstructed score vs Sets-sheet summary


def test_ingest_skips_serve_order_and_identity_checks_when_not_supplied(
    synthetic_non_pro_xlsx, import_config
):
    pipeline = SwingVisionImportPipeline(import_config)

    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx, date="2026-08-06", opponent="Alex", result="W"
    )
    record = load_pending(json_path)

    notes = " | ".join(record.import_notes)
    assert "reversed" not in notes
    assert "identified yourself" not in notes


def test_ingest_matching_identity_produces_no_identity_note(synthetic_non_pro_xlsx, import_config):
    pipeline = SwingVisionImportPipeline(import_config)

    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx,
        date="2026-08-06",
        opponent="Alex",
        result="W",
        tracked_identity="Test Player",
    )
    record = load_pending(json_path)

    assert "identified yourself" not in " | ".join(record.import_notes)


def test_ingest_reports_excluded_non_match_points_in_import_notes(tmp_path, import_config):
    workbook = Workbook()
    sets_sheet = workbook.active
    sets_sheet.title = "Sets"
    sets_sheet.append(["Set", "Host Score", "Guest Score", "Set Winner"])
    sets_sheet.append([1, 0, 0, "host"])
    workbook.create_sheet("Games").append(["Game", "Set", "Server", "Game Winner"])
    workbook.create_sheet("Points").append(
        ["Point", "Game", "Set", "Match Server", "Point Winner", "Detail"]
    )
    shot_rows = [
        [1, 1, "Test Player", "first_serve", "Serve", "In"],
        [1, 2, "Test Opponent", "first_return", "Backhand", "Net"],
        # Point 2: a fed ball, not real rallying - should be excluded, not
        # counted as a real point.
        [2, 0, "Test Player", "none", "Feed", "In"],
        [2, 1, "Test Opponent", "none", "Backhand", "Net"],
    ]
    add_settings_and_shots_sheets(workbook, shot_rows=shot_rows)
    path = tmp_path / "with_feed_point.xlsx"
    workbook.save(path)

    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(path, date="2026-08-06", opponent="Alex", result="W")
    record = load_pending(json_path)

    assert any("excluded as non-match activity" in note for note in record.import_notes)


def test_ingest_folds_a_first_serve_fault_into_its_second_serve_and_says_so(
    tmp_path, import_config
):
    workbook = Workbook()
    sets_sheet = workbook.active
    sets_sheet.title = "Sets"
    sets_sheet.append(["Set", "Host Score", "Guest Score", "Set Winner"])
    sets_sheet.append([1, 0, 0, "host"])
    workbook.create_sheet("Games").append(["Game", "Set", "Server", "Game Winner"])
    workbook.create_sheet("Points").append(
        ["Point", "Game", "Set", "Match Server", "Point Winner", "Detail"]
    )
    shot_rows = [
        # SwingVision numbers the fault and the second serve as two points.
        [1, 1, "Test Player", "first_serve", "Serve", "Net", 10.0, "near", "Deuce"],
        [2, 1, "Test Player", "first_serve", "Serve", "In", 18.0, "near", "deuce"],
        [2, 2, "Test Opponent", "first_return", "Backhand", "Net", 19.0, "far", "deuce"],
    ]
    add_settings_and_shots_sheets(workbook, shot_rows=shot_rows)
    path = tmp_path / "fault_then_second_serve.xlsx"
    workbook.save(path)

    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(path, date="2026-08-06", opponent="Alex", result="W")
    record = load_pending(json_path)

    points = [p for s in record.sets for p in s.points]
    assert len(points) == 1
    assert points[0].first_serve_in is False
    assert points[0].second_serve_in is True
    assert points[0].source_point_number == 2
    assert any("first-serve fault(s) were folded" in note for note in record.import_notes)


def test_confirm_point_clears_a_flag_directly_without_an_api_call(
    synthetic_non_pro_xlsx, import_config
):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx, date="2026-08-06", opponent="Alex", result="W"
    )
    record = load_pending(json_path)
    target = record.sets[0].points[0]

    confirmed = pipeline.confirm_point(
        json_path,
        set_number=1,
        game_number=target.game_number,
        point_number=target.point_number,
        point_end_type="ace",
        point_won=True,
        net_approach=False,
    )

    updated = confirmed.sets[0].points[0]
    assert updated.point_end_type == "ace"
    assert updated.point_won is True
    assert updated.needs_review is False

    # Persisted to the same pending file.
    reloaded = load_pending(json_path)
    assert reloaded.sets[0].points[0].needs_review is False


def test_confirm_point_every_flagged_point_then_finalize_succeeds(
    synthetic_non_pro_xlsx, import_config
):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx, date="2026-08-06", opponent="Alex", result="W"
    )
    record = load_pending(json_path)
    for point in record.sets[0].points:
        pipeline.confirm_point(
            json_path,
            set_number=1,
            game_number=point.game_number,
            point_number=point.point_number,
            point_end_type="winner",
            point_won=True,
            net_approach=False,
        )

    match_id = pipeline.finalize(json_path)
    assert match_id == 1


def test_confirm_point_rejects_an_inconsistent_pair_and_leaves_the_point_flagged(
    synthetic_non_pro_xlsx, import_config
):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx, date="2026-08-06", opponent="Alex", result="W"
    )
    record = load_pending(json_path)
    target = record.sets[0].points[0]

    with pytest.raises(ConfirmationError, match="requires point_won"):
        pipeline.confirm_point(
            json_path,
            set_number=1,
            game_number=target.game_number,
            point_number=target.point_number,
            point_end_type="ace",
            point_won=False,
            net_approach=False,
        )

    reloaded = load_pending(json_path)
    assert reloaded.sets[0].points[0].needs_review is True


class _FakeSuggestionMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.dumps(
            {"point_end_type": "forced_error", "reasoning": "Deep shot.", "confidence": "medium"}
        )
        return FakeMessage(content=[FakeTextBlock(text=payload)])


class _FakeSuggestionClient:
    def __init__(self):
        self.messages = _FakeSuggestionMessages()


def test_suggest_annotates_reconstructed_points_without_touching_needs_review(
    synthetic_non_pro_xlsx, import_config
):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx, date="2026-08-06", opponent="Alex", result="W"
    )
    client = _FakeSuggestionClient()

    record = pipeline.suggest(client, json_path)

    all_points = [p for s in record.sets for p in s.points]
    assert all(p.ai_suggested_point_end_type == "forced_error" for p in all_points)
    assert all(p.ai_suggestion_reasoning == "Deep shot." for p in all_points)
    assert all(p.needs_review for p in all_points)  # never cleared
    assert len(client.messages.calls) == len(all_points)

    # Suggestions persist to the same pending file.
    reloaded = load_pending(json_path)
    assert reloaded.sets[0].points[0].ai_suggested_point_end_type == "forced_error"


def test_suggest_raises_without_a_source_file_to_re_parse(import_config):
    pipeline = SwingVisionImportPipeline(import_config)
    record = MatchRecord(date="2026-08-06", opponent="Alex", result="W")
    json_path = save_pending(record, import_config.pending_dir)

    with pytest.raises(ValueError, match="source_file"):
        pipeline.suggest(_FakeSuggestionClient(), json_path)


def test_suggest_skips_direct_parse_points_with_no_shot_provenance(synthetic_xlsx, import_config):
    # synthetic_xlsx has real Points-sheet rows, so ingest() takes the
    # direct-parse path - those points have no source_point_number and no
    # raw shots to reason over, so suggest() must leave them alone.
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(synthetic_xlsx, date="2026-08-06", opponent="Alex", result="W")
    client = _FakeSuggestionClient()

    record = pipeline.suggest(client, json_path)

    all_points = [p for s in record.sets for p in s.points]
    assert all(p.source_point_number is None for p in all_points)
    assert all(p.ai_suggested_point_end_type is None for p in all_points)
    assert len(client.messages.calls) == 0


def test_ingest_raises_when_points_is_empty_and_settings_is_missing(tmp_path, import_config):
    workbook = Workbook()
    sets_sheet = workbook.active
    sets_sheet.title = "Sets"
    sets_sheet.append(["Set", "Host Score", "Guest Score", "Set Winner"])
    sets_sheet.append([1, 6, 4, "host"])
    workbook.create_sheet("Games").append(["Game", "Set", "Server", "Game Winner"])
    workbook.create_sheet("Points").append(
        ["Point", "Game", "Set", "Match Server", "Point Winner", "Detail"]
    )
    # A Settings sheet header with no data row at all -> raw.settings is None.
    workbook.create_sheet("Settings").append(["Host Team", "Guest Team"])
    workbook.create_sheet("Shots").append(["Point", "Shot", "Player", "Type", "Stroke", "Result"])
    path = tmp_path / "no_settings.xlsx"
    workbook.save(path)

    pipeline = SwingVisionImportPipeline(import_config)
    with pytest.raises(ValueError, match="Settings sheet"):
        pipeline.ingest(path, date="2026-08-06", opponent="Alex", result="W")


class _FlakySuggestionMessages:
    """Fails for the first point, succeeds for the rest."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return FakeMessage(content=[FakeTextBlock(text="not valid json")])
        payload = json.dumps(
            {"point_end_type": "winner", "reasoning": "Clean pass.", "confidence": "high"}
        )
        return FakeMessage(content=[FakeTextBlock(text=payload)])


class _FlakySuggestionClient:
    def __init__(self):
        self.messages = _FlakySuggestionMessages()


def test_suggest_continues_past_one_points_suggestion_failure(
    synthetic_non_pro_xlsx, import_config, caplog
):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx, date="2026-08-06", opponent="Alex", result="W"
    )
    client = _FlakySuggestionClient()

    with caplog.at_level(logging.ERROR):
        record = pipeline.suggest(client, json_path)

    all_points = [p for s in record.sets for p in s.points]
    succeeded = [p for p in all_points if p.ai_suggested_point_end_type == "winner"]
    failed = [p for p in all_points if p.ai_suggested_point_end_type is None]
    assert len(succeeded) == 1
    assert len(failed) == 1
    assert any("Suggestion failed" in r.message for r in caplog.records)


def _build_non_pro_xlsx(tmp_path, filename: str, shot_rows: list[list], **settings_kwargs):
    workbook = Workbook()
    sets_sheet = workbook.active
    sets_sheet.title = "Sets"
    sets_sheet.append(["Set", "Host Score", "Guest Score", "Set Winner"])
    workbook.create_sheet("Games").append(["Game", "Set", "Server", "Game Winner"])
    workbook.create_sheet("Points").append(
        ["Point", "Game", "Set", "Match Server", "Point Winner", "Detail"]
    )
    add_settings_and_shots_sheets(workbook, shot_rows=shot_rows, **settings_kwargs)
    path = tmp_path / filename
    workbook.save(path)
    return path


def test_ingest_multi_part_merges_a_game_spanning_the_file_boundary(tmp_path, import_config):
    # 3 straight host points in part 1 - not enough alone to complete a
    # game (needs 4). 1 more host point in part 2, whose own Point counter
    # restarts from 1 - only merging correctly completes the game 4-0.
    part1 = _build_non_pro_xlsx(
        tmp_path,
        "part1.xlsx",
        [
            [1, 1, "Test Player", "first_serve", "Serve", "In"],
            [2, 1, "Test Player", "first_serve", "Serve", "In"],
            [3, 1, "Test Player", "first_serve", "Serve", "In"],
        ],
    )
    part2 = _build_non_pro_xlsx(
        tmp_path, "part2.xlsx", [[1, 1, "Test Player", "first_serve", "Serve", "In"]]
    )

    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest_multi_part(
        [part1, part2], date="2026-08-18", opponent="Real Opponent", result="W"
    )
    record = load_pending(json_path)

    all_points = [p for s in record.sets for p in s.points]
    assert len(all_points) == 4
    assert record.source_files == [str(part1), str(part2)]
    assert record.sets[-1].games_won == 1
    assert record.sets[-1].games_lost == 0
    assert any("Merged from 2 files" in note for note in record.import_notes)


def test_ingest_multi_part_rejects_fewer_than_two_files(tmp_path, import_config):
    part1 = _build_non_pro_xlsx(
        tmp_path, "part1.xlsx", [[1, 1, "Test Player", "first_serve", "Serve", "In"]]
    )
    pipeline = SwingVisionImportPipeline(import_config)

    with pytest.raises(ValueError, match="at least 2 files"):
        pipeline.ingest_multi_part([part1], date="2026-08-18", opponent="Real Opponent", result="W")


def test_ingest_multi_part_rejects_files_that_disagree_on_host_name(tmp_path, import_config):
    part1 = _build_non_pro_xlsx(
        tmp_path,
        "part1.xlsx",
        [[1, 1, "Test Player", "first_serve", "Serve", "In"]],
        host_name="Rishi Gupta",
    )
    part2 = _build_non_pro_xlsx(
        tmp_path,
        "part2.xlsx",
        [[1, 1, "Someone Else", "first_serve", "Serve", "In"]],
        host_name="Someone Else",
    )
    pipeline = SwingVisionImportPipeline(import_config)

    with pytest.raises(ValueError, match="disagree"):
        pipeline.ingest_multi_part(
            [part1, part2], date="2026-08-18", opponent="Real Opponent", result="W"
        )


def test_ingest_multi_part_rejects_a_file_with_no_settings_sheet(tmp_path, import_config):
    part1 = _build_non_pro_xlsx(
        tmp_path, "part1.xlsx", [[1, 1, "Test Player", "first_serve", "Serve", "In"]]
    )

    workbook = Workbook()
    sets_sheet = workbook.active
    sets_sheet.title = "Sets"
    sets_sheet.append(["Set", "Host Score", "Guest Score", "Set Winner"])
    workbook.create_sheet("Games").append(["Game", "Set", "Server", "Game Winner"])
    workbook.create_sheet("Points").append(
        ["Point", "Game", "Set", "Match Server", "Point Winner", "Detail"]
    )
    # A Settings sheet with no data row at all -> raw.settings is None
    # (matches test_ingest_raises_when_points_is_empty_and_settings_is_missing's
    # pattern - the sheet must exist or parse() itself raises an unrelated
    # KeyError for the missing sheet).
    workbook.create_sheet("Settings").append(["Host Team", "Guest Team"])
    workbook.create_sheet("Shots").append(["Point", "Shot", "Player", "Type", "Stroke", "Result"])
    part2 = tmp_path / "part2_no_settings.xlsx"
    workbook.save(part2)

    pipeline = SwingVisionImportPipeline(import_config)

    with pytest.raises(ValueError, match="Settings sheet"):
        pipeline.ingest_multi_part(
            [part1, part2], date="2026-08-18", opponent="Real Opponent", result="W"
        )


def test_ingest_multi_part_rejects_a_pro_export_in_the_mix(tmp_path, import_config, synthetic_xlsx):
    part2 = _build_non_pro_xlsx(
        tmp_path, "part2.xlsx", [[1, 1, "Test Player", "first_serve", "Serve", "In"]]
    )
    pipeline = SwingVisionImportPipeline(import_config)

    with pytest.raises(ValueError, match="Points sheet"):
        pipeline.ingest_multi_part(
            [synthetic_xlsx, part2], date="2026-08-18", opponent="Real Opponent", result="W"
        )


def test_suggest_re_merges_multi_part_files_using_source_point_number(tmp_path, import_config):
    part1 = _build_non_pro_xlsx(
        tmp_path,
        "part1.xlsx",
        [
            [1, 1, "Test Player", "first_serve", "Serve", "In"],
            [2, 1, "Test Player", "first_serve", "Serve", "In"],
            [3, 1, "Test Player", "first_serve", "Serve", "In"],
        ],
    )
    part2 = _build_non_pro_xlsx(
        tmp_path, "part2.xlsx", [[1, 1, "Test Player", "first_serve", "Serve", "In"]]
    )
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest_multi_part(
        [part1, part2], date="2026-08-18", opponent="Real Opponent", result="W"
    )
    client = _FakeSuggestionClient()

    record = pipeline.suggest(client, json_path)

    all_points = [p for s in record.sets for p in s.points]
    assert all(p.ai_suggested_point_end_type == "forced_error" for p in all_points)
    assert len(client.messages.calls) == len(all_points)


class _FakeResolutionMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.dumps(
            {
                "point_end_type": "winner",
                "point_won": True,
                "net_approach": False,
                "reasoning": "The human said it was a clean winner.",
            }
        )
        return FakeMessage(content=[FakeTextBlock(text=payload)])


class _FakeResolutionClient:
    def __init__(self):
        self.messages = _FakeResolutionMessages()


class _FlakyResolutionMessages:
    """Fails for the first point, succeeds for the rest."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return FakeMessage(content=[FakeTextBlock(text="not valid json")])
        payload = json.dumps(
            {"point_end_type": "winner", "point_won": True, "net_approach": False, "reasoning": "x"}
        )
        return FakeMessage(content=[FakeTextBlock(text=payload)])


class _FlakyResolutionClient:
    def __init__(self):
        self.messages = _FlakyResolutionMessages()


def test_resolve_parses_review_answers_without_touching_needs_review(
    synthetic_non_pro_xlsx, import_config
):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx, date="2026-08-06", opponent="Alex", result="W"
    )
    record = load_pending(json_path)
    record.sets[0].points[0].review_answer = "She was way out of position - clean winner."
    save_pending(record, import_config.pending_dir)
    client = _FakeResolutionClient()

    resolved = pipeline.resolve(client, json_path)

    answered_point = resolved.sets[0].points[0]
    assert answered_point.resolved_point_end_type == "winner"
    assert answered_point.resolved_point_won is True
    assert answered_point.resolution_reasoning == "The human said it was a clean winner."
    assert answered_point.needs_review is True  # never cleared by resolve()
    assert answered_point.point_end_type != "winner"  # real field untouched

    # The other point had no review_answer - left completely alone.
    untouched_point = resolved.sets[0].points[1]
    assert untouched_point.resolved_point_end_type is None
    assert len(client.messages.calls) == 1


def test_resolve_works_for_direct_parse_points_with_no_shot_context(synthetic_xlsx, import_config):
    # synthetic_xlsx takes the direct-parse path - its points have no
    # source_point_number and no raw shots to enrich the prompt with, but
    # resolve() should still work from the review_answer alone.
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(synthetic_xlsx, date="2026-08-06", opponent="Alex", result="W")
    record = load_pending(json_path)
    flagged_point = next(p for s in record.sets for p in s.points if p.needs_review)
    flagged_point.review_answer = "That was actually a forced error, she was pulled wide."
    save_pending(record, import_config.pending_dir)
    client = _FakeResolutionClient()

    resolved = pipeline.resolve(client, json_path)

    answered = next(p for s in resolved.sets for p in s.points if p.review_answer)
    assert answered.resolved_point_end_type == "winner"
    assert answered.source_point_number is None


def test_resolve_continues_past_one_points_parse_failure(
    synthetic_non_pro_xlsx, import_config, caplog
):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx, date="2026-08-06", opponent="Alex", result="W"
    )
    record = load_pending(json_path)
    for point in record.sets[0].points:
        point.review_answer = "Clean winner."
    save_pending(record, import_config.pending_dir)
    client = _FlakyResolutionClient()

    with caplog.at_level(logging.ERROR):
        resolved = pipeline.resolve(client, json_path)

    all_points = [p for s in resolved.sets for p in s.points]
    succeeded = [p for p in all_points if p.resolved_point_end_type == "winner"]
    failed = [p for p in all_points if p.resolved_point_end_type is None]
    assert len(succeeded) == 1
    assert len(failed) == 1
    assert any("Failed to parse review_answer" in r.message for r in caplog.records)


def test_apply_resolutions_applies_fields_and_clears_needs_review(
    synthetic_non_pro_xlsx, import_config
):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx, date="2026-08-06", opponent="Alex", result="W"
    )
    record = load_pending(json_path)
    for point in record.sets[0].points:
        point.review_answer = "Clean winner, no net play."
    save_pending(record, import_config.pending_dir)
    pipeline.resolve(_FakeResolutionClient(), json_path)

    applied = pipeline.apply_resolutions(json_path)

    all_points = [p for s in applied.sets for p in s.points]
    assert all(p.point_end_type == "winner" for p in all_points)
    assert all(p.point_won is True for p in all_points)
    assert all(p.needs_review is False for p in all_points)

    # Fully resolved now - finalize() should succeed instead of raising.
    match_id = pipeline.finalize(json_path)
    assert match_id == 1


def test_apply_resolutions_leaves_unresolved_points_untouched(
    synthetic_non_pro_xlsx, import_config
):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx, date="2026-08-06", opponent="Alex", result="W"
    )
    record = load_pending(json_path)
    record.sets[0].points[0].review_answer = "Clean winner."
    save_pending(record, import_config.pending_dir)
    pipeline.resolve(_FakeResolutionClient(), json_path)

    applied = pipeline.apply_resolutions(json_path)

    untouched = applied.sets[0].points[1]
    assert untouched.needs_review is True
    assert untouched.resolved_point_end_type is None

    with pytest.raises(UnresolvedReviewError):
        pipeline.finalize(json_path)


class _PerPointMessages:
    """Answers with a classification derived from the prompt itself, so a
    result landing on the wrong point is detectable rather than invisible.

    Records calls under a lock: the point of these tests is that suggest()
    now runs concurrently, so the fake has to be safe to call that way."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls: list[dict] = []
        self.max_concurrent = 0
        self._active = 0

    def create(self, **kwargs):
        with self._lock:
            self.calls.append(kwargs)
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
        # Long enough that a sequential implementation could not reach the
        # concurrency this asserts, short enough not to slow the suite.
        time.sleep(0.02)
        # Echo the shot count back, so each answer is tied to its own point.
        shot_count = kwargs["system"].count('"shot_number"')
        end_type = "winner" if shot_count % 2 else "unforced_error"
        with self._lock:
            self._active -= 1
        payload = json.dumps(
            {
                "point_end_type": end_type,
                "reasoning": f"{shot_count} shots",
                "confidence": "medium",
            }
        )
        return FakeMessage(content=[FakeTextBlock(text=payload)])


class _PerPointClient:
    def __init__(self):
        self.messages = _PerPointMessages()


def test_suggestions_run_concurrently(synthetic_non_pro_xlsx, import_config):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx, date="2026-08-06", opponent="Alex", result="W"
    )
    client = _PerPointClient()

    pipeline.suggest(client, json_path)

    assert client.messages.max_concurrent > 1


def test_each_suggestion_lands_on_the_point_it_was_asked_about(
    synthetic_non_pro_xlsx, import_config
):
    """Concurrency must not shuffle answers between points - results are
    attached through an explicit future -> point map, not completion order."""
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx, date="2026-08-06", opponent="Alex", result="W"
    )
    shots_by_point = pipeline.shots_by_point(load_pending(json_path))
    client = _PerPointClient()

    record = pipeline.suggest(client, json_path)

    annotated = 0
    for set_record in record.sets:
        for point in set_record.points:
            if point.ai_suggestion_reasoning is None:
                continue
            annotated += 1
            expected_shots = len(shots_by_point[point.source_point_number])
            # The fake encoded this point's own shot count in its answer.
            assert point.ai_suggestion_reasoning == f"{expected_shots} shots"
            expected_type = "winner" if expected_shots % 2 else "unforced_error"
            assert point.ai_suggested_point_end_type == expected_type
    assert annotated > 0


def test_suggest_skips_points_the_reviewer_has_already_confirmed(
    synthetic_non_pro_xlsx, import_config
):
    """A settled point is not worth paying to re-ask about."""
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx, date="2026-08-06", opponent="Alex", result="W"
    )
    record = load_pending(json_path)
    flagged = [p for s in record.sets for p in s.points if p.needs_review]
    assert len(flagged) > 1
    flagged[0].needs_review = False
    save_pending(record, import_config.pending_dir)

    client = _FakeSuggestionClient()
    pipeline.suggest(client, json_path)

    assert len(client.messages.calls) == len(flagged) - 1


def test_archive_sources_packs_the_export_and_removes_the_loose_file(
    synthetic_non_pro_xlsx, import_config, tmp_path
):
    pipeline = SwingVisionImportPipeline(import_config)
    # Copy first: archive_sources deletes what it packs, and a caller must
    # only ever point it at files it owns.
    owned = tmp_path / "owned.xlsx"
    owned.write_bytes(synthetic_non_pro_xlsx.read_bytes())
    json_path = pipeline.ingest(owned, date="2026-08-06", opponent="Alex", result="W")

    archive_path = pipeline.archive_sources(json_path)

    assert archive_path is not None and archive_path.exists()
    assert not owned.exists()
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == ["owned.xlsx"]
    record = load_pending(json_path)
    assert record.source_archive == str(archive_path)
    assert record.source_files == ["owned.xlsx"]
    assert record.source_file is None


def test_an_archived_export_is_smaller_than_the_original(
    synthetic_non_pro_xlsx, import_config, tmp_path
):
    """The whole point of the step - a .xlsx is an OOXML zip whose parts are
    barely compressed, so deflating it again is a real saving."""
    pipeline = SwingVisionImportPipeline(import_config)
    owned = tmp_path / "owned.xlsx"
    owned.write_bytes(synthetic_non_pro_xlsx.read_bytes())
    original_size = owned.stat().st_size
    json_path = pipeline.ingest(owned, date="2026-08-06", opponent="Alex", result="W")

    archive_path = pipeline.archive_sources(json_path)

    assert archive_path.stat().st_size < original_size


def test_suggestions_still_work_once_the_export_is_archived(
    synthetic_non_pro_xlsx, import_config, tmp_path
):
    """The regression that matters: suggest() re-parses the original export,
    so archiving must not strand it."""
    pipeline = SwingVisionImportPipeline(import_config)
    owned = tmp_path / "owned.xlsx"
    owned.write_bytes(synthetic_non_pro_xlsx.read_bytes())
    json_path = pipeline.ingest(owned, date="2026-08-06", opponent="Alex", result="W")
    pipeline.archive_sources(json_path)

    record = pipeline.suggest(_FakeSuggestionClient(), json_path)

    annotated = [p for s in record.sets for p in s.points if p.ai_suggested_point_end_type]
    assert annotated, "no suggestions came back after archiving"


def test_archive_sources_is_idempotent(synthetic_non_pro_xlsx, import_config, tmp_path):
    pipeline = SwingVisionImportPipeline(import_config)
    owned = tmp_path / "owned.xlsx"
    owned.write_bytes(synthetic_non_pro_xlsx.read_bytes())
    json_path = pipeline.ingest(owned, date="2026-08-06", opponent="Alex", result="W")

    first = pipeline.archive_sources(json_path)
    second = pipeline.archive_sources(json_path)

    assert first == second
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["owned.xlsx"]  # not packed into itself


def test_archiving_a_multi_part_match_keeps_play_order(
    synthetic_non_pro_xlsx, import_config, tmp_path
):
    pipeline = SwingVisionImportPipeline(import_config)
    parts = []
    for name in ("set1.xlsx", "set2.xlsx"):
        part = tmp_path / name
        part.write_bytes(synthetic_non_pro_xlsx.read_bytes())
        parts.append(part)
    json_path = pipeline.ingest_multi_part(parts, date="2026-08-06", opponent="Alex", result="W")

    pipeline.archive_sources(json_path)

    record = load_pending(json_path)
    assert record.source_files == ["set1.xlsx", "set2.xlsx"]
    # And the merged re-parse still resolves every point's provenance.
    assert pipeline.shots_by_point(record)


# ---------------------------------------------------------------------------
# Footage verification, end to end
# ---------------------------------------------------------------------------


@pytest.fixture
def timed_xlsx(tmp_path):
    """A two-point non-Pro export whose shots carry footage timing."""
    workbook = Workbook()
    sets_sheet = workbook.active
    sets_sheet.title = "Sets"
    sets_sheet.append(["Set", "Host Score", "Guest Score", "Set Winner"])
    sets_sheet.append([1, 6, 4, "host"])
    workbook.create_sheet("Games").append(["Game", "Set", "Server", "Game Winner"])
    workbook.create_sheet("Points").append(
        ["Point", "Game", "Set", "Match Server", "Point Winner", "Detail"]
    )
    shot_rows = [
        # point, shot, player, type, stroke, result, video_time, hit_side
        [1, 1, "Test Player", "first_serve", "Serve", "In", 30.0, "near"],
        [1, 2, "Test Opponent", "first_return", "Backhand", "Net", 31.4, "far"],
        [2, 1, "Test Player", "first_serve", "Serve", "In", 60.0, "near"],
        [2, 2, "Test Opponent", "first_return", "Forehand", "In", 61.2, "far"],
        [2, 3, "Test Player", "in_play", "Forehand", "Out", 62.6, "near"],
    ]
    add_settings_and_shots_sheets(workbook, shot_rows=shot_rows)
    path = tmp_path / "timed_export.xlsx"
    workbook.save(path)
    return path


class _StubFootageClient:
    """Answers every point with a scripted verdict, keyed by call order."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.calls = 0
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.calls += 1
                payload = outer._verdicts.pop(0) if outer._verdicts else {}
                body = {
                    "footage_usable": True,
                    "server_seen": "tracked",
                    "winner": "tracked",
                    "point_end_type": "winner",
                    "tracked_at_net": "unclear",
                    "confidence": "high",
                    "reasoning": "scripted",
                }
                body.update(payload)
                message = FakeMessage(content=[FakeTextBlock(text=json.dumps(body))])
                message.usage = type("U", (), {"input_tokens": 1000, "output_tokens": 100})()
                message.stop_reason = "end_turn"
                return message

        class _Beta:
            messages = _Messages()

        self.beta = _Beta()


def _no_frames(_video, _window):
    """Frame extraction stand-in: the video itself isn't what's under test."""
    return [footage.Frame("finish", 0.0, b"\xff\xd8jpeg")]


def _stage_timed(pipeline, timed_xlsx):
    return pipeline.ingest(timed_xlsx, date="2026-08-06", opponent="Alex", result="W")


def test_footage_agreement_clears_the_flag_and_records_who_cleared_it(
    timed_xlsx, import_config, tmp_path, monkeypatch
):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = _stage_timed(pipeline, timed_xlsx)
    recorded = {
        (p.game_number, p.point_number): (p.point_won, p.point_end_type)
        for s in load_pending(json_path).sets
        for p in s.points
    }
    # Script each verdict to match what reconstruction recorded.
    verdicts = [
        {"winner": "tracked" if won else "opponent", "point_end_type": end_type}
        for won, end_type in recorded.values()
    ]
    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 10_000.0)

    summary = pipeline.verify_footage(
        _StubFootageClient(verdicts),
        json_path,
        [tmp_path / "video.mp4"],
        frame_source=_no_frames,
    )

    assert summary.agreed == len(recorded)
    assert summary.disagreed == 0
    record = load_pending(json_path)
    for set_record in record.sets:
        for point in set_record.points:
            assert point.needs_review is False
            assert point.confirmed_by == "footage"
            assert point.footage_check["verdict"] == "agrees"


def test_footage_disagreement_leaves_the_point_for_a_human(
    timed_xlsx, import_config, tmp_path, monkeypatch
):
    """A single source contradicting the record is never enough to change
    it - the verdict is attached as context and the flag stays up."""
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = _stage_timed(pipeline, timed_xlsx)
    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 10_000.0)

    summary = pipeline.verify_footage(
        _StubFootageClient(
            [{"winner": "opponent", "point_end_type": "forced_error", "reasoning": "passed"}] * 5
        ),
        json_path,
        [tmp_path / "video.mp4"],
        frame_source=_no_frames,
    )

    assert summary.agreed == 0
    record = load_pending(json_path)
    flagged = [p for s in record.sets for p in s.points if p.needs_review]
    assert flagged, "disagreement must not clear the flag"
    check = flagged[0].footage_check
    assert check["verdict"] == "disagrees"
    assert check["point_end_type"] == "forced_error"
    assert flagged[0].confirmed_by is None
    # And nothing the reconstruction recorded was overwritten.
    assert flagged[0].point_end_type != "forced_error"


def test_an_unclear_verdict_leaves_the_point_alone(
    timed_xlsx, import_config, tmp_path, monkeypatch
):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = _stage_timed(pipeline, timed_xlsx)
    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 10_000.0)

    summary = pipeline.verify_footage(
        _StubFootageClient([{"confidence": "low"}] * 5),
        json_path,
        [tmp_path / "video.mp4"],
        frame_source=_no_frames,
    )

    assert summary.inconclusive > 0 and summary.agreed == 0
    record = load_pending(json_path)
    assert all(p.needs_review for s in record.sets for p in s.points)


def test_condensed_footage_is_refused_before_any_call_is_made(
    timed_xlsx, import_config, tmp_path, monkeypatch
):
    """The guard has to fire before spending: frames from the wrong rally
    would produce confident nonsense."""
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = _stage_timed(pipeline, timed_xlsx)
    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 12.0)  # far too short
    client = _StubFootageClient([])

    with pytest.raises(footage.FootageAlignmentError):
        pipeline.verify_footage(
            client, json_path, [tmp_path / "video.mp4"], frame_source=_no_frames
        )

    assert client.calls == 0


def test_a_second_run_does_not_pay_to_recheck_the_same_points(
    timed_xlsx, import_config, tmp_path, monkeypatch
):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = _stage_timed(pipeline, timed_xlsx)
    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 10_000.0)
    args = dict(frame_source=_no_frames)

    first = pipeline.verify_footage(
        _StubFootageClient([{"confidence": "low"}] * 5),
        json_path,
        [tmp_path / "video.mp4"],
        **args,
    )
    second_client = _StubFootageClient([{"confidence": "low"}] * 5)
    second = pipeline.verify_footage(second_client, json_path, [tmp_path / "video.mp4"], **args)

    assert first.checked > 0
    assert second.checked == 0 and second_client.calls == 0


def test_the_run_reports_what_it_actually_cost(timed_xlsx, import_config, tmp_path, monkeypatch):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = _stage_timed(pipeline, timed_xlsx)
    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 10_000.0)

    summary = pipeline.verify_footage(
        _StubFootageClient([]),
        json_path,
        [tmp_path / "video.mp4"],
        frame_source=_no_frames,
    )

    assert summary.input_tokens == 1000 * summary.checked
    assert summary.usd > 0


def test_widespread_server_mismatches_are_called_out(
    timed_xlsx, import_config, tmp_path, monkeypatch
):
    """Every point reporting a different server than the export means the
    footage probably isn't this match - say so rather than quietly
    returning zero agreements."""
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = _stage_timed(pipeline, timed_xlsx)
    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 10_000.0)

    summary = pipeline.verify_footage(
        _StubFootageClient([{"server_seen": "opponent"}] * 5),
        json_path,
        [tmp_path / "video.mp4"],
        frame_source=_no_frames,
    )

    assert summary.server_mismatches == summary.checked
    assert any("may not line up" in note for note in summary.notes)


def test_progress_is_reported_as_points_complete(timed_xlsx, import_config, tmp_path, monkeypatch):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = _stage_timed(pipeline, timed_xlsx)
    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 10_000.0)
    seen = []

    pipeline.verify_footage(
        _StubFootageClient([]),
        json_path,
        [tmp_path / "video.mp4"],
        frame_source=_no_frames,
        progress=lambda s: seen.append(s.checked),
    )

    assert seen == sorted(seen) and seen[-1] > 0


def test_the_summary_reports_points_played_as_the_denominator(
    timed_xlsx, import_config, tmp_path, monkeypatch
):
    """Every coverage figure is a fraction of the points actually played.
    Showing a verdict count without that denominator - "20 confirmed" -
    reads as coverage. "20 confirmed of 100 points played" does not."""
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = _stage_timed(pipeline, timed_xlsx)
    record = load_pending(json_path)
    expected_played = sum(
        1 for s in record.sets for p in s.points if p.source_point_number is not None
    )
    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 10_000.0)

    summary = pipeline.verify_footage(
        _StubFootageClient([]),
        json_path,
        [tmp_path / "video.mp4"],
        frame_source=_no_frames,
    )

    assert expected_played > 0, "fixture must produce points"
    assert summary.points_played == expected_played
    # Even a run with no targets reports what Y is a fraction of.
    empty_summary = pipeline.verify_footage(
        _StubFootageClient([]),
        json_path,
        [tmp_path / "video.mp4"],
        frame_source=_no_frames,
    )
    assert empty_summary.points_played == expected_played


def test_the_summary_tracks_points_not_in_the_footage_as_their_own_bucket(
    timed_xlsx, import_config, tmp_path, monkeypatch
):
    """A point whose rally the footage does not contain is NOT fabricated
    by judging a nearby one - it goes in its own bucket and stays flagged
    for a human. The summary must say how many landed there."""
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = _stage_timed(pipeline, timed_xlsx)
    record = load_pending(json_path)

    # Simulate the retrieval miss by applying the helper directly, then
    # read the record back and confirm the fields survive the round trip.
    point = next(p for s in record.sets for p in s.points if p.needs_review)
    pipeline._apply_not_in_footage(point, "rally not present in condensed export")
    from swingvision_import.review import save_pending

    save_pending(record, import_config.pending_dir)

    reloaded = load_pending(json_path)
    touched = next(p for s in reloaded.sets for p in s.points if p.footage_check)
    assert touched.needs_review is True, "not_in_footage must never clear the flag"
    assert touched.footage_check["verdict"] == "not_in_footage"
    assert "condensed" in touched.footage_check["reasoning"]


# ---------------------------------------------------------------------------
# Condensed-rallies mode: retrieval + honest "not in footage" reporting
# ---------------------------------------------------------------------------


class _RetrievalClient:
    """Fake that answers both the retrieval and the judge call.

    The two calls run through the same client.beta.messages.create path -
    this fake tells them apart by the schema in the message content, so
    it can return a LocationVerdict JSON for the retrieval call and a
    FootageVerdict JSON for the judge call."""

    def __init__(self, found: bool = True, offset: float = 5.0, judge_outcome: str = "winner"):
        self.found = found
        self.offset = offset
        self.judge_outcome = judge_outcome
        self.messages = self
        self.beta = self
        self.calls: list[dict] = []

    class _Usage:
        input_tokens = 500
        output_tokens = 200

    class _Response:
        def __init__(self, text: str):
            from tests.ai.conftest import FakeMessage, FakeTextBlock

            self._msg = FakeMessage(content=[FakeTextBlock(text=text)])
            self._msg.usage = _RetrievalClient._Usage()
            self._msg.stop_reason = "end_turn"

        def __getattr__(self, name):
            return getattr(self._msg, name)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text_blocks = [b for b in kwargs["messages"][0]["content"] if b["type"] == "text"]
        prompt = " ".join(b["text"] for b in text_blocks)
        if "rally_present" in prompt:
            import json as _json

            return _RetrievalClient._Response(
                _json.dumps(
                    {
                        "rally_present": self.found,
                        "offset_seconds": self.offset,
                        "confidence": "high" if self.found else "low",
                        "reasoning": "fingerprint match" if self.found else "no match",
                    }
                )
            )
        # judge response
        import json as _json

        return _RetrievalClient._Response(
            _json.dumps(
                {
                    "footage_usable": True,
                    "server_seen": "tracked",
                    "winner": "tracked",
                    "point_end_type": self.judge_outcome,
                    "tracked_at_net": "unclear",
                    "confidence": "high",
                    "reasoning": "judged after retrieval",
                }
            )
        )


def test_condensed_mode_skips_the_alignment_guard(timed_xlsx, import_config, tmp_path, monkeypatch):
    """The guard refuses any video shorter than the shots. In condensed
    mode the video IS too short - that is the whole point. Running it
    at all requires the caller to opt in explicitly via config.source."""
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = _stage_timed(pipeline, timed_xlsx)
    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 5.0)  # way too short
    monkeypatch.setattr(footage, "extract_frames", lambda *a, **k: [b"\xff\xd8fake"])

    config = footage.FootageConfig(source="condensed_rallies")

    # Nothing to assert on the retrieval path yet - just that reaching
    # this point does not raise FootageAlignmentError, which is the gate
    # the condensed mode is built to bypass honestly.
    pipeline.verify_footage(
        _RetrievalClient(found=False),
        json_path,
        [tmp_path / "video.mp4"],
        footage_config=config,
        frame_source=_no_frames,
    )


def test_condensed_mode_not_found_marks_not_in_footage_and_leaves_the_flag(
    timed_xlsx, import_config, tmp_path, monkeypatch
):
    """The honesty rule: when retrieval cannot find a rally, the point
    is reported as "not in this footage" and left flagged for a human.
    The judge call never runs, and nothing fabricates an outcome."""
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = _stage_timed(pipeline, timed_xlsx)
    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 5.0)
    monkeypatch.setattr(footage, "extract_frames", lambda *a, **k: [b"\xff\xd8fake"])
    config = footage.FootageConfig(source="condensed_rallies")

    client = _RetrievalClient(found=False)
    summary = pipeline.verify_footage(
        client,
        json_path,
        [tmp_path / "video.mp4"],
        footage_config=config,
        frame_source=_no_frames,
    )

    assert summary.not_in_footage > 0
    assert summary.agreed == 0
    assert summary.checked == 0, "no judge call must have run"
    record = load_pending(json_path)
    flagged = [p for s in record.sets for p in s.points if p.footage_check]
    assert flagged, "at least one point must carry the not_in_footage context"
    assert all(p.needs_review for p in flagged), "not_in_footage never clears the flag"
    assert flagged[0].footage_check["verdict"] == "not_in_footage"


def test_condensed_mode_found_then_judged_can_still_auto_confirm(
    timed_xlsx, import_config, tmp_path, monkeypatch
):
    """When retrieval finds a rally, the judge call runs on frames seeked
    to the located offset. Agreement there still auto-confirms, exactly
    like full-mode - the condensed path does not loosen the two-source bar."""
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = _stage_timed(pipeline, timed_xlsx)
    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 100.0)
    monkeypatch.setattr(footage, "extract_frames", lambda *a, **k: [b"\xff\xd8fake"])
    config = footage.FootageConfig(source="condensed_rallies")

    # The stub xlsx's recorded end_type is "winner" for every point; the
    # judge fake returns the same, so compare() will auto-confirm once the
    # server/at-net consistency checks pass.
    client = _RetrievalClient(found=True, judge_outcome="winner")
    summary = pipeline.verify_footage(
        client,
        json_path,
        [tmp_path / "video.mp4"],
        footage_config=config,
        frame_source=_no_frames,
    )

    # The retrieval path produces location + judge calls; each retrieval
    # miss would have contributed to not_in_footage instead.
    assert summary.checked > 0, "at least one point must reach the judge"
    assert summary.not_in_footage == 0


def test_ingest_carries_the_rally_count_and_notes_the_coverage_ceiling(timed_xlsx, import_config):
    pipeline = SwingVisionImportPipeline(import_config)
    baseline = load_pending(_stage_timed(pipeline, timed_xlsx))
    points = sum(len(s.points) for s in baseline.sets)
    assert points > 1, "fixture must produce more than one point"
    assert not any("rallies" in note for note in baseline.import_notes)

    json_path = pipeline.ingest(
        timed_xlsx,
        date="2026-08-06",
        opponent="Alex",
        result="W",
        footage_rally_count=points - 1,
    )
    record = load_pending(json_path)

    assert record.footage_rally_count == points - 1
    assert any(f"at most {points - 1} of {points} points" in note for note in record.import_notes)


def test_the_summary_carries_the_rally_count_beside_the_denominator(
    timed_xlsx, import_config, tmp_path, monkeypatch
):
    """ "Of N points played" says what was played; the rally count says how
    much of that the footage can show at all. A summary that drops it
    leaves the reviewer to rediscover the ceiling from the misses."""
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(
        timed_xlsx, date="2026-08-06", opponent="Alex", result="W", footage_rally_count=2
    )
    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 10_000.0)

    summary = pipeline.verify_footage(
        _StubFootageClient([]), json_path, [tmp_path / "video.mp4"], frame_source=_no_frames
    )

    assert summary.footage_rally_count == 2
    assert summary.points_played > 0


def test_footage_frames_are_taken_on_the_videos_clock_not_the_sheets(
    timed_xlsx, import_config, tmp_path, monkeypatch
):
    """The real bug: the video ran 2s behind the sheet, so frames "after the
    last shot" stopped before the rally did."""
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = _stage_timed(pipeline, timed_xlsx)
    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 10_000.0)

    def last_shots(offset):
        seen = []

        def frames(_video, window):
            seen.append(window.last_shot)
            return _no_frames(_video, window)

        pipeline.verify_footage(
            _StubFootageClient([{"confidence": "low"}] * 50),
            json_path,
            [tmp_path / "video.mp4"],
            frame_source=frames,
            clock_offset=lambda _video, _times: offset,
        )
        for set_record in (record := load_pending(json_path)).sets:
            for point in set_record.points:
                point.footage_check = None
        save_pending(record, import_config.pending_dir)
        return sorted(seen)

    plain = last_shots(footage.ClockOffset(seconds=0.1, strength=12.0, shots=100))
    shifted = last_shots(footage.ClockOffset(seconds=2.0, strength=12.0, shots=100))

    assert plain
    assert shifted == [pytest.approx(t + 2.0) for t in plain]


def test_an_unconfirmed_clock_is_reported_and_the_sheets_times_are_kept(
    timed_xlsx, import_config, tmp_path, monkeypatch
):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = _stage_timed(pipeline, timed_xlsx)
    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 10_000.0)

    summary = pipeline.verify_footage(
        _StubFootageClient([]),
        json_path,
        [tmp_path / "video.mp4"],
        frame_source=_no_frames,
        clock_offset=lambda _video, _times: footage.ClockOffset(5.0, strength=2.0, shots=100),
    )

    assert any("could not confirm from the audio" in note for note in summary.notes)


def test_the_clock_is_measured_from_the_real_video_by_default(
    timed_xlsx, import_config, tmp_path, monkeypatch
):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = _stage_timed(pipeline, timed_xlsx)
    asked = []
    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 10_000.0)
    monkeypatch.setattr(
        footage, "measure_clock_offset", lambda video, times, exe: asked.append(video.name)
    )
    monkeypatch.setattr(footage, "frames_for_point", lambda *a, **k: _no_frames(None, None))

    pipeline.verify_footage(_StubFootageClient([]), json_path, [tmp_path / "video.mp4"])

    assert asked == ["video.mp4"]
