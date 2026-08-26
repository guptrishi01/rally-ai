from __future__ import annotations

from pathlib import Path

import pytest

from swingvision_import.records import MatchRecord, PointRecord, SetRecord
from swingvision_import.review import (
    ConfirmationError,
    confirm_point,
    find_pending_path,
    load_pending,
    save_pending,
    unresolved_flags,
)


def _sample_record(needs_review: bool) -> MatchRecord:
    point = PointRecord(
        game_number=1,
        point_number=1,
        is_serving=True,
        point_won=True,
        point_end_type="winner",
        needs_review=needs_review,
    )
    return MatchRecord(
        date="2026-08-06",
        opponent="Alex",
        result="W",
        sets=[SetRecord(set_number=1, games_won=6, games_lost=4, points=[point])],
    )


def test_unresolved_flags_reports_pending_points():
    flags = unresolved_flags(_sample_record(needs_review=True))
    assert len(flags) == 1
    assert "point 1" in flags[0]


def test_unresolved_flags_empty_once_confirmed():
    assert unresolved_flags(_sample_record(needs_review=False)) == []


def test_json_round_trip(tmp_path: Path):
    record = _sample_record(needs_review=False)
    path = save_pending(record, tmp_path)
    assert load_pending(path) == record


def test_unresolved_flags_reports_every_flagged_point_across_multiple_sets():
    flagged_a = PointRecord(1, 1, True, True, "winner", needs_review=True)
    clean = PointRecord(1, 2, True, True, "ace", needs_review=False)
    flagged_b = PointRecord(2, 1, False, False, "unforced_error", needs_review=True)
    record = MatchRecord(
        date="2026-08-06",
        opponent="Alex",
        result="W",
        sets=[
            SetRecord(set_number=1, games_won=6, games_lost=4, points=[flagged_a, clean]),
            SetRecord(set_number=2, games_won=6, games_lost=3, points=[flagged_b]),
        ],
    )

    flags = unresolved_flags(record)

    assert len(flags) == 2
    assert any("set 1" in f and "point 1" in f for f in flags)
    assert any("set 2" in f and "point 1" in f for f in flags)


def test_unresolved_flags_includes_the_ai_suggestion_when_present():
    point = PointRecord(
        game_number=1,
        point_number=1,
        is_serving=True,
        point_won=False,
        point_end_type="unforced_error",
        needs_review=True,
        ai_suggested_point_end_type="forced_error",
        ai_suggestion_reasoning="Wide angle return left no time to recover.",
    )
    record = MatchRecord(
        date="2026-08-06",
        opponent="Alex",
        result="W",
        sets=[SetRecord(set_number=1, games_won=6, games_lost=4, points=[point])],
    )

    flags = unresolved_flags(record)

    assert len(flags) == 1
    assert "Claude suggests 'forced_error'" in flags[0]
    assert "Wide angle return" in flags[0]


def test_unresolved_flags_omits_suggestion_text_when_none_was_generated():
    flags = unresolved_flags(_sample_record(needs_review=True))

    assert "Claude suggests" not in flags[0]


def test_save_pending_sanitizes_filesystem_unsafe_characters_in_opponent_name(tmp_path: Path):
    record = _sample_record(needs_review=False)
    record.opponent = 'Team A/B: "The Rematch"?'

    path = save_pending(record, tmp_path)

    assert path.exists()
    assert path.parent == tmp_path
    for unsafe_char in '<>:"/\\|?*':
        assert unsafe_char not in path.name
    assert load_pending(path).opponent == record.opponent


def test_save_pending_falls_back_to_a_placeholder_for_an_all_unsafe_opponent_name(tmp_path: Path):
    record = _sample_record(needs_review=False)
    record.opponent = "///"

    path = save_pending(record, tmp_path)

    assert path.exists()
    assert "unknown" in path.name


def test_save_pending_twice_overwrites_the_same_file_rather_than_duplicating(tmp_path: Path):
    record = _sample_record(needs_review=False)
    first_path = save_pending(record, tmp_path)

    record.pros = "Updated after re-review"
    second_path = save_pending(record, tmp_path)

    assert first_path == second_path
    assert list(tmp_path.glob("*.json")) == [first_path]
    assert load_pending(second_path).pros == "Updated after re-review"


def test_find_pending_path_locates_a_previously_saved_record(tmp_path: Path):
    record = _sample_record(needs_review=False)
    saved_path = save_pending(record, tmp_path)

    found = find_pending_path(record.date, record.opponent, tmp_path)

    assert found == saved_path


def test_find_pending_path_returns_none_when_nothing_was_ever_staged(tmp_path: Path):
    assert find_pending_path("2026-08-06", "Alex", tmp_path) is None


def test_confirm_point_sets_fields_and_clears_needs_review():
    record = _sample_record(needs_review=True)

    confirm_point(
        record,
        set_number=1,
        game_number=1,
        point_number=1,
        point_end_type="ace",
        point_won=True,
        net_approach=True,
    )

    point = record.sets[0].points[0]
    assert point.point_end_type == "ace"
    assert point.point_won is True
    assert point.net_approach is True
    assert point.needs_review is False


def test_confirm_point_rejects_an_unknown_point_end_type():
    record = _sample_record(needs_review=True)

    with pytest.raises(ConfirmationError, match="not a valid point_end_type"):
        confirm_point(
            record,
            set_number=1,
            game_number=1,
            point_number=1,
            point_end_type="let",
            point_won=True,
            net_approach=False,
        )


def test_confirm_point_rejects_a_winning_end_type_with_point_won_false():
    record = _sample_record(needs_review=True)

    with pytest.raises(ConfirmationError, match="requires point_won=true"):
        confirm_point(
            record,
            set_number=1,
            game_number=1,
            point_number=1,
            point_end_type="ace",
            point_won=False,
            net_approach=False,
        )


def test_confirm_point_rejects_a_losing_end_type_with_point_won_true():
    record = _sample_record(needs_review=True)

    with pytest.raises(ConfirmationError, match="requires point_won=false"):
        confirm_point(
            record,
            set_number=1,
            game_number=1,
            point_number=1,
            point_end_type="double_fault",
            point_won=True,
            net_approach=False,
        )


def test_confirm_point_raises_when_no_point_matches():
    record = _sample_record(needs_review=True)

    with pytest.raises(ConfirmationError, match="no point at set 2 game 1 point 1"):
        confirm_point(
            record,
            set_number=2,
            game_number=1,
            point_number=1,
            point_end_type="ace",
            point_won=True,
            net_approach=False,
        )
