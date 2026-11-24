from __future__ import annotations

from swingvision_import.quality_check import (
    check_footage_rally_count,
    check_score_against_sets_sheet,
    check_serve_order,
    check_tracked_identity,
)
from swingvision_import.raw import RawSetRow, RawSettings
from swingvision_import.records import PointRecord, SetRecord


def _set_record(
    set_number: int, games_won: int, games_lost: int, first_is_serving: bool
) -> SetRecord:
    return SetRecord(
        set_number=set_number,
        games_won=games_won,
        games_lost=games_lost,
        points=[
            PointRecord(
                game_number=1,
                point_number=1,
                is_serving=first_is_serving,
                point_won=True,
                point_end_type="ace",
                needs_review=True,
            )
        ],
    )


def test_check_score_against_sets_sheet_matching_score_produces_no_note():
    sets = [_set_record(1, 6, 4, True)]
    raw_sets = [RawSetRow(set_number=1, winner="host", games_won=6, games_lost=4)]

    assert check_score_against_sets_sheet(sets, raw_sets) == []


def test_check_score_against_sets_sheet_mismatch_produces_a_note():
    sets = [_set_record(2, 5, 3, True)]
    raw_sets = [RawSetRow(set_number=2, winner="host", games_won=6, games_lost=4)]

    notes = check_score_against_sets_sheet(sets, raw_sets)

    assert len(notes) == 1
    assert "5-3" in notes[0]
    assert "6-4" in notes[0]


def test_check_score_against_sets_sheet_skips_sets_with_no_matching_raw_row():
    sets = [_set_record(3, 6, 0, True)]

    assert check_score_against_sets_sheet(sets, raw_sets=[]) == []


def test_check_serve_order_returns_empty_when_nothing_supplied():
    sets = [_set_record(1, 6, 4, True)]

    assert check_serve_order(sets, first_server_by_set=None) == []


def test_check_serve_order_matching_ground_truth_produces_no_note():
    sets = [_set_record(1, 6, 4, True)]

    assert check_serve_order(sets, {1: "me"}) == []


def test_check_serve_order_mismatch_produces_a_high_severity_note():
    sets = [_set_record(1, 6, 4, True)]

    notes = check_serve_order(sets, {1: "opponent"})

    assert len(notes) == 1
    assert "reversed" in notes[0]


def test_check_serve_order_skips_sets_with_no_ground_truth_supplied():
    sets = [_set_record(1, 6, 4, True), _set_record(2, 6, 3, False)]

    # Only set 1 has ground truth supplied; set 2 is silently skipped.
    assert check_serve_order(sets, {1: "me"}) == []


def test_check_tracked_identity_returns_empty_when_nothing_supplied():
    settings = RawSettings(host_name="Rishi Gupta", guest_name="Opponent")

    assert check_tracked_identity(settings, claimed_identity=None) == []
    assert check_tracked_identity(None, claimed_identity="Rishi Gupta") == []


def test_check_tracked_identity_matches_case_and_whitespace_insensitively():
    settings = RawSettings(host_name="Rishi Gupta", guest_name="Opponent")

    assert check_tracked_identity(settings, "  rishi gupta  ") == []


def test_check_tracked_identity_mismatch_produces_a_note():
    settings = RawSettings(host_name="Rishi Gupta", guest_name="Opponent")

    notes = check_tracked_identity(settings, "Someone Else")

    assert len(notes) == 1
    assert "Someone Else" in notes[0]
    assert "Rishi Gupta" in notes[0]


def _sets_with_points(count: int) -> list[SetRecord]:
    record = _set_record(1, 6, 4, True)
    record.points = [record.points[0]] * count
    return [record]


def test_check_footage_rally_count_says_nothing_when_no_count_was_given():
    assert check_footage_rally_count(_sets_with_points(10), None) == []


def test_check_footage_rally_count_ignores_an_unusable_count():
    """A hand-edited pending JSON can hold anything; a non-positive count
    is treated as not supplied rather than producing a nonsense ceiling."""
    assert check_footage_rally_count(_sets_with_points(10), 0) == []
    assert check_footage_rally_count(_sets_with_points(10), -3) == []


def test_check_footage_rally_count_says_nothing_for_a_match_with_no_points():
    assert check_footage_rally_count([], 12) == []


def test_check_footage_rally_count_reports_the_coverage_ceiling():
    """Fewer rallies than points is the expected shape of a rallies-only
    export, so it is reported as a ceiling, not as a fault."""
    (note,) = check_footage_rally_count(_sets_with_points(10), 7)

    assert "at most 7 of 10 points (70%)" in note
    assert "At least 3 point(s) in this import have no rally" in note
    # The cause is not established, so the note must not name one.
    assert "has not been established" in note


def test_check_footage_rally_count_flags_more_rallies_than_points():
    """Every rally is a point, so this cannot be the export format - the
    import is short, or the count includes non-match rallies."""
    (note,) = check_footage_rally_count(_sets_with_points(10), 14)

    assert "14 rallies" in note and "only 10 points" in note
    assert "at least 4 point(s) are missing" in note


def test_check_footage_rally_count_reports_an_exact_match():
    (note,) = check_footage_rally_count(_sets_with_points(10), 10)

    assert "every point has a rally" in note
