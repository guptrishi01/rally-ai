from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from stats.models import (
    ClutchStats,
    MatchStats,
    NetStats,
    PointOutcomeStats,
    PointRow,
    ReceivingStats,
    SelfAssessment,
    ServingStats,
)
from stats.queries import (
    all_match_ids,
    all_match_stats,
    career_stats_from_matches,
    clutch_stats_from_points,
    match_stats,
    net_stats_from_points,
    point_outcome_stats_from_points,
    receiving_stats_from_points,
    serving_stats_from_points,
    update_journal_fields,
)
from swingvision_import.load import finalize_and_load
from swingvision_import.records import MatchRecord, PointRecord, SetRecord

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "data" / "schema.sql"


def _point(
    is_serving: bool,
    point_won: bool,
    point_end_type: str,
    *,
    first_serve_in: bool | None = True,
    second_serve_in: bool | None = None,
    net_approach: bool = False,
) -> PointRow:
    return PointRow(
        set_number=1,
        game_number=1,
        point_number=1,
        is_serving=is_serving,
        first_serve_in=first_serve_in,
        second_serve_in=second_serve_in,
        point_won=point_won,
        point_end_type=point_end_type,
        net_approach=net_approach,
        is_tiebreak_game=False,
    )


def test_serving_stats_from_points_counts_serves_aces_and_double_faults():
    points = [
        _point(True, True, "ace"),
        _point(True, False, "double_fault", first_serve_in=False, second_serve_in=False),
        _point(True, True, "winner", first_serve_in=False, second_serve_in=True),
        _point(False, True, "return_winner"),  # receiving point, shouldn't count as a serve
    ]

    stats = serving_stats_from_points(points)

    assert stats.first_serves_total == 3
    assert stats.first_serves_in == 1
    assert stats.second_serves_total == 2
    assert stats.second_serves_in == 1
    assert stats.aces == 1
    assert stats.double_faults == 1


def test_point_outcome_stats_from_points_computes_winner_to_ue_ratio():
    points = [
        _point(True, True, "winner"),
        _point(True, True, "winner"),
        _point(False, False, "unforced_error"),
    ]

    stats = point_outcome_stats_from_points(points)

    assert stats.winners == 2
    assert stats.unforced_errors == 1
    assert stats.winner_to_ue_ratio == 2.0


def test_point_outcome_stats_ratio_is_zero_not_a_crash_when_no_unforced_errors():
    points = [_point(True, True, "winner")]

    stats = point_outcome_stats_from_points(points)

    assert stats.unforced_errors == 0
    assert stats.winner_to_ue_ratio == 0.0


def test_net_stats_from_points_only_counts_flagged_approaches():
    points = [
        _point(True, True, "winner", net_approach=True),
        _point(True, False, "unforced_error", net_approach=True),
        _point(True, True, "ace"),  # no net approach at all
    ]

    stats = net_stats_from_points(points)

    assert stats.net_approaches == 2
    assert stats.net_points_won == 1
    assert stats.net_success_pct == 50.0


def test_receiving_and_clutch_stats_use_game_score_reconstruction():
    # Receiving, break point earned and converted at the 4th point (see
    # test_scoring.py for the detailed math); reused here to confirm
    # queries.py wires scoring.reconstruct in correctly end to end.
    points = [
        PointRow(
            1, 1, i + 1, False, True, None, won,
            "winner" if won else "forced_error", False, False,
        )
        for i, won in enumerate([True, True, True, True])
    ]

    receiving = receiving_stats_from_points(points)
    clutch = clutch_stats_from_points(points)

    assert receiving.break_points_total == 1
    assert receiving.break_points_converted == 1
    assert clutch.deuce_points_played == 0


def _seed_match(tmp_path: Path) -> tuple[sqlite3.Connection, int]:
    """Loads a small, fully-reviewed match via the SwingVision pipeline's own
    finalize_and_load, so this test exercises the real schema/join instead of
    a hand-rolled test DB."""
    connection = sqlite3.connect(tmp_path / "test.db")
    connection.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))

    record = MatchRecord(
        date="2026-08-06",
        opponent="Alex",
        result="W",
        energy_rating=4,
        mental_rating=3,
        pros="Served big",
        cons="Slow starts",
        sets=[
            SetRecord(
                set_number=1,
                games_won=6,
                games_lost=4,
                points=[
                    PointRecord(1, 1, True, True, "ace"),
                    PointRecord(1, 2, True, False, "double_fault", first_serve_in=False),
                    PointRecord(2, 1, False, True, "return_winner"),
                    PointRecord(2, 2, False, False, "unforced_error"),
                ],
            ),
        ],
    )
    match_id = finalize_and_load(connection, record)
    return connection, match_id


def test_match_stats_end_to_end_against_a_seeded_database(tmp_path: Path):
    connection, match_id = _seed_match(tmp_path)

    stats = match_stats(connection, match_id)

    assert stats.opponent == "Alex"
    assert stats.self_assessment.energy_rating == 4
    assert stats.self_assessment.pros == "Served big"
    assert stats.serving.aces == 1
    assert stats.serving.double_faults == 1
    assert stats.point_outcomes.total_points_played == 4


def test_tiebreak_games_excluded_from_service_and_return_game_counts():
    points = [
        # A completed, non-tiebreak service game the player holds.
        PointRow(1, 1, 1, True, True, None, True, "ace", False, False),
        PointRow(1, 1, 2, True, True, None, True, "ace", False, False),
        PointRow(1, 1, 3, True, True, None, True, "ace", False, False),
        PointRow(1, 1, 4, True, True, None, True, "ace", False, False),
        # A tiebreak game — must not count as a service or return game at all.
        PointRow(1, 2, 1, True, True, None, True, "ace", False, True),
        PointRow(1, 2, 2, False, True, None, False, "unforced_error", False, True),
    ]

    serving = serving_stats_from_points(points)
    receiving = receiving_stats_from_points(points)

    assert serving.service_games_total == 1
    assert serving.service_games_won == 1
    assert receiving.return_games_total == 0


def test_match_stats_raises_for_an_unknown_match_id(tmp_path: Path):
    connection, _ = _seed_match(tmp_path)

    with pytest.raises(ValueError, match="999"):
        match_stats(connection, match_id=999)


def test_all_match_ids_returns_empty_list_for_a_fresh_database(tmp_path: Path):
    connection = sqlite3.connect(tmp_path / "empty.db")
    connection.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))

    assert all_match_ids(connection) == []


def test_all_match_ids_orders_by_date(tmp_path: Path):
    connection, first_match_id = _seed_match(tmp_path)  # date=2026-08-06
    later_record = MatchRecord(
        date="2026-08-13",
        opponent="Jordan",
        result="L",
        sets=[SetRecord(set_number=1, games_won=4, games_lost=6, points=[])],
    )
    later_match_id = finalize_and_load(connection, later_record)
    earlier_record = MatchRecord(
        date="2026-07-30",
        opponent="Sam",
        result="W",
        sets=[SetRecord(set_number=1, games_won=6, games_lost=1, points=[])],
    )
    earlier_match_id = finalize_and_load(connection, earlier_record)

    assert all_match_ids(connection) == [earlier_match_id, first_match_id, later_match_id]


def test_all_match_stats_returns_empty_list_without_opening_a_connection():
    # A path that doesn't exist would raise if all_match_stats tried to
    # connect to it - the empty-match_ids short circuit must come first.
    assert all_match_stats(Path("does/not/exist.db"), []) == []


def test_all_match_stats_fetches_every_match_in_the_given_order(tmp_path: Path):
    connection, first_match_id = _seed_match(tmp_path)
    later_record = MatchRecord(
        date="2026-08-13",
        opponent="Jordan",
        result="L",
        sets=[SetRecord(set_number=1, games_won=4, games_lost=6, points=[])],
    )
    later_match_id = finalize_and_load(connection, later_record)

    results = all_match_stats(tmp_path / "test.db", [later_match_id, first_match_id])

    assert [m.match_id for m in results] == [later_match_id, first_match_id]
    assert [m.opponent for m in results] == ["Jordan", "Alex"]


def _match_stats(
    match_id: int,
    date: str,
    opponent: str,
    result: str,
    *,
    aces: int = 1,
    points_won_pct: float = 50.0,
    first_serve_pct: float = 60.0,
) -> MatchStats:
    return MatchStats(
        match_id=match_id,
        date=date,
        opponent=opponent,
        result=result,
        serving=ServingStats(10, 6, first_serve_pct, 4, 2, 50.0, aces, 1, 3, 4, 75.0),
        receiving=ReceivingStats(2, 1, 50.0, 2, 4, 50.0),
        point_outcomes=PointOutcomeStats(20, 12, points_won_pct, 6, 3, 2, 1, 1, 2.0),
        net=NetStats(3, 2, 66.7),
        clutch=ClutchStats(2, 1, 50.0),
        self_assessment=SelfAssessment(4, 3, None, None, None),
    )


def test_career_stats_from_matches_zeroed_for_an_empty_list():
    stats = career_stats_from_matches([])

    assert stats.total_matches == 0
    assert stats.win_pct == 0.0
    assert stats.current_streak_result is None
    assert stats.best_match_by_points_won_pct is None
    assert stats.most_aces_in_a_match is None


def test_career_stats_from_matches_computes_record_and_current_streak():
    matches = [
        _match_stats(1, "2026-08-01", "Alex", "W"),
        _match_stats(2, "2026-08-08", "Sam", "L"),
        _match_stats(3, "2026-08-15", "Jordan", "W"),
        _match_stats(4, "2026-08-22", "Casey", "W"),
    ]

    stats = career_stats_from_matches(matches)

    assert stats.total_matches == 4
    assert stats.wins == 3
    assert stats.losses == 1
    assert stats.win_pct == 75.0
    # Most recent two matches were both wins; the loss before that breaks the streak.
    assert stats.current_streak_result == "W"
    assert stats.current_streak_count == 2


def test_career_stats_from_matches_finds_highlights_regardless_of_input_order():
    matches = [
        _match_stats(1, "2026-08-01", "Alex", "W", aces=2, points_won_pct=55.0),
        _match_stats(2, "2026-08-08", "Sam", "L", aces=7, points_won_pct=40.0),
        _match_stats(3, "2026-08-15", "Jordan", "W", aces=1, points_won_pct=70.0),
    ]

    stats = career_stats_from_matches(matches)

    assert stats.best_match_by_points_won_pct.match_id == 3
    assert stats.best_match_by_points_won_pct.value == 70.0
    assert stats.most_aces_in_a_match.match_id == 2
    assert stats.most_aces_in_a_match.value == 7


def test_update_journal_fields_persists_new_pros_cons_notes(tmp_path: Path):
    connection, match_id = _seed_match(tmp_path)

    update_journal_fields(
        connection, match_id, pros="Better footwork", cons="Second serve", notes="Windy day"
    )

    updated = match_stats(connection, match_id).self_assessment
    assert updated.pros == "Better footwork"
    assert updated.cons == "Second serve"
    assert updated.notes == "Windy day"


def test_update_journal_fields_raises_for_an_unknown_match_id(tmp_path: Path):
    connection, _ = _seed_match(tmp_path)

    with pytest.raises(ValueError, match="999"):
        update_journal_fields(connection, 999, pros="x", cons="y", notes="z")
