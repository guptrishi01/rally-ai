"""Derived-stat aggregation from data/schema.sql, per docs/stat-definitions.md.

Each stat category has a pure `*_from_points` function (easy to unit test
with hand-built PointRow lists) and the module-level `match_stats` function
that fetches from SQLite and assembles the full MatchStats bundle.
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .models import (
    CareerHighlight,
    CareerStats,
    ClutchStats,
    MatchStats,
    NetStats,
    PointOutcomeStats,
    PointRow,
    ReceivingStats,
    SelfAssessment,
    ServingStats,
)
from .scoring import reconstruct

_FETCH_POINTS_SQL = """
    SELECT
        s.set_number,
        p.game_number,
        p.point_number,
        p.is_serving,
        p.first_serve_in,
        p.second_serve_in,
        p.point_won,
        p.point_end_type,
        p.net_approach,
        p.is_tiebreak_game
    FROM point p
    JOIN "set" s ON p.set_id = s.set_id
    WHERE s.match_id = ?{set_filter}
    ORDER BY s.set_number, p.game_number, p.point_number
"""


def _pct(numerator: int, denominator: int) -> float:
    """Computes a percentage, returning 0.0 rather than dividing by zero.

    Args:
        numerator: The count meeting the condition.
        denominator: The total count.

    Returns:
        `numerator / denominator * 100`, rounded to 1 decimal place, or 0.0
        if denominator is 0.
    """
    if denominator == 0:
        return 0.0
    return round(numerator / denominator * 100, 1)


def _fetch_points(
    conn: sqlite3.Connection, match_id: int, set_id: int | None = None
) -> list[PointRow]:
    """Fetches a match's (or one set's) points, ordered for aggregation.

    Args:
        conn: An open SQLite connection.
        match_id: The match to fetch points for.
        set_id: If given, restricts to this one set (set-level scope);
            otherwise all sets in the match (match-level scope).

    Returns:
        Ordered PointRow list, ready for the *_from_points functions or
        scoring.reconstruct.
    """
    set_filter = " AND s.set_id = ?" if set_id is not None else ""
    params = (match_id, set_id) if set_id is not None else (match_id,)
    cursor = conn.execute(_FETCH_POINTS_SQL.format(set_filter=set_filter), params)
    return [
        PointRow(
            set_number=row[0],
            game_number=row[1],
            point_number=row[2],
            is_serving=bool(row[3]),
            first_serve_in=None if row[4] is None else bool(row[4]),
            second_serve_in=None if row[5] is None else bool(row[5]),
            point_won=bool(row[6]),
            point_end_type=row[7],
            net_approach=bool(row[8]),
            is_tiebreak_game=bool(row[9]),
        )
        for row in cursor.fetchall()
    ]


def _non_tiebreak_games(points: list[PointRow]) -> list[list[PointRow]]:
    """Groups points into per-game lists, excluding tiebreak games.

    Args:
        points: Ordered point rows (see _fetch_points).

    Returns:
        One list of points per non-tiebreak game, in play order.
    """
    games: dict[tuple[int, int], list[PointRow]] = {}
    order: list[tuple[int, int]] = []
    for row in points:
        if row.is_tiebreak_game:
            continue
        key = (row.set_number, row.game_number)
        if key not in games:
            games[key] = []
            order.append(key)
        games[key].append(row)
    return [games[key] for key in order]


def serving_stats_from_points(points: list[PointRow]) -> ServingStats:
    """Computes serving stats from already-fetched point rows.

    Args:
        points: Ordered point rows for a match or a single set.

    Returns:
        The serving stats for that scope.
    """
    serve_points = [p for p in points if p.is_serving]
    first_serves_total = len(serve_points)
    first_serves_in = sum(1 for p in serve_points if p.first_serve_in)
    second_serve_points = [p for p in serve_points if p.first_serve_in is False]
    second_serves_total = len(second_serve_points)
    second_serves_in = sum(1 for p in second_serve_points if p.second_serve_in)
    aces = sum(1 for p in points if p.point_end_type == "ace")
    double_faults = sum(1 for p in points if p.point_end_type == "double_fault")

    service_games_won = 0
    service_games_total = 0
    for game in _non_tiebreak_games(points):
        if not game[0].is_serving:
            continue
        service_games_total += 1
        if game[-1].point_won:
            service_games_won += 1

    return ServingStats(
        first_serves_total=first_serves_total,
        first_serves_in=first_serves_in,
        first_serve_pct=_pct(first_serves_in, first_serves_total),
        second_serves_total=second_serves_total,
        second_serves_in=second_serves_in,
        second_serve_pct=_pct(second_serves_in, second_serves_total),
        aces=aces,
        double_faults=double_faults,
        service_games_won=service_games_won,
        service_games_total=service_games_total,
        service_hold_pct=_pct(service_games_won, service_games_total),
    )


def receiving_stats_from_points(points: list[PointRow]) -> ReceivingStats:
    """Computes receiving stats from already-fetched point rows.

    Args:
        points: Ordered point rows for a match or a single set.

    Returns:
        The receiving stats for that scope.
    """
    contexts = reconstruct(points)
    break_points_total = sum(1 for c in contexts if c.is_break_point)
    break_points_converted = sum(1 for c in contexts if c.is_break_point and c.point.point_won)

    return_games_won = 0
    return_games_total = 0
    for game in _non_tiebreak_games(points):
        if game[0].is_serving:
            continue
        return_games_total += 1
        if game[-1].point_won:
            return_games_won += 1

    return ReceivingStats(
        break_points_total=break_points_total,
        break_points_converted=break_points_converted,
        break_point_conversion_pct=_pct(break_points_converted, break_points_total),
        return_games_won=return_games_won,
        return_games_total=return_games_total,
        return_win_pct=_pct(return_games_won, return_games_total),
    )


def point_outcome_stats_from_points(points: list[PointRow]) -> PointOutcomeStats:
    """Computes point outcome stats from already-fetched point rows.

    Args:
        points: Ordered point rows for a match or a single set.

    Returns:
        The point outcome stats for that scope.
    """
    total_points_played = len(points)
    total_points_won = sum(1 for p in points if p.point_won)
    winners = sum(1 for p in points if p.point_end_type == "winner")
    unforced_errors = sum(1 for p in points if p.point_end_type == "unforced_error")
    forced_errors = sum(1 for p in points if p.point_end_type == "forced_error")
    return_winners = sum(1 for p in points if p.point_end_type == "return_winner")
    return_errors = sum(1 for p in points if p.point_end_type == "return_error")

    return PointOutcomeStats(
        total_points_played=total_points_played,
        total_points_won=total_points_won,
        points_won_pct=_pct(total_points_won, total_points_played),
        winners=winners,
        unforced_errors=unforced_errors,
        forced_errors=forced_errors,
        return_winners=return_winners,
        return_errors=return_errors,
        winner_to_ue_ratio=round(winners / unforced_errors, 2) if unforced_errors else 0.0,
    )


def net_stats_from_points(points: list[PointRow]) -> NetStats:
    """Computes net stats from already-fetched point rows.

    Args:
        points: Ordered point rows for a match or a single set.

    Returns:
        The net stats for that scope.
    """
    net_approaches = sum(1 for p in points if p.net_approach)
    net_points_won = sum(1 for p in points if p.net_approach and p.point_won)
    return NetStats(
        net_approaches=net_approaches,
        net_points_won=net_points_won,
        net_success_pct=_pct(net_points_won, net_approaches),
    )


def clutch_stats_from_points(points: list[PointRow]) -> ClutchStats:
    """Computes clutch stats from already-fetched point rows.

    Args:
        points: Ordered point rows for a match or a single set.

    Returns:
        The clutch stats for that scope.
    """
    contexts = reconstruct(points)
    deuce_points_played = sum(1 for c in contexts if c.is_deuce_point)
    deuces_converted = sum(1 for c in contexts if c.is_deuce_point and c.point.point_won)
    return ClutchStats(
        deuce_points_played=deuce_points_played,
        deuces_converted=deuces_converted,
        deuce_conversion_pct=_pct(deuces_converted, deuce_points_played),
    )


def match_stats(conn: sqlite3.Connection, match_id: int, set_id: int | None = None) -> MatchStats:
    """Assembles the full derived-stats bundle for a match or one of its sets.

    Args:
        conn: An open SQLite connection.
        match_id: The match to aggregate.
        set_id: If given, scopes every stat to this one set; otherwise
            aggregates across the whole match.

    Returns:
        The full MatchStats bundle.

    Raises:
        ValueError: If no match with this match_id exists.
    """
    row = conn.execute(
        "SELECT date, opponent, result, energy_rating, mental_rating, pros, cons, notes "
        "FROM match WHERE match_id = ?",
        (match_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"No match with match_id={match_id}")
    date, opponent, result, energy_rating, mental_rating, pros, cons, notes = row

    points = _fetch_points(conn, match_id, set_id)

    return MatchStats(
        match_id=match_id,
        date=date,
        opponent=opponent,
        result=result,
        serving=serving_stats_from_points(points),
        receiving=receiving_stats_from_points(points),
        point_outcomes=point_outcome_stats_from_points(points),
        net=net_stats_from_points(points),
        clutch=clutch_stats_from_points(points),
        self_assessment=SelfAssessment(
            energy_rating=energy_rating,
            mental_rating=mental_rating,
            pros=pros,
            cons=cons,
            notes=notes,
        ),
    )


def all_match_ids(conn: sqlite3.Connection) -> list[int]:
    """Lists every finalized match's id, oldest first.

    Args:
        conn: An open SQLite connection.

    Returns:
        match_id for every row in `match`, ordered by date. Empty if no
        match has been finalized yet.
    """
    rows = conn.execute("SELECT match_id FROM match ORDER BY date").fetchall()
    return [row[0] for row in rows]


def all_match_stats(db_path: Path, match_ids: list[int]) -> list[MatchStats]:
    """Fetches MatchStats for every given match_id concurrently.

    Each match's stats come from an independent, stateless SQLite read, so
    this mirrors ai/generate.py's ThreadPoolExecutor pattern (the other
    sanctioned use of it in this codebase — see CLAUDE.md's Code
    Conventions) instead of fetching one at a time, keeping the Overview/
    Statistics tabs' load time closer to linear as match count grows. A
    single sqlite3.Connection isn't safe to share across threads, so each
    worker opens (and closes) its own short-lived, read-only connection to
    the same database file rather than reusing a passed-in connection.

    Args:
        db_path: Path to the SQLite database file.
        match_ids: The matches to fetch, in the order results should come
            back in (all_match_ids' date-ascending order, typically).

    Returns:
        MatchStats in the same order as match_ids. Empty if match_ids is
        empty (no thread pool spun up for nothing).
    """
    if not match_ids:
        return []

    def _fetch(match_id: int) -> MatchStats:
        connection = sqlite3.connect(db_path)
        try:
            return match_stats(connection, match_id)
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=min(8, len(match_ids))) as executor:
        return list(executor.map(_fetch, match_ids))


def career_stats_from_matches(stats: list[MatchStats]) -> CareerStats:
    """Aggregates career-level stats from an already-fetched list of matches.

    Pure - no I/O, easy to test without a database or threading, matching
    this module's existing *_from_points convention (see
    serving_stats_from_points etc.): the I/O (all_match_stats, above) and
    the aggregation logic are two separate, independently testable steps.

    Args:
        stats: Every finalized match's MatchStats, in any order.

    Returns:
        The career-wide rollup. Every count/percentage field is 0/0.0 and
        both highlights are None if stats is empty, matching this
        package's "0.0 rather than raising" convention for empty scopes.
    """
    if not stats:
        return CareerStats(
            total_matches=0,
            wins=0,
            losses=0,
            win_pct=0.0,
            current_streak_result=None,
            current_streak_count=0,
            avg_first_serve_pct=0.0,
            avg_points_won_pct=0.0,
            best_match_by_points_won_pct=None,
            most_aces_in_a_match=None,
        )

    total = len(stats)
    wins = sum(1 for m in stats if m.result == "W")

    by_date = sorted(stats, key=lambda m: m.date)
    current_streak_result = by_date[-1].result
    current_streak_count = 0
    for m in reversed(by_date):
        if m.result != current_streak_result:
            break
        current_streak_count += 1

    best = max(stats, key=lambda m: m.point_outcomes.points_won_pct)
    most_aces = max(stats, key=lambda m: m.serving.aces)

    return CareerStats(
        total_matches=total,
        wins=wins,
        losses=total - wins,
        win_pct=_pct(wins, total),
        current_streak_result=current_streak_result,
        current_streak_count=current_streak_count,
        avg_first_serve_pct=round(sum(m.serving.first_serve_pct for m in stats) / total, 1),
        avg_points_won_pct=round(sum(m.point_outcomes.points_won_pct for m in stats) / total, 1),
        best_match_by_points_won_pct=CareerHighlight(
            match_id=best.match_id,
            date=best.date,
            opponent=best.opponent,
            value=best.point_outcomes.points_won_pct,
            label="Best points-won%",
        ),
        most_aces_in_a_match=CareerHighlight(
            match_id=most_aces.match_id,
            date=most_aces.date,
            opponent=most_aces.opponent,
            value=most_aces.serving.aces,
            label="Most aces",
        ),
    )


def update_journal_fields(
    conn: sqlite3.Connection,
    match_id: int,
    *,
    pros: str | None,
    cons: str | None,
    notes: str | None,
) -> None:
    """Updates a finalized match's pros/cons/notes — the Journal tab's sticky note.

    One sticky note per match, reusing these existing `match` columns
    rather than a separate journal_entry table — see CLAUDE.md's Journal
    design decision.

    Args:
        conn: An open SQLite connection.
        match_id: The match to update.
        pros: New "what went well" text, replacing the existing value.
        cons: New "what needs work" text, replacing the existing value.
        notes: New free-text notes, replacing the existing value.

    Raises:
        ValueError: If no match with this match_id exists.
    """
    cursor = conn.execute(
        "UPDATE match SET pros = ?, cons = ?, notes = ? WHERE match_id = ?",
        (pros, cons, notes, match_id),
    )
    if cursor.rowcount == 0:
        raise ValueError(f"No match with match_id={match_id}")
    conn.commit()
