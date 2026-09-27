"""Reconstructs point-level records from SwingVision's Shots sheet.

Used when the Points/Games sheets are empty — confirmed the case for two
real (non-Pro) exports: SwingVision's point-outcome rollup requires Pro,
but shot-level AI tracking doesn't.

Every point this produces gets `needs_review=True`, always — not just the
ambiguous winner/unforced_error/forced_error subset the direct-parse
transform.py path flags. This is a heuristic *we* invented from raw shot
geometry, not even SwingVision's own (already-distrusted) point classifier,
so it's held to at least the same bar. That's not just caution: the real
Shots data has at least one confirmed case of a `second_serve` shot marked
`Result='Out'` (i.e. a double fault) immediately followed by a
`second_return` shot in the same point, which shouldn't happen if the serve
actually faulted — ace/double-fault detection here uses the serve shot's
own Result as authoritative regardless of what follows, and that ambiguity
is exactly why nothing from this module is ever trusted un-reviewed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from .raw import RawShotRow
from .records import PointRecord, SetRecord

logger = logging.getLogger(__name__)

_SERVE_TYPES = frozenset({"first_serve", "second_serve"})
_NET_STROKES = frozenset({"Volley", "Forehand Volley", "Backhand Volley", "Overhead"})

# Every real Shots `Type` value that represents an actual rally shot. A point
# whose shots are all outside this set (Type == "none", e.g. a Stroke=="Feed"
# ball sent across between points) isn't part of live play at all - confirmed
# against real data: e.g. one point is exactly "host feeds the ball in,
# opponent nets a backhand," not a served point. Excluding these is different
# from a gap (zero shots): these points have shots, they're just not tennis.
_RALLY_SHOT_TYPES = frozenset(
    {
        "first_serve",
        "second_serve",
        "first_return",
        "second_return",
        "serve_plus_one",
        "return_plus_one",
        "in_play",
    }
)


@dataclass(frozen=True)
class ReconstructedPoint:
    """One point's derived facts, before game/set boundaries are assigned.

    Attributes:
        point_number: The original match-wide point counter from the Shots
            sheet — not the final per-game point_number,
            assign_game_set_boundaries renumbers that.
        is_serving: Whether the tracked (host) player served this point.
        point_won: Whether the tracked player won this point.
        first_serve_in: Whether the first serve landed in, if a
            first_serve shot exists in this point.
        second_serve_in: Whether the second serve landed in, if a
            second_serve shot exists; None if there wasn't one.
        point_end_type: A coarse point-outcome bucket — see
            _coarse_end_type.
        net_approach: Whether the tracked player hit a Volley/Overhead
            shot in this point.
    """

    point_number: int
    is_serving: bool
    point_won: bool
    first_serve_in: bool | None
    second_serve_in: bool | None
    point_end_type: str
    net_approach: bool


def group_shots_by_point(shots: list[RawShotRow]) -> dict[int, list[RawShotRow]]:
    """Groups shots by point number, sorted by shot number within each point.

    Args:
        shots: All shot rows for a match (or video-segment export).

    Returns:
        point_number -> its shots, ordered by shot_number.
    """
    grouped: dict[int, list[RawShotRow]] = {}
    for shot in shots:
        grouped.setdefault(shot.point_number, []).append(shot)
    for point_shots in grouped.values():
        point_shots.sort(key=lambda s: s.shot_number)
    return grouped


def merge_shots(shots_by_file: list[list[RawShotRow]]) -> list[RawShotRow]:
    """Concatenates multiple exports' Shots rows into one continuous sequence.

    For a match whose recording was interrupted and split into multiple
    SwingVision exports — each file's own `Point` counter restarts from 1,
    so simply concatenating the raw rows would collide. Each file's
    point_number is shifted to continue right after the previous file's
    highest point_number, so the combined sequence can be reconstructed as
    one continuous match (one Set 1 spanning a file boundary, rather than
    two independent, incorrectly-scored partial reconstructions).

    This assumes files are given in play order and that no points were
    lost exactly at a file boundary — there's no shot data to detect a gap
    there, so none is fabricated; if the recording actually skipped a
    point at the cut, this can't tell.

    Args:
        shots_by_file: Each file's raw Shots rows, in play order (file 0
            played before file 1, etc.).

    Returns:
        One flat list with globally-unique, increasing point numbers.
    """
    merged: list[RawShotRow] = []
    offset = 0
    for part, shots in enumerate(shots_by_file):
        # source_part records which file - and so which recording - each
        # shot's video_time is relative to.
        merged.extend(
            replace(shot, point_number=shot.point_number + offset, source_part=part)
            for shot in shots
        )
        file_max = max((shot.point_number for shot in shots), default=0)
        offset += file_max
    return merged


def _coarse_end_type(
    *,
    is_serving: bool,
    point_won: bool,
    shot_count: int,
    first_serve_shot: RawShotRow | None,
    second_serve_shot: RawShotRow | None,
) -> str:
    """Buckets a point into one of the point table's allowed end types.

    Best-guess only — every caller of reconstruct_point keeps
    needs_review=True regardless of which bucket this returns.

    Args:
        is_serving: Whether the tracked player served this point.
        point_won: Whether the tracked player won this point.
        shot_count: Total shots recorded for this point.
        first_serve_shot: The point's first_serve shot, if any.
        second_serve_shot: The point's second_serve shot, if any.

    Returns:
        One of the point table's allowed point_end_type values.
    """
    if is_serving:
        if shot_count == 1 and first_serve_shot is not None and first_serve_shot.result == "In":
            return "ace"
        if second_serve_shot is not None and second_serve_shot.result != "In":
            return "double_fault"
        return "winner" if point_won else "unforced_error"
    # Not serving: if the point ended at the return itself (server's serve +
    # our return, nothing else), "return_*" is the accurate term. If the
    # rally continued past the return, a later shot ending it is a regular
    # winner/error, not specifically a "return" outcome.
    if shot_count <= 2:
        return "return_winner" if point_won else "return_error"
    return "winner" if point_won else "unforced_error"


def reconstruct_point(
    point_number: int, shots: list[RawShotRow], host_name: str
) -> ReconstructedPoint | None:
    """Derives one point's facts from its shot sequence.

    Args:
        point_number: The match-wide point number these shots belong to.
        shots: This point's shots, ordered by shot_number. Empty means a
            gap (SwingVision tracked no shots at all for this point
            number) — the caller should skip and report it, never
            fabricate one.
        host_name: The tracked player's display name, from Settings, used
            to tell which side hit each shot (Shots' own Player column is
            a display name, not a "host"/"guest" token like Sets/Games/
            Points use).

    Returns:
        The reconstructed point, or None if `shots` is empty (a gap).
    """
    if not shots:
        return None

    serve_shot = next((s for s in shots if s.shot_type in _SERVE_TYPES), None)
    is_serving = serve_shot is not None and serve_shot.player == host_name

    first_serve_shot = next((s for s in shots if s.shot_type == "first_serve"), None)
    second_serve_shot = next((s for s in shots if s.shot_type == "second_serve"), None)
    first_serve_in = first_serve_shot.result == "In" if first_serve_shot else None
    second_serve_in = second_serve_shot.result == "In" if second_serve_shot else None

    last_shot = shots[-1]
    last_shot_by_host = last_shot.player == host_name
    # A shot that lands 'In' with no reply ends the point for its hitter
    # (a winner); anything else (Out/Net) ends it against its hitter.
    point_won = last_shot_by_host if last_shot.result == "In" else not last_shot_by_host

    point_end_type = _coarse_end_type(
        is_serving=is_serving,
        point_won=point_won,
        shot_count=len(shots),
        first_serve_shot=first_serve_shot,
        second_serve_shot=second_serve_shot,
    )

    net_approach = any(s.player == host_name and s.stroke in _NET_STROKES for s in shots)

    return ReconstructedPoint(
        point_number=point_number,
        is_serving=is_serving,
        point_won=point_won,
        first_serve_in=first_serve_in,
        second_serve_in=second_serve_in,
        point_end_type=point_end_type,
        net_approach=net_approach,
    )


def _to_point_record(
    point: ReconstructedPoint, game_number: int, point_number: int, is_tiebreak_game: bool
) -> PointRecord:
    """Converts a ReconstructedPoint into a staging PointRecord.

    Args:
        point: The reconstructed point.
        game_number: Its assigned game number within the set.
        point_number: Its assigned point number within the game (not the
            original match-wide counter).
        is_tiebreak_game: Whether this point was played in a 6-6 tiebreak
            game.

    Returns:
        A PointRecord with needs_review=True, always.
    """
    return PointRecord(
        game_number=game_number,
        point_number=point_number,
        is_serving=point.is_serving,
        point_won=point.point_won,
        point_end_type=point.point_end_type,
        first_serve_in=point.first_serve_in,
        second_serve_in=point.second_serve_in,
        net_approach=point.net_approach,
        is_tiebreak_game=is_tiebreak_game,
        needs_review=True,
        source_point_number=point.point_number,
    )


def assign_game_set_boundaries(
    points: list[ReconstructedPoint], *, ad_scoring: bool = True
) -> list[SetRecord]:
    """Forward-simulates standard scoring to assign game/set boundaries.

    SwingVision's Shots sheet carries no usable Game/Set columns (always
    0), so there's no rollup to read boundaries from — this replays the
    point sequence under standard tennis rules instead. A match that ends
    mid-game or mid-set (e.g. an interrupted recording) simply produces a
    trailing partial game/set with however many points it actually got,
    rather than requiring the sequence to end on a clean boundary.

    Args:
        points: Reconstructed points in original play order (gaps already
            excluded by the caller).
        ad_scoring: True for standard advantage scoring (a game must be
            won by 2 clear points from 3-3); False for no-ad, sudden-point
            scoring (first to 4 wins outright even at 3-3).

    Returns:
        SetRecords with their points' game_number/point_number assigned,
        and games_won/games_lost tallied from this simulation — which may
        differ from SwingVision's own (Pro-only, and in the real match
        this was built against, confirmed wrong) Sets-sheet summary.
    """
    sets: list[SetRecord] = []
    set_number = 1
    game_number = 1
    point_in_game = 0
    host_points, guest_points = 0, 0
    host_games, guest_games = 0, 0
    current_set_points: list[PointRecord] = []

    def _is_tiebreak_game() -> bool:
        return host_games == 6 and guest_games == 6

    def _game_won() -> bool | None:
        """Returns True/False if the current game just ended, else None."""
        if _is_tiebreak_game():
            leader, margin = max(host_points, guest_points), abs(host_points - guest_points)
            if leader >= 7 and margin >= 2:
                return host_points > guest_points
            return None
        if ad_scoring:
            leader, margin = max(host_points, guest_points), abs(host_points - guest_points)
            if leader >= 4 and margin >= 2:
                return host_points > guest_points
            return None
        if max(host_points, guest_points) >= 4:
            return host_points > guest_points
        return None

    for point in points:
        point_in_game += 1
        # host_games/guest_games only change at a game boundary (below), so
        # this is stable and correct for every point within the same game.
        current_game_is_tiebreak = _is_tiebreak_game()
        current_set_points.append(
            _to_point_record(point, game_number, point_in_game, current_game_is_tiebreak)
        )
        if point.point_won:
            host_points += 1
        else:
            guest_points += 1

        host_won_game = _game_won()
        if host_won_game is None:
            continue

        # Capture before mutating host_games/guest_games below: whether the
        # game that just ended *was* the 6-6 tiebreak, for the set record.
        was_tiebreak = current_game_is_tiebreak
        if host_won_game:
            host_games += 1
        else:
            guest_games += 1
        game_number += 1
        point_in_game = 0
        host_points, guest_points = 0, 0

        set_margin = abs(host_games - guest_games)
        set_leader = max(host_games, guest_games)
        set_over = (set_leader >= 6 and set_margin >= 2) or set_leader == 7
        if set_over:
            sets.append(
                SetRecord(
                    set_number=set_number,
                    games_won=host_games,
                    games_lost=guest_games,
                    is_tiebreak_set=was_tiebreak,
                    points=current_set_points,
                )
            )
            set_number += 1
            game_number = 1
            host_games, guest_games = 0, 0
            current_set_points = []

    if current_set_points:
        sets.append(
            SetRecord(
                set_number=set_number,
                games_won=host_games,
                games_lost=guest_games,
                is_tiebreak_set=False,
                points=current_set_points,
            )
        )

    return sets


@dataclass(frozen=True)
class ReconstructionResult:
    """The full output of reconstructing a match from its Shots sheet.

    Attributes:
        sets: The reconstructed sets, with game/set boundaries assigned.
        skipped_points: Match-wide point numbers with zero shots at all —
            a true gap (a recording dropout), never fabricated.
        excluded_points: Match-wide point numbers that had shots, but none
            of them were a rally-type shot (Type != "none") — e.g. a
            Stroke=="Feed" ball sent across between points. Not part of
            live play, so excluded rather than counted as a real point;
            distinct from skipped_points because these aren't a data gap,
            they're correctly-tracked non-match activity.
        points: The reconstructed points themselves, pre-boundary-
            assignment (still carrying their original match-wide
            point_number) — kept here so build_shot_pattern_summary can be
            called against them without re-deriving anything.
    """

    sets: list[SetRecord]
    skipped_points: list[int]
    excluded_points: list[int]
    points: list[ReconstructedPoint]


def _has_rally_shot(shots: list[RawShotRow]) -> bool:
    """Whether at least one shot in a point is an actual rally-type shot.

    Args:
        shots: A single point's shots.

    Returns:
        True if any shot's type is in _RALLY_SHOT_TYPES.
    """
    return any(s.shot_type in _RALLY_SHOT_TYPES for s in shots)


def reconstruct_all(
    shots: list[RawShotRow], host_name: str, *, ad_scoring: bool = True
) -> ReconstructionResult:
    """Full pipeline: Shots rows -> SetRecords, for use when Points is empty.

    Args:
        shots: All rows from the Shots sheet.
        host_name: The tracked player's display name, from Settings.
        ad_scoring: Passed through to assign_game_set_boundaries.

    Returns:
        A ReconstructionResult with the reconstructed sets plus the
        skipped (gap) and excluded (non-match) point numbers, both
        reported rather than silently dropped or fabricated.
    """
    grouped = group_shots_by_point(shots)
    points: list[ReconstructedPoint] = []
    skipped: list[int] = []
    excluded: list[int] = []
    if grouped:
        # A gap point number has zero shots, so it never becomes a key in
        # `grouped` at all - iterating its keys would silently skip past
        # it. Walking the full min..max range is what actually finds gaps.
        for point_number in range(min(grouped), max(grouped) + 1):
            point_shots = grouped.get(point_number, [])
            if not point_shots:
                skipped.append(point_number)
            elif not _has_rally_shot(point_shots):
                excluded.append(point_number)
            else:
                reconstructed = reconstruct_point(point_number, point_shots, host_name)
                points.append(reconstructed)

    if skipped:
        logger.warning(
            "Skipped %d point(s) with no shot data (likely a recording gap): %s",
            len(skipped),
            skipped,
        )
    if excluded:
        logger.info(
            "Excluded %d point(s) with no rally-type shot (not part of live play, "
            "e.g. a fed ball between points): %s",
            len(excluded),
            excluded,
        )

    sets = assign_game_set_boundaries(points, ad_scoring=ad_scoring)
    return ReconstructionResult(
        sets=sets, skipped_points=skipped, excluded_points=excluded, points=points
    )


_SHORT_RALLY_MAX_SHOTS = 4


def build_shot_pattern_summary(
    points: list[ReconstructedPoint], shots_by_point: dict[int, list[RawShotRow]]
) -> dict[str, float] | None:
    """Aggregates rally-length patterns for the AI coach's optional context.

    Rally-length only for now — SwingVision's Speed (MPH) column exists in
    the real Shots sheet but isn't parsed by raw.py/RawShotRow yet, so a
    speed-based metric is out of scope for this pass rather than silently
    assumed.

    Args:
        points: Reconstructed points (already excludes gaps and non-match
            points — see reconstruct_all).
        shots_by_point: point_number -> its shots, from group_shots_by_point.

    Returns:
        None if there are no points to summarize; otherwise avg_rally_length
        (mean shot count per point) and rally_win_rate_short/
        rally_win_rate_long (win rate split at _SHORT_RALLY_MAX_SHOTS shots).
    """
    if not points:
        return None

    rally_lengths = []
    short_won = short_total = 0
    long_won = long_total = 0
    for point in points:
        shot_count = len(shots_by_point.get(point.point_number, []))
        rally_lengths.append(shot_count)
        if shot_count <= _SHORT_RALLY_MAX_SHOTS:
            short_total += 1
            short_won += int(point.point_won)
        else:
            long_total += 1
            long_won += int(point.point_won)

    return {
        "avg_rally_length": round(sum(rally_lengths) / len(rally_lengths), 2),
        "rally_win_rate_short": round(short_won / short_total * 100, 1) if short_total else 0.0,
        "rally_win_rate_long": round(long_won / long_total * 100, 1) if long_total else 0.0,
    }
