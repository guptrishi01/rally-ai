"""Maps raw SwingVision rows onto RallyAI's MatchRecord staging schema."""

from __future__ import annotations

from .config import ImportConfig
from .raw import RawMatchExport
from .records import MatchRecord, PointRecord, SetRecord

_END_TYPE_ALIASES: dict[str, str] = {
    "winner": "winner",
    "unforced error": "unforced_error",
    "unforced_error": "unforced_error",
    "forced error": "forced_error",
    "forced_error": "forced_error",
    "ace": "ace",
    "double fault": "double_fault",
    "double_fault": "double_fault",
    "return winner": "return_winner",
    "return_winner": "return_winner",
    "return error": "return_error",
    "return_error": "return_error",
}


def _canonical_end_type(raw_end_type: str) -> str:
    """Maps a raw SwingVision point-outcome label onto RallyAI's canonical set.

    Args:
        raw_end_type: The raw label from the export (e.g. "Unforced Error").

    Returns:
        The canonical point_end_type value (e.g. "unforced_error").

    Raises:
        ValueError: If the label doesn't match any known alias, so an
            unrecognized SwingVision export doesn't get silently miscoded.
    """
    try:
        return _END_TYPE_ALIASES[raw_end_type.strip().lower()]
    except KeyError:
        raise ValueError(f"Unrecognized SwingVision point end type: {raw_end_type!r}") from None


def transform(
    raw: RawMatchExport,
    *,
    date: str,
    opponent: str,
    result: str,
    config: ImportConfig,
    source_file: str | None = None,
    **match_overrides: object,
) -> MatchRecord:
    """Builds a staged MatchRecord from a raw SwingVision export.

    Args:
        raw: The parsed sets, games, and points from SwingVisionParser.parse.
        date: ISO-format date of the match.
        opponent: Opponent's name.
        result: Match result, "W" or "L".
        config: Determines which point_end_type values require manual
            review before the resulting record can be finalized into SQL.
        source_file: Path to the original SwingVision export, stored on the
            record for traceability during review.
        **match_overrides: Additional MatchRecord fields to set directly
            (e.g. energy_rating, pros, cons, location).

    Returns:
        A MatchRecord with its points grouped into sets and ordered by
        (game_number, point_number), with needs_review flagged per
        config.needs_review_end_types.

    Raises:
        ValueError: If any point's end_type doesn't match a known
            SwingVision label.
    """
    points_by_set: dict[int, list[PointRecord]] = {}
    for raw_point in raw.points:
        end_type = _canonical_end_type(raw_point.end_type)
        point = PointRecord(
            game_number=raw_point.game_number,
            point_number=raw_point.point_number,
            is_serving=raw_point.server.strip().lower() == "host",
            point_won=raw_point.winner.strip().lower() == "host",
            point_end_type=end_type,
            first_serve_in=raw_point.first_serve_in,
            second_serve_in=raw_point.second_serve_in,
            needs_review=end_type in config.needs_review_end_types,
        )
        points_by_set.setdefault(raw_point.set_number, []).append(point)

    sets = [
        SetRecord(
            set_number=raw_set.set_number,
            games_won=raw_set.games_won,
            games_lost=raw_set.games_lost,
            points=sorted(
                points_by_set.get(raw_set.set_number, []),
                key=lambda p: (p.game_number, p.point_number),
            ),
        )
        for raw_set in sorted(raw.sets, key=lambda s: s.set_number)
    ]

    return MatchRecord(
        date=date,
        opponent=opponent,
        result=result,
        source_file=source_file,
        sets=sets,
        **match_overrides,
    )
