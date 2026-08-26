"""Staging-JSON persistence and the review-flag gate that guards SQL loading."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .records import LOSING_END_TYPES, VALID_END_TYPES, WINNING_END_TYPES, MatchRecord

# Characters invalid in Windows filenames (a superset of what's invalid on
# macOS/Linux), plus whitespace collapsed to a single underscore. Opponent
# names are free text, so this can't assume they're already filename-safe.
_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def slugify_filename(value: str) -> str:
    """Turns free text into a filesystem-safe filename component.

    Args:
        value: The raw text (e.g. an opponent's name) to sanitize.

    Returns:
        The text with filename-unsafe characters replaced by underscores,
        or "unknown" if nothing safe remains.
    """
    slug = _UNSAFE_FILENAME_CHARS.sub("_", value.strip())
    slug = re.sub(r"\s+", "_", slug).strip("._")
    return slug or "unknown"


def save_pending(record: MatchRecord, pending_dir: Path) -> Path:
    """Writes a MatchRecord to the pending-review JSON directory.

    Re-saving a record for the same date/opponent overwrites the existing
    file rather than creating a duplicate.

    Args:
        record: The match to stage for review.
        pending_dir: Directory to write the JSON file into; created if it
            doesn't already exist.

    Returns:
        Path to the written JSON file.
    """
    pending_dir.mkdir(parents=True, exist_ok=True)
    opponent_slug = slugify_filename(record.opponent)
    path = pending_dir / f"{record.date}_{opponent_slug}.json"
    path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return path


def find_pending_path(date: str, opponent: str, pending_dir: Path) -> Path | None:
    """Locates a match's pending JSON by the same naming save_pending uses.

    Used by ai/pipeline.py to look up a match's original staged data (for
    its optional shot_pattern_summary) using only the date/opponent already
    stored in the `match` table — no schema change needed to link SQL rows
    back to their staging JSON.

    Args:
        date: The match's date, as stored on the `match` row.
        opponent: The match's opponent, as stored on the `match` row.
        pending_dir: Directory pending JSON files are saved into.

    Returns:
        The path, if a file with that name still exists (it may have been
        deleted after finalize(), or never existed for a direct-parse
        match with a different provenance); None otherwise.
    """
    path = pending_dir / f"{date}_{slugify_filename(opponent)}.json"
    return path if path.exists() else None


def load_pending(path: Path) -> MatchRecord:
    """Loads a staged MatchRecord back from its JSON file.

    Args:
        path: Path to a JSON file previously written by save_pending.

    Returns:
        The reconstructed MatchRecord.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return MatchRecord.from_dict(data)


class ConfirmationError(ValueError):
    """Raised when a direct point confirmation is invalid.

    Covers both an unknown (set_number, game_number, point_number) lookup
    and a point_end_type/point_won pair that would violate
    data/schema.sql's CHECK constraint - the same consistency rule
    review_resolve.resolve_point_answer enforces on Claude's parsed output,
    applied here to a human's direct choice from the review UI instead.
    """


def confirm_point(
    record: MatchRecord,
    *,
    set_number: int,
    game_number: int,
    point_number: int,
    point_end_type: str,
    point_won: bool,
    net_approach: bool,
) -> MatchRecord:
    """Directly applies a human's manual confirmation for one flagged point.

    The browser review UI's fast path: the human picks the correct
    point_end_type/point_won/net_approach straight from controls in the
    UI, no Claude call needed. Distinct from apply_resolutions(), which
    only ever applies resolved_* fields that review_resolve.py already
    parsed from a review_answer - this sets the real fields immediately
    from the human's direct input and clears needs_review the same way.

    Args:
        record: The staged match record to update in place.
        set_number: The point's set number.
        game_number: The point's game number within that set.
        point_number: The point's number within that game.
        point_end_type: The confirmed outcome - must be one of
            records.VALID_END_TYPES.
        point_won: Whether the tracked player won the point.
        net_approach: Whether the tracked player approached the net.

    Returns:
        The same record, with the matching point's fields updated and
        needs_review cleared. Not saved to disk - the caller decides when
        to persist (see save_pending).

    Raises:
        ConfirmationError: If point_end_type isn't a valid value, if it's
            inconsistent with point_won (mirrors the point table's own
            CHECK constraint), or if no point matches the given
            set/game/point numbers.
    """
    if point_end_type not in VALID_END_TYPES:
        raise ConfirmationError(f"not a valid point_end_type: {point_end_type!r}")
    if point_end_type in WINNING_END_TYPES and not point_won:
        raise ConfirmationError(f"{point_end_type!r} requires point_won=true, got false")
    if point_end_type in LOSING_END_TYPES and point_won:
        raise ConfirmationError(f"{point_end_type!r} requires point_won=false, got true")

    for set_record in record.sets:
        if set_record.set_number != set_number:
            continue
        for point in set_record.points:
            if point.game_number != game_number or point.point_number != point_number:
                continue
            point.point_end_type = point_end_type
            point.point_won = point_won
            point.net_approach = net_approach
            point.needs_review = False
            return record

    raise ConfirmationError(
        f"no point at set {set_number} game {game_number} point {point_number}"
    )


def unresolved_flags(record: MatchRecord) -> list[str]:
    """Lists every point in a record that still needs manual review.

    Args:
        record: The match to check.

    Returns:
        One human-readable description per point with needs_review=True,
        identifying its set, game, point number, and outcome type. Empty
        if the record is fully resolved and safe to finalize into SQL.
    """
    flags = []
    for set_record in record.sets:
        for point in set_record.points:
            if not point.needs_review:
                continue
            message = (
                f"set {set_record.set_number} game {point.game_number} "
                f"point {point.point_number}: '{point.point_end_type}' needs confirmation"
            )
            if point.ai_suggested_point_end_type is not None:
                message += (
                    f" (Claude suggests '{point.ai_suggested_point_end_type}': "
                    f"{point.ai_suggestion_reasoning})"
                )
            flags.append(message)
    return flags
