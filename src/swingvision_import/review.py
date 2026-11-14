"""Staging-JSON persistence and the review-flag gate that guards SQL loading."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePath

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


def _source_names(record: MatchRecord) -> set[str]:
    """The basenames of the exports a record was built from.

    Basenames rather than full paths: the web form stores an upload as
    "<date>_<opponent>_<original name>.xlsx", which is stable for the same
    source file across re-uploads, and archiving later rewrites the entry
    to a bare member name. Comparing basenames survives both.

    Args:
        record: The staged match record.

    Returns:
        The set of source basenames; empty if the record has none.
    """
    sources = record.source_files or ([record.source_file] if record.source_file else [])
    return {PurePath(source).name for source in sources if source}


def allocate_pending_name(record: MatchRecord, pending_dir: Path) -> str:
    """Picks a staged filename, reusing one only for the very same match.

    Two competing rules meet here. CLAUDE.md says never reprocess or
    duplicate by default, so re-ingesting a match must land back on its own
    file. But two genuinely different matches against the same opponent on
    the same day used to collapse into one file and silently destroy the
    first.

    Both hold if the question is "is this the same match?" rather than "is
    the name taken?" - decided by the exports the record was built from. A
    re-ingest of the same files overwrites as before; a different match
    becomes "<date>_<opponent>_2.json".

    An interrupted recording is not this case at all: its parts arrive
    together and ingest_multi_part merges them into one record.

    Args:
        record: The match being staged.
        pending_dir: Where staged matches live.

    Returns:
        The filename to write to.
    """
    stem = f"{record.date}_{slugify_filename(record.opponent)}"
    names = _source_names(record)
    suffix = 1
    while True:
        candidate = f"{stem}.json" if suffix == 1 else f"{stem}_{suffix}.json"
        path = pending_dir / candidate
        if not path.exists():
            return candidate
        try:
            existing = load_pending(path)
        except (OSError, ValueError):
            return candidate  # unreadable: take it rather than orphan the name
        # Same exports means this is the same match being staged again.
        if names and _source_names(existing) == names:
            return candidate
        suffix += 1


def save_pending(record: MatchRecord, pending_dir: Path) -> Path:
    """Writes a MatchRecord to the pending-review JSON directory.

    A record that already knows its filename (record.pending_name) is
    written back to exactly that file, so the seven update paths - adding
    suggestions, footage verdicts, confirmations, archived sources - keep
    overwriting their own file rather than spawning copies. A record with
    no name yet is a new match, and gets one allocated without clobbering
    an existing match (see allocate_pending_name).

    Args:
        record: The match to stage for review.
        pending_dir: Directory to write the JSON file into; created if it
            doesn't already exist.

    Returns:
        Path to the written JSON file.
    """
    pending_dir.mkdir(parents=True, exist_ok=True)
    if not record.pending_name:
        record.pending_name = allocate_pending_name(record, pending_dir)
    path = pending_dir / record.pending_name
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
    record = MatchRecord.from_dict(data)
    # Records staged before pending_name existed carry no name; adopt the
    # file they actually came from, so saving one back cannot allocate a
    # fresh suffixed copy of itself.
    if not record.pending_name:
        record.pending_name = path.name
    return record


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
        The same record, with the matching point's fields updated,
        needs_review cleared, and confirmed_by set to "human" - so a
        point a person signed off on stays distinguishable from one the
        footage check cleared. Not saved to disk - the caller decides when
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
            point.confirmed_by = "human"
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
