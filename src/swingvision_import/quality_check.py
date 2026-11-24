"""Structured-data-only quality checks for a reconstructed match.

No video, no vision, no API calls — every check here reasons over data
SwingVision itself exported (the Sets-sheet summary, Settings) or ground
truth the user supplied through the intake UI. Results are informational
`MatchRecord.import_notes`, never a blocking gate like `needs_review` —
imperfection is expected and reported, not treated as a failure.
"""

from __future__ import annotations

from .raw import RawSetRow, RawSettings
from .records import SetRecord


def check_score_against_sets_sheet(
    sets: list[SetRecord], raw_sets: list[RawSetRow]
) -> list[str]:
    """Compares each reconstructed set's score against the Sets sheet's own.

    The Sets sheet is populated even without Pro and is SwingVision's own
    summary — not necessarily correct (a known real case: it showed 6-4
    for a set that was actually 5-3), but a mismatch is worth surfacing
    either way, since it means the two data sources disagree.

    Args:
        sets: Reconstructed sets, from reconstruct.assign_game_set_boundaries.
        raw_sets: Raw rows from the Sets sheet.

    Returns:
        One note per set_number present in both sides whose games_won/
        games_lost disagree. Empty if every matched set agrees, or if
        raw_sets has no row for a given set_number to compare against.
    """
    raw_by_set_number = {raw.set_number: raw for raw in raw_sets}
    notes = []
    for set_record in sets:
        raw = raw_by_set_number.get(set_record.set_number)
        if raw is None:
            continue
        if (set_record.games_won, set_record.games_lost) != (raw.games_won, raw.games_lost):
            notes.append(
                f"Set {set_record.set_number}: reconstructed score "
                f"{set_record.games_won}-{set_record.games_lost} does not match the "
                f"Sets sheet's own summary ({raw.games_won}-{raw.games_lost})."
            )
    return notes


def check_serve_order(
    sets: list[SetRecord], first_server_by_set: dict[int, str] | None
) -> list[str]:
    """Cross-checks reconstructed serve order against user-supplied ground truth.

    A mismatch is high-severity: since is_serving/point_won are derived
    relative to which side is "host", a wrong first server for a set
    usually means host/guest are reversed for that whole set, not just one
    point.

    Args:
        sets: Reconstructed sets.
        first_server_by_set: set_number -> "me" or "opponent", as supplied
            by the user at intake. None or missing entries are skipped
            (nothing to check against).

    Returns:
        One note per set whose first point's is_serving disagrees with the
        user-supplied server. Empty if nothing was supplied or everything
        agrees.
    """
    if not first_server_by_set:
        return []
    notes = []
    for set_record in sets:
        claimed = first_server_by_set.get(set_record.set_number)
        if claimed is None or not set_record.points:
            continue
        claimed_me = claimed == "me"
        actual_me = set_record.points[0].is_serving
        if claimed_me != actual_me:
            notes.append(
                f"Set {set_record.set_number}: you said "
                f"{'you' if claimed_me else 'the opponent'} served first, but "
                f"reconstruction has {'you' if actual_me else 'the opponent'} serving "
                "first — host/guest may be reversed for this set."
            )
    return notes


def check_tracked_identity(
    settings: RawSettings | None, claimed_identity: str | None
) -> list[str]:
    """Flags if the user's self-reported identity doesn't match Settings.

    Args:
        settings: Parsed Settings-sheet data, or None.
        claimed_identity: The name the user entered at intake as "who I am
            in this recording", or None if not supplied.

    Returns:
        A single note if both values are present and don't match
        (case/whitespace-insensitively); empty otherwise.
    """
    if settings is None or not claimed_identity:
        return []
    if claimed_identity.strip().lower() != settings.host_name.strip().lower():
        return [
            f"You identified yourself as '{claimed_identity}', but SwingVision's "
            f"Settings sheet has the tracked player as '{settings.host_name}' — "
            "double check this export is really yours before trusting is_serving/"
            "point_won."
        ]
    return []


def check_footage_rally_count(
    sets: list[SetRecord], footage_rally_count: int | None
) -> list[str]:
    """Cross-checks the uploader's rally count against the points rebuilt.

    The rallies-only video holds fewer rallies than the import has points,
    and why is not established: on the one export measured (2026-08-21),
    the sheet's own Point counter restarts at every serve attempt, so a
    first-serve fault is counted as a point of its own, and no single
    definition of "rally" tested against the sheet reproduces the count.
    So a count *below* the number of points is reported as the footage's
    coverage ceiling without naming a cause, while a count *above* it
    cannot be explained by the export format at all - the import is
    missing points (recording gaps), or the count includes non-match
    rallies such as a warm-up.

    Args:
        sets: The match's sets, reconstructed or direct-parsed.
        footage_rally_count: Rallies in the footage, as the uploader
            counted them, or None if they did not say.

    Returns:
        A single note when a usable count was supplied and the match has
        points; empty otherwise.
    """
    if footage_rally_count is None or footage_rally_count <= 0:
        return []
    points = sum(len(set_record.points) for set_record in sets)
    if points == 0:
        return []
    if footage_rally_count > points:
        return [
            f"You counted {footage_rally_count} rallies in the footage, but this import "
            f"has only {points} points. Every rally is a point, so at least "
            f"{footage_rally_count - points} point(s) are missing from the import "
            "(most likely recording gaps) - or the count includes rallies that were "
            "not part of the match, such as a warm-up."
        ]
    absent = points - footage_rally_count
    if absent == 0:
        return [
            f"The footage holds {footage_rally_count} rallies and this import has "
            f"{points} points - every point has a rally to check against."
        ]
    return [
        f"The footage holds {footage_rally_count} rallies against {points} points in "
        f"this import, so a footage check can reach at most {footage_rally_count} of "
        f"{points} points ({round(100 * footage_rally_count / points)}%). At least "
        f"{absent} point(s) in this import have no rally in the footage. Which ones "
        "has not been established - do not assume it is the serve- or return-decided "
        "points."
    ]
