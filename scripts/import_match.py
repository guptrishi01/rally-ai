"""CLI: SwingVision .xlsx (or multiple, for an interrupted recording) ->
staged pending-review JSON.

Usage:
    python scripts/import_match.py first-half.xlsx --date 2026-08-18 \\
        --opponent "Real Opponent" --result W [--suggest]

    python scripts/import_match.py first-half.xlsx second-half.xlsx \\
        --date 2026-08-18 --opponent "Real Opponent" --result W \\
        --first-server 1:opponent --first-server 2:me \\
        --tracked-identity "Rishi Gupta"

Only stages the match for review (ingest) - finalize() into SQL is a
separate, deliberate step (not run here), same two-stage design the
pipeline itself enforces. A single xlsx_path routes through
pipeline.ingest(); two or more route through pipeline.ingest_multi_part(),
which merges them into one continuous reconstruction for a match whose
recording was cut and resumed as separate files. --suggest additionally
runs Claude-assisted review suggestions on the staged points; it spends
real API money, so it's opt-in via this flag rather than always running.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from logging_config import configure_logging  # noqa: E402
from scripts.client import get_anthropic_client  # noqa: E402
from swingvision_import.pipeline import SwingVisionImportPipeline  # noqa: E402
from swingvision_import.records import parse_footage_rally_count  # noqa: E402
from swingvision_import.review import load_pending, unresolved_flags  # noqa: E402


def _parse_first_server(values: list[str] | None) -> dict[int, str] | None:
    """Parses repeated --first-server SET:WHO args into a set_number -> who map.

    Args:
        values: Raw "SET:WHO" strings from argparse, e.g. ["1:opponent", "2:me"].

    Returns:
        The parsed map, or None if no --first-server args were given.

    Raises:
        SystemExit: Via argparse.ArgumentTypeError-style failure if a value
            isn't in "SET:WHO" form.
    """
    if not values:
        return None
    result: dict[int, str] = {}
    for value in values:
        set_number_str, _, who = value.partition(":")
        if not who or not set_number_str.isdigit():
            raise argparse.ArgumentTypeError(
                f"--first-server value must be SET:WHO (e.g. 1:opponent), got {value!r}"
            )
        result[int(set_number_str)] = who
    return result


def _rally_count(value: str) -> int:
    """argparse type for --footage-rally-count.

    Args:
        value: The raw argument.

    Returns:
        The count as a positive int.

    Raises:
        argparse.ArgumentTypeError: If it is not a whole number of at least 1.
    """
    try:
        count = parse_footage_rally_count(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if count is None:
        raise argparse.ArgumentTypeError("The rally count must be a whole number of at least 1.")
    return count


def main() -> None:
    """Parses CLI args, runs ingest (and optionally suggest), and reports flags."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "xlsx_paths",
        type=Path,
        nargs="+",
        help="One export, or multiple (in play order) for a match split across files.",
    )
    parser.add_argument("--date", required=True, help="ISO-format match date.")
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--result", required=True, choices=["W", "L"])
    parser.add_argument(
        "--first-server",
        action="append",
        metavar="SET:WHO",
        help='Ground truth for a set\'s first server, e.g. "1:opponent" or "2:me". '
        "Repeatable, one per set.",
    )
    parser.add_argument(
        "--tracked-identity",
        help="Your name as tracked by SwingVision, cross-checked against the Settings sheet.",
    )
    parser.add_argument(
        "--player-appearance",
        help='What you were wearing, e.g. "grey shirt, black shorts" - how a footage '
        "check tells you apart from your opponent.",
    )
    parser.add_argument(
        "--footage-rally-count",
        type=_rally_count,
        metavar="N",
        help="Total rallies in the match, from the footage (SwingVision's "
        '"all rallies" export states it). Sets the ceiling a footage check can reach.',
    )
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Also run Claude-assisted review suggestions (spends real API money).",
    )
    args = parser.parse_args()

    configure_logging()

    pipeline = SwingVisionImportPipeline()
    ingest_kwargs = {
        "date": args.date,
        "opponent": args.opponent,
        "result": args.result,
        "first_server_by_set": _parse_first_server(args.first_server),
        "tracked_identity": args.tracked_identity,
    }
    for name in ("player_appearance", "footage_rally_count"):
        if getattr(args, name) is not None:
            ingest_kwargs[name] = getattr(args, name)
    if len(args.xlsx_paths) == 1:
        json_path = pipeline.ingest(args.xlsx_paths[0], **ingest_kwargs)
    else:
        json_path = pipeline.ingest_multi_part(args.xlsx_paths, **ingest_kwargs)
    print(f"Staged: {json_path}")

    if args.suggest:
        client = get_anthropic_client()
        pipeline.suggest(client, json_path)
        print("Suggestions added.")

    record = load_pending(json_path)
    flags = unresolved_flags(record)
    print(f"{len(flags)} point(s) still need review before this match can be finalized.")
    for note in record.import_notes:
        print(f"NOTE: {note}")


if __name__ == "__main__":
    main()
