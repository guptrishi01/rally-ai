"""CLI: check a staged match's flagged points against its footage.

Usage:
    # Judge through the signed-in Claude Code subscription (no API key):
    python scripts/verify_footage.py path/to/pending.json match.mp4 --via claude-code

    # Judge through the Anthropic API (spends ANTHROPIC_API_KEY credit):
    python scripts/verify_footage.py path/to/pending.json match.mp4 --via api

    # Try it on a few points first:
    python scripts/verify_footage.py path/to/pending.json match.mp4 --limit 10

Give one video per export part, in play order. Points that already carry
a footage verdict are skipped, so an interrupted run can simply be run
again. Verdicts are saved into the pending JSON as the run goes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from logging_config import configure_logging  # noqa: E402
from scripts.client import FOOTAGE_BACKENDS, get_footage_client  # noqa: E402
from swingvision_import import footage  # noqa: E402
from swingvision_import.config import ImportConfig  # noqa: E402
from swingvision_import.pipeline import SwingVisionImportPipeline  # noqa: E402
from swingvision_import.review import load_pending, save_pending  # noqa: E402


def _hold_back(json_path: Path, limit: int) -> list[tuple[int, int]]:
    """Unflags all but the first `limit` unchecked points, for a trial run.

    Args:
        json_path: The staged match.
        limit: How many unchecked flagged points to leave eligible.

    Returns:
        (set index, point index) of every point that was unflagged, so
        the flags can be put back afterwards.
    """
    record = load_pending(json_path)
    held: list[tuple[int, int]] = []
    eligible = 0
    for set_index, set_record in enumerate(record.sets):
        for point_index, point in enumerate(set_record.points):
            if not point.needs_review or point.footage_check is not None:
                continue
            eligible += 1
            if eligible > limit:
                point.needs_review = False
                held.append((set_index, point_index))
    save_pending(record, json_path.parent)
    return held


def _restore(json_path: Path, held: list[tuple[int, int]]) -> None:
    """Puts back the flags `_hold_back` took off."""
    record = load_pending(json_path)
    for set_index, point_index in held:
        record.sets[set_index].points[point_index].needs_review = True
    save_pending(record, json_path.parent)


def main() -> None:
    """Parses CLI args and runs the footage check."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("json_path", type=Path, help="Path to a staged pending-review JSON file.")
    parser.add_argument("videos", type=Path, nargs="+", help="The recording(s), in play order.")
    parser.add_argument(
        "--via",
        choices=sorted(FOOTAGE_BACKENDS),
        default="claude-code",
        help="claude-code uses the signed-in subscription; api spends API credit.",
    )
    parser.add_argument("--limit", type=int, help="Only check this many points (a trial run).")
    parser.add_argument("--workers", type=int, help="Points checked at once.")
    parser.add_argument("--model", help="Model id to judge with.")
    args = parser.parse_args()

    configure_logging()
    json_path = args.json_path.resolve()
    pipeline = SwingVisionImportPipeline(ImportConfig(pending_dir=json_path.parent))
    overrides = {
        key: value
        for key, value in (("workers", args.workers), ("model", args.model))
        if value is not None
    }
    config = footage.FootageConfig(**overrides)
    client = get_footage_client(args.via)

    held = _hold_back(json_path, args.limit) if args.limit else []
    try:
        summary = pipeline.verify_footage(
            client,
            json_path,
            args.videos,
            footage_config=config,
            progress=lambda s: print(
                f"  {s.checked} checked, {s.agreed} confirmed, {s.failed} failed", flush=True
            ),
        )
    finally:
        if held:
            _restore(json_path, held)

    print(
        f"{summary.checked} of {summary.points_played} points played were checked: "
        f"{summary.agreed} confirmed ({summary.corroborated} by two readings), "
        f"{summary.disagreed} disagreed, {summary.inconclusive} unclear, "
        f"{summary.failed} failed."
    )
    if args.via == "claude-code":
        print(
            f"Judged through Claude Code in {client.calls} call(s): no API credit spent "
            f"(about ${client.list_price_usd:.2f} at list price, for reference)."
        )
    else:
        print(f"API cost: ${summary.usd:.2f}")
    for note in summary.notes:
        print(f"Note: {note}")


if __name__ == "__main__":
    main()
