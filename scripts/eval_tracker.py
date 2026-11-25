"""CLI: score a tracked match against a SwingVision export and the reviewer's confirmations.

Usage:
    python scripts/eval_tracker.py <pending.json> --reference export.xlsx \\
        [--clock-offset 1.99] [--start 355 --end 600] [--tolerance 0.15] \\
        [--json results.json]

    # No export? Score against the points a person has confirmed in review:
    python scripts/eval_tracker.py <pending.json>

`<pending.json>` is a match staged by `scripts/track_match.py` - a whole
match, or a `--start/--end` stretch of one. The export is a *weak*
reference: SwingVision's own tracking has errors and gaps, so its numbers
are reported as agreement, never as accuracy. Confirmed points (the review
modal's "confirm") are human judgements and are always scored when there
are any.

`--clock-offset` is added to the export's `Video Time` to put it on the
video's clock (1.99s on the 2026-08-21 recording, measured from its audio).
Without it the offset is estimated by pairing shots, and the output says
so.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from logging_config import configure_logging  # noqa: E402
from swingvision_import.config import ImportConfig  # noqa: E402
from swingvision_import.parse import SwingVisionParser  # noqa: E402
from swingvision_import.raw import RawShotRow  # noqa: E402
from swingvision_import.review import load_pending  # noqa: E402
from tracking import evaluate  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """The command line.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pending", type=Path, help="a match staged by the tracker")
    parser.add_argument("--reference", type=Path, help="a SwingVision .xlsx of the same match")
    parser.add_argument("--clock-offset", type=float, help="added to the export's times")
    parser.add_argument("--start", type=float, help="score from here (video clock, seconds)")
    parser.add_argument("--end", type=float, help="score up to here")
    parser.add_argument("--tolerance", type=float, default=0.15, help="shot pairing, seconds")
    parser.add_argument("--serve-tolerance", type=float, default=0.3, help="serve pairing")
    parser.add_argument("--json", type=Path, help="also write the results here as JSON")
    return parser


def evaluate_staged(args: argparse.Namespace) -> dict:
    """Runs every comparison the inputs allow and prints them.

    Args:
        args: Parsed arguments.

    Returns:
        The results, JSON-ready: "reference" (or None) and "confirmed".

    Raises:
        ValueError: If the staged match was not made by the tracker.
    """
    record = load_pending(args.pending)
    if not record.tracking or record.tracking.get("shots") is None:
        raise ValueError(f"{args.pending.name} was not staged by the tracker: nothing to score.")
    tracking = record.tracking
    rows = [RawShotRow(**row) for row in tracking["shots"]]
    results: dict = {"pending": str(args.pending), "reference": None}
    if args.reference is not None:
        export = SwingVisionParser(ImportConfig()).parse(args.reference)
        tracked_range = tracking.get("range")
        window = None
        if args.start is not None or args.end is not None or tracked_range:
            low = args.start if args.start is not None else (tracked_range or [None, None])[0]
            high = args.end if args.end is not None else (tracked_range or [None, None])[1]
            window = (float(low), float(high))
        comparison = evaluate.compare_shots(
            export.shots,
            rows,
            reference_host=export.settings.host_name,
            tracker_host=(tracking.get("settings") or {}).get("host_name", ""),
            clock_offset=args.clock_offset,
            tolerance=args.tolerance,
            serve_tolerance=args.serve_tolerance,
            window=window,
            confidence=evaluate.tracker_confidence(tracking),
        )
        print("\n".join(comparison.lines()))
        results["reference"] = {"file": args.reference.name, **comparison.to_dict()}
    confirmed = evaluate.compare_confirmed(record)
    print("\n".join(confirmed.lines()))
    results["confirmed"] = confirmed.to_dict()
    return results


def main(argv: list[str] | None = None) -> int:
    """Scores the staged match and reports.

    Args:
        argv: Arguments; the process's own when omitted.

    Returns:
        The exit status: 0 when scored, 2 on a problem the user can fix.
    """
    args = build_parser().parse_args(argv)
    configure_logging()
    try:
        results = evaluate_staged(args)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
