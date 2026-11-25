"""CLI: a fixed-camera match video -> staged pending-review JSON, no SwingVision.

Usage:
    python scripts/track_match.py match.mp4 --host "Rishi" --guest "Alex" \\
        --host-starts near --result W [--date 2026-08-21] [--opponent Alex]

    # Part of the recording only, into a scratch folder:
    python scripts/track_match.py match.mp4 --host Rishi --guest Alex \\
        --host-starts far --result L --start 355 --end 600 \\
        --pending-dir C:/scratch/pending

    # Automatic calibration abstained? Mark the court by hand:
    python scripts/track_match.py match.mp4 ... --keypoints court.json

Tracking is slow - roughly 80ms a 1080p frame on a laptop CPU, hours for a
whole match - so it runs in chunks, each saved as it finishes. If the run
is interrupted (Ctrl+C, a crash, the machine sleeping), run the same
command again: finished chunks are not redone (--resume, the default).
--force discards the checkpoints and any staged file for this video and
starts over. A video that is already staged is skipped unless --force.

Only stages the match (ingest). Review, the footage check and finalize
are the existing steps, unchanged:
    python scripts/resolve_reviews.py <pending.json> ...
    python scripts/verify_footage.py <pending.json> match.mp4
then SwingVisionImportPipeline.finalize(<pending.json>).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date as dates
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from logging_config import configure_logging  # noqa: E402
from swingvision_import.config import ImportConfig  # noqa: E402
from swingvision_import.review import load_pending, unresolved_flags  # noqa: E402
from tracking import video as video_stage  # noqa: E402
from tracking.assemble import MatchSetup, TrackingBlock  # noqa: E402
from tracking.config import PipelineConfig, TrackerConfig  # noqa: E402
from tracking.pipeline import (  # noqa: E402
    ManualKeypoints,
    TrackerPipeline,
    TrackerStages,
    TrackingError,
    TrackingProgress,
)


def _seconds(value: str) -> float:
    """argparse type for a non-negative number of seconds.

    Args:
        value: The raw argument.

    Returns:
        The seconds.

    Raises:
        argparse.ArgumentTypeError: If it is not a number of at least 0.
    """
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not a number of seconds: {value!r}") from error
    if seconds < 0:
        raise argparse.ArgumentTypeError("seconds cannot be negative")
    return seconds


def _rating(value: str) -> int:
    """argparse type for a 1-5 self-assessment rating.

    Args:
        value: The raw argument.

    Returns:
        The rating.

    Raises:
        argparse.ArgumentTypeError: If it is not a whole number from 1 to 5,
            which data/schema.sql's CHECK would otherwise reject only at
            finalize, after every point had been reviewed.
    """
    try:
        rating = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not a whole number: {value!r}") from error
    if not 1 <= rating <= 5:
        raise argparse.ArgumentTypeError("ratings run from 1 to 5")
    return rating


def _first_server(value: str) -> tuple[int, str]:
    """argparse type for one set's first server, written SET=me|opponent.

    Args:
        value: The raw argument, e.g. "1=me".

    Returns:
        (set number, "me" or "opponent").

    Raises:
        argparse.ArgumentTypeError: If it is not in that form.
    """
    number, _, who = value.partition("=")
    if not number.strip().isdigit() or who.strip() not in ("me", "opponent"):
        raise argparse.ArgumentTypeError(f"expected SET=me or SET=opponent, got {value!r}")
    return int(number), who.strip()


def build_parser() -> argparse.ArgumentParser:
    """Defines the command line.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video", type=Path, help="The fixed-camera recording.")
    parser.add_argument("--host", required=True, help="Your name - the tracked player.")
    parser.add_argument("--guest", required=True, help="Your opponent's name.")
    parser.add_argument(
        "--host-starts",
        required=True,
        choices=["near", "far"],
        help="Which end you were at (near = the camera's end) when tracking begins.",
    )
    parser.add_argument("--host-hand", choices=["right", "left"], default="right")
    parser.add_argument("--guest-hand", choices=["right", "left"], default="right")
    parser.add_argument("--result", required=True, choices=["W", "L"], help="Match result.")
    parser.add_argument(
        "--date", help="ISO match date; the video file's modification date if omitted."
    )
    parser.add_argument("--opponent", help="Opponent's name for the record; --guest if omitted.")
    parser.add_argument("--start", type=_seconds, help="First second of video to track.")
    parser.add_argument("--end", type=_seconds, help="Last second of video to track.")
    again = parser.add_mutually_exclusive_group()
    again.add_argument(
        "--resume",
        action="store_true",
        help="Carry on from finished chunks (the default; here for clarity).",
    )
    again.add_argument(
        "--force",
        action="store_true",
        help="Discard checkpoints and any staged file for this video, and start over.",
    )
    parser.add_argument(
        "--keypoints",
        type=Path,
        help="JSON file of court keypoints marked by hand (name -> [x, y] pixels), used "
        "instead of automatic calibration.",
    )
    parser.add_argument("--tracked-identity", help="Your name, cross-checked against --host.")
    parser.add_argument(
        "--player-appearance",
        help='What you were wearing, e.g. "grey shirt" - helps the footage check.',
    )
    parser.add_argument(
        "--first-server",
        type=_first_server,
        action="append",
        default=[],
        metavar="SET=WHO",
        help="Who served first in a set (me or opponent), e.g. 1=me; repeat per set. "
        "Cross-checked against the tracked serve order.",
    )
    parser.add_argument("--energy-rating", type=_rating, help="Your energy, 1-5.")
    parser.add_argument("--mental-rating", type=_rating, help="Your mental state, 1-5.")
    parser.add_argument("--location", help="Where the match was played.")
    parser.add_argument(
        "--reviewer-note", help="Anything odd about this recording, shown during review."
    )
    parser.add_argument("--pending-dir", type=Path, help="Where to stage the match.")
    parser.add_argument("--work-dir", type=Path, help="Where to keep chunk checkpoints.")
    parser.add_argument(
        "--chunk-seconds", type=_seconds, default=PipelineConfig.chunk_seconds,
        help="Seconds of video per checkpointed chunk.",
    )
    parser.add_argument(
        "--overlap-seconds", type=_seconds, default=PipelineConfig.overlap_seconds,
        help="Extra seconds read on each side of a chunk.",
    )
    parser.add_argument(
        "--no-pose", action="store_true", help="Name strokes from the ball alone (faster)."
    )
    return parser


def build_pipeline(args: argparse.Namespace) -> TrackerPipeline:
    """Builds the pipeline the arguments ask for.

    Args:
        args: Parsed arguments.

    Returns:
        The pipeline.
    """
    pipeline = PipelineConfig(
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        work_dir=args.work_dir,
    )
    imports = ImportConfig(pending_dir=args.pending_dir) if args.pending_dir else ImportConfig()
    stages = TrackerStages(use_pose=not args.no_pose)
    return TrackerPipeline(TrackerConfig(pipeline=pipeline), imports, stages)


def _clock(seconds: float | None) -> str:
    """Formats a duration as h:mm:ss, or "?" when unknown."""
    if seconds is None:
        return "?"
    whole = int(round(seconds))
    return f"{whole // 3600}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


def format_progress(progress: TrackingProgress) -> str:
    """One line describing a progress report.

    Args:
        progress: The report.

    Returns:
        The line.
    """
    line = f"[{progress.stage}]"
    if progress.chunks_total and progress.video_seconds:
        line += (
            f" {progress.chunks_done}/{progress.chunks_total} chunks"
            f" ({progress.video_seconds_done / 60:.1f} of {progress.video_seconds / 60:.1f}"
            " min of video)"
        )
    if progress.seconds_per_video_minute is not None:
        line += (
            f", {progress.seconds_per_video_minute:.0f}s per video minute,"
            f" ETA {_clock(progress.eta_seconds)}"
        )
    return f"{line} {progress.message}".rstrip()


def _match_date(args: argparse.Namespace) -> str:
    """The match date: as given, or the video file's modification date."""
    if args.date:
        return args.date
    stamp = args.video.stat().st_mtime
    found = dates.fromtimestamp(stamp).isoformat()
    print(f"No --date given: using {found}, the video file's modification date.")
    return found


def main(argv: list[str] | None = None) -> int:
    """Parses the arguments, tracks and stages the match, and reports.

    Args:
        argv: Arguments; the process's own when omitted.

    Returns:
        The exit status: 0 when staged (or already staged), 2 on a problem
        the user can fix.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging()
    if not args.video.is_file():
        parser.exit(2, f"error: no such video: {args.video}\n")
    try:
        setup = MatchSetup.from_start_end(
            args.host,
            args.guest,
            args.host_starts,
            host_hand=args.host_hand,
            guest_hand=args.guest_hand,
        )
        keypoints = ManualKeypoints.from_file(args.keypoints) if args.keypoints else None
        overrides = {
            field: getattr(args, field)
            for field in (
                "player_appearance",
                "energy_rating",
                "mental_rating",
                "location",
                "reviewer_note",
            )
            if getattr(args, field) not in (None, "")
        }
        json_path = build_pipeline(args).ingest(
            args.video,
            setup,
            date=_match_date(args),
            opponent=args.opponent or args.guest,
            result=args.result,
            first_server_by_set=dict(args.first_server) or None,
            tracked_identity=args.tracked_identity,
            start=args.start,
            end=args.end,
            keypoints=keypoints,
            force=args.force,
            progress=lambda progress: print(format_progress(progress), flush=True),
            **overrides,
        )
    except (TrackingError, ValueError, OSError, video_stage.VideoError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Staged: {json_path}")
    record = load_pending(json_path)
    if record.tracking:
        block = TrackingBlock.from_dict(record.tracking)
        if block.notes:
            print(f"Coverage: {block.notes[0]}")
    flags = unresolved_flags(record)
    print(f"{len(flags)} point(s) need review before this match can be finalized.")
    for note in record.import_notes:
        print(f"NOTE: {note}")
    print(
        "Next: review the points (scripts/resolve_reviews.py or the browser), optionally "
        f"check them against the footage (scripts/verify_footage.py {json_path} "
        f"{args.video}), then finalize."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
