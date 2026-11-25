"""CLI: measure a recording's court conditions, and build the ball detector's reference.

The "retraining recommended" note compares a tracked match's court
brightness and colour with the conditions the ball detector was tuned or
trained on (`tracking.conditions`). This script measures those
conditions on a reference recording, cheaply - one seek per sample, never
a full decode - and writes the reference file.

Usage:
    # 1. Sample a stretch (repeat over several ranges; samples are appended,
    #    so a long match can be done in short, separate processes):
    python scripts/measure_conditions.py sample match.mp4 samples.jsonl \\
        --calibration calibration.json --start 0 --end 1200 --every 30

    # 2. Build the reference from every sample collected:
    python scripts/measure_conditions.py build samples.jsonl match.mp4 \\
        --detector motion-ball [--tracking-baseline baseline.json] \\
        [--out src/tracking/models/ball_conditions_reference.json]

After retraining the learned ball model (docs/ball-model-training.md), run
both steps over the recordings it was trained on, with `--detector` set to
the new model's registry name, so the note describes the model in use.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date as dates
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from logging_config import configure_logging  # noqa: E402
from tracking import conditions  # noqa: E402
from tracking import video as video_stage  # noqa: E402
from tracking.calibrate import Calibration, calibrate_video  # noqa: E402


def _load_calibration(args: argparse.Namespace) -> Calibration | None:
    """The court calibration the samples are taken over.

    Args:
        args: Parsed arguments.

    Returns:
        The calibration from `--calibration`, else one made at
        `--calibrate-at`, else None (the whole frame is summarised).

    Raises:
        ValueError: If automatic calibration abstains.
    """
    if args.calibration:
        return Calibration.from_dict(json.loads(Path(args.calibration).read_text("utf-8")))
    if args.calibrate_at is None:
        return None
    result = calibrate_video(args.video, args.calibrate_at)
    if not result.ok:
        raise ValueError(f"Calibration abstained: {result.reason}")
    return result.calibration


def sample(args: argparse.Namespace) -> int:
    """Samples one stretch and appends the samples to a JSON-lines file.

    Args:
        args: Parsed arguments.

    Returns:
        The exit status.
    """
    calibration = _load_calibration(args)
    times = conditions.sample_times(args.start, args.end, args.every)
    samples, failed = conditions.sample_video(args.video, times, calibration)
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    with args.samples.open("a", encoding="utf-8") as handle:
        for item in samples:
            handle.write(json.dumps(item.to_dict()) + "\n")
        for moment in failed:
            handle.write(json.dumps({"failed": moment}) + "\n")
    print(f"{len(samples)} sample(s) written to {args.samples}; {len(failed)} undecodable.")
    return 0


def read_samples(path: Path) -> tuple[list[conditions.ConditionSample], int]:
    """Reads a samples file.

    Args:
        path: The JSON-lines file `sample` appends to.

    Returns:
        (samples in time order, duplicates removed; count of failed
        moments).
    """
    found: dict[float, conditions.ConditionSample] = {}
    failed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if "failed" in data:
            failed.add(float(data["failed"]))
        else:
            item = conditions.ConditionSample.from_dict(data)
            found[round(item.pts, 3)] = item
    return [found[key] for key in sorted(found)], len(failed)


def build(args: argparse.Namespace) -> int:
    """Builds and writes the reference from a samples file.

    Args:
        args: Parsed arguments.

    Returns:
        The exit status.
    """
    samples, failed = read_samples(args.samples)
    info = video_stage.probe(args.video)
    identity = {
        "name": args.video.name,
        "sha256": args.sha256 or video_stage.file_checksum(args.video),
        "duration": info.duration,
        "width": info.width,
        "height": info.height,
    }
    tracking = None
    if args.tracking_baseline:
        tracking = json.loads(Path(args.tracking_baseline).read_text(encoding="utf-8"))
    span = (samples[0].pts, samples[-1].pts) if samples else (0.0, 0.0)
    reference = conditions.build_reference(
        samples,
        video=identity,
        detector=args.detector,
        measured=args.measured or dates.today().isoformat(),
        method=(
            f"{len(samples)} single frames, one seek each, from {span[0]:.0f}s to "
            f"{span[1]:.0f}s; court surface (near half, through the calibration) only."
        ),
        failed_samples=failed,
        tracking=tracking,
    )
    path = reference.save(args.out)
    print(f"Reference written to {path}:")
    print(json.dumps(reference.to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The command line.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("sample", help="sample a stretch of a recording")
    one.add_argument("video", type=Path)
    one.add_argument("samples", type=Path, help="JSON-lines file samples are appended to")
    one.add_argument("--calibration", type=Path, help="a Calibration.to_dict JSON file")
    one.add_argument("--calibrate-at", type=float, help="calibrate automatically here")
    one.add_argument("--start", type=float, default=0.0)
    one.add_argument("--end", type=float, required=True)
    one.add_argument("--every", type=float, default=30.0, help="seconds between samples")
    one.set_defaults(handler=sample)
    two = commands.add_parser("build", help="build the reference file from samples")
    two.add_argument("samples", type=Path)
    two.add_argument("video", type=Path, help="the reference recording, for its identity")
    two.add_argument("--detector", required=True, help="the ball detector it describes")
    two.add_argument("--sha256", help="the recording's SHA-256, if already known")
    two.add_argument("--tracking-baseline", type=Path, help="JSON of measured ball coverage")
    two.add_argument("--measured", help="ISO date of the measurement (today by default)")
    two.add_argument("--out", type=Path, default=conditions.REFERENCE_PATH)
    two.set_defaults(handler=build)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Runs the chosen step.

    Args:
        argv: Arguments; the process's own when omitted.

    Returns:
        The exit status: 0 on success, 2 on a problem the user can fix.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging()
    try:
        return args.handler(args)
    except (ValueError, OSError, video_stage.VideoError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
