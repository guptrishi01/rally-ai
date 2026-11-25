"""CLI: export a tracked match's ball labels (and frames) for training the ball model.

Usage:
    python scripts/export_ball_dataset.py <pending.json> <out_dir> \\
        [--split train] [--min-confidence 0.4] [--frames labelled|all|none] \\
        [--corrections corrected_labels.csv] [--confirmed-only]

`<pending.json>` is a match staged by `scripts/track_match.py`; its
checkpoint folder (kept by default) supplies the ball's per-frame track,
and the video named in it supplies the frames. Writes:

    <out_dir>/labels.csv      one row per frame (tracking.dataset.LABEL_COLUMNS)
    <out_dir>/manifest.json   source video SHA-256, detector, split, counts
    <out_dir>/frames/<clip>/<frame>.jpg

Labels from the tracker are pseudo-labels, marked as such. To correct them,
edit labels.csv in a labelling tool (or by hand): set label_source to
"human" on every row you checked or fixed, then re-export with
`--corrections` to merge them. Nothing is trained here - training runs on a
cloud GPU (docs/ball-model-training.md). The output folder holds frames of
the user's own footage: keep it out of the repository.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date as dates
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import cv2  # noqa: E402

from logging_config import configure_logging  # noqa: E402
from swingvision_import.review import load_pending  # noqa: E402
from tracking import dataset  # noqa: E402
from tracking import video as video_stage  # noqa: E402
from tracking.pipeline import CheckpointStore  # noqa: E402
from tracking.stitch import CompactBallTrack  # noqa: E402


def load_ball_track(work_folder: Path) -> CompactBallTrack:
    """Joins the ball track of every chunk checkpoint in a run's folder.

    Args:
        work_folder: The run's checkpoint folder.

    Returns:
        The whole run's ball state.

    Raises:
        ValueError: If no readable chunk checkpoint is there.
    """
    store = CheckpointStore(work_folder)
    chunks = []
    for path in sorted(Path(work_folder).glob("chunk_*.pkl")):
        chunk = store.load_chunk(int(path.stem.split("_")[1]))
        if chunk is not None:
            chunks.append(chunk)
    if not chunks:
        raise ValueError(
            f"No chunk checkpoints in {work_folder}: the ball track is needed to label frames. "
            "Track the match again with checkpoints kept (the default)."
        )
    return CompactBallTrack.concatenate([chunk.ball for chunk in chunks])


def write_frames(
    video: Path, rows: list[dataset.LabelRow], out_dir: Path, which: str, reader=None
) -> int:
    """Decodes the clips and saves the frames the labels name.

    Args:
        video: The recording.
        rows: The labels.
        out_dir: The dataset folder.
        which: "labelled" (frames with a position), "all", or "none".
        reader: (video, start, end) -> frames; `video.iter_frames` by
            default. Tests inject one.

    Returns:
        How many images were written.
    """
    if which == "none":
        return 0
    wanted = [r for r in rows if which == "all" or r.visibility == 1]
    read = reader or (lambda path, start, end: video_stage.iter_frames(path, start, end))
    written = 0
    by_clip: dict[str, list[dataset.LabelRow]] = {}
    for row in wanted:
        by_clip.setdefault(row.clip, []).append(row)
    for clip_rows in by_clip.values():
        files = {round(r.pts, 6): out_dir / r.file for r in clip_rows}
        low, high = min(r.pts for r in clip_rows), max(r.pts for r in clip_rows)
        for frame in read(video, low - 0.05, high + 0.05):
            target = files.get(round(frame.pts, 6))
            if target is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR))
            written += 1
    return written


def build_parser() -> argparse.ArgumentParser:
    """The command line.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pending", type=Path, help="a match staged by the tracker")
    parser.add_argument("out_dir", type=Path, help="the dataset folder (outside the repo)")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--frames", choices=("labelled", "all", "none"), default="labelled")
    parser.add_argument("--corrections", type=Path, help="a labels.csv with human rows")
    parser.add_argument("--confirmed-only", action="store_true", help="only reviewed points")
    parser.add_argument("--work-folder", type=Path, help="the run's checkpoints, if moved")
    parser.add_argument("--video", type=Path, help="the recording, if moved")
    return parser


def main(argv: list[str] | None = None, reader=None) -> int:
    """Exports the labels, manifest and frames.

    Args:
        argv: Arguments; the process's own when omitted.
        reader: Frame reader for `write_frames` (tests).

    Returns:
        The exit status: 0 on success, 2 on a problem the user can fix.
    """
    args = build_parser().parse_args(argv)
    configure_logging()
    try:
        record = load_pending(args.pending)
        clips = dataset.clips_from_record(record)
        if args.confirmed_only:
            clips = [clip for clip in clips if clip.confirmed]
        tracking = record.tracking
        run = tracking.get("run") or {}
        folder = args.work_folder or Path(run.get("work_folder", ""))
        ball = load_ball_track(folder)
        rows = dataset.pseudo_labels(
            ball, clips, min_confidence=args.min_confidence, split=args.split
        )
        if args.corrections is not None:
            rows = dataset.apply_corrections(rows, dataset.read_labels(args.corrections))
        video_info = tracking.get("video") or {}
        video = args.video or Path(video_info.get("path", ""))
        written = write_frames(video, rows, args.out_dir, args.frames, reader)
    except (ValueError, OSError, KeyError, video_stage.VideoError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    dataset.write_labels(args.out_dir / "labels.csv", rows)
    data = dataset.manifest(
        rows,
        video={
            "name": video_info.get("name"),
            "sha256": video_info.get("checksum"),
            "width": video_info.get("width"),
            "height": video_info.get("height"),
        },
        detector=(tracking.get("models") or {}).get("ball", "unknown"),
        split=args.split,
        min_confidence=args.min_confidence,
        calibration=tracking.get("calibration"),
        created=dates.today().isoformat(),
    )
    dataset.write_manifest(args.out_dir / "manifest.json", data)
    print(
        f"{data['frames']} frame label(s) in {data['clips']} clip(s): {data['by_source']}; "
        f"{written} image(s) written to {args.out_dir}."
    )
    print(
        "Tracker labels are pseudo-labels from the detector being replaced: check a sample "
        "and mark checked rows label_source=human before training on them."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
