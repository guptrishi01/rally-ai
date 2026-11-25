"""The tracker as a front door: one video in, one staged match out.

`TrackerPipeline.ingest` mirrors `SwingVisionImportPipeline.ingest`: it
stages a pending-review JSON file and never touches SQL. Everything after
staging - the review modal and `scripts/resolve_reviews.py`, the footage
check, `finalize` - runs unchanged, because the match is built exactly as
the SwingVision path builds one: the tracker's `RawShotRow`s go through
the same, unmodified `reconstruct.reconstruct_all`, every point arrives
`needs_review=True`, and the informational notes go into `import_notes`.

What is different about a tracked match
---------------------------------------
* `MatchRecord.source_file` is the video. The shot rows the tracker
  emitted live in `MatchRecord.tracking["shots"]`, and
  `swingvision_import.sources.parse_sources` hands them back wherever a
  later step would have re-read the export - so `verify_footage` works on
  a tracked match. Pass `clock_offset=assemble.video_clock_offset`: tracker
  times are the video's own timestamps, with nothing to measure.
* Each point carries its serve attempt's `tracking` block (a provenance
  trace per shot), and the match the run's: video, range, calibration,
  model versions, coverage, quality, and whether a retrained ball model is
  recommended (`retraining_advice`).

How a long recording is walked
------------------------------
At about 80ms a 1080p frame a 95-minute match is 7.5 hours of CPU, on a
machine short of memory. So the run is split into chunks (`stitch`); each
is read with overlap, tracked, and written to a checkpoint file as soon
as it is done. A run that is killed carries on from the first chunk with
no checkpoint - the repo's incremental-state rule: an existing output
means skip. Memory is bounded by one chunk's working set plus the
match's compact results, whatever the match's length. Progress (chunks
done, ETA) is reported through a callback, which is what a web request
would poll (see `TrackingProgress`).

Re-running `ingest` on a video that is already staged does nothing and
returns the staged file, unless `force` is given - which also discards
the checkpoints. Checkpoints written under different settings are
refused rather than mixed in (`CheckpointMismatchError`).

Nothing about the court is guessed: when calibration abstains, the run
stops with `CalibrationFailedError`, which says why and how to supply the
court's keypoints by hand (`ManualKeypoints`).
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import pickle
import shutil
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import chain
from pathlib import Path

import numpy as np

from swingvision_import import reconstruct, review
from swingvision_import.config import ImportConfig
from swingvision_import.pipeline import SwingVisionImportPipeline
from swingvision_import.records import MatchRecord

from . import assemble as assemble_stage
from . import ball as ball_stage
from . import calibrate, court, models
from . import conditions as conditions_stage
from . import deadtime as deadtime_stage
from . import events as events_stage
from . import players as players_stage
from . import strokes as strokes_stage
from . import video as video_stage
from .assemble import MatchSetup, TrackedMatch, TrackingBlock
from .ball import BallDetector, CandidateStream, SceneSampler, SceneStats, link_candidates
from .calibrate import Calibration, CalibrationResult
from .conditions import ConditionsReference
from .config import PipelineConfig, RetrainingConfig, TrackerConfig
from .deadtime import DeadTimePlan
from .events import AudioOnsets, EventQuality, EventTrack
from .players import PersonDetector, PlayerTracker, PlayerTracks
from .segment import EXCLUDED
from .stitch import (
    ChunkResult,
    ChunkWindow,
    CompactBallTrack,
    StitchedMatch,
    decode_holes,
    merge_chunks,
    plan_chunks,
    pose_key,
)
from .strokes import PoseEstimator, PoseSample, StrokeClassifier
from .video import PtsIndex, VideoFrame, VideoSource

logger = logging.getLogger(__name__)

SOURCE = "tracker-pipeline-v1"
PRODUCER = "independent-tracker"
CHECKPOINT_FORMAT = 1
_SEGMENT_STRIDE = 1_000_000


class TrackingError(Exception):
    """A tracking run could not go on; the message says what to do."""


class CalibrationFailedError(TrackingError):
    """The court could not be calibrated, so nothing on it can be placed."""


class CheckpointMismatchError(TrackingError):
    """Checkpoints exist from a run with different settings."""


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ManualKeypoints:
    """Court keypoints a person marked, for when automatic calibration abstains.

    Attributes:
        points: Keypoint name (`tracking.court.keypoint_names`, e.g.
            "near_baseline:left_singles") to its (x, y) pixel position.
        image_size: (width, height) of the frame they were marked on, or
            None for "the video's own size".
        template_kind: "doubles" or "singles".
    """

    points: Mapping[str, tuple[float, float]]
    image_size: tuple[int, int] | None = None
    template_kind: str = "doubles"

    @classmethod
    def from_dict(cls, data: Mapping) -> ManualKeypoints:
        """Reads keypoints from a JSON-shaped mapping.

        Two shapes are accepted: `{"points": {...}, "image_size": [w, h],
        "template": "doubles"}`, or the points mapping on its own.

        Args:
            data: The mapping.

        Returns:
            The keypoints.

        Raises:
            ValueError: If a point is not an (x, y) pair or the size is
                not (width, height).
        """
        if isinstance(data.get("points"), Mapping):
            raw, size = data["points"], data.get("image_size")
            template = data.get("template", "doubles")
        else:
            raw, size, template = data, None, "doubles"
        points = {}
        for name, value in raw.items():
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"Keypoint {name!r} must be an [x, y] pair, got {value!r}.")
            points[str(name)] = (float(value[0]), float(value[1]))
        if size is not None and (not isinstance(size, (list, tuple)) or len(size) != 2):
            raise ValueError(f"image_size must be [width, height], got {size!r}.")
        return cls(points, None if size is None else (int(size[0]), int(size[1])), template)

    @classmethod
    def from_file(cls, path: Path | str) -> ManualKeypoints:
        """Reads keypoints from a JSON file.

        Args:
            path: The file.

        Returns:
            The keypoints.

        Raises:
            ValueError: If the file is not valid JSON of either shape.
            OSError: If it cannot be read.
        """
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{Path(path).name}: not valid JSON ({error}).") from error
        if not isinstance(data, dict):
            raise ValueError(f"{Path(path).name}: expected a JSON object of keypoints.")
        return cls.from_dict(data)

    def identity(self) -> str:
        """A stable description, so checkpoints know which calibration made them."""
        payload = {
            "points": {k: list(v) for k, v in sorted(self.points.items())},
            "image_size": self.image_size,
            "template": self.template_kind,
        }
        return "manual:" + hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


@dataclass
class TrackerStages:
    """Every model and reader the pipeline uses, each injectable.

    Anything left None is built from the configuration when a run starts:
    the pinned YOLO person and pose models (falling back, with a note, to
    the classic motion detector and to no pose when they cannot be
    loaded), the classic motion ball detector, the rule stroke classifier,
    and the real video, audio and calibration readers. Tests pass fakes;
    no test needs torch or a model file.

    Attributes:
        person_detector: Finds people in a frame.
        pose_estimator: Finds a hitter's keypoints; None to load the
            pinned model (when `use_pose`).
        use_pose: False to judge strokes from the ball alone, without
            loading a pose model.
        ball_detector: (calibration, frame rate) -> a fresh detector for
            one chunk. Detectors keep state, so each chunk gets its own.
        stroke_classifier: Names each hit's stroke.
        event_detector: (ball, players, calibration, audio, config) ->
            `EventTrack`; `events.detect_events` by default.
        audio_reader: (video, start, end) -> onsets, or None when the
            audio cannot be read.
        frame_reader: (video, start, end, index) -> frames in order.
        video_loader: video -> `VideoSource` (header, PTS index, checksum).
        calibrator: (video, at) -> a `CalibrationResult`.
        audio_curve_reader: video -> the whole recording's raw onset curve
            (`deadtime.read_onset_curve`), or None without audio; used
            only by the dead-time pre-pass.
        alignment_judge: (video, current, candidate, start, end) ->
            (current's, candidate's) `calibrate.line_alignment_px` on the
            stretch's frames; decides whether a re-calibration is adopted.
    """

    person_detector: PersonDetector | None = None
    pose_estimator: PoseEstimator | None = None
    use_pose: bool = True
    ball_detector: Callable[[Calibration, float | None], BallDetector] | None = None
    stroke_classifier: StrokeClassifier | None = None
    event_detector: Callable[..., EventTrack] | None = None
    audio_reader: Callable[[Path, float, float], AudioOnsets | None] | None = None
    frame_reader: Callable[[Path, float, float, PtsIndex | None], Iterable[VideoFrame]] | None = (
        None
    )
    video_loader: Callable[[Path], VideoSource] | None = None
    calibrator: Callable[[Path, float], CalibrationResult] | None = None
    audio_curve_reader: Callable[[Path], np.ndarray | None] | None = None
    alignment_judge: (
        Callable[[Path, Calibration, Calibration, float, float], tuple[float | None, float | None]]
        | None
    ) = None


@dataclass
class TrackingProgress:
    """How far a run has got - what a CLI prints and a web job reports.

    A dataclass, so `webapp.jobs.JobRegistry.update` can store it as-is.

    Attributes:
        stage: "skipped", "opening", "calibrating", "tracking",
            "assembling" or "staged".
        chunks_total: Chunks in the run.
        chunks_done: Chunks finished, resumed ones included.
        chunks_resumed: Chunks taken from checkpoints rather than tracked.
        video_seconds: Length of the run's stretch of video.
        video_seconds_done: How much of it is finished.
        elapsed_seconds: Wall-clock seconds spent tracking in this run.
        eta_seconds: Estimated seconds still to go, once a chunk has been
            tracked in this run; None before.
        seconds_per_video_minute: Wall-clock seconds per minute of video,
            measured on the chunks tracked in this run.
        message: A sentence for a person.
    """

    stage: str
    chunks_total: int = 0
    chunks_done: int = 0
    chunks_resumed: int = 0
    video_seconds: float = 0.0
    video_seconds_done: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    seconds_per_video_minute: float | None = None
    message: str = ""


# --------------------------------------------------------------------------
# Retraining recommended?
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrainingAdvice:
    """Whether a ball model retrained on footage like this would help.

    Informational only: it never blocks staging, review or finalize.

    Attributes:
        recommended: Whether to say so.
        weak_tracking: Whether ball or event tracking was weak.
        conditions: How the footage differs from the reference the ball
            detector was tuned on, one plain phrase each; empty when it
            does not markedly (or could not be compared).
        measures: The numbers the decision was made on.
        note: The sentence for `import_notes`, or None.
        provisional: Whether any threshold used was a fallback rather
            than one derived from the measured reference.
        compared: Whether conditions were compared with a reference at
            all.
    """

    recommended: bool
    weak_tracking: bool
    conditions: tuple[str, ...]
    measures: dict
    note: str | None
    provisional: bool = False
    compared: bool = True

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field.
        """
        out = asdict(self)
        out["conditions"] = list(self.conditions)
        return out


def _weak_limits(
    reference: ConditionsReference | None, config: RetrainingConfig
) -> tuple[float, float, bool, dict | None]:
    """Coverage and confidence floors: relative to the reference when measured."""
    baseline = None if reference is None else reference.tracking
    if baseline and baseline.get("rally_coverage") and baseline.get("rally_mean_confidence"):
        return (
            config.relative_coverage * float(baseline["rally_coverage"]),
            config.relative_confidence * float(baseline["rally_mean_confidence"]),
            False,
            baseline,
        )
    return config.weak_coverage, config.weak_confidence, True, None


def retraining_advice(
    ball: CompactBallTrack,
    stretches: Iterable[tuple[float, float]],
    events: EventQuality,
    scene: SceneStats | None,
    config: RetrainingConfig | None = None,
    reference: ConditionsReference | None = None,
) -> RetrainingAdvice:
    """Decides whether to recommend retraining the ball model on this footage.

    Never blocking: the result is an informational note. The footage's
    look is compared with the measured reference (`conditions.compare`),
    and ball coverage with the reference's own measured coverage when it
    has one.

    Args:
        ball: The run's ball state.
        stretches: (start, end) of every point's rally - where coverage
            is judged, since most of a match is not play.
        events: The run's event quality, for how many results are unknown.
        scene: The run's court brightness and colour.
        config: Margins and fallbacks.
        reference: The conditions the detector was tuned on; None when
            there is no reference (conditions are then not compared, and
            the note says so).

    Returns:
        The advice, with the numbers it rests on.
    """
    config = config or RetrainingConfig()
    stretches = list(stretches)
    frames, seen, confidence = ball.coverage_over(stretches)
    coverage = seen / frames if frames else 0.0
    weak_points = sum(ball.quality(start, end).weak for start, end in stretches)
    floor, confidence_floor, fallback, baseline = _weak_limits(reference, config)
    reasons = []
    if frames and (coverage < floor or confidence < confidence_floor):
        against = (
            f" against {float(baseline['rally_coverage']):.0%} and "
            f"{float(baseline['rally_mean_confidence']):.2f} on the reference recording"
            if baseline
            else ""
        )
        reasons.append(
            f"a ball position in {coverage:.0%} of rally frames, mean confidence "
            f"{confidence:.2f}{against}"
        )
    if stretches and weak_points / len(stretches) > config.weak_point_share:
        reasons.append(f"weak on {weak_points} of {len(stretches)} points")
    if events.shots and events.abstention_rate > config.max_abstention:
        reasons.append(f"no result for {events.abstention_rate:.0%} of shots")
    check = conditions_stage.compare(scene, reference, config)
    measures = dict(check.measures)
    measures.update(
        {
            "rally_frames": frames,
            "rally_coverage": round(coverage, 3),
            "rally_mean_confidence": round(confidence, 3),
            "weak_points": weak_points,
            "points": len(stretches),
            "result_abstention": round(events.abstention_rate, 3),
            "coverage_floor": round(floor, 3),
            "confidence_floor": round(confidence_floor, 3),
            "weak_reasons": reasons,
        }
    )
    weak = bool(reasons)
    found = list(check.conditions)
    tail = " This is informational: nothing is blocked, and every point is reviewed as usual."
    if fallback and weak:
        tail = (
            " The coverage thresholds are provisional fallbacks: the reference has no measured "
            "tracking baseline." + tail
        )
    if not check.compared and scene is not None:
        tail = (
            " The footage's lighting and court colour could not be compared: no conditions "
            "reference is installed." + tail
        )
    if found:
        lead = f"Retraining the ball model is suggested: {'; '.join(found)}."
        if weak:
            lead += f" Ball tracking was also weak here ({'; '.join(reasons)})."
        else:
            lead += (
                " Ball tracking held up on this match, but the detector has not been tried in "
                "these conditions before."
            )
        note = lead + tail
    elif weak:
        note = (
            f"Ball tracking was weak on this match ({'; '.join(reasons)}), although the lighting "
            "and court colour are within the range the ball detector was tuned on. A ball model "
            "trained on more footage like this would help." + tail
        )
    else:
        note = None
    return RetrainingAdvice(
        recommended=note is not None,
        weak_tracking=weak,
        conditions=tuple(found),
        measures=measures,
        note=note,
        provisional=fallback,
        compared=check.compared,
    )


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------


def _atomic_write(path: Path, data: bytes) -> None:
    """Writes a file so a kill mid-write never leaves half of it behind."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


class CheckpointStore:
    """One run's work folder: a manifest, the calibration, a file per chunk.

    Chunk files are pickles of `stitch.ChunkResult`, written atomically.
    They are this program's own files in a folder it made; do not load
    checkpoints from anywhere else (unpickling runs code).

    Attributes:
        folder: The folder.
    """

    def __init__(self, folder: Path) -> None:
        """Initializes the store; nothing is created until written.

        Args:
            folder: The run's work folder.
        """
        self.folder = Path(folder)

    @property
    def manifest_path(self) -> Path:
        """Where the manifest is."""
        return self.folder / "manifest.json"

    @property
    def calibration_path(self) -> Path:
        """Where the calibration is."""
        return self.folder / "calibration.json"

    def chunk_path(self, index: int) -> Path:
        """Where one chunk's checkpoint is.

        Args:
            index: The chunk's index.

        Returns:
            Its file's path.
        """
        return self.folder / f"chunk_{index:04d}.pkl"

    def read_manifest(self) -> dict | None:
        """Reads the manifest.

        Returns:
            It, or None when there is none (or it is unreadable).
        """
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def write_manifest(self, manifest: dict) -> None:
        """Writes the manifest, creating the folder.

        Args:
            manifest: JSON-ready description of the run.
        """
        self.folder.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.manifest_path, json.dumps(manifest, indent=2).encode("utf-8"))

    def save_calibration(self, calibration: Calibration) -> None:
        """Keeps the run's calibration.

        Args:
            calibration: The calibration.
        """
        self.folder.mkdir(parents=True, exist_ok=True)
        data = json.dumps(calibration.to_dict(), indent=2).encode("utf-8")
        _atomic_write(self.calibration_path, data)

    def load_calibration(self) -> Calibration | None:
        """Reads the kept calibration.

        Returns:
            It, or None when there is none (or it is unreadable).
        """
        try:
            data = json.loads(self.calibration_path.read_text(encoding="utf-8"))
            return Calibration.from_dict(data)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def has_chunk(self, index: int) -> bool:
        """Whether a chunk's checkpoint exists.

        Args:
            index: The chunk's index.

        Returns:
            True when its file is there.
        """
        return self.chunk_path(index).is_file()

    def save_chunk(self, result: ChunkResult) -> Path:
        """Writes one chunk's checkpoint.

        Args:
            result: The chunk's output.

        Returns:
            The file written.
        """
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.chunk_path(result.window.index)
        payload = {"format": CHECKPOINT_FORMAT, "result": result}
        _atomic_write(path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        return path

    def load_chunk(self, index: int) -> ChunkResult | None:
        """Reads one chunk's checkpoint.

        Args:
            index: The chunk's index.

        Returns:
            The chunk's output, or None when the file is missing, from
            another format, or unreadable - it is then tracked again.
        """
        path = self.chunk_path(index)
        try:
            payload = pickle.loads(path.read_bytes())  # noqa: S301 - our own checkpoint
        except FileNotFoundError:
            return None
        except Exception as error:  # noqa: BLE001 - any unreadable file is re-tracked
            logger.warning("Checkpoint %s is unreadable (%s); tracking it again.", path.name, error)
            return None
        if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
            logger.warning("Checkpoint %s is from another format; tracking it again.", path.name)
            return None
        result = payload.get("result")
        return result if isinstance(result, ChunkResult) else None

    def clear(self) -> None:
        """Deletes the whole folder."""
        shutil.rmtree(self.folder, ignore_errors=True)


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------


@dataclass
class _Run:
    """Everything one `ingest` call works with (internal).

    Attributes:
        video: The recording, resolved.
        source: Its header, PTS index and checksum.
        start: The run's first moment.
        end: Its last moment.
        windows: The chunks.
        store: The checkpoint folder.
        calibration: The court calibration.
        calibration_source: How it was made ("auto@..." or "manual:...").
        person: The person detector.
        pose: The pose estimator, or None.
        ball_factory: Makes a fresh ball detector per chunk.
        ball_name: That detector's name.
        notes: Run notes gathered on the way, for `import_notes`.
        resumed: Indexes of chunks taken from checkpoints.
        dead_time: The dead-time pre-pass's plan, or None when it did not
            run (or heard nothing to plan from).
    """

    video: Path
    source: VideoSource
    start: float
    end: float
    windows: tuple[ChunkWindow, ...]
    store: CheckpointStore
    calibration: Calibration
    calibration_source: str
    person: PersonDetector
    pose: PoseEstimator | None
    ball_factory: Callable[[Calibration, float | None], BallDetector]
    ball_name: str
    notes: list[str] = field(default_factory=list)
    resumed: set[int] = field(default_factory=set)
    dead_time: DeadTimePlan | None = None


@dataclass(frozen=True)
class ChunkCalibration:
    """The calibration a chunk is tracked with, and how it was chosen (internal).

    Attributes:
        calibration: The calibration to use.
        drift_px: How far the court had moved from the run's own
            calibration at the chunk's start, or None when not measured.
        checked: Whether a re-calibration was attempted.
        recalibrated: Whether a re-calibration was adopted.
        alignment_px: (in use, re-calibration) line alignment, when judged.
    """

    calibration: Calibration
    drift_px: float | None
    checked: bool
    recalibrated: bool = False
    alignment_px: tuple[float | None, float | None] | None = None


def _slug(text: str) -> str:
    """A filename-safe version of a video's name."""
    return review.slugify_filename(text)[:40]


class TrackerPipeline:
    """Turns a fixed-camera match video into a staged, reviewable match.

    The tracker's counterpart of `SwingVisionImportPipeline`: `ingest`
    stages a pending JSON file and never touches SQL; review, the footage
    check and `finalize` are the existing pipeline's, unchanged.
    """

    def __init__(
        self,
        config: TrackerConfig | None = None,
        import_config: ImportConfig | None = None,
        stages: TrackerStages | None = None,
    ) -> None:
        """Initializes the pipeline.

        Args:
            config: Tracker settings, chunking included.
            import_config: Where staged matches go (`pending_dir`) and the
                database `finalize` writes to - the same object the
                SwingVision pipeline takes.
            stages: Injected models and readers; built from `config` when
                omitted.
        """
        self.config = config or TrackerConfig()
        self.import_config = import_config or ImportConfig()
        self.stages = stages or TrackerStages()

    # ------------------------------------------------------------------
    # Readers, defaulting to the real ones
    # ------------------------------------------------------------------

    def _load_video(self, video: Path) -> VideoSource:
        """Opens the recording: header, PTS index, checksum."""
        if self.stages.video_loader is not None:
            return self.stages.video_loader(video)
        return video_stage.load_video(video, self.config.video)

    def _frames(
        self, video: Path, start: float, end: float, index: PtsIndex | None
    ) -> Iterable[VideoFrame]:
        """Decodes a stretch of the recording."""
        if self.stages.frame_reader is not None:
            return self.stages.frame_reader(video, start, end, index)
        return video_stage.iter_frames(
            video, start=start, end=end, index=index, config=self.config.video
        )

    def _audio(self, video: Path, start: float, end: float) -> AudioOnsets | None:
        """Reads a stretch's sharp sounds."""
        if self.stages.audio_reader is not None:
            return self.stages.audio_reader(video, start, end)
        return events_stage.read_audio_onsets(video, start, end, self.config.events)

    def _calibrate_at(self, video: Path, at: float) -> CalibrationResult:
        """Calibrates the court from the recording at one moment."""
        if self.stages.calibrator is not None:
            return self.stages.calibrator(video, at)
        return calibrate.calibrate_video(video, at, self.config)

    def _detect_events(self, *args) -> EventTrack:
        """Runs the event rules (or the injected stand-in)."""
        detector = self.stages.event_detector or events_stage.detect_events
        return detector(*args)

    def _dead_time(self, video: Path, start: float, end: float) -> DeadTimePlan | None:
        """Runs the dead-time pre-pass over the run."""
        reader = self.stages.audio_curve_reader or deadtime_stage.read_onset_curve
        plan = deadtime_stage.plan_dead_time(reader(video), start, end, self.config.dead_time)
        if plan is not None:
            logger.info(
                "Dead-time pre-pass: skipping %.0fs of %.0fs in %d stretch(es).",
                plan.skipped_seconds,
                end - start,
                len(plan.dead),
            )
        return plan

    def _judge_alignment(
        self, video: Path, current: Calibration, candidate: Calibration, start: float, end: float
    ) -> tuple[float | None, float | None]:
        """How well each calibration lies on the court lines of a stretch's frames."""
        if self.stages.alignment_judge is not None:
            return self.stages.alignment_judge(video, current, candidate, start, end)
        count = max(1, self.config.pipeline.alignment_frames)
        step = (end - start) / (count + 1)
        frames = video_stage.frames_at(
            video, [start + step * (k + 1) for k in range(count)], config=self.config.video
        )
        median = court.median_frame([frame.image for frame in frames])
        return (
            calibrate.line_alignment_px(current, median, self.config),
            calibrate.line_alignment_px(candidate, median, self.config),
        )

    def work_folder(self, source: VideoSource, start: float, end: float) -> Path:
        """Names a run's checkpoint folder.

        The video's checksum and the run's range are in the name, so two
        recordings, or two ranges of one, never share checkpoints.

        Args:
            source: The opened recording.
            start: The run's first moment.
            end: Its last moment.

        Returns:
            The folder (not created here).
        """
        base = self.config.pipeline.work_dir or (self.import_config.pending_dir / "tracking_work")
        identity = source.checksum
        if identity is None:
            stat = source.info.path.stat()
            identity = hashlib.sha256(f"{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
        name = f"{_slug(source.info.path.stem)}_{identity[:12]}_{start:.2f}-{end:.2f}"
        return Path(base) / name

    def staged_path(self, video: Path | str, *, date: str, opponent: str, result: str) -> Path:
        """Says where `ingest` will stage (or has staged) this video's match.

        Known before any tracking starts, so a caller can key a background
        job on it and poll `progress_path`. The same rule as
        `review.allocate_pending_name`: the same video lands on the same
        file; a different match against the same opponent that day gets
        a suffixed name.

        Args:
            video: The recording.
            date: ISO-format date of the match.
            opponent: Opponent's name.
            result: Match result, "W" or "L".

        Returns:
            The pending JSON path; it exists only once staged.
        """
        video = Path(video).resolve()
        pending_dir = self.import_config.pending_dir
        probe = MatchRecord(date=date, opponent=opponent, result=result, source_file=str(video))
        return pending_dir / review.allocate_pending_name(probe, pending_dir)

    def progress_path(self, staged: Path) -> Path:
        """Where a run keeps its latest `TrackingProgress`, as JSON.

        Written on every report, atomically, so another process - a web
        server polling a tracking job it started as a subprocess - can
        read how far the run has got.

        Args:
            staged: The path `staged_path` returned.

        Returns:
            `<pending_dir>/tracking_work/<staged name>.progress.json`.
        """
        return self.import_config.pending_dir / "tracking_work" / f"{staged.stem}.progress.json"

    # ------------------------------------------------------------------
    # Setting up a run
    # ------------------------------------------------------------------

    def _calibration(
        self,
        video: Path,
        source: VideoSource,
        at: float,
        keypoints: ManualKeypoints | None,
    ) -> Calibration:
        """Calibrates the court, automatically or from marked keypoints.

        Raises:
            CalibrationFailedError: When the fit abstains, saying why and
                what to do.
        """
        size = (source.info.width, source.info.height)
        if keypoints is not None:
            marked = keypoints.image_size or size
            if tuple(marked) != size:
                raise CalibrationFailedError(
                    f"The keypoints were marked on a {marked[0]}x{marked[1]} frame but "
                    f"{video.name} is {size[0]}x{size[1]}. Mark them on a full-size frame of "
                    "this video."
                )
            try:
                result = calibrate.calibrate_from_points(
                    keypoints.points, size, self.config.calibration, keypoints.template_kind
                )
            except ValueError as error:
                raise CalibrationFailedError(str(error)) from error
            if not result.ok:
                raise CalibrationFailedError(
                    f"The keypoints supplied did not give a usable calibration: {result.reason} "
                    "Check each point's name and position against a full-size frame."
                )
            return result.calibration
        result = self._calibrate_at(video, at)
        if not result.ok:
            examples = ", ".join(court.keypoint_names("doubles")[:3])
            raise CalibrationFailedError(
                f"Court calibration abstained at {at:.1f}s: {result.reason} Re-aim the camera "
                "so both baselines and both sidelines are in view, or mark the court's "
                "keypoints by hand on a full-size frame and pass them as `keypoints` (on the "
                "command line: --keypoints file.json, a JSON object of keypoint name to [x, y] "
                f"pixels, with names such as {examples}; tracking.court.keypoint_names lists "
                "them all)."
            )
        return result.calibration

    def _models(
        self, calibration: Calibration
    ) -> tuple[PersonDetector, PoseEstimator | None, list[str]]:
        """Builds the person detector and pose estimator, noting any fallback."""
        config, stages, notes = self.config, self.stages, []
        person = stages.person_detector
        if person is None:
            try:
                loaded = models.load_person_model(config.models)
                person = players_stage.YoloPersonDetector(
                    loaded.model, config.players, config.models, loaded.source
                )
            except Exception as error:  # noqa: BLE001 - any failure falls back, with a note
                logger.warning("Person model unavailable (%s); using the motion detector.", error)
                person = players_stage.MotionPersonDetector(calibration, config.players)
                notes.append(
                    f"No learned person detector could be loaded ({error}). Players were "
                    "followed by the classic motion detector, which loses a player who stands "
                    "still: expect fewer hits attributed."
                )
        pose = stages.pose_estimator
        if pose is None and stages.use_pose:
            try:
                loaded = models.load_pose_model(config.models)
                pose = strokes_stage.YoloPoseEstimator(
                    loaded.model, config.strokes, config.models, loaded.source
                )
            except Exception as error:  # noqa: BLE001 - any failure falls back, with a note
                logger.warning("Pose model unavailable (%s); strokes from the ball alone.", error)
                notes.append(
                    f"No pose model could be loaded ({error}). Strokes were named from the "
                    "ball alone, so more of them are Unknown."
                )
        return person, pose, notes

    def _fingerprint(self, run: _Run) -> str:
        """Identifies the settings a chunk's output depends on."""
        config = self.config
        strokes = config.strokes
        parts = {
            "video": repr(config.video),
            "players": repr(config.players),
            "ball": repr(config.ball),
            "events": repr(config.events),
            "models": repr(config.models),
            "pose_window": [
                strokes.window_before,
                strokes.window_after,
                strokes.pose_frames,
                strokes.pose_size,
                strokes.crop_pad,
                strokes.min_player_height,
                strokes.min_keypoint_confidence,
            ],
            "person": run.person.name,
            "ball_detector": run.ball_name,
            "pose": None if run.pose is None else run.pose.name,
            "check_drift": config.pipeline.check_drift,
            "recalibrate": [config.pipeline.recalibrate, config.pipeline.recalibrate_px],
            "dead_time": None if run.dead_time is None else run.dead_time.to_dict(),
        }
        text = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _manifest(self, run: _Run) -> dict:
        """Describes the run, so a later run can tell whether to resume."""
        return {
            "format": CHECKPOINT_FORMAT,
            "video": run.video.name,
            "video_checksum": run.source.checksum,
            "range": [run.start, run.end],
            "plan": [window.to_dict() for window in run.windows],
            "calibration_source": run.calibration_source,
            "fingerprint": self._fingerprint(run),
        }

    def _check_manifest(self, run: _Run, manifest: dict) -> None:
        """Refuses to mix checkpoints from a run with other settings.

        Raises:
            CheckpointMismatchError: When the stored manifest differs and
                chunk checkpoints exist.
        """
        stored = run.store.read_manifest()
        if stored is None:
            return
        differ = sorted(key for key, value in manifest.items() if stored.get(key) != value)
        if not differ:
            return
        if any(run.store.has_chunk(window.index) for window in run.windows):
            raise CheckpointMismatchError(
                f"Checkpoints in {run.store.folder} were made with a different "
                f"{', '.join(differ)}. Pass force=True (--force) to discard them and track "
                "again."
            )
        logger.info("Stale manifest in %s (no chunks yet); replacing it.", run.store.folder)

    # ------------------------------------------------------------------
    # The front door
    # ------------------------------------------------------------------

    def ingest(
        self,
        video: Path | str,
        setup: MatchSetup,
        *,
        date: str,
        opponent: str,
        result: str,
        first_server_by_set: dict[int, str] | None = None,
        tracked_identity: str | None = None,
        start: float | None = None,
        end: float | None = None,
        keypoints: ManualKeypoints | None = None,
        force: bool = False,
        progress: Callable[[TrackingProgress], None] | None = None,
        **match_overrides: object,
    ) -> Path:
        """Tracks a match video and stages it for review.

        Args:
            video: The fixed-camera recording.
            setup: Names, who started at the near end, and hands.
            date: ISO-format date of the match.
            opponent: Opponent's name.
            result: Match result, "W" or "L".
            first_server_by_set: Optional set_number -> "me"/"opponent"
                ground truth, cross-checked against the reconstruction
                (`quality_check.check_serve_order`).
            tracked_identity: Optional "who I am", cross-checked against
                `setup.host_name`.
            start: First moment to track, seconds on the video's clock;
                the recording's start when omitted.
            end: Last moment to track; its end when omitted.
            keypoints: Court keypoints marked by hand, used instead of
                automatic calibration.
            force: Track again even when the video is already staged, and
                discard its checkpoints.
            progress: Called as the run advances.
            **match_overrides: Additional `MatchRecord` fields (e.g.
                energy_rating, pros, cons, player_appearance).

        Returns:
            Path to the staged pending-review JSON file - the existing one,
            untouched, when the video was already staged and `force` is
            not given.

        Raises:
            CalibrationFailedError: When the court cannot be calibrated.
            CheckpointMismatchError: When checkpoints from other settings
                exist and `force` is not given.
            ValueError: When the range is empty.
            tracking.video.VideoError: When the video cannot be read.
        """
        video = Path(video).resolve()
        staged = self.staged_path(video, date=date, opponent=opponent, result=result)
        name = staged.name
        status = self.progress_path(staged)

        def report(update: TrackingProgress) -> None:
            """Hands progress to the caller and to the progress file."""
            _write_progress(status, update)
            if progress is not None:
                progress(update)

        if staged.exists() and not force:
            message = f"{video.name} is already staged at {staged}; pass force to track it again."
            logger.info(message)
            report(TrackingProgress("skipped", message=message))
            return staged

        report(TrackingProgress("opening", message=f"Reading {video.name}'s frame timestamps."))
        source = self._load_video(video)
        first, last = float(source.index.pts[0]), float(source.index.pts[-1])
        run_start = first if start is None else max(first, float(start))
        run_end = last if end is None else min(last, float(end))
        if not run_end > run_start:
            raise ValueError(
                f"Nothing to track between {run_start:.2f}s and {run_end:.2f}s: the recording "
                f"runs from {first:.2f}s to {last:.2f}s."
            )
        windows = plan_chunks(run_start, run_end, self.config.pipeline)
        store = CheckpointStore(self.work_folder(source, run_start, run_end))
        if force:
            store.clear()

        report(
            TrackingProgress(
                "calibrating",
                len(windows),
                video_seconds=run_end - run_start,
                message="Calibrating the court.",
            )
        )
        at = run_start + self.config.pipeline.calibrate_at
        calibration_source = keypoints.identity() if keypoints else f"auto@{at:.3f}"
        stored = store.read_manifest()
        calibration = None
        if stored is not None and stored.get("calibration_source") == calibration_source:
            calibration = store.load_calibration()
        if calibration is None:
            calibration = self._calibration(video, source, at, keypoints)
        person, pose, notes = self._models(calibration)
        plan = None
        if self.config.dead_time.enabled:
            report(
                TrackingProgress(
                    "calibrating",
                    len(windows),
                    video_seconds=run_end - run_start,
                    message="Listening for stretches with no play (dead-time pre-pass).",
                )
            )
            plan = self._dead_time(video, run_start, run_end)
            if plan is None:
                notes.append(
                    "The dead-time pre-pass was asked for but the recording's audio gave it "
                    "nothing to go on, so the whole run was tracked."
                )
        ball_factory = self.stages.ball_detector or (
            lambda cal, fps: ball_stage.MotionBallDetector(cal, self.config.ball, fps)
        )
        run = _Run(
            video=video,
            source=source,
            start=run_start,
            end=run_end,
            windows=windows,
            store=store,
            calibration=calibration,
            calibration_source=calibration_source,
            person=person,
            pose=pose,
            ball_factory=ball_factory,
            ball_name=ball_factory(calibration, source.info.nominal_fps).name,
            notes=notes,
            dead_time=plan,
        )
        manifest = self._manifest(run)
        self._check_manifest(run, manifest)
        store.write_manifest(manifest)
        store.save_calibration(calibration)

        self._track_all(run, report)
        report(
            TrackingProgress(
                "assembling",
                len(windows),
                len(windows),
                len(run.resumed),
                run_end - run_start,
                run_end - run_start,
                message="Stitching the chunks and reconstructing the points.",
            )
        )
        path = self._stage(
            run,
            setup,
            name,
            date=date,
            opponent=opponent,
            result=result,
            first_server_by_set=first_server_by_set,
            tracked_identity=tracked_identity,
            match_overrides=match_overrides,
        )
        if not self.config.pipeline.keep_checkpoints:
            store.clear()
        report(
            TrackingProgress(
                "staged",
                len(windows),
                len(windows),
                len(run.resumed),
                run_end - run_start,
                run_end - run_start,
                message=f"Staged {path}.",
            )
        )
        return path

    # ------------------------------------------------------------------
    # Tracking, chunk by chunk
    # ------------------------------------------------------------------

    def _track_all(self, run: _Run, report: Callable[[TrackingProgress], None]) -> None:
        """Tracks every chunk without a checkpoint, in order, saving each."""
        total_video = run.end - run.start
        done_video = tracked_video = elapsed = 0.0
        previous: ChunkResult | None = None
        for count, window in enumerate(run.windows, start=1):
            length = window.end - window.start
            cached = run.store.load_chunk(window.index)
            if cached is not None and cached.window == window:
                previous = cached
                run.resumed.add(window.index)
                logger.info("Chunk %d/%d resumed from its checkpoint.", count, len(run.windows))
                message = f"Chunk {count}/{len(run.windows)} resumed from its checkpoint."
            else:
                report(
                    TrackingProgress(
                        "tracking",
                        len(run.windows),
                        count - 1,
                        len(run.resumed),
                        total_video,
                        done_video,
                        elapsed,
                        _eta(elapsed, tracked_video, total_video - done_video),
                        _rate(elapsed, tracked_video),
                        f"Tracking chunk {count}/{len(run.windows)} "
                        f"({window.start:.0f}-{window.end:.0f}s).",
                    )
                )
                previous = self._track_chunk(run, window, previous)
                run.store.save_chunk(previous)
                elapsed += previous.seconds
                tracked_video += length
                message = (
                    f"Chunk {count}/{len(run.windows)} tracked in {previous.seconds:.0f}s: "
                    f"{len(previous.hits)} hit(s), {len(previous.bounces)} bounce(s)."
                )
                logger.info(message)
            done_video += length
            report(
                TrackingProgress(
                    "tracking",
                    len(run.windows),
                    count,
                    len(run.resumed),
                    total_video,
                    done_video,
                    elapsed,
                    _eta(elapsed, tracked_video, total_video - done_video),
                    _rate(elapsed, tracked_video),
                    message,
                )
            )
            gc.collect()

    def _track_chunk(
        self, run: _Run, window: ChunkWindow, previous: ChunkResult | None
    ) -> ChunkResult:
        """Tracks one chunk: players, ball, events and poses."""
        config = self.config
        began = time.perf_counter()
        current = run.calibration
        if previous is not None and previous.calibration is not None:
            current = previous.calibration
        check = self._chunk_calibration(run, window, current)
        calibration = check.calibration
        if previous is None:
            tracker = PlayerTracker(run.person, calibration, config.players)
        else:
            tracker = PlayerTracker.resume(previous.tracker_state, run.person)
            tracker.recalibrate(calibration)
        detector = run.ball_factory(calibration, run.source.info.nominal_fps)
        stream = CandidateStream(detector)
        sampler = SceneSampler(calibration, config.ball.scene_sample_stride)
        plan = run.dead_time
        found = []
        core = state = swaps = None
        owned: list[float] = []
        skipped = 0
        live = True
        for frame in self._frames(run.video, window.read_start, window.read_end, run.source.index):
            mine = window.owns(frame.pts)
            if mine:
                owned.append(frame.pts)
            if plan is not None and not plan.is_active(frame.pts):
                skipped += mine
                if live:
                    # No frame difference is taken across a skipped stretch.
                    found.extend(stream.finish())
                    stream = CandidateStream(detector)
                    live = False
                continue
            live = True
            if frame.pts >= window.start:
                if not mine and core is None:
                    core, state, swaps = _hand_over(tracker)
                tracker.update(frame)
                if core is None:
                    sampler.add(frame.image)
            found.extend(stream.push(frame))
        found.extend(stream.finish())
        if core is None:
            core, state, swaps = _hand_over(tracker)
        after = tracker.take_observations()
        before = (
            ((), (), ())
            if previous is None
            else (
                tuple(o for o in previous.near if o.pts >= window.read_start),
                tuple(o for o in previous.far if o.pts >= window.read_start),
                tuple(t for t in previous.detector_times if t >= window.read_start),
            )
        )
        local = players_stage.label_tracks(
            near=before[0] + core.near + after.near,
            far=before[1] + core.far + after.far,
            swaps=tracker.swaps,
            detector_times=before[2] + core.detector_times + after.detector_times,
            source=tracker.detector_name,
            config=config.players,
        )
        ball = link_candidates(
            found,
            calibration.image_size,
            calibration,
            local,
            config.ball,
            source=detector.name,
            scene=sampler.result(),
        )
        del found
        audio = self._audio(run.video, window.read_start, window.read_end)
        events = self._detect_events(ball, local, calibration, audio, config.events)
        margin = config.pipeline.stitch_margin_seconds
        hits = tuple(h for h in events.hits if window.keeps(h.pts, margin))
        poses = self._poses(run, events, {pose_key(h.pts) for h in hits}, local)
        expected, holes = decode_holes(
            run.source.index, window, owned, config.pipeline.decode_hole_frames
        )
        return ChunkResult(
            window=window,
            hits=hits,
            bounces=tuple(b for b in events.bounces if window.keeps(b.pts, margin)),
            nets=tuple(n for n in events.nets if window.keeps(n.pts, margin)),
            ball=CompactBallTrack.from_track(
                events.ball, keep=window.owns, segment_offset=window.index * _SEGMENT_STRIDE
            ),
            near=core.near,
            far=core.far,
            detector_times=core.detector_times,
            rejected_people=core.rejected_people,
            swaps=swaps,
            poses=poses,
            scene=sampler.result(),
            discarded_kinks=events.discarded_kinks,
            drift_px=check.drift_px,
            drift_checked=check.checked,
            seconds=time.perf_counter() - began,
            tracker_state=state,
            detector_name=tracker.detector_name,
            calibration=calibration,
            recalibrated=check.recalibrated,
            alignment_px=check.alignment_px,
            decoded_frames=len(owned),
            expected_frames=expected,
            decode_holes=holes,
            skipped_frames=skipped,
        )

    def _poses(
        self,
        run: _Run,
        events: EventTrack,
        wanted: set[float],
        local: PlayerTracks,
    ) -> dict[float, tuple[PoseSample, ...]]:
        """Estimates poses around the chunk's kept hits, decoding only near them."""
        if run.pose is None or not wanted:
            return {}
        # Every kept hit is a shot, so `shots` is never empty here.
        shots = tuple(s for s in events.shots if pose_key(s.hit.pts) in wanted)
        strokes = self.config.strokes
        spans: list[list[float]] = []
        for shot in sorted(shots, key=lambda s: s.hit.pts):
            low = max(run.start, shot.hit.pts - strokes.window_before - 0.05)
            high = min(run.end, shot.hit.pts + strokes.window_after + 0.05)
            if spans and low <= spans[-1][1] + 1.0:
                spans[-1][1] = max(spans[-1][1], high)
            else:
                spans.append([low, high])
        frames = chain.from_iterable(
            self._frames(run.video, low, high, run.source.index) for low, high in spans
        )
        found = strokes_stage.estimate_poses(
            frames, replace(events, shots=shots), local, run.pose, strokes
        )
        by_index = {shot.index: pose_key(shot.hit.pts) for shot in shots}
        return {by_index[index]: samples for index, samples in found.items()}

    def _chunk_calibration(
        self, run: _Run, window: ChunkWindow, current: Calibration
    ) -> ChunkCalibration:
        """Re-calibrates at a chunk's start; adopts the result if it fits better.

        The run's first chunk keeps the run's calibration. Every later one
        is re-calibrated (when `check_drift`); `drift_px` is measured
        against the run's own calibration. When the court has moved
        further than `recalibrate_px` from the calibration in use, both
        are laid over frames from the middle of the chunk and the one
        whose lines sit closer to the painted lines is used - so a wrong
        re-calibration is never adopted just for being new.
        """
        pipeline = self.config.pipeline
        if not pipeline.check_drift or window.index == 0:
            return ChunkCalibration(current, None, False)
        try:
            result = self._calibrate_at(run.video, window.start)
        except video_stage.VideoError as error:
            logger.warning("Re-calibration at %.1fs failed: %s", window.start, error)
            return ChunkCalibration(current, None, True)
        if not result.ok:
            return ChunkCalibration(current, None, True)
        candidate = result.calibration
        drift = calibrate.drift_px(run.calibration, candidate)
        moved = calibrate.drift_px(current, candidate)
        if not pipeline.recalibrate or moved <= pipeline.recalibrate_px:
            return ChunkCalibration(current, drift, True)
        try:
            before, after = self._judge_alignment(
                run.video, current, candidate, window.start, window.end
            )
        except video_stage.VideoError as error:
            logger.warning("Could not judge the re-calibration at %.1fs: %s", window.start, error)
            return ChunkCalibration(current, drift, True, False, (None, None))
        adopt = before is not None and after is not None and after < before
        logger.info(
            "Chunk at %.0fs: court moved %.1fpx; line alignment %s -> %s px; %s.",
            window.start,
            moved,
            before,
            after,
            "re-calibrated" if adopt else "kept the calibration in use",
        )
        return ChunkCalibration(
            candidate if adopt else current, drift, True, adopt, (before, after)
        )

    # ------------------------------------------------------------------
    # One match from the chunks, staged
    # ------------------------------------------------------------------

    def _stage(
        self,
        run: _Run,
        setup: MatchSetup,
        name: str,
        *,
        date: str,
        opponent: str,
        result: str,
        first_server_by_set: dict[int, str] | None,
        tracked_identity: str | None,
        match_overrides: dict,
    ) -> Path:
        """Stitches the chunks, assembles, reconstructs and stages the match."""
        config = self.config
        chunks = [run.store.load_chunk(window.index) for window in run.windows]
        missing = [w.index for w, chunk in zip(run.windows, chunks) if chunk is None]
        if missing:
            raise TrackingError(
                f"Chunk checkpoint(s) {missing} in {run.store.folder} could not be read back; "
                "run again to track them."
            )
        stitched = merge_chunks(chunks, config.pipeline, config.events, config.players)
        stroke_config = setup.stroke_config(config.strokes)
        labels = strokes_stage.classify_strokes(
            stitched.events,
            stitched.players,
            stitched.poses,
            self.stages.stroke_classifier,
            stroke_config,
        )
        events = strokes_stage.apply_strokes(stitched.events, labels, stroke_config)
        tracked = assemble_stage.assemble(
            events,
            labels,
            setup,
            players=stitched.players,
            calibration=run.calibration,
            config=config,
            tracking_start=run.start,
        )
        reconstruction = reconstruct.reconstruct_all(tracked.export.shots, setup.host_name)
        by_point = reconstruct.group_shots_by_point(tracked.export.shots)
        blocks = tracked.point_blocks(reconstruction)
        for set_record in reconstruction.sets:
            for point in set_record.points:
                # Reconstruction already flags every point; a tracked point
                # is a guess from pixels, so this is stated, not assumed.
                point.needs_review = True
                point.tracking = blocks.get(point.source_point_number)
        record = MatchRecord(
            date=date,
            opponent=opponent,
            result=result,
            source_file=str(run.video),
            sets=reconstruction.sets,
            shot_pattern_summary=reconstruct.build_shot_pattern_summary(
                reconstruction.points, by_point
            ),
            **match_overrides,
        )
        playing = [a for a in tracked.segmentation.attempts if a.kind != EXCLUDED]
        advice = retraining_advice(
            events.ball,
            [(a.start - 0.5, a.end + 1.5) for a in playing],
            tracked.segmentation.events.quality(),
            stitched.scene,
            config.retraining,
            conditions_stage.load_reference(config.retraining.reference_path),
        )
        record.tracking = self._block(run, chunks, setup, tracked, stitched, advice)
        importer = SwingVisionImportPipeline(self.import_config)
        reconstruction_notes = importer._build_import_notes(
            record,
            [],
            tracked.export.settings,
            reconstruction,
            first_server_by_set,
            tracked_identity,
        )
        notes = self._notes(run, chunks, tracked, stitched, advice)
        record.import_notes = notes + reconstruction_notes
        record.pending_name = name
        path = review.save_pending(record, self.import_config.pending_dir)
        logger.info(
            "Staged %s vs %s at %s: %d shot rows, %d point(s), all need review",
            date,
            opponent,
            path,
            len(tracked.export.shots),
            sum(len(s.points) for s in record.sets),
        )
        return path

    def _notes(
        self,
        run: _Run,
        chunks: list[ChunkResult],
        tracked: TrackedMatch,
        stitched: StitchedMatch,
        advice: RetrainingAdvice,
    ) -> list[str]:
        """The tracker's import notes, most important first."""
        config = self.config.pipeline
        first, last = float(run.source.index.pts[0]), float(run.source.index.pts[-1])
        notes = [
            f"Staged by the independent tracker from {run.video.name}, not from a SwingVision "
            "export: every shot is a guess from pixels, and every point needs review."
        ]
        if run.start > first + 0.5 or run.end < last - 0.5:
            notes.append(
                f"Only {run.start:.0f}-{run.end:.0f}s of the {last - first:.0f}s recording was "
                f"tracked. The score is counted from 0-0 at {run.start:.0f}s, so set and game "
                "numbers are not the match's."
            )
        if advice.note is not None:
            notes.append(advice.note)
        setup = calibrate.setup_advice(run.calibration.quality, self.config.calibration)
        notes.append(setup.note())
        # The setup note says what these calibration warnings say, and more.
        covered = {w for w in run.calibration.quality.warnings if "far baseline" in w}
        notes.extend(note for note in tracked.notes if note not in covered)
        notes.extend(run.notes)
        if len(run.windows) > 1:
            report = stitched.report
            notes.append(
                f"Tracked in {len(run.windows)} chunks of up to {config.chunk_seconds:.0f}s, "
                f"each read with {config.overlap_seconds:.0f}s of overlap; {report.matched} "
                "event(s) seen by both chunks at a shared edge were kept once, "
                f"{report.dropped_unowned} seen only by the chunk with less context were left "
                "out."
            )
        notes.extend(_drift_notes(chunks, config))
        notes.extend(_footage_notes(run, chunks, tracked, self.config))
        return notes

    def _block(
        self,
        run: _Run,
        chunks: list[ChunkResult],
        setup: MatchSetup,
        tracked: TrackedMatch,
        stitched: StitchedMatch,
        advice: RetrainingAdvice,
    ) -> dict:
        """The match-level `tracking` block for the staged JSON."""
        info = run.source.info
        config = self.config.pipeline
        described = [
            {
                **chunk.window.to_dict(),
                "seconds": round(chunk.seconds, 2),
                "resumed": chunk.window.index in run.resumed,
                "drift_px": chunk.drift_px,
                "recalibrated": chunk.recalibrated,
                "alignment_px": None if chunk.alignment_px is None else list(chunk.alignment_px),
                "decoded_frames": chunk.decoded_frames,
                "expected_frames": chunk.expected_frames,
                "decode_holes": [list(hole) for hole in chunk.decode_holes],
                "skipped_frames": chunk.skipped_frames,
            }
            for chunk in chunks
        ]
        classifier = self.stages.stroke_classifier
        block = TrackingBlock.from_dict(tracked.tracking_block()).to_dict()
        block.update(
            {
                "producer": PRODUCER,
                "pipeline": SOURCE,
                "video": {
                    "path": str(run.video),
                    "name": run.video.name,
                    "checksum": run.source.checksum,
                    "width": info.width,
                    "height": info.height,
                    "nominal_fps": info.nominal_fps,
                    "duration": info.duration,
                    "start_time": info.start_time,
                    "frame_count": info.frame_count,
                    "is_vfr": info.is_vfr,
                },
                "clock": {
                    "kind": "pts",
                    "offset_seconds": 0.0,
                    "note": "video_time is the video's own presentation time: pass "
                    "tracking.assemble.video_clock_offset as verify_footage's clock_offset.",
                },
                "range": [run.start, run.end],
                "settings": {"host_name": setup.host_name, "guest_name": setup.guest_name},
                "setup": asdict(setup),
                "shots": [asdict(row) for row in tracked.export.shots],
                "calibration": run.calibration.to_dict(),
                "scene": None if stitched.scene is None else stitched.scene.to_dict(),
                "retraining": advice.to_dict(),
                "camera_setup": calibrate.setup_advice(
                    run.calibration.quality, self.config.calibration
                ).to_dict(),
                "dead_time": None if run.dead_time is None else run.dead_time.to_dict(),
                "frame_gaps": [
                    [round(a, 4), round(b, 4)]
                    for a, b in _frame_gaps(
                        run.source.index, run.start, run.end, self.config.ball.max_gap_seconds
                    )[1]
                ][:500],
                "models": {
                    "person": run.person.name,
                    "pose": None if run.pose is None else run.pose.name,
                    "ball": run.ball_name,
                    "strokes": "rule-strokes" if classifier is None else classifier.name,
                },
                "run": {
                    "chunk_seconds": config.chunk_seconds,
                    "overlap_seconds": config.overlap_seconds,
                    "work_folder": str(run.store.folder),
                    "chunks": described,
                    "stitch": stitched.report.to_dict(),
                    "notes": list(run.notes),
                },
            }
        )
        return block


def _drift_notes(chunks: Sequence[ChunkResult], config: PipelineConfig) -> list[str]:
    """What the camera moving mid-match did, and what was done about it."""
    moved = [c for c in chunks if c.drift_px is not None and c.drift_px > config.drift_warn_px]
    adopted = [c for c in chunks if c.recalibrated]
    judged = [c for c in chunks if c.alignment_px is not None and not c.recalibrated]
    refused = [c for c in judged if None not in c.alignment_px]
    unjudged = [c for c in judged if None in c.alignment_px]
    if not moved and not adopted:
        return []
    shifts = [c.drift_px for c in moved + adopted]
    since = min(c.window.start for c in moved + adopted)
    text = (
        f"The court moved in the picture by up to {max(shifts):.0f}px from where "
        f"it was first calibrated (from about {since:.0f}s): the camera crept, was bumped or "
        "slipped."
    )
    if adopted:
        shown = "; ".join(
            f"{c.window.start:.0f}s ({c.alignment_px[0]:.2f}px -> {c.alignment_px[1]:.2f}px)"
            for c in adopted[:8]
        )
        more = f" and {len(adopted) - 8} more" if len(adopted) > 8 else ""
        text += (
            f" The tracker re-calibrated at {len(adopted)} chunk start(s) where a fresh "
            "calibration lay closer to the painted lines (mean distance from them, before -> "
            f"after: {shown}{more})."
        )
    if refused:
        text += (
            f" At {len(refused)} chunk start(s) a fresh calibration fitted no better and was not "
            "used."
        )
    if unjudged:
        text += (
            f" At {len(unjudged)} chunk start(s) a fresh calibration could not be checked "
            "against the footage, so it was not used."
        )
    if not adopted:
        if not config.recalibrate:
            why = "re-calibration is turned off"
        elif refused or unjudged:
            why = "no fresh calibration was shown to fit better"
        else:
            why = f"each move was within the {config.recalibrate_px:.0f}px re-calibration threshold"
        text += (
            f" The calibration was not replaced ({why}), so positions, bounces and line calls "
            "after that may be off by up to that much."
        )
    return [text]


def _frame_gaps(
    index: PtsIndex, start: float, end: float, longer: float
) -> tuple[float, list[tuple[float, float]]]:
    """Dropped-frame gaps within a stretch that the ball track cannot bridge.

    A gap counts when it is longer than `longer` seconds and also longer
    than twice the recording's usual frame interval - so a low frame rate
    is never mistaken for dropped frames.

    Returns:
        (the threshold used, the gaps as (before, after) timestamps).
    """
    pts = np.asarray(index.pts)
    inside = pts[(pts >= start) & (pts <= end)]
    gaps = np.diff(inside)
    usual = float(np.median(np.diff(pts))) if len(pts) > 1 else 0.0
    threshold = max(longer, 2.0 * usual)
    where = np.flatnonzero(gaps > threshold)
    return threshold, [(float(inside[k]), float(inside[k + 1])) for k in where]


def _footage_notes(
    run: _Run, chunks: Sequence[ChunkResult], tracked: TrackedMatch, config: TrackerConfig
) -> list[str]:
    """What was wrong with the file itself, and what the pre-pass skipped."""
    notes = []
    info = run.source.info
    index = run.source.index
    first, last = float(index.pts[0]), float(index.pts[-1])
    if info.duration - (last - first) > config.video.truncation_seconds:
        notes.append(
            f"{run.video.name}'s header says it runs {info.duration:.0f}s but its frames stop at "
            f"{last - first:.0f}s: the file looks cut short (an interrupted recording or copy). "
            f"Only the {last - first:.0f}s that is there was tracked."
        )
    holes = [hole for chunk in chunks for hole in chunk.decode_holes]
    if holes:
        frames = sum(h[2] for h in holes)
        shown = ", ".join(f"{a:.1f}-{b:.1f}s" for a, b, _ in holes[:6])
        more = f" and {len(holes) - 6} more" if len(holes) > 6 else ""
        notes.append(
            f"{frames} frame(s) in {len(holes)} stretch(es) could not be decoded ({shown}{more}): "
            "the file is damaged there. Nothing was tracked in them, so any shot inside one is "
            "missing rather than guessed."
        )
    longer, gaps = _frame_gaps(index, run.start, run.end, config.ball.max_gap_seconds)
    playing = [a for a in tracked.segmentation.attempts if a.kind != EXCLUDED]
    hit = sorted(
        {a.number for a in playing for g0, g1 in gaps if g0 < a.end + 0.5 and g1 > a.start - 0.5}
    )
    if hit:
        worst = max(g1 - g0 for g0, g1 in gaps)
        shown = ", ".join(str(n) for n in hit[:15]) + (" and more" if len(hit) > 15 else "")
        notes.append(
            f"The recording drops frames: {len(gaps)} gap(s) longer than {longer:.2f}s (the "
            f"longest {worst:.2f}s) - longer than the ball track bridges - and some fall inside "
            f"the play of {len(hit)} point(s) (point numbers {shown}). A hit or bounce during "
            "one can be missed; real frame timestamps were used, so times and speeds allow for "
            "the gaps."
        )
    elif info.is_vfr:
        notes.append(
            f"{run.video.name} drops frames now and then; real frame timestamps were used "
            "throughout, so times and speeds allow for it, and no gap long enough to break the "
            "ball track fell inside a point."
        )
    if run.dead_time is not None:
        notes.append(run.dead_time.note())
    return notes


def _write_progress(path: Path, progress: TrackingProgress) -> None:
    """Writes the latest progress where another process can read it."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, json.dumps(asdict(progress)).encode("utf-8"))
    except OSError as error:  # progress is a courtesy; never stop a run for it
        logger.warning("Could not write progress to %s: %s", path, error)


def _hand_over(tracker: PlayerTracker) -> tuple:
    """Takes a tracker's observations and its state at a chunk's edge."""
    return tracker.take_observations(), tracker.snapshot(), tracker.swaps


def _rate(elapsed: float, tracked_video: float) -> float | None:
    """Wall-clock seconds per minute of video, once something was tracked."""
    return None if tracked_video <= 0 else elapsed / tracked_video * 60.0


def _eta(elapsed: float, tracked_video: float, remaining_video: float) -> float | None:
    """Estimated seconds left, from this run's own speed."""
    return None if tracked_video <= 0 else elapsed / tracked_video * remaining_video
