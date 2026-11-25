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
from collections.abc import Callable, Iterable, Mapping
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
from . import events as events_stage
from . import players as players_stage
from . import strokes as strokes_stage
from . import video as video_stage
from .assemble import MatchSetup, TrackedMatch, TrackingBlock
from .ball import BallDetector, CandidateStream, SceneSampler, SceneStats, link_candidates
from .calibrate import Calibration, CalibrationResult
from .config import RetrainingConfig, TrackerConfig
from .events import AudioOnsets, EventQuality, EventTrack
from .players import PersonDetector, PlayerTracker, PlayerTracks
from .segment import EXCLUDED
from .stitch import (
    ChunkResult,
    ChunkWindow,
    CompactBallTrack,
    StitchedMatch,
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

    Attributes:
        recommended: Whether to say so.
        weak_tracking: Whether ball or event tracking was weak.
        conditions: How conditions differ from the daylight reference,
            one phrase each; empty when they do not markedly.
        measures: The numbers the decision was made on.
        note: The sentence for `import_notes`, or None.
        provisional: Always True until a trained model sets a baseline.
    """

    recommended: bool
    weak_tracking: bool
    conditions: tuple[str, ...]
    measures: dict
    note: str | None
    provisional: bool = True

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field.
        """
        out = asdict(self)
        out["conditions"] = list(self.conditions)
        return out


def _conditions(scene: SceneStats | None, config: RetrainingConfig) -> tuple[list[str], dict]:
    """How the court's look differs from the reference footage."""
    if scene is None:
        return [], {"scene": None}
    reference = config.reference_brightness
    distance = float(np.linalg.norm(np.subtract(scene.court_rgb, config.reference_court_rgb)))
    measures = {
        "brightness": round(scene.brightness, 1),
        "reference_brightness": reference,
        "court_rgb": [round(v, 1) for v in scene.court_rgb],
        "reference_court_rgb": list(config.reference_court_rgb),
        "colour_distance": round(distance, 1),
    }
    found = []
    if scene.brightness < config.dim_ratio * reference:
        found.append(
            f"night or dusk lighting: court brightness {scene.brightness:.0f} against "
            f"{reference:.0f} in the daylight reference"
        )
    elif scene.brightness > config.bright_ratio * reference:
        found.append(
            f"much brighter light than the reference: court brightness {scene.brightness:.0f} "
            f"against {reference:.0f}"
        )
    if distance > config.colour_distance:
        rgb = ", ".join(f"{v:.0f}" for v in scene.court_rgb)
        found.append(f"a court colour unlike the reference's (RGB {rgb})")
    return found, measures


def retraining_advice(
    ball: CompactBallTrack,
    stretches: Iterable[tuple[float, float]],
    events: EventQuality,
    scene: SceneStats | None,
    config: RetrainingConfig | None = None,
) -> RetrainingAdvice:
    """Decides whether to recommend retraining the ball model on this footage.

    Never blocking: the result is an informational note. Every threshold
    is provisional (`RetrainingConfig`).

    Args:
        ball: The run's ball state.
        stretches: (start, end) of every point's rally - where coverage
            is judged, since most of a match is not play.
        events: The run's event quality, for how many results are unknown.
        scene: The run's court brightness and colour.
        config: Thresholds and the daylight reference.

    Returns:
        The advice, with the numbers it rests on.
    """
    config = config or RetrainingConfig()
    stretches = list(stretches)
    frames, seen, confidence = ball.coverage_over(stretches)
    coverage = seen / frames if frames else 0.0
    weak_points = sum(ball.quality(start, end).weak for start, end in stretches)
    reasons = []
    if frames and (coverage < config.weak_coverage or confidence < config.weak_confidence):
        reasons.append(
            f"a ball position in {coverage:.0%} of rally frames, mean confidence {confidence:.2f}"
        )
    if stretches and weak_points / len(stretches) > config.weak_point_share:
        reasons.append(f"weak on {weak_points} of {len(stretches)} points")
    if events.shots and events.abstention_rate > config.max_abstention:
        reasons.append(f"no result for {events.abstention_rate:.0%} of shots")
    conditions, measures = _conditions(scene, config)
    measures.update(
        {
            "rally_frames": frames,
            "rally_coverage": round(coverage, 3),
            "rally_mean_confidence": round(confidence, 3),
            "weak_points": weak_points,
            "points": len(stretches),
            "result_abstention": round(events.abstention_rate, 3),
        }
    )
    weak = bool(reasons)
    caveat = " (Thresholds are provisional: no trained ball model exists yet to set a baseline.)"
    if weak and conditions:
        note = (
            f"Ball tracking was weak on this match ({'; '.join(conditions)}; "
            f"{'; '.join(reasons)}). Retraining the ball model on footage like this is "
            "recommended." + caveat
        )
    elif weak:
        note = (
            f"Ball tracking was weak on this match ({'; '.join(reasons)}), in conditions close "
            "to the daylight reference footage. Retraining the ball model on footage like this "
            "is recommended." + caveat
        )
    elif conditions:
        note = (
            "Conditions on this match differ markedly from the daylight reference footage "
            f"({'; '.join(conditions)}). Ball tracking held up here, but retraining on footage "
            "like this is recommended before relying on it in these conditions." + caveat
        )
    else:
        note = None
    measures["weak_reasons"] = reasons
    return RetrainingAdvice(note is not None, weak, tuple(conditions), measures, note)


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
        calibration = run.calibration
        if previous is None:
            tracker = PlayerTracker(run.person, calibration, config.players)
        else:
            tracker = PlayerTracker.resume(previous.tracker_state, run.person)
        detector = run.ball_factory(calibration, run.source.info.nominal_fps)
        stream = CandidateStream(detector)
        sampler = SceneSampler(calibration, config.ball.scene_sample_stride)
        found = []
        core = state = swaps = None
        for frame in self._frames(run.video, window.read_start, window.read_end, run.source.index):
            if frame.pts >= window.start:
                if not window.owns(frame.pts) and core is None:
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
        drift, checked = self._drift(run, window)
        result = ChunkResult(
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
            drift_px=drift,
            drift_checked=checked,
            seconds=time.perf_counter() - began,
            tracker_state=state,
            detector_name=tracker.detector_name,
        )
        return result

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

    def _drift(self, run: _Run, window: ChunkWindow) -> tuple[float | None, bool]:
        """Re-calibrates at a chunk's start and measures how far the court moved."""
        if not self.config.pipeline.check_drift or window.index == 0:
            return None, False
        try:
            result = self._calibrate_at(run.video, window.start)
        except video_stage.VideoError as error:
            logger.warning("Re-calibration at %.1fs failed: %s", window.start, error)
            return None, True
        if not result.ok:
            return None, True
        return calibrate.drift_px(run.calibration, result.calibration), True


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
        info = run.source.info
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
        notes.extend(tracked.notes)
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
        moved = [
            (chunk.window, chunk.drift_px)
            for chunk in chunks
            if chunk.drift_px is not None and chunk.drift_px > config.drift_warn_px
        ]
        if moved:
            worst = max(px for _, px in moved)
            notes.append(
                f"The court moved in the picture by up to {worst:.0f}px (from about "
                f"{moved[0][0].start:.0f}s): the camera was bumped or slipped. Positions, "
                "bounces and line calls after that are off - the tracker does not re-calibrate "
                "mid-match."
            )
        if info.is_vfr:
            notes.append(
                f"{run.video.name} drops frames now and then; real frame timestamps were used "
                "throughout, so times and speeds allow for it."
            )
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
