"""Tests for the tracker pipeline: one video in, one staged match out.

No test loads a model, calls an API or decodes the real recording. Video
is either synthetic (`SyntheticVideo` hands out tiny frames and a PTS
index without ffmpeg) or a tiny lavfi clip; people, the ball, events and
strokes are injected fakes that replay a scripted set (`test_assemble`'s
6-4 set), so what the pipeline stages can be compared exactly with what
`assemble` makes of the same events in one pass.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from swingvision_import import footage, review
from swingvision_import.config import ImportConfig
from swingvision_import.load import UnresolvedReviewError
from swingvision_import.pipeline import SwingVisionImportPipeline
from tracking import assemble, court, events, models, players, stitch
from tracking import ball as ball_stage
from tracking import pipeline as tp
from tracking.assemble import MatchSetup, TrackingBlock
from tracking.calibrate import CalibrationResult
from tracking.config import PipelineConfig, PlayerConfig, RetrainingConfig, TrackerConfig
from tracking.events import EventTrack
from tracking.players import PersonDetection
from tracking.provenance import Provenance, Tagged
from tracking.strokes import Pose, StrokeLabel
from tracking.video import PtsIndex, VideoFrame, VideoInfo, VideoSource

from . import points as P
from .conftest import SIZE, exact_calibration, ground_homography, make_camera
from .test_assemble import a_set

SETUP = MatchSetup.from_start_end("Host", "Guest", "near")
REFERENCE = (70, 110, 140)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class SyntheticVideo:
    """A recording with no file behind it: a PTS index and tiny frames.

    Attributes:
        reads: (start, end) of every stretch decoded.
        loads: How many times the video was opened.
    """

    def __init__(
        self, duration=1200.0, fps=5.0, colour=REFERENCE, size=SIZE, vfr=False, shape=(9, 16)
    ):
        self.times = np.arange(0.0, duration + 1e-9, 1.0 / fps)
        self.fps, self.size, self.vfr = fps, size, vfr
        image = np.empty((*shape, 3), dtype=np.uint8)
        image[:] = colour
        image.setflags(write=False)
        self.image = image
        self.reads: list[tuple[float, float]] = []
        self.loads = 0

    def load(self, path):
        """Stands in for `video.load_video`."""
        self.loads += 1
        index = PtsIndex(self.times)
        info = VideoInfo(
            path=Path(path),
            width=self.size[0],
            height=self.size[1],
            rotation=0,
            codec="fake",
            nominal_fps=self.fps,
            average_fps=self.fps,
            duration=float(self.times[-1]),
            start_time=0.0,
            frame_count=len(self.times),
            is_vfr=self.vfr,
        )
        return VideoSource(info, index, index.stats(), "c0ffee" * 10)

    def frames(self, _path, start, end, index):
        """Stands in for `video.iter_frames`."""
        self.reads.append((start, end))
        for number in range(len(self.times)):
            pts = float(self.times[number])
            if start - 1e-9 <= pts <= end + 1e-9:
                yield VideoFrame(pts, self.image, number if index is not None else None)


class StandingPlayers:
    """A person detector that sees both players standing still at their ends."""

    name = "standing-v1"

    def __init__(self, calibration, near=(1.0, -0.5), far=(-1.0, 24.3)):
        left, top, _, _ = players.player_region(calibration)
        self.boxes = []
        for spot in (near, far):
            foot = calibration.to_image(np.array(spot, dtype=float))
            box = (foot[0] - 20, foot[1] - 80, foot[0] + 20, foot[1])
            self.boxes.append((box[0] - left, box[1] - top, box[2] - left, box[3] - top))
        self.calls = 0

    def detect(self, _image):
        """Both players, every time."""
        self.calls += 1
        return [PersonDetection(tuple(float(v) for v in box), 0.9) for box in self.boxes]


class NoBall:
    """A ball detector that never finds anything."""

    name = "no-ball-v1"
    span = 1

    def prepare(self, _image):
        """Nothing to prepare."""
        return None

    def detect(self, _before, _current, _after):
        """No candidates."""
        return []


class ScriptedEvents:
    """Replays a scripted event track, seen through each chunk's window.

    Attributes:
        windows: (first, last) frame time of every stretch it was asked
            about - one per chunk tracked.
    """

    def __init__(self, track: EventTrack, jitter=None):
        self.track = track
        self.jitter = jitter or (lambda pts, low, high: 0.0)
        self.windows: list[tuple[float, float]] = []

    def __call__(self, ball, _players, _calibration, _audio, config):
        """Gives the scripted events inside the ball track's stretch."""
        low, high = float(ball.frame_pts[0]), float(ball.frame_pts[-1])
        self.windows.append((low, high))
        hits = [
            replace(h, pts=h.pts + self.jitter(h.pts, low, high))
            for h in self.track.hits
            if low <= h.pts <= high
        ]
        bounces = [b for b in self.track.bounces if low <= b.pts <= high]
        nets = [n for n in self.track.nets if low <= n.pts <= high]
        shots, kept = events.join_shots(hits, bounces, nets, config)
        return EventTrack(tuple(hits), tuple(kept), tuple(nets), tuple(shots), ball, config)


class ScriptedStrokes:
    """A stroke classifier that names each hit as the script did."""

    name = "scripted-strokes-v1"

    def __init__(self, track, labels):
        self.by_time = [(h.pts, label.stroke.value) for h, label in zip(track.hits, labels)]
        self.contexts = []

    def classify(self, context):
        """The scripted stroke of the nearest scripted hit."""
        self.contexts.append(context)
        _, stroke = min(self.by_time, key=lambda pair: abs(pair[0] - context.hit.pts))
        sure = None if stroke == "unknown" else 0.7
        return StrokeLabel(
            context.index, Tagged(stroke, Provenance.MEASURED, sure, "test"), None, "scripted"
        )


class FixedPose:
    """A pose estimator that always finds the same keypoints."""

    name = "fixed-pose-v1"

    def __init__(self):
        self.calls = 0

    def estimate(self, _image, box):
        """Seventeen keypoints inside the box."""
        self.calls += 1
        points = np.tile([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], (17, 1))
        return Pose(keypoints=points, confidence=np.full(17, 0.9))


def _ok(calibration):
    """A calibrator that always succeeds with `calibration`."""
    calls = []

    def calibrator(_video, at):
        calls.append(at)
        return CalibrationResult(calibration, None, calibration.quality)

    calibrator.calls = calls
    return calibrator


@pytest.fixture
def scripted():
    """The 6-4 set, as events and their stroke labels."""
    return P.build(*a_set())


@pytest.fixture
def import_config(tmp_path):
    """Staging into a scratch folder, never the real pending folder."""
    return ImportConfig(
        pending_dir=tmp_path / "pending",
        db_path=tmp_path / "rally.db",
    )


def make_pipeline(
    calibration,
    scripted,
    import_config,
    video=None,
    config=None,
    jitter=None,
    **stages,
):
    """A pipeline wired to fakes; returns it with its fakes."""
    video = video or SyntheticVideo()
    track, labels = scripted
    fakes = SimpleNamespace(
        video=video,
        people=StandingPlayers(calibration),
        events=ScriptedEvents(track, jitter),
        strokes=ScriptedStrokes(track, labels),
        calibrator=_ok(calibration),
    )
    parts = dict(
        person_detector=fakes.people,
        use_pose=False,
        ball_detector=lambda _cal, _fps: NoBall(),
        stroke_classifier=fakes.strokes,
        event_detector=fakes.events,
        audio_reader=lambda *_: None,
        frame_reader=video.frames,
        video_loader=video.load,
        calibrator=fakes.calibrator,
    )
    parts.update(stages)
    config = config or TrackerConfig(pipeline=PipelineConfig(chunk_seconds=300.0))
    return tp.TrackerPipeline(config, import_config, tp.TrackerStages(**parts)), fakes


def ingest(pipeline, tmp_path, **kwargs):
    """Stages the scripted match from the synthetic video."""
    arguments = dict(date="2026-08-21", opponent="Guest", result="W")
    arguments.update(kwargs)
    return pipeline.ingest(tmp_path / "match.mp4", SETUP, **arguments)


def one_pass_rows(scripted):
    """What `assemble` makes of the scripted events directly, as dicts."""
    track, labels = scripted
    match = assemble.assemble(track, labels, SETUP, tracking_start=0.0)
    return [asdict(row) for row in match.export.shots]


# --------------------------------------------------------------------------
# Staging, end to end
# --------------------------------------------------------------------------


def test_a_tracked_video_is_staged_exactly_like_an_import(
    calibration, scripted, import_config, tmp_path
):
    pipeline, fakes = make_pipeline(calibration, scripted, import_config)
    path = ingest(pipeline, tmp_path, energy_rating=4)

    record = review.load_pending(path)
    assert path.parent == import_config.pending_dir
    assert record.source_file == str((tmp_path / "match.mp4").resolve())
    assert record.energy_rating == 4
    points = [p for s in record.sets for p in s.points]
    assert len(points) == 40
    (the_set,) = record.sets
    assert (the_set.games_won, the_set.games_lost) == (6, 4)
    assert all(p.needs_review for p in points)
    assert len(review.unresolved_flags(record)) == 40
    # The rows staged are the rows a single pass of assemble emits.
    assert record.tracking["shots"] == one_pass_rows(scripted)
    assert record.tracking["producer"] == "independent-tracker"
    assert record.tracking["clock"]["offset_seconds"] == 0.0
    assert record.tracking["models"]["strokes"] == "scripted-strokes-v1"
    assert len(fakes.events.windows) == 4  # 1200s in 300s chunks
    # Every point carries its own attempt's trace.
    assert all(p.tracking["number"] == p.source_point_number for p in points)
    assert all(p.tracking["shots"] for p in points)
    # The notes say where the match came from and how much was tracked.
    assert record.import_notes[0].startswith("Staged by the independent tracker")
    notes = record.import_notes
    assert any(n.startswith("Tracked ") and "with confidence" in n for n in notes)
    assert any("4 chunks" in note for note in record.import_notes)


def test_the_tracking_block_loads_back_and_round_trips(
    calibration, scripted, import_config, tmp_path
):
    pipeline, _ = make_pipeline(calibration, scripted, import_config)
    record = review.load_pending(ingest(pipeline, tmp_path))
    block = TrackingBlock.from_dict(record.tracking)
    assert block.source == assemble.SOURCE
    assert block.extra["producer"] == "independent-tracker"
    assert len(block.attempts) == 40
    first = block.attempts[0]
    assert first.shots[0].fields["video_time"].provenance is Provenance.MEASURED
    assert isinstance(first.shots[0].fields["hit_xy"].value, tuple)
    assert json.loads(json.dumps(block.to_dict())) == record.tracking
    # A point's block reads back the same way.
    point = record.sets[0].points[0]
    assert assemble.AttemptBlock.from_dict(point.tracking).number == point.source_point_number


def test_chunking_changes_nothing_the_reviewer_sees(calibration, scripted, import_config, tmp_path):
    whole, _ = make_pipeline(
        calibration,
        scripted,
        import_config,
        config=TrackerConfig(pipeline=PipelineConfig(chunk_seconds=5000.0)),
    )
    small, fakes = make_pipeline(
        calibration,
        scripted,
        replace(import_config, pending_dir=tmp_path / "other"),
        config=TrackerConfig(pipeline=PipelineConfig(chunk_seconds=97.0)),
    )
    one = review.load_pending(ingest(whole, tmp_path))
    many = review.load_pending(ingest(small, tmp_path))
    assert len(fakes.events.windows) == 13
    assert many.tracking["shots"] == one.tracking["shots"]
    assert [asdict(s) for s in many.sets] == [asdict(s) for s in one.sets]


def _edge_case(scripted):
    """A chunk length that puts a chunk edge exactly on a scripted hit."""
    track, _ = scripted
    hit = track.hits[25].pts
    return hit, PipelineConfig(chunk_seconds=hit)


@pytest.mark.parametrize(
    "left, right",
    [(-0.01, 0.01), (0.01, -0.01), (0.0, 0.0)],
    ids=["neither-chunk-owns-its-copy", "both-chunks-own-a-copy", "both-agree"],
)
def test_a_hit_on_a_chunk_edge_is_kept_exactly_once(
    calibration, scripted, import_config, tmp_path, left, right
):
    edge, config = _edge_case(scripted)

    def jitter(pts, low, high):
        """The chunk ending at the edge times the hit `left`, the next `right`."""
        if abs(pts - edge) > 1e-9:
            return 0.0
        return left if high < edge + 30 else right

    pipeline, fakes = make_pipeline(
        calibration, scripted, import_config, config=TrackerConfig(pipeline=config), jitter=jitter
    )
    record = review.load_pending(ingest(pipeline, tmp_path))
    times = [row["video_time"] for row in record.tracking["shots"]]
    near_edge = [t for t in times if abs(t - edge) < 0.1]
    assert len(near_edge) == 1
    assert len(times) == len(one_pass_rows(scripted))
    assert record.tracking["run"]["stitch"]["matched"] >= 1


def test_a_killed_run_resumes_without_redoing_finished_chunks(
    calibration, scripted, import_config, tmp_path
):
    pipeline, fakes = make_pipeline(calibration, scripted, import_config)

    class Killed(Exception):
        """Stands in for the OS killing the process."""

    def kill_after_two(progress):
        if progress.stage == "tracking" and progress.chunks_done == 2:
            raise Killed

    with pytest.raises(Killed):
        ingest(pipeline, tmp_path, progress=kill_after_two)
    assert len(fakes.events.windows) == 2
    assert not import_config.pending_dir.joinpath("2026-08-21_Guest.json").exists()

    seen = []
    resumed, again = make_pipeline(calibration, scripted, import_config)
    path = ingest(resumed, tmp_path, progress=seen.append)
    # Only the two unfinished chunks were tracked again.
    assert [round(low) for low, _ in again.events.windows] == [592, 892]
    assert seen[-1].stage == "staged" and seen[-1].chunks_resumed == 2
    record = review.load_pending(path)
    assert record.tracking["shots"] == one_pass_rows(scripted)
    chunks = record.tracking["run"]["chunks"]
    assert [c["resumed"] for c in chunks] == [True, True, False, False]


def test_progress_reports_chunks_and_an_eta(calibration, scripted, import_config, tmp_path):
    pipeline, _ = make_pipeline(calibration, scripted, import_config)
    seen = []
    ingest(pipeline, tmp_path, progress=seen.append)
    stages = [p.stage for p in seen]
    assert stages[:2] == ["opening", "calibrating"]
    assert stages[-2:] == ["assembling", "staged"]
    tracked = [p for p in seen if p.stage == "tracking"]
    assert tracked[0].eta_seconds is None
    after_one = next(p for p in tracked if p.chunks_done == 1)
    assert after_one.eta_seconds is not None and after_one.seconds_per_video_minute > 0
    assert after_one.video_seconds_done == pytest.approx(300.0)
    assert tracked[-1].chunks_done == tracked[-1].chunks_total == 4


def test_where_a_match_will_be_staged_is_known_before_tracking(
    calibration, scripted, import_config, tmp_path
):
    pipeline, fakes = make_pipeline(calibration, scripted, import_config)
    video = tmp_path / "match.mp4"
    where = pipeline.staged_path(video, date="2026-08-21", opponent="Guest", result="W")
    status = pipeline.progress_path(where)
    assert not where.exists() and fakes.video.loads == 0
    assert ingest(pipeline, tmp_path) == where
    final = json.loads(status.read_text())
    assert final["stage"] == "staged" and final["chunks_done"] == 4
    # Another match against the same opponent that day gets its own file.
    other = pipeline.staged_path(tmp_path / "other.mp4", date="2026-08-21", opponent="Guest",
                                 result="W")
    assert other != where and other.name == "2026-08-21_Guest_2.json"


def test_a_progress_file_that_cannot_be_written_is_only_logged(tmp_path, monkeypatch, caplog):
    tp._write_progress(tmp_path / "status" / "x.json", tp.TrackingProgress("opening"))
    assert json.loads((tmp_path / "status" / "x.json").read_text())["stage"] == "opening"

    def refuse(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(tp, "_atomic_write", refuse)
    tp._write_progress(tmp_path / "y.json", tp.TrackingProgress("opening"))
    assert "disk full" in caplog.text and not (tmp_path / "y.json").exists()


def test_a_staged_video_is_skipped_unless_forced(calibration, scripted, import_config, tmp_path):
    pipeline, fakes = make_pipeline(calibration, scripted, import_config)
    first = ingest(pipeline, tmp_path)
    seen = []
    assert ingest(pipeline, tmp_path, progress=seen.append) == first
    assert fakes.video.loads == 1  # not even opened again
    assert seen[0].stage == "skipped" and "force" in seen[0].message

    forced = ingest(pipeline, tmp_path, force=True)
    assert forced == first
    assert fakes.video.loads == 2
    assert len(fakes.events.windows) == 8  # every chunk tracked again


def test_checkpoints_from_other_settings_are_refused_until_forced(
    calibration, scripted, import_config, tmp_path
):
    pipeline, _ = make_pipeline(calibration, scripted, import_config)

    def kill(progress):
        if progress.stage == "tracking" and progress.chunks_done == 1:
            raise RuntimeError("killed")

    with pytest.raises(RuntimeError):
        ingest(pipeline, tmp_path, progress=kill)
    other = TrackerConfig(
        players=PlayerConfig(stride=6), pipeline=PipelineConfig(chunk_seconds=300.0)
    )
    changed, _ = make_pipeline(calibration, scripted, import_config, config=other)
    with pytest.raises(tp.CheckpointMismatchError, match="fingerprint.*--force"):
        ingest(changed, tmp_path)
    assert review.load_pending(ingest(changed, tmp_path, force=True)).sets


def test_a_stale_manifest_with_no_chunks_is_simply_replaced(
    calibration, scripted, import_config, tmp_path
):
    pipeline, _ = make_pipeline(calibration, scripted, import_config)
    source = SyntheticVideo().load(tmp_path / "match.mp4")
    folder = pipeline.work_folder(source, 0.0, 1200.0)
    tp.CheckpointStore(folder).write_manifest({"fingerprint": "something else"})
    assert ingest(pipeline, tmp_path).exists()


def test_checkpoints_can_be_discarded_once_staged(calibration, scripted, import_config, tmp_path):
    config = TrackerConfig(pipeline=PipelineConfig(chunk_seconds=300.0, keep_checkpoints=False))
    pipeline, _ = make_pipeline(calibration, scripted, import_config, config=config)
    record = review.load_pending(ingest(pipeline, tmp_path))
    assert not Path(record.tracking["run"]["work_folder"]).exists()
    kept, _ = make_pipeline(
        calibration, scripted, replace(import_config, pending_dir=tmp_path / "kept")
    )
    record = review.load_pending(ingest(kept, tmp_path))
    folder = Path(record.tracking["run"]["work_folder"])
    assert folder.parent == tmp_path / "kept" / "tracking_work"
    assert sorted(p.name for p in folder.iterdir()) == [
        "calibration.json",
        "chunk_0000.pkl",
        "chunk_0001.pkl",
        "chunk_0002.pkl",
        "chunk_0003.pkl",
        "manifest.json",
    ]


def test_finalize_refuses_while_any_flag_remains(calibration, scripted, import_config, tmp_path):
    pipeline, _ = make_pipeline(calibration, scripted, import_config)
    path = ingest(pipeline, tmp_path)
    importer = SwingVisionImportPipeline(import_config)
    with pytest.raises(UnresolvedReviewError):
        importer.finalize(path)
    record = review.load_pending(path)
    points = [(s, p) for s in record.sets for p in s.points]
    for set_record, point in points[:-1]:
        review.confirm_point(
            record,
            set_number=set_record.set_number,
            game_number=point.game_number,
            point_number=point.point_number,
            point_end_type=point.point_end_type,
            point_won=point.point_won,
            net_approach=point.net_approach,
        )
    review.save_pending(record, import_config.pending_dir)
    with pytest.raises(UnresolvedReviewError):
        importer.finalize(path)
    set_record, point = points[-1]
    review.confirm_point(
        record,
        set_number=set_record.set_number,
        game_number=point.game_number,
        point_number=point.point_number,
        point_end_type=point.point_end_type,
        point_won=point.point_won,
        net_approach=point.net_approach,
    )
    review.save_pending(record, import_config.pending_dir)
    assert importer.finalize(path) >= 1


# --------------------------------------------------------------------------
# Calibration: abstain, or the user's own keypoints
# --------------------------------------------------------------------------


def test_an_abstaining_calibration_stops_the_run_and_says_what_to_do(
    calibration, scripted, import_config, tmp_path
):
    refuse = lambda _video, _at: CalibrationResult(None, "The far baseline was not found.")  # noqa: E731
    pipeline, fakes = make_pipeline(calibration, scripted, import_config, calibrator=refuse)
    with pytest.raises(tp.CalibrationFailedError) as caught:
        ingest(pipeline, tmp_path, start=100.0)
    message = str(caught.value)
    assert "at 100.0s" in message and "The far baseline was not found." in message
    assert "Re-aim the camera" in message and "--keypoints" in message
    assert "near_baseline:" in message
    assert fakes.events.windows == []  # nothing tracked
    staged = import_config.pending_dir
    assert not staged.exists() or not list(staged.glob("*.json"))


def _keypoints(camera, size=SIZE):
    """Every doubles keypoint, as a person would mark it."""
    template = court.court_template("doubles")
    names = list(template.keypoints)
    marks = np.array([template.keypoints[name] for name in names], dtype=float)
    pixels = court.project(ground_homography(camera), marks)
    return {name: (float(x), float(y)) for name, (x, y) in zip(names, pixels)}


def test_marked_keypoints_calibrate_instead_of_detection(
    camera, calibration, scripted, import_config, tmp_path
):
    config = TrackerConfig(pipeline=PipelineConfig(chunk_seconds=300.0, check_drift=False))
    never = lambda *_: pytest.fail("automatic calibration was not wanted")  # noqa: E731
    pipeline, _ = make_pipeline(
        calibration, scripted, import_config, config=config, calibrator=never
    )
    marked = tp.ManualKeypoints(_keypoints(camera))
    record = review.load_pending(ingest(pipeline, tmp_path, keypoints=marked))
    assert record.tracking["calibration"]["provenance"] == "user_entered"
    assert record.tracking["shots"] == one_pass_rows(scripted)


@pytest.mark.parametrize(
    "marked, message",
    [
        (lambda kp: tp.ManualKeypoints(kp, image_size=(640, 360)), "640x360 frame"),
        (lambda kp: tp.ManualKeypoints({"nowhere": (1.0, 2.0)}), "Unknown court keypoint"),
        (
            lambda kp: tp.ManualKeypoints(dict(list(kp.items())[:3])),
            "did not give a usable calibration",
        ),
    ],
    ids=["wrong-frame-size", "unknown-name", "too-few-points"],
)
def test_unusable_keypoints_are_refused_with_the_reason(
    camera, calibration, scripted, import_config, tmp_path, marked, message
):
    pipeline, _ = make_pipeline(calibration, scripted, import_config)
    with pytest.raises(tp.CalibrationFailedError, match=message):
        ingest(pipeline, tmp_path, keypoints=marked(_keypoints(camera)))


def test_keypoints_read_from_either_json_shape(tmp_path):
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"near_baseline:left_singles": [10, 20]}))
    loaded = tp.ManualKeypoints.from_file(bare)
    assert loaded.points == {"near_baseline:left_singles": (10.0, 20.0)}
    assert loaded.image_size is None and loaded.template_kind == "doubles"
    full = tmp_path / "full.json"
    full.write_text(
        json.dumps({"points": {"far_service:center": [1, 2]}, "image_size": [960, 540],
                    "template": "singles"})
    )
    loaded = tp.ManualKeypoints.from_file(full)
    assert loaded.image_size == (960, 540) and loaded.template_kind == "singles"
    assert loaded.identity().startswith("manual:") and loaded.identity() != tp.ManualKeypoints(
        loaded.points
    ).identity()


@pytest.mark.parametrize(
    "text, message",
    [
        ("{not json", "not valid JSON"),
        ("[1, 2]", "expected a JSON object"),
        ('{"a": [1]}', "must be an \\[x, y\\] pair"),
        ('{"points": {"a": [1, 2]}, "image_size": [3]}', "image_size must be"),
    ],
)
def test_malformed_keypoint_files_are_refused(tmp_path, text, message):
    path = tmp_path / "keypoints.json"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        tp.ManualKeypoints.from_file(path)


# --------------------------------------------------------------------------
# Models: injected, loaded, or fallen back from
# --------------------------------------------------------------------------


def test_without_loadable_models_the_classic_detector_is_used_and_noted(
    calibration, scripted, import_config, tmp_path, monkeypatch
):
    def missing(_config=None):
        raise models.BackendMissingError("no ultralytics here")

    monkeypatch.setattr(models, "load_person_model", missing)
    monkeypatch.setattr(models, "load_pose_model", missing)
    full_size = SyntheticVideo(shape=(SIZE[1], SIZE[0]))
    pipeline, _ = make_pipeline(
        calibration, scripted, import_config, video=full_size, person_detector=None, use_pose=True
    )
    record = review.load_pending(ingest(pipeline, tmp_path))
    assert record.tracking["models"]["person"] == "motion-person-v1"
    assert record.tracking["models"]["pose"] is None
    notes = " ".join(record.import_notes)
    assert "classic motion detector" in notes and "no ultralytics here" in notes
    assert "named from the ball alone" in notes


class _Boxes:
    """Ultralytics-shaped detection output."""

    def __init__(self, boxes=(), scores=()):
        self.xyxy = np.array(boxes, dtype=float).reshape(-1, 4)
        self.conf = np.array(scores, dtype=float)


class _Yolo:
    """An ultralytics-shaped model that finds nobody."""

    def predict(self, *_args, **_kwargs):
        return [SimpleNamespace(boxes=_Boxes(), keypoints=None)]


def test_the_pinned_models_are_loaded_when_none_are_injected(
    calibration, scripted, import_config, tmp_path, monkeypatch
):
    def loaded(name):
        return lambda _config=None: SimpleNamespace(model=_Yolo(), source=f"{name}@fake-1")

    monkeypatch.setattr(models, "load_person_model", loaded("yolo11n"))
    monkeypatch.setattr(models, "load_pose_model", loaded("yolo11n-pose"))
    pipeline, _ = make_pipeline(
        calibration, scripted, import_config, person_detector=None, use_pose=True
    )
    record = review.load_pending(ingest(pipeline, tmp_path))
    assert record.tracking["models"]["person"] == "yolo11n@fake-1"
    assert record.tracking["models"]["pose"] == "yolo11n-pose@fake-1"


def test_poses_are_taken_only_around_kept_hits_and_reach_the_classifier(
    calibration, scripted, import_config, tmp_path
):
    config = TrackerConfig(
        players=PlayerConfig(stride=1), pipeline=PipelineConfig(chunk_seconds=300.0)
    )
    pose = FixedPose()
    pipeline, fakes = make_pipeline(
        calibration, scripted, import_config, config=config, pose_estimator=pose, use_pose=True
    )
    record = review.load_pending(ingest(pipeline, tmp_path))
    assert pose.calls > 0
    assert any(context.poses for context in fakes.strokes.contexts)
    # Pose decoding read short stretches around hits, not whole chunks.
    extra = [(low, high) for low, high in fakes.video.reads if high - low < 60]
    assert extra and all(high - low < 20 for low, high in extra)
    assert record.tracking["models"]["pose"] == "fixed-pose-v1"


def test_hits_close_together_share_one_decoded_stretch(calibration, import_config, tmp_path):
    quick = P.build(P.rally(P.NEAR, "deuce", 5, gap=0.8), start=20.0)
    config = TrackerConfig(
        players=PlayerConfig(stride=1), pipeline=PipelineConfig(chunk_seconds=300.0)
    )
    video = SyntheticVideo(duration=60.0)
    pipeline, _ = make_pipeline(
        calibration, quick, import_config, video=video, config=config,
        pose_estimator=FixedPose(), use_pose=True,
    )
    ingest(pipeline, tmp_path)
    # One read for the chunk, one stretch covering all five hits.
    assert len(video.reads) == 2
    low, high = video.reads[1]
    assert low == pytest.approx(20.0 - 0.35) and high == pytest.approx(23.2 + 0.15)


def test_a_video_without_a_checksum_is_identified_by_size_and_time(
    calibration, scripted, import_config, tmp_path
):
    pipeline, fakes = make_pipeline(calibration, scripted, import_config)
    video = tmp_path / "match.mp4"
    video.write_bytes(b"x")
    source = replace(fakes.video.load(video), checksum=None)
    folder = pipeline.work_folder(source, 0.0, 10.0)
    assert folder.name.startswith("match_") and folder.name.endswith("_0.00-10.00")
    video.write_bytes(b"xy")
    assert pipeline.work_folder(source, 0.0, 10.0) != folder


def test_a_scripted_model_that_never_sees_a_hit_needs_no_pose(
    calibration, scripted, import_config, tmp_path
):
    track, labels = scripted
    empty = (replace(track, hits=(), bounces=(), nets=(), shots=()), labels)
    pose = FixedPose()
    pipeline, _ = make_pipeline(
        calibration, empty, import_config, pose_estimator=pose, use_pose=True
    )
    record = review.load_pending(ingest(pipeline, tmp_path))
    assert pose.calls == 0
    assert record.sets == [] and record.tracking["shots"] == []


# --------------------------------------------------------------------------
# "Retraining recommended"
# --------------------------------------------------------------------------


def _ball(states, confidence=0.6, fps=10.0):
    """A compact ball track from a per-frame state string ("M", "B", ".")."""
    count = len(states)
    code = {"M": stitch.MEASURED_STATE, "B": stitch.BRIDGED_STATE, ".": stitch.UNKNOWN_STATE}
    state = np.array([code[c] for c in states], dtype=np.int8)
    return stitch.CompactBallTrack(
        frame_numbers=np.arange(count),
        frame_pts=np.arange(count) / fps,
        state=state,
        confidence=np.where(state > 0, confidence, 0.0),
        image_xy=np.zeros((count, 2)),
        segment=np.where(state > 0, 0, -1),
        image_size=SIZE,
        source="test",
    )


def _quality(shots=10, unknown=0):
    """An event quality with `unknown` of `shots` results unknown."""
    track = EventTrack((), (), (), (), stitch.CompactBallTrack.concatenate([_ball("M")]))
    found = track.quality()
    results = dict(found.results, unknown=unknown)
    return replace(found, shots=shots, results=results)


def _scene(brightness=120.0, rgb=REFERENCE):
    """A scene summary."""
    return ball_stage.SceneStats(30, brightness, 20.0, tuple(float(v) for v in rgb), -40.0)


def test_good_tracking_in_reference_conditions_recommends_nothing():
    advice = tp.retraining_advice(_ball("M" * 100), [(0.0, 9.9)], _quality(), _scene())
    assert not advice.recommended and advice.note is None
    assert advice.measures["rally_coverage"] == 1.0
    assert advice.provisional


def test_weak_tracking_at_night_recommends_retraining_and_says_why():
    advice = tp.retraining_advice(
        _ball("M" * 20 + "." * 80, confidence=0.2), [(0.0, 9.9)], _quality(), _scene(40.0)
    )
    assert advice.recommended and advice.weak_tracking
    assert advice.note.startswith("Ball tracking was weak on this match (night or dusk lighting")
    assert "Retraining the ball model on footage like this is recommended." in advice.note
    assert "provisional" in advice.note
    assert "20% of rally frames" in advice.note


def test_weak_tracking_in_daylight_is_still_flagged():
    advice = tp.retraining_advice(
        _ball("M" * 30 + "." * 70), [(0.0, 9.9)], _quality(), _scene()
    )
    assert advice.recommended and advice.conditions == ()
    assert "in conditions close to the daylight reference" in advice.note


def test_unfamiliar_conditions_alone_recommend_retraining():
    advice = tp.retraining_advice(
        _ball("M" * 100), [(0.0, 9.9)], _quality(), _scene(200.0, (200, 60, 60))
    )
    assert advice.recommended and not advice.weak_tracking
    assert len(advice.conditions) == 2
    assert "much brighter light" in advice.note and "court colour unlike" in advice.note
    assert "held up here" in advice.note


def test_many_weak_points_or_unknown_results_count_as_weak_tracking():
    patchy = _ball(("M" * 9 + ".") * 6 + "." * 40)
    stretches = [(k * 1.0, k * 1.0 + 0.9) for k in range(10)]
    advice = tp.retraining_advice(patchy, stretches, _quality(), _scene())
    assert "weak on 4 of 10 points" not in (advice.note or "")
    assert advice.measures["weak_points"] == 4
    advice = tp.retraining_advice(_ball("M" * 100), [(0.0, 9.9)], _quality(10, 8), _scene())
    assert advice.recommended and "no result for 80% of shots" in advice.note
    loose = RetrainingConfig(weak_point_share=0.3)
    advice = tp.retraining_advice(patchy, stretches, _quality(), _scene(), loose)
    assert "weak on 4 of 10 points" in advice.note


def test_no_scene_and_no_points_give_no_advice():
    advice = tp.retraining_advice(_ball("."), [], _quality(0), None)
    assert not advice.recommended and advice.measures["scene"] is None
    assert json.loads(json.dumps(advice.to_dict()))["conditions"] == []


def test_a_dark_weakly_tracked_match_carries_the_notice(
    calibration, scripted, import_config, tmp_path
):
    dark = SyntheticVideo(colour=(15, 20, 30))
    pipeline, _ = make_pipeline(calibration, scripted, import_config, video=dark)
    record = review.load_pending(ingest(pipeline, tmp_path))
    notice = record.import_notes[1]
    assert notice.startswith("Ball tracking was weak on this match (night or dusk lighting")
    assert record.tracking["retraining"]["recommended"] is True
    assert record.tracking["scene"]["brightness"] < 30


# --------------------------------------------------------------------------
# Notes, ranges and drift
# --------------------------------------------------------------------------


def test_a_partial_run_says_its_score_is_not_the_matchs(
    calibration, scripted, import_config, tmp_path
):
    pipeline, fakes = make_pipeline(calibration, scripted, import_config)
    record = review.load_pending(ingest(pipeline, tmp_path, start=100.0, end=700.0))
    assert record.tracking["range"] == [100.0, 700.0]
    assert any(n.startswith("Only 100-700s of the 1200s recording") for n in record.import_notes)
    assert fakes.events.windows[0][0] == pytest.approx(100.0)
    assert fakes.calibrator.calls[0] == pytest.approx(100.0)
    assert all(row["video_time"] >= 100.0 for row in record.tracking["shots"])


def test_an_empty_range_is_refused(calibration, scripted, import_config, tmp_path):
    pipeline, _ = make_pipeline(calibration, scripted, import_config)
    with pytest.raises(ValueError, match="Nothing to track"):
        ingest(pipeline, tmp_path, start=1300.0)


def test_dropped_frames_are_noted(calibration, scripted, import_config, tmp_path):
    dropping = SyntheticVideo(vfr=True)
    pipeline, _ = make_pipeline(calibration, scripted, import_config, video=dropping)
    record = review.load_pending(ingest(pipeline, tmp_path))
    assert any("drops frames now and then" in note for note in record.import_notes)


def test_a_camera_that_moved_mid_match_is_noted(
    calibration, scripted, import_config, tmp_path
):
    moved = exact_calibration(make_camera(lateral=0.4))

    def calibrator(_video, at):
        if at < 1.0:
            return CalibrationResult(calibration, None)
        if at < 400:
            return CalibrationResult(None, "players in the way")
        if at < 700:
            raise tp.video_stage.VideoUnreadableError("decode failed")
        return CalibrationResult(moved, None)

    pipeline, _ = make_pipeline(calibration, scripted, import_config, calibrator=calibrator)
    record = review.load_pending(ingest(pipeline, tmp_path))
    chunks = record.tracking["run"]["chunks"]
    assert [c["drift_px"] is None for c in chunks] == [True, True, True, False]
    assert chunks[3]["drift_px"] > 6
    note = next(n for n in record.import_notes if n.startswith("The court moved"))
    assert "from about 900s" in note


# --------------------------------------------------------------------------
# Checkpoints that cannot be read
# --------------------------------------------------------------------------


def test_unreadable_checkpoints_are_tracked_again(calibration, scripted, import_config, tmp_path):
    pipeline, fakes = make_pipeline(calibration, scripted, import_config)

    def kill(progress):
        if progress.stage == "tracking" and progress.chunks_done == 3:
            raise RuntimeError("killed")

    with pytest.raises(RuntimeError):
        ingest(pipeline, tmp_path, progress=kill)
    store = tp.CheckpointStore(
        pipeline.work_folder(fakes.video.load(tmp_path / "match.mp4"), 0.0, 1200.0)
    )
    store.chunk_path(0).write_bytes(b"not a pickle")
    store.chunk_path(1).write_bytes(pickle.dumps({"format": 999, "result": None}))
    store.chunk_path(2).write_bytes(pickle.dumps({"format": tp.CHECKPOINT_FORMAT, "result": 1}))
    again, fresh = make_pipeline(calibration, scripted, import_config)
    record = review.load_pending(ingest(again, tmp_path))
    assert len(fresh.events.windows) == 4
    assert record.tracking["shots"] == one_pass_rows(scripted)


def test_a_store_without_files_reads_as_empty(tmp_path):
    store = tp.CheckpointStore(tmp_path / "nothing")
    assert store.read_manifest() is None
    assert store.load_calibration() is None
    assert store.load_chunk(0) is None and not store.has_chunk(0)
    store.folder.mkdir()
    store.calibration_path.write_text("{}")
    assert store.load_calibration() is None
    store.clear()
    assert not store.folder.exists()


def test_a_chunk_lost_before_assembly_stops_the_run(
    calibration, scripted, import_config, tmp_path
):
    pipeline, fakes = make_pipeline(calibration, scripted, import_config)
    folder = pipeline.work_folder(fakes.video.load(tmp_path / "match.mp4"), 0.0, 1200.0)

    def lose(progress):
        if progress.stage == "assembling":
            tp.CheckpointStore(folder).chunk_path(1).unlink()

    with pytest.raises(tp.TrackingError, match=r"\[1\]"):
        ingest(pipeline, tmp_path, progress=lose)


# --------------------------------------------------------------------------
# The existing footage check, on a tracked match
# --------------------------------------------------------------------------


class _Verdicts:
    """A vision client that reports every point read clearly, server seen."""

    def __init__(self):
        self.calls = 0
        outer = self

        class _Messages:
            def create(self, **_kwargs):
                outer.calls += 1
                body = {
                    "footage_usable": True,
                    "server_seen": "tracked",
                    "winner": "tracked",
                    "point_end_type": "ace",
                    "tracked_at_net": "unclear",
                    "confidence": "high",
                    "reasoning": "scripted",
                }
                text = SimpleNamespace(type="text", text=json.dumps(body))
                usage = SimpleNamespace(input_tokens=100, output_tokens=10)
                return SimpleNamespace(content=[text], usage=usage, stop_reason="end_turn")

        self.beta = SimpleNamespace(messages=_Messages())


def test_the_footage_check_runs_on_a_tracked_match_on_the_videos_own_clock(
    calibration, scripted, import_config, tmp_path, monkeypatch
):
    pipeline, _ = make_pipeline(calibration, scripted, import_config)
    path = ingest(pipeline, tmp_path)
    record = review.load_pending(path)
    windows = []

    def frames(_video, window):
        windows.append(window)
        return [footage.Frame("finish", 0.0, b"\xff\xd8jpeg")]

    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 1200.0)
    client = _Verdicts()
    summary = SwingVisionImportPipeline(import_config).verify_footage(
        client,
        path,
        [Path(record.source_file)],
        footage_config=footage.FootageConfig(corroborate=False),
        frame_source=frames,
        clock_offset=assemble.video_clock_offset,
    )
    assert summary.points_played == 40
    assert client.calls == 40
    # The zero offset is trusted: no "could not confirm a shared clock" note.
    assert not any("share a clock" in note for note in summary.notes)
    first_times = sorted(w.first_shot for w in windows)
    serves = sorted(
        row["video_time"] for row in record.tracking["shots"] if row["shot_number"] == 1
    )
    assert first_times == pytest.approx(serves)
    # The clean holds - an unreturned serve by the host - are read as aces and confirmed.
    after = review.load_pending(path)
    confirmed = [p for s in after.sets for p in s.points if p.confirmed_by == "footage"]
    assert confirmed and all(p.point_end_type == "ace" and p.is_serving for p in confirmed)


def test_a_tracked_matchs_video_is_never_archived(calibration, scripted, import_config, tmp_path):
    pipeline, _ = make_pipeline(calibration, scripted, import_config)
    path = ingest(pipeline, tmp_path)
    video = tmp_path / "match.mp4"
    video.write_bytes(b"pretend footage")
    assert SwingVisionImportPipeline(import_config).archive_sources(path) is None
    assert video.read_bytes() == b"pretend footage"


# --------------------------------------------------------------------------
# A real (tiny) video, real decoding
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_clip(tmp_path_factory):
    """Forty seconds of testsrc at 10fps, 640x360, with no audio."""
    import subprocess

    path = tmp_path_factory.mktemp("clip") / "small.mp4"
    subprocess.run(
        [
            footage.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "testsrc=size=640x360:rate=10:duration=40", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def test_a_real_clip_is_decoded_chunk_by_chunk_and_staged(small_clip, import_config):
    size = (640, 360)
    camera = make_camera(size=size)
    calibration = exact_calibration(camera, size)
    script = P.build(
        P.rally(P.NEAR, "deuce", 4), P.serve(P.FAR, "ad"), P.serve(P.FAR, "deuce"),
        start=5.0, between=9.0,
    )
    config = TrackerConfig(pipeline=PipelineConfig(chunk_seconds=15.0, overlap_seconds=3.0))
    stages = tp.TrackerStages(
        person_detector=StandingPlayers(calibration),
        use_pose=False,
        ball_detector=lambda _cal, _fps: NoBall(),
        stroke_classifier=ScriptedStrokes(*script),
        event_detector=ScriptedEvents(script[0]),
    )
    pipeline = tp.TrackerPipeline(config, import_config, stages)
    # The real detector abstains on a test pattern - and says what to do.
    with pytest.raises(tp.CalibrationFailedError, match="--keypoints"):
        pipeline.ingest(small_clip, SETUP, date="2026-01-01", opponent="Guest", result="L")
    marked = tp.ManualKeypoints(_keypoints(camera, size))
    path = pipeline.ingest(
        small_clip, SETUP, date="2026-01-01", opponent="Guest", result="L", keypoints=marked
    )
    record = review.load_pending(path)
    assert record.tracking["video"]["width"] == 640 and record.tracking["video"]["checksum"]
    assert len(record.tracking["run"]["chunks"]) == 3
    assert len(stages.event_detector.windows) == 3
    track, labels = script
    direct = assemble.assemble(track, labels, SETUP, tracking_start=0.0)
    assert record.tracking["shots"] == [asdict(row) for row in direct.export.shots]
    assert record.tracking["scene"]["frames_sampled"] > 0
