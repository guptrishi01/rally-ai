"""Tests for stroke naming: the rules, the pose plumbing and the seams.

Every pose here is hand-made and every model a fake: no test loads a pose
model, imports torch or touches the network.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tracking import court, events, models, strokes
from tracking.config import ModelConfig, StrokeConfig
from tracking.events import HitEvent
from tracking.provenance import Provenance, Tagged
from tracking.strokes import HitContext, Pose, PoseSample, StrokeLabel
from tracking.video import VideoFrame

from . import rally
from .conftest import SIZE
from .test_events import BASE

NEAR_BOX = (400.0, 200.0, 480.0, 440.0)  # 240 px tall, centred on x = 440
FAR_BOX = (460.0, 150.0, 480.0, 200.0)  # 50 px tall, centred on x = 470


def hit(side="near", position=(1.0, -0.5), image_xy=(500.0, 330.0), **more):
    """A hand-made hit."""
    fields = dict(
        pts=10.0,
        time_uncertainty=0.02,
        side=side,
        player="A" if side == "near" else "B",
        zone="deuce",
        position=position,
        position_provenance=None if position is None else Provenance.MEASURED,
        foot_clipped=False,
        image_xy=image_xy,
        confidence=0.75,
        provenance=Provenance.MEASURED,
        cues=("kink",),
    )
    fields.update(more)
    return HitEvent(**fields)


def context(the_hit=None, box=NEAR_BOX, **more):
    """A hand-made context: mid-rally, from the back, nothing posed."""
    fields = dict(
        index=0,
        hit=the_hit or hit(),
        rival_quiet_before=1.5,
        watched_before=30.0,
        bounced_before=True,
        box=box,
        poses=(),
        handedness="right",
    )
    fields.update(more)
    return HitContext(**fields)


def pose(box=NEAR_BOX, wrist=(0.3, 0.5), sure=0.9, hand="right", **unsure):
    """A standing figure in a box, with one wrist placed by hand.

    Args:
        box: The figure's box.
        wrist: The hitting wrist, as (sideways offset from the body's
            centre in box heights, height down the box as a fraction).
        sure: Confidence of every keypoint...
        hand: Which wrist is the hitting one.
        **unsure: ...except these, by name: nose, shoulders, hips,
            wrist_sure.
    """
    tall = box[3] - box[1]
    centre = (box[0] + box[2]) / 2
    points = np.zeros((17, 2))
    confidence = np.full(17, sure)
    points[0] = (centre, box[1] + 0.08 * tall)
    points[5], points[6] = (
        (centre + 0.1 * tall, box[1] + 0.2 * tall),
        (
            centre - 0.1 * tall,
            box[1] + 0.2 * tall,
        ),
    )
    points[11], points[12] = (
        (centre + 0.08 * tall, box[1] + 0.5 * tall),
        (
            centre - 0.08 * tall,
            box[1] + 0.5 * tall,
        ),
    )
    hitting = 10 if hand == "right" else 9
    points[hitting] = (centre + wrist[0] * tall, box[1] + wrist[1] * tall)
    points[19 - hitting] = (centre, box[1] + 0.5 * tall)
    names = {"nose": [0], "shoulders": [5, 6], "hips": [11, 12], "wrist_sure": [hitting]}
    for name, value in unsure.items():
        confidence[names[name]] = value
    return Pose(points, confidence)


def sampled(*poses, at=9.9):
    """Poses as samples, all just before the hit unless told otherwise."""
    return tuple(PoseSample(at, one) for one in poses)


RULES = strokes.RuleStrokeClassifier()


def named(found: StrokeLabel) -> str:
    return found.stroke.value


# --------------------------------------------------------------------- pose


def test_a_keypoint_is_given_only_when_the_model_was_sure_of_it():
    figure = pose(wrist=(0.3, 0.5), nose=0.2)

    assert figure.point(0, 0.4) is None
    assert figure.point(10, 0.4) == pytest.approx((440 + 72, 320))


class _Tensor:
    """Mimics a torch tensor's route to numpy."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _FakePoseModel:
    """Mimics the one ultralytics call the pose wrapper makes."""

    def __init__(self, people, scores, with_confidence=True):
        self.calls = []
        self._people, self._scores, self._with = people, scores, with_confidence

    def predict(self, image, **options):
        self.calls.append((image.copy(), options))
        count = len(self._scores)
        points = np.zeros((count, 17, 2), dtype=np.float32)
        sure = np.zeros((count, 17), dtype=np.float32)
        for index, (spot, confidence) in enumerate(self._people):
            points[index] = spot
            sure[index] = confidence
        keypoints = SimpleNamespace(xy=_Tensor(points), conf=_Tensor(sure) if self._with else None)
        return [
            SimpleNamespace(keypoints=keypoints, boxes=SimpleNamespace(conf=_Tensor(self._scores)))
        ]


def test_the_pose_wrapper_crops_around_the_player_and_maps_keypoints_back():
    model = _FakePoseModel([((5.0, 7.0), 0.3), ((10.0, 20.0), 0.9)], [0.4, 0.8])
    estimator = strokes.YoloPoseEstimator(
        model, StrokeConfig(crop_pad=0.5, pose_size=256), name="fake@1"
    )
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    image[..., 0] = 200  # red, so the BGR flip can be seen

    found = estimator.estimate(image, (100.0, 100.0, 140.0, 200.0))

    crop, options = model.calls[0]
    # The 40 x 100 box grown by half its size each way: x 80..160, y 50..250.
    assert crop.shape[:2] == (201, 81)
    assert crop[0, 0].tolist() == [0, 0, 200]
    assert options == {"imgsz": 256, "verbose": False}
    # The more confident of the two people, shifted back into the whole frame.
    assert found.keypoints[0] == pytest.approx((90.0, 70.0))
    assert found.confidence[0] == pytest.approx(0.9)
    assert estimator.name == "fake@1"
    assert isinstance(estimator, strokes.PoseEstimator)


def test_the_pose_wrapper_says_none_when_nobody_is_found():
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    box = (100.0, 100.0, 140.0, 200.0)

    nobody = strokes.YoloPoseEstimator(_FakePoseModel([], []))
    no_confidence = strokes.YoloPoseEstimator(_FakePoseModel([((1.0, 1.0), 0.9)], [0.9], False))
    no_keypoints = SimpleNamespace(
        predict=lambda *a, **k: [SimpleNamespace(keypoints=None, boxes=None)]
    )

    assert nobody.estimate(image, box) is None
    assert no_confidence.estimate(image, box) is None
    assert strokes.YoloPoseEstimator(no_keypoints).estimate(image, box) is None
    # A box entirely off the frame leaves nothing to crop.
    assert nobody.estimate(image, (900.0, 900.0, 950.0, 990.0)) is None


def test_the_real_pose_model_is_loaded_lazily_and_once(monkeypatch):
    loads = []
    model = _FakePoseModel([((1.0, 1.0), 0.9)], [0.9])

    def fake_load(config):
        loads.append(config)
        return models.LoadedModel(models.POSE_YOLO, model, "yolo11n-pose@fake-1", None)

    monkeypatch.setattr(models, "load_pose_model", fake_load)
    estimator = strokes.YoloPoseEstimator()
    image = np.zeros((400, 600, 3), dtype=np.uint8)

    assert loads == [] and estimator.name == "yolo11n-pose"
    estimator.estimate(image, (100.0, 100.0, 140.0, 200.0))
    estimator.estimate(image, (100.0, 100.0, 140.0, 200.0))

    assert len(loads) == 1 and estimator.name == "yolo11n-pose@fake-1"
    assert "torch" not in sys.modules and "ultralytics" not in sys.modules


def test_the_pose_model_is_pinned_and_loaded_like_the_person_model(monkeypatch, tmp_path):
    created = []

    class FakeYolo:
        def __init__(self, path):
            created.append(path)

    backend = SimpleNamespace(YOLO=FakeYolo, __version__="9.9.9")
    monkeypatch.setattr(models, "import_backend", lambda name: backend)

    loaded = models.load_pose_model(ModelConfig(weights_dir=tmp_path, verify_checksum=False))

    assert models.model_spec("yolo11n-pose") is models.POSE_YOLO
    assert models.POSE_YOLO.licence == "AGPL-3.0" and len(models.POSE_YOLO.sha256) == 64
    assert loaded.source == "yolo11n-pose@ultralytics-9.9.9"
    assert created == [str((tmp_path / "yolo11n-pose.pt").resolve())]
    with pytest.raises(models.ModelError, match="not a pose model"):
        models.load_pose_model(ModelConfig(pose_model="yolo11n", weights_dir=tmp_path))


# ----------------------------------------------------- posing around hits


class _CountingEstimator:
    """A `PoseEstimator` that records what it was asked and answers as told."""

    name = "counting-v1"

    def __init__(self, answer=True):
        self.asked = []
        self._answer = answer

    def estimate(self, image, box):
        self.asked.append((image[0, 0, 0], box))
        return pose(box) if self._answer else None


@pytest.fixture(scope="module")
def base_events(camera, calibration):
    script = rally.fly(BASE)
    people = rally.player_tracks(calibration, script)
    track = rally.ball_track(camera, calibration, script)
    return script, people, events.detect_events(track, people, calibration)


def frames_of(script, every=1):
    """Blank frames at the script's times, numbered in their first pixel."""
    for number, moment in enumerate(script.times[::every]):
        image = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
        image[0, 0, 0] = number % 250
        yield VideoFrame(pts=float(moment), image=image, index=number)


def test_poses_are_estimated_on_a_few_frames_around_each_hit_only(base_events):
    script, people, found = base_events
    estimator = _CountingEstimator()
    config = StrokeConfig(pose_frames=4, min_player_height=0.0)

    poses = strokes.estimate_poses(frames_of(script), found, people, estimator, config)

    # The first hit is the first frame, so it has only what follows contact.
    assert sorted(poses) == [0, 1, 2] and len(poses[0]) <= 2
    assert len(poses[1]) == 4 and len(poses[2]) == 4
    assert len(estimator.asked) == sum(len(samples) for samples in poses.values())
    for index in (1, 2):
        when = found.shots[index].hit.pts
        times = [sample.pts for sample in poses[index]]
        assert times == sorted(times)
        assert min(times) >= when - config.window_before - 1e-6
        assert max(times) <= when + config.window_after + 1e-6
        # Mostly before contact: that is where the side of the swing shows.
        assert sum(moment <= when for moment in times) >= 3
    # Only the hitter's own box is handed over.
    assert estimator.asked[-1][1] == people.at(poses[2][-1].pts, "near").box


def test_a_player_too_small_or_not_tracked_is_not_posed(base_events, calibration):
    script, people, found = base_events
    far_tall = people.far[0].box[3] - people.far[0].box[1]
    near_tall = people.near[0].box[3] - people.near[0].box[1]
    between = (far_tall + near_tall) / 2 / SIZE[1]

    small = strokes.estimate_poses(
        frames_of(script),
        found,
        people,
        _CountingEstimator(),
        StrokeConfig(min_player_height=between),
    )
    untracked = strokes.estimate_poses(
        frames_of(script),
        found,
        rally.player_tracks(calibration, script, missing=("near",)),
        _CountingEstimator(),
        StrokeConfig(min_player_height=0.0),
    )

    assert sorted(small) == [0, 2]
    assert sorted(untracked) == [1]


def test_a_hit_the_estimator_finds_nobody_at_has_no_poses(base_events):
    script, people, found = base_events
    estimator = _CountingEstimator(answer=False)

    poses = strokes.estimate_poses(
        frames_of(script), found, people, estimator, StrokeConfig(min_player_height=0.0)
    )

    assert poses == {} and len(estimator.asked) > 0


def test_frames_far_apart_still_give_each_hit_what_they_can(base_events):
    script, people, found = base_events

    poses = strokes.estimate_poses(
        frames_of(script, every=9),
        found,
        people,
        _CountingEstimator(),
        StrokeConfig(min_player_height=0.0),
    )

    assert all(1 <= len(samples) <= 3 for samples in poses.values())


# ---------------------------------------------------------------- contexts


def test_contexts_carry_quiet_bounces_boxes_and_handedness(base_events):
    script, people, found = base_events
    config = StrokeConfig(handedness=(("A", "left"), ("B", "right")))

    first, second, third = strokes.hit_contexts(found, people, {1: sampled(pose())}, config)

    assert first.rival_quiet_before == float("inf") and first.watched_before == pytest.approx(0.0)
    assert second.rival_quiet_before == pytest.approx(script.hits[1] - script.hits[0], abs=0.03)
    assert third.rival_quiet_before == pytest.approx(script.hits[2] - script.hits[1], abs=0.03)
    # Each shot bounced on the receiver's side before it was hit back.
    assert not first.bounced_before and second.bounced_before and third.bounced_before
    assert (first.handedness, second.handedness) == ("left", "right")
    assert second.box == people.at(found.shots[1].hit.pts, "far").box
    assert len(second.poses) == 1 and first.poses == ()


def test_the_servers_own_earlier_hit_does_not_break_the_quiet(base_events):
    _, people, found = base_events
    twice = replace(found, shots=(found.shots[0], replace(found.shots[2], index=1)))

    _, again = strokes.hit_contexts(twice, people)

    assert again.rival_quiet_before == float("inf")


def test_an_unknown_handedness_is_refused_and_an_untracked_hitter_has_no_box(
    base_events, calibration
):
    script, people, found = base_events

    with pytest.raises(ValueError, match="handedness"):
        strokes.hit_contexts(found, people, config=StrokeConfig(handedness=(("A", "both"),)))
    nobody = rally.player_tracks(calibration, script, missing=("near", "far"))
    assert all(one.box is None for one in strokes.hit_contexts(found, nobody))


def test_no_shots_means_no_contexts(calibration):
    empty = rally.BallTrack((), (), (), np.zeros(0, dtype=int), np.zeros(0), SIZE, "none")
    nothing = events.detect_events(empty, None, calibration)

    assert (
        strokes.classify_strokes(nothing, rally.player_tracks(calibration, rally.fly(BASE))) == ()
    )


# ------------------------------------------------------------------- rules


def test_a_ball_struck_above_the_box_from_behind_the_baseline_is_a_serve():
    above = hit(image_xy=(445.0, 195.0))

    found = RULES.classify(context(above, rival_quiet_before=12.0))

    assert named(found) == "Serve"
    assert found.stroke.provenance is Provenance.MEASURED
    assert found.stroke.confidence == pytest.approx(0.85)
    assert "struck overhead (by the ball)" in found.reason and "12.0s" in found.reason
    assert found.spin is None and not found.pose_used


def test_a_serve_can_be_seen_by_the_pose_alone():
    unseen = hit(image_xy=None, provenance=Provenance.INFERRED, cues=("appearance",))
    reaching = sampled(pose(wrist=(0.05, 0.3)), pose(wrist=(0.05, -0.1)), pose(wrist=(-0.2, 0.4)))

    found = RULES.classify(context(unseen, rival_quiet_before=float("inf"), poses=reaching))

    assert named(found) == "Serve" and found.pose_used
    assert found.stroke.confidence == pytest.approx(0.75)
    assert "by the pose" in found.reason and "had not hit before it" in found.reason


def test_a_serve_nobody_saw_struck_is_inferred_from_where_and_when():
    unseen = hit(image_xy=None, provenance=Provenance.INFERRED)

    found = RULES.classify(context(unseen, rival_quiet_before=9.0))
    barely_watched = RULES.classify(
        context(unseen, rival_quiet_before=float("inf"), watched_before=1.0)
    )

    assert named(found) == "Serve" and found.stroke.provenance is Provenance.INFERRED
    assert found.stroke.confidence == pytest.approx(0.5)
    assert "height of contact was not seen" in found.reason
    assert barely_watched.stroke.confidence == pytest.approx(0.45)
    assert "too little was watched" in barely_watched.reason


def test_a_first_shot_not_seen_struck_overhead_is_still_a_serve_but_less_sure():
    waist_high = hit(image_xy=(520.0, 330.0))

    found = RULES.classify(context(waist_high, rival_quiet_before=9.0))

    assert named(found) == "Serve" and found.stroke.provenance is Provenance.INFERRED
    assert found.stroke.confidence == pytest.approx(0.45)
    assert "did not show an overhead contact" in found.reason


def test_the_far_player_serves_from_behind_the_far_baseline():
    far_serve = hit("far", position=(-1.0, 24.4), image_xy=(470.0, 148.0))
    inside = hit("far", position=(-1.0, 22.0), image_xy=(470.0, 148.0))

    assert named(RULES.classify(context(far_serve, FAR_BOX, rival_quiet_before=9.0))) == "Serve"
    assert named(RULES.classify(context(inside, FAR_BOX, rival_quiet_before=9.0))) == "Overhead"


def test_overhead_contact_in_the_middle_of_a_rally_is_an_overhead():
    above = hit(image_xy=(445.0, 195.0))
    reaching = sampled(pose(wrist=(0.05, -0.1)))
    unseen = hit(image_xy=None, provenance=Provenance.INFERRED)

    by_ball = RULES.classify(context(above))
    by_pose = RULES.classify(context(unseen, poses=reaching))

    assert named(by_ball) == "Overhead" and by_ball.stroke.confidence == pytest.approx(0.7)
    assert named(by_pose) == "Overhead" and by_pose.stroke.confidence == pytest.approx(0.55)
    assert "middle of a rally" in by_ball.reason


@pytest.mark.parametrize(
    ("side", "box", "ball_x", "hand", "expected"),
    [
        # The near player has their back to the camera: their right is the picture's right.
        ("near", NEAR_BOX, 500.0, "right", "Forehand"),
        ("near", NEAR_BOX, 380.0, "right", "Backhand"),
        ("near", NEAR_BOX, 500.0, "left", "Backhand"),
        ("near", NEAR_BOX, 380.0, "left", "Forehand"),
        # The far player faces it: their right is the picture's left.
        ("far", FAR_BOX, 455.0, "right", "Forehand"),
        ("far", FAR_BOX, 485.0, "right", "Backhand"),
        ("far", FAR_BOX, 455.0, "left", "Backhand"),
    ],
)
def test_forehand_or_backhand_is_the_side_of_contact_and_the_playing_hand(
    side, box, ball_x, hand, expected
):
    position = (1.0, -0.5) if side == "near" else (-1.0, 24.3)
    mid_box = (box[1] + box[3]) / 2

    found = RULES.classify(
        context(hit(side, position, image_xy=(ball_x, mid_box)), box, handedness=hand)
    )

    assert named(found) == expected
    assert found.stroke.confidence == pytest.approx(0.65)
    assert f"taken as {hand}-handed" in found.reason and "by the ball" in found.reason


def test_ball_and_pose_agreeing_is_the_most_confident_reading():
    swing = sampled(pose(wrist=(0.3, 0.5)))

    found = RULES.classify(context(hit(image_xy=(500.0, 330.0)), poses=swing))

    assert named(found) == "Forehand" and found.pose_used
    assert found.stroke.confidence == pytest.approx(0.8)
    assert "ball and pose" in found.reason


def test_when_they_disagree_the_ball_decides_at_lower_confidence():
    other_side = sampled(pose(wrist=(-0.3, 0.5)))

    found = RULES.classify(context(hit(image_xy=(500.0, 330.0)), poses=other_side))

    assert named(found) == "Forehand"
    assert found.stroke.confidence == pytest.approx(0.5)
    assert "the pose disagreed" in found.reason


def test_the_pose_alone_names_a_stroke_whose_contact_was_not_tracked():
    unseen = hit(image_xy=(900.0, 100.0), provenance=Provenance.INFERRED, cues=("appearance",))

    backhand = RULES.classify(context(unseen, poses=sampled(pose(wrist=(-0.3, 0.5)))))
    left_handed = RULES.classify(
        context(unseen, poses=sampled(pose(wrist=(-0.3, 0.5), hand="left")), handedness="left")
    )

    assert named(backhand) == "Backhand" and backhand.stroke.confidence == pytest.approx(0.55)
    assert "pose alone" in backhand.reason
    assert named(left_handed) == "Forehand"


def test_the_side_is_read_before_contact_not_from_the_follow_through():
    unseen = hit(image_xy=None, provenance=Provenance.INFERRED)
    swing = (
        PoseSample(9.75, pose(wrist=(-0.35, 0.5))),
        PoseSample(9.9, pose(wrist=(-0.25, 0.5))),
        # After contact a backhand's racket hand finishes on the other side.
        PoseSample(10.08, pose(wrist=(0.4, 0.3))),
    )

    assert named(RULES.classify(context(unseen, poses=swing))) == "Backhand"
    only_after = (PoseSample(10.08, pose(wrist=(0.4, 0.3))),)
    assert named(RULES.classify(context(unseen, poses=only_after))) == "unknown"


def test_unsure_keypoints_are_ignored_and_the_stroke_left_unknown():
    unseen = hit(image_xy=None, provenance=Provenance.INFERRED)
    no_wrist = sampled(pose(wrist=(0.3, 0.5), wrist_sure=0.1))
    no_shoulders = sampled(pose(wrist=(0.3, 0.5), shoulders=0.1))

    for poses in (no_wrist, no_shoulders, ()):
        found = RULES.classify(context(unseen, poses=poses))
        assert named(found) == "unknown" and found.stroke.confidence is None
        assert "side of contact was not seen" in found.reason and not found.pose_used


def test_a_pose_without_hips_or_a_nose_still_reads_overhead():
    unseen = hit(image_xy=None, provenance=Provenance.INFERRED)
    reaching = sampled(pose(wrist=(0.05, -0.15), hips=0.1, nose=0.1))
    level = sampled(pose(wrist=(0.3, 0.45), hips=0.1, nose=0.1))

    assert named(RULES.classify(context(unseen, poses=reaching))) == "Overhead"
    assert named(RULES.classify(context(unseen, poses=level))) == "Forehand"


def test_a_wrist_or_ball_in_line_with_the_body_decides_nothing():
    central = hit(image_xy=(445.0, 330.0))

    assert named(RULES.classify(context(central))) == "unknown"
    assert (
        named(RULES.classify(context(central, poses=sampled(pose(wrist=(0.02, 0.5)))))) == "unknown"
    )


def test_no_box_means_neither_the_ball_nor_the_pose_can_be_read():
    found = RULES.classify(context(box=None, poses=sampled(pose())))

    assert named(found) == "unknown" and not found.pose_used


def test_a_hit_at_the_net_with_no_bounce_seen_is_a_volley_but_never_a_sure_one():
    at_net = hit(position=(1.0, court.NET_Y_M - 3.0), image_xy=(500.0, 330.0))

    volley = RULES.classify(context(at_net, bounced_before=False))
    after_bounce = RULES.classify(context(at_net, bounced_before=True))
    backhand = RULES.classify(
        context(replace(at_net, image_xy=(380.0, 330.0)), bounced_before=False)
    )
    from_the_back = RULES.classify(context(bounced_before=False))

    assert named(volley) == "Forehand Volley" and named(backhand) == "Backhand Volley"
    assert volley.stroke.confidence == pytest.approx(0.65 * 0.85)
    assert "no bounce seen" in volley.reason
    assert named(after_bounce) == "Forehand" and named(from_the_back) == "Forehand"


def test_a_volley_whose_side_was_not_seen_is_unknown():
    at_net = hit(position=(1.0, court.NET_Y_M - 3.0), image_xy=(445.0, 330.0))

    found = RULES.classify(context(at_net, bounced_before=False))

    assert named(found) == "unknown" and "at the net" in found.reason


def test_an_untracked_hitter_gets_no_stroke():
    found = RULES.classify(context(hit(position=None)))

    assert named(found) == "unknown" and "not tracked" in found.reason
    assert found.stroke.provenance is Provenance.INFERRED


def test_a_stroke_scored_below_the_floor_is_reported_unknown_with_what_it_would_have_been():
    strict = strokes.RuleStrokeClassifier(StrokeConfig(min_confidence=0.7))

    found = strict.classify(context(hit(image_xy=(500.0, 330.0))))

    assert named(found) == "unknown" and found.stroke.confidence is None
    assert "too unsure (0.65) to name it Forehand" in found.reason


def test_a_label_serialises_and_never_claims_a_spin():
    stored = json.loads(json.dumps(RULES.classify(context()).to_dict()))

    assert stored["stroke"]["value"] == "Forehand" and stored["spin"] is None
    assert stored["stroke"]["source"] == "stroke-rules-v1" and stored["pose_used"] is False
    with_spin = StrokeLabel(
        0, Tagged("Forehand", Provenance.MEASURED), Tagged("Flat", Provenance.USER_ENTERED), ""
    )
    assert with_spin.to_dict()["spin"]["value"] == "Flat"


# ------------------------------------------------- whole rally, and seams


def test_a_scripted_rally_is_named_from_the_ball_alone(base_events):
    _, people, found = base_events

    labels = strokes.classify_strokes(found, people)

    # Near serves (inferred: the contact itself was not tracked), then two forehands.
    assert [named(one) for one in labels] == ["Serve", "Forehand", "Forehand"]
    assert labels[0].stroke.provenance is Provenance.INFERRED
    assert [one.index for one in labels] == [0, 1, 2]
    assert isinstance(RULES, strokes.StrokeClassifier)


def test_a_left_handed_player_swaps_forehand_and_backhand(base_events):
    _, people, found = base_events
    config = StrokeConfig(handedness=(("A", "right"), ("B", "left")))

    labels = strokes.classify_strokes(found, people, config=config)

    assert [named(one) for one in labels] == ["Serve", "Backhand", "Forehand"]


def test_another_classifier_can_take_the_rules_place(base_events):
    _, people, found = base_events

    class Always:
        name = "learned-v0"

        def classify(self, one):
            return StrokeLabel(
                one.index, Tagged("Backhand", Provenance.MEASURED, 0.9, self.name), None, ""
            )

    labels = strokes.classify_strokes(found, people, classifier=Always())

    assert [named(one) for one in labels] == ["Backhand"] * 3
    assert labels[0].stroke.source == "learned-v0"


def test_naming_the_serve_rejudges_it_and_strokes_refine_directions(base_events):
    _, people, found = base_events
    labels = strokes.classify_strokes(found, people)

    applied = strokes.apply_strokes(found, labels)

    # As a rally shot the first ball is judged against the sideline; as a serve, its box.
    assert found.shots[0].call.line == "sideline" and not found.shots[0].serve
    assert applied.shots[0].serve and applied.shots[0].call.line == "service line"
    assert applied.shots[1].result.value == "In" and not applied.shots[1].serve
    # A forehand from the deuce half is an ordinary cross-court shot...
    assert applied.shots[1].direction.value == "cross court"
    backhands = [replace(one, stroke=replace(one.stroke, value="Backhand")) for one in labels[1:]]
    renamed = strokes.apply_strokes(found, backhands)
    # ...the same shot hit as a backhand from there is an inside-out one.
    assert renamed.shots[1].direction.value == "inside out"
    assert renamed.shots[2].direction.value == "inside in"
    assert not renamed.shots[0].serve


@pytest.mark.parametrize(
    ("direction", "stroke", "zone", "hand", "expected"),
    [
        ("cross court", "Forehand", "deuce", "right", "cross court"),
        ("cross court", "Forehand", "ad", "right", "inside out"),
        ("down the line", "Forehand", "ad", "right", "inside in"),
        ("cross court", "Backhand", "deuce", "right", "inside out"),
        ("cross court", "Backhand", "ad", "right", "cross court"),
        ("cross court", "Forehand", "ad", "left", "cross court"),
        ("down the line", "Backhand Volley", "ad", "left", "inside in"),
        ("cross court", "unknown", "ad", "right", "cross court"),
        ("cross court", "Forehand", None, "right", "cross court"),
        ("down the t", "Forehand", "ad", "right", "down the t"),
    ],
)
def test_inside_shots_are_renamed(direction, stroke, zone, hand, expected):
    assert strokes.refine_direction(direction, stroke, zone, hand) == expected


def test_stroke_quality_counts_names_unknowns_and_pose_use(base_events):
    _, people, found = base_events
    labels = list(strokes.classify_strokes(found, people))
    labels[1] = replace(
        labels[1], stroke=replace(labels[1].stroke, value="unknown"), pose_used=True
    )

    quality = strokes.stroke_quality(found, labels)

    assert quality["strokes"]["Serve"] == 1 and quality["strokes"]["unknown"] == 1
    assert quality["strokes"]["Overhead"] == 0
    assert quality["by_side"]["far"] == {"shots": 1, "unknown": 1, "pose_used": 1}
    assert quality["by_side"]["near"] == {"shots": 2, "unknown": 0, "pose_used": 0}
    assert json.dumps(quality)
