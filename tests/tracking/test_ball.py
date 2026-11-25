"""Tests for ball candidates and track linking, on rendered and hand-made data."""

from __future__ import annotations

import json
import logging

import cv2
import numpy as np
import pytest

from tracking import ball, court, models
from tracking.config import BallConfig
from tracking.players import PlayerObservation, PlayerTracks
from tracking.provenance import Provenance
from tracking.video import VideoFrame

from .conftest import SIZE, render_court

FPS = 60.0
FRAMES = 84
BALL_RGB = (215, 235, 60)


def flight(count=FRAMES, fps=FPS):
    """A shot from the near baseline that bounces in the far service box.

    Returns:
        (count, 3) court positions in metres, gravity and one bounce
        included.
    """
    spot = np.array([1.5, 1.0, 1.0])
    speed = np.array([-2.0, 17.0, 3.0])
    step = 1.0 / fps
    out = []
    for _ in range(count):
        out.append(spot.copy())
        speed[2] -= 9.81 * step
        spot = spot + speed * step
        if spot[2] < 0.033:
            spot[2] = 0.033 + (0.033 - spot[2])
            speed[2] = -0.75 * speed[2]
    return np.array(out)


def to_pixels(camera, points):
    """Projects (N, 3) court points through a 3x4 camera."""
    lifted = np.hstack([points, np.ones((len(points), 1))])
    mapped = lifted @ camera.T
    return mapped[:, :2] / mapped[:, 2:]


def ball_radius(camera, points):
    """The ball's radius in pixels at each of (N, 3) court points (at least 1.5)."""
    depth = (np.hstack([points, np.ones((len(points), 1))]) @ camera.T)[:, 2]
    scale = np.linalg.norm(camera[2, :3])
    focal = 0.68 * SIZE[0]
    return np.maximum(1.5, 0.0335 * focal / (depth / scale))


_NOISE: list[np.ndarray] = []


def _noise(number):
    """One of a few fixed noise fields, so rendering a frame stays cheap."""
    if not _NOISE:
        rng = np.random.default_rng(11)
        _NOISE.extend(
            rng.normal(0, 3.0, (SIZE[1], SIZE[0], 3)).astype(np.int16) for _ in range(5)
        )
    return np.roll(_NOISE[number % 5], 37 * number, axis=1)


def rally_frames(camera, background, hidden=(), dropped=(), distract=True, seed=3, count=FRAMES):
    """Renders the flight over a court, one frame at a time.

    Args:
        camera: 3x4 projection.
        background: The court image without noise.
        hidden: Frame numbers in which the ball is not drawn (occluded).
        dropped: Frame numbers that are left out of the stream entirely.
        distract: Draw a walking player-sized block and a jittering
            yellow speck.
        seed: Seed for the distractor's jitter.
        count: How many frames of the flight to render.

    Yields:
        `VideoFrame`s with real timestamps and their frame numbers.
    """
    rng = np.random.default_rng(seed)
    path = to_pixels(camera, flight())
    radius = ball_radius(camera, flight())
    for number in range(count):
        if number in dropped:
            continue
        image = background.copy()
        if distract:
            left = 100 + 3 * number
            cv2.rectangle(image, (left, 330), (left + 40, 450), (190, 140, 110), -1)
            speck = (700 + int(rng.integers(-25, 25)), 420 + int(rng.integers(-25, 25)))
            cv2.circle(image, speck, 3, BALL_RGB, -1, cv2.LINE_AA)
        if number not in hidden:
            here = path[number]
            before = path[max(number - 1, 0)]
            middle = (here + before) / 2
            thick = max(2, round(2 * radius[number]))
            cv2.line(
                image,
                tuple(np.round(middle * 16).astype(int)),
                tuple(np.round(here * 16).astype(int)),
                BALL_RGB,
                thick,
                cv2.LINE_AA,
                shift=4,
            )
        noisy = np.clip(image.astype(np.int16) + _noise(number), 0, 255).astype(np.uint8)
        yield VideoFrame(pts=number / FPS, image=noisy, index=number)


@pytest.fixture(scope="module")
def background(camera):
    """The court with no noise; noise is added per frame."""
    return render_court(camera, noise=0)


@pytest.fixture(scope="module")
def truth(camera):
    """Where the ball really is in each frame, pixels."""
    return to_pixels(camera, flight())


@pytest.fixture(scope="module")
def clean_track(camera, calibration, background):
    """The flight tracked with nothing hidden and two frames dropped."""
    return ball.track_ball(rally_frames(camera, background, dropped=(30, 31)), calibration)


def _errors(track, truth):
    """Distance from each observation to the true ball position, by frame."""
    return {
        o.frame: float(np.hypot(*np.subtract(o.image_xy, truth[o.frame])))
        for o in track.observations
    }


def line_frames(count=30, start=(100.0, 200.0), step=(8.0, 2.0), extra=None, fps=60.0):
    """Hand-made candidates: one thing moving in a straight line.

    Args:
        count: Frames.
        start: Position in frame 0.
        step: Movement per frame.
        extra: {frame: [BallCandidate, ...]} added to those frames.
        fps: Frame rate of the timestamps.

    Returns:
        A list of `FrameCandidates`.
    """
    extra = extra or {}
    out = []
    for number in range(count):
        spot = (start[0] + step[0] * number, start[1] + step[1] * number)
        found = [ball.BallCandidate(spot, 0.8, 4.0), *extra.get(number, [])]
        out.append(ball.FrameCandidates(number, number / fps, tuple(found)))
    return out


# ---------------------------------------------------------------- end to end


def test_the_rendered_ball_is_recovered_despite_noise_and_distractors(clean_track, truth):
    errors = _errors(clean_track, truth)
    drawn = [n for n in range(4, FRAMES - 4) if n not in (30, 31)]
    close = [n for n in drawn if n in errors and errors[n] < 3.0]

    assert len(close) / len(drawn) > 0.85
    # Whatever is tracked is the ball: nothing sits on the distractors.
    assert max(errors.values()) < 6.0
    assert np.median(list(errors.values())) < 1.5


def test_dropped_frames_are_absent_not_invented(clean_track):
    frames = {o.frame for o in clean_track.observations}

    assert not frames & {30, 31}
    assert 30 not in clean_track.frame_numbers
    # The path carries straight on across the missing frames.
    assert {29, 32} <= frames
    assert clean_track.at(29 / FPS).segment == clean_track.at(32 / FPS).segment


def test_every_observation_is_tagged_and_scored(clean_track):
    for observation in clean_track.observations:
        assert observation.provenance in (Provenance.MEASURED, Provenance.INTERPOLATED)
        assert 0.0 <= observation.confidence <= 1.0
        measured = observation.provenance is Provenance.MEASURED
        assert (observation.visibility is ball.Visibility.VISIBLE) == measured
        assert (observation.extent is not None) == measured
    assert clean_track.source == "motion-ball-v1"
    assert clean_track.image_size == SIZE


SHORT_HIDE = (20, 21, 22)
LONG_HIDE = tuple(range(36, 56))


@pytest.fixture(scope="module")
def occluded_track(camera, calibration, background):
    """The flight with the ball hidden for three frames, then for twenty."""
    frames = rally_frames(camera, background, hidden=SHORT_HIDE + LONG_HIDE)
    return ball.track_ball(frames, calibration)


def test_a_short_occlusion_is_bridged_and_tagged_interpolated(occluded_track, truth):
    track = occluded_track

    bridged = {o.frame: o for o in track.observations if o.provenance is Provenance.INTERPOLATED}
    assert set(SHORT_HIDE) <= set(bridged)
    neighbour = track.at(19 / FPS)
    for number in SHORT_HIDE:
        observation = bridged[number]
        assert observation.visibility is ball.Visibility.BRIDGED
        assert np.hypot(*np.subtract(observation.image_xy, truth[number])) < 6.0
        assert observation.confidence < neighbour.confidence
        assert observation.segment == neighbour.segment
    assert track.quality().bridged >= 3


def test_a_long_occlusion_is_left_unknown_not_filled(occluded_track):
    track = occluded_track

    frames = {o.frame for o in track.observations}
    assert not frames & set(range(38, 54))
    gap = next(g for g in track.gaps if g.start_frame <= 45 <= g.end_frame)
    assert gap.frames >= 16
    assert gap.seconds == pytest.approx((gap.frames - 1) / FPS, abs=1e-6)
    before, after = track.at(33 / FPS), track.at(58 / FPS)
    assert before is not None and after is not None
    assert before.segment != after.segment
    assert track.quality().gap_histogram["0.1-0.5s"] >= 1


def test_court_position_is_right_for_a_ball_on_the_ground(clean_track):
    path = flight()
    lowest = min(range(35, 70), key=lambda n: path[n][2])

    observation = clean_track.at(lowest / FPS)

    assert observation is not None
    assert np.hypot(*np.subtract(observation.ground_ray_xy, path[lowest][:2])) < 1.0
    # In the air the same projection lands well beyond the ball: it is the
    # sightline's ground point, not the ball's position.
    airborne = clean_track.at(20 / FPS)
    assert airborne.ground_ray_xy[1] - path[20][1] > 2.0
    # ...which is why no observation claims a court position: only a later
    # phase that knows the ball is on the ground may fill it in.
    assert all(o.court_xy is None for o in clean_track.observations)
    assert observation.to_dict()["court_xy"] is None
    assert observation.to_dict()["ground_ray_xy"] == list(observation.ground_ray_xy)


def test_a_sightline_that_never_meets_the_court_has_no_court_position(calibration):
    above_horizon = ball.link_candidates(line_frames(start=(100.0, 40.0)), SIZE, calibration)
    grazing = ball.link_candidates(
        line_frames(start=(100.0, 139.0), step=(8.0, 0.0)), SIZE, calibration
    )
    on_court = ball.link_candidates(line_frames(start=(300.0, 300.0)), SIZE, calibration)

    assert len(above_horizon.observations) == 30
    assert all(o.ground_ray_xy is None for o in above_horizon.observations)
    assert all(o.ground_ray_xy is None for o in grazing.observations)
    assert all(o.ground_ray_xy is not None for o in on_court.observations)


def test_a_calibration_that_maps_a_pixel_nowhere_gives_no_court_position():
    class Nowhere:
        def to_court(self, points):
            return np.array([np.nan, np.nan])

    assert ball._on_ground(Nowhere(), np.array([10.0, 10.0])) is None
    assert ball._on_ground(None, np.array([10.0, 10.0])) is None


# -------------------------------------------------------------------- linking


def test_one_straight_path_becomes_one_segment(calibration):
    track = ball.link_candidates(line_frames(), SIZE, source="hand")

    assert len(track.segments) == 1
    segment = track.segments[0]
    assert (segment.start_frame, segment.end_frame) == (0, 29)
    assert segment.measured == 30 and segment.bridged == 0
    assert segment.roughness < 0.05
    assert track.gaps == ()
    assert track.observations[5].ground_ray_xy is None
    assert track.observations[5].velocity == pytest.approx((480.0, 120.0), rel=0.05)
    assert track.source == "hand"


def test_an_impossible_jump_is_never_linked():
    # In frame 12 the only detection is 400 pixels off the path.
    frames = line_frames()
    frames[12] = ball.FrameCandidates(12, 12 / 60, (ball.BallCandidate((700.0, 60.0), 1.0, 4.0),))

    track = ball.link_candidates(frames, SIZE)

    at_jump = track.at(12 / 60)
    assert at_jump.provenance is Provenance.INTERPOLATED
    assert np.hypot(at_jump.image_xy[0] - 196.0, at_jump.image_xy[1] - 224.0) < 3.0
    assert all(abs(o.image_xy[1] - (200 + 2 * o.frame)) < 3.0 for o in track.observations)


def test_a_teleport_splits_the_track_in_two():
    first = line_frames(15)
    second = line_frames(15, start=(700.0, 400.0))
    frames = first + [
        ball.FrameCandidates(15 + f.frame, (15 + f.frame) / 60, f.candidates) for f in second
    ]

    track = ball.link_candidates(frames, SIZE)

    assert len(track.segments) == 2
    assert track.at(14 / 60).segment != track.at(15 / 60).segment


def test_a_speed_over_the_physical_limit_is_refused(calibration):
    # 400 pixels a frame at the far end of the court is hundreds of m/s.
    fast = line_frames(12, start=(60.0, 150.0), step=(70.0, 0.0))

    slow_limit = ball.link_candidates(fast, SIZE, calibration, config=BallConfig(max_speed_mps=5.0))
    generous = ball.link_candidates(fast, SIZE, calibration)

    assert slow_limit.observations == ()
    assert len(generous.segments) == 1


def test_the_coherent_path_wins_over_scattered_stronger_candidates():
    rng = np.random.default_rng(0)
    extra = {
        number: [
            ball.BallCandidate((float(rng.uniform(0, 960)), float(rng.uniform(0, 540))), 1.0, 4.0)
            for _ in range(6)
        ]
        for number in range(30)
    }

    track = ball.link_candidates(line_frames(extra=extra), SIZE)

    on_path = [
        o for o in track.observations if abs(o.image_xy[1] - (200 + 2 * o.frame)) < 1e-6
    ]
    assert len(on_path) >= 28
    assert len(on_path) / len(track.observations) > 0.95


def test_of_two_overlapping_paths_the_one_with_more_evidence_is_kept():
    weak = [ball.BallCandidate((500.0 - 6 * n, 300.0), 0.2, 4.0) for n in range(30)]
    frames = line_frames(extra={n: [weak[n]] for n in range(30)})

    track = ball.link_candidates(frames, SIZE)

    assert len(track.segments) == 1
    assert all(o.image_xy[1] != 300.0 for o in track.observations)


def test_too_little_too_still_or_too_weak_is_not_a_segment():
    short = ball.link_candidates(line_frames(4), SIZE)
    still = ball.link_candidates(line_frames(30, step=(0.3, 0.0)), SIZE)
    faint = [
        ball.FrameCandidates(f.frame, f.pts, (ball.BallCandidate(f.candidates[0].xy, 0.05, 4.0),))
        for f in line_frames(12)
    ]

    assert short.observations == ()
    assert still.observations == ()
    assert ball.link_candidates(faint, SIZE).observations == ()


def test_a_jittering_speck_is_not_a_flight():
    rng = np.random.default_rng(4)
    frames = [
        ball.FrameCandidates(
            n,
            n / 60,
            (
                ball.BallCandidate(
                    (400.0 + float(rng.uniform(-12, 12)), 300.0 + float(rng.uniform(-12, 12))),
                    0.9,
                    4.0,
                ),
            ),
        )
        for n in range(60)
    ]

    track = ball.link_candidates(
        frames, SIZE, config=BallConfig(gate=0.02, max_roughness=0.2)
    )
    lenient = ball.link_candidates(
        frames, SIZE, config=BallConfig(gate=0.02, max_roughness=5.0)
    )

    assert track.observations == ()
    assert lenient.observations != ()


def test_a_path_that_barely_travels_is_scenery():
    # Moves fast enough to seed, but only back and forth within a few pixels.
    frames = [
        ball.FrameCandidates(
            n, n / 60, (ball.BallCandidate((300.0 + 2.0 * (n % 2), 200.0 + 2.0 * (n % 2)), 0.9, 4),)
        )
        for n in range(30)
    ]

    assert ball.link_candidates(frames, SIZE, config=BallConfig(gate=0.02)).observations == ()


def _still_players(box):
    """A `PlayerTracks` with one near player standing in `box` throughout."""
    seen = tuple(
        PlayerObservation(
            frame=n,
            pts=n / 60,
            side="near",
            player="A",
            box=box,
            foot_image=((box[0] + box[2]) / 2, box[3]),
            foot_court=(0.0, 1.0),
            zone="deuce",
            confidence=0.9,
            provenance=Provenance.MEASURED,
            track_id=1,
        )
        for n in range(0, 31, 6)
    )
    return PlayerTracks(
        near=seen,
        far=(),
        swaps=(),
        detector_times=tuple(o.pts for o in seen),
        source="fake",
        max_interpolation_seconds=0.6,
    )


def test_candidates_inside_a_player_box_are_discounted_or_dropped():
    # The only path runs through the player's box.
    players = _still_players((50.0, 150.0, 400.0, 300.0))
    frames = line_frames()

    kept = ball.link_candidates(frames, SIZE, players=players)
    plain = ball.link_candidates(frames, SIZE)
    dropped = ball.link_candidates(
        frames, SIZE, players=players, config=BallConfig(player_box_weight=0.0)
    )

    inside = [o for o in kept.observations if o.image_xy[0] < 400]
    assert inside and all(
        o.confidence < plain.at(o.pts).confidence for o in inside if o.extent is not None
    )
    assert all(o.image_xy[0] > 400 for o in dropped.observations)


def test_observations_and_segments_say_when_the_ball_was_inside_a_player_box():
    players = _still_players((50.0, 150.0, 200.0, 300.0))

    track = ball.link_candidates(line_frames(), SIZE, players=players)
    plain = ball.link_candidates(line_frames(), SIZE)

    flagged = [o for o in track.observations if o.in_player_box]
    assert flagged and all(o.image_xy[0] < 230 for o in flagged)
    assert any(not o.in_player_box for o in track.observations)
    assert 0.0 < track.segments[0].in_player_box < 1.0
    assert track.segments[0].to_dict()["in_player_box"] == track.segments[0].in_player_box
    assert flagged[0].to_dict()["in_player_box"] is True
    # Without player tracks nothing can be said, so nothing is flagged.
    assert not any(o.in_player_box for o in plain.observations)
    assert plain.segments[0].in_player_box == 0.0


def test_a_bridged_observation_is_never_flagged_as_inside_a_player_box():
    players = _still_players((0.0, 0.0, float(SIZE[0]), float(SIZE[1])))
    frames = line_frames()
    frames[10] = ball.FrameCandidates(frames[10].frame, frames[10].pts, ())

    track = ball.link_candidates(frames, SIZE, players=players)

    bridged = [o for o in track.observations if o.provenance is Provenance.INTERPOLATED]
    assert bridged and not any(o.in_player_box for o in bridged)
    assert track.segments[0].in_player_box == 1.0


def test_the_comparison_spacing_follows_the_frame_rate_when_it_is_known():
    config = BallConfig(difference_seconds=0.067, difference_frames=3)

    assert ball.MotionBallDetector(config=config).span == 3
    assert ball.MotionBallDetector(config=config, frame_rate=60.0).span == 4
    assert ball.MotionBallDetector(config=config, frame_rate=30.0).span == 2
    assert ball.MotionBallDetector(config=config, frame_rate=5.0).span == 1
    assert ball.MotionBallDetector(config=config, frame_rate=0.0).span == 3


def test_track_ball_hands_the_frame_rate_to_its_default_detector(monkeypatch):
    seen = {}

    class Spy(ball.MotionBallDetector):
        def __init__(self, calibration=None, config=None, frame_rate=None):
            super().__init__(calibration, config, frame_rate)
            seen["span"] = self.span

    monkeypatch.setattr(ball, "MotionBallDetector", Spy)
    frames = [
        VideoFrame(pts=n / 60, image=np.zeros((40, 60, 3), dtype=np.uint8), index=n)
        for n in range(12)
    ]

    track = ball.track_ball(frames, frame_rate=60.0)

    assert seen["span"] == 4
    assert len(track.frame_pts) == 12 and track.observations == ()


def test_no_frames_and_no_candidates_give_an_empty_honest_track():
    nothing = ball.link_candidates([], SIZE)
    blank = ball.link_candidates(
        [ball.FrameCandidates(n, n / 60, ()) for n in range(10)], SIZE
    )

    assert nothing.observations == () and nothing.gaps == ()
    assert nothing.quality().frames == 0 and nothing.quality().coverage == 0.0
    assert nothing.quality().weak
    assert len(blank.gaps) == 1 and blank.gaps[0].frames == 10
    assert blank.quality().unknown == 10
    assert blank.quality().longest_gap_seconds == pytest.approx(10 / 60, abs=1e-6)


def test_a_single_frame_is_handled():
    track = ball.link_candidates(line_frames(1), SIZE)

    assert track.observations == ()
    assert track.gaps[0].frames == 1 and track.gaps[0].seconds == 0.0


# -------------------------------------------------------------- track queries


def test_the_track_answers_at_between_and_segment(clean_track):
    first = clean_track.observations[0]

    assert clean_track.at(first.pts) is first
    assert clean_track.at(first.pts + 0.004) is None
    assert clean_track.at(999.0) is None
    stretch = clean_track.between(0.2, 0.4)
    assert stretch and all(0.2 <= o.pts <= 0.4 for o in stretch)
    assert set(clean_track.segment(first.segment)) <= set(clean_track.observations)


def test_quality_reports_coverage_gaps_and_confidence(clean_track):
    quality = clean_track.quality()

    assert quality.frames == FRAMES - 2
    assert quality.measured + quality.bridged + quality.unknown == quality.frames
    assert quality.coverage == pytest.approx((quality.measured + quality.bridged) / quality.frames)
    assert set(quality.gap_histogram) == {"under 0.1s", "0.1-0.5s", "0.5-2s", "2-10s", "over 10s"}
    assert quality.coverage > 0.8 and not quality.weak
    assert quality.notice() is None
    assert quality.coherence > 0.9
    part = clean_track.quality(0.5, 0.8)
    assert part.frames < quality.frames


def test_weak_tracking_says_so_in_words():
    frames = line_frames(12) + [
        ball.FrameCandidates(n, n / 60, ()) for n in range(12, 2000)
    ]

    quality = ball.link_candidates(frames, SIZE).quality()

    assert quality.weak
    assert "weak" in quality.notice() and "1%" in quality.notice()
    assert quality.gap_histogram["over 10s"] == 1
    assert sum(quality.gap_histogram.values()) == 1


def test_gaps_of_every_length_land_in_their_bucket():
    def run(start, seconds):
        base = line_frames(12, start=(100.0 + start, 200.0))
        return [
            ball.FrameCandidates(int(start * 60) + f.frame, start + f.pts, f.candidates)
            for f in base
        ], start + 12 / 60 + seconds

    frames, cursor = [], 0.0
    for seconds in (0.3, 1.0, 5.0, 12.0):
        burst, following = run(cursor, seconds)
        frames += burst
        count = round((following - burst[-1].pts) * 60) - 1
        frames += [
            ball.FrameCandidates(burst[-1].frame + 1 + k, burst[-1].pts + (k + 1) / 60, ())
            for k in range(count)
        ]
        cursor = following
    frames += run(cursor, 0)[0]

    histogram = ball.link_candidates(frames, SIZE).quality().gap_histogram

    assert histogram == {
        "under 0.1s": 0, "0.1-0.5s": 1, "0.5-2s": 1, "2-10s": 1, "over 10s": 1,
    }


def test_the_track_serialises_to_json(clean_track):
    payload = json.loads(json.dumps(clean_track.to_dict()))

    assert payload["source"] == "motion-ball-v1"
    assert payload["frames"] == FRAMES - 2
    assert payload["observations"][0]["provenance"] in ("measured", "interpolated")
    assert payload["segments"][0]["measured"] > 0
    assert payload["scene"]["frames_sampled"] >= 1
    assert payload["quality"]["coverage"] > 0.8
    assert ball.link_candidates([], SIZE).to_dict()["scene"] is None
    gap = ball.BallGap(1, 3, 0.1, 0.3, 3)
    assert gap.to_dict()["frames"] == 3


def test_coherence_counts_only_plausible_steps():
    def seen(frame, x):
        return ball.BallObservation(
            frame, frame / 30, (x, 100.0), None, (0.0, 0.0), 0.9,
            ball.Visibility.VISIBLE, Provenance.MEASURED, 0, 4.0,
        )

    steady = [seen(n, 50.0 * n) for n in range(5)]
    jumpy = [seen(0, 0.0), seen(1, 900.0), seen(2, 0.0), seen(3, 0.5), seen(20, 40.0)]

    assert ball.trajectory_coherence(steady, 1920) == 1.0
    assert ball.trajectory_coherence(jumpy, 1920) == 0.0
    assert ball.trajectory_coherence(steady[:1], 1920) == 0.0


# ------------------------------------------------------------------- detectors


def test_yellowness_separates_ball_from_court_and_lines():
    pixels = np.array([[BALL_RGB, (40, 80, 175), (245, 245, 245)]], dtype=np.uint8)

    values = ball.yellowness(pixels)[0]

    assert values.dtype == np.int16
    assert values[0] > 100 and values[1] < -80 and abs(values[2]) < 5


def _stamp(background, spot, radius=3):
    """The court with a ball painted at `spot`."""
    image = background.copy()
    cv2.circle(image, spot, radius, BALL_RGB, -1, cv2.LINE_AA)
    return image


def test_the_motion_detector_finds_a_moving_ball_and_not_a_still_one(calibration, background):
    detector = ball.MotionBallDetector(calibration)
    still = (300, 400)
    frames = [_stamp(_stamp(background, still), (500 + 12 * n, 380)) for n in range(5)]
    ready = [detector.prepare(frame) for frame in frames]

    found = detector.detect(ready[0], ready[2], ready[4])

    assert isinstance(detector, ball.BallDetector)
    assert len(found) == 1
    assert np.hypot(found[0].xy[0] - 524, found[0].xy[1] - 380) < 1.5
    assert found[0].score > 0.6
    assert 4 <= found[0].extent <= 9
    assert detector.detect(ready[0], ready[0], ready[0]) == []


def test_without_a_calibration_the_whole_frame_is_searched(background):
    detector = ball.MotionBallDetector()
    frames = [_stamp(background, (500 + 12 * n, 20)) for n in range(3)]
    ready = [detector.prepare(frame) for frame in frames]

    found = detector.detect(*ready)

    assert len(found) == 1 and found[0].xy[1] < 25


def test_a_calibration_for_another_frame_size_is_ignored_with_a_warning(
    calibration, background, caplog
):
    detector = ball.MotionBallDetector(calibration)
    small = cv2.resize(background, (480, 270))

    with caplog.at_level(logging.WARNING, logger="tracking.ball"):
        ready = detector.prepare(small)

    assert ready.shape == (270, 480)
    assert "searching the whole frame" in caplog.text


def test_blobs_far_too_big_or_too_small_for_a_ball_are_not_candidates(calibration, background):
    detector = ball.MotionBallDetector(calibration)
    big = [_stamp(background, (300 + 60 * n, 300), radius=25) for n in range(3)]
    ready = [detector.prepare(frame) for frame in big]

    assert detector.detect(*ready) == []

    # Low in the frame a ball is close and large; a two-pixel speck there
    # is not one.
    detector = ball.MotionBallDetector(calibration)
    speck = [_stamp(background, (300 + 30 * n, 520), radius=1) for n in range(3)]
    ready = [detector.prepare(frame) for frame in speck]

    assert detector.detect(*ready) == []


def test_only_the_best_candidates_are_kept(calibration, background):
    detector = ball.MotionBallDetector(calibration, BallConfig(max_candidates=3))

    def scatter(shift):
        image = background.copy()
        for k in range(8):
            cv2.circle(image, (100 + 100 * k + shift, 300), 3, BALL_RGB, -1, cv2.LINE_AA)
        return image

    ready = [detector.prepare(scatter(shift)) for shift in (0, 14, 28)]

    found = detector.detect(*ready)

    assert len(found) == 3
    assert [c.score for c in found] == sorted((c.score for c in found), reverse=True)


def test_something_that_is_always_moving_is_treated_as_clutter(calibration, background):
    detector = ball.MotionBallDetector(calibration, BallConfig(clutter_limit=3.0))
    on, off = _stamp(background, (400, 300)), background
    ready = [detector.prepare(on), detector.prepare(off)]
    counts = [len(detector.detect(ready[1], ready[0], ready[1])) for _ in range(8)]

    assert counts[0] == 1
    assert counts[-1] == 0


def test_the_flight_volume_knows_which_depths_a_pixel_can_show(camera, calibration):
    volume = ball.FlightVolume(calibration)
    on_court = to_pixels(camera, np.array([[0.0, 12.0, 1.0]]))

    near, far = volume.depth_span(on_court)

    assert volume.ok and volume.focal == pytest.approx(0.68 * SIZE[0], rel=0.01)
    # The camera is 6m behind the baseline and the point 12m down the court.
    assert near[0] < 18.0 < far[0]
    assert ball.flight_region(calibration).all()


def test_a_low_roof_cuts_the_sky_out_of_the_search(calibration):
    config = BallConfig(flight_height_m=2.0)
    volume = ball.FlightVolume(calibration, config)

    near, far = volume.depth_span(np.array([[480.0, 5.0], [480.0, 400.0]]))
    region = ball.flight_region(calibration, config)

    assert np.isnan(near[0]) and np.isnan(far[0])
    assert np.isfinite(near[1])
    assert not region[:20].any() and region[400].all()
    assert volume.size_fit(np.array([[480.0, 5.0]]), np.array([4.0]))[0] == 0.0


def test_a_small_ball_where_only_a_near_one_could_be_is_a_neighbours(camera, calibration):
    volume = ball.FlightVolume(calibration)
    # Far right, low: the sightline is over this court only close to the
    # camera, where a ball would be more than ten pixels across.
    spot = np.array([[940.0, 420.0]])

    _, far = volume.depth_span(spot)
    expected = volume.focal * 0.067 / far[0]

    assert expected > 6
    assert volume.size_fit(spot, np.array([2.0]))[0] == 0.0
    assert volume.size_fit(spot, np.array([expected * 1.5]))[0] == 1.0
    assert volume.size_fit(spot, np.array([400.0]))[0] == 0.0


def test_a_blob_no_bigger_than_the_ball_itself_is_too_small_to_be_it(calibration):
    # What removed the neighbouring court's ball on the real recording: a
    # moving blob is smeared to twice the ball or more, so one only as big
    # as the ball could look at the far edge of this court's space is a
    # ball further away than that - on somebody else's court.
    volume = ball.FlightVolume(calibration)
    spot = np.array([[940.0, 420.0]])
    _, far = volume.depth_span(spot)
    smallest = volume.focal * 0.067 / far[0]

    assert volume.size_fit(spot, np.array([smallest * 0.9]))[0] == 0.0
    assert volume.size_fit(spot, np.array([smallest]))[0] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < volume.size_fit(spot, np.array([smallest * 1.25]))[0] < 1.0
    assert volume.size_fit(spot, np.array([smallest * 2.0]))[0] == 1.0
    lenient = ball.FlightVolume(calibration, BallConfig(min_blob_ratio=1.0))
    assert lenient.size_fit(spot, np.array([smallest]))[0] == 1.0
    assert smallest >= BallConfig().resolved_ball_px
    assert BallConfig().min_blob_ratio == 1.5 and BallConfig().flight_height_m == 10.0


def test_the_size_of_a_blob_says_nothing_about_a_ball_too_small_to_resolve(calibration):
    # A far-baseline ball is a couple of pixels wide; whichever of them
    # cleared the threshold is the blob. Asking it to be 1.5 balls wide
    # threw away real far-court detections on the recording.
    volume = ball.FlightVolume(calibration)
    far_court = np.array(calibration.to_image(np.array([[0.0, 23.0]])), dtype=float)
    _, far = volume.depth_span(far_court)
    smallest = volume.focal * 0.067 / far[0]

    assert smallest < BallConfig().resolved_ball_px / 2
    assert volume.size_fit(far_court, np.array([smallest]))[0] == pytest.approx(1.0)
    assert volume.size_fit(far_court, np.array([smallest * 0.5]))[0] == 0.0
    strict = ball.FlightVolume(calibration, BallConfig(resolved_ball_px=0.001))
    assert strict.size_fit(far_court, np.array([smallest]))[0] == pytest.approx(0.0, abs=1e-9)


def test_without_a_camera_the_volume_allows_everything(calibration, monkeypatch):
    blind = ball.FlightVolume(None)
    near, far = blind.depth_span(np.array([[10.0, 10.0]]))

    assert not blind.ok and blind.focal == 0.0
    assert near[0] == 0.5 and np.isinf(far[0])
    assert blind.size_fit(np.array([[10.0, 10.0]]), np.array([500.0]))[0] == 1.0
    assert blind.region().all()

    monkeypatch.setattr(ball, "camera_from_homography", lambda *_: None)
    assert not ball.FlightVolume(calibration).ok
    assert ball.flight_region(calibration).shape == (SIZE[1], SIZE[0])
    assert ball.flight_region(calibration).all()


def test_the_volume_is_the_same_whichever_way_the_camera_matrix_is_signed(
    calibration, monkeypatch
):
    real = court.camera_from_homography
    spots = np.array([[480.0, 300.0], [200.0, 450.0]])
    usual = ball.FlightVolume(calibration).depth_span(spots)

    monkeypatch.setattr(ball, "camera_from_homography", lambda *args: -real(*args))
    flipped = ball.FlightVolume(calibration).depth_span(spots)

    assert np.allclose(usual, flipped)


def test_row_scale_grows_toward_the_camera(calibration):
    scale = ball.row_scale(calibration)

    assert scale.shape == (SIZE[1],)
    assert scale[-1] > scale[SIZE[1] // 2] > scale[0] > 0
    assert scale[0] == scale[5]


def test_row_scale_survives_a_calibration_that_maps_nowhere():
    class Broken:
        image_size = (64, 48)

        def to_image(self, points):
            return np.full((len(points), 2), np.nan)

    class Flat:
        image_size = (64, 48)

        def to_image(self, points):
            points = np.asarray(points, dtype=float)
            return np.stack([points[:, 0] * 10.0, np.full(len(points), 20.0)], axis=1)

    assert (ball.row_scale(Broken()) == 1.0).all()
    assert (ball.row_scale(Flat()) == 10.0).all()


class _FakeHeatmap:
    """Stands in for a trained model: two peaks, one stronger."""

    def __init__(self):
        self.seen = None

    def predict(self, frames):
        self.seen = frames.shape
        heat = np.zeros(frames.shape[1:3])
        heat[10:13, 20:23] = 0.9
        heat[11, 21] = 0.99
        heat[30:32, 5:7] = 0.6
        heat[40, 40] = 0.3
        return heat


def test_the_learned_detector_slot_turns_a_heatmap_into_candidates():
    model = _FakeHeatmap()
    detector = ball.HeatmapBallDetector(model, name="fake-heatmap@1", span=2)
    image = np.zeros((48, 64, 3), dtype=np.uint8)

    found = detector.detect(detector.prepare(image), image, image)

    assert isinstance(detector, ball.BallDetector) and isinstance(model, ball.HeatmapModel)
    assert model.seen == (3, 48, 64, 3)
    assert [round(c.score, 2) for c in found] == [0.99, 0.6]
    assert found[0].xy == pytest.approx((21.0, 11.0), abs=0.1)
    assert found[0].extent == 3.0
    assert detector.name == "fake-heatmap@1" and detector.span == 2


def test_the_learned_detector_cannot_be_loaded_because_nothing_is_trained():
    with pytest.raises(models.WeightsMissingError):
        models.load_ball_heatmap()


# --------------------------------------------------------------------- streams


class _CountingDetector:
    """Reports which frames it was asked to compare."""

    name = "counting"

    def __init__(self, span):
        self.span = span

    def prepare(self, image):
        return int(image[0, 0, 0])

    def detect(self, before, current, after):
        return [ball.BallCandidate((float(before), float(after)), 0.5, float(current))]


def _numbered(count, indexed=True):
    """Frames whose first pixel holds their own number."""
    for number in range(count):
        image = np.full((2, 2, 3), number, dtype=np.uint8)
        yield VideoFrame(pts=number / 30, image=image, index=number + 100 if indexed else None)


def test_the_stream_compares_each_frame_with_those_span_away():
    found = list(ball.find_candidates(_numbered(7), _CountingDetector(2)))

    assert [f.frame for f in found] == [100, 101, 102, 103, 104, 105, 106]
    assert [f.searched for f in found] == [False, False, True, True, True, False, False]
    middle = found[3]
    assert middle.candidates[0].xy == (1.0, 5.0) and middle.candidates[0].extent == 3.0
    assert found[0].candidates == ()


def test_frames_without_an_index_are_numbered_as_they_come():
    found = list(ball.find_candidates(_numbered(3, indexed=False), _CountingDetector(1)))

    assert [f.frame for f in found] == [0, 1, 2]
    assert [f.searched for f in found] == [False, True, False]


def test_a_stream_shorter_than_the_window_searches_nothing():
    stream = ball.CandidateStream(_CountingDetector(3))
    pushed = [stream.push(frame) for frame in _numbered(2)]

    assert pushed == [[], []]
    assert [f.searched for f in stream.finish()] == [False, False]
    assert stream.finish() == []


def test_scene_statistics_describe_the_court_surface(calibration, background):
    sampler = ball.SceneSampler(calibration, stride=2)
    for _ in range(4):
        sampler.add(background)

    scene = sampler.result()

    assert scene.frames_sampled == 2
    # The court is blue (40, 80, 175) with a few white lines on it.
    assert scene.court_rgb[2] > scene.court_rgb[1] > scene.court_rgb[0]
    assert 60 < scene.brightness < 110 and scene.contrast > 0
    assert scene.court_yellowness < -80
    assert json.loads(json.dumps(scene.to_dict()))["frames_sampled"] == 2


def test_scene_statistics_fall_back_to_the_whole_frame(calibration):
    grey = np.full((40, 60, 3), 100, dtype=np.uint8)
    plain = ball.SceneSampler()
    mismatched = ball.SceneSampler(calibration)
    plain.add(grey)
    mismatched.add(grey)

    assert ball.SceneSampler().result() is None
    assert plain.result().court_rgb == (100.0, 100.0, 100.0)
    assert mismatched.result().brightness == pytest.approx(100.0)


def test_tracking_needs_a_frame_or_a_calibration(calibration):
    with pytest.raises(ValueError, match="at least one frame"):
        ball.track_ball([])

    empty = ball.track_ball([], calibration)
    assert empty.observations == () and empty.image_size == SIZE and empty.scene is None


def test_tracking_without_a_calibration_takes_the_size_from_the_frames(background, camera):
    track = ball.track_ball(rally_frames(camera, background, distract=False, count=24))

    assert track.image_size == SIZE
    assert len(track.observations) > 12
    assert all(o.ground_ray_xy is None for o in track.observations)


def test_an_empty_search_region_falls_back_to_the_whole_frame():
    assert ball._bounds(np.zeros((4, 6), dtype=bool)) == (0, 4, 0, 6)
    mask = np.zeros((4, 6), dtype=bool)
    mask[1:3, 2:5] = True
    assert ball._bounds(mask) == (1, 3, 2, 5)


def test_linking_survives_a_calibration_that_cannot_place_the_net():
    class Lost:
        image_size = SIZE

        def to_image(self, points):
            points = np.asarray(points, dtype=float)
            if points.ndim == 1:
                return np.array([np.nan, np.nan])
            return np.stack([points[:, 0] * 10.0, 300.0 - points[:, 1] * 5.0], axis=1)

        def to_court(self, points):
            return np.array([np.nan, np.nan])

    track = ball.link_candidates(line_frames(), SIZE, Lost())

    assert len(track.segments) == 1
    assert all(o.ground_ray_xy is None for o in track.observations)
