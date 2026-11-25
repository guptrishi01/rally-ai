"""Tests for hit / bounce / net events and shot results, on scripted rallies.

Every trajectory here is flown by `rally.fly` under gravity, so the true
hit and bounce times and places are known exactly. No frame is decoded and
no model is loaded.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from tracking import court, events
from tracking.config import EventConfig
from tracking.events import (
    BounceEvent,
    HitEvent,
    NetEvent,
    _Kink,
    _Reading,
)
from tracking.provenance import Provenance, weakest

from . import rally
from .rally import Stroke

BASE = (
    Stroke("near", (-2.0, 17.5), 0.9, 0.5),
    Stroke("far", (1.5, 5.0), 1.0, 0.45),
    Stroke("near", (0.5, 26.5), 0.9, 0.4),
)


@pytest.fixture(scope="module")
def base_script():
    """Near hits in, far returns in, near hits long."""
    return rally.fly(BASE)


@pytest.fixture(scope="module")
def base_players(calibration, base_script):
    return rally.player_tracks(calibration, base_script)


@pytest.fixture(scope="module")
def base(camera, calibration, base_script, base_players):
    """The base rally's events, from one unbroken ball segment."""
    track = rally.ball_track(camera, calibration, base_script)
    return events.detect_events(track, base_players, calibration)


def detect(camera, calibration, script, audio=None, config=None, missing=(), **track):
    """Runs event detection on a scripted rally."""
    people = rally.player_tracks(calibration, script, missing=missing)
    ball = rally.ball_track(camera, calibration, script, **track)
    return events.detect_events(ball, people, calibration, audio, config)


# ------------------------------------------------------------------- audio


def clicks(times, seconds=6.0, rate=8000):
    """Quiet noise with a sharp click at each of `times`."""
    rng = np.random.default_rng(1)
    samples = rng.normal(0, 20, int(seconds * rate))
    for moment in times:
        at = int(moment * rate)
        samples[at : at + 40] += 12000 * np.sign(rng.normal(size=40))
    return samples.astype(np.int16)


def test_sharp_sounds_become_onsets_on_the_video_clock():
    found = events.audio_onsets(clicks([1.0, 2.5, 4.0]), start=100.0)
    loud = found.times[found.strengths >= 2.5]

    # Strength is relative to the recording's own 99th percentile, so a few
    # noise peaks always clear 1.0; the clicks are the ones far above it.
    assert np.allclose(loud, [101.0, 102.5, 104.0], atol=0.011)
    assert (found.strengths >= EventConfig().audio_min_strength).all()
    assert (np.diff(found.times) > EventConfig().audio_separation_seconds).all()


def test_silence_and_too_little_audio_have_no_onsets():
    assert len(events.audio_onsets(np.zeros(8000, dtype=np.int16)).times) == 0
    assert len(events.audio_onsets(np.zeros(10, dtype=np.int16)).times) == 0


def test_the_nearest_onset_is_found_only_inside_the_window():
    onsets = events.AudioOnsets(np.array([1.0, 2.0]), np.array([1.5, 2.5]))

    assert onsets.nearest(1.04, 0.06) == (1.0, 1.5)
    assert onsets.nearest(1.5, 0.06) is None
    assert events.AudioOnsets(np.zeros(0), np.zeros(0)).nearest(1.0, 1.0) is None


def test_audio_is_read_through_ffmpeg_for_just_the_stretch_asked(monkeypatch, tmp_path):
    asked = []

    def fake_run(command, capture_output):
        asked.append(command)
        return subprocess.CompletedProcess(command, 0, clicks([0.5, 1.5], seconds=3).tobytes(), b"")

    monkeypatch.setattr(events.subprocess, "run", fake_run)

    found = events.read_audio_onsets(tmp_path / "match.mp4", 60.0, 63.0, ffmpeg="ffmpeg")
    whole = events.read_audio_onsets(tmp_path / "match.mp4", ffmpeg="ffmpeg")

    assert asked[0][asked[0].index("-ss") + 1] == "60.000"
    assert asked[0][asked[0].index("-t") + 1] == "3.000"
    assert "-t" not in asked[1]
    assert np.allclose(found.times[found.strengths >= 2.5], [60.5, 61.5], atol=0.011)
    assert np.allclose(whole.times[whole.strengths >= 2.5], [0.5, 1.5], atol=0.011)


def test_unreadable_audio_is_reported_as_none_not_raised(monkeypatch, tmp_path):
    failed = subprocess.CompletedProcess([], 1, b"", b"no audio")
    monkeypatch.setattr(events.subprocess, "run", lambda *a, **k: failed)
    assert events.read_audio_onsets(tmp_path / "silent.mp4", ffmpeg="ffmpeg") is None

    def missing(*args, **kwargs):
        raise OSError("no ffmpeg")

    monkeypatch.setattr(events.subprocess, "run", missing)
    assert events.read_audio_onsets(tmp_path / "silent.mp4", ffmpeg="ffmpeg") is None


# ---------------------------------------------------------------- geometry

HALF_SINGLES = court.SINGLES_WIDTH_M / 2
HALF_DOUBLES = court.DOUBLES_WIDTH_M / 2
NET = court.NET_Y_M
FAR_SERVICE = NET + court.SERVICE_LINE_FROM_NET_M
NEAR_SERVICE = NET - court.SERVICE_LINE_FROM_NET_M


def test_a_rally_shot_must_land_in_the_other_half_of_the_court():
    assert events.target_region("near") == (-HALF_SINGLES, HALF_SINGLES, NET, court.COURT_LENGTH_M)
    assert events.target_region("far") == (-HALF_SINGLES, HALF_SINGLES, 0.0, NET)
    assert events.target_region("near", "doubles")[:2] == (-HALF_DOUBLES, HALF_DOUBLES)


@pytest.mark.parametrize(
    ("side", "zone", "expected"),
    [
        ("near", "deuce", (-HALF_SINGLES, 0.0, NET, FAR_SERVICE)),
        ("near", "ad", (0.0, HALF_SINGLES, NET, FAR_SERVICE)),
        ("far", "deuce", (0.0, HALF_SINGLES, NEAR_SERVICE, NET)),
        ("far", "ad", (-HALF_SINGLES, 0.0, NEAR_SERVICE, NET)),
    ],
)
def test_a_serve_must_land_in_the_box_diagonally_opposite(side, zone, expected):
    assert events.target_region(side, "singles", zone) == pytest.approx(expected)
    # The alleys never belong to a service box, doubles or not.
    assert events.target_region(side, "doubles", zone) == pytest.approx(expected)


def test_unknown_sides_courts_and_zones_are_refused():
    with pytest.raises(ValueError, match="hitter_side"):
        events.target_region("left")
    with pytest.raises(ValueError, match="court"):
        events.target_region("near", "mixed")
    with pytest.raises(ValueError, match="serve_zone"):
        events.target_region("near", "singles", "middle")


SURE = (0.1, 0.1)


@pytest.mark.parametrize(
    ("spot", "side", "value", "line"),
    [
        ((0.0, 18.0), "near", "In", "sideline"),
        ((HALF_SINGLES + 0.5, 18.0), "near", "Out", "sideline"),
        ((-HALF_SINGLES - 0.5, 18.0), "near", "Out", "sideline"),
        ((0.0, court.COURT_LENGTH_M + 0.5), "near", "Out", "baseline"),
        ((0.0, NET - 0.5), "near", "Out", "net"),
        ((0.0, -0.5), "far", "Out", "baseline"),
        ((0.0, NET + 0.5), "far", "Out", "net"),
        ((1.0, 0.5), "far", "In", "baseline"),
    ],
)
def test_a_rally_bounce_is_called_against_each_line(spot, side, value, line):
    call = events.judge_bounce(spot, SURE, side)

    assert (call.value, call.line) == (value, line)
    assert (call.margin_m > 0) == (value == "In")


def test_the_alley_is_out_in_singles_and_in_in_doubles():
    alley = (HALF_SINGLES + 0.6, 18.0)

    assert events.judge_bounce(alley, SURE, "near", "singles").value == "Out"
    assert events.judge_bounce(alley, SURE, "near", "doubles").value == "In"


@pytest.mark.parametrize(
    ("spot", "value", "line"),
    [
        ((-2.0, 15.0), "In", "centre line"),
        ((0.5, 15.0), "Out", "centre line"),
        ((-2.0, FAR_SERVICE + 0.5), "Out", "service line"),
        ((-HALF_SINGLES - 0.5, 15.0), "Out", "sideline"),
    ],
)
def test_a_serve_is_called_against_its_service_box(spot, value, line):
    # A near server in the deuce court serves to the camera's left.
    call = events.judge_bounce(spot, SURE, "near", serve_zone="deuce")

    assert (call.value, call.line) == (value, line)
    # The same bounce is a perfectly good rally shot, bar the one in the alley.
    assert events.judge_bounce(spot, SURE, "near").value == ("Out" if line == "sideline" else "In")


def test_a_far_servers_box_is_on_the_near_side():
    assert events.judge_bounce((2.0, 8.0), SURE, "far", serve_zone="deuce").value == "In"
    assert events.judge_bounce((2.0, 4.0), SURE, "far", serve_zone="deuce").line == "service line"
    assert events.judge_bounce((-2.0, 8.0), SURE, "far", serve_zone="deuce").line == "centre line"


def test_a_bounce_within_its_own_uncertainty_of_a_line_is_not_called():
    long_by_a_little = (0.0, court.COURT_LENGTH_M + 0.3)

    vague = events.judge_bounce(long_by_a_little, (0.1, 0.6), "near")
    sharp = events.judge_bounce(long_by_a_little, (0.1, 0.1), "near")
    stricter = events.judge_bounce(long_by_a_little, (0.1, 0.2), "near", sigmas=2.0)

    assert vague.value is None and vague.line == "baseline"
    assert vague.margin_m == pytest.approx(-0.3) and vague.uncertainty_m == 0.6
    assert sharp.value == "Out"
    assert stricter.value is None
    # Depth is vague but width is not: a ball well wide is still out.
    assert events.judge_bounce((HALF_SINGLES + 0.5, 23.6), (0.1, 0.6), "near").value == "Out"


def test_a_pixel_is_worth_far_more_court_at_the_far_end(calibration):
    near = events.ground_uncertainty(calibration, tuple(calibration.to_image([0.0, 2.0])), 2.0)
    far = events.ground_uncertainty(calibration, tuple(calibration.to_image([0.0, 22.0])), 2.0)

    assert far[1] > 5 * near[1]
    assert far[1] > far[0]
    assert near[1] == pytest.approx(2 * 0.028, rel=0.5)


def test_a_spot_that_maps_nowhere_has_infinite_uncertainty(calibration):
    broken = replace(calibration, inverse=np.full((3, 3), np.nan))

    assert events.ground_uncertainty(broken, (100.0, 100.0), 2.0) == (float("inf"), float("inf"))


def test_speed_is_distance_over_time_and_is_an_average():
    # 22.352 m in one second is exactly 50 mph.
    assert events.average_speed_mph((0.0, 0.0), 10.0, (0.0, 22.352), 11.0) == pytest.approx(50.0)
    assert events.average_speed_mph((0.0, 0.0), 10.0, (3.0, 4.0), 10.5) == pytest.approx(
        22.37, 0.01
    )


def test_a_speed_is_not_reported_when_it_cannot_be_believed():
    assert events.average_speed_mph((0.0, 0.0), 10.0, (0.0, 20.0), 10.05) is None
    assert events.average_speed_mph((0.0, 0.0), 10.0, (0.0, 20.0), 10.2) is None
    assert events.average_speed_mph((0.0, 0.0), 10.0, (0.0, 20.0), 9.0) is None


def test_direction_comes_from_which_side_of_the_centre_line_the_ball_lands():
    assert events.shot_direction((2.0, 0.0), (-2.0, 20.0)) == "cross court"
    assert events.shot_direction((2.0, 0.0), (3.0, 20.0)) == "down the line"
    assert events.shot_direction((-2.0, 24.0), (-1.0, 4.0)) == "down the line"
    assert events.shot_direction((2.0, 0.0), (0.2, 20.0), uncertainty_m=0.3) is None
    assert events.shot_direction((0.0, 0.0), (2.0, 20.0)) is None


def test_a_serves_direction_is_the_t_or_wide():
    assert events.shot_direction((1.0, 0.0), (-0.5, 17.0), serve=True) == "down the t"
    assert events.shot_direction((1.0, 0.0), (-3.5, 17.0), serve=True) == "out wide"
    assert events.shot_direction((1.0, 0.0), (-2.1, 17.0), serve=True, uncertainty_m=0.3) is None


# ------------------------------------------------------- the scripted rally


def test_every_hit_is_found_at_its_time_and_given_to_the_right_player(base, base_script):
    assert len(base.hits) == 3
    for hit, when, feet in zip(base.hits, base_script.hits, base_script.feet):
        assert hit.pts == pytest.approx(when, abs=0.02)
        assert hit.position == pytest.approx(feet, abs=0.05)
        assert 0.0 < hit.confidence <= 1.0
    assert [hit.side for hit in base.hits] == ["near", "far", "near"]
    assert [hit.player for hit in base.hits] == ["A", "B", "A"]
    assert [hit.zone for hit in base.hits] == ["deuce", "deuce", "deuce"]


def test_a_hit_says_how_it_was_seen(base):
    first, second, third = base.hits

    # The track begins at the first contact: the ball "appears" at the player.
    assert first.cues == ("appearance",) and first.provenance is Provenance.INFERRED
    assert "appeared" in first.reason
    assert second.cues == ("kink",) and second.provenance is Provenance.MEASURED
    assert third.time_uncertainty == pytest.approx(0.02)
    assert first.confidence < second.confidence


def test_every_bounce_is_found_where_it_landed(base, base_script):
    assert len(base.bounces) == 3
    for bounce, when, stroke in zip(base.bounces, base_script.bounces, base_script.strokes):
        assert bounce.pts == pytest.approx(when, abs=0.02)
        assert bounce.court_xy == pytest.approx(stroke.target, abs=0.25)
        assert bounce.provenance is Provenance.MEASURED
    assert [bounce.half for bounce in base.bounces] == ["far", "near", "far"]


def test_a_bounce_carries_its_uncertainty_in_metres(base):
    far, near, _ = base.bounces

    assert far.uncertainty_m[1] > near.uncertainty_m[1]
    assert far.uncertainty_m[1] > far.uncertainty_m[0]
    assert min(near.uncertainty_m) >= EventConfig().bounce_floor_m


def test_court_position_is_filled_at_the_bounces_and_nowhere_else(base, camera, calibration):
    original = rally.ball_track(camera, calibration, rally.fly(BASE))
    landed = [o for o in base.ball.observations if o.court_xy is not None]

    assert all(o.court_xy is None for o in original.observations)
    assert len(landed) == 3
    for observation, bounce in zip(landed, base.bounces):
        assert observation.pts == pytest.approx(bounce.pts)
        assert observation.court_xy == bounce.court_xy == observation.ground_ray_xy
    assert len(base.ball.observations) == len(original.observations)


def test_shots_are_judged_in_or_out_against_the_lines(base):
    first, second, third = base.shots

    assert [shot.result.value for shot in base.shots] == ["In", "In", "Out"]
    assert first.returned and second.returned and not third.returned
    assert first.call.line == "sideline" and "inside" in first.result_reason
    assert third.call.line == "baseline" and third.call.margin_m == pytest.approx(-2.73, abs=0.3)
    assert "beyond the baseline" in third.result_reason
    assert all(shot.result.provenance is Provenance.MEASURED for shot in base.shots)
    assert not any(shot.serve for shot in base.shots)


def test_speed_is_the_average_from_the_hitters_feet_to_the_bounce(base, base_script):
    for shot, feet, hit, bounce, stroke in zip(
        base.shots, base_script.feet, base_script.hits, base_script.bounces, base_script.strokes
    ):
        metres = np.hypot(stroke.target[0] - feet[0], stroke.target[1] - feet[1])
        assert shot.speed_mph.value == pytest.approx(metres / (bounce - hit) * 2.2369, rel=0.04)
    # The first hit was only inferred, so its speed is no better than that.
    assert base.shots[0].speed_mph.provenance is Provenance.INFERRED
    # A speed is only as good as the worst thing it was computed from.
    second = base.shots[1]
    assert second.speed_mph.provenance is weakest(
        [second.hit.provenance, second.hit.position_provenance, second.bounce.provenance]
    )
    assert second.speed_mph.confidence == min(second.hit.confidence, second.bounce.confidence)


def test_direction_is_reported_per_shot(base):
    assert [shot.direction.value for shot in base.shots] == [
        "cross court",
        "cross court",
        "down the line",
    ]


def test_quality_counts_what_was_found(base):
    quality = base.quality()

    assert (quality.hits, quality.bounces, quality.nets, quality.shots) == (3, 3, 0, 3)
    assert quality.hits_by_side == {"near": 2, "far": 1}
    assert quality.bounces_by_half == {"near": 1, "far": 2}
    assert quality.results == {"In": 2, "Out": 1, "Net": 0, "unknown": 0}
    assert quality.bounce_seen_by_landing_half == {"near": (1, 1), "far": (2, 2)}
    assert quality.hits_inferred == 1 and quality.speeds == 3
    assert quality.abstention_rate == 0.0 and quality.notice() is None


def test_events_serialise_to_json(base):
    stored = json.loads(json.dumps(base.to_dict()))

    assert stored["source"] == "events-v1" and stored["court"] == "singles"
    assert len(stored["hits"]) == len(stored["shots"]) == 3
    assert stored["shots"][2]["result"]["value"] == "Out"
    assert stored["shots"][0]["call"]["line"] == "sideline"
    assert stored["bounces"][0]["provenance"] == "measured"
    assert stored["quality"]["bounce_seen_by_landing_half"]["far"] == [2, 2]
    assert stored["hits"][0]["position_provenance"] == "measured"


def test_segments_that_break_at_each_hit_give_the_same_events(camera, calibration, base_script):
    found = detect(camera, calibration, base_script, breaks=base_script.hits[1:])

    assert [hit.side for hit in found.hits] == ["near", "far", "near"]
    for hit, when in zip(found.hits, base_script.hits):
        assert hit.pts == pytest.approx(when, abs=0.03)
    assert found.hits[1].cues == ("junction",)
    assert found.hits[1].provenance is Provenance.INTERPOLATED
    assert [shot.result.value for shot in found.shots] == ["In", "In", "Out"]


def test_a_racket_taken_for_the_ball_changes_nothing(camera, calibration, base_script, base):
    people = rally.player_tracks(calibration, base_script)
    box = people.at(base_script.hits[2], "near").box
    where = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    blip = (base_script.hits[2] - 0.3, base_script.hits[2] - 0.05, where)

    found = detect(camera, calibration, base_script, blips=[blip])

    assert [round(hit.pts, 2) for hit in found.hits] == [round(hit.pts, 2) for hit in base.hits]
    assert [bounce.court_xy for bounce in found.bounces] == [b.court_xy for b in base.bounces]
    assert [shot.result.value for shot in found.shots] == ["In", "In", "Out"]


# -------------------------------------------------- gaps, and abstaining


def test_a_short_gap_across_a_bounce_is_bridged_and_tagged(camera, calibration, base_script, base):
    landing = base_script.bounces[0]

    found = detect(camera, calibration, base_script, hidden=[(landing - 0.07, landing + 0.07)])
    bounce = found.shots[0].bounce

    assert bounce.provenance is Provenance.INTERPOLATED
    assert bounce.pts == pytest.approx(landing, abs=0.08)
    assert bounce.court_xy == pytest.approx(BASE[0].target, abs=1.5)
    assert bounce.uncertainty_m[1] > base.shots[0].bounce.uncertainty_m[1]
    assert bounce.confidence < base.shots[0].bounce.confidence
    assert "unseen" in bounce.reason
    assert found.shots[0].result.provenance in (Provenance.INTERPOLATED, Provenance.INFERRED)
    # No frame showed the ball on the ground, so no observation claims a court position there.
    assert not any(
        o.court_xy is not None and abs(o.pts - landing) < 0.2 for o in found.ball.observations
    )


def test_a_long_gap_across_a_bounce_reports_no_bounce(camera, calibration, base_script):
    first, last = base_script.bounces[0], base_script.bounces[2]

    found = detect(
        camera,
        calibration,
        base_script,
        hidden=[(first - 0.3, first + 0.3), (last - 0.25, last + 5.0)],
    )
    returned, _, unreturned = found.shots

    assert returned.bounce is None and unreturned.bounce is None
    # The far player hit the first shot back: it was in, by inference only.
    assert returned.result.value == "In"
    assert returned.result.provenance is Provenance.INFERRED
    assert "no bounce was seen" in returned.result_reason
    assert returned.speed_mph is None and returned.direction is None
    # Nothing at all is known about how the last shot ended.
    assert unreturned.result.value is None
    assert "no bounce, net contact or return" in unreturned.result_reason


def test_unknown_results_are_counted_by_the_end_they_were_hit_toward(camera, calibration):
    script = rally.fly(BASE[:1])
    landing = script.bounces[0]

    found = detect(camera, calibration, script, hidden=[(landing - 0.3, landing + 5.0)])
    quality = found.quality()

    assert quality.results["unknown"] == 1
    assert quality.unknown_by_landing_half == {"near": 0, "far": 1}
    assert quality.bounce_seen_by_landing_half["far"] == (0, 1)
    assert quality.abstention_rate == 1.0
    assert "1 of 1 shots have no result" in quality.notice()
    assert "1 hit toward the far end" in quality.notice()


def test_without_players_there_are_bounces_but_no_hits_or_shots(camera, calibration, base_script):
    ball = rally.ball_track(camera, calibration, base_script)

    found = events.detect_events(ball, None, calibration)
    quality = found.quality()

    assert found.hits == () and found.shots == ()
    assert len(found.bounces) == 3
    assert quality.abstention_rate == 0.0 and quality.notice() is None


def test_a_ball_in_the_net_is_a_net_result(camera, calibration):
    # A flat drive toward a far target, stopped dead where it reaches the net.
    script = rally.fly([Stroke("near", (0.0, 20.0), 0.6, 0.5, stops_at=0.35)])

    found = detect(camera, calibration, script)
    shot = found.shots[0]

    assert len(found.nets) == 1
    assert found.nets[0].pts == pytest.approx(script.hits[0] + 0.35, abs=0.05)
    assert "image speed at the net" in found.nets[0].reason
    assert shot.result.value == "Net" and shot.result.provenance is Provenance.MEASURED
    assert shot.net is found.nets[0] and shot.bounce is None
    assert shot.speed_mph is None
    assert found.quality().results["Net"] == 1
    assert json.dumps(found.to_dict()["nets"])


def test_the_far_player_can_net_it_too(camera, calibration):
    script = rally.fly(
        [
            Stroke("near", (-2.0, 17.5), 0.9, 0.5),
            Stroke("far", (3.0, 4.0), 0.7, 0.5, stops_at=0.44),
        ]
    )

    found = detect(camera, calibration, script)

    assert [shot.result.value for shot in found.shots] == ["In", "Net"]
    assert found.shots[1].hit.side == "far"
    # A ball that stops at the net in front of the far player was not hit by them again.
    assert len(found.hits) == 2


def test_a_bounce_too_close_to_the_line_is_not_called(camera, calibration):
    # 0.2 m long at the far baseline, where one pixel is 0.45 m of depth.
    script = rally.fly([Stroke("near", (0.0, court.COURT_LENGTH_M + 0.2), 0.9, 0.3)])

    shot = detect(camera, calibration, script).shots[0]

    assert shot.bounce is not None and shot.call.value is None
    assert shot.result.value is None
    assert "too close to call" in shot.result_reason
    assert shot.result.provenance is Provenance.MEASURED


def test_a_close_bounce_the_other_player_hit_is_in_by_inference(camera, calibration):
    script = rally.fly(
        [
            Stroke("near", (-2.0, court.COURT_LENGTH_M - 0.1), 0.9, 0.3),
            Stroke("far", (3.0, 6.0), 1.1, 0.3),
        ]
    )

    shot = detect(camera, calibration, script).shots[0]

    assert shot.call.value is None
    assert shot.result.value == "In" and shot.result.provenance is Provenance.INFERRED
    assert "the other player hit it next" in shot.result_reason


def test_a_measured_out_that_was_played_on_is_left_unknown_unless_a_serve():
    hits = [made_hit(1.0, "near"), made_hit(2.3, "far", (-2.0, 25.5))]
    long_ball = made_bounce(1.9, (-2.0, court.COURT_LENGTH_M + 2.5))

    shots, _ = events._assemble(hits, [long_ball], [], EventConfig())
    as_rally, as_serve = shots[0], events._judge(shots[0], True, EventConfig())

    assert as_rally.call.value == "Out" and as_rally.result.value is None
    assert as_rally.result.provenance is Provenance.INFERRED
    assert "the two disagree" in as_rally.result_reason
    # A fault is often hit back; that does not make it good.
    assert as_serve.serve and as_serve.result.value == "Out"
    assert as_serve.call.line == "service line"


def test_serves_are_rejudged_against_the_service_box(base):
    # The first shot landed at (-2, 17.5), inside the box a deuce-court serve must find.
    served = base.rejudged([0])
    wrong_box = base.rejudged([1])

    assert base.shots[0].call.line == "sideline" and not base.shots[0].serve
    assert served.shots[0].serve and served.shots[0].result.value == "In"
    assert served.shots[0].call.line == "service line"
    # ...and too near the middle of the box to say "down the t" or "out wide".
    assert served.shots[0].direction is None
    assert served.shots[1].result.value == "In" and not served.shots[1].serve
    assert served.hits == base.hits and served.bounces == base.bounces
    # The far player's shot landed short of the service line: a fine rally shot, a fault as a serve.
    assert base.shots[1].result.value == "In"
    assert wrong_box.shots[1].serve and wrong_box.shots[1].result.value == "Out"
    assert wrong_box.shots[1].call.line == "service line"
    assert not wrong_box.shots[0].serve


# ----------------------------------------------------- inference, tagged


def test_a_hit_that_was_never_seen_is_added_and_tagged_inferred(camera, calibration, base_script):
    away, back = base_script.bounces[0], base_script.bounces[1]

    found = detect(camera, calibration, base_script, hidden=[(away + 0.15, back - 0.15)])
    unseen = found.hits[1]

    assert [hit.side for hit in found.hits] == ["near", "far", "near"]
    assert unseen.provenance is Provenance.INFERRED and unseen.cues == ("returned",)
    assert unseen.image_xy is None and unseen.confidence == pytest.approx(0.2)
    assert "was returned in between" in unseen.reason
    # Bracketed by the bounce at the far end and the bounce on the way back.
    assert away <= unseen.pts <= back - EventConfig().min_flight_seconds
    assert abs(unseen.pts - base_script.hits[1]) <= unseen.time_uncertainty + 0.05
    assert unseen.position is not None and unseen.player == "B"
    assert found.shots[0].returned and found.shots[1].returned
    assert found.quality().hits_inferred == 2


def test_with_no_bounces_seen_the_unseen_hit_sits_mid_way(camera, calibration, base_script):
    first, third = base_script.hits[0], base_script.hits[2]

    found = detect(camera, calibration, base_script, hidden=[(first + 0.3, third - 0.3)])
    unseen = found.hits[1]

    assert unseen.cues == ("returned",)
    assert unseen.pts == pytest.approx((first + third) / 2, abs=0.15)
    assert unseen.time_uncertainty > 0.5


def test_two_hits_by_one_player_far_apart_imply_nothing(camera, calibration, base_script):
    first, third = base_script.hits[0], base_script.hits[2]
    config = EventConfig(max_exchange_seconds=1.0)

    found = detect(
        camera, calibration, base_script, config=config, hidden=[(first + 0.3, third - 0.3)]
    )

    assert [hit.side for hit in found.hits] == ["near", "near"]
    assert not found.shots[0].returned


def test_a_ball_that_lands_on_the_hitters_own_side_did_not_cross(camera, calibration):
    script = rally.fly([Stroke("near", (0.5, 8.0), 0.6, 0.3)])

    shot = detect(camera, calibration, script).shots[0]

    assert shot.bounce is not None and shot.bounce.half == "near" and shot.call is None
    assert shot.result.value == "Net" and shot.result.provenance is Provenance.INFERRED
    assert "own side of the net" in shot.result_reason


def test_an_unreturned_bounce_beside_a_player_is_a_bounce_not_their_hit(camera, calibration, base):
    # An ace: it lands beside where the near player stands, and they never touch it.
    script = rally.fly(
        [
            Stroke("near", (-2.0, 17.5), 0.9, 0.5),
            Stroke("far", (3.0, 2.0), 1.1, 0.3),
        ]
    )
    people = rally.player_tracks(calibration, script)
    box = rally.person_box(calibration, (2.5, 2.0))
    standing = replace(
        people,
        near=tuple(
            replace(o, box=box, foot_court=(2.5, 2.0), foot_image=((box[0] + box[2]) / 2, box[3]))
            if o.pts > script.hits[1]
            else o
            for o in people.near
        ),
    )
    ball = rally.ball_track(camera, calibration, script)

    found = events.detect_events(ball, standing, calibration)

    assert [hit.side for hit in found.hits] == ["near", "far"]
    assert found.shots[1].bounce.court_xy == pytest.approx((3.0, 2.0), abs=0.2)
    assert found.shots[1].result.value == "In" and not found.shots[1].returned
    # Within reach of a player, a bounce is less certain to be one.
    assert found.shots[1].bounce.confidence == pytest.approx(0.8 * 0.7)
    assert base.bounces[2].confidence == pytest.approx(0.8)


# --------------------------------------------------------------- audio cue


def test_a_sound_where_the_hit_would_be_heard_confirms_it(camera, calibration, base_script, base):
    scene = events._Scene(base.ball, None, calibration, None, EventConfig())
    heard = [
        when + scene.sound_delay(feet, stroke.side)
        for when, feet, stroke in zip(base_script.hits, base_script.feet, base_script.strokes)
    ]
    audio = events.AudioOnsets(np.array(heard), np.array([2.0, 1.2, 2.5]))

    found = detect(camera, calibration, base_script, audio=audio)

    assert all("audio" in hit.cues for hit in found.hits)
    assert [hit.audio_strength for hit in found.hits] == [2.0, 1.2, 2.5]
    assert found.hits[1].confidence > base.hits[1].confidence
    assert found.quality().hits_with_audio == 3
    # The far player is some 30 m from the camera: their sound arrives late.
    assert heard[1] - base_script.hits[1] > 0.05 > heard[0] - base_script.hits[0]


def test_a_sound_retimes_only_a_hit_the_picture_was_vague_about(camera, calibration, base_script):
    scene = events._Scene(
        rally.ball_track(camera, calibration, base_script), None, calibration, None, EventConfig()
    )
    late = 0.04
    heard = [
        when + late + scene.sound_delay(feet, stroke.side)
        for when, feet, stroke in zip(base_script.hits, base_script.feet, base_script.strokes)
    ]
    audio = events.AudioOnsets(np.array(heard), np.ones(3))

    found = detect(camera, calibration, base_script, audio=audio)

    # The first hit is an appearance (vague): it moves onto the sound.
    assert found.hits[0].pts == pytest.approx(base_script.hits[0] + late, abs=0.005)
    assert found.hits[0].time_uncertainty == EventConfig().audio_retime_seconds
    # The second was seen as a kink in the track: confirmed, not moved.
    assert found.hits[1].pts == pytest.approx(base_script.hits[1], abs=0.02)
    assert "audio" in found.hits[1].cues


def test_a_sound_at_the_wrong_moment_confirms_nothing(camera, calibration, base_script):
    audio = events.AudioOnsets(np.array([h + 0.4 for h in base_script.hits]), np.ones(3) * 3)

    found = detect(camera, calibration, base_script, audio=audio)

    assert not any("audio" in hit.cues for hit in found.hits)
    assert all(hit.audio_strength is None for hit in found.hits)


# --------------------------------------------------- the rules, one by one


def kink(pts=1.0, xy=(480.0, 300.0), before=(0.0, 100.0), after=(0.0, -100.0), **more):
    """A hand-made kink; an appearance gets the straight path its velocity implies."""
    kind = more.pop("kind", "kink")
    if kind == "appearance":
        more.setdefault("path", (np.array([after[0], xy[0]]), np.array([after[1], xy[1]])))
    return _Kink(
        pts=pts,
        xy=np.array(xy, dtype=float),
        before=None if before is None else np.array(before, dtype=float),
        after=np.array(after, dtype=float),
        kind=kind,
        **more,
    )


def scene_for(camera, calibration, script=None, config=None, audio=None, **people):
    """A `_Scene` over a scripted rally."""
    script = script or rally.fly(BASE)
    return events._Scene(
        rally.ball_track(camera, calibration, script),
        rally.player_tracks(calibration, script, **people),
        calibration,
        audio,
        config or EventConfig(),
    )


def test_a_ball_that_looks_too_small_is_not_within_the_near_players_reach(camera, calibration):
    scene = scene_for(camera, calibration)
    person = scene.player(1.0, "near")
    head = ((person.box[0] + person.box[2]) / 2, person.box[1] - 5.0)
    expected = EventConfig().ball_diameter_m * scene.row_scale(person.foot_image[1])

    right_size = events._read(kink(xy=head, extent=1.5 * expected), scene)
    far_ball = events._read(kink(xy=head, extent=0.3 * expected), scene)
    unsized = events._read(kink(xy=head, extent=None), scene)

    assert right_size.reach["near"] < 0.5
    assert far_ball.reach["near"] == float("inf")
    assert unsized.reach["near"] == right_size.reach["near"]
    assert not right_size.boxed
    assert events._read(kink(xy=(head[0], head[1] + 30.0)), scene).boxed


def test_a_player_not_tracked_at_that_moment_is_looked_up_nearby_or_not_at_all(camera, calibration):
    scene = scene_for(camera, calibration)
    people = scene.players
    sparse = replace(
        people,
        near=tuple(o for o in people.near if not 2.0 < o.pts < 4.5),
        max_interpolation_seconds=0.3,
        _times={},
    )
    scene = events._Scene(scene.ball, sparse, calibration, None, EventConfig())
    nobody = events._Scene(scene.ball, None, calibration, None, EventConfig())
    empty = events._Scene(
        scene.ball, replace(people, near=(), _times={}), calibration, None, EventConfig()
    )

    assert scene.player(2.5, "near").pts == pytest.approx(2.0, abs=0.11)
    assert scene.player(4.2, "near").pts == pytest.approx(4.5, abs=0.11)
    assert scene.player(3.25, "near") is None
    assert scene.player(-50.0, "near") is None
    assert nobody.player(2.5, "near") is None
    assert empty.player(2.5, "near") is None


def test_without_a_recoverable_camera_there_is_no_net_and_a_default_delay(
    camera, calibration, monkeypatch
):
    monkeypatch.setattr(events, "camera_from_homography", lambda *args: None)
    scene = scene_for(camera, calibration)
    on_the_net = calibration.to_image(np.array([0.0, court.NET_Y_M]))

    assert scene.net is None and not scene.at_net(on_the_net)
    assert scene.sound_delay(None, "near") == pytest.approx(6.5 / 343, abs=0.005)
    assert scene.sound_delay(None, "far") == pytest.approx(30.0 / 343, abs=0.01)


def test_the_net_is_where_the_camera_sees_it(camera, calibration):
    scene = scene_for(camera, calibration)
    foot = calibration.to_image(np.array([0.0, court.NET_Y_M]))

    assert scene.at_net(foot + np.array([0.0, -5.0]))
    assert not scene.at_net(foot + np.array([0.0, 60.0]))


def reading(scene, side, **more):
    """A reading of a hand-made kink at one player's racket."""
    person = scene.player(more.get("pts", 1.0), side)
    spot = ((person.box[0] + person.box[2]) / 2 + 5.0, (person.box[1] + person.box[3]) / 2)
    return events._read(kink(xy=more.pop("xy", spot), **more), scene)


def test_a_ball_leaving_the_near_player_downward_is_not_their_hit(camera, calibration):
    scene = scene_for(camera, calibration)
    hit = reading(scene, "near", after=(100.0, -900.0))
    rolling = reading(scene, "near", before=None, after=(600.0, 150.0), kind="appearance", gap=1.0)

    assert events._strength(hit, "near", scene) > 1.0
    assert rolling.reach["near"] <= EventConfig().reach_m
    assert events._strength(rolling, "near", scene) == 0.0
    assert events._choose([rolling], [], "near", scene) is None


def test_a_serve_struck_above_the_horizon_may_leave_downward(camera, calibration):
    scene = scene_for(camera, calibration)
    person = scene.player(1.0, "near")
    above = ((person.box[0] + person.box[2]) / 2, 20.0)
    serve = events._read(kink(xy=above, before=(0.0, 50.0), after=(-300.0, 400.0)), scene)

    assert serve.ground is None
    assert events._strength(serve, "near", scene) > 1.0


def test_a_seen_bounce_at_the_far_players_feet_is_not_their_hit(camera, calibration):
    scene = scene_for(camera, calibration)
    person = scene.player(1.0, "far")
    feet = (person.foot_image[0] + 12.0, person.foot_image[1] + 2.0)
    racket = (person.foot_image[0] + 12.0, person.foot_image[1] - 25.0)

    at_feet = events._read(kink(xy=feet, before=(0.0, 60.0), after=(0.0, -80.0)), scene)
    at_racket = events._read(kink(xy=racket, before=(0.0, 60.0), after=(0.0, -80.0)), scene)

    assert at_feet.bounce_like and events._strength(at_feet, "far", scene) == 0.0
    assert events._strength(at_racket, "far", scene) > 0.0


def test_the_players_take_turns_and_cannot_hit_twice_in_a_hurry(camera, calibration, base):
    scene = scene_for(camera, calibration)
    config = EventConfig()
    near_hit = base.hits[0]
    at_near = reading(scene, "near", pts=near_hit.pts + 0.5)
    at_far = reading(scene, "far", pts=near_hit.pts + 0.1)
    nowhere = events._read(kink(xy=(480.0, 10.0)), scene)

    assert events._eligible(at_near, None, config) == "near"
    assert events._eligible(at_far, None, config) == "far"
    assert events._eligible(nowhere, None, config) is None
    # 0.5 s after their own hit the near player cannot have hit it again...
    assert events._eligible(at_near, near_hit, config) is None
    # ...nor can the far player have reached it after 0.1 s.
    assert events._eligible(at_far, near_hit, config) is None
    assert events._eligible(replace_pts(at_far, near_hit.pts + 0.6), near_hit, config) == "far"
    assert events._eligible(replace_pts(at_near, near_hit.pts + 1.5), near_hit, config) == "near"


def replace_pts(found: _Reading, pts: float) -> _Reading:
    """The same reading at another time."""
    return replace(found, kink=replace(found.kink, pts=pts))


def test_in_one_visit_the_last_hard_kink_is_the_hit(camera, calibration):
    scene = scene_for(camera, calibration)
    bounce = reading(scene, "near", before=(100.0, 500.0), after=(80.0, -300.0))
    hit = reading(scene, "near", pts=1.4, before=(300.0, 400.0), after=(-900.0, -1500.0))
    wobble = reading(scene, "near", pts=1.5, before=(-900.0, -1500.0), after=(-880.0, -1400.0))

    assert bounce.bounce_like and not hit.bounce_like
    assert events._choose([bounce, hit], [], "near", scene) == 1
    # A faint kink after the hit does not take its place.
    assert events._choose([bounce, hit, wobble], [], "near", scene) == 1
    assert events._choose([wobble], [], "near", scene) is None


def test_a_lone_bounce_like_kink_is_a_hit_only_if_the_ball_then_goes_away(camera, calibration):
    scene = scene_for(camera, calibration)
    lone = reading(scene, "near", before=(100.0, 500.0), after=(80.0, -300.0))
    at_far = reading(scene, "far", pts=2.0)
    far_bounce = events._read(
        kink(
            pts=2.0,
            xy=tuple(calibration.to_image(np.array([3.0, 17.0]))),
            before=(0.0, 60.0),
            after=(0.0, -80.0),
        ),
        scene,
    )
    too_late = replace_pts(at_far, 9.0)

    assert lone.bounce_like
    assert events._choose([lone], [], "near", scene) is None
    assert events._choose([lone], [at_far], "near", scene) == 0
    assert events._choose([lone], [far_bounce], "near", scene) == 0
    assert events._choose([lone], [too_late], "near", scene) is None


def test_a_sideways_reversal_or_a_dead_stop_is_not_a_bounce(camera, calibration):
    scene = scene_for(camera, calibration)
    spot = tuple(calibration.to_image(np.array([2.0, 6.0])))

    assert events._read(
        kink(xy=spot, before=(50.0, 300.0), after=(40.0, -200.0)), scene
    ).bounce_like
    assert not events._read(
        kink(xy=spot, before=(300.0, 300.0), after=(-300.0, -200.0)), scene
    ).bounce_like
    assert not events._read(
        kink(xy=spot, before=(50.0, 300.0), after=(5.0, 10.0)), scene
    ).bounce_like
    assert not events._read(
        kink(xy=spot, before=(50.0, -300.0), after=(40.0, -500.0)), scene
    ).bounce_like
    assert not events._read(
        kink(xy=spot, before=(50.0, 300.0), after=(400.0, 200.0)), scene
    ).bounce_like
    # Far beyond the court is not a place a ball in play bounces.
    beyond = tuple(calibration.to_image(np.array([0.0, 40.0])))
    assert not events._read(
        kink(xy=beyond, before=(5.0, 30.0), after=(4.0, -20.0)), scene
    ).bounce_like
    assert not events._read(kink(xy=spot, before=None, after=(0.0, -200.0)), scene).bounce_like


def test_a_stop_inside_a_players_box_or_away_from_the_net_is_not_a_net_contact(
    camera, calibration, base
):
    scene = scene_for(camera, calibration)
    last = base.hits[0]
    on_net = tuple(calibration.to_image(np.array([3.0, court.NET_Y_M])) + np.array([0.0, -8.0]))
    stop = dict(before=(0.0, -300.0), after=(0.0, 5.0))

    assert events._net_from(events._read(kink(xy=on_net, **stop), scene), last, scene) is not None
    assert events._net_from(events._read(kink(xy=on_net, **stop), scene), None, scene) is None
    junction = events._read(kink(xy=on_net, kind="junction", **stop), scene)
    assert events._net_from(junction, last, scene) is None
    wrong_way = events._read(kink(xy=on_net, before=(0.0, 300.0), after=(0.0, 5.0)), scene)
    assert events._net_from(wrong_way, last, scene) is None
    person = scene.player(1.0, "near")
    inside = ((person.box[0] + person.box[2]) / 2, (person.box[1] + person.box[3]) / 2)
    boxed = events._read(kink(xy=inside, **stop), scene)
    assert boxed.boxed and events._net_from(boxed, last, scene) is None
    still_moving = events._read(kink(xy=on_net, before=(0.0, -300.0), after=(0.0, -200.0)), scene)
    assert events._net_from(still_moving, last, scene) is None


# ------------------------------------------------------------ finding kinks


def test_two_segments_that_do_not_meet_are_not_joined(camera, calibration, base_script):
    # Hide the far hit and 0.2 s either side: the paths before and after never meet.
    gone = (base_script.hits[1] - 0.1, base_script.hits[1] + 0.1)
    ball = rally.ball_track(camera, calibration, base_script, hidden=[gone])
    moved = tuple(
        replace(o, image_xy=(o.image_xy[0] + 200.0, o.image_xy[1])) if o.segment == 1 else o
        for o in ball.observations
    )

    kinks = events.find_kinks(replace(ball, observations=moved), calibration)
    kinds = [k.kind for k in kinks if abs(k.pts - base_script.hits[1]) < 0.2]

    assert kinds == ["appearance"]


def test_a_smooth_path_across_a_short_gap_is_no_event(camera, calibration, base_script, base):
    mid_flight = base_script.hits[0] + 0.4

    with_gap = detect(camera, calibration, base_script, hidden=[(mid_flight, mid_flight + 0.08)])

    assert [hit.pts for hit in with_gap.hits] == pytest.approx([hit.pts for hit in base.hits])
    assert len(with_gap.bounces) == 3


def test_a_gap_too_long_to_reach_across_starts_an_appearance(camera, calibration, base_script):
    mid_flight = base_script.hits[0] + 0.3
    ball = rally.ball_track(
        camera, calibration, base_script, hidden=[(mid_flight, mid_flight + 0.4)]
    )

    kinks = events.find_kinks(ball, calibration)
    resumed = [k for k in kinks if k.kind == "appearance" and k.pts > 1.5]

    assert len(resumed) == 1 and resumed[0].gap == pytest.approx(0.42, abs=0.03)
    assert resumed[0].path is not None and resumed[0].before is None


def test_runs_too_short_to_fit_are_ignored_and_sparse_ones_still_fit(camera, calibration):
    script = rally.fly(BASE[:1])
    ball = rally.ball_track(camera, calibration, script)
    first = ball.observations[0]
    lonely = replace(first, pts=first.pts - 2.0, segment=1)
    sparse = [
        replace(first, pts=first.pts - 1.0 + 0.3 * step, segment=2, image_xy=(100.0 + step, 100.0))
        for step in range(3)
    ]
    segments = ball.segments + tuple(
        replace(ball.segments[0], index=number, in_player_box=0.0) for number in (1, 2)
    )
    crowded = replace(
        ball,
        observations=tuple(sorted((lonely, *sparse, *ball.observations), key=lambda o: o.pts)),
        segments=segments,
    )

    kinks = events.find_kinks(crowded, calibration)

    assert not any(k.pts == pytest.approx(lonely.pts) for k in kinks)
    assert any(k.pts == pytest.approx(sparse[0].pts) and k.kind == "appearance" for k in kinks)
    assert events._end(sparse, EventConfig(), first=False) == sparse[-2:]


def test_an_empty_track_has_no_events(calibration):
    empty = rally.BallTrack(
        observations=(),
        segments=(),
        gaps=(),
        frame_numbers=np.zeros(0, dtype=int),
        frame_pts=np.zeros(0),
        image_size=rally.SIZE,
        source="none",
    )

    found = events.detect_events(empty, None, calibration)

    assert found.hits == found.bounces == found.nets == found.shots == ()
    assert found.quality().shots == 0


# ------------------------------------------------- judging, on made events


def made_hit(pts=1.0, side="near", position=(1.0, 0.0), zone="deuce"):
    return HitEvent(
        pts=pts,
        time_uncertainty=0.02,
        side=side,
        player="A" if side == "near" else "B",
        zone=zone,
        position=position,
        position_provenance=None if position is None else Provenance.MEASURED,
        foot_clipped=False,
        image_xy=(100.0, 100.0),
        confidence=0.75,
        provenance=Provenance.MEASURED,
        cues=("kink",),
    )


def made_bounce(pts=1.9, court_xy=(-2.0, 19.0), uncertainty=(0.3, 0.5)):
    return BounceEvent(
        pts=pts,
        time_uncertainty=0.02,
        court_xy=court_xy,
        uncertainty_m=uncertainty,
        half="near" if court_xy[1] < court.NET_Y_M else "far",
        image_xy=(200.0, 200.0),
        confidence=0.8,
        provenance=Provenance.MEASURED,
    )


def test_a_bounce_on_the_hitters_side_is_dropped_when_the_ball_plainly_went_over():
    hits = [made_hit(1.0, "near"), made_hit(2.4, "far", (0.0, 24.0))]
    wobble, real = made_bounce(1.4, (1.0, 9.0)), made_bounce(1.9)

    shots, kept = events._assemble(hits, [wobble, real], [], EventConfig())
    returned_only, kept_again = events._assemble(hits, [wobble], [], EventConfig())

    assert kept == [real] and shots[0].bounce is real
    assert kept_again == [] and returned_only[0].bounce is None
    assert returned_only[0].result.value == "In"


def test_a_net_contact_followed_by_a_bounce_over_the_net_is_not_a_net_result():
    hit = made_hit(1.0, "near")
    cord = NetEvent(1.5, (300.0, 300.0), 0.6, Provenance.MEASURED)

    over, _ = events._assemble([hit], [made_bounce(1.9)], [cord], EventConfig())
    stopped, _ = events._assemble([hit], [], [cord], EventConfig())

    assert over[0].net is None and over[0].result.value == "In"
    assert stopped[0].net is cord and stopped[0].result.value == "Net"


def test_a_shot_by_an_untracked_hitter_has_a_result_but_no_speed():
    hit = made_hit(1.0, "near", position=None, zone=None)

    shots, _ = events._assemble([hit], [made_bounce(1.9)], [], EventConfig())

    assert shots[0].result.value == "In"
    assert shots[0].speed_mph is None and shots[0].direction is None


def test_an_implausibly_fast_shot_gets_no_speed_but_keeps_its_direction():
    shots, _ = events._assemble([made_hit(1.0, "near")], [made_bounce(1.15)], [], EventConfig())

    assert shots[0].speed_mph is None
    assert shots[0].direction.value == "cross court"


def test_a_return_must_come_from_the_other_player_and_soon():
    slow = [made_hit(1.0, "near"), made_hit(9.0, "far", (0.0, 24.0))]
    same = [made_hit(1.0, "near"), made_hit(2.5, "near")]

    assert not events._assemble(slow, [], [], EventConfig())[0][0].returned
    assert not events._assemble(same, [], [], EventConfig())[0][0].returned


def test_a_kink_just_after_the_ball_appears_replaces_the_appearance(
    camera, calibration, base_script
):
    # The track picks the ball up five frames before the far player hits it.
    late = base_script.hits[1] - 0.085
    ball = rally.ball_track(camera, calibration, base_script, hidden=[(0.0, late)])

    kinks = events.find_kinks(ball, calibration)

    assert kinks[0].kind == "kink"
    assert kinks[0].pts == pytest.approx(base_script.hits[1], abs=0.02)
    assert not any(k.kind == "appearance" for k in kinks)


def test_a_stop_on_the_net_but_inside_a_players_box_is_not_trusted(camera, calibration, base):
    scene = scene_for(camera, calibration)
    on_net = tuple(calibration.to_image(np.array([3.0, court.NET_Y_M])) + np.array([0.0, -8.0]))
    stop = events._read(kink(xy=on_net, before=(0.0, -300.0), after=(0.0, 5.0)), scene)

    assert events._net_from(stop, base.hits[0], scene) is not None
    assert events._net_from(replace(stop, boxed=True), base.hits[0], scene) is None


def test_nothing_seen_on_the_nets_tape_is_a_bounce_or_a_hit(camera, calibration):
    scene = scene_for(camera, calibration)
    lifted = np.array([[2.0, court.NET_Y_M, court.NET_HEIGHT_CENTRE_M + 0.03, 1.0]]) @ camera.T
    on_tape = tuple(lifted[0, :2] / lifted[0, 2])
    below_tape = (on_tape[0], on_tape[1] + 15.0)
    bounce = dict(before=(10.0, 60.0), after=(8.0, -80.0))

    assert scene.behind_tape(np.array(on_tape)) and not scene.behind_tape(np.array(below_tape))
    assert events._read(kink(xy=below_tape, **bounce), scene).bounce_like
    # The same kink where the tape is: the ground there cannot be seen at all.
    hidden = events._read(kink(xy=on_tape, **bounce), scene)
    assert not hidden.bounce_like
    assert set(hidden.reach.values()) == {float("inf")}


def test_a_ball_landing_behind_the_tape_has_no_bounce_to_report(camera, calibration):
    # From this camera the tape hides the ground from about 19 m to 22 m down the court.
    script = rally.fly(
        [
            Stroke("near", (-2.0, 20.5), 0.9, 0.5),
            Stroke("far", (3.0, 5.0), 1.1, 0.3),
        ]
    )

    found = detect(camera, calibration, script)

    assert [hit.side for hit in found.hits] == ["near", "far"]
    assert found.shots[0].bounce is None
    assert found.shots[0].result.value == "In"
    assert found.shots[0].result.provenance is Provenance.INFERRED
    assert found.discarded_kinks >= 1


def test_a_serve_by_an_untracked_server_is_not_judged_against_a_guessed_box():
    server = made_hit(1.0, "near", position=None, zone=None)
    receiver = made_hit(2.4, "far", (0.0, 24.0))

    alone, _ = events._assemble([server], [made_bounce(1.9)], [], EventConfig())
    played, _ = events._assemble([server, receiver], [made_bounce(1.9)], [], EventConfig())
    unreturned = events._judge(alone[0], True, EventConfig())
    returned = events._judge(played[0], True, EventConfig())

    # Which service box it had to find depends on where the server stood.
    assert unreturned.call is None and unreturned.result.value is None
    assert "where the server stood is not known" in unreturned.result_reason
    assert returned.call is None and returned.result.value == "In"
    assert returned.result.provenance is Provenance.INFERRED
    assert "the other player hit the ball next" in returned.result_reason
