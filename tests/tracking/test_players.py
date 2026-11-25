"""Tests for player detection plumbing, association, selection and identity.

Every detector here is a fake: no test loads a model or touches the network.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tracking import models, players
from tracking.config import ModelConfig, PlayerConfig
from tracking.players import PersonDetection
from tracking.provenance import Provenance
from tracking.video import VideoFrame

from .conftest import SIZE

RED = (200, 40, 40)
BLUE = (40, 60, 200)
GREEN = (40, 180, 60)
RATE = 10.0


def person_box(calibration, spot, tall=1.8, wide=0.6):
    """The image box of a person standing at a court position."""
    foot = calibration.to_image(np.array(spot, dtype=float))
    scale = players._ground_scale(calibration, (float(foot[0]), float(foot[1])))
    return (
        float(foot[0] - wide * scale / 2),
        float(foot[1] - tall * scale),
        float(foot[0] + wide * scale / 2),
        float(foot[1]),
    )


class ScriptedDetector:
    """A `PersonDetector` that replays prepared detections, one list per call."""

    name = "scripted-v1"

    def __init__(self, script):
        self._script = list(script)
        self.shapes = []

    def detect(self, image):
        self.shapes.append(image.shape)
        return self._script.pop(0) if self._script else []


def scene(calibration, steps, confidence=0.9):
    """Builds frames and matching detections from people's court positions.

    Args:
        calibration: Places each person in the image.
        steps: One dict per frame: {name: (court position, colour)}.
        confidence: Detection confidence given to everyone.

    Returns:
        (frames, detections per frame). Each person is painted as a block
        of their colour, so appearance can tell them apart.
    """
    frames, script = [], []
    for number, people in enumerate(steps):
        image = np.full((SIZE[1], SIZE[0], 3), 90, dtype=np.uint8)
        found = []
        for spot, colour in people.values():
            box = person_box(calibration, spot)
            corner_one = (int(round(box[0])) - 1, int(round(box[1])) - 1)
            corner_two = (int(round(box[2])) + 1, int(round(box[3])) + 1)
            cv2.rectangle(image, corner_one, corner_two, colour, -1)
            found.append(PersonDetection(box, confidence))
        frames.append(VideoFrame(pts=number / RATE, image=image, index=number))
        script.append(found)
    return frames, script


def run(calibration, steps, config=None, **kwargs):
    """Tracks a scripted scene with the detector run on every frame."""
    config = config or PlayerConfig(stride=1)
    frames, script = scene(calibration, steps)
    detector = ScriptedDetector(script)
    tracks = players.track_players(frames, detector, calibration, config, crop=False, **kwargs)
    return tracks, detector


NEAR_SPOT = (1.0, 1.0)
FAR_SPOT = (-2.0, 22.0)


def standing(count, near=RED, far=BLUE, near_spot=NEAR_SPOT, far_spot=FAR_SPOT):
    """`count` frames of two players standing at their ends."""
    return [{"near": (near_spot, near), "far": (far_spot, far)} for _ in range(count)]


# ------------------------------------------------------------------ conventions


def test_side_and_zone_follow_the_documented_convention():
    assert players.side_of(3.0) == "near" and players.side_of(20.0) == "far"
    assert players.side_of(11.885) == "far"
    # From each player's own view, facing the net: deuce is their right.
    assert players.zone_of("near", 2.0) == "deuce" and players.zone_of("near", -2.0) == "ad"
    assert players.zone_of("far", -2.0) == "deuce" and players.zone_of("far", 2.0) == "ad"
    assert players.zone_of("near", 0.0) == "deuce" and players.zone_of("far", 0.0) == "ad"
    assert players.foot_of((10.0, 20.0, 30.0, 80.0)) == (20.0, 80.0)


# -------------------------------------------------------------------- detectors


class _Tensor:
    """Mimics a torch tensor's route to numpy."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _FakeYolo:
    """Mimics the one ultralytics call the wrapper makes."""

    def __init__(self, boxes, scores):
        self.calls = []
        self._boxes, self._scores = boxes, scores

    def predict(self, image, **options):
        self.calls.append((image.copy(), options))
        result = SimpleNamespace(
            boxes=SimpleNamespace(xyxy=_Tensor(self._boxes), conf=_Tensor(self._scores))
        )
        return [result]


def test_the_yolo_wrapper_asks_for_people_and_returns_plain_boxes():
    model = _FakeYolo([[10, 20, 30, 80], [100, 50, 140, 150]], [0.9, 0.4])
    detector = players.YoloPersonDetector(
        model, PlayerConfig(detector_size=640, min_confidence=0.2), name="fake-yolo@1"
    )
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, :, 0] = 200

    found = detector.detect(image)

    assert isinstance(detector, players.PersonDetector)
    assert found == [
        PersonDetection((10.0, 20.0, 30.0, 80.0), pytest.approx(0.9)),
        PersonDetection((100.0, 50.0, 140.0, 150.0), pytest.approx(0.4)),
    ]
    sent, options = model.calls[0]
    # RGB in, BGR to the model.
    assert sent[0, 0, 2] == 200 and sent[0, 0, 0] == 0
    assert options == {"classes": [0], "conf": 0.2, "imgsz": 640, "verbose": False}
    assert detector.name == "fake-yolo@1"


def test_the_yolo_wrapper_handles_a_frame_with_nobody_in_it():
    detector = players.YoloPersonDetector(_FakeYolo(np.zeros((0, 4)), np.zeros(0)))

    assert detector.detect(np.zeros((8, 8, 3), dtype=np.uint8)) == []


def test_the_real_model_is_loaded_lazily_and_once(monkeypatch):
    loads = []
    model = _FakeYolo([[1, 2, 3, 4]], [0.8])

    def fake_load(config):
        loads.append(config)
        return models.LoadedModel(models.PERSON_YOLO, model, "yolo11n@ultralytics-0.0", None)

    monkeypatch.setattr(players.models, "load_person_model", fake_load)
    settings = ModelConfig(person_model="yolo11n")
    detector = players.YoloPersonDetector(models_config=settings)

    assert loads == [] and detector.name == "yolo11n"
    detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))
    detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))

    assert loads == [settings]
    assert detector.name == "yolo11n@ultralytics-0.0"


def test_a_missing_backend_surfaces_only_when_detection_is_asked_for(monkeypatch, tmp_path):
    import sys

    monkeypatch.setitem(sys.modules, "ultralytics", None)
    detector = players.YoloPersonDetector(models_config=ModelConfig(weights_dir=tmp_path))

    with pytest.raises(models.BackendMissingError):
        detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))


def _moving_block(court_image, left, top=300, wide=40, tall=110):
    """The court with a dark, person-shaped block on it."""
    image = court_image.copy()
    cv2.rectangle(image, (left, top), (left + wide, top + tall), (20, 20, 20), -1)
    return image


def test_the_motion_fallback_finds_a_person_sized_mover(calibration, court_image):
    detector = players.MotionPersonDetector(calibration)

    assert isinstance(detector, players.PersonDetector)
    assert detector.detect(court_image) == []
    found = []
    for step in range(1, 5):
        found = detector.detect(_moving_block(court_image, 300 + 30 * step))

    assert detector.name == "motion-person-v1"
    centres = [((d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2) for d in found]
    assert any(410 < x < 470 and 320 < y < 400 for x, y in centres)
    assert all(0.3 <= d.confidence <= 0.9 for d in found)


def test_the_motion_fallback_sizes_blobs_in_metres_when_calibrated(calibration, court_image):
    def small_mover(detector):
        detector.detect(court_image)
        # A 20-pixel-tall thing low in the frame is about 0.3m: not a person.
        return detector.detect(_moving_block(court_image, 400, top=420, wide=12, tall=24))

    assert small_mover(players.MotionPersonDetector(calibration)) == []
    assert len(small_mover(players.MotionPersonDetector())) == 1


def test_the_motion_fallback_ignores_wide_and_tiny_blobs(court_image):
    detector = players.MotionPersonDetector()
    detector.detect(court_image)

    wide = detector.detect(_moving_block(court_image, 300, wide=160, tall=40))
    detector = players.MotionPersonDetector()
    detector.detect(court_image)
    tiny = detector.detect(_moving_block(court_image, 300, wide=3, tall=5))

    assert wide == [] and tiny == []


def test_the_motion_fallback_skips_a_blob_it_cannot_place(court_image):
    class Nowhere:
        def to_court(self, points):
            return np.array([np.nan, np.nan])

        def to_image(self, points):
            return np.full((2, 2), np.nan)

    detector = players.MotionPersonDetector(Nowhere())
    detector.detect(court_image)

    assert detector.detect(_moving_block(court_image, 300)) == []


# ------------------------------------------------------------------ association


def test_tracks_keep_their_numbers_as_people_move():
    associator = players.PersonAssociator()
    ids = []
    for step in range(6):
        found = [
            PersonDetection((100.0 + 8 * step, 300.0, 160.0 + 8 * step, 480.0), 0.9),
            PersonDetection((500.0 - 3 * step, 150.0, 515.0 - 3 * step, 190.0), 0.8),
        ]
        ids.append([person.track_id for person in associator.update(found, step / RATE)])

    assert all(pair == ids[0] for pair in ids)
    assert len(set(ids[0])) == 2


def test_a_weak_detection_continues_a_track_but_never_starts_one():
    associator = players.PersonAssociator(PlayerConfig(high_confidence=0.5, min_confidence=0.1))
    box = (100.0, 300.0, 160.0, 480.0)

    started = associator.update([PersonDetection(box, 0.9)], 0.0)
    continued = associator.update([PersonDetection(box, 0.2)], 0.1)
    stranger = associator.update(
        [PersonDetection(box, 0.9), PersonDetection((700.0, 100.0, 720.0, 160.0), 0.2)], 0.2
    )
    noise = associator.update([PersonDetection(box, 0.05)], 0.3)

    assert continued[0].track_id == started[0].track_id and continued[0].hits == 2
    assert [person.track_id for person in stranger] == [started[0].track_id]
    assert noise == []


def test_a_track_unseen_for_too_long_is_closed_and_its_number_retired():
    associator = players.PersonAssociator(PlayerConfig(max_track_age_seconds=1.0))
    box = (100.0, 300.0, 160.0, 480.0)

    first = associator.update([PersonDetection(box, 0.9)], 0.0)
    later = associator.update([PersonDetection(box, 0.9)], 5.0)

    assert later[0].track_id != first[0].track_id
    assert later[0].hits == 1


def test_the_near_and_far_player_do_not_trade_tracks_when_they_overlap():
    associator = players.PersonAssociator()
    near = (300.0, 200.0, 420.0, 520.0)
    far = (350.0, 230.0, 366.0, 270.0)
    first = associator.update([PersonDetection(near, 0.9), PersonDetection(far, 0.9)], 0.0)
    # Listed the other way round, and overlapping in the image.
    second = associator.update([PersonDetection(far, 0.9), PersonDetection(near, 0.9)], 0.1)

    by_box = {person.box: person.track_id for person in first}
    assert {person.box: person.track_id for person in second} == by_box


def test_two_detections_at_the_same_instant_do_not_divide_by_zero():
    associator = players.PersonAssociator()
    box = (100.0, 300.0, 160.0, 480.0)
    associator.update([PersonDetection(box, 0.9)], 1.0)

    again = associator.update([PersonDetection(box, 0.9)], 1.0)

    assert again[0].hits == 2


# -------------------------------------------------------- selection and identity


def test_exactly_the_two_players_of_this_court_are_kept(calibration):
    steps = []
    for number in range(12):
        steps.append(
            {
                "near": ((1.0 + 0.1 * number, 1.0), RED),
                "far": ((-2.0, 22.0 - 0.1 * number), BLUE),
                # Standing on the next court over, and walking past behind the far fence.
                "neighbour": ((12.5, 6.0), GREEN),
                "passer": ((-3.0 + 0.2 * number, 36.0), GREEN),
            }
        )

    tracks, _ = run(calibration, steps)
    quality = tracks.quality()

    assert len(tracks.near) == len(tracks.far) == 10  # min_track_hits=3 delays the start
    assert all(o.side == "near" and o.player == "A" for o in tracks.near)
    assert all(o.side == "far" and o.player == "B" for o in tracks.far)
    assert all(0.9 < o.foot_court[0] < 2.2 and abs(o.foot_court[1] - 1.0) < 0.05
               for o in tracks.near)
    assert all(abs(o.foot_court[0] + 2.0) < 0.05 for o in tracks.far)
    assert all(o.zone == "deuce" for o in tracks.near + tracks.far)
    assert all(o.provenance is Provenance.MEASURED for o in tracks.near)
    assert quality.rejected_people == 24
    assert quality.both_found == pytest.approx(10 / 12)
    assert quality.near_track_changes == quality.far_track_changes == 0
    assert quality.end_swaps == 0
    assert tracks.source == "scripted-v1"
    assert tracks.swaps == ()


def test_a_brief_false_person_on_court_is_not_taken_for_a_player(calibration):
    steps = [{"far": (FAR_SPOT, BLUE)} for _ in range(8)]
    steps[4]["ghost"] = ((0.0, 3.0), GREEN)
    steps[5]["ghost"] = ((0.0, 3.0), GREEN)

    tracks, _ = run(calibration, steps)

    assert tracks.near == ()
    assert tracks.quality().near_found == 0.0


def test_the_player_already_chosen_is_kept_over_a_newcomer(calibration):
    steps = standing(10)
    for number in range(4, 10):
        steps[number]["ballkid"] = ((0.0, 4.0), GREEN)

    tracks, _ = run(calibration, steps)

    assert tracks.quality().near_track_changes == 0
    assert all(abs(o.foot_court[1] - 1.0) < 0.05 for o in tracks.near)


def test_of_two_new_people_the_one_inside_the_court_is_chosen(calibration):
    steps = [
        {"inside": ((0.0, 3.0), RED), "outside": ((7.5, 3.0), GREEN), "far": (FAR_SPOT, BLUE)}
        for _ in range(6)
    ]

    tracks, _ = run(calibration, steps)

    assert all(abs(o.foot_court[0]) < 0.1 for o in tracks.near)


def test_feet_cut_off_by_the_frame_are_flagged(calibration):
    steps = standing(6, near_spot=(1.0, -2.6))

    tracks, _ = run(calibration, steps)

    assert tracks.near and all(o.foot_clipped for o in tracks.near)
    assert not any(o.foot_clipped for o in tracks.far)
    assert tracks.quality().near_clipped == 1.0


def test_a_net_rush_does_not_change_which_end_a_player_is_at(calibration):
    steps = [
        {"near": ((1.0, min(1.0 + 1.3 * number, 12.6)), RED), "far": (FAR_SPOT, BLUE)}
        for number in range(14)
    ]

    tracks, _ = run(calibration, steps)

    assert tracks.swaps == ()
    assert len(tracks.near) == 12
    assert tracks.near[-1].foot_court[1] > 11.885
    assert tracks.near[-1].side == "near"


def test_positions_between_detections_are_interpolated_and_tagged(calibration):
    steps = [
        {"near": ((1.0 + 0.2 * number, 1.0), RED), "far": (FAR_SPOT, BLUE)}
        for number in range(30)
    ]
    frames, script = scene(calibration, steps)
    detector = ScriptedDetector(script[::3])
    config = PlayerConfig(stride=3, max_interpolation_seconds=0.5)

    tracks = players.track_players(frames, detector, calibration, config, crop=False)

    assert len(detector.shapes) == 10
    assert [o.frame for o in tracks.near] == [6, 9, 12, 15, 18, 21, 24, 27]
    exact = tracks.at(0.9, "near")
    between = tracks.at(1.0, "near")
    assert exact.provenance is Provenance.MEASURED and exact.frame == 9
    assert between.provenance is Provenance.INTERPOLATED
    assert between.frame == 10
    assert between.foot_court[0] == pytest.approx(1.0 + 0.2 * 10, abs=0.05)
    assert between.confidence == pytest.approx(0.81)
    assert between.track_id == exact.track_id and between.player == "A"
    assert tracks.at(1.0, "near", frame=77).frame == 77
    assert tracks.at(0.1, "near") is None  # before the first detection
    assert tracks.at(9.0, "near") is None  # after the last
    assert len(tracks.boxes_at(1.0)) == 2 and tracks.boxes_at(50.0) == []


def test_a_long_gap_between_detections_is_not_bridged(calibration):
    steps = standing(4) + [{"far": (FAR_SPOT, BLUE)} for _ in range(12)] + standing(4)

    tracks, _ = run(calibration, steps, PlayerConfig(stride=1, max_interpolation_seconds=0.6))

    assert tracks.at(1.0, "near") is None
    assert tracks.at(1.0, "far") is not None
    assert tracks.quality().near_track_changes == 0  # same track: it only aged 1.3s
    with pytest.raises(ValueError, match="near.*far"):
        tracks.at(1.0, "left")


def test_interpolating_across_two_different_tracks_claims_no_track(calibration):
    gap = [{"far": (FAR_SPOT, BLUE)} for _ in range(20)]
    steps = standing(5) + gap + standing(5)
    config = PlayerConfig(stride=1, max_interpolation_seconds=5.0, swap_absence_seconds=60.0)

    tracks, _ = run(calibration, steps, config)

    bridged = tracks.at(1.5, "near")
    assert bridged.provenance is Provenance.INTERPOLATED and bridged.track_id is None
    assert tracks.quality().near_track_changes == 1


def test_frames_without_an_index_are_numbered_as_they_come(calibration):
    frames, script = scene(calibration, standing(5))
    frames = [VideoFrame(pts=f.pts, image=f.image) for f in frames]

    tracks = players.track_players(
        frames, ScriptedDetector(script), calibration, PlayerConfig(stride=1), crop=False
    )

    assert [o.frame for o in tracks.near] == [2, 3, 4]


# -------------------------------------------------------------------- end swaps


def _walk(count, start, end):
    """`count` positions walking from one court spot to another."""
    start, end = np.array(start), np.array(end)
    return [tuple(start + (end - start) * k / (count - 1)) for k in range(count)]


# An arrangement counts as settled after one second here (ten frames), so
# the scripted scenes can stay short. The default is several seconds.
QUICK = PlayerConfig(stride=1, swap_settle_seconds=1.0)


def test_a_change_of_ends_followed_by_the_tracker_is_one_measured_event(calibration):
    one = _walk(45, (-7.0, 2.0), (-7.0, 22.0))
    two = _walk(45, (7.0, 22.0), (7.0, 2.0))
    steps = standing(4, near_spot=one[0], far_spot=two[0])
    steps += [{"near": (a, RED), "far": (b, BLUE)} for a, b in zip(one, two)]
    steps += standing(4, near=BLUE, far=RED, near_spot=two[-1], far_spot=one[-1])

    tracks, _ = run(calibration, steps, QUICK)

    assert len(tracks.swaps) == 1
    swap = tracks.swaps[0]
    assert swap.provenance is Provenance.MEASURED and "followed" in swap.reason
    # Dated to when the new arrangement began, not to when it was judged.
    assert 2.0 < swap.pts < 3.5
    assert swap.last_settled_pts < swap.pts
    assert tracks.near[0].player == "A" and tracks.far[0].player == "B"
    assert tracks.near[-1].player == "B" and tracks.far[-1].player == "A"
    assert tracks.player_at(0.5, "near") == "A" and tracks.player_at(5.0, "near") == "B"
    assert tracks.player_at(5.0, "far") == "A"
    # The same association track carried the person across.
    assert tracks.near[0].track_id == tracks.far[-1].track_id
    assert tracks.quality().end_swaps == 1


def test_wandering_across_the_net_and_back_is_not_a_swap(calibration):
    # What the real recording showed at a changeover: a player crosses the
    # net (to the bench, to fetch a ball) and comes back. Nothing changed.
    out = _walk(45, (-7.0, 2.0), (-7.0, 22.0))
    steps = standing(20, near_spot=out[0])
    steps += [{"near": (spot, RED)} for spot in out + out[::-1]]
    steps += standing(20, near_spot=out[0])

    tracks, _ = run(calibration, steps, QUICK)

    assert tracks.swaps == ()
    # The wanderer really was seen at the far end for a while...
    assert any(o.track_id == tracks.near[0].track_id for o in tracks.far)
    # ...and is still player A, back at the near end.
    assert tracks.near[-1].player == "A" and tracks.far[-1].player == "B"


def test_an_arrangement_that_has_not_settled_is_not_judged(calibration):
    one = _walk(45, (-7.0, 2.0), (-7.0, 22.0))
    two = _walk(45, (7.0, 22.0), (7.0, 2.0))
    steps = standing(20, near_spot=one[0], far_spot=two[0])
    steps += [{"near": (a, RED), "far": (b, BLUE)} for a, b in zip(one, two)]

    # With the default settle time the stream ends before the new
    # arrangement has held long enough: no swap is claimed.
    tracks, _ = run(calibration, steps, PlayerConfig(stride=1, swap_settle_seconds=6.0))

    assert tracks.swaps == ()
    assert PlayerConfig().swap_settle_seconds >= 5.0


def _break(seconds, before, after):
    """Both players seen, nobody for `seconds`, then both seen again."""
    empty = [{} for _ in range(int(seconds * RATE))]
    return before + empty + after


def test_a_swap_during_a_break_is_inferred_from_what_the_players_look_like(calibration):
    steps = _break(7.0, standing(20), standing(20, near=BLUE, far=RED))

    tracks, _ = run(calibration, steps, QUICK)

    assert len(tracks.swaps) == 1
    swap = tracks.swaps[0]
    assert swap.provenance is Provenance.INFERRED
    assert 0.5 < swap.confidence <= 1.0
    assert "look like" in swap.reason
    # It applies from the first sighting after the break, not from when it was decided.
    assert swap.pts == pytest.approx(tracks.near[20 - 2].pts)
    assert swap.last_settled_pts == pytest.approx(1.9)
    assert all(o.player == "A" for o in tracks.near if o.pts < 5.0)
    assert all(o.player == "B" for o in tracks.near if o.pts > 8.0)
    assert all(o.player == "A" for o in tracks.far if o.pts > 8.0)
    assert json.loads(json.dumps(swap.to_dict()))["provenance"] == "inferred"
    assert swap.to_dict()["last_settled_pts"] == swap.last_settled_pts


def test_a_break_with_no_change_of_ends_records_nothing(calibration):
    tracks, _ = run(calibration, _break(7.0, standing(20), standing(20)), QUICK)

    assert tracks.swaps == ()
    assert {o.player for o in tracks.near} == {"A"}


def test_two_swaps_bring_the_labels_back(calibration):
    steps = _break(7.0, standing(20), standing(20, near=BLUE, far=RED))
    steps = _break(7.0, steps, standing(20))

    tracks, _ = run(calibration, steps, QUICK)

    assert len(tracks.swaps) == 2
    assert tracks.near[-1].player == "A" and tracks.far[-1].player == "B"
    assert tracks.player_at(10.0, "near") == "B"


def test_one_end_emptying_and_refilling_is_not_a_swap(calibration):
    away = [{"near": (NEAR_SPOT, RED)} for _ in range(70)]
    steps = standing(20) + away + standing(20)

    tracks, _ = run(calibration, steps, QUICK)

    assert tracks.swaps == ()
    assert tracks.far[-1].player == "B"


def test_a_short_absence_is_not_even_examined(calibration):
    # Swapped colours, but only two seconds away: players do not change ends mid-rally.
    steps = _break(2.0, standing(20), standing(20, near=BLUE, far=RED))

    tracks, _ = run(calibration, steps, QUICK)

    assert tracks.swaps == ()


def test_without_an_earlier_look_at_both_players_nothing_can_be_inferred(calibration):
    alone = [{"near": (NEAR_SPOT, RED)} for _ in range(10)]
    steps = alone + [{} for _ in range(70)] + standing(20, near=BLUE, far=RED)

    tracks, _ = run(calibration, steps, QUICK)

    assert tracks.swaps == ()


def test_a_followed_swap_of_players_dressed_alike_is_measured_but_less_sure(calibration):
    one = _walk(45, (-7.0, 2.0), (-7.0, 22.0))
    two = _walk(45, (7.0, 22.0), (7.0, 2.0))
    steps = standing(4, far=RED, near_spot=one[0], far_spot=two[0])
    steps += [{"near": (a, RED), "far": (b, RED)} for a, b in zip(one, two)]

    tracks, _ = run(calibration, steps, QUICK)

    assert len(tracks.swaps) == 1
    swap = tracks.swaps[0]
    assert swap.provenance is Provenance.MEASURED
    assert swap.confidence == pytest.approx(0.6)
    assert "could not confirm" in swap.reason


def _judged(calibration, ids, looks, away=60.0):
    """Judges a hand-built arrangement against a settled red-near, blue-far one.

    Args:
        calibration: Any calibration; the tracker is never fed a frame.
        ids: {"near": set, "far": set} of tracks in the new arrangement;
            before it, track 1 was near and track 2 far.
        looks: (near colour, far colour) of the new arrangement.
        away: Seconds since the old arrangement was last confirmed.

    Returns:
        The tracker, after judging.
    """
    def look(colour):
        image = np.full((40, 40, 3), colour, dtype=np.uint8)
        return players.appearance(image, (0.0, 0.0, 40.0, 40.0))

    tracker = players.PlayerTracker(ScriptedDetector([]), calibration, QUICK, crop=False)
    tracker._settled_ids = {"near": {1}, "far": {2}}
    tracker._settled_last = 10.0
    tracker._known = {"near": look(RED), "far": look(BLUE)}
    arrangement = players._Run(
        pts=10.0 + away,
        frame=700,
        ids=ids,
        looks={"near": look(looks[0]), "far": look(looks[1])},
    )
    tracker._judge(arrangement)
    return tracker


def test_a_track_that_hopped_between_players_does_not_hide_a_swap(calibration):
    # The real recording's first changeover: the players passed at the bench,
    # the far player's track number ended up on the other player, and so "the
    # same track is still at the far end" - yet the people had changed ends.
    tracker = _judged(calibration, {"near": {16}, "far": {2}}, (BLUE, RED))

    (swap,) = tracker.finish().swaps
    assert swap.provenance is Provenance.INFERRED
    assert "tracks did not show it" in swap.reason
    assert swap.pts == 70.0 and swap.last_settled_pts == 10.0


def test_a_track_changing_ends_while_the_players_look_the_same_is_a_tracking_slip(
    calibration, caplog
):
    with caplog.at_level(logging.WARNING, logger="tracking.players"):
        tracker = _judged(calibration, {"near": {2}, "far": {1}}, (RED, BLUE))

    assert tracker.finish().swaps == ()
    assert "tracking slip" in caplog.text


def test_tracks_that_disagree_leave_the_decision_to_appearance(calibration):
    # Track 1 is still near, yet track 2 has turned up there as well.
    muddled = {"near": {1, 2}, "far": {3}}

    assert _judged(calibration, muddled, (RED, BLUE)).finish().swaps == ()
    assert len(_judged(calibration, muddled, (BLUE, RED)).finish().swaps) == 1
    # ...but never on appearance alone when hardly any time has passed.
    assert _judged(calibration, muddled, (BLUE, RED), away=2.0).finish().swaps == ()


def test_a_player_whose_look_cannot_be_taken_never_settles_an_arrangement(
    calibration, monkeypatch
):
    monkeypatch.setattr(players, "appearance", lambda *args, **kwargs: None)

    tracks, _ = run(calibration, _break(7.0, standing(20), standing(20)), QUICK)

    assert tracks.swaps == ()
    assert len(tracks.near) > 30


def test_a_box_with_no_pixels_has_no_appearance(calibration):
    image = np.zeros((40, 60, 3), dtype=np.uint8)

    assert players.appearance(image, (100.0, 100.0, 120.0, 160.0)) is None
    assert players.appearance(image, (10.0, 10.0, 10.0, 10.0)) is None
    look = players.appearance(image, (10.0, 5.0, 30.0, 35.0))
    assert look.sum() == pytest.approx(1.0)
    assert players.similarity(look, look) == pytest.approx(1.0)


# ------------------------------------------------------------- region and output


def test_the_detector_is_given_only_the_part_of_the_frame_players_can_be_in(calibration):
    region = players.player_region(calibration)
    frames, script = scene(calibration, standing(5))
    top = int(region[1])
    shifted = [
        [PersonDetection((d.box[0], d.box[1] - top, d.box[2], d.box[3] - top), 0.9) for d in found]
        for found in script
    ]
    detector = ScriptedDetector(shifted)

    tracks = players.track_players(frames, detector, calibration, PlayerConfig(stride=1))

    assert 0 < top < SIZE[1] / 2
    assert region == (0.0, float(top), float(SIZE[0]), float(SIZE[1]))
    assert detector.shapes[0] == (SIZE[1] - top, SIZE[0], 3)
    # Boxes come back in full-frame coordinates.
    assert tracks.near[0].foot_court == pytest.approx(NEAR_SPOT, abs=0.05)


def test_the_region_is_a_tight_box_when_everything_is_in_front_of_the_camera(calibration):
    # A region narrower than the court, so that its sides are inside the frame.
    left, top, right, bottom = players.player_region(
        calibration, PlayerConfig(behind_margin_m=0.0, lateral_margin_m=-2.0)
    )

    assert 0 < left < right < SIZE[0]
    assert 0 < top < bottom <= SIZE[1]


def test_the_region_is_the_whole_frame_when_the_camera_cannot_be_recovered(
    calibration, monkeypatch
):
    whole = (0.0, 0.0, float(SIZE[0]), float(SIZE[1]))
    monkeypatch.setattr(players, "camera_from_homography", lambda *_: None)
    assert players.player_region(calibration) == whole

    behind = np.hstack([np.eye(3), [[0.0], [0.0], [-1000.0]]])
    monkeypatch.setattr(players, "camera_from_homography", lambda *_: behind)
    assert players.player_region(calibration) == whole

    speck = np.array([[0, 0, 0, 5.0], [0, 0, 0, 5.0], [0, 0, 0, 1.0]])
    monkeypatch.setattr(players, "camera_from_homography", lambda *_: speck)
    assert players.player_region(calibration) == whole


def test_tracks_and_quality_serialise_to_json(calibration):
    tracks, _ = run(calibration, standing(8))

    payload = json.loads(json.dumps(tracks.to_dict()))

    assert payload["source"] == "scripted-v1"
    assert payload["near"][0]["side"] == "near" and payload["near"][0]["provenance"] == "measured"
    assert payload["far"][0]["zone"] == "deuce"
    assert payload["quality"]["both_found"] == pytest.approx(6 / 8)
    assert payload["swaps"] == []
    assert tracks.quality().mean_confidence == pytest.approx(0.9)


def test_an_empty_stream_gives_empty_tracks(calibration):
    tracks = players.track_players([], ScriptedDetector([]), calibration)
    quality = tracks.quality()

    assert tracks.near == () and tracks.far == ()
    assert quality.detector_frames == 0 and quality.both_found == 0.0
    assert quality.mean_confidence == 0.0 and quality.near_clipped == 0.0
    assert tracks.at(0.0, "near") is None


def test_a_person_whose_feet_map_nowhere_is_rejected(calibration, monkeypatch):
    frames, script = scene(calibration, standing(4))
    tracker = players.PlayerTracker(
        ScriptedDetector(script), calibration, PlayerConfig(stride=1), crop=False
    )
    monkeypatch.setattr(
        type(calibration), "to_court", lambda self, points: np.array([np.nan, np.nan])
    )
    for frame in frames:
        tracker.update(frame)

    tracks = tracker.finish()

    assert tracks.near == () and tracks.rejected_people == 8
