"""Tests for chunk planning, the compact ball track and stitching chunks back.

Also covers what the pipeline needed from earlier phases: carrying a
player tracker across chunks (`PlayerTracker.snapshot` / `resume`),
labelling observations gathered in pieces, and reading tagged values and
calibrations back from JSON.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from tracking import assemble, calibrate, players, stitch
from tracking.ball import BallObservation, BallSegment, BallTrack, SceneStats, Visibility
from tracking.config import PipelineConfig, PlayerConfig
from tracking.events import EventTrack, join_shots
from tracking.players import FAR, NEAR, EndSwap
from tracking.provenance import Provenance, Tagged
from tracking.stitch import ChunkResult, CompactBallTrack

from . import points as P
from .conftest import SIZE
from .test_players import BLUE, QUICK, RED, ScriptedDetector, _walk, scene, standing

# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------


def test_a_run_is_cut_into_chunks_with_overlap_clipped_to_the_run():
    windows = stitch.plan_chunks(10.0, 320.0, PipelineConfig(chunk_seconds=100.0))
    assert [(w.start, w.end) for w in windows] == [
        (10.0, 110.0), (110.0, 210.0), (210.0, 310.0), (310.0, 320.0)
    ]
    assert windows[0].read_start == 10.0 and windows[0].read_end == 118.0
    assert windows[1].read_start == 102.0
    assert windows[-1].read_end == 320.0 and windows[-1].last
    assert not any(w.last for w in windows[:-1])


def test_every_moment_belongs_to_exactly_one_chunk():
    windows = stitch.plan_chunks(0.0, 30.0, PipelineConfig(chunk_seconds=10.0))
    for moment in (0.0, 9.999, 10.0, 25.0, 30.0):
        assert sum(w.owns(moment) for w in windows) == 1
    assert not any(w.owns(30.01) or w.owns(-0.01) for w in windows)
    first = windows[0]
    assert first.keeps(10.5, 1.0) and not first.keeps(11.0, 1.0)
    assert not first.keeps(-0.5, 1.0)  # before anything was read
    assert json.loads(json.dumps(first.to_dict()))["read_end"] == 18.0


def test_a_short_run_is_one_chunk():
    (only,) = stitch.plan_chunks(0.0, 5.0)
    assert only.last and only.owns(5.0) and only.read_end == 5.0


@pytest.mark.parametrize(
    "start, end, config, message",
    [
        (5.0, 5.0, PipelineConfig(), "must end after"),
        (0.0, 5.0, PipelineConfig(chunk_seconds=0.0), "chunk_seconds"),
        (0.0, 5.0, PipelineConfig(overlap_seconds=1.0), "overlap_seconds"),
    ],
)
def test_impossible_plans_are_refused(start, end, config, message):
    with pytest.raises(ValueError, match=message):
        stitch.plan_chunks(start, end, config)


# --------------------------------------------------------------------------
# The compact ball track answers exactly as the real one
# --------------------------------------------------------------------------


def _track(seed=3, frames=400, fps=60.0):
    """A ball track with measured runs, bridged frames and gaps of every length."""
    rng = np.random.default_rng(seed)
    times = np.arange(frames) / fps + 100.0
    state = np.zeros(frames, dtype=int)
    k = 0
    segment = -1
    segments = []
    observations = []
    while k < frames:
        if rng.random() < 0.6:
            length = int(rng.integers(5, 40))
            segment += 1
            first = k
            for j in range(k, min(frames, k + length)):
                bridged = rng.random() < 0.15
                state[j] = 2 if bridged else 1
                observations.append(
                    BallObservation(
                        frame=1000 + j,
                        pts=float(times[j]),
                        image_xy=(float(100 + 7 * j), float(300 - 3 * j + rng.normal())),
                        ground_ray_xy=None,
                        velocity=(1.0, 1.0),
                        confidence=float(rng.uniform(0.1, 0.9)),
                        visibility=Visibility.BRIDGED if bridged else Visibility.VISIBLE,
                        provenance=Provenance.INTERPOLATED if bridged else Provenance.MEASURED,
                        segment=segment,
                    )
                )
            k = min(frames, k + length)
            segments.append(BallSegment(segment, 1000 + first, 1000 + k - 1, 0, 0, 1, 0, 1.0, 0.0))
        else:
            k += int(rng.integers(1, 700))
    gaps = []
    j = 0
    while j < frames:
        if state[j]:
            j += 1
            continue
        end = j
        while end + 1 < frames and not state[end + 1]:
            end += 1
        from tracking.ball import BallGap

        gaps.append(BallGap(1000 + j, 1000 + end, float(times[j]), float(times[end]), end - j + 1))
        j = end + 1
    scene = SceneStats(5, 100.0, 10.0, (1.0, 2.0, 3.0), -5.0)
    return BallTrack(
        tuple(observations), tuple(segments), tuple(gaps), 1000 + np.arange(frames), times, SIZE,
        "test-ball", scene,
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
@pytest.mark.parametrize("stretch", [(None, None), (100.5, 102.0), (101.0, 106.0), (90.0, 99.0)])
def test_the_compact_track_reports_the_same_quality_as_the_real_one(seed, stretch):
    track = _track(seed)
    compact = CompactBallTrack.from_track(track)
    assert compact.quality(*stretch) == track.quality(*stretch)
    assert compact.frame_pts is not track.frame_pts and compact.scene == track.scene


def test_chunks_of_a_track_join_back_into_the_whole():
    track = _track(5)
    cut = 102.0
    parts = [
        CompactBallTrack.from_track(track, keep=lambda pts: pts < cut),
        CompactBallTrack.from_track(track, keep=lambda pts: pts >= cut),
    ]
    joined = CompactBallTrack.concatenate(parts)
    assert joined.quality() == track.quality()
    assert joined.quality(101.0, 104.0) == track.quality(101.0, 104.0)
    with pytest.raises(ValueError):
        CompactBallTrack.concatenate([])


def test_segment_numbers_from_different_chunks_never_collide():
    track = _track(6)
    one = CompactBallTrack.from_track(track, segment_offset=0)
    two = CompactBallTrack.from_track(track, segment_offset=10_000)
    joined = CompactBallTrack.concatenate([one, two])
    assert joined.quality().segments == 2 * track.quality().segments


def test_an_observation_off_the_frame_list_is_ignored_and_empty_is_empty():
    track = _track(5)
    assert track.observations
    stray = replace(track.observations[0], pts=5000.0)
    odd = replace(track, observations=track.observations + (stray,))
    assert CompactBallTrack.from_track(odd).quality() == track.quality()
    empty = BallTrack((), (), (), np.zeros(0, dtype=int), np.zeros(0), SIZE, "none")
    compact = CompactBallTrack.from_track(empty)
    assert compact.quality().frames == 0 and compact.gaps() == []
    assert compact.coverage_over([(0.0, 1.0)]) == (0, 0, 0.0)


def test_coverage_over_several_stretches_counts_each_frame_once():
    track = _track(8)
    compact = CompactBallTrack.from_track(track)
    frames, seen, confidence = compact.coverage_over([(100.0, 101.0), (100.5, 102.0)])
    whole = track.quality(100.0, 102.0)
    assert frames == whole.frames and seen == whole.measured + whole.bridged
    assert confidence == pytest.approx(whole.mean_confidence)


def test_coherence_with_no_close_measured_steps_is_zero():
    sparse = CompactBallTrack(
        frame_numbers=np.arange(3),
        frame_pts=np.array([0.0, 1.0, 2.0]),
        state=np.array([1, 1, 0], dtype=np.int8),
        confidence=np.array([0.5, 0.5, 0.0]),
        image_xy=np.array([[0.0, 0.0], [5.0, 5.0], [np.nan, np.nan]]),
        segment=np.array([0, 0, -1]),
        image_size=SIZE,
        source="test",
    )
    found = sparse.quality()
    assert found.coherence == 0.0 and found.measured == 2 and found.longest_gap_seconds == 1.0


# --------------------------------------------------------------------------
# Stitching events across chunk edges
# --------------------------------------------------------------------------


def _chunk(window, hits=(), bounces=(), nets=(), **more):
    """A chunk result carrying only events (and empty everything else)."""
    empty = BallTrack((), (), (), np.zeros(0, dtype=int), np.zeros(0), SIZE, "test")
    fields = dict(
        window=window,
        hits=tuple(hits),
        bounces=tuple(bounces),
        nets=tuple(nets),
        ball=CompactBallTrack.from_track(empty),
        near=(),
        far=(),
        detector_times=(),
        rejected_people=0,
        swaps=(),
        poses={},
        scene=None,
        discarded_kinks=1,
        drift_px=None,
        drift_checked=False,
        seconds=1.0,
        tracker_state=b"",
        detector_name="test-people",
    )
    fields.update(more)
    return ChunkResult(**fields)


WINDOWS = stitch.plan_chunks(0.0, 20.0, PipelineConfig(chunk_seconds=10.0, overlap_seconds=3.0))


def _times(events):
    return [round(e.pts, 3) for e in events]


def test_an_event_seen_by_both_chunks_is_kept_once_whatever_each_timed_it():
    # The copy kept is the one from the chunk owning the pair's mean time;
    # a mean exactly on the edge belongs to the later chunk.
    cases = [(9.99, 10.01, 10.01), (10.01, 9.99, 9.99), (9.95, 9.97, 9.95), (10.02, 10.03, 10.03)]
    for left, right, kept in cases:
        chunks = [
            _chunk(WINDOWS[0], hits=[P.hit(left, NEAR)]),
            _chunk(WINDOWS[1], hits=[P.hit(right, NEAR)]),
        ]
        hits, _, _, report = stitch.stitch_events(chunks, PipelineConfig(overlap_seconds=3.0))
        assert _times(hits) == [kept], (left, right)
        assert report.matched == 1


def test_an_event_only_the_owner_saw_is_kept_and_only_the_other_saw_is_dropped():
    chunks = [
        _chunk(WINDOWS[0], hits=[P.hit(9.9, NEAR), P.hit(10.3, FAR)]),
        _chunk(WINDOWS[1], hits=[P.hit(9.6, FAR), P.hit(10.5, NEAR)]),
    ]
    hits, _, _, report = stitch.stitch_events(chunks)
    assert _times(hits) == [9.9, 10.5]
    assert report.matched == 0 and report.dropped_unowned == 2


def test_hits_at_different_ends_are_never_the_same_hit():
    chunks = [
        _chunk(WINDOWS[0], hits=[P.hit(9.99, NEAR)]),
        _chunk(WINDOWS[1], hits=[P.hit(10.01, FAR)]),
    ]
    hits, _, _, report = stitch.stitch_events(chunks)
    assert [(h.side, round(h.pts, 2)) for h in hits] == [(NEAR, 9.99), (FAR, 10.01)]
    assert report.matched == 0


def test_bounces_and_net_contacts_are_stitched_the_same_way():
    chunks = [
        _chunk(WINDOWS[0], bounces=[P.bounce(10.01, 1.0, 20.0)], nets=[P.net(9.98)]),
        _chunk(WINDOWS[1], bounces=[P.bounce(9.99, 1.0, 20.0)], nets=[P.net(10.02)]),
    ]
    _, bounces, nets, report = stitch.stitch_events(chunks)
    assert len(bounces) == 1 and len(nets) == 1 and report.matched == 2
    assert report.kept_from_neighbour == 1


def test_one_event_is_matched_with_at_most_one_other():
    chunks = [
        _chunk(WINDOWS[0], hits=[P.hit(9.95, NEAR), P.hit(10.06, NEAR)]),
        _chunk(WINDOWS[1], hits=[P.hit(10.0, NEAR)]),
    ]
    hits, _, _, report = stitch.stitch_events(chunks)
    assert report.matched == 1
    assert _times(hits) == [9.95]  # 10.0 and 9.95 paired; 10.06 left unowned


def test_far_from_an_edge_each_chunk_keeps_only_its_own():
    chunks = [
        _chunk(WINDOWS[0], hits=[P.hit(5.0, NEAR), P.hit(14.0, FAR)]),
        _chunk(WINDOWS[1], hits=[P.hit(5.0, NEAR), P.hit(14.0, FAR)]),
    ]
    hits, _, _, _ = stitch.stitch_events(chunks)
    assert _times(hits) == [5.0, 14.0]


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------


def _observation(pts, side):
    """An unlabelled detection."""
    return players.PlayerObservation(
        frame=int(pts * 10), pts=pts, side=side, player="", box=(0, 0, 1, 1),
        foot_image=(0, 0), foot_court=(0.0, 1.0 if side == NEAR else 23.0), zone="deuce",
        confidence=0.9, provenance=Provenance.MEASURED, track_id=1,
    )


def test_merging_relabels_hits_with_a_change_of_ends_judged_in_a_later_chunk():
    swap = EndSwap(9.5, 95, Provenance.MEASURED, 0.9, "followed", 8.0)
    first = _chunk(
        WINDOWS[0],
        hits=[P.hit(3.0, NEAR), P.hit(9.7, NEAR)],
        near=(_observation(3.0, NEAR),),
        detector_times=(3.0,),
        rejected_people=2,
        poses={stitch.pose_key(3.0): ("pose",)},
        scene=SceneStats(10, 100.0, 5.0, (10.0, 20.0, 30.0), -1.0),
    )
    second = _chunk(
        WINDOWS[1],
        hits=[P.hit(12.0, FAR, 1.0)],
        far=(_observation(12.0, FAR),),
        detector_times=(12.0,),
        swaps=(swap,),
        rejected_people=1,
        scene=SceneStats(30, 200.0, 5.0, (30.0, 40.0, 50.0), -3.0),
    )
    merged = stitch.merge_chunks([first, second])
    assert [h.player for h in merged.events.hits] == ["A", "B", "A"]
    assert [o.player for o in merged.players.near] == ["A"]
    assert [o.player for o in merged.players.far] == ["A"]
    assert merged.players.rejected_people == 3 and merged.players.source == "test-people"
    assert merged.poses == {0: ("pose",)}
    assert merged.events.discarded_kinks == 2
    assert len(merged.events.shots) == 3
    assert merged.scene.frames_sampled == 40 and merged.scene.brightness == pytest.approx(175.0)
    assert merged.scene.court_rgb == pytest.approx((25.0, 35.0, 45.0))
    assert merged.events.ball.scene == merged.scene


def test_merging_needs_the_whole_run_in_order():
    with pytest.raises(ValueError, match="No chunks"):
        stitch.merge_chunks([])
    with pytest.raises(ValueError, match="whole run"):
        stitch.merge_chunks([_chunk(WINDOWS[0])])
    with pytest.raises(ValueError, match="whole run"):
        stitch.merge_chunks([_chunk(WINDOWS[1]), _chunk(WINDOWS[0])])


def test_scenes_are_merged_only_from_chunks_that_sampled_any():
    assert stitch.merge_scenes([None]) is None
    empty = SceneStats(0, 1.0, 1.0, (1.0, 1.0, 1.0), 1.0)
    assert stitch.merge_scenes([empty, None]) is None
    report = stitch.StitchReport(chunks=2, matched=1)
    assert json.loads(json.dumps(report.to_dict()))["matched"] == 1


def test_join_shots_is_what_detect_events_joins_with():
    track, _ = P.build(P.rally(NEAR, "deuce", 3, end="out"))
    shots, kept = join_shots(track.hits, track.bounces, track.nets)
    assert [s.result.value for s in shots] == [s.result.value for s in track.shots]
    assert kept == list(track.bounces)
    assert isinstance(EventTrack(track.hits, tuple(kept), (), tuple(shots), track.ball), EventTrack)


# --------------------------------------------------------------------------
# A player tracker carried across chunks
# --------------------------------------------------------------------------


def _swap_scene():
    """Two players changing ends, as in the players tests."""
    one = _walk(45, (-7.0, 2.0), (-7.0, 22.0))
    two = _walk(45, (7.0, 22.0), (7.0, 2.0))
    steps = standing(4, near_spot=one[0], far_spot=two[0])
    steps += [{"near": (a, RED), "far": (b, BLUE)} for a, b in zip(one, two)]
    steps += standing(20, near=BLUE, far=RED, near_spot=two[-1], far_spot=one[-1])
    return steps


@pytest.mark.parametrize("cut", [10, 30, 50])
def test_a_tracker_resumed_from_a_snapshot_carries_on_exactly(calibration, cut):
    frames, script = scene(calibration, _swap_scene())
    whole = players.PlayerTracker(ScriptedDetector(list(script)), calibration, QUICK, crop=False)
    for frame in frames:
        whole.update(frame)
    expected = whole.finish()

    detector = ScriptedDetector(list(script))
    first = players.PlayerTracker(detector, calibration, QUICK, crop=False)
    for frame in frames[:cut]:
        first.update(frame)
    taken = first.take_observations()
    resumed = players.PlayerTracker.resume(first.snapshot(), detector)
    for frame in frames[cut:]:
        resumed.update(frame)
    rest = resumed.take_observations()
    assert resumed.swaps == expected.swaps and resumed.detector_name == "scripted-v1"
    merged = players.label_tracks(
        taken.near + rest.near,
        taken.far + rest.far,
        resumed.swaps,
        taken.detector_times + rest.detector_times,
        "scripted-v1",
        QUICK,
        taken.rejected_people + rest.rejected_people,
    )
    assert merged.near == expected.near and merged.far == expected.far
    assert merged.detector_times == expected.detector_times
    assert merged.quality() == expected.quality()


def test_a_stateful_detector_is_kept_in_the_snapshot_and_a_stateless_one_is_not(
    calibration,
):
    motion = players.MotionPersonDetector(calibration)
    tracker = players.PlayerTracker(motion, calibration)
    restored = players.PlayerTracker.resume(tracker.snapshot())
    assert restored.detector_name == "motion-person-v1"
    stateless = players.PlayerTracker(ScriptedDetector([]), calibration)
    with pytest.raises(ValueError, match="kept no detector"):
        players.PlayerTracker.resume(stateless.snapshot())
    import pickle

    with pytest.raises(TypeError):
        players.PlayerTracker.resume(pickle.dumps({"not": "a tracker"}))


def test_observations_taken_out_are_forgotten(calibration):
    frames, script = scene(calibration, standing(6))
    tracker = players.PlayerTracker(ScriptedDetector(script), calibration, PlayerConfig(stride=1))
    for frame in frames:
        tracker.update(frame)
    taken = tracker.take_observations()
    assert len(taken.near) == 4 and all(o.player == "" for o in taken.near)
    assert tracker.take_observations().near == ()
    assert tracker.finish().near == ()


# --------------------------------------------------------------------------
# Reading back from JSON
# --------------------------------------------------------------------------


def test_tagged_values_read_back_from_their_dicts():
    tagged = Tagged((1.5, 2.5), Provenance.INTERPOLATED, 0.4, "src")
    back = Tagged.from_dict(json.loads(json.dumps(tagged.to_dict())))
    assert back == tagged
    assert Tagged.from_dict({"value": "In", "provenance": "inferred"}) == Tagged(
        "In", Provenance.INFERRED
    )
    with pytest.raises(ValueError):
        Tagged.from_dict({"value": 1, "provenance": "guessed"})


def test_a_calibration_reads_back_from_its_dict(calibration):
    back = calibrate.Calibration.from_dict(json.loads(json.dumps(calibration.to_dict())))
    assert np.allclose(back.homography, calibration.homography)
    assert np.allclose(back.inverse, calibration.inverse)
    assert back.quality == calibration.quality and back.provenance is calibration.provenance
    assert back.image_size == calibration.image_size
    detected = replace(calibration.quality, lines_found=("a", "b"))
    assert calibrate.CalibrationQuality.from_dict(detected.to_dict()) == detected


def test_an_attempt_block_with_a_folded_fault_round_trips():
    from swingvision_import import reconstruct

    from .test_assemble import golden

    match = golden()
    result = reconstruct.reconstruct_all(match.export.shots, "Host")
    blocks = match.point_blocks(result)
    folded = next(block for block in blocks.values() if "folded_fault" in block)
    stored = json.loads(json.dumps(folded))
    back = assemble.AttemptBlock.from_dict(stored)
    assert back.folded_fault is not None and back.folded_fault.fault
    assert back.to_dict() == stored
