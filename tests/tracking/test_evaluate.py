"""Tests for scoring tracker output against a reference export and against review."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from swingvision_import import reconstruct
from swingvision_import.raw import RawShotRow
from swingvision_import.records import MatchRecord
from tracking import evaluate

OFFSET = 2.0


def _row(point, shot, t, player, side, stroke, shot_type, result="In", bounce=None, speed=None):
    return RawShotRow(
        point_number=point,
        shot_number=shot,
        player=player,
        shot_type=shot_type,
        stroke=stroke,
        result=result,
        video_time=t,
        hit_side=side,
        bounce_x=None if bounce is None else bounce[0],
        bounce_y=None if bounce is None else bounce[1],
        speed_mph=speed,
    )


def _reference():
    """Three points, on SwingVision's clock (OFFSET behind the video)."""
    r = []
    # Point 1: host serves from near, three shots, host's forehand wins.
    r += [
        _row(1, 1, 8.0, "Ref", "near", "Serve", "first_serve", bounce=(1.0, 18.0), speed=90),
        _row(1, 2, 9.5, "Opp", "far", "Backhand", "first_return", bounce=(0.5, 5.0), speed=50),
        _row(1, 3, 11.0, "Ref", "near", "Forehand", "serve_plus_one", bounce=(-2.0, 21.0)),
    ]
    # Point 2: guest serves from far; host returns into the net.
    r += [
        _row(2, 1, 38.0, "Opp", "far", "Serve", "first_serve", bounce=(-1.0, 7.0), speed=80),
        _row(2, 2, 39.2, "Ref", "near", "Forehand", "first_return", result="Net"),
    ]
    # Point 3: host serves; only the serve and a return.
    r += [
        _row(3, 1, 68.0, "Ref", "near", "Serve", "first_serve", bounce=(1.5, 17.0), speed=95),
        _row(3, 2, 69.4, "Opp", "far", "Forehand", "first_return", bounce=(0.0, 4.0)),
    ]
    return r


def _tracker():
    """The tracker's view: on the video clock, with its own mistakes."""
    t = []
    t += [
        _row(1, 1, 10.02, "Me", "near", "Serve", "first_serve", bounce=(1.2, 18.5), speed=86),
        _row(1, 2, 11.48, "Them", "far", "Unclear", "first_return", bounce=(0.5, 5.4)),
        _row(1, 3, 13.05, "Me", "far", "Backhand", "serve_plus_one", bounce=(-2.0, 22.0)),
    ]
    # Point 2's return was missed entirely; an extra hit was seen at 45s.
    t += [
        _row(2, 1, 40.1, "Them", "far", "Serve", "first_serve", bounce=(-1.0, 7.5), speed=70),
        _row(2, 2, 45.0, "Me", "near", "Forehand", "in_play", result="Out"),
    ]
    t += [
        _row(3, 1, 70.0, "Me", "near", "Serve", "first_serve", bounce=(1.5, 17.0)),
        _row(3, 2, 71.4, "Them", "far", "Forehand", "first_return", bounce=(0.0, 4.0)),
    ]
    return t


def _compare(**kwargs):
    arguments = dict(reference_host="Ref", tracker_host="Me", clock_offset=OFFSET)
    arguments.update(kwargs)
    return evaluate.compare_shots(_reference(), _tracker(), **arguments)


def test_ratios_and_spreads_carry_their_denominators():
    assert str(evaluate.Ratio(7, 9)) == "7/9 (78%)"
    assert str(evaluate.Ratio(0, 0)) == "0/0 (n/a)" and evaluate.Ratio(0, 0).value is None
    assert evaluate.Ratio(1, 4).to_dict() == {"hits": 1, "total": 4, "value": 0.25}
    spread = evaluate.Distribution.of([1.0, 2.0, 3.0])
    assert spread.median == 2.0 and spread.count == 3
    assert str(evaluate.Distribution.of([])) == "n/a (n=0)"
    assert str(spread).startswith("median 2.00, p90 2.80")


def test_stroke_families_compare_both_vocabularies():
    assert evaluate.stroke_family("Forehand Volley") == "volley"
    assert evaluate.stroke_family("Backhand") == "backhand"
    assert evaluate.stroke_family("Unclear") is None and evaluate.stroke_family(None) is None
    assert evaluate.stroke_family("Lob") == "lob"


def test_shots_pair_closest_first_one_to_one():
    reference = [_row(1, 1, 10.0, "R", "near", "Serve", "first_serve")]
    tracker = [
        _row(1, 1, 12.1, "T", "near", "Serve", "first_serve"),
        _row(1, 2, 11.95, "T", "near", "Serve", "first_serve"),
    ]
    assert evaluate.match_shots(reference, tracker, OFFSET, 0.15) == [(0, 1)]
    assert evaluate.match_shots([], tracker, OFFSET, 0.15) == []


def test_every_measure_is_reported_with_its_denominator():
    result = _compare()
    assert result.reference_shots == 7 and result.tracker_shots == 7
    assert str(result.shot_recall) == "6/7 (86%)"
    assert str(result.shot_precision) == "6/7 (86%)"
    assert {k: str(v) for k, v in result.recall_by_side.items()} == {
        "near": "3/4 (75%)",
        "far": "3/3 (100%)",
    }
    assert str(result.hitter_side) == "5/6 (83%)"
    assert str(result.stroke) == "4/5 (80%)"
    assert str(result.stroke_abstained) == "1/6 (17%)"
    assert result.stroke_confusion == {
        "serve->serve": 3,
        "forehand->backhand": 1,
        "forehand->forehand": 1,
    }
    assert str(result.shot_type) == "6/6 (100%)"
    assert str(result.serve_recall) == "3/3 (100%)"
    assert str(result.server_side) == "3/3 (100%)"
    # Point 2: the reference says the server won (return in the net); the
    # tracker's last shot is an extra hit by the receiver that went out -
    # it agrees for a different reason. Points 1 and 3 agree.
    assert str(result.winner) == "3/3 (100%)"
    assert result.bounce_error_m.count == 6
    assert result.bounce_error_by_half["far"].count == 3
    assert result.speed_difference_mph.count == 2
    assert result.speed_difference_mph.mean == pytest.approx(-7.0)
    assert result.speed_abs_difference_mph.mean == pytest.approx(7.0)
    assert not result.offset_estimated and result.window == (9.02, 72.4)


def test_the_confidence_table_shows_how_often_each_band_was_right():
    confidence = {(1, 1): 0.9, (1, 2): 0.5, (1, 3): 0.5, (2, 2): 0.1, (3, 1): 0.95}
    result = _compare(confidence=confidence)
    assert result.confidence_calibration == [
        {"bin": "0.0-0.2", "shots": 1, "paired": 0},
        {"bin": "0.4-0.6", "shots": 2, "paired": 2},
        {"bin": "0.8-1.0", "shots": 2, "paired": 2},
    ]


def test_the_lines_read_as_agreement_with_a_weak_reference_and_serialise():
    result = _compare(confidence={(1, 1): 0.9})
    lines = result.lines()
    assert lines[0].startswith("Against a SwingVision export (a weak reference, not ground truth)")
    assert "clock offset +2.00s (given)" in lines[0]
    assert any(line.strip().startswith("serve recall     3/3 (100%)") for line in lines)
    assert any("landing far" in line for line in lines)
    assert any("0.8-1.0" in line for line in lines)
    assert any(line.startswith("Note: Only 6 paired bounce(s)") for line in lines)
    data = json.loads(json.dumps(result.to_dict()))
    assert data["shot_recall"] == {"hits": 6, "total": 7, "value": pytest.approx(6 / 7)}
    assert data["recall_by_side"]["near"]["total"] == 4
    assert data["bounce_error_by_half"]["far"]["count"] == 3


def test_an_unknown_offset_is_estimated_and_said_to_be():
    result = _compare(clock_offset=None)
    assert result.offset_estimated and result.clock_offset == pytest.approx(2.0, abs=0.03)
    assert any("estimated by pairing" in note for note in result.notes)
    assert "(estimated)" in result.lines()[0]
    assert evaluate.estimate_clock_offset([], _tracker()) == (0.0, 0)


def test_a_window_with_no_reference_shots_says_nothing_could_be_scored():
    result = _compare(window=(500.0, 600.0))
    assert result.reference_shots == 0 and result.shot_recall.value is None
    assert any("nothing could be scored" in note for note in result.notes)
    assert result.lines()[-1].startswith("Note:")
    empty = evaluate.compare_shots(
        _reference(), [], reference_host="Ref", tracker_host="Me", clock_offset=OFFSET
    )
    assert empty.window is None and empty.reference_shots == 7


def test_rows_without_a_time_are_left_out_of_a_window():
    rows = [_row(1, 1, None, "R", "near", "Serve", "first_serve"), _reference()[0]]
    assert evaluate.in_window(rows, 0.0, 100.0, 0.0) == [rows[1]]
    assert evaluate.is_serve(_row(1, 1, 1.0, "R", "near", "Serve", "none"))


def _staged(confirm=True):
    rows = _tracker()
    result = reconstruct.reconstruct_all(rows, "Me")
    record = MatchRecord(date="2026-08-21", opponent="Them", result="W", sets=result.sets)
    record.tracking = {"shots": [asdict(r) for r in rows], "settings": {"host_name": "Me"}}
    points = [p for s in record.sets for p in s.points]
    if confirm:
        points[0].confirmed_by = "human"
        points[1].confirmed_by = "human"
        points[1].point_won = not points[1].point_won
        points[1].point_end_type = "forced_error"
    return record, points


def test_confirmed_points_score_the_trackers_original_guess():
    record, points = _staged()
    result = evaluate.compare_confirmed(record)
    assert result.confirmed_points == 2 and result.traced_points == 2
    assert str(result.point_won) == "1/2 (50%)" and str(result.end_type) == "1/2 (50%)"
    assert sum(result.end_type_pairs.values()) == 2
    lines = result.lines()
    assert lines[0].startswith("Against the reviewer's confirmations (human judgements)")
    assert any("pairs" in line for line in lines)
    assert json.loads(json.dumps(result.to_dict()))["point_won"]["hits"] == 1


def test_confirmations_that_cannot_be_traced_or_matches_not_tracked_say_so():
    record, points = _staged()
    points[0].source_point_number = 999
    result = evaluate.compare_confirmed(record)
    assert result.traced_points == 1
    assert "1 confirmed point(s) could not be traced" in result.notes[0]
    imported, _ = _staged()
    imported.tracking = None
    result = evaluate.compare_confirmed(imported)
    assert result.traced_points == 0 and "not staged by the tracker" in result.notes[0]
    assert "Note:" in result.lines()[-1]
    unconfirmed, _ = _staged(confirm=False)
    assert evaluate.compare_confirmed(unconfirmed).confirmed_points == 0
    assert not any("pairs" in line for line in evaluate.compare_confirmed(unconfirmed).lines())


def test_tracker_confidence_reads_every_trace_including_folded_faults():
    tracking = {
        "attempts": [
            {
                "shots": [{"point_number": 2, "shot_number": 1, "confidence": 0.7}],
                "folded_fault": {
                    "shots": [{"point_number": 1, "shot_number": 1, "confidence": 0.3}]
                },
            },
            {"shots": [{"point_number": 3, "shot_number": 1, "confidence": 0.9}]},
        ]
    }
    assert evaluate.tracker_confidence(tracking) == {(2, 1): 0.7, (1, 1): 0.3, (3, 1): 0.9}
    assert evaluate.tracker_confidence(None) == {}
