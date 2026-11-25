"""Tests for the review modal's summaries of a staged tracking block."""

from __future__ import annotations

from tracking import review_view
from tracking.assemble import ShotTrace
from tracking.provenance import Provenance, Tagged


def _trace(shot_number, fields, guesses=(), notes=(), tracked=True):
    return ShotTrace(
        point_number=7,
        shot_number=shot_number,
        event_index=shot_number,
        pts=10.0 + shot_number,
        time_uncertainty=0.04,
        confidence=0.8,
        tracked=tracked,
        fields=fields,
        guesses=tuple(guesses),
        notes=tuple(notes),
    ).to_dict()


def _attempt(shots, **extra):
    block = {
        "number": 7,
        "kind": "served",
        "start": 10.0,
        "end": 14.0,
        "server_side": "near",
        "server": "Rishi",
        "serve_zone": "deuce",
        "zone_source": "feet",
        "second_serve": False,
        "fault": False,
        "confidence": 0.6,
        "reasons": ["serve seen"],
        "shots": shots,
    }
    block.update(extra)
    return block


SERVE = _trace(
    1,
    {
        "stroke": Tagged("Serve", Provenance.MEASURED, 0.9, "pose"),
        "player": Tagged("Rishi", Provenance.MEASURED, 0.95),
        "result": Tagged("In", Provenance.INFERRED, 0.3, "assemble"),
        "bounce_xy": Tagged((1.0, 18.0), Provenance.INTERPOLATED, 0.5),
        "mystery": Tagged("x", Provenance.INFERRED),
    },
    guesses=["result"],
    notes=["result guessed: the ball came back"],
)


def test_a_shot_lists_fields_in_reading_order_with_guesses_marked():
    shot = review_view.shot_provenance(ShotTrace.from_dict(SERVE))
    assert [f["field"] for f in shot["fields"]] == [
        "player",
        "stroke",
        "result",
        "bounce_xy",
        "mystery",
    ]
    by = {f["field"]: f for f in shot["fields"]}
    assert by["player"]["provenance"] == "measured" and by["player"]["guess"] is False
    assert by["result"] == {
        "field": "result",
        "label": "result",
        "provenance": "inferred",
        "confidence": 0.3,
        "guess": True,
    }
    assert by["bounce_xy"]["label"] == "bounce"
    assert by["mystery"]["label"] == "mystery"
    assert shot["notes"] == ["result guessed: the ball came back"]
    assert shot["tracked"] is True and shot["confidence"] == 0.8


def test_a_guess_with_no_field_of_its_own_is_still_reported_as_a_guess():
    trace = _trace(1, {"player": Tagged("Rishi", Provenance.INFERRED)}, guesses=["serve_row"])
    shot = review_view.shot_provenance(ShotTrace.from_dict(trace))
    assert shot["fields"][-1] == {
        "field": "serve_row",
        "label": "serve row",
        "provenance": "inferred",
        "confidence": None,
        "guess": True,
    }


def test_a_point_without_a_block_has_no_provenance():
    assert review_view.point_provenance(None) is None
    assert review_view.point_provenance({}) is None


def test_a_point_summary_puts_a_folded_faults_shots_first():
    fault = _attempt([_trace(1, {"stroke": Tagged("Unclear", Provenance.INFERRED)},
                             guesses=["stroke"], tracked=False)], fault=True, number=6)
    block = _attempt([SERVE], second_serve=True, folded_fault=fault)
    point = review_view.point_provenance(block)
    assert point["kind"] == "served"
    assert point["folded_fault"] is True
    assert point["second_serve"] is True
    assert point["server_side"] == "near"
    assert point["reasons"] == ["serve seen"]
    assert [s["tracked"] for s in point["shots"]] == [False, True]
    assert [s["part"] for s in point["shots"]] == ["folded_fault", "point"]
    assert point["shots"][0]["fields"][0]["guess"] is True


def test_a_point_without_a_fault_lists_only_its_own_shots():
    point = review_view.point_provenance(_attempt([SERVE]))
    assert point["folded_fault"] is False
    assert len(point["shots"]) == 1


def test_a_match_without_a_block_has_no_provenance():
    assert review_view.match_provenance(None) is None


def test_a_match_summary_carries_the_coverage_line_and_its_denominators():
    block = {
        "source": "assemble-v1",
        "summary": {
            "shots": 40,
            "tracked_shots": 25,
            "guesses": {"result": 9, "stroke": 4, "serve_row": 2},
            "models": {"ball": "summary-ball"},
        },
        "notes": ["Tracked 25 of 38 shots with confidence."],
        "attempts": [],
        "models": {"ball": "classic-ball"},
        "video": {"name": "match.mp4"},
        "range": [0.0, 600.0],
    }
    match = review_view.match_provenance(block)
    assert match == {
        "notes": ["Tracked 25 of 38 shots with confidence."],
        "shots": 40,
        "tracked_shots": 25,
        "guessed_results": 9,
        "guessed_strokes": 4,
        "inferred_serves": 2,
        "models": {"ball": "classic-ball"},
        "video": "match.mp4",
        "range": [0.0, 600.0],
    }


def test_a_sparse_match_block_still_summarises():
    match = review_view.match_provenance({"summary": {"models": {"ball": "b"}}})
    assert match["notes"] == []
    assert match["models"] == {"ball": "b"}
    assert match["video"] is None and match["shots"] is None
