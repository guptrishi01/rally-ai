"""Tests for segmentation: serves, serve attempts, roles, faults, confidence.

Every event track here is hand-built (`points`): no video, no model.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from tracking import segment
from tracking.ball import BallTrack
from tracking.config import SegmentConfig
from tracking.players import FAR, NEAR
from tracking.provenance import Provenance
from tracking.segment import (
    EXCLUDED,
    FIRST_RETURN,
    FIRST_SERVE,
    IN_PLAY,
    NONE,
    RETURN_PLUS_ONE,
    SECOND_RETURN,
    SECOND_SERVE,
    SERVE_PLUS_ONE,
    SERVE_UNSEEN,
    SERVED,
)
from tracking.strokes import BACKHAND, FOREHAND, OVERHEAD, SERVE, UNKNOWN

from . import points as P
from .conftest import SIZE


def roles(attempt):
    """An attempt's shot roles, in order."""
    return [shot.role for shot in attempt.shots]


def kinds(segmentation):
    """Every attempt's kind, in order."""
    return [attempt.kind for attempt in segmentation.attempts]


def run(*pieces, config=None, **build):
    """Builds pieces and segments them."""
    events, labels = P.build(*pieces, **build)
    return segment.segment(events, labels, config)


# --------------------------------------------------------------------------
# Roles
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "position, second, expected",
    [
        (0, False, FIRST_SERVE),
        (0, True, SECOND_SERVE),
        (1, False, FIRST_RETURN),
        (1, True, SECOND_RETURN),
        (2, True, SERVE_PLUS_ONE),
        (3, False, RETURN_PLUS_ONE),
        (4, False, IN_PLAY),
        (9, True, IN_PLAY),
    ],
)
def test_a_shot_is_named_by_its_place_in_the_rally(position, second, expected):
    assert segment.role_for(position, second) == expected


def test_a_negative_place_is_refused():
    with pytest.raises(ValueError, match="position"):
        segment.role_for(-1)


def test_a_clean_rally_is_one_attempt_with_the_sheets_roles():
    found = run(P.rally(NEAR, "deuce", 6))
    (attempt,) = found.attempts
    assert attempt.kind == SERVED and attempt.number == 1
    assert roles(attempt) == [
        FIRST_SERVE,
        FIRST_RETURN,
        SERVE_PLUS_ONE,
        RETURN_PLUS_ONE,
        IN_PLAY,
        IN_PLAY,
    ]
    assert [s.shot_number for s in attempt.shots] == [1, 2, 3, 4, 5, 6]
    assert attempt.server_side == NEAR and attempt.server == "A"
    assert (attempt.serve_zone, attempt.zone_source) == ("deuce", "feet")
    assert attempt.serve is attempt.shots[0]
    assert attempt.start == attempt.shots[0].shot.hit.pts
    assert attempt.end == attempt.shots[-1].shot.hit.pts


def test_a_missed_hit_moves_the_rally_count_on_by_two():
    piece = P.rally(FAR, "ad", 5)
    del piece.hits[1]  # the near player's return was not seen
    (attempt,) = run(piece).attempts
    assert roles(attempt) == [FIRST_SERVE, SERVE_PLUS_ONE, RETURN_PLUS_ONE, IN_PLAY]
    assert [s.position for s in attempt.shots] == [0, 2, 3, 4]


# --------------------------------------------------------------------------
# Which hits are serves
# --------------------------------------------------------------------------


def test_points_are_numbered_one_per_serve_attempt_in_time_order():
    found = run(P.rally(NEAR, "deuce", 3), P.rally(NEAR, "ad", 4), P.rally(FAR, "deuce", 2))
    assert [a.number for a in found.attempts] == [1, 2, 3]
    assert kinds(found) == [SERVED] * 3
    assert [a.serve_zone for a in found.attempts] == ["deuce", "ad", "deuce"]
    assert found.attempts[2].server == "B"


def test_the_servers_pre_serve_bounce_does_not_steal_the_serve():
    # Seen on four real windows: the ball bounced before serving is taken
    # for a weak "serve", the events stage infers a 0.2 return from it,
    # and the real serve that follows is labelled an overhead.
    early = P.Piece(
        [
            (P.hit(0.0, NEAR, confidence=0.45, provenance=Provenance.INFERRED), SERVE),
            (P.hit(1.3, FAR, 2.0, confidence=0.2, provenance=Provenance.INFERRED), UNKNOWN),
        ]
    )
    main = P.rally(NEAR, "deuce", 3)
    main.hits[0] = (main.hits[0][0], OVERHEAD)
    found = run(early, (1.2, main))
    assert kinds(found) == [EXCLUDED, SERVED]
    junk, real = found.attempts
    assert roles(junk) == [NONE, NONE]
    assert [s.position for s in junk.shots] == [None, None]
    assert real.shots[0].shot.hit.pts == pytest.approx(12.5)
    assert real.shots[0].label.stroke.value == SERVE
    assert real.shots[0].label.stroke.provenance is Provenance.INFERRED
    assert "classifier said Overhead" in real.shots[0].label.reason
    assert any("also looked like the serve" in r for r in real.reasons)
    # The weak serve label that lost is not left claiming to be a serve.
    assert junk.shots[0].stroke == UNKNOWN
    assert "did not take it for a serve" in junk.shots[0].label.reason


def test_a_rally_going_on_through_weak_inferred_returns_is_not_a_new_serve():
    piece = P.Piece([(P.hit(0.0, NEAR), SERVE), (P.hit(1.2, FAR, 2.5), FOREHAND)])
    for n, t in enumerate((2.6, 4.0, 5.4, 6.8, 8.2)):
        if n % 2:
            weak = P.hit(t, FAR, -2.0, confidence=0.2, provenance=Provenance.INFERRED)
            piece.hits.append((weak, FOREHAND))
        else:
            piece.hits.append((P.hit(t, NEAR, 0.3), UNKNOWN))
    (attempt,) = run(piece).attempts
    assert len(attempt.shots) == 7
    assert any("weakly inferred" in r for r in attempt.reasons)


@pytest.mark.parametrize(
    "where, stroke",
    [
        ({"x": 2.6}, SERVE),  # a returner's spot, not a server's
        ({"y": 3.0}, SERVE),  # inside the court
        ({}, FOREHAND),  # a named groundstroke after a quiet spell is not a serve
        ({"seen": False}, OVERHEAD),  # no feet to check an overhead against
    ],
)
def test_a_hit_that_cannot_be_a_serve_is_not_taken_for_one(where, stroke):
    piece = P.Piece([(P.hit(0.0, NEAR, **where), stroke), (P.hit(1.2, FAR, 2.5), FOREHAND)])
    found = run(piece)
    assert kinds(found) == [SERVE_UNSEEN]


QUIET = segment.LeadIn(9.0, 0, True)


def test_a_serve_label_without_feet_still_counts_at_half_weight():
    events, labels = P.build(P.Piece([(P.hit(0.0, NEAR, seen=False), SERVE)]))
    score = segment.serve_score(events.shots[0], labels[0], QUIET)
    assert score == pytest.approx(0.5 * 0.7 * 0.75)
    assert kinds(segment.segment(events, labels)) == [SERVED]


def test_serve_scores_weigh_the_label_and_what_led_in():
    events, _ = P.build(P.Piece([(P.hit(0.0, NEAR), SERVE)]))
    shot = events.shots[0]
    score = segment.serve_score
    assert score(shot, P.label(0, SERVE, 0.8), QUIET) == pytest.approx(0.8 * 0.75)
    assert score(shot, P.label(0, OVERHEAD, 0.8), QUIET) == pytest.approx(0.9 * 0.8 * 0.75)
    assert score(shot, P.label(0, UNKNOWN), QUIET) == pytest.approx(0.6 * 0.3 * 0.75)
    assert score(shot, None, QUIET) == pytest.approx(0.6 * 0.3 * 0.75)
    # Too soon after the other player; too many hits leading in; hits that
    # lead in from the other player's hit rather than from a silence.
    assert score(shot, P.label(0, SERVE), segment.LeadIn(3.0, 0, True)) == 0.0
    assert score(shot, P.label(0, SERVE), segment.LeadIn(9.0, 3, True)) == 0.0
    assert score(shot, P.label(0, SERVE), segment.LeadIn(9.0, 1, False)) == 0.0
    assert score(shot, P.label(0, SERVE), segment.LeadIn(9.0, 2, True)) > 0.0


def test_a_far_serve_is_allowed_more_slack_behind_the_baseline():
    far = P.Piece([(P.hit(0.0, FAR, -0.6, 22.2), SERVE), (P.hit(1.2, NEAR, 2.5), FOREHAND)])
    assert kinds(run(far)) == [SERVED]
    strict = SegmentConfig(serve_behind_far_m=0.6)
    assert kinds(run(far, config=strict)) == [SERVE_UNSEEN]


def test_lead_in_counts_hits_since_the_last_silence_or_confident_rival_hit():
    events, _ = P.build(
        P.Piece([(P.hit(0.0, FAR), SERVE)]),
        (
            9.0,
            P.Piece(
                [
                    (P.hit(0.0, NEAR), UNKNOWN),
                    (P.hit(1.0, FAR, confidence=0.2), UNKNOWN),
                    (P.hit(2.0, NEAR), SERVE),
                ]
            ),
        ),
    )
    shots = list(events.shots)
    assert segment.lead_in(shots, 0) == segment.LeadIn(float("inf"), 0, True)
    assert segment.lead_in(shots, 3) == segment.LeadIn(pytest.approx(11.0), 2, True)
    # A confident rival hit stops the count, sets the quiet, and the hits
    # after it are a rally, not a lead-in.
    sure = SegmentConfig(quiet_breaker_confidence=0.1)
    assert segment.lead_in(shots, 3, sure) == segment.LeadIn(pytest.approx(1.0), 0, True)
    rallying, _ = P.build(
        P.Piece([(P.hit(0.0, FAR), SERVE), (P.hit(1.0, NEAR, 3.0), FOREHAND)]),
        (1.5, P.Piece([(P.hit(0.0, NEAR), UNKNOWN)])),
    )
    assert segment.lead_in(list(rallying.shots), 2) == segment.LeadIn(
        pytest.approx(2.5), 1, False
    )


def test_two_serves_by_different_players_are_never_one_cluster():
    events, labels = P.build(
        P.Piece([(P.hit(0.0, NEAR), SERVE)]), (1.0, P.Piece([(P.hit(0.0, FAR), SERVE)]))
    )
    config = SegmentConfig(serve_quiet_seconds=0.5)
    assert len(segment.pick_serves(list(events.shots), dict(enumerate(labels)), config)) == 2


def test_serve_candidates_far_apart_are_separate_serves():
    events, labels = P.build(P.serve(NEAR, "deuce"), P.serve(NEAR, "ad"))
    picked = segment.pick_serves(list(events.shots), dict(enumerate(labels)))
    assert sorted(picked) == [0, 1]
    assert all(not contested for _, contested in picked.values())


# --------------------------------------------------------------------------
# Serving side
# --------------------------------------------------------------------------


def test_the_serve_bounce_names_the_side_when_the_feet_are_on_the_centre_line():
    piece = P.serve(NEAR, "ad", landing="in")
    server = P.hit(0.0, NEAR, 0.05)  # deuce by a few centimetres
    piece.hits[0] = (server, SERVE)
    events, _ = P.build(piece)
    assert segment.serve_zone(events.shots[0]) == ("ad", "bounce")
    clear = replace(events.shots[0], hit=P.hit(events.shots[0].hit.pts, NEAR, 1.5))
    assert segment.serve_zone(clear) == ("deuce", "feet")


def test_a_serve_with_no_feet_takes_its_side_from_the_bounce_or_has_none():
    landed, _ = P.build(P.serve(NEAR, "deuce", landing="in", seen=False))
    assert segment.serve_zone(landed.shots[0]) == ("deuce", "bounce")
    blind, _ = P.build(P.serve(NEAR, "deuce", landing="net", seen=False))
    assert segment.serve_zone(blind.shots[0]) == (None, None)


# --------------------------------------------------------------------------
# Faults and second serves
# --------------------------------------------------------------------------


def test_a_fault_then_the_same_side_again_is_a_second_serve():
    found = run(P.serve(NEAR, "ad", landing="net"), (7.0, P.rally(NEAR, "ad", 3)))
    fault, second = found.attempts
    assert fault.fault and not fault.second_serve
    assert second.second_serve and not second.fault
    assert roles(second) == [SECOND_SERVE, SECOND_RETURN, SERVE_PLUS_ONE]
    assert any("first-serve fault" in r for r in fault.reasons)
    assert found.counts()["faults"] == 1 and found.counts()["second_serves"] == 1


def test_a_fault_the_receiver_swatted_back_is_still_a_fault():
    swatted = P.serve(NEAR, "deuce", landing="unseen")
    swatted.hits.append((P.hit(1.0, FAR, -2.5), FOREHAND))
    fault, second = run(swatted, (6.0, P.rally(NEAR, "deuce", 2))).attempts
    assert fault.fault and second.second_serve
    assert fault.shots[0].shot.result.value == "In"  # inferred: the receiver hit it


@pytest.mark.parametrize(
    "first, then, gap",
    [
        (P.serve(NEAR, "deuce", landing="net"), P.rally(NEAR, "ad", 2), 7.0),  # other side
        (P.serve(NEAR, "deuce", landing="in"), P.rally(NEAR, "deuce", 2), 7.0),  # seen in: a let
        (P.serve(NEAR, "deuce", landing="net"), P.rally(NEAR, "deuce", 2), 30.0),  # too long
        (P.serve(NEAR, "deuce", landing="net"), P.rally(FAR, "deuce", 2), 7.0),  # other server
        (P.rally(NEAR, "deuce", 3, serve_landing="unseen"), P.rally(NEAR, "deuce", 2), 7.0),
    ],
    ids=["other side", "seen in", "too long", "other server", "played on"],
)
def test_a_serve_is_not_a_fault_without_the_evidence(first, then, gap):
    one, two = run(first, (gap, then)).attempts
    assert not one.fault and not two.second_serve
    assert roles(two)[0] == FIRST_SERVE


def test_a_bridged_bounce_in_the_box_blocks_a_fault_too():
    bridged = P.serve(NEAR, "deuce", landing="unseen")
    x, y = P.SERVE_LANDING[NEAR]["deuce"]
    bridged.bounces.append(P.bounce(0.6, x, y, provenance=Provenance.INTERPOLATED))
    one, two = run(bridged, (7.0, P.rally(NEAR, "deuce", 2))).attempts
    assert not one.fault and not two.second_serve


def test_a_second_serve_cannot_fault_into_a_third():
    found = run(
        P.serve(NEAR, "deuce", landing="net"),
        (7.0, P.serve(NEAR, "deuce", landing="net")),
        (7.0, P.serve(NEAR, "deuce", landing="net")),
    )
    assert [(a.fault, a.second_serve) for a in found.attempts] == [
        (True, False),
        (False, True),
        (False, False),  # nothing follows it to make it a fault
    ]


def test_without_a_serving_side_a_fault_is_judged_by_time_alone():
    quick = run(
        P.serve(NEAR, "deuce", landing="net", seen=False),
        (8.0, P.rally(NEAR, "deuce", 2)),
    )
    one, two = quick.attempts
    assert one.fault and two.second_serve
    assert any("by time alone" in r for r in two.reasons)
    slow = run(
        P.serve(NEAR, "deuce", landing="net", seen=False),
        (15.0, P.rally(NEAR, "deuce", 2)),
    )
    assert not slow.attempts[0].fault


def test_a_person_mismatch_at_the_same_end_is_not_folded():
    found = run(
        P.serve(NEAR, "deuce", landing="net"),
        (7.0, P.rally(NEAR, "deuce", 2, person="B")),
    )
    assert not found.attempts[0].fault


def test_play_between_points_does_not_stop_a_fault_folding():
    swat = P.Piece([(P.hit(0.0, FAR, 3.0, confidence=0.3), FOREHAND)])
    found = run(P.serve(NEAR, "ad", landing="net"), (6.0, swat), (6.0, P.rally(NEAR, "ad", 2)))
    assert kinds(found) == [SERVED, EXCLUDED, SERVED]
    assert found.attempts[0].fault and found.attempts[2].second_serve


# --------------------------------------------------------------------------
# Rallies with no serve, and play between points
# --------------------------------------------------------------------------


def test_a_rally_whose_serve_was_missed_is_a_point_without_a_serve_row():
    piece = P.Piece(
        [
            (P.hit(0.0, NEAR, 3.0), FOREHAND),
            (P.hit(1.5, FAR, -2.0), BACKHAND),
            (P.hit(3.0, NEAR, -2.0), FOREHAND),
        ]
    )
    (attempt,) = run(piece).attempts
    assert attempt.kind == SERVE_UNSEEN
    assert roles(attempt) == [FIRST_RETURN, SERVE_PLUS_ONE, RETURN_PLUS_ONE]
    assert [s.shot_number for s in attempt.shots] == [2, 3, 4]  # the serve is shot 1
    assert attempt.server_side == FAR and attempt.server == "B"
    assert attempt.serve is None
    # The returner stood in their deuce half, so the serve came from deuce.
    assert (attempt.serve_zone, attempt.zone_source) == ("deuce", "return")
    assert attempt.serve_estimate == pytest.approx(10.0 - 1.14)
    assert attempt.confidence == pytest.approx(0.25 * 0.75, abs=1e-3)
    assert any("no serve was seen" in r for r in attempt.reasons)
    far_first = run(P.Piece([(P.hit(0.0, FAR, 3.0), FOREHAND), (P.hit(1.5, NEAR, 2.0), BACKHAND)]))
    assert far_first.attempts[0].server_side == NEAR and far_first.attempts[0].server == "A"


def test_a_lone_hit_or_one_players_hits_are_play_between_points():
    lone = run(P.Piece([(P.hit(0.0, FAR, 3.0), FOREHAND)]))
    assert kinds(lone) == [EXCLUDED]
    assert "taken for play between points" in lone.attempts[0].reasons[0]
    stray = P.Piece([(P.hit(0.0, FAR, 3.0), FOREHAND)])
    two = run(stray, P.Piece([(P.hit(0.0, NEAR, 3.0), FOREHAND)]))
    assert kinds(two) == [EXCLUDED, EXCLUDED]
    one_sided = run(P.Piece([(P.hit(0.0, NEAR, 3.0), UNKNOWN), (P.hit(2.0, NEAR, 3.0), UNKNOWN)]))
    assert kinds(one_sided) == [EXCLUDED]
    assert one_sided.attempts[0].confidence == 0.0
    assert one_sided.attempts[0].server is None


def test_play_before_the_match_start_is_warm_up():
    rallying = P.Piece([(P.hit(0.0, NEAR, 3.0), FOREHAND), (P.hit(1.5, FAR, -2.0), BACKHAND)])
    found = run(rallying, P.rally(NEAR, "deuce", 2), config=SegmentConfig(match_start=20.0))
    assert kinds(found) == [EXCLUDED, SERVED]
    assert found.attempts[0].reasons == ("before the match started: warm-up",)


def test_a_long_silence_ends_a_rally_and_the_rest_waits_for_a_serve():
    piece = P.rally(NEAR, "deuce", 2)
    piece.hits.append((P.hit(8.0, FAR, 3.0), FOREHAND))
    piece.hits.append((P.hit(9.5, NEAR, -3.0), FOREHAND))
    found = run(piece)
    assert kinds(found) == [SERVED, SERVE_UNSEEN]
    assert len(found.attempts[0].shots) == 2


def test_a_boundary_near_the_silence_threshold_lowers_the_confidence():
    sure = run(P.rally(NEAR, "deuce", 3)).attempts[0]
    piece = P.rally(NEAR, "deuce", 3, gap=4.5)
    unsure = run(piece).attempts[0]
    assert len(unsure.shots) == 3
    assert unsure.confidence == pytest.approx(sure.confidence * 0.8, abs=1e-3)
    assert any("boundary is unsure" in r for r in unsure.reasons)
    rallying = P.Piece([(P.hit(0.0, NEAR, 3.0), FOREHAND), (P.hit(4.6, FAR, -2.0), BACKHAND)])
    assert run(rallying).attempts[0].confidence == pytest.approx(0.25 * 0.75 * 0.8, abs=1e-3)


# --------------------------------------------------------------------------
# Labels, events and serialisation
# --------------------------------------------------------------------------


def test_segmentation_relabels_and_rejudges_with_its_own_serves():
    piece = P.rally(NEAR, "deuce", 2)
    piece.hits[0] = (piece.hits[0][0], OVERHEAD)
    events, labels = P.build(piece)
    # Judged as a rally shot, the serve's landing in the far service box is in;
    # judged as a serve from deuce it is still in - and now it is a serve.
    found = segment.segment(events, labels)
    serve_shot = found.attempts[0].shots[0].shot
    assert serve_shot.serve
    assert found.labels[0].stroke.value == SERVE
    assert found.events.shots[0].serve


def test_a_shot_with_no_label_is_an_unknown_stroke_and_can_still_be_the_serve():
    events, _ = P.build(P.rally(NEAR, "deuce", 3))
    found = segment.segment(events, ())
    first, second, _ = found.attempts[0].shots
    assert first.label.stroke.value == SERVE  # chosen, so labelled
    assert first.label.stroke.confidence == pytest.approx(0.3)  # 0.6 x 0.3 floored
    assert second.label is None and second.stroke == UNKNOWN


def test_segmentation_serialises_to_plain_json():
    found = run(P.serve(NEAR, "ad", landing="net"), (7.0, P.rally(NEAR, "ad", 3)))
    data = json.loads(json.dumps(found.to_dict()))
    assert data["source"] == "segment-v1"
    assert data["counts"] == {
        SERVED: 2,
        SERVE_UNSEEN: 0,
        EXCLUDED: 0,
        "faults": 1,
        "second_serves": 1,
    }
    assert data["attempts"][1]["shots"][0]["role"] == SECOND_SERVE
    assert data["attempts"][0]["fault"] is True


def test_no_shots_means_no_attempts():
    events, labels = P.build()
    found = segment.segment(events, labels)
    assert found.attempts == () and found.counts()[SERVED] == 0


def test_a_quiet_spell_nobody_watched_vouches_only_for_a_contact_seen_struck():
    unwatched = segment.LeadIn(float("inf"), 0, True, watched=0.5)
    events, _ = P.build(P.Piece([(P.hit(0.0, NEAR), SERVE)]))
    shot = events.shots[0]
    guessed = P.label(0, SERVE, 0.45, Provenance.INFERRED)
    assert segment.serve_score(shot, guessed, unwatched) == 0.0
    assert segment.serve_score(shot, P.label(0, UNKNOWN), unwatched) == 0.0
    assert segment.serve_score(shot, P.label(0, OVERHEAD, 0.7), unwatched) > 0.0
    # Seen on the real recording: an inferred hit 0.12s into a tracked
    # stretch, read as a serve and then as a fault of the serve after it.
    frames = np.arange(300)
    ball = BallTrack((), (), (), frames, 9.9 + frames / 30.0, SIZE, "motion")
    early = P.serve(NEAR, "ad", landing="unseen", confidence=0.45, provenance=Provenance.INFERRED)
    early.hits[0] = (early.hits[0][0], SERVE)
    events, labels = P.build(early, (7.0, P.rally(NEAR, "ad", 2)), ball=ball)
    labels = (P.label(0, SERVE, 0.45, Provenance.INFERRED),) + labels[1:]
    found = segment.segment(events, labels)
    assert kinds(found) == [EXCLUDED, SERVED]
    assert not found.attempts[1].second_serve
    # Told the stretch before it was quiet, the same serve stands.
    told = segment.segment(events, labels, tracking_start=0.0)
    assert kinds(told) == [SERVED, SERVED] and told.attempts[1].second_serve
