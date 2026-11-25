"""Tests for assembly: the tracker's shots as `RawShotRow`s, and their provenance.

Every event track is hand-built (`points`); reconstruction is the real,
unchanged `swingvision_import.reconstruct`.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from swingvision_import import reconstruct
from swingvision_import.raw import RawMatchExport, RawSettings, RawShotRow
from swingvision_import.reconstruct import ReconstructedPoint, ReconstructionResult
from tracking import assemble, segment
from tracking import events as events_stage
from tracking.assemble import MatchSetup
from tracking.ball import BallTrack
from tracking.config import EventConfig, StrokeConfig, TrackerConfig
from tracking.players import FAR, NEAR, EndSwap, PlayerTracks
from tracking.provenance import Provenance
from tracking.strokes import FOREHAND, OVERHEAD, SERVE, UNKNOWN

from . import points as P
from .conftest import SIZE

SETUP = MatchSetup("Host", "Guest", "A")


def golden_events():
    """A short hand-annotated sequence: warm-up junk, a rally, a fault, a second serve, an ace."""
    junk = P.Piece(
        [
            (P.hit(0.0, NEAR, confidence=0.45, provenance=Provenance.INFERRED), SERVE),
            (P.hit(1.3, FAR, 2.0, confidence=0.2, provenance=Provenance.INFERRED), UNKNOWN),
        ]
    )
    first = P.rally(NEAR, "deuce", 4, end="net")
    first.hits[0] = (first.hits[0][0], OVERHEAD)
    first.hits[2] = (first.hits[2][0], UNKNOWN)
    fault = P.serve(NEAR, "ad", landing="net")
    second = P.rally(NEAR, "ad", 3, end="out")
    ace = P.serve(NEAR, "deuce", landing="in")
    return P.build(junk, (1.2, first), fault, (7.0, second), ace)


def golden():
    """The golden sequence, assembled."""
    return assemble.assemble(*golden_events(), SETUP)


# (point, shot, player, shot_type, stroke, result, video_time, side, zone,
#  hit_x, hit_y, bounce_x, bounce_y, speed) - checked by hand against the rules.
GOLDEN = [
    (1, 1, "Host", "none", "Unclear", "In", 10.0, "near", "deuce", 0.5, -0.4, None, None, None),
    (1, 2, "Guest", "none", "Unclear", "In", 11.3, "far", "ad", 2.0, 24.2, None, None, None),
    (2, 1, "Host", "first_serve", "Serve", "In", 12.5, "near", "deuce", 0.5, -0.4, -2.0, 16.5,
     63.69),
    (2, 2, "Guest", "first_return", "Forehand", "In", 14.0, "far", "ad", 2.5, 24.2, None, None,
     None),
    (2, 3, "Host", "serve_plus_one", "Unclear", "In", 15.5, "near", "ad", -2.5, -0.4, None, None,
     None),
    (2, 4, "Guest", "return_plus_one", "Forehand", "Net", 17.0, "far", "ad", 2.5, 24.2, None,
     None, None),
    (3, 1, "Host", "first_serve", "Serve", "Net", 42.4, "near", "ad", -0.5, -0.4, None, None,
     None),
    (4, 1, "Host", "second_serve", "Serve", "In", 49.8, "near", "ad", -0.5, -0.4, 2.0, 16.5,
     63.69),
    (4, 2, "Guest", "second_return", "Forehand", "In", 51.3, "far", "ad", 2.5, 24.2, None, None,
     None),
    (4, 3, "Host", "serve_plus_one", "Forehand", "Out", 52.8, "near", "ad", -2.5, -0.4, 0.0,
     25.5, 72.76),
    (5, 1, "Host", "first_serve", "Serve", "In", 78.6, "near", "deuce", 0.5, -0.4, -2.0, 16.5,
     63.69),
]


def as_tuple(row: RawShotRow):
    """The golden fields of a row."""
    return (
        row.point_number,
        row.shot_number,
        row.player,
        row.shot_type,
        row.stroke,
        row.result,
        row.video_time,
        row.hit_side,
        row.hit_zone,
        row.hit_x,
        row.hit_y,
        row.bounce_x,
        row.bounce_y,
        row.speed_mph,
    )


def made(*pieces, setup=SETUP, players=None, calibration=None, labels=None, **build):
    """Builds pieces and assembles them."""
    events, given = P.build(*pieces, **build)
    return assemble.assemble(
        events,
        given if labels is None else labels,
        setup,
        players=players,
        calibration=calibration,
    )


def trace_of(match, point, shot):
    """The trace of one row."""
    return next(t for t in match.traces if (t.point_number, t.shot_number) == (point, shot))


# --------------------------------------------------------------------------
# The golden fixture, end to end
# --------------------------------------------------------------------------


def test_the_golden_sequence_emits_exactly_these_rows():
    match = golden()
    assert [as_tuple(row) for row in match.export.shots] == GOLDEN
    assert match.export.settings == RawSettings("Host", "Guest")
    assert all(row.spin is None and row.source_part == 0 for row in match.export.shots)
    assert assemble.validate_export(match.export) == []


def test_the_golden_sequence_drives_reconstruction_to_the_expected_points():
    match = golden()
    result = reconstruct.reconstruct_all(match.export.shots, "Host")
    assert result.excluded_points == [1]
    assert result.merged_fault_points == [3]
    assert result.skipped_points == []
    found = [
        (p.point_number, p.is_serving, p.point_won, p.point_end_type, p.first_serve_in)
        for p in result.points
    ]
    assert found == [
        (2, True, True, "winner", True),
        (4, True, False, "unforced_error", False),
        (5, True, True, "ace", True),
    ]
    assert all(p.needs_review for s in result.sets for p in s.points)


def test_every_row_has_a_trace_saying_how_each_value_was_got():
    match = golden()
    assert len(match.traces) == len(match.export.shots)
    serve = trace_of(match, 2, 1)
    assert serve.fields["stroke"].value == SERVE
    assert serve.fields["stroke"].provenance is Provenance.INFERRED  # the classifier said overhead
    assert serve.fields["shot_type"].source == "segment-v1"
    assert serve.fields["video_time"].value == pytest.approx(12.5)
    assert serve.fields["bounce_xy"].value == (-2.0, 16.5)
    assert serve.tracked and serve.guesses == ()
    unknown = trace_of(match, 2, 3)
    assert unknown.guesses == ("stroke",)
    assert unknown.fields["stroke"].value == "Unclear"
    assert unknown.fields["stroke"].confidence is None
    assert "too low-confidence to name" in unknown.notes[0]
    junk = trace_of(match, 1, 2)
    assert not junk.tracked  # inferred, weak


# --------------------------------------------------------------------------
# A whole set, and graceful degradation
# --------------------------------------------------------------------------


def hold(server, person):
    """Four points the server wins: alternating sides, each an unreturned serve."""
    return [P.serve(server, zone, person=person) for zone in ("deuce", "ad", "deuce", "ad")]


def broken(server, person):
    """Four points the receiver wins: the server's next shot goes in the net."""
    return [
        P.rally(server, zone, 3, end="net", person=person)
        for zone in ("deuce", "ad", "deuce", "ad")
    ]


def a_set():
    """Host at the near end serving odd games; host breaks the guest in game 10: 6-4."""
    pieces = []
    for game in range(10):
        if game % 2 == 0:
            pieces += hold(NEAR, "A")
        elif game < 9:
            pieces += hold(FAR, "B")
        else:
            pieces += broken(FAR, "B")
    return pieces


def test_a_tracked_set_reconstructs_to_its_score():
    match = made(*a_set())
    assert assemble.validate_export(match.export) == []
    result = reconstruct.reconstruct_all(match.export.shots, "Host")
    (the_set,) = result.sets
    assert (the_set.games_won, the_set.games_lost) == (6, 4)
    assert len(result.points) == 40


def test_missing_far_hits_and_spurious_inferred_hits_degrade_gracefully():
    events, labels = P.build(*a_set())
    rng = np.random.default_rng(7)
    hits = list(events.hits)
    # Drop most far-end rally hits (the real far recall is about a third)...
    kept = [h for h in hits if h.side == NEAR or rng.random() < 0.3 or _is_serve(h)]
    # ...and add weak "returned" hits the events stage infers, at bad times.
    extra = [
        P.hit(h.pts + 0.7, FAR, 1.0, confidence=0.2, provenance=Provenance.INFERRED)
        for h in hits[::7]
        if h.side == NEAR
    ]
    mixed = sorted(kept + extra, key=lambda h: h.pts)
    shots, bounces = events_stage._assemble(
        mixed, list(events.bounces), list(events.nets), EventConfig()
    )
    damaged = replace(events, hits=tuple(mixed), shots=tuple(shots), bounces=tuple(bounces))
    stroke_of = {h.pts: label.stroke.value for h, label in zip(events.hits, labels)}
    relabelled = tuple(
        P.label(i, stroke_of.get(shot.hit.pts, UNKNOWN)) for i, shot in enumerate(shots)
    )
    match = assemble.assemble(damaged, relabelled, SETUP)
    assert assemble.validate_export(match.export) == []
    result = reconstruct.reconstruct_all(match.export.shots, "Host")
    # Every serve was still seen, so every point is still there.
    assert match.segmentation.counts()["served"] == 40
    assert len(result.points) == 40
    assert any("guessed" in note for note in match.notes)


def _is_serve(hit):
    """Whether a scripted hit stands where a server does."""
    return abs(hit.position[0]) <= 0.5 and (hit.position[1] < 0 or hit.position[1] > 24)


def test_nothing_tracked_assembles_to_nothing():
    match = made()
    assert match.export.shots == []
    assert match.notes[0].startswith("Tracked 0 of 0 shots")
    assert match.tracking_block()["attempts"] == []


# --------------------------------------------------------------------------
# Results: kept, or guessed with a reason
# --------------------------------------------------------------------------


def test_a_fault_swatted_back_overrides_the_inferred_in():
    swatted = P.serve(NEAR, "deuce", landing="unseen")
    swatted.hits.append((P.hit(1.0, FAR, -2.5), FOREHAND))
    match = made(swatted, (6.0, P.rally(NEAR, "deuce", 2)))
    row = match.export.shots[0]
    assert (row.shot_type, row.result) == ("first_serve", "Out")
    trace = trace_of(match, 1, 1)
    assert trace.guesses == ("result",)
    assert "overrides an inferred In" in trace.notes[0]
    assert trace.fields["result"].provenance is Provenance.INFERRED
    result = reconstruct.reconstruct_all(match.export.shots, "Host")
    assert result.merged_fault_points == [1]


def test_a_fault_not_seen_past_the_net_is_guessed_into_it():
    match = made(P.serve(NEAR, "deuce", landing="unseen"), (6.0, P.rally(NEAR, "deuce", 2)))
    assert match.export.shots[0].result == "Net"
    assert "was not seen to cross" in trace_of(match, 1, 1).notes[0]


def test_a_fault_seen_out_keeps_its_measured_result():
    match = made(P.serve(NEAR, "deuce", landing="out"), (6.0, P.rally(NEAR, "deuce", 2)))
    assert match.export.shots[0].result == "Out"
    assert trace_of(match, 1, 1).guesses == ()
    assert trace_of(match, 1, 1).fields["result"].provenance is Provenance.MEASURED


def test_a_lone_first_serve_not_served_again_is_taken_as_unreturned():
    match = made(P.serve(NEAR, "deuce", landing="unseen"), P.rally(NEAR, "ad", 2))
    assert match.export.shots[0].result == "In"
    assert "unreturned" in trace_of(match, 1, 1).notes[0]


def test_a_lone_second_serve_is_read_as_a_double_fault():
    match = made(
        P.serve(NEAR, "deuce", landing="net"),
        (6.0, P.serve(NEAR, "deuce", landing="unseen")),
        P.rally(NEAR, "ad", 2),
    )
    second = match.export.shots[1]
    assert (second.shot_type, second.result) == ("second_serve", "Net")
    assert "double fault" in trace_of(match, 2, 1).notes[0]
    result = reconstruct.reconstruct_all(match.export.shots, "Host")
    assert result.points[0].point_end_type == "double_fault"


def test_a_rally_ending_unseen_is_an_error_net_or_out_by_whether_it_crossed():
    into_net = made(P.rally(NEAR, "deuce", 3, end="unknown"))
    assert into_net.export.shots[-1].result == "Net"
    assert "read as an error" in trace_of(into_net, 1, 3).notes[0]
    beyond = P.rally(NEAR, "deuce", 3, end="unknown")
    beyond.bounces.append(P.bounce(3.8, 0.0, 26.0))  # far past the baseline...
    gone = made(beyond)
    assert gone.export.shots[-1].result == "Out"  # ...a measured Out, not a guess


@pytest.mark.parametrize("y, lean, where", [(23.72, "In", "inside"), (23.82, "Out", "beyond")])
def test_a_bounce_too_close_to_call_leans_the_way_its_margin_points(y, lean, where):
    close = P.rally(NEAR, "deuce", 3, end="unknown")
    close.bounces.append(P.bounce(3.8, 0.0, y))  # within 0.1m of the baseline
    match = made(close)
    assert match.export.shots[-1].result == lean
    trace = trace_of(match, 1, 3)
    assert trace.fields["result"].confidence == 0.4
    assert f"{where} the baseline" in trace.notes[0] and f"leaning {lean}" in trace.notes[0]


def test_a_shot_followed_by_the_same_player_is_in_the_ball_came_back():
    piece = P.rally(NEAR, "deuce", 4, end="net")
    del piece.hits[2]  # the server's +1 was not seen: the receiver hits twice
    match = made(piece)
    assert [row.result for row in match.export.shots] == ["In", "In", "Net"]
    assert "the ball came back" in trace_of(match, 1, 2).notes[0]


def test_play_between_points_is_not_judged():
    match = made(P.Piece([(P.hit(0.0, FAR, 3.0), FOREHAND)]))
    assert (match.export.shots[0].shot_type, match.export.shots[0].result) == ("none", "In")
    assert "not judged" in trace_of(match, 1, 1).notes[0]


# --------------------------------------------------------------------------
# Strokes, zones, positions, people
# --------------------------------------------------------------------------


def test_a_shot_with_no_label_is_emitted_as_unknown_and_says_so():
    events, _ = P.build(P.rally(NEAR, "deuce", 2))
    match = assemble.assemble(events, (), SETUP)
    assert [row.stroke for row in match.export.shots] == ["Serve", "Unclear"]
    assert "no stroke label" in trace_of(match, 1, 2).notes[0]
    assert any(
        n.startswith("Low confidence: 1 of 2 strokes") and "(point 1 shot 2)" in n
        for n in match.notes
    )
    renamed = replace(TrackerConfig().assemble, unknown_stroke="Unclassified")
    assert TrackerConfig().assemble.unknown_stroke == "Unclear"
    custom = assemble.assemble(events, (), SETUP, config=TrackerConfig(assemble=renamed))
    assert custom.export.shots[1].stroke == "Unclassified"
    assert assemble.validate_export(custom.export, "Unclassified") == []


def test_zones_are_widened_to_the_sheets_vocabulary_except_on_serves():
    piece = P.Piece(
        [
            (P.hit(0.0, NEAR, 0.05), SERVE),  # on the centre mark: still deuce, a serve
            (P.hit(1.5, FAR, -4.6), FOREHAND),  # beyond the singles sideline
            (P.hit(3.0, NEAR, 0.05), FOREHAND),  # on the centre line
            (P.hit(4.5, FAR, 4.6), FOREHAND),
            (P.hit(6.0, NEAR, 2.0, seen=False), FOREHAND),
        ]
    )
    match = made(piece)
    assert [row.hit_zone for row in match.export.shots] == [
        "deuce",
        "deuce_alley",
        "center_line",
        "ad_alley",
        None,
    ]
    assert match.export.shots[4].hit_x is None and "hit_xy" not in trace_of(match, 1, 5).fields


def test_a_serving_side_named_by_the_bounce_is_tagged_inferred():
    match = made(P.serve(NEAR, "ad", landing="in", seen=False))
    row = match.export.shots[0]
    assert row.hit_zone == "ad"
    zone = trace_of(match, 1, 1).fields["hit_zone"]
    assert zone.provenance is Provenance.INFERRED and zone.source == "segment-v1"


def test_a_serve_with_no_side_at_all_has_no_zone():
    match = made(P.serve(NEAR, "ad", landing="net", seen=False))
    assert match.export.shots[0].hit_zone is None


def test_clipped_feet_are_noted():
    piece = P.rally(NEAR, "deuce", 2)
    piece.hits[0] = (P.hit(0.0, NEAR, clipped=True), SERVE)
    assert "feet were cut off" in " ".join(trace_of(made(piece), 1, 1).notes)


def swap(pts, provenance):
    """Player tracks holding only one change of ends."""
    change = EndSwap(pts, 0, provenance, 0.8, "test")
    return PlayerTracks((), (), (change,), (), "fake-players", 0.5)


def test_after_an_inferred_change_of_ends_who_hit_is_only_inferred():
    match = made(P.rally(NEAR, "deuce", 2), players=swap(11.0, Provenance.INFERRED))
    assert trace_of(match, 1, 1).fields["player"].provenance is Provenance.MEASURED
    assert trace_of(match, 1, 2).fields["player"].provenance is Provenance.INFERRED
    assert any("inferred from the players' clothing" in note for note in match.notes)
    seen = made(P.rally(NEAR, "deuce", 2), players=swap(11.0, Provenance.MEASURED))
    assert trace_of(seen, 1, 2).fields["player"].provenance is Provenance.MEASURED
    assert not any("clothing" in note for note in seen.notes)


def test_the_host_is_who_the_setup_says():
    far_host = MatchSetup.from_start_end("Host", "Guest", "far")
    match = made(P.rally(NEAR, "deuce", 2), setup=far_host)
    assert [row.player for row in match.export.shots] == ["Guest", "Host"]
    result = reconstruct.reconstruct_all(match.export.shots, "Host")
    assert not result.points[0].is_serving


# --------------------------------------------------------------------------
# The setup
# --------------------------------------------------------------------------


def test_the_setup_maps_people_to_names_and_hands():
    setup = MatchSetup.from_start_end("Host", "Guest", "near", host_hand="left")
    assert setup.host_player == "A"
    assert (setup.name("A"), setup.name("B")) == ("Host", "Guest")
    assert setup.handedness() == (("A", "left"), ("B", "right"))
    config = setup.stroke_config(StrokeConfig(serve_quiet_seconds=3.0))
    assert dict(config.handedness) == {"A": "left", "B": "right"}
    assert config.serve_quiet_seconds == 3.0
    flipped = MatchSetup("Host", "Guest", "B", guest_hand="left")
    assert dict(flipped.stroke_config().handedness) == {"B": "right", "A": "left"}
    assert flipped.settings() == RawSettings("Host", "Guest")


@pytest.mark.parametrize(
    "fields, message",
    [
        (("", "Guest", "A"), "both be given"),
        (("Same", "Same", "A"), "must differ"),
        (("Host", "Guest", "C"), "host_player"),
        (("Host", "Guest", "A", "both"), "a hand"),
    ],
)
def test_a_setup_reconstruct_could_not_use_is_refused(fields, message):
    with pytest.raises(ValueError, match=message):
        MatchSetup(*fields)


def test_unknown_ends_and_people_are_refused():
    with pytest.raises(ValueError, match="host_starts"):
        MatchSetup.from_start_end("Host", "Guest", "left")
    with pytest.raises(ValueError, match="person"):
        SETUP.name("C")


# --------------------------------------------------------------------------
# Schema validation
# --------------------------------------------------------------------------


def row(**changes):
    """A valid first serve, changed as asked."""
    fields = dict(
        point_number=1,
        shot_number=1,
        player="Host",
        shot_type="first_serve",
        stroke="Serve",
        result="In",
        video_time=1.0,
        hit_side="near",
        hit_zone="deuce",
    )
    fields.update(changes)
    return RawShotRow(**fields)


SETTINGS = RawSettings("Host", "Guest")


@pytest.mark.parametrize(
    "changes, problem",
    [
        ({"point_number": 0}, "point_number"),
        ({"shot_number": 0}, "shot_number"),
        ({"player": "Somebody"}, "player"),
        ({"shot_type": "serve1"}, "shot_type"),
        ({"stroke": "Lob", "shot_type": "in_play"}, "stroke"),
        ({"result": "in"}, "result"),
        ({"video_time": None}, "video_time"),
        ({"hit_side": "left"}, "hit_side"),
        ({"hit_zone": "middle", "shot_type": "in_play"}, "hit_zone"),
        ({"hit_zone": "deuce_alley"}, "a serve's zone"),
        ({"stroke": "Forehand"}, "a serve's stroke"),
        ({"source_part": 1}, "source_part"),
        ({"spin": "Topspin"}, "spin"),
        ({"speed_mph": -1.0}, "speed_mph"),
        ({"direction": "sideways"}, "direction"),
        ({"hit_x": 1.0}, "hit_x and hit_y"),
        ({"bounce_y": 1.0}, "bounce_x and bounce_y"),
    ],
)
def test_a_row_outside_the_vocabulary_is_reported(changes, problem):
    found = assemble.validate_row(row(**changes), SETTINGS)
    assert len(found) == 1 and problem in found[0]


def test_valid_rows_have_no_problems():
    assert assemble.validate_row(row(), SETTINGS) == []
    rally = row(shot_type="in_play", stroke="Feed", hit_zone="center_line", direction="inside out")
    assert assemble.validate_row(rally, SETTINGS) == []


def test_order_and_numbering_problems_between_rows_are_reported():
    shots = [
        row(),
        row(shot_number=3, shot_type="first_return", stroke="Forehand", video_time=2.0),
        row(point_number=2, video_time=0.5),
        row(point_number=3, shot_type="in_play", stroke="Forehand", video_time=5.0),
        row(point_number=3, shot_number=2, video_time=6.0),
        row(point_number=4, shot_type="none", stroke="Feed", video_time=7.0),
        row(point_number=4, shot_number=2, shot_type="in_play", stroke="Feed", video_time=8.0),
    ]
    found = assemble.validate_export(RawMatchExport(SETTINGS, shots=shots))
    assert found == [
        "point 1: shot numbers are not 1..2 in order",
        "point 2: shots are not in time order",
        "point 3: a serve that is not its first shot",
        "point 4: mixes play between points with rally shots",
    ]
    assert assemble.validate_export(RawMatchExport(None, shots=shots)) == [
        "no settings: shots cannot be told apart by player"
    ]


# --------------------------------------------------------------------------
# Notes, the tracking block and the clock
# --------------------------------------------------------------------------


def test_the_notes_count_against_their_denominators():
    missed = P.Piece([(P.hit(0.0, NEAR, 3.0), FOREHAND), (P.hit(1.5, FAR, -2.0), FOREHAND)])
    match = made(P.rally(NEAR, "deuce", 3, end="unknown"), missed)
    notes = match.notes
    assert notes[0] == (
        "Tracked 5 of 5 shots with confidence (a hit seen or bridged, not inferred, at "
        "confidence 0.40 or more)."
    )
    assert notes[1] == "2 serve attempts: the serve was seen on 1 and missed on 1."
    assert notes[2].startswith("1 serve row(s) were inferred, not seen")
    assert any(n.startswith("2 of 5 shot results were guessed") for n in notes)
    assert not any("far end is weak" in n for n in notes)


def test_a_weak_far_end_is_noted():
    piece = P.rally(NEAR, "deuce", 4)
    weak = [P.hit(h.pts, h.side, confidence=0.2) if h.side == FAR else h for h, _ in piece.hits]
    piece.hits = [(h, s) for h, (_, s) in zip(weak, piece.hits)]
    match = made(piece)
    assert any(n.startswith("The far end is weak: 0 far-end hits") for n in match.notes)


def test_weak_ball_tracking_and_many_unknown_results_are_noted():
    frames = np.arange(600)
    ball = BallTrack((), (), (), frames, 10.0 + frames / 30.0, SIZE, "motion")
    piece = P.Piece(
        [(P.hit(0.0, NEAR), SERVE)]
        + [(P.hit(t, FAR if n % 2 else NEAR, 2.0), FOREHAND) for n, t in enumerate((5.0, 6.5))]
    )
    match = made(piece, ball=ball)
    assert any(n.startswith("Ball tracking was weak here") for n in match.notes)
    assert "Ball tracking was weak on 1 of 1 points." in match.notes
    assert any("shots have no result" in n for n in match.notes)


def test_calibration_warnings_and_quality_reach_the_notes_and_summary(calibration):
    warned = replace(calibration, quality=replace(calibration.quality, warnings=("Raise it.",)))
    match = made(P.rally(NEAR, "deuce", 2), calibration=warned)
    assert match.notes[-1] == "Raise it."
    assert match.summary["calibration"]["warnings"] == ["Raise it."]


def test_the_tracking_block_is_plain_json_with_versions_and_provenance():
    match = assemble.assemble(*golden_events(), SETUP, players=swap(1000.0, Provenance.MEASURED))
    block = json.loads(json.dumps(match.tracking_block()))
    summary = block["summary"]
    assert summary["video_clock"] == "pts"
    assert summary["shots"] == 11 and summary["tracked_shots"] == 9
    assert summary["attempts"]["faults"] == 1
    assert summary["guesses"] == {
        "result": 0,
        "stroke": 3,
        "serve_row": 0,
        "settled_by_scoring": 0,
    }
    assert summary["provenance"]["stroke"] == {
        "measured": 7,
        "interpolated": 0,
        "inferred": 4,  # the relabelled serve and three unclear strokes
        "user_entered": 0,
    }
    assert summary["models"] == {
        "ball": "none",
        "events": "events-v1",
        "strokes": ["segment-v1", "test"],
        "players": "fake-players",
        "segment": "segment-v1",
        "endings": "endings-v1",
        "assemble": "assemble-v1",
    }
    assert summary["calibration"] is None and summary["player_quality"]["end_swaps"] == 1
    assert block["notes"] == list(match.notes)
    first = block["attempts"][1]
    assert first["server"] == "Host" and first["kind"] == "served"
    assert first["shots"][0]["fields"]["hit_xy"]["value"] == [0.5, -0.4]


def test_each_reconstructed_point_gets_its_block_with_the_fault_it_absorbed():
    match = golden()
    result = reconstruct.reconstruct_all(match.export.shots, "Host")
    blocks = match.point_blocks(result)
    assert sorted(blocks) == [2, 4, 5]
    assert blocks[4]["second_serve"] is True
    assert blocks[4]["folded_fault"]["number"] == 3
    assert blocks[4]["folded_fault"]["fault"] is True
    assert "folded_fault" not in blocks[2]
    json.dumps(blocks)


def test_blocks_for_a_result_from_some_other_export_are_left_out():
    match = golden()
    stray = ReconstructedPoint(99, True, True, True, None, "ace", False)
    result = ReconstructionResult([], [], [], [stray], merged_fault_points=[98, 1000])
    assert match.point_blocks(result) == {}
    assert match.attempt_block(99) is None


def test_tracker_times_need_no_clock_offset():
    offset = assemble.video_clock_offset(None, [1.0] * 25)
    assert offset.seconds == 0.0 and offset.trusted
    assert not assemble.video_clock_offset(None, [1.0]).trusted


def test_segmentation_and_assembly_share_the_configured_thresholds():
    config = TrackerConfig(segment=replace(TrackerConfig().segment, serve_max_x_m=0.2))
    events, labels = P.build(P.rally(NEAR, "deuce", 2))
    match = assemble.assemble(events, labels, SETUP, config=config)
    assert match.segmentation.attempts[0].kind == segment.SERVE_UNSEEN
    # The near server, too wide to be one, is taken for a returner: the far
    # player's serve is inferred in front of it.
    shots = match.export.shots
    assert [r.shot_type for r in shots] == ["first_serve", "first_return", "serve_plus_one"]
    assert shots[0].player == "Guest" and shots[0].hit_side == "far"


def test_assembly_passes_the_tracking_start_through():
    frames = np.arange(300)
    ball = BallTrack((), (), (), frames, 9.9 + frames / 30.0, SIZE, "motion")
    events, _ = P.build(P.serve(NEAR, "ad", landing="unseen"), ball=ball)
    labels = (P.label(0, SERVE, 0.45, Provenance.INFERRED),)
    assert assemble.assemble(events, labels, SETUP).export.shots[0].shot_type == "none"
    told = assemble.assemble(events, labels, SETUP, tracking_start=0.0)
    assert told.export.shots[0].shot_type == "first_serve"


# --------------------------------------------------------------------------
# Missed serves: an inferred serve row
# --------------------------------------------------------------------------


def missed_serve(side, returner_x=2.5, shots=3):
    """A rally whose serve (by the other end) was not seen: it opens with the return."""
    piece = P.Piece()
    hitter = side
    for n in range(shots):
        x = returner_x if n == 0 else (-2.5 if n % 2 else 2.5)
        piece.hits.append((P.hit(1.5 * n, hitter, x, P.BASELINE_Y[hitter]), FOREHAND))
        hitter = P.other(hitter)
    piece.nets.append(P.net(1.5 * (shots - 1) + 0.4))
    return piece


def test_a_missed_serve_gets_an_inferred_serve_row_so_the_server_is_right():
    match = made(missed_serve(FAR, returner_x=-2.5))  # the far receiver in their deuce half
    serve, back = match.export.shots[:2]
    assert (serve.shot_number, serve.shot_type, serve.stroke, serve.result) == (
        1,
        "first_serve",
        "Serve",
        "In",
    )
    assert (serve.player, serve.hit_side, serve.hit_zone) == ("Host", "near", "deuce")
    assert serve.video_time == pytest.approx(10.0 - 1.14)
    assert serve.hit_x is None and serve.bounce_x is None and serve.speed_mph is None
    assert (back.shot_number, back.shot_type) == (2, "first_return")
    trace = trace_of(match, 1, 1)
    assert trace.event_index is None and not trace.tracked and trace.guesses == ("serve_row",)
    assert all(f.provenance is Provenance.INFERRED for f in trace.fields.values())
    assert "serve not seen: inferred from the return at 10.00s" in trace.notes[0]
    assert assemble.validate_export(match.export) == []
    result = reconstruct.reconstruct_all(match.export.shots, "Host")
    assert result.points[0].is_serving  # the host served, as the far return implies


def test_a_seen_fault_folds_into_a_second_serve_inferred_from_its_return():
    match = made(P.serve(NEAR, "deuce", landing="net"), (6.0, missed_serve(FAR, -2.5)))
    second = match.export.shots[1]
    assert (second.shot_type, second.hit_zone) == ("second_serve", "deuce")
    assert match.export.shots[2].shot_type == "second_return"
    result = reconstruct.reconstruct_all(match.export.shots, "Host")
    assert result.merged_fault_points == [1]
    assert result.points[0].first_serve_in is False and result.points[0].second_serve_in


def test_the_inferred_serve_row_reads_back_through_the_loader():
    match = made(missed_serve(FAR))
    block = json.loads(json.dumps(match.tracking_block()))
    back = assemble.TrackingBlock.from_dict(block)
    (attempt,) = back.attempts
    assert attempt.serve_estimate == pytest.approx(10.0 - 1.14)
    assert attempt.shots[0].event_index is None
    assert attempt.shots[0] == match.traces[0]
    assert attempt.to_dict() == block["attempts"][0]


# --------------------------------------------------------------------------
# Endings settled by scoring
# --------------------------------------------------------------------------


def test_scoring_settles_endings_the_tracker_did_not_see():
    hold_near = hold(NEAR, "A")
    unseen = [
        P.serve(FAR, "deuce", person="B"),
        P.rally(FAR, "ad", 3, end="unknown", person="B"),
        P.serve(FAR, "deuce", person="B"),
        P.rally(FAR, "ad", 3, end="unknown", person="B"),
    ]
    match = made(*hold_near, *unseen, *hold_near)
    # The guest's four-point game: two aces were seen, so all four were theirs.
    rallies = [r for r in match.export.shots if r.point_number in (6, 8)]
    lasts = [r for r in rallies if r.shot_number == 3]
    assert [r.result for r in lasts] == ["In", "In"]  # the guest's own last shots won
    trace = trace_of(match, 6, 3)
    assert trace.fields["result"].source == "endings-v1"
    assert "settled by game scoring: Guest must have won" in trace.notes[-1]
    assert "result" in trace.guesses  # still not seen
    assert match.summary["guesses"]["settled_by_scoring"] == 2
    assert any(n.startswith("2 of those unseen point endings were settled") for n in match.notes)
    result = reconstruct.reconstruct_all(match.export.shots, "Host")
    assert [(s.games_won, s.games_lost) for s in result.sets] == [(2, 1)]


def test_a_game_that_does_not_add_up_keeps_its_guesses_and_says_so():
    odd = [
        P.serve(FAR, "ad", person="B"),  # a game cannot start from the ad side
        P.rally(FAR, "deuce", 3, end="unknown", person="B"),
        P.serve(FAR, "ad", person="B"),
        P.serve(FAR, "deuce", person="B"),
    ]
    match = made(*hold(NEAR, "A"), *odd, *hold(NEAR, "A"))
    assert match.summary["guesses"]["settled_by_scoring"] == 0
    assert any(n.startswith("1 game(s) do not add up as tracked") for n in match.notes)
    assert trace_of(match, 6, 3).fields["result"].source == "assemble-v1"


def test_a_double_fault_the_rules_say_was_lost_stays_an_error():
    unseen = [
        P.serve(FAR, "deuce", person="B", landing="net"),
        (6.0, P.serve(FAR, "deuce", person="B", landing="unseen")),
        P.rally(FAR, "ad", 3, end="net", person="B"),
        P.rally(FAR, "deuce", 3, end="net", person="B"),
        P.rally(FAR, "ad", 3, end="net", person="B"),
    ]
    match = made(*hold(NEAR, "A"), *unseen, *hold(NEAR, "A"))
    # The guest lost three seen points, so the game (and the double fault) went to the host.
    second = next(r for r in match.export.shots if r.shot_type == "second_serve")
    assert second.result == "Net"
    assert "settled by game scoring: Host must have won" in trace_of(
        match, second.point_number, 1
    ).notes[-1]
