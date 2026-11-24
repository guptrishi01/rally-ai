from __future__ import annotations

from swingvision_import.raw import RawShotRow
from swingvision_import.reconstruct import (
    ReconstructedPoint,
    assign_game_set_boundaries,
    build_shot_pattern_summary,
    group_shots_by_point,
    is_second_serve_attempt,
    merge_shots,
    reconstruct_all,
    reconstruct_point,
)

HOST = "Rishi Gupta"
GUEST = "Opponent"


def _shot(point, shot, player, shot_type, stroke, result) -> RawShotRow:
    return RawShotRow(point, shot, player, shot_type, stroke, result)


def test_reconstruct_point_clean_host_winner_at_the_serve():
    shots = [
        _shot(1, 1, HOST, "first_serve", "Serve", "In"),
        _shot(1, 2, GUEST, "first_return", "Forehand", "In"),
        _shot(1, 3, HOST, "serve_plus_one", "Forehand", "In"),
    ]

    point = reconstruct_point(1, shots, HOST)

    assert point.is_serving is True
    assert point.point_won is True
    assert point.point_end_type == "winner"


def test_reconstruct_point_host_unforced_error_after_own_serve():
    shots = [
        _shot(1, 1, HOST, "first_serve", "Serve", "In"),
        _shot(1, 2, GUEST, "first_return", "Forehand", "In"),
        _shot(1, 3, HOST, "in_play", "Backhand", "Net"),
    ]

    point = reconstruct_point(1, shots, HOST)

    assert point.point_won is False
    assert point.point_end_type == "unforced_error"


def test_reconstruct_point_ace():
    shots = [_shot(5, 1, HOST, "first_serve", "Serve", "In")]

    point = reconstruct_point(5, shots, HOST)

    assert point.point_end_type == "ace"
    assert point.point_won is True


def test_reconstruct_point_double_fault_ignores_the_ambiguous_trailing_shot():
    # Mirrors the real, confirmed-ambiguous pattern: a second_serve marked
    # Out (a fault) followed by a second_return shot in the same point,
    # which shouldn't happen if the serve genuinely faulted. The serve
    # shot's own Result is authoritative regardless.
    shots = [
        _shot(6, 1, HOST, "second_serve", "Serve", "Out"),
        _shot(6, 2, GUEST, "second_return", "Backhand", "In"),
    ]

    point = reconstruct_point(6, shots, HOST)

    assert point.point_end_type == "double_fault"
    assert point.point_won is False


def test_reconstruct_point_net_approach_via_volley_stroke():
    shots = [
        _shot(1, 1, GUEST, "first_serve", "Serve", "In"),
        _shot(1, 2, HOST, "first_return", "Forehand", "In"),
        _shot(1, 3, GUEST, "serve_plus_one", "Forehand", "In"),
        _shot(1, 4, HOST, "in_play", "Forehand Volley", "In"),
    ]

    point = reconstruct_point(1, shots, HOST)

    assert point.net_approach is True


def test_reconstruct_point_no_net_approach_without_a_volley_or_overhead():
    shots = [
        _shot(1, 1, HOST, "first_serve", "Serve", "In"),
        _shot(1, 2, GUEST, "first_return", "Forehand", "Net"),
    ]

    point = reconstruct_point(1, shots, HOST)

    assert point.net_approach is False


def test_reconstruct_point_returns_none_for_a_gap():
    assert reconstruct_point(4, [], HOST) is None


def test_reconstruct_point_return_winner_when_the_point_ends_at_the_return():
    shots = [
        _shot(1, 1, GUEST, "first_serve", "Serve", "In"),
        _shot(1, 2, HOST, "first_return", "Forehand", "In"),
    ]

    point = reconstruct_point(1, shots, HOST)

    assert point.is_serving is False
    assert point.point_end_type == "return_winner"


def test_reconstruct_point_return_error_when_the_return_itself_misses():
    shots = [
        _shot(1, 1, GUEST, "first_serve", "Serve", "In"),
        _shot(1, 2, HOST, "first_return", "Forehand", "Net"),
    ]

    point = reconstruct_point(1, shots, HOST)

    assert point.point_end_type == "return_error"


def test_reconstruct_point_uses_generic_winner_not_return_winner_past_the_return():
    shots = [
        _shot(1, 1, GUEST, "first_serve", "Serve", "In"),
        _shot(1, 2, HOST, "first_return", "Forehand", "In"),
        _shot(1, 3, GUEST, "serve_plus_one", "Forehand", "In"),
        _shot(1, 4, HOST, "return_plus_one", "Backhand", "In"),
    ]

    point = reconstruct_point(1, shots, HOST)

    assert point.point_end_type == "winner"


def test_group_shots_by_point_sorts_by_shot_number():
    shots = [
        _shot(1, 2, HOST, "first_return", "Forehand", "In"),
        _shot(1, 1, GUEST, "first_serve", "Serve", "In"),
        _shot(2, 1, HOST, "first_serve", "Serve", "In"),
    ]

    grouped = group_shots_by_point(shots)

    assert [s.shot_number for s in grouped[1]] == [1, 2]
    assert list(grouped[2]) == [shots[2]]


def _point(point_won: bool) -> ReconstructedPoint:
    return ReconstructedPoint(
        point_number=0,
        is_serving=True,
        point_won=point_won,
        first_serve_in=True,
        second_serve_in=None,
        point_end_type="winner" if point_won else "unforced_error",
        net_approach=False,
    )


def _quick_game(host_wins: bool) -> list[ReconstructedPoint]:
    """4 points, all won by one side - the fastest possible game."""
    return [_point(host_wins) for _ in range(4)]


def test_assign_game_set_boundaries_numbers_points_within_a_game():
    sets = assign_game_set_boundaries(_quick_game(host_wins=True))

    assert len(sets) == 1
    assert sets[0].games_won == 1
    assert sets[0].games_lost == 0
    assert [p.point_number for p in sets[0].points] == [1, 2, 3, 4]
    assert all(p.game_number == 1 for p in sets[0].points)
    assert all(not p.is_tiebreak_game for p in sets[0].points)


def test_assign_game_set_boundaries_handles_a_deuce_game():
    # 3-3 (deuce), then host wins the next two points to close the game.
    points = [_point(True), _point(False), _point(True), _point(False), _point(True), _point(False)]
    points += [_point(True), _point(True)]

    sets = assign_game_set_boundaries(points)

    assert len(sets) == 1
    assert sets[0].games_won == 1
    assert sets[0].games_lost == 0
    assert len(sets[0].points) == 8


def test_assign_game_set_boundaries_flags_points_in_a_six_all_tiebreak():
    # 6 games to host, 6 to guest (4 points each, alternating), then a
    # tiebreak-game's worth of points on top.
    points: list[ReconstructedPoint] = []
    for i in range(12):
        points += _quick_game(host_wins=(i % 2 == 0))
    tiebreak_points = [_point(True) for _ in range(7)]
    points += tiebreak_points

    sets = assign_game_set_boundaries(points)

    # 12 regular games (6-6) plus a completed 7-0 tiebreak game -> the set
    # closes 7-6 to host.
    assert len(sets) == 1
    assert sets[0].games_won == 7
    assert sets[0].games_lost == 6
    assert sets[0].is_tiebreak_set is True
    tiebreak_game_points = [p for p in sets[0].points if p.game_number == 13]
    assert len(tiebreak_game_points) == 7
    assert all(p.is_tiebreak_game for p in tiebreak_game_points)
    non_tiebreak_points = [p for p in sets[0].points if p.game_number != 13]
    assert all(not p.is_tiebreak_game for p in non_tiebreak_points)


def test_assign_game_set_boundaries_leaves_a_trailing_partial_game_uncounted_as_a_set():
    # Only 2 points played in game 1 - an interrupted recording. Still
    # returned as a (partial) set rather than dropped.
    sets = assign_game_set_boundaries([_point(True), _point(False)])

    assert len(sets) == 1
    assert len(sets[0].points) == 2
    assert sets[0].games_won == 0
    assert sets[0].games_lost == 0


def test_assign_game_set_boundaries_no_ad_scoring_wins_at_four_regardless_of_margin():
    # 3-3 (deuce under ad-scoring), then one more point - under no-ad this
    # wins the game outright at 4-3, not requiring a 2-point margin.
    points = [_point(True), _point(False), _point(True), _point(False), _point(True), _point(False)]
    points.append(_point(True))

    sets = assign_game_set_boundaries(points, ad_scoring=False)

    assert len(sets) == 1
    assert sets[0].games_won == 1
    assert len(sets[0].points) == 7


def test_reconstruct_all_skips_gaps_and_reports_them():
    shots = [
        _shot(1, 1, HOST, "first_serve", "Serve", "In"),
        # Point 2: a gap, no shots at all.
        _shot(3, 1, HOST, "first_serve", "Serve", "In"),
    ]

    result = reconstruct_all(shots, HOST)

    assert result.skipped_points == [2]
    assert result.excluded_points == []
    total_points = sum(len(s.points) for s in result.sets)
    assert total_points == 2


def test_reconstruct_all_excludes_points_with_no_rally_shot_but_does_not_skip_them():
    # Point 2 here is exactly the real pattern found in production data: the
    # host feeds a ball across (Type=="none") and the guest nets it back -
    # never a served point, so it must not be treated as a gap or as a real
    # point that shifts subsequent point/game numbering.
    shots = [
        _shot(1, 1, HOST, "first_serve", "Serve", "In"),
        _shot(1, 2, GUEST, "first_return", "Forehand", "Net"),
        _shot(2, 0, HOST, "none", "Feed", "In"),
        _shot(2, 1, GUEST, "none", "Backhand", "Net"),
        _shot(3, 1, HOST, "first_serve", "Serve", "In"),
        _shot(3, 2, GUEST, "first_return", "Forehand", "Net"),
    ]

    result = reconstruct_all(shots, HOST)

    assert result.excluded_points == [2]
    assert result.skipped_points == []
    total_points = sum(len(s.points) for s in result.sets)
    assert total_points == 2


def _reconstructed_point(point_number: int, point_won: bool) -> ReconstructedPoint:
    return ReconstructedPoint(
        point_number=point_number,
        is_serving=True,
        point_won=point_won,
        first_serve_in=True,
        second_serve_in=None,
        point_end_type="winner" if point_won else "unforced_error",
        net_approach=False,
    )


def test_build_shot_pattern_summary_returns_none_for_no_points():
    assert build_shot_pattern_summary([], {}) is None


def test_build_shot_pattern_summary_computes_rally_length_and_win_rate_split():
    points = [
        _reconstructed_point(1, point_won=True),  # 2 shots -> short, won
        _reconstructed_point(2, point_won=False),  # 3 shots -> short, lost
        _reconstructed_point(3, point_won=True),  # 6 shots -> long, won
        _reconstructed_point(4, point_won=False),  # 8 shots -> long, lost
    ]
    shots_by_point = {
        1: [_shot(1, i, HOST, "in_play", "Forehand", "In") for i in range(2)],
        2: [_shot(2, i, HOST, "in_play", "Forehand", "In") for i in range(3)],
        3: [_shot(3, i, HOST, "in_play", "Forehand", "In") for i in range(6)],
        4: [_shot(4, i, HOST, "in_play", "Forehand", "In") for i in range(8)],
    }

    summary = build_shot_pattern_summary(points, shots_by_point)

    assert summary == {
        "avg_rally_length": 4.75,
        "rally_win_rate_short": 50.0,
        "rally_win_rate_long": 50.0,
    }


def test_build_shot_pattern_summary_treats_a_missing_shot_lookup_as_zero_length():
    points = [_reconstructed_point(1, point_won=True)]

    summary = build_shot_pattern_summary(points, shots_by_point={})

    assert summary["avg_rally_length"] == 0.0
    assert summary["rally_win_rate_short"] == 100.0
    assert summary["rally_win_rate_long"] == 0.0


def test_merge_shots_shifts_later_files_point_numbers_to_continue_the_sequence():
    file1 = [
        _shot(1, 1, HOST, "first_serve", "Serve", "In"),
        _shot(2, 1, HOST, "first_serve", "Serve", "In"),
    ]
    file2 = [
        # This file's own Point counter restarts from 1, like a real
        # second-half export.
        _shot(1, 1, GUEST, "first_serve", "Serve", "In"),
        _shot(2, 1, GUEST, "first_serve", "Serve", "In"),
    ]

    merged = merge_shots([file1, file2])

    assert [s.point_number for s in merged] == [1, 2, 3, 4]
    # Original per-shot data is otherwise untouched.
    assert merged[2].player == GUEST


def test_merge_shots_with_a_single_file_leaves_point_numbers_unchanged():
    file1 = [_shot(1, 1, HOST, "first_serve", "Serve", "In")]

    merged = merge_shots([file1])

    assert [s.point_number for s in merged] == [1]


def test_merge_shots_handles_an_empty_file_without_disrupting_the_offset():
    file1 = [_shot(1, 1, HOST, "first_serve", "Serve", "In")]
    empty_file: list[RawShotRow] = []
    file2 = [_shot(1, 1, GUEST, "first_serve", "Serve", "In")]

    merged = merge_shots([file1, empty_file, file2])

    assert [s.point_number for s in merged] == [1, 2]


def _serve(point, player, shot_type, result, zone=None) -> RawShotRow:
    return RawShotRow(point, 1, player, shot_type, "Serve", result, hit_zone=zone)


def test_a_fault_and_the_serve_after_it_from_the_same_side_are_one_point():
    # The real pattern: SwingVision numbers each serve attempt as its own
    # "point", and here labels both attempts first_serve.
    shots = [
        _serve(1, HOST, "first_serve", "Out", "deuce"),
        _serve(2, HOST, "first_serve", "In", "deuce"),
        _shot(2, 2, GUEST, "first_return", "Forehand", "Net"),
        _serve(3, HOST, "first_serve", "In", "ad"),
    ]

    result = reconstruct_all(shots, HOST)

    assert result.merged_fault_points == [1]
    assert [p.point_number for p in result.points] == [2, 3]
    merged = result.points[0]
    assert merged.first_serve_in is False
    assert merged.second_serve_in is True
    assert merged.point_won is True


def test_a_fault_then_a_missed_second_serve_is_a_double_fault():
    shots = [
        _serve(1, HOST, "first_serve", "Net", "ad"),
        _serve(2, HOST, "second_serve", "Out", "ad"),
    ]

    result = reconstruct_all(shots, HOST)

    assert len(result.points) == 1
    assert result.points[0].point_end_type == "double_fault"
    assert result.points[0].point_won is False


def test_a_fault_returned_anyway_still_folds_into_the_second_serve():
    # A returner often hits a faulted serve back; the stray return must not
    # make the fault a point of its own.
    shots = [
        _serve(1, HOST, "first_serve", "Out", "deuce"),
        _shot(1, 2, GUEST, "first_return", "Backhand", "In"),
        _serve(2, HOST, "first_serve", "In", "deuce"),
    ]

    assert reconstruct_all(shots, HOST).merged_fault_points == [1]


def test_serves_from_alternating_sides_are_separate_points():
    shots = [
        _serve(1, HOST, "first_serve", "Out", "deuce"),
        _serve(2, HOST, "first_serve", "In", "ad"),
    ]

    result = reconstruct_all(shots, HOST)

    assert result.merged_fault_points == []
    assert len(result.points) == 2


def test_a_first_serve_that_landed_in_is_not_folded_into_the_next_serve():
    # Same side twice with the first serve in is a let or a tracking slip -
    # ambiguous, so it is left as two points for the reviewer.
    previous = [_serve(1, HOST, "first_serve", "In", "deuce")]
    current = [_serve(2, HOST, "first_serve", "In", "deuce")]

    assert is_second_serve_attempt(previous, current) is False


def test_swingvisions_own_second_serve_label_outweighs_an_in_first_serve():
    previous = [_serve(1, HOST, "first_serve", "In", "deuce")]
    current = [_serve(2, HOST, "second_serve", "In", "deuce")]

    assert is_second_serve_attempt(previous, current) is True


def test_different_servers_are_never_one_point():
    previous = [_serve(1, HOST, "first_serve", "Out", "deuce")]
    current = [_serve(2, GUEST, "second_serve", "In", "deuce")]

    assert is_second_serve_attempt(previous, current) is False


def test_attempts_from_different_export_parts_are_never_one_point():
    previous = [_serve(1, HOST, "first_serve", "Out", "deuce")]
    current = [
        RawShotRow(2, 1, HOST, "second_serve", "Serve", "In", hit_zone="deuce", source_part=1)
    ]

    assert is_second_serve_attempt(previous, current) is False


def test_without_a_serving_side_the_second_serve_label_decides():
    fault = [_serve(1, HOST, "first_serve", "Out")]

    assert is_second_serve_attempt(fault, [_serve(2, HOST, "second_serve", "In")]) is True
    assert is_second_serve_attempt(fault, [_serve(2, HOST, "first_serve", "In")]) is False


def test_an_attempt_with_no_serve_is_never_a_second_serve():
    previous = [_serve(1, HOST, "first_serve", "Out", "deuce")]
    current = [_shot(2, 1, GUEST, "in_play", "Forehand", "In")]

    assert is_second_serve_attempt(previous, current) is False


def test_a_third_attempt_starts_a_new_point():
    # Fault, fault (a double fault), then the next point's serve: only the
    # first pair is one point.
    shots = [
        _serve(1, HOST, "first_serve", "Out", "deuce"),
        _serve(2, HOST, "second_serve", "Out", "deuce"),
        _serve(3, HOST, "first_serve", "In", "ad"),
    ]

    result = reconstruct_all(shots, HOST)

    assert result.merged_fault_points == [1]
    assert [p.point_end_type for p in result.points] == ["double_fault", "ace"]
