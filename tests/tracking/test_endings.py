"""Tests for settling unseen point winners from the rules of scoring."""

from __future__ import annotations

import pytest

from tracking import endings
from tracking.endings import PointFacts


def game(*won, server="A", zones=True, start=0):
    """A run of points by one server; None marks an unseen winner."""
    sides = ("deuce", "ad")
    return [
        PointFacts(start + i, server, sides[i % 2] if zones else None, w)
        for i, w in enumerate(won)
    ]


@pytest.mark.parametrize(
    "won, ad, ends",
    [
        ((True,) * 4, True, 4),
        ((True, False, True, True, True), True, 5),
        ((True, True, True, False, False, False, True), True, None),  # deuce, then advantage
        ((True, True, True, False, False, False, True, True), True, 8),
        ((True, True, True, False, False, False, True), False, 7),  # no-ad: sudden death
        ((True, True), True, None),
    ],
)
def test_a_game_ends_four_points_two_clear_or_at_sudden_death(won, ad, ends):
    assert endings.game_over_at(won, ad) == ends
    assert endings.legal_game(won, ad) == (ends == len(won))


def test_a_game_that_ended_early_is_not_one_legal_game():
    assert not endings.legal_game((True,) * 5)


def test_a_four_point_game_leaves_no_choice():
    verdict = endings.settle_game(game(True, None, True, None))
    assert verdict.settled == {1: True, 3: True}
    assert verdict.reason == ""


def test_a_five_point_game_settles_only_what_every_completion_agrees_on():
    verdict = endings.settle_game(game(True, True, None, None, None))
    # The server won two, so they won the game (the loser has at most one),
    # and the last point; one of the middle two went the other way.
    assert verdict.settled == {4: True}
    assert verdict.reason == "2 unseen winner(s) could go either way"


@pytest.mark.parametrize(
    "points, reason",
    [
        (game(True, True, True, True), "every winner was seen"),
        (game(True, None), "a 2-point game cannot happen"),
        (game(*([None] * 14)), "14 unseen winners: too many to check"),
        (game(True, None, True, True, zones=False)[:1] + [PointFacts(9, "A", "deuce", None)]
         + game(True, True, start=10), "point 2 was served from the deuce side"),
        (game(True, True, True, True, None), "no way for these points to make one game"),
    ],
)
def test_a_game_that_cannot_be_trusted_settles_nothing(points, reason):
    verdict = endings.settle_game(points)
    assert verdict.settled == {}
    assert reason in verdict.reason


def test_unknown_zones_do_not_count_against_a_game():
    assert endings.settle_game(game(True, None, True, True, zones=False)).settled == {1: True}


def test_points_split_into_runs_of_one_server():
    points = game(True, True, server="A") + game(None, start=5, server="B")
    points += [PointFacts(9, None, None, None)] + game(True, start=10, server="B")
    runs = endings.games(points)
    assert [[p.key for p in run] for run in runs] == [[0, 1], [5], [], [10]]


def test_only_games_bounded_by_other_games_are_settled():
    before = game(True, True, True, True, server="A")
    middle = game(None, True, True, True, server="B", start=10)
    after = game(None, None, None, None, server="A", start=20)
    verdicts = endings.settle_games(before + middle + after)
    assert [v.settled for v in verdicts] == [{}, {10: True}, {}]
    assert verdicts[0].reason.startswith("may be cut short")
    assert verdicts[2].reason.startswith("may be cut short")


def test_a_game_next_to_a_point_with_no_server_is_not_trusted():
    before = game(True, True, True, True, server="A")
    middle = game(None, True, True, True, server="B", start=10)
    nobody = [PointFacts(30, None, None, None)]
    verdicts = endings.settle_games(before + middle + nobody + before)
    assert all(v.settled == {} for v in verdicts)


def test_a_no_ad_game_settles_by_its_own_rule():
    points = game(True, True, True, False, False, False, None)
    assert endings.settle_game(points, ad_scoring=True).reason.startswith("no way")
    assert endings.settle_game(points, ad_scoring=False).settled == {}
    assert "either way" in endings.settle_game(points, ad_scoring=False).reason
