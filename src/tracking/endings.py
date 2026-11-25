"""Endings: settling unseen point winners from the rules of scoring.

The tracker often cannot see how a point ended - far-end bounces and balls
dying in the net are mostly lost. But a game is not a free sequence of
points: one player serves all of it, it ends the moment someone reaches
four points two clear, and the server alternates deuce and ad sides from
deuce. So a game's length and the points whose winner *was* seen can leave
exactly one way for the unseen ones to have gone: in a four-point game
every point went to the game's winner; in a five-point game where the
server won two seen points, the server won the game and lost one of the
other three.

`settle_games` enumerates every assignment of the unknown winners and keeps
an unknown only when all the assignments that make a legal game agree on
it. Nothing is settled in a game that

* is not bounded by a change of server on both sides - the first and last
  games of a stretch may be cut off by the recording or by a missed serve;
* has fewer than four points (a tiebreak's run of one server, or a cut
  short game) or more unknowns than `max_unknown`;
* serves from the wrong side somewhere (deuce, ad, deuce... from the first
  point) - a point was missed or added, so the count is wrong;
* has no legal completion at all - the same.

Those are reported, not forced. A tiebreak changes server every two
points, so its "games" are one or two points long and are never touched.
Everything settled here is a conclusion from rules, tagged inferred, never
a sighting; and it is only as right as the count of points, which the
review screen shows.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SOURCE = "endings-v1"
DEUCE_SIDE = "deuce"
AD_SIDE = "ad"


@dataclass(frozen=True)
class PointFacts:
    """What is known about one point, for scoring.

    Attributes:
        key: The caller's identifier for the point.
        server: Who served it ("A" or "B"); None when unknown.
        zone: The side served from, "deuce" or "ad"; None when unknown.
        server_won: Whether the server won it; None when not seen.
    """

    key: int
    server: str | None
    zone: str | None
    server_won: bool | None


@dataclass(frozen=True)
class GameVerdict:
    """What scoring said about one game.

    Attributes:
        keys: The game's points, in order.
        settled: Point key -> whether the server won, for every unknown
            point the rules decide.
        reason: Why nothing (or not everything) was settled; "" when the
            game was checked and every decidable unknown was decided.
    """

    keys: tuple[int, ...]
    settled: dict[int, bool]
    reason: str = ""


def game_over_at(server_won: Sequence[bool], ad_scoring: bool = True) -> int | None:
    """Says after which point a game ended.

    Args:
        server_won: Each point's winner, True for the server.
        ad_scoring: Two clear after deuce (True) or sudden death at 3-3.

    Returns:
        The 1-based number of the point that ended the game, or None if
        it had not ended.
    """
    mine = theirs = 0
    for number, won in enumerate(server_won, start=1):
        mine += won
        theirs += not won
        lead = max(mine, theirs)
        if lead >= 4 and (abs(mine - theirs) >= 2 or not ad_scoring):
            return number
    return None


def legal_game(server_won: Sequence[bool], ad_scoring: bool = True) -> bool:
    """Whether a sequence of points is exactly one complete game.

    Args:
        server_won: Each point's winner, True for the server.
        ad_scoring: As `game_over_at`.

    Returns:
        True when the game ends on its last point and not before.
    """
    return game_over_at(server_won, ad_scoring) == len(server_won)


def settle_game(
    points: Sequence[PointFacts], ad_scoring: bool = True, max_unknown: int = 12
) -> GameVerdict:
    """Decides what the rules allow about one game's unseen winners.

    Args:
        points: The game's points, in order, all served by one player.
        ad_scoring: As `game_over_at`.
        max_unknown: More unknown points than this are not enumerated.

    Returns:
        The settled winners, or the reason none could be.
    """
    keys = tuple(p.key for p in points)
    unknown = [i for i, p in enumerate(points) if p.server_won is None]
    if not unknown:
        return GameVerdict(keys, {}, "every winner was seen")
    if len(points) < 4:
        return GameVerdict(keys, {}, f"a {len(points)}-point game cannot happen")
    if len(unknown) > max_unknown:
        return GameVerdict(keys, {}, f"{len(unknown)} unseen winners: too many to check")
    for i, point in enumerate(points):
        expected = DEUCE_SIDE if i % 2 == 0 else AD_SIDE
        if point.zone is not None and point.zone != expected:
            return GameVerdict(
                keys, {}, f"point {i + 1} was served from the {point.zone} side: a point is missing"
            )
    outcomes = []
    for guess in itertools.product((True, False), repeat=len(unknown)):
        winners = [p.server_won for p in points]
        for i, won in zip(unknown, guess):
            winners[i] = won
        if legal_game(winners, ad_scoring):
            outcomes.append(guess)
    if not outcomes:
        return GameVerdict(keys, {}, "no way for these points to make one game")
    settled = {
        points[i].key: outcomes[0][n]
        for n, i in enumerate(unknown)
        if all(outcome[n] == outcomes[0][n] for outcome in outcomes)
    }
    left = len(unknown) - len(settled)
    reason = "" if not left else f"{left} unseen winner(s) could go either way"
    return GameVerdict(keys, settled, reason)


def games(points: Sequence[PointFacts]) -> list[list[PointFacts]]:
    """Splits points into runs served by one player - games, or pieces of one.

    Args:
        points: Every point in order (faults already folded).

    Returns:
        Runs of consecutive points with the same known server; a point
        with no known server ends a run and belongs to none.
    """
    runs: list[list[PointFacts]] = []
    current: list[PointFacts] = []
    for point in points:
        if point.server is None:
            if current:
                runs.append(current)
            runs.append([])
            current = []
            continue
        if current and current[-1].server != point.server:
            runs.append(current)
            current = []
        current.append(point)
    if current:
        runs.append(current)
    return runs


def settle_games(
    points: Sequence[PointFacts], ad_scoring: bool = True, max_unknown: int = 12
) -> list[GameVerdict]:
    """Settles what scoring decides across a stretch of points.

    Args:
        points: Every point in order, faults folded into their second
            serve.
        ad_scoring: As `game_over_at`.
        max_unknown: As `settle_game`.

    Returns:
        One verdict per run of one server's points. The first and last
        runs, and any run next to a point with no known server, are not
        settled: they may be cut short.
    """
    runs = games(points)
    verdicts = []
    for n, run in enumerate(runs):
        if not run:
            continue
        keys = tuple(p.key for p in run)
        bounded = 0 < n < len(runs) - 1 and runs[n - 1] and runs[n + 1]
        if not bounded:
            verdicts.append(GameVerdict(keys, {}, "may be cut short: not between two others"))
            continue
        verdicts.append(settle_game(run, ad_scoring, max_unknown))
    settled = sum(len(v.settled) for v in verdicts)
    logger.info("Endings: %d unseen winner(s) settled by scoring", settled)
    return verdicts
