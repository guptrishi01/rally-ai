"""Hand-built event tracks for the segmentation and assembly tests.

No ball is flown and nothing is rendered: hits, bounces and net contacts
are placed by hand on the video's clock and joined into shots by the
events stage's own `_assemble`, so results come out exactly as the real
stage would judge them. A `Piece` is one serve attempt (or any stretch of
play); `build` lays pieces end to end with realistic silences between.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from tracking import events
from tracking.ball import BallTrack
from tracking.config import EventConfig
from tracking.events import BounceEvent, EventTrack, HitEvent, NetEvent
from tracking.players import FAR, NEAR, side_of, zone_of
from tracking.provenance import Provenance, Tagged
from tracking.strokes import FOREHAND, SERVE, UNKNOWN, StrokeLabel

from .conftest import SIZE

EMPTY_BALL = BallTrack((), (), (), np.zeros(0, dtype=int), np.zeros(0), SIZE, "none")
SERVER_X = {NEAR: {"deuce": 0.5, "ad": -0.5}, FAR: {"deuce": -0.5, "ad": 0.5}}
BASELINE_Y = {NEAR: -0.4, FAR: 24.2}
# Where a serve from each side lands in: the box diagonally opposite.
SERVE_LANDING = {
    NEAR: {"deuce": (-2.0, 16.5), "ad": (2.0, 16.5)},
    FAR: {"deuce": (2.0, 7.5), "ad": (-2.0, 7.5)},
}


def other(side: str) -> str:
    """The other end."""
    return FAR if side == NEAR else NEAR


def hit(
    pts,
    side,
    x=None,
    y=None,
    *,
    confidence=0.75,
    provenance=Provenance.MEASURED,
    person=None,
    seen=True,
    clipped=False,
    cues=("kink",),
):
    """A hand-made hit; by default a server's, behind the baseline near the centre."""
    x = SERVER_X[side]["deuce"] if x is None else x
    y = BASELINE_Y[side] if y is None else y
    return HitEvent(
        pts=pts,
        time_uncertainty=0.03,
        side=side,
        player=person or ("A" if side == NEAR else "B"),
        zone=zone_of(side, x) if seen else None,
        position=(x, y) if seen else None,
        position_provenance=Provenance.MEASURED if seen else None,
        foot_clipped=clipped,
        image_xy=None,
        confidence=confidence,
        provenance=provenance,
        cues=cues,
    )


def bounce(pts, x, y, confidence=0.8, provenance=Provenance.MEASURED):
    """A seen bounce, judged with 0.1m of uncertainty."""
    return BounceEvent(
        pts, 0.02, (x, y), (0.1, 0.1), side_of(y), (0.0, 0.0), confidence, provenance
    )


def net(pts):
    """A seen stop at the net."""
    return NetEvent(pts, (0.0, 0.0), 0.7, Provenance.MEASURED)


def label(index, stroke, confidence=0.7, provenance=Provenance.MEASURED):
    """A stroke label; an unknown stroke carries no confidence."""
    sure = None if stroke == UNKNOWN else confidence
    return StrokeLabel(index, Tagged(stroke, provenance, sure, "test"), None, f"test {stroke}")


@dataclass
class Piece:
    """A stretch of play, timed from its own start.

    Attributes:
        hits: (hit, stroke) pairs.
        bounces: Bounces.
        nets: Net contacts.
    """

    hits: list[tuple[HitEvent, str]] = field(default_factory=list)
    bounces: list[BounceEvent] = field(default_factory=list)
    nets: list[NetEvent] = field(default_factory=list)

    @property
    def length(self) -> float:
        """Seconds from the start to the last event."""
        times = [h.pts for h, _ in self.hits] + [b.pts for b in self.bounces]
        return max(times + [n.pts for n in self.nets], default=0.0)

    def shifted(self, by: float) -> Piece:
        """The same piece, starting `by` seconds later."""
        return Piece(
            [(_moved(h, by), s) for h, s in self.hits],
            [_moved(b, by) for b in self.bounces],
            [_moved(n, by) for n in self.nets],
        )


def _moved(event, by):
    """An event `by` seconds later."""
    return replace(event, pts=event.pts + by)


def serve(side, zone, *, landing="in", stroke=SERVE, person=None, **more) -> Piece:
    """A serve attempt that is only a serve.

    Args:
        side: The server's end.
        zone: "deuce" or "ad".
        landing: "in" (in the box, seen), "out" (long, seen), "net",
            or "unseen".
        stroke: Its label.
        person: The server, "A" or "B".
        **more: Passed to `hit`.
    """
    piece = Piece([(hit(0.0, side, SERVER_X[side][zone], person=person, **more), stroke)])
    x, y = SERVE_LANDING[side][zone]
    if landing == "in":
        piece.bounces.append(bounce(0.6, x, y))
    elif landing == "out":
        piece.bounces.append(bounce(0.6, x, 21.0 if side == NEAR else 2.7))
    elif landing == "net":
        piece.nets.append(net(0.4))
    return piece


def rally(side, zone, shots, *, end="net", gap=1.5, serve_landing="in", person=None) -> Piece:
    """A served rally of `shots` hits, alternating ends from the server.

    Args:
        side: The server's end.
        zone: The side served from.
        shots: Hits in all, the serve included (at least 1).
        end: How the last shot ended: "net", "out", "winner" or "unknown".
        gap: Seconds between hits.
        serve_landing: As `serve`'s `landing`, for the serve.
        person: The server, "A" or "B".
    """
    piece = serve(side, zone, landing=serve_landing, person=person)
    hitter = side
    returner = {"A": "B", "B": "A"}.get(person or "")
    for n in range(1, shots):
        hitter = other(hitter)
        x = 2.5 if n % 2 else -2.5
        who = (person if hitter == side else returner) if person else None
        piece.hits.append((hit(n * gap, hitter, x, BASELINE_Y[hitter], person=who), FOREHAND))
    last = piece.hits[-1][0]
    if shots == 1:
        return piece
    t = last.pts
    deep = 22.0 if last.side == NEAR else 2.0
    if end == "net":
        piece.nets.append(net(t + 0.4))
    elif end == "out":
        piece.bounces.append(bounce(t + 0.8, 0.0, 25.5 if last.side == NEAR else -1.5))
    elif end == "winner":
        piece.bounces.append(bounce(t + 0.8, 1.0, deep))
    return piece


def build(*pieces, start=10.0, between=25.0, ball=None):
    """Lays pieces end to end and joins them into events and labels.

    Args:
        *pieces: Pieces, or (silence before it, piece) pairs to override
            the silence.
        start: When the first piece starts.
        between: Default silence between pieces, seconds.
        ball: A ball track to attach; an empty one if omitted.

    Returns:
        (EventTrack, labels).
    """
    hits, bounces, nets = [], [], []
    clock = start
    for n, item in enumerate(pieces):
        gap, piece = item if isinstance(item, tuple) else (between, item)
        if n:
            clock += gap
        moved = piece.shifted(clock)
        hits.extend(moved.hits)
        bounces.extend(moved.bounces)
        nets.extend(moved.nets)
        clock += piece.length
    hits.sort(key=lambda pair: pair[0].pts)
    bounces.sort(key=lambda b: b.pts)
    nets.sort(key=lambda n: n.pts)
    shots, kept = events._assemble([h for h, _ in hits], bounces, nets, EventConfig())
    track = EventTrack(
        tuple(h for h, _ in hits), tuple(kept), tuple(nets), tuple(shots), ball or EMPTY_BALL
    )
    labels = tuple(label(i, stroke) for i, (_, stroke) in enumerate(hits))
    return track, labels

