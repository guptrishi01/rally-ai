"""Segmentation: which hits are serves, and which point each hit belongs to.

`segment` reads the events (`events.detect_events`) and the stroke labels
(`strokes.classify_strokes`) and splits the hits into **serve attempts**,
numbered match-wide exactly as SwingVision's `Point` counter is: one number
per serve attempt, so a first-serve fault and the second serve that follows
are two numbers. `reconstruct.reconstruct_all` folds the fault back into
its point; this module only has to get the sequence and the roles right.

How a serve is found
--------------------
A hit is a serve candidate when the other player has not hit it
confidently for `SegmentConfig.serve_quiet_seconds`, the hitter's feet are
behind their baseline and near the centre line, and its stroke label is a
serve, an overhead or unknown (a named forehand or backhand after a quiet
spell is a return of a serve nobody saw, or a ball fed in). Each
candidate is scored from its label's confidence and the hit's.

A quiet spell nobody watched proves nothing: within `serve_quiet_seconds`
of the start of tracking (or `tracking_start`) only a contact seen struck
as a serve or overhead counts. On the real recording an inferred hit 0.12s
into a tracked stretch was otherwise read as a serve, then as a fault.

Candidates by the same player close together with no confident hit by the
other player between them are one serve, and the strongest is kept. That
rule exists because of what the real recording showed: on four of fifteen
windows the server's ball-bounce before serving was taken for a hit, the
events stage then inferred a 0.2-confidence "return" by the other player,
and the real serve that followed lost its quiet and was labelled an
overhead. The weak inferred hit does not break the quiet here, and the
later, stronger candidate wins.

Segmentation, not the classifier, has the last word on which hits are
serves: chosen serves are relabelled `Serve` (tagged inferred when the
classifier said otherwise) and the events are re-judged against the
service box accordingly (`strokes.apply_strokes`).

Points, gaps and play between points
------------------------------------
A serve's attempt runs through the hits that follow it until a silence of
`rally_gap_seconds` or the next serve. Hits outside every attempt are
grouped by the same silence. A group with confident hits by **both**
players (at least `min_unserved_hits`) after `match_start` is a point whose
serve was missed - kind `serve_unseen`; its first hit is taken for the
return. Anything else - a ball hit back to the server, the pre-serve
bounce, warm-up - is kind `excluded` and is emitted with shot type `none`,
so `reconstruct` reports it as a non-match point rather than scoring it.
No serve is ever invented: a `serve_unseen` point has no serve row, and
`reconstruct` then treats it as served by the guest - a known error,
counted in the notes.

Roles
-----
`shot_type` follows SwingVision's positional roles: serve, return, serve
+1, return +1, then `in_play`. A rally's position is counted **including**
hits the tracker missed: two hits in a row by the same player mean the
other player's hit between them was not seen, so the count moves on by
two. Far-end hits are found about a third of the time on the real
recording, so this matters.

Faults
------
A serve attempt is judged a first-serve fault when the next attempt is
the same player serving again from the same side within
`second_serve_max_seconds`, the server did not play on, and no seen
(or bridged) bounce put the serve in. The next attempt's serve is then a
`second_serve` (its return `second_return`) - the label `reconstruct`
uses to fold the two, alongside the serving side. A seen "In" blocks it,
as it does in `reconstruct`: a let is replayed from the same side.

Confidence
----------
Every attempt carries a confidence from 0 to 1 and the reasons that
lowered it: a contested serve, a boundary near the silence threshold,
weak inferred hits inside, a fault judged by time alone. It is a ranking
heuristic for review, not a probability. Every point still arrives
`needs_review=True`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from .config import SegmentConfig, StrokeConfig
from .court import COURT_LENGTH_M
from .events import RESULT_IN, EventTrack, Shot
from .players import FAR, NEAR, zone_of
from .provenance import Provenance, Tagged
from .strokes import OVERHEAD, SERVE, UNKNOWN, StrokeLabel, apply_strokes

logger = logging.getLogger(__name__)

SOURCE = "segment-v1"

FIRST_SERVE = "first_serve"
SECOND_SERVE = "second_serve"
FIRST_RETURN = "first_return"
SECOND_RETURN = "second_return"
SERVE_PLUS_ONE = "serve_plus_one"
RETURN_PLUS_ONE = "return_plus_one"
IN_PLAY = "in_play"
NONE = "none"
SHOT_TYPES = (
    FIRST_SERVE,
    SECOND_SERVE,
    FIRST_RETURN,
    SECOND_RETURN,
    SERVE_PLUS_ONE,
    RETURN_PLUS_ONE,
    IN_PLAY,
    NONE,
)

SERVED = "served"
SERVE_UNSEEN = "serve_unseen"
EXCLUDED = "excluded"
KINDS = (SERVED, SERVE_UNSEEN, EXCLUDED)

_BOUNDARY_DOUBT_SECONDS = 1.0
_CONTESTED = 0.85
_NEAR_THRESHOLD = 0.8
_WEAK_HIT = 0.9
_UNZONED = 0.85
_UNSERVED_BASE = 0.25
_MAX_WEAK_PENALTIES = 3
# A result read off a bounce - seen, or bridged across a few unseen frames -
# as opposed to one inferred from the other player hitting the ball back.
_BOUNCE_SEEN = (Provenance.MEASURED, Provenance.INTERPOLATED)


def role_for(position: int, second_serve: bool = False) -> str:
    """Names a shot's role from its place in the rally.

    Args:
        position: 0 for the serve, 1 for the return, and so on - counting
            hits the tracker missed.
        second_serve: Whether the rally was played off a second serve.

    Returns:
        One of `SHOT_TYPES` other than "none".

    Raises:
        ValueError: If `position` is negative.
    """
    if position < 0:
        raise ValueError(f"position must be 0 or more, got {position}.")
    if position == 0:
        return SECOND_SERVE if second_serve else FIRST_SERVE
    if position == 1:
        return SECOND_RETURN if second_serve else FIRST_RETURN
    return {2: SERVE_PLUS_ONE, 3: RETURN_PLUS_ONE}.get(position, IN_PLAY)


@dataclass(frozen=True)
class SegmentedShot:
    """One hit, placed in its serve attempt.

    Attributes:
        shot: The shot, as re-judged with this segmentation's serves.
        label: Its stroke label, relabelled where segmentation decided
            whether it is a serve; None when no label was given.
        shot_number: 1-based order among this attempt's emitted shots.
        role: Its `shot_type` - one of `SHOT_TYPES`.
        position: Its place in the rally counting missed hits (0 is the
            serve), or None for play between points.
    """

    shot: Shot
    label: StrokeLabel | None
    shot_number: int
    role: str
    position: int | None

    @property
    def stroke(self) -> str:
        """The stroke's name, "unknown" when there is no label."""
        return UNKNOWN if self.label is None else self.label.stroke.value

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            The shot's index and time, its number, role, place and stroke.
        """
        return {
            "index": self.shot.index,
            "pts": self.shot.hit.pts,
            "shot_number": self.shot_number,
            "role": self.role,
            "position": self.position,
            "stroke": self.stroke,
        }


@dataclass(frozen=True)
class Attempt:
    """One serve attempt - or one group of hits that is not one.

    Attributes:
        number: The match-wide point number, one per attempt, as
            SwingVision's `Point` counter.
        kind: "served", "serve_unseen" (a rally whose serve was missed)
            or "excluded" (play between points, warm-up).
        shots: Its hits, in order.
        server_side: The end the server was at; for a `serve_unseen`
            point, the end opposite the first hit. None when excluded.
        server: The serving person, "A" or "B"; None when unknown.
        serve_zone: "deuce" or "ad", the side served from; None when
            unknown or not served.
        zone_source: "feet" when the server's position named the side,
            "bounce" when the serve's landing did; None without a zone.
        second_serve: Whether this attempt is the second serve after a
            fault.
        fault: Whether this attempt's serve was judged a first-serve
            fault - the next attempt being its second serve.
        confidence: 0 to 1, a ranking heuristic (0 for excluded groups,
            which are not scored).
        reasons: What set or lowered the confidence, in words.
    """

    number: int
    kind: str
    shots: tuple[SegmentedShot, ...]
    server_side: str | None
    server: str | None
    serve_zone: str | None
    zone_source: str | None
    second_serve: bool = False
    fault: bool = False
    confidence: float = 0.0
    reasons: tuple[str, ...] = ()

    @property
    def start(self) -> float:
        """The first hit's time, seconds on the video's clock."""
        return self.shots[0].shot.hit.pts

    @property
    def end(self) -> float:
        """The last hit's time."""
        return self.shots[-1].shot.hit.pts

    @property
    def serve(self) -> SegmentedShot | None:
        """The serve, for a served attempt; None otherwise."""
        return self.shots[0] if self.kind == SERVED else None

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field, shots as their own dicts.
        """
        return {
            "number": self.number,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "server_side": self.server_side,
            "server": self.server,
            "serve_zone": self.serve_zone,
            "zone_source": self.zone_source,
            "second_serve": self.second_serve,
            "fault": self.fault,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "shots": [shot.to_dict() for shot in self.shots],
        }


@dataclass(frozen=True)
class Segmentation:
    """A stretch of play split into serve attempts.

    Attributes:
        attempts: Every attempt, in time order, numbered from 1.
        events: The events re-judged with the chosen serves.
        labels: The stroke labels, relabelled where segmentation decided
            whether a hit was a serve.
        config: The settings used.
        source: What produced it.
    """

    attempts: tuple[Attempt, ...]
    events: EventTrack
    labels: tuple[StrokeLabel, ...]
    config: SegmentConfig = field(default_factory=SegmentConfig)
    source: str = SOURCE

    def counts(self) -> dict[str, int]:
        """Counts attempts by kind, plus faults and second serves.

        Returns:
            Attempts per kind (every kind present), "faults" and
            "second_serves".
        """
        found = {kind: sum(a.kind == kind for a in self.attempts) for kind in KINDS}
        found["faults"] = sum(a.fault for a in self.attempts)
        found["second_serves"] = sum(a.second_serve for a in self.attempts)
        return found

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            The source, the counts and every attempt.
        """
        return {
            "source": self.source,
            "counts": self.counts(),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


# --------------------------------------------------------------------------
# Serves
# --------------------------------------------------------------------------


def _stroke_of(label: StrokeLabel | None) -> str:
    """A label's stroke name, "unknown" for no label."""
    return UNKNOWN if label is None else label.stroke.value


def _confident(shot: Shot, config: SegmentConfig) -> bool:
    """Whether a hit is confident enough to count as the player having hit."""
    return shot.hit.confidence >= config.quiet_breaker_confidence


@dataclass(frozen=True)
class LeadIn:
    """What came just before a hit, as far as a serve is concerned.

    Attributes:
        quiet: Seconds since the other player last hit confidently;
            infinity when they have not.
        hits: Hits by anyone just before it, back to the last silence
            longer than `rally_gap_seconds` or the other player's last
            confident hit, whichever came later.
        settled: Whether those hits followed a silence (or the start of
            the track) rather than the other player's hit - a pre-serve
            bounce follows a silence; a rally's next shot follows a hit.
        watched: Seconds of tracked video before the hit; infinity when
            not known. A quiet spell longer than what was watched is not
            a quiet spell anyone saw.
    """

    quiet: float
    hits: int
    settled: bool
    watched: float = float("inf")


def lead_in(
    shots: Sequence[Shot],
    index: int,
    config: SegmentConfig | None = None,
    start: float = float("-inf"),
) -> LeadIn:
    """Measures what came just before one hit.

    Args:
        shots: Shots in time order.
        index: The hit's position in `shots`.
        config: Supplies what counts as confident and as a silence.
        start: When tracking began, seconds; unknown if omitted.

    Returns:
        The quiet, the hits leading in, and whether a silence came first.
    """
    config = config or SegmentConfig()
    hit = shots[index].hit
    quiet, hits, counting, settled, later = float("inf"), 0, True, True, hit.pts
    for earlier in reversed(shots[:index]):
        counting = counting and later - earlier.hit.pts <= config.rally_gap_seconds
        if earlier.hit.side != hit.side and _confident(earlier, config):
            quiet = hit.pts - earlier.hit.pts
            settled = not counting or hits == 0
            break
        hits += counting
        later = earlier.hit.pts
    return LeadIn(quiet, hits, settled, hit.pts - start)


def serve_score(
    shot: Shot,
    label: StrokeLabel | None,
    lead: LeadIn,
    config: SegmentConfig | None = None,
) -> float:
    """Scores how much a hit looks like a serve; 0 when it cannot be one.

    Args:
        shot: The shot.
        label: Its stroke label, if any.
        lead: What came just before it (see `lead_in`).
        config: Thresholds; see `SegmentConfig`.

    Returns:
        The label's confidence (0.3 when it has none) times the hit's,
        weighted 1.0 for a `Serve` label, 0.9 for an `Overhead` and 0.6
        for an unknown stroke. 0 when the stroke cannot be a serve, the
        other player hit too recently, a rally is going on (hits lead in
        from the other player's hit, or too many lead in), or the feet
        were seen somewhere a server does not stand. A hit whose feet
        were not seen counts only if labelled `Serve`, at half weight:
        nothing is left to check that against. Within
        `serve_quiet_seconds` of the start of tracking the quiet was not
        watched, so only a measured serve or overhead contact counts - on
        the real recording an inferred hit 0.12s into a tracked stretch
        was otherwise read as a serve, then as a fault.
    """
    config = config or SegmentConfig()
    stroke = _stroke_of(label)
    if stroke not in config.serve_strokes or lead.quiet < config.serve_quiet_seconds:
        return 0.0
    if lead.hits > config.serve_lead_hits or (lead.hits and not lead.settled):
        return 0.0
    seen_struck = (
        label is not None
        and stroke in (SERVE, OVERHEAD)
        and label.stroke.provenance is Provenance.MEASURED
    )
    if lead.watched < config.serve_quiet_seconds and not seen_struck:
        return 0.0
    hit = shot.hit
    sure = 0.3 if label is None or label.stroke.confidence is None else label.stroke.confidence
    if hit.position is None:
        return 0.5 * sure * hit.confidence if stroke == SERVE else 0.0
    x, y = hit.position
    if hit.side == NEAR:
        behind = y <= config.serve_behind_near_m
    else:
        behind = y >= COURT_LENGTH_M - config.serve_behind_far_m
    if not behind or abs(x) > config.serve_max_x_m:
        return 0.0
    weight = {SERVE: 1.0, OVERHEAD: 0.9}.get(stroke, 0.6)
    return weight * sure * hit.confidence


def _same_serve(shots: Sequence[Shot], first: int, second: int, config: SegmentConfig) -> bool:
    """Whether two serve candidates are one serve and its preparation."""
    one, two = shots[first].hit, shots[second].hit
    if one.side != two.side or one.player != two.player:
        return False
    if two.pts - one.pts > config.serve_cluster_seconds:
        return False
    between = shots[first + 1 : second]
    return not any(s.hit.side != one.side and _confident(s, config) for s in between)


def pick_serves(
    shots: Sequence[Shot],
    labels: dict[int, StrokeLabel],
    config: SegmentConfig | None = None,
    start: float = float("-inf"),
) -> dict[int, tuple[float, bool]]:
    """Chooses which hits are serves.

    Args:
        shots: Shots in time order.
        labels: Stroke label per shot index (`Shot.index`).
        config: Thresholds; see `SegmentConfig`.
        start: When tracking began, seconds; unknown if omitted.

    Returns:
        Position in `shots` -> (serve score, whether another candidate
        competed with it), for every chosen serve.
    """
    config = config or SegmentConfig()
    scores = [
        serve_score(shot, labels.get(shot.index), lead_in(shots, i, config, start), config)
        for i, shot in enumerate(shots)
    ]
    chosen: dict[int, tuple[float, bool]] = {}
    group: list[int] = []

    def close() -> None:
        """Keeps the strongest of a group of candidates - the later on a tie."""
        best = max(group, key=lambda i: (scores[i], i))
        chosen[best] = (scores[best], len(group) > 1)

    for i in (i for i, score in enumerate(scores) if score > 0):
        if group and _same_serve(shots, group[-1], i, config):
            group.append(i)
            continue
        if group:
            close()
        group = [i]
    if group:
        close()
    return chosen


def _relabel(
    shots: Sequence[Shot],
    labels: dict[int, StrokeLabel],
    serves: dict[int, float],
) -> tuple[StrokeLabel, ...]:
    """Makes the labels agree with segmentation about which hits are serves.

    Args:
        shots: Every shot.
        labels: Stroke label per shot index.
        serves: Shot index -> serve score, for every chosen serve.

    Returns:
        One label per shot, in shot-index order. A chosen serve the
        classifier named otherwise becomes an inferred `Serve`; a hit the
        classifier called a serve but segmentation did not becomes
        unknown. Every other label is unchanged; a shot with no label and
        not a serve keeps none.
    """
    out = []
    for shot in sorted(shots, key=lambda s: s.index):
        label = labels.get(shot.index)
        said = _stroke_of(label)
        if shot.index in serves and said != SERVE:
            sure = round(min(0.9, max(0.3, serves[shot.index] / max(shot.hit.confidence, 1e-9))), 3)
            label = StrokeLabel(
                index=shot.index,
                stroke=Tagged(SERVE, Provenance.INFERRED, sure, SOURCE),
                spin=None,
                reason=(
                    "segmentation: the first hit after a quiet spell, from behind the baseline "
                    f"near the centre; the classifier said {said}"
                ),
                pose_used=label is not None and label.pose_used,
            )
        elif shot.index not in serves and said == SERVE:
            label = replace(
                label,
                stroke=Tagged(UNKNOWN, Provenance.INFERRED, None, SOURCE),
                reason=f"{label.reason}; segmentation did not take it for a serve",
            )
        if label is not None:
            out.append(label)
    return tuple(out)


def serve_zone(shot: Shot, config: SegmentConfig | None = None) -> tuple[str | None, str | None]:
    """Says which side a serve was struck from.

    A serve lands in the box diagonally opposite, and the receiver's half
    has the same name as the server's ("deuce" to "deuce"), so a seen
    landing names the side too. It decides when the feet were not seen or
    stood within `zone_doubt_m` of the centre line, where a metre-scale
    footpoint cannot.

    Args:
        shot: The serve.
        config: Supplies `zone_doubt_m`.

    Returns:
        (zone, "feet" or "bounce"), or (None, None) when neither says.
    """
    config = config or SegmentConfig()
    hit = shot.hit
    landing = None
    if shot.bounce is not None and shot.bounce.half != hit.side:
        landing = zone_of(shot.bounce.half, shot.bounce.court_xy[0])
    feet = hit.zone if hit.position is not None else None
    doubtful = hit.position is None or abs(hit.position[0]) < config.zone_doubt_m
    if landing is not None and (feet is None or (doubtful and landing != feet)):
        return landing, "bounce"
    if feet is not None:
        return feet, "feet"
    return None, None


# --------------------------------------------------------------------------
# Rallies, faults and attempts
# --------------------------------------------------------------------------


@dataclass
class _Group:
    """Hits that form one attempt, while it is being worked out."""

    kind: str
    members: list[int]
    contested: bool = False
    doubt: bool = False
    reason: str = ""
    fault: bool = False
    second: bool = False
    unzoned: bool = False


def _doubtful_gap(gap: float, config: SegmentConfig) -> bool:
    """Whether a silence is within a second of the rally threshold."""
    return abs(gap - config.rally_gap_seconds) <= _BOUNDARY_DOUBT_SECONDS


def _rallies(
    shots: Sequence[Shot], serves: dict[int, tuple[float, bool]], config: SegmentConfig
) -> list[_Group]:
    """Groups the hits into served rallies and everything else.

    Args:
        shots: Shots in time order.
        serves: Position -> (score, contested) for every chosen serve.
        config: Thresholds.

    Returns:
        Every group, in time order, each kind decided.
    """
    anchors = sorted(serves)
    groups: list[_Group] = []
    taken: set[int] = set()
    for k, anchor in enumerate(anchors):
        stop = anchors[k + 1] if k + 1 < len(anchors) else len(shots)
        members = [anchor]
        doubt = False
        for j in range(anchor + 1, stop):
            gap = shots[j].hit.pts - shots[members[-1]].hit.pts
            doubt = doubt or _doubtful_gap(gap, config)
            if gap > config.rally_gap_seconds:
                break
            members.append(j)
        groups.append(_Group(SERVED, members, contested=serves[anchor][1], doubt=doubt))
        taken.update(members)

    current: list[int] = []
    leftovers: list[list[int]] = []
    for i in (i for i in range(len(shots)) if i not in taken):
        joined = bool(current) and i == current[-1] + 1
        if joined and shots[i].hit.pts - shots[current[-1]].hit.pts <= config.rally_gap_seconds:
            current.append(i)
            continue
        if current:
            leftovers.append(current)
        current = [i]
    if current:
        leftovers.append(current)
    for members in leftovers:
        groups.append(_unserved(shots, members, config))
    return sorted(groups, key=lambda g: g.members[0])


def _unserved(shots: Sequence[Shot], members: list[int], config: SegmentConfig) -> _Group:
    """Decides whether hits outside every served rally are a point."""
    if config.match_start is not None and shots[members[0]].hit.pts < config.match_start:
        return _Group(EXCLUDED, members, reason="before the match started: warm-up")
    confident = [shots[i] for i in members if _confident(shots[i], config)]
    sides = {shot.hit.side for shot in confident}
    if len(confident) >= config.min_unserved_hits and len(sides) == 2:
        gaps = [shots[b].hit.pts - shots[a].hit.pts for a, b in zip(members, members[1:])]
        doubt = any(_doubtful_gap(gap, config) for gap in gaps)
        return _Group(SERVE_UNSEEN, members, doubt=doubt)
    return _Group(
        EXCLUDED,
        members,
        reason=(
            f"{len(confident)} confident hit(s), by {len(sides)} player(s): too little for a "
            "point - taken for play between points"
        ),
    )


def _judge_faults(shots: Sequence[Shot], groups: list[_Group], config: SegmentConfig) -> None:
    """Marks first-serve faults and the second serves that follow them.

    Only consecutive attempts that are not excluded are compared, exactly
    as `reconstruct.reconstruct_all` does when it folds them.
    """
    playing = [g for g in groups if g.kind != EXCLUDED]
    for one, two in zip(playing, playing[1:]):
        if one.kind != SERVED or two.kind != SERVED or one.second:
            continue
        first, second = shots[one.members[0]], shots[two.members[0]]
        if first.hit.player != second.hit.player or first.hit.side != second.hit.side:
            continue
        if any(shots[i].hit.side == first.hit.side for i in one.members[1:]):
            continue  # the server played on: the serve was in
        result = first.result
        if result.value == RESULT_IN and result.provenance in _BOUNCE_SEEN:
            continue  # a bounce put it in: a let, or a point, as reconstruct reads it
        gap = second.hit.pts - first.hit.pts
        zones = serve_zone(first, config)[0], serve_zone(second, config)[0]
        if zones[0] is not None and zones[1] is not None:
            fault = zones[0] == zones[1] and gap <= config.second_serve_max_seconds
        else:
            fault = gap <= config.second_serve_unzoned_seconds
            two.unzoned = fault
        one.fault = two.second = fault


def _positions(shots: Sequence[Shot], start: int) -> list[int]:
    """Places each hit in the rally, counting the other player's missed hits."""
    places = [start]
    for previous, current in zip(shots, shots[1:]):
        places.append(places[-1] + (1 if current.hit.side != previous.hit.side else 2))
    return places


def _other(person: str | None) -> str | None:
    """The other person's label, "A" <-> "B"; None for anything else."""
    return {"A": "B", "B": "A"}.get(person or "")


def _attempt(
    number: int,
    group: _Group,
    shots: Sequence[Shot],
    labels: dict[int, StrokeLabel],
    config: SegmentConfig,
) -> Attempt:
    """Builds one numbered attempt from a decided group."""
    mine = [shots[i] for i in group.members]
    reasons = [group.reason] if group.reason else []
    if group.kind == EXCLUDED:
        segmented = tuple(
            SegmentedShot(shot, labels.get(shot.index), n, NONE, None)
            for n, shot in enumerate(mine, start=1)
        )
        return Attempt(number, EXCLUDED, segmented, None, None, None, None, reasons=tuple(reasons))
    places = _positions(mine, 0 if group.kind == SERVED else 1)
    segmented = tuple(
        SegmentedShot(shot, labels.get(shot.index), n, role_for(place, group.second), place)
        for n, (shot, place) in enumerate(zip(mine, places), start=1)
    )
    first = mine[0].hit
    if group.kind == SERVED:
        zone, zone_source = serve_zone(mine[0], config)
        label = labels[mine[0].index]
        confidence = first.confidence * (label.stroke.confidence or 0.0)
        side, server = first.side, first.player
        reasons.append(f"serve seen at {first.pts:.2f}s ({label.stroke.provenance.value})")
    else:
        zone = zone_source = None
        confidence = _UNSERVED_BASE * max(shot.hit.confidence for shot in mine)
        side = FAR if first.side == NEAR else NEAR
        server = _other(first.player)
        reasons.append("no serve was seen; the first hit is taken for the return")
    if group.contested:
        confidence *= _CONTESTED
        reasons.append("another hit just before it also looked like the serve")
    if group.doubt:
        confidence *= _NEAR_THRESHOLD
        reasons.append("a silence within a second of the rally threshold: the boundary is unsure")
    weak = sum(
        shot.hit.provenance is Provenance.INFERRED and not _confident(shot, config)
        for shot in mine[1:]
    )
    if weak:
        confidence *= _WEAK_HIT ** min(weak, _MAX_WEAK_PENALTIES)
        reasons.append(f"{weak} weakly inferred hit(s) inside")
    if group.unzoned:
        confidence *= _UNZONED
        reasons.append("second serve judged by time alone: a serving side was not seen")
    if group.fault:
        reasons.append("judged a first-serve fault: the same server served again from that side")
    return Attempt(
        number=number,
        kind=group.kind,
        shots=segmented,
        server_side=side,
        server=server,
        serve_zone=zone,
        zone_source=zone_source,
        second_serve=group.second,
        fault=group.fault,
        confidence=round(confidence, 3),
        reasons=tuple(reasons),
    )


def segment(
    events: EventTrack,
    labels: Sequence[StrokeLabel],
    config: SegmentConfig | None = None,
    stroke_config: StrokeConfig | None = None,
    tracking_start: float | None = None,
) -> Segmentation:
    """Splits a stretch of play into numbered serve attempts.

    Args:
        events: The events, from `events.detect_events` (with or without
            `strokes.apply_strokes` - they are re-judged here).
        labels: Stroke labels, from `strokes.classify_strokes`; a shot
            with no label is treated as an unknown stroke.
        config: Thresholds; see `SegmentConfig`.
        stroke_config: Handedness, for re-judging directions.
        tracking_start: When the watched stretch began, seconds - the ball
            track's first frame if omitted. Pass an earlier time only when
            the video before the tracked stretch is known to be quiet.

    Returns:
        The attempts, numbered from 1 in time order, with the re-judged
        events and the relabelled strokes they were built on.
    """
    config = config or SegmentConfig()
    given = {label.index: label for label in labels}
    ordered = sorted(events.shots, key=lambda s: (s.hit.pts, s.index))
    frames = events.ball.frame_pts
    start = float(frames[0]) if len(frames) else float("-inf")
    start = start if tracking_start is None else tracking_start
    picked = pick_serves(ordered, given, config, start)
    relabelled = _relabel(
        events.shots, given, {ordered[i].index: score for i, (score, _) in picked.items()}
    )
    judged = apply_strokes(events, relabelled, stroke_config)
    by_index = {shot.index: shot for shot in judged.shots}
    shots = [by_index[shot.index] for shot in ordered]
    groups = _rallies(shots, picked, config)
    _judge_faults(shots, groups, config)
    final = {label.index: label for label in relabelled}
    attempts = tuple(
        _attempt(number, group, shots, final, config)
        for number, group in enumerate(groups, start=1)
    )
    segmentation = Segmentation(attempts, judged, relabelled, config)
    logger.info("Segmentation: %s", segmentation.counts())
    return segmentation
