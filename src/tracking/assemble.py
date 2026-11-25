"""Assembly: the tracker's shots as `RawShotRow`s - the seam (spec section 3).

`assemble` takes the events, the stroke labels and the match setup, runs
`segment.segment`, and emits a `RawMatchExport` that
`reconstruct.reconstruct_all` consumes unchanged, plus a `TrackedMatch`
sidecar saying, for every value emitted, how it was obtained.

Field by field
--------------
* `point_number` / `shot_number` / `shot_type`: from `segment` - one point
  number per serve attempt, as SwingVision's `Point` counter.
* `player`: the host's or guest's name. Which tracked person ("A", at the
  near end when tracking began, or "B") is the host is **an input**
  (`MatchSetup.host_player`), never guessed. A wrong `EndSwap` in the
  player tracks swaps host and guest from then on; the sidecar tags the
  name `inferred` after an inferred swap.
* `stroke`: the label's stroke. A stroke the classifier could not name
  with any confidence is emitted as `AssembleConfig.unknown_stroke`
  ("Unclear") - not one of SwingVision's strokes, so nobody reads it as a
  finding - and the notes list every one as low-confidence. `reconstruct`
  only looks for volleys and overheads in it, so "Unclear" is tolerated.
* A `serve_unseen` point gets an **inferred serve row** as shot 1 (see
  `segment`): who served, from which side and roughly when, all tagged
  inferred. Without it `reconstruct` reads the point as the guest's serve.
* `result`: the events stage's result where it has one. Where it does not
  (`None`), a guess from context, **always** listed in the sidecar's
  `guesses` with its reason: a shot hit back is `In`; a serve judged a
  fault is `Out` if the ball crossed the net (the receiver swatted it or a
  bounce was seen beyond it) and `Net` otherwise; a lone first serve not
  served again from that side is `In` (unreturned); a rally's last shot
  is `Out` if it crossed and `Net` otherwise - on the real sheet 62% of
  points end on an error. Before those priors, a bounce too close to call
  leans the way its measured margin points. After them, the rules of
  scoring settle what they can (`endings`): a game's length and its seen
  winners often leave one possible winner for an unseen point, and that
  point's last result is set to match, tagged `endings-v1`. A fault
  overrides an *inferred* `In` (the events stage infers `In` whenever the
  other player hits next, and receivers swat faults back); an `In` read
  off a bounce is never overridden.
* `video_time`: the hit's time **on the video's own presentation clock**
  (PTS, seconds from the start of the file), not SwingVision's wall-clock
  `Video Time`. So the footage check needs no offset for tracker output:
  pass `video_clock_offset` as `verify_footage`'s `clock_offset`, which
  reports a zero offset instead of measuring one from the audio.
* `hit_side`: the hitter's end. `hit_zone`: widened to the sheet's
  vocabulary - `deuce` / `ad`, `deuce_alley` / `ad_alley` beyond the
  singles sideline, `center_line` within `center_line_m` of it. A serve's
  zone is always plain `deuce` / `ad` (the side served from, as the sheet
  names it): `reconstruct` folds a fault into its second serve only when
  the two zones are equal strings.
* `hit_x` / `hit_y`: the hitter's feet; `bounce_x` / `bounce_y`: the
  bounce, when seen. Both in `court.py`'s frame, which matches
  SwingVision's on the one real export compared (metres, origin at the
  near baseline's centre mark, +x to the camera's right), so no
  conversion is applied.
* `direction`, `speed_mph`: from the events (speed is an average over the
  flight, not a radar speed). `spin`: always None - not attempted.
* `source_part`: 0. One video is one part.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from swingvision_import.footage import ClockOffset
from swingvision_import.raw import RawMatchExport, RawSettings, RawShotRow
from swingvision_import.reconstruct import ReconstructionResult

from . import endings
from .calibrate import Calibration
from .config import AssembleConfig, StrokeConfig, TrackerConfig
from .court import SINGLES_WIDTH_M
from .events import (
    CROSS_COURT,
    DOWN_THE_LINE,
    DOWN_THE_T,
    OUT_WIDE,
    RESULT_IN,
    RESULT_NET,
    RESULT_OUT,
    RESULTS,
    EventTrack,
    Shot,
)
from .players import FAR, NEAR, SIDES, PlayerTracks
from .provenance import Provenance, Tagged, provenance_counts
from .segment import (
    EXCLUDED,
    FIRST_SERVE,
    SECOND_SERVE,
    SERVE_UNSEEN,
    SERVED,
    SHOT_TYPES,
    Attempt,
    Segmentation,
    SegmentedShot,
    segment,
)
from .strokes import INSIDE_IN, INSIDE_OUT, STROKES, UNKNOWN, StrokeLabel

logger = logging.getLogger(__name__)

SOURCE = "assemble-v1"

CENTER_LINE = "center_line"
ZONES = ("deuce", "ad", "deuce_alley", "ad_alley", CENTER_LINE)
DIRECTIONS = (CROSS_COURT, DOWN_THE_LINE, DOWN_THE_T, OUT_WIDE, INSIDE_OUT, INSIDE_IN)
FEED = "Feed"
PEOPLE = ("A", "B")
HANDS = ("right", "left")
_HALF_SINGLES = SINGLES_WIDTH_M / 2


def video_clock_offset(video: Path, shot_times: list[float]) -> ClockOffset:
    """Reports the tracker's clock offset: none, by construction.

    Shaped like `footage.measure_clock_offset`, so it can be passed as
    `SwingVisionImportPipeline.verify_footage(clock_offset=...)` for a
    tracker-produced match. Tracker `video_time` is the video's own PTS,
    so there is nothing to measure. With fewer than 20 shots in a part,
    `verify_footage` still treats the offset as unconfirmed and notes it.

    Args:
        video: The recording (unused).
        shot_times: Each shot's `video_time` in it.

    Returns:
        A zero offset, as certain as an offset can be.
    """
    return ClockOffset(seconds=0.0, strength=float("inf"), shots=len(shot_times))


@dataclass(frozen=True)
class MatchSetup:
    """What the person who recorded the match says about it.

    Attributes:
        host_name: The tracked player's display name (the account
            holder - `RawSettings.host_name`).
        guest_name: The opponent's display name.
        host_player: Which tracked person is the host: "A" (at the near
            end when tracking began) or "B".
        host_hand: The host's playing hand, "right" or "left".
        guest_hand: The guest's.
    """

    host_name: str
    guest_name: str
    host_player: str
    host_hand: str = "right"
    guest_hand: str = "right"

    def __post_init__(self) -> None:
        """Rejects a setup `reconstruct` could not tell the players apart by.

        Raises:
            ValueError: If a name is blank, the names are equal, the host
                is not "A" or "B", or a hand is not "right" or "left".
        """
        if not self.host_name.strip() or not self.guest_name.strip():
            raise ValueError("host_name and guest_name must both be given.")
        if self.host_name == self.guest_name:
            raise ValueError("host_name and guest_name must differ: shots are told apart by name.")
        if self.host_player not in PEOPLE:
            raise ValueError(f"host_player must be 'A' or 'B', got {self.host_player!r}.")
        for hand in (self.host_hand, self.guest_hand):
            if hand not in HANDS:
                raise ValueError(f"a hand must be 'right' or 'left', got {hand!r}.")

    @classmethod
    def from_start_end(
        cls, host_name: str, guest_name: str, host_starts: str, **hands: str
    ) -> MatchSetup:
        """Builds a setup from the end the host was at when tracking began.

        Args:
            host_name: The host's name.
            guest_name: The guest's name.
            host_starts: "near" or "far".
            **hands: `host_hand` / `guest_hand`, optional.

        Returns:
            The setup: the host is "A" when they started at the near end.

        Raises:
            ValueError: If `host_starts` is neither end.
        """
        if host_starts not in SIDES:
            raise ValueError(f"host_starts must be 'near' or 'far', got {host_starts!r}.")
        player = "A" if host_starts == NEAR else "B"
        return cls(host_name, guest_name, player, **hands)

    def name(self, person: str) -> str:
        """Gives a tracked person's display name.

        Args:
            person: "A" or "B".

        Returns:
            The host's or guest's name.

        Raises:
            ValueError: If `person` is neither.
        """
        if person not in PEOPLE:
            raise ValueError(f"person must be 'A' or 'B', got {person!r}.")
        return self.host_name if person == self.host_player else self.guest_name

    def handedness(self) -> tuple[tuple[str, str], ...]:
        """Each tracked person's hand, in `StrokeConfig.handedness` form."""
        guest = "B" if self.host_player == "A" else "A"
        return tuple(sorted(((self.host_player, self.host_hand), (guest, self.guest_hand))))

    def stroke_config(self, base: StrokeConfig | None = None) -> StrokeConfig:
        """A stroke config with this setup's hands.

        Args:
            base: The config to start from; defaults if omitted.

        Returns:
            `base` with its handedness replaced - pass it to
            `strokes.classify_strokes` too, or forehands and backhands of
            a left-hander come out swapped.
        """
        return replace(base or StrokeConfig(), handedness=self.handedness())

    def settings(self) -> RawSettings:
        """The `RawSettings` a SwingVision export would carry."""
        return RawSettings(host_name=self.host_name, guest_name=self.guest_name)


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def validate_row(
    row: RawShotRow, settings: RawSettings, unknown_stroke: str = "Unclear"
) -> list[str]:
    """Checks one row against the vocabularies `reconstruct` and the sheet use.

    Args:
        row: The row.
        settings: The match's names.
        unknown_stroke: The stroke emitted when none could be named.

    Returns:
        One sentence per problem; empty when the row is valid.
    """
    problems = []

    def check(ok: bool, what: str) -> None:
        """Records a problem when `ok` is false."""
        if not ok:
            problems.append(f"point {row.point_number} shot {row.shot_number}: {what}")

    check(isinstance(row.point_number, int) and row.point_number >= 1, "point_number below 1")
    check(isinstance(row.shot_number, int) and row.shot_number >= 1, "shot_number below 1")
    check(row.player in (settings.host_name, settings.guest_name), f"player {row.player!r}")
    check(row.shot_type in SHOT_TYPES, f"shot_type {row.shot_type!r}")
    check(row.stroke in STROKES + (FEED, unknown_stroke), f"stroke {row.stroke!r}")
    check(row.result in RESULTS, f"result {row.result!r}")
    check(row.video_time is not None and row.video_time >= 0, f"video_time {row.video_time!r}")
    check(row.hit_side in SIDES, f"hit_side {row.hit_side!r}")
    check(row.hit_zone is None or row.hit_zone in ZONES, f"hit_zone {row.hit_zone!r}")
    serve = row.shot_type in SHOT_TYPES[:2]
    check(not serve or row.hit_zone in (None, "deuce", "ad"), "a serve's zone is deuce or ad")
    check(not serve or row.stroke == STROKES[0], "a serve's stroke is Serve")
    check(row.source_part == 0, f"source_part {row.source_part!r}")
    check(row.spin is None, "spin is not attempted")
    check(row.speed_mph is None or row.speed_mph >= 0, f"speed_mph {row.speed_mph!r}")
    check(row.direction is None or row.direction in DIRECTIONS, f"direction {row.direction!r}")
    check((row.hit_x is None) == (row.hit_y is None), "hit_x and hit_y come together")
    check((row.bounce_x is None) == (row.bounce_y is None), "bounce_x and bounce_y come together")
    return problems


def validate_export(export: RawMatchExport, unknown_stroke: str = "Unclear") -> list[str]:
    """Checks every row, and the order and numbering between them.

    Args:
        export: The tracker's export.
        unknown_stroke: The stroke emitted when none could be named.

    Returns:
        One sentence per problem; empty when the export is valid.
    """
    if export.settings is None:
        return ["no settings: shots cannot be told apart by player"]
    problems = []
    for row in export.shots:
        problems.extend(validate_row(row, export.settings, unknown_stroke))
    by_point: dict[int, list[RawShotRow]] = {}
    for row in export.shots:
        by_point.setdefault(row.point_number, []).append(row)
    previous_end = float("-inf")
    for number in sorted(by_point):
        rows = by_point[number]
        if [r.shot_number for r in rows] != list(range(1, len(rows) + 1)):
            problems.append(f"point {number}: shot numbers are not 1..{len(rows)} in order")
        times = [r.video_time or 0.0 for r in rows]
        if times != sorted(times) or times[0] < previous_end:
            problems.append(f"point {number}: shots are not in time order")
        previous_end = times[-1]
        serves = [i for i, r in enumerate(rows) if r.shot_type in SHOT_TYPES[:2]]
        if serves and serves != [0]:
            problems.append(f"point {number}: a serve that is not its first shot")
        kinds = {r.shot_type == SHOT_TYPES[-1] for r in rows}
        if len(kinds) > 1:
            problems.append(f"point {number}: mixes play between points with rally shots")
    return problems


# --------------------------------------------------------------------------
# One row, and how each of its values was obtained
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ShotTrace:
    """How every value of one emitted row was obtained (spec section 6.1).

    Attributes:
        point_number: The row's point number.
        shot_number: The row's shot number.
        event_index: The shot's index in `EventTrack.shots`; None for a
            serve inferred from its return (no event exists for it).
        pts: The hit's time, seconds on the video's clock (estimated for
            an inferred serve).
        time_uncertainty: How far off that may be, seconds.
        confidence: The hit's own confidence, 0 to 1.
        tracked: Whether the hit counts as tracked with confidence (seen
            or bridged, not inferred, and confident enough).
        fields: Per emitted field, the value with its provenance,
            confidence and source. A field emitted as None is absent.
        guesses: Fields whose emitted value is a guess from context with
            no evidence of its own - "result", "stroke".
        notes: Why each guess was made, and anything else a reviewer
            should know about this row.
    """

    point_number: int
    shot_number: int
    event_index: int | None
    pts: float
    time_uncertainty: float
    confidence: float
    tracked: bool
    fields: dict[str, Tagged]
    guesses: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every attribute, each field as its tagged dict.
        """
        return {
            "point_number": self.point_number,
            "shot_number": self.shot_number,
            "event_index": self.event_index,
            "pts": self.pts,
            "time_uncertainty": self.time_uncertainty,
            "confidence": self.confidence,
            "tracked": self.tracked,
            "fields": {name: _plain(tagged) for name, tagged in self.fields.items()},
            "guesses": list(self.guesses),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ShotTrace:
        """Reads a trace back from `to_dict`'s form (e.g. the staged JSON).

        Args:
            data: A dict as `to_dict` writes it.

        Returns:
            The trace; court positions come back as tuples.
        """
        return cls(
            point_number=int(data["point_number"]),
            shot_number=int(data["shot_number"]),
            event_index=None if data["event_index"] is None else int(data["event_index"]),
            pts=float(data["pts"]),
            time_uncertainty=float(data["time_uncertainty"]),
            confidence=float(data["confidence"]),
            tracked=bool(data["tracked"]),
            fields={name: Tagged.from_dict(value) for name, value in data["fields"].items()},
            guesses=tuple(data.get("guesses", ())),
            notes=tuple(data.get("notes", ())),
        )


@dataclass(frozen=True)
class AttemptBlock:
    """One serve attempt's `tracking` block, read back from the staged JSON.

    Attributes:
        number: Its point number (one per serve attempt).
        kind: "served", "serve_unseen" or "excluded".
        start: Its first hit's time, seconds on the video's clock.
        end: Its last hit's time.
        server_side: The server's end, or None.
        server: The server's display name, or None.
        serve_zone: "deuce" / "ad", or None.
        zone_source: "feet" / "bounce", or None.
        second_serve: Whether it is a second serve.
        fault: Whether its serve was judged a first-serve fault.
        confidence: Segmentation's 0..1 ranking heuristic.
        reasons: What set or lowered that confidence.
        shots: A trace per emitted row, in order.
        folded_fault: The first-serve fault reconstruction folded into
            this attempt, when there was one.
        serve_estimate: For a serve inferred from its return, when it was
            struck (seconds); None otherwise.
    """

    number: int
    kind: str
    start: float
    end: float
    server_side: str | None
    server: str | None
    serve_zone: str | None
    zone_source: str | None
    second_serve: bool
    fault: bool
    confidence: float
    reasons: tuple[str, ...]
    shots: tuple[ShotTrace, ...]
    folded_fault: AttemptBlock | None = None
    serve_estimate: float | None = None

    def to_dict(self) -> dict:
        """Serialises to the form `TrackedMatch.attempt_block` writes.

        Returns:
            Every attribute; `folded_fault` only when there is one.
        """
        out = {
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
            "serve_estimate": self.serve_estimate,
            "shots": [trace.to_dict() for trace in self.shots],
        }
        if self.folded_fault is not None:
            out["folded_fault"] = self.folded_fault.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: dict) -> AttemptBlock:
        """Reads an attempt block back.

        Args:
            data: A dict as `TrackedMatch.attempt_block` (or `to_dict`)
                writes it.

        Returns:
            The block.
        """
        folded = data.get("folded_fault")
        return cls(
            number=int(data["number"]),
            kind=data["kind"],
            start=float(data["start"]),
            end=float(data["end"]),
            server_side=data.get("server_side"),
            server=data.get("server"),
            serve_zone=data.get("serve_zone"),
            zone_source=data.get("zone_source"),
            second_serve=bool(data.get("second_serve", False)),
            fault=bool(data.get("fault", False)),
            confidence=float(data.get("confidence", 0.0)),
            reasons=tuple(data.get("reasons", ())),
            shots=tuple(ShotTrace.from_dict(shot) for shot in data.get("shots", ())),
            folded_fault=None if folded is None else cls.from_dict(folded),
            serve_estimate=data.get("serve_estimate"),
        )


@dataclass(frozen=True)
class TrackingBlock:
    """A match's `tracking` block, read back from the staged JSON.

    Attributes:
        source: What produced it (`SOURCE`).
        summary: Counts, provenance, model versions and quality.
        notes: The tracker's informational notes.
        attempts: Every serve attempt's block, in order.
        extra: Every other key the block carried (the pipeline adds the
            video, the run and the emitted rows), kept as stored.
    """

    source: str
    summary: dict
    notes: tuple[str, ...]
    attempts: tuple[AttemptBlock, ...]
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialises back to the stored form.

        Returns:
            The block, `extra`'s keys included.
        """
        return {
            **self.extra,
            "source": self.source,
            "summary": self.summary,
            "notes": list(self.notes),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrackingBlock:
        """Reads a match-level block back.

        Args:
            data: A dict as `TrackedMatch.tracking_block` writes it,
                possibly with more keys.

        Returns:
            The block.
        """
        known = {"source", "summary", "notes", "attempts"}
        return cls(
            source=data.get("source", SOURCE),
            summary=dict(data.get("summary", {})),
            notes=tuple(data.get("notes", ())),
            attempts=tuple(AttemptBlock.from_dict(a) for a in data.get("attempts", ())),
            extra={key: value for key, value in data.items() if key not in known},
        )


def _plain(tagged: Tagged) -> dict:
    """A tagged value as JSON-ready types, tuples as lists."""
    out = tagged.to_dict()
    if isinstance(out["value"], tuple):
        out["value"] = [float(v) for v in out["value"]]
    return out


def _crossed(shot: Shot) -> bool:
    """Whether the ball is known to have got past the net."""
    landed_beyond = shot.bounce is not None and shot.bounce.half != shot.hit.side
    return shot.returned or landed_beyond


def _guess(value: str, reason: str, confidence: float = 0.3) -> tuple[Tagged, str]:
    """A guessed result, tagged inferred at low confidence, with its reason."""
    return Tagged(value, Provenance.INFERRED, confidence, SOURCE), reason


def _error(shot: Shot) -> str:
    """How a shot that lost the point most likely failed: long if it crossed."""
    return RESULT_OUT if _crossed(shot) else RESULT_NET


def _result(attempt: Attempt, place: int) -> tuple[Tagged, str | None]:
    """Decides the result emitted for one shot of an attempt.

    Args:
        attempt: The attempt.
        place: The shot's position in `attempt.shots`.

    Returns:
        (the result, tagged), and the reason when it is a guess - None
        when the events stage's own result is kept.
    """
    segmented = attempt.shots[place]
    shot = segmented.shot
    known = shot.result
    after = place + 1 < len(attempt.shots)
    if attempt.fault and place == 0:
        if known.value in (RESULT_OUT, RESULT_NET):
            return known, None
        value = RESULT_OUT if _crossed(shot) else RESULT_NET
        was = "" if known.value is None else f"; overrides an inferred {known.value}"
        reason = (
            f"judged a first-serve fault (the same server served again from that side): "
            f"{value} because the ball {'did' if value == RESULT_OUT else 'was not seen to'} "
            f"cross the net{was}"
        )
        return _guess(value, reason)
    if known.value is not None:
        return known, None
    if attempt.kind == EXCLUDED:
        return _guess(RESULT_IN, "play between points: not judged")
    if after:
        return _guess(RESULT_IN, "the ball came back: another hit followed in the rally")
    call = shot.call
    if call is not None and call.value is None:
        lean = RESULT_IN if call.margin_m >= 0 else RESULT_OUT
        where = "inside" if call.margin_m >= 0 else "beyond"
        return _guess(
            lean,
            f"the bounce was {abs(call.margin_m):.2f}m {where} the {call.line}, within its "
            f"{call.uncertainty_m:.2f}m uncertainty: leaning {lean}",
            0.4,
        )
    if segmented.position == 0 and not attempt.second_serve:
        return _guess(RESULT_IN, "a lone first serve not served again from that side: unreturned")
    value = RESULT_OUT if _crossed(shot) else RESULT_NET
    if segmented.position == 0:
        what = "a lone second serve, read as a double fault"
    else:
        what = "the last shot of the rally, read as an error (62% of points end on one)"
    side = "crossed the net" if value == RESULT_OUT else "was not seen past the net"
    return _guess(value, f"{what}: {value} because the ball {side}")


def _zone(attempt: Attempt, segmented: SegmentedShot, config: AssembleConfig) -> Tagged | None:
    """The sheet-vocabulary zone of one shot, tagged; None when unknown."""
    hit = segmented.shot.hit
    if attempt.kind == SERVED and segmented.position == 0:
        if attempt.serve_zone is None:
            return None
        if attempt.zone_source == "bounce":
            return Tagged(attempt.serve_zone, Provenance.INFERRED, None, "segment-v1")
        return Tagged(attempt.serve_zone, hit.position_provenance or Provenance.INFERRED)
    if hit.position is None or hit.zone is None:
        return None
    x = hit.position[0]
    if abs(x) > _HALF_SINGLES:
        value = f"{hit.zone}_alley"
    elif abs(x) < config.center_line_m:
        value = CENTER_LINE
    else:
        value = hit.zone
    return Tagged(value, hit.position_provenance or Provenance.INFERRED)


def _who(hit_pts: float, provenance: Provenance, players: PlayerTracks | None) -> Provenance:
    """How well a hit's person is known: no better than an inferred end swap."""
    swaps = () if players is None else players.swaps
    inferred = any(s.pts <= hit_pts and s.provenance is Provenance.INFERRED for s in swaps)
    return Provenance.INFERRED if inferred else provenance


def _emit(
    attempt: Attempt,
    place: int,
    setup: MatchSetup,
    players: PlayerTracks | None,
    config: AssembleConfig,
) -> tuple[RawShotRow, ShotTrace]:
    """Emits one row and its trace."""
    segmented = attempt.shots[place]
    shot, hit = segmented.shot, segmented.shot.hit
    guesses, notes = [], []
    tagged: dict[str, Tagged] = {}

    name = setup.name(hit.player)
    tagged["player"] = Tagged(name, _who(hit.pts, hit.provenance, players), hit.confidence)
    tagged["hit_side"] = Tagged(hit.side, hit.provenance, hit.confidence)
    tagged["video_time"] = Tagged(hit.pts, hit.provenance, hit.confidence)
    tagged["shot_type"] = Tagged(
        segmented.role, Provenance.INFERRED, attempt.confidence, "segment-v1"
    )

    stroke = segmented.stroke
    if stroke == UNKNOWN:
        stroke = config.unknown_stroke
        guesses.append("stroke")
        why = "no stroke label" if segmented.label is None else segmented.label.reason
        notes.append(f"stroke too low-confidence to name ({why}): shown as {stroke}")
        tagged["stroke"] = Tagged(stroke, Provenance.INFERRED, None, SOURCE)
    else:
        tagged["stroke"] = segmented.label.stroke

    result, reason = _result(attempt, place)
    tagged["result"] = result
    if reason is not None:
        guesses.append("result")
        notes.append(f"result guessed: {reason}")

    zone = _zone(attempt, segmented, config)
    if zone is not None:
        tagged["hit_zone"] = zone
    if hit.position is not None:
        tagged["hit_xy"] = Tagged(tuple(hit.position), hit.position_provenance)
        if hit.foot_clipped:
            notes.append("the hitter's feet were cut off: they stood further back than hit_y")
    if shot.bounce is not None:
        bounce = shot.bounce
        tagged["bounce_xy"] = Tagged(tuple(bounce.court_xy), bounce.provenance, bounce.confidence)
    if shot.speed_mph is not None:
        tagged["speed_mph"] = shot.speed_mph
    if shot.direction is not None:
        tagged["direction"] = shot.direction

    position = hit.position
    row = RawShotRow(
        point_number=attempt.number,
        shot_number=segmented.shot_number,
        player=name,
        shot_type=segmented.role,
        stroke=stroke,
        result=result.value,
        video_time=round(float(hit.pts), 3),
        hit_side=hit.side,
        hit_zone=None if zone is None else zone.value,
        source_part=0,
        spin=None,
        speed_mph=None if shot.speed_mph is None else round(float(shot.speed_mph.value), 2),
        bounce_x=None if shot.bounce is None else round(float(shot.bounce.court_xy[0]), 3),
        bounce_y=None if shot.bounce is None else round(float(shot.bounce.court_xy[1]), 3),
        hit_x=None if position is None else round(float(position[0]), 3),
        hit_y=None if position is None else round(float(position[1]), 3),
        direction=None if shot.direction is None else shot.direction.value,
    )
    tracked = hit.provenance is not Provenance.INFERRED and hit.confidence >= config.confident_hit
    trace = ShotTrace(
        point_number=attempt.number,
        shot_number=segmented.shot_number,
        event_index=shot.index,
        pts=float(hit.pts),
        time_uncertainty=float(hit.time_uncertainty),
        confidence=float(hit.confidence),
        tracked=tracked,
        fields=tagged,
        guesses=tuple(guesses),
        notes=tuple(notes),
    )
    return row, trace


def _inferred_serve(
    attempt: Attempt, setup: MatchSetup, config: AssembleConfig
) -> tuple[RawShotRow, ShotTrace]:
    """Emits the serve of a `serve_unseen` point, inferred from its return.

    Who served is the player at the other end from the returner, the side
    the returner's half, the time the return's less the measured delay
    (see `segment`). It is a conclusion, not a sighting: every field is
    tagged inferred, and it has no position, bounce, speed or direction.
    It is shot 1; the return is shot 2.
    """
    returned = attempt.shots[0].shot.hit
    role = SECOND_SERVE if attempt.second_serve else FIRST_SERVE
    name = setup.name(attempt.server)
    pts = float(attempt.serve_estimate)
    origin = Provenance.INFERRED
    tagged = {
        "player": Tagged(name, origin, None, "segment-v1"),
        "hit_side": Tagged(attempt.server_side, origin, None, "segment-v1"),
        "video_time": Tagged(pts, origin, None, "segment-v1"),
        "shot_type": Tagged(role, origin, attempt.confidence, "segment-v1"),
        "stroke": Tagged(STROKES[0], origin, None, "segment-v1"),
        "result": Tagged(RESULT_IN, origin, 0.5, SOURCE),
    }
    if attempt.serve_zone is not None:
        tagged["hit_zone"] = Tagged(attempt.serve_zone, origin, None, "segment-v1")
    row = RawShotRow(
        point_number=attempt.number,
        shot_number=1,
        player=name,
        shot_type=role,
        stroke=STROKES[0],
        result=RESULT_IN,
        video_time=round(pts, 3),
        hit_side=attempt.server_side,
        hit_zone=attempt.serve_zone,
        source_part=0,
    )
    trace = ShotTrace(
        point_number=attempt.number,
        shot_number=1,
        event_index=None,
        pts=pts,
        time_uncertainty=0.3,
        confidence=0.0,
        tracked=False,
        fields=tagged,
        guesses=("serve_row",),
        notes=(
            f"serve not seen: inferred from the return at {returned.pts:.2f}s - served by "
            f"{name} from the {attempt.server_side} end, {attempt.serve_zone or 'unknown'} "
            "side, about 1.1s before the return; In because it was returned",
        ),
    )
    return row, trace


# --------------------------------------------------------------------------
# Endings: what the scoring rules decide
# --------------------------------------------------------------------------


def _settle_endings(
    segmentation: Segmentation,
    spans: dict[int, tuple[int, int]],
    rows: list[RawShotRow],
    traces: list[ShotTrace],
    setup: MatchSetup,
    config: AssembleConfig,
) -> list[endings.GameVerdict]:
    """Settles unseen point endings the rules of a game leave no choice about.

    Changes, in place, the last row of every point `endings.settle_games`
    decides: its result becomes whatever makes the decided player win (In
    for the hitter's winner, else the error the ball most likely made),
    tagged inferred by `endings-v1` with the reason in the trace.

    Args:
        segmentation: The attempts.
        spans: Attempt number -> (first row index, end row index).
        rows: Every emitted row.
        traces: Their traces.
        setup: For telling the host from the guest.
        config: Supplies `ad_scoring`.

    Returns:
        The verdict on every run of one server's points.
    """
    facts, last = [], {}
    shot_of = {}
    for attempt in segmentation.attempts:
        if attempt.kind == EXCLUDED or attempt.fault:
            continue
        first, end = spans[attempt.number]
        index = end - 1
        last[attempt.number] = index
        shot_of[attempt.number] = attempt.shots[-1].shot
        hitter = _person(rows[index].player, setup)
        if "result" in traces[index].guesses:
            won = None
        else:
            winner = hitter if rows[index].result == RESULT_IN else _other_person(hitter)
            won = winner == attempt.server
        facts.append(endings.PointFacts(attempt.number, attempt.server, attempt.serve_zone, won))
    verdicts = endings.settle_games(facts, config.ad_scoring)
    servers = {f.key: f.server for f in facts}
    for verdict in verdicts:
        for number, server_won in verdict.settled.items():
            index = last[number]
            hitter = _person(rows[index].player, setup)
            winner = servers[number] if server_won else _other_person(servers[number])
            value = RESULT_IN if winner == hitter else _error(shot_of[number])
            rows[index] = replace(rows[index], result=value)
            trace = traces[index]
            fields = dict(trace.fields)
            fields["result"] = Tagged(value, Provenance.INFERRED, 0.6, endings.SOURCE)
            note = (
                f"result settled by game scoring: {setup.name(winner)} must have won this point "
                f"for its {len(verdict.keys)}-point game to add up"
            )
            traces[index] = replace(trace, fields=fields, notes=trace.notes + (note,))
    return verdicts


def _person(name: str, setup: MatchSetup) -> str:
    """The tracked person ("A"/"B") behind a display name."""
    return setup.host_player if name == setup.host_name else _other_person(setup.host_player)


def _other_person(person: str) -> str:
    """The other tracked person."""
    return "B" if person == "A" else "A"


# --------------------------------------------------------------------------
# The match
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackedMatch:
    """The tracker's output: the export, and how every value of it was got.

    Attributes:
        export: Settings and shots for `reconstruct.reconstruct_all`.
        segmentation: The serve attempts the shots were split into.
        traces: One per emitted row, in row order.
        setup: The match setup it was built with.
        notes: Informational notes for the reviewer (never blocking).
        summary: Counts, provenance, model versions and quality - the
            match-level `tracking` block.
        source: What produced it.
    """

    export: RawMatchExport
    segmentation: Segmentation
    traces: tuple[ShotTrace, ...]
    setup: MatchSetup
    notes: tuple[str, ...] = ()
    summary: dict = field(default_factory=dict)
    source: str = SOURCE

    def attempt_block(self, number: int) -> dict | None:
        """The `tracking` block of one serve attempt.

        Args:
            number: The attempt's point number.

        Returns:
            The attempt (kind, server's name, confidence, reasons) with
            its shots' traces; None when no attempt has that number.
        """
        attempt = next((a for a in self.segmentation.attempts if a.number == number), None)
        if attempt is None:
            return None
        block = attempt.to_dict()
        del block["shots"]
        block["server"] = None if attempt.server is None else self.setup.name(attempt.server)
        block["shots"] = [t.to_dict() for t in self.traces if t.point_number == number]
        return block

    def point_blocks(self, result: ReconstructionResult) -> dict[int, dict]:
        """The `tracking` block of every reconstructed point (spec section 6.3).

        `reconstruct` folds a first-serve fault into the attempt after it,
        keeping the later number as `PointRecord.source_point_number`;
        the fault's own block is kept under `folded_fault`.

        Args:
            result: `reconstruct.reconstruct_all` on this match's export.

        Returns:
            source_point_number -> its block.
        """
        numbers = sorted(point.point_number for point in result.points)
        blocks = {n: self.attempt_block(n) for n in numbers}
        for fault in result.merged_fault_points:
            into = next((n for n in numbers if n > fault), None)
            if into is not None and blocks[into] is not None:
                blocks[into]["folded_fault"] = self.attempt_block(fault)
        return {n: block for n, block in blocks.items() if block is not None}

    def tracking_block(self) -> dict:
        """The match-level `tracking` block for the staged JSON.

        Returns:
            The summary, the notes and every attempt's block, JSON-ready.
        """
        return {
            "source": self.source,
            "summary": self.summary,
            "notes": list(self.notes),
            "attempts": [self.attempt_block(a.number) for a in self.segmentation.attempts],
        }


def _ball_notes(events: EventTrack, attempts: Sequence[Attempt]) -> list[str]:
    """Ball-tracking notes over the stretches that were play."""
    playing = [a for a in attempts if a.kind != EXCLUDED]
    overall = None
    if playing:
        overall = events.ball.quality(playing[0].start - 0.5, playing[-1].end + 1.5)
    if overall is None or overall.frames == 0:
        return []  # no frames were tracked here: nothing to say about the ball
    notes = [] if overall.notice() is None else [overall.notice()]
    weak = sum(events.ball.quality(a.start - 0.5, a.end + 1.5).weak for a in playing)
    if weak:
        notes.append(f"Ball tracking was weak on {weak} of {len(playing)} points.")
    return notes


def tracking_notes(
    traces: Sequence[ShotTrace],
    segmentation: Segmentation,
    players: PlayerTracks | None = None,
    calibration: Calibration | None = None,
    config: AssembleConfig | None = None,
    verdicts: Sequence[endings.GameVerdict] = (),
) -> list[str]:
    """Informational notes on what the tracker could and could not see.

    Like `quality_check`'s notes: for the reviewer's awareness, never a
    gate. Every count is stated against its denominator.

    Args:
        traces: One per emitted row.
        segmentation: The attempts.
        players: The player tracks, for end-swap notes.
        calibration: The calibration, for its warnings.
        config: Supplies the far-end threshold and the unknown stroke.
        verdicts: What scoring settled about unseen endings, if checked.

    Returns:
        Sentences, most important first.
    """
    config = config or AssembleConfig()
    attempts = segmentation.attempts
    counts = segmentation.counts()
    playing = [t for t in traces if t.fields["shot_type"].value != "none"]
    inferred_serves = sum("serve_row" in t.guesses for t in playing)
    playing = [t for t in playing if "serve_row" not in t.guesses]
    tracked = sum(t.tracked for t in playing)
    notes = [
        f"Tracked {tracked} of {len(playing)} shots with confidence (a hit seen or bridged, "
        f"not inferred, at confidence {config.confident_hit:.2f} or more)."
    ]
    points = counts[SERVED] + counts[SERVE_UNSEEN]
    notes.append(
        f"{points} serve attempts: the serve was seen on {counts[SERVED]} and missed on "
        f"{counts[SERVE_UNSEEN]}."
    )
    if inferred_serves:
        notes.append(
            f"{inferred_serves} serve row(s) were inferred, not seen: the server is the player "
            "at the other end from the return, the side is the returner's half, and the time "
            "is estimated about 1.1s before the return. They have no position or speed."
        )
    if counts["faults"]:
        notes.append(
            f"{counts['faults']} serve(s) read as a first-serve fault because the same player "
            "served again from the same side; the second serve is its own point number, as "
            "SwingVision numbers them, and reconstruction folds the two."
        )
    if counts[EXCLUDED]:
        hits = sum(len(a.shots) for a in attempts if a.kind == EXCLUDED)
        notes.append(
            f"{hits} hit(s) in {counts[EXCLUDED]} group(s) were taken for play between points "
            "or warm-up and are excluded from the score."
        )
    by_side = {
        side: sum(t.tracked for t in playing if t.fields["hit_side"].value == side)
        for side in SIDES
    }
    if by_side[NEAR] and by_side[FAR] < config.far_weak_ratio * by_side[NEAR]:
        notes.append(
            f"The far end is weak: {by_side[FAR]} far-end hits tracked with confidence against "
            f"{by_side[NEAR]} at the near end. Far-end strokes, results and positions need a "
            "person to look."
        )
    guessed_results = sum("result" in t.guesses for t in playing)
    if guessed_results:
        notes.append(
            f"{guessed_results} of {len(playing)} shot results were guessed from context, not "
            "seen; each says why in its trace."
        )
    settled = sum(len(v.settled) for v in verdicts)
    if settled:
        notes.append(
            f"{settled} of those unseen point endings were settled by the scoring rules: the "
            "only way their game adds up."
        )
    broken = [v for v in verdicts if "missing" in v.reason or "no way" in v.reason]
    if broken:
        notes.append(
            f"{len(broken)} game(s) do not add up as tracked - a point was probably missed or "
            "an extra one found - so their unseen endings stay guesses."
        )
    unclear = [t for t in playing if "stroke" in t.guesses]
    if unclear:
        named = ", ".join(f"point {t.point_number} shot {t.shot_number}" for t in unclear[:15])
        more = f" and {len(unclear) - 15} more" if len(unclear) > 15 else ""
        notes.append(
            f"Low confidence: {len(unclear)} of {len(playing)} strokes could not be named with "
            f"any confidence and are shown as {config.unknown_stroke!r} ({named}{more}). "
            "Treat them as unknown."
        )
    notes.extend(_ball_notes(segmentation.events, attempts))
    quality = segmentation.events.quality().notice()
    if quality is not None:
        notes.append(quality)
    if players is not None:
        inferred = sum(s.provenance is Provenance.INFERRED for s in players.swaps)
        if inferred:
            notes.append(
                f"{inferred} change(s) of ends were inferred from the players' clothing; a wrong "
                "one swaps host and guest from that moment on."
            )
    if calibration is not None:
        notes.extend(calibration.quality.warnings)
    return notes


def _summary(
    traces: Sequence[ShotTrace],
    segmentation: Segmentation,
    players: PlayerTracks | None,
    calibration: Calibration | None,
    verdicts: Sequence[endings.GameVerdict] = (),
) -> dict:
    """The match-level numbers of the `tracking` block."""
    events = segmentation.events
    names = sorted({name for t in traces for name in t.fields})
    stroke_sources = sorted({s.stroke.source for s in segmentation.labels if s.stroke.source})
    return {
        "video_clock": "pts",
        "shots": len(traces),
        "tracked_shots": sum(t.tracked for t in traces),
        "attempts": segmentation.counts(),
        "guesses": {
            "result": sum("result" in t.guesses for t in traces),
            "stroke": sum("stroke" in t.guesses for t in traces),
            "serve_row": sum("serve_row" in t.guesses for t in traces),
            "settled_by_scoring": sum(len(v.settled) for v in verdicts),
        },
        "provenance": {
            name: provenance_counts(t.fields[name].provenance for t in traces if name in t.fields)
            for name in names
        },
        "models": {
            "ball": events.ball.source,
            "events": events.source,
            "strokes": stroke_sources,
            "players": None if players is None else players.source,
            "segment": segmentation.source,
            "endings": endings.SOURCE,
            "assemble": SOURCE,
        },
        "calibration": None if calibration is None else calibration.quality.to_dict(),
        "event_quality": events.quality().to_dict(),
        "player_quality": None if players is None else players.quality().to_dict(),
    }


def assemble(
    events: EventTrack,
    labels: Sequence[StrokeLabel],
    setup: MatchSetup,
    *,
    players: PlayerTracks | None = None,
    calibration: Calibration | None = None,
    config: TrackerConfig | None = None,
    tracking_start: float | None = None,
) -> TrackedMatch:
    """Turns the tracker's events and strokes into a `RawMatchExport`.

    Args:
        events: From `events.detect_events` (and optionally
            `strokes.apply_strokes`).
        labels: From `strokes.classify_strokes`, classified with
            `setup.stroke_config(...)` so the hands are right.
        setup: Names, which tracked person is the host, and hands.
        players: The player tracks, for identity provenance and notes.
        calibration: The calibration, for the `tracking` block.
        config: Segmentation and assembly settings.
        tracking_start: Passed to `segment.segment`.

    Returns:
        The export, the segmentation, a trace per row, notes and summary.
    """
    config = config or TrackerConfig()
    segmentation = segment(
        events, labels, config.segment, setup.stroke_config(config.strokes), tracking_start
    )
    rows, traces = [], []
    spans = {}
    for attempt in segmentation.attempts:
        begin = len(rows)
        if attempt.kind == SERVE_UNSEEN and attempt.server is not None:
            row, trace = _inferred_serve(attempt, setup, config.assemble)
            rows.append(row)
            traces.append(trace)
        for place in range(len(attempt.shots)):
            row, trace = _emit(attempt, place, setup, players, config.assemble)
            rows.append(row)
            traces.append(trace)
        spans[attempt.number] = (begin, len(rows))
    verdicts = _settle_endings(segmentation, spans, rows, traces, setup, config.assemble)
    export = RawMatchExport(settings=setup.settings(), shots=rows)
    notes = tracking_notes(
        traces, segmentation, players, calibration, config.assemble, verdicts
    )
    summary = _summary(traces, segmentation, players, calibration, verdicts)
    logger.info("Assembled %d shot rows in %d attempts", len(rows), len(segmentation.attempts))
    return TrackedMatch(export, segmentation, tuple(traces), setup, tuple(notes), summary)
