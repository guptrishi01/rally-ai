"""Events: when the ball was hit, where it bounced, and how each shot ended.

`detect_events` reads a `BallTrack` and the `PlayerTracks` and reports hits,
bounces, net contacts and - joining them - shots, each with a result
(`In` / `Out` / `Net`, or unknown), an average speed and a direction.
Splitting shots into points and emitting `RawShotRow`s is the next phase.

How an event is found
---------------------
The ball's image path is smooth while it flies and has a *kink* where
something touched it. Kinks are looked for in three places:

* inside a segment, by fitting the path just before and just after each
  frame and comparing the two velocities (`kink`, a measured event);
* between two segments separated by a short unseen stretch (a quarter of
  a second at most), by extending both paths into the gap and seeing
  where they meet (`junction`, tagged `interpolated`);
* where a segment starts with nothing leading into it (`appearance`):
  the ball is usually lost for a few frames around contact, because it is
  inside the hitter's box and a racket looks like a ball. Its path is
  followed back to where it came from.

A kink is then **a hit, a bounce, a net contact or nothing**:

* *Hit.* The kinks within reach of one player during one visit of the ball
  to their end are taken together, and at most one is the hit: the last
  that is hard enough (a bounce in front of a player comes before their
  hit, never after). "Within reach" is judged in the picture **and by the
  ball's apparent size**, because a far-court ball just above the near
  player's head is on the same sightline as a ball at their racket. The
  players take turns, a player cannot hit again before the ball could
  have come back, and a seen bounce at the level of the far player's feet
  is a bounce, not their hit.
* *Bounce.* The ball was falling in the image, was pushed upward, kept its
  sideways motion and did not stop dead, and its sightline meets the
  ground on or near the court. Only there is `ground_ray_xy` the ball's
  real position, and only there is `BallObservation.court_xy` filled in.
* *Net.* The ball lost nearly all its image speed in front of the net and
  did not rebound. **This is rarely seen on real footage**: a ball that
  dies in the net stops moving, the motion detector loses it, and the
  track simply ends - indistinguishable from a ball lost over the far
  court. Such a shot gets an unknown result, not a guessed `Net`.

What is measured, what is inferred, and what is left unknown
-----------------------------------------------------------
Every event carries a provenance and a confidence, and every bounce an
uncertainty in metres - the calibration's metres-per-pixel at that spot
times the pixel uncertainty, so a far-end bounce (0.28m of depth per pixel
on the real recording) is honestly vague. `judge_bounce` calls a bounce in
or out only when it is further from the line than that uncertainty;
otherwise the call is unknown.

Two conclusions are drawn from context and tagged `inferred`, with the
reasoning recorded on the shot:

* a shot whose bounce was not seen (or was too close to call) but which
  the other player then hit is `In` - they played it;
* when one player hits twice with the ball plainly having gone and come
  back, the other player's unseen hit is added in between.

Nothing else is invented. A shot with no bounce, no net contact and no
return has an unknown result, and `EventQuality` counts those by court
half so the weakness of the far end is a number, not an impression.

Audio
-----
A struck ball is a sharp sound. `audio_onsets` turns the recording's audio
into onset times (using `swingvision_import.footage.onset_envelope`, the
same envelope the clock-offset measurement uses). Tennis audio is full of
other sounds - the neighbouring court, footsteps - so an onset never
creates a hit. One that falls where sound from that player would arrive
*confirms* a hit already seen, and replaces its time only when the picture
was vague about it. Measured on the real recording, a hit seen in the
track is already timed to about 0.03s and audio did not improve on that;
far-end hits are barely louder than the background.

Speed is an average
-------------------
`Shot.speed_mph` is the straight court-plane distance from the hitter's
feet to the bounce, divided by the time between them. It is **not** a
radar speed off the racket: the ball slows in flight, the contact point is
not above the feet, and height is ignored.
"""

from __future__ import annotations

import bisect
import logging
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from swingvision_import import footage

from .ball import BallObservation, BallTrack, _on_ground, row_scale
from .calibrate import Calibration
from .config import EventConfig
from .court import (
    COURT_LENGTH_M,
    DOUBLES_WIDTH_M,
    NET_HEIGHT_CENTRE_M,
    NET_HEIGHT_POST_M,
    NET_Y_M,
    SERVICE_LINE_FROM_NET_M,
    SINGLES_WIDTH_M,
    Point,
    camera_from_homography,
)
from .players import AD, DEUCE, FAR, NEAR, SIDES, PlayerObservation, PlayerTracks, side_of
from .provenance import Provenance, Tagged, weakest

logger = logging.getLogger(__name__)

RESULT_IN = "In"
RESULT_OUT = "Out"
RESULT_NET = "Net"
RESULTS = (RESULT_IN, RESULT_OUT, RESULT_NET)

CROSS_COURT = "cross court"
DOWN_THE_LINE = "down the line"
DOWN_THE_T = "down the t"
OUT_WIDE = "out wide"

SOURCE = "events-v1"
_MPS_TO_MPH = 2.2369363
_AUDIO_RATE = 8000


def _other(side: str) -> str:
    """Names the opposite end of the court."""
    return FAR if side == NEAR else NEAR


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioOnsets:
    """Moments the recording's sound rose sharply.

    Attributes:
        times: When each onset happened, seconds on the video's clock.
        strengths: How sharp each was, in units of the audio's
            99th-percentile onset (capped at 3).
    """

    times: np.ndarray
    strengths: np.ndarray

    def nearest(self, moment: float, window: float) -> tuple[float, float] | None:
        """Finds the onset closest to a moment.

        Args:
            moment: The moment, seconds.
            window: How far from it an onset may be.

        Returns:
            (time, strength) of the closest onset within `window`, or
            None when there is none.
        """
        if not len(self.times):
            return None
        best = int(np.argmin(np.abs(self.times - moment)))
        if abs(self.times[best] - moment) > window:
            return None
        return float(self.times[best]), float(self.strengths[best])


def audio_onsets(
    samples: np.ndarray,
    sample_rate: int = _AUDIO_RATE,
    start: float = 0.0,
    config: EventConfig | None = None,
) -> AudioOnsets:
    """Finds the sharp sounds in a stretch of audio.

    Args:
        samples: Mono audio samples.
        sample_rate: Samples per second.
        start: The video time of the first sample, seconds.
        config: Supplies the weakest onset kept and how close two may be.

    Returns:
        The local peaks of the onset envelope at or above
        `audio_min_strength`; empty when the audio is silent or too short.
    """
    config = config or EventConfig()
    envelope = footage.onset_envelope(np.asarray(samples), sample_rate)
    if envelope is None:
        return AudioOnsets(np.zeros(0), np.zeros(0))
    step = max(1, sample_rate // 100) / sample_rate
    reach = max(1, round(config.audio_separation_seconds / step))
    padded = np.pad(envelope, reach)
    highest = np.max(
        [padded[shift : shift + len(envelope)] for shift in range(2 * reach + 1)], axis=0
    )
    peaks = np.flatnonzero((envelope >= highest) & (envelope >= config.audio_min_strength))
    return AudioOnsets(start + peaks * step, envelope[peaks].astype(float))


def read_audio_onsets(
    video: Path | str,
    start: float = 0.0,
    end: float | None = None,
    config: EventConfig | None = None,
    ffmpeg: str | None = None,
) -> AudioOnsets | None:
    """Reads a stretch of a recording's audio and finds its sharp sounds.

    Never raises: a recording with no audio track simply has no onsets to
    offer, and events are then timed from the picture alone.

    Args:
        video: The recording.
        start: First moment wanted, seconds.
        end: Last moment wanted; to the end if omitted.
        config: Onset settings.
        ffmpeg: ffmpeg binary; located automatically if not given.

    Returns:
        The onsets, on the video's clock, or None when the audio could
        not be read.
    """
    command = [ffmpeg or footage.ffmpeg_exe(), "-hide_banner", "-loglevel", "error"]
    command += ["-ss", f"{max(0.0, start):.3f}"]
    if end is not None:
        command += ["-t", f"{max(0.0, end - start):.3f}"]
    command += ["-i", str(video), "-vn", "-ac", "1", "-ar", str(_AUDIO_RATE), "-f", "s16le", "-"]
    try:
        result = subprocess.run(command, capture_output=True)
    except OSError:
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    samples = np.frombuffer(result.stdout[: len(result.stdout) // 2 * 2], dtype=np.int16)
    return audio_onsets(samples, _AUDIO_RATE, max(0.0, start), config)


# --------------------------------------------------------------------------
# What is reported
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HitEvent:
    """One contact between a racket and the ball.

    Attributes:
        pts: When, seconds on the video's clock.
        time_uncertainty: How far off that may be, seconds.
        side: The end the hitter was at, "near" or "far".
        player: The person, "A" or "B" (see `players`).
        zone: "deuce" or "ad", the half the hitter stood in.
        position: The hitter's feet on the court, (x, y) metres - the
            shot's `hit_x` / `hit_y`. None when the player was not
            tracked there.
        position_provenance: How that position was obtained.
        foot_clipped: True when the hitter's feet were cut off by the
            frame, so they stood further from the net than `position`.
        image_xy: Where the ball was in the picture at contact, pixels.
        confidence: 0 to 1.
        provenance: `MEASURED` for a kink seen in the track,
            `INTERPOLATED` when the contact fell in a short unseen
            stretch, `INFERRED` when it was concluded from the ball
            appearing at a player or from the ball coming back.
        cues: What supported it: "kink", "junction", "appearance",
            "returned" and, when a sound fixed its time, "audio".
        audio_strength: The matched onset's strength, or None.
        reason: The reasoning, in words - always given for an inferred
            hit.
    """

    pts: float
    time_uncertainty: float
    side: str
    player: str
    zone: str | None
    position: Point | None
    position_provenance: Provenance | None
    foot_clipped: bool
    image_xy: Point | None
    confidence: float
    provenance: Provenance
    cues: tuple[str, ...]
    audio_strength: float | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field, enums as their values.
        """
        origin = self.position_provenance
        return {
            "pts": self.pts,
            "time_uncertainty": self.time_uncertainty,
            "side": self.side,
            "player": self.player,
            "zone": self.zone,
            "position": None if self.position is None else list(self.position),
            "position_provenance": None if origin is None else origin.value,
            "foot_clipped": self.foot_clipped,
            "image_xy": None if self.image_xy is None else list(self.image_xy),
            "confidence": self.confidence,
            "provenance": self.provenance.value,
            "cues": list(self.cues),
            "audio_strength": self.audio_strength,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BounceEvent:
    """One contact between the ball and the ground.

    Attributes:
        pts: When, seconds.
        time_uncertainty: How far off that may be, seconds.
        court_xy: Where on the court, (x, y) metres.
        uncertainty_m: One standard deviation of that position across the
            court (x) and along it (y), metres - from the calibration's
            metres-per-pixel at that spot.
        half: The end of the court it landed at, "near" or "far".
        image_xy: Where it was in the picture, pixels.
        confidence: 0 to 1.
        provenance: `MEASURED` when the bounce frame was seen,
            `INTERPOLATED` when it fell in a short unseen stretch.
        reason: What it was recognised by.
    """

    pts: float
    time_uncertainty: float
    court_xy: Point
    uncertainty_m: Point
    half: str
    image_xy: Point
    confidence: float
    provenance: Provenance
    reason: str = ""

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field, enums as their values.
        """
        return {
            "pts": self.pts,
            "time_uncertainty": self.time_uncertainty,
            "court_xy": list(self.court_xy),
            "uncertainty_m": list(self.uncertainty_m),
            "half": self.half,
            "image_xy": list(self.image_xy),
            "confidence": self.confidence,
            "provenance": self.provenance.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class NetEvent:
    """The ball stopping at the net.

    Attributes:
        pts: When, seconds.
        image_xy: Where in the picture, pixels.
        confidence: 0 to 1.
        provenance: `MEASURED`: the stop was seen.
        reason: What it was recognised by.
    """

    pts: float
    image_xy: Point
    confidence: float
    provenance: Provenance
    reason: str = ""

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field, enums as their values.
        """
        return {
            "pts": self.pts,
            "image_xy": list(self.image_xy),
            "confidence": self.confidence,
            "provenance": self.provenance.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LineCall:
    """Whether a bounce was in, judged against the lines.

    Attributes:
        value: "In", "Out", or None when the bounce is closer to a line
            than its own uncertainty.
        margin_m: Distance inside the deciding line, metres - negative
            when outside it.
        line: The deciding line: "sideline", "centre line", "baseline",
            "service line" or "net".
        uncertainty_m: The bounce's uncertainty toward that line.
    """

    value: str | None
    margin_m: float
    line: str
    uncertainty_m: float


@dataclass(frozen=True)
class Shot:
    """One hit and what became of the ball.

    Attributes:
        index: Its position in `EventTrack.shots`.
        hit: The contact that started it.
        bounce: Where the ball first landed on the other side, if seen.
        net: The ball stopping at the net, if seen.
        returned: Whether the other player hit the ball next.
        result: "In" / "Out" / "Net", or a None value when it could not
            be told - always tagged with how it was obtained.
        result_reason: The reasoning behind the result, in words.
        call: The line call on `bounce`, when there is one.
        speed_mph: Average speed from the hitter's feet to the bounce -
            not a radar speed - or None when either end is unknown.
        direction: "cross court" / "down the line" (or, for a serve,
            "down the t" / "out wide"), or None when not derivable.
        serve: Whether it was judged as a serve, against a service box.
    """

    index: int
    hit: HitEvent
    bounce: BounceEvent | None
    net: NetEvent | None
    returned: bool
    result: Tagged[str | None]
    result_reason: str
    call: LineCall | None = None
    speed_mph: Tagged[float] | None = None
    direction: Tagged[str] | None = None
    serve: bool = False

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field; events nested as their own dicts.
        """
        return {
            "index": self.index,
            "hit": self.hit.to_dict(),
            "bounce": None if self.bounce is None else self.bounce.to_dict(),
            "net": None if self.net is None else self.net.to_dict(),
            "returned": self.returned,
            "result": self.result.to_dict(),
            "result_reason": self.result_reason,
            "call": None if self.call is None else dict(self.call.__dict__),
            "speed_mph": None if self.speed_mph is None else self.speed_mph.to_dict(),
            "direction": None if self.direction is None else self.direction.to_dict(),
            "serve": self.serve,
        }


@dataclass(frozen=True)
class EventQuality:
    """How much of a stretch's events were found, and how.

    Attributes:
        hits: Hits reported.
        hits_by_side: ...per end of the court.
        hits_inferred: Hits concluded from context rather than seen.
        hits_with_audio: Hits whose time a sound fixed.
        bounces: Bounces reported.
        bounces_by_half: ...per end of the court they landed at.
        nets: Net contacts reported.
        shots: Shots (one per hit).
        results: Shots per result; "unknown" counts the abstentions.
        results_inferred: Results concluded from what happened next.
        unknown_by_landing_half: Unknown results per end the ball was
            hit toward - where the tracker could not see.
        bounce_seen_by_landing_half: Per end the ball was hit toward,
            (shots with a bounce, shots).
        speeds: Shots with a speed.
        discarded_kinks: Kinks in the track that were taken for none of
            a hit, a bounce or a net contact.
    """

    hits: int
    hits_by_side: dict[str, int]
    hits_inferred: int
    hits_with_audio: int
    bounces: int
    bounces_by_half: dict[str, int]
    nets: int
    shots: int
    results: dict[str, int]
    results_inferred: int
    unknown_by_landing_half: dict[str, int]
    bounce_seen_by_landing_half: dict[str, tuple[int, int]]
    speeds: int
    discarded_kinks: int

    @property
    def abstention_rate(self) -> float:
        """Share of shots whose result is unknown; 0.0 with no shots."""
        return self.results["unknown"] / self.shots if self.shots else 0.0

    def notice(self) -> str | None:
        """Words the result for a person, when much was left unknown.

        Returns:
            A sentence saying how many shots have no result and at which
            end, or None when fewer than a quarter are unknown.
        """
        if self.shots == 0 or self.abstention_rate < 0.25:
            return None
        far, near = self.unknown_by_landing_half[FAR], self.unknown_by_landing_half[NEAR]
        return (
            f"{self.results['unknown']} of {self.shots} shots have no result: the ball's landing "
            f"was not seen ({far} hit toward the far end, {near} toward the near end). "
            "Those shots need a person to look."
        )

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field, plus the abstention rate.
        """
        out = dict(self.__dict__)
        out["bounce_seen_by_landing_half"] = {
            half: list(pair) for half, pair in self.bounce_seen_by_landing_half.items()
        }
        out["abstention_rate"] = self.abstention_rate
        return out


@dataclass(frozen=True)
class EventTrack:
    """Everything that happened to the ball over a stretch of video.

    Attributes:
        hits: Racket contacts, in time order.
        bounces: Ground contacts, in time order.
        nets: Net contacts, in time order.
        shots: One per hit, in time order.
        ball: The ball track the events were read from, with `court_xy`
            filled in on the observation at each measured bounce (and
            nowhere else).
        config: The settings the shots were judged with.
        discarded_kinks: Kinks taken for no event.
        source: What produced the events.
    """

    hits: tuple[HitEvent, ...]
    bounces: tuple[BounceEvent, ...]
    nets: tuple[NetEvent, ...]
    shots: tuple[Shot, ...]
    ball: BallTrack
    config: EventConfig = field(default_factory=EventConfig)
    discarded_kinks: int = 0
    source: str = SOURCE

    def rejudged(self, serves: Iterable[int]) -> EventTrack:
        """Judges the named shots as serves and every other as a rally shot.

        Which hits are serves is the stroke classifier's decision, made
        after the events exist; a serve must land in the service box
        diagonally opposite, not merely in the court. Which box that is
        depends on the half the server stood in, so a serve by a player
        who was not tracked at contact is not judged against either.

        Args:
            serves: Indexes into `shots` of the shots that are serves.

        Returns:
            The same events with every shot's result, direction and
            `serve` flag recomputed.
        """
        chosen = set(serves)
        shots = tuple(_judge(shot, shot.index in chosen, self.config) for shot in self.shots)
        return replace(self, shots=shots)

    def quality(self) -> EventQuality:
        """Summarises what was found and what was left unknown.

        Returns:
            Counts of events, results and abstentions, by court half.
        """
        results = {name: 0 for name in RESULTS} | {"unknown": 0}
        unknown = {NEAR: 0, FAR: 0}
        seen = {NEAR: [0, 0], FAR: [0, 0]}
        for shot in self.shots:
            landing = _other(shot.hit.side)
            value = shot.result.value
            results[value or "unknown"] += 1
            unknown[landing] += value is None
            seen[landing][1] += 1
            seen[landing][0] += shot.bounce is not None
        inferred = (Provenance.INFERRED,)
        return EventQuality(
            hits=len(self.hits),
            hits_by_side={side: sum(hit.side == side for hit in self.hits) for side in SIDES},
            hits_inferred=sum(hit.provenance in inferred for hit in self.hits),
            hits_with_audio=sum("audio" in hit.cues for hit in self.hits),
            bounces=len(self.bounces),
            bounces_by_half={
                side: sum(bounce.half == side for bounce in self.bounces) for side in SIDES
            },
            nets=len(self.nets),
            shots=len(self.shots),
            results=results,
            results_inferred=sum(
                shot.result.value is not None and shot.result.provenance in inferred
                for shot in self.shots
            ),
            unknown_by_landing_half=unknown,
            bounce_seen_by_landing_half={side: (seen[side][0], seen[side][1]) for side in SIDES},
            speeds=sum(shot.speed_mph is not None for shot in self.shots),
            discarded_kinks=self.discarded_kinks,
        )

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Hits, bounces, nets, shots and the quality summary. The ball
            track is not repeated here; it serialises itself.
        """
        return {
            "source": self.source,
            "court": self.config.court,
            "hits": [hit.to_dict() for hit in self.hits],
            "bounces": [bounce.to_dict() for bounce in self.bounces],
            "nets": [net.to_dict() for net in self.nets],
            "shots": [shot.to_dict() for shot in self.shots],
            "quality": self.quality().to_dict(),
        }


# --------------------------------------------------------------------------
# Court geometry: where a shot must land, how sure a bounce is, how fast
# --------------------------------------------------------------------------


def target_region(
    hitter_side: str, court: str = "singles", serve_zone: str | None = None
) -> tuple[float, float, float, float]:
    """Gives the rectangle a shot must land in to be in.

    Args:
        hitter_side: The end the shot was hit from, "near" or "far".
        court: "singles" or "doubles" - which sidelines bound a rally
            shot. A serve is bounded by the singles sideline either way.
        serve_zone: None for a rally shot. For a serve, the half the
            server stood in ("deuce" or "ad"): the target is the service
            box diagonally opposite.

    Returns:
        (x_min, x_max, y_min, y_max) in court metres.

    Raises:
        ValueError: If the side, court or zone is not a known one.
    """
    if hitter_side not in SIDES:
        raise ValueError(f"hitter_side must be 'near' or 'far', got {hitter_side!r}.")
    if court not in ("singles", "doubles"):
        raise ValueError(f"court must be 'singles' or 'doubles', got {court!r}.")
    if serve_zone not in (None, DEUCE, AD):
        raise ValueError(f"serve_zone must be 'deuce', 'ad' or None, got {serve_zone!r}.")
    half = (SINGLES_WIDTH_M if court == "singles" or serve_zone else DOUBLES_WIDTH_M) / 2
    low, high = -half, half
    if serve_zone is not None:
        # Deuce is the server's own right. A near server's right is +x, so
        # the diagonal box is at -x; for a far server it is the reverse.
        to_camera_left = (hitter_side == NEAR) == (serve_zone == DEUCE)
        low, high = (-half, 0.0) if to_camera_left else (0.0, half)
    depth = SERVICE_LINE_FROM_NET_M if serve_zone is not None else COURT_LENGTH_M / 2
    if hitter_side == NEAR:
        return (low, high, NET_Y_M, NET_Y_M + depth)
    return (low, high, NET_Y_M - depth, NET_Y_M)


def judge_bounce(
    court_xy: Point,
    uncertainty_m: Point,
    hitter_side: str,
    court: str = "singles",
    serve_zone: str | None = None,
    sigmas: float = 1.0,
) -> LineCall:
    """Calls a bounce in or out - or declines to.

    The call is made on the line the bounce is *least clearly* inside,
    measured in units of the bounce's own uncertainty toward that line.
    Painted lines are a few centimetres wide and count as in; that width
    is far below any uncertainty here and is ignored.

    Args:
        court_xy: Where the ball landed, (x, y) court metres.
        uncertainty_m: Its uncertainty across (x) and along (y) the court.
        hitter_side: The end it was hit from.
        court: "singles" or "doubles".
        serve_zone: The server's half for a serve, None for a rally shot.
        sigmas: How many uncertainties from the line a call needs.

    Returns:
        The call, the margin and the deciding line. `value` is None when
        the bounce is within `sigmas` uncertainties of that line.

    Raises:
        ValueError: As `target_region`.
    """
    low, high, front, back = target_region(hitter_side, court, serve_zone)
    x, y = court_xy
    across = max(float(uncertainty_m[0]), 1e-6)
    along = max(float(uncertainty_m[1]), 1e-6)
    serving = serve_zone is not None
    far_line = "service line" if serving else "baseline"
    if hitter_side == NEAR:
        depth = [(back - y, along, far_line), (y - front, along, "net")]
    else:
        depth = [(y - front, along, far_line), (back - y, along, "net")]
    sides = [
        (x - low, across, "centre line" if serving and low == 0.0 else "sideline"),
        (high - x, across, "centre line" if serving and high == 0.0 else "sideline"),
    ]
    margin, spread, line = min(depth + sides, key=lambda item: item[0] / item[1])
    ratio = margin / spread
    value = RESULT_IN if ratio >= sigmas else RESULT_OUT if ratio <= -sigmas else None
    return LineCall(value=value, margin_m=float(margin), line=line, uncertainty_m=float(spread))


def ground_uncertainty(calibration: Calibration, image_xy: Point, pixels: float) -> Point:
    """Turns a pixel uncertainty at an image spot into metres on the court.

    Args:
        calibration: The court calibration.
        image_xy: The spot, pixels.
        pixels: One standard deviation of the position there, pixels, the
            same in every direction.

    Returns:
        (across, along): one standard deviation of the court position in
        x and in y, metres, with the calibration's own median residual
        added in quadrature. Infinite where the spot maps nowhere.
    """
    spot = np.asarray(image_xy, dtype=float)
    steps = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    mapped = calibration.to_court(spot + steps)
    if not np.isfinite(mapped).all():
        return (float("inf"), float("inf"))
    by_column = (mapped[0] - mapped[1]) / 2.0
    by_row = (mapped[2] - mapped[3]) / 2.0
    floor = calibration.quality.residual_m_median
    across = float(np.hypot(pixels * np.hypot(by_column[0], by_row[0]), floor))
    along = float(np.hypot(pixels * np.hypot(by_column[1], by_row[1]), floor))
    return (across, along)


def average_speed_mph(
    start_xy: Point,
    start_pts: float,
    end_xy: Point,
    end_pts: float,
    config: EventConfig | None = None,
) -> float | None:
    """Computes a shot's average speed over the court plane.

    Straight-line distance between two court positions over the time
    between them. An average over the flight, ignoring height - not the
    speed off the racket.

    Args:
        start_xy: Where it was hit from, court metres.
        start_pts: When, seconds.
        end_xy: Where it bounced.
        end_pts: When.
        config: Supplies the shortest time and the fastest speed believed.

    Returns:
        Miles per hour, or None when the time is too short to divide by
        or the answer is faster than a ball can travel.
    """
    config = config or EventConfig()
    elapsed = end_pts - start_pts
    if elapsed < config.min_speed_seconds:
        return None
    metres = float(np.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]))
    speed = metres / elapsed * _MPS_TO_MPH
    return speed if speed <= config.max_speed_mph else None


def shot_direction(
    hit_xy: Point, bounce_xy: Point, serve: bool = False, uncertainty_m: float = 0.0
) -> str | None:
    """Names where a shot was aimed, from where it was hit and landed.

    A rally shot is "cross court" when it lands on the other side of the
    centre line from where it was hit, "down the line" when on the same
    side. A serve is "down the t" when it lands in the half of the
    service box nearer the centre line, "out wide" otherwise.
    ("Inside out" and "inside in" need the stroke as well; see
    `strokes.refine_direction`.)

    Args:
        hit_xy: The hitter's feet, court metres.
        bounce_xy: The bounce.
        serve: Whether the shot is a serve.
        uncertainty_m: The bounce's uncertainty across the court.

    Returns:
        The direction, or None when the bounce is too close to the
        deciding line to tell.
    """
    if serve:
        middle = SINGLES_WIDTH_M / 4
        if abs(abs(bounce_xy[0]) - middle) <= uncertainty_m:
            return None
        return DOWN_THE_T if abs(bounce_xy[0]) < middle else OUT_WIDE
    if abs(bounce_xy[0]) <= uncertainty_m or hit_xy[0] == 0.0:
        return None
    return CROSS_COURT if (hit_xy[0] > 0) != (bounce_xy[0] > 0) else DOWN_THE_LINE


def _judge(shot: Shot, serve: bool, config: EventConfig) -> Shot:
    """Works out a shot's result, speed and direction from its events."""
    hit, bounce = shot.hit, shot.bounce
    call = None
    over = bounce is not None and bounce.half != hit.side
    boxless = serve and hit.zone is None
    if over and not boxless:
        zone = hit.zone if serve else None
        call = judge_bounce(
            bounce.court_xy, bounce.uncertainty_m, hit.side, config.court, zone, config.call_sigmas
        )
    unseen = (
        "where the server stood is not known, so the bounce cannot be judged against a service box"
        if over
        else "no bounce was seen"
    )
    guess = config.inferred_confidence
    if call is not None and call.value == RESULT_IN:
        result = Tagged(RESULT_IN, bounce.provenance, bounce.confidence, SOURCE)
        reason = f"bounced {call.margin_m:.2f}m inside the {call.line}"
    elif call is not None and call.value == RESULT_OUT and shot.returned and not serve:
        result = Tagged(None, Provenance.INFERRED, None, SOURCE)
        reason = (
            f"bounce measured {-call.margin_m:.2f}m beyond the {call.line}, "
            "but the other player hit it and play went on - the two disagree"
        )
    elif call is not None and call.value == RESULT_OUT:
        result = Tagged(RESULT_OUT, bounce.provenance, bounce.confidence, SOURCE)
        reason = f"bounced {-call.margin_m:.2f}m beyond the {call.line}"
    elif call is not None and shot.returned:
        result = Tagged(RESULT_IN, Provenance.INFERRED, guess, SOURCE)
        reason = (
            f"bounce within {call.uncertainty_m:.2f}m uncertainty of the {call.line} "
            f"({call.margin_m:+.2f}m); the other player hit it next"
        )
    elif call is not None:
        result = Tagged(None, bounce.provenance, None, SOURCE)
        reason = (
            f"bounce within {call.uncertainty_m:.2f}m uncertainty of the {call.line} "
            f"({call.margin_m:+.2f}m): too close to call"
        )
    elif shot.net is not None and not shot.returned:
        result = Tagged(RESULT_NET, shot.net.provenance, shot.net.confidence, SOURCE)
        reason = "the ball stopped at the net"
    elif bounce is not None and not over and not shot.returned:
        result = Tagged(RESULT_NET, Provenance.INFERRED, guess * bounce.confidence, SOURCE)
        reason = "the ball landed on the hitter's own side of the net"
    elif shot.returned:
        result = Tagged(RESULT_IN, Provenance.INFERRED, guess, SOURCE)
        reason = f"{unseen}; the other player hit the ball next"
    elif over:
        result = Tagged(None, Provenance.INFERRED, None, SOURCE)
        reason = unseen
    else:
        result = Tagged(None, Provenance.INFERRED, None, SOURCE)
        reason = "no bounce, net contact or return was seen"
    speed = direction = None
    if call is not None and hit.position is not None:
        origin = weakest(
            [hit.provenance, hit.position_provenance or hit.provenance, bounce.provenance]
        )
        sure = min(hit.confidence, bounce.confidence)
        mph = average_speed_mph(hit.position, hit.pts, bounce.court_xy, bounce.pts, config)
        if mph is not None:
            speed = Tagged(mph, origin, sure, SOURCE)
        name = shot_direction(hit.position, bounce.court_xy, serve, bounce.uncertainty_m[0])
        if name is not None:
            direction = Tagged(name, origin, sure, SOURCE)
    return replace(
        shot,
        result=result,
        result_reason=reason,
        call=call,
        speed_mph=speed,
        direction=direction,
        serve=serve,
    )


# --------------------------------------------------------------------------
# Kinks: where the ball's path stops being smooth
# --------------------------------------------------------------------------

_KINK = "kink"
_JUNCTION = "junction"
_APPEARANCE = "appearance"
_RETURNED = "returned"


@dataclass
class _Kink:
    """A place the ball's image path changes abruptly.

    Attributes:
        pts: When.
        xy: Where in the image.
        before: Image velocity leading in, px/s; None for an appearance.
        after: Image velocity leading out, px/s.
        kind: `_KINK`, `_JUNCTION` or `_APPEARANCE`.
        gap: Seconds of unseen ball it sits in (0 for a seen kink).
        blur: Extra position uncertainty, pixels: how far the ball moves
            in half a frame, plus how far the two paths miss at a
            junction.
        confidence: Mean confidence of the ball observations around it.
        extent: Median size, in pixels, of the ball's blob around it;
            None when no measured observation is close.
        path: For an appearance, the fitted path the ball then follows
            (x and y polynomials in seconds since `pts`), so it can be
            traced back to where it came from.
    """

    pts: float
    xy: np.ndarray
    before: np.ndarray | None
    after: np.ndarray
    kind: str
    gap: float = 0.0
    blur: float = 0.0
    confidence: float = 0.0
    extent: float | None = None
    path: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def jump(self) -> float:
        """Size of the velocity change, px/s (the speed out for an appearance)."""
        if self.before is None:
            return float(np.hypot(*self.after))
        return float(np.hypot(*(self.after - self.before)))

    def eventful(self, scale: np.ndarray, config: EventConfig) -> bool:
        """Whether the change is big enough, in metres and against the ball's speed."""
        row = int(np.clip(round(float(self.xy[1])), 0, len(scale) - 1))
        fastest = max(float(np.hypot(*self.before)), float(np.hypot(*self.after)), 1e-9)
        return (
            self.jump / scale[row] >= config.min_kink_mps
            and self.jump / fastest >= config.min_kink_ratio
        )


def _fit(times: np.ndarray, points: np.ndarray, origin: float) -> tuple[np.ndarray, np.ndarray]:
    """Fits x(t) and y(t) around `origin`: a parabola, or a line when short."""
    degree = 2 if len(times) >= 5 else 1
    shifted = times - origin
    return np.polyfit(shifted, points[:, 0], degree), np.polyfit(shifted, points[:, 1], degree)


def _position(fit: tuple[np.ndarray, np.ndarray], offset: float) -> np.ndarray:
    """Evaluates a fitted path `offset` seconds from its origin."""
    return np.array([np.polyval(fit[0], offset), np.polyval(fit[1], offset)])


def _velocity(fit: tuple[np.ndarray, np.ndarray], offset: float) -> np.ndarray:
    """Evaluates a fitted path's velocity `offset` seconds from its origin."""
    return np.array(
        [np.polyval(np.polyder(fit[0]), offset), np.polyval(np.polyder(fit[1]), offset)]
    )


def _arrays(observations: Sequence[BallObservation]) -> tuple[np.ndarray, np.ndarray]:
    """Splits observations into their times and their image positions."""
    times = np.array([o.pts for o in observations])
    return times, np.array([o.image_xy for o in observations], dtype=float)


def _extent(observations: Sequence[BallObservation]) -> float | None:
    """The median blob size of the measured observations among these."""
    sizes = [o.extent for o in observations if o.extent is not None]
    return float(np.median(sizes)) if sizes else None


def _end(
    observations: Sequence[BallObservation], config: EventConfig, first: bool
) -> list[BallObservation]:
    """Gives the first or last stretch of a run that a path is fitted to."""
    if first:
        limit = observations[0].pts + config.fit_seconds
        chosen = [o for o in observations if o.pts <= limit]
        return chosen if len(chosen) >= 2 else list(observations[:2])
    limit = observations[-1].pts - config.fit_seconds
    chosen = [o for o in observations if o.pts >= limit]
    return chosen if len(chosen) >= 2 else list(observations[-2:])


def _segment_kinks(
    observations: Sequence[BallObservation], scale: np.ndarray, config: EventConfig
) -> list[_Kink]:
    """Finds the kinks inside one segment of the ball's track."""
    times, points = _arrays(observations)
    found: list[_Kink] = []
    for index in range(len(times)):
        left = (times >= times[index] - config.fit_seconds) & (times <= times[index])
        right = (times >= times[index]) & (times <= times[index] + config.fit_seconds)
        if left.sum() < config.min_fit_points or right.sum() < config.min_fit_points:
            continue
        around = observations[max(0, index - 6) : index + 7]
        kink = _Kink(
            pts=float(times[index]),
            xy=points[index],
            before=_velocity(_fit(times[left], points[left], times[index]), 0.0),
            after=_velocity(_fit(times[right], points[right], times[index]), 0.0),
            kind=_KINK,
            blur=float(np.hypot(*(points[index + 1] - points[index - 1]))) / 4,
            confidence=float(np.mean([o.confidence for o in around])),
            extent=_extent(around),
        )
        if kink.eventful(scale, config):
            found.append(kink)
    # Around a real kink several neighbouring frames qualify; keep the sharpest.
    return [
        kink
        for kink in found
        if kink.jump
        >= max(
            other.jump
            for other in found
            if abs(other.pts - kink.pts) <= config.kink_separation_seconds
        )
    ]


def _start_of(observations: Sequence[BallObservation], config: EventConfig, gap: float) -> _Kink:
    """Describes a segment's first moment: the ball appearing."""
    head = _end(observations, config, first=True)
    times, points = _arrays(head)
    path = _fit(times, points, times[0])
    return _Kink(
        pts=float(times[0]),
        xy=points[0],
        before=None,
        after=_velocity(path, 0.0),
        kind=_APPEARANCE,
        gap=gap,
        confidence=float(np.mean([o.confidence for o in head])),
        extent=_extent(head),
        path=path,
    )


def _junction(
    earlier: Sequence[BallObservation],
    later: Sequence[BallObservation],
    image_width: int,
    config: EventConfig,
) -> _Kink | None:
    """Looks for the one event that joins two segments across an unseen gap.

    Both paths are extended into the gap; where they pass closest is
    where the ball was touched. Returns None when they never come close -
    more than one thing happened in the gap, or these are not the same
    ball - or when the gap is too long to extend across: on the real
    recording two paths either side of a 0.4s gap "met" a quarter of a
    second before the hit that actually separated them.
    """
    start, end = earlier[-1].pts, later[0].pts
    gap = end - start
    if gap > config.max_junction_seconds:
        return None
    tail, head = _end(earlier, config, first=False), _end(later, config, first=True)
    one, two = _fit(*_arrays(tail), start), _fit(*_arrays(head), start)
    offsets = np.linspace(0.0, gap, 41)
    misses = [float(np.hypot(*(_position(one, at) - _position(two, at)))) for at in offsets]
    best = int(np.argmin(misses))
    at = float(offsets[best])
    before, after = _velocity(one, at), _velocity(two, at)
    fastest = max(float(np.hypot(*before)), float(np.hypot(*after)))
    allowed = config.junction_miss * image_width + config.junction_miss_seconds * fastest
    if misses[best] > allowed:
        return None
    return _Kink(
        pts=start + at,
        xy=(_position(one, at) + _position(two, at)) / 2,
        before=before,
        after=after,
        kind=_JUNCTION,
        gap=gap,
        blur=misses[best] / 2 + fastest * gap / 40,
        confidence=float(np.mean([o.confidence for o in tail + head])) * 0.5,
        extent=_extent(tail + head),
    )


def find_kinks(
    ball: BallTrack, calibration: Calibration, config: EventConfig | None = None
) -> list[_Kink]:
    """Lists every abrupt change in the ball's path, in time order.

    Segments that never left a player's box (`confusion_fraction`) are a
    racket or a hand and are left out.

    Args:
        ball: The ball's track.
        calibration: The court calibration; gives the metres a pixel of
            velocity change is worth at each image row.
        config: Thresholds; see `EventConfig`.

    Returns:
        Kinks inside segments, junctions between segments, and
        appearances where a segment starts with nothing leading in.
    """
    config = config or EventConfig()
    scale = row_scale(calibration)
    runs: dict[int, list[BallObservation]] = {}
    for observation in ball.observations:
        runs.setdefault(observation.segment, []).append(observation)
    kept = [
        runs.get(segment.index, [])
        for segment in ball.segments
        if segment.in_player_box < config.confusion_fraction
    ]
    kept = sorted((run for run in kept if len(run) >= 2), key=lambda run: run[0].pts)
    kinks: list[_Kink] = []
    for number, run in enumerate(kept):
        previous = kept[number - 1] if number else None
        joined = None
        if previous is not None and run[0].pts > previous[-1].pts:
            joined = _junction(previous, run, ball.image_size[0], config)
        if joined is None:
            gap = float("inf") if previous is None else run[0].pts - previous[-1].pts
            kinks.append(_start_of(run, config, gap))
        elif joined.eventful(scale, config):
            kinks.append(joined)
        kinks.extend(_segment_kinks(run, scale, config))
    kinks.sort(key=lambda kink: kink.pts)
    merged: list[_Kink] = []
    for kink in kinks:
        if not merged or kink.pts - merged[-1].pts > config.kink_separation_seconds:
            merged.append(kink)
        elif (kink.kind != _APPEARANCE, kink.jump) > (
            merged[-1].kind != _APPEARANCE,
            merged[-1].jump,
        ):
            merged[-1] = kink
    return merged


# --------------------------------------------------------------------------
# Reading the kinks: hit, bounce, net or nothing
# --------------------------------------------------------------------------


class _Scene:
    """The court, the camera and the players, as the event rules need them."""

    def __init__(
        self,
        ball: BallTrack,
        players: PlayerTracks | None,
        calibration: Calibration,
        audio: AudioOnsets | None,
        config: EventConfig,
    ) -> None:
        """Prepares lookups: row scale, the net's outline, player times."""
        self.ball, self.players, self.calibration = ball, players, calibration
        self.audio, self.config = audio, config
        self.scale = row_scale(calibration)
        self.width = ball.image_size[0]
        self.times = {side: [o.pts for o in self._side(side)] for side in SIDES}
        self.centre = np.array([0.0, -6.0, 2.5])
        self.net: np.ndarray | None = None
        self.tape: np.ndarray | None = None
        camera = camera_from_homography(calibration.homography, calibration.image_size)
        if camera is not None:
            self.centre = -np.linalg.inv(camera[:, :3]) @ camera[:, 3]
            reach = DOUBLES_WIDTH_M / 2 + 0.914

            def outline(low: float, high: float) -> np.ndarray:
                """The image outline of a band of the net between two heights."""
                corners = np.array(
                    [
                        (-reach, NET_Y_M, low, 1.0),
                        (reach, NET_Y_M, low, 1.0),
                        (reach, NET_Y_M, high, 1.0),
                        (-reach, NET_Y_M, high, 1.0),
                    ]
                )
                mapped = corners @ camera.T
                return (mapped[:, :2] / mapped[:, 2:]).astype(np.float32)

            self.net = outline(0.0, NET_HEIGHT_POST_M)
            self.tape = outline(NET_HEIGHT_CENTRE_M - 0.06, NET_HEIGHT_POST_M + 0.02)

    def _side(self, side: str) -> tuple[PlayerObservation, ...]:
        """One end's detections, or nothing when there are no players."""
        return () if self.players is None else self.players.side(side)

    def player(self, pts: float, side: str) -> PlayerObservation | None:
        """The player at an end at a moment, or their nearest detection in time."""
        if self.players is None:
            return None
        seen = self.players.at(pts, side)
        if seen is not None:
            return seen
        times = self.times[side]
        after = bisect.bisect_left(times, pts)
        nearby = [index for index in (after - 1, after) if 0 <= index < len(times)]
        if not nearby:
            return None
        best = min(nearby, key=lambda index: abs(times[index] - pts))
        if abs(times[best] - pts) > self.config.player_lookup_seconds:
            return None
        return self._side(side)[best]

    def row_scale(self, row: float) -> float:
        """Pixels per metre on the ground at an image row."""
        return float(self.scale[int(np.clip(round(row), 0, len(self.scale) - 1))])

    def _inside(self, outline: np.ndarray | None, xy: np.ndarray, pad: float) -> bool:
        """Whether an image point lies in (or within `pad` image widths of) an outline."""
        if outline is None:
            return False
        spot = (float(xy[0]), float(xy[1]))
        return cv2.pointPolygonTest(outline, spot, True) >= -pad * self.width

    def at_net(self, xy: np.ndarray) -> bool:
        """Whether an image point lies on the net as the camera sees it."""
        return self._inside(self.net, xy, self.config.net_pad)

    def behind_tape(self, xy: np.ndarray) -> bool:
        """Whether an image point lies on the net's white tape.

        Nothing on the ground can be seen through the tape, and a ball
        crossing it merges with it for a frame or two and seems to jerk:
        on the real recording that jerk was read as a bounce four metres
        from where the ball actually landed.
        """
        return self._inside(self.tape, xy, self.config.tape_pad)

    def sound_delay(self, position: Point | None, side: str) -> float:
        """Seconds a hit's sound takes to reach the camera."""
        if position is None:
            position = (0.0, 0.0) if side == NEAR else (0.0, COURT_LENGTH_M)
        away = np.linalg.norm(np.array([position[0], position[1], 1.0]) - self.centre)
        return float(away) / self.config.speed_of_sound_mps


@dataclass
class _Reading:
    """One kink with everything the rules ask about it.

    Attributes:
        kink: The kink.
        ground: Where its sightline meets the court, or None.
        half: The end of the court that point is at, or None.
        people: The player at each end then, or None.
        reach: Metres from the ball to each player's box - infinite when
            that player is unknown, or when the ball looks the wrong size
            to be at their distance. For an appearance, the nearest the
            ball's path comes when followed back.
        moment: When the ball was that near each player: the kink's own
            time, or earlier for an appearance followed back.
        boxed: Whether the kink lies inside a player's box in the
            picture, whatever the ball's size says about its distance -
            where the track is least reliable.
        bounce_like: Whether it has a bounce's signature.
    """

    kink: _Kink
    ground: Point | None
    half: str | None
    people: dict[str, PlayerObservation | None]
    reach: dict[str, float]
    moment: dict[str, float]
    boxed: bool
    bounce_like: bool


def _read(kink: _Kink, scene: _Scene) -> _Reading:
    """Measures a kink against the court and the players.

    One pixel cannot say how far away the ball is: a far-court ball just
    above the near player's head in the picture is on the same sightline
    as a ball at their racket. Its apparent *size* can - a ball at the
    near baseline looks four or five times as wide as one at the far
    baseline - so a ball that looks the wrong size to be at a player's
    distance is not within their reach, wherever it is in the picture.

    A ball that appears mid-flight was usually lost around contact, so
    its fitted path is followed backward - no further than it went unseen
    - to the latest moment it was nearest each player.

    A kink on the net's tape is nothing at all - neither a bounce nor
    within anyone's reach; see `_Scene.behind_tape`.
    """
    config = scene.config
    ground = _on_ground(scene.calibration, np.asarray(kink.xy, dtype=float))
    people: dict[str, PlayerObservation | None] = {}
    reach: dict[str, float] = {}
    moment: dict[str, float] = {}
    boxed = False
    trail = [(kink.pts, kink.xy)]
    if kink.kind == _APPEARANCE:
        offsets = np.linspace(-min(kink.gap, config.max_lookback_seconds), 0.0, 31)
        trail = [(kink.pts + float(at), _position(kink.path, float(at))) for at in offsets]
    hidden = scene.behind_tape(kink.xy)
    for side in SIDES:
        person = scene.player(kink.pts, side)
        people[side] = person
        reach[side] = float("inf")
        if person is None or hidden:
            continue
        scale = scene.row_scale(person.foot_image[1])
        box = person.box
        boxed = boxed or (box[0] <= kink.xy[0] <= box[2] and box[1] <= kink.xy[1] <= box[3])
        if kink.extent is not None:
            looks = kink.extent / (config.ball_diameter_m * scale)
            if not config.min_size_ratio <= looks <= config.max_size_ratio:
                continue
        away = [
            float(
                np.hypot(
                    max(box[0] - spot[0], 0.0, spot[0] - box[2]),
                    max(box[1] - spot[1], 0.0, spot[1] - box[3]),
                )
            )
            for _, spot in trail
        ]
        latest = max(index for index, far in enumerate(away) if far <= min(away) + 1e-6)
        reach[side] = away[latest] / scale
        moment[side] = trail[latest][0]
    wide = DOUBLES_WIDTH_M / 2 + config.bounce_margin_m
    on_court = (
        ground is not None
        and abs(ground[0]) <= wide
        and -config.bounce_margin_m <= ground[1] <= COURT_LENGTH_M + config.bounce_margin_m
    )
    bounce_like = False
    if on_court and kink.before is not None and not hidden:
        turn = float(kink.after[1] - kink.before[1])
        sideways = abs(float(kink.after[0] - kink.before[0]))
        reversed_sideways = kink.before[0] * kink.after[0] < 0 and min(
            abs(kink.before[0]), abs(kink.after[0])
        ) > config.bounce_reversal * abs(turn)
        kept = np.hypot(*kink.after) >= config.bounce_min_keep * np.hypot(*kink.before)
        bounce_like = bool(
            kink.before[1] > 0
            and turn < 0
            and sideways <= config.bounce_sideways * abs(turn)
            and not reversed_sideways
            and kept
        )
    return _Reading(
        kink=kink,
        ground=ground,
        half=None if ground is None else side_of(ground[1]),
        people=people,
        reach=reach,
        moment=moment,
        boxed=bool(boxed),
        bounce_like=bounce_like,
    )


def _strength(reading: _Reading, side: str, scene: _Scene) -> float:
    """How hard a kink is as a hit by one player, in m/s at their distance.

    The smaller of the change in the ball's image velocity and the speed
    it leaves at: a hit changes the ball's motion *and* sends it
    somewhere, so a ball that merely stops (in the net, in a hand) is no
    hit however abruptly it does so.

    Zero for a seen bounce at or below the level of the far player's feet
    in the picture: from a camera behind the near baseline the ground in
    front of the far player lies *below* their feet in the image, where a
    ball bounces and no racket reaches.

    Zero, too, for a ball that leaves the near player moving *down* the
    picture while below the horizon: a ball hit away from the camera
    from below camera height moves up the image. (A serve or a smash is
    struck above the horizon and is exempt.) This is what keeps a stray
    ball rolling past the near player from being read as their shot.
    """
    person = reading.people[side]
    scale = scene.row_scale(person.foot_image[1])
    if side == NEAR and reading.ground is not None and reading.kink.after[1] >= 0:
        return 0.0
    if side == FAR and reading.bounce_like and reading.kink.kind == _KINK:
        if reading.kink.xy[1] >= person.foot_image[1] - scene.config.ground_level_m * scale:
            return 0.0
    return min(reading.kink.jump, float(np.hypot(*reading.kink.after))) / scale


def _eligible(reading: _Reading, last: HitEvent | None, config: EventConfig) -> str | None:
    """Says which player, if either, could have hit the ball at a kink."""
    if last is None:
        order = sorted(SIDES, key=lambda side: reading.reach[side])
    else:
        order = [_other(last.side), last.side]
    for side in order:
        if reading.reach[side] > config.reach_m:
            continue
        if last is not None:
            since = reading.kink.pts - last.pts
            needed = config.rehit_seconds if side == last.side else config.min_flight_seconds
            if since < needed:
                continue
        return side
    return None


def _choose(
    visit: list[_Reading], following: list[_Reading], side: str, scene: _Scene
) -> int | None:
    """Picks the hit among the kinks of one visit of the ball to a player.

    A bounce in front of a player and their hit both happen within their
    reach, bounce first. So the hit is the *last* kink of the visit that
    is hard enough - at least a quarter as hard as the hardest that does
    not look like a bounce. When the only hard kink is a seen one that looks
    like a bounce, it is the hit only if the ball is then seen at the
    other end; otherwise it was a bounce nobody returned.
    """
    config = scene.config
    hard = [_strength(reading, side, scene) for reading in visit]
    plain = [hard[index] for index, reading in enumerate(visit) if not reading.bounce_like]
    floor = max(config.min_hit_mps, 0.25 * max(plain or hard))
    strong = [index for index in range(len(visit)) if hard[index] >= floor]
    if not strong:
        return None
    pick = strong[-1]
    if len(strong) > 1 or not visit[pick].bounce_like or visit[pick].kink.kind != _KINK:
        return pick
    other = _other(side)
    horizon = visit[pick].kink.pts + config.max_return_seconds
    went = any(
        later.kink.pts <= horizon
        and (later.reach[other] <= config.reach_m or (later.bounce_like and later.half == other))
        for later in visit[pick + 1 :] + following
    )
    return pick if went else None


def _hit_from(
    reading: _Reading, side: str, scene: _Scene, cue: str | None = None, reason: str = ""
) -> HitEvent:
    """Builds the hit a kink stands for, confirmed by the audio when it can be.

    A sharp sound arriving when this player's hit would be heard confirms
    the hit. It also replaces the hit's time, but only when the picture
    was vague about it (an appearance, or a junction across more than a
    few frames): measured on the real recording, a hit seen in the track
    is already timed to about 0.03s, and moving it onto the nearest sound
    - often the neighbouring court's - made it worse, not better.
    """
    config, kink = scene.config, reading.kink
    person = reading.people[side]
    cue = cue or kink.kind
    pts = reading.moment.get(side, kink.pts)
    doubt = {_KINK: 0.02, _JUNCTION: max(0.03, kink.gap / 2), _APPEARANCE: 0.1}.get(cue, kink.gap)
    sure = {_KINK: 0.75, _JUNCTION: 0.6, _APPEARANCE: 0.45}.get(cue, 0.2)
    origin = {_KINK: Provenance.MEASURED, _JUNCTION: Provenance.INTERPOLATED}.get(
        cue, Provenance.INFERRED
    )
    cues: tuple[str, ...] = (cue,)
    strength = None
    position = None if person is None else person.foot_court
    if scene.audio is not None and cue != _RETURNED:
        delay = scene.sound_delay(position, side)
        window = config.audio_window_seconds
        heard = scene.audio.nearest(pts + delay, window)
        if heard is not None:
            strength, sure, cues = heard[1], min(0.95, sure + 0.15), cues + ("audio",)
            if doubt > config.audio_retime_seconds:
                pts, doubt = heard[0] - delay, config.audio_retime_seconds
    if not reason and cue == _APPEARANCE:
        reason = "the ball appeared leaving this player; the contact itself was not tracked"
    return HitEvent(
        pts=float(pts),
        time_uncertainty=float(doubt),
        side=side,
        player=scene.players.player_at(pts, side),
        zone=None if person is None else person.zone,
        position=position,
        position_provenance=None if person is None else person.provenance,
        foot_clipped=False if person is None else person.foot_clipped,
        image_xy=(float(kink.xy[0]), float(kink.xy[1])),
        confidence=float(sure),
        provenance=origin,
        cues=cues,
        audio_strength=strength,
        reason=reason,
    )


def _unseen_hit(
    last: HitEvent, hit: HitEvent, bounces: list[BounceEvent], scene: _Scene
) -> HitEvent:
    """Adds the other player's hit that must lie between two by one player.

    The ball went away and came back, so it was hit at the other end. It
    happened after the ball could have got there - and after it bounced
    there, if that was seen - and before it could have come back - and
    before it bounced on the way back, if that was seen. It is placed in
    the middle of that interval, with half the interval as its
    uncertainty.
    """
    config = scene.config
    start, end = last.pts + config.min_flight_seconds, hit.pts - config.min_flight_seconds
    between = [bounce for bounce in bounces if last.pts < bounce.pts < hit.pts]
    there = [bounce.pts for bounce in between if bounce.half != hit.side]
    back = [bounce.pts for bounce in between if bounce.half == hit.side and bounce.pts > start]
    if there:
        start = max(start, there[0])
    if back and back[-1] - config.min_flight_seconds > start:
        end = back[-1] - config.min_flight_seconds
    longest, begins = end - start, start
    pts = begins + longest / 2
    side = _other(hit.side)
    blank = _Kink(
        pts=pts,
        xy=np.array([np.nan, np.nan]),
        before=None,
        after=np.zeros(2),
        kind=_RETURNED,
        gap=longest / 2,
    )
    reading = _Reading(blank, None, None, {side: scene.player(pts, side)}, {}, {}, False, False)
    made = _hit_from(
        reading,
        side,
        scene,
        cue=_RETURNED,
        reason=(
            f"the {hit.side} player hit at {last.pts:.2f}s and again at {hit.pts:.2f}s, so the "
            "ball was returned in between; that contact was not seen"
        ),
    )
    return replace(made, image_xy=None)


def _bounce_from(reading: _Reading, scene: _Scene) -> BounceEvent:
    """Builds the bounce a kink stands for, with its uncertainty in metres."""
    config, kink = scene.config, reading.kink
    pixels = config.pixel_sigma * scene.width + kink.blur
    spot = (float(kink.xy[0]), float(kink.xy[1]))
    across, along = ground_uncertainty(scene.calibration, spot, pixels)
    floor = config.bounce_floor_m
    seen = kink.kind == _KINK
    in_reach = min(reading.reach.values()) <= config.reach_m
    confidence = (0.8 if seen else 0.5) * (0.7 if in_reach else 1.0)
    return BounceEvent(
        pts=float(kink.pts),
        time_uncertainty=0.02 if seen else max(0.03, kink.gap / 2),
        court_xy=reading.ground,
        uncertainty_m=(float(np.hypot(across, floor)), float(np.hypot(along, floor))),
        half=reading.half,
        image_xy=spot,
        confidence=confidence,
        provenance=Provenance.MEASURED if seen else Provenance.INTERPOLATED,
        reason=(
            "the ball's path turned upward at the ground"
            if seen
            else f"the paths either side of a {kink.gap:.2f}s unseen stretch meet at the ground"
        ),
    )


def _net_from(reading: _Reading, last: HitEvent | None, scene: _Scene) -> NetEvent | None:
    """Builds a net contact from a kink, if it is one.

    The ball must have been travelling toward the net from the last
    hitter's end, be on the net in the picture, be clear of both players'
    boxes (where the track is unreliable) and have lost nearly all its
    speed there.
    """
    config, kink = scene.config, reading.kink
    if last is None or kink.kind != _KINK or not scene.at_net(kink.xy):
        return None
    if reading.boxed:
        return None
    arriving, leaving = float(np.hypot(*kink.before)), float(np.hypot(*kink.after))
    toward_net = kink.before[1] < 0 if last.side == NEAR else kink.before[1] > 0
    fast = arriving / scene.row_scale(kink.xy[1]) >= config.min_net_approach_mps
    if not (toward_net and fast and leaving <= config.net_stop_ratio * arriving):
        return None
    return NetEvent(
        pts=float(kink.pts),
        image_xy=(float(kink.xy[0]), float(kink.xy[1])),
        confidence=0.6,
        provenance=Provenance.MEASURED,
        reason=f"the ball lost {1 - leaving / arriving:.0%} of its image speed at the net",
    )


def _decode(
    readings: list[_Reading], scene: _Scene
) -> tuple[list[HitEvent], list[BounceEvent], list[NetEvent], int]:
    """Walks the kinks in time order, naming each a hit, bounce, net or nothing."""
    config = scene.config
    hits: list[HitEvent] = []
    bounces: list[BounceEvent] = []
    nets: list[NetEvent] = []
    discarded = 0
    last: HitEvent | None = None

    def settle(reading: _Reading) -> int:
        """Files a kink that is not a hit; returns 1 when it is nothing."""
        if reading.bounce_like:
            bounces.append(_bounce_from(reading, scene))
            return 0
        net = _net_from(reading, last, scene)
        if net is not None:
            nets.append(net)
            return 0
        return 1

    index = 0
    while index < len(readings):
        reading = readings[index]
        side = _eligible(reading, last, config)
        pick = None
        if side is not None:
            end = index + 1
            while (
                end < len(readings)
                and readings[end].reach[side] <= config.reach_m
                and readings[end].kink.pts - readings[end - 1].kink.pts <= config.visit_gap_seconds
            ):
                end += 1
            pick = _choose(readings[index:end], readings[end:], side, scene)
        if pick is None:
            discarded += settle(reading)
            index += 1
            continue
        for earlier in readings[index : index + pick]:
            discarded += settle(earlier)
        hit = _hit_from(readings[index + pick], side, scene)
        if last is not None and hit.side == last.side:
            if hit.pts - last.pts <= config.max_exchange_seconds:
                hits.append(_unseen_hit(last, hit, bounces, scene))
        hits.append(hit)
        last = hit
        index += pick + 1
    return hits, bounces, nets, discarded


def _assemble(
    hits: list[HitEvent], bounces: list[BounceEvent], nets: list[NetEvent], config: EventConfig
) -> tuple[list[Shot], list[BounceEvent]]:
    """Joins each hit to what became of the ball, and drops impossible bounces.

    A shot runs from its hit to the next hit. Its bounce is the first on
    the other side of the net; a "bounce" on the hitter's own side is
    kept only when nothing says the ball went over - otherwise it was a
    wobble in the track, and is dropped.
    """
    shots: list[Shot] = []
    dropped: list[BounceEvent] = []
    for index, hit in enumerate(hits):
        following = hits[index + 1] if index + 1 < len(hits) else None
        until = float("inf") if following is None else following.pts
        returned = (
            following is not None
            and following.side != hit.side
            and following.pts - hit.pts <= config.max_return_seconds
        )
        landed = [bounce for bounce in bounces if hit.pts < bounce.pts <= until]
        over = [bounce for bounce in landed if bounce.half != hit.side]
        short = [bounce for bounce in landed if bounce.half == hit.side]
        bounce = over[0] if over else None
        if returned or over:
            dropped.extend(short)
        elif short:
            bounce = short[0]
        stopped = [net for net in nets if hit.pts < net.pts <= until]
        blank = Shot(
            index=index,
            hit=hit,
            bounce=bounce,
            net=stopped[0] if stopped and not over else None,
            returned=returned,
            result=Tagged(None, Provenance.INFERRED, None, SOURCE),
            result_reason="",
        )
        shots.append(_judge(blank, False, config))
    kept = [bounce for bounce in bounces if not any(bounce is gone for gone in dropped)]
    return shots, kept


def join_shots(
    hits: Sequence[HitEvent],
    bounces: Sequence[BounceEvent],
    nets: Sequence[NetEvent],
    config: EventConfig | None = None,
) -> tuple[list[Shot], list[BounceEvent]]:
    """Joins hits to what became of the ball - for events gathered in pieces.

    `detect_events` does this for one stretch. When a recording is
    tracked in chunks, a shot hit at the end of one chunk may bounce in
    the next, so the shots are joined again once every chunk's hits,
    bounces and net contacts are together.

    Args:
        hits: Every hit, in time order.
        bounces: Every bounce, in time order.
        nets: Every net contact, in time order.
        config: Thresholds; see `EventConfig`.

    Returns:
        One shot per hit, judged as a rally shot (`EventTrack.rejudged`
        judges serves), and the bounces that remain - a bounce on the
        hitter's own side is dropped when the ball plainly went over.
    """
    return _assemble(list(hits), list(bounces), list(nets), config or EventConfig())


def detect_events(
    ball: BallTrack,
    players: PlayerTracks | None,
    calibration: Calibration,
    audio: AudioOnsets | None = None,
    config: EventConfig | None = None,
) -> EventTrack:
    """Finds the hits, bounces, net contacts and shots in a stretch of video.

    Every shot is judged here as a rally shot, against the whole court;
    call `EventTrack.rejudged` once the serves are known.

    Args:
        ball: The ball's track (`ball.track_ball`).
        players: The players' tracks. Without them no kink can be
            attributed to anyone, so no hits - and no shots - are
            reported; bounces still are.
        calibration: The court calibration the tracks were made with.
        audio: Sharp sounds in the same stretch (`audio_onsets`), used
            only to fix the time of hits already seen.
        config: Thresholds; see `EventConfig`.

    Returns:
        The events, the shots and a copy of the ball track with
        `court_xy` filled in at each measured bounce.
    """
    config = config or EventConfig()
    scene = _Scene(ball, players, calibration, audio, config)
    readings = [_read(kink, scene) for kink in find_kinks(ball, calibration, config)]
    hits, bounces, nets, discarded = _decode(readings, scene)
    shots, kept = _assemble(hits, bounces, nets, config)
    discarded += len(bounces) - len(kept)
    moments = [observation.pts for observation in ball.observations]
    landed = {
        bisect.bisect_left(moments, bounce.pts - 1e-6): bounce
        for bounce in kept
        if bounce.provenance is Provenance.MEASURED
    }
    observations = tuple(
        replace(observation, court_xy=landed[index].court_xy) if index in landed else observation
        for index, observation in enumerate(ball.observations)
    )
    logger.info(
        "Events: %d hits, %d bounces, %d net contacts, %d kinks discarded",
        len(hits),
        len(kept),
        len(nets),
        discarded,
    )
    return EventTrack(
        hits=tuple(hits),
        bounces=tuple(kept),
        nets=tuple(nets),
        shots=tuple(shots),
        ball=replace(ball, observations=observations, _times=[]),
        config=config,
        discarded_kinks=discarded,
    )
