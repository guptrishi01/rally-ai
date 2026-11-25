"""A scripted rally for the events and strokes tests: physics in, tracks out.

Nothing here is rendered. A ball is flown through court space under
gravity from hit to bounce to the next hit, projected through the test
camera into the picture, and wrapped in the same `BallTrack` and
`PlayerTracks` the real stages produce - so the event rules are tested on
trajectories whose hits and bounces are known exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tracking import ball, players
from tracking.ball import BallObservation, BallSegment, BallTrack, Visibility
from tracking.players import PlayerObservation, PlayerTracks
from tracking.provenance import Provenance

from .conftest import SIZE

FPS = 60.0
GRAVITY = 9.81
BALL_RADIUS_M = 0.0335
FOCAL = 0.68


@dataclass(frozen=True)
class Stroke:
    """One scripted shot.

    Attributes:
        side: Who hits it, "near" or "far".
        target: Where it lands, (x, y) court metres.
        flight: Seconds from contact to that bounce.
        carry: Seconds the ball travels on after the bounce before the
            next shot is hit (or the script ends).
        stops_at: If set, the ball stops dead this many seconds after
            contact instead of flying on (a ball into the net).
        feet: Where the hitter's feet are relative to the ball at
            contact, (dx, dy) metres; a default to one side when None.
    """

    side: str
    target: tuple[float, float]
    flight: float
    carry: float = 0.5
    stops_at: float | None = None
    feet: tuple[float, float] | None = None


@dataclass(frozen=True)
class Script:
    """A rally flown through court space.

    Attributes:
        times: Each frame's time, seconds.
        points: (N, 3) ball positions in court metres.
        hits: When each shot was hit.
        bounces: When each shot bounced (None for a ball that stopped).
        contacts: The ball's (x, y, z) at each hit.
        feet: The hitter's feet at each hit, (x, y).
        strokes: The shots, as scripted.
    """

    times: np.ndarray
    points: np.ndarray
    hits: tuple[float, ...]
    bounces: tuple[float | None, ...]
    contacts: tuple[tuple[float, float, float], ...]
    feet: tuple[tuple[float, float], ...]
    strokes: tuple[Stroke, ...]


def fly(strokes, start=(1.6, 0.3, 1.0), begins=1.0, fps=FPS) -> Script:
    """Flies a ball through a sequence of shots.

    Each shot leaves the ball's current position with exactly the
    velocity that lands it on its target after `flight` seconds; the ball
    then bounces (keeping 75% of its vertical and 70% of its horizontal
    speed) and carries on for `carry` seconds, where the next shot is hit.

    Args:
        strokes: The shots, in order.
        start: Where the first contact is, (x, y, z) court metres.
        begins: The time of the first contact, seconds.
        fps: Frames per second.

    Returns:
        The frame-by-frame path with the true hit and bounce times.
    """
    pieces = []  # (start time, end time, origin, velocity)
    hits, bounces, contacts, feet = [], [], [], []
    spot = np.array(start, dtype=float)
    clock = begins
    for stroke in strokes:
        hits.append(clock)
        contacts.append(tuple(spot))
        aside = stroke.feet
        if aside is None:
            aside = (-0.6, -0.2) if stroke.side == "near" else (0.6, 0.2)
        feet.append((spot[0] + aside[0], spot[1] + aside[1]))
        away = np.array(
            [
                (stroke.target[0] - spot[0]) / stroke.flight,
                (stroke.target[1] - spot[1]) / stroke.flight,
                (0.5 * GRAVITY * stroke.flight**2 - spot[2]) / stroke.flight,
            ]
        )
        if stroke.stops_at is not None:
            stop = stroke.stops_at
            pieces.append((clock, clock + stop, spot.copy(), away))
            spot = spot + away * stop - np.array([0, 0, 0.5 * GRAVITY * stop**2])
            pieces.append((clock + stop, clock + stop + stroke.carry, spot.copy(), np.zeros(3)))
            spot = spot - np.array([0, 0, 0.5 * GRAVITY * stroke.carry**2])
            bounces.append(None)
            clock += stop + stroke.carry
            continue
        pieces.append((clock, clock + stroke.flight, spot.copy(), away))
        landed = np.array([stroke.target[0], stroke.target[1], 0.0])
        falling = away[2] - GRAVITY * stroke.flight
        after = np.array([0.7 * away[0], 0.7 * away[1], -0.75 * falling])
        clock += stroke.flight
        bounces.append(clock)
        pieces.append((clock, clock + stroke.carry, landed, after))
        spot = landed + after * stroke.carry - np.array([0, 0, 0.5 * GRAVITY * stroke.carry**2])
        spot[2] = max(spot[2], 0.2)
        clock += stroke.carry
    times = np.arange(round(begins * fps), round(clock * fps)) / fps
    points = np.zeros((len(times), 3))
    for index, moment in enumerate(times):
        piece = next(p for p in pieces if p[0] - 1e-9 <= moment < p[1] + 1e-9 or p is pieces[-1])
        since = moment - piece[0]
        points[index] = piece[2] + piece[3] * since - np.array([0, 0, 0.5 * GRAVITY * since**2])
    points[:, 2] = np.maximum(points[:, 2], 0.0)
    return Script(
        times, points, tuple(hits), tuple(bounces), tuple(contacts), tuple(feet), tuple(strokes)
    )


def to_image(camera, points) -> np.ndarray:
    """Projects (N, 3) court points through a 3x4 camera."""
    lifted = np.hstack([np.asarray(points, dtype=float), np.ones((len(points), 1))])
    mapped = lifted @ camera.T
    return mapped[:, :2] / mapped[:, 2:]


def ball_extent(camera, points, size=SIZE) -> np.ndarray:
    """How wide the ball's blob looks at each of (N, 3) court points, pixels.

    One and a half times the ball's true image diameter: motion blur and
    differencing always smear it a little.
    """
    lifted = np.hstack([np.asarray(points, dtype=float), np.ones((len(points), 1))])
    depth = (lifted @ camera.T)[:, 2] / np.linalg.norm(camera[2, :3])
    return 1.5 * 2 * BALL_RADIUS_M * FOCAL * size[0] / depth


def ball_track(
    camera,
    calibration,
    script: Script,
    hidden=(),
    breaks=(),
    blips=(),
    size=SIZE,
) -> BallTrack:
    """Wraps a scripted path in a `BallTrack`, as the linker would produce it.

    Args:
        camera: The 3x4 test camera.
        calibration: Its calibration.
        script: The path.
        hidden: (start, end) stretches of time with no observation.
        breaks: Moments at which a new segment starts although the ball
            stayed visible (the linker breaks segments at hits).
        blips: (start, end, (x, y) pixels) - extra segments of something
            that never leaves a player's box: a racket taken for the ball.
        size: (width, height) of the picture.

    Returns:
        The track. Every observation is measured; a hidden stretch simply
        has none, and starts a new segment.
    """
    pixels = to_image(camera, script.points)
    extents = ball_extent(camera, script.points, size)
    speeds = np.gradient(pixels, script.times, axis=0)
    observations: list[BallObservation] = []
    segment = 0
    cuts = sorted(breaks)
    was_hidden = False
    for index, moment in enumerate(script.times):
        if any(start <= moment <= end for start, end in hidden):
            was_hidden = True
            continue
        crossed = cuts and moment >= cuts[0] - 1e-9
        if observations and (was_hidden or crossed):
            segment += 1
        while cuts and moment >= cuts[0] - 1e-9:
            cuts.pop(0)
        was_hidden = False
        spot = (float(pixels[index, 0]), float(pixels[index, 1]))
        observations.append(
            BallObservation(
                frame=index,
                pts=float(moment),
                image_xy=spot,
                ground_ray_xy=ball._on_ground(calibration, pixels[index]),
                velocity=(float(speeds[index, 0]), float(speeds[index, 1])),
                confidence=0.6,
                visibility=Visibility.VISIBLE,
                provenance=Provenance.MEASURED,
                segment=segment,
                extent=float(extents[index]),
            )
        )
    boxed = {}
    for start, end, where in blips:
        segment += 1
        boxed[segment] = 1.0
        for step, moment in enumerate(np.arange(start, end, 1 / FPS)):
            observations.append(
                BallObservation(
                    frame=10_000 + step,
                    pts=float(moment),
                    image_xy=(where[0] + 3.0 * (step % 3), where[1] - 2.0 * (step % 2)),
                    ground_ray_xy=None,
                    velocity=(0.0, 0.0),
                    confidence=0.3,
                    visibility=Visibility.VISIBLE,
                    provenance=Provenance.MEASURED,
                    segment=segment,
                    extent=12.0,
                    in_player_box=True,
                )
            )
    observations.sort(key=lambda observation: observation.pts)
    segments = []
    for number in range(segment + 1):
        own = [o for o in observations if o.segment == number]
        segments.append(
            BallSegment(
                index=number,
                start_frame=own[0].frame,
                end_frame=own[-1].frame,
                start_pts=own[0].pts,
                end_pts=own[-1].pts,
                measured=len(own),
                bridged=0,
                score=float(len(own)),
                roughness=0.0,
                in_player_box=boxed.get(number, 0.0),
            )
        )
    return BallTrack(
        observations=tuple(observations),
        segments=tuple(segments),
        gaps=(),
        frame_numbers=np.arange(len(script.times)),
        frame_pts=script.times,
        image_size=size,
        source="scripted",
    )


def person_box(calibration, spot, tall=1.8, wide=0.6):
    """The image box of a person standing at a court position."""
    foot = calibration.to_image(np.array(spot, dtype=float))
    scale = players._ground_scale(calibration, (float(foot[0]), float(foot[1])))
    return (
        float(foot[0] - wide * scale / 2),
        float(foot[1] - tall * scale),
        float(foot[0] + wide * scale / 2),
        float(foot[1]),
    )


def player_tracks(calibration, script: Script, step=0.1, missing=()) -> PlayerTracks:
    """Builds both players' tracks for a scripted rally.

    Each player stands where they hit each of their shots and walks in a
    straight line between; before their first shot they stand where they
    will hit it.

    Args:
        calibration: Places each player in the picture.
        script: The rally.
        step: Seconds between detections.
        missing: Ends ("near" / "far") to leave untracked.

    Returns:
        The tracks, every observation measured.
    """
    moments = np.arange(script.times[0] - 0.5, script.times[-1] + 0.5, step)
    seen = {"near": [], "far": []}
    homes = {"near": (0.0, -0.5), "far": (0.0, 24.3)}
    for side in ("near", "far"):
        if side in missing:
            continue
        own = [
            (hit, feet)
            for hit, feet, stroke in zip(script.hits, script.feet, script.strokes)
            if stroke.side == side
        ] or [(0.0, homes[side])]
        when = np.array([hit for hit, _ in own])
        across = np.interp(moments, when, [feet[0] for _, feet in own])
        along = np.interp(moments, when, [feet[1] for _, feet in own])
        for number, moment in enumerate(moments):
            spot = (float(across[number]), float(along[number]))
            box = person_box(calibration, spot)
            seen[side].append(
                PlayerObservation(
                    frame=number,
                    pts=float(moment),
                    side=side,
                    player="A" if side == "near" else "B",
                    box=box,
                    foot_image=((box[0] + box[2]) / 2, box[3]),
                    foot_court=spot,
                    zone=players.zone_of(side, spot[0]),
                    confidence=0.9,
                    provenance=Provenance.MEASURED,
                    track_id=1 if side == "near" else 2,
                )
            )
    return PlayerTracks(
        near=tuple(seen["near"]),
        far=tuple(seen["far"]),
        swaps=(),
        detector_times=tuple(float(moment) for moment in moments),
        source="scripted",
        max_interpolation_seconds=0.6,
    )
