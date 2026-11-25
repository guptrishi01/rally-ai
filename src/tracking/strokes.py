"""Strokes: naming each hit a serve, forehand, backhand, volley or overhead.

There is **no labelled training data** for this camera, so nothing here is
learned from examples. `RuleStrokeClassifier` names a stroke from geometry
the earlier stages already measured, and says `unknown` when that geometry
does not decide it. The `StrokeClassifier` protocol is the seam a trained
classifier would later plug into.

The rules
---------
* **Serve** - hit from behind the baseline when the other player has not
  hit for `serve_quiet_seconds`. (The server's own earlier "hits" - the
  ball bounced or tossed before serving - do not break the quiet.) A serve
  whose overhead contact was seen is `measured`; one named from position
  and timing alone is `inferred`, at lower confidence.
* **Overhead** - struck overhead in the middle of a rally.
* **Volley** - hit from the service-line area or closer to the net with no
  bounce seen on the hitter's side since the other player's hit. A bounce
  that was *not seen* is not the same as no bounce - bounces go unseen
  about half the time - so a volley is never confident.
* **Forehand / backhand** - which side of the body the contact is on, from
  the hitter's own point of view, combined with the hand they play with.

"Struck overhead" and "which side of the body" come from up to two
measurements:

* the **ball**: where the ball was in the picture at contact relative to
  the hitter's box (only when the contact itself was tracked);
* the **pose**: where the hitting wrist was relative to the head and the
  shoulders, from a pose model run on a few frames around the hit - the
  side from the frames *up to* contact, because the racket hand crosses
  the body in the follow-through.

Where the ball speaks it decides, and the pose raises or lowers the
confidence; the pose alone is a weaker fallback. That order was measured,
not assumed (see `RuleStrokeClassifier._contact`).

Handedness is an input, not a finding
-------------------------------------
Forehand and backhand cannot be told apart without knowing which hand the
player holds the racket in. `StrokeConfig.handedness` supplies it per
person and **defaults to right-handed for both**. A wrong value swaps
every forehand and backhand of that player; it must become an intake
question.

Mirror convention: the near player has their back to the camera, so their
right is the picture's right; the far player faces the camera, so their
right is the picture's left.

Pose is used where it can be
----------------------------
`PoseEstimator` is a protocol; `YoloPoseEstimator` wraps the pinned
Ultralytics pose model, loaded lazily (this module imports without torch).
It runs only on the hitter's crop, on a handful of frames around each hit
(`estimate_poses`). A pose whose wrist or shoulders are unsure is ignored,
and a hitter below `StrokeConfig.min_player_height` is not posed at all;
strokes are then judged from the ball alone, or left `unknown`. The far
player on the real recording is about 85 pixels tall: expected to be
useless, the pose there turned out to name the side correctly on the
shots tried, so the floor sits below that height.

Spin is not attempted
---------------------
Topspin, slice and flat differ in the ball's rotation and in a few degrees
of racket path - neither visible at this resolution - and no defensible
cue was found in the trajectory either. `StrokeLabel.spin` is always None.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np

from . import models
from .config import ModelConfig, StrokeConfig
from .court import COURT_LENGTH_M, NET_Y_M, SERVICE_LINE_FROM_NET_M
from .events import (
    CROSS_COURT,
    DOWN_THE_LINE,
    EventTrack,
    HitEvent,
)
from .players import AD, DEUCE, FAR, NEAR, Box, PlayerTracks, _as_array
from .provenance import Provenance, Tagged
from .video import VideoFrame

logger = logging.getLogger(__name__)

SERVE = "Serve"
FOREHAND = "Forehand"
BACKHAND = "Backhand"
FOREHAND_VOLLEY = "Forehand Volley"
BACKHAND_VOLLEY = "Backhand Volley"
OVERHEAD = "Overhead"
UNKNOWN = "unknown"
STROKES = (SERVE, FOREHAND, BACKHAND, FOREHAND_VOLLEY, BACKHAND_VOLLEY, OVERHEAD)

INSIDE_OUT = "inside out"
INSIDE_IN = "inside in"

SOURCE = "stroke-rules-v1"

# COCO keypoint order, as the pose model reports it.
_NOSE, _LEFT_SHOULDER, _RIGHT_SHOULDER = 0, 5, 6
_LEFT_WRIST, _RIGHT_WRIST, _LEFT_HIP, _RIGHT_HIP = 9, 10, 11, 12
_KEYPOINTS = 17


@dataclass(frozen=True)
class Pose:
    """One person's body keypoints in a frame.

    Attributes:
        keypoints: (17, 2) image positions in pixels of the whole frame,
            in COCO order (nose, eyes, ears, shoulders, elbows, wrists,
            hips, knees, ankles; left before right).
        confidence: (17,) the model's confidence in each, 0 to 1.
    """

    keypoints: np.ndarray
    confidence: np.ndarray

    def point(self, index: int, floor: float) -> np.ndarray | None:
        """Gives one keypoint if the model was sure enough of it.

        Args:
            index: Which keypoint, COCO order.
            floor: Least confidence accepted.

        Returns:
            Its (x, y), or None when its confidence is below `floor`.
        """
        if self.confidence[index] < floor:
            return None
        return self.keypoints[index]


@runtime_checkable
class PoseEstimator(Protocol):
    """Anything that can find one person's keypoints inside a box.

    Attributes:
        name: The estimator's name and version, kept as provenance.
    """

    name: str

    def estimate(self, image: np.ndarray, box: Box) -> Pose | None:
        """Estimates the pose of the person in a box.

        Args:
            image: (H, W, 3) uint8 RGB frame.
            box: The person's (x0, y0, x1, y1) in that frame.

        Returns:
            Their keypoints in whole-frame pixels, or None when nobody
            was found.
        """
        ...


class YoloPoseEstimator:
    """The pinned Ultralytics pose model, run on a crop around one player.

    The model is loaded on first use, so building one costs nothing and
    needs no backend.

    Attributes:
        name: "name@backend-version" once loaded; the model's registry
            name before that.
    """

    def __init__(
        self,
        model: Any | None = None,
        config: StrokeConfig | None = None,
        models_config: ModelConfig | None = None,
        name: str | None = None,
    ) -> None:
        """Initializes the estimator.

        Args:
            model: An already-loaded model with ultralytics' `predict`
                interface - tests pass a fake. Loaded lazily from the
                registry when omitted.
            config: Supplies the crop padding and inference size.
            models_config: Which pinned model to load and from where.
            name: Provenance name for an injected model.
        """
        self._model = model
        self._config = config or StrokeConfig()
        self._models_config = models_config or ModelConfig()
        self.name = name or self._models_config.pose_model

    def _ensure(self) -> Any:
        """Returns the model, loading the pinned one on first use."""
        if self._model is None:
            loaded = models.load_pose_model(self._models_config)
            self._model, self.name = loaded.model, loaded.source
        return self._model

    def estimate(self, image: np.ndarray, box: Box) -> Pose | None:
        """Estimates the pose of the person in a box.

        Args:
            image: (H, W, 3) uint8 RGB frame.
            box: The person's (x0, y0, x1, y1) in that frame.

        Returns:
            The keypoints of the most confident person found in the
            padded crop, in whole-frame pixels; None when nobody was.

        Raises:
            tracking.models.ModelError: If the model cannot be loaded.
        """
        model = self._ensure()
        tall, wide = image.shape[:2]
        pad_x = self._config.crop_pad * (box[2] - box[0])
        pad_y = self._config.crop_pad * (box[3] - box[1])
        left, top = max(0, int(box[0] - pad_x)), max(0, int(box[1] - pad_y))
        right, bottom = min(wide, int(box[2] + pad_x) + 1), min(tall, int(box[3] + pad_y) + 1)
        if right - left < 2 or bottom - top < 2:
            return None
        crop = np.ascontiguousarray(image[top:bottom, left:right, ::-1])
        found = model.predict(crop, imgsz=self._config.pose_size, verbose=False)[0]
        if found.keypoints is None or found.keypoints.conf is None:
            return None
        points = _as_array(found.keypoints.xy).reshape(-1, _KEYPOINTS, 2)
        sure = _as_array(found.keypoints.conf).reshape(-1, _KEYPOINTS)
        scores = _as_array(found.boxes.conf).reshape(-1)
        if not len(scores):
            return None
        best = int(np.argmax(scores))
        return Pose(
            keypoints=points[best] + np.array([left, top], dtype=float), confidence=sure[best]
        )


@dataclass(frozen=True)
class PoseSample:
    """A pose at one moment.

    Attributes:
        pts: The frame's timestamp, seconds.
        pose: The hitter's keypoints in it.
    """

    pts: float
    pose: Pose


def estimate_poses(
    frames: Iterable[VideoFrame],
    events: EventTrack,
    players: PlayerTracks,
    estimator: PoseEstimator,
    config: StrokeConfig | None = None,
) -> dict[int, tuple[PoseSample, ...]]:
    """Runs the pose estimator on a few frames around each hit.

    Frames are consumed one at a time and never held. Only frames from
    `window_before` seconds before a hit to `window_after` after it are
    looked at, at most `pose_frames` per hit, and only the hitter's crop
    is given to the estimator. A hitter whose box is shorter than
    `min_player_height` of the frame is skipped.

    Args:
        frames: Frames in display order, covering the events' stretch.
        events: The events whose hits are to be posed.
        players: The players' tracks, for the hitter's box.
        estimator: Finds keypoints in a box.
        config: Window, frame count and size floor; see `StrokeConfig`.

    Returns:
        Shot index to the poses found around that hit, in time order.
        A hit with no usable pose has no entry.
    """
    config = config or StrokeConfig()
    before, after = config.window_before, config.window_after
    targets = {
        shot.index: list(
            np.linspace(shot.hit.pts - before, shot.hit.pts + after, config.pose_frames)
        )
        for shot in events.shots
    }
    order = sorted(events.shots, key=lambda shot: shot.hit.pts)
    found: dict[int, list[PoseSample]] = {}
    first_open = 0
    for frame in frames:
        while first_open < len(order) and order[first_open].hit.pts + after < frame.pts - 1e-6:
            first_open += 1
        for shot in order[first_open:]:
            if shot.hit.pts - before > frame.pts + 1e-6:
                break
            wanted = targets[shot.index]
            if not wanted or frame.pts < wanted[0] - 1e-6 or frame.pts > shot.hit.pts + after:
                continue
            while wanted and wanted[0] <= frame.pts + 1e-6:
                wanted.pop(0)
            person = players.at(frame.pts, shot.hit.side)
            if person is None:
                continue
            height = person.box[3] - person.box[1]
            if height < config.min_player_height * frame.image.shape[0]:
                continue
            pose = estimator.estimate(frame.image, person.box)
            if pose is not None:
                found.setdefault(shot.index, []).append(PoseSample(frame.pts, pose))
    logger.info("Pose: %d of %d hits have at least one pose", len(found), len(order))
    return {index: tuple(samples) for index, samples in found.items()}


@dataclass(frozen=True)
class HitContext:
    """Everything a stroke classifier is given about one hit.

    Attributes:
        index: The shot's index in `EventTrack.shots`.
        hit: The contact.
        rival_quiet_before: Seconds since the *other* player last hit;
            infinite when they have not. A player bouncing the ball
            before serving does not break their own quiet.
        watched_before: Seconds of tracked video before this hit - how
            long a quiet spell before it could have been observed.
        bounced_before: Whether a bounce was seen on the hitter's side
            since the previous hit.
        box: The hitter's box at contact, or None when not tracked.
        poses: Poses of the hitter around contact, if any were estimated.
        handedness: "right" or "left": the hand this person plays with.
    """

    index: int
    hit: HitEvent
    rival_quiet_before: float
    watched_before: float
    bounced_before: bool
    box: Box | None
    poses: tuple[PoseSample, ...]
    handedness: str


@dataclass(frozen=True)
class StrokeLabel:
    """The stroke named for one hit.

    Attributes:
        index: The shot's index in `EventTrack.shots`.
        stroke: One of `STROKES`, or "unknown" - tagged with how it was
            obtained and how sure it is.
        spin: Always None: spin is not attempted (see the module
            docstring).
        reason: The reasoning, in words.
        pose_used: Whether a pose contributed.
    """

    index: int
    stroke: Tagged[str]
    spin: Tagged[str] | None
    reason: str
    pose_used: bool = False

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field; the stroke as its tagged dict.
        """
        return {
            "index": self.index,
            "stroke": self.stroke.to_dict(),
            "spin": None if self.spin is None else self.spin.to_dict(),
            "reason": self.reason,
            "pose_used": self.pose_used,
        }


@runtime_checkable
class StrokeClassifier(Protocol):
    """Anything that can name a hit's stroke - rules today, a model later.

    Attributes:
        name: The classifier's name and version, kept as provenance.
    """

    name: str

    def classify(self, context: HitContext) -> StrokeLabel:
        """Names one hit's stroke.

        Args:
            context: The hit and what is known around it.

        Returns:
            The label; its stroke is "unknown" when the classifier cannot
            tell.
        """
        ...


@dataclass(frozen=True)
class _Contact:
    """What the ball and the pose say about how a ball was struck.

    Attributes:
        overhead: True / False, or None when not known.
        overhead_by: "pose", "ball" or "" - what `overhead` rests on.
        side: +1 when contact was to the picture's right of the body, -1
            to its left, None when neither measurement says.
        side_by: What `side` rests on, in words.
        confidence: How far `side` is trusted, 0 to 1.
        pose_used: Whether a pose gave either answer.
    """

    overhead: bool | None
    overhead_by: str
    side: int | None
    side_by: str
    confidence: float
    pose_used: bool


class RuleStrokeClassifier:
    """Names strokes from geometry, abstaining when geometry does not decide.

    Attributes:
        name: "stroke-rules-v1".
    """

    name = SOURCE

    def __init__(self, config: StrokeConfig | None = None) -> None:
        """Initializes the classifier.

        Args:
            config: Thresholds; see `StrokeConfig`.
        """
        self._config = config or StrokeConfig()

    def _from_ball(self, context: HitContext) -> tuple[bool | None, int | None]:
        """Reads overhead / side from where the ball was against the box.

        Only when the contact itself was tracked: for a hit inferred from
        the ball appearing, the ball's first position is already on its
        way and says nothing about the side it was struck on.
        """
        hit, box = context.hit, context.box
        if box is None or hit.image_xy is None or hit.provenance is Provenance.INFERRED:
            return None, None
        tall = box[3] - box[1]
        overhead = None
        if hit.image_xy[1] <= box[1] + 0.05 * tall:
            overhead = True
        elif hit.image_xy[1] >= box[1] + self._config.overhead_margin * tall:
            overhead = False
        offset = (hit.image_xy[0] - (box[0] + box[2]) / 2) / tall
        side = None if abs(offset) < self._config.side_margin else int(np.sign(offset))
        return overhead, side

    def _from_pose(self, context: HitContext) -> tuple[bool | None, int | None]:
        """Reads overhead / side from the hitting wrist against head and shoulders.

        Overhead if the wrist is above the head in *any* frame of the
        window (a serve's contact is brief and its time uncertain). Side
        from the frames up to contact only: the racket hand is on the
        stroke's own side through the backswing and the swing, then
        crosses the body in the follow-through.
        """
        config = self._config
        floor = config.min_keypoint_confidence
        wrist_index = _RIGHT_WRIST if context.handedness == "right" else _LEFT_WRIST
        if context.box is None:
            return None, None
        tall = context.box[3] - context.box[1]
        overhead = None
        offsets = []
        for sample in context.poses:
            pose = sample.pose
            wrist = pose.point(wrist_index, floor)
            left, right = pose.point(_LEFT_SHOULDER, floor), pose.point(_RIGHT_SHOULDER, floor)
            if wrist is None or left is None or right is None:
                continue
            shoulders = (left + right) / 2
            hips = [pose.point(index, floor) for index in (_LEFT_HIP, _RIGHT_HIP)]
            torso = 0.3 * tall
            if hips[0] is not None and hips[1] is not None:
                torso = float(np.hypot(*((hips[0] + hips[1]) / 2 - shoulders)))
            nose = pose.point(_NOSE, floor)
            head = nose[1] if nose is not None else shoulders[1] - 0.5 * torso
            overhead = bool(overhead or wrist[1] < head - config.overhead_margin * torso)
            if sample.pts <= context.hit.pts + 0.02:
                offsets.append((wrist[0] - shoulders[0]) / tall)
        side = None
        if offsets and abs(float(np.median(offsets))) >= config.side_margin:
            side = int(np.sign(np.median(offsets)))
        return overhead, side

    def _contact(self, context: HitContext) -> _Contact:
        """Combines the ball's and the pose's readings of one contact.

        Where the ball says which side, the ball decides: measured on the
        real recording it agreed with the reference on every shot it
        spoke for, while pose alone was wrong on three near-end backhands
        in eight. Pose is the fallback, and a second opinion.
        """
        ball_over, ball_side = self._from_ball(context)
        pose_over, pose_side = self._from_pose(context)
        overhead, by = (ball_over, "ball") if ball_over is not None else (pose_over, "pose")
        if ball_side is not None and pose_side is None:
            side, side_by, sure = ball_side, "ball", 0.65
        elif ball_side is not None and pose_side == ball_side:
            side, side_by, sure = ball_side, "ball and pose", 0.8
        elif ball_side is not None:
            side, side_by, sure = ball_side, "ball; the pose disagreed", 0.5
        elif pose_side is not None:
            side, side_by, sure = pose_side, "pose alone", 0.55
        else:
            side, side_by, sure = None, "", 0.0
        return _Contact(
            overhead=overhead,
            overhead_by=by if overhead is not None else "",
            side=side,
            side_by=side_by,
            confidence=sure,
            pose_used=pose_over is not None or pose_side is not None,
        )

    def classify(self, context: HitContext) -> StrokeLabel:
        """Names one hit's stroke.

        Args:
            context: The hit and what is known around it.

        Returns:
            The label. Its stroke is "unknown" - with the reason - when
            the hitter was not tracked, the contact's side could not be
            told, or the result is below `StrokeConfig.min_confidence`.
        """
        config, hit = self._config, context.hit
        contact = self._contact(context)

        def label(stroke: str, confidence: float, origin: Provenance, reason: str) -> StrokeLabel:
            """Builds the label, falling back to unknown when unsure."""
            if stroke != UNKNOWN and confidence < config.min_confidence:
                reason = f"{reason}; too unsure ({confidence:.2f}) to name it {stroke}"
                stroke = UNKNOWN
            sure = None if stroke == UNKNOWN else float(confidence)
            return StrokeLabel(
                index=context.index,
                stroke=Tagged(stroke, origin, sure, self.name),
                spin=None,
                reason=reason,
                pose_used=contact.pose_used,
            )

        if hit.position is None:
            return label(UNKNOWN, 0.0, Provenance.INFERRED, "the hitter was not tracked at contact")
        y = hit.position[1]
        if hit.side == NEAR:
            behind = y <= config.serve_behind_m
        else:
            behind = y >= COURT_LENGTH_M - config.serve_behind_m
        if behind and context.rival_quiet_before >= config.serve_quiet_seconds:
            never = context.rival_quiet_before == float("inf")
            waited = (
                "the other player had not hit before it"
                if never
                else f"{context.rival_quiet_before:.1f}s after the other player last hit"
            )
            if contact.overhead:
                confidence = 0.85 if contact.overhead_by == "ball" else 0.75
                origin = Provenance.MEASURED
                how = f"struck overhead (by the {contact.overhead_by})"
            elif contact.overhead is None:
                confidence, origin = 0.5, Provenance.INFERRED
                how = "the height of contact was not seen"
            else:
                confidence, origin = 0.45, Provenance.INFERRED
                how = f"the {contact.overhead_by} did not show an overhead contact"
            if never and context.watched_before < config.serve_quiet_seconds:
                confidence *= 0.9
                how += "; too little was watched before it to be sure of the quiet"
            return label(SERVE, confidence, origin, f"from behind the baseline, {waited}; {how}")
        if contact.overhead:
            return label(
                OVERHEAD,
                0.7 if contact.overhead_by == "ball" else 0.55,
                Provenance.MEASURED,
                f"struck overhead (by the {contact.overhead_by}) in the middle of a rally",
            )
        at_net = abs(y - NET_Y_M) <= SERVICE_LINE_FROM_NET_M + config.volley_line_m
        volley = at_net and not context.bounced_before
        where = "at the net with no bounce seen on their side" if volley else "a groundstroke"
        if contact.side is None:
            reason = f"{where}; the side of contact was not seen"
            return label(UNKNOWN, 0.0, Provenance.INFERRED, reason)
        own_right = (contact.side > 0) == (hit.side == NEAR)
        forehand = own_right == (context.handedness == "right")
        confidence = contact.confidence
        if volley:
            confidence *= 0.85
            stroke = FOREHAND_VOLLEY if forehand else BACKHAND_VOLLEY
        else:
            stroke = FOREHAND if forehand else BACKHAND
        return label(
            stroke,
            confidence,
            Provenance.MEASURED,
            f"{where}; contact on the player's {'right' if own_right else 'left'} "
            f"(by the {contact.side_by}), taken as {context.handedness}-handed",
        )


def hit_contexts(
    events: EventTrack,
    players: PlayerTracks,
    poses: Mapping[int, Sequence[PoseSample]] | None = None,
    config: StrokeConfig | None = None,
) -> tuple[HitContext, ...]:
    """Gathers, for every shot, what a stroke classifier needs.

    Args:
        events: The events.
        players: The players' tracks, for the hitter's box at contact.
        poses: Poses per shot index, from `estimate_poses`; optional.
        config: Supplies each person's handedness.

    Returns:
        One context per shot, in shot order.

    Raises:
        ValueError: If a configured handedness is neither "right" nor
            "left".
    """
    config = config or StrokeConfig()
    hands = dict(config.handedness)
    wrong = sorted(hand for hand in hands.values() if hand not in ("right", "left"))
    if wrong:
        raise ValueError(f"handedness must be 'right' or 'left', got {wrong[0]!r}.")
    poses = poses or {}
    start = float(events.ball.frame_pts[0]) if len(events.ball.frame_pts) else float("-inf")
    contexts = []
    last: dict[str, float] = {}
    previous = float("-inf")
    for shot in events.shots:
        hit = shot.hit
        rival = last.get(FAR if hit.side == NEAR else NEAR)
        person = players.at(hit.pts, hit.side)
        contexts.append(
            HitContext(
                index=shot.index,
                hit=hit,
                rival_quiet_before=float("inf") if rival is None else hit.pts - rival,
                watched_before=hit.pts - start,
                bounced_before=any(
                    previous < bounce.pts < hit.pts and bounce.half == hit.side
                    for bounce in events.bounces
                ),
                box=None if person is None else person.box,
                poses=tuple(poses.get(shot.index, ())),
                handedness=hands.get(hit.player, "right"),
            )
        )
        last[hit.side] = previous = hit.pts
    return tuple(contexts)


def classify_strokes(
    events: EventTrack,
    players: PlayerTracks,
    poses: Mapping[int, Sequence[PoseSample]] | None = None,
    classifier: StrokeClassifier | None = None,
    config: StrokeConfig | None = None,
) -> tuple[StrokeLabel, ...]:
    """Names the stroke of every shot.

    Args:
        events: The events.
        players: The players' tracks.
        poses: Poses per shot index, from `estimate_poses`; without them
            strokes are judged from the ball alone.
        classifier: Names one hit; `RuleStrokeClassifier` if omitted.
        config: Thresholds and handedness; see `StrokeConfig`.

    Returns:
        One label per shot, in shot order.
    """
    config = config or StrokeConfig()
    classifier = classifier or RuleStrokeClassifier(config)
    labels = tuple(
        classifier.classify(context) for context in hit_contexts(events, players, poses, config)
    )
    named = sum(label.stroke.value != UNKNOWN for label in labels)
    logger.info("Strokes: %d of %d named by %s", named, len(labels), classifier.name)
    return labels


def refine_direction(direction: str, stroke: str, zone: str | None, handedness: str) -> str:
    """Renames a rally shot's direction when it was hit "inside".

    A forehand hit from the player's backhand half of the court (or a
    backhand from their forehand half) is an inside shot: cross court
    becomes "inside out", down the line "inside in".

    Args:
        direction: "cross court" or "down the line" (anything else is
            returned unchanged).
        stroke: The stroke's name.
        zone: The half the hitter stood in, "deuce" or "ad".
        handedness: "right" or "left".

    Returns:
        The direction, renamed when the shot was an inside one.
    """
    forehand = stroke in (FOREHAND, FOREHAND_VOLLEY)
    backhand = stroke in (BACKHAND, BACKHAND_VOLLEY)
    if zone not in (DEUCE, AD) or not (forehand or backhand):
        return direction
    forehand_half = DEUCE if handedness == "right" else AD
    inside = forehand != (zone == forehand_half)
    if not inside:
        return direction
    return {CROSS_COURT: INSIDE_OUT, DOWN_THE_LINE: INSIDE_IN}.get(direction, direction)


def apply_strokes(
    events: EventTrack, labels: Sequence[StrokeLabel], config: StrokeConfig | None = None
) -> EventTrack:
    """Folds the strokes back into the events.

    Shots named a serve are re-judged against the service box, and rally
    shots' directions are renamed "inside out" / "inside in" where the
    stroke makes them so.

    Args:
        events: The events the labels were made for.
        labels: One label per shot, from `classify_strokes`.
        config: Supplies each person's handedness.

    Returns:
        The events with results and directions updated.
    """
    config = config or StrokeConfig()
    hands = dict(config.handedness)
    named = {label.index: label.stroke.value for label in labels}
    judged = events.rejudged(index for index, stroke in named.items() if stroke == SERVE)
    shots = []
    for shot in judged.shots:
        stroke = named.get(shot.index, UNKNOWN)
        if shot.direction is not None and not shot.serve:
            renamed = refine_direction(
                shot.direction.value, stroke, shot.hit.zone, hands.get(shot.hit.player, "right")
            )
            shot = replace(shot, direction=replace(shot.direction, value=renamed))
        shots.append(shot)
    return replace(judged, shots=tuple(shots))


def stroke_quality(events: EventTrack, labels: Sequence[StrokeLabel]) -> dict:
    """Counts what was named, what was left unknown, and where pose helped.

    Args:
        events: The events the labels were made for.
        labels: One label per shot.

    Returns:
        Counts per stroke (every stroke present, "unknown" included), and
        per end of the court: shots, unknowns and how many used a pose.
    """
    counts = {name: 0 for name in STROKES + (UNKNOWN,)}
    ends = {side: {"shots": 0, "unknown": 0, "pose_used": 0} for side in (NEAR, FAR)}
    sides = {shot.index: shot.hit.side for shot in events.shots}
    for label in labels:
        counts[label.stroke.value] += 1
        end = ends[sides[label.index]]
        end["shots"] += 1
        end["unknown"] += label.stroke.value == UNKNOWN
        end["pose_used"] += label.pose_used
    return {"strokes": counts, "by_side": ends}
