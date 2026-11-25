"""The two players: finding people, following them, and saying who is where.

Three layers, each usable and testable on its own:

1. **Detection** - anything satisfying the `PersonDetector` protocol turns a
   frame into person boxes. `YoloPersonDetector` wraps the pinned learned
   model (loaded lazily, so this module imports without torch);
   `MotionPersonDetector` is the classic fallback - a background model and
   person-sized foreground blobs - so the stage degrades rather than dies
   when no learned model is available. Tests inject fakes.
2. **Association** - `PersonAssociator` links boxes across frames the
   ByteTrack way: confident detections are matched to predicted track
   positions first, then low-confidence ones are allowed to *continue* a
   track but never start one. Greedy matching on padded-box overlap, in
   numpy; two to six people do not need the Hungarian algorithm.
3. **Selection and identity** - `PlayerTracker` keeps exactly two people:
   per end of the court, the track whose **feet** (lower-centre of the box,
   through the calibration) stand on or near *this* court. People on a
   neighbouring court, behind the fence or walking past have feet somewhere
   else and are never candidates.

Identity is the end of the court
--------------------------------
A player is identified by the end they are at: `side` is `"near"` (the
camera's end, court y below the net) or `"far"`. That is exactly what
`RawShotRow.hit_side` needs. The *person* is tracked separately as
`player`: `"A"` is whoever was at the near end when tracking began, `"B"`
the other. They are never silently exchanged: when the players change
ends, an `EndSwap` event is recorded and every observation after it carries
the swapped label. Nothing is concluded from a player crossing the net:
on the real recording players wander to the bench, round the net post and
back during a changeover, and reading each crossing as a swap recorded
three for one. A swap is judged only once a player has been at **both**
ends, unbroken, for `PlayerConfig.swap_settle_seconds`, by comparing that
arrangement with the last settled one. It is `measured` when an
association track that was at one end is now at the other (and the
players' appearance does not contradict it), and `inferred` when the
people now at each end look (by clothing-colour histogram) like each
other's earlier selves - the usual changeover, where the tracks are lost
or, as measured on the real recording, hop from one player to the other
at the bench. Appearance outranks tracks when it is decisive; when the
players are dressed alike it decides nothing and only a followed track can
show a swap. It is dated to when the new arrangement began; `player`
labels between `EndSwap.last_settled_pts` and `EndSwap.pts` mean little.

Deuce and ad
------------
`zone` is the half of the court the player stands in **from their own point
of view, facing the net**: `"deuce"` is their right, `"ad"` their left. In
the court frame (`court.py`: +x to the camera's right) that is

* near end: deuce when x >= 0, ad when x < 0;
* far end: deuce when x < 0, ad when x >= 0.

Checked against SwingVision's `Hit Zone` on the one real export available:
applying this rule to the sheet's own `Hit (x)` reproduces the half its
`Hit Zone` names on 673 of 683 shots. The ten exceptions are four shots
within centimetres of the centre line (which the sheet calls
`center_line`) and six serves hit within 0.6m of the centre mark, which
the sheet labels by the side served from. The sheet's vocabulary is wider
than this module's: it also says `deuce_alley` / `ad_alley` beyond the
singles sideline (|x| > 4.115m) and `center_line`. Only the half is
produced here; widening it is for the phase that emits `RawShotRow`.

What a position means
---------------------
`foot_court` is where the *feet* are on the court plane. It is as good as
the box's lower edge: when the box runs off the bottom of the frame (the
near player, from a low camera) the feet are not visible, `foot_clipped` is
set, and the true position is further from the net than reported. At the
far end one pixel is a quarter of a metre of depth, so a far footpoint is a
metre-scale estimate, not a line call.

The detector need not run on every frame. `PlayerConfig.stride` picks every
n-th; `PlayerTracks.at` interpolates between two detections of the same
end when they are close enough in time (tagged `interpolated`) and returns
nothing across a longer gap.
"""

from __future__ import annotations

import bisect
import logging
import pickle
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

from . import models
from .calibrate import Calibration
from .config import ModelConfig, PlayerConfig
from .court import COURT_LENGTH_M, DOUBLES_WIDTH_M, NET_Y_M, Point, camera_from_homography
from .provenance import Provenance
from .video import VideoFrame

logger = logging.getLogger(__name__)

Box = tuple[float, float, float, float]

NEAR = "near"
FAR = "far"
DEUCE = "deuce"
AD = "ad"
SIDES = (NEAR, FAR)

_PERSON_HEIGHT_M = 2.3
_COCO_PERSON = 0


@dataclass(frozen=True)
class PersonDetection:
    """One person found in one frame.

    Attributes:
        box: (x0, y0, x1, y1) in pixels, y down.
        confidence: The detector's confidence, 0 to 1.
    """

    box: Box
    confidence: float


@runtime_checkable
class PersonDetector(Protocol):
    """Anything that finds people in a frame.

    Attributes:
        name: What it is and its version, recorded as the provenance
            `source` of everything derived from it.
    """

    name: str

    def detect(self, image: np.ndarray) -> list[PersonDetection]:
        """Finds the people in one frame.

        Args:
            image: (H, W, 3) uint8 RGB frame.

        Returns:
            Their boxes, in that frame's pixels.
        """
        ...


def _as_array(values: Any) -> np.ndarray:
    """Turns a backend tensor (or anything array-like) into a numpy array."""
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values, dtype=float)


class YoloPersonDetector:
    """The pinned Ultralytics YOLO person detector.

    The model is loaded on first use, not on construction, so building one
    costs nothing and needs no backend - a missing backend surfaces as
    `models.BackendMissingError` at the first `detect`.

    Attributes:
        name: "name@backend-version" once loaded; the model's registry
            name before that.
    """

    def __init__(
        self,
        model: Any | None = None,
        config: PlayerConfig | None = None,
        models_config: ModelConfig | None = None,
        name: str | None = None,
    ) -> None:
        """Initializes the detector.

        Args:
            model: An already-loaded model object with ultralytics'
                `predict` interface - tests pass a fake. Loaded lazily
                from the registry when omitted.
            config: Supplies the inference size and confidence floor.
            models_config: Which pinned model to load and from where.
            name: Provenance name for an injected model.
        """
        self._model = model
        self._config = config or PlayerConfig()
        self._models_config = models_config or ModelConfig()
        self.name = name or self._models_config.person_model

    def _ensure(self) -> Any:
        """Returns the model, loading the pinned one on first use."""
        if self._model is None:
            loaded = models.load_person_model(self._models_config)
            self._model, self.name = loaded.model, loaded.source
        return self._model

    def detect(self, image: np.ndarray) -> list[PersonDetection]:
        """Finds the people in one frame.

        Args:
            image: (H, W, 3) uint8 RGB frame.

        Returns:
            Person boxes at or above `PlayerConfig.min_confidence`.

        Raises:
            tracking.models.ModelError: If the model cannot be loaded.
        """
        model = self._ensure()
        # Ultralytics takes numpy images as BGR.
        results = model.predict(
            np.ascontiguousarray(image[:, :, ::-1]),
            classes=[_COCO_PERSON],
            conf=self._config.min_confidence,
            imgsz=self._config.detector_size,
            verbose=False,
        )
        boxes = results[0].boxes
        corners = _as_array(boxes.xyxy).reshape(-1, 4)
        scores = _as_array(boxes.conf).reshape(-1)
        return [
            PersonDetection(tuple(float(v) for v in corner), float(score))
            for corner, score in zip(corners, scores)
        ]


def _ground_scale(calibration: Calibration, foot: Point) -> float:
    """Pixels per metre, sideways on the ground, at an image point."""
    spot = calibration.to_court(np.array(foot, dtype=float))
    pair = calibration.to_image(np.array([spot - (0.5, 0.0), spot + (0.5, 0.0)]))
    scale = float(np.linalg.norm(pair[1] - pair[0]))
    return scale if np.isfinite(scale) else 0.0


class MotionPersonDetector:
    """The classic fallback: people are person-sized things that moved.

    A running-average background at reduced size; pixels that differ from
    it are foreground; foreground blobs taller than wide and (given a
    calibration) between about one and two and a half metres tall at their
    own distance are reported as people.

    Honest limits: a player who stands still fades into the background at
    `motion_learning_rate` and is lost until they move; two people who
    overlap become one box; shadows widen boxes. It exists so the stage
    yields *something* without a learned model, and its detections say so
    through `name`.

    Attributes:
        name: "motion-person-v1".
        stateful: True - a checkpoint of a `PlayerTracker` using it keeps
            its background model.
    """

    name = "motion-person-v1"
    # Its background model is state: `PlayerTracker.snapshot` keeps it.
    stateful = True

    def __init__(
        self, calibration: Calibration | None = None, config: PlayerConfig | None = None
    ) -> None:
        """Initializes the detector.

        Args:
            calibration: When given, blobs are sized in metres at their
                own distance; without it only shape and area are checked.
            config: Motion thresholds and the working width.
        """
        self._calibration = calibration
        self._config = config or PlayerConfig()
        self._background: np.ndarray | None = None

    def detect(self, image: np.ndarray) -> list[PersonDetection]:
        """Finds moving people in one frame and updates the background.

        Frames must be fed in order; the first only seeds the background.

        Args:
            image: (H, W, 3) uint8 RGB frame.

        Returns:
            Boxes of person-sized moving blobs, in full-frame pixels.
        """
        config = self._config
        tall, wide = image.shape[:2]
        scale = min(1.0, config.motion_width / wide)
        size = (max(1, round(wide * scale)), max(1, round(tall * scale)))
        small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if self._background is None:
            self._background = grey
            return []
        moving = np.abs(grey - self._background) > config.motion_threshold
        # Foreground is learned ten times slower, so a player pausing
        # between shots is not absorbed at once.
        rate = np.where(moving, config.motion_learning_rate / 10, config.motion_learning_rate)
        self._background += rate.astype(np.float32) * (grey - self._background)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        joined = cv2.morphologyEx(moving.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
        found = []
        for label in range(1, count):
            left, top, width, height, area = (int(v) for v in stats[label])
            if area < config.motion_min_area * size[0] * size[1] or height < width * 0.8:
                continue
            box = (left / scale, top / scale, (left + width) / scale, (top + height) / scale)
            if self._calibration is not None:
                pixels_per_metre = _ground_scale(self._calibration, foot_of(box))
                if pixels_per_metre <= 0:
                    continue
                metres = (box[3] - box[1]) / pixels_per_metre
                if not config.motion_min_height_m <= metres <= config.motion_max_height_m:
                    continue
            fill = area / float(width * height)
            found.append(PersonDetection(box, float(np.clip(0.3 + 0.6 * fill, 0.0, 1.0))))
        return found


def foot_of(box: Box) -> Point:
    """Gives a box's footpoint: the middle of its lower edge.

    Args:
        box: (x0, y0, x1, y1) in pixels.

    Returns:
        (x, y) in pixels.
    """
    return ((box[0] + box[2]) / 2.0, box[3])


def side_of(court_y: float) -> str:
    """Names the end of the court a court-plane position is at.

    Args:
        court_y: Distance down the court from the near baseline, metres.

    Returns:
        "near" below the net, "far" at or beyond it.
    """
    return NEAR if court_y < NET_Y_M else FAR


def zone_of(side: str, court_x: float) -> str:
    """Names the half of the court a player stands in, from their own view.

    See the module docstring for the convention.

    Args:
        side: "near" or "far".
        court_x: Position across the court, metres, +x to the camera's
            right.

    Returns:
        "deuce" or "ad".
    """
    on_camera_right = court_x >= 0
    if side == NEAR:
        return DEUCE if on_camera_right else AD
    return AD if on_camera_right else DEUCE


def _iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Overlap of every box in `first` (N, 4) with every box in `second` (M, 4)."""
    left = np.maximum(first[:, None, 0], second[None, :, 0])
    top = np.maximum(first[:, None, 1], second[None, :, 1])
    right = np.minimum(first[:, None, 2], second[None, :, 2])
    bottom = np.minimum(first[:, None, 3], second[None, :, 3])
    shared = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    area_one = (first[:, 2] - first[:, 0]) * (first[:, 3] - first[:, 1])
    area_two = (second[:, 2] - second[:, 0]) * (second[:, 3] - second[:, 1])
    union = area_one[:, None] + area_two[None, :] - shared
    return np.where(union > 0, shared / np.where(union > 0, union, 1.0), 0.0)


def _padded(boxes: np.ndarray, pad: float) -> np.ndarray:
    """Grows (N, 4) boxes by a fraction of their own size on every side."""
    grow_x = (boxes[:, 2] - boxes[:, 0]) * pad / 2
    grow_y = (boxes[:, 3] - boxes[:, 1]) * pad / 2
    return boxes + np.stack([-grow_x, -grow_y, grow_x, grow_y], axis=1)


@dataclass
class _Track:
    """One person being followed (mutable, internal).

    Attributes:
        track_id: Its number, never reused.
        box: Its last box, (4,) pixels.
        velocity: The box centre's velocity, pixels per second.
        pts: When it was last detected.
        hits: How many detections it has been given.
        confidence: Its last detection's confidence.
    """

    track_id: int
    box: np.ndarray
    velocity: np.ndarray
    pts: float
    hits: int
    confidence: float

    def predicted(self, pts: float) -> np.ndarray:
        """Where its box should be at a later moment, at constant velocity."""
        shift = self.velocity * (pts - self.pts)
        return self.box + np.array([shift[0], shift[1], shift[0], shift[1]])


@dataclass(frozen=True)
class TrackedPerson:
    """A detection together with the track it was given to.

    Attributes:
        track_id: The track's number.
        box: The detection's box.
        confidence: The detection's confidence.
        hits: Detections the track has had, this one included.
    """

    track_id: int
    box: Box
    confidence: float
    hits: int


class PersonAssociator:
    """Links person boxes across frames into numbered tracks.

    ByteTrack's two-pass idea with a constant-velocity prediction: tracks
    are first matched against confident detections, then the tracks still
    unmatched against the low-confidence ones, which rescues a player the
    detector half-lost (motion blur, the far end) without letting noise
    start tracks. A pair may only be linked if their box areas are within a
    factor of three - the near and far player overlap in the image when
    both are at the net, and must not trade tracks.
    """

    def __init__(self, config: PlayerConfig | None = None) -> None:
        """Initializes the associator.

        Args:
            config: Thresholds for matching and for closing tracks.
        """
        self._config = config or PlayerConfig()
        self._tracks: list[_Track] = []
        self._next_id = 1

    def _match(
        self, tracks: list[_Track], detections: list[PersonDetection], pts: float
    ) -> tuple[list[tuple[_Track, PersonDetection]], list[_Track], list[PersonDetection]]:
        """Greedily pairs tracks with detections by padded-box overlap."""
        if not tracks or not detections:
            return [], list(tracks), list(detections)
        config = self._config
        predicted = np.array([track.predicted(pts) for track in tracks])
        seen = np.array([detection.box for detection in detections], dtype=float)
        overlap = _iou(_padded(predicted, config.match_pad), _padded(seen, config.match_pad))
        area_track = (predicted[:, 2] - predicted[:, 0]) * (predicted[:, 3] - predicted[:, 1])
        area_seen = (seen[:, 2] - seen[:, 0]) * (seen[:, 3] - seen[:, 1])
        ratio = area_track[:, None] / np.maximum(area_seen[None, :], 1e-9)
        overlap[(ratio > 3.0) | (ratio < 1 / 3.0)] = 0.0
        pairs, used_tracks, used_seen = [], set(), set()
        for flat in np.argsort(-overlap, axis=None):
            row, column = divmod(int(flat), overlap.shape[1])
            if overlap[row, column] < config.match_iou:
                break
            if row in used_tracks or column in used_seen:
                continue
            used_tracks.add(row)
            used_seen.add(column)
            pairs.append((tracks[row], detections[column]))
        spare_tracks = [t for index, t in enumerate(tracks) if index not in used_tracks]
        spare_seen = [d for index, d in enumerate(detections) if index not in used_seen]
        return pairs, spare_tracks, spare_seen

    def update(self, detections: Iterable[PersonDetection], pts: float) -> list[TrackedPerson]:
        """Assigns one frame's detections to tracks.

        Args:
            detections: The people found in the frame.
            pts: The frame's timestamp, seconds. Must not go backwards.

        Returns:
            Every detection that now belongs to a track, with the track's
            number. Low-confidence detections matching no track are
            dropped.
        """
        config = self._config
        kept = [d for d in detections if d.confidence >= config.min_confidence]
        strong = [d for d in kept if d.confidence >= config.high_confidence]
        weak = [d for d in kept if d.confidence < config.high_confidence]
        self._tracks = [
            track for track in self._tracks if pts - track.pts <= config.max_track_age_seconds
        ]
        first, spare_tracks, unclaimed = self._match(self._tracks, strong, pts)
        second, _, _ = self._match(spare_tracks, weak, pts)
        out = []
        for track, detection in first + second:
            box = np.array(detection.box, dtype=float)
            elapsed = pts - track.pts
            if elapsed > 0:
                moved = (box[:2] + box[2:]) / 2 - (track.box[:2] + track.box[2:]) / 2
                track.velocity = 0.5 * track.velocity + 0.5 * moved / elapsed
            track.box, track.pts, track.confidence = box, pts, detection.confidence
            track.hits += 1
            out.append(
                TrackedPerson(track.track_id, detection.box, detection.confidence, track.hits)
            )
        for detection in unclaimed:
            track = _Track(
                self._next_id, np.array(detection.box, dtype=float), np.zeros(2), pts, 1,
                detection.confidence,
            )
            self._next_id += 1
            self._tracks.append(track)
            out.append(TrackedPerson(track.track_id, detection.box, detection.confidence, 1))
        return out


@dataclass(frozen=True)
class PlayerObservation:
    """Where one player was at one moment.

    Attributes:
        frame: The frame's number in the recording (or in the stream it
            came from, when the frames carried no index).
        pts: The frame's timestamp, seconds.
        side: The end of the court: "near" or "far". This is the player's
            identity for attributing shots.
        player: The person: "A" (at the near end when tracking began) or
            "B". Changes relative to `side` only at a recorded `EndSwap`.
        box: (x0, y0, x1, y1) in pixels.
        foot_image: The footpoint - middle of the box's lower edge.
        foot_court: The footpoint on the court plane, (x, y) metres in
            `court.py`'s frame.
        zone: "deuce" or "ad", from the player's own point of view.
        confidence: The detection's confidence; for an interpolated
            observation, 0.9 of the weaker of the two it lies between.
        provenance: `MEASURED` for a detection, `INTERPOLATED` between two.
        foot_clipped: True when the box runs off the bottom of the frame,
            so the feet were not seen and the true position is further
            from the net than `foot_court` says.
        track_id: The association track this came from, or None when
            interpolated between two different tracks.
    """

    frame: int
    pts: float
    side: str
    player: str
    box: Box
    foot_image: Point
    foot_court: Point
    zone: str
    confidence: float
    provenance: Provenance
    foot_clipped: bool = False
    track_id: int | None = None

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field, tuples as lists and provenance as its value.
        """
        return {
            "frame": self.frame,
            "pts": self.pts,
            "side": self.side,
            "player": self.player,
            "box": list(self.box),
            "foot_image": list(self.foot_image),
            "foot_court": list(self.foot_court),
            "zone": self.zone,
            "confidence": self.confidence,
            "provenance": self.provenance.value,
            "foot_clipped": self.foot_clipped,
            "track_id": self.track_id,
        }


@dataclass(frozen=True)
class EndSwap:
    """The players changed ends.

    Attributes:
        pts: From when the swapped labelling applies, seconds.
        frame: The frame at that moment.
        provenance: `MEASURED` when a track was followed across the net;
            `INFERRED` when concluded from appearance after both ends had
            been empty.
        confidence: 0 to 1. For an inferred swap, how decisively the
            appearance comparison favoured it.
        reason: A sentence saying what was seen.
        last_settled_pts: The last moment the previous arrangement was
            confirmed. Between it and `pts` the players were between
            arrangements (walking, at the bench), and the `player` label
            of an observation in that stretch means little.
    """

    pts: float
    frame: int
    provenance: Provenance
    confidence: float
    reason: str
    last_settled_pts: float = 0.0

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field, provenance as its value.
        """
        return {
            "pts": self.pts,
            "frame": self.frame,
            "provenance": self.provenance.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "last_settled_pts": self.last_settled_pts,
        }


@dataclass(frozen=True)
class PlayerQuality:
    """How well the players were followed over a stretch.

    Attributes:
        detector_frames: Frames the detector was run on.
        near_found: Fraction of those with a near-end player.
        far_found: Fraction with a far-end player.
        both_found: Fraction with both.
        near_track_changes: Times the near end's observation passed from
            one association track to another - each is a moment identity
            rested on court position alone.
        far_track_changes: The same for the far end.
        end_swaps: Recorded changes of ends.
        mean_confidence: Mean detection confidence of the observations.
        near_clipped: Fraction of near-end observations whose feet were
            cut off by the frame edge.
        rejected_people: Tracked detections discarded because their feet
            were not on this court (neighbours, passers-by).
    """

    detector_frames: int
    near_found: float
    far_found: float
    both_found: float
    near_track_changes: int
    far_track_changes: int
    end_swaps: int
    mean_confidence: float
    near_clipped: float
    rejected_people: int

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field.
        """
        return dict(self.__dict__)


def _blend(first: PlayerObservation, second: PlayerObservation, frame: int, pts: float):
    """Interpolates one player's observation between two detections."""
    share = (pts - first.pts) / (second.pts - first.pts)

    def mix(one, two):
        """Linear blend of two equal-length tuples at `share`."""
        return tuple(float(a + (b - a) * share) for a, b in zip(one, two))

    court = mix(first.foot_court, second.foot_court)
    return PlayerObservation(
        frame=frame,
        pts=pts,
        side=first.side,
        player=first.player,
        box=mix(first.box, second.box),
        foot_image=mix(first.foot_image, second.foot_image),
        foot_court=court,
        zone=zone_of(first.side, court[0]),
        confidence=0.9 * min(first.confidence, second.confidence),
        provenance=Provenance.INTERPOLATED,
        foot_clipped=first.foot_clipped or second.foot_clipped,
        track_id=first.track_id if first.track_id == second.track_id else None,
    )


@dataclass(frozen=True)
class PlayerTracks:
    """Both players over a stretch of video.

    Only detections are stored (`near`, `far`); `at` produces the position
    at any other moment by interpolation, or declines to.

    Attributes:
        near: The near-end player's detections, in time order.
        far: The far-end player's.
        swaps: Recorded changes of ends, in time order.
        detector_times: Timestamp of every frame the detector ran on -
            the denominator for "how often was each player found".
        source: The detector that produced the boxes.
        max_interpolation_seconds: Longest gap `at` will bridge.
        rejected_people: Tracked detections discarded as not on this
            court.
    """

    near: tuple[PlayerObservation, ...]
    far: tuple[PlayerObservation, ...]
    swaps: tuple[EndSwap, ...]
    detector_times: tuple[float, ...]
    source: str
    max_interpolation_seconds: float
    rejected_people: int = 0
    _times: dict[str, list[float]] = field(default_factory=dict, repr=False, compare=False)

    def side(self, side: str) -> tuple[PlayerObservation, ...]:
        """Returns one end's detections.

        Args:
            side: "near" or "far".

        Returns:
            That end's observations in time order.

        Raises:
            ValueError: If `side` is neither.
        """
        if side not in SIDES:
            raise ValueError(f"side must be 'near' or 'far', got {side!r}.")
        return self.near if side == NEAR else self.far

    def at(self, pts: float, side: str, frame: int | None = None) -> PlayerObservation | None:
        """Gives one end's player at a moment.

        Args:
            pts: The moment, seconds.
            side: "near" or "far".
            frame: The frame number to stamp on an interpolated result;
                interpolated from the neighbours' when omitted.

        Returns:
            The detection made at that moment (to within a millisecond);
            otherwise an `INTERPOLATED` observation between the detections
            either side, if they are no more than
            `max_interpolation_seconds` apart; otherwise None. Never
            extrapolated beyond the first or last detection.
        """
        observations = self.side(side)
        if side not in self._times:
            self._times[side] = [observation.pts for observation in observations]
        times = self._times[side]
        after = bisect.bisect_left(times, pts - 1e-3)
        if after < len(times) and abs(times[after] - pts) <= 1e-3:
            return observations[after]
        if after == 0 or after == len(times):
            return None
        first, second = observations[after - 1], observations[after]
        if second.pts - first.pts > self.max_interpolation_seconds:
            return None
        if frame is None:
            share = (pts - first.pts) / (second.pts - first.pts)
            frame = round(first.frame + (second.frame - first.frame) * share)
        return _blend(first, second, frame, pts)

    def boxes_at(self, pts: float) -> list[Box]:
        """Gives the boxes of whichever players are known at a moment.

        Args:
            pts: The moment, seconds.

        Returns:
            Zero, one or two boxes - what a ball detector should treat as
            "probably a limb, not the ball".
        """
        found = (self.at(pts, side) for side in SIDES)
        return [observation.box for observation in found if observation is not None]

    def player_at(self, pts: float, side: str) -> str:
        """Says which person is at an end at a moment.

        Args:
            pts: The moment, seconds.
            side: "near" or "far".

        Returns:
            "A" or "B", from the recorded end swaps alone.
        """
        swapped = sum(1 for swap in self.swaps if swap.pts <= pts) % 2 == 1
        return _label(side, swapped)

    def quality(self) -> PlayerQuality:
        """Summarises how well the players were followed.

        Returns:
            Found fractions, track changes, swaps and confidence.
        """
        frames = len(self.detector_times)
        near_times = {round(o.pts, 3) for o in self.near}
        far_times = {round(o.pts, 3) for o in self.far}

        def changes(observations: Sequence[PlayerObservation]) -> int:
            """Counts hand-overs from one association track to another."""
            ids = [o.track_id for o in observations]
            return sum(1 for one, two in zip(ids, ids[1:]) if one != two)

        everything = self.near + self.far
        return PlayerQuality(
            detector_frames=frames,
            near_found=len(near_times) / frames if frames else 0.0,
            far_found=len(far_times) / frames if frames else 0.0,
            both_found=len(near_times & far_times) / frames if frames else 0.0,
            near_track_changes=changes(self.near),
            far_track_changes=changes(self.far),
            end_swaps=len(self.swaps),
            mean_confidence=(
                float(np.mean([o.confidence for o in everything])) if everything else 0.0
            ),
            near_clipped=(
                sum(o.foot_clipped for o in self.near) / len(self.near) if self.near else 0.0
            ),
            rejected_people=self.rejected_people,
        )

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Observations per end, swaps, source and the quality summary.
        """
        return {
            "source": self.source,
            "near": [observation.to_dict() for observation in self.near],
            "far": [observation.to_dict() for observation in self.far],
            "swaps": [swap.to_dict() for swap in self.swaps],
            "quality": self.quality().to_dict(),
        }


def _label(side: str, swapped: bool) -> str:
    """Names the person at an end, given whether ends are currently swapped."""
    return "A" if (side == NEAR) != swapped else "B"


def appearance(image: np.ndarray, box: Box, bins: tuple[int, int] = (12, 4)) -> np.ndarray | None:
    """Summarises what a person is wearing as a colour histogram.

    A hue-and-saturation histogram of the middle of the box's upper half -
    torso, not court. Normalised, so the huge near player and the tiny far
    one are comparable. Crude on purpose: two players dressed alike cannot
    be told apart by it, and it then decides nothing.

    Args:
        image: (H, W, 3) uint8 RGB frame.
        box: The person's box in that frame.
        bins: (hue bins, saturation bins).

    Returns:
        A flat histogram summing to 1, or None when the box holds no
        pixels.
    """
    tall, wide = image.shape[:2]
    x0, y0, x1, y1 = box
    left = int(np.clip(round(x0 + 0.25 * (x1 - x0)), 0, wide))
    right = int(np.clip(round(x1 - 0.25 * (x1 - x0)), 0, wide))
    top = int(np.clip(round(y0 + 0.15 * (y1 - y0)), 0, tall))
    bottom = int(np.clip(round(y0 + 0.55 * (y1 - y0)), 0, tall))
    if right <= left or bottom <= top:
        return None
    patch = cv2.cvtColor(np.ascontiguousarray(image[top:bottom, left:right]), cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist([patch], [0, 1], None, list(bins), [0, 180, 0, 256]).ravel()
    total = float(histogram.sum())
    return (histogram / total).astype(np.float64) if total > 0 else None


def similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Compares two appearance histograms.

    Args:
        first: One histogram.
        second: Another, the same shape.

    Returns:
        Their intersection: 1 for identical distributions, 0 for disjoint.
    """
    return float(np.minimum(first, second).sum())


def player_region(calibration: Calibration, config: PlayerConfig | None = None) -> Box:
    """Finds the part of the frame a player of this court can appear in.

    The court plus its margins, lifted by a person's height, projected
    into the image and boxed. Running the detector on this crop instead of
    the whole frame saves the sky and the trees.

    Args:
        calibration: The court calibration.
        config: Supplies the margins around the court.

    Returns:
        (x0, y0, x1, y1) in whole pixels, clipped to the frame; the whole
        frame when the camera cannot be recovered.
    """
    config = config or PlayerConfig()
    wide, tall = calibration.image_size
    camera = camera_from_homography(calibration.homography, calibration.image_size)
    if camera is None:
        return (0.0, 0.0, float(wide), float(tall))
    reach = DOUBLES_WIDTH_M / 2 + config.lateral_margin_m
    corners = np.array(
        [
            (x, y, z, 1.0)
            for x in (-reach, reach)
            for y in (-config.behind_margin_m, COURT_LENGTH_M + config.behind_margin_m)
            for z in (0.0, _PERSON_HEIGHT_M)
        ]
    )
    mapped = corners @ camera.T
    ahead = mapped[:, 2] > 1e-6
    if not ahead.any():
        return (0.0, 0.0, float(wide), float(tall))
    pixels = mapped[ahead, :2] / mapped[ahead, 2:]
    top = float(np.clip(np.floor(pixels[:, 1].min()), 0, tall))
    if ahead.all():
        left = float(np.clip(np.floor(pixels[:, 0].min()), 0, wide))
        right = float(np.clip(np.ceil(pixels[:, 0].max()), 0, wide))
        bottom = float(np.clip(np.ceil(pixels[:, 1].max()), 0, tall))
    else:
        # Part of the region is behind the camera: it runs off the bottom
        # and both sides of the frame.
        left, right, bottom = 0.0, float(wide), float(tall)
    if right - left < 2 or bottom - top < 2:
        return (0.0, 0.0, float(wide), float(tall))
    return (left, top, right, bottom)


@dataclass
class _Slot:
    """What is known about one end of the court while tracking (mutable).

    Attributes:
        track_id: The track currently chosen for this end, or None.
        last_pts: When this end last had a player, or None.
        last_look: The appearance seen at `last_pts`, or None.
    """

    track_id: int | None = None
    last_pts: float | None = None
    last_look: np.ndarray | None = None


@dataclass
class _Run:
    """An unbroken stretch with a player at both ends (mutable, internal).

    Attributes:
        pts: When it began.
        frame: The frame it began at.
        ids: Association tracks seen at each end during it.
        looks: Running appearance of whoever was at each end during it.
        counts: Sightings each of `looks` blends.
        judged: Whether it has lasted long enough to have been compared
            with the arrangement before it.
    """

    pts: float
    frame: int
    ids: dict[str, set[int]] = field(default_factory=lambda: {NEAR: set(), FAR: set()})
    looks: dict[str, np.ndarray | None] = field(
        default_factory=lambda: {NEAR: None, FAR: None}
    )
    counts: dict[str, int] = field(default_factory=lambda: {NEAR: 0, FAR: 0})
    judged: bool = False


@dataclass(frozen=True)
class ObservationBatch:
    """What a `PlayerTracker` recorded since its observations were last taken.

    Attributes:
        near: Near-end detections, in time order, **not yet labelled**
            with the person (`player` is ""): a change of ends judged
            later can still relabel them. Label them with `label_tracks`.
        far: Far-end detections, likewise.
        detector_times: When the detector ran.
        rejected_people: Detections discarded as not on this court.
    """

    near: tuple[PlayerObservation, ...]
    far: tuple[PlayerObservation, ...]
    detector_times: tuple[float, ...]
    rejected_people: int


class PlayerTracker:
    """Follows the two players of one court through a stream of frames.

    Feed frames in order with `update`; collect the result with `finish`.
    The detector runs on every `PlayerConfig.stride`-th frame fed.

    A long recording can be followed in pieces without keeping every
    observation in memory: `take_observations` hands over (and forgets)
    what was recorded so far, and `snapshot` / `resume` save and restore
    everything else - the association tracks, the settled arrangement, the
    swaps, the stride's phase - so a tracker restored from a snapshot
    carries on exactly as the original would have.
    """

    def __init__(
        self,
        detector: PersonDetector,
        calibration: Calibration,
        config: PlayerConfig | None = None,
        crop: bool = True,
    ) -> None:
        """Initializes the tracker.

        Args:
            detector: Finds people in a frame.
            calibration: Places their feet on the court.
            config: Thresholds; see `PlayerConfig`.
            crop: Hand the detector only `player_region`, not the whole
                frame.
        """
        self._detector = detector
        self._calibration = calibration
        self._config = config or PlayerConfig()
        self._associator = PersonAssociator(self._config)
        self._region = player_region(calibration, self._config) if crop else None
        self._fed = 0
        self._times: list[float] = []
        self._observations: dict[str, list[PlayerObservation]] = {NEAR: [], FAR: []}
        self._slots = {NEAR: _Slot(), FAR: _Slot()}
        self._known: dict[str, np.ndarray] = {}
        self._track_side: dict[int, str] = {}
        self._swaps: list[EndSwap] = []
        self._run: _Run | None = None
        self._settled_ids: dict[str, set[int]] | None = None
        self._settled_last = 0.0
        self._rejected = 0

    def recalibrate(self, calibration: Calibration) -> None:
        """Places feet through a new calibration from now on.

        Used when the camera has moved mid-match and the pipeline adopted
        a fresh calibration: identities, tracks and swaps carry on; only
        where the court is in the picture changes.

        Args:
            calibration: The calibration to use from now on.
        """
        self._calibration = calibration
        if self._region is not None:
            self._region = player_region(calibration, self._config)

    def _detect(self, image: np.ndarray) -> list[PersonDetection]:
        """Runs the detector, on the player region when cropping."""
        if self._region is None:
            return self._detector.detect(image)
        left, top, right, bottom = (int(v) for v in self._region)
        found = self._detector.detect(image[top:bottom, left:right])
        return [
            PersonDetection(
                (d.box[0] + left, d.box[1] + top, d.box[2] + left, d.box[3] + top), d.confidence
            )
            for d in found
        ]

    def _on_court(self, person: TrackedPerson) -> tuple[Point, float] | None:
        """Places a person's feet on the court, or rejects them.

        Returns:
            (court position, metres outside the doubles court) when the
            feet are within this court's margins, else None.
        """
        config = self._config
        spot = self._calibration.to_court(np.array(foot_of(person.box)))
        if not np.isfinite(spot).all():
            return None
        across = abs(spot[0]) - DOUBLES_WIDTH_M / 2
        along = max(-spot[1], spot[1] - COURT_LENGTH_M)
        if across > config.lateral_margin_m or along > config.behind_margin_m:
            return None
        outside = float(np.hypot(max(across, 0.0), max(along, 0.0)))
        return (float(spot[0]), float(spot[1])), outside

    def _side_for(self, person: TrackedPerson, court_y: float) -> str:
        """Decides a person's end, holding their previous one near the net."""
        previous = self._track_side.get(person.track_id)
        if previous is not None and abs(court_y - NET_Y_M) < self._config.net_band_m:
            return previous
        return side_of(court_y)

    def update(self, frame: VideoFrame) -> None:
        """Takes the next frame; runs the detector if it is this frame's turn.

        Args:
            frame: The frame, in display order.
        """
        turn = self._fed % max(1, self._config.stride) == 0
        number = frame.index if frame.index is not None else self._fed
        self._fed += 1
        if not turn:
            return
        config = self._config
        pts = frame.pts
        self._times.append(pts)
        people = self._associator.update(self._detect(frame.image), pts)
        tall = self._calibration.image_size[1]
        candidates: dict[str, list[tuple[TrackedPerson, Point, float]]] = {NEAR: [], FAR: []}
        for person in people:
            placed = self._on_court(person)
            if placed is None:
                self._rejected += 1
                continue
            spot, outside = placed
            side = self._side_for(person, spot[1])
            candidates[side].append((person, spot, outside))
        for side in SIDES:
            slot = self._slots[side]
            options = candidates[side]
            held = [option for option in options if option[0].track_id == slot.track_id]
            ready = [option for option in options if option[0].hits >= config.min_track_hits]
            if held:
                person, spot, _ = held[0]
            elif ready:
                person, spot, _ = min(ready, key=lambda o: (round(o[2], 1), -o[0].confidence))
            else:
                continue
            self._record(frame, number, side, person, spot, tall)
        self._watch(pts, number)

    def _record(
        self,
        frame: VideoFrame,
        number: int,
        side: str,
        person: TrackedPerson,
        spot: Point,
        tall: int,
    ) -> None:
        """Stores one end's observation and updates that end's state."""
        config = self._config
        slot = self._slots[side]
        came_from = self._track_side.get(person.track_id)
        if came_from is not None and came_from != side:
            # A followed track turned up at the other end: whatever
            # arrangement held until now is over. What replaces it is
            # judged only once it has settled (`_watch`).
            self._run = None
        self._track_side[person.track_id] = side
        slot.track_id, slot.last_pts = person.track_id, frame.pts
        slot.last_look = appearance(frame.image, person.box, config.appearance_bins)
        clipped = person.box[3] >= tall * (1 - config.foot_edge_margin)
        self._observations[side].append(
            PlayerObservation(
                frame=number,
                pts=frame.pts,
                side=side,
                player="",
                box=tuple(float(v) for v in person.box),
                foot_image=foot_of(person.box),
                foot_court=spot,
                zone=zone_of(side, spot[0]),
                confidence=float(person.confidence),
                provenance=Provenance.MEASURED,
                foot_clipped=bool(clipped),
                track_id=person.track_id,
            )
        )

    def _watch(self, pts: float, number: int) -> None:
        """Follows the current arrangement and judges it once it has settled.

        A *run* is an unbroken stretch with a player at both ends. Players
        wander during a changeover - to the bench, round the net post and
        back - so nothing is concluded from a crossing itself. Only when a
        run has lasted `swap_settle_seconds` is it compared with the last
        settled arrangement.
        """
        config = self._config
        occupied = all(
            slot.last_pts is not None and pts - slot.last_pts <= config.max_track_age_seconds
            for slot in self._slots.values()
        )
        if not occupied:
            self._run = None
            return
        if self._run is None:
            self._run = _Run(pts, number)
        run = self._run
        for side in SIDES:
            slot = self._slots[side]
            if slot.last_pts != pts:
                continue
            run.ids[side].add(slot.track_id)
            if slot.last_look is not None:
                run.counts[side] += 1
                rate = max(config.appearance_rate, 1.0 / run.counts[side])
                held = run.looks[side]
                run.looks[side] = (
                    slot.last_look if held is None else (1 - rate) * held + rate * slot.last_look
                )
        if not run.judged:
            seen_enough = all(
                run.looks[side] is not None and run.counts[side] >= config.min_track_hits
                for side in SIDES
            )
            if pts - run.pts < config.swap_settle_seconds or not seen_enough:
                return
            run.judged = True
            self._judge(run)
        self._settled_ids = {side: set(run.ids[side]) for side in SIDES}
        self._settled_last = min(slot.last_pts for slot in self._slots.values())
        self._known = {side: run.looks[side].copy() for side in SIDES}

    def _judge(self, run: _Run) -> None:
        """Compares a newly settled arrangement with the one before it.

        Two kinds of evidence. **Appearance**: do the people now at each
        end look like each other's earlier selves? **Tracks**: is an
        association track that was at one end now at the other? On the
        real recording the tracks alone were wrong - the players passed
        each other at the bench and one track hopped from one to the
        other - so appearance, when it is decisive, outranks them:

        * tracks crossed, appearance does not contradict: `MEASURED`;
        * appearance says swapped after a long enough interruption,
          whatever the tracks say: `INFERRED`;
        * tracks crossed but the players plainly look the same as before:
          an identity switch, not a swap - nothing recorded.
        """
        config = self._config
        before = self._settled_ids
        if before is None:
            logger.debug("First settled arrangement at %.2fs: %s", run.pts, run.ids)
            return
        crossed = bool(run.ids[NEAR] & before[FAR] or run.ids[FAR] & before[NEAR])
        stayed = bool(run.ids[NEAR] & before[NEAR] or run.ids[FAR] & before[FAR])
        followed = crossed and not stayed
        now, known = run.looks, self._known
        stay = similarity(now[NEAR], known[NEAR]) + similarity(now[FAR], known[FAR])
        swap = similarity(now[NEAR], known[FAR]) + similarity(now[FAR], known[NEAR])
        margin = (swap - stay) / 2
        away = run.pts - self._settled_last
        logger.debug(
            "Arrangement from %.2fs, %.1fs after the last: tracks crossed=%s stayed=%s, "
            "appearance margin %.2f.",
            run.pts, away, crossed, stayed, margin,
        )
        if followed and margin > -config.swap_margin:
            agrees = margin >= config.swap_margin
            self._swap(
                run,
                Provenance.MEASURED,
                0.9 if agrees else 0.6,
                "A track was followed from one end to the other and the new arrangement held"
                + (
                    f"; the players' appearance agrees (margin {margin:.2f})."
                    if agrees
                    else "; their appearance could not confirm it."
                ),
            )
        elif followed:
            logger.warning(
                "A track changed ends at %.2fs but both players look as before "
                "(margin %.2f): taken as a tracking slip, not a change of ends.",
                run.pts, margin,
            )
        elif margin >= config.swap_margin and away >= config.swap_absence_seconds:
            self._swap(
                run,
                Provenance.INFERRED,
                float(np.clip(margin, 0.0, 1.0)),
                "The players now at each end look like each other's earlier selves "
                f"(appearance margin {margin:.2f})"
                + ("; the tracks did not show it." if stayed else "; the tracks were lost."),
            )

    def _swap(self, run: _Run, provenance: Provenance, confidence: float, reason: str) -> None:
        """Records a change of ends, dated to when the new arrangement began."""
        self._swaps.append(
            EndSwap(
                pts=run.pts,
                frame=run.frame,
                provenance=provenance,
                confidence=confidence,
                reason=reason,
                last_settled_pts=self._settled_last,
            )
        )
        logger.info("End swap (%s) from %.2fs: %s", provenance.value, run.pts, reason)

    @property
    def swaps(self) -> tuple[EndSwap, ...]:
        """Every change of ends recorded so far, in time order."""
        return tuple(self._swaps)

    @property
    def detector_name(self) -> str:
        """The detector's provenance name."""
        return self._detector.name

    def take_observations(self) -> ObservationBatch:
        """Hands over what was recorded since the last call, and forgets it.

        Returns:
            The unlabelled observations, detector times and rejection
            count accumulated since the tracker was made, resumed, or
            last asked.
        """
        batch = ObservationBatch(
            near=tuple(self._observations[NEAR]),
            far=tuple(self._observations[FAR]),
            detector_times=tuple(self._times),
            rejected_people=self._rejected,
        )
        self._observations = {NEAR: [], FAR: []}
        self._times = []
        self._rejected = 0
        return batch

    def snapshot(self) -> bytes:
        """Saves the tracker's state, to carry on from later.

        The detector is saved with it only when it says it is `stateful`
        (the motion detector's background model); a learned detector is
        stateless and is handed back to `resume` instead.

        Returns:
            The pickled state. Load it only with `resume`, and only from
            a file this program wrote.
        """
        detector = self._detector
        if not getattr(detector, "stateful", False):
            self._detector = None
        try:
            return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            self._detector = detector

    @classmethod
    def resume(cls, snapshot: bytes, detector: PersonDetector | None = None) -> PlayerTracker:
        """Restores a tracker from `snapshot`.

        Args:
            snapshot: What `snapshot` returned.
            detector: The detector to carry on with, when the snapshot did
                not keep its own.

        Returns:
            The tracker, as it was when the snapshot was taken.

        Raises:
            ValueError: If the snapshot kept no detector and none is given.
            TypeError: If the snapshot is not of a `PlayerTracker`.
        """
        tracker = pickle.loads(snapshot)  # noqa: S301 - our own checkpoint file
        if not isinstance(tracker, cls):
            raise TypeError("The snapshot is not of a player tracker.")
        if tracker._detector is None:
            if detector is None:
                raise ValueError("This snapshot kept no detector; pass the one to resume with.")
            tracker._detector = detector
        return tracker

    def finish(self) -> PlayerTracks:
        """Returns everything tracked so far.

        Returns:
            Both ends' observations, labelled with the person ("A"/"B")
            according to the recorded swaps, plus the swaps themselves.
        """
        tracks = PlayerTracks(
            near=(),
            far=(),
            swaps=tuple(self._swaps),
            detector_times=tuple(self._times),
            source=self._detector.name,
            max_interpolation_seconds=self._config.max_interpolation_seconds,
            rejected_people=self._rejected,
        )
        labelled = {
            side: tuple(
                replace(observation, player=tracks.player_at(observation.pts, side))
                for observation in self._observations[side]
            )
            for side in SIDES
        }
        return replace(tracks, near=labelled[NEAR], far=labelled[FAR])


def label_tracks(
    near: Sequence[PlayerObservation],
    far: Sequence[PlayerObservation],
    swaps: Sequence[EndSwap],
    detector_times: Sequence[float],
    source: str,
    config: PlayerConfig | None = None,
    rejected_people: int = 0,
) -> PlayerTracks:
    """Builds `PlayerTracks` from observations gathered in pieces.

    Every observation is labelled with its person from `swaps` alone, as
    `PlayerTracker.finish` does - so observations taken out of a tracker
    before a later change of ends was judged come out right.

    Args:
        near: Near-end observations, in time order.
        far: Far-end observations, in time order.
        swaps: Every change of ends, in time order.
        detector_times: When the detector ran.
        source: The detector's name.
        config: Supplies the longest gap `PlayerTracks.at` bridges.
        rejected_people: Detections discarded as not on this court.

    Returns:
        The labelled tracks.
    """
    config = config or PlayerConfig()
    tracks = PlayerTracks(
        near=(),
        far=(),
        swaps=tuple(swaps),
        detector_times=tuple(detector_times),
        source=source,
        max_interpolation_seconds=config.max_interpolation_seconds,
        rejected_people=rejected_people,
    )
    return replace(
        tracks,
        near=tuple(replace(o, player=tracks.player_at(o.pts, NEAR)) for o in near),
        far=tuple(replace(o, player=tracks.player_at(o.pts, FAR)) for o in far),
    )


def track_players(
    frames: Iterable[VideoFrame],
    detector: PersonDetector,
    calibration: Calibration,
    config: PlayerConfig | None = None,
    crop: bool = True,
) -> PlayerTracks:
    """Follows the two players through a stream of frames.

    Frames are consumed one at a time and never held, so this can walk a
    whole match from `video.iter_frames`.

    Args:
        frames: Frames in display order.
        detector: Finds people in a frame.
        calibration: Places their feet on the court.
        config: Thresholds; see `PlayerConfig`.
        crop: Hand the detector only the part of the frame a player can
            be in.

    Returns:
        Both players' observations and any changes of ends.
    """
    tracker = PlayerTracker(detector, calibration, config, crop)
    for frame in frames:
        tracker.update(frame)
    return tracker.finish()
