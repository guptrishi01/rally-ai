"""The ball: candidates in every frame, then the one path through them.

Why two steps
-------------
The Phase 0 spike took "the smallest moving blob in each frame" as the ball
and got a scatter over limbs, the net cord and people in the background: no
single frame says which moving speck is the ball. So this module never asks
a frame that question. A `BallDetector` proposes **several** candidates per
frame, each with a score; `link_candidates` then finds the ball as the
candidates that, taken together over time, move the way a ball can: a
constant-velocity (Kalman) model predicts where a track should continue,
only a detection inside that gate may extend it, a step implying an
impossible speed is never linked, and among the tracks that compete for the
same stretch of time the set with the best total evidence wins.

Detectors
---------
* `MotionBallDetector` (Track A, classic, no weights). A pixel is a
  candidate when its *yellowness* (min(R, G) - B: high for a tennis ball,
  low for court, sky and most clothing) is higher than in both the frame a
  little earlier and the frame a little later - which marks a moving
  yellow thing where it **is**, with no ghost where it was. It ignores
  pixels that are always moving (foliage, a windscreen) and scores each
  blob by colour, by contrast and - through `FlightVolume` - by whether a
  ball anywhere above *this* court along that line of sight could look
  that size. That last test is what discounts a neighbouring court's
  ball. This grew out of the three-frame difference in
  `swingvision_import.footage.compose_ball_path`.
* `HeatmapBallDetector` (Track B, learned). The slot for a TrackNet-style
  model: it turns a model's heatmap into the same candidates. **No such
  model has been trained**; `models.load_ball_heatmap` raises
  `WeightsMissingError`.

What the track does and does not claim
--------------------------------------
* A position is `measured` only where a detection was linked.
* A silence of at most `BallConfig.max_gap_seconds` inside a segment is
  bridged from the filter (tagged `interpolated`, visibility `bridged`,
  lower confidence). A longer silence ends the segment: those frames get
  **no observation at all** and are listed in `BallTrack.gaps`.
* `image_xy` is where the ball is in the picture. `ground_ray_xy` is
  where the line of sight through it meets the court plane. **That is the
  ball's court position only while the ball is on the ground** - at a
  bounce. For a ball in the air it is a point further from the camera
  than the ball really is, which is why it is not called a position.
* `court_xy` - the ball's actual place on the court - is therefore **None
  on everything this module produces**. Deciding *when* the ball is on
  the ground is the events phase's job; at a bounce it will copy
  `ground_ray_xy` into `court_xy`, and nowhere else.
* A segment is a stretch of smooth motion. Segments tend to break at hits
  (the velocity changes abruptly) and sometimes at bounces; the boundaries
  are evidence for the events phase, not events.

Quality is reported, not assumed: `BallTrack.quality` gives coverage, the
gap histogram and mean confidence over any stretch, and `BallQuality.weak`
/ `notice` are what a later "ball tracking was weak on this match" message
is built from. `SceneStats` records how bright and what colour the court
was, since that is what a retrained model would need to match.
"""

from __future__ import annotations

import bisect
import logging
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

from .calibrate import Calibration
from .config import BallConfig
from .court import COURT_LENGTH_M, DOUBLES_WIDTH_M, NET_Y_M, Point, camera_from_homography
from .players import PlayerTracks
from .provenance import Provenance, provenance_counts
from .video import VideoFrame

logger = logging.getLogger(__name__)

_GAP_BUCKETS = ((0.1, "under 0.1s"), (0.5, "0.1-0.5s"), (2.0, "0.5-2s"), (10.0, "2-10s"))
_LONGEST_BUCKET = "over 10s"


class Visibility(str, Enum):
    """Whether the ball was seen in a frame.

    Attributes:
        VISIBLE: A detection was linked into the track.
        BRIDGED: Not detected; position bridged across a short gap.
        UNKNOWN: No position is claimed for this frame.
    """

    VISIBLE = "visible"
    BRIDGED = "bridged"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BallCandidate:
    """One thing in one frame that might be the ball.

    Attributes:
        xy: Its centre, (x, y) pixels.
        score: How ball-like it looks on its own, 0 to 1.
        extent: Its longest side in pixels.
    """

    xy: Point
    score: float
    extent: float = 0.0


@dataclass(frozen=True)
class FrameCandidates:
    """Every candidate found in one frame.

    Attributes:
        frame: The frame's number.
        pts: Its timestamp, seconds.
        candidates: The candidates, best first. Empty is a valid answer.
        searched: False for the frames at either end of a stream, which
            have no earlier or later frame to compare with and so were
            not searched at all.
    """

    frame: int
    pts: float
    candidates: tuple[BallCandidate, ...]
    searched: bool = True


@runtime_checkable
class BallDetector(Protocol):
    """Anything that proposes ball candidates from a short run of frames.

    A detector looks at a frame together with one earlier and one later
    frame, `span` frames away. `prepare` is called once per frame so
    per-frame work is not repeated for each window the frame appears in.

    Attributes:
        name: What it is and its version, for provenance.
        span: How many frames before and after the judged frame it
            compares with.
    """

    name: str
    span: int

    def prepare(self, image: np.ndarray) -> Any:
        """Computes whatever the detector needs from one frame.

        Args:
            image: (H, W, 3) uint8 RGB frame.

        Returns:
            Anything; it is handed back to `detect`.
        """
        ...

    def detect(self, before: Any, current: Any, after: Any) -> list[BallCandidate]:
        """Proposes candidates for the middle frame.

        Args:
            before: `prepare`'s result for the frame `span` earlier.
            current: The same for the frame being judged.
            after: The same for the frame `span` later.

        Returns:
            Candidates in full-frame pixels, best first.
        """
        ...


def row_scale(calibration: Calibration, config: BallConfig | None = None) -> np.ndarray:
    """Gives pixels-per-metre on the ground for every image row.

    Measured sideways across the court's centre line. Rows above the far
    end of the search region take the far end's value, rows below the near
    end the near end's - so this is the scale of the *ground* at that row;
    a ball in the air there is nearer the camera and looks bigger.

    Args:
        calibration: The court calibration.
        config: Supplies how far beyond the baselines to go.

    Returns:
        (image height,) array of pixels per metre.
    """
    config = config or BallConfig()
    tall = calibration.image_size[1]
    depth = np.linspace(-config.court_margin_m, COURT_LENGTH_M + config.court_margin_m, 80)
    left = calibration.to_image(np.stack([np.full_like(depth, -0.5), depth], axis=1))
    right = calibration.to_image(np.stack([np.full_like(depth, 0.5), depth], axis=1))
    scale = np.linalg.norm(right - left, axis=1)
    rows = (left[:, 1] + right[:, 1]) / 2
    fine = np.isfinite(scale) & np.isfinite(rows) & (scale > 0)
    # Keep only the stretch where going down the court goes up the image:
    # past the horizon the mapping folds back and means nothing.
    keep = fine & (np.concatenate([[True], np.diff(rows) < 0]))
    if keep.sum() < 2:
        return np.full(tall, float(np.nanmedian(scale[fine])) if fine.any() else 1.0)
    order = np.argsort(rows[keep])
    return np.interp(np.arange(tall), rows[keep][order], scale[keep][order])


class FlightVolume:
    """The box of space a ball in play can be in, as this camera sees it.

    The box is the court plus `side_margin_m` beside it and
    `court_margin_m` behind each baseline, up to `flight_height_m`. Given
    the full camera (recovered from the homography) every pixel is a line
    of sight, and that line is inside the box for some stretch of depth -
    or never. This is what turns "is this blob ball-sized?" into a
    question with an answer: a ball seen through a pixel must be at one of
    those depths, so it must look between two known sizes. A ball on the
    neighbouring court is seen along lines that cross this court only
    close to the camera, where a ball would look far bigger than it does.

    Attributes:
        ok: False when the camera could not be recovered; every query
            then answers "anything goes".
        focal: The focal length in pixels (0 when not `ok`).
    """

    def __init__(self, calibration: Calibration | None, config: BallConfig | None = None) -> None:
        """Recovers the camera and lays out the box.

        Args:
            calibration: The court calibration, or None for no volume.
            config: Supplies the margins, the height and the ball's size.
        """
        self._config = config or BallConfig()
        self.ok = False
        self.focal = 0.0
        if calibration is None:
            return
        self._size = calibration.image_size
        camera = camera_from_homography(calibration.homography, calibration.image_size)
        if camera is None:
            return
        camera = camera / np.linalg.norm(camera[2, :3])
        if camera[2] @ np.array([0.0, NET_Y_M, 0.0, 1.0]) < 0:
            camera = -camera
        matrix = camera[:, :3]
        self._camera = camera
        self._inverse = np.linalg.inv(matrix)
        self._centre = -self._inverse @ camera[:, 3]
        self.focal = float(np.linalg.norm(matrix[0] - (matrix[0] @ matrix[2]) * matrix[2]))
        across = DOUBLES_WIDTH_M / 2 + self._config.side_margin_m
        back = self._config.court_margin_m
        self._low = np.array([-across, -back, 0.0])
        self._high = np.array([across, COURT_LENGTH_M + back, self._config.flight_height_m])
        self.ok = True

    def depth_span(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Finds the depths at which each pixel's line of sight is inside the box.

        Args:
            points: (N, 2) image positions.

        Returns:
            (nearest, farthest) depth in metres along the camera's axis,
            each (N,). Both NaN where the line never enters the box.
            (0.5, inf) everywhere when there is no camera.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if not self.ok:
            return np.full(len(points), 0.5), np.full(len(points), np.inf)
        rays = np.hstack([points, np.ones((len(points), 1))]) @ self._inverse.T
        with np.errstate(divide="ignore", invalid="ignore"):
            first = (self._low - self._centre) / rays
            second = (self._high - self._centre) / rays
        enter = np.nanmax(np.minimum(first, second), axis=1)
        leave = np.nanmin(np.maximum(first, second), axis=1)
        enter = np.maximum(enter, 0.5)
        missed = ~(leave > enter)
        return np.where(missed, np.nan, enter), np.where(missed, np.nan, leave)

    def size_fit(self, points: np.ndarray, extents: np.ndarray) -> np.ndarray:
        """Scores how well each blob's size suits a ball inside the box.

        A blob may be up to `size_slack` times longer than the ball
        (motion blur), and is never much smaller than `min_blob_ratio`
        times the ball: measured on real footage, a moving-pixel blob is
        two to three times the ball's true diameter at any distance. The
        lower bound is what removes a neighbouring court's ball: its line
        of sight crosses this court's space only close to the camera,
        where this court's ball would make a blob twice the size. It is
        relaxed to "no smaller than the ball" where the ball itself would
        be under `resolved_ball_px` wide: a far-court ball two pixels
        across leaves a blob of whatever pixels cleared the threshold,
        and its size says nothing.

        Args:
            points: (N, 2) blob centres.
            extents: (N,) blob longest sides, pixels.

        Returns:
            (N,) scores: 1 where a ball somewhere along that line of sight
            inside the box would look like this, falling to 0 as the size
            is off by `size_tolerance`, and 0 where the line misses the box.
        """
        extents = np.asarray(extents, dtype=float)
        if not self.ok:
            return np.ones(len(extents))
        config = self._config
        near, far = self.depth_span(points)
        ball = self.focal * config.ball_diameter_m
        with np.errstate(invalid="ignore", divide="ignore"):
            too_big = (extents / config.size_slack) / (ball / near)
            least = ball / far
            resolved = np.clip(2.0 * least / config.resolved_ball_px - 1.0, 0.0, 1.0)
            ratio = 1.0 + (config.min_blob_ratio - 1.0) * resolved
            too_small = ratio * least / extents
        off = np.maximum(np.maximum(too_big, too_small), 1.0)
        room = max(config.size_tolerance - 1.0, 1e-6)
        fit = np.clip((config.size_tolerance - off) / room, 0.0, 1.0)
        return np.where(np.isnan(near), 0.0, fit)

    def region(self, step: int = 8) -> np.ndarray:
        """Marks the pixels whose line of sight enters the box at all.

        Args:
            step: Grid spacing, pixels, the test is made at.

        Returns:
            (H, W) boolean mask; all True when there is no camera.
        """
        wide, tall = self._size if self.ok else (1, 1)
        if not self.ok:
            return np.ones((tall, wide), dtype=bool)
        columns = np.arange(step / 2, wide, step)
        rows = np.arange(step / 2, tall, step)
        grid = np.stack(np.meshgrid(columns, rows), axis=-1).reshape(-1, 2)
        near, _ = self.depth_span(grid)
        coarse = (~np.isnan(near)).reshape(len(rows), len(columns)).astype(np.uint8)
        coarse = cv2.dilate(coarse, np.ones((3, 3), np.uint8))
        return cv2.resize(coarse, (wide, tall), interpolation=cv2.INTER_NEAREST).astype(bool)


def flight_region(calibration: Calibration, config: BallConfig | None = None) -> np.ndarray:
    """Marks where in the frame a ball in play over this court can appear.

    See `FlightVolume`. From a low camera close behind the baseline this
    is nearly the whole frame - a ball just in front of the lens can be
    anywhere - which is why size, not this mask, is what tells this
    court's ball from a neighbour's.

    Args:
        calibration: The court calibration.
        config: Supplies the margins and the height.

    Returns:
        (H, W) boolean mask; all True when the camera cannot be recovered.
    """
    volume = FlightVolume(calibration, config)
    if not volume.ok:
        wide, tall = calibration.image_size
        return np.ones((tall, wide), dtype=bool)
    return volume.region()


def _bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding rows and columns (top, bottom, left, right) of a mask's True area."""
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        return 0, mask.shape[0], 0, mask.shape[1]
    return int(rows[0]), int(rows[-1]) + 1, int(columns[0]), int(columns[-1]) + 1


def yellowness(image: np.ndarray) -> np.ndarray:
    """Measures how tennis-ball-coloured each pixel is.

    Args:
        image: (H, W, 3) uint8 RGB.

    Returns:
        (H, W) int16 of min(R, G) - B: about +100 for a lit ball, strongly
        negative for blue court and sky, near zero for white lines and
        grey.
    """
    red, green, blue = cv2.split(np.ascontiguousarray(image))
    return cv2.subtract(cv2.min(red, green), blue, dtype=cv2.CV_16S)


class MotionBallDetector:
    """Track A: moving, ball-coloured, ball-sized blobs.

    Stateful - it keeps a map of pixels that are always moving - so one
    instance should be fed one stream, in order.

    Attributes:
        name: "motion-ball-v1".
        span: Frames between the judged frame and each comparison frame.
    """

    name = "motion-ball-v1"

    def __init__(
        self,
        calibration: Calibration | None = None,
        config: BallConfig | None = None,
        frame_rate: float | None = None,
    ) -> None:
        """Initializes the detector.

        Args:
            calibration: When given, candidates are sized against
                `FlightVolume` and lines of sight that miss it are not
                searched. Without it the whole frame is searched and
                only a crude size limit applies.
            config: Thresholds; see `BallConfig`.
            frame_rate: Frames per second of the stream it will be fed.
                When given, `span` is `difference_seconds` at that rate;
                otherwise it is `difference_frames`.
        """
        self._config = config or BallConfig()
        self._calibration = calibration
        if frame_rate is not None and frame_rate > 0:
            self.span = max(1, round(self._config.difference_seconds * frame_rate))
        else:
            self.span = max(1, self._config.difference_frames)
        self._volume = FlightVolume(None, self._config)
        self._region: np.ndarray | None = None
        self._window = (0, 0, 0, 0)
        self._width = 0
        self._clutter: np.ndarray | None = None

    def _setup(self, shape: tuple[int, int]) -> None:
        """Builds the search region for this frame size."""
        calibration = self._calibration
        fits = calibration is not None and calibration.image_size == (shape[1], shape[0])
        if calibration is not None and not fits:
            logger.warning(
                "Calibration is for %sx%s but frames are %sx%s; searching the whole frame.",
                *calibration.image_size,
                shape[1],
                shape[0],
            )
        self._volume = FlightVolume(calibration if fits else None, self._config)
        region = self._volume.region() if self._volume.ok else np.ones(shape, dtype=bool)
        self._width = shape[1]
        self._window = _bounds(region)
        top, bottom, left, right = self._window
        self._region = np.ascontiguousarray(region[top:bottom, left:right]).astype(np.uint8)
        block = max(1, self._config.clutter_block)
        tall, wide = self._region.shape
        self._clutter = np.zeros((max(1, tall // block), max(1, wide // block)), dtype=np.float32)

    def prepare(self, image: np.ndarray) -> np.ndarray:
        """Computes the yellowness of the search window of one frame.

        Args:
            image: (H, W, 3) uint8 RGB frame.

        Returns:
            int16 yellowness of the cropped search window.
        """
        if self._region is None:
            self._setup(image.shape[:2])
        top, bottom, left, right = self._window
        return yellowness(image[top:bottom, left:right])

    def _quiet(self, moving: np.ndarray) -> np.ndarray:
        """Updates the always-moving map and returns `moving` without clutter."""
        config = self._config
        small = self._clutter.shape[::-1]
        busy = cv2.resize(moving, small, interpolation=cv2.INTER_AREA) > 0
        self._clutter *= config.clutter_decay
        self._clutter += busy
        noisy = (self._clutter > config.clutter_limit).astype(np.uint8)
        if not noisy.any():
            return moving
        full = cv2.resize(noisy, moving.shape[::-1], interpolation=cv2.INTER_NEAREST)
        return cv2.bitwise_and(moving, cv2.bitwise_not(full * 255))

    def detect(
        self, before: np.ndarray, current: np.ndarray, after: np.ndarray
    ) -> list[BallCandidate]:
        """Proposes candidates for the middle frame.

        Args:
            before: Yellowness of the frame `span` earlier.
            current: Yellowness of the frame being judged.
            after: Yellowness of the frame `span` later.

        Returns:
            Up to `max_candidates` candidates in full-frame pixels, best
            first.
        """
        config = self._config
        rise = cv2.min(cv2.subtract(current, before), cv2.subtract(current, after))
        moving = cv2.compare(rise, config.change_threshold, cv2.CMP_GT)
        moving = cv2.bitwise_and(self._quiet(moving), moving, mask=self._region)
        count, labels, stats, centres = cv2.connectedComponentsWithStats(moving, connectivity=8)
        if count <= 1:
            return []
        top, _, left, _ = self._window
        areas = stats[1:, cv2.CC_STAT_AREA]
        extents = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
        keep = np.flatnonzero(
            (areas >= config.min_blob_px) & (extents <= config.max_blob * self._width)
        )
        if not len(keep):
            return []
        spots = centres[keep + 1] + (left, top)
        sizes = self._volume.size_fit(spots, extents[keep])
        order = np.argsort(-sizes, kind="stable")[: 4 * config.max_candidates]
        found = []
        for index, spot, size in zip(keep[order], spots[order], sizes[order]):
            if size <= 0:
                break
            label = index + 1
            x0, y0 = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP]
            x1 = x0 + stats[label, cv2.CC_STAT_WIDTH]
            y1 = y0 + stats[label, cv2.CC_STAT_HEIGHT]
            inside = labels[y0:y1, x0:x1] == label
            colour = float(np.clip(current[y0:y1, x0:x1][inside].max() / 80.0, 0.0, 1.0))
            change = float(rise[y0:y1, x0:x1][inside].mean())
            contrast = float(np.clip(change / (4.0 * config.change_threshold), 0.0, 1.0))
            look = (config.colour_weight * colour + config.contrast_weight * contrast) / (
                config.colour_weight + config.contrast_weight
            )
            found.append(
                BallCandidate(
                    (float(spot[0]), float(spot[1])),
                    float(np.clip(size * look, 0.0, 1.0)),
                    float(extents[index]),
                )
            )
        found.sort(key=lambda candidate: -candidate.score)
        return found[: config.max_candidates]


@runtime_checkable
class HeatmapModel(Protocol):
    """A learned model that maps three frames to a ball heatmap."""

    def predict(self, frames: np.ndarray) -> np.ndarray:
        """Scores every pixel of the middle frame.

        Args:
            frames: (3, H, W, 3) uint8 RGB: earlier, judged, later.

        Returns:
            (H, W) float array in 0..1, high where the ball is.
        """
        ...


class HeatmapBallDetector:
    """Track B: candidates from a learned temporal heatmap model.

    The plumbing only. It needs a trained model, and none exists yet -
    `tracking.models.load_ball_heatmap` raises `WeightsMissingError`.

    Attributes:
        name: The model's provenance name.
        span: Frames between the judged frame and each comparison frame.
    """

    def __init__(
        self,
        model: HeatmapModel,
        config: BallConfig | None = None,
        name: str = "ball-heatmap",
        span: int = 1,
    ) -> None:
        """Initializes the detector.

        Args:
            model: The loaded model.
            config: Supplies the peak threshold and the candidate limit.
            name: Provenance name, e.g. `LoadedModel.source`.
            span: Frame spacing the model was trained with.
        """
        self._model = model
        self._config = config or BallConfig()
        self.name = name
        self.span = span

    def prepare(self, image: np.ndarray) -> np.ndarray:
        """Passes the frame through; the model takes raw frames.

        Args:
            image: (H, W, 3) uint8 RGB frame.

        Returns:
            The same frame.
        """
        return image

    def detect(
        self, before: np.ndarray, current: np.ndarray, after: np.ndarray
    ) -> list[BallCandidate]:
        """Turns the model's heatmap for the middle frame into candidates.

        Args:
            before: The frame `span` earlier.
            current: The frame being judged.
            after: The frame `span` later.

        Returns:
            One candidate per heatmap peak above `heatmap_threshold`, at
            the peak's weighted centre, scored by its height.
        """
        config = self._config
        heat = np.asarray(self._model.predict(np.stack([before, current, after])), dtype=float)
        hot = (heat >= config.heatmap_threshold).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
        found = []
        for label in range(1, count):
            rows, columns = np.nonzero(labels == label)
            weights = heat[rows, columns]
            centre = (
                float((columns * weights).sum() / weights.sum()),
                float((rows * weights).sum() / weights.sum()),
            )
            extent = float(max(stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]))
            found.append(BallCandidate(centre, float(np.clip(weights.max(), 0.0, 1.0)), extent))
        found.sort(key=lambda candidate: -candidate.score)
        return found[: config.max_candidates]


class CandidateStream:
    """Feeds frames to a `BallDetector` one at a time, holding only its window.

    `push` a frame, get back the candidates of whichever earlier frame can
    now be judged; `finish` flushes the frames at the end that never get a
    later frame to compare with.
    """

    def __init__(self, detector: BallDetector) -> None:
        """Initializes the stream.

        Args:
            detector: The detector to feed.
        """
        self._detector = detector
        self._span = max(1, int(detector.span))
        self._held: deque[tuple[int, float, Any]] = deque()
        self._fed = 0
        self._judged = 0

    def push(self, frame: VideoFrame) -> list[FrameCandidates]:
        """Takes the next frame.

        Args:
            frame: The frame, in display order.

        Returns:
            The candidates for at most one earlier frame (none until the
            window is full). Frames come out in the order they went in.
        """
        number = frame.index if frame.index is not None else self._fed
        self._fed += 1
        self._held.append((number, frame.pts, self._detector.prepare(frame.image)))
        span = self._span
        out = []
        if len(self._held) <= span:
            return out
        position = len(self._held) - 1 - span
        number, pts, current = self._held[position]
        if self._judged < span:
            # The first frames have nothing earlier to compare with.
            out.append(FrameCandidates(number, pts, (), searched=False))
        else:
            found = self._detector.detect(self._held[0][2], current, self._held[-1][2])
            out.append(FrameCandidates(number, pts, tuple(found)))
        self._judged += 1
        if len(self._held) > 2 * span:
            self._held.popleft()
        return out

    def finish(self) -> list[FrameCandidates]:
        """Flushes the frames that never got a later frame.

        Returns:
            Unsearched entries for the last `span` frames (fewer if fewer
            frames were ever pushed).
        """
        left = self._fed - self._judged
        waiting = list(self._held)[-left:] if left else []
        self._held.clear()
        self._judged = self._fed
        return [FrameCandidates(number, pts, (), searched=False) for number, pts, _ in waiting]


def find_candidates(
    frames: Iterable[VideoFrame], detector: BallDetector
) -> Iterator[FrameCandidates]:
    """Streams ball candidates for every frame.

    Args:
        frames: Frames in display order. Only the detector's window is
            held at any time.
        detector: Proposes the candidates.

    Yields:
        One `FrameCandidates` per frame, in order.
    """
    stream = CandidateStream(detector)
    for frame in frames:
        yield from stream.push(frame)
    yield from stream.finish()


@dataclass(frozen=True)
class SceneStats:
    """What the court looked like: the conditions a model would have to match.

    Attributes:
        frames_sampled: Frames the summary was taken over.
        brightness: Mean luminance of the court surface, 0 to 255.
        contrast: Standard deviation of that luminance across the surface.
        court_rgb: Mean (R, G, B) of the court surface.
        court_yellowness: Mean `yellowness` of the court surface - what a
            ball has to stand out from.
    """

    frames_sampled: int
    brightness: float
    contrast: float
    court_rgb: tuple[float, float, float]
    court_yellowness: float

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field, the colour as a list.
        """
        return {
            "frames_sampled": self.frames_sampled,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "court_rgb": list(self.court_rgb),
            "court_yellowness": self.court_yellowness,
        }


class SceneSampler:
    """Accumulates `SceneStats` from occasional frames."""

    def __init__(self, calibration: Calibration | None = None, stride: int = 30) -> None:
        """Initializes the sampler.

        Args:
            calibration: When given, only the court surface is summarised;
                otherwise the whole frame.
            stride: Use every `stride`-th frame offered.
        """
        self._calibration = calibration
        self._stride = max(1, stride)
        self._offered = 0
        self._mask: np.ndarray | None = None
        self._sums = np.zeros(6)
        self._count = 0

    def _surface(self, shape: tuple[int, int]) -> np.ndarray:
        """Mask of the court surface for this frame size."""
        mask = np.zeros(shape, dtype=np.uint8)
        calibration = self._calibration
        if calibration is not None and calibration.image_size == (shape[1], shape[0]):
            half = DOUBLES_WIDTH_M / 2
            corners = calibration.to_image(
                [(-half, 0.0), (half, 0.0), (half, NET_Y_M), (-half, NET_Y_M)]
            )
            if np.isfinite(corners).all():
                cv2.fillPoly(mask, [np.round(np.clip(corners, -1e5, 1e5)).astype(np.int32)], 1)
        if not mask.any():
            mask[:] = 1
        return mask.astype(bool)

    def add(self, image: np.ndarray) -> None:
        """Offers a frame; every `stride`-th one is summarised.

        Args:
            image: (H, W, 3) uint8 RGB frame.
        """
        turn = self._offered % self._stride == 0
        self._offered += 1
        if not turn:
            return
        if self._mask is None:
            self._mask = self._surface(image.shape[:2])
        pixels = image[self._mask].astype(np.float64)
        luma = pixels @ np.array([0.299, 0.587, 0.114])
        yellow = np.minimum(pixels[:, 0], pixels[:, 1]) - pixels[:, 2]
        self._sums += [luma.mean(), luma.std(), *pixels.mean(axis=0), yellow.mean()]
        self._count += 1

    def result(self) -> SceneStats | None:
        """Returns the summary so far.

        Returns:
            The statistics, or None if no frame was sampled.
        """
        if not self._count:
            return None
        mean = self._sums / self._count
        return SceneStats(
            frames_sampled=self._count,
            brightness=float(mean[0]),
            contrast=float(mean[1]),
            court_rgb=(float(mean[2]), float(mean[3]), float(mean[4])),
            court_yellowness=float(mean[5]),
        )


@dataclass(frozen=True)
class BallObservation:
    """Where the ball was in one frame.

    Only frames with a position have an observation; see `BallTrack.gaps`
    for the rest.

    Attributes:
        frame: The frame's number.
        pts: Its timestamp, seconds.
        image_xy: The ball's centre in the image, pixels. The detection's
            own centre when measured; the smoothed filter estimate when
            bridged.
        ground_ray_xy: Where the line of sight through the ball meets the
            court plane, (x, y) metres. **Not the ball's position** unless
            the ball is on the ground: for a ball in the air it lies
            beyond the ball. None when that line never meets the ground
            in front of the camera (the ball is above the horizon) or
            there is no calibration.
        velocity: The smoothed image velocity, pixels per second.
        confidence: 0 to 1. For a measured observation, the detection's
            score scaled by how smooth its segment is; for a bridged one,
            half of that of its measured neighbours.
        visibility: `VISIBLE` or `BRIDGED`.
        provenance: `MEASURED` or `INTERPOLATED`, matching `visibility`.
        segment: Index into `BallTrack.segments`.
        extent: The detected blob's longest side in pixels; None when
            bridged.
        in_player_box: True when a measured detection lay inside a
            player's (padded) box - where a hand, a racket or a shoe is
            as likely as the ball. Always False without player tracks,
            and for bridged observations.
        court_xy: The ball's position on the court, (x, y) metres. Only
            meaningful where the ball is known to be on the ground, which
            this module never knows: `link_candidates` always leaves it
            None, and the events phase fills it (from `ground_ray_xy`) at
            a bounce.
    """

    frame: int
    pts: float
    image_xy: Point
    ground_ray_xy: Point | None
    velocity: Point
    confidence: float
    visibility: Visibility
    provenance: Provenance
    segment: int
    extent: float | None = None
    in_player_box: bool = False
    court_xy: Point | None = None

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field, enums as their values.
        """
        ray = self.ground_ray_xy
        return {
            "frame": self.frame,
            "pts": self.pts,
            "image_xy": list(self.image_xy),
            "ground_ray_xy": None if ray is None else list(ray),
            "court_xy": None if self.court_xy is None else list(self.court_xy),
            "velocity": list(self.velocity),
            "confidence": self.confidence,
            "visibility": self.visibility.value,
            "provenance": self.provenance.value,
            "segment": self.segment,
            "extent": self.extent,
            "in_player_box": self.in_player_box,
        }


@dataclass(frozen=True)
class BallSegment:
    """One stretch of smooth ball motion.

    Attributes:
        index: Its position in `BallTrack.segments`.
        start_frame: First frame (measured).
        end_frame: Last frame (measured).
        start_pts: Timestamp of the first frame.
        end_pts: Timestamp of the last.
        measured: Frames with a linked detection.
        bridged: Frames bridged inside it.
        score: The evidence it was chosen on (sum of detection scores,
            discounted by roughness).
        roughness: Mean distance between where the filter predicted each
            detection and where it was, as a fraction of the gate. 0 is
            perfectly smooth.
        in_player_box: Fraction of its measured detections that lay
            inside a player's (padded) box. A segment near 1.0 never left
            a player: a ball in the hand, or not the ball at all.
    """

    index: int
    start_frame: int
    end_frame: int
    start_pts: float
    end_pts: float
    measured: int
    bridged: int
    score: float
    roughness: float
    in_player_box: float = 0.0

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field.
        """
        return dict(self.__dict__)


@dataclass(frozen=True)
class BallGap:
    """A run of frames with no ball position.

    Attributes:
        start_frame: First frame of the run.
        end_frame: Last frame of the run.
        start_pts: Timestamp of the first.
        end_pts: Timestamp of the last.
        frames: How many frames it spans.
    """

    start_frame: int
    end_frame: int
    start_pts: float
    end_pts: float
    frames: int

    @property
    def seconds(self) -> float:
        """How long the silence lasted, first unknown frame to last."""
        return self.end_pts - self.start_pts

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field.
        """
        return dict(self.__dict__)


@dataclass(frozen=True)
class BallQuality:
    """How much of a stretch the ball was actually tracked for.

    Attributes:
        frames: Frames in the stretch.
        measured: Frames with a linked detection.
        bridged: Frames bridged across a short gap.
        unknown: Frames with no position.
        coverage: (measured + bridged) / frames.
        measured_fraction: measured / frames.
        mean_confidence: Mean confidence of the observations; 0 if none.
        segments: Segments touching the stretch.
        gap_histogram: Count of unknown runs by duration bucket. Every
            bucket is present, zero included.
        longest_gap_seconds: The longest unknown run.
        coherence: Fraction of consecutive measured steps (at most 0.1s
            apart) that move at a plausible ball speed - the Phase 0
            spike's measure. Linking enforces plausible steps, so this is
            near 1 by construction: it shows the track is a path, **not**
            that the path is the ball.
        weak: True when coverage or mean confidence is under the
            configured floor.
    """

    frames: int
    measured: int
    bridged: int
    unknown: int
    coverage: float
    measured_fraction: float
    mean_confidence: float
    segments: int
    gap_histogram: dict[str, int]
    longest_gap_seconds: float
    coherence: float
    weak: bool

    def notice(self) -> str | None:
        """Words the result for a person, when it is weak.

        Returns:
            A sentence saying how little was tracked, or None when the
            tracking was not weak.
        """
        if not self.weak:
            return None
        return (
            f"Ball tracking was weak here: a position for {self.coverage:.0%} of frames "
            f"(mean confidence {self.mean_confidence:.2f}). Shot and bounce data from this "
            "stretch is unreliable; a ball model trained on this camera would help."
        )

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field.
        """
        return dict(self.__dict__)


def trajectory_coherence(observations: Sequence[BallObservation], image_width: int) -> float:
    """Measures how path-like a set of ball positions is, as the spike did.

    The spike called a step plausible when the ball moved between 2 and
    180 pixels across consecutive 30fps frames of a 1920-wide picture.
    The same bounds, per second and per image width, are used here.

    Args:
        observations: Positions in time order; only measured ones count.
        image_width: Width of the frames, pixels.

    Returns:
        Plausible steps / steps, over consecutive measured observations at
        most 0.1s apart. 0.0 when there are no such pairs.
    """
    seen = [o for o in observations if o.provenance is Provenance.MEASURED]
    slow, fast = 2 * 30 / 1920 * image_width, 180 * 30 / 1920 * image_width
    steps = plausible = 0
    for first, second in zip(seen, seen[1:]):
        elapsed = second.pts - first.pts
        if elapsed <= 0 or elapsed > 0.1:
            continue
        steps += 1
        speed = float(np.hypot(*np.subtract(second.image_xy, first.image_xy))) / elapsed
        plausible += slow <= speed <= fast
    return plausible / steps if steps else 0.0


@dataclass(frozen=True)
class BallTrack:
    """The ball over a stretch of video.

    Attributes:
        observations: Every frame with a position, in time order.
        segments: The stretches of smooth motion they belong to.
        gaps: Every run of frames with no position, in time order.
        frame_numbers: The number of every frame processed.
        frame_pts: The timestamp of every frame processed - the
            denominator for coverage.
        image_size: (width, height) of the frames.
        source: The detector that proposed the candidates.
        scene: Brightness and colour summary, when one was collected.
        weak_coverage: Coverage under which a stretch is called weak.
        weak_confidence: Mean confidence under which it is.
    """

    observations: tuple[BallObservation, ...]
    segments: tuple[BallSegment, ...]
    gaps: tuple[BallGap, ...]
    frame_numbers: np.ndarray
    frame_pts: np.ndarray
    image_size: tuple[int, int]
    source: str
    scene: SceneStats | None = None
    weak_coverage: float = BallConfig.weak_coverage
    weak_confidence: float = BallConfig.weak_confidence
    _times: list[float] = field(default_factory=list, repr=False, compare=False)

    def at(self, pts: float, tolerance: float = 1e-3) -> BallObservation | None:
        """Gives the observation at a moment, if there is one.

        Args:
            pts: The moment, seconds.
            tolerance: How far off an observation's timestamp may be.

        Returns:
            The observation of the frame at that moment, or None when the
            ball's position there is unknown.
        """
        if not self._times:
            self._times.extend(observation.pts for observation in self.observations)
        times = self._times
        index = bisect.bisect_left(times, pts - tolerance)
        if index < len(times) and abs(times[index] - pts) <= tolerance:
            return self.observations[index]
        return None

    def between(self, start: float, end: float) -> tuple[BallObservation, ...]:
        """Gives the observations in a stretch of time.

        Args:
            start: First moment, seconds (inclusive).
            end: Last moment, seconds (inclusive).

        Returns:
            The observations with `start <= pts <= end`, in order.
        """
        return tuple(o for o in self.observations if start <= o.pts <= end)

    def segment(self, index: int) -> tuple[BallObservation, ...]:
        """Gives one segment's observations.

        Args:
            index: The segment's index.

        Returns:
            Its observations in time order.
        """
        return tuple(o for o in self.observations if o.segment == index)

    def quality(self, start: float | None = None, end: float | None = None) -> BallQuality:
        """Summarises tracking quality over the whole track or part of it.

        Coverage over a whole recording is low for an honest reason - most
        of a match is not rally - so the events phase should ask about the
        stretches it believes are rallies.

        Args:
            start: First moment of the stretch; the beginning if omitted.
            end: Last moment; the end if omitted.

        Returns:
            Coverage, gaps, confidence and whether that counts as weak.
        """
        low = -np.inf if start is None else start
        high = np.inf if end is None else end
        inside = (self.frame_pts >= low) & (self.frame_pts <= high)
        frames = int(inside.sum())
        seen = [o for o in self.observations if low <= o.pts <= high]
        counts = provenance_counts(o.provenance for o in seen)
        measured, bridged = counts["measured"], counts["interpolated"]
        histogram = {label: 0 for _, label in _GAP_BUCKETS} | {_LONGEST_BUCKET: 0}
        longest = 0.0
        step = float(np.median(np.diff(self.frame_pts))) if len(self.frame_pts) > 1 else 0.0
        for gap in self.gaps:
            if gap.end_pts < low or gap.start_pts > high:
                continue
            seconds = min(gap.end_pts, high) - max(gap.start_pts, low) + step
            longest = max(longest, seconds)
            label = next((n for limit, n in _GAP_BUCKETS if seconds < limit), _LONGEST_BUCKET)
            histogram[label] += 1
        coverage = (measured + bridged) / frames if frames else 0.0
        confidence = float(np.mean([o.confidence for o in seen])) if seen else 0.0
        return BallQuality(
            frames=frames,
            measured=measured,
            bridged=bridged,
            unknown=frames - measured - bridged,
            coverage=coverage,
            measured_fraction=measured / frames if frames else 0.0,
            mean_confidence=confidence,
            segments=len({o.segment for o in seen}),
            gap_histogram=histogram,
            longest_gap_seconds=longest,
            coherence=trajectory_coherence(seen, self.image_size[0]),
            weak=coverage < self.weak_coverage or confidence < self.weak_confidence,
        )

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Observations, segments, gaps, the quality summary and the
            scene summary.
        """
        return {
            "source": self.source,
            "image_size": list(self.image_size),
            "frames": int(len(self.frame_pts)),
            "observations": [observation.to_dict() for observation in self.observations],
            "segments": [segment.to_dict() for segment in self.segments],
            "gaps": [gap.to_dict() for gap in self.gaps],
            "quality": self.quality().to_dict(),
            "scene": None if self.scene is None else self.scene.to_dict(),
        }


class _Filter:
    """A constant-velocity Kalman filter over image position (internal).

    State is (x, y, vx, vy). Time steps are real timestamp differences, so
    a dropped frame is a longer step, not a wrong one.
    """

    def __init__(self, start: np.ndarray, velocity: np.ndarray, accel: float, jitter: float):
        """Starts the filter at a position with a velocity.

        Args:
            start: (2,) position in pixels.
            velocity: (2,) pixels per second.
            accel: Process noise: pixels per second squared.
            jitter: Measurement noise: pixels.
        """
        self.state = np.concatenate([start, velocity]).astype(float)
        self.spread = np.diag([jitter**2, jitter**2, (accel * 0.05) ** 2, (accel * 0.05) ** 2])
        self._accel = accel
        self._noise = np.eye(2) * jitter**2

    def predict(self, elapsed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Works out where the state will be, without moving the filter.

        Args:
            elapsed: Seconds ahead.

        Returns:
            (transition matrix, predicted state, predicted covariance).
        """
        move = np.eye(4)
        move[0, 2] = move[1, 3] = elapsed
        half = 0.5 * elapsed**2
        push = np.array([[half, 0], [0, half], [elapsed, 0], [0, elapsed]])
        state = move @ self.state
        spread = move @ self.spread @ move.T + push @ push.T * self._accel**2
        return move, state, spread

    def correct(self, seen: np.ndarray) -> None:
        """Pulls the state toward a detection.

        Args:
            seen: (2,) detected position in pixels.
        """
        gain = self.spread[:, :2] @ np.linalg.inv(self.spread[:2, :2] + self._noise)
        self.state = self.state + gain @ (seen - self.state[:2])
        self.spread = self.spread - gain @ self.spread[:2, :]


@dataclass
class _Path:
    """A candidate track through the frames (internal).

    Attributes:
        nodes: (frame position, candidate index) of each linked detection.
        errors: Per link, prediction error as a fraction of the gate.
        value: The evidence score it competes on.
        roughness: Mean of `errors`.
    """

    nodes: list[tuple[int, int]]
    errors: list[float]
    value: float = 0.0
    roughness: float = 0.0


class _Linker:
    """Builds candidate paths and picks the best consistent set (internal)."""

    def __init__(
        self,
        frames: Sequence[FrameCandidates],
        image_size: tuple[int, int],
        calibration: Calibration | None,
        players: PlayerTracks | None,
        config: BallConfig,
    ) -> None:
        """Lays the candidates out as arrays and works out the motion limits."""
        self.config = config
        self.frames = frames
        self.width = image_size[0]
        self.pts = np.array([frame.pts for frame in frames], dtype=float)
        steps = np.diff(self.pts)
        self.frame_step = float(np.median(steps)) if len(steps) else 1.0 / 30
        self.reach = config.max_gap_seconds + 1.5 * self.frame_step
        self.scale = row_scale(calibration, config) if calibration is not None else None
        if self.scale is not None:
            # A ball above the net line's image row may be anywhere in front
            # of the net, so the limit there is never tighter than at the net.
            at_net = float(calibration.to_image(np.array([0.0, NET_Y_M]))[1])
            row = round(at_net) if np.isfinite(at_net) else len(self.scale) // 2
            self.scale_floor = float(self.scale[int(np.clip(row, 0, len(self.scale) - 1))])
        self.points: list[np.ndarray] = []
        self.scores: list[np.ndarray] = []
        self.extents: list[np.ndarray] = []
        self.boxed: list[np.ndarray] = []
        for frame in frames:
            boxes = players.boxes_at(frame.pts) if players is not None else []
            kept = [(c, self._inside(c, boxes)) for c in frame.candidates]
            if config.player_box_weight <= 0:
                kept = [(c, inside) for c, inside in kept if not inside]
            self.boxed.append(np.array([inside for _, inside in kept], dtype=bool))
            kept = [(c, config.player_box_weight if inside else 1.0) for c, inside in kept]
            self.points.append(np.array([c.xy for c, _ in kept], dtype=float).reshape(-1, 2))
            self.scores.append(np.array([c.score * weight for c, weight in kept], dtype=float))
            self.extents.append(np.array([c.extent for c, _ in kept], dtype=float))
        self.limits = [self.speed_limit(points) for points in self.points]
        self.used: dict[tuple[int, int, int, int], tuple[_Path, int]] = {}

    def _inside(self, candidate: BallCandidate, boxes: Sequence[Sequence[float]]) -> bool:
        """Says whether a candidate lies inside a (padded) player box."""
        pad = self.config.player_box_pad
        x, y = candidate.xy
        for x0, y0, x1, y1 in boxes:
            # Padded by a share of the player's height: an arm and a racket
            # reach about that far beyond the body on every side.
            grow = (y1 - y0) * pad
            if x0 - grow <= x <= x1 + grow and y0 - grow <= y <= y1 + grow:
                return True
        return False

    def speed_limit(self, points: np.ndarray) -> np.ndarray:
        """Fastest image speed, pixels per second, a ball at each point may have."""
        config = self.config
        if self.scale is None:
            return np.full(len(points), config.max_step * self.width)
        rows = np.clip(np.round(points[:, 1]).astype(int), 0, len(self.scale) - 1)
        scale = np.maximum(self.scale[rows], self.scale_floor)
        return config.max_speed_mps * config.speed_slack * scale

    def seeds(self, first: int) -> list[tuple[int, int, int, int, int, int]]:
        """Finds triples of mutually consistent detections starting at a frame.

        Returns:
            (frame a, candidate a, frame b, candidate b, frame c,
            candidate c) for every triple whose third point lies where the
            first two predict, that breaks no speed limit, and whose
            second link is not already part of a path.
        """
        config = self.config
        start = self.points[first]
        if not len(start):
            return []
        found = []
        for second in range(first + 1, min(first + 3, len(self.frames))):
            middle = self.points[second]
            gap_one = self.pts[second] - self.pts[first]
            if not len(middle) or gap_one <= 0 or gap_one > self.reach:
                continue
            hop = middle[None, :, :] - start[:, None, :]
            speed = np.linalg.norm(hop, axis=2) / gap_one
            allowed = (speed >= config.min_step * self.width) & (
                speed <= np.maximum(self.limits[first][:, None], self.limits[second])
            )
            if not allowed.any():
                continue
            for third in range(second + 1, min(second + 3, len(self.frames))):
                end = self.points[third]
                gap_two = self.pts[third] - self.pts[second]
                if not len(end) or gap_two <= 0 or gap_two > self.reach:
                    continue
                guess = middle[None, :, :] + hop * (gap_two / gap_one)
                miss = np.linalg.norm(end[None, None, :, :] - guess[:, :, None, :], axis=3)
                reach = config.gate * self.width + config.gate_speed_factor * (
                    np.linalg.norm(hop, axis=2) * (gap_two / gap_one)
                )
                follow = np.linalg.norm(end[None, :, :] - middle[:, None, :], axis=2) / gap_two
                legal = follow <= self.limits[second][:, None]
                good = allowed[:, :, None] & (miss <= reach[:, :, None]) & legal[None, :, :]
                for a, b, c in zip(*np.nonzero(good)):
                    if (second, int(b), third, int(c)) not in self.used:
                        found.append((first, int(a), second, int(b), third, int(c)))
        return found

    def grow(self, seed: tuple[int, int, int, int, int, int]) -> _Path:
        """Extends a seed forward for as long as detections keep fitting.

        Prediction here is a plain constant-velocity follower with fixed
        gains - thousands of these are grown and most are thrown away, so
        it must be cheap. The paths that survive are filtered properly
        (`_smooth`). A path that runs into a link another path already
        made takes over that path's remainder instead of recomputing it.
        """
        config = self.config
        first, a, second, b, third, c = seed
        spot = self.points[second][b]
        velocity = (spot - self.points[first][a]) / (self.pts[second] - self.pts[first])
        path = _Path(nodes=[(first, a), (second, b)], errors=[])
        last, forced = second, (third, c)
        position = second + 1
        base = config.gate * self.width
        while position < len(self.frames):
            waited = self.pts[position] - self.pts[last]
            if waited > self.reach:
                break
            points = self.points[position]
            if not len(points):
                position += 1
                continue
            guess = spot + velocity * waited
            radius = base + config.gate_speed_factor * float(np.hypot(*velocity)) * waited
            offset = points - guess
            miss = np.hypot(offset[:, 0], offset[:, 1])
            came_from = self.points[last][path.nodes[-1][1]]
            hop = points - came_from
            fits = (miss <= radius) & (
                np.hypot(hop[:, 0], hop[:, 1]) <= self.limits[position] * waited
            )
            if forced is not None and forced[0] == position:
                choice = forced[1]
            elif fits.any():
                cost = np.where(fits, miss / radius - 0.5 * self.scores[position], np.inf)
                choice = int(np.argmin(cost))
            else:
                position += 1
                continue
            forced = None
            owner = self.used.get((last, path.nodes[-1][1], position, choice))
            if owner is not None:
                other, index = owner
                path.nodes.extend(other.nodes[index:])
                path.errors.extend(other.errors[max(index - 2, 0) :])
                break
            spot = guess + config.follow_gain * offset[choice]
            velocity = velocity + config.velocity_gain * offset[choice] / waited
            path.errors.append(float(min(miss[choice] / radius, 2.0)))
            path.nodes.append((position, choice))
            last = position
            position += 1
        for index, ((one, i), (two, j)) in enumerate(zip(path.nodes, path.nodes[1:])):
            self.used.setdefault((one, i, two, j), (path, index + 1))
        return path

    def judge(self, path: _Path) -> bool:
        """Scores a path and says whether it is worth keeping."""
        config = self.config
        if len(path.nodes) < config.min_segment_frames:
            return False
        points = np.array([self.points[position][index] for position, index in path.nodes])
        reach = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        if reach < config.min_travel * self.width:
            return False
        path.roughness = float(np.mean(path.errors)) if path.errors else 0.0
        if path.roughness > config.max_roughness:
            return False
        evidence = sum(float(self.scores[position][index]) for position, index in path.nodes)
        path.value = evidence * float(np.exp(-config.smoothness_weight * path.roughness))
        return path.value >= config.min_segment_score

    def paths(self) -> list[_Path]:
        """Builds every acceptable candidate path."""
        kept = []
        for first in range(len(self.frames)):
            for seed in self.seeds(first):
                if (seed[2], seed[3], seed[4], seed[5]) in self.used:
                    continue
                path = self.grow(seed)
                if self.judge(path):
                    kept.append(path)
        return kept


def _choose(paths: list[_Path]) -> list[_Path]:
    """Picks the non-overlapping set of paths with the greatest total value.

    Weighted interval scheduling: sort by end, and for each path either
    take it together with the best set ending before it starts, or skip it.
    """
    if not paths:
        return []
    ordered = sorted(paths, key=lambda path: path.nodes[-1][0])
    ends = [path.nodes[-1][0] for path in ordered]
    best = [0.0] * (len(ordered) + 1)
    taken: list[bool] = []
    before: list[int] = []
    for index, path in enumerate(ordered):
        previous = bisect.bisect_left(ends, path.nodes[0][0], 0, index)
        before.append(previous)
        with_it = best[previous] + path.value
        taken.append(with_it > best[index])
        best[index + 1] = max(best[index], with_it)
    chosen = []
    index = len(ordered)
    while index > 0:
        if taken[index - 1]:
            chosen.append(ordered[index - 1])
            index = before[index - 1]
        else:
            index -= 1
    return chosen[::-1]


def _smooth(linker: _Linker, path: _Path) -> tuple[np.ndarray, np.ndarray]:
    """Runs the filter forward and a smoother backward over one path.

    Returns:
        (frame positions covered, (N, 4) smoothed states) for every frame
        from the path's first to its last, detected or not.
    """
    config = linker.config
    seen = dict(path.nodes)
    first, last = path.nodes[0][0], path.nodes[-1][0]
    positions = np.arange(first, last + 1)
    start = linker.points[first][seen[first]]
    second = path.nodes[1][0]
    elapsed = linker.pts[second] - linker.pts[first]
    velocity = (linker.points[second][seen[second]] - start) / elapsed
    tracker = _Filter(
        start,
        velocity,
        config.process_noise * linker.width,
        config.measurement_noise * linker.width,
    )
    states, spreads = [tracker.state.copy()], [tracker.spread.copy()]
    moves, guesses, guess_spreads = [], [], []
    for position in positions[1:]:
        move, guess, guess_spread = tracker.predict(
            linker.pts[position] - linker.pts[position - 1]
        )
        tracker.state, tracker.spread = guess.copy(), guess_spread.copy()
        if position in seen:
            tracker.correct(linker.points[position][seen[position]])
        moves.append(move)
        guesses.append(guess)
        guess_spreads.append(guess_spread)
        states.append(tracker.state.copy())
        spreads.append(tracker.spread.copy())
    smooth = [None] * len(states)
    smooth[-1] = states[-1]
    for index in range(len(states) - 2, -1, -1):
        gain = spreads[index] @ moves[index].T @ np.linalg.inv(guess_spreads[index])
        smooth[index] = states[index] + gain @ (smooth[index + 1] - guesses[index])
    return positions, np.array(smooth)


def _on_ground(calibration: Calibration | None, xy: np.ndarray) -> Point | None:
    """Where the sightline through an image point meets the court, if it does.

    A pixel above the horizon still "maps" through the homography - to a
    point behind the camera. That is told apart by the sign of the
    projective depth, compared with a point known to be in front (the
    middle of the net).
    """
    if calibration is None:
        return None
    spot = calibration.to_court(xy)
    if not np.isfinite(spot).all():
        return None
    depth = calibration.homography[2] @ np.array([spot[0], spot[1], 1.0])
    ahead = calibration.homography[2] @ np.array([0.0, NET_Y_M, 1.0])
    if depth * ahead <= 0 or abs(spot[0]) > 60 or not -60 <= spot[1] <= 120:
        return None
    return (float(spot[0]), float(spot[1]))


def link_candidates(
    frames: Sequence[FrameCandidates],
    image_size: tuple[int, int],
    calibration: Calibration | None = None,
    players: PlayerTracks | None = None,
    config: BallConfig | None = None,
    source: str = "unknown",
    scene: SceneStats | None = None,
) -> BallTrack:
    """Finds the ball's track through per-frame candidates.

    Args:
        frames: Candidates for consecutive frames, in time order.
        image_size: (width, height) of those frames.
        calibration: The court calibration; sets the physical speed limit
            per image row and gives `ground_ray_xy`. Without it a crude
            image-width speed limit applies and `ground_ray_xy` is None.
        players: The players' tracks; candidates inside a player's box
            are down-weighted (or dropped) per `player_box_weight`.
        config: Linking thresholds; see `BallConfig`.
        source: The detector's name, recorded on the track.
        scene: A scene summary to attach.

    Returns:
        The track: observations where a position is claimed, segments,
        and explicit gaps everywhere else.
    """
    config = config or BallConfig()
    numbers = np.array([frame.frame for frame in frames], dtype=np.int64)
    linker = _Linker(frames, image_size, calibration, players, config)
    chosen = _choose(linker.paths()) if len(frames) else []
    observations: list[BallObservation] = []
    segments: list[BallSegment] = []
    known = np.zeros(len(frames), dtype=bool)
    for index, path in enumerate(chosen):
        positions, states = _smooth(linker, path)
        seen = dict(path.nodes)
        quality = float(np.exp(-config.smoothness_weight * path.roughness))
        scores = {p: float(linker.scores[p][i]) for p, i in path.nodes}
        measured_positions = sorted(seen)
        for position, state in zip(positions, states):
            position = int(position)
            known[position] = True
            if position in seen:
                xy = linker.points[position][seen[position]]
                confidence = float(np.clip(scores[position] * quality, 0.0, 1.0))
                visible, origin = Visibility.VISIBLE, Provenance.MEASURED
                extent = float(linker.extents[position][seen[position]])
                boxed = bool(linker.boxed[position][seen[position]])
            else:
                xy = state[:2]
                after = bisect.bisect_left(measured_positions, position)
                around = (measured_positions[after - 1], measured_positions[after])
                confidence = float(
                    np.clip(0.5 * quality * min(scores[around[0]], scores[around[1]]), 0.0, 1.0)
                )
                visible, origin = Visibility.BRIDGED, Provenance.INTERPOLATED
                extent, boxed = None, False
            observations.append(
                BallObservation(
                    frame=int(numbers[position]),
                    pts=float(linker.pts[position]),
                    image_xy=(float(xy[0]), float(xy[1])),
                    ground_ray_xy=_on_ground(calibration, np.asarray(xy, dtype=float)),
                    velocity=(float(state[2]), float(state[3])),
                    confidence=confidence,
                    visibility=visible,
                    provenance=origin,
                    segment=index,
                    extent=extent,
                    in_player_box=boxed,
                )
            )
        first, last = int(positions[0]), int(positions[-1])
        segments.append(
            BallSegment(
                index=index,
                start_frame=int(numbers[first]),
                end_frame=int(numbers[last]),
                start_pts=float(linker.pts[first]),
                end_pts=float(linker.pts[last]),
                measured=len(seen),
                bridged=len(positions) - len(seen),
                score=path.value,
                roughness=path.roughness,
                in_player_box=float(np.mean([linker.boxed[p][i] for p, i in path.nodes])),
            )
        )
    gaps = []
    position = 0
    while position < len(frames):
        if known[position]:
            position += 1
            continue
        end = position
        while end + 1 < len(frames) and not known[end + 1]:
            end += 1
        gaps.append(
            BallGap(
                start_frame=int(numbers[position]),
                end_frame=int(numbers[end]),
                start_pts=float(linker.pts[position]),
                end_pts=float(linker.pts[end]),
                frames=end - position + 1,
            )
        )
        position = end + 1
    logger.info(
        "Ball linked: %d segments, %d observations over %d frames, %d gaps",
        len(segments),
        len(observations),
        len(frames),
        len(gaps),
    )
    return BallTrack(
        observations=tuple(observations),
        segments=tuple(segments),
        gaps=tuple(gaps),
        frame_numbers=numbers,
        frame_pts=linker.pts,
        image_size=image_size,
        source=source,
        scene=scene,
        weak_coverage=config.weak_coverage,
        weak_confidence=config.weak_confidence,
    )


def track_ball(
    frames: Iterable[VideoFrame],
    calibration: Calibration | None = None,
    detector: BallDetector | None = None,
    players: PlayerTracks | None = None,
    config: BallConfig | None = None,
    frame_rate: float | None = None,
) -> BallTrack:
    """Tracks the ball through a stream of frames.

    Frames are consumed one at a time; only the detector's short window
    is ever held. The candidates (a few numbers per frame) are kept for
    the whole stretch and linked at the end.

    Args:
        frames: Frames in display order, at full size - a far-court ball
            is a few pixels wide and does not survive downscaling.
        calibration: The court calibration. Strongly recommended: it
            limits the search to the court and makes the speed limit
            physical.
        detector: Proposes candidates; `MotionBallDetector` if omitted.
        players: The players' tracks, to discount limbs and rackets.
        config: Detection and linking thresholds.
        frame_rate: The stream's frames per second (e.g.
            `VideoInfo.nominal_fps`), handed to the default detector so
            its comparison spacing suits this recording. Ignored when a
            `detector` is given.

    Returns:
        The ball's track, with gaps and quality.

    Raises:
        ValueError: If no frames were given and there is no calibration
            to take the image size from.
    """
    config = config or BallConfig()
    detector = detector or MotionBallDetector(calibration, config, frame_rate)
    stream = CandidateStream(detector)
    sampler = SceneSampler(calibration, config.scene_sample_stride)
    found: list[FrameCandidates] = []
    size = calibration.image_size if calibration is not None else None
    for frame in frames:
        if size is None:
            size = (frame.image.shape[1], frame.image.shape[0])
        sampler.add(frame.image)
        found.extend(stream.push(frame))
    found.extend(stream.finish())
    if size is None:
        raise ValueError("track_ball needs at least one frame or a calibration.")
    return link_candidates(
        found, size, calibration, players, config, source=detector.name, scene=sampler.result()
    )
