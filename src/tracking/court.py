"""The court: its regulation template, and finding its painted lines in a frame.

The court-plane coordinate frame is defined here, once
--------------------------------------------------------
Every court-plane coordinate the tracker produces - a bounce, a strike, a
player's feet - is in this frame, in **metres**:

* The origin is the **centre mark of the near baseline** ("near" meaning the
  end the camera stands behind).
* **+x runs to the camera's right** along the baseline. The singles
  sidelines are at x = -4.115 and +4.115, the doubles sidelines at -5.485
  and +5.485, and the centre service line is x = 0.
* **+y runs away from the camera**, down the court. The near baseline is
  y = 0, the near service line 5.485, the net 11.885, the far service line
  18.285 and the far baseline 23.77.

Relation to SwingVision's own frame (`RawShotRow.bounce_x/bounce_y`,
`hit_x/hit_y`): read off the one real export available (683 shots), its
values behave exactly like this frame - metres, y from about 0 at the near
baseline to about 23.8 at the far one (a far-end hitter's `Hit (y)` medians
24.6, a near-end hitter's about 0), x centred on the centre line (range
-6.2..7.2) and positive toward the camera's right (a near-end player in the
deuce court has x > 0; a far-end player in *their* deuce court has x < 0).
SwingVision does not document its frame, so that agreement is an observation
on one export, not a guarantee; this module's definition above is the
contract, and it was not bent to fit.

Detection, and what it assumes
------------------------------
Classic OpenCV only. A frame becomes a mask of thin bright pixels, the mask
becomes Hough segments, segments merge into lines, and the lines split into
two families: *transverse* (baselines and service lines, close to horizontal)
and *longitudinal* (sidelines and the centre service line, converging on one
vanishing point). Every way of naming two lines from each family as two
template lines gives a candidate homography; each is scored by projecting the
**whole** template and measuring how much of it lands on line pixels, with a
penalty for template lines that land where nothing is painted. That penalty
is the guard against the classic failure - a tidy fit to wrongly-named lines.
The winner's lines are then refitted to the pixels and intersected into
named keypoints, which `calibrate.py` turns into the final homography.

Two further guards exist because of what the real recording showed. A net's
mesh is a field of bright dots, so support is counted against straight
*segments* of the right orientation (`evidence_maps`), never raw pixels. And
the net's white tape is itself a long straight near-horizontal band which,
from a fence-height camera, sits exactly where the far baseline is: the
camera is recovered from the homography (`camera_from_homography`), the tape
is projected (`net_tape_zone`), and its pixels are barred from being read as
a painted line. A line also must not look far thicker than paint would.

This assumes the camera stands **behind one baseline looking down the
court**, roughly upright: baselines within about 12 degrees of horizontal,
sidelines converging upward. A side-on or overhead view is not supported and
makes `detect_court` abstain rather than guess.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import CourtConfig

logger = logging.getLogger(__name__)

COURT_LENGTH_M = 23.77
SINGLES_WIDTH_M = 8.23
DOUBLES_WIDTH_M = 10.97
SERVICE_LINE_FROM_NET_M = 6.40
NET_Y_M = COURT_LENGTH_M / 2
NET_HEIGHT_CENTRE_M = 0.914
NET_HEIGHT_POST_M = 1.07

_HALF_SINGLES = SINGLES_WIDTH_M / 2
_HALF_DOUBLES = DOUBLES_WIDTH_M / 2
_NEAR_SERVICE_Y = NET_Y_M - SERVICE_LINE_FROM_NET_M
_FAR_SERVICE_Y = NET_Y_M + SERVICE_LINE_FROM_NET_M

TRANSVERSE = "transverse"
LONGITUDINAL = "longitudinal"

Point = tuple[float, float]


@dataclass(frozen=True)
class CourtLine:
    """One painted line of the court, in court-plane metres.

    Attributes:
        name: Its stable name, e.g. "near_baseline" or "left_singles".
        family: `TRANSVERSE` (runs across the court, constant y) or
            `LONGITUDINAL` (runs down the court, constant x).
        start: One end, (x, y) in metres.
        end: The other end.
    """

    name: str
    family: str
    start: Point
    end: Point

    @property
    def position(self) -> float:
        """The line's constant coordinate: y if transverse, x if longitudinal."""
        return self.start[1] if self.family == TRANSVERSE else self.start[0]


@dataclass(frozen=True)
class CourtTemplate:
    """A regulation court as named lines and named keypoints.

    Attributes:
        kind: "doubles" (a full court, alleys painted) or "singles" (no
            doubles sidelines; baselines stop at the singles sidelines).
        lines: Every painted line.
        keypoints: Named line intersections, "<transverse>:<longitudinal>"
            to (x, y) in metres - the points a calibration is fitted to,
            and the names the manual fallback accepts. The two
            "<baseline>:center" entries are the centre marks.
    """

    kind: str
    lines: tuple[CourtLine, ...]
    keypoints: dict[str, Point] = field(default_factory=dict)

    def line(self, name: str) -> CourtLine:
        """Looks a line up by name.

        Args:
            name: The line's name.

        Returns:
            The line.

        Raises:
            KeyError: If the template has no such line.
        """
        for line in self.lines:
            if line.name == name:
                return line
        raise KeyError(name)

    def family(self, family: str) -> tuple[CourtLine, ...]:
        """Lists one family's lines, ordered by position (near/left first).

        Args:
            family: `TRANSVERSE` or `LONGITUDINAL`.

        Returns:
            The lines of that family.
        """
        return tuple(
            sorted((ln for ln in self.lines if ln.family == family), key=lambda ln: ln.position)
        )

    @property
    def half_width(self) -> float:
        """Half the painted width: to the doubles or the singles sideline."""
        return _HALF_DOUBLES if self.kind == "doubles" else _HALF_SINGLES

    @property
    def corners(self) -> tuple[Point, Point, Point, Point]:
        """The outer corners: near-left, near-right, far-right, far-left."""
        half = self.half_width
        return ((-half, 0.0), (half, 0.0), (half, COURT_LENGTH_M), (-half, COURT_LENGTH_M))


def court_template(kind: str = "doubles") -> CourtTemplate:
    """Builds the regulation court template.

    Args:
        kind: "doubles" or "singles".

    Returns:
        The template, in the court frame this module's docstring defines.

    Raises:
        ValueError: If `kind` is neither.
    """
    if kind not in ("doubles", "singles"):
        raise ValueError(f"Unknown court template {kind!r}; expected 'doubles' or 'singles'.")
    half = _HALF_DOUBLES if kind == "doubles" else _HALF_SINGLES
    lines = [
        CourtLine("near_baseline", TRANSVERSE, (-half, 0.0), (half, 0.0)),
        CourtLine(
            "near_service",
            TRANSVERSE,
            (-_HALF_SINGLES, _NEAR_SERVICE_Y),
            (_HALF_SINGLES, _NEAR_SERVICE_Y),
        ),
        CourtLine(
            "far_service",
            TRANSVERSE,
            (-_HALF_SINGLES, _FAR_SERVICE_Y),
            (_HALF_SINGLES, _FAR_SERVICE_Y),
        ),
        CourtLine("far_baseline", TRANSVERSE, (-half, COURT_LENGTH_M), (half, COURT_LENGTH_M)),
        CourtLine(
            "left_singles", LONGITUDINAL, (-_HALF_SINGLES, 0.0), (-_HALF_SINGLES, COURT_LENGTH_M)
        ),
        CourtLine("center", LONGITUDINAL, (0.0, _NEAR_SERVICE_Y), (0.0, _FAR_SERVICE_Y)),
        CourtLine(
            "right_singles", LONGITUDINAL, (_HALF_SINGLES, 0.0), (_HALF_SINGLES, COURT_LENGTH_M)
        ),
    ]
    if kind == "doubles":
        lines.append(
            CourtLine(
                "left_doubles",
                LONGITUDINAL,
                (-_HALF_DOUBLES, 0.0),
                (-_HALF_DOUBLES, COURT_LENGTH_M),
            )
        )
        lines.append(
            CourtLine(
                "right_doubles",
                LONGITUDINAL,
                (_HALF_DOUBLES, 0.0),
                (_HALF_DOUBLES, COURT_LENGTH_M),
            )
        )
    keypoints: dict[str, Point] = {}
    for across in (ln for ln in lines if ln.family == TRANSVERSE):
        for along in sorted(
            (ln for ln in lines if ln.family == LONGITUDINAL), key=lambda ln: ln.position
        ):
            if "service" in across.name and "doubles" in along.name:
                continue  # the service line stops at the singles sideline
            keypoints[f"{across.name}:{along.name}"] = (along.position, across.position)
    return CourtTemplate(kind=kind, lines=tuple(lines), keypoints=keypoints)


def keypoint_names(kind: str = "doubles") -> tuple[str, ...]:
    """Lists the keypoint names a template defines.

    Args:
        kind: "doubles" or "singles".

    Returns:
        The names, e.g. "near_baseline:left_singles" - what the manual
        calibration fallback accepts.
    """
    return tuple(court_template(kind).keypoints)


# --------------------------------------------------------------------------
# Lines in the image
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageLine:
    """A straight line found in the image.

    Stored as `a*x + b*y + c = 0` with `a*a + b*b == 1`, so `distance` is in
    pixels.

    Attributes:
        a: Normal x component.
        b: Normal y component.
        c: Offset.
        start: One end of the stretch the line was seen over, (x, y).
        end: The other end.
        support: Pixels of that stretch actually backed by evidence.
    """

    a: float
    b: float
    c: float
    start: Point
    end: Point
    support: float

    @classmethod
    def through(cls, start: Point, end: Point, support: float | None = None) -> ImageLine:
        """Builds the line through two points.

        Args:
            start: One point, (x, y).
            end: Another, distinct point.
            support: Evidence in pixels; the points' separation if omitted.

        Returns:
            The line.

        Raises:
            ValueError: If the two points coincide.
        """
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            raise ValueError("A line needs two distinct points.")
        a, b = -dy / length, dx / length
        c = -(a * start[0] + b * start[1])
        return cls(
            a,
            b,
            c,
            (float(start[0]), float(start[1])),
            (float(end[0]), float(end[1])),
            length if support is None else support,
        )

    @property
    def angle_deg(self) -> float:
        """The line's angle from horizontal, in (-90, 90] degrees."""
        angle = math.degrees(math.atan2(-self.a, self.b))
        if angle > 90:
            angle -= 180
        elif angle <= -90:
            angle += 180
        return angle

    @property
    def length(self) -> float:
        """The length of the stretch the line was seen over, in pixels."""
        return math.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Measures signed perpendicular distance from points to the line.

        Args:
            points: (..., 2) array of (x, y).

        Returns:
            Signed distances in pixels, same leading shape.
        """
        points = np.asarray(points, dtype=float)
        return self.a * points[..., 0] + self.b * points[..., 1] + self.c

    def intersect(self, other: ImageLine) -> Point | None:
        """Finds where two lines cross.

        Args:
            other: The other line.

        Returns:
            The crossing point, or None if the lines are parallel.
        """
        det = self.a * other.b - other.a * self.b
        if abs(det) < 1e-9:
            return None
        x = (self.b * other.c - other.b * self.c) / det
        y = (other.a * self.c - self.a * other.c) / det
        return (x, y)

    def y_at(self, x: float) -> float:
        """Reads the line's y at a given x (for a line that is not vertical)."""
        return -(self.a * x + self.c) / self.b

    def x_at(self, y: float) -> float:
        """Reads the line's x at a given y (for a line that is not horizontal)."""
        return -(self.b * y + self.c) / self.a


@dataclass(frozen=True)
class LineFamilies:
    """Image lines sorted into the court's two families.

    Attributes:
        transverse: Baseline/service-line candidates, strongest first.
        longitudinal: Sideline/centre-line candidates, strongest first.
        vanishing_point: Where the longitudinal lines converge, (x, y), or
            None if no such point was found.
    """

    transverse: tuple[ImageLine, ...]
    longitudinal: tuple[ImageLine, ...]
    vanishing_point: Point | None


@dataclass(frozen=True)
class DetectedLine:
    """A template line located in the image.

    Attributes:
        name: The template line's name.
        line: Where it is in the image.
        coverage: Fraction of the line's in-frame length backed by pixels.
        thickness_px: How thick it looks.
    """

    name: str
    line: ImageLine
    coverage: float
    thickness_px: float


@dataclass(frozen=True)
class Keypoint:
    """A named court point located in the image.

    Attributes:
        name: "<transverse>:<longitudinal>", as in `CourtTemplate.keypoints`.
        court: Its position on the court, (x, y) metres.
        image: Its position in the image, (x, y) pixels.
        in_frame: Whether that position is inside the frame. A point just
            outside it is still the crossing of two measured lines, but is
            extrapolated.
    """

    name: str
    court: Point
    image: Point
    in_frame: bool


@dataclass(frozen=True)
class CourtDetection:
    """What `detect_court` found - or why it found nothing.

    Attributes:
        reason: None when a court was found; otherwise a sentence a person
            can act on, and every other field is empty.
        image_size: (width, height) of the frame examined.
        template_kind: "doubles" or "singles", whichever fitted.
        lines: The template lines located.
        keypoints: Their named intersections.
        support: Fraction of the in-frame template lying on line pixels
            under the detection's own homography, 0 to 1.
        runner_up: Best score reached by a *different* fit, as a fraction of
            the chosen one's. Near 1 would mean the choice was a coin toss.
        rejected: Template lines not located, name to the reason.
        homography: The detection's working court-to-image homography
            (3x3). `calibrate.py` fits the one to rely on.
    """

    reason: str | None
    image_size: tuple[int, int]
    template_kind: str | None = None
    lines: tuple[DetectedLine, ...] = ()
    keypoints: tuple[Keypoint, ...] = ()
    support: float = 0.0
    runner_up: float = 0.0
    rejected: dict[str, str] = field(default_factory=dict)
    homography: np.ndarray | None = None

    @property
    def ok(self) -> bool:
        """Whether a court was found."""
        return self.reason is None

    def line(self, name: str) -> DetectedLine | None:
        """Looks a located line up by name.

        Args:
            name: A template line name.

        Returns:
            The located line, or None if it was not found.
        """
        for line in self.lines:
            if line.name == name:
                return line
        return None


def median_frame(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Blends frames by per-pixel median, which removes whatever moved.

    A player standing on a line hides it in one frame; over several frames
    spread across a few seconds the line is what is usually there.

    Args:
        frames: Same-sized (H, W, 3) uint8 frames.

    Returns:
        One frame of the same shape.

    Raises:
        ValueError: If there are no frames, or their shapes differ.
    """
    stack = [np.asarray(frame) for frame in frames]
    if not stack:
        raise ValueError("median_frame() needs at least one frame.")
    if any(frame.shape != stack[0].shape for frame in stack):
        raise ValueError("median_frame() needs frames of one size.")
    if len(stack) == 1:
        return stack[0].copy()
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def _shifted(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Returns `image` moved by (dx, dy), edges repeated."""
    height, width = image.shape
    padded = cv2.copyMakeBorder(
        image, abs(dy), abs(dy), abs(dx), abs(dx), borderType=cv2.BORDER_REPLICATE
    )
    y0, x0 = abs(dy) - dy, abs(dx) - dx
    return padded[y0 : y0 + height, x0 : x0 + width]


def line_mask(frame: np.ndarray, config: CourtConfig | None = None) -> np.ndarray:
    """Marks pixels that look like painted court line.

    A line pixel is bright, near-white, and brighter than the pixels a short
    reach away on *both* sides - horizontally or vertically. The two-sided
    test is what separates a thin line from the edge of a large bright area
    such as the sky.

    Args:
        frame: (H, W, 3) uint8 RGB frame.
        config: Detection settings.

    Returns:
        (H, W) uint8 mask, 255 on line pixels.
    """
    config = config or CourtConfig()
    frame = np.asarray(frame)
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.int16)
    full = max(2, round(config.line_reach * frame.shape[1]))
    contrast = config.line_contrast
    standing_out = np.zeros(grey.shape, dtype=bool)
    # Two reaches: the long one spans a thick near line; the short one still
    # isolates a thin far line that has another bright band a few pixels away.
    for reach in {full, max(2, full // 3)}:
        standing_out |= ((grey - _shifted(grey, reach, 0)) > contrast) & (
            (grey - _shifted(grey, -reach, 0)) > contrast
        )
        standing_out |= ((grey - _shifted(grey, 0, reach)) > contrast) & (
            (grey - _shifted(grey, 0, -reach)) > contrast
        )
    spread = frame.max(axis=2).astype(np.int16) - frame.min(axis=2).astype(np.int16)
    keep = (
        standing_out & (grey >= config.line_min_brightness) & (spread <= config.line_max_saturation)
    )
    return keep.astype(np.uint8) * 255


def find_segments(mask: np.ndarray, config: CourtConfig | None = None) -> np.ndarray:
    """Finds straight segments in a line mask.

    Args:
        mask: (H, W) uint8 mask from `line_mask`.
        config: Detection settings.

    Returns:
        (N, 4) float array of x1, y1, x2, y2; N is 0 when nothing is found.
    """
    config = config or CourtConfig()
    width = mask.shape[1]
    found = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 360,
        threshold=max(8, round(config.hough_threshold * width)),
        minLineLength=max(8, round(config.hough_min_length * width)),
        maxLineGap=max(2, round(config.hough_max_gap * width)),
    )
    if found is None or not len(found):
        return np.zeros((0, 4), dtype=float)
    # OpenCV 4 returns (N, 1, 4); OpenCV 5 returns (N, 4).
    return np.asarray(found, dtype=float).reshape(-1, 4)


def _fit_line(points: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    """Fits a line to weighted points by total least squares.

    Returns:
        (a, b, c) with a*a + b*b == 1.
    """
    total = weights.sum()
    centre = (points * weights[:, None]).sum(axis=0) / total
    offset = (points - centre) * np.sqrt(weights)[:, None]
    _, _, vt = np.linalg.svd(offset, full_matrices=False)
    a, b = vt[-1]
    return float(a), float(b), float(-(a * centre[0] + b * centre[1]))


def merge_segments(
    segments: np.ndarray, image_size: tuple[int, int], config: CourtConfig | None = None
) -> list[ImageLine]:
    """Merges Hough segments lying on the same straight line.

    One painted line yields many segments - both edges of a thick line, and
    separate pieces either side of whatever crosses it.

    Args:
        segments: (N, 4) array of x1, y1, x2, y2.
        image_size: (width, height) of the frame.
        config: Detection settings.

    Returns:
        One `ImageLine` per distinct line, strongest support first.
    """
    config = config or CourtConfig()
    segments = np.asarray(segments, dtype=float).reshape(-1, 4)
    if not len(segments):
        return []
    tolerance = config.merge_distance * math.hypot(*image_size)
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    order = np.argsort(-lengths, kind="stable")
    seeds: list[ImageLine] = []
    members: list[list[int]] = []
    for index in order:
        x1, y1, x2, y2 = segments[index]
        candidate = ImageLine.through((x1, y1), (x2, y2))
        for seed, group in zip(seeds, members):
            turn = abs(candidate.angle_deg - seed.angle_deg)
            turn = min(turn, 180 - turn)
            if turn > config.merge_angle_deg:
                continue
            if np.abs(seed.distance(np.array([[x1, y1], [x2, y2]]))).max() <= tolerance:
                group.append(index)
                break
        else:
            seeds.append(candidate)
            members.append([index])
    lines = []
    for group in members:
        ends = segments[group].reshape(-1, 2)
        a, b, c = _fit_line(ends, np.repeat(lengths[group], 2))
        direction = np.array([b, -a])
        along = ends @ direction
        start, end = ends[int(np.argmin(along))], ends[int(np.argmax(along))]
        base = along.min()
        covered = np.zeros(int(along.max() - base) + 2, dtype=bool)
        for lo, hi in (along.reshape(-1, 2) - base).tolist():
            covered[int(min(lo, hi)) : int(max(lo, hi)) + 1] = True
        normal = np.array([a, b])
        first = start - (start @ normal + c) * normal  # the ends, dropped onto the fitted line
        last = end - (end @ normal + c) * normal
        lines.append(
            ImageLine(
                a,
                b,
                c,
                (float(first[0]), float(first[1])),
                (float(last[0]), float(last[1])),
                float(covered.sum()),
            )
        )
    lines.sort(key=lambda ln: -ln.support)
    return lines


def _vanishing_point(lines: list[ImageLine], tolerance: float) -> tuple[Point, list[int]] | None:
    """Finds the point most lines pass through, and which lines those are.

    Only a point *above* the lines that define it is considered: sidelines
    seen from behind a baseline converge upward.
    """
    if len(lines) < 2:
        return None
    normals = np.array([[ln.a, ln.b, ln.c] for ln in lines])
    tops = np.array([min(ln.start[1], ln.end[1]) for ln in lines])
    supports = np.array([ln.support for ln in lines])
    first, second = np.triu_indices(len(lines), k=1)
    meets = np.cross(normals[first], normals[second])
    with np.errstate(divide="ignore", invalid="ignore"):
        points = meets[:, :2] / meets[:, 2:]
    usable = (np.abs(meets[:, 2]) > 1e-9) & (points[:, 1] < np.minimum(tops[first], tops[second]))
    if not usable.any():
        return None
    points = points[usable]
    gaps = np.abs(points @ normals[:, :2].T + normals[:, 2])  # (pairs, lines)
    through = (gaps <= tolerance) & (tops[None, :] > points[:, 1:2] - tolerance)
    scores = (through * supports).sum(axis=1)
    best = int(np.argmax(scores))
    inliers = np.nonzero(through[best])[0].tolist()
    point = (float(points[best, 0]), float(points[best, 1]))
    weights = np.sqrt(supports[inliers])
    solved, *_ = np.linalg.lstsq(
        normals[inliers, :2] * weights[:, None], -normals[inliers, 2] * weights, rcond=None
    )
    refined = (float(solved[0]), float(solved[1]))
    if math.hypot(refined[0] - point[0], refined[1] - point[1]) <= 4 * tolerance:
        point = refined
    return point, inliers


def classify_lines(
    lines: list[ImageLine], image_size: tuple[int, int], config: CourtConfig | None = None
) -> LineFamilies:
    """Sorts image lines into baseline-like and sideline-like candidates.

    Args:
        lines: Lines from `merge_segments`.
        image_size: (width, height) of the frame.
        config: Detection settings.

    Returns:
        The two families. Either may be empty; `vanishing_point` is None
        when no two steep lines converge above themselves.
    """
    config = config or CourtConfig()
    flat = [ln for ln in lines if abs(ln.angle_deg) <= config.transverse_max_angle_deg]
    steep = [ln for ln in lines if abs(ln.angle_deg) > config.transverse_max_angle_deg]
    tolerance = config.vanishing_tolerance * math.hypot(*image_size)
    found = _vanishing_point(steep, tolerance)
    if found is None:
        return LineFamilies(tuple(flat[: config.max_transverse]), (), None)
    point, inliers = found
    longitudinal = [steep[k] for k in inliers]
    # A ground line cannot sit above the horizon, which passes through the
    # sidelines' vanishing point.
    transverse = [ln for ln in flat if min(ln.start[1], ln.end[1]) > point[1]]
    return LineFamilies(
        tuple(transverse[: config.max_transverse]),
        tuple(longitudinal[: config.max_longitudinal]),
        point,
    )


# --------------------------------------------------------------------------
# Naming the lines: the candidate-homography search
# --------------------------------------------------------------------------


def project(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Maps points through a homography.

    Args:
        homography: 3x3 matrix.
        points: (..., 2) array.

    Returns:
        (..., 2) array of mapped points. A point that maps to infinity
        comes back as inf/nan rather than raising.
    """
    points = np.asarray(points, dtype=float)
    mapped = points @ homography[:2, :2].T + homography[:2, 2]
    scale = points @ homography[2, :2] + homography[2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return mapped / scale[..., None]


def _project_many(homographies: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Maps the same points through many homographies.

    Args:
        homographies: (N, 3, 3) matrices.
        points: (P, 2) points.

    Returns:
        (N, P, 2) mapped points; inf/nan where a point maps to infinity.
    """
    lifted = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    mapped = np.einsum("nij,pj->npi", homographies, lifted)
    with np.errstate(divide="ignore", invalid="ignore"):
        return mapped[..., :2] / mapped[..., 2:]


def _homographies(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solves many 4-point homographies at once.

    Args:
        source: (N, 4, 2) points.
        target: (N, 4, 2) points they map to.

    Returns:
        (N, 3, 3) matrices with the bottom-right entry 1.
    """
    count = len(source)
    system = np.zeros((count, 8, 8))
    x, y = source[..., 0], source[..., 1]
    u, v = target[..., 0], target[..., 1]
    system[:, 0::2, 0], system[:, 0::2, 1], system[:, 0::2, 2] = x, y, 1.0
    system[:, 0::2, 6], system[:, 0::2, 7] = -u * x, -u * y
    system[:, 1::2, 3], system[:, 1::2, 4], system[:, 1::2, 5] = x, y, 1.0
    system[:, 1::2, 6], system[:, 1::2, 7] = -v * x, -v * y
    wanted = np.zeros((count, 8))
    wanted[:, 0::2], wanted[:, 1::2] = u, v
    solved = np.linalg.solve(system, wanted[..., None])[..., 0]
    return np.concatenate([solved, np.ones((count, 1))], axis=1).reshape(count, 3, 3)


def _template_samples(template: CourtTemplate, per_line: int) -> tuple[np.ndarray, np.ndarray]:
    """Breaks each template line into equal steps.

    Returns:
        A (lines, per_line + 1, 2) array of court points, and a (lines,)
        array saying which evidence map each line is checked against: 0
        for transverse, 1 for longitudinal.
    """
    steps = np.linspace(0.0, 1.0, per_line + 1)[:, None]
    points = np.stack(
        [
            np.array(ln.start) + steps * (np.array(ln.end) - np.array(ln.start))
            for ln in template.lines
        ]
    )
    kinds = np.array([0 if ln.family == TRANSVERSE else 1 for ln in template.lines])
    return points, kinds


def evidence_maps(
    segments: np.ndarray, image_size: tuple[int, int], config: CourtConfig | None = None
) -> np.ndarray:
    """Draws the found segments as two masks, one per line family.

    A candidate fit is supported by *straight segments running the right
    way*, not by any bright pixel: a net's mesh is a field of bright dots
    that would otherwise "support" any line drawn through it.

    Args:
        segments: (N, 4) array of x1, y1, x2, y2 from `find_segments`.
        image_size: (width, height) of the frame.
        config: Detection settings.

    Returns:
        (2, H, W) boolean array: [0] is True near a near-horizontal
        segment (evidence for a baseline or service line), [1] near a
        steep one (evidence for a sideline or the centre line).
    """
    config = config or CourtConfig()
    width, height = image_size
    reach = max(1, round(config.hit_tolerance * math.hypot(width, height)))
    maps = np.zeros((2, height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in np.asarray(segments, dtype=float).reshape(-1, 4):
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        angle = min(angle, 180 - angle)
        which = 0 if angle <= config.transverse_max_angle_deg else 1
        cv2.line(
            maps[which], (round(x1), round(y1)), (round(x2), round(y2)), 1, thickness=2 * reach + 1
        )
    return maps.astype(bool)


def _score(
    homographies: np.ndarray,
    samples: np.ndarray,
    kinds: np.ndarray,
    hit: np.ndarray,
    miss_penalty: float,
    weight_cap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scores candidate homographies by how much template lands on found lines.

    `samples` and `kinds` come from `_template_samples`; `hit` from
    `evidence_maps`. A transverse template line is only supported by
    near-horizontal segments, a longitudinal one only by steep segments.

    Each template line is walked in steps; a step counts by its length in
    pixels, for the line if its midpoint is on a line pixel and against it
    (times `miss_penalty`) if it is in frame but on nothing. A line's net
    share of supported length is then weighted by how much of it is in
    frame, saturating at `weight_cap` pixels - so the score is, roughly,
    *the number of template lines found where expected*. Counting lines
    rather than raw pixels matters: one long baseline must not outvote
    three shorter lines that are missing.

    Returns:
        (score, support) per homography: the line-count score, and the
        supported fraction of the in-frame template by pixel length. A
        homography that puts any of the court behind the camera scores
        -inf.
    """
    height, width = hit.shape[1:]
    flat = samples.reshape(-1, 2)
    scale = flat @ homographies[:, 2, :2].T + homographies[:, 2, 2]  # (points, N)
    mapped = np.einsum("pk,nik->pni", flat, homographies[:, :2, :2]) + homographies[:, :2, 2]
    behind = (scale <= 1e-9).any(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = mapped / scale[..., None]
    pixels = pixels.reshape(samples.shape[0], samples.shape[1], len(homographies), 2)
    pixels = np.nan_to_num(pixels, nan=-1e9, posinf=1e9, neginf=-1e9)
    step = np.linalg.norm(pixels[:, 1:] - pixels[:, :-1], axis=-1)
    middle = (pixels[:, 1:] + pixels[:, :-1]) / 2
    column = np.rint(middle[..., 0])
    row = np.rint(middle[..., 1])
    inside = (column >= 0) & (column < width) & (row >= 0) & (row < height)
    column = np.clip(column, 0, width - 1).astype(np.intp)
    row = np.clip(row, 0, height - 1).astype(np.intp)
    on_line = hit[kinds[:, None, None], row, column] & inside
    step = np.where(inside, np.minimum(step, float(width + height)), 0.0)
    line_supported = (step * on_line).sum(axis=1)  # (lines, N)
    line_visible = step.sum(axis=1)
    net = line_supported - miss_penalty * (line_visible - line_supported)
    share = np.divide(net, line_visible, out=np.zeros_like(net), where=line_visible > 0)
    score = (share * np.minimum(line_visible, weight_cap) / weight_cap).sum(axis=0)
    supported = line_supported.sum(axis=0)
    visible = line_visible.sum(axis=0)
    support = np.divide(supported, visible, out=np.zeros_like(supported), where=visible > 0)
    score[behind] = -np.inf
    support[behind] = 0.0
    return score, support


def _crossings(
    transverse: tuple[ImageLine, ...], longitudinal: tuple[ImageLine, ...]
) -> np.ndarray:
    """Tabulates where every transverse line crosses every longitudinal one.

    Returns:
        (transverse, longitudinal, 2) array; nan where a pair is parallel.
    """
    table = np.full((len(transverse), len(longitudinal), 2), np.nan)
    for i, across in enumerate(transverse):
        for k, along in enumerate(longitudinal):
            point = across.intersect(along)
            if point is not None:
                table[i, k] = point
    return table


def _candidates(
    families: LineFamilies, template: CourtTemplate, image_size: tuple[int, int]
) -> np.ndarray:
    """Builds every homography that names two lines of each family.

    Image lines keep their order: of two transverse lines the lower one in
    the image is the nearer, and of two longitudinal lines the one further
    left stays further left.

    Returns:
        (N, 3, 3) court-to-image homographies; N may be 0.
    """
    width, height = image_size
    transverse = tuple(sorted(families.transverse, key=lambda ln: -ln.y_at(width / 2)))
    longitudinal = families.longitudinal
    table = _crossings(transverse, longitudinal)
    across_pairs = list(itertools.combinations(range(len(transverse)), 2))
    along_pairs = list(itertools.combinations(range(len(longitudinal)), 2))
    across_named = list(itertools.combinations(template.family(TRANSVERSE), 2))
    along_named = list(itertools.combinations(template.family(LONGITUDINAL), 2))
    if not (across_pairs and along_pairs):
        return np.zeros((0, 3, 3))
    quads = []
    for near, far in across_pairs:
        for one, two in along_pairs:
            corners = table[[near, near, far, far], [one, two, two, one]]
            if not np.isfinite(corners).all():
                continue
            if corners[0, 0] > corners[1, 0]:
                corners = corners[[1, 0, 3, 2]]
            edges = np.roll(corners, -1, axis=0) - corners
            turns = (
                edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1]
                - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
            )
            # Near-left, near-right, far-right, far-left must wind one way
            # (negative with y pointing down) and enclose real area.
            if (turns < -1e-4 * width * height).all():
                quads.append(corners)
    if not quads:
        return np.zeros((0, 3, 3))
    targets = np.array(quads)
    sources = np.array(
        [
            [
                (left.position, near.position),
                (right.position, near.position),
                (right.position, far.position),
                (left.position, far.position),
            ]
            for near, far in across_named
            for left, right in along_named
        ]
    )
    source = np.repeat(sources[None], len(targets), axis=0).reshape(-1, 4, 2)
    target = np.repeat(targets[:, None], len(sources), axis=1).reshape(-1, 4, 2)
    return _homographies(source, target)


def plausible_camera(
    homographies: np.ndarray, image_size: tuple[int, int], config: CourtConfig | None = None
) -> np.ndarray:
    """Tells which court-to-image homographies a real camera could produce.

    A camera with square pixels looking at a flat court maps the court's two
    axes to two directions that, once the focal length is undone, are
    perpendicular and equally long. Two equations, one unknown (the focal
    length): solve for it, then check it is a focal length a camera could
    have and that both equations hold. Most wrong ways of naming the lines
    fail this outright - above all the ones that flatten the whole court
    onto a single painted line, which would otherwise score perfectly.

    Assumes the principal point is at the image centre and ignores lens
    distortion, hence the loose tolerances.

    Args:
        homographies: (N, 3, 3) court-to-image matrices.
        image_size: (width, height) of the frame.
        config: Detection settings (`min_focal`, `max_focal`,
            `max_axis_skew`, `max_axis_stretch`).

    Returns:
        (N,) boolean array, True where the homography is plausible.
    """
    config = config or CourtConfig()
    width, height = image_size
    centred = np.array(homographies, dtype=float)
    centred[:, 0, :] -= (width / 2) * centred[:, 2, :]
    centred[:, 1, :] -= (height / 2) * centred[:, 2, :]
    one, two = centred[:, :, 0], centred[:, :, 1]
    cross = one[:, 0] * two[:, 0] + one[:, 1] * two[:, 1]
    cross_depth = one[:, 2] * two[:, 2]
    size = (one[:, 0] ** 2 + one[:, 1] ** 2) - (two[:, 0] ** 2 + two[:, 1] ** 2)
    size_depth = one[:, 2] ** 2 - two[:, 2] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        focal_squared = -(cross * cross_depth + size * size_depth) / (
            cross_depth**2 + size_depth**2
        )
        focal = np.sqrt(focal_squared)
        first = np.stack([one[:, 0] / focal, one[:, 1] / focal, one[:, 2]], axis=1)
        second = np.stack([two[:, 0] / focal, two[:, 1] / focal, two[:, 2]], axis=1)
        first_length = np.linalg.norm(first, axis=1)
        second_length = np.linalg.norm(second, axis=1)
        skew = np.abs((first * second).sum(axis=1)) / (first_length * second_length)
        stretch = first_length / second_length
    return (
        np.isfinite(focal)
        & (focal >= config.min_focal * width)
        & (focal <= config.max_focal * width)
        & np.isfinite(skew)
        & (skew <= config.max_axis_skew)
        & (stretch <= config.max_axis_stretch)
        & (stretch >= 1 / config.max_axis_stretch)
    )


def camera_from_homography(
    homography: np.ndarray, image_size: tuple[int, int]
) -> np.ndarray | None:
    """Recovers the full camera from a court-plane homography.

    The homography only maps the ground. Assuming square pixels and a
    principal point at the image centre, it also fixes the focal length
    and the camera's pose, which is what lets things *above* the ground -
    the net tape - be placed in the image.

    Args:
        homography: 3x3 court-to-image matrix.
        image_size: (width, height) of the frame.

    Returns:
        A 3x4 projection taking (x, y, z, 1) court metres (z up) to
        homogeneous pixels, or None when no real camera fits.
    """
    width, height = image_size
    centred = np.array(homography, dtype=float)
    centred[0] -= (width / 2) * centred[2]
    centred[1] -= (height / 2) * centred[2]
    one, two, three = centred[:, 0], centred[:, 1], centred[:, 2]
    cross, cross_depth = one[0] * two[0] + one[1] * two[1], one[2] * two[2]
    size = (one[0] ** 2 + one[1] ** 2) - (two[0] ** 2 + two[1] ** 2)
    size_depth = one[2] ** 2 - two[2] ** 2
    weight = cross_depth**2 + size_depth**2
    if weight < 1e-30:
        return None
    focal_squared = -(cross * cross_depth + size * size_depth) / weight
    if not np.isfinite(focal_squared) or focal_squared <= 0:
        return None
    focal = math.sqrt(focal_squared)
    undo = np.array([1 / focal, 1 / focal, 1.0])
    first, second, shift = one * undo, two * undo, three * undo
    scale = 2.0 / (np.linalg.norm(first) + np.linalg.norm(second))
    first, second = first / np.linalg.norm(first), second / np.linalg.norm(second)
    pose = np.stack([first, second, np.cross(first, second), scale * shift], axis=1)
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
    return intrinsics @ pose


def net_tape_zone(
    homography: np.ndarray, image_size: tuple[int, int], pad: float
) -> np.ndarray | None:
    """Marks where the net's white tape should appear in the frame.

    The tape is a long, straight, white, near-horizontal band - everything
    a far baseline or service line is, except painted on the ground. From
    a low camera it sits right where those lines are, so its pixels must
    not be mistaken for theirs.

    Args:
        homography: 3x3 court-to-image matrix.
        image_size: (width, height) of the frame.
        pad: Pixels to grow the zone by, to allow for the homography
            being a first estimate.

    Returns:
        (H, W) boolean mask, True on the tape, or None when the camera
        cannot be recovered (see `camera_from_homography`).
    """
    width, height = image_size
    camera = camera_from_homography(homography, image_size)
    if camera is None:
        return None
    reach = DOUBLES_WIDTH_M / 2 + 0.914
    corners = np.array(
        [
            (-reach, NET_Y_M, NET_HEIGHT_CENTRE_M - 0.06, 1.0),
            (reach, NET_Y_M, NET_HEIGHT_CENTRE_M - 0.06, 1.0),
            (reach, NET_Y_M, NET_HEIGHT_POST_M + 0.02, 1.0),
            (-reach, NET_Y_M, NET_HEIGHT_POST_M + 0.02, 1.0),
        ]
    )
    mapped = corners @ camera.T
    if (mapped[:, 2] <= 1e-9).any():
        return None
    outline = mapped[:, :2] / mapped[:, 2:]
    zone = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(zone, [np.round(np.clip(outline, -1e6, 1e6)).astype(np.int32)], 1)
    grow = max(1, round(pad))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * grow + 1, 2 * grow + 1))
    return cv2.dilate(zone, kernel).astype(bool)


@dataclass(frozen=True)
class _Fit:
    """The search's winner for one template."""

    homography: np.ndarray
    template: CourtTemplate
    score: float
    support: float
    runner_up: float


def _best_fit(
    families: LineFamilies,
    template: CourtTemplate,
    hit: np.ndarray,
    config: CourtConfig,
) -> _Fit | None:
    """Searches one template's candidate homographies for the best-supported."""
    height, width = hit.shape[1:]
    candidates = _candidates(families, template, (width, height))
    if len(candidates):
        candidates = candidates[plausible_camera(candidates, (width, height), config)]
    if not len(candidates):
        return None
    samples, kinds = _template_samples(template, config.samples_per_line)
    scores = np.empty(len(candidates))
    supports = np.empty(len(candidates))
    for begin in range(0, len(candidates), 4000):
        chunk = slice(begin, begin + 4000)
        scores[chunk], supports[chunk] = _score(
            candidates[chunk],
            samples,
            kinds,
            hit,
            config.miss_penalty,
            config.line_weight_cap * width,
        )
    winner = int(np.argmax(scores))
    if not np.isfinite(scores[winner]) or scores[winner] <= 0:
        return None
    # Two fits are "different" when they disagree about where the court is
    # *inside the frame*; a corner far outside it is extrapolation.
    marks = np.array(list(template.keypoints.values()))
    here = project(candidates[winner], marks)
    seen = (here[:, 0] >= 0) & (here[:, 0] < width) & (here[:, 1] >= 0) & (here[:, 1] < height)
    marks = marks[seen] if seen.sum() >= 4 else np.array(template.corners)
    placed = _project_many(candidates, marks)
    placed = np.nan_to_num(placed, nan=1e9, posinf=1e9, neginf=-1e9)
    moved = np.linalg.norm(placed - placed[winner], axis=-1).max(axis=-1)
    different = (moved > config.distinct_fit * math.hypot(width, height)) & np.isfinite(scores)
    runner_up = float(scores[different].max()) if different.any() else 0.0
    return _Fit(
        homography=candidates[winner],
        template=template,
        score=float(scores[winner]),
        support=float(supports[winner]),
        runner_up=max(0.0, runner_up) / float(scores[winner]),
    )


# --------------------------------------------------------------------------
# Refitting the named lines to the pixels
# --------------------------------------------------------------------------


def _clip_to_frame(start: np.ndarray, end: np.ndarray, width: int, height: int):
    """Clips a segment to the frame (Liang-Barsky).

    Returns:
        The clipped (start, end), or None if no part is inside.
    """
    delta = end - start
    low, high = 0.0, 1.0
    for moved, offset, limit in (
        (-delta[0], start[0], 0.0),
        (delta[0], width - 1 - start[0], 0.0),
        (-delta[1], start[1], 0.0),
        (delta[1], height - 1 - start[1], 0.0),
    ):
        if abs(moved) < 1e-12:
            if offset < limit:
                return None
            continue
        ratio = offset / moved
        if moved < 0:
            low = max(low, ratio)
        else:
            high = min(high, ratio)
    if low >= high:
        return None
    return start + low * delta, start + high * delta


def _expected_thickness(
    homography: np.ndarray, inverse: np.ndarray, line: CourtLine, at: np.ndarray, paint: float
) -> float:
    """Predicts how thick a painted line should look at an image point."""
    on_court = project(inverse, at)
    offset = np.array([0.0, paint]) if line.family == TRANSVERSE else np.array([paint, 0.0])
    shifted = project(homography, on_court + offset)
    separation = float(np.linalg.norm(shifted - at))
    return separation if math.isfinite(separation) else 0.0


def _refit_line(
    line: CourtLine,
    homography: np.ndarray,
    inverse: np.ndarray,
    pixels: np.ndarray,
    image_size: tuple[int, int],
    config: CourtConfig,
) -> DetectedLine | str:
    """Locates one template line among the line pixels near its projection.

    Returns:
        The located line, or a sentence saying why it was not accepted.
    """
    width, height = image_size
    diagonal = math.hypot(width, height)
    ends = project(homography, np.array([line.start, line.end]))
    scale = np.array([line.start, line.end]) @ homography[2, :2] + homography[2, 2]
    if not np.isfinite(ends).all() or (scale <= 1e-9).any():
        return "projects behind the camera"
    clipped = _clip_to_frame(ends[0], ends[1], width, height)
    minimum = config.min_line_length * width
    if clipped is None or np.linalg.norm(clipped[1] - clipped[0]) < minimum:
        return "outside the frame"
    start, end = clipped
    length = float(np.linalg.norm(end - start))
    direction = (end - start) / length
    normal = np.array([-direction[1], direction[0]])
    band = max(2.5, config.refit_band * diagonal)
    relative = pixels - start
    along = relative @ direction
    off = relative @ normal
    near = (np.abs(off) <= band) & (along >= 0) & (along <= length)
    if near.sum() < minimum / 2:
        return "no line pixels where the template expects it"
    vx, vy, x0, y0 = cv2.fitLine(
        pixels[near].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
    ).ravel()
    fitted_normal = np.array([-vy, vx], dtype=float)
    fitted_off = (pixels[near] - np.array([x0, y0], dtype=float)) @ fitted_normal
    core = np.abs(fitted_off) <= max(2.0, band / 2)
    chosen = pixels[near][core]
    if len(chosen) >= 2:
        vx, vy, x0, y0 = cv2.fitLine(
            chosen.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
        ).ravel()
        fitted_normal = np.array([-vy, vx], dtype=float)
        fitted_off = (pixels[near] - np.array([x0, y0], dtype=float)) @ fitted_normal
    bins = np.zeros(int(length // 2) + 1, dtype=bool)
    bins[(along[near][np.abs(fitted_off) <= band] // 2).astype(int)] = True
    covered = float(bins.sum() * 2)
    coverage = min(1.0, covered / length)
    if covered < minimum or coverage < config.min_line_coverage:
        return f"only {coverage:.0%} of its visible length is backed by line pixels"
    thickness = float((np.abs(fitted_off) <= band).sum() / covered)
    expected = _expected_thickness(
        homography, inverse, line, (start + end) / 2, config.paint_width_m
    )
    allowed = config.thickness_ratio * expected + config.thickness_slack_px
    if thickness > allowed:
        return (
            f"too thick to be a painted line ({thickness:.1f}px where about "
            f"{expected:.1f}px is expected) - probably the net tape or another object"
        )
    centre = np.array([x0, y0], dtype=float)
    travel = np.array([vx, vy], dtype=float)
    first = centre + ((start - centre) @ travel) * travel
    last = centre + ((end - centre) @ travel) * travel
    return DetectedLine(
        name=line.name,
        line=ImageLine.through(tuple(first), tuple(last), support=covered),
        coverage=coverage,
        thickness_px=thickness,
    )


def _keypoints(
    found: dict[str, DetectedLine],
    template: CourtTemplate,
    image_size: tuple[int, int],
    margin: float,
) -> list[Keypoint]:
    """Intersects located lines into the template's named keypoints."""
    width, height = image_size
    keypoints = []
    for name, court in template.keypoints.items():
        across, along = name.split(":")
        if across not in found or along not in found:
            continue
        point = found[across].line.intersect(found[along].line)
        if point is None:
            continue
        x, y = point
        if not (-margin * width <= x <= (1 + margin) * width):
            continue
        if not (-margin * height <= y <= (1 + margin) * height):
            continue
        keypoints.append(
            Keypoint(name, court, (float(x), float(y)), 0 <= x < width and 0 <= y < height)
        )
    return keypoints


def _refine(
    fit: _Fit, mask: np.ndarray, config: CourtConfig
) -> tuple[np.ndarray, dict[str, DetectedLine], dict[str, str], list[Keypoint]]:
    """Alternates refitting the lines and the homography until settled."""
    height, width = mask.shape
    rows, columns = np.nonzero(mask)
    pixels = np.stack([columns, rows], axis=1).astype(float)
    homography = fit.homography
    found: dict[str, DetectedLine] = {}
    rejected: dict[str, str] = {}
    keypoints: list[Keypoint] = []
    pad = config.net_zone_pad * math.hypot(width, height)
    for _ in range(max(1, config.refit_iterations)):
        inverse = np.linalg.inv(homography)
        tape = net_tape_zone(homography, (width, height), pad)
        usable = pixels
        if tape is not None:
            usable = pixels[~tape[rows, columns]]
        found, rejected = {}, {}
        for line in fit.template.lines:
            outcome = _refit_line(line, homography, inverse, usable, (width, height), config)
            if isinstance(outcome, DetectedLine):
                found[line.name] = outcome
                continue
            middle = project(homography, (np.array(line.start) + np.array(line.end)) / 2)
            if tape is not None and np.isfinite(middle).all():
                column, row = round(float(middle[0])), round(float(middle[1]))
                if 0 <= column < width and 0 <= row < height and tape[row, column]:
                    outcome += " - from this camera position it lies behind the net tape"
            rejected[line.name] = outcome
        keypoints = _keypoints(found, fit.template, (width, height), config.keypoint_margin)
        if len(keypoints) < 4:
            break
        refitted, _ = cv2.findHomography(
            np.array([kp.court for kp in keypoints]), np.array([kp.image for kp in keypoints]), 0
        )
        if refitted is None or not np.isfinite(refitted).all():
            break
        homography = refitted
    return homography, found, rejected, keypoints


def detect_court(frame: np.ndarray, config: CourtConfig | None = None) -> CourtDetection:
    """Finds the court's painted lines and named keypoints in one frame.

    Pass a `median_frame` of several frames where possible, so players do
    not hide the lines.

    Args:
        frame: (H, W, 3) uint8 RGB frame.
        config: Detection settings.

    Returns:
        A `CourtDetection`. When the court cannot be found with confidence
        its `reason` says why and nothing is fabricated - check `ok`.
    """
    config = config or CourtConfig()
    frame = np.asarray(frame)
    height, width = frame.shape[:2]
    size = (width, height)
    mask = line_mask(frame, config)
    segments = find_segments(mask, config)
    lines = merge_segments(segments, size, config)
    if not lines:
        return CourtDetection("No straight painted lines were found in the frame.", size)
    families = classify_lines(lines, size, config)
    if len(families.longitudinal) < 2:
        return CourtDetection(
            "Could not find two sidelines converging down the court - the camera "
            "needs to stand behind a baseline, looking along the court.",
            size,
        )
    if len(families.transverse) < 2:
        return CourtDetection(
            "Could not find two lines running across the court (a baseline and a service line).",
            size,
        )
    hit = evidence_maps(segments, size, config)
    fits = [
        fit
        for fit in (
            _best_fit(families, court_template(kind), hit, config) for kind in config.templates
        )
        if fit is not None
    ]
    if not fits:
        return CourtDetection("The lines found do not fit a tennis court in any arrangement.", size)
    fit = max(fits, key=lambda item: item.score)
    logger.info(
        "Court search: %s template, support %.2f, runner-up %.2f",
        fit.template.kind,
        fit.support,
        fit.runner_up,
    )
    if fit.runner_up > config.max_runner_up:
        return CourtDetection(
            f"Two different court placements fit the lines almost equally well "
            f"(the second scores {fit.runner_up:.0%} of the first), so neither is trusted.",
            size,
        )
    homography, found, rejected, keypoints = _refine(fit, mask, config)
    across = sum(1 for name in found if fit.template.line(name).family == TRANSVERSE)
    along = len(found) - across
    if len(keypoints) < 4 or across < 2 or along < 2:
        return CourtDetection(
            f"Only {across} line(s) across the court and {along} down it could be "
            f"confirmed against the pixels - not enough to locate the court.",
            size,
        )
    samples, kinds = _template_samples(fit.template, 4 * config.samples_per_line)
    _, support = _score(
        homography[None], samples, kinds, hit, config.miss_penalty, config.line_weight_cap * width
    )
    if support[0] < config.min_support:
        return CourtDetection(
            f"Only {support[0]:.0%} of the court's lines are where the best fit "
            f"expects them, which is too little to trust.",
            size,
        )
    return CourtDetection(
        reason=None,
        image_size=size,
        template_kind=fit.template.kind,
        lines=tuple(found[name] for name in sorted(found)),
        keypoints=tuple(keypoints),
        support=float(support[0]),
        runner_up=fit.runner_up,
        rejected=rejected,
        homography=homography,
    )
