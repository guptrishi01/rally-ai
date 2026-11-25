"""Fitting, checking and using the image<->court homography.

A calibration is a 3x3 homography taking court-plane metres (the frame
`court.py` defines) to image pixels, plus its inverse. It is fitted with
RANSAC from named correspondences - found automatically by
`court.detect_court`, or clicked by a person (`calibrate_from_points`, the
fallback the review UI needs when detection abstains).

**A fit is never returned unchecked.** `fit_homography` abstains - returns a
`CalibrationResult` with no calibration and a sentence saying why - when the
points are too few or collinear, when too many disagree, when the court comes
out mirrored, behind the camera or with its ends swapped, when
the reprojection residual is too large in pixels or in metres, or when the
far baseline is so foreshortened that the far end is unusable. A fit that
passes still carries its `CalibrationQuality`: the residuals, how many metres
one pixel covers at each end, a 0-100 setup score and plain-language
warnings. The planar map says nothing about height: a ball in the air maps
to where its line of sight meets the ground, not to where it is.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import court as court_module
from . import video
from .config import CalibrationConfig, TrackerConfig
from .court import COURT_LENGTH_M, CourtDetection, Point, court_template, project
from .provenance import Provenance

logger = logging.getLogger(__name__)


class CalibrationAbstained(Exception):
    """Raised by `CalibrationResult.require` when there is no calibration."""


@dataclass(frozen=True)
class Correspondence:
    """One court point and where it is in the image.

    Attributes:
        name: The keypoint's name, e.g. "near_baseline:left_singles".
        court: Its court-plane position, (x, y) metres.
        image: Its image position, (x, y) pixels.
    """

    name: str
    court: Point
    image: Point


@dataclass(frozen=True)
class CalibrationQuality:
    """How far a calibration can be trusted, in numbers and in words.

    Attributes:
        points: Correspondences supplied.
        inliers: How many of them agree with the fit.
        residual_px_median: Median distance, in pixels, between where the
            fit puts an agreeing point and where it was seen.
        residual_px_max: The largest such distance.
        residual_m_median: The same error measured on the court, in
            metres. Large at the far end even when the pixels agree,
            because a far pixel covers a lot of court.
        residual_m_max: The largest.
        near_depth_m_per_px: Metres of court depth one pixel covers at the
            near baseline.
        far_depth_m_per_px: The same at the far baseline - the number that
            says how well a far-end in/out call can possibly be made.
        far_lateral_m_per_px: Metres of court width one pixel covers at
            the far baseline.
        lines_found: Names of the template lines detection located, or
            None for a calibration made from clicked points.
        far_baseline_found: Whether the far baseline itself was located
            (None for clicked points). When False its position is
            extrapolated from the other lines.
        support: Fraction of the in-frame template lying on line pixels
            (None for clicked points).
        setup_score: 0-100 summary of the camera setup: half from the far
            depth resolution, the rest from the residual and from how many
            lines were found. A heuristic for ranking setups, not a
            probability.
        warnings: Sentences for the user about what is weak.
    """

    points: int
    inliers: int
    residual_px_median: float
    residual_px_max: float
    residual_m_median: float
    residual_m_max: float
    near_depth_m_per_px: float
    far_depth_m_per_px: float
    far_lateral_m_per_px: float
    lines_found: tuple[str, ...] | None
    far_baseline_found: bool | None
    support: float | None
    setup_score: float
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types, for the staged `tracking` block.

        Returns:
            Every field, tuples as lists.
        """
        return {
            "points": self.points,
            "inliers": self.inliers,
            "residual_px_median": self.residual_px_median,
            "residual_px_max": self.residual_px_max,
            "residual_m_median": self.residual_m_median,
            "residual_m_max": self.residual_m_max,
            "near_depth_m_per_px": self.near_depth_m_per_px,
            "far_depth_m_per_px": self.far_depth_m_per_px,
            "far_lateral_m_per_px": self.far_lateral_m_per_px,
            "lines_found": None if self.lines_found is None else list(self.lines_found),
            "far_baseline_found": self.far_baseline_found,
            "support": self.support,
            "setup_score": self.setup_score,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CalibrationQuality:
        """Reads a quality back from `to_dict`'s form.

        Args:
            data: A dict as `to_dict` writes it.

        Returns:
            The quality, lists back as tuples.
        """
        lines = data.get("lines_found")
        return cls(
            points=int(data["points"]),
            inliers=int(data["inliers"]),
            residual_px_median=float(data["residual_px_median"]),
            residual_px_max=float(data["residual_px_max"]),
            residual_m_median=float(data["residual_m_median"]),
            residual_m_max=float(data["residual_m_max"]),
            near_depth_m_per_px=float(data["near_depth_m_per_px"]),
            far_depth_m_per_px=float(data["far_depth_m_per_px"]),
            far_lateral_m_per_px=float(data["far_lateral_m_per_px"]),
            lines_found=None if lines is None else tuple(lines),
            far_baseline_found=data.get("far_baseline_found"),
            support=data.get("support"),
            setup_score=float(data["setup_score"]),
            warnings=tuple(data.get("warnings", ())),
        )


@dataclass(frozen=True)
class Calibration:
    """A checked mapping between the image and the court plane.

    Attributes:
        homography: 3x3 matrix taking court metres to image pixels.
        inverse: 3x3 matrix taking image pixels to court metres.
        image_size: (width, height) of the frames it applies to.
        template_kind: "doubles" or "singles".
        provenance: `MEASURED` when found from the pixels, `USER_ENTERED`
            when fitted to clicked points.
        quality: How far to trust it.
    """

    homography: np.ndarray
    inverse: np.ndarray
    image_size: tuple[int, int]
    template_kind: str
    provenance: Provenance
    quality: CalibrationQuality

    def to_image(self, points: np.ndarray | Iterable[Point]) -> np.ndarray:
        """Maps court-plane points into the image.

        Args:
            points: (..., 2) court positions in metres.

        Returns:
            (..., 2) image positions in pixels.
        """
        return project(self.homography, np.asarray(points, dtype=float))

    def to_court(self, points: np.ndarray | Iterable[Point]) -> np.ndarray:
        """Maps image points onto the court plane.

        Only meaningful for things touching the ground - a bounce, a
        player's feet. See the module docstring.

        Args:
            points: (..., 2) image positions in pixels.

        Returns:
            (..., 2) court positions in metres.
        """
        return project(self.inverse, np.asarray(points, dtype=float))

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            The homography as nested lists, with size, template,
            provenance and quality.
        """
        return {
            "homography": self.homography.tolist(),
            "image_size": list(self.image_size),
            "template_kind": self.template_kind,
            "provenance": self.provenance.value,
            "quality": self.quality.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Calibration:
        """Reads a calibration back from `to_dict`'s form.

        The stored calibration was checked when it was made; it is not
        re-checked here. The inverse is recomputed from the homography.

        Args:
            data: A dict as `to_dict` writes it.

        Returns:
            The calibration.
        """
        homography = np.asarray(data["homography"], dtype=float)
        width, height = data["image_size"]
        return cls(
            homography=homography,
            inverse=np.linalg.inv(homography),
            image_size=(int(width), int(height)),
            template_kind=data["template_kind"],
            provenance=Provenance.parse(data["provenance"]),
            quality=CalibrationQuality.from_dict(data["quality"]),
        )


@dataclass(frozen=True)
class CalibrationResult:
    """A calibration, or an explicit refusal to give one.

    Attributes:
        calibration: The calibration, or None when abstaining.
        reason: None when calibrated; otherwise why not, in a sentence a
            person can act on.
        quality: The measurements behind the decision when the fit got far
            enough to be measured - present even on some abstentions, so
            the UI can show *how* bad it was.
        detection: The court detection it came from, if any.
    """

    calibration: Calibration | None
    reason: str | None
    quality: CalibrationQuality | None = None
    detection: CourtDetection | None = None

    @property
    def ok(self) -> bool:
        """Whether there is a calibration to use."""
        return self.calibration is not None

    def require(self) -> Calibration:
        """Returns the calibration or refuses loudly.

        Returns:
            The calibration.

        Raises:
            CalibrationAbstained: If there is none, carrying the reason.
        """
        if self.calibration is None:
            raise CalibrationAbstained(self.reason)
        return self.calibration


def _spread(points: np.ndarray) -> float:
    """Measures how far a point set is from lying on one line (0 = on a line)."""
    values = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return float(values[1] / values[0]) if values[0] > 0 else 0.0


def _metres_per_pixel(homography: np.ndarray, at: Point, toward: Point) -> float:
    """Measures how much court one pixel covers at a court point, in a direction."""
    here = project(homography, np.array(at))
    there = project(homography, np.array(at) + np.array(toward))
    pixels = float(np.linalg.norm(there - here))
    return math.hypot(*toward) / pixels if pixels > 1e-9 else math.inf


def _shape_problem(
    homography: np.ndarray, template: court_module.CourtTemplate, config: CalibrationConfig
) -> str | None:
    """Checks the fitted court is a court seen from above, the right way round.

    Returns:
        None if it is; otherwise what is wrong.
    """
    corners = np.array(template.corners)
    depth = corners @ homography[2, :2] + homography[2, 2]
    if (depth <= 1e-9).any() or not np.isfinite(homography).all():
        return (
            "The fit puts part of the court behind the camera, so the points "
            "cannot be where they were marked."
        )
    placed = project(homography, corners)
    edges = np.roll(placed, -1, axis=0) - placed
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    # With every corner in front of the camera the outline stays convex, so
    # it winds one way or the other: the right way, or mirrored.
    if not (turns < 0).all():
        return (
            "The fit shows the court mirrored - left and right (or near and far) "
            "points appear to be swapped."
        )
    near = float(np.linalg.norm(placed[1] - placed[0]))
    far = float(np.linalg.norm(placed[2] - placed[3]))
    if near < config.min_near_far_ratio * far:
        return (
            "The far baseline comes out longer than the near one - the two ends "
            "appear to be labelled the wrong way round."
        )
    return None


def _setup_score(
    far_depth: float, residual_px: float, limit_px: float, lines: float | None, worst: float
) -> float:
    """Combines far-end resolution, residual and lines found into 0-100."""
    best = 0.05
    far = 1.0 - math.log(max(far_depth, best) / best) / math.log(worst / best)
    far = min(1.0, max(0.0, far))
    fit = min(1.0, max(0.0, 1.0 - residual_px / limit_px))
    if lines is None:
        return round(100.0 * (2 * far + fit) / 3, 1)
    return round(100.0 * (0.5 * far + 0.25 * fit + 0.25 * lines), 1)


def fit_homography(
    correspondences: Iterable[Correspondence],
    image_size: tuple[int, int],
    config: CalibrationConfig | None = None,
    template_kind: str = "doubles",
    provenance: Provenance = Provenance.MEASURED,
    detection: CourtDetection | None = None,
) -> CalibrationResult:
    """Fits a homography to court<->image correspondences, and checks it.

    Args:
        correspondences: At least four named court points with their image
            positions, not all on one line.
        image_size: (width, height) of the frame the image positions are in.
        config: Fit and validation settings.
        template_kind: "doubles" or "singles" - which court the points are
            on; decides which outline the sanity checks use.
        provenance: How the image positions were obtained.
        detection: The automatic detection they came from, if any; its
            line findings go into the quality summary.

    Returns:
        A `CalibrationResult`: a calibration with its quality, or an
        abstention with the reason.
    """
    config = config or CalibrationConfig()
    template = court_template(template_kind)
    items = list(correspondences)
    width, height = image_size
    diagonal = math.hypot(width, height)

    def abstain(reason: str, quality: CalibrationQuality | None = None) -> CalibrationResult:
        """Builds the refusal, logging why."""
        logger.info("Calibration abstained: %s", reason)
        return CalibrationResult(None, reason, quality, detection)

    if len(items) < max(4, config.min_points):
        return abstain(
            f"A calibration needs at least {max(4, config.min_points)} court points; "
            f"{len(items)} were given."
        )
    on_court = np.array([item.court for item in items], dtype=float)
    in_image = np.array([item.image for item in items], dtype=float)
    if not (np.isfinite(on_court).all() and np.isfinite(in_image).all()):
        return abstain("A court point has a position that is not a number.")
    if min(_spread(on_court), _spread(in_image)) < config.min_spread:
        return abstain(
            "The court points all lie along one line, which cannot fix a court's position."
        )
    cv2.setRNGSeed(0)
    homography, flags = cv2.findHomography(
        on_court, in_image, cv2.RANSAC, config.ransac_threshold * diagonal
    )
    if homography is None or not np.isfinite(homography).all():
        return abstain("No consistent court position fits these points.")
    agree = flags.ravel().astype(bool)
    needed = max(4, math.ceil(config.min_inlier_fraction * len(items)))
    if agree.sum() < needed:
        return abstain(
            f"Only {int(agree.sum())} of the {len(items)} court points agree on one "
            f"court position; at least {needed} must."
        )
    if abs(np.linalg.det(homography)) < 1e-12:
        return abstain("The fit collapses the court onto a line.")
    problem = _shape_problem(homography, template, config)
    if problem is not None:
        return abstain(problem)
    inverse = np.linalg.inv(homography)
    error_px = np.linalg.norm(project(homography, on_court[agree]) - in_image[agree], axis=1)
    error_m = np.linalg.norm(project(inverse, in_image[agree]) - on_court[agree], axis=1)
    far_depth = _metres_per_pixel(homography, (0.0, COURT_LENGTH_M), (0.0, -0.5))
    near_depth = _metres_per_pixel(homography, (0.0, 0.0), (0.0, 0.5))
    far_lateral = _metres_per_pixel(homography, (0.0, COURT_LENGTH_M), (0.5, 0.0))
    limit_px = config.max_residual_px * diagonal

    warnings = []
    if not agree.all():
        dropped = ", ".join(item.name for item, kept in zip(items, agree) if not kept)
        warnings.append(
            f"Ignored {int((~agree).sum())} point(s) that disagree with the rest: {dropped}."
        )
    if len(items) == 4:
        warnings.append(
            "Fitted from exactly four points, so the residual is zero by construction "
            "and cannot reveal a misplaced point. Mark more points to check the fit."
        )
    outside = int(
        (
            (in_image[agree, 0] < 0)
            | (in_image[agree, 0] >= width)
            | (in_image[agree, 1] < 0)
            | (in_image[agree, 1] >= height)
        ).sum()
    )
    if outside:
        warnings.append(
            f"{outside} of the points used lie outside the frame, where two found "
            f"lines would cross - extrapolated rather than seen."
        )
    lines_found = None
    far_found = None
    support = None
    share = None
    if detection is not None and detection.ok:
        lines_found = tuple(line.name for line in detection.lines)
        far_found = "far_baseline" in lines_found
        support = detection.support
        share = len(lines_found) / len(template.lines)
        if not far_found:
            why = detection.rejected.get("far_baseline", "it was not found")
            warnings.append(
                f"The far baseline was not located ({why}); its position is "
                f"extrapolated from the other lines."
            )
    if far_depth > config.warn_far_depth_m_per_px:
        warnings.append(
            f"At the far baseline one pixel covers {far_depth:.2f} m of court depth, so "
            f"in/out calls at the far end will be weak. Raise the camera or move it back."
        )
    quality = CalibrationQuality(
        points=len(items),
        inliers=int(agree.sum()),
        residual_px_median=float(np.median(error_px)),
        residual_px_max=float(error_px.max()),
        residual_m_median=float(np.median(error_m)),
        residual_m_max=float(error_m.max()),
        near_depth_m_per_px=near_depth,
        far_depth_m_per_px=far_depth,
        far_lateral_m_per_px=far_lateral,
        lines_found=lines_found,
        far_baseline_found=far_found,
        support=support,
        setup_score=_setup_score(
            far_depth, float(np.median(error_px)), limit_px, share, config.max_far_depth_m_per_px
        ),
        warnings=tuple(warnings),
    )
    if quality.residual_px_median > limit_px:
        return abstain(
            f"The points disagree with the best-fitting court by {quality.residual_px_median:.1f} "
            f"pixels (median); the limit is {limit_px:.1f}.",
            quality,
        )
    if quality.residual_m_median > config.max_residual_m:
        return abstain(
            f"The points disagree with the best-fitting court by "
            f"{quality.residual_m_median:.2f} m on the court (median); the limit is "
            f"{config.max_residual_m:.2f} m.",
            quality,
        )
    if far_depth > config.max_far_depth_m_per_px:
        return abstain(
            f"The camera is too low or too flat to the court: at the far baseline one "
            f"pixel covers {far_depth:.1f} m of court depth, so the far end cannot be "
            f"located. Raise the camera and re-aim.",
            quality,
        )
    return CalibrationResult(
        Calibration(
            homography=homography,
            inverse=inverse,
            image_size=(width, height),
            template_kind=template_kind,
            provenance=provenance,
            quality=quality,
        ),
        None,
        quality,
        detection,
    )


def calibrate_detection(
    detection: CourtDetection, config: CalibrationConfig | None = None
) -> CalibrationResult:
    """Turns a court detection into a checked calibration.

    Args:
        detection: The output of `court.detect_court`.
        config: Fit and validation settings.

    Returns:
        A calibration fitted to the detection's keypoints, or an
        abstention - with the detection's own reason if it found no court.
    """
    if not detection.ok:
        return CalibrationResult(None, detection.reason, None, detection)
    return fit_homography(
        [Correspondence(kp.name, kp.court, kp.image) for kp in detection.keypoints],
        detection.image_size,
        config,
        template_kind=detection.template_kind,
        provenance=Provenance.MEASURED,
        detection=detection,
    )


def calibrate_frame(frame: np.ndarray, config: TrackerConfig | None = None) -> CalibrationResult:
    """Calibrates the court from one frame.

    Args:
        frame: (H, W, 3) uint8 RGB frame - ideally a `court.median_frame`
            of several, so players do not hide lines.
        config: Tracker settings (its `court` and `calibration` parts).

    Returns:
        A calibration, or an abstention with the reason.
    """
    config = config or TrackerConfig()
    return calibrate_detection(court_module.detect_court(frame, config.court), config.calibration)


def calibrate_frames(
    frames: Iterable[np.ndarray], config: TrackerConfig | None = None
) -> CalibrationResult:
    """Calibrates the court from several frames blended by median.

    Args:
        frames: Same-sized RGB frames spread over a few seconds.
        config: Tracker settings.

    Returns:
        A calibration, or an abstention with the reason.
    """
    return calibrate_frame(court_module.median_frame(frames), config)


def calibrate_video(
    path: Path | str, at: float = 0.0, config: TrackerConfig | None = None
) -> CalibrationResult:
    """Calibrates the court as it appears at one moment of a recording.

    Samples `calibration.frames_for_median` frames across
    `calibration.median_span_seconds` starting at `at`, blends them, and
    calibrates. Call it again later in the match and compare with
    `drift_px` to notice a camera that has moved.

    Args:
        path: The video file.
        at: Where the sampled stretch begins, as a PTS in seconds.
        config: Tracker settings.

    Returns:
        A calibration, or an abstention with the reason.

    Raises:
        tracking.video.VideoUnreadableError: If frames cannot be decoded
            there.
    """
    config = config or TrackerConfig()
    count = max(1, config.calibration.frames_for_median)
    step = config.calibration.median_span_seconds / count
    frames = video.frames_at(path, [at + k * step for k in range(count)], config=config.video)
    return calibrate_frames([frame.image for frame in frames], config)


def calibrate_from_points(
    points: Mapping[str, Point],
    image_size: tuple[int, int],
    config: CalibrationConfig | None = None,
    template_kind: str = "doubles",
) -> CalibrationResult:
    """Calibrates from keypoints a person marked - the manual fallback.

    Args:
        points: Keypoint name to its (x, y) pixel position. Names are those
            of `court.keypoint_names(template_kind)`, e.g.
            "near_baseline:left_singles", "far_service:center".
        image_size: (width, height) of the frame the positions are in.
        config: Fit and validation settings.
        template_kind: "doubles" or "singles".

    Returns:
        A calibration tagged `USER_ENTERED`, or an abstention with the
        reason - a person's clicks are checked exactly as a detection's
        points are.

    Raises:
        ValueError: If a name is not a keypoint of that template.
    """
    template = court_template(template_kind)
    unknown = sorted(set(points) - set(template.keypoints))
    if unknown:
        raise ValueError(
            f"Unknown court keypoint(s) {', '.join(unknown)}; expected names such as "
            f"{', '.join(list(template.keypoints)[:3])} (see court.keypoint_names)."
        )
    return fit_homography(
        [
            Correspondence(name, template.keypoints[name], (float(x), float(y)))
            for name, (x, y) in points.items()
        ],
        image_size,
        config,
        template_kind=template_kind,
        provenance=Provenance.USER_ENTERED,
    )


def drift_px(first: Calibration, second: Calibration) -> float:
    """Measures how far the court moved in the image between two calibrations.

    A fixed camera gives a few pixels at most (detection noise); a bumped
    tripod gives far more, and the later calibration should replace the
    earlier one from then on.

    Args:
        first: One calibration.
        second: Another, of the same recording.

    Returns:
        The largest displacement, in pixels, of any court keypoint that is
        inside the frame under `first` (of the court's corners if fewer
        than four are).
    """
    template = court_template(first.template_kind)
    marks = np.array(list(template.keypoints.values()))
    here = first.to_image(marks)
    width, height = first.image_size
    seen = (here[:, 0] >= 0) & (here[:, 0] < width) & (here[:, 1] >= 0) & (here[:, 1] < height)
    if seen.sum() < 4:
        marks = np.array(template.corners)
        here = first.to_image(marks)
    else:
        marks, here = marks[seen], here[seen]
    return float(np.linalg.norm(second.to_image(marks) - here, axis=1).max())


def draw_overlay(
    frame: np.ndarray,
    calibration: Calibration,
    detection: CourtDetection | None = None,
) -> np.ndarray:
    """Draws the template court over a frame, for a person to check by eye.

    A low residual on wrongly-named lines is the failure numbers cannot
    show; a picture can. Template lines are drawn in red through the
    calibration; the detection's keypoints, if given, as green rings.

    Args:
        frame: (H, W, 3) uint8 RGB frame.
        calibration: The calibration to draw.
        detection: Optionally, the detection whose keypoints to mark.

    Returns:
        A new annotated RGB frame; the input is not modified.
    """
    canvas = np.ascontiguousarray(np.asarray(frame).copy())
    template = court_template(calibration.template_kind)
    for line in template.lines:
        path = calibration.to_image(np.linspace(line.start, line.end, 60))
        path = path[np.isfinite(path).all(axis=1)]
        cv2.polylines(canvas, [np.round(path).astype(np.int32)], False, (255, 0, 0), 1, cv2.LINE_AA)
    if detection is not None:
        for keypoint in detection.keypoints:
            centre = (round(keypoint.image[0]), round(keypoint.image[1]))
            cv2.circle(canvas, centre, 7, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas
