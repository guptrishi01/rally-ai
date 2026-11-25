"""Viewing conditions: what the court looked like, measured against a reference.

The ball detector - the classic motion detector today, the learned heatmap
model (Track B) once one is trained - was tuned on particular footage. A
recording that looks very different (night play under floodlights, a
clay or green court, glare) is footage it has never been tried on, and the
honest thing to say is so, with the numbers. This module holds:

* `ConditionSample` / `sample_image` / `sample_video` - the court
  surface's brightness and colour in one frame, sampled cheaply (one seek
  per sample, never a full decode).
* `ConditionsReference` - the spread of those values over the footage the
  detector was tuned (or trained) on, plus that footage's identity and, when
  measured, the tracker's own ball coverage on it. It is a JSON file
  (`REFERENCE_PATH`) a future retrain overwrites with its own training
  set's spread, so the flag always describes the model actually in use.
* `compare` - how far a run's conditions sit outside the reference.

How "far" is decided (see docs/ball-model-training.md for the measured
numbers): a run is outside the reference when it is further beyond the
**range seen across the whole reference match** than the larger of

* `RetrainingConfig.sigmas` times the reference's own spread (a measured
  margin: the normal variation within one match), and
* a floor (`RetrainingConfig.brightness_floor` of the reference's median
  brightness; `RetrainingConfig.chroma_floor` in chromaticity). The floors
  are a judgement, not a measurement: no footage in other conditions
  exists yet to say where the detector actually starts to fail.

Colour is compared as **chromaticity** - each channel's share of
R + G + B - so a darker picture of the same court is a brightness
difference only, never also a colour difference.

A flag from here says "this footage is unlike anything the detector was
tuned on", never "the detector failed": whether it failed is what the
tracking measures beside it say.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import video as video_stage
from .ball import SceneSampler, SceneStats
from .calibrate import Calibration
from .config import RetrainingConfig, VideoConfig

logger = logging.getLogger(__name__)

REFERENCE_PATH = Path(__file__).resolve().parent / "models" / "ball_conditions_reference.json"
REFERENCE_FORMAT = 1


def chromaticity(rgb: Sequence[float]) -> tuple[float, float, float]:
    """Each channel's share of the total, so brightness drops out.

    Args:
        rgb: (R, G, B).

    Returns:
        (r, g, b) summing to 1; equal thirds for black.
    """
    total = float(sum(rgb))
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return tuple(float(v) / total for v in rgb)  # type: ignore[return-value]


@dataclass(frozen=True)
class ConditionSample:
    """The court surface in one frame.

    Attributes:
        pts: When the frame was shown, seconds on the video's clock.
        brightness: Mean luminance of the court surface, 0 to 255.
        contrast: Standard deviation of that luminance.
        court_rgb: Mean (R, G, B) of the surface.
        court_yellowness: Mean `ball.yellowness` of the surface.
    """

    pts: float
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
            "pts": self.pts,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "court_rgb": list(self.court_rgb),
            "court_yellowness": self.court_yellowness,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConditionSample:
        """Reads a sample back from `to_dict`'s form.

        Args:
            data: The mapping.

        Returns:
            The sample.
        """
        rgb = data["court_rgb"]
        return cls(
            pts=float(data["pts"]),
            brightness=float(data["brightness"]),
            contrast=float(data["contrast"]),
            court_rgb=(float(rgb[0]), float(rgb[1]), float(rgb[2])),
            court_yellowness=float(data["court_yellowness"]),
        )


def sample_image(image: np.ndarray, calibration: Calibration | None, pts: float) -> ConditionSample:
    """Measures the court surface in one frame.

    Uses `ball.SceneSampler`, so a sample means exactly what a tracking
    run's own scene summary means.

    Args:
        image: (H, W, 3) uint8 RGB frame.
        calibration: The court's calibration; the whole frame without one.
        pts: When the frame was shown.

    Returns:
        The sample.
    """
    sampler = SceneSampler(calibration, 1)
    sampler.add(image)
    stats = sampler.result()
    return ConditionSample(
        pts=float(pts),
        brightness=stats.brightness,
        contrast=stats.contrast,
        court_rgb=stats.court_rgb,
        court_yellowness=stats.court_yellowness,
    )


def sample_times(start: float, end: float, every: float) -> list[float]:
    """Evenly spaced sampling moments.

    Args:
        start: First moment, seconds.
        end: Last moment; included only when it falls on the grid.
        every: Spacing, seconds.

    Returns:
        start, start + every, ... up to `end`.

    Raises:
        ValueError: If `every` is not positive.
    """
    if every <= 0:
        raise ValueError("The sampling interval must be positive.")
    count = int(math.floor((end - start) / every + 1e-9)) + 1 if end >= start else 0
    return [start + k * every for k in range(count)]


def sample_video(
    video: Path | str,
    times: Iterable[float],
    calibration: Calibration | None,
    config: VideoConfig | None = None,
    reader: Callable[[Path, float], tuple[float, np.ndarray]] | None = None,
) -> tuple[list[ConditionSample], list[float]]:
    """Measures the court surface at several moments, one seek each.

    Args:
        video: The recording.
        times: The moments, seconds on its clock.
        calibration: The court's calibration.
        config: Decode settings.
        reader: (video, moment) -> (pts, image); `video.frame_at` by
            default. Tests inject one.

    Returns:
        (samples, failed): one sample per moment that could be decoded, and
        the moments that could not - reported, never silently dropped.
    """

    def decode(path: Path, moment: float) -> tuple[float, np.ndarray]:
        """Reads the frame at a moment with the real decoder."""
        frame = video_stage.frame_at(path, moment, config=config)
        return frame.pts, frame.image

    read = reader or decode
    samples, failed = [], []
    for moment in times:
        try:
            pts, image = read(Path(video), float(moment))
        except video_stage.VideoError as error:
            logger.warning("No frame for a conditions sample at %.1fs: %s", moment, error)
            failed.append(float(moment))
            continue
        samples.append(sample_image(image, calibration, pts))
    return samples, failed


@dataclass(frozen=True)
class Spread:
    """How one measured value varied across a recording.

    Attributes:
        mean: Its mean.
        std: Its standard deviation.
        minimum: The lowest value seen.
        p05: 5th percentile.
        p50: Median.
        p95: 95th percentile.
        maximum: The highest value seen.
    """

    mean: float
    std: float
    minimum: float
    p05: float
    p50: float
    p95: float
    maximum: float

    @classmethod
    def of(cls, values: Sequence[float]) -> Spread:
        """Summarises values.

        Args:
            values: At least one value.

        Returns:
            Their spread.

        Raises:
            ValueError: If there are none.
        """
        data = np.asarray(list(values), dtype=float)
        if not len(data):
            raise ValueError("A spread needs at least one value.")
        p05, p50, p95 = np.percentile(data, [5, 50, 95])
        return cls(
            mean=round(float(data.mean()), 4),
            std=round(float(data.std()), 4),
            minimum=round(float(data.min()), 4),
            p05=round(float(p05), 4),
            p50=round(float(p50), 4),
            p95=round(float(p95), 4),
            maximum=round(float(data.max()), 4),
        )

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field.
        """
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> Spread:
        """Reads a spread back.

        Args:
            data: As `to_dict` writes it.

        Returns:
            The spread.
        """
        return cls(**{name: float(data[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ConditionsReference:
    """The conditions the ball detector was tuned or trained on.

    Attributes:
        video: Identity of the reference footage: name, SHA-256, duration,
            width and height (several entries' worth is a list in
            `extra["videos"]` once a training set spans recordings).
        detector: The ball detector these conditions describe, e.g.
            "motion-ball" (tuned) or a `models.REGISTRY` name (trained).
        measured: ISO date the reference was measured.
        method: How it was sampled, in words.
        samples: Frames it rests on.
        failed_samples: Moments that could not be decoded.
        brightness: Spread of court brightness.
        contrast: Spread of court contrast.
        court_rgb: Mean court (R, G, B).
        chroma: Mean court chromaticity.
        chroma_distance: Spread of each sample's chromaticity distance
            from `chroma` - how much the colour varied within the footage.
        tracking: The tracker's own ball measures on this footage
            (`rally_coverage`, `rally_mean_confidence`, where and how they
            were measured), or None when not measured.
        extra: Anything else a retrain wants to keep (e.g. its dataset).
    """

    video: dict
    detector: str
    measured: str
    method: str
    samples: int
    failed_samples: int
    brightness: Spread
    contrast: Spread
    court_rgb: tuple[float, float, float]
    chroma: tuple[float, float, float]
    chroma_distance: Spread
    tracking: dict | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field, with the file format's version.
        """
        return {
            "format": REFERENCE_FORMAT,
            "video": dict(self.video),
            "detector": self.detector,
            "measured": self.measured,
            "method": self.method,
            "samples": self.samples,
            "failed_samples": self.failed_samples,
            "brightness": self.brightness.to_dict(),
            "contrast": self.contrast.to_dict(),
            "court_rgb": [round(v, 3) for v in self.court_rgb],
            "chroma": [round(v, 5) for v in self.chroma],
            "chroma_distance": self.chroma_distance.to_dict(),
            "tracking": self.tracking,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConditionsReference:
        """Reads a reference back.

        Args:
            data: As `to_dict` writes it.

        Returns:
            The reference.

        Raises:
            ValueError: If the format is not one this code reads.
        """
        if data.get("format") != REFERENCE_FORMAT:
            raise ValueError(f"Unknown conditions reference format {data.get('format')!r}.")
        rgb, chroma = data["court_rgb"], data["chroma"]
        return cls(
            video=dict(data["video"]),
            detector=str(data["detector"]),
            measured=str(data["measured"]),
            method=str(data["method"]),
            samples=int(data["samples"]),
            failed_samples=int(data.get("failed_samples", 0)),
            brightness=Spread.from_dict(data["brightness"]),
            contrast=Spread.from_dict(data["contrast"]),
            court_rgb=(float(rgb[0]), float(rgb[1]), float(rgb[2])),
            chroma=(float(chroma[0]), float(chroma[1]), float(chroma[2])),
            chroma_distance=Spread.from_dict(data["chroma_distance"]),
            tracking=data.get("tracking"),
            extra=dict(data.get("extra") or {}),
        )

    def save(self, path: Path | str = REFERENCE_PATH) -> Path:
        """Writes the reference as JSON.

        Args:
            path: Where; the packaged reference by default.

        Returns:
            The path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def build_reference(
    samples: Sequence[ConditionSample],
    *,
    video: dict,
    detector: str,
    measured: str,
    method: str,
    failed_samples: int = 0,
    tracking: dict | None = None,
    extra: dict | None = None,
) -> ConditionsReference:
    """Summarises samples into a reference.

    Args:
        samples: The court measured across the reference footage.
        video: The footage's identity (name, sha256, duration, ...).
        detector: Which ball detector it describes.
        measured: ISO date.
        method: How it was sampled.
        failed_samples: Moments that could not be decoded.
        tracking: The tracker's ball measures on it, if measured.
        extra: Anything else to keep.

    Returns:
        The reference.

    Raises:
        ValueError: If there are no samples.
    """
    if not samples:
        raise ValueError("A conditions reference needs at least one sample.")
    rgb = np.array([s.court_rgb for s in samples], dtype=float)
    chromas = np.array([chromaticity(s.court_rgb) for s in samples])
    centre = chromas.mean(axis=0)
    return ConditionsReference(
        video=dict(video),
        detector=detector,
        measured=measured,
        method=method,
        samples=len(samples),
        failed_samples=failed_samples,
        brightness=Spread.of([s.brightness for s in samples]),
        contrast=Spread.of([s.contrast for s in samples]),
        court_rgb=tuple(round(float(v), 3) for v in rgb.mean(axis=0)),  # type: ignore[arg-type]
        chroma=tuple(round(float(v), 5) for v in centre),  # type: ignore[arg-type]
        chroma_distance=Spread.of(np.linalg.norm(chromas - centre, axis=1)),
        tracking=tracking,
        extra=dict(extra or {}),
    )


def load_reference(path: Path | str | None = REFERENCE_PATH) -> ConditionsReference | None:
    """Reads a reference, or says there is none.

    Args:
        path: The file; None means "no reference".

    Returns:
        The reference, or None when the file is missing or unreadable -
        logged, since without it no conditions can be compared.
    """
    if path is None:
        return None
    try:
        return ConditionsReference.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError) as error:
        logger.warning("No usable conditions reference at %s: %s", path, error)
        return None


@dataclass(frozen=True)
class ConditionsCheck:
    """How a run's conditions sit against the reference.

    Attributes:
        conditions: One plain phrase per way they differ markedly; empty
            when they do not, or when nothing could be compared.
        measures: The numbers behind it: the run's values, the reference's
            range and the limits used.
        compared: Whether a comparison was made at all (False with no
            scene summary or no reference).
    """

    conditions: tuple[str, ...]
    measures: dict
    compared: bool


def limits(reference: ConditionsReference, config: RetrainingConfig) -> dict:
    """The brightness and colour limits a run is held to.

    Args:
        reference: The reference.
        config: Supplies `sigmas` and the floors.

    Returns:
        {"dim_below", "bright_above", "chroma_above"}.
    """
    b = reference.brightness
    margin = max(config.sigmas * b.std, config.brightness_floor * b.p50)
    d = reference.chroma_distance
    chroma_margin = max(config.sigmas * d.std, config.chroma_floor)
    return {
        "dim_below": round(b.minimum - margin, 2),
        "bright_above": round(b.maximum + margin, 2),
        "chroma_above": round(d.maximum + chroma_margin, 5),
    }


def compare(
    scene: SceneStats | None,
    reference: ConditionsReference | None,
    config: RetrainingConfig | None = None,
) -> ConditionsCheck:
    """Says how a run's court looks against the reference footage.

    Args:
        scene: The run's scene summary.
        reference: The reference; None when there is none.
        config: Margins.

    Returns:
        The check. Without a scene or a reference nothing is compared and
        `compared` is False - "unknown", never "the same".
    """
    config = config or RetrainingConfig()
    if scene is None or reference is None:
        return ConditionsCheck((), {"scene": None if scene is None else scene.to_dict()}, False)
    bounds = limits(reference, config)
    chroma = chromaticity(scene.court_rgb)
    distance = float(np.linalg.norm(np.subtract(chroma, reference.chroma)))
    measures = {
        "brightness": round(scene.brightness, 1),
        "court_rgb": [round(v, 1) for v in scene.court_rgb],
        "chroma_distance": round(distance, 5),
        "reference_brightness_range": [reference.brightness.minimum, reference.brightness.maximum],
        "reference_brightness_median": reference.brightness.p50,
        "reference_court_rgb": [round(v, 1) for v in reference.court_rgb],
        "reference_video": reference.video.get("name"),
        "reference_detector": reference.detector,
        "limits": bounds,
    }
    found = []
    median = reference.brightness.p50
    if scene.brightness < bounds["dim_below"]:
        found.append(
            f"footage much darker than the conditions the ball detector was tuned on: court "
            f"brightness {scene.brightness:.0f} against {reference.brightness.minimum:.0f}-"
            f"{reference.brightness.maximum:.0f} on the reference recording "
            f"({scene.brightness / median:.0%} of its median) - night, dusk or heavy shade"
        )
    elif scene.brightness > bounds["bright_above"]:
        found.append(
            f"footage much brighter than the conditions the ball detector was tuned on: court "
            f"brightness {scene.brightness:.0f} against {reference.brightness.minimum:.0f}-"
            f"{reference.brightness.maximum:.0f} on the reference recording - glare, "
            "overexposure or a pale surface"
        )
    if distance > bounds["chroma_above"]:
        rgb = ", ".join(f"{v:.0f}" for v in scene.court_rgb)
        ref = ", ".join(f"{v:.0f}" for v in reference.court_rgb)
        found.append(
            f"a court colour unlike the reference's (RGB {rgb} against {ref}) - a different "
            "surface, or very different light"
        )
    return ConditionsCheck(tuple(found), measures, True)
