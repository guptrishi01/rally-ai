"""A cheap pre-pass that finds the stretches of a recording with no play in them.

Most of a tennis recording is not rallies: points are separated by walking,
towelling and changeovers. Tracking costs about five seconds of CPU per
second of video, so time spent on stretches with nothing in them is the
largest avoidable cost. This pre-pass listens instead of looking: a ball
strike is a sharp sound, and reading a whole match's audio takes seconds.

How a stretch is judged
-----------------------
1. The audio is read once (8 kHz mono) and reduced to its onset curve - the
   rise in high-frequency energy every 10 ms, the same quantity
   `swingvision_import.footage.onset_envelope` uses - scaled by the
   **whole recording's** 99th percentile, so a quiet stretch is quiet
   against the match, not against itself.
2. A *strike* is a local peak at or above `DeadTimeConfig.min_strength`.
3. Strikes no more than `join_seconds` apart form a burst; a burst of at
   least `min_strikes` is play, padded by `pad_before` / `pad_after`.
4. What is left between play, if at least `min_dead_seconds` long, is dead
   time and is not tracked.

It errs towards tracking: neighbouring courts, voices and a ball bounced
before a serve all keep a stretch active. What it can get wrong is a quiet
point - a ball struck too softly to register - and that is why it is off
by default, why every skipped stretch is listed in the staged match, and
why the measured numbers (docs/ball-model-training.md, "Dead-time
pre-pass") are stated with the caveat that the settings were chosen on the
same recording they were measured on.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from swingvision_import import footage

from .config import DeadTimeConfig

logger = logging.getLogger(__name__)

SAMPLE_RATE = 8000
STEP_SECONDS = 0.01
_HOP = int(SAMPLE_RATE * STEP_SECONDS)


class OnsetCurve:
    """Builds the raw onset curve from audio handed over in blocks.

    The curve is the rise, step to step, in the RMS of the audio's first
    difference over 10 ms steps - unscaled, so blocks can be joined and the
    whole recording scaled at the end.
    """

    def __init__(self) -> None:
        """Starts an empty curve."""
        self._carry = np.zeros(0, dtype=np.float32)
        self._last_sample: float | None = None
        self._last_energy: float | None = None
        self._parts: list[np.ndarray] = []

    def feed(self, samples: np.ndarray) -> None:
        """Adds the next block of samples.

        Args:
            samples: Mono samples, any length.
        """
        data = np.concatenate([self._carry, np.asarray(samples, dtype=np.float32)])
        steps = len(data) // _HOP
        self._carry = data[steps * _HOP :]
        if not steps:
            return
        block = data[: steps * _HOP]
        first = block[0] if self._last_sample is None else self._last_sample
        edges = np.diff(block, prepend=np.float32(first))
        self._last_sample = float(block[-1])
        energy = np.sqrt((edges.reshape(steps, _HOP) ** 2).mean(axis=1))
        before = energy[0] if self._last_energy is None else self._last_energy
        self._last_energy = float(energy[-1])
        self._parts.append(np.maximum(0.0, np.diff(energy, prepend=np.float32(before))))

    def result(self) -> np.ndarray:
        """The curve so far.

        Returns:
            One value per 10 ms step from the first sample.
        """
        if not self._parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._parts).astype(np.float32)


def read_onset_curve(
    video: Path | str, ffmpeg: str | None = None, block_seconds: float = 30.0
) -> np.ndarray | None:
    """Reads a recording's whole audio track into its onset curve.

    Streams the audio in blocks, so memory stays at one block plus the
    curve (some 2MB for a 95-minute match).

    Args:
        video: The recording.
        ffmpeg: ffmpeg binary; located automatically if not given.
        block_seconds: Audio read per block.

    Returns:
        The raw curve (step 10 ms, from time 0), or None when the
        recording has no readable audio.
    """
    command = [ffmpeg or footage.ffmpeg_exe(), "-hide_banner", "-loglevel", "error"]
    command += ["-i", str(video), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError as error:
        logger.warning("Could not start ffmpeg for the dead-time pre-pass: %s", error)
        return None
    curve = OnsetCurve()
    size = int(block_seconds * SAMPLE_RATE) * 2
    try:
        while block := process.stdout.read(size):
            curve.feed(np.frombuffer(block[: len(block) // 2 * 2], dtype=np.int16))
    finally:
        process.stdout.close()
        code = process.wait()
    result = curve.result()
    if code != 0 or not len(result):
        logger.warning("No usable audio in %s for the dead-time pre-pass.", Path(video).name)
        return None
    return result


def strikes(curve: np.ndarray, config: DeadTimeConfig) -> np.ndarray:
    """Finds the sharp sounds in a whole recording's onset curve.

    Args:
        curve: The raw onset curve.
        config: Strength, separation and scaling percentile.

    Returns:
        Their times, seconds from the curve's start.
    """
    if not len(curve):
        return np.zeros(0)
    ceiling = float(np.percentile(curve, config.percentile))
    if ceiling <= 0:
        return np.zeros(0)
    scaled = curve / ceiling
    reach = max(1, round(config.separation_seconds / STEP_SECONDS))
    padded = np.pad(scaled, reach)
    highest = np.max([padded[k : k + len(scaled)] for k in range(2 * reach + 1)], axis=0)
    peaks = np.flatnonzero((scaled >= highest) & (scaled >= config.min_strength))
    return peaks * STEP_SECONDS


def _spans(times: np.ndarray, config: DeadTimeConfig) -> list[tuple[float, float]]:
    """Play spans from strike times: bursts, padded, joined across short gaps."""
    spans: list[list[float]] = []
    first = 0
    for k in range(1, len(times) + 1):
        if k < len(times) and times[k] - times[k - 1] <= config.join_seconds:
            continue
        if k - first >= config.min_strikes:
            low, high = times[first] - config.pad_before, times[k - 1] + config.pad_after
            if spans and low - spans[-1][1] < config.min_dead_seconds:
                spans[-1][1] = max(spans[-1][1], high)
            else:
                spans.append([low, high])
        first = k
    return [(float(a), float(b)) for a, b in spans]


@dataclass(frozen=True)
class DeadTimePlan:
    """Which stretches of a run are tracked and which are skipped.

    Attributes:
        start: The run's first moment, seconds on the video's clock.
        end: Its last moment.
        active: Stretches that are tracked, in order, within the run.
        dead: Stretches that are skipped, in order.
        strikes: Sharp sounds found in the run.
        settings: The `DeadTimeConfig` values used.
    """

    start: float
    end: float
    active: tuple[tuple[float, float], ...]
    dead: tuple[tuple[float, float], ...]
    strikes: int
    settings: dict

    def is_active(self, pts: float) -> bool:
        """Whether a moment is tracked.

        Args:
            pts: The moment.

        Returns:
            False inside a dead stretch, True otherwise.
        """
        return not any(low <= pts < high for low, high in self.dead)

    @property
    def skipped_seconds(self) -> float:
        """Total length of the dead stretches."""
        return float(sum(high - low for low, high in self.dead))

    @property
    def skipped_share(self) -> float:
        """The dead stretches' share of the run."""
        length = self.end - self.start
        return self.skipped_seconds / length if length > 0 else 0.0

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field, the stretches rounded to 10 ms, and the totals.
        """
        return {
            "start": self.start,
            "end": self.end,
            "active": [[round(a, 2), round(b, 2)] for a, b in self.active],
            "dead": [[round(a, 2), round(b, 2)] for a, b in self.dead],
            "strikes": self.strikes,
            "skipped_seconds": round(self.skipped_seconds, 2),
            "skipped_share": round(self.skipped_share, 4),
            "settings": dict(self.settings),
        }

    def note(self) -> str:
        """The import note describing what was skipped.

        Returns:
            One sentence with the totals and the caveat.
        """
        return (
            f"Dead-time pre-pass: {self.skipped_seconds:.0f}s of the {self.end - self.start:.0f}s "
            f"tracked ({self.skipped_share:.0%}) in {len(self.dead)} stretch(es) with no "
            "ball-strike sounds was skipped, not tracked. Any play in those stretches is "
            "missing from this match; if a point is missing, track again without the "
            "pre-pass. The stretches are listed in the tracking block."
        )


def plan_dead_time(
    curve: np.ndarray | None,
    start: float,
    end: float,
    config: DeadTimeConfig | None = None,
) -> DeadTimePlan | None:
    """Works out which stretches of a run to skip.

    Args:
        curve: The recording's raw onset curve, from time 0 (step 10 ms);
            None when its audio could not be read.
        start: The run's first moment, seconds on the video's clock.
        end: Its last moment.
        config: The pre-pass settings.

    Returns:
        The plan, or None when there is no audio to plan from - the whole
        run is then tracked.
    """
    config = config or DeadTimeConfig()
    if curve is None or not len(curve):
        return None
    times = strikes(curve, config)
    play = _spans(times, config)
    active, dead = [], []
    cursor = start
    for low, high in play:
        low, high = max(low, start), min(high, end)
        if high <= low:
            continue
        if low - cursor >= config.min_dead_seconds:
            dead.append((cursor, low))
        else:
            # Too short a quiet stretch to skip: tracked with the play after it.
            # (`_spans` already joined bursts closer than that, so this is only
            # ever the stretch before the run's first burst.)
            low = cursor
        active.append((low, high))
        cursor = high
    if end - cursor >= config.min_dead_seconds:
        dead.append((cursor, end))
    elif active:
        active[-1] = (active[-1][0], end)
    else:
        active.append((start, end))
    if not active:
        logger.warning(
            "The dead-time pre-pass heard no play at all between %.0fs and %.0fs; tracking the "
            "whole run instead of skipping it.",
            start,
            end,
        )
        return None
    inside = int(((times >= start) & (times <= end)).sum())
    return DeadTimePlan(
        start=start,
        end=end,
        active=tuple(active),
        dead=tuple(dead),
        strikes=inside,
        settings=dict(config.__dict__),
    )


def outside_active(plan: DeadTimePlan, moments: Iterable[float]) -> list[float]:
    """Moments a plan would skip - for checking it against known shot times.

    Args:
        plan: The plan.
        moments: Times on the video clock.

    Returns:
        Those inside a dead stretch.
    """
    return [float(m) for m in moments if not plan.is_active(float(m))]
