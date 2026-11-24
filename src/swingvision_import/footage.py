"""Checking reconstructed points against the match footage itself.

A reconstructed point's outcome is a guess from shot geometry, which is why
every one of them starts with needs_review=True. This module adds a second,
*independent* source: Claude looks at still frames pulled from the video
around each point and judges, on its own, who won and how. A point is
auto-confirmed only when that judgement and the reconstruction agree exactly
and the judgement is high-confidence - two sources agreeing is the bar, never
one source on its own. Anything else stays flagged for a human, now with the
footage verdict attached as context.

Three decisions matter for whether the result can be trusted:

- **Independence.** The model is told who served and which end each player
  is at - never what the spreadsheet recorded as the outcome. Showing it the
  recorded answer would invite it to agree, which would make "two sources
  agree" meaningless.
- **Alignment.** SwingVision's `Video Time` indexes the *full-length*
  recording. Its condensed rallies-only export is about a fifth as long and
  has no timestamp map, so seeking into it lands on the wrong rally. Frames
  from the wrong rally would still produce a confident-sounding verdict, so
  check_alignment() refuses footage that can't contain the recorded times,
  and the model independently reports who it saw serving - a mismatch with
  the spreadsheet marks that point inconclusive rather than trusted.
- **Vocabulary.** point_end_type is from the tracked player's side, and its
  buckets are the ones reconstruct.py actually assigns (an opponent's double
  fault is `return_winner`; an ace against us is `return_error`). The prompt
  defines them the same way, or nothing could ever agree.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import anthropic
import numpy as np

from ai.client import extract_text

from .raw import RawShotRow
from .records import VALID_END_TYPES, WINNING_END_TYPES, PointRecord
from .review_assist import is_transient_api_error

logger = logging.getLogger(__name__)

_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Per-token list prices (USD per million), for the pre-run estimate only.
_PRICES = {
    "claude-fable-5-2": (8.00, 40.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}


class FootageError(Exception):
    """Footage couldn't be read or a frame couldn't be extracted."""


class FootageAlignmentError(FootageError):
    """The footage can't contain the times the export says shots happened.

    Almost always SwingVision's condensed export: its `Video Time` values
    refer to the full-length recording, which is several times longer.
    """


@dataclass(frozen=True)
class FootageConfig:
    """Settings for a footage verification run.

    Attributes:
        model: Claude model id. Opus by default: this is a judgement from
            small, fast-moving objects in still frames, and a wrong
            agreement auto-confirms a point.
        effort: output_config effort. `high` because the call is
            intelligence-sensitive, not a formatting task.
        max_tokens: Output ceiling, which adaptive thinking also counts
            against.
        frame_width: Pixel width frames are scaled to (height follows the
            aspect ratio). 1280x720 costs 1,196 visual tokens a frame and
            leaves a far-court ball a few pixels wide; smaller is cheaper
            but the ball starts to disappear.
        end_window_before: Seconds before the last shot to start sampling.
        end_window_after: Seconds after the last shot to keep sampling -
            long enough to see the bounce and whether it was reached.
        end_frames: How many frames to take across that window.
        ball_path: Add one composite image of everything that moved
            around the finish (see compose_ball_path). Stills alone left
            the ball's landing unseen: a first real sample reached no
            high-confidence verdict, most saying so in as many words.
        ball_path_before: Seconds before the last shot the composite
            starts.
        ball_path_after: Seconds after the last shot it runs to.
        ball_path_width: Pixel width of the composite. Wider than the
            stills on purpose - a far-court ball is a few pixels across
            and the path is the one thing this image is for.
        corroborate: When the first reading falls short of high
            confidence, take a second reading from different evidence (the
            stills without the ball-path image) and let the two together
            decide - see corroborate(). Needs `ball_path`, since without
            it a second reading would only repeat the first.
        corroboration_confidence: The least confidence either reading may
            have for the pair to count: "medium" or "low". "low" is what
            the prompt calls a guess, so it is not the default.
        workers: Points checked concurrently. Kept below suggestion
            workers because each request carries ~10 large images, which
            reaches input-token rate limits sooner.
        max_attempts: Tries per point for transient API failures.
        retry_base_delay: First backoff in seconds; doubles per attempt.
        use_fallbacks: Send the server-side refusal fallback. If the API
            rejects it, the run switches it off and carries on.
        estimated_output_tokens: Output (incl. thinking) assumed per point
            for the *estimate only*; the run reports the real figure.
        source: What kind of footage is being checked.

            - "full": the full-length recording Video Time indexes.
              check_alignment() refuses anything shorter; frames are
              seeked to by time and judged directly.
            - "condensed_rallies": SwingVision's rallies-only export.
              Only ~70% of points are present (no serve- or
              return-decided ones), so the alignment guard is skipped
              AND every flagged point is first run through
              locate_point_in_condensed(). A point whose rally the
              retrieval cannot find is marked `not_in_footage` and
              stays flagged honestly, rather than being judged against
              some other rally in the window. Picked explicitly by
              the caller - there is no auto-detection.
        retrieval_window: Seconds on each side of the predicted point
            position to extract retrieval frames from. Only used when
            `source == "condensed_rallies"`.
        retrieval_frames: How many frames to sample across that window.
        retrieval_min_duration_padding: Floor on per-point video time
            when fitting the concatenation model, to account for
            single-shot points (a serve straight out still takes real
            seconds on screen).
    """

    model: str = "claude-opus-5"
    effort: str = "high"
    max_tokens: int = 16000
    frame_width: int = 1280
    end_window_before: float = 0.5
    end_window_after: float = 2.5
    end_frames: int = 8
    ball_path: bool = True
    ball_path_before: float = 0.3
    ball_path_after: float = 3.0
    ball_path_width: int = 1920
    corroborate: bool = True
    corroboration_confidence: str = "medium"
    workers: int = 4
    max_attempts: int = 4
    retry_base_delay: float = 2.0
    use_fallbacks: bool = True
    estimated_output_tokens: int = 4000
    source: str = "full"
    retrieval_window: float = 25.0
    retrieval_frames: int = 6
    retrieval_min_duration_padding: float = 4.0


@dataclass(frozen=True)
class Frame:
    """One still pulled from the footage.

    Attributes:
        label: What this frame shows, for the prompt ("serve", "end").
        offset: Seconds relative to the point's last shot (negative is
            before it), so the model can reason about order and timing.
        jpeg: The encoded image.
        caption: Text shown before the image in place of the usual
            "Frame N (label, t=...)" line, for an image that is not a
            single moment and needs explaining. `{n}` is replaced by the
            frame's number.
    """

    label: str
    offset: float
    jpeg: bytes
    caption: str | None = None


@dataclass(frozen=True)
class PointWindow:
    """Where in the footage one point lives, and who is where.

    Attributes:
        part: Which recording (source_part) the point is in.
        first_shot: video_time of the point's first shot (the serve).
        prev_shot: video_time of the second-to-last shot, if any - the
            shot the final one answered, which is what "forced" hinges on.
        last_shot: video_time of the point's final shot.
        tracked_serving: Whether the tracked player served.
        tracked_side: "near"/"far" end for the tracked player, if known.
        shot_count: Shots SwingVision recorded for the point.
    """

    part: int
    first_shot: float
    prev_shot: float | None
    last_shot: float
    tracked_serving: bool
    tracked_side: str | None
    shot_count: int


@dataclass(frozen=True)
class FootageVerdict:
    """What Claude concluded from the frames alone.

    Attributes:
        usable: Whether the frames actually showed how the point ended.
        server_seen: "tracked", "opponent" or "unclear" - an alignment
            check against the spreadsheet's own record of who served.
        tracked_seen_at: Which end the tracked player was identified at from
            their described appearance - "near", "far" or "unclear". A second
            alignment check, independent of the server one: it disagreeing
            with the sheet means the frames are of somebody else.
        winner: "tracked", "opponent" or "unclear".
        point_end_type: One of the point table's values, or "unclear".
        tracked_at_net: "yes", "no" or "unclear".
        confidence: "high", "medium" or "low".
        reasoning: One or two sentences, shown to the reviewer.
        refused: The model declined; the point is simply left for a human.
        input_tokens: Billed input tokens for the call.
        output_tokens: Billed output tokens for the call.
    """

    usable: bool
    server_seen: str
    winner: str
    point_end_type: str
    tracked_at_net: str
    confidence: str
    reasoning: str
    tracked_seen_at: str = "unclear"
    refused: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


VERDICT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "footage_usable": {"type": "boolean"},
        "server_seen": {"type": "string", "enum": ["tracked", "opponent", "unclear"]},
        "winner": {"type": "string", "enum": ["tracked", "opponent", "unclear"]},
        "point_end_type": {
            "type": "string",
            "enum": sorted(VALID_END_TYPES) + ["unclear"],
        },
        "tracked_at_net": {"type": "string", "enum": ["yes", "no", "unclear"]},
        "tracked_seen_at": {"type": "string", "enum": ["near", "far", "unclear"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reasoning": {"type": "string"},
    },
    "required": [
        "footage_usable",
        "server_seen",
        "winner",
        "point_end_type",
        "tracked_at_net",
        "confidence",
        "reasoning",
    ],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------
# Footage access
# --------------------------------------------------------------------------


def ffmpeg_exe() -> str:
    """Locates an ffmpeg binary.

    Prefers the one bundled by imageio-ffmpeg, so nothing has to be
    installed system-wide; falls back to whatever is on PATH.

    Returns:
        Path to an ffmpeg executable.

    Raises:
        FootageError: If neither is available.
    """
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        found = shutil.which("ffmpeg")
        if found:
            return found
    raise FootageError(
        "No ffmpeg available - install it with `pip install imageio-ffmpeg`."
    )


def probe_duration(video: Path, ffmpeg: str | None = None) -> float:
    """Reads a video's duration in seconds without decoding it.

    Args:
        video: The video file.
        ffmpeg: ffmpeg binary; located automatically if not given.

    Returns:
        The container's reported duration.

    Raises:
        FootageError: If the file can't be read as video.
    """
    result = subprocess.run(
        [ffmpeg or ffmpeg_exe(), "-hide_banner", "-i", str(video)],
        capture_output=True,
        text=True,
        errors="replace",
    )
    match = _DURATION.search(result.stderr)
    if not match:
        raise FootageError(f"{video.name}: couldn't read a duration from this file.")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


@dataclass(frozen=True)
class ClockOffset:
    """How far a recording's clock runs behind the sheet's `Video Time`.

    Attributes:
        seconds: Add this to a `Video Time` to get the moment in the video.
        strength: How far the best offset stands out from every other
            offset tried, in standard deviations. A match with no clear
            peak has nothing to trust.
        shots: How many shot times the measurement used.
    """

    seconds: float
    strength: float
    shots: int

    @property
    def trusted(self) -> bool:
        """Whether the peak is clear enough to shift frames by."""
        return self.strength >= _MIN_OFFSET_STRENGTH and self.shots >= _MIN_OFFSET_SHOTS


# Measured on a real 95-minute recording: the true offset scored 9-13
# standard deviations above the rest in every sixth of the match.
_MIN_OFFSET_STRENGTH = 6.0
_MIN_OFFSET_SHOTS = 20
_AUDIO_RATE = 8000
_AUDIO_HOP = 80  # 10 ms


def offset_from_audio(
    samples: np.ndarray,
    shot_times: list[float],
    *,
    sample_rate: int = _AUDIO_RATE,
    max_lag: float = 6.0,
) -> ClockOffset | None:
    """Finds the clock offset that best lines shot times up with ball strikes.

    A ball being struck is a sharp sound. Sliding the sheet's shot times
    along the recording's audio and scoring how much sudden sound sits
    under them gives one clear peak at the true offset. Confirmed on a
    real match: the sheet's times ran 1.99s ahead of the video, the same
    in every sixth of a 95-minute recording - enough that the frames
    "around the last shot" ended before the rally did.

    Args:
        samples: Mono audio samples.
        shot_times: `Video Time` of each shot, in seconds.
        sample_rate: Samples per second.
        max_lag: Largest offset, either way, to consider.

    Returns:
        The best offset and how clearly it stands out, or None when there
        is no audio or no shot inside the recording.
    """
    hop = max(1, sample_rate // 100)
    frames = len(samples) // hop
    if frames < 3 or not shot_times:
        return None
    edges = np.diff(samples.astype(np.float32), prepend=np.float32(samples[0]))
    energy = np.sqrt((edges[: frames * hop].reshape(frames, hop) ** 2).mean(axis=1))
    onset = np.maximum(0.0, np.diff(energy, prepend=energy[0]))
    ceiling = float(np.percentile(onset, 99))
    if ceiling <= 0:
        return None
    onset = np.minimum(onset / ceiling, 3.0)
    # A strike within 20 ms either side still counts.
    padded = np.pad(onset, 2)
    near = np.maximum.reduce([padded[i : i + frames] for i in range(5)])

    index = np.round(np.asarray(shot_times, dtype=np.float64) * 100).astype(np.int64)
    lags = np.arange(-int(max_lag * 100), int(max_lag * 100) + 1)
    scores = np.zeros(len(lags))
    for position, lag in enumerate(lags):
        at = index + lag
        at = at[(at >= 0) & (at < frames)]
        if len(at):
            scores[position] = near[at].mean()
    spread = float(scores.std())
    if spread <= 0:
        return None
    best = int(scores.argmax())
    return ClockOffset(
        seconds=float(lags[best]) / 100,
        strength=float((scores[best] - np.median(scores)) / spread),
        shots=len(shot_times),
    )


def measure_clock_offset(
    video: Path, shot_times: list[float], ffmpeg: str | None = None
) -> ClockOffset | None:
    """Measures a recording's clock offset from its own audio track.

    Free and deterministic: ffmpeg decodes the audio, nothing is sent
    anywhere. Never raises - a recording with no audio simply cannot be
    measured, and the caller then uses the sheet's times as they are.

    Args:
        video: The recording.
        shot_times: `Video Time` of each shot in that recording.
        ffmpeg: ffmpeg binary; located automatically if not given.

    Returns:
        The measured offset, or None if the audio could not be read.
    """
    try:
        result = subprocess.run(
            [
                ffmpeg or ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(_AUDIO_RATE),
                "-f",
                "s16le",
                "-",
            ],
            capture_output=True,
        )
    except OSError:
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    samples = np.frombuffer(result.stdout[: len(result.stdout) // 2 * 2], dtype=np.int16)
    return offset_from_audio(samples, shot_times)


def shift_window(window: PointWindow, seconds: float) -> PointWindow:
    """Moves a point's shot times onto the recording's own clock.

    Args:
        window: The point's window, in the sheet's `Video Time`.
        seconds: The recording's measured clock offset.

    Returns:
        The same window with every time shifted, never below zero.
    """
    return replace(
        window,
        first_shot=max(0.0, window.first_shot + seconds),
        prev_shot=None if window.prev_shot is None else max(0.0, window.prev_shot + seconds),
        last_shot=max(0.0, window.last_shot + seconds),
    )


def extract_frames(
    video: Path,
    start: float,
    duration: float,
    count: int,
    width: int,
    ffmpeg: str | None = None,
) -> list[bytes]:
    """Pulls `count` evenly spaced JPEG frames from a stretch of video.

    One ffmpeg process per stretch: it seeks to the keyframe before
    `start`, decodes forward, and samples with the fps filter - far cheaper
    than one seek per frame on long-GOP phone footage.

    Args:
        video: The video file.
        start: Seconds from the start of the file; clamped at 0.
        duration: Length of the stretch to sample, in seconds.
        count: Frames wanted across it.
        width: Output width in pixels; height keeps the aspect ratio.
        ffmpeg: ffmpeg binary; located automatically if not given.

    Returns:
        Up to `count` JPEGs in time order. Fewer if the stretch runs past
        the end of the file.

    Raises:
        FootageError: If ffmpeg itself fails.
    """
    start = max(0.0, start)
    rate = count / duration
    with tempfile.TemporaryDirectory(prefix="rallyai-frames-") as tmp:
        pattern = Path(tmp) / "f%03d.jpg"
        result = subprocess.run(
            [
                ffmpeg or ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(video),
                "-t",
                f"{duration:.3f}",
                "-an",
                "-vf",
                f"fps={rate:.6f},scale={width}:-2",
                "-frames:v",
                str(count),
                "-q:v",
                "3",
                str(pattern),
            ],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if result.returncode != 0:
            raise FootageError(f"{video.name}: ffmpeg failed: {result.stderr.strip()[:300]}")
        return [p.read_bytes() for p in sorted(Path(tmp).glob("f*.jpg"))]


# Tuned by eye on a real 1920x1080 recording, 30 frames a second.
_PATH_FPS = 30
_PATH_CHANGE = 20  # brightness change, of 255, that counts as movement
_PATH_BLOCK = 40  # pixels; movement is counted per block of this size
_PATH_CROWDED = 220  # moving pixels in a 3x3 block neighbourhood: a player
_PPM_HEADER = re.compile(rb"P6\s+(\d+)\s+(\d+)\s+255\s")


def _time_colour(fraction: float) -> np.ndarray:
    """Blue at the start of the stretch, through green, to red at its end."""
    fraction = min(1.0, max(0.0, fraction))
    return np.array(
        [255 * fraction, 255 * (1 - abs(fraction - 0.5) * 2), 255 * (1 - fraction)],
        dtype=np.float32,
    )


def _crowded(moving: np.ndarray) -> np.ndarray:
    """Marks the parts of a frame where a lot is moving at once.

    A ball is a few dozen moving pixels with nothing moving around it; a
    player is thousands. Counting per block and looking at each block's
    neighbourhood tells them apart without any shape analysis.

    Args:
        moving: Boolean mask of pixels that moved in this frame.

    Returns:
        A mask the same shape, True where the movement is player-sized.
    """
    height, width = moving.shape
    rows, cols = height // _PATH_BLOCK, width // _PATH_BLOCK
    crowded = np.zeros_like(moving)
    if not rows or not cols:
        return crowded
    inner = moving[: rows * _PATH_BLOCK, : cols * _PATH_BLOCK]
    counts = inner.reshape(rows, _PATH_BLOCK, cols, _PATH_BLOCK).sum(axis=(1, 3))
    padded = np.pad(counts, 1)
    near = sum(padded[dy : dy + rows, dx : dx + cols] for dy in range(3) for dx in range(3))
    blocks = np.repeat(np.repeat(near >= _PATH_CROWDED, _PATH_BLOCK, axis=0), _PATH_BLOCK, axis=1)
    crowded[: rows * _PATH_BLOCK, : cols * _PATH_BLOCK] = blocks
    return crowded


def compose_ball_path(frames: Iterable[np.ndarray], expected: int) -> np.ndarray | None:
    """Draws everything that moved across a run of frames onto the first one.

    A still frame shows where the ball is; it does not show where it went,
    and at this camera distance the bounce usually falls between frames.
    This puts the whole flight in one picture: each frame's moving pixels
    are painted in a colour for their time (blue first, red last), so a
    ball becomes a dotted line whose colour says which way it travelled
    and a bounce is the corner in that line. Ball-sized movement is
    painted solid; player-sized movement only tints, so a player running
    through does not bury the path.

    Movement is "differs from the frame before AND the frame after", which
    marks a thing where it is in that frame and leaves no ghost where it
    was. Deterministic, and it never sees the sheet.

    Args:
        frames: RGB frames (height x width x 3, uint8), in time order.
        expected: How many frames the stretch should hold, to scale the
            colours by; a short read still ends at the right colour.

    Returns:
        The composite as an RGB uint8 array, or None if there were fewer
        than three frames to compare.
    """
    canvas: np.ndarray | None = None
    recent: list[np.ndarray] = []
    index = 0
    compared = 0
    for frame in frames:
        if canvas is None:
            canvas = frame.astype(np.float32) * 0.6
        # Green carries most of a tennis ball's brightness.
        recent.append(frame[:, :, 1].astype(np.int16))
        if len(recent) == 3:
            before, current, after = recent
            moving = (np.abs(current - before) > _PATH_CHANGE) & (
                np.abs(current - after) > _PATH_CHANGE
            )
            compared += 1
            if moving.any():
                colour = _time_colour((index - 1) / max(1, expected - 1))
                crowded = moving & _crowded(moving)
                canvas[crowded] = canvas[crowded] * 0.7 + colour * 0.3
                small = moving & ~crowded
                # One pixel thicker, so a far-court ball survives any downscaling.
                thick = small.copy()
                thick[1:, :] |= small[:-1, :]
                thick[:-1, :] |= small[1:, :]
                thick[:, 1:] |= small[:, :-1]
                thick[:, :-1] |= small[:, 1:]
                canvas[thick] = colour
            recent.pop(0)
        index += 1
    if canvas is None or not compared:
        return None
    return canvas.clip(0, 255).astype(np.uint8)


def _read_ppm_frames(stream) -> Iterator[np.ndarray]:
    """Yields RGB frames from a stream of binary PPM images.

    PPM carries its own width and height, so nothing has to be known about
    the video (or its rotation) in advance.
    """
    while True:
        header = b""
        while header.count(b"\n") < 3:
            byte = stream.read(1)
            if not byte:
                return
            header += byte
        match = _PPM_HEADER.match(header)
        if match is None:
            return
        width, height = int(match.group(1)), int(match.group(2))
        raw = stream.read(width * height * 3)
        if len(raw) < width * height * 3:
            return
        yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)


def ball_path_image(
    video: Path, start: float, duration: float, width: int, ffmpeg: str | None = None
) -> bytes | None:
    """Builds the ball-path composite for one stretch of video, as a JPEG.

    Never raises: the composite is an aid on top of the stills, so a
    stretch it cannot be built for is judged from the stills alone.

    Args:
        video: The video file.
        start: Seconds from the start of the file; clamped at 0.
        duration: Length of the stretch, in seconds.
        width: Widest the composite may be; never upscaled.
        ffmpeg: ffmpeg binary; located automatically if not given.

    Returns:
        The JPEG, or None if it could not be built.
    """
    exe = ffmpeg or ffmpeg_exe()
    start = max(0.0, start)
    try:
        decode = subprocess.Popen(
            [
                exe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(video),
                "-t",
                f"{duration:.3f}",
                "-an",
                "-vf",
                f"fps={_PATH_FPS},scale='min({width},iw)':-2",
                "-f",
                "image2pipe",
                "-vcodec",
                "ppm",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    try:
        composite = compose_ball_path(
            _read_ppm_frames(decode.stdout), round(duration * _PATH_FPS)
        )
    finally:
        decode.stdout.close()
        decode.wait()
    if composite is None:
        return None
    height, wide = composite.shape[:2]
    try:
        encode = subprocess.run(
            [
                exe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{wide}x{height}",
                "-i",
                "-",
                "-frames:v",
                "1",
                "-q:v",
                "3",
                "-f",
                "image2",
                "-vcodec",
                "mjpeg",
                "-",
            ],
            input=composite.tobytes(),
            capture_output=True,
        )
    except OSError:
        return None
    if encode.returncode != 0 or not encode.stdout:
        return None
    return encode.stdout


def ball_path_caption(before: float, after: float) -> str:
    """Explains the composite to the judge. States no outcome.

    Args:
        before: Seconds before the last shot the composite starts.
        after: Seconds after the last shot it ends.

    Returns:
        The caption, with `{n}` where the frame number goes.
    """
    return (
        "Frame {n} (ball path) is NOT a single moment. It is the frame at "
        f"t={-before:+.1f}s with everything that moved between then and "
        f"t={after:+.1f}s drawn on top, coloured by when it moved: BLUE at the "
        "start, through GREEN, to RED at the end. A ball in flight shows as a "
        "line of small solid dots - its path. The colours give its direction "
        "(it travels from bluer to redder), a sharp corner low in the path is "
        "a bounce, and widely spaced dots mean it was moving fast. Players show "
        "as faint tinted smears covering where they moved. Use it for what the "
        "stills cannot show: where the final shot went, whether it cleared the "
        "net, where it bounced relative to the court lines, and whether the "
        "other player got to it or it bounced again. Be careful: balls and "
        "people on neighbouring courts leave paths too, as does a spare ball "
        "being cleared away after the point - read only a path that connects "
        "to this point's players, and check it against the stills."
    )


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def check_alignment(
    latest_time_by_part: dict[int, float],
    durations: list[float],
    tolerance: float = 2.0,
) -> None:
    """Refuses footage that cannot contain the export's shot times.

    A necessary check, not a sufficient one: it catches the condensed
    export (the common case) and a wrong or truncated file, but not a
    different full-length recording that happens to be long enough - the
    per-point server check in compare() is what catches that.

    Args:
        latest_time_by_part: Part index -> the latest video_time recorded
            for any shot in that part.
        durations: Each part's video duration, in part order.
        tolerance: Seconds of slack for container rounding.

    Raises:
        FootageAlignmentError: If any part's recorded shots run past the
            end of its video, or a part has no video at all.
    """
    problems = []
    for part, latest in sorted(latest_time_by_part.items()):
        if part >= len(durations):
            problems.append(f"part {part + 1} has shots but no matching video")
            continue
        if latest > durations[part] + tolerance:
            problems.append(
                f"part {part + 1}: the export records a shot at {latest:.0f}s "
                f"({latest / 60:.0f} min), but the video is only "
                f"{durations[part]:.0f}s ({durations[part] / 60:.0f} min) long"
            )
    if problems:
        # Name the cause, not just the symptom. Duration mismatch is the
        # actual signal; the filename isn't, since users commonly rename on
        # the way through cloud storage.
        raise FootageAlignmentError(
            "This footage doesn't line up with the export's timestamps ("
            + "; ".join(problems)
            + "). The usual cause is SwingVision's rally filter being on when "
            "the video is saved: the saved file then holds only the rallies, "
            "but Video Time still indexes the full-length recording. It is "
            "also a biased sample - the longest ~45% of points and none that "
            "ended on a serve or return - so it cannot check an ace, a double "
            "fault or a return error. Clear the rally filter in SwingVision "
            "before saving, and the saved video should run the full match "
            "length."
        )


def plan_point(shots: list[RawShotRow], host_name: str) -> PointWindow | None:
    """Works out where a point lives in the footage and who is where.

    Args:
        shots: The point's shots, as grouped by reconstruct.
        host_name: The tracked player's name as it appears on shots.

    Returns:
        The window, or None if the shots carry no video_time to locate it.
    """
    timed = sorted(
        (s for s in shots if s.video_time is not None), key=lambda s: s.video_time
    )
    if not timed:
        return None
    tracked_side = next((s.hit_side for s in timed if s.player == host_name and s.hit_side), None)
    if tracked_side is None:
        # Only the opponent hit a ball this point: we're at the other end.
        opponent_side = next((s.hit_side for s in timed if s.hit_side), None)
        tracked_side = {"near": "far", "far": "near"}.get(opponent_side or "")
    return PointWindow(
        part=timed[0].source_part,
        first_shot=timed[0].video_time,
        prev_shot=timed[-2].video_time if len(timed) >= 2 else None,
        last_shot=timed[-1].video_time,
        tracked_serving=timed[0].player == host_name,
        tracked_side=tracked_side,
        shot_count=len(timed),
    )


def frames_for_point(
    video: Path, window: PointWindow, config: FootageConfig, ffmpeg: str | None = None
) -> list[Frame]:
    """Extracts the frames a verdict is judged from.

    One frame at the serve (who is serving, from which end), one at the
    shot the final shot answered (was the player under pressure), then a
    run across the finish (where the ball went, and whether it was
    reached), and last one composite of everything that moved across the
    finish (the ball's whole path, which stills sample too thinly to show).

    Args:
        video: The recording this point is in.
        window: Where the point is.
        config: Frame settings.
        ffmpeg: ffmpeg binary; located automatically if not given.

    Returns:
        Frames in time order.
    """
    exe = ffmpeg or ffmpeg_exe()
    frames: list[Frame] = []

    serve = extract_frames(video, window.first_shot + 0.2, 0.2, 1, config.frame_width, exe)
    frames += [Frame("serve", window.first_shot + 0.2 - window.last_shot, j) for j in serve]

    if window.prev_shot is not None and window.prev_shot > window.first_shot:
        prev = extract_frames(video, window.prev_shot, 0.2, 1, config.frame_width, exe)
        frames += [Frame("previous shot", window.prev_shot - window.last_shot, j) for j in prev]

    span = config.end_window_before + config.end_window_after
    start = window.last_shot - config.end_window_before
    finish = extract_frames(video, start, span, config.end_frames, config.frame_width, exe)
    step = span / config.end_frames
    frames += [
        Frame("finish", -config.end_window_before + i * step, jpeg)
        for i, jpeg in enumerate(finish)
    ]

    if config.ball_path:
        composite = ball_path_image(
            video,
            window.last_shot - config.ball_path_before,
            config.ball_path_before + config.ball_path_after,
            config.ball_path_width,
            exe,
        )
        if composite is None:
            logger.warning("%s: no ball-path image at %.1fs", video.name, window.last_shot)
        else:
            frames.append(
                Frame(
                    "ball path",
                    -config.ball_path_before,
                    composite,
                    ball_path_caption(config.ball_path_before, config.ball_path_after),
                )
            )
    return frames


# --------------------------------------------------------------------------
# Judging
# --------------------------------------------------------------------------


def _prompt(
    window: PointWindow,
    reviewer_note: str | None,
    player_appearance: str | None = None,
) -> str:
    """The instructions that follow the frames.

    Deliberately omits what the spreadsheet recorded as the outcome.

    Appearance is given as a way to FIND the tracked player, never as a way
    to judge the point. It is worth stating separately from the court end
    because the two disagree in a useful way: ends swap at changeovers, so a
    described shirt that turns up at the end the sheet did not expect means
    the frames are of the wrong person or the wrong point.
    """
    side = window.tracked_side or "unknown"
    other = {"near": "far", "far": "near"}.get(side, "unknown")
    server = "the TRACKED player" if window.tracked_serving else "the OPPONENT"
    note = (
        "\nThe person who recorded the match wrote this about the recording "
        "itself - background on how it was captured, not an answer:\n"
        f"{reviewer_note.strip()}\n"
        if reviewer_note and reviewer_note.strip()
        else ""
    )
    lines = [
        "These are frames from a fixed camera behind one end of a tennis "
        "court, all from a single point. Each is labelled with its time in "
        "seconds relative to the point's final shot (0.0); any frame that is "
        "not a single moment says so in its own label.",
        "",
        "Context from the shot tracker (not an outcome):",
        f"- The TRACKED player is at the {side.upper()} end of the court; the "
        f"OPPONENT is at the {other.upper()} end.",
        *(
            [
                f"- The TRACKED player is described as: {player_appearance.strip()}. "
                "Use this to work out which player is which, and say where you "
                "actually saw them - do not assume the end above is right."
            ]
            if player_appearance and player_appearance.strip()
            else []
        ),
        f"- {server} serves this point.",
        f"- The tracker recorded {window.shot_count} shot(s) in the point.",
        note,
        "Judge from the frames alone how this point ended. Report:",
        '- server_seen: who you can see serving - "tracked", "opponent", or "unclear".',
        *(
            [
                "- tracked_seen_at: which end you actually saw the player matching "
                'that description at - "near", "far", or "unclear".'
            ]
            if player_appearance and player_appearance.strip()
            else []
        ),
        '- winner: who won the point - "tracked", "opponent", or "unclear".',
        "- point_end_type: always from the TRACKED player's side, using exactly "
        "these buckets:",
        "  - ace: the tracked player served and the point ended on the serve, "
        "won by the tracked player.",
        "  - double_fault: the tracked player served and both serves were faults.",
        "  - return_winner: the opponent served, the point ended on the serve or "
        "the tracked player's return, and the tracked player won it (including "
        "the opponent double-faulting).",
        "  - return_error: the opponent served, the point ended on the serve or "
        "the tracked player's return, and the tracked player lost it (including "
        "being aced).",
        "  - winner: the tracked player won the point any other way (an "
        "unreachable shot, or the opponent missing).",
        "  - unforced_error: the tracked player lost any other way by missing a "
        "shot they should have made, without real pressure.",
        "  - forced_error: the tracked player lost any other way under real "
        "pressure - including the opponent hitting a shot the tracked player "
        "couldn't reach.",
        "  - unclear: the frames don't show enough to say.",
        "- tracked_at_net: whether the tracked player came forward to the net "
        'during what you can see - "yes", "no", or "unclear".',
        "- footage_usable: false if the frames don't actually show the end of a "
        "point (wrong moment, ball out of view, camera blocked).",
        '- confidence: "high" only if the frames plainly show the outcome; a '
        'guess is "low".',
        "- reasoning: one or two sentences on what in the frames shows it.",
        "",
        "Where the frames don't settle something, say \"unclear\" rather than "
        "guessing - an unclear answer is sent to a person, and a wrong confident "
        "one is not.",
    ]
    return "\n".join(lines)


def _content(
    frames: list[Frame],
    window: PointWindow,
    reviewer_note: str | None,
    player_appearance: str | None = None,
) -> list[dict]:
    """Builds the user turn: labelled frames first, then the instructions."""
    blocks: list[dict] = []
    for i, frame in enumerate(frames, start=1):
        if frame.caption:
            caption = frame.caption.replace("{n}", str(i))
        else:
            caption = f"Frame {i} ({frame.label}, t={frame.offset:+.2f}s):"
        blocks.append({"type": "text", "text": caption})
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(frame.jpeg).decode("ascii"),
                },
            }
        )
    blocks.append(
        {"type": "text", "text": _prompt(window, reviewer_note, player_appearance)}
    )
    return blocks


class FallbackSwitch:
    """Remembers, across worker threads, whether fallbacks were rejected."""

    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def disable(self) -> None:
        with self._lock:
            self._enabled = False


def judge_point(
    client,
    config: FootageConfig,
    window: PointWindow,
    frames: list[Frame],
    reviewer_note: str | None = None,
    fallbacks: FallbackSwitch | None = None,
    player_appearance: str | None = None,
) -> FootageVerdict:
    """Asks Claude for a verdict on one point from its frames.

    Args:
        client: An anthropic.Anthropic-shaped client (injected, so tests
            never call the real API). Needs `beta.messages.create`.
        config: Model and retry settings.
        window: Where the point is, and who is where.
        frames: The frames to judge from.
        reviewer_note: The uploader's note on this import, if any.
        fallbacks: Shared switch for the refusal-fallback parameter.
        player_appearance: How the tracked player looked, if the uploader
            said - used to identify them, never to judge the point.

    Returns:
        The verdict. A refusal comes back as an unusable verdict rather
        than an exception, so the point simply stays with the reviewer.

    Raises:
        FootageError: On a non-transient API failure, repeated transient
            failures, or a response that doesn't match the schema.
    """
    switch = fallbacks or FallbackSwitch(config.use_fallbacks)
    content = _content(frames, window, reviewer_note, player_appearance)
    attempts = max(1, config.max_attempts)
    response = None
    attempt = 0
    while attempt < attempts:
        attempt += 1
        kwargs: dict = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "output_config": {
                "effort": config.effort,
                "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
            },
            "messages": [{"role": "user", "content": content}],
        }
        if switch.enabled:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        try:
            response = client.beta.messages.create(**kwargs)
            break
        except anthropic.BadRequestError as exc:
            if switch.enabled and "fallback" in str(exc).lower():
                logger.warning("API rejected refusal fallbacks; continuing without them: %s", exc)
                switch.disable()
                attempt -= 1  # not the call's fault - don't spend an attempt on it
                continue
            raise FootageError(f"Footage check rejected by the API: {exc}") from exc
        except anthropic.APIError as exc:
            if attempt >= attempts or not is_transient_api_error(exc):
                raise FootageError(f"Footage check failed: {exc}") from exc
            delay = config.retry_base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Transient footage-check failure (attempt %d/%d), retrying in %.0fs: %s",
                attempt,
                attempts,
                delay,
                exc,
            )
            time.sleep(delay)

    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    if getattr(response, "stop_reason", None) == "refusal":
        return FootageVerdict(
            usable=False,
            server_seen="unclear",
            winner="unclear",
            point_end_type="unclear",
            tracked_at_net="unclear",
            confidence="low",
            reasoning="The model declined to assess these frames.",
            refused=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    raw = ""
    try:
        raw = extract_text(response)
        data = json.loads(raw)
        return FootageVerdict(
            usable=bool(data["footage_usable"]),
            server_seen=data["server_seen"],
            tracked_seen_at=data.get("tracked_seen_at", "unclear"),
            winner=data["winner"],
            point_end_type=data["point_end_type"],
            tracked_at_net=data["tracked_at_net"],
            confidence=data["confidence"],
            reasoning=str(data["reasoning"]),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise FootageError(f"Footage check returned an unreadable verdict: {raw[:200]!r}") from exc


def compare(verdict: FootageVerdict, point: PointRecord, window: PointWindow) -> str:
    """Decides whether the footage corroborates the recorded outcome.

    This is the rule for one reading standing alone; corroborate() is the
    rule for two. "agrees" is deliberately hard to reach: the verdict must
    be usable and high-confidence, must have seen the same server the spreadsheet
    recorded (otherwise the frames may be from another point), must have
    found the tracked player at the end the sheet puts them (when their
    appearance was described), must be internally consistent, must not
    contradict the recorded net approach, and must match the recorded
    outcome exactly.

    Args:
        verdict: The footage verdict.
        point: The reconstructed point as currently recorded.
        window: The point's window (for who served).

    Returns:
        "agrees", "disagrees", or "inconclusive".
    """
    if verdict.confidence != "high":
        return "inconclusive"
    return _reading(verdict, point, window)


_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def _confident_enough(verdict: FootageVerdict, minimum: str) -> bool:
    """Whether a reading's own confidence reaches `minimum`."""
    return _CONFIDENCE_RANK.get(verdict.confidence, 0) >= _CONFIDENCE_RANK.get(minimum, 1)


def wants_second_reading(
    first: FootageVerdict, point: PointRecord, window: PointWindow, minimum: str = "medium"
) -> bool:
    """Whether a second reading could change what happens to this point.

    Not when the first is already high-confidence (it decides alone), not
    when it is below the floor (no pair including it can count), and not
    when it is unreadable or failed an alignment check (a second look at
    the wrong rally settles nothing). So the second call is only paid for
    where it can clear a flag or confirm a disagreement.

    Args:
        first: The first reading.
        point: The reconstructed point as currently recorded.
        window: The point's window.
        minimum: The corroboration floor, "medium" or "low".

    Returns:
        True if a second reading is worth taking.
    """
    if first.confidence == "high" or not _confident_enough(first, minimum):
        return False
    return _reading(first, point, window) != "inconclusive"


def corroborate(
    first: FootageVerdict,
    second: FootageVerdict,
    point: PointRecord,
    window: PointWindow,
    minimum: str = "medium",
) -> str:
    """Decides a point from two readings that were given different evidence.

    Measured on a real recording, no single reading reached high
    confidence: most points turn on a line call at the far baseline, which
    one camera behind the near baseline does not settle. What a single
    reading cannot supply, agreement can - so a flag also clears when
    three things line up: the sheet's reconstruction, a reading from the
    stills plus the ball-path image, and a reading from the stills alone.

    This is deliberately not as independent as it sounds, and the limit is
    stated rather than hidden: both readings come from the same model and
    the same video, so a blind spot they share would fool both. The sheet
    is the genuinely independent third. Each reading must still pass every
    check compare() makes except the high-confidence one, and neither may
    fall below `minimum`.

    Two readings that agree with each other *against* the sheet come back
    "disagrees": the point stays flagged with both attached, and nothing
    is overwritten.

    Args:
        first: The reading that saw the ball-path image.
        second: The reading from the stills alone.
        point: The reconstructed point as currently recorded.
        window: The point's window.
        minimum: The least confidence either reading may have.

    Returns:
        "agrees", "disagrees", or "inconclusive".
    """
    if not (_confident_enough(first, minimum) and _confident_enough(second, minimum)):
        return "inconclusive"
    outcomes = (_reading(first, point, window), _reading(second, point, window))
    if outcomes == ("agrees", "agrees"):
        return "agrees"
    same_story = (first.winner, first.point_end_type) == (second.winner, second.point_end_type)
    if outcomes == ("disagrees", "disagrees") and same_story:
        return "disagrees"
    return "inconclusive"


def _reading(verdict: FootageVerdict, point: PointRecord, window: PointWindow) -> str:
    """Whether one reading matches the recorded outcome, confidence aside.

    Every check compare() describes except the confidence bar, which the
    callers apply: compare() demands high of a reading standing alone, and
    corroborate() accepts less of two that agree.

    Args:
        verdict: The footage verdict.
        point: The reconstructed point as currently recorded.
        window: The point's window (for who served).

    Returns:
        "agrees", "disagrees", or "inconclusive".
    """
    if verdict.refused or not verdict.usable:
        return "inconclusive"
    if "unclear" in (verdict.winner, verdict.point_end_type):
        return "inconclusive"
    expected_server = "tracked" if window.tracked_serving else "opponent"
    if verdict.server_seen != expected_server:
        return "inconclusive"
    # Identity by appearance, when it was given: ends swap at changeovers,
    # so the player described turning up at the other end means the frames
    # are of the wrong person - not something to resolve by judging harder.
    if (
        verdict.tracked_seen_at != "unclear"
        and window.tracked_side
        and verdict.tracked_seen_at != window.tracked_side
    ):
        return "inconclusive"
    footage_won = verdict.winner == "tracked"
    if (verdict.point_end_type in WINNING_END_TYPES) != footage_won:
        return "inconclusive"  # the verdict contradicts itself
    # One-directional on purpose: the frames skip the middle of a rally, so
    # "didn't see them at the net" is no evidence a volley never happened,
    # but "saw them at the net" does contradict a record that says no.
    if verdict.tracked_at_net == "yes" and not point.net_approach:
        return "disagrees"
    if footage_won == point.point_won and verdict.point_end_type == point.point_end_type:
        return "agrees"
    return "disagrees"


# --------------------------------------------------------------------------
# Content-addressed retrieval for the condensed rallies-only export
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ShotFingerprint:
    """One shot's content-addressable features, from the sheet only.

    The retrieval prompt uses these to answer "is the point whose shots
    are listed present anywhere in these frames?" without ever seeing the
    recorded outcome - the independence that makes a positive answer
    trustworthy.

    Attributes:
        who: "tracked" or "opponent" - who hit this shot.
        stroke: "Serve", "Forehand", "Backhand", etc.
        spin: "Flat", "Topspin", "Slice", or None.
        speed_mph: Shot speed rounded to the nearest MPH, or None.
    """

    who: str
    stroke: str
    spin: str | None
    speed_mph: int | None


@dataclass(frozen=True)
class LocationVerdict:
    """What the retrieval call concluded for one point.

    Attributes:
        found: Whether the rally was identified in the frames.
        offset_s: Seconds from the predicted window start to the located
            rally, when found. Meaningless when found is False.
        confidence: "high", "medium" or "low".
        reasoning: One or two sentences, shown to the reviewer.
        input_tokens: Billed input tokens for the call.
        output_tokens: Billed output tokens for the call.
    """

    found: bool
    offset_s: float
    confidence: str
    reasoning: str
    input_tokens: int = 0
    output_tokens: int = 0


LOCATION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "rally_present": {"type": "boolean"},
        "offset_seconds": {"type": "number"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reasoning": {"type": "string"},
    },
    "required": ["rally_present", "offset_seconds", "confidence", "reasoning"],
    "additionalProperties": False,
}


def fingerprint_point(shots: list[RawShotRow], host_name: str) -> list[ShotFingerprint]:
    """Builds the shot fingerprint sequence for one point.

    Reads only sheet fields the retrieval prompt may legitimately see.
    The point outcome (winner/error/at-net) is NOT in the fingerprint
    and must never reach the prompt from any other path - the retrieval
    independence test pins the parameter allowlist exactly.

    Args:
        shots: The point's shots, in order.
        host_name: The tracked player's name as it appears on shots.

    Returns:
        One fingerprint per shot, in shot order.
    """
    fingerprints = []
    for shot in sorted(shots, key=lambda s: s.video_time or 0.0):
        if shot.video_time is None:
            continue
        who = "tracked" if shot.player == host_name else "opponent"
        speed = round(shot.speed_mph) if shot.speed_mph is not None else None
        fingerprints.append(
            ShotFingerprint(
                who=who,
                stroke=shot.stroke or "",
                spin=shot.spin,
                speed_mph=speed,
            )
        )
    return fingerprints


def predict_condensed_center(
    shots_by_point: dict[int, list[RawShotRow]],
    target_point: int,
    video_duration: float,
    min_padding: float,
) -> float | None:
    """Predicts the midpoint of a point's rally in the condensed video.

    Separated from the window computation so the caller can pair this
    midpoint with the config-controlled retrieval_window instead of a
    locally-derived half-width.

    Args:
        shots_by_point: Point number -> its shots.
        target_point: Which point to predict a midpoint for.
        video_duration: Seconds. Used as the sum to solve padding against.
        min_padding: Floor on a single clip's duration.

    Returns:
        Seconds into the video, or None if the point isn't in the sheet.
    """
    if target_point not in shots_by_point:
        return None
    order = sorted(shots_by_point)
    spans: list[float] = []
    for point in order:
        shots = shots_by_point[point]
        times = [s.video_time for s in shots if s.video_time is not None]
        spans.append(max(times) - min(times) if len(times) >= 2 else 0.0)
    lo, hi = 0.0, 60.0
    for _ in range(60):
        mid = (lo + hi) / 2
        total = sum(max(span + mid, min_padding) for span in spans)
        if total < video_duration:
            lo = mid
        else:
            hi = mid
    padding = (lo + hi) / 2
    cursor = 0.0
    for point, span in zip(order, spans, strict=True):
        duration = max(span + padding, min_padding)
        if point == target_point:
            return cursor + duration / 2
        cursor += duration
    return None


def _retrieval_prompt(
    fingerprints: list[ShotFingerprint], window_seconds: float, frame_count: int
) -> str:
    """Instructions for the content-addressed retrieval call.

    Describes the shot fingerprint to look for, asks whether these frames
    show that rally, and - if yes - roughly when within the window.
    Deliberately does NOT include the point's recorded outcome, which the
    judge call still has to form independently later.
    """
    shot_lines = []
    for i, shot in enumerate(fingerprints, start=1):
        bits = [shot.who.upper(), shot.spin or "", shot.stroke]
        label = " ".join(bit for bit in bits if bit)
        if shot.speed_mph is not None:
            label += f" ({shot.speed_mph} MPH)"
        shot_lines.append(f"  {i}. {label}")

    return "\n".join(
        [
            "These are still frames from a fixed camera behind one end of a "
            "tennis court, spanning roughly {:.0f} seconds of SwingVision's "
            "'filtered rallies' export. The export is the match's rallies "
            "concatenated with no overlay and no between-point footage, so "
            "adjacent frames may cross from one rally into the next.".format(
                window_seconds
            ),
            "",
            "Find the ONE rally whose shots match this sequence (from the "
            "spreadsheet, not the video):",
            *shot_lines,
            "",
            "Report:",
            '- rally_present: true if the described rally is in these frames.',
            "- offset_seconds: if present, approximately where in these "
            f"frames the rally starts. The frames span 0..{window_seconds:.0f}s "
            f"across {frame_count} evenly-spaced samples; count from the "
            "first frame.",
            '- confidence: "high", "medium" or "low".',
            "- reasoning: one or two sentences on how you decided.",
            "",
            "If no rally in these frames matches the described shot "
            "sequence - including when the frames cover a completely "
            "different rally - answer rally_present=false. Do NOT guess at "
            "the outcome, and do NOT substitute the nearest rally; a "
            "missing rally stays flagged for a human rather than being "
            "judged from the wrong footage.",
            "",
            "Respond with ONLY a JSON object matching this schema. No prose, "
            "no markdown code fences.",
            json.dumps(LOCATION_SCHEMA, indent=2),
        ]
    )


def _retrieval_content(
    frames: list[Frame],
    fingerprints: list[ShotFingerprint],
    window_seconds: float,
) -> list[dict]:
    """Builds the retrieval API message content: frames then instructions."""
    blocks: list[dict] = []
    for i, frame in enumerate(frames):
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(frame.jpeg).decode("ascii"),
                },
            }
        )
        blocks.append({"type": "text", "text": f"Frame {i + 1}"})
    blocks.append(
        {"type": "text", "text": _retrieval_prompt(fingerprints, window_seconds, len(frames))}
    )
    return blocks


def locate_point_in_condensed(
    client,
    config: FootageConfig,
    fingerprints: list[ShotFingerprint],
    frames: list[Frame],
    window_seconds: float,
    fallbacks: FallbackSwitch | None = None,
) -> LocationVerdict:
    """Asks Claude whether a specific rally is present in a window of frames.

    Args:
        client: An anthropic.Anthropic-shaped client.
        config: Model / retry settings.
        fingerprints: The shot sequence that identifies the target rally.
        frames: Frames across the predicted window, in time order.
        window_seconds: Total span of the window the frames cover.
        fallbacks: Shared refusal-fallback switch.

    Returns:
        The retrieval verdict. A LocationVerdict with found=False is the
        honest answer when the rally isn't in these frames; nothing
        downstream is allowed to turn it into a judgement.

    Raises:
        FootageError: On a hard failure that no retry can recover.
    """
    switch = fallbacks or FallbackSwitch(config.use_fallbacks)
    content = _retrieval_content(frames, fingerprints, window_seconds)
    attempt = 0
    last_exc: Exception | None = None
    while attempt < config.max_attempts:
        attempt += 1
        # Same request shape as judge_point: output_config with an effort and
        # a JSON schema, not thinking={"type":"enabled"}. claude-opus-5
        # rejects the old shape, and keeping the two call sites aligned
        # means a future shape change only needs one edit.
        kwargs: dict = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "output_config": {
                "effort": config.effort,
                "format": {"type": "json_schema", "schema": LOCATION_SCHEMA},
            },
            "messages": [{"role": "user", "content": content}],
        }
        if switch.enabled:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        try:
            response = client.beta.messages.create(**kwargs)
            raw = extract_text(response)
            data = json.loads(_strip_markdown_fence(raw))
            return LocationVerdict(
                found=bool(data["rally_present"]),
                offset_s=float(data.get("offset_seconds") or 0.0),
                confidence=data.get("confidence", "low"),
                reasoning=data.get("reasoning", ""),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        except anthropic.BadRequestError as exc:
            if switch.enabled and "fallback" in str(exc).lower():
                switch.disable()
                attempt -= 1  # not the call's fault
                continue
            last_exc = exc
            break
        except (anthropic.APIError, OSError) as exc:
            last_exc = exc
            if not is_transient_api_error(exc):
                break
            if attempt < config.max_attempts:
                time.sleep(config.retry_base_delay * (2 ** (attempt - 1)))
        except (ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            last_exc = exc
            break
    raise FootageError(f"Retrieval call failed: {last_exc}") from last_exc


# Alias so a local test fake can reuse the same helper rather than
# re-implementing JSON extraction twice. The judge call defines its own
# version at module level; both strip the same markdown-fence artefact.
def _strip_markdown_fence(text: str) -> str:
    """Strips a leading/trailing markdown code fence, if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.split("\n")
    if len(lines) > 1 and lines[-1].strip() == "```":
        lines = lines[1:-1]
    else:
        lines = lines[1:]
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def frame_tokens(width: int, height: int) -> int:
    """Visual tokens for one image: ceil(w/28) x ceil(h/28).

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        The token count the API bills for it (below the downscale limits).
    """
    return math.ceil(width / 28) * math.ceil(height / 28)


@dataclass(frozen=True)
class CostEstimate:
    """A pre-run estimate. The run itself reports real usage.

    Attributes:
        points: Points that would be checked.
        input_tokens_per_point: Frames plus instructions.
        output_tokens_per_point: The configured assumption.
        usd: Estimated total at list price.
        model: Which model the estimate is for.
    """

    points: int
    input_tokens_per_point: int
    output_tokens_per_point: int
    usd: float
    model: str


def estimate_cost(points: int, config: FootageConfig, aspect: float = 16 / 9) -> CostEstimate:
    """Estimates what checking `points` points will cost.

    Assumes the most frames a point can get (serve + previous shot +
    finish) and, with corroboration on, that every point needs its second
    reading; single-shot points use fewer frames and a confident or
    unreadable first reading takes no second, so this errs high.

    Args:
        points: How many points would be checked.
        config: The run's settings.
        aspect: Width / height of the footage.

    Returns:
        The estimate.
    """
    height = math.ceil(config.frame_width / aspect / 2) * 2
    per_frame = frame_tokens(config.frame_width, height)
    frames = config.end_frames + 2
    input_per_point = frames * per_frame + frames * 12 + 1100  # images + labels + prompt
    if config.ball_path:
        path_height = math.ceil(config.ball_path_width / aspect / 2) * 2
        input_per_point += frame_tokens(config.ball_path_width, path_height) + 260
    output_per_point = config.estimated_output_tokens
    if config.ball_path and config.corroborate:
        # Worst case: every point gets the second, stills-only reading.
        input_per_point += frames * per_frame + frames * 12 + 1100
        output_per_point *= 2
    in_price, out_price = _PRICES.get(config.model, _PRICES["claude-opus-5"])
    usd = points * (
        input_per_point * in_price + output_per_point * out_price
    ) / 1_000_000
    return CostEstimate(
        points=points,
        input_tokens_per_point=input_per_point,
        output_tokens_per_point=output_per_point,
        usd=round(usd, 2),
        model=config.model,
    )


def actual_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """What a run actually cost at list price, from billed usage.

    Args:
        input_tokens: Total billed input tokens.
        output_tokens: Total billed output tokens.
        model: The model used.

    Returns:
        USD, rounded to cents.
    """
    in_price, out_price = _PRICES.get(model, _PRICES["claude-opus-5"])
    return round((input_tokens * in_price + output_tokens * out_price) / 1_000_000, 2)


@dataclass
class FootageRunSummary:
    """What a verification run did, with coverage against points played.

    Every surface that shows these figures must also show the denominator
    (`points_played`) alongside them. A "20 confirmed" read without the
    "of 123 points played, 53 were not in this footage" context is the
    exact way the tool misleads about its own reach - see the
    feedback-honest-coverage rule in memory.

    Attributes:
        points_played: Total points with shots in this match - the
            denominator for every coverage figure below. Set once per run
            from the record, not accumulated as the run proceeds.
        footage_rally_count: Rallies the footage itself contains, as the
            uploader counted them (MatchRecord.footage_rally_count), or
            None if they did not say. The ceiling on `checked`: shown
            beside `points_played` so the gap between "played" and "in
            the footage at all" is stated rather than discovered.
        checked: Points a verdict was obtained for.
        agreed: Auto-confirmed - footage and reconstruction matched.
        disagreed: Footage contradicted the record; left for review.
        inconclusive: Footage couldn't settle it; left for review.
        not_in_footage: Points whose rally the condensed export does not
            contain, reported honestly instead of guessed at. Stays
            `needs_review=True` for the human.
        failed: Points whose check errored; left untouched.
        server_mismatches: Verdicts that saw a different server than the
            spreadsheet recorded - a sign of misaligned footage.
        input_tokens: Total billed input tokens.
        output_tokens: Total billed output tokens.
        usd: Actual cost at list price.
        notes: Anything the reviewer should know about the run itself.
    """

    points_played: int = 0
    footage_rally_count: int | None = None
    checked: int = 0
    agreed: int = 0
    corroborated: int = 0
    second_readings: int = 0
    disagreed: int = 0
    inconclusive: int = 0
    not_in_footage: int = 0
    failed: int = 0
    server_mismatches: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    notes: list[str] = field(default_factory=list)
