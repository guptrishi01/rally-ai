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
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from ai.client import extract_text

from .raw import RawShotRow
from .records import VALID_END_TYPES, WINNING_END_TYPES, PointRecord
from .review_assist import is_transient_api_error

logger = logging.getLogger(__name__)

_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Per-token list prices (USD per million), for the pre-run estimate only.
_PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
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
        workers: Points checked concurrently. Kept below suggestion
            workers because each request carries ~10 large images, which
            reaches input-token rate limits sooner.
        max_attempts: Tries per point for transient API failures.
        retry_base_delay: First backoff in seconds; doubles per attempt.
        use_fallbacks: Send the server-side refusal fallback. If the API
            rejects it, the run switches it off and carries on.
        estimated_output_tokens: Output (incl. thinking) assumed per point
            for the *estimate only*; the run reports the real figure.
    """

    model: str = "claude-opus-5"
    effort: str = "high"
    max_tokens: int = 16000
    frame_width: int = 1280
    end_window_before: float = 0.5
    end_window_after: float = 2.5
    end_frames: int = 8
    workers: int = 4
    max_attempts: int = 4
    retry_base_delay: float = 2.0
    use_fallbacks: bool = True
    estimated_output_tokens: int = 2500


@dataclass(frozen=True)
class Frame:
    """One still pulled from the footage.

    Attributes:
        label: What this frame shows, for the prompt ("serve", "end").
        offset: Seconds relative to the point's last shot (negative is
            before it), so the model can reason about order and timing.
        jpeg: The encoded image.
    """

    label: str
    offset: float
    jpeg: bytes


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
                f"part {part + 1}: the export records a shot at {latest:.0f}s, "
                f"but the video is only {durations[part]:.0f}s long"
            )
    if problems:
        raise FootageAlignmentError(
            "This footage doesn't line up with the export's timestamps ("
            + "; ".join(problems)
            + "). SwingVision's condensed rallies-only video is the usual cause: "
            "its Video Time values refer to the full-length recording. Upload the "
            "full-length video for each part to check points against footage."
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
    reached).

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
    return frames


# --------------------------------------------------------------------------
# Judging
# --------------------------------------------------------------------------


def _prompt(window: PointWindow, reviewer_note: str | None) -> str:
    """The instructions that follow the frames.

    Deliberately omits what the spreadsheet recorded as the outcome.
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
        "These are still frames from a fixed camera behind one end of a tennis "
        "court, all from a single point. Each is labelled with its time in "
        "seconds relative to the point's final shot (0.0).",
        "",
        "Context from the shot tracker (not an outcome):",
        f"- The TRACKED player is at the {side.upper()} end of the court; the "
        f"OPPONENT is at the {other.upper()} end.",
        f"- {server} serves this point.",
        f"- The tracker recorded {window.shot_count} shot(s) in the point.",
        note,
        "Judge from the frames alone how this point ended. Report:",
        '- server_seen: who you can see serving - "tracked", "opponent", or "unclear".',
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


def _content(frames: list[Frame], window: PointWindow, reviewer_note: str | None) -> list[dict]:
    """Builds the user turn: labelled frames first, then the instructions."""
    blocks: list[dict] = []
    for i, frame in enumerate(frames, start=1):
        blocks.append(
            {"type": "text", "text": f"Frame {i} ({frame.label}, t={frame.offset:+.2f}s):"}
        )
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
    blocks.append({"type": "text", "text": _prompt(window, reviewer_note)})
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

    Returns:
        The verdict. A refusal comes back as an unusable verdict rather
        than an exception, so the point simply stays with the reviewer.

    Raises:
        FootageError: On a non-transient API failure, repeated transient
            failures, or a response that doesn't match the schema.
    """
    switch = fallbacks or FallbackSwitch(config.use_fallbacks)
    content = _content(frames, window, reviewer_note)
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

    "agrees" is deliberately hard to reach: the verdict must be usable and
    high-confidence, must have seen the same server the spreadsheet
    recorded (otherwise the frames may be from another point), must be
    internally consistent, must not contradict the recorded net approach,
    and must match the recorded outcome exactly.

    Args:
        verdict: The footage verdict.
        point: The reconstructed point as currently recorded.
        window: The point's window (for who served).

    Returns:
        "agrees", "disagrees", or "inconclusive".
    """
    if verdict.refused or not verdict.usable or verdict.confidence != "high":
        return "inconclusive"
    if "unclear" in (verdict.winner, verdict.point_end_type):
        return "inconclusive"
    expected_server = "tracked" if window.tracked_serving else "opponent"
    if verdict.server_seen != expected_server:
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
    finish); single-shot points use fewer, so this errs high.

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
    in_price, out_price = _PRICES.get(config.model, _PRICES["claude-opus-5"])
    usd = points * (
        input_per_point * in_price + config.estimated_output_tokens * out_price
    ) / 1_000_000
    return CostEstimate(
        points=points,
        input_tokens_per_point=input_per_point,
        output_tokens_per_point=config.estimated_output_tokens,
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
    """What a verification run did.

    Attributes:
        checked: Points a verdict was obtained for.
        agreed: Auto-confirmed - footage and reconstruction matched.
        disagreed: Footage contradicted the record; left for review.
        inconclusive: Footage couldn't settle it; left for review.
        failed: Points whose check errored; left untouched.
        server_mismatches: Verdicts that saw a different server than the
            spreadsheet recorded - a sign of misaligned footage.
        input_tokens: Total billed input tokens.
        output_tokens: Total billed output tokens.
        usd: Actual cost at list price.
        notes: Anything the reviewer should know about the run itself.
    """

    checked: int = 0
    agreed: int = 0
    disagreed: int = 0
    inconclusive: int = 0
    failed: int = 0
    server_mismatches: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    notes: list[str] = field(default_factory=list)
