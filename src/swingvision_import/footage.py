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
    workers: int = 4
    max_attempts: int = 4
    retry_base_delay: float = 2.0
    use_fallbacks: bool = True
    estimated_output_tokens: int = 2500
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
        "These are still frames from a fixed camera behind one end of a tennis "
        "court, all from a single point. Each is labelled with its time in "
        "seconds relative to the point's final shot (0.0).",
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

    "agrees" is deliberately hard to reach: the verdict must be usable and
    high-confidence, must have seen the same server the spreadsheet
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
    if verdict.refused or not verdict.usable or verdict.confidence != "high":
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
    disagreed: int = 0
    inconclusive: int = 0
    not_in_footage: int = 0
    failed: int = 0
    server_mismatches: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    notes: list[str] = field(default_factory=list)
