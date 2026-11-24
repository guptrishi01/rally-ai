"""Reading a recording: what it is, when each frame is, and the frames themselves.

Everything timing-related downstream - a shot's `video_time`, a ball's speed -
depends on *when* each frame was actually shown, not on an assumed constant
frame rate. So this module never invents a timestamp: `build_pts_index`
reads every frame's presentation timestamp (PTS) from the container without
decoding, and `iter_frames` hands each decoded frame out with the PTS ffmpeg
itself reports for it.

**The clock.** A PTS here is the video stream's own container timestamp, in
seconds - the same clock in the index and in the frame iterator. That is not
quite "seconds since the first frame", nor quite what `ffmpeg -ss t` (and so
`swingvision_import.footage`) seeks to: ffmpeg adds the container's
`start_time` to `-ss`. On the real recording this was built against that
offset is one 60fps frame (0.0167s); `VideoInfo.start_time` carries it so a
later stage can convert exactly instead of assuming zero.

**Variable frame rate.** Phone recordings drop frames. The real recording is
nominally 60fps but 610 of its 342,928 frame gaps are 2 to 21 frames long.
`PtsIndex.stats` measures this; `load_video` flags it on `VideoInfo.is_vfr`
and, only if `VideoConfig.reject_vfr` asks, refuses the file with a
`VariableFrameRateError` that says how irregular it is.

Decoding reuses `swingvision_import.footage`'s ffmpeg helpers (`ffmpeg_exe`,
`probe_duration`, `_read_ppm_frames`) rather than duplicating them. Frames
come out display-oriented: ffmpeg applies the container's rotation itself.
"""

from __future__ import annotations

import hashlib
import logging
import queue
import re
import subprocess
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from swingvision_import import footage

from .config import VideoConfig

logger = logging.getLogger(__name__)

_VIDEO_STREAM = re.compile(r"Stream #\d+:\d+.*?: Video: (\w+).*?[ ,](\d{2,5})x(\d{2,5})[ ,\[]")
_AVERAGE_FPS = re.compile(r"([\d.]+) fps")
_NOMINAL_FPS = re.compile(r"([\d.]+)(k?) tbr")
_START = re.compile(r"start:\s*(-?[\d.]+)")
_ROTATION = re.compile(r"rotation of (-?[\d.]+) degrees")
_ROTATE_TAG = re.compile(r"^\s*rotate\s*:\s*(-?\d+)", re.MULTILINE)
_SHOWINFO_BASE = re.compile(r"showinfo.*config in time_base:\s*(\d+)/(\d+)")
_SHOWINFO_FRAME = re.compile(r"showinfo.*\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:\s*(-?[\d.]+)")
_CRC_BASE = re.compile(r"#tb 0:\s*(\d+)/(\d+)")


class VideoError(Exception):
    """A recording could not be used. Subclasses say why."""


class VideoUnreadableError(VideoError):
    """The file is missing, is not a video, or ffmpeg could not decode it."""


class VariableFrameRateError(VideoError):
    """The recording's frame timing is irregular and the config refuses that."""


@dataclass(frozen=True)
class VideoInfo:
    """What a recording is, read from its header.

    Attributes:
        path: The file.
        width: Frame width in pixels **as displayed** (after rotation) -
            the width of every frame `iter_frames` yields at full size.
        height: Frame height as displayed.
        rotation: Clockwise rotation the container asks a player to apply,
            one of 0, 90, 180, 270. Already applied to decoded frames.
        codec: The video codec's name, e.g. "hevc".
        nominal_fps: The frame rate the stream declares ("tbr") - what the
            camera was set to.
        average_fps: The frame rate actually achieved over the whole file.
            Lower than `nominal_fps` when frames were dropped. None if the
            header does not say.
        duration: Length in seconds, as the container reports it.
        start_time: The container's start offset in seconds; see the
            module docstring's note on the clock.
        frame_count: Exact number of video frames, or None until an index
            has been built (`load_video` fills it in).
        is_vfr: Whether the frame timing is irregular, or None until an
            index has been built.
    """

    path: Path
    width: int
    height: int
    rotation: int
    codec: str
    nominal_fps: float
    average_fps: float | None
    duration: float
    start_time: float
    frame_count: int | None = None
    is_vfr: bool | None = None


@dataclass(frozen=True)
class PtsStats:
    """How regular a recording's frame timing is.

    Attributes:
        frames: Number of frames.
        median_gap: The usual time between frames, in seconds.
        min_gap: The shortest.
        max_gap: The longest - a long one is a run of dropped frames.
        irregular: How many gaps sit further from the median than the
            configured jitter allows.
        irregular_fraction: That count as a share of all gaps.
        effective_fps: Frames per second actually delivered, start to end.
    """

    frames: int
    median_gap: float
    min_gap: float
    max_gap: float
    irregular: int
    irregular_fraction: float
    effective_fps: float

    @property
    def is_vfr(self) -> bool:
        """Whether any gap is irregular."""
        return self.irregular > 0


class PtsIndex:
    """Every frame's presentation timestamp, in display order.

    Frame `n` is the n-th frame shown (0-based), which is the n-th frame
    `iter_frames` yields when reading from the start with stride 1.

    Attributes:
        pts: Sorted (N,) float array of timestamps in seconds; read-only.
        time_base: Seconds per container tick, kept exact.
    """

    def __init__(self, pts: Iterable[float], time_base: Fraction = Fraction(1, 1000)) -> None:
        """Builds an index from timestamps.

        Args:
            pts: Timestamps in seconds, in any order.
            time_base: Seconds per container tick.

        Raises:
            ValueError: If there are no timestamps.
        """
        values = np.sort(np.asarray(list(pts), dtype=float))
        if not len(values):
            raise ValueError("A PTS index needs at least one frame.")
        values.setflags(write=False)
        self.pts = values
        self.time_base = time_base

    def __len__(self) -> int:
        """The number of frames."""
        return len(self.pts)

    def pts_of(self, frame: int) -> float:
        """Reads a frame's timestamp.

        Args:
            frame: 0-based frame number.

        Returns:
            Its PTS in seconds.

        Raises:
            IndexError: If there is no such frame.
        """
        if not 0 <= frame < len(self.pts):
            raise IndexError(f"Frame {frame} is outside 0..{len(self.pts) - 1}.")
        return float(self.pts[frame])

    def frame_at(self, seconds: float) -> int:
        """Finds the frame shown nearest a moment.

        Args:
            seconds: A PTS in seconds; clamped to the recording.

        Returns:
            The 0-based number of the frame whose timestamp is closest.
        """
        after = int(np.searchsorted(self.pts, seconds))
        if after <= 0:
            return 0
        if after >= len(self.pts):
            return len(self.pts) - 1
        before = after - 1
        return before if seconds - self.pts[before] <= self.pts[after] - seconds else after

    def frame_on_screen(self, seconds: float) -> int:
        """Finds the frame being shown at a moment: the last one at or before it.

        Args:
            seconds: A PTS in seconds; clamped to the recording.

        Returns:
            The 0-based frame number.
        """
        return max(0, int(np.searchsorted(self.pts, seconds, side="right")) - 1)

    def frames_between(self, start: float, end: float) -> range:
        """Lists the frames whose timestamps fall in a closed time range.

        Args:
            start: First moment, in seconds.
            end: Last moment, in seconds.

        Returns:
            The frame numbers, possibly empty.
        """
        first = int(np.searchsorted(self.pts, start, side="left"))
        last = int(np.searchsorted(self.pts, end, side="right"))
        return range(first, max(first, last))

    def stats(self, jitter: float = VideoConfig.vfr_jitter) -> PtsStats:
        """Measures how regular the frame timing is.

        Args:
            jitter: How far a gap may sit from the median gap, as a
                fraction of it, before it counts as irregular.

        Returns:
            The measurements. A one-frame recording has no gaps, so its
            gaps read as 0 and it is not irregular.
        """
        gaps = np.diff(self.pts)
        if not len(gaps):
            return PtsStats(len(self.pts), 0.0, 0.0, 0.0, 0, 0.0, 0.0)
        median = float(np.median(gaps))
        irregular = int((np.abs(gaps - median) > jitter * median).sum())
        span = float(self.pts[-1] - self.pts[0])
        return PtsStats(
            frames=len(self.pts),
            median_gap=median,
            min_gap=float(gaps.min()),
            max_gap=float(gaps.max()),
            irregular=irregular,
            irregular_fraction=irregular / len(gaps),
            effective_fps=len(gaps) / span if span > 0 else 0.0,
        )


@dataclass(frozen=True)
class VideoFrame:
    """One decoded frame and when it was shown.

    Attributes:
        pts: Its presentation timestamp in seconds (see the module
            docstring's note on the clock).
        image: (H, W, 3) uint8 RGB pixels, display-oriented. Read-only;
            copy before drawing on it.
        index: Its 0-based frame number in the whole recording, when a
            `PtsIndex` was supplied to look it up; otherwise None.
    """

    pts: float
    image: np.ndarray
    index: int | None = None


@dataclass(frozen=True)
class VideoSource:
    """A recording opened for tracking: header, timestamps and identity.

    Attributes:
        info: The header, with `frame_count` and `is_vfr` filled in.
        index: Every frame's timestamp.
        stats: How regular those timestamps are.
        checksum: The file's SHA-256, or None if it was not asked for.
    """

    info: VideoInfo
    index: PtsIndex
    stats: PtsStats
    checksum: str | None


def _require_file(path: Path) -> None:
    """Raises `VideoUnreadableError` unless `path` is an existing file."""
    if not path.is_file():
        raise VideoUnreadableError(f"{path.name}: no such video file.")


def probe(path: Path | str, ffmpeg: str | None = None) -> VideoInfo:
    """Reads a recording's header without decoding it.

    Args:
        path: The video file.
        ffmpeg: ffmpeg binary; located automatically if not given.

    Returns:
        The header. `frame_count` and `is_vfr` are None: both need the
        timestamp index, which `load_video` builds.

    Raises:
        VideoUnreadableError: If the file is missing or has no video
            stream ffmpeg can describe.
    """
    path = Path(path)
    _require_file(path)
    exe = ffmpeg or footage.ffmpeg_exe()
    try:
        duration = footage.probe_duration(path, exe)
    except footage.FootageError as error:
        raise VideoUnreadableError(str(error)) from error
    header = subprocess.run(
        [exe, "-hide_banner", "-i", str(path)], capture_output=True, text=True, errors="replace"
    ).stderr
    stream = _VIDEO_STREAM.search(header)
    if stream is None:
        raise VideoUnreadableError(f"{path.name}: no video stream found in this file.")
    line = header[stream.start() : header.find("\n", stream.start())]
    nominal = _NOMINAL_FPS.search(line)
    average = _AVERAGE_FPS.search(line)
    if nominal is None:
        raise VideoUnreadableError(f"{path.name}: the video stream declares no frame rate.")
    turned = _ROTATION.search(header)
    tagged = _ROTATE_TAG.search(header)
    rotation = 0
    if turned is not None:
        # The display matrix states the counter-clockwise turn stored in the
        # file's favour; a player applies the opposite.
        rotation = round(-float(turned.group(1))) % 360
    elif tagged is not None:
        rotation = int(tagged.group(1)) % 360
    width, height = int(stream.group(2)), int(stream.group(3))
    if rotation in (90, 270):
        width, height = height, width
    start = _START.search(header)
    return VideoInfo(
        path=path,
        width=width,
        height=height,
        rotation=rotation,
        codec=stream.group(1),
        nominal_fps=float(nominal.group(1)) * (1000.0 if nominal.group(2) else 1.0),
        average_fps=float(average.group(1)) if average else None,
        duration=duration,
        start_time=float(start.group(1)) if start else 0.0,
    )


def _start_offset(path: Path, exe: str) -> float:
    """Reads the container's start offset in seconds (0 if it states none)."""
    header = subprocess.run(
        [exe, "-hide_banner", "-i", str(path)], capture_output=True, text=True, errors="replace"
    ).stderr
    found = _START.search(header)
    return float(found.group(1)) if found else 0.0


def build_pts_index(path: Path | str, ffmpeg: str | None = None) -> PtsIndex:
    """Reads every frame's presentation timestamp without decoding.

    Walks the container's packets (a 95-minute, 3.9GB recording takes about
    seven seconds) and sorts their timestamps into display order, which
    differs from file order whenever the codec uses B-frames.

    Args:
        path: The video file.
        ffmpeg: ffmpeg binary; located automatically if not given.

    Returns:
        The index.

    Raises:
        VideoUnreadableError: If the file is missing, ffmpeg fails on it,
            or it carries frames with no timestamp.
    """
    path = Path(path)
    _require_file(path)
    result = subprocess.run(
        [
            ffmpeg or footage.ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-copyts",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "disabled",
            "-f",
            "framecrc",
            "-",
        ],
        capture_output=True,
        text=True,
        errors="replace",
    )
    base = _CRC_BASE.search(result.stdout)
    if result.returncode != 0 or base is None:
        raise VideoUnreadableError(
            f"{path.name}: couldn't read frame timestamps: {result.stderr.strip()[:300]}"
        )
    time_base = Fraction(int(base.group(1)), int(base.group(2)))
    ticks = []
    for line in result.stdout.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        stamp = line.split(",")[2].strip()
        if not stamp.lstrip("-").isdigit():
            raise VideoUnreadableError(f"{path.name}: a frame has no presentation timestamp.")
        ticks.append(int(stamp))
    if not ticks:
        raise VideoUnreadableError(f"{path.name}: the video stream has no frames.")
    return PtsIndex(np.array(ticks, dtype=float) * float(time_base), time_base)


def _drain_timestamps(stream, stamps: queue.Queue, log: list[str]) -> None:
    """Reads ffmpeg's stderr, queueing each showinfo timestamp in order.

    Runs on its own thread so a full stderr pipe can never stall the
    decode. Puts None when the stream ends.
    """
    base: Fraction | None = None
    for raw in stream:
        line = raw.decode("utf-8", errors="replace")
        frame = _SHOWINFO_FRAME.search(line)
        if frame is not None:
            # Ticks times the time base is exact; pts_time is rounded.
            stamps.put(float(int(frame.group(1)) * base) if base else float(frame.group(2)))
            continue
        declared = _SHOWINFO_BASE.search(line)
        if declared is not None:
            base = Fraction(int(declared.group(1)), int(declared.group(2)))
        elif "showinfo" not in line and line.strip():
            log.append(line.strip())
            del log[:-20]
    stamps.put(None)


def iter_frames(
    path: Path | str,
    start: float | None = None,
    end: float | None = None,
    stride: int = 1,
    width: int | None = None,
    index: PtsIndex | None = None,
    config: VideoConfig | None = None,
    ffmpeg: str | None = None,
) -> Iterator[VideoFrame]:
    """Streams decoded frames, each with its real presentation timestamp.

    One ffmpeg process decodes forward; frames are yielded as they arrive
    and never held, so a whole match can be walked in constant memory. No
    frame is duplicated or dropped to fake a constant rate.

    Args:
        path: The video file.
        start: First moment wanted, as a PTS in seconds; from the first
            frame if omitted. Exact: the first frame yielded is the first
            whose PTS is at or after it.
        end: Last moment wanted (inclusive); to the last frame if omitted.
        stride: Yield every `stride`-th frame of those in range, starting
            with the first.
        width: Scale frames to this width (height keeps the aspect ratio);
            full size if omitted.
        index: A timestamp index for this file; when given, each frame's
            `index` is filled in.
        config: Decode settings.
        ffmpeg: ffmpeg binary; located automatically if not given.

    Yields:
        `VideoFrame`s in display order.

    Raises:
        ValueError: If `stride` is below 1 or the range is backwards.
        VideoUnreadableError: If the file is missing or ffmpeg fails.
    """
    config = config or VideoConfig()
    path = Path(path)
    if stride < 1:
        raise ValueError("stride must be at least 1.")
    if start is not None and end is not None and end < start:
        raise ValueError("end must not be before start.")
    _require_file(path)
    command = [ffmpeg or footage.ffmpeg_exe(), "-hide_banner", "-nostats", "-copyts"]
    exe = command[0]
    offset = _start_offset(path, exe) if (start is not None or end is not None) else 0.0
    seek = 0.0
    if start is not None and start - offset > 1.0:
        # ffmpeg adds the container's start offset to -ss, so take it off;
        # land a second early and let the exact select below choose the
        # first frame.
        seek = start - offset - 1.0
        command += ["-ss", f"{seek:.6f}"]
    if end is not None:
        command += ["-t", f"{max(0.0, end - offset - seek) + 1.0:.6f}"]
    filters = []
    wanted = []
    if start is not None:
        wanted.append(f"gte(t,{start - 1e-6:.6f})")
    if end is not None:
        wanted.append(f"lte(t,{end + 1e-6:.6f})")
    if wanted:
        filters.append(f"select='{'*'.join(wanted)}'")
    if stride > 1:
        filters.append(f"select='not(mod(n,{stride}))'")
    if width is not None:
        filters.append(f"scale={int(width)}:-2")
    filters.append("showinfo=checksum=0")
    command += [
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-fps_mode",
        "passthrough",
        "-vf",
        ",".join(filters),
        "-f",
        "image2pipe",
        "-vcodec",
        "ppm",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stamps: queue.Queue = queue.Queue()
    log: list[str] = []
    reader = threading.Thread(
        target=_drain_timestamps, args=(process.stderr, stamps, log), daemon=True
    )
    reader.start()
    produced = 0
    try:
        for image in footage._read_ppm_frames(process.stdout):
            try:
                pts = stamps.get(timeout=config.decode_timeout_seconds)
            except queue.Empty:
                pts = None
            if pts is None:
                raise VideoUnreadableError(f"{path.name}: ffmpeg gave a frame with no timestamp.")
            produced += 1
            yield VideoFrame(pts, image, index.frame_at(pts) if index is not None else None)
        process.stdout.close()
        code = process.wait(timeout=config.decode_timeout_seconds)
        reader.join(timeout=config.decode_timeout_seconds)
        if code != 0 and not produced:
            raise VideoUnreadableError(
                f"{path.name}: ffmpeg could not decode this file: {' | '.join(log[-3:])[:300]}"
            )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for pipe in (process.stdout, process.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()


def frame_at(
    path: Path | str,
    seconds: float,
    width: int | None = None,
    config: VideoConfig | None = None,
    ffmpeg: str | None = None,
) -> VideoFrame:
    """Decodes the first frame at or after a moment.

    Args:
        path: The video file.
        seconds: The moment, as a PTS in seconds.
        width: Scale to this width; full size if omitted.
        config: Decode settings.
        ffmpeg: ffmpeg binary; located automatically if not given.

    Returns:
        That frame, with its own PTS (at or shortly after `seconds`).

    Raises:
        VideoUnreadableError: If the file cannot be decoded, or holds no
            frame within two seconds after that moment.
    """
    frames = iter_frames(
        path, start=seconds, end=seconds + 2.0, width=width, config=config, ffmpeg=ffmpeg
    )
    try:
        for frame in frames:
            return frame
    finally:
        frames.close()
    raise VideoUnreadableError(f"{Path(path).name}: no frame at {seconds:.3f}s.")


def frames_at(
    path: Path | str,
    times: Iterable[float],
    width: int | None = None,
    config: VideoConfig | None = None,
    ffmpeg: str | None = None,
) -> list[VideoFrame]:
    """Decodes one frame at each of several moments.

    One seek per moment, which is far cheaper than decoding through when
    the moments are seconds apart.

    Args:
        path: The video file.
        times: The moments, as PTS in seconds.
        width: Scale to this width; full size if omitted.
        config: Decode settings.
        ffmpeg: ffmpeg binary; located automatically if not given.

    Returns:
        One frame per moment, in the order given.

    Raises:
        VideoUnreadableError: As `frame_at`.
    """
    return [frame_at(path, moment, width, config, ffmpeg) for moment in times]


def file_checksum(path: Path | str, config: VideoConfig | None = None) -> str:
    """Hashes a recording, so results can name exactly which file they came from.

    Args:
        path: The file.
        config: Supplies the read chunk size.

    Returns:
        The SHA-256 of its bytes, as lowercase hex.

    Raises:
        VideoUnreadableError: If the file is missing.
    """
    config = config or VideoConfig()
    path = Path(path)
    _require_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(config.checksum_chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_video(
    path: Path | str,
    config: VideoConfig | None = None,
    checksum: bool = True,
    ffmpeg: str | None = None,
) -> VideoSource:
    """Opens a recording for tracking: header, timestamp index, checksum.

    Args:
        path: The video file.
        config: Decode settings, including whether irregular frame timing
            is refused.
        checksum: Hash the file too. Reads every byte, so it can be
            skipped when only the timing is wanted.
        ffmpeg: ffmpeg binary; located automatically if not given.

    Returns:
        The opened recording; `info.frame_count` and `info.is_vfr` are
        exact, taken from the index.

    Raises:
        VideoUnreadableError: If the file is missing or not decodable.
        VariableFrameRateError: If the frame timing is irregular and
            `config.reject_vfr` is set.
    """
    config = config or VideoConfig()
    path = Path(path)
    info = probe(path, ffmpeg)
    index = build_pts_index(path, ffmpeg)
    stats = index.stats(config.vfr_jitter)
    if stats.is_vfr:
        message = (
            f"{path.name}: frame timing is irregular - {stats.irregular} of "
            f"{stats.frames - 1} frame gaps ({stats.irregular_fraction:.2%}) differ from the "
            f"usual {stats.median_gap * 1000:.1f}ms, the longest being "
            f"{stats.max_gap * 1000:.0f}ms."
        )
        if config.reject_vfr:
            raise VariableFrameRateError(message + " This configuration refuses such files.")
        logger.warning("%s Real timestamps are used throughout.", message)
    return VideoSource(
        info=replace(info, frame_count=len(index), is_vfr=stats.is_vfr),
        index=index,
        stats=stats,
        checksum=file_checksum(path, config) if checksum else None,
    )
