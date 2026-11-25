"""Walking a recording in chunks, and stitching the chunks back into one match.

A 95-minute match is about 340,000 frames and some 7.5 hours of CPU, on a
machine short of memory. So `pipeline.TrackerPipeline` tracks it in
**chunks**: each owns a stretch of the recording (`ChunkWindow.start` to
`end`) but reads `PipelineConfig.overlap_seconds` more on either side, so
the ball linker, the event rules and the motion detector see a chunk's
edges with context. This module holds the parts that are pure bookkeeping,
testable without any video:

* `plan_chunks` - the windows.
* `CompactBallTrack` - the ball's per-frame state for the whole match in a
  few numpy arrays (some 40 bytes a frame), answering `quality` exactly as
  `BallTrack.quality` would, without keeping a Python object per frame.
* `ChunkResult` - what one chunk hands on: its events, its frames' ball
  state, its players' observations, the poses around its hits.
* `stitch_events` / `merge_chunks` - one match from the chunks.

Who owns what
-------------
Per-frame data (ball state, player detections, scene samples) belongs to
the chunk whose stretch the frame is in - one owner per frame, no
judgement needed. **Events** are different: two chunks reading the same
hit near their shared edge, each with different context, can time it
slightly differently - 149.99s in one and 150.01s in the next. Kept by
owner alone, that hit would be kept twice (each owns its own copy) or not
at all (each copy lies in the other's stretch). So every chunk also keeps
the events it found within `stitch_margin_seconds` beyond its edges, and
at each edge the two chunks' events are matched (same kind, same end of
the court, within `stitch_tolerance_seconds`): a matched pair becomes one
event - the copy from the chunk owning their mean time - and an unmatched
event is kept only by its owner, the chunk with more context around it.

The players are followed by **one** tracker carried from chunk to chunk
(`players.PlayerTracker.snapshot`), not one per chunk: who is "A" and who
"B", and every change of ends, then hold across the whole match. A
chunk's own copy reads on into the overlap for that chunk's events, then
is discarded; the next chunk carries on from the snapshot taken at the
edge. Every observation is labelled with its person only at the end
(`players.label_tracks`), once every change of ends is known.

Honest limits
-------------
* The ball linker chooses paths over a whole chunk, so near an edge its
  choice can differ from what one long pass would have made. The overlap
  makes this rare; it is not impossible, and matched duplicates are
  counted (`StitchReport`) so how often edges disagreed is a number.
* `discarded_kinks` is summed over chunks, overlap included, so it
  slightly overcounts.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from .ball import _GAP_BUCKETS, _LONGEST_BUCKET, BallQuality, BallTrack, SceneStats
from .config import BallConfig, EventConfig, PipelineConfig, PlayerConfig
from .events import BounceEvent, EventTrack, HitEvent, NetEvent, join_shots
from .players import EndSwap, PlayerObservation, PlayerTracks, label_tracks
from .provenance import Provenance
from .strokes import PoseSample

logger = logging.getLogger(__name__)

SOURCE = "stitch-v1"

UNKNOWN_STATE, MEASURED_STATE, BRIDGED_STATE = 0, 1, 2
# Segment numbers from different chunks must not collide.
_SEGMENT_STRIDE = 1_000_000


# --------------------------------------------------------------------------
# The windows
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkWindow:
    """One chunk's place in the run.

    Attributes:
        index: 0-based position in the run.
        start: First moment it owns, seconds on the video's clock.
        end: Where its stretch ends; the moment itself belongs to the
            next chunk, except for the run's last chunk, which owns it.
        read_start: First moment it reads (`start` less the overlap,
            never before the run).
        read_end: Last moment it reads.
        last: Whether it is the run's last chunk.
    """

    index: int
    start: float
    end: float
    read_start: float
    read_end: float
    last: bool = False

    def owns(self, pts: float) -> bool:
        """Whether a moment belongs to this chunk's stretch.

        Args:
            pts: The moment, seconds.

        Returns:
            True for `start <= pts < end` (`<= end` for the last chunk).
        """
        return self.start <= pts < self.end or (self.last and pts == self.end)

    def keeps(self, pts: float, margin: float) -> bool:
        """Whether an event at a moment is handed on for stitching.

        Args:
            pts: The event's time.
            margin: How far beyond the stretch events are kept.

        Returns:
            True within `margin` of the stretch, inside what was read.
        """
        inside = self.start - margin <= pts < self.end + margin
        return inside and self.read_start <= pts <= self.read_end

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field.
        """
        return dict(self.__dict__)


def plan_chunks(start: float, end: float, config: PipelineConfig | None = None) -> tuple[
    ChunkWindow, ...
]:
    """Splits a run into chunks of `chunk_seconds`, the last one shorter.

    Args:
        start: The run's first moment, seconds on the video's clock.
        end: Its last moment.
        config: Chunk length, overlap and stitch margin.

    Returns:
        The windows, in order; one when the run is shorter than a chunk.

    Raises:
        ValueError: If the run is empty, a chunk has no length, or the
            overlap does not exceed the stitch margin.
    """
    config = config or PipelineConfig()
    if not end > start:
        raise ValueError(f"The run must end after it starts ({start:.2f}s to {end:.2f}s).")
    if config.chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive.")
    if config.overlap_seconds <= config.stitch_margin_seconds:
        raise ValueError("overlap_seconds must exceed stitch_margin_seconds.")
    count = max(1, math.ceil((end - start) / config.chunk_seconds - 1e-9))
    windows = []
    for index in range(count):
        first = start + index * config.chunk_seconds
        last = index == count - 1
        until = end if last else start + (index + 1) * config.chunk_seconds
        windows.append(
            ChunkWindow(
                index=index,
                start=first,
                end=until,
                read_start=max(start, first - config.overlap_seconds),
                read_end=min(end, until + config.overlap_seconds),
                last=last,
            )
        )
    return tuple(windows)


# --------------------------------------------------------------------------
# The ball, compactly
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactBallTrack:
    """The ball's state in every frame of a run, as arrays.

    Stands in for `BallTrack` as `EventTrack.ball` once chunks are merged:
    it has the `frame_pts`, `source`, `scene` and `image_size` the later
    stages read, and a `quality` that gives the same answer
    `BallTrack.quality` gives on the same frames. It keeps no per-frame
    Python objects, so a whole match costs some 15MB rather than hundreds.

    Attributes:
        frame_numbers: (N,) frame numbers, in time order.
        frame_pts: (N,) their timestamps, seconds.
        state: (N,) `UNKNOWN_STATE`, `MEASURED_STATE` or `BRIDGED_STATE`.
        confidence: (N,) the observation's confidence; 0 where unknown.
        image_xy: (N, 2) the ball's image position; NaN where unknown.
        segment: (N,) the segment it belongs to, unique across chunks;
            -1 where unknown.
        image_size: (width, height) of the frames.
        source: The ball detector's name.
        scene: Brightness and colour summary, when one was collected.
        weak_coverage: Coverage under which a stretch is called weak.
        weak_confidence: Mean confidence under which it is.
    """

    frame_numbers: np.ndarray
    frame_pts: np.ndarray
    state: np.ndarray
    confidence: np.ndarray
    image_xy: np.ndarray
    segment: np.ndarray
    image_size: tuple[int, int]
    source: str
    scene: SceneStats | None = None
    weak_coverage: float = BallConfig.weak_coverage
    weak_confidence: float = BallConfig.weak_confidence
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def from_track(
        cls,
        track: BallTrack,
        keep: Callable[[float], bool] | None = None,
        segment_offset: int = 0,
    ) -> CompactBallTrack:
        """Packs a `BallTrack`'s frames into arrays.

        Args:
            track: The track.
            keep: Which frames to keep, by timestamp; all when omitted.
            segment_offset: Added to every segment number.

        Returns:
            The packed frames.
        """
        pts = np.asarray(track.frame_pts, dtype=float)
        count = len(pts)
        state = np.zeros(count, dtype=np.int8)
        confidence = np.zeros(count, dtype=float)
        image_xy = np.full((count, 2), np.nan, dtype=float)
        segment = np.full(count, -1, dtype=np.int64)
        for observation in track.observations:
            position = int(np.searchsorted(pts, observation.pts - 1e-9))
            if position >= count or abs(pts[position] - observation.pts) > 1e-6:
                continue  # not one of the processed frames: nothing to place
            measured = observation.provenance is Provenance.MEASURED
            state[position] = MEASURED_STATE if measured else BRIDGED_STATE
            confidence[position] = observation.confidence
            image_xy[position] = observation.image_xy
            segment[position] = observation.segment + segment_offset
        mask = (
            np.ones(count, dtype=bool)
            if keep is None
            else np.array([keep(float(moment)) for moment in pts], dtype=bool)
        )
        return cls(
            frame_numbers=np.asarray(track.frame_numbers, dtype=np.int64)[mask],
            frame_pts=pts[mask],
            state=state[mask],
            confidence=confidence[mask],
            image_xy=image_xy[mask],
            segment=segment[mask],
            image_size=tuple(track.image_size),
            source=track.source,
            scene=track.scene,
            weak_coverage=track.weak_coverage,
            weak_confidence=track.weak_confidence,
        )

    @classmethod
    def concatenate(
        cls, parts: Sequence[CompactBallTrack], scene: SceneStats | None = None
    ) -> CompactBallTrack:
        """Joins consecutive pieces into one.

        Args:
            parts: Pieces in time order, not overlapping.
            scene: The scene summary for the whole; the first piece's when
                omitted.

        Returns:
            One track over all their frames.

        Raises:
            ValueError: If there are no pieces.
        """
        if not parts:
            raise ValueError("Nothing to join.")
        first = parts[0]
        return cls(
            frame_numbers=np.concatenate([p.frame_numbers for p in parts]),
            frame_pts=np.concatenate([p.frame_pts for p in parts]),
            state=np.concatenate([p.state for p in parts]),
            confidence=np.concatenate([p.confidence for p in parts]),
            image_xy=np.concatenate([p.image_xy for p in parts]),
            segment=np.concatenate([p.segment for p in parts]),
            image_size=first.image_size,
            source=first.source,
            scene=scene if scene is not None else first.scene,
            weak_coverage=first.weak_coverage,
            weak_confidence=first.weak_confidence,
        )

    def _gap_runs(self) -> tuple[np.ndarray, np.ndarray, float]:
        """First and last positions of every unknown run, and the frame step.

        Computed once and cached: `quality` is asked once per point.
        """
        if "runs" not in self._cache:
            unknown = np.concatenate([[False], self.state == UNKNOWN_STATE, [False]])
            edges = np.flatnonzero(np.diff(unknown.astype(np.int8)))
            pts = self.frame_pts
            step = float(np.median(np.diff(pts))) if len(pts) > 1 else 0.0
            self._cache["runs"] = (edges[::2], edges[1::2] - 1, step)
        return self._cache["runs"]

    def gaps(self) -> list[tuple[int, int]]:
        """Runs of frames with no ball position.

        Returns:
            (first, last) frame positions of each run, in order.
        """
        firsts, lasts, _ = self._gap_runs()
        return list(zip(firsts.tolist(), lasts.tolist()))

    def _coherence(self, inside: np.ndarray) -> float:
        """`ball.trajectory_coherence` over the measured frames in a stretch."""
        seen = np.flatnonzero(inside & (self.state == MEASURED_STATE))
        if len(seen) < 2:
            return 0.0
        elapsed = np.diff(self.frame_pts[seen])
        steps = (elapsed > 0) & (elapsed <= 0.1)
        if not steps.any():
            return 0.0
        width = self.image_size[0]
        slow, fast = 2 * 30 / 1920 * width, 180 * 30 / 1920 * width
        moved = np.hypot(*np.diff(self.image_xy[seen], axis=0).T)
        speed = moved[steps] / elapsed[steps]
        return float(((speed >= slow) & (speed <= fast)).sum() / steps.sum())

    def quality(self, start: float | None = None, end: float | None = None) -> BallQuality:
        """Summarises tracking quality, exactly as `BallTrack.quality` does.

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
        measured = int((inside & (self.state == MEASURED_STATE)).sum())
        bridged = int((inside & (self.state == BRIDGED_STATE)).sum())
        seen = inside & (self.state != UNKNOWN_STATE)
        firsts, lasts, step = self._gap_runs()
        begins, ends = self.frame_pts[firsts], self.frame_pts[lasts]
        touching = ~((ends < low) | (begins > high))
        # Each overlapping run's length inside the stretch, as BallTrack does it.
        seconds = np.minimum(ends[touching], high) - np.maximum(begins[touching], low) + step
        limits = np.array([limit for limit, _ in _GAP_BUCKETS])
        buckets = np.bincount(
            np.searchsorted(limits, seconds, side="right"), minlength=len(limits) + 1
        )
        labels = [label for _, label in _GAP_BUCKETS] + [_LONGEST_BUCKET]
        histogram = {label: int(count) for label, count in zip(labels, buckets)}
        longest = float(seconds.max()) if len(seconds) else 0.0
        coverage = (measured + bridged) / frames if frames else 0.0
        confidence = float(np.mean(self.confidence[seen])) if seen.any() else 0.0
        return BallQuality(
            frames=frames,
            measured=measured,
            bridged=bridged,
            unknown=frames - measured - bridged,
            coverage=coverage,
            measured_fraction=measured / frames if frames else 0.0,
            mean_confidence=confidence,
            segments=int(len(np.unique(self.segment[seen]))),
            gap_histogram=histogram,
            longest_gap_seconds=float(longest),
            coherence=self._coherence(inside),
            weak=coverage < self.weak_coverage or confidence < self.weak_confidence,
        )

    def coverage_over(self, stretches: Sequence[tuple[float, float]]) -> tuple[int, int, float]:
        """Measures tracking over several stretches together (overlaps once).

        Args:
            stretches: (start, end) moments, e.g. every point's rally.

        Returns:
            (frames, frames with a ball position, mean confidence of those
            positions) over the union of the stretches.
        """
        inside = np.zeros(len(self.frame_pts), dtype=bool)
        for start, end in stretches:
            inside |= (self.frame_pts >= start) & (self.frame_pts <= end)
        seen = inside & (self.state != UNKNOWN_STATE)
        confidence = float(np.mean(self.confidence[seen])) if seen.any() else 0.0
        return int(inside.sum()), int(seen.sum()), confidence


# --------------------------------------------------------------------------
# One chunk's output
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkResult:
    """Everything one chunk hands on - what its checkpoint file holds.

    Attributes:
        window: The chunk's place in the run.
        hits: Hits it found within the stitch margin of its stretch.
        bounces: Bounces, likewise.
        nets: Net contacts, likewise.
        ball: The ball's state in the frames it owns.
        near: Near-end player detections in its stretch, unlabelled.
        far: Far-end ones.
        detector_times: When the person detector ran in its stretch.
        rejected_people: Detections discarded as not on this court.
        swaps: Every change of ends judged so far in the run (cumulative,
            so the last chunk's list is the whole run's).
        poses: Hit time (rounded to the microsecond) to the poses found
            around that hit.
        scene: Its scene summary, or None.
        discarded_kinks: Kinks it took for no event (overlap included).
        drift_px: How far the court had moved in the picture at its
            start, or None when not checked or calibration abstained.
        drift_checked: Whether a re-calibration was attempted.
        seconds: Wall-clock seconds it took.
        tracker_state: The player tracker's snapshot at the end of its
            stretch - where the next chunk carries on from.
        detector_name: The person detector's name.
    """

    window: ChunkWindow
    hits: tuple[HitEvent, ...]
    bounces: tuple[BounceEvent, ...]
    nets: tuple[NetEvent, ...]
    ball: CompactBallTrack
    near: tuple[PlayerObservation, ...]
    far: tuple[PlayerObservation, ...]
    detector_times: tuple[float, ...]
    rejected_people: int
    swaps: tuple[EndSwap, ...]
    poses: dict[float, tuple[PoseSample, ...]]
    scene: SceneStats | None
    discarded_kinks: int
    drift_px: float | None
    drift_checked: bool
    seconds: float
    tracker_state: bytes
    detector_name: str


def pose_key(pts: float) -> float:
    """The key a hit's poses are stored under.

    Args:
        pts: The hit's time.

    Returns:
        It, rounded to the microsecond.
    """
    return round(float(pts), 6)


# --------------------------------------------------------------------------
# Stitching
# --------------------------------------------------------------------------


@dataclass
class StitchReport:
    """How the chunk edges were reconciled.

    Attributes:
        chunks: Chunks stitched.
        matched: Events seen by both chunks at an edge, kept once.
        kept_from_neighbour: Of those, how many were kept although the
            chunk owning their mean time had timed its own copy in the
            other chunk's stretch.
        dropped_unowned: Events seen only by the chunk that did not own
            their time - left out, since the owner saw more around them.
    """

    chunks: int = 0
    matched: int = 0
    kept_from_neighbour: int = 0
    dropped_unowned: int = 0

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field.
        """
        return dict(self.__dict__)


def _hit_key(hit: HitEvent) -> str:
    """Hits match only at the same end."""
    return hit.side


def _bounce_key(bounce: BounceEvent) -> str:
    """Bounces match only in the same half."""
    return bounce.half


def _net_key(_net: NetEvent) -> str:
    """Net contacts match any net contact."""
    return "net"


def _stitch_kind(
    per_chunk: Sequence[Sequence],
    windows: Sequence[ChunkWindow],
    key: Callable[[object], str],
    config: PipelineConfig,
    report: StitchReport,
) -> list:
    """Stitches one kind of event; see the module docstring.

    Args:
        per_chunk: Each chunk's events of this kind, in chunk order.
        windows: The chunks' windows.
        key: What two events must share to be the same event.
        config: Margin and tolerance.
        report: Counts what happened at the edges.

    Returns:
        The events, each once, in time order.
    """
    kept: dict[int, object] = {}
    for events, window in zip(per_chunk, windows):
        for event in events:
            if window.owns(event.pts):
                kept[id(event)] = event
    margin, tolerance = config.stitch_margin_seconds, config.stitch_tolerance_seconds
    for left in range(len(windows) - 1):
        edge = windows[left].end
        near = [
            [e for e in per_chunk[side] if edge - margin <= e.pts < edge + margin]
            for side in (left, left + 1)
        ]
        pairs = sorted(
            (abs(one.pts - two.pts), i, j)
            for i, one in enumerate(near[0])
            for j, two in enumerate(near[1])
            if key(one) == key(two) and abs(one.pts - two.pts) <= tolerance
        )
        used: tuple[set[int], set[int]] = (set(), set())
        for _, i, j in pairs:
            if i in used[0] or j in used[1]:
                continue
            used[0].add(i)
            used[1].add(j)
            one, two = near[0][i], near[1][j]
            winner, loser = (one, two) if (one.pts + two.pts) / 2 < edge else (two, one)
            report.matched += 1
            if id(winner) not in kept:
                report.kept_from_neighbour += 1
            kept.pop(id(loser), None)
            kept[id(winner)] = winner
        for side in (0, 1):
            window = windows[left + side]
            report.dropped_unowned += sum(
                1
                for i, event in enumerate(near[side])
                if i not in used[side] and not window.owns(event.pts)
            )
    return sorted(kept.values(), key=lambda event: event.pts)


def stitch_events(
    chunks: Sequence[ChunkResult], config: PipelineConfig | None = None
) -> tuple[list[HitEvent], list[BounceEvent], list[NetEvent], StitchReport]:
    """Joins every chunk's hits, bounces and net contacts, each once.

    Args:
        chunks: The chunks, in order.
        config: Stitch margin and tolerance.

    Returns:
        (hits, bounces, nets) in time order, and what happened at the
        edges.
    """
    config = config or PipelineConfig()
    windows = [chunk.window for chunk in chunks]
    report = StitchReport(chunks=len(chunks))
    hits = _stitch_kind([c.hits for c in chunks], windows, _hit_key, config, report)
    bounces = _stitch_kind([c.bounces for c in chunks], windows, _bounce_key, config, report)
    nets = _stitch_kind([c.nets for c in chunks], windows, _net_key, config, report)
    return hits, bounces, nets, report


def merge_scenes(scenes: Sequence[SceneStats | None]) -> SceneStats | None:
    """Averages scene summaries, weighted by the frames each sampled.

    Args:
        scenes: Per-chunk summaries; None entries are skipped.

    Returns:
        One summary, or None when there were none.
    """
    found = [scene for scene in scenes if scene is not None and scene.frames_sampled > 0]
    if not found:
        return None
    weights = np.array([scene.frames_sampled for scene in found], dtype=float)

    def mean(values: list[float]) -> float:
        """The frames-weighted mean of one statistic."""
        return float(np.average(values, weights=weights))

    return SceneStats(
        frames_sampled=int(weights.sum()),
        brightness=mean([s.brightness for s in found]),
        contrast=mean([s.contrast for s in found]),
        court_rgb=tuple(mean([s.court_rgb[k] for s in found]) for k in range(3)),
        court_yellowness=mean([s.court_yellowness for s in found]),
    )


@dataclass(frozen=True)
class StitchedMatch:
    """The run's chunks as one stretch, ready for strokes and assembly.

    Attributes:
        events: Every hit, bounce and net contact once, joined into shots
            (rally-judged; serves are judged at segmentation), over a
            `CompactBallTrack` of the whole run.
        players: Both players over the whole run, labelled with every
            change of ends.
        poses: Shot index to the poses around its hit.
        scene: The run's scene summary.
        report: What happened at the chunk edges.
    """

    events: EventTrack
    players: PlayerTracks
    poses: dict[int, tuple[PoseSample, ...]]
    scene: SceneStats | None
    report: StitchReport


def merge_chunks(
    chunks: Sequence[ChunkResult],
    pipeline: PipelineConfig | None = None,
    events: EventConfig | None = None,
    players: PlayerConfig | None = None,
) -> StitchedMatch:
    """Turns a run's chunks back into one match.

    Args:
        chunks: Every chunk of the run, in order.
        pipeline: Stitch settings.
        events: How shots are joined and judged.
        players: Supplies how far `PlayerTracks.at` interpolates.

    Returns:
        The match's events, players, poses and scene.

    Raises:
        ValueError: If there are no chunks or they are not the run's
            consecutive chunks.
    """
    if not chunks:
        raise ValueError("No chunks to merge.")
    if [c.window.index for c in chunks] != list(range(len(chunks))) or not chunks[-1].window.last:
        raise ValueError("The chunks are not the whole run, in order.")
    events = events or EventConfig()
    tracks = label_tracks(
        near=[o for chunk in chunks for o in chunk.near],
        far=[o for chunk in chunks for o in chunk.far],
        swaps=chunks[-1].swaps,
        detector_times=[t for chunk in chunks for t in chunk.detector_times],
        source=chunks[0].detector_name,
        config=players,
        rejected_people=sum(chunk.rejected_people for chunk in chunks),
    )
    hits, bounces, nets, report = stitch_events(chunks, pipeline)
    # A chunk named each hitter from the changes of ends it knew of; a
    # change judged in a later chunk can still rename them.
    hits = [replace(hit, player=tracks.player_at(hit.pts, hit.side)) for hit in hits]
    shots, kept = join_shots(hits, bounces, nets, events)
    scene = merge_scenes([chunk.scene for chunk in chunks])
    ball = CompactBallTrack.concatenate([chunk.ball for chunk in chunks], scene)
    track = EventTrack(
        hits=tuple(hits),
        bounces=tuple(kept),
        nets=tuple(nets),
        shots=tuple(shots),
        ball=ball,
        config=events,
        discarded_kinks=sum(chunk.discarded_kinks for chunk in chunks),
    )
    found: dict[float, tuple[PoseSample, ...]] = {}
    for chunk in chunks:
        found.update(chunk.poses)
    poses = {
        shot.index: found[pose_key(shot.hit.pts)]
        for shot in shots
        if pose_key(shot.hit.pts) in found
    }
    logger.info(
        "Stitched %d chunk(s): %d hits, %d bounces, %d nets; %d edge event(s) matched",
        len(chunks),
        len(hits),
        len(kept),
        len(nets),
        report.matched,
    )
    return StitchedMatch(track, tracks, poses, scene, report)
