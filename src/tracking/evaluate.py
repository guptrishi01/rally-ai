"""Scoring tracker output against a reference - how accuracy work gets measured.

Two references are supported, and neither is called ground truth unless it
is one:

* **A SwingVision export of the same match** (`compare_shots`). A *weak*
  reference: SwingVision's own tracking has errors, gaps and its own
  clock. Agreement with it is reported as agreement, never as accuracy.
* **The reviewer's own confirmations** (`compare_confirmed`). When a
  person confirms a tracked point in the review modal, the confirmed
  outcome is a human judgement; the tracker's original guess is still
  recoverable from the shot rows staged with the match, so every
  confirmed point is a free labelled example. This is the correction loop:
  review feeds evaluation without any new storage.

Everything is reported with its denominator (`Ratio`), so "90%" never
appears without "of how many", and a metric with nothing to measure says
`n/a` rather than 0% or 100%.

Matching. A reference shot and a tracker shot are the same shot when their
times (the reference's moved onto the video clock by `clock_offset`) are
within `tolerance` seconds; pairs are made closest-first, one-to-one.
`estimate_clock_offset` finds the offset that pairs the most shots when it
is not known - an estimate, and reported as one.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from swingvision_import import reconstruct
from swingvision_import.raw import RawShotRow

SERVE = "serve"
_SERVE_TYPES = {"first_serve", "second_serve"}
_NET_Y_M = 11.885
CONFIDENCE_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0001)


@dataclass(frozen=True)
class Ratio:
    """A count out of a denominator.

    Attributes:
        hits: How many satisfied the measure.
        total: How many it was measured on.
    """

    hits: int
    total: int

    @property
    def value(self) -> float | None:
        """The fraction, or None when there was nothing to measure."""
        return self.hits / self.total if self.total else None

    def __str__(self) -> str:
        """Reads as "7/9 (78%)", or "0/0 (n/a)"."""
        value = self.value
        shown = "n/a" if value is None else f"{value:.0%}"
        return f"{self.hits}/{self.total} ({shown})"

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            hits, total and the fraction (None when undefined).
        """
        value = self.value
        return {"hits": self.hits, "total": self.total, "value": value}


@dataclass(frozen=True)
class Distribution:
    """A spread of errors.

    Attributes:
        count: How many values.
        mean: Their mean, or None when there are none.
        median: Their median.
        p90: Their 90th percentile.
    """

    count: int
    mean: float | None
    median: float | None
    p90: float | None

    @classmethod
    def of(cls, values: Iterable[float]) -> Distribution:
        """Summarises values.

        Args:
            values: The values; may be empty.

        Returns:
            Their spread; None statistics when empty.
        """
        data = np.asarray(list(values), dtype=float)
        if not len(data):
            return cls(0, None, None, None)
        return cls(
            len(data),
            round(float(data.mean()), 3),
            round(float(np.median(data)), 3),
            round(float(np.percentile(data, 90)), 3),
        )

    def __str__(self) -> str:
        """Reads as "median 0.40, p90 1.10, mean 0.52 (n=12)"."""
        if not self.count:
            return "n/a (n=0)"
        return (
            f"median {self.median:.2f}, p90 {self.p90:.2f}, mean {self.mean:.2f} (n={self.count})"
        )

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field.
        """
        return dict(self.__dict__)


def stroke_family(stroke: str | None) -> str | None:
    """Reduces a stroke name to its family, so both vocabularies compare.

    Args:
        stroke: A stroke as either source writes it, e.g. "Forehand Volley".

    Returns:
        "serve", "forehand", "backhand", "volley", "overhead", "feed", or
        None for no stroke or the tracker's "Unclear".
    """
    if not stroke:
        return None
    text = stroke.strip().lower()
    if text in ("unclear", "unknown", "none"):
        return None
    for family in ("volley", "overhead", "serve", "feed", "forehand", "backhand"):
        if family in text:
            return family
    return text


def is_serve(row: RawShotRow) -> bool:
    """Whether a row is a serve, by role or by stroke.

    Args:
        row: The row.

    Returns:
        True for a first or second serve.
    """
    return row.shot_type in _SERVE_TYPES or stroke_family(row.stroke) == SERVE


def in_window(rows: Iterable[RawShotRow], start: float | None, end: float | None, offset: float):
    """Rows whose time, plus `offset`, lies in [start, end].

    Args:
        rows: The rows.
        start: First moment on the video clock; open when None.
        end: Last moment; open when None.
        offset: Added to each row's `video_time`.

    Returns:
        The rows inside, in time order; rows without a time are left out.
    """
    kept = []
    for row in rows:
        if row.video_time is None:
            continue
        moment = row.video_time + offset
        if (start is None or moment >= start) and (end is None or moment <= end):
            kept.append(row)
    return sorted(kept, key=lambda r: r.video_time)


def match_shots(
    reference: Sequence[RawShotRow],
    tracker: Sequence[RawShotRow],
    clock_offset: float,
    tolerance: float,
) -> list[tuple[int, int]]:
    """Pairs reference shots with tracker shots by time, closest first.

    Args:
        reference: Reference rows.
        tracker: Tracker rows (their `video_time` is the video clock).
        clock_offset: Added to a reference time to put it on the video
            clock.
        tolerance: Largest time difference a pair may have, seconds.

    Returns:
        (reference index, tracker index) pairs, one-to-one.
    """
    ref = np.array([r.video_time + clock_offset for r in reference], dtype=float)
    ours = np.array([t.video_time for t in tracker], dtype=float)
    if not len(ref) or not len(ours):
        return []
    candidates = []
    for i, moment in enumerate(ref):
        low = int(np.searchsorted(ours, moment - tolerance, side="left"))
        high = int(np.searchsorted(ours, moment + tolerance, side="right"))
        candidates.extend((abs(ours[j] - moment), i, j) for j in range(low, high))
    used_ref, used_ours, pairs = set(), set(), []
    for _, i, j in sorted(candidates):
        if i not in used_ref and j not in used_ours:
            used_ref.add(i)
            used_ours.add(j)
            pairs.append((i, j))
    return sorted(pairs)


def estimate_clock_offset(
    reference: Sequence[RawShotRow],
    tracker: Sequence[RawShotRow],
    search: float = 6.0,
    step: float = 0.01,
    tolerance: float = 0.15,
) -> tuple[float, int]:
    """Finds the clock offset that pairs the most shots.

    Among offsets pairing equally many shots, the one whose pairs sit
    closest in time is chosen.

    Args:
        reference: Reference rows with times.
        tracker: Tracker rows with times.
        search: Offsets from -search to +search seconds are tried.
        step: Spacing of the offsets tried.
        tolerance: Pairing tolerance.

    Returns:
        (offset, pairs at that offset), and (0.0, 0) when either side has
        no shots.
    """
    ref = in_window(reference, None, None, 0.0)
    ours = in_window(tracker, None, None, 0.0)
    if not ref or not ours:
        return 0.0, 0
    best = (0.0, -1, math.inf)
    count = int(round(search / step))
    for k in range(-count, count + 1):
        offset = round(k * step, 6)
        pairs = match_shots(ref, ours, offset, tolerance)
        spread = sum(abs(ours[j].video_time - ref[i].video_time - offset) for i, j in pairs)
        if len(pairs) > best[1] or (len(pairs) == best[1] and spread < best[2]):
            best = (offset, len(pairs), spread)
    return best[0], best[1]


def _server_won(point_number: int, rows: list[RawShotRow], host: str) -> bool:
    """Whether the server won a point, by the same rule for both sources."""
    point = reconstruct.reconstruct_point(point_number, rows, host)
    return point.is_serving == point.point_won


@dataclass(frozen=True)
class ShotComparison:
    """How tracker shots compare with a reference export over one stretch.

    Attributes:
        clock_offset: Offset added to reference times.
        offset_estimated: Whether it was estimated rather than given.
        tolerance: Pairing tolerance, seconds.
        window: (start, end) on the video clock, or None for all of it.
        reference_shots: Reference shots in the window.
        tracker_shots: Tracker shots in the window.
        shot_recall: Reference shots paired with a tracker shot.
        shot_precision: Tracker shots paired with a reference shot.
        recall_by_side: Shot recall split by the reference's hitter end.
        hitter_side: Paired shots whose hitter end agrees.
        stroke: Paired shots, both with a named stroke, whose stroke
            family agrees.
        stroke_abstained: Paired shots the tracker could not name a
            stroke for, out of all paired shots.
        stroke_confusion: "reference->tracker" family pairs and counts.
        shot_type: Paired shots whose role (first_serve, first_return,
            in_play...) agrees.
        serve_recall: Reference serves paired with a tracker serve.
        server_side: Paired serves whose server end agrees.
        winner: Points (paired by their serve) where both say the same
            thing about whether the server won.
        bounce_error_m: Bounce distance error, metres, over paired shots
            that both placed; also split by landing half.
        bounce_error_by_half: Per landing half ("near"/"far").
        speed_difference_mph: Tracker minus reference speed, over paired
            shots with both speeds (signed).
        speed_abs_difference_mph: The same, as absolute differences.
        confidence_calibration: Per tracker-confidence bin, how many of
            its shots were paired with a reference shot.
        notes: Plain caveats.
    """

    clock_offset: float
    offset_estimated: bool
    tolerance: float
    window: tuple[float, float] | None
    reference_shots: int
    tracker_shots: int
    shot_recall: Ratio
    shot_precision: Ratio
    recall_by_side: dict[str, Ratio]
    hitter_side: Ratio
    stroke: Ratio
    stroke_abstained: Ratio
    stroke_confusion: dict[str, int]
    shot_type: Ratio
    serve_recall: Ratio
    server_side: Ratio
    winner: Ratio
    bounce_error_m: Distribution
    bounce_error_by_half: dict[str, Distribution]
    speed_difference_mph: Distribution
    speed_abs_difference_mph: Distribution
    confidence_calibration: list[dict]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field.
        """
        out = {}
        for name, value in self.__dict__.items():
            if isinstance(value, (Ratio, Distribution)):
                out[name] = value.to_dict()
            elif isinstance(value, dict):
                out[name] = {
                    k: v.to_dict() if isinstance(v, (Ratio, Distribution)) else v
                    for k, v in value.items()
                }
            elif isinstance(value, tuple):
                out[name] = list(value)
            else:
                out[name] = value
        return out

    def lines(self) -> list[str]:
        """The comparison as lines for a person.

        Returns:
            One line per measure, each with its denominator.
        """
        window = (
            "the whole recording"
            if self.window is None
            else (f"{self.window[0]:.0f}-{self.window[1]:.0f}s")
        )
        how = "estimated" if self.offset_estimated else "given"
        out = [
            f"Against a SwingVision export (a weak reference, not ground truth) over {window}; "
            f"clock offset {self.clock_offset:+.2f}s ({how}), pairing tolerance "
            f"{self.tolerance:.2f}s.",
            f"Shots: {self.reference_shots} in the reference, {self.tracker_shots} from the "
            "tracker.",
            f"  shot recall      {self.shot_recall}",
            f"  shot precision   {self.shot_precision}",
        ]
        for side, ratio in sorted(self.recall_by_side.items()):
            out.append(f"  recall, {side:4s} end {ratio}")
        out += [
            f"  hitter end       {self.hitter_side}",
            f"  stroke family    {self.stroke}  (tracker abstained on {self.stroke_abstained})",
            f"  shot role        {self.shot_type}",
            f"  serve recall     {self.serve_recall}",
            f"  server end       {self.server_side}",
            f"  server won point {self.winner}",
            f"  bounce error, m  {self.bounce_error_m}",
        ]
        for half, spread in sorted(self.bounce_error_by_half.items()):
            out.append(f"    landing {half:4s}   {spread}")
        out += [
            f"  speed diff, mph  {self.speed_difference_mph}  (tracker minus reference)",
            f"  |speed diff|     {self.speed_abs_difference_mph}",
        ]
        if self.stroke_confusion:
            shown = ", ".join(f"{k} {v}" for k, v in sorted(self.stroke_confusion.items()))
            out.append(f"  stroke pairs     {shown}")
        if self.confidence_calibration:
            out.append("  confidence calibration (tracker shots paired with a reference shot):")
            for row in self.confidence_calibration:
                out.append(f"    {row['bin']:>9s}  {Ratio(row['paired'], row['shots'])}")
        out.extend(f"Note: {note}" for note in self.notes)
        return out


def _calibration(
    tracker: Sequence[RawShotRow],
    paired: set[int],
    confidence: Mapping[tuple[int, int], float],
) -> list[dict]:
    """Paired share per tracker-confidence bin."""
    rows = []
    edges = CONFIDENCE_BINS
    for low, high in zip(edges[:-1], edges[1:]):
        inside = [
            j
            for j, row in enumerate(tracker)
            if (row.point_number, row.shot_number) in confidence
            and low <= confidence[(row.point_number, row.shot_number)] < high
        ]
        if inside:
            rows.append(
                {
                    "bin": f"{low:.1f}-{min(high, 1.0):.1f}",
                    "shots": len(inside),
                    "paired": sum(j in paired for j in inside),
                }
            )
    return rows


def compare_shots(
    reference: Sequence[RawShotRow],
    tracker: Sequence[RawShotRow],
    *,
    reference_host: str,
    tracker_host: str,
    clock_offset: float | None = None,
    tolerance: float = 0.15,
    serve_tolerance: float = 0.3,
    window: tuple[float, float] | None = None,
    confidence: Mapping[tuple[int, int], float] | None = None,
) -> ShotComparison:
    """Scores tracker shot rows against a reference export's.

    Args:
        reference: The export's shot rows.
        tracker: The tracker's shot rows (video-clock times).
        reference_host: The export's host name, for reconstructing its
            points.
        tracker_host: The tracker's host name.
        clock_offset: Added to reference times to put them on the video
            clock; estimated from the shots when None.
        tolerance: Pairing tolerance for shots, seconds.
        serve_tolerance: Pairing tolerance for serves - a serve's moment of
            contact is harder to time than a groundstroke's.
        window: (start, end) on the video clock to score over; the
            tracker's own span when None.
        confidence: (point, shot) -> the tracker's hit confidence, for the
            calibration table; omitted when None.

    Returns:
        The comparison.
    """
    notes = []
    estimated = clock_offset is None
    tracker_all = in_window(tracker, None, None, 0.0)
    if window is None and tracker_all:
        window = (tracker_all[0].video_time - 1.0, tracker_all[-1].video_time + 1.0)
    if estimated:
        nearby = in_window(reference, None, None, 0.0)
        clock_offset, _ = estimate_clock_offset(nearby, tracker_all, tolerance=tolerance)
        notes.append(
            "The clock offset was estimated by pairing as many shots as possible; with few "
            "shots it can lock onto a wrong offset - pass the known one when there is one."
        )
    start, end = window if window is not None else (None, None)
    ref = in_window(reference, start, end, clock_offset)
    ours = in_window(tracker, start, end, 0.0)
    pairs = match_shots(ref, ours, clock_offset, tolerance)
    paired_ref = {i for i, _ in pairs}
    paired_ours = {j for _, j in pairs}

    sides: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for i, row in enumerate(ref):
        tally = sides[row.hit_side or "unknown"]
        tally[1] += 1
        tally[0] += i in paired_ref

    side_ok = side_n = stroke_ok = stroke_n = abstained = role_ok = role_n = 0
    confusion: Counter = Counter()
    bounce, bounce_half, speed = [], defaultdict(list), []
    for i, j in pairs:
        a, b = ref[i], ours[j]
        if a.hit_side and b.hit_side:
            side_n += 1
            side_ok += a.hit_side == b.hit_side
        fa, fb = stroke_family(a.stroke), stroke_family(b.stroke)
        if fb is None:
            abstained += 1
        elif fa is not None:
            stroke_n += 1
            stroke_ok += fa == fb
            confusion[f"{fa}->{fb}"] += 1
        if a.shot_type and b.shot_type and a.shot_type != "none":
            role_n += 1
            role_ok += a.shot_type == b.shot_type
        if None not in (a.bounce_x, a.bounce_y, b.bounce_x, b.bounce_y):
            error = math.hypot(a.bounce_x - b.bounce_x, a.bounce_y - b.bounce_y)
            bounce.append(error)
            bounce_half["far" if a.bounce_y > _NET_Y_M else "near"].append(error)
        if a.speed_mph and b.speed_mph is not None:
            speed.append(b.speed_mph - a.speed_mph)

    ref_serves = [i for i, row in enumerate(ref) if is_serve(row)]
    our_serves = [j for j, row in enumerate(ours) if is_serve(row)]
    serve_pairs = match_shots(
        [ref[i] for i in ref_serves], [ours[j] for j in our_serves], clock_offset, serve_tolerance
    )
    server_ok = sum(
        ref[ref_serves[a]].hit_side == ours[our_serves[b]].hit_side for a, b in serve_pairs
    )
    winner_ok = winner_n = 0
    ref_points = reconstruct.group_shots_by_point(ref)
    our_points = reconstruct.group_shots_by_point(ours)
    for a, b in serve_pairs:
        ref_row, our_row = ref[ref_serves[a]], ours[our_serves[b]]
        mine = _server_won(ref_row.point_number, ref_points[ref_row.point_number], reference_host)
        theirs = _server_won(our_row.point_number, our_points[our_row.point_number], tracker_host)
        winner_n += 1
        winner_ok += mine == theirs
    if not ref:
        notes.append("No reference shots fall in the window: nothing could be scored.")
    if bounce and len(bounce) < 10:
        notes.append(f"Only {len(bounce)} paired bounce(s): the error spread is not reliable.")
    return ShotComparison(
        clock_offset=float(clock_offset),
        offset_estimated=estimated,
        tolerance=tolerance,
        window=None if window is None else (float(window[0]), float(window[1])),
        reference_shots=len(ref),
        tracker_shots=len(ours),
        shot_recall=Ratio(len(paired_ref), len(ref)),
        shot_precision=Ratio(len(paired_ours), len(ours)),
        recall_by_side={side: Ratio(*tally) for side, tally in sides.items()},
        hitter_side=Ratio(side_ok, side_n),
        stroke=Ratio(stroke_ok, stroke_n),
        stroke_abstained=Ratio(abstained, len(pairs)),
        stroke_confusion=dict(confusion),
        shot_type=Ratio(role_ok, role_n),
        serve_recall=Ratio(len(serve_pairs), len(ref_serves)),
        server_side=Ratio(server_ok, len(serve_pairs)),
        winner=Ratio(winner_ok, winner_n),
        bounce_error_m=Distribution.of(bounce),
        bounce_error_by_half={half: Distribution.of(v) for half, v in bounce_half.items()},
        speed_difference_mph=Distribution.of(speed),
        speed_abs_difference_mph=Distribution.of(abs(v) for v in speed),
        confidence_calibration=_calibration(ours, paired_ours, confidence or {}),
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class ConfirmedComparison:
    """How the tracker's own guesses compare with what a reviewer confirmed.

    Attributes:
        confirmed_points: Points a person confirmed (`confirmed_by` is
            "human").
        traced_points: Of those, points whose tracker guess could be
            recovered from the staged shot rows.
        point_won: Points where the tracker's guess of who won agrees.
        end_type: Points where its `point_end_type` agrees.
        end_type_pairs: "tracker->confirmed" counts.
        notes: Plain caveats.
    """

    confirmed_points: int
    traced_points: int
    point_won: Ratio
    end_type: Ratio
    end_type_pairs: dict[str, int] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            Every field.
        """
        return {
            "confirmed_points": self.confirmed_points,
            "traced_points": self.traced_points,
            "point_won": self.point_won.to_dict(),
            "end_type": self.end_type.to_dict(),
            "end_type_pairs": dict(self.end_type_pairs),
            "notes": list(self.notes),
        }

    def lines(self) -> list[str]:
        """The comparison as lines for a person.

        Returns:
            One line per measure.
        """
        out = [
            f"Against the reviewer's confirmations (human judgements): {self.confirmed_points} "
            f"point(s) confirmed by a person, {self.traced_points} traceable to the tracker's "
            "guess.",
            f"  who won          {self.point_won}",
            f"  how it ended     {self.end_type}",
        ]
        if self.end_type_pairs:
            shown = ", ".join(f"{k} {v}" for k, v in sorted(self.end_type_pairs.items()))
            out.append(f"  pairs            {shown}")
        out.extend(f"Note: {note}" for note in self.notes)
        return out


def compare_confirmed(record) -> ConfirmedComparison:
    """Scores a staged tracked match's original guesses against its confirmations.

    The staged points carry the confirmed values; the tracker's original
    guess is rebuilt from the shot rows staged with the match
    (`MatchRecord.tracking["shots"]`) by the same reconstruction that
    produced it, matched by `source_point_number`.

    Args:
        record: A `swingvision_import.records.MatchRecord`.

    Returns:
        The comparison; empty counts for a match with no tracking block
        or no confirmed points.
    """
    confirmed = [
        point
        for set_record in record.sets
        for point in set_record.points
        if point.confirmed_by == "human"
    ]
    notes = []
    shots = (record.tracking or {}).get("shots") or []
    host = ((record.tracking or {}).get("settings") or {}).get("host_name", "")
    rows = [RawShotRow(**row) for row in shots]
    guesses = {}
    if rows:
        result = reconstruct.reconstruct_all(rows, host)
        guesses = {
            point.source_point_number: point
            for set_record in result.sets
            for point in set_record.points
        }
    elif record.tracking is None:
        notes.append("This match was not staged by the tracker: there is no guess to compare.")
    won_ok = end_ok = traced = 0
    pairs: Counter = Counter()
    for point in confirmed:
        guess = guesses.get(point.source_point_number)
        if guess is None:
            continue
        traced += 1
        won_ok += guess.point_won == point.point_won
        end_ok += guess.point_end_type == point.point_end_type
        pairs[f"{guess.point_end_type}->{point.point_end_type}"] += 1
    if confirmed and traced < len(confirmed):
        notes.append(
            f"{len(confirmed) - traced} confirmed point(s) could not be traced to a tracker guess."
        )
    return ConfirmedComparison(
        confirmed_points=len(confirmed),
        traced_points=traced,
        point_won=Ratio(won_ok, traced),
        end_type=Ratio(end_ok, traced),
        end_type_pairs=dict(pairs),
        notes=tuple(notes),
    )


def tracker_confidence(tracking: Mapping | None) -> dict[tuple[int, int], float]:
    """Reads each staged tracker shot's hit confidence.

    Args:
        tracking: `MatchRecord.tracking`.

    Returns:
        (point number, shot number) -> confidence, from every serve
        attempt's shot traces (folded faults included).
    """
    found = {}
    for attempt in (tracking or {}).get("attempts") or []:
        traces = list(attempt.get("shots") or [])
        folded = attempt.get("folded_fault")
        if folded:
            traces += list(folded.get("shots") or [])
        for trace in traces:
            found[(int(trace["point_number"]), int(trace["shot_number"]))] = float(
                trace["confidence"]
            )
    return found
