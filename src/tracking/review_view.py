"""Plain summaries of a staged `tracking` block, for a person reviewing it.

The staged JSON keeps every tracker value with its provenance (spec
section 6.1), but in a shape built for audit, not for reading at a glance.
The review modal needs two smaller things: the match's coverage line and
notes, and per shot which fields were *measured* and which were bridged,
inferred or plainly guessed. This module reduces the stored block to
exactly that and nothing more.

It decides nothing. A summary never clears `needs_review`, never ranks a
point as safe to skip, and never turns a guess into a measurement: a field
in a shot's `guesses` is reported as a guess whatever its tag says.
"""

from __future__ import annotations

from collections.abc import Mapping

from .assemble import AttemptBlock, ShotTrace, TrackingBlock
from .calibrate import CalibrationQuality, setup_advice
from .provenance import Provenance

# The order fields are listed in, so the same shot always reads the same
# way: who and when first, then what was hit, then where it went.
_FIELD_ORDER = (
    "player",
    "hit_side",
    "video_time",
    "shot_type",
    "stroke",
    "result",
    "hit_zone",
    "hit_xy",
    "bounce_xy",
    "speed_mph",
    "direction",
)

# What each stored field is called on screen.
FIELD_LABELS = {
    "player": "who hit it",
    "hit_side": "court end",
    "video_time": "time",
    "shot_type": "shot type",
    "stroke": "stroke",
    "result": "result",
    "hit_zone": "zone",
    "hit_xy": "hit position",
    "bounce_xy": "bounce",
    "speed_mph": "speed",
    "direction": "direction",
}


def _ordered(names) -> list[str]:
    """Sorts field names into display order, unknown ones last by name."""
    known = [name for name in _FIELD_ORDER if name in names]
    return known + sorted(name for name in names if name not in _FIELD_ORDER)


def shot_provenance(trace: ShotTrace, part: str = "point") -> dict:
    """Says how each value of one emitted shot row was obtained.

    Args:
        trace: The shot's trace from the staged JSON.
        part: "point" for the point's own shots, "folded_fault" for the
            shots of a first-serve fault folded into it (whose rows carry
            another point number).

    Returns:
        A dict with the part, the shot number, whether the hit counts as tracked
        with confidence, its confidence, and `fields`: one entry per
        stored field in display order, each with its label, provenance
        ("measured", "interpolated", "inferred" or "user_entered"),
        confidence (or None) and `guess` - True when the value is a guess
        from context with no evidence of its own. Plus the trace's notes.
    """
    fields = []
    for name in _ordered(trace.fields):
        tagged = trace.fields[name]
        fields.append(
            {
                "field": name,
                "label": FIELD_LABELS.get(name, name.replace("_", " ")),
                "provenance": tagged.provenance.value,
                "confidence": tagged.confidence,
                "guess": name in trace.guesses,
            }
        )
    # A guess the trace names but that has no field of its own (an
    # inferred serve row) is still a guess, and is still said.
    for name in trace.guesses:
        if name not in trace.fields:
            fields.append(
                {
                    "field": name,
                    "label": FIELD_LABELS.get(name, name.replace("_", " ")),
                    "provenance": Provenance.INFERRED.value,
                    "confidence": None,
                    "guess": True,
                }
            )
    return {
        "part": part,
        "shot_number": trace.shot_number,
        "tracked": trace.tracked,
        "confidence": trace.confidence,
        "fields": fields,
        "notes": list(trace.notes),
    }


def point_provenance(block: Mapping | None) -> dict | None:
    """Summarises one point's `tracking` block for the review modal.

    Args:
        block: `PointRecord.tracking` - the serve attempt's block as the
            tracker staged it - or None for a point that did not come
            from the tracker.

    Returns:
        None when there is no block. Otherwise the attempt's kind, its
        segmentation confidence and the reasons behind it, the server's
        end, whether a first-serve fault was folded into it, and one
        `shot_provenance` per shot - the folded fault's shots first (part
        "folded_fault"), as they happened first.
    """
    if not block:
        return None
    attempt = AttemptBlock.from_dict(dict(block))
    shots = [shot_provenance(trace) for trace in attempt.shots]
    if attempt.folded_fault is not None:
        folded = [shot_provenance(t, "folded_fault") for t in attempt.folded_fault.shots]
        shots = folded + shots
    return {
        "kind": attempt.kind,
        "confidence": attempt.confidence,
        "reasons": list(attempt.reasons),
        "server_side": attempt.server_side,
        "second_serve": attempt.second_serve,
        "folded_fault": attempt.folded_fault is not None,
        "shots": shots,
    }


def match_provenance(block: Mapping | None) -> dict | None:
    """Summarises a match's `tracking` block for the review modal.

    Args:
        block: `MatchRecord.tracking`, or None for an imported match.

    Returns:
        None for an imported match. Otherwise the tracker's notes (the
        first is its coverage line, "Tracked N of M shots with
        confidence"), the shot counts behind it, how many results and
        strokes were guessed, the models that produced it - so a
        reviewer sees the denominator beside every number - and two
        informational panels: `setup` (`calibrate.setup_advice`: the
        camera position's score and what it makes weak) and `retraining`
        (whether a ball model retrained on footage like this is
        suggested, and why). Either is None when the block lacks it.
    """
    if not block:
        return None
    parsed = TrackingBlock.from_dict(dict(block))
    summary = parsed.summary
    guesses = summary.get("guesses", {})
    video = parsed.extra.get("video") or {}
    return {
        "notes": list(parsed.notes),
        "shots": summary.get("shots"),
        "tracked_shots": summary.get("tracked_shots"),
        "guessed_results": guesses.get("result"),
        "guessed_strokes": guesses.get("stroke"),
        "inferred_serves": guesses.get("serve_row"),
        "models": parsed.extra.get("models") or summary.get("models") or {},
        "video": video.get("name"),
        "range": parsed.extra.get("range"),
        "setup": _setup(parsed.extra.get("calibration")),
        "retraining": _retraining(parsed.extra.get("retraining")),
    }


def _setup(calibration: Mapping | None) -> dict | None:
    """The camera-setup advice, from the stored calibration's quality."""
    quality = (calibration or {}).get("quality")
    if not quality:
        return None
    return setup_advice(CalibrationQuality.from_dict(dict(quality))).to_dict()


def _retraining(advice: Mapping | None) -> dict | None:
    """The retraining suggestion, reduced to what a reviewer reads."""
    if not advice:
        return None
    return {
        "recommended": bool(advice.get("recommended")),
        "note": advice.get("note"),
        "conditions": list(advice.get("conditions") or []),
        "weak_tracking": bool(advice.get("weak_tracking")),
        "informational": True,
    }
