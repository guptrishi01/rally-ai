"""Parses the human reviewer's own free-text answer into structured point fields.

A different trust relationship than review_assist.py's suggestions: there,
Claude reasons independently over shot data with no human input at all —
its own best guess, always secondary to the human's judgment. Here, the
human has already stated their own conclusion in PointRecord.review_answer
(e.g. "she was way out of position, that was a clean winner down the
line") — Claude's job is purely to translate that plain-language answer
into the point table's structured vocabulary (point_end_type/point_won/
net_approach), not to make an independent judgment call.

Still never clears needs_review by itself: the parsed result lands in the
point's resolved_* fields, and a separate, explicit
pipeline.apply_resolutions() step is what actually copies them onto the
real fields and clears the flag — one more human checkpoint, since a
parsing mistake here is a correctness bug just like anywhere else in this
pipeline, and the human should see what Claude understood before it's
applied.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import anthropic

from ai.client import AnthropicClientLike, extract_text, strip_markdown_fence

from .raw import RawShotRow
from .records import LOSING_END_TYPES, VALID_END_TYPES, WINNING_END_TYPES, PointRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolutionConfig:
    """Settings for parsing a review answer into structured fields.

    Attributes:
        model: Claude model id.
        max_tokens: Max output tokens per call.
    """

    model: str = "claude-sonnet-5"
    max_tokens: int = 512


@dataclass(frozen=True)
class PointResolution:
    """Claude's structured parse of the human's free-text review answer.

    Attributes:
        point_end_type: One of the point table's allowed values.
        point_won: Whether the tracked player won the point.
        net_approach: Whether the tracked player approached the net.
        reasoning: How Claude read the human's answer into these values,
            shown back to the human before pipeline.apply_resolutions().
    """

    point_end_type: str
    point_won: bool
    net_approach: bool
    reasoning: str


class ResolutionError(RuntimeError):
    """Raised when parsing a review answer fails.

    Covers both the API call itself failing (auth, rate limit, network —
    an `anthropic.APIError`) and a response that parses but doesn't match
    the expected shape — mirrors review_assist.SuggestionError in spirit.

    Attributes:
        raw_response: The raw text that failed to parse, or "" if the API
            call itself failed before any response was received.
    """

    def __init__(self, raw_response: str, cause: Exception) -> None:
        super().__init__(f"Review-answer parsing failed: {cause}")
        self.raw_response = raw_response


def _build_prompt(point: PointRecord, shot_context: list[RawShotRow]) -> str:
    """Builds the prompt asking Claude to parse the human's review answer.

    Args:
        point: The flagged point, carrying the human's review_answer text
            plus whatever current point_end_type/point_won it has (from
            SwingVision, reconstruct.py, or a prior AI suggestion — all
            explicitly labeled as unreliable, secondary to review_answer).
        shot_context: The point's raw shots, if any were traceable — extra
            structured context, not required for the human's answer to be
            parseable on its own.

    Returns:
        The complete system prompt.
    """
    shots_json = json.dumps(
        [
            {
                "shot_number": s.shot_number,
                "player": s.player,
                "type": s.shot_type,
                "stroke": s.stroke,
                "result": s.result,
            }
            for s in shot_context
        ],
        indent=2,
    )
    return (
        "You are transcribing a tennis coach's own review notes into "
        "structured data. The human has already reviewed this point and "
        "written their own conclusion below — your job is ONLY to "
        "translate their plain-language answer into the fields below, "
        "not to second-guess or independently re-judge the point. If "
        "their answer is ambiguous or doesn't clearly resolve one of the "
        "fields, make your best-effort reading and say so plainly in "
        "your reasoning.\n\n"
        f'The human\'s review answer: "{point.review_answer}"\n\n'
        f'Current (unreliable, pre-review) point_end_type: "{point.point_end_type}", '
        f"point_won: {point.point_won}, net_approach: {point.net_approach}.\n\n"
        f"Shot sequence, if available (JSON):\n{shots_json}\n\n"
        "IMPORTANT: point_end_type is always from the tracked player's own "
        "perspective, and must always pair with point_won consistently: "
        '"ace", "winner", and "return_winner" always mean the tracked '
        'player WON the point; "double_fault", "unforced_error", '
        '"forced_error", and "return_error" always mean they LOST it. '
        "There is no category for \"the opponent hit an ace/winner past "
        "me\" — if the human's answer describes the opponent hitting an "
        "untouchable serve or shot, that means the tracked player lost the "
        'point on a return they couldn\'t make, i.e. "return_error" with '
        "point_won=false — not \"ace\"/\"winner\" with point_won=false, "
        "which is never a valid combination.\n\n"
        "Respond with ONLY a JSON object (no prose, no markdown fences) with "
        "exactly these keys:\n"
        '  "point_end_type": one of "winner", "unforced_error", '
        '"forced_error", "ace", "double_fault", "return_winner", '
        '"return_error"\n'
        '  "point_won": boolean — did the tracked player win this point\n'
        '  "net_approach": boolean — did the tracked player approach the net\n'
        '  "reasoning": string — how you read the human\'s answer into these values'
    )


def resolve_point_answer(
    client: AnthropicClientLike,
    config: ResolutionConfig,
    point: PointRecord,
    shot_context: list[RawShotRow],
) -> PointResolution:
    """Asks Claude to parse the human's review_answer into structured fields.

    Args:
        client: An anthropic.Anthropic-shaped client (injected so tests
            never hit the real API or spend real money).
        config: Model/token settings.
        point: The flagged point, with a non-empty review_answer.
        shot_context: The point's raw shots, if traceable (may be empty).

    Returns:
        The parsed resolution. Never applied automatically — the caller
        attaches it to the point's resolved_* fields and leaves
        needs_review untouched; pipeline.apply_resolutions() is the only
        thing that actually applies it.

    Raises:
        ResolutionError: If the API call itself fails, or its response
            isn't valid JSON matching the expected shape.
    """
    system_prompt = _build_prompt(point, shot_context)
    try:
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": "Parse the review answer now."}],
        )
    except anthropic.APIError as exc:
        raise ResolutionError("", exc) from exc

    raw_text = ""
    try:
        raw_text = extract_text(response)
        data = json.loads(strip_markdown_fence(raw_text))
        point_end_type = data["point_end_type"]
        if point_end_type not in VALID_END_TYPES:
            raise ValueError(f"not a valid point_end_type: {point_end_type!r}")
        point_won = bool(data["point_won"])
        # Must match data/schema.sql's CHECK constraint, or apply_resolutions()
        # would stage a point that blows up finalize() with a confusing
        # IntegrityError far removed from this point - confirmed against
        # the live API: it produced exactly this inconsistent combination
        # once (point_end_type="ace" with point_won=false) despite the
        # prompt's explicit guidance against it.
        if point_end_type in WINNING_END_TYPES and not point_won:
            raise ValueError(f"{point_end_type!r} requires point_won=true, got false")
        if point_end_type in LOSING_END_TYPES and point_won:
            raise ValueError(f"{point_end_type!r} requires point_won=false, got true")
        return PointResolution(
            point_end_type=point_end_type,
            point_won=point_won,
            net_approach=bool(data["net_approach"]),
            reasoning=data["reasoning"],
        )
    except (ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ResolutionError(raw_text, exc) from exc
