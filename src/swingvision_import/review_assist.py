"""Claude-assisted suggestions for flagged SwingVision points.

Reuses `ai.client`'s `AnthropicClientLike` Protocol and injected-client
pattern rather than reinventing it — tests never touch the real API here
either. A suggestion is always additional context for the human reviewer,
never a resolution: `suggest_point_resolution` attaches its result to a
point's `ai_suggested_point_end_type`/`ai_suggestion_reasoning` fields, but
never touches `needs_review` — per the user's explicit rule, a Claude
suggestion is one more thing to confirm, not a second silent auto-tagger.

Reasoning is over structured shot data already tracked by SwingVision
(stroke, speed context, direction, in/out/net result per shot) — no video
or vision involved.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import anthropic

from ai.client import AnthropicClientLike, extract_text, strip_markdown_fence

from .raw import RawShotRow
from .records import VALID_END_TYPES, PointRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SuggestionConfig:
    """Settings for generating a point-resolution suggestion.

    Attributes:
        model: Claude model id.
        max_tokens: Max output tokens per call.
    """

    model: str = "claude-sonnet-5"
    max_tokens: int = 512


@dataclass(frozen=True)
class PointSuggestion:
    """A Claude-suggested refinement of a flagged point's classification.

    Attributes:
        point_end_type: The suggested value — one of the point table's
            allowed point_end_type values.
        reasoning: Why Claude suggested it, shown to the user during review.
        confidence: How confident Claude is: "high", "medium", or "low".
    """

    point_end_type: str
    reasoning: str
    confidence: str


class SuggestionError(RuntimeError):
    """Raised when a point-suggestion call fails.

    Covers both the API call itself failing (auth, rate limit, network —
    an `anthropic.APIError`) and a response that parses but doesn't match
    the expected shape — mirrors ai/client.py's SpecialistError in spirit.

    Attributes:
        raw_response: The raw text that failed to parse, or "" if the API
            call itself failed before any response was received.
    """

    def __init__(self, raw_response: str, cause: Exception) -> None:
        super().__init__(f"Point suggestion failed: {cause}")
        self.raw_response = raw_response


def _build_prompt(point: PointRecord, shot_context: list[RawShotRow]) -> str:
    """Builds the prompt describing one flagged point's shot sequence.

    Args:
        point: The flagged point. Its current point_end_type (SwingVision's
            or reconstruct.py's best guess) is shown as context, explicitly
            labeled as unreliable.
        shot_context: The point's raw shots — already-tracked structured
            data (stroke, direction, in/out/net result), not video.

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
        "You are a tennis point-outcome analyst. Given the shot-by-shot "
        "sequence for one point, suggest the most likely classification.\n\n"
        f'Current best guess: "{point.point_end_type}" — unreliable, this is '
        "exactly what you're being asked to double check.\n\n"
        f"Shot sequence (JSON):\n{shots_json}\n\n"
        "Respond with ONLY a JSON object (no prose, no markdown fences) with "
        "exactly these keys:\n"
        '  "point_end_type": one of "winner", "unforced_error", '
        '"forced_error", "ace", "double_fault", "return_winner", '
        '"return_error"\n'
        '  "reasoning": string — why, in one or two sentences\n'
        '  "confidence": "high" | "medium" | "low"'
    )


def suggest_point_resolution(
    client: AnthropicClientLike,
    config: SuggestionConfig,
    point: PointRecord,
    shot_context: list[RawShotRow],
) -> PointSuggestion:
    """Asks Claude to suggest a refined classification for one flagged point.

    Args:
        client: An anthropic.Anthropic-shaped client (injected so tests
            never hit the real API or spend real money).
        config: Model/token settings.
        point: The flagged point to get a suggestion for.
        shot_context: The point's raw shots, for Claude to reason over.

    Returns:
        The suggested resolution. Never applied automatically — the caller
        is responsible for attaching it to the point and leaving
        needs_review untouched.

    Raises:
        SuggestionError: If the API call itself fails, or its response
            isn't valid JSON matching the expected shape.
    """
    system_prompt = _build_prompt(point, shot_context)
    try:
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": "Suggest the classification now."}],
        )
    except anthropic.APIError as exc:
        raise SuggestionError("", exc) from exc

    raw_text = ""
    try:
        raw_text = extract_text(response)
        data = json.loads(strip_markdown_fence(raw_text))
        point_end_type = data["point_end_type"]
        if point_end_type not in VALID_END_TYPES:
            raise ValueError(f"not a valid point_end_type: {point_end_type!r}")
        return PointSuggestion(
            point_end_type=point_end_type,
            reasoning=data["reasoning"],
            confidence=data["confidence"],
        )
    except (ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SuggestionError(raw_text, exc) from exc
