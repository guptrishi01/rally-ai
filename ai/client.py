"""Calls one coaching specialist and parses its structured JSON output.

The Anthropic client is always passed in, never constructed here — that's
what lets tests inject a fake client and never touch the real API or spend
real money.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

import anthropic

from .config import AICoachConfig
from .records import Category, CoachingItem, FitnessItem, PracticeIdeaItem, SupportingStat

logger = logging.getLogger(__name__)

_ITEM_CLASSES: dict[str, type] = {
    "strategy": CoachingItem,
    "practice_idea": PracticeIdeaItem,
    "fitness": FitnessItem,
}

_EXTRA_FIELDS: dict[str, list[str]] = {
    "strategy": [],
    "practice_idea": ["practice_name", "how_to_try"],
    "fitness": ["focus_area"],
}


class _Message(Protocol):
    content: list[object]


class _Messages(Protocol):
    def create(self, **kwargs: object) -> _Message: ...


def extract_text(response: _Message) -> str:
    """Finds the first text block in a response, skipping non-text blocks.

    Claude Sonnet 5 emits extended-thinking blocks ahead of its text block
    by default, so `response.content[0]` is not reliably the text — a real
    call against the live API confirmed `content[0]` can be a ThinkingBlock
    with no `.text` attribute at all. Shared here rather than duplicated
    because both ai/client.py and swingvision_import/review_assist.py parse
    responses the same way.

    Args:
        response: The API response (or test fake) to extract text from.

    Returns:
        The first text block's content.

    Raises:
        ValueError: If the response has no block with type "text".
    """
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError("Response contained no text block")


def strip_markdown_fence(text: str) -> str:
    """Strips a leading/trailing markdown code fence, if present.

    Claude Sonnet 5 sometimes wraps a JSON response in ```json ... ``` even
    when the prompt explicitly says "no markdown fences" — confirmed
    against the live API, not hypothetical: a real response came back as
    '```json\\n{...}\\n```', which json.loads() rejects outright (fails on
    the leading backtick). Stripping this first is what actually makes
    "respond with ONLY a JSON object" reliable rather than aspirational.
    Shared here since every JSON-parsing call site in this codebase
    (call_specialist below, review_assist.py, review_resolve.py) hits the
    same real failure mode.

    Args:
        text: The raw response text, possibly fenced.

    Returns:
        The text with a single leading/trailing ``` fence (and optional
        language tag on the opening line) removed; the original text
        unchanged if it wasn't fenced.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.split("\n")
    if len(lines) > 1 and lines[-1].strip() == "```":
        lines = lines[1:-1]
    else:
        lines = lines[1:]
    return "\n".join(lines).strip()


class AnthropicClientLike(Protocol):
    """The minimal shape of anthropic.Anthropic this module depends on.

    Defined as a Protocol (structural typing) rather than importing
    anthropic.Anthropic directly, so a test's fake client only needs to
    match this shape, not subclass or mock the real SDK class.
    """

    messages: _Messages


class SpecialistError(RuntimeError):
    """Raised when a coaching specialist call fails.

    Covers both failure modes: the API call itself failing (auth, rate
    limit, network — an `anthropic.APIError`), and the API call succeeding
    but returning a response that isn't valid JSON matching the expected
    item schema. Both are handled identically by generate.py — one
    specialist failing degrades that section of the report, not the whole
    thing — so they share one exception type rather than two.

    Attributes:
        category: Which specialist failed.
        raw_response: The raw text that failed to parse, or "" if the API
            call itself failed before any response was received.
    """

    def __init__(self, category: str, raw_response: str, cause: Exception) -> None:
        super().__init__(f"{category} specialist failed: {cause}")
        self.category = category
        self.raw_response = raw_response


def call_specialist(
    client: AnthropicClientLike,
    config: AICoachConfig,
    category: Category,
    system_prompt: str,
) -> list[CoachingItem]:
    """Calls one coaching specialist and parses its structured JSON output.

    Args:
        client: An anthropic.Anthropic-shaped client.
        config: Model/token settings.
        category: "strategy", "practice_idea", or "fitness" — selects which
            dataclass to parse items into.
        system_prompt: The specialist's full system prompt, already built
            by prompts.py (includes the context and the response schema).

    Returns:
        The parsed list of items for this specialist.

    Raises:
        SpecialistError: If the API call itself fails (auth, rate limit,
            network — any `anthropic.APIError`), or if it succeeds but the
            response isn't valid JSON matching the expected item schema.
    """
    try:
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": "Generate the tips now."}],
        )
    except anthropic.APIError as exc:
        raise SpecialistError(category, "", exc) from exc

    item_class = _ITEM_CLASSES[category]
    extra_fields = _EXTRA_FIELDS[category]
    raw_text = ""
    try:
        raw_text = extract_text(response)
        raw_items = json.loads(strip_markdown_fence(raw_text))
        return [
            item_class(
                category=category,
                observation=raw["observation"],
                recommendation=raw["recommendation"],
                supporting_stat=SupportingStat(**raw["supporting_stat"]),
                priority=raw["priority"],
                **{field: raw[field] for field in extra_fields},
            )
            for raw in raw_items
        ]
    except (ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SpecialistError(category, raw_text, exc) from exc
