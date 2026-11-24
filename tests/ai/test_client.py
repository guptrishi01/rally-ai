from __future__ import annotations

import json

import pytest

from ai.client import SpecialistError, call_specialist, strip_markdown_fence
from ai.config import AICoachConfig
from ai.records import FitnessItem, PracticeIdeaItem
from tests.ai.conftest import (
    FakeAnthropicClient,
    FakeMessage,
    FakeTextBlock,
    FakeThinkingBlock,
    canned_item,
)


def test_call_specialist_parses_a_valid_strategy_response():
    client = FakeAnthropicClient()

    items = call_specialist(client, AICoachConfig(), "strategy", "You are a strategy analyst.")

    assert len(items) == 1
    assert items[0].category == "strategy"
    assert items[0].supporting_stat.stat == "FS%"


def test_call_specialist_parses_practice_idea_specific_fields():
    client = FakeAnthropicClient()

    items = call_specialist(
        client,
        AICoachConfig(),
        "practice_idea",
        "You suggest practice ideas a player could try",
    )

    assert isinstance(items[0], PracticeIdeaItem)
    assert items[0].practice_name == "Cross-court consistency"
    # Prescriptive-frequency fields are gone from the data model; what
    # replaces them is a one-sentence "how to try it" suggestion.
    assert "deep cross-court" in items[0].how_to_try


def test_call_specialist_parses_fitness_specific_fields():
    client = FakeAnthropicClient()

    items = call_specialist(
        client,
        AICoachConfig(),
        "fitness",
        "You surface physical and mental patterns worth noticing",
    )

    assert isinstance(items[0], FitnessItem)
    assert items[0].focus_area == "endurance"


def test_call_specialist_raises_specialist_error_on_invalid_json():
    client = FakeAnthropicClient(raise_for={"strategy"})

    with pytest.raises(SpecialistError) as exc_info:
        call_specialist(client, AICoachConfig(), "strategy", "You are a strategy analyst.")

    assert exc_info.value.category == "strategy"
    assert exc_info.value.raw_response == "not valid json"


def test_call_specialist_raises_specialist_error_on_missing_required_key():
    # Missing supporting_stat/priority.
    incomplete_item = {"observation": "x", "recommendation": "y"}
    client = FakeAnthropicClient(response_for={"strategy": [incomplete_item]})

    with pytest.raises(SpecialistError):
        call_specialist(client, AICoachConfig(), "strategy", "You are a strategy analyst.")


def test_call_specialist_passes_model_and_token_config_through():
    client = FakeAnthropicClient()
    config = AICoachConfig(model="claude-sonnet-5", max_tokens=512)

    call_specialist(client, config, "strategy", "You are a strategy analyst.")

    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-5"
    assert call["max_tokens"] == 512
    assert "temperature" not in call


class _ThinkingFirstMessages:
    """Stands in for a real Claude Sonnet 5 response: a ThinkingBlock ahead
    of the text block, confirmed against the live API - content[0] is not
    reliably the text block."""

    def create(self, **kwargs: object) -> FakeMessage:
        payload = json.dumps([canned_item("strategy")])
        return FakeMessage(
            content=[FakeThinkingBlock(thinking="reasoning..."), FakeTextBlock(text=payload)]
        )


class _ThinkingFirstClient:
    def __init__(self) -> None:
        self.messages = _ThinkingFirstMessages()


class _ThinkingOnlyMessages:
    """A response with no text block at all."""

    def create(self, **kwargs: object) -> FakeMessage:
        return FakeMessage(content=[FakeThinkingBlock(thinking="reasoning...")])


class _ThinkingOnlyClient:
    def __init__(self) -> None:
        self.messages = _ThinkingOnlyMessages()


def test_call_specialist_skips_a_leading_thinking_block_to_find_the_text():
    items = call_specialist(
        _ThinkingFirstClient(), AICoachConfig(), "strategy", "You are a strategy analyst."
    )

    assert len(items) == 1
    assert items[0].category == "strategy"


def test_call_specialist_raises_specialist_error_when_no_text_block_exists():
    with pytest.raises(SpecialistError) as exc_info:
        call_specialist(
            _ThinkingOnlyClient(), AICoachConfig(), "strategy", "You are a strategy analyst."
        )

    assert exc_info.value.raw_response == ""


def test_call_specialist_wraps_a_real_api_failure_as_specialist_error():
    # The API call itself failing (auth, rate limit, network) is a
    # different failure mode than a malformed response, but must be caught
    # the same way - not left to propagate and kill the whole report.
    client = FakeAnthropicClient(api_error_for={"strategy"})

    with pytest.raises(SpecialistError) as exc_info:
        call_specialist(client, AICoachConfig(), "strategy", "You are a strategy analyst.")

    assert exc_info.value.category == "strategy"
    assert exc_info.value.raw_response == ""
    assert "simulated connection failure" in str(exc_info.value)


class _FencedMessages:
    """Stands in for a real (confirmed, not hypothetical) Claude Sonnet 5
    response that wraps its JSON in a markdown code fence despite the
    prompt explicitly saying not to."""

    def create(self, **kwargs):
        payload = json.dumps([canned_item("strategy")])
        return FakeMessage(content=[FakeTextBlock(text=f"```json\n{payload}\n```")])


class _FencedClient:
    def __init__(self):
        self.messages = _FencedMessages()


def test_call_specialist_strips_a_markdown_fence_before_parsing():
    items = call_specialist(
        _FencedClient(), AICoachConfig(), "strategy", "You are a strategy analyst."
    )

    assert len(items) == 1
    assert items[0].category == "strategy"


def test_strip_markdown_fence_removes_a_fence_with_a_language_tag():
    fenced = '```json\n{"a": 1}\n```'
    assert strip_markdown_fence(fenced) == '{"a": 1}'


def test_strip_markdown_fence_removes_a_bare_fence():
    fenced = '```\n{"a": 1}\n```'
    assert strip_markdown_fence(fenced) == '{"a": 1}'


def test_strip_markdown_fence_leaves_unfenced_text_unchanged():
    assert strip_markdown_fence('{"a": 1}') == '{"a": 1}'


def test_strip_markdown_fence_handles_a_fence_with_no_closing_line():
    # Malformed/truncated, but shouldn't crash - falls back to dropping
    # just the opening line.
    assert strip_markdown_fence('```json\n{"a": 1}').strip() == '{"a": 1}'
