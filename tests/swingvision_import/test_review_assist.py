from __future__ import annotations

import json

import anthropic
import httpx
import pytest

from swingvision_import.raw import RawShotRow
from swingvision_import.records import PointRecord
from swingvision_import.review_assist import (
    SuggestionConfig,
    SuggestionError,
    suggest_point_resolution,
)
from tests.ai.conftest import FakeMessage, FakeTextBlock, FakeThinkingBlock


class _FakeMessages:
    """A purpose-built fake for review_assist's single-shot-suggestion
    prompt/response shape - reuses tests/ai/conftest.py's FakeMessage/
    FakeTextBlock response primitives rather than redefining them, but
    review_assist's response schema (point_end_type/reasoning/confidence)
    doesn't match ai/'s specialist item schema, so FakeAnthropicClient
    itself isn't a fit here."""

    def __init__(self, response_text: str | None = None, raise_error: Exception | None = None):
        self.response_text = response_text
        self.raise_error = raise_error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_error:
            raise self.raise_error
        return FakeMessage(content=[FakeTextBlock(text=self.response_text)])


class _FakeClient:
    def __init__(self, response_text: str | None = None, raise_error: Exception | None = None):
        self.messages = _FakeMessages(response_text, raise_error)


class _ThinkingFirstMessages:
    """Stands in for a real Claude Sonnet 5 response: a ThinkingBlock ahead
    of the text block, confirmed against the live API - content[0] is not
    reliably the text block."""

    def __init__(self, content: list):
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeMessage(content=self.content)


class _ThinkingFirstClient:
    def __init__(self, content: list):
        self.messages = _ThinkingFirstMessages(content)


def _point() -> PointRecord:
    return PointRecord(
        game_number=1,
        point_number=1,
        is_serving=True,
        point_won=False,
        point_end_type="unforced_error",
        needs_review=True,
    )


def _shots() -> list[RawShotRow]:
    return [
        RawShotRow(1, 1, "Rishi Gupta", "first_serve", "Serve", "In"),
        RawShotRow(1, 2, "Opponent", "first_return", "Forehand", "In"),
        RawShotRow(1, 3, "Rishi Gupta", "serve_plus_one", "Forehand", "Net"),
    ]


def test_suggest_point_resolution_parses_a_valid_response():
    valid = json.dumps(
        {"point_end_type": "forced_error", "reasoning": "Wide angle return.", "confidence": "high"}
    )
    client = _FakeClient(response_text=valid)

    suggestion = suggest_point_resolution(client, SuggestionConfig(), _point(), _shots())

    assert suggestion.point_end_type == "forced_error"
    assert suggestion.confidence == "high"
    assert "angle" in suggestion.reasoning


def test_suggest_point_resolution_includes_shot_context_in_the_prompt():
    valid = json.dumps({"point_end_type": "unforced_error", "reasoning": "x", "confidence": "low"})
    client = _FakeClient(response_text=valid)

    suggest_point_resolution(client, SuggestionConfig(), _point(), _shots())

    system = client.messages.calls[0]["system"]
    assert "serve_plus_one" in system
    assert '"result": "Net"' in system


def test_suggest_point_resolution_raises_on_api_failure():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.APIConnectionError(message="boom", request=request)
    client = _FakeClient(raise_error=error)

    with pytest.raises(SuggestionError) as exc_info:
        suggest_point_resolution(
            client, SuggestionConfig(max_attempts=1), _point(), _shots()
        )

    assert exc_info.value.raw_response == ""
    assert "boom" in str(exc_info.value)
    assert len(client.messages.calls) == 1  # max_attempts=1 means no retry


def test_suggest_point_resolution_raises_on_invalid_json():
    client = _FakeClient(response_text="not json")

    with pytest.raises(SuggestionError) as exc_info:
        suggest_point_resolution(client, SuggestionConfig(), _point(), _shots())

    assert exc_info.value.raw_response == "not json"


def test_suggest_point_resolution_raises_on_an_invalid_point_end_type():
    invalid = json.dumps({"point_end_type": "let", "reasoning": "x", "confidence": "low"})
    client = _FakeClient(response_text=invalid)

    with pytest.raises(SuggestionError):
        suggest_point_resolution(client, SuggestionConfig(), _point(), _shots())


def test_suggest_point_resolution_raises_on_missing_required_key():
    incomplete = json.dumps({"point_end_type": "winner"})
    client = _FakeClient(response_text=incomplete)

    with pytest.raises(SuggestionError):
        suggest_point_resolution(client, SuggestionConfig(), _point(), _shots())


def test_suggest_point_resolution_passes_model_and_token_config_through():
    valid = json.dumps({"point_end_type": "winner", "reasoning": "x", "confidence": "high"})
    client = _FakeClient(response_text=valid)
    config = SuggestionConfig(model="claude-sonnet-5", max_tokens=256)

    suggest_point_resolution(client, config, _point(), _shots())

    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-5"
    assert call["max_tokens"] == 256
    assert "temperature" not in call


def test_suggest_point_resolution_skips_a_leading_thinking_block_to_find_the_text():
    valid = json.dumps({"point_end_type": "winner", "reasoning": "x", "confidence": "high"})
    client = _ThinkingFirstClient(
        content=[FakeThinkingBlock(thinking="reasoning..."), FakeTextBlock(text=valid)]
    )

    suggestion = suggest_point_resolution(client, SuggestionConfig(), _point(), _shots())

    assert suggestion.point_end_type == "winner"


def test_suggest_point_resolution_raises_when_no_text_block_exists():
    client = _ThinkingFirstClient(content=[FakeThinkingBlock(thinking="reasoning...")])

    with pytest.raises(SuggestionError) as exc_info:
        suggest_point_resolution(client, SuggestionConfig(), _point(), _shots())

    assert exc_info.value.raw_response == ""


def test_suggest_point_resolution_strips_a_markdown_fence_before_parsing():
    # Confirmed against the live API, not hypothetical: a real response
    # came back wrapped in ```json ... ``` despite the prompt explicitly
    # saying not to.
    valid = json.dumps({"point_end_type": "forced_error", "reasoning": "x", "confidence": "high"})
    client = _FakeClient(response_text=f"```json\n{valid}\n```")

    suggestion = suggest_point_resolution(client, SuggestionConfig(), _point(), _shots())

    assert suggestion.point_end_type == "forced_error"


class _FlakyMessages:
    """Fails the first `failures` calls with `error`, then succeeds."""

    def __init__(self, error: Exception, failures: int, response_text: str):
        self.error = error
        self.failures = failures
        self.response_text = response_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures:
            raise self.error
        return FakeMessage(content=[FakeTextBlock(text=self.response_text)])


class _FlakyClient:
    def __init__(self, error: Exception, failures: int, response_text: str):
        self.messages = _FlakyMessages(error, failures, response_text)


def _valid_response() -> str:
    return json.dumps(
        {"point_end_type": "winner", "reasoning": "clean", "confidence": "high"}
    )


def test_a_transient_failure_is_retried_and_can_still_succeed():
    """Concurrency is what provokes 429s, so a rate-limited point has to be
    able to recover - otherwise going parallel would quietly cost coverage."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.RateLimitError(
        message="slow down",
        response=httpx.Response(429, request=request),
        body=None,
    )
    client = _FlakyClient(error, failures=2, response_text=_valid_response())

    suggestion = suggest_point_resolution(
        client,
        SuggestionConfig(max_attempts=3, retry_base_delay=0.0),
        _point(),
        _shots(),
    )

    assert suggestion.point_end_type == "winner"
    assert len(client.messages.calls) == 3


def test_a_non_transient_failure_is_not_retried():
    """A 400 fails the same way every time - retrying it only spends money."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.BadRequestError(
        message="bad", response=httpx.Response(400, request=request), body=None
    )
    client = _FlakyClient(error, failures=99, response_text=_valid_response())

    with pytest.raises(SuggestionError):
        suggest_point_resolution(
            client,
            SuggestionConfig(max_attempts=5, retry_base_delay=0.0),
            _point(),
            _shots(),
        )

    assert len(client.messages.calls) == 1


def test_a_malformed_response_is_not_retried():
    """Nothing about the request was wrong, so asking again just pays twice
    for the same deterministic mistake."""
    client = _FlakyClient(RuntimeError("unused"), failures=0, response_text="not json")

    with pytest.raises(SuggestionError):
        suggest_point_resolution(
            client,
            SuggestionConfig(max_attempts=5, retry_base_delay=0.0),
            _point(),
            _shots(),
        )

    assert len(client.messages.calls) == 1


def test_the_reviewers_own_note_reaches_the_prompt_as_context_not_instruction():
    client = _FakeClient(response_text=_valid_response())

    suggest_point_resolution(
        client,
        SuggestionConfig(),
        _point(),
        _shots(),
        "part 2 restarts mid-game",
    )

    system = client.messages.calls[0]["system"]
    assert "part 2 restarts mid-game" in system
    assert "not as an answer" in system


def test_no_note_means_no_note_block_in_the_prompt():
    client = _FakeClient(response_text=_valid_response())

    suggest_point_resolution(client, SuggestionConfig(), _point(), _shots(), "   ")

    assert "background on how the data was" not in client.messages.calls[0]["system"]
