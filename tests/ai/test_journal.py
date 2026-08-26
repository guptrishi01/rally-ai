from __future__ import annotations

import json

import anthropic
import httpx
import pytest

from ai.config import AICoachConfig
from ai.journal import JournalFeedbackError, generate_journal_feedback
from tests.ai.conftest import FakeMessage, FakeTextBlock


def _fake_api_error() -> anthropic.APIError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(message="simulated connection failure", request=request)


class _FakeJournalMessages:
    def __init__(
        self,
        feedback_text: str = "Great serving today.",
        raise_bad_json: bool = False,
        raise_api_error: bool = False,
    ):
        self.calls: list[dict] = []
        self._feedback_text = feedback_text
        self._raise_bad_json = raise_bad_json
        self._raise_api_error = raise_api_error

    def create(self, **kwargs: object) -> FakeMessage:
        self.calls.append(kwargs)
        if self._raise_api_error:
            raise _fake_api_error()
        if self._raise_bad_json:
            return FakeMessage(content=[FakeTextBlock(text="not valid json")])
        payload = json.dumps({"feedback": self._feedback_text})
        return FakeMessage(content=[FakeTextBlock(text=payload)])


class _FakeJournalClient:
    def __init__(self, **kwargs: object) -> None:
        self.messages = _FakeJournalMessages(**kwargs)


def test_generate_journal_feedback_calls_the_api_and_saves_it(tmp_path, sample_match_stats):
    client = _FakeJournalClient()
    config = AICoachConfig(journal_feedback_dir=tmp_path / "journal_feedback")

    feedback = generate_journal_feedback(
        sample_match_stats, client, "Served big today", config=config
    )

    assert feedback.match_id == sample_match_stats.match_id
    assert feedback.feedback == "Great serving today."
    assert len(client.messages.calls) == 1
    assert (tmp_path / "journal_feedback" / f"{sample_match_stats.match_id}.json").exists()


def test_generate_journal_feedback_skips_the_api_when_cached_and_text_unchanged(
    tmp_path, sample_match_stats
):
    client = _FakeJournalClient()
    config = AICoachConfig(journal_feedback_dir=tmp_path / "journal_feedback")

    generate_journal_feedback(sample_match_stats, client, "Served big today", config=config)
    generate_journal_feedback(sample_match_stats, client, "Served big today", config=config)

    assert len(client.messages.calls) == 1


def test_generate_journal_feedback_regenerates_when_journal_text_changed(
    tmp_path, sample_match_stats
):
    client = _FakeJournalClient()
    config = AICoachConfig(journal_feedback_dir=tmp_path / "journal_feedback")

    first = generate_journal_feedback(
        sample_match_stats, client, "Served big today", config=config
    )
    second = generate_journal_feedback(
        sample_match_stats, client, "Actually struggled on return games", config=config
    )

    assert len(client.messages.calls) == 2
    assert first.journal_text == "Served big today"
    assert second.journal_text == "Actually struggled on return games"


def test_generate_journal_feedback_force_true_regenerates_even_with_matching_text(
    tmp_path, sample_match_stats
):
    client = _FakeJournalClient()
    config = AICoachConfig(journal_feedback_dir=tmp_path / "journal_feedback")

    generate_journal_feedback(sample_match_stats, client, "Served big today", config=config)
    generate_journal_feedback(
        sample_match_stats, client, "Served big today", config=config, force=True
    )

    assert len(client.messages.calls) == 2


def test_generate_journal_feedback_raises_on_unparseable_response(tmp_path, sample_match_stats):
    client = _FakeJournalClient(raise_bad_json=True)
    config = AICoachConfig(journal_feedback_dir=tmp_path / "journal_feedback")

    with pytest.raises(JournalFeedbackError):
        generate_journal_feedback(sample_match_stats, client, "Served big today", config=config)


def test_generate_journal_feedback_raises_when_the_api_call_itself_fails(
    tmp_path, sample_match_stats
):
    client = _FakeJournalClient(raise_api_error=True)
    config = AICoachConfig(journal_feedback_dir=tmp_path / "journal_feedback")

    with pytest.raises(JournalFeedbackError):
        generate_journal_feedback(sample_match_stats, client, "Served big today", config=config)
