from __future__ import annotations

import logging

from ai.config import AICoachConfig
from ai.context import build_context
from ai.generate import generate_all
from tests.ai.conftest import FakeAnthropicClient


def test_generate_all_invokes_all_three_specialists(sample_match_stats):
    client = FakeAnthropicClient()
    context = build_context(sample_match_stats)

    generate_all(client, AICoachConfig(), context)

    assert len(client.messages.calls) == 3
    systems = {call["system"] for call in client.messages.calls}
    assert any("tactical patterns" in s for s in systems)
    assert any("practice ideas" in s for s in systems)
    assert any("physical and mental patterns" in s for s in systems)


def test_generate_all_returns_items_in_the_right_category_slot(sample_match_stats):
    client = FakeAnthropicClient()
    context = build_context(sample_match_stats)

    strategy, practice_ideas, fitness = generate_all(client, AICoachConfig(), context)

    assert strategy[0].category == "strategy"
    assert practice_ideas[0].category == "practice_idea"
    assert fitness[0].category == "fitness"


def test_generate_all_does_not_fail_the_whole_report_when_one_specialist_fails(
    sample_match_stats, caplog
):
    client = FakeAnthropicClient(raise_for={"practice_idea"})
    context = build_context(sample_match_stats)

    with caplog.at_level(logging.ERROR):
        strategy, practice_ideas, fitness = generate_all(client, AICoachConfig(), context)

    assert len(strategy) == 1
    assert practice_ideas == []
    assert len(fitness) == 1
    assert any("practice_idea" in record.message for record in caplog.records)


def test_generate_all_survives_a_real_api_failure_on_one_specialist(sample_match_stats, caplog):
    # Distinct from the malformed-JSON case above: here the API call itself
    # raises (auth/rate-limit/network), which used to propagate uncaught
    # and kill generate_all() entirely instead of degrading gracefully.
    client = FakeAnthropicClient(api_error_for={"fitness"})
    context = build_context(sample_match_stats)

    with caplog.at_level(logging.ERROR):
        strategy, practice_ideas, fitness = generate_all(client, AICoachConfig(), context)

    assert len(strategy) == 1
    assert len(practice_ideas) == 1
    assert fitness == []
