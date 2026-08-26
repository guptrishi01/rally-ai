from __future__ import annotations

from ai.config import AICoachConfig
from ai.context import build_context
from ai.prompts import coach_prompt, drill_prompt, fitness_prompt, strategy_prompt


def test_strategy_prompt_embeds_the_actual_match_stats(sample_match_stats):
    context = build_context(sample_match_stats)
    prompt = strategy_prompt(context, AICoachConfig())

    assert '"FS%": 60.0' in prompt
    assert '"opponent": "Alex"' in prompt


def test_strategy_prompt_states_the_configured_item_bounds(sample_match_stats):
    context = build_context(sample_match_stats)
    config = AICoachConfig(strategy_item_bounds=(2, 5))

    prompt = strategy_prompt(context, config)

    assert "Produce 2-5 items" in prompt


def test_drill_prompt_schema_includes_drill_specific_fields(sample_match_stats):
    context = build_context(sample_match_stats)
    prompt = drill_prompt(context, AICoachConfig())

    assert "drill_name" in prompt
    assert "frequency" in prompt


def test_fitness_prompt_schema_includes_focus_area(sample_match_stats):
    context = build_context(sample_match_stats)
    prompt = fitness_prompt(context, AICoachConfig())

    assert "focus_area" in prompt


def test_prompts_never_ask_for_a_category_key_in_the_response(sample_match_stats):
    context = build_context(sample_match_stats)
    for prompt in (
        strategy_prompt(context, AICoachConfig()),
        drill_prompt(context, AICoachConfig()),
        fitness_prompt(context, AICoachConfig()),
    ):
        assert 'do not include a "category" key' in prompt


def test_coach_prompt_embeds_the_match_stats_and_journal_text(sample_match_stats):
    context = build_context(sample_match_stats)

    prompt = coach_prompt(context, "Served big today, but faded in the third set.")

    assert '"FS%": 60.0' in prompt
    assert "Served big today, but faded in the third set." in prompt
    assert '"feedback": string' in prompt


def test_coach_prompt_never_asks_for_a_structured_item_list(sample_match_stats):
    context = build_context(sample_match_stats)

    prompt = coach_prompt(context, "Some notes.")

    assert "JSON array" not in prompt
    assert '"observation"' not in prompt
