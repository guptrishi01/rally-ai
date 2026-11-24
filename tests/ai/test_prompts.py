from __future__ import annotations

from ai.config import AICoachConfig
from ai.context import build_context
from ai.prompts import (
    fitness_prompt,
    journal_feedback_prompt,
    practice_idea_prompt,
    strategy_prompt,
)


def test_strategy_prompt_embeds_the_actual_match_stats(sample_match_stats):
    context = build_context(sample_match_stats)
    prompt = strategy_prompt(context, AICoachConfig())

    assert '"FS%": 60.0' in prompt
    assert '"opponent": "Alex"' in prompt


def test_strategy_prompt_states_the_configured_item_bounds(sample_match_stats):
    context = build_context(sample_match_stats)
    config = AICoachConfig(strategy_item_bounds=(2, 5))

    prompt = strategy_prompt(context, config)

    # Items are framed as "tips" throughout, so the count line matches.
    assert "Produce 2-5 tips" in prompt


def test_practice_idea_prompt_schema_includes_idea_specific_fields(sample_match_stats):
    """Formerly asserted drill_name/frequency. Renamed so the schema reads
    as "an idea to try", not "a drill to run" - frequency is deliberately
    removed, since prescribing reps/weekly cadence is what tipped the
    output into coach-replacement territory."""
    context = build_context(sample_match_stats)
    prompt = practice_idea_prompt(context, AICoachConfig())

    assert "practice_name" in prompt
    assert "how_to_try" in prompt
    # The old "frequency" field is gone. Must stay gone.
    assert '"frequency"' not in prompt
    # And the prompt itself must tell the model not to re-invent it.
    assert "Do NOT prescribe a frequency or duration" in prompt


def test_fitness_prompt_schema_includes_focus_area(sample_match_stats):
    context = build_context(sample_match_stats)
    prompt = fitness_prompt(context, AICoachConfig())

    assert "focus_area" in prompt


def test_prompts_never_ask_for_a_category_key_in_the_response(sample_match_stats):
    context = build_context(sample_match_stats)
    for prompt in (
        strategy_prompt(context, AICoachConfig()),
        practice_idea_prompt(context, AICoachConfig()),
        fitness_prompt(context, AICoachConfig()),
    ):
        assert 'do not include a "category" key' in prompt


def test_every_specialist_prompt_states_the_tips_not_coach_rule(sample_match_stats):
    """The one-line "you are not their coach" rule is the durable anchor
    for the tone pass; it must survive at the top of every specialist's
    prompt or the whole wording risks drifting back."""
    context = build_context(sample_match_stats)
    for prompt in (
        strategy_prompt(context, AICoachConfig()),
        practice_idea_prompt(context, AICoachConfig()),
        fitness_prompt(context, AICoachConfig()),
    ):
        assert "You are NOT their coach" in prompt
        assert "replace working with a real coach" in prompt


def test_specialist_prompts_never_prescribe_a_training_plan(sample_match_stats):
    """Banned-phrase check. These are the patterns the user pushed back
    against; any of them reaching a prompt is a tone drift.

    Careful about the baseline: parts of the current prompt deliberately
    QUOTE the banned patterns inside its own instructions ("do NOT
    prescribe a frequency like '3x/week'"). A literal substring check
    would trip on those correct instructions. Instead: assert the stable
    opt-out clauses are present, AND the prescriptive persona openings
    are absent from the role-description portion of the prompt.
    """
    context = build_context(sample_match_stats)
    for prompt in (
        strategy_prompt(context, AICoachConfig()),
        practice_idea_prompt(context, AICoachConfig()),
        fitness_prompt(context, AICoachConfig()),
    ):
        # Prescriptive persona openings - the old wording said "You are a
        # world-class tennis coach ...", which is the exact framing to
        # keep out. "You are NOT their coach" is allowed (it's the rule).
        assert "world-class tennis coach" not in prompt
        assert "You are a tennis strategy analyst" not in prompt  # old wording
        assert "practice-drill designer" not in prompt  # old wording
        # The rule's phrasing must survive untouched. If this disappears,
        # the prompt has drifted off the tone pass regardless of what
        # replaced it.
        assert "Do not prescribe a training plan" in prompt
        assert "do not tell them how often to practise" in prompt


def test_journal_feedback_prompt_embeds_the_match_stats_and_journal_text(sample_match_stats):
    context = build_context(sample_match_stats)

    prompt = journal_feedback_prompt(context, "Served big today, but faded in the third set.")

    assert '"FS%": 60.0' in prompt
    assert "Served big today, but faded in the third set." in prompt
    assert '"feedback": string' in prompt


def test_journal_feedback_prompt_frames_the_voice_as_reflection_not_coaching(
    sample_match_stats,
):
    """Voice regression guard. The journal response used to open with a
    Mouratoglou persona ("direct, encouraging, tactically sharp" coach);
    the new framing is reflective, grounded in the stats, and explicitly
    not a training plan."""
    context = build_context(sample_match_stats)

    prompt = journal_feedback_prompt(context, "Some notes.")

    assert "Mouratoglou" not in prompt
    assert "reflective" in prompt or "reflection" in prompt
    assert "You are NOT their coach" in prompt
    assert "next-week training prescriptions" in prompt


def test_journal_feedback_prompt_never_asks_for_a_structured_item_list(sample_match_stats):
    context = build_context(sample_match_stats)

    prompt = journal_feedback_prompt(context, "Some notes.")

    assert "JSON array" not in prompt
    assert '"observation"' not in prompt
