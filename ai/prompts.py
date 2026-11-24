"""System-prompt builders for each AI specialist.

Each builder produces a self-contained system prompt: the match context as
JSON, the item-count bounds, and the exact JSON schema the specialist must
respond with (no prose, no markdown fences) — so client.py can parse the
response directly.

**Framing is tips, not coaching.** Per the user's directive (saved in
memory as feedback-tips-not-coach), these prompts never speak as a coach
writing a training plan. Items read as "something to try" or "worth
looking at" with a stat attached. The persona reminder below is enforced
at the top of every prompt and asserted by tests so this does not drift.
"""

from __future__ import annotations

import json

from .config import AICoachConfig
from .context import CoachContext

_ITEM_SCHEMA = (
    '  "observation": string — what stands out in this match\'s data\n'
    '  "recommendation": string — one concrete tip the stat supports, '
    'framed as "something to try" or "worth looking at", never as '
    '"I recommend you do X" or "your training plan"\n'
    '  "supporting_stat": {"stat": string, "value": number, '
    '"comparison_label": string|null, "comparison_value": number|null}\n'
    '  "priority": "high" | "medium" | "low"'
)

_PRACTICE_IDEA_EXTRA_SCHEMA = (
    '\n  "practice_name": string — short name for the idea\n'
    '  "how_to_try": string — one sentence on how a player could try this '
    'at practice. Do NOT prescribe a frequency or duration ("3x/week", '
    '"15 min a day"). Leave timing and reps to the player or their '
    'actual coach.'
)

_FITNESS_EXTRA_SCHEMA = (
    '\n  "focus_area": string — e.g. "endurance", "recovery", '
    '"composure under pressure"'
)

_RESPONSE_INSTRUCTIONS = (
    "Respond with ONLY a JSON array — no prose, no markdown code fences. "
    "Each array element is an object with exactly these keys "
    '(do not include a "category" key, it is fixed by this prompt):\n{schema}'
)

# The one-line assertion that survives every prompt. Short enough that it
# can be asserted on by substring, strict enough that the tone pass has a
# single durable anchor the tests can key off of.
_TIPS_NOT_COACH_RULE = (
    "You are giving the player tips and advice from their own match "
    "numbers. You are NOT their coach. Do not prescribe a training plan, "
    "do not tell them how often to practise, and do not suggest you can "
    "replace working with a real coach."
)

_JOURNAL_PERSONA = (
    "You read one match's numbers alongside the player's own written "
    "reflection on it and respond with a short, grounded read on what "
    "stands out. Think of yourself as a thoughtful friend who knows "
    "tennis and has the stats in front of them - reflective, specific, "
    "and honest about what the numbers do and do not say. Speak in the "
    "second person, naturally."
)

_JOURNAL_RESPONSE_INSTRUCTIONS = (
    "Respond with ONLY a JSON object — no prose, no markdown code fences. "
    'The object has exactly this key:\n'
    '  "feedback": string — your reflection, 2-4 short paragraphs. Match '
    'what the player actually wrote; if they mentioned something that '
    'the numbers back up, say so; if the numbers tell a different story, '
    'say so gently. Never give next-week training prescriptions.'
)


def _build_prompt(
    role_description: str,
    context: CoachContext,
    item_bounds: tuple[int, int],
    schema: str,
) -> str:
    """Assembles a specialist's full system prompt.

    Args:
        role_description: The specialist's persona and task, in prose.
        context: The fixed match context to embed as JSON.
        item_bounds: (min, max) items to request.
        schema: The JSON schema block describing each response item.

    Returns:
        The complete system prompt string.
    """
    min_items, max_items = item_bounds
    context_json = json.dumps(context.to_dict(), indent=2)
    return (
        f"{_TIPS_NOT_COACH_RULE}\n\n"
        f"{role_description}\n\n"
        f"Match context (JSON):\n{context_json}\n\n"
        "Ground every tip in the specific stats above — never give generic "
        "advice that would apply to any player, and never restate a raw "
        "number without saying what stands out about it.\n\n"
        f"Produce {min_items}-{max_items} tips.\n\n"
        f"{_RESPONSE_INSTRUCTIONS.format(schema=schema)}"
    )


def strategy_prompt(context: CoachContext, config: AICoachConfig) -> str:
    """Builds the system prompt for the strategy specialist.

    Args:
        context: The fixed match context.
        config: Supplies strategy_item_bounds.

    Returns:
        The strategy specialist's system prompt.
    """
    return _build_prompt(
        "You surface tactical patterns worth looking at in this match's "
        "data - things a player might try differently next time, not a "
        "set of orders. Phrase each one as a tip ('worth looking at', "
        "'something to try'), never as a prescribed training plan.",
        context,
        config.strategy_item_bounds,
        _ITEM_SCHEMA,
    )


def practice_idea_prompt(context: CoachContext, config: AICoachConfig) -> str:
    """Builds the system prompt for the practice-idea specialist.

    Formerly the "drills" specialist; renamed and reframed so the output
    reads as ideas a player could try - not prescribed drills with
    "3x/week" frequencies. See feedback-tips-not-coach in memory.

    Args:
        context: The fixed match context.
        config: Supplies practice_idea_item_bounds.

    Returns:
        The practice-idea specialist's system prompt.
    """
    return _build_prompt(
        "You suggest practice ideas a player could try if a pattern in "
        "this match stands out. These are IDEAS, not drills: do not "
        "prescribe reps, duration or weekly frequency - leave that to "
        "the player and their actual coach. Each idea is one thing they "
        "might try at practice, framed as a suggestion.",
        context,
        config.practice_idea_item_bounds,
        _ITEM_SCHEMA + _PRACTICE_IDEA_EXTRA_SCHEMA,
    )


def fitness_prompt(context: CoachContext, config: AICoachConfig) -> str:
    """Builds the system prompt for the fitness specialist.

    Args:
        context: The fixed match context.
        config: Supplies fitness_item_bounds.

    Returns:
        The fitness specialist's system prompt.
    """
    return _build_prompt(
        "You surface physical and mental patterns worth noticing from "
        "this match - the energy/mental ratings and the player's own "
        "notes. These are tips, not a conditioning program. Describe "
        "what stands out and what might be worth paying attention to, "
        "never a prescribed training regimen.",
        context,
        config.fitness_item_bounds,
        _ITEM_SCHEMA + _FITNESS_EXTRA_SCHEMA,
    )


def journal_feedback_prompt(context: CoachContext, journal_text: str) -> str:
    """Builds the system prompt for the Journal tab's reflection voice.

    Distinct from the three specialists above: those produce structured
    tips concurrently. This is one voice responding in prose to the
    player's own journal entry for one match, grounded in the same match
    stats. Reflective rather than prescriptive - see
    feedback-tips-not-coach in memory for the rule.

    Args:
        context: The fixed match context (the same CoachContext the three
            specialists see).
        journal_text: The player's own journal entry for this match (their
            pros/cons/notes, submitted from the Journal tab).

    Returns:
        The complete system prompt.
    """
    context_json = json.dumps(context.to_dict(), indent=2)
    return (
        f"{_TIPS_NOT_COACH_RULE}\n\n"
        f"{_JOURNAL_PERSONA}\n\n"
        f"Match context (JSON):\n{context_json}\n\n"
        f'The player\'s own journal entry for this match: "{journal_text}"\n\n'
        "Ground your reflection in the specific stats above and what the "
        "player themselves wrote. Never give generic advice that would "
        "apply to any player, and never just restate a number without "
        "saying what stands out about it. Keep it reflective: this is a "
        "read on what happened in this one match, not a plan for next "
        "week.\n\n"
        f"{_JOURNAL_RESPONSE_INSTRUCTIONS}"
    )


# Backwards-compatibility aliases. coach_prompt and drill_prompt still exist
# as function names for anything that imported them directly, but both
# delegate to the renamed implementations above. Delete at next major bump.
coach_prompt = journal_feedback_prompt
drill_prompt = practice_idea_prompt
