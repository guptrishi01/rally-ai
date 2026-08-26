"""System-prompt builders for each AI coaching specialist.

Each builder produces a self-contained system prompt: the match context as
JSON, the item-count bounds, and the exact JSON schema the specialist must
respond with (no prose, no markdown fences) — so client.py can parse the
response directly.
"""

from __future__ import annotations

import json

from .config import AICoachConfig
from .context import CoachContext

_ITEM_SCHEMA = (
    '  "observation": string — the specific pattern in the data\n'
    '  "recommendation": string — one concrete, actionable adjustment\n'
    '  "supporting_stat": {"stat": string, "value": number, '
    '"comparison_label": string|null, "comparison_value": number|null}\n'
    '  "priority": "high" | "medium" | "low"'
)

_DRILL_EXTRA_SCHEMA = (
    '\n  "drill_name": string — short name for the drill\n'
    '  "frequency": string — e.g. "15 min, 3x/week"'
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

_COACH_PERSONA = (
    "You are a world-class tennis coach in the mold of Patrick Mouratoglou: "
    "direct, encouraging, and tactically sharp. Speak to the player in the "
    "second person, like a coach writing them a personal note right after "
    "watching the match — confident and specific, never generic "
    "motivational filler."
)

_COACH_RESPONSE_INSTRUCTIONS = (
    "Respond with ONLY a JSON object — no prose, no markdown code fences. "
    'The object has exactly this key:\n'
    '  "feedback": string — your coaching response, 2-4 short paragraphs'
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
        f"{role_description}\n\n"
        f"Match context (JSON):\n{context_json}\n\n"
        "Ground every item in the specific stats above — never give generic "
        "advice that would apply to any player, and never restate a raw "
        "number without interpreting what it means.\n\n"
        f"Produce {min_items}-{max_items} items.\n\n"
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
        "You are a tennis strategy analyst. Identify tactical patterns in "
        "this match's data that are actionable for the player's next match.",
        context,
        config.strategy_item_bounds,
        _ITEM_SCHEMA,
    )


def drill_prompt(context: CoachContext, config: AICoachConfig) -> str:
    """Builds the system prompt for the drills specialist.

    Args:
        context: The fixed match context.
        config: Supplies drill_item_bounds.

    Returns:
        The drills specialist's system prompt.
    """
    return _build_prompt(
        "You are a tennis practice-drill designer. Identify technical or "
        "tactical weaknesses in this match's data and design one targeted "
        "practice drill per weakness.",
        context,
        config.drill_item_bounds,
        _ITEM_SCHEMA + _DRILL_EXTRA_SCHEMA,
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
        "You are a tennis physical-conditioning coach. Using energy/mental "
        "ratings and the player's own notes, identify conditioning "
        "recommendations grounded in this specific match, not generic "
        "fitness advice.",
        context,
        config.fitness_item_bounds,
        _ITEM_SCHEMA + _FITNESS_EXTRA_SCHEMA,
    )


def coach_prompt(context: CoachContext, journal_text: str) -> str:
    """Builds the system prompt for the Journal tab's single coaching voice.

    Distinct from strategy_prompt/drill_prompt/fitness_prompt above: those
    three each produce a structured list of items in their own narrow
    lane, run concurrently (see ai/generate.py). This is one persona
    responding in prose directly to the player's own freshly-written
    journal entry for one match, grounded in the same match stats.

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
        f"{_COACH_PERSONA}\n\n"
        f"Match context (JSON):\n{context_json}\n\n"
        f'The player\'s own journal entry for this match: "{journal_text}"\n\n'
        "Ground your feedback in the specific stats above and what the "
        "player themselves wrote — never give generic advice that would "
        "apply to any player, and never just restate a number without "
        "interpreting what it means for how they should train or play "
        "next.\n\n"
        f"{_COACH_RESPONSE_INSTRUCTIONS}"
    )
