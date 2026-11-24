"""Runs the 3 tip specialists concurrently.

This is the one place CLAUDE.md's code conventions name for
ThreadPoolExecutor: three independent, stateless API calls against the same
fixed context, not a blanket concurrency convention.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from . import prompts
from .client import AnthropicClientLike, SpecialistError, call_specialist
from .config import AICoachConfig
from .context import CoachContext
from .records import CoachingItem, FitnessItem, PracticeIdeaItem

logger = logging.getLogger(__name__)


def generate_all(
    client: AnthropicClientLike, config: AICoachConfig, context: CoachContext
) -> tuple[list[CoachingItem], list[PracticeIdeaItem], list[FitnessItem]]:
    """Runs the strategy / practice-ideas / fitness specialists concurrently.

    Args:
        client: An anthropic.Anthropic-shaped client.
        config: Model/token/item-count settings.
        context: The fixed context all three specialists see.

    Returns:
        (strategy tips, practice ideas, fitness tips). A specialist whose
        response fails to parse contributes an empty list for its category
        rather than failing the whole report — the failure is logged, not
        silently swallowed.
    """
    prompt_builders = {
        "strategy": prompts.strategy_prompt,
        "practice_idea": prompts.practice_idea_prompt,
        "fitness": prompts.fitness_prompt,
    }

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            category: executor.submit(
                call_specialist, client, config, category, build_prompt(context, config)
            )
            for category, build_prompt in prompt_builders.items()
        }
        results: dict[str, list] = {}
        for category, future in futures.items():
            try:
                results[category] = future.result()
            except SpecialistError:
                logger.exception(
                    "Specialist %s failed to produce parseable output; "
                    "report will have an empty %s section",
                    category,
                    category,
                )
                results[category] = []

    return results["strategy"], results["practice_idea"], results["fitness"]
