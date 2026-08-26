"""Generates the Journal tab's single Mouratoglou-style coaching response.

Distinct from AICoachPipeline (pipeline.py): that's 3 parallel specialists
producing structured strategy/drill/fitness items; this is one voice
responding directly to the player's own freshly-written journal entry, in
prose. Same filesystem-cache pattern as AICoachPipeline.generate() - one
JSON file per match, gitignored, skipped on a re-request unless
force=True. A cached response is also regenerated if the journal entry it
was generated from has since changed, since serving stale feedback for an
edited entry would be actively misleading, not just an unnecessary re-spend.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic

from stats.models import MatchStats

from .client import AnthropicClientLike, extract_text, strip_markdown_fence
from .config import AICoachConfig
from .context import build_context
from .prompts import coach_prompt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JournalFeedback:
    """One match's cached Journal-tab coaching response.

    Attributes:
        match_id: The match this feedback is for.
        generated_at: ISO-format timestamp of generation.
        model: The Claude model id used to generate this response.
        journal_text: The journal entry this feedback was generated from -
            cached alongside the response so a later edit to the journal
            entry can be detected as stale (see generate_journal_feedback).
        feedback: The coach's response.
    """

    match_id: int
    generated_at: str
    model: str
    journal_text: str
    feedback: str

    def to_dict(self) -> dict:
        """Converts this feedback to a plain, JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "JournalFeedback":
        """Rebuilds a JournalFeedback from a dict produced by to_dict."""
        return cls(**data)


class JournalFeedbackError(RuntimeError):
    """Raised when generating Journal-tab coaching feedback fails.

    Covers both the API call itself failing (auth, rate limit, network —
    an `anthropic.APIError`) and a response that parses but doesn't match
    the expected shape — mirrors ai/client.py's SpecialistError in spirit,
    kept as its own type since this call site isn't one of the three
    strategy/drill/fitness specialists.

    Attributes:
        raw_response: The raw text that failed to parse, or "" if the API
            call itself failed before any response was received.
    """

    def __init__(self, raw_response: str, cause: Exception) -> None:
        super().__init__(f"Journal feedback generation failed: {cause}")
        self.raw_response = raw_response


def _report_path(config: AICoachConfig, match_id: int) -> Path:
    return config.journal_feedback_dir / f"{match_id}.json"


def generate_journal_feedback(
    stats: MatchStats,
    client: AnthropicClientLike,
    journal_text: str,
    *,
    config: AICoachConfig | None = None,
    force: bool = False,
) -> JournalFeedback:
    """Generates (or loads a cached) Journal-tab coaching response.

    Args:
        stats: The match's derived-stats bundle (stats.queries.match_stats).
        client: An anthropic.Anthropic-shaped client (injected so tests
            never hit the real API or spend real money).
        journal_text: The player's freshly-written journal entry for this
            match (their current pros/cons/notes).
        config: Model/path settings. Defaults to AICoachConfig() if not
            given.
        force: If True, regenerates even if a cached response with a
            matching journal_text exists, overwriting it.

    Returns:
        The match's JournalFeedback, newly generated or loaded from cache.

    Raises:
        JournalFeedbackError: If the API call itself fails, or its
            response isn't valid JSON matching the expected shape.
    """
    config = config or AICoachConfig()
    path = _report_path(config, stats.match_id)
    if path.exists() and not force:
        cached = JournalFeedback.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if cached.journal_text == journal_text:
            logger.info(
                "Journal feedback for match_id=%d already exists; skipping", stats.match_id
            )
            return cached

    context = build_context(stats)
    system_prompt = coach_prompt(context, journal_text)
    try:
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": "Give your coaching feedback now."}],
        )
    except anthropic.APIError as exc:
        raise JournalFeedbackError("", exc) from exc

    raw_text = ""
    try:
        raw_text = extract_text(response)
        data = json.loads(strip_markdown_fence(raw_text))
        feedback_text = data["feedback"]
    except (ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise JournalFeedbackError(raw_text, exc) from exc

    feedback = JournalFeedback(
        match_id=stats.match_id,
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        model=config.model,
        journal_text=journal_text,
        feedback=feedback_text,
    )
    config.journal_feedback_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(feedback.to_dict(), indent=2), encoding="utf-8")
    logger.info("Generated and saved journal feedback for match_id=%d", stats.match_id)
    return feedback
