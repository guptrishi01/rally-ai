"""Configuration for the AI coaching engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = "claude-sonnet-5"


@dataclass(frozen=True)
class AICoachConfig:
    """Settings for generating one match's coaching report.

    Attributes:
        model: Claude model id used for all three specialists.
        max_tokens: Max output tokens per specialist call.
        strategy_item_bounds: (min, max) strategy items to request.
        drill_item_bounds: (min, max) drill items to request.
        fitness_item_bounds: (min, max) fitness items to request — smaller
            than the others since there's less data density to draw on
            (energy/mental rating + cons, vs. the full point-outcome stats).
        reports_dir: Directory generated CoachingReport JSON files are
            written to (gitignored — personal data, not code).
        pending_dir: Directory swingvision_import stages pending-review
            match JSON into — mirrors ImportConfig.pending_dir's default.
            Used to best-effort look up a match's shot_pattern_summary by
            date/opponent; harmless if nothing's found there.
        journal_feedback_dir: Directory generated JournalFeedback JSON
            files (see journal.py) are written to — gitignored, personal
            AI output, same convention as reports_dir.
    """

    model: str = DEFAULT_MODEL
    max_tokens: int = 1024
    strategy_item_bounds: tuple[int, int] = (2, 4)
    drill_item_bounds: tuple[int, int] = (2, 4)
    fitness_item_bounds: tuple[int, int] = (1, 3)
    reports_dir: Path = _REPO_ROOT / "ai" / "reports"
    pending_dir: Path = _REPO_ROOT / "src" / "swingvision_import" / "pending"
    journal_feedback_dir: Path = _REPO_ROOT / "ai" / "journal_feedback"
