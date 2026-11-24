"""Orchestrates the AI coach: context -> generate (concurrent) -> save.

Incremental state tracking (per CLAUDE.md's code conventions) happens via
the filesystem: a match_id with an existing report file is skipped unless
force=True, the same pattern swingvision_import uses for its pending JSON.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from pathlib import Path

from stats.models import MatchStats
from stats.queries import match_stats
from swingvision_import.review import find_pending_path, load_pending

from .client import AnthropicClientLike
from .config import AICoachConfig
from .context import build_context
from .generate import generate_all
from .records import CoachingReport

logger = logging.getLogger(__name__)


def _lookup_shot_pattern_summary(
    stats: MatchStats, pending_dir: Path
) -> dict[str, float] | None:
    """Best-effort lookup of a match's shot-pattern summary from its
    original staged JSON, by the date/opponent already on the match row.

    This is additive-only: a missing pending file (deleted after finalize,
    or never existed for a direct-parse match with no reconstruction) or
    any read/parse problem just means no enrichment - never fails coaching
    generation over an optional context field.

    Args:
        stats: The match's derived-stats bundle (has date/opponent).
        pending_dir: Directory pending JSON files are staged into.

    Returns:
        The summary dict, or None if unavailable for any reason.
    """
    path = find_pending_path(stats.date, stats.opponent, pending_dir)
    if path is None:
        return None
    try:
        return load_pending(path).shot_pattern_summary
    except (OSError, ValueError, TypeError) as exc:
        logger.info("Couldn't load shot_pattern_summary from %s: %s", path, exc)
        return None


class AICoachPipeline:
    """Generates and persists a match's AI coaching report.

    Attributes:
        config: Model/token/item-count/output-path settings.
    """

    def __init__(self, config: AICoachConfig | None = None) -> None:
        """Initializes the pipeline.

        Args:
            config: Settings to use. Defaults to AICoachConfig() if not
                given.
        """
        self.config = config or AICoachConfig()

    def _report_path(self, match_id: int) -> Path:
        return self.config.reports_dir / f"{match_id}.json"

    def generate(
        self,
        connection: sqlite3.Connection,
        client: AnthropicClientLike,
        match_id: int,
        *,
        force: bool = False,
    ) -> CoachingReport:
        """Generates (or loads a cached) coaching report for one match.

        Args:
            connection: An open SQLite connection with the match already
                loaded (via swingvision_import.load.finalize_and_load).
            client: An anthropic.Anthropic-shaped client.
            match_id: The match to generate a report for.
            force: If True, regenerates even if a report already exists,
                overwriting it. Otherwise an existing report is loaded from
                disk and returned without calling the API again.

        Returns:
            The match's CoachingReport, newly generated or loaded from disk.
        """
        report_path = self._report_path(match_id)
        if report_path.exists() and not force:
            logger.info("Coaching report for match_id=%d already exists; skipping", match_id)
            return CoachingReport.from_dict(json.loads(report_path.read_text(encoding="utf-8")))

        stats = match_stats(connection, match_id)
        shot_pattern_summary = _lookup_shot_pattern_summary(stats, self.config.pending_dir)
        context = build_context(stats, shot_pattern_summary)
        strategy, practice_ideas, fitness = generate_all(client, self.config, context)

        report = CoachingReport(
            match_id=match_id,
            generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            model=self.config.model,
            strategy=strategy,
            practice_ideas=practice_ideas,
            fitness=fitness,
        )

        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        logger.info("Generated and saved coaching report for match_id=%d", match_id)
        return report
