"""Builds the deterministic context object handed to all 3 specialists.

No LLM involved here — this is pure data assembly, so the same match always
produces the same context regardless of how many times it's built.

Despite the `CoachContext` name, the context is used for generating **tips
from the match's own numbers**, not for emulating a coach. The name is
kept because every import site in `ai/` uses it; see
feedback-tips-not-coach in memory for the framing rule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from stats.models import MatchStats


@dataclass(frozen=True)
class CoachContext:
    """The fixed, JSON-serializable context every specialist call sees.

    Attributes:
        match_id: The match this context describes.
        opponent: Opponent's name.
        result: "W" or "L".
        stats: A flat dict of stat name -> value, using the same
            abbreviations as docs/stat-definitions.md so prompts can
            reference them directly. Includes shot_pattern_summary's keys
            (avg_rally_length, rally_win_rate_short/long) when one was
            available for this match — see build_context.
        pros: Self-reported "what went well."
        cons: Self-reported "what needs work."
        energy_rating: 1-5, if provided.
        mental_rating: 1-5, if provided.
    """

    match_id: int
    opponent: str
    result: str
    stats: dict[str, float]
    pros: str | None
    cons: str | None
    energy_rating: int | None
    mental_rating: int | None

    def to_dict(self) -> dict:
        """Converts this context to a plain, JSON-serializable dict.

        Returns:
            A dict suitable for embedding directly into a prompt.
        """
        return asdict(self)


def build_context(
    match_stats: MatchStats, shot_pattern_summary: dict[str, float] | None = None
) -> CoachContext:
    """Builds the deterministic coaching context from a match's derived stats.

    Args:
        match_stats: The full derived-stats bundle for one match.
        shot_pattern_summary: Optional rally-length aggregates from
            swingvision_import.reconstruct.build_shot_pattern_summary,
            looked up by ai/pipeline.py from the match's original staged
            JSON if it's still present. Folded into `stats` alongside the
            SQL-derived numbers when given; omitted entirely otherwise —
            this is additive-only, never required.

    Returns:
        The fixed context object every specialist call receives, unchanged.
    """
    stats = {
        "FS%": match_stats.serving.first_serve_pct,
        "SS%": match_stats.serving.second_serve_pct,
        "aces": match_stats.serving.aces,
        "double_faults": match_stats.serving.double_faults,
        "SH%": match_stats.serving.service_hold_pct,
        "BP%": match_stats.receiving.break_point_conversion_pct,
        "RW%": match_stats.receiving.return_win_pct,
        "PW%": match_stats.point_outcomes.points_won_pct,
        "winners": match_stats.point_outcomes.winners,
        "unforced_errors": match_stats.point_outcomes.unforced_errors,
        "forced_errors": match_stats.point_outcomes.forced_errors,
        "W/UE": match_stats.point_outcomes.winner_to_ue_ratio,
        "NS%": match_stats.net.net_success_pct,
        "DC%": match_stats.clutch.deuce_conversion_pct,
    }
    if shot_pattern_summary:
        stats.update(shot_pattern_summary)
    return CoachContext(
        match_id=match_stats.match_id,
        opponent=match_stats.opponent,
        result=match_stats.result,
        stats=stats,
        pros=match_stats.self_assessment.pros,
        cons=match_stats.self_assessment.cons,
        energy_rating=match_stats.self_assessment.energy_rating,
        mental_rating=match_stats.self_assessment.mental_rating,
    )
