"""Renders MatchStats (+ AI coaching output) into self-contained HTML reports.

Jinja2's autoescape is on, so free text from the database (opponent names,
pros/cons) or from the AI coach (observation/recommendation text) is always
HTML-escaped when interpolated with `{{ }}`. The only template values marked
`| safe` are chart fragments built by charts.py, which does its own
attribute-level escaping internally — never raw user/AI text.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ai.records import CoachingReport
from stats.models import MatchStats

from . import charts, palette
from .config import ReportConfig

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

_env = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    autoescape=select_autoescape(["html", "jinja"]),
)


def _point_outcome_chart(stats: MatchStats) -> str:
    outcomes = stats.point_outcomes
    data = [
        charts.BarDatum("Winners", outcomes.winners),
        charts.BarDatum("Unforced errors", outcomes.unforced_errors),
        charts.BarDatum("Forced errors", outcomes.forced_errors),
        charts.BarDatum("Return winners", outcomes.return_winners),
        charts.BarDatum("Return errors", outcomes.return_errors),
        charts.BarDatum("Aces", stats.serving.aces),
        charts.BarDatum("Double faults", stats.serving.double_faults),
    ]
    return charts.bar_chart("point-outcomes", "Point Outcome Breakdown", data)


def _serve_chart(stats: MatchStats) -> str:
    serving = stats.serving
    rows = [
        charts.StackedBarRow(
            "1st serve",
            serving.first_serves_in,
            serving.first_serves_total - serving.first_serves_in,
        ),
        charts.StackedBarRow(
            "2nd serve",
            serving.second_serves_in,
            serving.second_serves_total - serving.second_serves_in,
        ),
    ]
    return charts.stacked_bar_chart(
        "serve-in-out", "First/Second Serve In vs. Out", "In", "Out", rows
    )


def _stat_tiles(stats: MatchStats) -> list[str]:
    return [
        charts.stat_tile("First serve %", f"{stats.serving.first_serve_pct:g}%"),
        charts.stat_tile("Service hold %", f"{stats.serving.service_hold_pct:g}%"),
        charts.stat_tile("Return win %", f"{stats.receiving.return_win_pct:g}%"),
        charts.stat_tile("Break point %", f"{stats.receiving.break_point_conversion_pct:g}%"),
        charts.stat_tile("Winner : UE", f"{stats.point_outcomes.winner_to_ue_ratio:g}"),
        charts.stat_tile("Points won %", f"{stats.point_outcomes.points_won_pct:g}%"),
    ]


def render_match_report(
    match_stats: MatchStats, coaching_report: CoachingReport | None = None
) -> str:
    """Renders one match's full report: stats + AI coaching insights.

    Args:
        match_stats: The match's derived stats bundle.
        coaching_report: The match's AI coaching report, if one has been
            generated. If None, the coaching sections are omitted entirely.

    Returns:
        The complete, self-contained HTML document as a string.
    """
    template = _env.get_template("match_report.html.jinja")

    net_meter_html = None
    if match_stats.net.net_approaches:
        net_meter_html = charts.meter(
            "net-success", "Net Success Rate", match_stats.net.net_success_pct
        )

    context = {
        "match": match_stats,
        "energy_rating": match_stats.self_assessment.energy_rating,
        "mental_rating": match_stats.self_assessment.mental_rating,
        "pros": match_stats.self_assessment.pros,
        "cons": match_stats.self_assessment.cons,
        "palette_css": palette.CSS_VARS,
        "stat_tiles": _stat_tiles(match_stats),
        "point_outcome_chart": _point_outcome_chart(match_stats),
        "serve_chart": _serve_chart(match_stats),
        "net_meter": net_meter_html,
        "strategy": coaching_report.strategy if coaching_report else [],
        "practice_ideas": coaching_report.practice_ideas if coaching_report else [],
        "fitness": coaching_report.fitness if coaching_report else [],
        "improvement_plan": coaching_report.improvement_plan if coaching_report else [],
    }
    return template.render(**context)


def render_match_report_to_file(
    match_stats: MatchStats,
    output_path: Path | None = None,
    coaching_report: CoachingReport | None = None,
    config: ReportConfig | None = None,
) -> Path:
    """Renders and writes a match report to disk.

    Args:
        match_stats: The match's derived stats bundle.
        output_path: Where to write the .html file. Defaults to
            `config.output_dir / f"{match_stats.match_id}.html"` if not
            given.
        coaching_report: The match's AI coaching report, if any.
        config: Supplies the default output directory when output_path
            isn't given. Defaults to ReportConfig() if not given.

    Returns:
        The path written to.
    """
    if output_path is None:
        config = config or ReportConfig()
        output_path = config.output_dir / f"{match_stats.match_id}.html"
    html_text = render_match_report(match_stats, coaching_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def render_history_report(matches: list[MatchStats]) -> str:
    """Renders a cross-match trend report.

    Only single-stat trend lines ship this round (FS%, W/UE ratio, BP%)
    plus the win/loss strip. The combined hold%/return% 2-series line chart
    from the original ideation needs a genuine multi-series line_chart
    variant that doesn't exist yet in charts.py — cut for v1 rather than
    rushed, since there's no real multi-match data yet to validate it
    against either.

    Args:
        matches: MatchStats for each match, ordered oldest to newest.

    Returns:
        The complete, self-contained HTML document as a string.
    """
    template = _env.get_template("history_report.html.jinja")

    fs_points = [charts.LinePoint(m.date, m.serving.first_serve_pct) for m in matches]
    ue_points = [charts.LinePoint(m.date, m.point_outcomes.winner_to_ue_ratio) for m in matches]
    bp_points = [
        charts.LinePoint(m.date, m.receiving.break_point_conversion_pct) for m in matches
    ]
    win_loss_entries = [
        charts.StatusEntry(f"{m.date} vs. {m.opponent}", m.result == "W") for m in matches
    ]

    context = {
        "palette_css": palette.CSS_VARS,
        "match_count": len(matches),
        "fs_chart": charts.line_chart("fs-trend", "First Serve % Over Time", fs_points),
        "ue_chart": charts.line_chart(
            "ue-trend", "Winner : UE Ratio Over Time", ue_points
        ),
        "bp_chart": charts.line_chart(
            "bp-trend", "Break Point % Over Time", bp_points
        ),
        "win_loss_strip": charts.status_strip("win-loss", "Match Results", win_loss_entries),
    }
    return template.render(**context)


def render_history_report_to_file(
    matches: list[MatchStats],
    output_path: Path | None = None,
    config: ReportConfig | None = None,
) -> Path:
    """Renders and writes a cross-match history report to disk.

    Args:
        matches: MatchStats for each match, ordered oldest to newest.
        output_path: Where to write the .html file. Defaults to
            `config.output_dir / "history.html"` if not given.
        config: Supplies the default output directory when output_path
            isn't given. Defaults to ReportConfig() if not given.

    Returns:
        The path written to.
    """
    if output_path is None:
        config = config or ReportConfig()
        output_path = config.output_dir / "history.html"
    html_text = render_history_report(matches)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path
