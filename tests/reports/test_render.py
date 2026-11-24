from __future__ import annotations

from pathlib import Path

from ai.records import (
    CoachingItem,
    CoachingReport,
    FitnessItem,
    PracticeIdeaItem,
    SupportingStat,
)
from reports.config import ReportConfig
from reports.render import (
    render_history_report,
    render_history_report_to_file,
    render_match_report,
    render_match_report_to_file,
)
from tests.reports.conftest import make_match_stats


def _stat() -> SupportingStat:
    return SupportingStat(stat="FS%", value=60.0)


def _sample_report() -> CoachingReport:
    return CoachingReport(
        match_id=1,
        generated_at="2026-08-06T12:00:00+00:00",
        model="claude-sonnet-5",
        strategy=[CoachingItem("strategy", "obs", "Attack the second serve", _stat(), "high")],
        practice_ideas=[
            PracticeIdeaItem(
                "practice_idea",
                "obs",
                "Cross-court rally idea",
                _stat(),
                "medium",
                "CC Rally",
                "trade cross-court shots until one breaks down",
            )
        ],
        fitness=[
            FitnessItem("fitness", "obs", "Notice sprint recovery", _stat(), "low", "endurance")
        ],
    )


def test_render_match_report_includes_stat_tiles_and_charts(sample_match_stats):
    html_text = render_match_report(sample_match_stats)

    assert "Alex" in html_text
    assert "First serve %" in html_text
    assert "Point Outcome Breakdown" in html_text
    assert "First/Second Serve In vs. Out" in html_text


def test_render_match_report_omits_net_section_when_no_net_approaches():
    stats = make_match_stats(net_approaches=0)

    html_text = render_match_report(stats)

    assert "Net Play" not in html_text


def test_render_match_report_includes_net_section_when_net_approaches_exist(sample_match_stats):
    html_text = render_match_report(sample_match_stats)

    assert "Net Play" in html_text


def test_render_match_report_omits_coaching_sections_without_a_report(sample_match_stats):
    html_text = render_match_report(sample_match_stats, coaching_report=None)

    assert "Tips from this match" not in html_text
    assert "<h2>Strategy tips</h2>" not in html_text


def test_render_match_report_includes_all_three_coaching_sections(sample_match_stats):
    html_text = render_match_report(sample_match_stats, coaching_report=_sample_report())

    assert "Attack the second serve" in html_text
    assert "Cross-court rally idea" in html_text
    assert "CC Rally" in html_text
    assert "Notice sprint recovery" in html_text
    assert "endurance" in html_text
    # The whole section is framed as "tips", not as a prescribed plan.
    assert "Tips from this match" in html_text
    assert "<h2>Practice ideas</h2>" in html_text
    assert "<h2>Fitness tips</h2>" in html_text


def test_render_match_report_escapes_free_text_from_the_database():
    stats = make_match_stats(
        opponent='<script>alert(1)</script>',
        pros="<b>bold pros</b>",
    )

    html_text = render_match_report(stats)

    assert "<script>alert(1)</script>" not in html_text
    assert "&lt;script&gt;" in html_text
    assert "<b>bold pros</b>" not in html_text


def test_render_match_report_to_file_writes_the_html(tmp_path: Path, sample_match_stats):
    output_path = tmp_path / "nested" / "report.html"

    written = render_match_report_to_file(sample_match_stats, output_path)

    assert written == output_path
    assert output_path.exists()
    assert "Alex" in output_path.read_text(encoding="utf-8")


def test_render_match_report_to_file_uses_config_output_dir_by_default(
    tmp_path: Path, sample_match_stats
):
    config = ReportConfig(output_dir=tmp_path / "generated")

    written = render_match_report_to_file(sample_match_stats, config=config)

    assert written == tmp_path / "generated" / "1.html"
    assert written.exists()


def test_render_history_report_includes_trend_charts_and_win_loss_strip():
    matches = [
        make_match_stats(match_id=1, date="2026-07-01", result="W"),
        make_match_stats(match_id=2, date="2026-07-15", result="L"),
        make_match_stats(match_id=3, date="2026-08-01", result="W"),
    ]

    html_text = render_history_report(matches)

    assert "3 matches tracked" in html_text
    assert "First Serve % Trend" in html_text
    assert "Match Results" in html_text


def test_render_history_report_handles_zero_matches_without_crashing():
    html_text = render_history_report([])

    assert "0 matches tracked" in html_text


def test_render_history_report_to_file_writes_the_html(tmp_path: Path):
    output_path = tmp_path / "history.html"

    written = render_history_report_to_file([make_match_stats()], output_path)

    assert written == output_path


def test_render_history_report_to_file_uses_config_output_dir_by_default(tmp_path: Path):
    config = ReportConfig(output_dir=tmp_path / "generated")

    written = render_history_report_to_file([make_match_stats()], config=config)

    assert written == tmp_path / "generated" / "history.html"
    assert written.exists()
