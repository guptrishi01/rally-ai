from __future__ import annotations

from ai.records import (
    CoachingItem,
    CoachingReport,
    FitnessItem,
    PracticeIdeaItem,
    SupportingStat,
)


def _stat(value: float) -> SupportingStat:
    return SupportingStat(stat="FS%", value=value)


def _report() -> CoachingReport:
    return CoachingReport(
        match_id=1,
        generated_at="2026-08-06T12:00:00+00:00",
        model="claude-sonnet-5",
        strategy=[
            CoachingItem("strategy", "obs1", "rec1", _stat(1), "low"),
            CoachingItem("strategy", "obs2", "rec2", _stat(2), "high"),
        ],
        practice_ideas=[
            PracticeIdeaItem(
                "practice_idea",
                "obs3",
                "rec3",
                _stat(3),
                "medium",
                "Deep cross-court",
                "trade deep cross-court balls until one breaks down",
            )
        ],
        fitness=[FitnessItem("fitness", "obs4", "rec4", _stat(4), "high", "endurance")],
    )


def test_improvement_plan_sorts_across_all_categories_by_priority():
    report = _report()

    plan = report.improvement_plan

    assert [item.priority for item in plan] == ["high", "high", "medium", "low"]
    assert len(plan) == 4


def test_improvement_plan_is_derived_not_stored():
    report = _report()
    plan_before = report.improvement_plan
    report.strategy.append(CoachingItem("strategy", "obs5", "rec5", _stat(5), "high"))

    plan_after = report.improvement_plan

    assert len(plan_after) == len(plan_before) + 1


def test_to_dict_then_from_dict_round_trips_every_category():
    report = _report()

    rebuilt = CoachingReport.from_dict(report.to_dict())

    assert rebuilt == report
    assert rebuilt.practice_ideas[0].practice_name == "Deep cross-court"
    assert "cross-court" in rebuilt.practice_ideas[0].how_to_try
    assert rebuilt.fitness[0].focus_area == "endurance"
    assert [item.priority for item in rebuilt.improvement_plan] == [
        "high",
        "high",
        "medium",
        "low",
    ]


def test_to_dict_omits_the_derived_improvement_plan_field():
    data = _report().to_dict()

    assert "improvement_plan" not in data


def test_to_dict_uses_the_practice_ideas_key_not_drills():
    """Serialization regression guard: a cached report under "drills" would
    not round-trip through the renamed loader, so the key rename has to be
    pinned at the serialization boundary - not just at the dataclass."""
    data = _report().to_dict()

    assert "practice_ideas" in data
    assert "drills" not in data
