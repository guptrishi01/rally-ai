from __future__ import annotations

import pytest

from stats.models import (
    ClutchStats,
    MatchStats,
    NetStats,
    PointOutcomeStats,
    ReceivingStats,
    SelfAssessment,
    ServingStats,
)


def make_match_stats(
    match_id: int = 1,
    date: str = "2026-08-06",
    opponent: str = "Alex",
    result: str = "W",
    net_approaches: int = 3,
    pros: str | None = "Served big",
    cons: str | None = "Slow starts",
) -> MatchStats:
    return MatchStats(
        match_id=match_id,
        date=date,
        opponent=opponent,
        result=result,
        serving=ServingStats(10, 6, 60.0, 4, 2, 50.0, 2, 1, 3, 4, 75.0),
        receiving=ReceivingStats(2, 1, 50.0, 2, 4, 50.0),
        point_outcomes=PointOutcomeStats(20, 12, 60.0, 6, 3, 2, 1, 1, 2.0),
        net=NetStats(net_approaches, 2, 66.7 if net_approaches else 0.0),
        clutch=ClutchStats(2, 1, 50.0),
        self_assessment=SelfAssessment(4, 3, pros, cons, None),
    )


@pytest.fixture
def sample_match_stats() -> MatchStats:
    return make_match_stats()
