"""Shared fixtures for ai/ tests — a fake Anthropic client that never makes
a real API call, so the test suite never touches the network or spends
real money.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import httpx
import pytest

from ai.config import AICoachConfig
from stats.models import (
    ClutchStats,
    MatchStats,
    NetStats,
    PointOutcomeStats,
    ReceivingStats,
    SelfAssessment,
    ServingStats,
)


def canned_item(category_hint: str) -> dict:
    """Builds a valid, minimal response item for the given specialist.

    Args:
        category_hint: "strategy", "drill", or "fitness".

    Returns:
        A dict matching that specialist's expected JSON item shape.
    """
    item = {
        "observation": f"Sample {category_hint} observation",
        "recommendation": f"Sample {category_hint} recommendation",
        "supporting_stat": {
            "stat": "FS%",
            "value": 55.0,
            "comparison_label": None,
            "comparison_value": None,
        },
        "priority": "medium",
    }
    if category_hint == "drill":
        item["drill_name"] = "Cross-court consistency"
        item["frequency"] = "15 min, 3x/week"
    elif category_hint == "fitness":
        item["focus_area"] = "endurance"
    return item


def _category_from_system_prompt(system: str) -> str:
    if "drill designer" in system:
        return "drill"
    if "conditioning coach" in system:
        return "fitness"
    return "strategy"


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeThinkingBlock:
    thinking: str
    type: str = "thinking"


@dataclass
class FakeMessage:
    content: list


def _fake_api_error() -> anthropic.APIError:
    """Builds a real anthropic.APIError, minimal but genuine.

    Using the SDK's own exception class (rather than a lookalike) is what
    makes the `except anthropic.APIError` in client.py actually exercised
    by a test, not just assumed to work.

    Returns:
        An anthropic.APIConnectionError, the simplest APIError subclass to
        construct (no real HTTP response needed).
    """
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(message="simulated connection failure", request=request)


@dataclass
class FakeMessages:
    """Stands in for anthropic.Anthropic().messages.

    Attributes:
        response_for: category -> list of raw item dicts to return as JSON.
            Categories not present here get one default canned_item().
        raise_for: categories whose response should be unparseable JSON,
            to exercise the SpecialistError "bad response" path.
        api_error_for: categories whose call() should raise a real
            anthropic.APIError, to exercise the SpecialistError "API call
            itself failed" path.
        calls: every kwargs dict passed to create(), for assertions.
    """

    response_for: dict[str, list[dict]] = field(default_factory=dict)
    raise_for: set[str] = field(default_factory=set)
    api_error_for: set[str] = field(default_factory=set)
    calls: list[dict] = field(default_factory=list)

    def create(self, **kwargs: object) -> FakeMessage:
        self.calls.append(kwargs)
        category = _category_from_system_prompt(kwargs["system"])
        if category in self.api_error_for:
            raise _fake_api_error()
        if category in self.raise_for:
            return FakeMessage(content=[FakeTextBlock(text="not valid json")])
        items = self.response_for.get(category, [canned_item(category)])
        return FakeMessage(content=[FakeTextBlock(text=json.dumps(items))])


class FakeAnthropicClient:
    """An anthropic.Anthropic-shaped fake for tests."""

    def __init__(
        self,
        response_for: dict[str, list[dict]] | None = None,
        raise_for: set[str] | None = None,
        api_error_for: set[str] | None = None,
    ) -> None:
        self.messages = FakeMessages(response_for or {}, raise_for or set(), api_error_for or set())


@pytest.fixture
def fake_client() -> FakeAnthropicClient:
    return FakeAnthropicClient()


@pytest.fixture
def ai_config(tmp_path: Path) -> AICoachConfig:
    return AICoachConfig(reports_dir=tmp_path / "reports")


@pytest.fixture
def sample_match_stats() -> MatchStats:
    return MatchStats(
        match_id=1,
        date="2026-08-06",
        opponent="Alex",
        result="W",
        serving=ServingStats(10, 6, 60.0, 4, 2, 50.0, 2, 1, 3, 4, 75.0),
        receiving=ReceivingStats(2, 1, 50.0, 2, 4, 50.0),
        point_outcomes=PointOutcomeStats(20, 12, 60.0, 6, 3, 2, 1, 1, 2.0),
        net=NetStats(3, 2, 66.7),
        clutch=ClutchStats(2, 1, 50.0),
        self_assessment=SelfAssessment(4, 3, "Served big", "Slow starts", None),
    )
