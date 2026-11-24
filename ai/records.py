"""JSON-serializable dataclasses for the AI's output.

CoachingReport is the artifact saved to ai/reports/<match_id>.json. Unlike
swingvision_import's MatchRecord, there's no review gate here — a report is
complete the moment it's generated.

Terminology note. The code calls this a "coaching report", but the output is
**tips and advice** grounded in the match's own stats - not a replacement
for a real coach, and never written in the voice of one. Items are framed
as "something to try" / "worth looking at", with a stat attached, not as
training prescriptions. See the feedback-tips-not-coach memory for the rule.

CoachingReport / CoachingItem survive as internal names because the module
and file structure (ai/, generate.py, journal.py) is still "coaching
engine" in CLAUDE.md. The user-facing wording in prompts, prose, reports
and the UI is the one that changed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Category = Literal["strategy", "practice_idea", "fitness"]
Priority = Literal["high", "medium", "low"]

_PRIORITY_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}


@dataclass(frozen=True)
class SupportingStat:
    """A stat that grounds a tip in this match's actual numbers.

    This is what makes "specific to this match's stats, not generic
    boilerplate" (the UAT checklist item in CLAUDE.md) something a test can
    assert on, rather than eyeballing prose.

    Attributes:
        stat: The stat's name/abbreviation (e.g. "BP%"), matching
            docs/stat-definitions.md.
        value: The stat's value for this match.
        comparison_label: What it's being compared to, if anything (e.g.
            "your season average").
        comparison_value: The comparison value, if comparison_label is set.
    """

    stat: str
    value: float
    comparison_label: str | None = None
    comparison_value: float | None = None


@dataclass(frozen=True)
class CoachingItem:
    """One tip grounded in this match's data.

    Attributes:
        category: Which specialist produced this item.
        observation: The pattern in the data that prompted this item.
        recommendation: One suggestion the stat supports - a tip to try,
            not a prescription to follow.
        supporting_stat: The stat that grounds this item in this match.
        priority: Impact ranking, used to build the derived improvement plan.
    """

    category: Category
    observation: str
    recommendation: str
    supporting_stat: SupportingStat
    priority: Priority


@dataclass(frozen=True)
class PracticeIdeaItem(CoachingItem):
    """A tip with a specific practice idea the player could try.

    Attributes:
        practice_name: Short name for the idea (e.g. "Deep cross-court rally").
        how_to_try: One sentence on how to try it - a suggestion, not a
            prescribed frequency or duration. Leaving reps and timing to
            the player or their actual coach is deliberate.
    """

    practice_name: str
    how_to_try: str


@dataclass(frozen=True)
class FitnessItem(CoachingItem):
    """A tip tied to a physical/mental conditioning theme.

    Attributes:
        focus_area: The conditioning theme (e.g. "endurance", "recovery",
            "composure under pressure").
    """

    focus_area: str


def _item_to_dict(item: CoachingItem) -> dict:
    """Converts a single item to a plain dict.

    Args:
        item: A CoachingItem, PracticeIdeaItem, or FitnessItem.

    Returns:
        Its dict representation, suitable for json.dumps.
    """
    return asdict(item)


@dataclass
class CoachingReport:
    """The full AI-generated tips bundle for one match.

    Attributes:
        match_id: The match this report is for.
        generated_at: ISO-format timestamp of generation.
        model: The Claude model id used to generate this report.
        strategy: Strategy tips.
        practice_ideas: Practice ideas - things to try if a pattern stands
            out. Deliberately framed as "ideas" rather than "drills" to
            stop it reading as a training prescription.
        fitness: Fitness/conditioning tips.
    """

    match_id: int
    generated_at: str
    model: str
    strategy: list[CoachingItem] = field(default_factory=list)
    practice_ideas: list[PracticeIdeaItem] = field(default_factory=list)
    fitness: list[FitnessItem] = field(default_factory=list)

    @property
    def improvement_plan(self) -> list[CoachingItem]:
        """All items across every category, sorted by priority (high first).

        Named "improvement plan" because that is how it has been referred
        to in the code and docs since v1; the content is a prioritised
        list of tips, not a prescribed training plan. See
        feedback-tips-not-coach in memory for the wording rule.

        This is a derived view, not a separately generated category — see
        CLAUDE.md: there is deliberately no 4th "synthesis" LLM call, so
        this can never drift out of sync with strategy/practice_ideas/fitness.

        Returns:
            All items from strategy, practice_ideas, and fitness combined,
            sorted by priority.
        """
        all_items: list[CoachingItem] = [
            *self.strategy,
            *self.practice_ideas,
            *self.fitness,
        ]
        return sorted(all_items, key=lambda item: _PRIORITY_ORDER[item.priority])

    def to_dict(self) -> dict:
        """Converts this report to a plain, JSON-serializable dict.

        Returns:
            A nested dict representation suitable for json.dumps. The
            derived improvement_plan is not included — recompute it from
            strategy/practice_ideas/fitness after loading.
        """
        return {
            "match_id": self.match_id,
            "generated_at": self.generated_at,
            "model": self.model,
            "strategy": [_item_to_dict(item) for item in self.strategy],
            "practice_ideas": [_item_to_dict(item) for item in self.practice_ideas],
            "fitness": [_item_to_dict(item) for item in self.fitness],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CoachingReport":
        """Rebuilds a CoachingReport from a dict produced by to_dict.

        Args:
            data: A dict shaped like the output of to_dict.

        Returns:
            The reconstructed CoachingReport.
        """

        def _rebuild(item_cls: type, raw: dict) -> CoachingItem:
            kwargs = {**raw, "supporting_stat": SupportingStat(**raw["supporting_stat"])}
            return item_cls(**kwargs)

        return cls(
            match_id=data["match_id"],
            generated_at=data["generated_at"],
            model=data["model"],
            strategy=[_rebuild(CoachingItem, raw) for raw in data.get("strategy", [])],
            practice_ideas=[
                _rebuild(PracticeIdeaItem, raw) for raw in data.get("practice_ideas", [])
            ],
            fitness=[_rebuild(FitnessItem, raw) for raw in data.get("fitness", [])],
        )
