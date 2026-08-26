"""The JSON-serializable staging schema for a single match.

A MatchRecord is the artifact that gets hand-reviewed before anything reaches
SQL: transform.py builds one from a SwingVision export, review.py saves/loads
it as JSON, and load.py refuses to write it to the database until every
PointRecord.needs_review flag is cleared.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# The point table's full CHECK-constraint vocabulary for point_end_type,
# shared by every module that validates or prompts around it
# (review_assist.py, review_resolve.py, review.py's confirm_point) so
# there's exactly one source of truth instead of parallel copies.
VALID_END_TYPES: frozenset[str] = frozenset(
    {
        "winner",
        "unforced_error",
        "forced_error",
        "ace",
        "double_fault",
        "return_winner",
        "return_error",
    }
)

# Mirrors data/schema.sql's point CHECK constraint: point_end_type
# functionally determines point_won, and the two must never disagree.
WINNING_END_TYPES: frozenset[str] = frozenset({"ace", "winner", "return_winner"})
LOSING_END_TYPES: frozenset[str] = frozenset(
    {"double_fault", "unforced_error", "forced_error", "return_error"}
)


@dataclass
class PointRecord:
    """A single point within the staged, JSON-serializable match schema.

    Attributes:
        game_number: 1-based game number within the set.
        point_number: 1-based point number within the game.
        is_serving: Whether the tracked player was serving this point.
        point_won: Whether the tracked player won this point.
        point_end_type: Canonical point-outcome type (e.g. "ace",
            "unforced_error") — one of the values allowed by the point
            table's CHECK constraint.
        first_serve_in: Whether the first serve landed in, if known.
        second_serve_in: Whether the second serve landed in, if known.
        net_approach: Whether the tracked player approached the net this
            point. SwingVision reports nothing for this; it stays False
            until filled in by hand during review. A net approach's own
            success is always just point_won restricted to net_approach
            points — there's no separate net_point_won column; it was pure
            redundant storage (always exactly point_won when net_approach
            was true) and was dropped as a 3NF cleanup.
        is_tiebreak_game: Whether this point was played in a tiebreak game.
        needs_review: True when point_end_type came from one of
            SwingVision's less reliable AI-guessed categories (see
            config.NEEDS_REVIEW_END_TYPES) or from reconstruct.py's
            Shots-based heuristic (which flags every point it produces,
            not just the ambiguous ones), and hasn't been confirmed by
            hand yet. finalize_and_load refuses to write a match while this
            is True on any of its points.
        notes: Free-text notes for this point.
        ai_suggested_point_end_type: A Claude-suggested refinement of
            point_end_type, from review_assist.py — set only when the user
            has explicitly run that (opt-in, costs an API call) step.
            Never clears needs_review by itself: per the user's explicit
            rule, a Claude suggestion is one more thing to confirm, not a
            second silent auto-tagger.
        ai_suggestion_reasoning: The reasoning Claude gave for that
            suggestion, shown alongside it during review.
        source_point_number: The original SwingVision match-wide "Point"
            number this was reconstructed from (see reconstruct.py), or
            None for a point that came from a direct Points-sheet parse.
            Lets pipeline.suggest() re-fetch this point's raw shots from
            the source export without re-deriving the mapping.
        review_answer: The human reviewer's own free-text explanation of
            what actually happened at this point (e.g. "she was way out of
            position, that was a clean winner down the line"), written
            during manual review instead of hand-editing point_end_type/
            point_won directly. Distinct from ai_suggested_point_end_type:
            that's Claude's independent guess from shot data alone, before
            any human input; this is the human's own stated conclusion,
            which pipeline.resolve() (review_resolve.py) parses into the
            resolved_* fields below.
        resolved_point_end_type: Claude's parse of review_answer into the
            point table's canonical point_end_type vocabulary — a
            translation of what the human already said, not an independent
            judgment call. Set only after pipeline.resolve() runs.
        resolved_point_won: Same, parsed for point_won.
        resolved_net_approach: Same, parsed for net_approach.
        resolution_reasoning: Claude's explanation of how it parsed
            review_answer into the resolved_* fields, for the human to
            sanity-check before applying.
        needs_review still isn't cleared by any of the above — a separate,
        explicit pipeline.apply_resolutions() call is what copies
        resolved_* onto the real fields and clears needs_review, so a
        parsing mistake here still gets one more human checkpoint before
        it reaches finalize().
    """

    game_number: int
    point_number: int
    is_serving: bool
    point_won: bool
    point_end_type: str
    first_serve_in: bool | None = None
    second_serve_in: bool | None = None
    net_approach: bool = False
    is_tiebreak_game: bool = False
    needs_review: bool = False
    notes: str | None = None
    ai_suggested_point_end_type: str | None = None
    ai_suggestion_reasoning: str | None = None
    source_point_number: int | None = None
    review_answer: str | None = None
    resolved_point_end_type: str | None = None
    resolved_point_won: bool | None = None
    resolved_net_approach: bool | None = None
    resolution_reasoning: str | None = None


@dataclass
class SetRecord:
    """A single set within the staged, JSON-serializable match schema.

    Attributes:
        set_number: 1-based set number within the match.
        games_won: Games won by the tracked player in this set.
        games_lost: Games lost by the tracked player in this set.
        is_tiebreak_set: Whether this set was decided by a tiebreak.
        points: All points played in this set, in play order.
    """

    set_number: int
    games_won: int
    games_lost: int
    is_tiebreak_set: bool = False
    points: list[PointRecord] = field(default_factory=list)


@dataclass
class MatchRecord:
    """The staged, JSON-serializable representation of one full match.

    This is the artifact that gets hand-reviewed before anything reaches
    SQL: transform.py builds one from a SwingVision export, review.py
    saves/loads it as JSON, and load.py refuses to write it to the database
    until every PointRecord.needs_review flag is cleared.

    Attributes:
        date: ISO-format date of the match.
        opponent: Opponent's name.
        result: Match result, "W" or "L".
        match_type: "competitive" or "practice".
        location: Where the match was played, if known.
        energy_rating: Self-reported energy rating (1-5), if provided.
        mental_rating: Self-reported mental rating (1-5), if provided.
        pros: Self-reported "what went well" notes.
        cons: Self-reported "what needs work" notes.
        notes: Free-text notes for the match.
        source_file: Path to the SwingVision export this record was built
            from, for traceability during review. None for a match merged
            from multiple exports — see source_files instead.
        source_files: Paths to the SwingVision exports this record was
            merged from (in play order), for a match whose recording was
            interrupted and split into multiple files — see
            pipeline.ingest_multi_part and reconstruct.merge_shots. None
            for a single-file match (the common case), which uses
            source_file instead. suggest() checks this first before
            falling back to source_file.
        sets: All sets played in this match, in play order.
        import_notes: Informational data-quality notes from quality_check.py
            (e.g. a reconstructed set score that doesn't match the Sets
            sheet's own summary, a serve-order mismatch against user-
            supplied ground truth, gap/exclusion counts). Unlike
            needs_review, these never block finalize() — they're for the
            reviewer's awareness, not a per-point confirmation gate.
        shot_pattern_summary: Aggregate shot-sequence stats (e.g. average
            rally length) computed once at ingest time from the raw Shots
            data, for the AI coach to optionally cite. None for a
            direct-parse match (no raw shots involved) or when there was
            nothing to summarize.
    """

    date: str
    opponent: str
    result: str
    match_type: str = "competitive"
    location: str | None = None
    energy_rating: int | None = None
    mental_rating: int | None = None
    pros: str | None = None
    cons: str | None = None
    notes: str | None = None
    source_file: str | None = None
    source_files: list[str] | None = None
    sets: list[SetRecord] = field(default_factory=list)
    import_notes: list[str] = field(default_factory=list)
    shot_pattern_summary: dict[str, float] | None = None

    def to_dict(self) -> dict:
        """Converts this record to a plain, JSON-serializable dict.

        Returns:
            A nested dict representation suitable for json.dumps.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MatchRecord":
        """Rebuilds a MatchRecord from a dict produced by to_dict.

        Tolerates a dict missing optional keys (e.g. a user hand-trimmed
        the pending JSON while resolving a review flag) by falling back to
        each dataclass field's default.

        Args:
            data: A dict shaped like the output of to_dict, with a "sets"
                key containing a list of set dicts, each with a "points"
                key containing a list of point dicts.

        Returns:
            The reconstructed MatchRecord.
        """
        sets = []
        for raw_set in data.get("sets", []):
            points = [PointRecord(**p) for p in raw_set.get("points", [])]
            set_kwargs = {k: v for k, v in raw_set.items() if k != "points"}
            sets.append(SetRecord(points=points, **set_kwargs))
        match_kwargs = {k: v for k, v in data.items() if k != "sets"}
        return cls(sets=sets, **match_kwargs)
