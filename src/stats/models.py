"""Dataclasses for derived stats, mirroring docs/stat-definitions.md exactly.

Every percentage/ratio field returns 0.0 when its denominator is 0 (e.g. no
first serves attempted) rather than raising or returning None — that keeps
every field a plain float, renderable without Optional-handling throughout
the AI context builder and the HTML report, at the cost of not distinguishing
"0%" from "undefined." Revisit only if that distinction turns out to matter.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PointRow:
    """One point, as fetched from the database for aggregation.

    Attributes:
        set_number: 1-based set number within the match.
        game_number: 1-based game number within the set (resets each set).
        point_number: 1-based point number within the game.
        is_serving: Whether the tracked player was serving this point.
        first_serve_in: Whether the first serve landed in, if known.
        second_serve_in: Whether the second serve landed in, if known.
        point_won: Whether the tracked player won this point.
        point_end_type: Canonical point-outcome type (e.g. "ace").
        net_approach: Whether the tracked player approached the net. A net
            approach's own success is always just point_won restricted to
            net_approach points - there's no separate net_point_won field.
        is_tiebreak_game: Whether this point was played in a tiebreak game.
    """

    set_number: int
    game_number: int
    point_number: int
    is_serving: bool
    first_serve_in: bool | None
    second_serve_in: bool | None
    point_won: bool
    point_end_type: str
    net_approach: bool
    is_tiebreak_game: bool


@dataclass(frozen=True)
class ServingStats:
    """Serving stats for a match or a single set. See stat-definitions.md.

    Attributes:
        first_serves_total: FST.
        first_serves_in: FSI.
        first_serve_pct: FS%.
        second_serves_total: SST.
        second_serves_in: SSI.
        second_serve_pct: SS%.
        aces: ACE.
        double_faults: DF.
        service_games_won: SGW.
        service_games_total: SGT.
        service_hold_pct: SH%.
    """

    first_serves_total: int
    first_serves_in: int
    first_serve_pct: float
    second_serves_total: int
    second_serves_in: int
    second_serve_pct: float
    aces: int
    double_faults: int
    service_games_won: int
    service_games_total: int
    service_hold_pct: float


@dataclass(frozen=True)
class ReceivingStats:
    """Receiving stats for a match or a single set. See stat-definitions.md.

    Attributes:
        break_points_total: BPT.
        break_points_converted: BPC.
        break_point_conversion_pct: BP%.
        return_games_won: RGW.
        return_games_total: RGT.
        return_win_pct: RW%.
    """

    break_points_total: int
    break_points_converted: int
    break_point_conversion_pct: float
    return_games_won: int
    return_games_total: int
    return_win_pct: float


@dataclass(frozen=True)
class PointOutcomeStats:
    """Point outcome stats for a match or a single set. See stat-definitions.md.

    Attributes:
        total_points_played: TPP.
        total_points_won: TPW.
        points_won_pct: PW%.
        winners: W.
        unforced_errors: UE.
        forced_errors: FE.
        return_winners: RW.
        return_errors: RE.
        winner_to_ue_ratio: W/UE. 0.0 when UE is 0.
    """

    total_points_played: int
    total_points_won: int
    points_won_pct: float
    winners: int
    unforced_errors: int
    forced_errors: int
    return_winners: int
    return_errors: int
    winner_to_ue_ratio: float


@dataclass(frozen=True)
class NetStats:
    """Net stats for a match or a single set. See stat-definitions.md.

    Attributes:
        net_approaches: NA.
        net_points_won: NPW.
        net_success_pct: NS%.
    """

    net_approaches: int
    net_points_won: int
    net_success_pct: float


@dataclass(frozen=True)
class ClutchStats:
    """Clutch stats for a match or a single set. See stat-definitions.md.

    Attributes:
        deuce_points_played: DPP.
        deuces_converted: DC — deuce points where point_won is True (by
            analogy with break_points_converted; stat-definitions.md's
            "led to winning the game" wording isn't stricter than that
            elsewhere, so this follows the same convention as BPC).
        deuce_conversion_pct: DC%.
    """

    deuce_points_played: int
    deuces_converted: int
    deuce_conversion_pct: float


@dataclass(frozen=True)
class SelfAssessment:
    """Self-reported, always-manual fields from the match table.

    Attributes:
        energy_rating: 1-5, if provided.
        mental_rating: 1-5, if provided.
        pros: Free-text "what went well."
        cons: Free-text "what needs work."
        notes: Free-text general notes - the third leg of the Journal tab's
            sticky note, alongside pros/cons. match_stats() didn't select
            it until the Journal feature needed it; the column has existed
            on `match` from the start.
    """

    energy_rating: int | None
    mental_rating: int | None
    pros: str | None
    cons: str | None
    notes: str | None


@dataclass(frozen=True)
class MatchStats:
    """The full derived-stats bundle for one match.

    Attributes:
        match_id: The match's database id.
        date: ISO-format match date.
        opponent: Opponent's name.
        result: "W" or "L".
        serving: Serving stats.
        receiving: Receiving stats.
        point_outcomes: Point outcome stats.
        net: Net stats.
        clutch: Clutch stats.
        self_assessment: Self-reported fields.
    """

    match_id: int
    date: str
    opponent: str
    result: str
    serving: ServingStats
    receiving: ReceivingStats
    point_outcomes: PointOutcomeStats
    net: NetStats
    clutch: ClutchStats
    self_assessment: SelfAssessment


@dataclass(frozen=True)
class CareerHighlight:
    """One standout match, surfaced on the Overview tab.

    Attributes:
        match_id: The match's database id.
        date: ISO-format match date.
        opponent: Opponent's name.
        value: The stat value that earned this highlight.
        label: What the value is (e.g. "Best points-won%", "Most aces").
    """

    match_id: int
    date: str
    opponent: str
    value: float
    label: str


@dataclass(frozen=True)
class CareerStats:
    """Aggregate stats across every finalized match, for the Overview tab.

    Unlike MatchStats/docs/stat-definitions.md, this isn't a per-match
    stat category - it's a cross-match rollup, computed by
    queries.career_stats_from_matches over an already-fetched list of
    MatchStats (see queries.all_match_stats for the concurrent fetch).

    Attributes:
        total_matches: Count of finalized matches.
        wins: Count with result == "W".
        losses: Count with result == "L".
        win_pct: wins / total_matches, 0.0 if there are no matches.
        current_streak_result: "W" or "L" — the most recent match's
            result — or None if there are no matches.
        current_streak_count: How many matches in a row ended in
            current_streak_result, counting back from the most recent.
        avg_first_serve_pct: Mean FS% across every match.
        avg_points_won_pct: Mean PW% across every match.
        best_match_by_points_won_pct: The match with the highest PW%, or
            None if there are no matches.
        most_aces_in_a_match: The match with the most aces, or None if
            there are no matches.
    """

    total_matches: int
    wins: int
    losses: int
    win_pct: float
    current_streak_result: str | None
    current_streak_count: int
    avg_first_serve_pct: float
    avg_points_won_pct: float
    best_match_by_points_won_pct: CareerHighlight | None
    most_aces_in_a_match: CareerHighlight | None
