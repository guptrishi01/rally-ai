"""Dataclasses mirroring SwingVision's raw exported rows, pre-transformation.

Confirmed against two real exports: Sets/Games/Points/Settings/Shots/Stats
all exist, but Points/Games/Stats are empty without a Pro subscription —
only Sets (final scores) and Shots (full per-shot AI tracking) are
populated. RawShotRow/RawSettings exist specifically so reconstruct.py can
build point-level data from Shots when Points is empty; Stats isn't modeled
at all since nothing in it survives past being a zeroed-out placeholder in
what we've seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RawSettings:
    """Match metadata from SwingVision's "Settings" sheet.

    Attributes:
        host_name: Display name of the "host" side — the account that
            recorded the session, always the tracked player (confirmed:
            "Host Team" held the real user's name in both real exports
            inspected). Compare RawShotRow.player against this to tell
            which side hit a given shot.
        guest_name: Display name of the "guest" side (the opponent).
    """

    host_name: str
    guest_name: str


@dataclass
class RawSetRow:
    """One row from SwingVision's "Sets" sheet, before transformation.

    Attributes:
        set_number: 1-based set number within the match.
        winner: Raw winner label from the export — confirmed real values
            are "host"/"guest" (not "player"/"opponent"), not yet
            normalized.
        games_won: Games won by the tracked player in this set (SwingVision
            calls this column "Host Score").
        games_lost: Games lost by the tracked player in this set
            ("Guest Score").
    """

    set_number: int
    winner: str
    games_won: int
    games_lost: int


@dataclass
class RawGameRow:
    """One row from SwingVision's "Games" sheet, before transformation.

    Only usable with a Pro export — confirmed empty (one placeholder row)
    in both non-Pro exports inspected.

    Attributes:
        set_number: 1-based set number this game belongs to.
        game_number: 1-based game number within the set.
        server: Raw server label from the export ("host"/"guest"), not yet
            normalized.
        winner: Raw game-winner label from the export, not yet normalized.
    """

    set_number: int
    game_number: int
    server: str
    winner: str


@dataclass
class RawPointRow:
    """One row from SwingVision's "Points" sheet, before transformation.

    Only usable with a Pro export — confirmed empty (0 rows) in both
    non-Pro exports inspected; reconstruct.py is the fallback when this
    list is empty.

    Attributes:
        set_number: 1-based set number this point belongs to.
        game_number: 1-based game number within the set.
        point_number: 1-based point number within the game.
        server: Raw server label from the export ("host"/"guest"), not yet
            normalized.
        winner: Raw point-winner label from the export, not yet normalized.
        end_type: Raw point-outcome label from the export, not yet mapped
            onto Rally AI's canonical set. Column is confirmed to be
            "Detail" (not "Shot Type"), but its value vocabulary is still
            unverified — no Pro export has been seen to check against.
        first_serve_in: Whether the first serve landed in, or None if the
            export didn't report it for this point.
        second_serve_in: Whether the second serve landed in, or None if
            there was no second serve or it wasn't reported.
    """

    set_number: int
    game_number: int
    point_number: int
    server: str
    winner: str
    end_type: str
    first_serve_in: bool | None = None
    second_serve_in: bool | None = None


@dataclass
class RawShotRow:
    """One row from SwingVision's "Shots" sheet — real per-shot AI tracking.

    Unlike Sets/Games/Points, this sheet is fully populated even without a
    Pro subscription. `reconstruct.py` builds point-level `PointRecord`s
    from these when the Points sheet itself is empty.

    Attributes:
        point_number: The match-wide point counter this shot belongs to
            (not renumbered per-game — reconstruct.py handles that
            separately, since SwingVision's own Game/Set columns on this
            sheet are always 0 and unusable).
        shot_number: 1-based order of this shot within its point.
        player: The raw display name of who hit this shot (e.g.
            "Rishi Gupta"), not yet compared against RawSettings.host_name.
        shot_type: SwingVision's shot-role label — confirmed real values
            include "first_serve", "second_serve", "first_return",
            "second_return", "serve_plus_one", "return_plus_one",
            "in_play", "none".
        stroke: The stroke used — confirmed real values include "Serve",
            "Forehand", "Backhand", "Forehand Volley", "Backhand Volley",
            "Overhead", "Feed". A Volley/Overhead stroke is the only
            usable signal for net_approach — SwingVision has no dedicated
            column for it at all.
        result: Whether the shot landed — confirmed real values are "In",
            "Out", "Net" (not just In/Out).
        video_time: Seconds from the start of the *full-length* recording
            to this shot ("Video Time"). Confirmed against a real export:
            it tracks the wall clock exactly (Settings start + video_time ==
            the shot's own Start Time, zero drift over 180 points), so it
            indexes the original recording - **not** SwingVision's
            condensed rallies-only export, which is ~1/5 as long. Matching
            the wall clock is not matching the video file: one real
            recording ran 1.99s behind these times, which
            footage.measure_clock_offset corrects for. None when the
            column is absent.
        hit_side: Which end of the court the hitter stood at, "near" or
            "far" relative to the camera ("Hit Side"). Lets footage checks
            say who is who without naming anyone. None when absent.
        hit_zone: Which half of the baseline the hitter stood in, "deuce"
            or "ad" ("Hit Zone"). On a serve this is the serving side,
            which is what tells a second serve (same server, same side
            again) from the next point (other side). None when absent.
        source_part: Index of the export this shot came from, for a match
            merged from several files (see reconstruct.merge_shots). Each
            part's video_time is relative to *that part's own* recording,
            so this is what says which video a timestamp belongs to. 0 for
            a single-file match.
    """

    point_number: int
    shot_number: int
    player: str
    shot_type: str
    stroke: str
    result: str
    video_time: float | None = None
    hit_side: str | None = None
    hit_zone: str | None = None
    source_part: int = 0
    # Spin and speed are only read by footage retrieval; they give the
    # shot a content-addressable fingerprint when seeking into the
    # condensed rallies-only export. None when a sheet doesn't carry
    # them (older exports or sheets where they are blank).
    spin: str | None = None
    speed_mph: float | None = None


@dataclass
class RawMatchExport:
    """The full set of raw rows parsed from one SwingVision .xlsx export.

    Attributes:
        settings: Match metadata (host/guest names). None only if a Settings
            sheet somehow can't be read — callers needing shot-player
            normalization should treat that as an error, not silently skip.
        sets: All rows from the "Sets" sheet.
        games: All rows from the "Games" sheet (Pro only; expect empty).
        points: All rows from the "Points" sheet (Pro only; expect empty —
            empty is the signal to fall back to reconstruct.py on `shots`).
        shots: All rows from the "Shots" sheet — populated even without Pro.
    """

    settings: RawSettings | None = None
    sets: list[RawSetRow] = field(default_factory=list)
    games: list[RawGameRow] = field(default_factory=list)
    points: list[RawPointRow] = field(default_factory=list)
    shots: list[RawShotRow] = field(default_factory=list)
