"""Configuration for parsing SwingVision's exported .xlsx match file.

Sheet and column names for Sets/Games/Points/Settings/Shots are now
confirmed against two real (non-Pro) exports — see raw.py's module
docstring. Points/Games header *names* are confirmed too (the sheets exist
with real headers), but their *value* vocabulary (e.g. what "Serve State"
or "Detail" actually contain) is still unverified, since both real exports
had zero data rows in those sheets — Pro is required to populate them.
Correct that once a Pro export is available, not the parsing logic in
parse.py, which stays alias-driven specifically so this doesn't require a
rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SHEET_NAMES: dict[str, str] = {
    "settings": "Settings",
    "sets": "Sets",
    "games": "Games",
    "points": "Points",
    "shots": "Shots",
}

DEFAULT_COLUMN_ALIASES: dict[str, dict[str, list[str]]] = {
    "settings": {
        "host_name": ["Host Team"],
        "guest_name": ["Guest Team"],
    },
    "sets": {
        "set_number": ["Set", "Set #", "SetNumber"],
        "winner": ["Set Winner", "Winner"],
        "games_won": ["Host Score", "Games Won"],
        "games_lost": ["Guest Score", "Games Lost"],
    },
    "games": {
        "set_number": ["Set", "Set #"],
        "game_number": ["Game", "Game #", "GameNumber"],
        "server": ["Server", "Serving Player"],
        "winner": ["Game Winner", "Winner"],
    },
    "points": {
        "set_number": ["Set", "Set #"],
        "game_number": ["Game", "Game #", "GameNumber"],
        "point_number": ["Point", "Point #", "PointNumber"],
        "server": ["Match Server", "Server", "Serving Player"],
        "winner": ["Point Winner", "Winner"],
        "end_type": ["Detail", "Shot Type", "Point End Type", "Outcome"],
        # No confirmed real column maps to these yet — real Points headers
        # carry serve detail in a single "Serve State" column instead of
        # separate 1st/2nd-in flags, but its value vocabulary is unverified
        # (no Pro data to check against), so it isn't wired in. These stay
        # optional (see parse.py's _OPTIONAL_FIELDS) and will just be None.
        "first_serve_in": ["1st Serve In", "First Serve In"],
        "second_serve_in": ["2nd Serve In", "Second Serve In"],
    },
    "shots": {
        "point_number": ["Point"],
        "shot_number": ["Shot"],
        "player": ["Player"],
        "shot_type": ["Type"],
        "stroke": ["Stroke"],
        "result": ["Result"],
    },
}

# SwingVision's AI classification for these point-outcome labels is unreliable
# enough that a match may not be finalized into SQL until a human confirms
# them (see review.unresolved_flags / load.finalize_and_load). Applies to the
# Pro-only direct-parse Points path; the Shots-reconstruction fallback path
# (reconstruct.py) flags every point regardless of end_type — see its module
# docstring.
NEEDS_REVIEW_END_TYPES: frozenset[str] = frozenset(
    {"winner", "unforced_error", "forced_error"}
)


@dataclass(frozen=True)
class ImportConfig:
    """Configuration for parsing a SwingVision export and loading it into SQL.

    Attributes:
        sheet_names: Maps each logical sheet key ("sets", "games", "points",
            "settings", "shots") to the sheet name expected in the .xlsx
            workbook.
        column_aliases: Maps each sheet key to a mapping of canonical field
            name -> list of header strings that may represent it in a real
            export. Update this, not the parsing logic in parse.py, if a
            future export uses different header text.
        needs_review_end_types: Point-outcome labels whose AI classification
            is unreliable enough that a match may not be finalized into SQL
            until a human confirms them.
        pending_dir: Directory where staged (pre-review) match JSON files
            are written by review.save_pending.
        db_path: Path to the SQLite database file.
        schema_path: Path to the SQL schema script applied on first use.
    """

    sheet_names: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SHEET_NAMES))
    column_aliases: dict[str, dict[str, list[str]]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_COLUMN_ALIASES.items()}
    )
    needs_review_end_types: frozenset[str] = NEEDS_REVIEW_END_TYPES
    pending_dir: Path = _REPO_ROOT / "src" / "swingvision_import" / "pending"
    db_path: Path = _REPO_ROOT / "data" / "rallyai.db"
    schema_path: Path = _REPO_ROOT / "data" / "schema.sql"
