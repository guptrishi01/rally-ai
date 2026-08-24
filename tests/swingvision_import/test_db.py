from __future__ import annotations

from pathlib import Path

from swingvision_import.db import get_connection

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "data" / "schema.sql"


def test_get_connection_creates_and_initializes_a_new_database(tmp_path: Path):
    db_path = tmp_path / "rallyai.db"
    connection = get_connection(db_path, _SCHEMA_PATH)

    assert db_path.exists()
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"match", "set", "point"} <= tables
    connection.close()


def test_get_connection_is_idempotent_on_an_existing_database(tmp_path: Path):
    """Calling get_connection twice must not re-run schema.sql against an
    existing file - CREATE TABLE would fail the second time around."""
    db_path = tmp_path / "rallyai.db"
    first = get_connection(db_path, _SCHEMA_PATH)
    first.execute("INSERT INTO match (date, opponent, result) VALUES ('2026-08-06', 'Alex', 'W')")
    first.commit()
    first.close()

    second = get_connection(db_path, _SCHEMA_PATH)
    count = second.execute("SELECT COUNT(*) FROM match").fetchone()[0]
    second.close()

    assert count == 1


def test_get_connection_enables_foreign_key_enforcement(tmp_path: Path):
    connection = get_connection(tmp_path / "rallyai.db", _SCHEMA_PATH)
    fk_status = connection.execute("PRAGMA foreign_keys").fetchone()[0]
    connection.close()

    assert fk_status == 1
