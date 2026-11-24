from __future__ import annotations

import pytest

from swingvision_import.records import (
    MatchRecord,
    PointRecord,
    SetRecord,
    parse_footage_rally_count,
)


def _multi_set_record() -> MatchRecord:
    return MatchRecord(
        date="2026-08-06",
        opponent="Alex",
        result="W",
        energy_rating=4,
        pros="Good serving",
        sets=[
            SetRecord(
                set_number=1,
                games_won=6,
                games_lost=4,
                points=[
                    PointRecord(1, 1, True, True, "ace"),
                    PointRecord(1, 2, True, True, "winner", needs_review=True),
                ],
            ),
            SetRecord(
                set_number=2,
                games_won=6,
                games_lost=2,
                points=[PointRecord(1, 1, False, False, "double_fault")],
            ),
        ],
    )


def test_to_dict_then_from_dict_round_trips_a_multi_set_match():
    record = _multi_set_record()
    rebuilt = MatchRecord.from_dict(record.to_dict())

    assert rebuilt == record
    assert len(rebuilt.sets) == 2
    assert len(rebuilt.sets[0].points) == 2
    assert rebuilt.sets[1].points[0].point_end_type == "double_fault"


def test_from_dict_applies_defaults_for_a_hand_trimmed_dict():
    """A user hand-editing the pending JSON to resolve a flag may delete keys
    they don't care about rather than retyping the whole object; from_dict
    must not require every optional field to be present."""
    minimal = {
        "date": "2026-08-06",
        "opponent": "Alex",
        "result": "W",
        "sets": [
            {
                "set_number": 1,
                "games_won": 6,
                "games_lost": 4,
                "points": [
                    {
                        "game_number": 1,
                        "point_number": 1,
                        "is_serving": True,
                        "point_won": True,
                        "point_end_type": "ace",
                    }
                ],
            }
        ],
    }

    record = MatchRecord.from_dict(minimal)

    assert record.match_type == "competitive"
    assert record.energy_rating is None
    point = record.sets[0].points[0]
    assert point.needs_review is False
    assert point.net_approach is False


def test_records_with_identical_field_values_are_equal():
    assert _multi_set_record() == _multi_set_record()


def test_records_differing_only_in_a_nested_point_are_not_equal():
    a = _multi_set_record()
    b = _multi_set_record()
    b.sets[1].points[0].needs_review = True

    assert a != b


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_parse_footage_rally_count_treats_blank_as_not_supplied(blank):
    assert parse_footage_rally_count(blank) is None


@pytest.mark.parametrize("value, expected", [("123", 123), (" 7 ", 7), (123, 123)])
def test_parse_footage_rally_count_accepts_whole_numbers(value, expected):
    assert parse_footage_rally_count(value) == expected


@pytest.mark.parametrize("bad", ["0", "-4", "12.5", "lots", 0, -1, True, 12.5, [3]])
def test_parse_footage_rally_count_rejects_anything_else(bad):
    with pytest.raises(ValueError, match="whole number of at least 1"):
        parse_footage_rally_count(bad)


def test_a_pending_json_written_before_the_rally_count_existed_still_loads():
    """Staged matches outlive the code that wrote them: a file with no
    footage_rally_count key must load, with the count simply unknown."""
    data = _multi_set_record().to_dict()
    del data["footage_rally_count"]

    assert MatchRecord.from_dict(data).footage_rally_count is None
