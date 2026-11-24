"""Tests for provenance tags: the measured/interpolated/inferred/user split."""

import pytest

from tracking import provenance
from tracking.provenance import Provenance


def test_the_four_tags_serialise_as_plain_strings():
    assert [member.value for member in Provenance] == [
        "measured",
        "interpolated",
        "inferred",
        "user_entered",
    ]
    assert Provenance.MEASURED == "measured"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("measured", Provenance.MEASURED),
        (" User-Entered ", Provenance.USER_ENTERED),
        ("user entered", Provenance.USER_ENTERED),
        (Provenance.INFERRED, Provenance.INFERRED),
    ],
)
def test_parse_reads_stored_forms(text, expected):
    assert Provenance.parse(text) is expected


def test_parse_rejects_an_unknown_tag_and_lists_the_real_ones():
    with pytest.raises(ValueError, match="measured, interpolated"):
        Provenance.parse("guessed")


def test_only_detections_and_people_count_as_observed():
    assert Provenance.MEASURED.is_observed
    assert Provenance.USER_ENTERED.is_observed
    assert not Provenance.INTERPOLATED.is_observed
    assert not Provenance.INFERRED.is_observed


def test_a_derived_value_takes_its_least_trustworthy_input():
    assert provenance.weakest([Provenance.MEASURED, Provenance.INTERPOLATED]) is (
        Provenance.INTERPOLATED
    )
    assert provenance.weakest([Provenance.INFERRED, Provenance.USER_ENTERED]) is Provenance.INFERRED
    assert provenance.weakest([Provenance.USER_ENTERED, Provenance.MEASURED]) is Provenance.MEASURED


def test_weakest_of_nothing_is_an_error():
    with pytest.raises(ValueError):
        provenance.weakest([])


def test_trust_runs_from_person_to_rule():
    ranks = [provenance.trust_rank(member) for member in Provenance]
    assert ranks == [1, 2, 3, 0]


def test_counts_state_zero_rather_than_omit_it():
    counts = provenance.provenance_counts([Provenance.MEASURED, Provenance.MEASURED])
    assert counts == {"measured": 2, "interpolated": 0, "inferred": 0, "user_entered": 0}


def test_helpers_tag_values():
    assert provenance.measured(1.5, 0.9, "ball-v1").provenance is Provenance.MEASURED
    assert provenance.interpolated(1.5).provenance is Provenance.INTERPOLATED
    assert provenance.inferred("serve", 0.4).provenance is Provenance.INFERRED
    entered = provenance.user_entered("Forehand", "review modal")
    assert entered.provenance is Provenance.USER_ENTERED
    assert entered.confidence is None


def test_a_tagged_value_serialises_to_json_ready_types():
    tagged = provenance.measured([1.0, 2.0], 0.75, "court-v1")
    assert tagged.to_dict() == {
        "value": [1.0, 2.0],
        "provenance": "measured",
        "confidence": 0.75,
        "source": "court-v1",
    }


def test_confidence_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError):
        provenance.measured(1.0, 1.2)
