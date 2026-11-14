"""Unit tests for the submission store, pairing store, and the --lan flag.

These cover the pieces the API tests exercise end to end: chunk range
bookkeeping, metadata validation, idempotency, on-disk persistence, and
token lifetime - none of which need a Flask app to be checked.
"""

from __future__ import annotations

import pytest

from webapp.app import resolve_host
from webapp.pairing import PairingStore
from webapp.submissions import (
    SubmissionError,
    SubmissionStore,
    merge_range,
    parse_content_range,
    validate_metadata,
)

_METADATA = {"date": "2026-08-06", "opponent": "Alex", "result": "W"}


@pytest.fixture
def store(tmp_path) -> SubmissionStore:
    return SubmissionStore(tmp_path / "submissions")


# --- Content-Range parsing -------------------------------------------------


def test_parse_content_range_reads_start_end_total():
    assert parse_content_range("bytes 0-1023/4096") == (0, 1023, 4096)


@pytest.mark.parametrize(
    "header",
    [None, "", "bytes */4096", "bytes 0-1023", "0-1023/4096", "bytes 10-5/4096", "bytes 0-9/5"],
)
def test_parse_content_range_rejects_anything_else(header):
    with pytest.raises(SubmissionError):
        parse_content_range(header)


# --- Range bookkeeping ----------------------------------------------------


def test_merge_range_joins_adjacent_and_overlapping_spans():
    assert merge_range([], (0, 9)) == [(0, 9)]
    assert merge_range([(0, 9)], (10, 19)) == [(0, 19)]
    assert merge_range([(0, 9)], (5, 14)) == [(0, 14)]
    assert merge_range([(0, 9), (20, 29)], (10, 19)) == [(0, 29)]


def test_merge_range_keeps_a_gap_as_two_spans():
    assert merge_range([(0, 9)], (20, 29)) == [(0, 9), (20, 29)]


def test_out_of_order_chunks_assemble_the_original_bytes(store):
    submission, _ = store.create(_METADATA, "key-1")
    payload = bytes(range(256))

    store.write_chunk(submission.submission_id, "clip.mp4", 128, 255, 256, payload[128:])
    store.write_chunk(submission.submission_id, "clip.mp4", 0, 127, 256, payload[:128])

    slot = store.get(submission.submission_id).slots["clip.mp4"]
    assert slot.complete is True
    assert store.slot_path(submission.submission_id, "clip.mp4").read_bytes() == payload


def test_a_repeated_chunk_is_reported_as_a_duplicate(store):
    submission, _ = store.create(_METADATA, "key-1")

    first = store.write_chunk(submission.submission_id, "clip.mp4", 0, 4, 10, b"abcde")
    second = store.write_chunk(submission.submission_id, "clip.mp4", 0, 4, 10, b"abcde")

    assert first.duplicate is False
    assert second.duplicate is True
    assert second.slot.received_bytes == 5


def test_next_offset_points_at_the_first_missing_byte(store):
    submission, _ = store.create(_METADATA, "key-1")
    store.write_chunk(submission.submission_id, "clip.mp4", 0, 9, 100, b"0" * 10)

    slot = store.get(submission.submission_id).slots["clip.mp4"]

    assert slot.next_offset == 10
    assert slot.complete is False


def test_next_offset_skips_back_to_a_hole_left_by_out_of_order_chunks(store):
    submission, _ = store.create(_METADATA, "key-1")
    store.write_chunk(submission.submission_id, "clip.mp4", 50, 99, 100, b"0" * 50)

    slot = store.get(submission.submission_id).slots["clip.mp4"]

    assert slot.next_offset == 0


def test_a_chunk_past_the_declared_total_is_refused(store):
    submission, _ = store.create(_METADATA, "key-1")

    with pytest.raises(SubmissionError):
        store.write_chunk(submission.submission_id, "clip.mp4", 95, 104, 100, b"0" * 10)


def test_changing_a_slots_total_is_refused(store):
    submission, _ = store.create(_METADATA, "key-1")
    store.write_chunk(submission.submission_id, "clip.mp4", 0, 4, 10, b"abcde")

    with pytest.raises(SubmissionError):
        store.write_chunk(submission.submission_id, "clip.mp4", 5, 9, 99, b"fghij")


def test_a_hostile_slot_name_is_refused(store):
    submission, _ = store.create(_METADATA, "key-1")

    for slot in ("../evil.xlsx", "..", "", "/etc/passwd"):
        with pytest.raises(SubmissionError):
            store.write_chunk(submission.submission_id, slot, 0, 2, 3, b"abc")


# --- Metadata validation --------------------------------------------------


def test_validate_metadata_keeps_only_known_fields():
    cleaned = validate_metadata({**_METADATA, "surprise": "ignored", "notes": "Windy"})

    assert cleaned["notes"] == "Windy"
    assert "surprise" not in cleaned


def test_validate_metadata_normalises_the_result():
    assert validate_metadata({**_METADATA, "result": "l"})["result"] == "L"


@pytest.mark.parametrize(
    "bad",
    [
        {"date": ""},
        {"date": "06/08/2026"},
        {"date": "2026-13-40"},
        {"opponent": "   "},
        {"result": "draw"},
        {"energy_rating": 0},
        {"energy_rating": 6},
        {"energy_rating": "high"},
        {"mental_rating": 9},
        {"first_server_set1": "nobody"},
        {"first_server_set4": "me"},
    ],
)
def test_validate_metadata_rejects_bad_values(bad):
    with pytest.raises(SubmissionError):
        validate_metadata({**_METADATA, **bad})


def test_validate_metadata_drops_blank_optional_text():
    cleaned = validate_metadata({**_METADATA, "notes": "", "location": "   "})

    assert "notes" not in cleaned
    assert "location" not in cleaned


# --- Idempotency and persistence ------------------------------------------


def test_the_same_idempotency_key_returns_the_first_submission(store):
    first, created_first = store.create(_METADATA, "key-1")
    second, created_second = store.create({**_METADATA, "opponent": "Other"}, "key-1")

    assert created_first is True
    assert created_second is False
    assert second.submission_id == first.submission_id
    assert second.metadata["opponent"] == "Alex"
    assert len(store.list()) == 1


def test_different_keys_create_different_submissions(store):
    first, _ = store.create(_METADATA, "key-1")
    second, _ = store.create(_METADATA, "key-2")

    assert first.submission_id != second.submission_id
    assert len(store.list()) == 2


def test_state_is_read_back_from_disk_by_a_fresh_store(tmp_path):
    root = tmp_path / "submissions"
    submission, _ = SubmissionStore(root).create(_METADATA, "key-1")
    SubmissionStore(root).write_chunk(submission.submission_id, "clip.mp4", 0, 4, 10, b"abcde")

    reloaded = SubmissionStore(root).get(submission.submission_id)

    assert reloaded.state == "uploading"
    assert reloaded.slots["clip.mp4"].received_bytes == 5
    assert reloaded.idempotency_key == "key-1"


def test_a_half_written_submission_file_is_skipped_not_fatal(store, tmp_path):
    store.create(_METADATA, "key-1")
    broken = store.root / "deadbeef"
    broken.mkdir()
    (broken / "submission.json").write_text("{not json", encoding="utf-8")

    assert len(store.list()) == 1


def test_delete_removes_the_whole_submission_directory(store):
    submission, _ = store.create(_METADATA, "key-1")
    store.write_chunk(submission.submission_id, "clip.mp4", 0, 2, 3, b"abc")

    assert store.delete(submission.submission_id) is True

    assert store.get(submission.submission_id) is None
    assert store.delete(submission.submission_id) is False


def test_an_unknown_submission_id_is_none_not_an_error(store):
    assert store.get("0" * 32) is None
    assert store.get("../../etc") is None


# --- Pairing --------------------------------------------------------------


def test_a_code_is_redeemed_once_for_a_token(tmp_path):
    store = PairingStore(tmp_path / "pairing.json")
    code = store.new_code()

    device = store.redeem(code, "iPhone")

    assert store.is_paired(device.token) is True
    assert store.redeem(code, "iPhone") is None


def test_an_expired_code_is_not_redeemable(tmp_path):
    store = PairingStore(tmp_path / "pairing.json", code_ttl_seconds=-1)

    assert store.redeem(store.new_code(), "iPhone") is None


def test_a_token_survives_a_fresh_store(tmp_path):
    path = tmp_path / "pairing.json"
    device = PairingStore(path).redeem(PairingStore(path).new_code(), "iPhone")

    assert PairingStore(path).is_paired(device.token) is True


def test_revoking_a_token_unpairs_that_device(tmp_path):
    store = PairingStore(tmp_path / "pairing.json")
    device = store.redeem(store.new_code(), "iPhone")

    assert store.revoke(device.token) is True
    assert store.is_paired(device.token) is False
    assert store.revoke(device.token) is False


def test_an_unknown_token_is_never_paired(tmp_path):
    store = PairingStore(tmp_path / "pairing.json")
    store.redeem(store.new_code(), "iPhone")

    assert store.is_paired("") is False
    assert store.is_paired("nope") is False


def test_devices_lists_what_is_paired(tmp_path):
    store = PairingStore(tmp_path / "pairing.json")
    store.redeem(store.new_code(), "iPhone")

    assert [d.name for d in store.devices()] == ["iPhone"]


# --- The --lan flag -------------------------------------------------------


def test_the_default_host_is_still_localhost_only():
    assert resolve_host([]) == "127.0.0.1"


def test_lan_opts_into_binding_every_interface():
    assert resolve_host(["--lan"]) == "0.0.0.0"
