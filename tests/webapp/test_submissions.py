"""API tests for the phone companion's submissions surface (Phase 1).

Everything here goes through the real Flask app built by create_app() with
injected test config, exactly like tests/webapp/test_app.py - no route
logic is exercised in isolation, and no test touches the real Anthropic
API (the submissions path never constructs a client at all).
"""

from __future__ import annotations

import io

import pytest

from swingvision_import.review import load_pending, unresolved_flags
from webapp.app import create_app
from webapp.pairing import PairingStore

_METADATA = {
    "date": "2026-08-06",
    "opponent": "Alex",
    "result": "W",
    "location": "Riverside",
    "energy_rating": 4,
    "mental_rating": 3,
    "pros": "Served well",
    "cons": "Second serve",
    "notes": "Windy",
    "reviewer_note": "part 2 restarts mid-game",
}


def _mint_code(webapp_config) -> str:
    """Mints a pairing code the way the laptop's own UI will."""
    return PairingStore(
        webapp_config.pairing_path, code_ttl_seconds=webapp_config.pairing_code_ttl_seconds
    ).new_code()


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _create(client, token: str, *, key: str = "key-1", **overrides):
    body = dict(_METADATA)
    body.update(overrides)
    return client.post(
        "/api/submissions",
        json={"idempotency_key": key, "metadata": body},
        headers=_auth(token),
    )


def _put_chunk(client, token, submission_id, slot, data, start, total):
    end = start + len(data) - 1
    return client.put(
        f"/api/submissions/{submission_id}/files/{slot}",
        data=data,
        headers={**_auth(token), "Content-Range": f"bytes {start}-{end}/{total}"},
    )


def _upload_whole(client, token, submission_id, slot, payload, *, chunk_size=4096):
    """Pushes a whole file as consecutive chunks, like the PWA will."""
    total = len(payload)
    for start in range(0, total, chunk_size):
        response = _put_chunk(
            client, token, submission_id, slot, payload[start : start + chunk_size], start, total
        )
        assert response.status_code == 200, response.get_json()
    return response


@pytest.fixture
def token(client, webapp_config) -> str:
    response = client.post(
        "/api/pair", json={"code": _mint_code(webapp_config), "device_name": "iPhone"}
    )
    assert response.status_code == 200
    return response.get_json()["token"]


# --- Pairing ---------------------------------------------------------------


def test_pairing_exchanges_a_code_for_a_device_token(client, webapp_config):
    response = client.post(
        "/api/pair", json={"code": _mint_code(webapp_config), "device_name": "iPhone"}
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["token"]
    assert body["device_name"] == "iPhone"


def test_pairing_rejects_an_unknown_code(client):
    response = client.post("/api/pair", json={"code": "000000", "device_name": "iPhone"})

    assert response.status_code == 401
    assert "code" in response.get_json()["error"].lower()


def test_a_pairing_code_can_only_be_redeemed_once(client, webapp_config):
    code = _mint_code(webapp_config)
    assert client.post("/api/pair", json={"code": code}).status_code == 200

    second = client.post("/api/pair", json={"code": code})

    assert second.status_code == 401


def test_pairing_rejects_an_expired_code(client, webapp_config):
    store = PairingStore(webapp_config.pairing_path, code_ttl_seconds=-1)
    code = store.new_code()

    response = client.post("/api/pair", json={"code": code})

    assert response.status_code == 401


# --- Token auth ------------------------------------------------------------


def test_submissions_reject_an_unpaired_request(client):
    assert client.get("/api/submissions").status_code == 401
    assert client.post("/api/submissions", json={}).status_code == 401
    assert client.put("/api/submissions/abc/files/x.xlsx", data=b"x").status_code == 401
    assert client.post("/api/submissions/abc/complete").status_code == 401
    assert client.delete("/api/submissions/abc").status_code == 401


def test_submissions_reject_a_bogus_token(client):
    response = client.get("/api/submissions", headers=_auth("not-a-real-token"))

    assert response.status_code == 401


def test_a_revoked_token_stops_working(client, webapp_config, token):
    assert client.get("/api/submissions", headers=_auth(token)).status_code == 200
    store = PairingStore(webapp_config.pairing_path)
    assert store.revoke(token) is True

    assert client.get("/api/submissions", headers=_auth(token)).status_code == 401


# --- Create / list / validate ---------------------------------------------


def test_create_submission_lands_in_awaiting_files(client, token):
    response = _create(client, token)

    assert response.status_code == 201
    body = response.get_json()
    assert body["state"] == "awaiting_files"
    assert body["submission_id"]
    assert body["metadata"]["opponent"] == "Alex"
    assert body["metadata"]["reviewer_note"] == "part 2 restarts mid-game"


def test_create_submission_requires_valid_metadata(client, token):
    assert _create(client, token, date="06/08/2026").status_code == 400
    assert _create(client, token, opponent="").status_code == 400
    assert _create(client, token, result="draw").status_code == 400
    assert _create(client, token, energy_rating=9).status_code == 400
    assert _create(client, token, first_server_set1="nobody").status_code == 400


def test_create_submission_requires_an_idempotency_key(client, token):
    response = client.post("/api/submissions", json={"metadata": _METADATA}, headers=_auth(token))

    assert response.status_code == 400


def test_a_repeated_idempotency_key_returns_the_existing_submission(client, token):
    first = _create(client, token, key="same-key")
    assert first.status_code == 201

    second = _create(client, token, key="same-key", opponent="Someone Else")

    assert second.status_code == 200
    assert second.get_json()["submission_id"] == first.get_json()["submission_id"]
    # The original metadata wins - a retry must not rewrite what was accepted.
    assert second.get_json()["metadata"]["opponent"] == "Alex"
    assert len(client.get("/api/submissions", headers=_auth(token)).get_json()) == 1


def test_list_submissions_reports_state_and_per_slot_progress(client, token, xlsx_bytes):
    submission_id = _create(client, token).get_json()["submission_id"]
    _put_chunk(client, token, submission_id, "export.xlsx", xlsx_bytes[:10], 0, len(xlsx_bytes))

    listed = client.get("/api/submissions", headers=_auth(token)).get_json()

    assert len(listed) == 1
    assert listed[0]["state"] == "uploading"
    slot = listed[0]["files"][0]
    assert slot["slot"] == "export.xlsx"
    assert slot["received_bytes"] == 10
    assert slot["total_bytes"] == len(xlsx_bytes)
    assert slot["complete"] is False
    assert slot["next_offset"] == 10


# --- Chunked upload --------------------------------------------------------


def test_chunks_may_arrive_out_of_order(client, token):
    submission_id = _create(client, token).get_json()["submission_id"]
    payload = b"0123456789"

    _put_chunk(client, token, submission_id, "clip.mp4", payload[5:], 5, 10)
    _put_chunk(client, token, submission_id, "clip.mp4", payload[:5], 0, 10)

    slot = client.get("/api/submissions", headers=_auth(token)).get_json()[0]["files"][0]
    assert slot["complete"] is True
    assert slot["received_bytes"] == 10


def test_resending_an_identical_chunk_is_a_no_op(client, token):
    submission_id = _create(client, token).get_json()["submission_id"]

    _put_chunk(client, token, submission_id, "clip.mp4", b"abcde", 0, 10)
    repeat = _put_chunk(client, token, submission_id, "clip.mp4", b"abcde", 0, 10)

    assert repeat.status_code == 200
    assert repeat.get_json()["received_bytes"] == 5  # not 10
    assert repeat.get_json()["duplicate"] is True


def test_an_interrupted_upload_resumes_from_the_reported_offset(client, token, xlsx_bytes):
    """The phone's recovery path: it lost the connection part-way, asks the
    server what it already has, and sends only the rest."""
    submission_id = _create(client, token).get_json()["submission_id"]
    total = len(xlsx_bytes)
    _put_chunk(client, token, submission_id, "export.xlsx", xlsx_bytes[:1000], 0, total)

    slot = client.get("/api/submissions", headers=_auth(token)).get_json()[0]["files"][0]
    resume_at = slot["next_offset"]
    assert resume_at == 1000
    rest = xlsx_bytes[resume_at:]
    _put_chunk(client, token, submission_id, "export.xlsx", rest, resume_at, total)

    done = client.get("/api/submissions", headers=_auth(token)).get_json()[0]["files"][0]
    assert done["complete"] is True
    assert done["received_bytes"] == total


def test_a_malformed_content_range_is_rejected(client, token):
    submission_id = _create(client, token).get_json()["submission_id"]

    missing = client.put(
        f"/api/submissions/{submission_id}/files/clip.mp4", data=b"abc", headers=_auth(token)
    )
    assert missing.status_code == 400

    nonsense = client.put(
        f"/api/submissions/{submission_id}/files/clip.mp4",
        data=b"abc",
        headers={**_auth(token), "Content-Range": "bytes lots"},
    )
    assert nonsense.status_code == 400


def test_a_chunk_whose_length_disagrees_with_its_range_is_rejected(client, token):
    submission_id = _create(client, token).get_json()["submission_id"]

    response = client.put(
        f"/api/submissions/{submission_id}/files/clip.mp4",
        data=b"abc",
        headers={**_auth(token), "Content-Range": "bytes 0-9/10"},
    )

    assert response.status_code == 400


def test_a_chunk_changing_a_slots_total_size_is_rejected(client, token):
    submission_id = _create(client, token).get_json()["submission_id"]
    _put_chunk(client, token, submission_id, "clip.mp4", b"abcde", 0, 10)

    response = _put_chunk(client, token, submission_id, "clip.mp4", b"fghij", 5, 99)

    assert response.status_code == 409


def test_a_chunk_for_an_unknown_submission_404s(client, token):
    response = _put_chunk(client, token, "0" * 32, "clip.mp4", b"abc", 0, 3)

    assert response.status_code == 404


def test_a_hostile_slot_name_cannot_escape_the_submissions_directory(
    client, token, webapp_config, tmp_path
):
    submission_id = _create(client, token).get_json()["submission_id"]

    # A Windows-style traversal survives URL routing as one path segment,
    # so it reaches the handler and has to be refused there.
    backslashed = client.put(
        f"/api/submissions/{submission_id}/files/..%5C..%5Cevil.xlsx",
        data=b"abc",
        headers={**_auth(token), "Content-Range": "bytes 0-2/3"},
    )
    assert backslashed.status_code == 400

    # A POSIX-style one is collapsed by URL normalisation before it ever
    # matches this route - so it can't reach the handler, and doesn't.
    slashed = client.put(
        f"/api/submissions/{submission_id}/files/..%2F..%2Fevil.xlsx",
        data=b"abc",
        headers={**_auth(token), "Content-Range": "bytes 0-2/3"},
    )
    assert slashed.status_code != 200

    assert not (tmp_path / "evil.xlsx").exists()
    assert not (webapp_config.submissions_dir / "evil.xlsx").exists()
    assert [p.name for p in (webapp_config.submissions_dir / submission_id).iterdir()] == [
        "submission.json"
    ]


# --- Complete --------------------------------------------------------------


def test_complete_refuses_while_a_slot_is_incomplete(client, token, xlsx_bytes):
    submission_id = _create(client, token).get_json()["submission_id"]
    _put_chunk(client, token, submission_id, "export.xlsx", xlsx_bytes[:50], 0, len(xlsx_bytes))

    response = client.post(f"/api/submissions/{submission_id}/complete", headers=_auth(token))

    assert response.status_code == 409
    assert "export.xlsx" in response.get_json()["error"]


def test_complete_refuses_a_submission_with_no_export(client, token):
    submission_id = _create(client, token).get_json()["submission_id"]
    _upload_whole(client, token, submission_id, "clip.mp4", b"fake video bytes")

    response = client.post(f"/api/submissions/{submission_id}/complete", headers=_auth(token))

    assert response.status_code == 409
    assert ".xlsx" in response.get_json()["error"]


def test_the_full_lifecycle_stages_a_pending_json(
    client, token, xlsx_bytes, import_config, webapp_config
):
    submission_id = _create(client, token).get_json()["submission_id"]
    _upload_whole(client, token, submission_id, "export.xlsx", xlsx_bytes, chunk_size=2048)
    _upload_whole(client, token, submission_id, "clip.mp4", b"fake video bytes")

    response = client.post(f"/api/submissions/{submission_id}/complete", headers=_auth(token))

    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["state"] == "staged"
    assert body["json_filename"] == "2026-08-06_Alex.json"
    assert body["flags"]

    json_path = import_config.pending_dir / "2026-08-06_Alex.json"
    assert json_path.exists()
    record = load_pending(json_path)
    assert record.location == "Riverside"
    assert record.energy_rating == 4
    assert record.reviewer_note == "part 2 restarts mid-game"
    # The same archiving and media rules the browser upload path uses.
    assert sorted(p.name for p in webapp_config.uploads_dir.iterdir()) == [
        "2026-08-06_Alex_exports.zip"
    ]
    clip = webapp_config.media_dir / "2026-08-06_Alex" / "clip.mp4"
    assert clip.read_bytes() == b"fake video bytes"
    # The staging area is emptied once the files have moved.
    assert not any((webapp_config.submissions_dir / submission_id).glob("*.xlsx"))


def test_complete_does_not_weaken_the_review_gate(client, token, xlsx_bytes):
    submission_id = _create(client, token).get_json()["submission_id"]
    _upload_whole(client, token, submission_id, "export.xlsx", xlsx_bytes)
    completed = client.post(
        f"/api/submissions/{submission_id}/complete", headers=_auth(token)
    ).get_json()

    finalize = client.post(f"/api/pending/{completed['json_filename']}/finalize")

    assert finalize.status_code == 409
    assert finalize.get_json()["flags"]


def test_completing_twice_returns_the_same_staged_submission(client, token, xlsx_bytes):
    submission_id = _create(client, token).get_json()["submission_id"]
    _upload_whole(client, token, submission_id, "export.xlsx", xlsx_bytes)

    first = client.post(f"/api/submissions/{submission_id}/complete", headers=_auth(token))
    second = client.post(f"/api/submissions/{submission_id}/complete", headers=_auth(token))

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.get_json()["json_filename"] == first.get_json()["json_filename"]


def test_a_multi_part_submission_preserves_part_order(
    client, token, xlsx_bytes, import_config, webapp_config
):
    """Two exports and two clips - a real match that was recorded in two
    files. Order and pairing come from the filenames, same rule as the
    browser upload path, and part2 must sort before part10."""
    submission_id = _create(client, token).get_json()["submission_id"]
    for slot in ("part10.xlsx", "part2.xlsx"):
        _upload_whole(client, token, submission_id, slot, xlsx_bytes, chunk_size=4096)
    for slot in ("part10.mp4", "part2.mp4"):
        _upload_whole(client, token, submission_id, slot, b"clip " + slot.encode())

    response = client.post(f"/api/submissions/{submission_id}/complete", headers=_auth(token))

    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert [part["export"] for part in body["parts"]] == ["part2.xlsx", "part10.xlsx"]
    assert body["parts"][0]["videos"] == ["part2.mp4"]
    assert body["parts"][1]["videos"] == ["part10.mp4"]

    record = load_pending(import_config.pending_dir / "2026-08-06_Alex.json")
    assert record.source_files is not None and len(record.source_files) == 2

    saved = sorted(p.name for p in (webapp_config.media_dir / "2026-08-06_Alex").iterdir())
    assert saved == ["part01__part2.mp4", "part02__part10.mp4"]


def test_a_staged_submission_refuses_further_chunks(client, token, xlsx_bytes):
    """Its files have moved into data/uploads/ and someone may be reviewing
    the pending JSON - a late chunk must not quietly rewrite either."""
    submission_id = _create(client, token).get_json()["submission_id"]
    _upload_whole(client, token, submission_id, "export.xlsx", xlsx_bytes)
    assert (
        client.post(f"/api/submissions/{submission_id}/complete", headers=_auth(token)).status_code
        == 200
    )

    late = _put_chunk(client, token, submission_id, "extra.mp4", b"abc", 0, 3)

    assert late.status_code == 409


def test_complete_on_an_unknown_submission_404s(client, token):
    assert (
        client.post(f"/api/submissions/{'0' * 32}/complete", headers=_auth(token)).status_code
        == 404
    )


# --- Delete ----------------------------------------------------------------


def test_delete_removes_the_submission_and_its_chunks(client, token, webapp_config):
    submission_id = _create(client, token).get_json()["submission_id"]
    _upload_whole(client, token, submission_id, "clip.mp4", b"abc")

    response = client.delete(f"/api/submissions/{submission_id}", headers=_auth(token))

    assert response.status_code == 200
    assert client.get("/api/submissions", headers=_auth(token)).get_json() == []
    assert not (webapp_config.submissions_dir / submission_id).exists()


def test_deleting_an_unknown_submission_404s(client, token):
    assert client.delete(f"/api/submissions/{'0' * 32}", headers=_auth(token)).status_code == 404


# --- Persistence -----------------------------------------------------------


def test_a_submission_and_its_token_survive_a_restart(
    client, token, xlsx_bytes, import_config, webapp_config
):
    """jobs.py keeps its state in memory because a restart genuinely ends a
    footage check. A half-uploaded 11 GB recording is the opposite: losing
    it to a restart would mean re-sending everything."""
    submission_id = _create(client, token).get_json()["submission_id"]
    _put_chunk(client, token, submission_id, "export.xlsx", xlsx_bytes[:1000], 0, len(xlsx_bytes))

    restarted = create_app(import_config, webapp_config).test_client()

    listed = restarted.get("/api/submissions", headers=_auth(token)).get_json()
    assert [s["submission_id"] for s in listed] == [submission_id]
    assert listed[0]["files"][0]["next_offset"] == 1000

    _put_chunk(
        restarted, token, submission_id, "export.xlsx", xlsx_bytes[1000:], 1000, len(xlsx_bytes)
    )
    completed = restarted.post(f"/api/submissions/{submission_id}/complete", headers=_auth(token))
    assert completed.status_code == 200
    record = load_pending(import_config.pending_dir / "2026-08-06_Alex.json")
    assert unresolved_flags(record)


# --- Failure reporting -----------------------------------------------------


def test_an_unparseable_export_marks_the_submission_failed(client, token):
    submission_id = _create(client, token).get_json()["submission_id"]
    _upload_whole(client, token, submission_id, "export.xlsx", b"not really a spreadsheet")

    response = client.post(f"/api/submissions/{submission_id}/complete", headers=_auth(token))

    assert response.status_code == 400
    listed = client.get("/api/submissions", headers=_auth(token)).get_json()
    assert listed[0]["state"] == "failed"
    assert listed[0]["error"]


def test_the_browser_upload_path_is_unchanged_by_submissions(client, xlsx_bytes):
    """The existing intake form carries no token and must keep working."""
    response = client.post(
        "/api/import",
        data={
            "xlsx_file": (io.BytesIO(xlsx_bytes), "export.xlsx"),
            "date": "2026-08-06",
            "opponent": "Alex",
            "result": "W",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
