from __future__ import annotations

import io
import json
import zipfile

from swingvision_import import footage
from swingvision_import.review import load_pending, unresolved_flags
from tests.ai.conftest import FakeMessage, FakeTextBlock


def _upload(client, xlsx_bytes: bytes, *, filename: str = "export.xlsx", **extra_form):
    data = {
        "xlsx_file": (io.BytesIO(xlsx_bytes), filename),
        "date": "2026-08-06",
        "opponent": "Alex",
        "result": "W",
    }
    data.update(extra_form)
    return client.post("/api/import", data=data, content_type="multipart/form-data")


class _FakeSuggestionMessages:
    def create(self, **kwargs):
        payload = json.dumps(
            {"point_end_type": "forced_error", "reasoning": "Deep shot.", "confidence": "medium"}
        )
        return FakeMessage(content=[FakeTextBlock(text=payload)])


class _FakeSuggestionClient:
    def __init__(self):
        self.messages = _FakeSuggestionMessages()


class _FakeCoachMessages:
    def create(self, **kwargs):
        payload = json.dumps({"feedback": "Great serving today - keep it up."})
        return FakeMessage(content=[FakeTextBlock(text=payload)])


class _FakeCoachClient:
    def __init__(self):
        self.messages = _FakeCoachMessages()


def test_report_renders_a_finalized_match(client, finalized_match_id):
    response = client.get(f"/report/{finalized_match_id}")

    assert response.status_code == 200
    assert b"Alex" in response.data


def test_report_404s_for_an_unknown_match(client):
    response = client.get("/report/999")

    assert response.status_code == 404


def test_report_never_constructs_a_real_client(client, finalized_match_id, monkeypatch):
    def _fail_if_called():
        raise AssertionError("view_report must never construct a real Anthropic client")

    monkeypatch.setattr("scripts.client.get_anthropic_client", _fail_if_called)

    response = client.get(f"/report/{finalized_match_id}")

    assert response.status_code == 200


def test_import_stages_pending_json_and_returns_flags_as_json(
    client, xlsx_bytes, import_config
):
    response = _upload(client, xlsx_bytes)

    assert response.status_code == 200
    body = response.get_json()
    assert body["date"] == "2026-08-06"
    assert body["opponent"] == "Alex"
    assert body["staged_label"] == 1
    assert len(body["flags"]) > 0
    assert body["import_notes"]  # the fixture's gap point produces one

    json_path = import_config.pending_dir / "2026-08-06_Alex.json"
    assert json_path.exists()
    record = load_pending(json_path)
    assert len(unresolved_flags(record)) == len(body["flags"])


def test_import_missing_xlsx_returns_400_with_error(client):
    response = client.post(
        "/api/import",
        data={"date": "2026-08-06", "opponent": "Alex", "result": "W"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "at least one SwingVision" in response.get_json()["error"]


def test_import_saves_uploaded_video_into_a_per_match_directory(
    client, xlsx_bytes, webapp_config
):
    response = _upload(
        client, xlsx_bytes, video_files=(io.BytesIO(b"fake video bytes"), "clip.mp4")
    )

    assert response.status_code == 200
    saved_video = webapp_config.media_dir / "2026-08-06_Alex" / "clip.mp4"
    assert saved_video.exists()
    assert saved_video.read_bytes() == b"fake video bytes"


def test_hostile_xlsx_filename_does_not_escape_uploads_dir(
    client, xlsx_bytes, webapp_config, tmp_path
):
    response = _upload(client, xlsx_bytes, filename="../../../evil.xlsx")

    assert response.status_code == 200
    assert not (tmp_path / "evil.xlsx").exists()
    assert list(webapp_config.uploads_dir.iterdir())  # something safe landed inside


def test_hostile_video_filename_does_not_escape_media_dir(client, xlsx_bytes, tmp_path):
    response = _upload(client, xlsx_bytes, video_files=(io.BytesIO(b"x"), "../../../evil.mp4"))

    assert response.status_code == 200
    assert not (tmp_path / "evil.mp4").exists()


def test_ingest_first_server_and_tracked_identity_feed_quality_check(client, xlsx_bytes):
    response = _upload(
        client,
        xlsx_bytes,
        first_server_set1="opponent",  # wrong on purpose - point 1 was really host serving
        tracked_identity="Someone Else",
    )

    assert response.status_code == 200
    notes = " | ".join(response.get_json()["import_notes"])
    assert "reversed" in notes
    assert "Someone Else" in notes


def test_pending_detail_rejects_path_traversal_in_json_filename(client):
    response = client.get("/api/pending/../../../../etc/passwd")

    assert response.status_code == 404


def test_pending_detail_returns_404_for_a_nonexistent_pending_file(client):
    response = client.get("/api/pending/does_not_exist.json")

    assert response.status_code == 404


def test_pending_detail_lists_only_flagged_points_with_shot_context(client, xlsx_bytes):
    _upload(client, xlsx_bytes)

    response = client.get("/api/pending/2026-08-06_Alex.json")

    assert response.status_code == 200
    body = response.get_json()
    assert body["date"] == "2026-08-06"
    assert len(body["points"]) > 0
    assert all(p["shots"] for p in body["points"])  # reconstructed points carry shot context


def test_suggest_runs_claude_assist_and_updates_pending_detail(client, xlsx_bytes, monkeypatch):
    _upload(client, xlsx_bytes)
    monkeypatch.setattr("webapp.api.get_anthropic_client", lambda: _FakeSuggestionClient())

    response = client.post("/api/pending/2026-08-06_Alex.json/suggest")

    assert response.status_code == 200
    body = response.get_json()
    assert all(p["ai_suggested_point_end_type"] == "forced_error" for p in body["points"])


def test_confirm_point_then_finalize_succeeds(client, xlsx_bytes, import_config):
    _upload(client, xlsx_bytes)
    record = load_pending(import_config.pending_dir / "2026-08-06_Alex.json")
    flagged_points = [p for s in record.sets for p in s.points if p.needs_review]

    for point in flagged_points:
        response = client.post(
            "/api/pending/2026-08-06_Alex.json/confirm-point",
            json={
                "set_number": record.sets[0].set_number,
                "game_number": point.game_number,
                "point_number": point.point_number,
                "point_end_type": "winner",
                "point_won": True,
                "net_approach": False,
            },
        )
        assert response.status_code == 200

    assert response.get_json()["flags_remaining"] == 0

    finalize_response = client.post("/api/pending/2026-08-06_Alex.json/finalize")
    assert finalize_response.status_code == 200
    assert "match_id" in finalize_response.get_json()


def test_finalize_refuses_while_flags_remain(client, xlsx_bytes):
    _upload(client, xlsx_bytes)

    response = client.post("/api/pending/2026-08-06_Alex.json/finalize")

    assert response.status_code == 409
    assert response.get_json()["flags"]


def test_confirm_point_rejects_an_inconsistent_pair(client, xlsx_bytes, import_config):
    _upload(client, xlsx_bytes)
    record = load_pending(import_config.pending_dir / "2026-08-06_Alex.json")
    target = record.sets[0].points[0]

    response = client.post(
        "/api/pending/2026-08-06_Alex.json/confirm-point",
        json={
            "set_number": record.sets[0].set_number,
            "game_number": target.game_number,
            "point_number": target.point_number,
            "point_end_type": "ace",
            "point_won": False,
            "net_approach": False,
        },
    )

    assert response.status_code == 400


def test_overview_reflects_a_finalized_match(client, finalized_match_id):
    response = client.get("/api/overview")

    assert response.status_code == 200
    body = response.get_json()
    assert body["total_matches"] == 1
    assert body["wins"] == 1


def test_overview_zeroed_state_before_any_match_is_finalized(client):
    response = client.get("/api/overview")

    assert response.status_code == 200
    assert response.get_json()["total_matches"] == 0


def test_matches_lists_finalized_matches(client, finalized_match_id):
    response = client.get("/api/matches")

    assert response.status_code == 200
    body = response.get_json()
    assert len(body) == 1
    assert body[0]["match_id"] == finalized_match_id


def test_match_detail_404s_for_an_unknown_match(client):
    response = client.get("/api/matches/999")

    assert response.status_code == 404


def test_update_journal_persists_pros_cons_notes(client, finalized_match_id):
    response = client.put(
        f"/api/matches/{finalized_match_id}/journal",
        json={"pros": "Better footwork", "cons": "Second serve", "notes": "Windy day"},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["self_assessment"]["pros"] == "Better footwork"
    assert body["self_assessment"]["notes"] == "Windy day"


def test_coach_generates_feedback_from_the_journal_entry(
    client, finalized_match_id, monkeypatch
):
    monkeypatch.setattr("webapp.api.get_anthropic_client", lambda: _FakeCoachClient())

    response = client.post(
        f"/api/matches/{finalized_match_id}/coach",
        json={"journal_text": "Served big today"},
    )

    assert response.status_code == 200
    assert response.get_json()["feedback"] == "Great serving today - keep it up."


def test_shot_embeddings_returns_points_for_a_finalized_reconstructed_match(
    client, xlsx_bytes, import_config
):
    _upload(client, xlsx_bytes)
    record = load_pending(import_config.pending_dir / "2026-08-06_Alex.json")
    for point in [p for s in record.sets for p in s.points if p.needs_review]:
        client.post(
            "/api/pending/2026-08-06_Alex.json/confirm-point",
            json={
                "set_number": record.sets[0].set_number,
                "game_number": point.game_number,
                "point_number": point.point_number,
                "point_end_type": "winner",
                "point_won": True,
                "net_approach": False,
            },
        )
    match_id = client.post("/api/pending/2026-08-06_Alex.json/finalize").get_json()["match_id"]

    response = client.get("/api/shots/embeddings")

    assert response.status_code == 200
    points = response.get_json()
    assert len(points) > 0
    assert all(p["match_id"] == match_id for p in points)
    assert all("x" in p and "y" in p and "z" in p for p in points)


def test_shot_embeddings_empty_before_any_match_is_finalized(client):
    response = client.get("/api/shots/embeddings")

    assert response.status_code == 200
    assert response.get_json() == []


def test_media_returns_empty_list_when_no_video_was_uploaded(client, finalized_match_id):
    response = client.get(f"/api/matches/{finalized_match_id}/media")

    assert response.status_code == 200
    assert response.get_json()["videos"] == []


def test_media_lists_and_serves_an_uploaded_video(client, xlsx_bytes, import_config):
    _upload(client, xlsx_bytes, video_files=(io.BytesIO(b"fake video bytes"), "clip.mp4"))
    record = load_pending(import_config.pending_dir / "2026-08-06_Alex.json")
    for point in [p for s in record.sets for p in s.points if p.needs_review]:
        client.post(
            "/api/pending/2026-08-06_Alex.json/confirm-point",
            json={
                "set_number": record.sets[0].set_number,
                "game_number": point.game_number,
                "point_number": point.point_number,
                "point_end_type": "winner",
                "point_won": True,
                "net_approach": False,
            },
        )
    match_id = client.post("/api/pending/2026-08-06_Alex.json/finalize").get_json()["match_id"]

    response = client.get(f"/api/matches/{match_id}/media")

    assert response.status_code == 200
    videos = response.get_json()["videos"]
    assert videos == ["/media/2026-08-06_Alex/clip.mp4"]

    video_response = client.get(videos[0])
    assert video_response.status_code == 200
    assert video_response.data == b"fake video bytes"


def _upload_parts(client, xlsx_bytes: bytes, *, exports: list[str], videos: list[str] = ()):
    """Posts a multi-part submission: several exports under the xlsx_files
    field, plus footage whose filenames are what pairs it to them."""
    data = {
        "xlsx_files": [(io.BytesIO(xlsx_bytes), name) for name in exports],
        "date": "2026-08-06",
        "opponent": "Alex",
        "result": "W",
    }
    if videos:
        data["video_files"] = [(io.BytesIO(b"clip bytes"), name) for name in videos]
    return client.post("/api/import", data=data, content_type="multipart/form-data")


def test_multiple_exports_are_merged_into_one_match(client, xlsx_bytes, import_config):
    response = _upload_parts(client, xlsx_bytes, exports=["set1.xlsx", "set2.xlsx"])

    assert response.status_code == 200
    body = response.get_json()
    assert [part["index"] for part in body["parts"]] == [1, 2]
    assert [part["export"] for part in body["parts"]] == ["set1.xlsx", "set2.xlsx"]

    record = load_pending(import_config.pending_dir / "2026-08-06_Alex.json")
    # ingest_multi_part records every source file; ingest() records one.
    assert record.source_files is not None
    assert len(record.source_files) == 2


def test_parts_are_ordered_naturally_not_lexicographically(client, xlsx_bytes):
    response = _upload_parts(client, xlsx_bytes, exports=["part10.xlsx", "part2.xlsx"])

    assert response.status_code == 200
    assert [p["export"] for p in response.get_json()["parts"]] == ["part2.xlsx", "part10.xlsx"]


def test_footage_is_paired_to_the_export_sharing_its_filename(
    client, xlsx_bytes, webapp_config
):
    response = _upload_parts(
        client,
        xlsx_bytes,
        exports=["set1.xlsx", "set2.xlsx"],
        videos=["set2.mp4", "set1.mp4"],
    )

    assert response.status_code == 200
    parts = response.get_json()["parts"]
    assert parts[0]["videos"] == ["set1.mp4"]
    assert parts[1]["videos"] == ["set2.mp4"]

    # Saved names carry the part order, so the media listing's plain
    # sorted() plays them back in the order they were recorded.
    saved = sorted(p.name for p in (webapp_config.media_dir / "2026-08-06_Alex").iterdir())
    assert saved == ["part01__set1.mp4", "part02__set2.mp4"]


def test_footage_matching_no_export_is_kept_and_reported(client, xlsx_bytes, webapp_config):
    response = _upload_parts(
        client, xlsx_bytes, exports=["set1.xlsx", "set2.xlsx"], videos=["set1.mp4", "random.mp4"]
    )

    assert response.status_code == 200
    body = response.get_json()
    saved = sorted(p.name for p in (webapp_config.media_dir / "2026-08-06_Alex").iterdir())
    assert saved == ["part01__set1.mp4", "unpaired__random.mp4"]
    notes = " ".join(body["upload_notes"])
    assert "random.mp4" in notes  # the clip that matched nothing
    assert "set2.xlsx" in notes  # the export left without footage


def test_single_export_keeps_its_footage_filename_unchanged(
    client, xlsx_bytes, webapp_config
):
    """A one-part match has no order to encode, so nothing is renamed and
    the clip's filename never has to match the export's."""
    response = _upload_parts(client, xlsx_bytes, exports=["export.xlsx"], videos=["anything.mp4"])

    assert response.status_code == 200
    body = response.get_json()
    assert body["parts"][0]["videos"] == ["anything.mp4"]
    assert body["upload_notes"] == []
    saved = [p.name for p in (webapp_config.media_dir / "2026-08-06_Alex").iterdir()]
    assert saved == ["anything.mp4"]


def test_uploaded_exports_are_archived_after_staging(client, xlsx_bytes, webapp_config):
    """The .xlsx copies in data/uploads/ have done their job once the match
    is staged, so they get packed into one zip (~40% smaller)."""
    response = _upload(client, xlsx_bytes)

    assert response.status_code == 200
    contents = sorted(p.name for p in webapp_config.uploads_dir.iterdir())
    assert contents == ["2026-08-06_Alex_exports.zip"]
    with zipfile.ZipFile(webapp_config.uploads_dir / contents[0]) as archive:
        assert archive.namelist() == ["2026-08-06_Alex_export.xlsx"]


def test_archiving_the_exports_leaves_the_footage_alone(
    client, xlsx_bytes, webapp_config
):
    """Footage is already entropy-coded - zipping it measured at -0.0% and
    would break Film Review, which streams the file straight off disk."""
    response = _upload(
        client, xlsx_bytes, video_files=(io.BytesIO(b"fake video bytes"), "clip.mp4")
    )

    assert response.status_code == 200
    clip = webapp_config.media_dir / "2026-08-06_Alex" / "clip.mp4"
    assert clip.exists()
    assert clip.read_bytes() == b"fake video bytes"  # byte-for-byte, playable


def test_review_still_works_against_an_archived_export(client, xlsx_bytes):
    """The flagged-point walk re-parses the export for shot context, so it
    has to keep working once that export lives inside a zip."""
    upload = _upload(client, xlsx_bytes).get_json()

    detail = client.get(f"/api/pending/{upload['json_filename']}").get_json()

    assert detail["points"], "no flagged points came back"
    assert any(p["shots"] for p in detail["points"]), "no shot context after archiving"


def test_an_over_cap_upload_returns_a_json_error_not_an_html_page(
    xlsx_bytes, import_config, tmp_path
):
    """Flask's stock 413 is HTML, which the frontend's fetch wrapper can't
    read a message out of - the user just saw the bare status line."""
    from webapp.app import create_app
    from webapp.config import WebAppConfig

    tiny_cap = WebAppConfig(
        uploads_dir=tmp_path / "uploads",
        media_dir=tmp_path / "media",
        max_content_length=1024,  # 1 KB, so the fixture export is over it
    )
    app = create_app(import_config=import_config, webapp_config=tiny_cap)
    with app.test_client() as client:
        response = _upload(client, xlsx_bytes)

    assert response.status_code == 413
    assert response.is_json
    error = response.get_json()["error"]
    assert "GB limit" in error
    assert "RALLYAI_MAX_UPLOAD_MB" in error  # tells them how to raise it


def test_the_upload_cap_can_be_raised_by_environment_variable(monkeypatch):
    from webapp.config import WebAppConfig

    monkeypatch.setenv("RALLYAI_MAX_UPLOAD_MB", "64")
    assert WebAppConfig().max_content_length == 64 * 1024 * 1024


def test_a_nonsense_upload_cap_falls_back_to_the_default(monkeypatch):
    """A typo in an env var must not silently shrink the limit to something
    that rejects every real upload."""
    from webapp.config import WebAppConfig

    monkeypatch.setenv("RALLYAI_MAX_UPLOAD_MB", "lots")
    assert WebAppConfig().max_content_length == 32 * 1024 * 1024 * 1024


def _timed_xlsx_bytes() -> bytes:
    """A non-Pro export whose shots carry footage timing."""
    from openpyxl import Workbook

    from tests.swingvision_import.conftest import add_settings_and_shots_sheets

    workbook = Workbook()
    sets_sheet = workbook.active
    sets_sheet.title = "Sets"
    sets_sheet.append(["Set", "Host Score", "Guest Score", "Set Winner"])
    sets_sheet.append([1, 6, 4, "host"])
    workbook.create_sheet("Games").append(["Game", "Set", "Server", "Game Winner"])
    workbook.create_sheet("Points").append(
        ["Point", "Game", "Set", "Match Server", "Point Winner", "Detail"]
    )
    add_settings_and_shots_sheets(
        workbook,
        shot_rows=[
            [1, 1, "Test Player", "first_serve", "Serve", "In", 30.0, "near"],
            [1, 2, "Test Opponent", "first_return", "Backhand", "Net", 31.4, "far"],
        ],
    )
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def test_footage_status_reports_no_footage_when_none_was_uploaded(client):
    upload = _upload(client, _timed_xlsx_bytes()).get_json()

    status = client.get(f"/api/pending/{upload['json_filename']}/footage").get_json()

    assert status["available"] is False
    assert "No footage" in status["reason"]
    assert status["eligible_points"] > 0


def test_footage_status_refuses_a_condensed_export(client, monkeypatch):
    """The real-world case: SwingVision's rallies-only video is a fraction
    of the recording its timestamps refer to, so the check must refuse it
    and say why rather than silently checking the wrong rallies."""
    upload = _upload(
        client,
        _timed_xlsx_bytes(),
        video_files=(io.BytesIO(b"fake"), "clip.mp4"),
    ).get_json()
    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 5.0)  # way too short

    status = client.get(f"/api/pending/{upload['json_filename']}/footage").get_json()

    assert status["available"] is False
    assert "rallies-only" in status["reason"]


def test_footage_status_quotes_a_cost_before_anything_is_spent(client, monkeypatch):
    upload = _upload(
        client,
        _timed_xlsx_bytes(),
        video_files=(io.BytesIO(b"fake"), "clip.mp4"),
    ).get_json()
    monkeypatch.setattr(footage, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(footage, "probe_duration", lambda *a, **k: 10_000.0)

    status = client.get(f"/api/pending/{upload['json_filename']}/footage").get_json()

    assert status["available"] is True
    assert status["estimate_usd"] > 0
    assert status["model"] == "claude-opus-5"


def test_starting_a_check_without_footage_is_refused(client):
    upload = _upload(client, _timed_xlsx_bytes()).get_json()

    response = client.post(f"/api/pending/{upload['json_filename']}/footage")

    assert response.status_code == 400
    assert "No footage" in response.get_json()["error"]


def test_a_missing_pending_file_404s(client):
    assert client.get("/api/pending/nope.json/footage").status_code == 404
