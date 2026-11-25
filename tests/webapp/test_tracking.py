"""Tests for the "no export - track the video" intake, its jobs, and the
placement and provenance routes. No test ever starts a real tracking run:
the launcher is a fake that records what it was asked to start."""

from __future__ import annotations

import io
import json
import os
import subprocess
from pathlib import Path

import pytest

from swingvision_import.records import MatchRecord, PointRecord, SetRecord
from swingvision_import.review import save_pending
from tracking.assemble import ShotTrace
from tracking.pipeline import TrackerPipeline
from tracking.provenance import Provenance, Tagged
from webapp import tracking_jobs
from webapp.app import create_app
from webapp.tracking_jobs import SubprocessLauncher, TrackingJobs, pid_alive

STAGED = "2026-08-21_Alex.json"


class FakeProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.returncode: int | None = None

    def poll(self):
        return self.returncode


class FakeLauncher:
    """Starts nothing; remembers every launch and which pids are 'alive'."""

    def __init__(self):
        self.launches: list[tuple[list[str], Path]] = []
        self.processes: list[FakeProcess] = []
        self.alive_pids: set[int] = set()

    def launch(self, argv, log_path):
        self.launches.append((list(argv), log_path))
        process = FakeProcess(4000 + len(self.launches))
        self.processes.append(process)
        return process

    def is_alive(self, pid):
        return pid in self.alive_pids


@pytest.fixture
def launcher():
    return FakeLauncher()


@pytest.fixture
def app(import_config, webapp_config, launcher):
    flask_app = create_app(import_config, webapp_config, tracking_launcher=launcher)
    flask_app.config["TESTING"] = True
    return flask_app


def _form(**changes):
    form = {
        "date": "2026-08-21",
        "opponent": "Alex",
        "result": "W",
        "host_name": "Rishi",
        "guest_name": "Alex B",
        "host_starts": "far",
        "host_hand": "left",
        "guest_hand": "right",
        "video_file": (io.BytesIO(b"fake video bytes"), "../../Match Day.MP4"),
    }
    form.update(changes)
    return {k: v for k, v in form.items() if v is not None}


def _start(client, **changes):
    return client.post("/api/track", data=_form(**changes), content_type="multipart/form-data")


def _media_files(webapp_config):
    root = webapp_config.media_dir
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) \
        if root.exists() else []


def test_starting_saves_the_footage_and_launches_the_cli(client, launcher, webapp_config,
                                                         import_config):
    response = _start(
        client,
        player_appearance="grey shirt",
        energy_rating="4",
        reviewer_note="cut at 3-2",
        first_server_set2="opponent",
        first_server_set1="me",
        tracked_identity="ignored for tracking",
    )
    assert response.status_code == 202
    status = response.get_json()
    assert status["json_filename"] == STAGED
    assert status["state"] == "running" and status["alive"] is True
    assert status["video"] == "Match_Day.MP4"
    assert status["progress"] is None

    # Saved under data/media/<match>/ with a sanitised name, so Film Review
    # finds it - and nowhere else.
    assert _media_files(webapp_config) == ["2026-08-21_Alex/Match_Day.MP4"]
    argv, log_path = launcher.launches[0]
    video = webapp_config.media_dir / "2026-08-21_Alex" / "Match_Day.MP4"
    assert argv == [
        str(video),
        "--host", "Rishi", "--guest", "Alex B", "--host-starts", "far",
        "--host-hand", "left", "--guest-hand", "right", "--result", "W",
        "--date", "2026-08-21", "--opponent", "Alex",
        "--pending-dir", str(import_config.pending_dir),
        "--player-appearance", "grey shirt", "--energy-rating", "4",
        "--reviewer-note", "cut at 3-2",
        "--first-server", "1=me", "--first-server", "2=opponent",
    ]
    assert log_path == import_config.pending_dir / "tracking_work" / "2026-08-21_Alex.log"
    assert "tracking launch 1" in log_path.read_text("utf-8")
    assert client.get("/api/tracking").get_json()[0]["json_filename"] == STAGED


def test_the_guest_defaults_to_the_opponent(client, launcher):
    assert _start(client, guest_name="").status_code == 202
    argv = launcher.launches[0][0]
    assert argv[argv.index("--guest") + 1] == "Alex"


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"video_file": None}, "exactly one match video"),
        ({"date": "21/08/2026"}, "match date"),
        ({"opponent": " "}, "opponent's name"),
        ({"host_name": ""}, "your own name"),
        ({"guest_name": "rishi"}, "must differ"),
        ({"result": "D"}, "won or lost"),
        ({"host_starts": "middle"}, "which end"),
        ({"guest_hand": "both"}, "guest hand must be right or left"),
        ({"energy_rating": "9"}, "energy_rating"),
        ({"first_server_set1": "nobody"}, "served first"),
        ({"video_file": (io.BytesIO(b"x"), "???")}, "nothing usable"),
    ],
)
def test_a_bad_answer_is_refused_before_anything_is_written(
    client, launcher, webapp_config, changes, message
):
    response = _start(client, **changes)
    assert response.status_code == 400
    assert message in response.get_json()["error"]
    assert launcher.launches == []
    assert _media_files(webapp_config) == []


def test_two_videos_are_refused(client, launcher):
    data = _form()
    data["video_file"] = [(io.BytesIO(b"a"), "a.mp4"), (io.BytesIO(b"b"), "b.mp4")]
    response = client.post("/api/track", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert launcher.launches == []


def test_one_run_at_a_time_and_the_refusal_writes_nothing(client, launcher, webapp_config):
    assert _start(client).status_code == 202
    again = _start(client)
    assert again.status_code == 409
    assert "already being tracked" in again.get_json()["error"]

    other = _start(client, opponent="Sam", video_file=(io.BytesIO(b"v"), "other.mp4"))
    assert other.status_code == 409
    assert "one runs at a time" in other.get_json()["error"]
    assert _media_files(webapp_config) == ["2026-08-21_Alex/Match_Day.MP4"]
    assert len(launcher.launches) == 1


def _tracked_record(source_file: str) -> MatchRecord:
    trace = ShotTrace(
        point_number=1,
        shot_number=1,
        event_index=0,
        pts=12.0,
        time_uncertainty=0.04,
        confidence=0.7,
        tracked=True,
        fields={
            "player": Tagged("Rishi", Provenance.MEASURED, 0.9),
            "stroke": Tagged("Unclear", Provenance.INFERRED),
            "result": Tagged("In", Provenance.INFERRED, 0.3),
        },
        guesses=("stroke", "result"),
        notes=("stroke too low-confidence to name",),
    ).to_dict()
    attempt = {
        "number": 1, "kind": "served", "start": 12.0, "end": 12.0, "server_side": "near",
        "server": "Rishi", "serve_zone": "deuce", "zone_source": "feet",
        "second_serve": False, "fault": False, "confidence": 0.5, "reasons": ["seen"],
        "shots": [trace],
    }
    point = PointRecord(1, 1, True, True, "ace", needs_review=True, source_point_number=1,
                        tracking=attempt)
    return MatchRecord(
        date="2026-08-21",
        opponent="Alex",
        result="W",
        source_file=source_file,
        sets=[SetRecord(1, 6, 4, points=[point])],
        import_notes=["Staged by the independent tracker."],
        tracking={
            "source": "assemble-v1",
            "summary": {"shots": 1, "tracked_shots": 1, "guesses": {"result": 1, "stroke": 1}},
            "notes": ["Tracked 1 of 1 shots with confidence."],
            "attempts": [attempt],
            "settings": {"host_name": "Rishi", "guest_name": "Alex"},
            "shots": [
                {"point_number": 1, "shot_number": 1, "player": "Rishi",
                 "shot_type": "first_serve", "stroke": "Unclear", "result": "In",
                 "hit_side": "near", "bounce_x": 1.0, "bounce_y": 17.0}
            ],
            "models": {"ball": "classic"},
            "video": {"name": "Match_Day.MP4"},
        },
    )


def test_a_finished_run_is_staged_and_listed_for_review(client, launcher, import_config,
                                                        webapp_config):
    _start(client)
    progress = {"stage": "tracking", "chunks_total": 8, "chunks_done": 3, "chunks_resumed": 1,
                "video_seconds": 4800.0, "video_seconds_done": 1800.0, "elapsed_seconds": 900.0,
                "eta_seconds": 1500.0, "seconds_per_video_minute": 30.0, "message": "Chunk 3/8."}
    work = import_config.pending_dir / "tracking_work"
    (work / "2026-08-21_Alex.progress.json").write_text(json.dumps(progress), "utf-8")
    assert client.get(f"/api/tracking/{STAGED}").get_json()["progress"] == progress

    video = webapp_config.media_dir / "2026-08-21_Alex" / "Match_Day.MP4"
    save_pending(_tracked_record(str(video)), import_config.pending_dir)
    # Staged while the process is still finishing up: not "staged" until
    # it exits, unless the tracker itself said so.
    assert client.get(f"/api/tracking/{STAGED}").get_json()["state"] == "running"
    launcher.processes[0].returncode = 0
    status = client.get(f"/api/tracking/{STAGED}").get_json()
    assert status["state"] == "staged" and status["alive"] is False

    listed = client.get("/api/pending").get_json()
    assert [(m["json_filename"], m["source"]) for m in listed] == [(STAGED, "tracker")]


def test_a_failed_run_reports_its_error_and_resumes_with_the_same_arguments(
    client, launcher, import_config
):
    _start(client)
    log = import_config.pending_dir / "tracking_work" / "2026-08-21_Alex.log"
    with open(log, "a", encoding="utf-8") as handle:
        handle.write("[calibrating] Calibrating the court.\n")
        handle.write("error: Court calibration abstained at 1.0s: no lines.\n")
    launcher.processes[0].returncode = 2
    status = client.get(f"/api/tracking/{STAGED}").get_json()
    assert status["state"] == "failed"
    assert status["error"].startswith("error: Court calibration abstained")

    resumed = client.post(f"/api/tracking/{STAGED}/resume")
    assert resumed.status_code == 202
    assert resumed.get_json()["state"] == "running"
    assert resumed.get_json()["launches"] == 2
    assert launcher.launches[1][0] == launcher.launches[0][0]
    # The earlier attempt's error is not reported against this one.
    launcher.processes[1].returncode = None
    assert client.get(f"/api/tracking/{STAGED}").get_json()["error"] is None


def test_an_exit_with_no_message_is_still_a_failure(client, launcher):
    _start(client)
    launcher.processes[0].returncode = 1
    status = client.get(f"/api/tracking/{STAGED}").get_json()
    assert status["state"] == "failed"
    assert "exit code 1" in status["error"]


def test_a_traceback_names_the_exception(client, launcher, import_config):
    _start(client)
    log = import_config.pending_dir / "tracking_work" / "2026-08-21_Alex.log"
    with open(log, "a", encoding="utf-8") as handle:
        handle.write("Traceback (most recent call last):\n  File x\nMemoryError: out\n")
    launcher.processes[0].returncode = 1
    assert client.get(f"/api/tracking/{STAGED}").get_json()["error"] == "MemoryError: out"


def test_after_a_server_restart_a_dead_run_waits_for_resume(
    client, launcher, import_config, webapp_config
):
    _start(client)
    pid = launcher.processes[0].pid
    # A new server process: no Popen handles, only the record on disk.
    restarted = create_app(import_config, webapp_config, tracking_launcher=launcher)
    other = restarted.test_client()
    launcher.alive_pids.add(pid)
    assert other.get(f"/api/tracking/{STAGED}").get_json()["state"] == "running"
    launcher.alive_pids.clear()
    status = other.get(f"/api/tracking/{STAGED}").get_json()
    assert status["state"] == "interrupted" and status["error"] is None
    # Never relaunched by itself.
    assert len(launcher.launches) == 1
    assert other.post(f"/api/tracking/{STAGED}/resume").status_code == 202
    assert len(launcher.launches) == 2


def test_resume_and_dismiss_refuse_what_they_cannot_do(client, launcher, import_config,
                                                       webapp_config):
    assert client.post("/api/tracking/nope.json/resume").status_code == 404
    assert client.delete("/api/tracking/nope.json").status_code == 404
    assert client.get("/api/tracking/nope.json").status_code == 404

    _start(client)
    assert client.post(f"/api/tracking/{STAGED}/resume").status_code == 409
    assert client.delete(f"/api/tracking/{STAGED}").status_code == 409

    launcher.processes[0].returncode = 0
    video = webapp_config.media_dir / "2026-08-21_Alex" / "Match_Day.MP4"
    save_pending(_tracked_record(str(video)), import_config.pending_dir)
    staged = client.post(f"/api/tracking/{STAGED}/resume")
    assert staged.status_code == 409 and "already staged" in staged.get_json()["error"]

    # Starting the same video again once staged points at the review instead.
    again = _start(client)
    assert again.status_code == 409 and again.get_json()["json_filename"] == STAGED

    assert client.delete(f"/api/tracking/{STAGED}").status_code == 200
    assert client.get(f"/api/tracking/{STAGED}").status_code == 404
    # The staged match itself is untouched.
    assert (import_config.pending_dir / STAGED).exists()


def test_resume_waits_while_another_match_is_tracked(import_config, launcher):
    jobs = TrackingJobs(import_config.pending_dir, launcher)
    jobs.start("a.json", ["a.mp4"], date="d", opponent="A", video="a.mp4")
    launcher.processes[0].returncode = 1
    jobs.start("b.json", ["b.mp4"], date="d", opponent="B", video="b.mp4")
    with pytest.raises(tracking_jobs.TrackingJobConflict, match="Another match"):
        jobs.resume("a.json")


def test_the_job_list_survives_a_damaged_file(import_config, launcher):
    jobs = TrackingJobs(import_config.pending_dir, launcher)
    assert jobs.list() == []
    jobs.start("a.json", ["a.mp4"], date="d", opponent="A", video="a.mp4")
    (jobs.work_dir / "broken.job.json").write_text("{", "utf-8")
    jobs.progress_path("a.json").write_text("{", "utf-8")
    assert [s["json_filename"] for s in jobs.list()] == ["a.json"]
    assert jobs.status("a.json")["progress"] is None


def test_progress_is_read_where_the_tracker_writes_it(import_config):
    staged = import_config.pending_dir / STAGED
    expected = TrackerPipeline(import_config=import_config).progress_path(staged)
    assert TrackingJobs(import_config.pending_dir).progress_path(STAGED) == expected


def test_the_subprocess_launcher_runs_the_cli_detached(tmp_path, monkeypatch):
    seen = {}

    def fake_popen(command, **options):
        seen["command"], seen["options"] = command, options
        return FakeProcess(1)

    monkeypatch.setattr(tracking_jobs.subprocess, "Popen", fake_popen)
    process = SubprocessLauncher().launch(["v.mp4", "--host", "R"], tmp_path / "w" / "x.log")
    assert process.pid == 1
    assert seen["command"][1:] == [str(tracking_jobs.TRACK_SCRIPT), "v.mp4", "--host", "R"]
    assert seen["options"]["env"]["PYTHONUNBUFFERED"] == "1"
    assert seen["options"]["stdin"] is subprocess.DEVNULL
    assert "creationflags" in seen["options"] or seen["options"].get("start_new_session")
    assert SubprocessLauncher().is_alive(os.getpid())


def test_pid_alive_knows_this_process_and_rejects_nonsense():
    assert pid_alive(os.getpid())
    assert not pid_alive(0)
    assert not pid_alive(-5)


def test_pending_detail_shows_tracker_provenance(client, import_config):
    save_pending(_tracked_record("C:/nowhere/Match_Day.MP4"), import_config.pending_dir)
    detail = client.get(f"/api/pending/{STAGED}").get_json()
    assert detail["source"] == "tracker"
    assert detail["tracking"]["notes"] == ["Tracked 1 of 1 shots with confidence."]
    assert detail["tracking"]["guessed_strokes"] == 1
    (point,) = detail["points"]
    assert point["shots"][0]["stroke"] == "Unclear"
    fields = {f["field"]: f for f in point["tracking"]["shots"][0]["fields"]}
    assert fields["player"]["provenance"] == "measured"
    assert fields["stroke"]["guess"] is True
    # Informational only: the point is still flagged.
    assert client.get("/api/pending").get_json()[0]["flags_remaining"] == 1


def test_pending_detail_for_an_export_has_no_tracker_block(client, xlsx_bytes):
    imported = client.post(
        "/api/import",
        data={"xlsx_files": (io.BytesIO(xlsx_bytes), "e.xlsx"), "date": "2026-08-06",
              "opponent": "Alex", "result": "W"},
        content_type="multipart/form-data",
    ).get_json()
    detail = client.get(f"/api/pending/{imported['json_filename']}").get_json()
    assert detail["source"] == "export" and detail["tracking"] is None
    assert all(p["tracking"] is None for p in detail["points"])


def test_placement_route(client, finalized_match_id, import_config):
    assert client.get("/api/matches/999/placement").status_code == 404
    body = client.get(f"/api/matches/{finalized_match_id}/placement").get_json()
    assert body["available"] is False and "no longer on disk" in body["reason"]


def test_placement_route_reads_a_tracked_match(client, finalized_match_id, import_config):
    record = _tracked_record("C:/nowhere/Match_Day.MP4")
    record.date, record.opponent = "2026-08-06", "Alex"
    save_pending(record, import_config.pending_dir)
    body = client.get(f"/api/matches/{finalized_match_id}/placement").get_json()
    assert body["available"] and body["source"] == "tracker"
    assert body["bounces"][0]["by"] == "me" and body["bounces"][0]["kind"] == "serve"

