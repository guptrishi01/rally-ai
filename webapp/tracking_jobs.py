"""Tracking a match video as a separate OS process, and following it.

Tracking a whole match takes hours of CPU (roughly 8 h on a laptop) and
~730 MB of memory, so unlike the footage check (`webapp/jobs.py`, a
thread) it cannot live inside the web server: it runs as its own process -
`scripts/track_match.py`, the same command a person would type - and the
server only starts it and reads how far it has got.

What survives a restart, and why:

- **The job record** (`<pending_dir>/tracking_work/<staged stem>.job.json`)
  holds the exact arguments the process was started with and its pid, so
  a server that restarts can still describe the job and, if the process
  is gone, start it again with the same arguments.
- **The progress file** (`TrackerPipeline.progress_path`) is written by
  the tracker itself on every report, so it is true whichever server
  process reads it.
- **The checkpoints** make a relaunch cheap: finished chunks are not redone.

A dead job is never relaunched silently. The status says "interrupted" and
the person decides, with a resume button - a run that died because the
machine ran out of memory would only die again unattended.

The job is keyed on the staged JSON's filename (`TrackerPipeline
.staged_path`), which is known before tracking starts: when that file
appears the match is staged, and the existing review flow takes over.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_SCRIPT = _REPO_ROOT / "scripts" / "track_match.py"

# One tracking run at a time: each one holds ~730 MB and every core it can
# get for hours, and two at once on a laptop just makes both slower.
MAX_RUNNING = 1

_LOG_TAIL_BYTES = 4096
# Written into the log before every launch, so an error from an earlier
# attempt is never reported against the current one.
_LAUNCH_MARKER = "=== rally-ai: tracking launch"


class TrackingJobError(Exception):
    """A request about a tracking job that cannot be honoured as asked."""


class TrackingJobConflict(TrackingJobError):
    """The job is in a state that forbids the request (e.g. still running)."""


class Process(Protocol):
    """The part of `subprocess.Popen` the job store uses."""

    pid: int

    def poll(self) -> int | None:
        """Returns the exit code, or None while still running."""


class Launcher(Protocol):
    """Starts a tracking process and says whether a pid is still running.

    Injected so tests never start a real tracking run.
    """

    def launch(self, argv: list[str], log_path: Path) -> Process:
        """Starts `scripts/track_match.py` with these arguments.

        Args:
            argv: Arguments after the script's path.
            log_path: Where its output goes.

        Returns:
            The running process.
        """

    def is_alive(self, pid: int) -> bool:
        """Whether a process with this pid is running."""


def pid_alive(pid: int) -> bool:
    """Whether a process with this pid is running, without extra packages.

    Used only for a job this server process did not start (it restarted
    since): a job it started is asked through its own `Popen`, which is
    exact. A pid can in principle be reused by an unrelated process after
    the tracker exits; the progress file going stale is the tell, and the
    resume button stays available once the pid is gone.

    Args:
        pid: The process id.

    Returns:
        True if it is running.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:  # pragma: no cover - exercised on POSIX only
        os.kill(pid, 0)
    except ProcessLookupError:  # pragma: no cover
        return False
    except PermissionError:  # pragma: no cover
        return True
    return True  # pragma: no cover


class SubprocessLauncher:
    """Starts `scripts/track_match.py` as a detached child process.

    Detached so it outlives the web server: a restart (or the Flask
    reloader) must not kill hours of tracking. Output goes to a log file
    beside the progress file, unbuffered, so its last error line is there
    to show if it stops.
    """

    def launch(self, argv: list[str], log_path: Path) -> Process:
        """Starts the tracker.

        Args:
            argv: Arguments after the script's path.
            log_path: Where its stdout and stderr go (appended).

        Returns:
            The `Popen`.
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)
        options: dict = {}
        if sys.platform == "win32":  # pragma: no cover - platform branch
            options["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            )
        else:  # pragma: no cover - platform branch
            options["start_new_session"] = True
        with open(log_path, "ab") as log:
            return subprocess.Popen(
                [sys.executable, str(TRACK_SCRIPT), *argv],
                cwd=_REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                **options,
            )

    def is_alive(self, pid: int) -> bool:
        """Whether the pid is running (see `pid_alive`)."""
        return pid_alive(pid)


@dataclass
class TrackingJobs:
    """The tracking jobs this app started, on disk, and their processes.

    Attributes:
        pending_dir: Where staged matches land; the job records, progress
            files and logs live in its `tracking_work` folder.
        launcher: Starts processes; `SubprocessLauncher` unless injected.
        clock: Wall-clock time, injectable for tests.
    """

    pending_dir: Path
    launcher: Launcher = field(default_factory=SubprocessLauncher)
    clock: Callable[[], float] = time.time
    _processes: dict[str, Process] = field(default_factory=dict, init=False, repr=False)

    @property
    def work_dir(self) -> Path:
        """The tracker's own work folder, shared with its progress files."""
        return self.pending_dir / "tracking_work"

    def _record_path(self, json_filename: str) -> Path:
        return self.work_dir / f"{Path(json_filename).stem}.job.json"

    def progress_path(self, json_filename: str) -> Path:
        """Where the tracker writes this job's progress.

        The same rule as `tracking.pipeline.TrackerPipeline.progress_path`
        (a test holds the two together), restated so reading a status
        never imports the tracker and its OpenCV dependency.

        Args:
            json_filename: The staged JSON's filename.

        Returns:
            The progress file's path.
        """
        return self.work_dir / f"{Path(json_filename).stem}.progress.json"

    def log_path(self, json_filename: str) -> Path:
        """Where the tracking process's output goes.

        Args:
            json_filename: The staged JSON's filename.

        Returns:
            The log file's path.
        """
        return self.work_dir / f"{Path(json_filename).stem}.log"

    # ------------------------------------------------------------------

    def _read_record(self, json_filename: str) -> dict | None:
        path = self._record_path(json_filename)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write_record(self, record: dict) -> None:
        path = self._record_path(record["json_filename"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    def _alive(self, record: dict) -> bool:
        """Whether the job's process is running."""
        process = self._processes.get(record["json_filename"])
        if process is not None and process.pid == record.get("pid"):
            return process.poll() is None
        return bool(record.get("pid")) and self.launcher.is_alive(int(record["pid"]))

    def _exit_code(self, record: dict) -> int | None:
        """The exit code of a process this server started, when it has one."""
        process = self._processes.get(record["json_filename"])
        if process is None or process.pid != record.get("pid"):
            return None
        return process.poll()

    def _progress(self, json_filename: str) -> dict | None:
        try:
            return json.loads(self.progress_path(json_filename).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _last_error(self, json_filename: str) -> str | None:
        """The last line of the log that reads as an error, if any."""
        try:
            with open(self.log_path(json_filename), "rb") as log:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - _LOG_TAIL_BYTES))
                tail = log.read().decode("utf-8", "replace")
        except OSError:
            return None
        lines = [line.strip() for line in tail.splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith(_LAUNCH_MARKER):
                return None
            if line.startswith("error:") or "Error" in line.split(":")[0]:
                return line
        return None

    # ------------------------------------------------------------------

    def status(self, json_filename: str) -> dict | None:
        """Describes one job.

        Args:
            json_filename: The staged JSON's filename the job is keyed on.

        Returns:
            None for an unknown job. Otherwise its record's description
            (date, opponent, video, started_at, launches) plus `state`:
            "running", "staged" (the staged file is there - review it),
            "failed" (the process exited with an error, shown in
            `error`) or "interrupted" (the process is gone without
            staging or an error - the server or machine restarted; resume
            it). `alive` and `progress` (the tracker's latest
            `TrackingProgress`, or None) are included either way.
        """
        record = self._read_record(json_filename)
        if record is None:
            return None
        alive = self._alive(record)
        progress = self._progress(json_filename)
        staged = (self.pending_dir / json_filename).exists()
        error = None
        if staged and (not alive or (progress or {}).get("stage") in ("staged", "skipped")):
            state = "staged"
        elif alive:
            state = "running"
        else:
            code = self._exit_code(record)
            error = self._last_error(json_filename)
            if code not in (None, 0) or error:
                state = "failed"
                error = error or f"The tracking process stopped with exit code {code}."
            else:
                state = "interrupted"
        return {
            "json_filename": json_filename,
            "date": record.get("date"),
            "opponent": record.get("opponent"),
            "video": record.get("video"),
            "started_at": record.get("started_at"),
            "launches": record.get("launches", 1),
            "state": state,
            "alive": alive,
            "error": error,
            "progress": progress,
        }

    def list(self) -> list[dict]:
        """Describes every job, most recently started first.

        Returns:
            One `status` per job record on disk.
        """
        if not self.work_dir.is_dir():
            return []
        statuses = [
            self.status(path.name[: -len(".job.json")] + ".json")
            for path in self.work_dir.glob("*.job.json")
        ]
        found = [status for status in statuses if status is not None]
        return sorted(found, key=lambda s: s.get("started_at") or 0, reverse=True)

    def running(self) -> list[dict]:
        """The jobs whose process is alive right now.

        Returns:
            Their statuses.
        """
        return [status for status in self.list() if status["alive"]]

    def _launch(self, record: dict) -> None:
        """Starts (or restarts) the record's process and saves its pid."""
        json_filename = record["json_filename"]
        log_path = self.log_path(json_filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"{_LAUNCH_MARKER} {record.get('launches', 1)} ===\n")
        process = self.launcher.launch(record["argv"], log_path)
        self._processes[json_filename] = process
        record["pid"] = process.pid
        record["started_at"] = self.clock()
        self._write_record(record)
        logger.info("Started tracking for %s (pid %s)", json_filename, process.pid)

    def start(self, json_filename: str, argv: list[str], *, date: str, opponent: str,
              video: str) -> dict:
        """Starts tracking a match.

        Args:
            json_filename: The staged JSON's filename, from
                `TrackerPipeline.staged_path`.
            argv: `scripts/track_match.py`'s arguments, kept so the same
                run can be resumed later.
            date: The match date, for the job list.
            opponent: The opponent, for the job list.
            video: The video's filename, for the job list.

        Returns:
            The new job's status.

        Raises:
            TrackingJobConflict: If this match is already being tracked, or
                another match is (one run at a time).
        """
        current = self.status(json_filename)
        if current is not None and current["alive"]:
            raise TrackingJobConflict("This match is already being tracked.")
        if len(self.running()) >= MAX_RUNNING:
            raise TrackingJobConflict(
                "Another match is being tracked right now. Tracking holds about 730 MB of "
                "memory and every core it can get for hours, so one runs at a time - start "
                "this one when that finishes."
            )
        record = {
            "json_filename": json_filename,
            "argv": list(argv),
            "date": date,
            "opponent": opponent,
            "video": video,
            "launches": 1,
        }
        self._launch(record)
        return self.status(json_filename)

    def resume(self, json_filename: str) -> dict:
        """Starts a stopped job again with the arguments it was started with.

        Finished chunks are taken from checkpoints, so this carries on
        rather than starting over.

        Args:
            json_filename: The job's staged JSON filename.

        Returns:
            Its status after relaunching.

        Raises:
            TrackingJobError: If there is no such job.
            TrackingJobConflict: If it is still running, already staged,
                or another job is running.
        """
        record = self._read_record(json_filename)
        if record is None:
            raise TrackingJobError("No tracking job for that match.")
        current = self.status(json_filename)
        if current["state"] == "running":
            raise TrackingJobConflict("This match is still being tracked.")
        if current["state"] == "staged":
            raise TrackingJobConflict("This match is already staged - review it instead.")
        if self.running():
            raise TrackingJobConflict(
                "Another match is being tracked right now; resume this one when it finishes."
            )
        record["launches"] = int(record.get("launches", 1)) + 1
        self._launch(record)
        return self.status(json_filename)

    def dismiss(self, json_filename: str) -> bool:
        """Forgets a job that is no longer running.

        Removes the job record and its log only. The progress file and
        the checkpoints stay, so starting the same video again still
        resumes; the staged match, if any, is untouched.

        Args:
            json_filename: The job's staged JSON filename.

        Returns:
            False if there was no such job.

        Raises:
            TrackingJobConflict: If it is still running - stop the process
                first; the server never kills a run it cannot restart.
        """
        current = self.status(json_filename)
        if current is None:
            return False
        if current["alive"]:
            raise TrackingJobConflict("This match is still being tracked.")
        self._record_path(json_filename).unlink(missing_ok=True)
        self.log_path(json_filename).unlink(missing_ok=True)
        self._processes.pop(json_filename, None)
        return True
