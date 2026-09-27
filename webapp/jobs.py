"""In-process tracking for the one long-running job the app has.

A footage check is minutes of work and real money, so it can't be a
synchronous request: the browser needs to show progress, and a reload
mustn't start a second one. This keeps that state in memory, deliberately -
it describes a run happening in *this* process, and a restart genuinely
ends it. Anything worth keeping (each point's verdict) is already written
into the pending JSON as the run goes, so a lost job costs progress
reporting, never results.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import asdict, is_dataclass

logger = logging.getLogger(__name__)


class JobRegistry:
    """Tracks at most one running job per staged match.

    Attributes:
        _jobs: Pending-JSON filename -> the latest snapshot of its job.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def start(self, key: str, run: Callable[[], object]) -> bool:
        """Starts `run` on a background thread, unless one is already going.

        Args:
            key: The pending JSON filename the job belongs to.
            run: The work to do. Its return value, if a dataclass, becomes
                the job's final summary.

        Returns:
            True if a job was started, False if one was already running.
        """
        with self._lock:
            existing = self._jobs.get(key)
            if existing and existing["state"] == "running":
                return False
            self._jobs[key] = {"state": "running", "progress": None, "summary": None, "error": None}

        def worker() -> None:
            try:
                result = run()
            except Exception as exc:  # noqa: BLE001 - reported to the UI, not swallowed
                logger.exception("Background job failed for %s", key)
                with self._lock:
                    self._jobs[key] = {
                        "state": "failed",
                        "progress": self._jobs.get(key, {}).get("progress"),
                        "summary": None,
                        "error": str(exc),
                    }
                return
            with self._lock:
                self._jobs[key] = {
                    "state": "done",
                    "progress": None,
                    "summary": asdict(result) if is_dataclass(result) else None,
                    "error": None,
                }

        threading.Thread(target=worker, name=f"job-{key}", daemon=True).start()
        return True

    def update(self, key: str, progress: object) -> None:
        """Records a running job's latest progress.

        Args:
            key: The job's key.
            progress: A dataclass snapshot of how far along it is.
        """
        with self._lock:
            job = self._jobs.get(key)
            if job and job["state"] == "running":
                job["progress"] = asdict(progress) if is_dataclass(progress) else None

    def snapshot(self, key: str) -> dict | None:
        """Returns the current state of a job, or None if there isn't one.

        Args:
            key: The job's key.

        Returns:
            A JSON-safe copy of the job's state.
        """
        with self._lock:
            job = self._jobs.get(key)
            return dict(job) if job else None
