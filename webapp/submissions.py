"""Disk-backed submissions: what the phone owns until the server accepts it.

A submission is the phone companion's unit of work. Metadata (how the
match felt, entered at the court) is sent first and alone, because it is
irreplaceable and tiny; the files follow later over Wi-Fi, in resumable
chunks, because a full recording is ~11 GB and a single POST that dies at
80% is not acceptable.

**Why this is on disk and webapp/jobs.py is not.** jobs.py keeps its state
in memory on purpose: it describes a footage check running in *this*
process, and a restart genuinely ends that run - everything worth keeping
is already in the pending JSON. A submission is the opposite. It outlives
any one request by design: the metadata may sit for hours before the files
arrive, and a half-uploaded recording that a restart forgot would have to
be re-sent from byte zero. So this follows the pending-JSON pattern
instead - one JSON file per submission, re-read rather than cached, so
state after a restart is the state before it.

Layout, all under submissions_dir (gitignored - personal match data):

    <id>/submission.json   the record: metadata, state, per-slot progress
    <id>/<slot>            one sparse file per slot, written at offsets

A **slot is the filename** the phone is sending ("set2.xlsx", "set2.mp4").
That is not incidental: webapp/uploads.py pairs an export with the footage
sharing its stem and reads part order off the same names, so keying slots
by filename means /complete can hand pair_uploads exactly what the browser
form hands it, and the pairing rule stays in one place.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

STATE_AWAITING_FILES = "awaiting_files"
STATE_UPLOADING = "uploading"
STATE_STAGED = "staged"
STATE_FAILED = "failed"

_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_SUBMISSION_ID = re.compile(r"^[0-9a-f]{32}$")
_RECORD_NAME = "submission.json"

_REQUIRED_FIELDS = ("date", "opponent", "result")
_TEXT_FIELDS = ("location", "pros", "cons", "notes", "reviewer_note", "tracked_identity")
_RATING_FIELDS = ("energy_rating", "mental_rating")
_SET_NUMBERS = (1, 2, 3)
_SERVERS = ("me", "opponent")
_RESULTS = ("W", "L")


class SubmissionError(ValueError):
    """A submission was malformed: bad metadata, bad slot, or a bad chunk."""


class SubmissionConflict(SubmissionError):
    """A chunk contradicts what the server already holds for that slot.

    Separate from SubmissionError so the API can answer 409 (the client's
    own state is out of step) rather than 400 (the request is nonsense).
    """


def _now() -> str:
    """Returns the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def parse_content_range(header: str | None) -> tuple[int, int, int]:
    """Parses a `Content-Range: bytes start-end/total` upload header.

    Only the fully-specified form is accepted. "bytes */total" and a
    missing header are both refused rather than guessed at, because
    guessing an offset is how a resumed upload silently corrupts a file.

    Args:
        header: The raw header value, or None if the client sent none.

    Returns:
        (start, end, total), with end inclusive, as HTTP defines it.

    Raises:
        SubmissionError: If the header is missing, malformed, inverted, or
            reaches past the declared total.
    """
    match = _CONTENT_RANGE.match((header or "").strip())
    if match is None:
        raise SubmissionError(
            "A chunk must carry a Content-Range header of the form 'bytes start-end/total'."
        )
    start, end, total = (int(group) for group in match.groups())
    if end < start:
        raise SubmissionError(f"Content-Range {start}-{end} ends before it starts.")
    if total <= 0 or end >= total:
        raise SubmissionError(f"Content-Range {start}-{end}/{total} reaches past the file's size.")
    return start, end, total


def merge_range(spans: list[tuple[int, int]], new: tuple[int, int]) -> list[tuple[int, int]]:
    """Folds one received byte span into a sorted, non-overlapping list.

    Adjacent spans join ((0, 9) and (10, 19) become (0, 19)), so a file
    that arrived in order is held as one span however many chunks it took,
    and a genuine gap stays visible as two.

    Args:
        spans: The spans already received, sorted and disjoint.
        new: The (start, end) span just received, end inclusive.

    Returns:
        A new sorted, disjoint list including `new`.
    """
    merged: list[tuple[int, int]] = []
    for span_start, span_end in sorted([*spans, new]):
        if not merged:
            merged.append((span_start, span_end))
            continue
        last_start, last_end = merged[-1]
        if span_start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, span_end))
        else:
            merged.append((span_start, span_end))
    return merged


def validate_metadata(raw: object) -> dict[str, object]:
    """Validates and normalises the metadata the phone submitted.

    Checked here rather than at ingest() time because this is the one
    moment the phone is still holding the user's typed answers: a 400 now
    is a fixable form error, while a bad date discovered at /complete is a
    staged match with the wrong name.

    Args:
        raw: The submitted metadata, expected to be a JSON object.

    Returns:
        Only the recognised, non-blank fields, with the result upper-cased
        and ratings coerced to int - ready to splat into ingest().

    Raises:
        SubmissionError: If a required field is missing or any value is out
            of range for data/schema.sql's own CHECK constraints.
    """
    if not isinstance(raw, dict):
        raise SubmissionError("Metadata must be a JSON object.")

    cleaned: dict[str, object] = {}
    for name in _REQUIRED_FIELDS:
        value = str(raw.get(name) or "").strip()
        if not value:
            raise SubmissionError(f"Missing required field: {name}.")
        cleaned[name] = value

    if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(cleaned["date"])):
        raise SubmissionError("date must be an ISO-8601 date (YYYY-MM-DD).")
    try:
        datetime.strptime(str(cleaned["date"]), "%Y-%m-%d")
    except ValueError as exc:
        raise SubmissionError(f"date is not a real date: {cleaned['date']}.") from exc

    cleaned["result"] = str(cleaned["result"]).upper()
    if cleaned["result"] not in _RESULTS:
        raise SubmissionError("result must be 'W' or 'L'.")

    for name in _TEXT_FIELDS:
        value = str(raw.get(name) or "").strip()
        if value:
            cleaned[name] = value

    for name in _RATING_FIELDS:
        value = raw.get(name)
        if value in (None, ""):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise SubmissionError(f"{name} must be a whole number from 1 to 5.")
        try:
            rating = int(value)
        except ValueError as exc:
            raise SubmissionError(f"{name} must be a whole number from 1 to 5.") from exc
        if not 1 <= rating <= 5:
            raise SubmissionError(f"{name} must be a whole number from 1 to 5.")
        cleaned[name] = rating

    for key, value in raw.items():
        if not key.startswith("first_server_set"):
            continue
        if not value:
            continue
        suffix = key[len("first_server_set") :]
        if not suffix.isdigit() or int(suffix) not in _SET_NUMBERS:
            raise SubmissionError(f"Unknown field: {key}.")
        if value not in _SERVERS:
            raise SubmissionError(f"{key} must be 'me' or 'opponent'.")
        cleaned[key] = value

    return cleaned


@dataclass
class SlotState:
    """Upload progress for one file of a submission.

    Attributes:
        slot: The sanitized filename this slot holds, which is also the
            name /complete hands to the export/footage pairing rule.
        total_bytes: The file's full size, as every chunk's Content-Range
            declares it.
        received: Sorted, disjoint (start, end) spans already on disk.
    """

    slot: str
    total_bytes: int
    received: list[tuple[int, int]] = field(default_factory=list)

    @property
    def received_bytes(self) -> int:
        """How many bytes of this file the server actually holds."""
        return sum(end - start + 1 for start, end in self.received)

    @property
    def complete(self) -> bool:
        """Whether the whole file has arrived, with no gaps."""
        return self.received == [(0, self.total_bytes - 1)]

    @property
    def next_offset(self) -> int:
        """The first byte still missing - where a resume should carry on.

        Returns:
            0 when nothing contiguous has arrived yet (out-of-order chunks
            left a hole at the front), total_bytes when the file is
            complete, otherwise the byte after the leading run.
        """
        if self.complete:
            return self.total_bytes
        if self.received and self.received[0][0] == 0:
            return self.received[0][1] + 1
        return 0

    @property
    def missing_ranges(self) -> list[tuple[int, int]]:
        """Every span the server is still waiting on, in order."""
        missing: list[tuple[int, int]] = []
        cursor = 0
        for start, end in self.received:
            if start > cursor:
                missing.append((cursor, start - 1))
            cursor = end + 1
        if cursor < self.total_bytes:
            missing.append((cursor, self.total_bytes - 1))
        return missing

    def to_dict(self) -> dict[str, object]:
        """Returns the JSON-safe view the phone and dashboard both read."""
        return {
            "slot": self.slot,
            "total_bytes": self.total_bytes,
            "received_bytes": self.received_bytes,
            "complete": self.complete,
            "next_offset": self.next_offset,
            "missing_ranges": [list(span) for span in self.missing_ranges],
        }


@dataclass
class Submission:
    """One match the phone has handed, or is handing, to the server.

    Attributes:
        submission_id: Server-assigned id, also its directory name.
        idempotency_key: The phone's own per-submission UUID. A retry
            carrying a key already seen returns the original submission
            rather than creating a second one.
        metadata: The validated intake fields (see validate_metadata).
        state: One of awaiting_files / uploading / staged / failed.
        created_at: ISO-8601 timestamp of when the metadata arrived.
        updated_at: ISO-8601 timestamp of the last change.
        slots: Slot name -> its upload progress.
        json_filename: The staged pending JSON, once /complete has run.
        error: Why it failed, if it did.
    """

    submission_id: str
    idempotency_key: str
    metadata: dict[str, object]
    state: str = STATE_AWAITING_FILES
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    slots: dict[str, SlotState] = field(default_factory=dict)
    json_filename: str | None = None
    error: str | None = None

    @property
    def export_slots(self) -> list[str]:
        """The .xlsx slots - a submission needs at least one to be staged."""
        return sorted(name for name in self.slots if name.lower().endswith(".xlsx"))

    @property
    def video_slots(self) -> list[str]:
        """Every non-export slot, i.e. the footage."""
        return sorted(name for name in self.slots if not name.lower().endswith(".xlsx"))

    @property
    def incomplete_slots(self) -> list[str]:
        """Slots still missing bytes, so /complete must refuse."""
        return sorted(name for name, slot in self.slots.items() if not slot.complete)

    def to_dict(self) -> dict[str, object]:
        """Returns the JSON-safe view used both on disk and over the API."""
        return {
            "submission_id": self.submission_id,
            "idempotency_key": self.idempotency_key,
            "metadata": dict(self.metadata),
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "files": [self.slots[name].to_dict() for name in sorted(self.slots)],
            "json_filename": self.json_filename,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Submission:
        """Rebuilds a submission from its on-disk record.

        Args:
            data: The parsed submission.json contents.

        Returns:
            The reconstructed Submission.

        Raises:
            KeyError, TypeError, ValueError: If the record is malformed -
            the store treats any of these as "skip this one".
        """
        slots = {}
        for raw in data.get("files", []):
            slots[raw["slot"]] = SlotState(
                slot=raw["slot"],
                total_bytes=int(raw["total_bytes"]),
                received=[(int(start), int(end)) for start, end in raw.get("received", [])],
            )
        return cls(
            submission_id=data["submission_id"],
            idempotency_key=data["idempotency_key"],
            metadata=dict(data.get("metadata", {})),
            state=data.get("state", STATE_AWAITING_FILES),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            slots=slots,
            json_filename=data.get("json_filename"),
            error=data.get("error"),
        )


@dataclass(frozen=True)
class ChunkResult:
    """The outcome of one chunk upload.

    Attributes:
        slot: That slot's progress after the chunk.
        duplicate: True if the server already held every byte of it, so
            nothing was written. Re-sending a chunk is a no-op by design:
            a phone that lost the response has no way to know whether its
            bytes landed, and must be free to just send them again.
    """

    slot: SlotState
    duplicate: bool

    def to_dict(self) -> dict[str, object]:
        """Returns the JSON-safe view the phone reads to decide what's next."""
        return {**self.slot.to_dict(), "duplicate": self.duplicate}


class SubmissionStore:
    """Submissions on disk, one directory each.

    Reads always come off disk rather than a cache, so a restart, a second
    worker thread, or a hand-edited record all see the same state. The lock
    only serialises this process's writes.

    Attributes:
        root: The directory holding one subdirectory per submission.
    """

    def __init__(self, root: Path) -> None:
        """Builds a store over a directory, which need not exist yet.

        Args:
            root: Where submissions live (WebAppConfig.submissions_dir).
        """
        self.root = root
        self._lock = Lock()

    # --- paths ---------------------------------------------------------

    def _directory(self, submission_id: str) -> Path | None:
        """Resolves a submission id to its directory.

        Args:
            submission_id: The id from the URL.

        Returns:
            The directory, or None if the id isn't one this store mints -
            which is also the path-traversal guard, since nothing but 32
            hex characters is ever turned into a path.
        """
        if not _SUBMISSION_ID.match(submission_id or ""):
            return None
        return self.root / submission_id

    def slot_path(self, submission_id: str, slot: str) -> Path:
        """Returns where one slot's bytes are assembled.

        Args:
            submission_id: The submission's id.
            slot: The slot name.

        Returns:
            The file path.

        Raises:
            SubmissionError: If the id or slot name is unusable.
        """
        directory = self._directory(submission_id)
        if directory is None:
            raise SubmissionError(f"Unknown submission: {submission_id}.")
        return directory / safe_slot(slot)

    # --- records -------------------------------------------------------

    def _record_path(self, submission_id: str) -> Path | None:
        directory = self._directory(submission_id)
        return None if directory is None else directory / _RECORD_NAME

    def _save(self, submission: Submission) -> None:
        """Writes a submission's record atomically.

        Args:
            submission: The record to persist.
        """
        submission.updated_at = _now()
        path = self._record_path(submission.submission_id)
        if path is None:
            raise SubmissionError(f"Unknown submission: {submission.submission_id}.")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = submission.to_dict()
        # The API view reports derived progress only; the record has to
        # keep the raw spans, since they are what makes a resume exact.
        payload["files"] = [
            {**submission.slots[name].to_dict(),
             "received": [list(span) for span in submission.slots[name].received]}
            for name in sorted(submission.slots)
        ]
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)

    def get(self, submission_id: str) -> Submission | None:
        """Loads one submission.

        Args:
            submission_id: The id from the URL.

        Returns:
            The submission, or None if the id is unknown or unreadable.
        """
        path = self._record_path(submission_id)
        if path is None or not path.exists():
            return None
        try:
            return Submission.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, KeyError):
            logger.warning("Skipping an unreadable submission record: %s", path)
            return None

    def list(self) -> list[Submission]:
        """Lists every submission on disk, newest last.

        Returns:
            One Submission per readable record; a half-written or
            hand-edited one is skipped rather than fatal, same as
            api.py's /api/pending listing.
        """
        if not self.root.is_dir():
            return []
        found = []
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir():
                continue
            submission = self.get(directory.name)
            if submission is not None:
                found.append(submission)
        return sorted(found, key=lambda s: s.created_at)

    def create(self, metadata: dict[str, object], idempotency_key: str) -> tuple[Submission, bool]:
        """Records a new submission, or returns the one this key already made.

        Args:
            metadata: Already-validated metadata (see validate_metadata).
            idempotency_key: The phone's per-submission UUID.

        Returns:
            (submission, created). `created` is False when the key was
            already accepted - the original record is returned untouched,
            so a retry can never rewrite metadata the server has, nor
            stage the same match twice under two names.

        Raises:
            SubmissionError: If no idempotency key was given.
        """
        key = str(idempotency_key or "").strip()
        if not key:
            raise SubmissionError("Missing required field: idempotency_key.")
        with self._lock:
            for existing in self.list():
                if existing.idempotency_key == key:
                    return existing, False
            submission = Submission(
                submission_id=uuid.uuid4().hex,
                idempotency_key=key,
                metadata=dict(metadata),
            )
            self._save(submission)
        logger.info("Accepted submission %s (awaiting files)", submission.submission_id)
        return submission, True

    def delete(self, submission_id: str) -> bool:
        """Abandons a submission, chunks and all.

        Args:
            submission_id: The submission to remove.

        Returns:
            True if something was removed, False if the id was unknown.
        """
        directory = self._directory(submission_id)
        if directory is None or not directory.is_dir():
            return False
        with self._lock:
            shutil.rmtree(directory)
        logger.info("Deleted submission %s", submission_id)
        return True

    # --- chunks --------------------------------------------------------

    def write_chunk(
        self, submission_id: str, slot: str, start: int, end: int, total: int, data: bytes
    ) -> ChunkResult:
        """Writes one chunk of one slot at its byte offset.

        The file is written sparsely (seek, then write), so chunks may
        arrive in any order and the assembled bytes are identical either
        way. A chunk the server already holds in full is skipped.

        Args:
            submission_id: The submission being uploaded to.
            slot: The slot (filename) this chunk belongs to.
            start: First byte offset, inclusive.
            end: Last byte offset, inclusive.
            total: The whole file's size, from Content-Range.
            data: The chunk's bytes.

        Returns:
            The slot's progress, and whether this chunk was a duplicate.

        Raises:
            SubmissionError: If the submission or slot name is unusable,
                or the body's length disagrees with the declared range.
            SubmissionConflict: If the slot's total size changed between
                chunks, or the submission has already been staged.
        """
        if len(data) != end - start + 1:
            raise SubmissionError(
                f"Chunk is {len(data)} bytes but Content-Range declares {end - start + 1}."
            )
        if end >= total or end < start or total <= 0:
            raise SubmissionError(
                f"Content-Range {start}-{end}/{total} reaches past the file's size."
            )
        name = safe_slot(slot)
        with self._lock:
            submission = self.get(submission_id)
            if submission is None:
                raise SubmissionError(f"Unknown submission: {submission_id}.")
            if submission.state == STATE_STAGED:
                raise SubmissionConflict(
                    "This submission has already been staged; its files can't be changed."
                )
            state = submission.slots.get(name)
            if state is None:
                state = SlotState(slot=name, total_bytes=total)
                submission.slots[name] = state
            elif state.total_bytes != total:
                raise SubmissionConflict(
                    f"{name} was declared as {state.total_bytes} bytes, now {total}."
                )

            duplicate = any(
                span_start <= start and end <= span_end for span_start, span_end in state.received
            )
            if not duplicate:
                path = self.slot_path(submission_id, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "r+b" if path.exists() else "wb") as handle:
                    handle.seek(start)
                    handle.write(data)
                state.received = merge_range(state.received, (start, end))
            if submission.state == STATE_AWAITING_FILES:
                submission.state = STATE_UPLOADING
            self._save(submission)
        return ChunkResult(slot=state, duplicate=duplicate)

    # --- completion ----------------------------------------------------

    def mark_staged(self, submission_id: str, json_filename: str) -> Submission | None:
        """Records that /complete staged this submission's pending JSON.

        Args:
            submission_id: The submission.
            json_filename: The staged pending-JSON filename.

        Returns:
            The updated submission, or None if the id is unknown.
        """
        with self._lock:
            submission = self.get(submission_id)
            if submission is None:
                return None
            submission.state = STATE_STAGED
            submission.json_filename = json_filename
            submission.error = None
            self._save(submission)
        return submission

    def mark_failed(self, submission_id: str, error: str) -> Submission | None:
        """Records why a submission couldn't be staged, so it can be retried.

        Args:
            submission_id: The submission.
            error: The reason, shown verbatim to the user.

        Returns:
            The updated submission, or None if the id is unknown.
        """
        with self._lock:
            submission = self.get(submission_id)
            if submission is None:
                return None
            submission.state = STATE_FAILED
            submission.error = error
            self._save(submission)
        return submission

    def clear_files(self, submission_id: str) -> None:
        """Deletes a submission's assembled files, keeping its record.

        Called once the files have moved into data/uploads/ and
        data/media/, where the existing archiving and pairing rules own
        them. The record itself stays, because the dashboard still lists
        the submission and names the pending JSON it became.

        Args:
            submission_id: The submission whose staging area to empty.
        """
        directory = self._directory(submission_id)
        if directory is None or not directory.is_dir():
            return
        with self._lock:
            for path in directory.iterdir():
                if path.is_file() and path.name != _RECORD_NAME:
                    path.unlink()


def safe_slot(slot: str) -> str:
    """Validates a slot name, which is also the filename it will be saved as.

    Sanitizing is not enough on its own here: a name that *changes* under
    secure_filename means the client sent something it shouldn't have
    (a path, a traversal, a reserved name), and silently accepting the
    sanitized version would also break the filename-pairing rule the slot
    name exists to feed. So anything that isn't already safe is refused.

    Args:
        slot: The slot name from the URL.

    Returns:
        The slot name, unchanged.

    Raises:
        SubmissionError: If the name is empty, unsafe, or reserved.
    """
    name = (slot or "").strip()
    if not name or name == _RECORD_NAME or secure_filename(name) != name:
        raise SubmissionError(f"Unusable filename for an upload slot: {slot!r}.")
    return name


@contextmanager
def assembled_uploads(store: SubmissionStore, submission: Submission):
    """Presents a submission's assembled files as uploaded files.

    This is what lets /complete go through exactly the same staging path
    as the browser's intake form: webapp/uploads.py pairs and saves
    FileStorage objects, so the assembled chunks are handed over as
    FileStorage objects too, named as the phone named them. No parallel
    ingest or pairing logic exists.

    Args:
        store: The store holding the assembled files.
        submission: The submission to open.

    Yields:
        (xlsx_files, video_files), both lists of FileStorage, in slot-name
        order - pair_uploads applies the natural sort and pairing itself.
    """
    streams = []
    try:
        xlsx_files = []
        for name in submission.export_slots:
            stream = store.slot_path(submission.submission_id, name).open("rb")
            streams.append(stream)
            xlsx_files.append(FileStorage(stream=stream, filename=name))
        video_files = []
        for name in submission.video_slots:
            stream = store.slot_path(submission.submission_id, name).open("rb")
            streams.append(stream)
            video_files.append(FileStorage(stream=stream, filename=name))
        yield xlsx_files, video_files
    finally:
        for stream in streams:
            stream.close()
