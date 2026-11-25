"""Re-opening a staged match's original export(s), wherever they now live.

A staged record points at its exports one of two ways: loose paths on
disk (how ingest() leaves them, and how the CLI keeps them, since those
are the user's own files), or member names inside a zip once
pipeline.archive_sources() has packed them. Every consumer that re-parses
a match - the suggestion pass, the shot map - needs to handle both, so
the handling lives here once rather than being copied per caller.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from .parse import SwingVisionParser
from .raw import RawMatchExport, RawSettings, RawShotRow
from .records import MatchRecord


def source_names(record: MatchRecord) -> list[str]:
    """Lists a record's export references, in play order.

    Args:
        record: The staged match record.

    Returns:
        Paths (unarchived) or archive member names (archived); empty if
        the record has no source reference at all.
    """
    if record.source_files:
        return list(record.source_files)
    if record.source_file:
        return [record.source_file]
    return []


def tracker_export(record: MatchRecord) -> RawMatchExport | None:
    """Gives back the shots the independent tracker staged a match from.

    A match staged by `tracking.pipeline` has no export: its source is a
    video, and the shot rows the tracker emitted are kept in the record's
    `tracking` block. They are what every re-reading consumer - the
    suggestion pass, the footage check - needs, so they stand in for the
    export here.

    Args:
        record: The staged match record.

    Returns:
        The tracker's rows and names, or None for a match imported from a
        SwingVision export.
    """
    block = record.tracking
    if not block or block.get("shots") is None:
        return None
    names = block.get("settings") or {}
    settings = None
    if names.get("host_name") and names.get("guest_name"):
        settings = RawSettings(host_name=names["host_name"], guest_name=names["guest_name"])
    return RawMatchExport(settings=settings, shots=[RawShotRow(**row) for row in block["shots"]])


def parse_sources(record: MatchRecord, parser: SwingVisionParser) -> list[RawMatchExport]:
    """Re-parses every export a staged record was built from, in play order.

    Args:
        record: The staged match record.
        parser: The parser to re-read with.

    Returns:
        One RawMatchExport per source file, in play order - or, for a
        match the independent tracker staged, its own rows
        (`tracker_export`), never the video re-read. Empty if the
        record has nothing to re-parse - callers treat that as "nothing
        usable here", not an error, exactly as before.
    """
    tracked = tracker_export(record)
    if tracked is not None:
        return [tracked]
    names = source_names(record)
    if not names:
        return []
    if record.source_archive:
        # Members are read wholly into memory rather than extracted: an
        # export is tens of KB, and it keeps re-parsing free of temp files.
        with zipfile.ZipFile(record.source_archive) as archive:
            return [parser.parse(io.BytesIO(archive.read(name))) for name in names]
    return [parser.parse(Path(name)) for name in names]
