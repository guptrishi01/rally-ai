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
from .raw import RawMatchExport
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


def parse_sources(record: MatchRecord, parser: SwingVisionParser) -> list[RawMatchExport]:
    """Re-parses every export a staged record was built from, in play order.

    Args:
        record: The staged match record.
        parser: The parser to re-read with.

    Returns:
        One RawMatchExport per source file, in play order. Empty if the
        record has nothing to re-parse - callers treat that as "nothing
        usable here", not an error, exactly as before.
    """
    names = source_names(record)
    if not names:
        return []
    if record.source_archive:
        # Members are read wholly into memory rather than extracted: an
        # export is tens of KB, and it keeps re-parsing free of temp files.
        with zipfile.ZipFile(record.source_archive) as archive:
            return [parser.parse(io.BytesIO(archive.read(name))) for name in names]
    return [parser.parse(Path(name)) for name in names]
