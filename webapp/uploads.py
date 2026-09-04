"""Saves uploaded files to disk with sanitized, non-traversable paths.

werkzeug's secure_filename strips path separators, ".." segments, and
non-ASCII-unsafe characters from user-supplied filenames before anything
touches the filesystem - a hostile upload filename (e.g. containing "../")
can't escape uploads_dir/media_dir.

Also the home of the export/footage pairing rule for a multi-part match
(a recording cut and resumed): each SwingVision export is matched to the
video of the same stem, so "set2.xlsx" and "set2.mp4" travel together and
the parts reach ingest_multi_part in the order their filenames imply -
not the order the browser happened to hand them over in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from swingvision_import.review import slugify_filename

_DIGITS = re.compile(r"(\d+)")


def part_key(filename: str) -> str:
    """Reduces an uploaded filename to the token that pairs it with others.

    The sanitized stem, lowercased: "Set 2.xlsx" and "set_2.MP4" both
    reduce to "set_2" and so belong to the same part. Extension and case
    are dropped because the two files come off different devices (the
    SwingVision app and a phone's camera roll) and rarely agree on either.

    Args:
        filename: The client-supplied filename.

    Returns:
        The pairing key, or "" if nothing survived sanitizing.
    """
    return Path(secure_filename(filename or "")).stem.lower()


def _natural_key(key: str) -> tuple[object, ...]:
    """Sort key that orders embedded numbers numerically.

    Plain lexicographic order puts "part10" before "part2", which would
    hand ingest_multi_part the parts out of play order - the one thing it
    can't recover from on its own.

    Args:
        key: A pairing key from part_key.

    Returns:
        A tuple of alternating text/number segments, safe to sort.
    """
    return tuple(int(part) if part.isdigit() else part for part in _DIGITS.split(key))


@dataclass(frozen=True)
class MatchPart:
    """One export and the footage that belongs with it.

    Attributes:
        index: 1-based position in play order.
        key: The shared pairing key (see part_key).
        xlsx: The SwingVision export for this part.
        videos: Footage whose filename stem matches this part's export.
    """

    index: int
    key: str
    xlsx: FileStorage
    videos: list[FileStorage]


@dataclass(frozen=True)
class UploadPairing:
    """The result of pairing a submission's exports with its footage.

    Attributes:
        parts: Every export, in play order, with its matched footage.
        unpaired_videos: Footage that matched no export's filename.
        notes: Human-readable, informational only - never blocks an
            import, same as MatchRecord.import_notes.
    """

    parts: list[MatchPart]
    unpaired_videos: list[FileStorage]
    notes: list[str]

    @property
    def is_multi_part(self) -> bool:
        """Whether this submission is an interrupted, multi-file recording."""
        return len(self.parts) > 1


def pair_uploads(
    xlsx_files: list[FileStorage], video_files: list[FileStorage]
) -> UploadPairing:
    """Pairs each export with the footage sharing its filename, in play order.

    A single export is the trivial case: it is the whole match, so every
    uploaded clip belongs to it and filenames are not required to agree.
    With two or more exports the filename *is* the ordering and the link
    to footage, so anything that doesn't line up gets reported back.

    Args:
        xlsx_files: The uploaded SwingVision exports (at least one).
        video_files: The uploaded footage; may be empty (video is optional).

    Returns:
        The pairing, including notes about anything that didn't match.
    """
    ordered = sorted(xlsx_files, key=lambda f: _natural_key(part_key(f.filename or "")))
    notes: list[str] = []

    if len(ordered) < 2:
        parts = [
            MatchPart(index=1, key=part_key(f.filename or ""), xlsx=f, videos=list(video_files))
            for f in ordered
        ]
        return UploadPairing(parts=parts, unpaired_videos=[], notes=notes)

    by_key: dict[str, list[FileStorage]] = {}
    for video in video_files:
        by_key.setdefault(part_key(video.filename or ""), []).append(video)

    parts: list[MatchPart] = []
    for index, xlsx in enumerate(ordered, start=1):
        key = part_key(xlsx.filename or "")
        parts.append(MatchPart(index=index, key=key, xlsx=xlsx, videos=by_key.pop(key, [])))

    unpaired = [video for videos in by_key.values() for video in videos]

    notes.append(
        "Part order read from filenames: "
        + ", ".join(f"{part.index}. {part.xlsx.filename}" for part in parts)
    )
    if video_files:
        missing = [part for part in parts if not part.videos]
        if missing:
            notes.append(
                "No footage matched these exports by filename: "
                + ", ".join(part.xlsx.filename or "(unnamed)" for part in missing)
            )
        if unpaired:
            notes.append(
                "These clips matched no export filename and were kept unpaired: "
                + ", ".join(video.filename or "(unnamed)" for video in unpaired)
            )
        if not missing and not unpaired:
            notes.append(
                f"Every part's export and footage matched by filename ({len(parts)} parts)."
            )

    return UploadPairing(parts=parts, unpaired_videos=unpaired, notes=notes)


def match_slug(date: str, opponent: str) -> str:
    """Builds the same filesystem-safe match identifier the savers below use.

    Public (not just an internal helper) since api.py's Film Review media
    lookup needs to reconstruct the same per-match subdirectory name that
    save_uploaded_videos originally saved into, from just a match's
    date/opponent.

    Args:
        date: The match date.
        opponent: The opponent's name.

    Returns:
        A filesystem-safe "<date>_<slugified opponent>" identifier.
    """
    return f"{date}_{slugify_filename(opponent)}"


def save_uploaded_xlsx(
    file: FileStorage, *, date: str, opponent: str, uploads_dir: Path
) -> Path:
    """Saves an uploaded SwingVision export to uploads_dir.

    Args:
        file: The uploaded .xlsx file.
        date: The match date, used to namespace the saved filename.
        opponent: The opponent's name, used to namespace the saved filename.
        uploads_dir: Directory to save into; created if it doesn't exist.

    Returns:
        Path to the saved file, safe to pass to SwingVisionParser.parse.
    """
    uploads_dir.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file.filename or "") or "match.xlsx"
    dest = uploads_dir / f"{match_slug(date, opponent)}_{filename}"
    file.save(dest)
    return dest


def save_uploaded_videos(
    files: list[FileStorage],
    *,
    date: str,
    opponent: str,
    media_dir: Path,
    prefix: str = "",
) -> list[Path]:
    """Saves uploaded match video into a per-match subdirectory of media_dir.

    Args:
        files: Uploaded video files (may be empty - video is optional).
        date: The match date, used to namespace the per-match subdirectory.
        opponent: The opponent's name, used to namespace the subdirectory.
        media_dir: Base directory for all matches' video; created if it
            doesn't exist.
        prefix: Optional sanitized prefix prepended to each saved filename
            (see save_paired_videos). Empty by default, so a single-part
            match keeps the exact filename the user uploaded.

    Returns:
        Paths to every file actually saved (skips empty file inputs).
    """
    match_dir = media_dir / match_slug(date, opponent)
    saved: list[Path] = []
    for file in files:
        if not file or not file.filename:
            continue
        filename = secure_filename(file.filename)
        if not filename:
            continue
        match_dir.mkdir(parents=True, exist_ok=True)
        dest = match_dir / f"{prefix}{filename}"
        file.save(dest)
        saved.append(dest)
    return saved


def save_paired_videos(
    pairing: UploadPairing, *, date: str, opponent: str, media_dir: Path
) -> list[Path]:
    """Saves a pairing's footage so its filenames carry the part order.

    A multi-part match's clips are saved as "partNN__<original name>", and
    anything that matched no export as "unpaired__<original name>". Film
    Review lists a match's media with a plain sorted() over the directory,
    so encoding the order into the name is what makes clips play back in
    the order they were recorded - and "p" sorting before "u" keeps the
    unpaired stragglers at the end rather than interleaved.

    A single-part match is written exactly as before, with no prefix:
    there is only one part, so there is no order to encode and no reason
    to rename the user's file.

    Args:
        pairing: The result of pair_uploads.
        date: The match date, used to namespace the per-match subdirectory.
        opponent: The opponent's name, used to namespace the subdirectory.
        media_dir: Base directory for all matches' video.

    Returns:
        Paths to every file actually saved.
    """
    saved: list[Path] = []
    for part in pairing.parts:
        prefix = f"part{part.index:02d}__" if pairing.is_multi_part else ""
        saved.extend(
            save_uploaded_videos(
                part.videos, date=date, opponent=opponent, media_dir=media_dir, prefix=prefix
            )
        )
    saved.extend(
        save_uploaded_videos(
            pairing.unpaired_videos,
            date=date,
            opponent=opponent,
            media_dir=media_dir,
            prefix="unpaired__",
        )
    )
    return saved
