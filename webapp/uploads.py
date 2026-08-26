"""Saves uploaded files to disk with sanitized, non-traversable paths.

werkzeug's secure_filename strips path separators, ".." segments, and
non-ASCII-unsafe characters from user-supplied filenames before anything
touches the filesystem - a hostile upload filename (e.g. containing "../")
can't escape uploads_dir/media_dir.
"""

from __future__ import annotations

from pathlib import Path

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from swingvision_import.review import slugify_filename


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
    files: list[FileStorage], *, date: str, opponent: str, media_dir: Path
) -> list[Path]:
    """Saves uploaded match video into a per-match subdirectory of media_dir.

    Args:
        files: Uploaded video files (may be empty - video is optional).
        date: The match date, used to namespace the per-match subdirectory.
        opponent: The opponent's name, used to namespace the subdirectory.
        media_dir: Base directory for all matches' video; created if it
            doesn't exist.

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
        dest = match_dir / filename
        file.save(dest)
        saved.append(dest)
    return saved
