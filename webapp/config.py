"""Configuration for the intake web UI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_MAX_UPLOAD_MB = 32 * 1024  # 32 GB


def _max_upload_bytes() -> int:
    """Reads the upload cap from RALLYAI_MAX_UPLOAD_MB, in megabytes.

    Returns:
        The cap in bytes, falling back to the default if the variable is
        unset or isn't a positive integer - a typo in an env var should
        not silently shrink the limit to something that rejects uploads.
    """
    raw = os.environ.get("RALLYAI_MAX_UPLOAD_MB", "")
    try:
        megabytes = int(raw)
    except ValueError:
        megabytes = _DEFAULT_MAX_UPLOAD_MB
    if megabytes <= 0:
        megabytes = _DEFAULT_MAX_UPLOAD_MB
    return megabytes * 1024 * 1024


@dataclass(frozen=True)
class WebAppConfig:
    """Settings for where uploaded files land.

    Attributes:
        uploads_dir: Where uploaded SwingVision .xlsx exports are staged
            before parsing (gitignored - personal match data, not code).
        media_dir: Where uploaded match video is stored, one subdirectory
            per match (gitignored - personal match data, not code).
        submissions_dir: Where the phone companion's in-flight submissions
            live - one subdirectory per submission, holding its record and
            its partially-uploaded files (gitignored - personal match data,
            not code). Files move out of here into uploads_dir/media_dir on
            completion, so the existing archiving and pairing rules apply
            unchanged.
        pairing_code_ttl_seconds: How long a pairing code shown on the
            laptop stays redeemable. Short on purpose: it is a one-time
            secret typed or scanned off the screen, and the long-lived
            thing it hands back (a device token) is what the phone keeps.
        max_content_length: Flask's upload size cap, in bytes, covering
            the whole multipart body - every export and every clip in one
            request. Match footage is what makes this large: an hour of
            1080p is comfortably several GB, and two matches at once
            doubles it. Override with RALLYAI_MAX_UPLOAD_MB when even this
            isn't enough; the cap exists at all so a runaway request
            fails fast instead of filling the disk.
    """

    uploads_dir: Path = _REPO_ROOT / "data" / "uploads"
    media_dir: Path = _REPO_ROOT / "data" / "media"
    submissions_dir: Path = _REPO_ROOT / "data" / "submissions"
    pairing_code_ttl_seconds: int = 300
    max_content_length: int = field(default_factory=_max_upload_bytes)

    @property
    def pairing_path(self) -> Path:
        """Where paired device tokens and outstanding codes are kept.

        Derived from submissions_dir rather than configured separately, so
        the one gitignored directory covers every bit of phone-companion
        state and a test only has to redirect one field.

        Returns:
            The pairing state file's path.
        """
        return self.submissions_dir / "pairing.json"
