"""Pairing codes and device tokens for the phone companion.

The phone has to reach the laptop over the LAN, which is a real loosening
of this app's "localhost only" stance (see webapp/app.py's --lan flag), so
every /api/submissions* call carries a token and nothing mints one without
a code typed or scanned off the laptop first.

Two kinds of secret, with deliberately different lifetimes:

- a **pairing code**, short-lived and single-use, shown on the laptop;
- a **device token**, long-lived, held by one phone, revocable by deleting
  it on the laptop.

State lives in one JSON file on disk, not in memory. A token the user
paired last week has to keep working after a restart - unlike
webapp/jobs.py, whose in-memory state describes a run that a restart
genuinely ends.
"""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

# 8 digits, not 6: the code is normally scanned out of a QR rather than
# typed, so length costs almost nothing, while on a shared Wi-Fi the only
# thing standing between an attacker and a device token is guessing a live
# code inside its 5-minute window. 10^8 makes that hopeless at any request
# rate this single-threaded dev server could answer.
_CODE_DIGITS = 8
_TOKEN_BYTES = 32


def _now() -> datetime:
    """Returns the current UTC time.

    Returns:
        An aware datetime, so stored expiry timestamps round-trip through
        ISO text without losing their offset.
    """
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Device:
    """One paired phone.

    Attributes:
        token: The long-lived bearer token this device sends.
        name: What the device called itself when it paired, for display.
        paired_at: ISO-8601 timestamp of when it paired.
    """

    token: str
    name: str
    paired_at: str


class PairingStore:
    """Disk-backed store of outstanding pairing codes and paired devices.

    Every method re-reads the file rather than caching: the state is tiny
    (a handful of records), and a stale cache would be the difference
    between a revoked token still working and not.

    Attributes:
        path: The JSON file this store reads and writes.
        code_ttl_seconds: How long a freshly minted pairing code stays
            redeemable. A negative value makes every code already expired,
            which is only useful in tests.
    """

    def __init__(self, path: Path, code_ttl_seconds: int = 300) -> None:
        """Builds a store over a state file, which need not exist yet.

        Args:
            path: Where to keep the JSON state.
            code_ttl_seconds: Lifetime of a new pairing code, in seconds.
        """
        self.path = path
        self.code_ttl_seconds = code_ttl_seconds
        self._lock = Lock()

    def _read(self) -> dict:
        """Loads the state file.

        Returns:
            The parsed state, or an empty one if the file is missing or
            unreadable - a corrupt file costs the pairings, not the app.
        """
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"codes": {}, "devices": []}
        if not isinstance(data, dict):
            return {"codes": {}, "devices": []}
        data.setdefault("codes", {})
        data.setdefault("devices", [])
        return data

    def _write(self, data: dict) -> None:
        """Writes the state file atomically.

        Args:
            data: The state to persist.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def new_code(self) -> str:
        """Mints a single-use pairing code for the laptop to display.

        Returns:
            A numeric code, redeemable once within code_ttl_seconds.
        """
        code = f"{secrets.randbelow(10**_CODE_DIGITS):0{_CODE_DIGITS}d}"
        expires_at = (_now() + timedelta(seconds=self.code_ttl_seconds)).isoformat()
        with self._lock:
            data = self._read()
            data["codes"] = {
                existing: value
                for existing, value in data["codes"].items()
                if value.get("expires_at", "") > _now().isoformat()
            }
            data["codes"][code] = {"expires_at": expires_at}
            self._write(data)
        logger.info("Minted a pairing code valid for %ss", self.code_ttl_seconds)
        return code

    def redeem(self, code: str, device_name: str = "phone") -> Device | None:
        """Exchanges a pairing code for a long-lived device token.

        Args:
            code: The code shown on the laptop.
            device_name: What to call this device on the laptop.

        Returns:
            The paired Device, or None if the code is unknown, already
            used, or expired - the three are deliberately indistinguishable
            to the caller, so a wrong guess learns nothing.
        """
        with self._lock:
            data = self._read()
            entry = data["codes"].pop(code, None) if code else None
            if entry is None or entry.get("expires_at", "") <= _now().isoformat():
                if entry is not None:
                    self._write(data)  # drop the expired code either way
                return None
            device = Device(
                token=secrets.token_urlsafe(_TOKEN_BYTES),
                name=device_name or "phone",
                paired_at=_now().isoformat(),
            )
            data["devices"].append(
                {"token": device.token, "name": device.name, "paired_at": device.paired_at}
            )
            self._write(data)
        logger.info("Paired a new device: %s", device.name)
        return device

    def is_paired(self, token: str) -> bool:
        """Whether a bearer token belongs to a paired device.

        Args:
            token: The token from the request's Authorization header.

        Returns:
            True if it matches a paired device. Compared with
            compare_digest so a near-miss token can't be found by timing.
        """
        if not token:
            return False
        return any(
            secrets.compare_digest(token, device["token"]) for device in self._read()["devices"]
        )

    def devices(self) -> list[Device]:
        """Lists every paired device.

        Returns:
            One Device per pairing, oldest first.
        """
        return [
            Device(token=d["token"], name=d.get("name", "phone"), paired_at=d.get("paired_at", ""))
            for d in self._read()["devices"]
        ]

    def revoke(self, token: str) -> bool:
        """Unpairs one device.

        Args:
            token: The token to remove.

        Returns:
            True if a device was removed, False if nothing matched.
        """
        with self._lock:
            data = self._read()
            remaining = [d for d in data["devices"] if d["token"] != token]
            if len(remaining) == len(data["devices"]):
                return False
            data["devices"] = remaining
            self._write(data)
        logger.info("Revoked a device token")
        return True
