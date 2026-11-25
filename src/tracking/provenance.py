"""Where a tracker number came from: measured, interpolated, inferred, typed in.

The tracker's rule (spec section 6.1) is that a smooth-looking value must
never pass for a detected one. Every observation the tracker keeps therefore
carries one of four `Provenance` tags, and anything derived from several
observations carries the *least* trustworthy tag among them (`weakest`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

T = TypeVar("T")


class Provenance(str, Enum):
    """How a value was obtained.

    A `str` enum, so it serialises into the staged JSON as its plain value.

    Attributes:
        MEASURED: Read off the pixels by a detector that was confident.
        INTERPOLATED: Bridged across a short gap between measured values.
        INFERRED: Concluded from rules or context, not observed.
        USER_ENTERED: Typed or clicked by a person during review.
    """

    MEASURED = "measured"
    INTERPOLATED = "interpolated"
    INFERRED = "inferred"
    USER_ENTERED = "user_entered"

    @classmethod
    def parse(cls, value: str | Provenance) -> Provenance:
        """Reads a provenance tag back from its stored form.

        Args:
            value: A `Provenance`, or its string value in any case, with
                spaces or hyphens allowed in place of the underscore.

        Returns:
            The matching member.

        Raises:
            ValueError: If the text names no provenance.
        """
        if isinstance(value, Provenance):
            return value
        cleaned = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        try:
            return cls(cleaned)
        except ValueError:
            known = ", ".join(member.value for member in cls)
            raise ValueError(f"Unknown provenance {value!r}; expected one of: {known}.") from None

    @property
    def is_observed(self) -> bool:
        """Whether a detector or a person actually saw this value."""
        return self in (Provenance.MEASURED, Provenance.USER_ENTERED)


# Most to least trustworthy. A person's correction outranks a detection; a
# rule's conclusion ranks below a bridged gap between two detections.
_TRUST_ORDER = (
    Provenance.USER_ENTERED,
    Provenance.MEASURED,
    Provenance.INTERPOLATED,
    Provenance.INFERRED,
)


def trust_rank(provenance: Provenance) -> int:
    """Ranks a provenance tag, 0 being the most trustworthy.

    Args:
        provenance: The tag to rank.

    Returns:
        Its position in user_entered, measured, interpolated, inferred.
    """
    return _TRUST_ORDER.index(provenance)


def weakest(provenances: Iterable[Provenance]) -> Provenance:
    """Finds the tag a value derived from several inputs must carry.

    A speed computed from one measured and one interpolated position is
    interpolated: a derived value is only as good as its worst input.

    Args:
        provenances: The inputs' tags.

    Returns:
        The least trustworthy tag among them.

    Raises:
        ValueError: If there are no inputs.
    """
    items = list(provenances)
    if not items:
        raise ValueError("weakest() needs at least one provenance.")
    return max(items, key=trust_rank)


def provenance_counts(provenances: Iterable[Provenance]) -> dict[str, int]:
    """Counts tags, for the staged JSON's informational `tracking` block.

    Args:
        provenances: Any number of tags.

    Returns:
        A count for every tag, zero included, keyed by its string value -
        so "nothing was interpolated" is stated rather than left out.
    """
    seen = Counter(provenances)
    return {member.value: seen.get(member, 0) for member in Provenance}


@dataclass(frozen=True)
class Tagged(Generic[T]):
    """A value together with how it was obtained.

    Attributes:
        value: The value itself.
        provenance: How it was obtained.
        confidence: The producer's confidence from 0 to 1, or None when the
            producer does not estimate one.
        source: What produced it - a detector or model name with its
            version - so a result can be reproduced and audited.
    """

    value: T
    provenance: Provenance
    confidence: float | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        """Rejects a confidence outside 0..1."""
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be within 0..1, got {self.confidence}.")

    def to_dict(self) -> dict:
        """Serialises to plain JSON-ready types.

        Returns:
            The four fields, with the provenance as its string value.
        """
        return {
            "value": self.value,
            "provenance": self.provenance.value,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Tagged:
        """Reads a tagged value back from `to_dict`'s form.

        A list value (how JSON stores a tuple, e.g. a court position)
        comes back as a tuple.

        Args:
            data: A dict with "value" and "provenance", and optionally
                "confidence" and "source".

        Returns:
            The tagged value.

        Raises:
            ValueError: If the provenance is unknown or the confidence is
                outside 0..1.
            KeyError: If "value" or "provenance" is missing.
        """
        value = data["value"]
        if isinstance(value, list):
            value = tuple(value)
        return cls(
            value,
            Provenance.parse(data["provenance"]),
            data.get("confidence"),
            data.get("source"),
        )


def measured(value: T, confidence: float | None = None, source: str | None = None) -> Tagged[T]:
    """Tags a value as read off the pixels.

    Args:
        value: The value.
        confidence: The detector's confidence, 0 to 1.
        source: The detector and version that produced it.

    Returns:
        The tagged value.
    """
    return Tagged(value, Provenance.MEASURED, confidence, source)


def interpolated(value: T, confidence: float | None = None, source: str | None = None) -> Tagged[T]:
    """Tags a value as bridged across a short gap.

    Args:
        value: The value.
        confidence: Confidence in the bridge, 0 to 1.
        source: What did the bridging.

    Returns:
        The tagged value.
    """
    return Tagged(value, Provenance.INTERPOLATED, confidence, source)


def inferred(value: T, confidence: float | None = None, source: str | None = None) -> Tagged[T]:
    """Tags a value as concluded from rules or context.

    Args:
        value: The value.
        confidence: Confidence in the conclusion, 0 to 1.
        source: The rule that concluded it.

    Returns:
        The tagged value.
    """
    return Tagged(value, Provenance.INFERRED, confidence, source)


def user_entered(value: T, source: str | None = None) -> Tagged[T]:
    """Tags a value as supplied by a person.

    Args:
        value: The value.
        source: Where it was entered, e.g. "review modal".

    Returns:
        The tagged value, with no confidence: a person's entry is not a
        probability.
    """
    return Tagged(value, Provenance.USER_ENTERED, None, source)
