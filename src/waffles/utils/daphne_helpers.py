"""
Shared helpers for DAPHNE-specific tooling.

The utilities in this module are intentionally tiny so they can be reused by
multiple scripts without having to duplicate the same bookkeeping logic.
"""

from __future__ import annotations

from typing import Optional, Sequence, TypeVar

from daqdataformats import FragmentType, fragment_type_to_string

T = TypeVar("T")

# Historical fragment type codes that carried DAPHNE Ethernet payloads before an
# explicit enum value was added to daqdataformats.
LEGACY_DAPHNE_ETH_TYPES = {
    18,  # kVD_CathodePDS in fddaq-v5.4.x still carries DAPHNE Ethernet payloads
}


def looks_like_daphne_eth(fragment_type: int) -> bool:
    """
    Return ``True`` when ``fragment_type`` encodes a DAPHNE Ethernet fragment.

    The logic mirrors ``rawdatautils`` and falls back to a conservative
    heuristic so that unknown enum values result in a soft failure rather than
    an exception.
    """
    raw_value = int(fragment_type)

    try:
        enum_value = FragmentType(raw_value)
    except ValueError:
        return raw_value in LEGACY_DAPHNE_ETH_TYPES

    enum_name = enum_value.name.lower()
    if "daphne" in enum_name and "eth" in enum_name:
        return True

    if raw_value in LEGACY_DAPHNE_ETH_TYPES:
        return True

    try:
        enum_string = fragment_type_to_string(enum_value).lower()
    except Exception:  # pragma: no cover - defensive fallback
        return False

    return "daphne" in enum_string and "eth" in enum_string


def select_records(records: Sequence[T], skip: int, limit: Optional[int]) -> list[T]:
    """
    Slice the sequence of Trigger Records according to the caller request.

    Parameters
    ----------
    records:
        Ordered collection returned by ``HDF5RawDataFile.get_all_record_ids``.
    skip:
        Number of entries to drop from the start of ``records``.
    limit:
        Optional maximum number of entries to keep after skipping.

    Returns
    -------
    list[T]
        The selected records, preserving the original ordering.
    """
    if skip < 0:
        raise ValueError("skip_records must be non-negative")

    start = min(skip, len(records))
    if limit is None:
        return list(records[start:])
    if limit < 0:
        raise ValueError("max_records must be non-negative when provided")

    end = min(len(records), start + limit)
    return list(records[start:end])
