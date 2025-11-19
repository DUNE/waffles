"""
Helpers to compute high-level statistics for DAPHNE Ethernet waveforms.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import detdataformats
from hdf5libs import HDF5RawDataFile

from rawdatautils.unpack.utils import DAPHNEEthStreamUnpacker

from waffles.data_classes.Waveform import Waveform
from waffles.utils.daphne_helpers import looks_like_daphne_eth, select_records


@dataclass(frozen=True)
class LinkKey:
    """Identify a readout link via detector, crate, slot, and stream numbers."""

    det_id: int
    crate_id: int
    slot_id: int
    stream_id: int

    def __str__(self) -> str:
        return (
            f"det={self.det_id} crate={self.crate_id} "
            f"slot={self.slot_id} stream={self.stream_id}"
        )


@dataclass
class LinkInventory:
    """Summary of which source IDs map onto which hardware links."""

    source_to_link: Dict[int, LinkKey]
    fragment_counts: Counter
    records_considered: int
    fragments_considered: int


@dataclass
class WaveformStats:
    """Container with various aggregate counters derived from a WaveformSet."""

    total_waveforms: int
    endpoint_counts: Counter
    channel_counts: Counter
    record_counts: Counter


def compute_waveform_stats(waveforms: Sequence[Waveform]) -> WaveformStats:
    """Return aggregate counters for the provided waveforms."""
    endpoint_counts: Counter = Counter()
    channel_counts: Counter = Counter()
    record_counts: Counter = Counter()

    for wf in waveforms:
        endpoint_counts[int(wf.endpoint)] += 1
        channel_counts[(int(wf.endpoint), int(wf.channel))] += 1
        record_counts[(int(wf.run_number), int(wf.record_number))] += 1

    return WaveformStats(
        total_waveforms=len(waveforms),
        endpoint_counts=endpoint_counts,
        channel_counts=channel_counts,
        record_counts=record_counts,
    )


def collect_link_inventory(
    filepath: str,
    detector: str,
    *,
    channel_map: Optional[str] = None,
    skip_records: int = 0,
    max_records: Optional[int] = None,
) -> LinkInventory:
    """
    Walk through the requested HDF5 file and build a mapping from source IDs to links.

    The iteration order and skip/limit handling matches ``load_daphne_eth_waveforms`` so
    that statistics can be correlated with the decoded WaveformSet.
    """
    det_enum = detdataformats.DetID.string_to_subdetector(detector)
    h5_file = HDF5RawDataFile(filepath)
    records = list(h5_file.get_all_record_ids())
    selected_records = select_records(records, skip_records, max_records)

    unpacker = DAPHNEEthStreamUnpacker(
        channel_map=channel_map,
        ana_data_prescale=1,
        wvfm_data_prescale=1,
    )

    source_to_link: Dict[int, LinkKey] = {}
    fragment_counts: Counter = Counter()
    fragments_considered = 0

    for record in selected_records:
        try:
            geo_ids = list(h5_file.get_geo_ids_for_subdetector(record, det_enum))
        except Exception:
            continue

        for geo_id in geo_ids:
            try:
                fragment = h5_file.get_frag(record, geo_id)
            except Exception:
                continue
            if fragment.get_data_size() == 0:
                continue
            if not looks_like_daphne_eth(fragment.get_fragment_type()):
                continue

            try:
                det_id, crate_id, slot_id, stream_id = unpacker.get_det_crate_slot_stream(fragment)
            except Exception:
                continue

            link = LinkKey(int(det_id), int(crate_id), int(slot_id), int(stream_id))
            fragment_counts[link] += 1
            fragments_considered += 1

            source_id = int(fragment.get_header().element_id.id)
            previous = source_to_link.get(source_id)
            if previous is None:
                source_to_link[source_id] = link
            elif previous != link:
                raise RuntimeError(
                    f"Source ID {source_id} maps to multiple links: {previous} vs {link}"
                )

    return LinkInventory(
        source_to_link=source_to_link,
        fragment_counts=fragment_counts,
        records_considered=len(selected_records),
        fragments_considered=fragments_considered,
    )


def map_waveforms_to_links(
    stats: WaveformStats,
    inventory: LinkInventory,
) -> Tuple[Counter, set[int]]:
    """
    Use the endpoint counts in ``stats`` to build a per-link waveform counter.

    Returns a tuple consisting of (link_counts, missing_endpoints).
    """
    link_counts: Counter = Counter()
    missing: set[int] = set()

    for endpoint, count in stats.endpoint_counts.items():
        link = inventory.source_to_link.get(endpoint)
        if link is None:
            missing.add(endpoint)
            continue
        link_counts[link] += count

    return link_counts, missing


def compute_slot_channel_counts(
    waveforms: Sequence[Waveform],
    inventory: LinkInventory,
) -> Counter:
    """
    Combine slot information from ``inventory`` with the raw channel stored on each waveform.
    """
    slot_channel_counts: Counter = Counter()

    for wf in waveforms:
        raw_channel = getattr(wf, "raw_channel", None)
        if raw_channel is None:
            continue
        link = inventory.source_to_link.get(int(wf.endpoint))
        if link is None:
            continue
        slot_channel_counts[(int(link.slot_id), int(raw_channel))] += 1

    return slot_channel_counts
