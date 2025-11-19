#!/usr/bin/env python3
"""
Print a mapping between raw DAPHNE Ethernet channels and offline channel IDs.

Fragment type detection decides whether to use stream or self-trigger unpackers,
and the detchannelmaps plugin translates each raw channel into an offline ID.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Iterable, Optional, Tuple

import numpy as np

import detchannelmaps
import detdataformats
from daqdataformats import FragmentType
from fddetdataformats import (
    DAPHNEEthFrame,
    DAPHNEEthStreamFrame,
    DAPHNEFrame,
    DAPHNEStreamFrame,
)
from hdf5libs import HDF5RawDataFile
from rawdatautils.unpack import daphne as daphne_unpack
try:
    from rawdatautils.unpack import daphneeth as daphneeth_unpack
except ImportError:  # pragma: no cover
    daphneeth_unpack = None

from waffles.utils.daphne_helpers import select_records

STREAM_TYPES = {
    FragmentType.kDAPHNEStream,
    FragmentType.kDAPHNEEthStream,
}


def _first_frame_header(fragment_type: FragmentType, payload: bytes):
    if fragment_type == FragmentType.kDAPHNE:
        return DAPHNEFrame(payload[: DAPHNEFrame.sizeof()])
    if fragment_type == FragmentType.kDAPHNEEth:
        return DAPHNEEthFrame(payload[: DAPHNEEthFrame.sizeof()])
    if fragment_type == FragmentType.kDAPHNEEthStream:
        return DAPHNEEthStreamFrame(payload[: DAPHNEEthStreamFrame.sizeof()])
    if fragment_type == FragmentType.kDAPHNEStream:
        return DAPHNEStreamFrame(payload[: DAPHNEStreamFrame.sizeof()])
    raise ValueError(f"Unsupported fragment type: {fragment_type}")


def _raw_channels_for_fragment(fragment, fragment_type: FragmentType, max_frames: Optional[int]) -> Iterable[int]:
    module = daphne_unpack
    if fragment_type not in {FragmentType.kDAPHNE, FragmentType.kDAPHNEStream}:
        if daphneeth_unpack is None:
            raise RuntimeError(
                "rawdatautils.unpack.daphneeth is required to decode DAPHNE Ethernet fragments"
            )
        module = daphneeth_unpack

    if fragment_type in STREAM_TYPES:
        channels = module.np_array_channels_stream(fragment)
        if max_frames is not None:
            channels = channels[:max_frames, :]
        return np.asarray(channels).reshape(-1)

    channels = module.np_array_channels(fragment)
    if max_frames is not None:
        channels = channels[:max_frames]
    return np.asarray(channels).reshape(-1)


def _iter_fragment_mappings(
    filepath: str,
    detector: str,
    channel_map_plugin: str,
    *,
    skip_records: int = 0,
    max_records: Optional[int] = None,
    frames_per_fragment: Optional[int] = None,
) -> Iterable[Tuple[int, int, int]]:
    det_enum = detdataformats.DetID.string_to_subdetector(detector)
    channel_map = detchannelmaps.make_pds_map(channel_map_plugin)

    h5_file = HDF5RawDataFile(filepath)
    records = list(h5_file.get_all_record_ids())
    selected_records = select_records(records, skip_records, max_records)

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

            header = fragment.get_header()
            fragment_type = FragmentType(header.fragment_type)

            if fragment_type not in {
                FragmentType.kDAPHNE,
                FragmentType.kDAPHNEEth,
                FragmentType.kDAPHNEStream,
                FragmentType.kDAPHNEEthStream,
            }:
                continue

            payload = fragment.get_data_bytes()
            if not payload:
                continue

            frame = _first_frame_header(fragment_type, payload)
            daq_header = frame.get_daqheader()
            det_id = int(daq_header.det_id)
            crate_id = int(daq_header.crate_id)
            slot_id = int(daq_header.slot_id)
            stream_id = int(getattr(daq_header, "stream_id", getattr(daq_header, "link_id", 0)))

            raw_channels = _raw_channels_for_fragment(fragment, fragment_type, frames_per_fragment)
            for raw_channel in raw_channels:
                offline = channel_map.get_offline_channel_from_det_crate_slot_stream_chan(
                    det_id, crate_id, slot_id, stream_id, int(raw_channel)
                )
                yield slot_id, int(raw_channel), int(offline)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize raw-versus-offline DAPHNE channels for a raw HDF5 file.",
    )
    parser.add_argument("hdf5_file", help="Path to the input raw HDF5 file.")
    parser.add_argument(
        "--detector",
        default="HD_PDS",
        help="Detector string understood by detdataformats (default: %(default)s).",
    )
    parser.add_argument(
        "--channel-map-plugin",
        default="SimplePDSChannelMap",
        help="detchannelmaps plugin used to translate channels (default: %(default)s).",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Only inspect the first N Trigger Records.",
    )
    parser.add_argument(
        "--skip-records",
        type=int,
        default=0,
        help="Number of Trigger Records to skip from the beginning of the file.",
    )
    parser.add_argument(
        "--frames-per-fragment",
        type=int,
        default=None,
        help="Limit the number of frames examined per fragment (default: all).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Maximum number of rows to print from the mapping table.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    counts: Counter = Counter()
    total_entries = 0

    for slot, raw_channel, offline_channel in _iter_fragment_mappings(
        filepath=args.hdf5_file,
        detector=args.detector,
        channel_map_plugin=args.channel_map_plugin,
        skip_records=args.skip_records,
        max_records=args.max_records,
        frames_per_fragment=args.frames_per_fragment,
    ):
        counts[(slot, raw_channel, offline_channel)] += 1
        total_entries += 1

    if not counts:
        print("No DAPHNE Ethernet data found.")
        return 1

    print(
        f"Processed {total_entries} frame/channel entries across "
        f"{len(counts)} unique slot/raw/offline combinations."
    )
    print("\nslot  raw_ch  offline_ch  count   share")
    print("-------------------------------------------")

    rows_shown = 0
    for (slot, raw_ch, offline_ch), count in counts.most_common():
        if args.top is not None and rows_shown >= args.top:
            remaining = len(counts) - rows_shown
            if remaining > 0:
                print(f"... ({remaining} additional entries)")
            break
        share = 100.0 * count / total_entries if total_entries > 0 else 0.0
        print(f"{slot:4d}  {raw_ch:6d}  {offline_ch:11d}  {count:5d}  {share:6.2f}%")
        rows_shown += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
