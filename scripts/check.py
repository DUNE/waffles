#!/usr/bin/env python3
"""
Low-level inspector for a single fragment inside a raw DAQ HDF5 file.

The script is intentionally verbose: it prints the Trigger Record/GeoID that
produced the fragment, dumps header information, optionally writes the raw
payload to disk, and runs the DAPHNE Ethernet unpacker so that the decoded
waveform content can be examined without any additional glue code.

Typical usage:

    ./scripts/check.py /path/to/file.hdf5 --detector VD_CathodePDS

You can pick which Trigger Record / GeoID pair to inspect via
``--record-index`` and ``--geo-index`` once the summary lists the available
entries.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional

import numpy as np

import detdataformats
from daqdataformats import FragmentType, fragment_type_to_string
from hdf5libs import HDF5RawDataFile

from rawdatautils.unpack.utils import DAPHNEEthUnpacker


def _hexdump(data: bytes, *, width: int = 16, limit: Optional[int] = 512) -> str:
    """
    Render ``data`` in a human-friendly hex dump format.

    Parameters
    ----------
    data:
        Byte-like object to print.
    width:
        Number of bytes per line.
    limit:
        Maximum number of bytes to print. Use ``None`` to disable truncation.
    """
    mv = memoryview(data).cast("B")
    end = len(mv) if limit is None else min(len(mv), limit)

    lines = []
    for offset in range(0, end, width):
        chunk = mv[offset : offset + width]
        hex_repr = " ".join(f"{byte:02x}" for byte in chunk)
        ascii_repr = "".join(chr(byte) if 32 <= byte < 127 else "." for byte in chunk)
        lines.append(f"{offset:08x}  {hex_repr:<{width * 3 - 1}}  {ascii_repr}")

    if limit is not None and len(mv) > limit:
        lines.append(f"... truncated after {limit} bytes (payload has {len(mv)} bytes total) ...")
    return "\n".join(lines)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a single fragment (payload included) in a raw HDF5 file."
    )
    parser.add_argument("hdf5_file", help="Path to the input HDF5 file.")
    parser.add_argument(
        "--detector",
        default="HD_PDS",
        help="Detector string understood by detdataformats (default: %(default)s).",
    )
    parser.add_argument(
        "--record-index",
        type=int,
        default=0,
        help="Zero-based index inside the Trigger Record list (default: %(default)s).",
    )
    parser.add_argument(
        "--geo-index",
        type=int,
        default=0,
        help="Zero-based index inside the GeoID list for the chosen record (default: %(default)s).",
    )
    parser.add_argument(
        "--channel-map",
        default=None,
        help="Optional channel map name forwarded to DAPHNEEthUnpacker.",
    )
    parser.add_argument(
        "--payload-dump",
        default=None,
        help="If set, write the raw fragment payload to this path.",
    )
    parser.add_argument(
        "--payload-preview",
        type=int,
        default=256,
        help="Number of payload bytes to show in the console (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _select(items: list, index: int, label: str) -> tuple[int, object]:
    if not items:
        raise RuntimeError(f"No {label} entries available to inspect.")
    if index < 0 or index >= len(items):
        raise IndexError(f"{label} index {index} is outside [0, {len(items) - 1}]")
    return index, items[index]


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    if not os.path.isfile(args.hdf5_file):
        raise FileNotFoundError(f"Input file '{args.hdf5_file}' not found.")

    det_enum = detdataformats.DetID.string_to_subdetector(args.detector)
    h5_file = HDF5RawDataFile(args.hdf5_file)

    records = list(h5_file.get_all_record_ids())
    print(f"File contains {len(records)} Trigger Records.")
    rec_idx, record = _select(records, args.record_index, "record")
    print(f"Selected record #{rec_idx}: {record}")

    geo_ids = list(h5_file.get_geo_ids_for_subdetector(record, det_enum))
    print(f"Record has {len(geo_ids)} GeoIDs for detector '{args.detector}'.")
    geo_idx, geo_id = _select(geo_ids, args.geo_index, "GeoID")
    print(f"Selected GeoID #{geo_idx}: {geo_id}")

    fragment = h5_file.get_frag(record, geo_id)
    header = fragment.get_header()
    fragment_type = fragment.get_fragment_type()
    frag_enum = FragmentType(fragment_type)
    frag_string = fragment_type_to_string(frag_enum)

    print("\n=== Fragment header ===")
    print(f"  Run / Trigger / Seq : {header.run_number} / {header.trigger_number} / {header.sequence_number}")
    print(f"  Detector ID         : {header.detector_id} ({detdataformats.DetID.Subdetector(header.detector_id).name})")
    print(f"  Fragment type       : {frag_enum.name} ({frag_string})")
    print(f"  Source ID           : {header.element_id.id}")
    print(f"  Window [begin, end] : {header.window_begin} - {header.window_end}")
    print(f"  Trigger timestamp   : {fragment.get_trigger_timestamp()}")
    print(f"  Data size           : {fragment.get_data_size()} bytes")

    payload = fragment.get_data_bytes()
    preview_len = len(payload) if args.payload_preview < 0 else args.payload_preview
    print(f"\n=== Payload preview ({min(len(payload), preview_len)} bytes) ===")
    print(_hexdump(payload, limit=preview_len))

    if args.payload_dump:
        with open(args.payload_dump, "wb") as dump_fh:
            dump_fh.write(payload)
        print(f"\nWrote full payload ({len(payload)} bytes) to {args.payload_dump}")

    print("\n=== Attempting DAPHNE Ethernet unpack ===")
    unpacker = DAPHNEEthUnpacker(
        channel_map=args.channel_map,
        ana_data_prescale=1,
        wvfm_data_prescale=1,
    )
    ana_data, wvfm_data = unpacker.get_det_data_all(fragment)

    if ana_data is None and wvfm_data is None:
        print("Unpacker returned no data (check fragment type / prescale settings).")
    else:
        print(f"  Analysis entries : {len(ana_data) if ana_data else 0}")
        print(f"  Waveform entries : {len(wvfm_data) if wvfm_data else 0}")
        if wvfm_data:
            first = wvfm_data[0]
            print(
                f"  First waveform   : timestamp={first.timestamp_dts} "
                f"endpoint={first.src_id} channel={first.channel} "
                f"samples={len(first.adcs)}"
            )
            print(f"  ADC sample preview: {first.adcs[: min(16, len(first.adcs))]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
