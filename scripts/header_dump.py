#!/usr/bin/env python3
"""
Quickly dump the header of the first fragment in a raw DAQ HDF5 file.

Unlike ``check.py`` this helper makes no assumptions about the detector: it
opens the file, grabs the first Trigger Record, scans every available
subdetector, and prints the richest header summary it can assemble for the
first fragment that shows up. Optional flags let you override the record,
detector (if you happen to know it), or GeoID index if needed.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional, Sequence, Tuple

import detdataformats
from daqdataformats import FragmentType, fragment_type_to_string
from hdf5libs import HDF5RawDataFile


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the very first fragment header in a raw HDF5 file."
    )
    parser.add_argument("hdf5_file", help="Path to the input HDF5 file.")
    parser.add_argument(
        "--record-index",
        type=int,
        default=0,
        help="Zero-based index inside the Trigger Record list (default: %(default)s).",
    )
    parser.add_argument(
        "--detector",
        help="Optional detector string understood by detdataformats (default: scan all).",
    )
    parser.add_argument(
        "--geo-index",
        type=int,
        default=0,
        help="Zero-based index inside the GeoID list for the chosen detector (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _select(items: Sequence, index: int, label: str):
    if not items:
        raise RuntimeError(f"No {label} entries available to inspect.")
    if index < 0 or index >= len(items):
        raise IndexError(f"{label} index {index} is outside [0, {len(items) - 1}]")
    return items[index]


def _gather_geo_ids(
    h5_file: HDF5RawDataFile, record, detectors: Sequence[detdataformats.DetID.Subdetector]
) -> list[Tuple[detdataformats.DetID.Subdetector, Sequence]]:
    summary: list[Tuple[detdataformats.DetID.Subdetector, Sequence]] = []
    for detector in detectors:
        try:
            geo_ids = list(h5_file.get_geo_ids_for_subdetector(record, detector))
        except RuntimeError:
            continue
        if geo_ids:
            summary.append((detector, geo_ids))
    return summary


def _format_subdetector(detector_id: int) -> str:
    try:
        subdet = detdataformats.DetID.Subdetector(detector_id)
    except ValueError:
        return f"{detector_id} (unknown)"
    return f"{detector_id} ({subdet.name})"


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    if not os.path.isfile(args.hdf5_file):
        raise FileNotFoundError(f"Input file '{args.hdf5_file}' not found.")

    h5_file = HDF5RawDataFile(args.hdf5_file)

    records = list(h5_file.get_all_record_ids())
    if not records:
        raise RuntimeError("File contains no Trigger Records.")
    print(f"File contains {len(records)} Trigger Records.")

    record = _select(records, args.record_index, "record")
    print(f"Selected record #{args.record_index}: {record}")

    if args.detector:
        try:
            detectors = [detdataformats.DetID.string_to_subdetector(args.detector)]
        except Exception as exc:  # detdataformats raises RuntimeError on bad input
            raise ValueError(f"Unknown detector string '{args.detector}'") from exc
    else:
        detectors = list(detdataformats.DetID.Subdetector)

    geo_summary = _gather_geo_ids(h5_file, record, detectors)
    if not geo_summary:
        raise RuntimeError("No fragments found in the selected record.")

    print("\nGeoID inventory for this record:")
    for detector, geo_ids in geo_summary:
        print(f"  {detector.name:<16} : {len(geo_ids)} entries")

    detector, geo_ids = geo_summary[0]
    geo_id = _select(geo_ids, args.geo_index, f"GeoID for detector {detector.name}")
    print(f"\nInspecting GeoID #{args.geo_index} for detector {detector.name}: {geo_id}")

    fragment = h5_file.get_frag(record, geo_id)
    header = fragment.get_header()
    fragment_type = FragmentType(fragment.get_fragment_type())
    fragment_label = fragment_type_to_string(fragment_type)

    source_id = header.element_id
    source_repr = (
        f"{source_id.subsystem.name} (enum value {source_id.subsystem.value}), id=0x{source_id.id:08x}"
    )

    print("\n=== Fragment header ===")
    print(f"  Marker / version      : 0x{header.fragment_header_marker:08x} / {header.version}")
    print(f"  Size (header/payload) : {header.size} bytes total / {fragment.get_data_size()} payload bytes")
    print(f"  Run / Trigger / Seq   : {header.run_number} / {header.trigger_number} / {header.sequence_number}")
    print(f"  Trigger timestamp     : {header.trigger_timestamp}")
    print(f"  Window [begin, end]   : {header.window_begin} - {header.window_end}")
    print(f"  Detector ID           : {_format_subdetector(header.detector_id)}")
    print(f"  Fragment type         : {fragment_type.name} ({fragment_label})")
    print(f"  Source subsystem / id : {source_repr}")
    print(f"  Error bits            : 0x{header.error_bits:08x}")
    print(f"  GeoID selection       : {detector.name} / {geo_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
