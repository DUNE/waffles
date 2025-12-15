#!/usr/bin/env python3
"""
Profile the raw HDF5 reader to spot hotspots in decoding/per-record loops.

Example:
    python scripts/profile_raw_hdf5_reader.py /path/to/file.hdf5 --wvfm-count 2000
"""

import argparse
import cProfile
import pstats
import sys
from pathlib import Path

from waffles.input_output import raw_hdf5_reader as reader
from waffles.input_output.daphne_eth_reader import load_daphne_eth_waveforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="cProfile wrapper around raw_hdf5_reader.WaveformSet_from_hdf5_file or the ETH reader.",
    )
    parser.add_argument("filepath", help="Path to the raw HDF5 file to read.")
    parser.add_argument(
        "--det",
        default="HD_PDS",
        help="Detector string (e.g. HD_PDS, VD_Membrane_PDS, VD_Cathode_PDS, VD_CathodePDS).",
    )
    parser.add_argument(
        "--eth",
        action="store_true",
        help="Use the DAPHNE Ethernet reader (load_daphne_eth_waveforms).",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Keep 1 of every N waveforms (default: 1, i.e., keep all).",
    )
    parser.add_argument(
        "--wvfm-count",
        type=int,
        default=int(1e9),
        help="Maximum number of waveforms to read (default: very large).",
    )
    parser.add_argument(
        "--nrecord-start-fraction",
        type=float,
        default=0.0,
        help="Fraction of records to skip from the start (default: 0.0).",
    )
    parser.add_argument(
        "--nrecord-stop-fraction",
        type=float,
        default=1.0,
        help="Fraction of records to keep up to (default: 1.0).",
    )
    parser.add_argument(
        "--record-chunk-size",
        type=int,
        default=200,
        help="Records per chunk inside the reader (default: 200).",
    )
    parser.add_argument(
        "--save-stats",
        type=Path,
        help="Optional path to dump the raw cProfile stats for later inspection.",
    )
    parser.add_argument(
        "--print-top",
        type=int,
        default=40,
        help="How many lines of the sorted cProfile output to show (default: 40).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    profile = cProfile.Profile()
    profile.enable()

    if args.eth:
        wfset = load_daphne_eth_waveforms(
            args.filepath,
            detector=args.det,
            max_waveforms=args.wvfm_count,
            max_records=None,
            skip_records=0,
        )
    else:
        wfset = reader.WaveformSet_from_hdf5_file(
            args.filepath,
            nrecord_start_fraction=args.nrecord_start_fraction,
            nrecord_stop_fraction=args.nrecord_stop_fraction,
            subsample=args.subsample,
            wvfm_count=args.wvfm_count,
            det=args.det,
            record_chunk_size=args.record_chunk_size,
        )

    profile.disable()

    if wfset is None:
        print("No waveforms found")
        return 1

    stats = pstats.Stats(profile).strip_dirs().sort_stats("cumulative")
    stats.print_stats(args.print_top)

    if args.save_stats:
        stats.dump_stats(str(args.save_stats))
        print(f"Saved cProfile stats to {args.save_stats}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
