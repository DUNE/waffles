#!/usr/bin/env python3
"""
CLI helper that demonstrates how to use `waffles.input_output.daphne_eth_reader`.

Example
-------
    ./scripts/read_daphne_eth.py /path/to/file.hdf5 --max-waveforms 10

The script prints a short summary of the DAPHNE Ethernet waveforms it finds and
echoes a few example entries so you can double-check the decoding quickly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from waffles.input_output.daphne_eth_reader import load_daphne_eth_waveforms


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect DAPHNE Ethernet fragments stored in a raw HDF5 file."
    )
    parser.add_argument("hdf5_file", help="Path to the input raw HDF5 file.")
    parser.add_argument(
        "--detector",
        default="HD_PDS",
        help="Detector string understood by detdataformats (default: %(default)s).",
    )
    parser.add_argument(
        "--channel-map",
        default=None,
        help="Optional channel map name forwarded to rawdatautils.",
    )
    parser.add_argument(
        "--max-waveforms",
        type=int,
        default=None,
        help="Stop after decoding this many waveforms.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Only examine the first N Trigger Records (after skipping).",
    )
    parser.add_argument(
        "--skip-records",
        type=int,
        default=0,
        help="Number of Trigger Records to skip from the beginning.",
    )
    parser.add_argument(
        "--time-step-ns",
        type=float,
        default=16.0,
        help="Sampling period (ns) passed to the Waveform objects (default: %(default)s).",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=5,
        help="Print up to this many example waveforms (default: %(default)s).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress info-level logging; only warnings/errors will be shown.",
    )
    return parser


def _configure_logger(quiet: bool) -> logging.Logger:
    logger = logging.getLogger("daphne_eth_reader")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.WARNING if quiet else logging.INFO)
    return logger


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logger = _configure_logger(args.quiet)

    wfset = load_daphne_eth_waveforms(
        filepath=args.hdf5_file,
        detector=args.detector,
        channel_map=args.channel_map,
        max_waveforms=args.max_waveforms,
        max_records=args.max_records,
        skip_records=args.skip_records,
        time_step_ns=args.time_step_ns,
        logger=logger,
    )

    if wfset is None:
        print("No DAPHNE Ethernet waveforms found.", file=sys.stderr)
        return 1

    print(f"Decoded {len(wfset.waveforms)} waveforms from {args.hdf5_file}")

    unique_channels = sorted({(wf.endpoint, wf.channel) for wf in wfset.waveforms})
    print(f"Observed {len(unique_channels)} unique (endpoint, channel) pairs.")

    to_show = min(args.show, len(wfset.waveforms))
    if to_show > 0:
        print(f"\nDisplaying the first {to_show} waveforms:")
        for wf in wfset.waveforms[:to_show]:
            print(
                f"  run={wf.run_number} record={wf.record_number} "
                f"endpoint={wf.endpoint} channel={wf.channel} "
                f"timestamp={wf.timestamp} nsamples={len(wf.adcs)}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())

