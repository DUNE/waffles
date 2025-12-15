#!/usr/bin/env python3
"""
Inspect a raw/ETH/structured HDF5 file and print a concise summary.

Examples:
  python scripts/inspect_raw_file.py /path/to/file.hdf5 --max-records 5 --max-waveforms 200
  python scripts/inspect_raw_file.py /path/to/file.hdf5 --eth --det VD_CathodePDS --debug
"""

import argparse
from collections import Counter
from pprint import pprint
from typing import Dict, Iterable, Optional, Tuple

import detchannelmaps
from daqdataformats import FragmentType
from hdf5libs import HDF5RawDataFile

from waffles.input_output.waveform_loader import load_waveforms, normalize_detector
from waffles.utils.daphne_decoders import (
    decode_fragment_arrays,
    extract_daq_link,
    get_fragment_decoder,
)
from waffles.utils.daphne_helpers import select_records


def summarize_fragments(filepath: str, detector: str, max_records: Optional[int], skip_records: int, debug: bool) -> Dict[str, Counter]:
    """Collect fragment type/source/geo_id counts from the first records."""
    h5_file = HDF5RawDataFile(filepath)
    records = list(h5_file.get_all_record_ids())
    records = select_records(records, skip_records, max_records)

    frag_types = Counter()
    source_ids = Counter()
    geo_ids = Counter()

    for record in records:
        try:
            geos = list(h5_file.get_geo_ids_for_subdetector(record, detector))
        except Exception:
            continue

        for gid in geos:
            try:
                frag = h5_file.get_frag(record, gid)
            except Exception:
                continue
            if frag.get_data_size() == 0:
                continue
            try:
                ftype = FragmentType(frag.get_fragment_type())
            except ValueError:
                continue

            frag_types[str(ftype)] += 1
            source_ids[frag.get_header().get_source_id().id] += 1
            geo_ids[gid.id] += 1

            if debug:
                hdr = frag.get_header()
                print(
                    f"[DEBUG] record={record} geo_id={gid.id} "
                    f"type={ftype.name} size={frag.get_data_size()} "
                    f"run={hdr.run_number} ts={hdr.trigger_timestamp}"
                )
                decoder = get_fragment_decoder(ftype)
                if decoder:
                    try:
                        chs, adcs, ts = decode_fragment_arrays(frag, decoder)
                        print(f"        channels={list(chs[:8])} nsamples={adcs.shape}")
                    except Exception as err:
                        print(f"        decode failed: {err}")

    return {
        "fragment_types": frag_types,
        "source_ids": source_ids,
        "geo_ids": geo_ids,
        "records_considered": Counter({"records": len(records)}),
    }


def summarize_waveforms(filepath: str, det: str, max_waveforms: int, force_eth: bool, force_raw: bool, structured: bool) -> None:
    """Load a small sample of waveforms and print channel/endpoint counts."""
    wfset = load_waveforms(
        filepath,
        det=det,
        force_eth=force_eth,
        force_raw=force_raw,
        force_structured=structured,
        wvfm_count=max_waveforms,
        nrecord_stop_fraction=0.2,
    )
    if wfset is None or len(wfset.waveforms) == 0:
        print("No waveforms decoded.")
        return

    endpoints = Counter()
    channels = Counter()
    for wf in wfset.waveforms:
        endpoints[int(wf.endpoint)] += 1
        channels[(int(wf.endpoint), int(wf.channel))] += 1

    print(f"Decoded waveforms: {len(wfset.waveforms)}")
    print("Endpoints (top 10):")
    for ep, cnt in endpoints.most_common(10):
        print(f"  endpoint {ep}: {cnt}")
    print("Channels (top 10):")
    for (ep, ch), cnt in channels.most_common(10):
        print(f"  ep {ep} ch {ch}: {cnt}")


def offline_channel_table(
    filepath: str,
    detector: str,
    channel_map_plugin: str,
    skip_records: int,
    max_records: Optional[int],
    frames_per_fragment: Optional[int],
    limit_rows: int,
) -> None:
    """
    Build and print a raw->offline channel table using detchannelmaps.
    """
    det_norm = normalize_detector(detector)
    channel_map = detchannelmaps.make_pds_map(channel_map_plugin)
    h5_file = HDF5RawDataFile(filepath)
    records = select_records(list(h5_file.get_all_record_ids()), skip_records, max_records)

    counts = Counter()
    for record in records:
        try:
            geos = list(h5_file.get_geo_ids_for_subdetector(record, det_norm))
        except Exception:
            continue

        for gid in geos:
            try:
                frag = h5_file.get_frag(record, gid)
            except Exception:
                continue
            if frag.get_data_size() == 0:
                continue
            try:
                ftype = FragmentType(frag.get_fragment_type())
            except ValueError:
                continue

            decoder = get_fragment_decoder(ftype)
            if decoder is None:
                continue

            try:
                raw_channels, _, _ = decode_fragment_arrays(frag, decoder)
                det_id, crate_id, slot_id, stream_id = extract_daq_link(frag, decoder)
            except Exception:
                continue

            raw_channels = raw_channels.reshape(-1)
            if frames_per_fragment is not None and frames_per_fragment > 0 and raw_channels.size > frames_per_fragment:
                raw_channels = raw_channels[:frames_per_fragment]

            for raw_ch in raw_channels:
                offline = channel_map.get_offline_channel_from_det_crate_slot_stream_chan(
                    det_id, crate_id, slot_id, stream_id, int(raw_ch)
                )
                counts[(slot_id, int(raw_ch), int(offline))] += 1

    if not counts:
        print("No raw->offline mapping entries found.")
        return

    print("Raw -> offline channel mapping (top entries):")
    for (slot, raw_ch, offline), cnt in counts.most_common(limit_rows):
        print(f"  slot={slot:2d} raw_ch={raw_ch:3d} offline={offline:4d} (count={cnt})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect raw/ETH/structured HDF5 and print summary.")
    p.add_argument("filepath", help="Path to HDF5 file.")
    p.add_argument("--det", default="AUTO", help="Detector (HD_PDS, VD_MembranePDS, VD_CathodePDS, or AUTO).")
    p.add_argument("--eth", action="store_true", help="Force ETH reader.")
    p.add_argument("--force-raw", action="store_true", help="Force RAW reader.")
    p.add_argument("--structured", action="store_true", help="Force structured reader.")
    p.add_argument("--max-records", type=int, help="Max records to scan (headers/mapping).")
    p.add_argument("--skip-records", type=int, default=0, help="Records to skip from start.")
    p.add_argument("--frames-per-fragment", type=int, help="Limit frames per fragment when mapping channels.")
    p.add_argument("--max-waveforms", type=int, default=200, help="Max waveforms to load for summary.")
    p.add_argument("--debug", action="store_true", help="Print fragment headers/channels during scan.")
    p.add_argument(
        "--channel-map",
        default="SimplePDSChannelMap",
        help="detchannelmaps plugin for raw->offline mapping (default: SimplePDSChannelMap).",
    )
    p.add_argument(
        "--offline-table",
        action="store_true",
        help="Print raw->offline channel mapping summary.",
    )
    p.add_argument(
        "--offline-limit",
        type=int,
        default=30,
        help="Limit rows in the raw->offline table output (default: 30).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    det = normalize_detector(args.det) if args.det else "AUTO"
    print(f"== Inspecting {args.filepath}")
    print(f"Detector: {det} (AUTO triggers detection)\n")

    # Fragment-level summary
    frag_stats = summarize_fragments(
        args.filepath,
        det if det != "AUTO" else "HD_PDS",
        args.max_records,
        args.skip_records,
        args.debug,
    )
    print("Fragment types:")
    pprint(frag_stats["fragment_types"])
    print("Source IDs:")
    pprint(frag_stats["source_ids"])
    print("Geo IDs:")
    pprint(frag_stats["geo_ids"])
    print(f"Records considered: {frag_stats['records_considered']['records']}\n")

    # Waveform-level summary
    summarize_waveforms(
        args.filepath,
        det,
        max_waveforms=args.max_waveforms,
        force_eth=args.eth,
        force_raw=args.force_raw,
        structured=args.structured,
    )

    if args.offline_table:
        offline_channel_table(
            args.filepath,
            det if det != "AUTO" else "HD_PDS",
            args.channel_map,
            skip_records=args.skip_records,
            max_records=args.max_records,
            frames_per_fragment=args.frames_per_fragment,
            limit_rows=args.offline_limit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
