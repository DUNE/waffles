"""
Unified waveform loader that auto-selects the appropriate reader (ETH vs RAW).

It tries to keep a single entry point for downstream code:
    load_waveforms(...)

The dispatcher first normalizes the detector string, then, unless forced, probes a
small number of records with the ETH reader. If ETH waveforms are found, it uses
the ETH reader; otherwise it falls back to the standard raw reader.
"""

from __future__ import annotations

from typing import Optional

import h5py

from waffles.input_output import raw_hdf5_reader
from waffles.input_output.daphne_eth_reader import load_daphne_eth_waveforms
from waffles.input_output.hdf5_structured import load_structured_waveformset


def normalize_detector(det: str) -> str:
    """Accept a variety of detector spellings and normalize to the expected strings."""
    det_clean = det.replace("-", "_")
    mapping = {
        "VD_CathodePDS": "VD_Cathode_PDS",
        "VD_MembranePDS": "VD_Membrane_PDS",
        "VD_PMT_PDS": "VD_PMT_PDS",
        "VD_PMT": "VD_PMT_PDS",
        "HD_PDS": "HD_PDS",
    }
    return mapping.get(det_clean, det_clean)


def _probe_eth(filepath: str, det: str, max_records: int = 5) -> bool:
    """
    Try a tiny ETH read to see if the file contains ETH fragments for the given detector.
    Returns True if at least one waveform is decoded.
    """
    wfset = load_daphne_eth_waveforms(
        filepath,
        detector=det,
        max_waveforms=1,
        max_records=max_records,
        skip_records=0,
    )
    return wfset is not None and len(wfset.waveforms) > 0


def _is_structured_file(filepath: str) -> bool:
    """Heuristic check for structured HDF5 (saved via hdf5_structured)."""
    try:
        with h5py.File(filepath, "r") as f:
            return all(key in f for key in ("adcs", "timestamps", "channels", "endpoints"))
    except Exception:
        return False


def load_waveforms(
    filepath: str,
    *,
    det: str = "HD_PDS",
    force_eth: bool = False,
    force_raw: bool = False,
    force_structured: bool = False,
    # Raw-reader options
    subsample: int = 1,
    wvfm_count: Optional[int] = None,
    nrecord_start_fraction: float = 0.0,
    nrecord_stop_fraction: float = 1.0,
    record_chunk_size: int = 200,
    # ETH-reader options
    max_records: Optional[int] = None,
    skip_records: int = 0,
    channel_map: Optional[str] = None,
    time_step_ns: float = 16.0,
    logger=None,
) -> Optional[object]:
    """
    Unified entry point to load waveforms from a raw HDF5 file.

    If neither force flag is set, a small ETH probe is attempted; if ETH waveforms
    are found, the ETH reader is used, otherwise it falls back to the raw reader.
    """
    if force_eth and force_raw:
        raise ValueError("force_eth and force_raw are mutually exclusive")
    if (force_structured and force_eth) or (force_structured and force_raw):
        raise ValueError("force_structured cannot be combined with force_eth or force_raw")

    det_norm = normalize_detector(det)
    use_eth = force_eth
    use_structured = force_structured

    if not (force_eth or force_raw or force_structured):
        use_structured = _is_structured_file(filepath)

    if use_structured:
        return load_structured_waveformset(
            filepath,
            max_waveforms=wvfm_count,
            verbose=False,
        )

    if not (force_eth or force_raw or use_structured):
        use_eth = _probe_eth(filepath, det_norm)

    if use_eth:
        return load_daphne_eth_waveforms(
            filepath,
            detector=det_norm,
            channel_map=channel_map,
            max_waveforms=wvfm_count,
            max_records=max_records,
            skip_records=skip_records,
            time_step_ns=time_step_ns,
            logger=logger,
        )

    # Raw reader path
    raw_wvfm_count = wvfm_count if wvfm_count is not None else int(1e9)
    return raw_hdf5_reader.WaveformSet_from_hdf5_file(
        filepath,
        nrecord_start_fraction=nrecord_start_fraction,
        nrecord_stop_fraction=nrecord_stop_fraction,
        subsample=subsample,
        wvfm_count=raw_wvfm_count,
        det=det_norm,
        record_chunk_size=record_chunk_size,
    )
