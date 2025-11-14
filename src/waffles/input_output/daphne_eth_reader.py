"""
Lightweight helpers to pull DAPHNE Ethernet waveforms out of raw HDF5 files.

The implementation is intentionally minimal: it opens the file with
`hdf5libs.HDF5RawDataFile`, walks through the Trigger Records, and converts the
decoded waveforms into the Waffles `Waveform`/`WaveformSet` primitives so that
downstream analysis code can stay unchanged.

Example
-------
>>> from waffles.input_output.daphne_eth_reader import load_daphne_eth_waveforms
>>> wfset = load_daphne_eth_waveforms("/path/to/file.hdf5")
>>> len(wfset.waveforms)
42
"""

from __future__ import annotations

import logging
import importlib
from typing import Iterable, Optional

import numpy as np

import detdataformats
from hdf5libs import HDF5RawDataFile

from rawdatautils.unpack.utils import DAPHNEEthUnpacker

from waffles.Exceptions import GenerateExceptionMessage
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.utils.daphne_helpers import looks_like_daphne_eth, select_records


def _alias_method(target: object, alias_name: str, source_name: str) -> bool:
    """Try to add ``alias_name`` to ``target`` by pointing it to ``source_name``."""
    if target is None or hasattr(target, alias_name):
        return False if target is None else True

    original = getattr(target, source_name, None)
    if original is None:
        return False

    try:
        setattr(target, alias_name, original)
    except Exception:
        return False
    return True


def _ensure_daphne_header_compat() -> None:
    """
    Older rawdatautils versions expect DAPHNEEt.get_header(), newer
    fddetdataformats builds renamed it to get_daqheader(). Patch in an alias.
    """
    for module_name in ("fddetdataformats",):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

        candidates = [module]
        submodule = getattr(module, "_daq_fddetdataformats_py", None)
        if submodule is not None:
            candidates.append(submodule)

        for candidate in candidates:
            for type_name in ("DAPHNEEt", "DAPHNEEthFrame"):
                target = getattr(candidate, type_name, None)
                _alias_method(target, "get_header", "get_daqheader")


_ensure_daphne_header_compat()


def _ensure_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    """Provide a logger with a sensible default formatting if the caller passed None."""
    if logger is not None:
        return logger

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            fmt='[%(levelname)s] %(asctime)s - %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_daphne_eth_waveforms(
    filepath: str,
    *,
    detector: str = "HD_PDS",
    channel_map: Optional[str] = None,
    max_waveforms: Optional[int] = None,
    max_records: Optional[int] = None,
    skip_records: int = 0,
    time_step_ns: float = 16.0,
    logger: Optional[logging.Logger] = None,
) -> Optional[WaveformSet]:
    """
    Decode DAPHNE Ethernet fragments from a raw HDF5 file and return them as a WaveformSet.

    Parameters
    ----------
    filepath:
        Path to the input HDF5 file.
    detector:
        Name understood by `detdataformats.DetID.string_to_subdetector` (defaults to "HD_PDS").
    channel_map:
        Optional channel map name forwarded to `DAPHNEEthUnpacker`.
    max_waveforms:
        If set, stop once this many waveforms have been produced.
    max_records:
        If set, only inspect the first `max_records` Trigger Records (after skipping).
    skip_records:
        Number of Trigger Records to skip from the start of the file.
    time_step_ns:
        Sampling period passed to the Waffles `Waveform` ctor (defaults to 16 ns).
    logger:
        Optional logger; if omitted a module-level logger is configured automatically.

    Returns
    -------
    Optional[WaveformSet]
        A WaveformSet populated with the decoded DAPHNE Ethernet waveforms, or ``None`` if
        the file does not contain any matching fragments.
    """
    log = _ensure_logger(logger)

    try:
        det_enum = detdataformats.DetID.string_to_subdetector(detector)
    except Exception as err:
        raise Exception(GenerateExceptionMessage(
            1,
            "load_daphne_eth_waveforms()",
            f"Failed to parse detector '{detector}': {err}",
        )) from err

    h5_file = HDF5RawDataFile(filepath)
    records = list(h5_file.get_all_record_ids())
    selected_records = select_records(records, skip_records, max_records)

    unpacker = DAPHNEEthUnpacker(channel_map=channel_map, ana_data_prescale=1, wvfm_data_prescale=1)

    waveforms = []
    for record in selected_records:
        try:
            geo_ids = list(h5_file.get_geo_ids_for_subdetector(record, det_enum))
        except Exception as err:
            log.warning("Skipping record %s: unable to resolve geo IDs (%s)", record, err)
            continue

        for geo_id in geo_ids:
            try:
                fragment = h5_file.get_frag(record, geo_id)
            except Exception as err:
                log.warning("Skipping fragment (record=%s, geo_id=%s): %s", record, geo_id, err)
                continue

            if fragment.get_data_size() == 0:
                continue
            if not looks_like_daphne_eth(fragment.get_fragment_type()):
                continue

            _, det_waveforms = unpacker.get_det_data_all(fragment)
            if not det_waveforms:
                continue

            header = fragment.get_header()
            run_number = int(header.run_number)
            record_number = int(record[0]) if isinstance(record, (tuple, list)) else int(record)
            daq_timestamp = int(fragment.get_trigger_timestamp())

            for wfdata in det_waveforms:
                wf = Waveform(
                    timestamp=int(wfdata.timestamp_dts),
                    time_step_ns=time_step_ns,
                    daq_window_timestamp=daq_timestamp,
                    adcs=np.array(wfdata.adcs, copy=True),
                    run_number=run_number,
                    record_number=record_number,
                    endpoint=int(wfdata.src_id),
                    channel=int(wfdata.channel),
                )
                waveforms.append(wf)

                if max_waveforms is not None and len(waveforms) >= max_waveforms:
                    log.info("Reached max_waveforms=%s; returning early", max_waveforms)
                    return WaveformSet(*waveforms)

    if not waveforms:
        log.warning("No DAPHNE Ethernet waveforms found in %s", filepath)
        return None

    return WaveformSet(*waveforms)
