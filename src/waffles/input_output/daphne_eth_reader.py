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
import detchannelmaps
from daqdataformats import FragmentType
from hdf5libs import HDF5RawDataFile

from waffles.Exceptions import GenerateExceptionMessage
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.utils.daphne_decoders import (
    decode_fragment_arrays,
    extract_daq_link,
    get_fragment_decoder,
)
from waffles.utils.daphne_helpers import select_records


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
        Optional detchannelmaps plugin used to translate raw channels into offline IDs.
        Defaults to ``SimplePDSChannelMap`` when omitted.
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

    channel_map_plugin = channel_map or "SimplePDSChannelMap"
    channel_mapper = None
    if channel_map_plugin:
        try:
            channel_mapper = detchannelmaps.make_pds_map(channel_map_plugin)
        except Exception as err:
            log.warning(
                "Unable to load channel map '%s': %s (raw channels will be used)",
                channel_map_plugin,
                err,
            )
            channel_mapper = None

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

            try:
                fragment_type = FragmentType(fragment.get_fragment_type())
            except ValueError:
                continue

            decoder = get_fragment_decoder(fragment_type)
            if decoder is None:
                continue

            try:
                raw_channels, adcs_matrix, timestamps = decode_fragment_arrays(
                    fragment, decoder
                )
            except Exception as err:
                log.warning(
                    "Skipping fragment (record=%s, geo_id=%s): failed to decode (%s)",
                    record,
                    geo_id,
                    err,
                )
                continue

            if raw_channels.size == 0 or adcs_matrix.size == 0:
                continue

            if decoder.is_stream:
                adcs_matrix = np.asarray(adcs_matrix).transpose()
                first_ts = (
                    int(timestamps[0])
                    if np.size(timestamps) > 0
                    else int(fragment.get_trigger_timestamp())
                )
                per_waveform_ts = np.full(len(raw_channels), first_ts, dtype=np.int64)
            else:
                per_waveform_ts = np.asarray(timestamps).astype(np.int64, copy=False)
            per_waveform_ts = per_waveform_ts.reshape(-1)

            try:
                det_id, crate_id, slot_id, stream_id = extract_daq_link(fragment, decoder)
            except Exception as err:
                log.warning(
                    "Skipping fragment (record=%s, geo_id=%s): unable to parse DAQ header (%s)",
                    record,
                    geo_id,
                    err,
                )
                continue

            header = fragment.get_header()
            endpoint = int(header.element_id.id)
            run_number = int(header.run_number)
            record_number = (
                int(record[0]) if isinstance(record, (tuple, list)) else int(record)
            )
            daq_timestamp = int(fragment.get_trigger_timestamp())

            for idx, raw_channel in enumerate(raw_channels):
                if idx >= adcs_matrix.shape[0]:
                    break
                offline_channel = int(raw_channel)
                if channel_mapper is not None:
                    try:
                        offline_channel = channel_mapper.get_offline_channel_from_det_crate_slot_stream_chan(
                            det_id, crate_id, slot_id, stream_id, int(raw_channel)
                        )
                    except Exception:
                        offline_channel = int(raw_channel)

                timestamp_value = (
                    int(per_waveform_ts[idx])
                    if idx < len(per_waveform_ts)
                    else daq_timestamp
                )
                wf = Waveform(
                    timestamp=timestamp_value,
                    time_step_ns=time_step_ns,
                    daq_window_timestamp=daq_timestamp,
                    adcs=np.array(adcs_matrix[idx], copy=True),
                    run_number=run_number,
                    record_number=record_number,
                    endpoint=endpoint,
                    channel=int(offline_channel),
                )
                wf.offline_channel = int(offline_channel)
                wf.raw_channel = int(raw_channel)
                wf.slot = int(slot_id)
                wf.stream_id = int(stream_id)
                wf.detector_id = int(det_id)
                wf.crate_id = int(crate_id)
                waveforms.append(wf)

                if max_waveforms is not None and len(waveforms) >= max_waveforms:
                    log.info("Reached max_waveforms=%s; returning early", max_waveforms)
                    return WaveformSet(*waveforms)

    if not waveforms:
        log.warning("No DAPHNE Ethernet waveforms found in %s", filepath)
        return None

    lengths = {len(wf.adcs) for wf in waveforms}
    if len(lengths) > 1:
        min_len = min(lengths)
        log.warning(
            "Waveforms have non-uniform lengths %s; truncating all to %d samples",
            sorted(lengths),
            min_len,
        )
        for wf in waveforms:
            wf._WaveformAdcs__slice_adcs(0, min_len)

    return WaveformSet(*waveforms)
