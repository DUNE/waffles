"""
Helpers that select the correct rawdatautils bindings for each DAPHNE fragment type.

All four fragment types (kDAPHNE, kDAPHNEStream, kDAPHNEEth, kDAPHNEEthStream)
require different frame descriptors even though they share similar numpy helper
names. This module centralizes the dispatch so other code can stay concise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from daqdataformats import FragmentType
from fddetdataformats import (
    DAPHNEEthFrame,
    DAPHNEEthStreamFrame,
    DAPHNEFrame,
    DAPHNEStreamFrame,
)

from rawdatautils.unpack import daphne as daphne_unpack

try:
    from rawdatautils.unpack import daphneeth as daphneeth_unpack
except ImportError:  # pragma: no cover - optional dependency
    daphneeth_unpack = None


@dataclass(frozen=True)
class FragmentDecoder:
    """Describe how to decode a particular fragment type."""

    module: object
    frame_class: type
    is_stream: bool


_FRAGMENT_DECODERS: dict[FragmentType, FragmentDecoder] = {
    FragmentType.kDAPHNE: FragmentDecoder(daphne_unpack, DAPHNEFrame, False),
    FragmentType.kDAPHNEStream: FragmentDecoder(daphne_unpack, DAPHNEStreamFrame, True),
}

if daphneeth_unpack is not None:
    _FRAGMENT_DECODERS.update(
        {
            FragmentType.kDAPHNEEth: FragmentDecoder(
                daphneeth_unpack, DAPHNEEthFrame, False
            ),
            FragmentType.kDAPHNEEthStream: FragmentDecoder(
                daphneeth_unpack, DAPHNEEthStreamFrame, True
            ),
        }
    )


def get_fragment_decoder(fragment_type: FragmentType) -> Optional[FragmentDecoder]:
    """Return the decoder configuration for ``fragment_type`` if supported."""
    return _FRAGMENT_DECODERS.get(fragment_type)


def decode_fragment_arrays(
    fragment,
    decoder: FragmentDecoder,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert ``fragment`` into (raw_channels, adcs_matrix, timestamps).

    ``adcs_matrix`` always has shape ``(n_waveforms, n_samples)`` so each row can
    be passed to :class:`waffles.data_classes.Waveform`.
    """
    module = decoder.module

    if decoder.is_stream:
        channels = np.asarray(module.np_array_channels_stream(fragment))
        if channels.size == 0:
            return (
                np.empty(0, dtype=np.int16),
                np.empty((0, 0), dtype=np.int16),
                np.empty(0, dtype=np.int64),
            )

        raw_channels = (
            channels[0].reshape(-1) if channels.ndim > 1 else channels.reshape(-1)
        )
        adcs = np.asarray(module.np_array_adc_stream(fragment)).transpose()
        timestamps = np.asarray(module.np_array_timestamp_stream(fragment))
        first_ts = (
            int(timestamps[0]) if timestamps.size > 0 else int(fragment.get_trigger_timestamp())
        )
        per_waveform_ts = np.full(len(raw_channels), first_ts, dtype=np.int64)
        return raw_channels, adcs, per_waveform_ts

    raw_channels = np.asarray(module.np_array_channels(fragment)).reshape(-1)
    adcs = np.asarray(module.np_array_adc(fragment))
    timestamps = np.asarray(module.np_array_timestamp(fragment)).reshape(-1)
    if timestamps.size == raw_channels.size:
        per_waveform_ts = timestamps.astype(np.int64, copy=False)
    else:
        per_waveform_ts = np.full(
            len(raw_channels), int(fragment.get_trigger_timestamp()), dtype=np.int64
        )
    return raw_channels, adcs, per_waveform_ts


def extract_daq_link(fragment, decoder: FragmentDecoder) -> Tuple[int, int, int, int]:
    """Return (det, crate, slot, stream) identifiers from the first frame header."""
    payload = fragment.get_data_bytes()
    frame_type = decoder.frame_class
    frame = frame_type(payload[: frame_type.sizeof()])
    daq_header = frame.get_daqheader()

    det_id = int(daq_header.det_id)
    crate_id = int(daq_header.crate_id)
    slot_id = int(daq_header.slot_id)
    stream_id = int(getattr(daq_header, "stream_id", getattr(daq_header, "link_id", 0)))
    return det_id, crate_id, slot_id, stream_id


def iter_raw_channels(
    fragment,
    decoder: FragmentDecoder,
    *,
    frames_per_fragment: Optional[int] = None,
) -> Sequence[int]:
    """Yield raw channel IDs for the fragment payload."""
    module = decoder.module
    if decoder.is_stream:
        channels = np.asarray(module.np_array_channels_stream(fragment))
        if frames_per_fragment is not None:
            channels = channels[:frames_per_fragment, :]
        return np.asarray(channels).reshape(-1)

    channels = np.asarray(module.np_array_channels(fragment))
    if frames_per_fragment is not None:
        channels = channels[:frames_per_fragment]
    return np.asarray(channels).reshape(-1)
