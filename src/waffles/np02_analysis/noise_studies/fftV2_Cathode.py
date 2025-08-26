#!/usr/bin/env python3
"""
avg_wf_fft_plotter.py  –  compute and plot the mean FFT per file or channel across multiple HDF5 files,
                keeping memory usage minimal, and support plotting multiple files together.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import logging
import sys
import warnings
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from waffles.input_output.hdf5_structured import load_structured_waveformset
from waffles.np02_utils.AutoMap import generate_ChannelMap, dict_uniqch_to_module, dict_module_to_uniqch
from waffles.data_classes.UniqueChannel import UniqueChannel

# -------------------- Helpers --------------------
def iter_files(paths: list[Path]) -> list[Path]:
    """
    Expand directories into *.hdf5, leave files untouched.
    """
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob("*.hdf5")))
        elif p.suffix == ".hdf5":
            out.append(p)
    return out

# FFT helper unchanged
def fft(sig: np.ndarray, dt: float = 16e-9) -> tuple[np.ndarray, np.ndarray]:
    np.seterr(divide='ignore')
    if dt is None:
        dt = 1
        t = np.arange(sig.shape[-1])
    else:
        t = np.arange(sig.shape[-1]) * dt
    if sig.shape[0] % 2 != 0:
        warnings.warn("signal preferred to be even in size, autoFixing it...", RuntimeWarning)
        t = t[:-1]
        sig = sig[:-1]
    sigFFT = np.fft.fft(sig) / t.shape[0]
    freq = np.fft.fftfreq(t.shape[0], d=dt)
    firstNegInd = np.argmax(freq < 0)
    freqAxisPos = freq[:firstNegInd]
    sigFFTPos = 2 * sigFFT[:firstNegInd]

    x = freqAxisPos / 1e6
    y = 20 * np.log10(np.abs(sigFFTPos) / 2**14)
    return x, y

# -------------------- Main Script --------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute and plot mean FFT per file and channel across HDF5 files."
    )
    ap.add_argument("inputs", nargs="+",
                    help="Structured HDF5 file(s) or directory(ies)")
    ap.add_argument("--channel", type=int, default=None,
                    help="Channel index to include in FFT (default: all)")
    #ap.add_argument("--out-html", help="Write plot to HTML instead of showing")
    ap.add_argument("--out-png", help="Write plot to png instead of showing")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    args = ap.parse_args()

    logging.basicConfig(level=max(10, 30 - args.verbose * 10),
                        format="%(levelname)s: %(message)s")

    files = iter_files([Path(p).resolve() for p in args.inputs])
    if not files:
        sys.exit("No .hdf5 files found.")

    # Prepare figure
    fig = go.Figure()

    # Cathode HV ramp up
    '''
    custom_labels = {
    "processed_np02vd_raw_run037291_0000_df-s04-d0_dw_0_20250716T125207.hdf5.copied_structured_cathode": "Cathode HV on 35kV",
    "processed_np02vd_raw_run037293_0000_df-s04-d0_dw_0_20250716T125827.hdf5.copied_structured_cathode": "Cathode HV on 70kV",
    "processed_np02vd_raw_run037295_0000_df-s04-d0_dw_0_20250716T130616.hdf5.copied_structured_cathode": "Cathode HV on 110kV",
    "processed_np02vd_raw_run037297_0000_df-s04-d0_dw_0_20250716T131710.hdf5.copied_structured_cathode": "Cathode HV on 154kV"
    }
    '''
    # UPS Test
    '''
    custom_labels = {
    "processed_np02vd_raw_run037286_0000_df-s04-d0_dw_0_20250716T115517.hdf5.copied_structured_cathode": "UPS ON",
    "processed_np02vd_raw_run037287_0000_df-s04-d0_dw_0_20250716T122141.hdf5.copied_structured_cathode": "UPS OFF"
    }
    '''
    # Check low frequency
    custom_labels = {
    "processed_np02vd_raw_run038549_0000_df-s05-d0_dw_0_20250804T144848.hdf5.copied_structured_cathode.hdf5": "Cathode Full streaming",
    }

    #print("...", dict_module_to_uniqch["M1(1)"])
    #print("...",dict_uniqch_to_module[str(UniqueChannel(107,47))])

    for f in files:
        logging.info("Processing %s", f.name)
        wfset = load_structured_waveformset(f.as_posix(), max_waveforms=1000)
        # Filter waveforms by channel if specified
        if args.channel is None:
            sel_wfs = wfset.waveforms
        else:
            sel_wfs = [wf for wf in wfset.waveforms if wf.channel == args.channel]
        if not sel_wfs:
            logging.warning("No waveforms on channel %s in %s", args.channel, f.name)
            continue

        wf_corr_list = []
        for wf in sel_wfs:
            baseline = np.median(wf.adcs[:100])
            wf_corr = wf.adcs - baseline
            wf_corr_list.append(wf_corr)

        # Compute FFT for each waveform
        #fft_list = [fft(wf.adcs) for wf in sel_wfs[:10]]
        fft_list = [fft(wf) for wf in wf_corr_list[:100]]
        freqs, powers = zip(*fft_list)
        freqs = freqs[0]
        mean_power = np.mean(powers, axis=0)

        # Add a trace per file
        #label = f.stem + (f" (chan {args.channel})" if args.channel is not None else "")
        label = custom_labels.get(f.stem, f.stem)
        fig.add_trace(
            go.Scatter(x=freqs, y=mean_power, mode="lines", name=label)
        )
        del wfset  # free memory

    # Layout
    fig.update_layout(
        #title=f"Mean FFT{(' on channel ' + str(args.channel)) if args.channel is not None else ''}",
        title=f"Mean FFT{(' on ' + dict_uniqch_to_module[str(UniqueChannel(106,str(args.channel)))]) if args.channel is not None else ''}",
        #yaxis_range=[-120,-60],
        yaxis_range=[-145,-20],
        xaxis_type="log",
        xaxis_title=r'$Frequency (MHz)$',
        yaxis_title=r'$\text{Power (dB}_{FS})$',
        template="plotly_white",
        #showlegend=True,
        height=800, width=1200
    )

    if args.out_png:
        # Save as PNG (requires kaleido)
        out_png = Path(args.out_png)
        if out_png.suffix.lower() != '.png':
            out_png = out_png.with_suffix('.png')
        fig.write_image(out_png.as_posix())
        logging.info("Plot saved as PNG to %s", out_png)
    else:
        # Default: display interactively
        fig.show()

    '''
    if args.out_html:
        fig.write_html(Path(args.out_html).as_posix())
        logging.info("Plot written to %s", args.out_html)
    else:
        fig.show()
    '''
if __name__ == "__main__":
    main()
