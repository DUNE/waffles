#!/usr/bin/env python3
from __future__ import annotations            # postpone type-hint eval

import argparse, getpass, json, logging, sys
from pathlib import Path
import subprocess

import paramiko                               # SSH / SFTP
import numpy as np
import matplotlib.pyplot as plt
import plotly.subplots as psu

# ── waffles imports ──────────────────────────────────────────────────────────
from waffles.input_output.hdf5_structured import load_structured_waveformset
from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.np02_data.ProtoDUNE_VD_maps import mem_geometry_map
from waffles.np02_data.ProtoDUNE_VD_maps import cat_geometry_map
from waffles.plotting.plot import plot_ChannelWsGrid
# ─────────────────────────────────────────────────────────────────────────────


# ╭────────────────────────── Helper utilities ───────────────────────────────╮
def parse_run_list(txt: str) -> list[int]:
    out: set[int] = set()
    for chunk in txt.split(","):
        if "-" in chunk:
            lo, hi = map(int, chunk.split("-"))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(chunk))
    return sorted(out)


def ssh_connect(host: str, port: int, user: str,
                *, kerberos=False, key: str | None = None,
                passwd: str | None = None) -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if kerberos:
        c.connect(host, port=port, username=user,
                  gss_auth=True, gss_kex=True, gss_host=host)
    elif key:
        pkey = paramiko.RSAKey.from_private_key_file(key, password=passwd)
        c.connect(host, port=port, username=user, pkey=pkey)
    else:
        c.connect(host, port=port, username=user, password=passwd)
    return c


def remote_hdf5_files(ssh: paramiko.SSHClient,
                      remote_dir: str, run: int) -> list[str]:
    """Return list of raw-data chunks for *run* (accept .hdf5[.gz][.copied])."""
    cmd = f"ls -1 {remote_dir}/np02vd_raw_run{run:06d}*.hdf5* 2>/dev/null"
    _in, out, err = ssh.exec_command(cmd)
    if err.read():
        return []
    files = out.read().decode().splitlines()

    # whitelist endings
    keep: list[str] = []
    for f in files:
        if any(f.endswith(sfx) for sfx in
               (".hdf5", ".hdf5.gz", ".hdf5.copied", ".hdf5.gz.copied")):
            keep.append(f)

    # deduplicate plain vs .copied (prefer plain)
    uniq: dict[str, str] = {}
    for f in keep:
        base = f.removesuffix(".copied")
        if base not in uniq or not uniq[base].endswith(".copied"):
            uniq[base] = f
    return sorted(uniq.values())


def download_all(sftp: paramiko.SFTPClient,
                 files: list[str], dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for fr in files:
        attr = sftp.stat(fr)
        tgt = dest / Path(fr).name
        if tgt.is_file() and tgt.stat().st_size == attr.st_size:
            logging.info("⏩  %s already present (%d B)", tgt.name, attr.st_size)
        else:
            logging.info("⬇️  %s → %s", fr, tgt)
            sftp.get(fr, tgt.as_posix())
        out.append(tgt)
    return out
# ╰────────────────────────────────────────────────────────────────────────────╯


# ╭────────────────── Waveform analysis & plotting helpers ────────────────────╮
def _analyse(wfset):
    ip = IPDict(
        baseline_limits=[0, 50, 900, 1000],
        baseline_method="EasyMedian",      # ← NEW (or "Mean", "Fit", …)
        int_ll=50, int_ul=120,
        amp_ll=50, amp_ul=120,
    )
    wfset.analyse("std", BasicWfAna, ip,
                  analysis_kwargs={},
                  checks_kwargs=dict(points_no=wfset.points_per_wf),
                  overwrite=True)

def _grids(wfset, detector:str):

    if detector == 'VD_Membrane_PDS':
        return dict(
            TCO=ChannelWsGrid(mem_geometry_map[2], wfset,
                              bins_number=115,
                              domain=np.array([-1e4, 5e4]),
                              variable="integral"),
            nTCO=ChannelWsGrid(mem_geometry_map[1], wfset,
                               bins_number=115,
                               domain=np.array([-1e4, 5e4]),
                               variable="integral"))
    elif detector == 'VD_Cathode_PDS':
        return dict(
            TCO=ChannelWsGrid(cat_geometry_map[2], wfset,
                              bins_number=115,
                              domain=np.array([-1e4, 5e4]),
                              variable="integral"),
            nTCO=ChannelWsGrid(cat_geometry_map[1], wfset,
                               bins_number=115,
                               domain=np.array([-1e4, 5e4]),
                               variable="integral"))


def plot_grid(grid, title, html: Path | None, detector:str):

    if detector == 'VD_Membrane_PDS':
        fig = psu.make_subplots(rows=4, cols=2)
    elif detector == 'VD_Cathode_PDS':
        fig = psu.make_subplots(rows=8, cols=4)
    
    plot_ChannelWsGrid( grid, figure=fig, share_x_scale=True,
                       share_y_scale=True, mode="overlay", wfs_per_axes=50)
    fig.update_layout(title=title, template="plotly_white",
                      width=1000, height=800, showlegend=True)
    if html:
        fig.write_html(html.as_posix())
        logging.info("💾 %s", html)
# ╰────────────────────────────────────────────────────────────────────────────╯


def process_structured(h5: Path, outdir: Path,
                       max_wfs: int, headless: bool, detector: str):
    
    wfset = load_structured_waveformset(h5.as_posix(),
                                        max_waveforms=max_wfs)
    _analyse(wfset)
    for n, g in _grids(wfset, detector).items():
        html = outdir / f"{n}.html" if headless else None
        plot_grid(g, n, html, detector)


# ╭─────────────────────────────── main() ─────────────────────────────────────╮
def main() -> None:
    ap = argparse.ArgumentParser(description="NP02-VD multi-run processor")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--remote-dir", default="/data0")
    ap.add_argument("--out", default=".", help="Local root output directory")
    ap.add_argument("--hostname", default="np04-srv-004")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--user", required=True)
    auth = ap.add_mutually_exclusive_group()
    auth.add_argument("--kerberos", action="store_true")
    auth.add_argument("--ssh-key", help="Path to private key")
    ap.add_argument("--all-chunks", action="store_true")
    ap.add_argument("--max-waveforms", type=int, default=100, help="Maximum waveforms to be plotted")
    ap.add_argument("--config-template", default="config.json")
    ap.add_argument("--headless", action="store_true", help="Set it to save html plots instead of showing them")
    ap.add_argument("-v", "--verbose", action="count", default=1)
    args = ap.parse_args()

    logging.basicConfig(level=max(10, 30 - 10*args.verbose),
                        format="%(levelname)s: %(message)s")

    runs = parse_run_list(args.runs)
    out_root = Path(args.out).resolve()
    raw_dir = out_root / "raw"
    list_dir = out_root / "raw_lists"
    processed_dir = out_root / "processed"
    plot_root = out_root / "plots"

    for d in (list_dir, processed_dir, plot_root):
        d.mkdir(parents=True, exist_ok=True)

    # ── SSH login ───────────────────────────────────────────────────────────
    pw = None
    if not args.kerberos and not args.ssh_key:
        pw = getpass.getpass(f"{args.user}@{args.hostname} password: ")
    ssh = ssh_connect(args.hostname, args.port, args.user,
                      kerberos=args.kerberos, key=args.ssh_key, passwd=pw)
    sftp = ssh.open_sftp()
    logging.info("✅ SSH connected")

 #   ok_runs: list[int] = []
 #   for run in runs:
    # -----------------------------------------------------------------
    # runs that already have a processed file -> skip EVERYTHING
    # -----------------------------------------------------------------
    have_struct = {
        run for run in runs
        if any(processed_dir.glob(f"processed_np02vd_raw_run{run:06d}_*.hdf5"))
    }
    for r in sorted(have_struct):
        logging.info("run %d: processed file exists – nothing to do", r)

    runs_to_fetch = [r for r in runs if r not in have_struct]
    ok_runs: list[int] = []

    for run in runs_to_fetch:
        try:
            rem = remote_hdf5_files(ssh, args.remote_dir, run)
            if not rem:
                logging.warning("run %d: no remote files", run)
                continue
            loc = download_all(sftp, rem, raw_dir / f"run{run:06d}")
            (list_dir / f"{run:06d}.txt").write_text(
                "\n".join(p.as_posix() for p in loc) + "\n")
            ok_runs.append(run)
        except Exception as e:
            logging.error("run %d: %s", run, e)
    sftp.close()
    ssh.close()

    # ── Skip already-processed runs ─────────────────────────────────────────
    pending = []
    for r in ok_runs:
        if any(processed_dir.glob(f"processed_np02vd_raw_run{r:06d}_*.hdf5")):
            logging.info("run %d already processed – skip", r)
        else:
            pending.append(r)
    pending=ok_runs
    if not pending:
        logging.warning("Nothing to process; all runs already done.")
    else:
        # ── Build config for 07_save_structured_from_config.py ──────────────
        cfg = json.load(open(args.config_template))
        detector = cfg.get("det")
        cfg.update(dict(
            runs=pending,
            rucio_dir=list_dir.as_posix(),
            output_dir=processed_dir.as_posix()))
        tmp_cfg = out_root / "temp_config.json"
        tmp_cfg.write_text(json.dumps(cfg, indent=4))

        logging.info("🚀 07_save_structured_from_config.py …")
        subprocess.run(["python3", "07_save_structured_from_config.py",
                        "--config", tmp_cfg.as_posix()], check=True)

    # ── Plot each run (new or existing) ─────────────────────────────────────
    for r in ok_runs:
        prod = list(processed_dir.glob(
            f"processed_np02vd_raw_run{r:06d}_*.hdf5"))
        if not prod:
            logging.warning("run %d: processed file missing", r)
            continue
        pr_dir = plot_root / f"run{r:06d}"
        pr_dir.mkdir(parents=True, exist_ok=True)
        process_structured(prod[0], pr_dir,
                           args.max_waveforms, args.headless, detector)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
