"""Plot accepted ColdBox VGAIN-scan results from one or more scan_summary.csv files.

Each CSV row is one module/bias/VGAIN/channel acquisition.  Only rows whose
fit_status is ``ok`` contribute measurements to the plots.  The nominal
DVbias is used for legends; it is never relabelled as overvoltage without a
calibrated bias conversion.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = {"module", "bias_folder", "vgain", "channel", "fit_status", "snr"}


def finite_number(value: Optional[str]) -> Optional[float]:
    """Return a finite CSV number, or None for empty/non-numeric values."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def bias_value(bias: str) -> Optional[float]:
    match = re.search(r"(?:DVbias_)?(\d+V\d+)(?:$|[^\d])", bias)
    if match is None:
        return None
    return finite_number(match.group(1).replace("V", "."))


def bias_label(bias: str) -> str:
    value = bias_value(bias)
    return f"DVbias {value:g} V" if value is not None else bias


def bias_matches(bias: str, requested: str) -> bool:
    if bias == requested or bias.casefold() == requested.casefold():
        return True
    value = bias_value(bias)
    requested_value = bias_value(requested)
    return value is not None and requested_value is not None and math.isclose(
        value, requested_value, abs_tol=1e-8
    )


def read_rows(csv_paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in csv_paths:
        with csv_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{csv_path}: missing columns: {', '.join(sorted(missing))}")
            for line, row in enumerate(reader, start=2):
                row["_source"] = f"{csv_path}:{line}"
                rows.append(row)
    if not rows:
        raise ValueError("The supplied scan_summary.csv files contain no rows")
    return rows


def select_rows(
    rows: list[dict[str, str]], module: Optional[str], channel: Optional[int],
    biases: Optional[list[str]],
) -> tuple[list[dict[str, str]], str, int]:
    modules = sorted({row["module"] for row in rows if row["module"]})
    if module is None:
        if len(modules) != 1:
            raise ValueError(f"Choose --module from {modules}")
        module = modules[0]
    rows = [row for row in rows if row["module"] == module]
    if not rows:
        raise ValueError(f"No rows for module {module}; available: {modules}")
    channels = sorted({int(row["channel"]) for row in rows if row["channel"]})
    if channel is None:
        if len(channels) != 1:
            raise ValueError(f"Choose --channel from {channels}")
        channel = channels[0]
    rows = [row for row in rows if row["channel"] == str(channel)]
    if not rows:
        raise ValueError(f"No rows for {module}, channel {channel}; available: {channels}")
    available_biases = sorted({row["bias_folder"] for row in rows})
    if biases:
        rows = [row for row in rows if any(bias_matches(row["bias_folder"], query)
                                            for query in biases)]
        if not rows:
            raise ValueError(f"No rows for --bias {biases}; available: {available_biases}")
    return rows, module, channel


def accepted_groups(rows: list[dict[str, str]]) -> dict[str, list[dict[str, float]]]:
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    seen: dict[tuple[str, int], str] = {}
    for row in rows:
        vgain = finite_number(row["vgain"])
        if vgain is None or not vgain.is_integer():
            raise ValueError(f"{row['_source']}: invalid VGAIN {row['vgain']!r}")
        key = (row["bias_folder"], int(vgain))
        if key in seen:
            raise ValueError(f"Duplicate point {key}: {seen[key]} and {row['_source']}")
        seen[key] = row["_source"]
        if row["fit_status"].strip().lower() != "ok":
            continue
        snr = finite_number(row["snr"])
        if snr is None or snr <= 0:
            raise ValueError(f"{row['_source']}: fit_status=ok but SNR is invalid")
        snr_error = finite_number(row.get("snr_error"))
        dynamic_range = finite_number(row.get("dynamic_range_pe"))
        dynamic_range_error = finite_number(row.get("dynamic_range_error_pe"))
        groups[row["bias_folder"]].append({
            "vgain": vgain,
            "snr": snr,
            "snr_error": snr_error if snr_error is not None and snr_error >= 0 else math.nan,
            "dynamic_range_pe": dynamic_range if dynamic_range is not None and dynamic_range > 0 else math.nan,
            "dynamic_range_error_pe": dynamic_range_error
            if dynamic_range_error is not None and dynamic_range_error >= 0 else math.nan,
        })
    if not groups:
        raise ValueError("No fit_status=ok points with valid SNR in the selected scan")
    return {bias: sorted(points, key=lambda p: p["vgain"])
            for bias, points in sorted(groups.items(), key=lambda item:
                (bias_value(item[0]) is None, bias_value(item[0]) or 0, item[0]))}


def draw_error_bars(ax, x: np.ndarray, y: np.ndarray, xerr: np.ndarray,
                    yerr: np.ndarray, color: str) -> None:
    """Only draw errors that exist in the CSV; do not invent zero errors."""
    ymask = np.isfinite(yerr)
    if np.any(ymask):
        ax.errorbar(x[ymask], y[ymask], yerr=yerr[ymask], fmt="none",
                    color=color, elinewidth=1, capsize=2, zorder=2)
    xmask = np.isfinite(xerr)
    if np.any(xmask):
        ax.errorbar(x[xmask], y[xmask], xerr=xerr[xmask], fmt="none",
                    color=color, elinewidth=1, capsize=2, zorder=2)


def create_figure(
    groups: dict[str, list[dict[str, float]]], rows: list[dict[str, str]],
    module: str, channel: int,
) -> plt.Figure:
    dynamic_count = sum(math.isfinite(point["dynamic_range_pe"])
                        for points in groups.values() for point in points)
    has_dynamic_range = dynamic_count > 0
    fig, axes = plt.subplots(1, 2 if has_dynamic_range else 1,
                             figsize=(12.4, 5.8) if has_dynamic_range else (8.2, 5.8))
    axes = np.atleast_1d(axes)
    palette = plt.get_cmap("tab10")
    markers = ("o", "s", "^", "D", "v", "P", "X", "h")
    handles = []
    for i, (bias, points) in enumerate(groups.items()):
        color = palette(i % 10)
        x = np.array([point["vgain"] for point in points])
        y = np.array([point["snr"] for point in points])
        yerr = np.array([point["snr_error"] for point in points])
        handle, = axes[0].plot(x, y, marker=markers[i % len(markers)],
                               linestyle="-", color=color, markersize=4.5,
                               linewidth=1.4, label=bias_label(bias), zorder=3)
        handles.append(handle)
        draw_error_bars(axes[0], x, y, np.full_like(x, math.nan), yerr, color)
        if has_dynamic_range:
            dr = np.array([point["dynamic_range_pe"] for point in points])
            drerr = np.array([point["dynamic_range_error_pe"] for point in points])
            mask = np.isfinite(dr)
            if np.any(mask):
                axes[1].plot(dr[mask], y[mask], marker=markers[i % len(markers)],
                             linestyle="-", color=color, markersize=4.5,
                             linewidth=1.4, zorder=3)
                draw_error_bars(axes[1], dr[mask], y[mask], drerr[mask], yerr[mask], color)
    axes[0].set(xlabel="VGAIN setting", ylabel=r"SNR = $G/\sigma_0$",
                title="SNR versus VGAIN")
    if has_dynamic_range:
        axes[1].set(xlabel="Dynamic range [PE]", ylabel=r"SNR = $G/\sigma_0$",
                    title="SNR versus dynamic range")
    for ax in axes:
        ax.grid(True, which="major", alpha=0.22)
        ax.set_ylim(bottom=0)
    fig.suptitle(f"{module}  ·  channel {channel}  ·  VGAIN scan", fontsize=14)
    legend_rows = math.ceil(len(handles) / 4)
    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc="lower center", bbox_to_anchor=(0.5, 0.09),
               ncol=min(4, len(handles)), frameon=False)
    accepted = sum(len(points) for points in groups.values())
    rejected = len(rows) - accepted
    review = sum(row["fit_status"].strip().lower() == "review_required" for row in rows)
    note = (f"Accepted fits: {accepted}/{len(rows)}; review required: {review}; "
            f"other excluded: {rejected-review}. ")
    note += "Nominal DVbias; overvoltage conversion not applied."
    if has_dynamic_range and dynamic_count < accepted:
        note += f" Dynamic range available for {dynamic_count}/{accepted} accepted points."
    fig.text(0.5, 0.025, note, ha="center", va="center", fontsize=8, color="#444444")
    fig.subplots_adjust(left=0.09 if has_dynamic_range else 0.12,
                        right=0.97, top=0.85,
                        bottom=0.30 if legend_rows > 1 else 0.24, wspace=0.28)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path,
                        help="One or more scan_summary.csv files; multiple biases may be combined")
    parser.add_argument("--module", help="Module name, e.g. M1 (required if CSVs contain several)")
    parser.add_argument("--channel", type=int,
                        help="Channel number (required if CSVs contain several)")
    parser.add_argument("--bias", action="append",
                        help="Keep only this nominal bias, e.g. 42V85; repeat as needed")
    parser.add_argument("--output-prefix", type=Path, required=True,
                        help="Write PREFIX.png and PREFIX.pdf")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing plots")
    args = parser.parse_args()
    if args.output_prefix.suffix:
        parser.error("--output-prefix must have no file extension")
    output_paths = [args.output_prefix.with_suffix(".png"), args.output_prefix.with_suffix(".pdf")]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        parser.error(f"Output already exists: {', '.join(existing)}; use --overwrite to replace")
    rows, module, channel = select_rows(read_rows(args.csv), args.module, args.channel, args.bias)
    groups = accepted_groups(rows)
    fig = create_figure(groups, rows, module, channel)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_paths[0], dpi=220, facecolor="white")
    fig.savefig(output_paths[1], facecolor="white")
    plt.close(fig)
    print(f"Saved {output_paths[0]} and {output_paths[1]}")
    print(f"Accepted {sum(map(len, groups.values()))}/{len(rows)} fits across {len(groups)} biases")


if __name__ == "__main__":
    main()
