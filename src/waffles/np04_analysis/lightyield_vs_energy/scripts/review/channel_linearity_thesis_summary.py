#!/usr/bin/env python3
r"""Create thesis-ready channel-linearity tables and the APA 1--APA 2 light-yield map.

The program is a post-processing step.  It reads the CSV files created by
``channel_calorimetric_linearity.py`` and never refits any distribution.
Only channels with all five successful nominal Langauss response fits, a
successful nominal 1--7 GeV/c linear fit, and finite threshold-selection
systematics are used in the thesis products.

Output files
------------
    apa12_channel_light_yield_map.png
    apa12_channel_light_yield_map.pdf
    apa1_channel_peak_results.tex
    apa2_channel_peak_results.tex
    apa1_channel_linearity_results.tex
    apa2_channel_linearity_results.tex
    channel_linearity_thesis_channel_selection.csv
    channel_linearity_thesis_summary.txt

The first uncertainty in the map and linear-fit tables is the ODR fit
uncertainty.  The second is the selection-threshold systematic, evaluated as
the envelope of the threshold variations produced by the channel analysis.

Run from ``scripts/review`` on LXPlus, after the channel analysis:

    python channel_linearity_thesis_summary.py \
        --results-dir ../../output/review/channel_calorimetric_linearity_01
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.patches import Rectangle

from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map


MOMENTA = (1, 2, 3, 5, 7)
NOMINAL_SCENARIO = "nominal"
CURRENT_FIT_RANGE = "1_to_7_GeV_c"
LEGACY_FIT_RANGE = "including_1_GeV_c"
COLORS = {
    "text": "#222222",
    "missing": "#E8E8E8",
    "missing_edge": "#777777",
}

DISTRIBUTION_REQUIRED = {
    "threshold_scenario", "apa", "endpoint", "channel", "momentum_GeV_c",
    "status", "peak_PE", "peak_error_PE",
}
LINEARITY_REQUIRED = {
    "threshold_scenario", "apa", "endpoint", "channel", "fit_range", "status",
    "available_momenta_GeV_c", "slope_PE_per_GeV", "slope_error_PE_per_GeV",
    "intercept_PE", "intercept_error_PE",
}
SYSTEMATIC_REQUIRED = {
    "apa", "endpoint", "channel", "fit_range", "nominal_status",
    "slope_threshold_systematic_PE_per_GeV", "intercept_threshold_systematic_PE",
}


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    """Read a CSV and fail early when it does not match the analysis schema."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing columns in {path.name}: {', '.join(missing)}")
        return list(reader)


def as_int(value: object) -> int:
    number = float(str(value).strip())
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"Expected a finite integer, received {value!r}")
    return int(number)


def as_finite_float(value: object) -> float:
    number = float(str(value).strip())
    if not math.isfinite(number):
        raise ValueError(f"Expected a finite number, received {value!r}")
    return number


def status_is_success(row: dict[str, str], response: bool = False) -> bool:
    """Return true for a numerically usable fit, without filtering quality flags.

    The original analysis marks a fit for visual review also when its
    goodness-of-fit is weak.  Such a flag is retained in the selection CSV,
    but it is not an automatic exclusion criterion here because the requested
    tables deliberately do not use a chi-square cut.
    """
    if row.get("status", "").strip() != "success":
        return False
    if response and "response_valid" in row:
        return row["response_valid"].strip() in {"1", "true", "True"}
    return True


def parse_momenta(value: str) -> tuple[int, ...]:
    try:
        return tuple(sorted(as_int(item) for item in value.split(";") if item.strip()))
    except ValueError:
        return ()


def choose_fit_range(linearity_rows: list[dict[str, str]], systematic_rows: list[dict[str, str]]) -> str:
    """Select one coherent analysis version when legacy rows share the CSV.

    Some review directories contain rows from the old ``including_1_GeV_c``
    output together with the current ``1_to_7_GeV_c`` output.  They must not
    be merged channel by channel.  The current name is preferred whenever it
    is present in both source files; the legacy name is used only as fallback.
    """
    linearity_ranges = {row["fit_range"].strip() for row in linearity_rows}
    systematic_ranges = {row["fit_range"].strip() for row in systematic_rows}
    for candidate in (CURRENT_FIT_RANGE, LEGACY_FIT_RANGE):
        if candidate in linearity_ranges and candidate in systematic_ranges:
            return candidate
    raise ValueError(
        "No compatible 1--7 GeV/c fit range is available in both the linearity and systematic CSV files"
    )


def map_channels() -> dict[int, dict[tuple[int, int], tuple[int, int]]]:
    """Return the row and column of every APA 1 and APA 2 endpoint-channel."""
    output: dict[int, dict[tuple[int, int], tuple[int, int]]] = {1: {}, 2: {}}
    for apa in (1, 2):
        for row_index, row in enumerate(APA_map[apa].data):
            for column_index, unique_channel in enumerate(row):
                key = (int(unique_channel.endpoint), int(unique_channel.channel))
                output[apa][key] = (row_index, column_index)
    return output


def collect_records(
    distribution_rows: list[dict[str, str]],
    linearity_rows: list[dict[str, str]],
    systematic_rows: list[dict[str, str]],
    fit_range: str,
) -> tuple[dict[int, list[dict]], list[dict[str, object]]]:
    """Build strict, reusable channel records for the four thesis tables."""
    distributions: dict[tuple[int, int, int, int], dict[str, str]] = {}
    duplicate_distributions: set[tuple[int, int, int, int]] = set()
    for row in distribution_rows:
        if row["threshold_scenario"].strip() != NOMINAL_SCENARIO:
            continue
        key = (
            as_int(row["apa"]), as_int(row["endpoint"]),
            as_int(row["channel"]), as_int(row["momentum_GeV_c"]),
        )
        if key in distributions:
            duplicate_distributions.add(key)
        distributions[key] = row

    linearity: dict[tuple[int, int, int], dict[str, str]] = {}
    duplicate_linearity: set[tuple[int, int, int]] = set()
    for row in linearity_rows:
        if row["threshold_scenario"].strip() != NOMINAL_SCENARIO:
            continue
        if row["fit_range"].strip() != fit_range:
            continue
        key = (as_int(row["apa"]), as_int(row["endpoint"]), as_int(row["channel"]))
        if key in linearity:
            duplicate_linearity.add(key)
        linearity[key] = row

    systematics: dict[tuple[int, int, int], dict[str, str]] = {}
    duplicate_systematics: set[tuple[int, int, int]] = set()
    for row in systematic_rows:
        if row["fit_range"].strip() != fit_range:
            continue
        key = (as_int(row["apa"]), as_int(row["endpoint"]), as_int(row["channel"]))
        if key in systematics:
            duplicate_systematics.add(key)
        systematics[key] = row

    all_keys = sorted(set(linearity) | {(apa, endpoint, channel) for apa, endpoint, channel, _ in distributions})
    successful: dict[int, list[dict]] = {1: [], 2: []}
    selection_rows: list[dict[str, object]] = []

    for apa, endpoint, channel in all_keys:
        if apa not in successful:
            continue
        key = (apa, endpoint, channel)
        reasons: list[str] = []
        peaks: dict[int, tuple[float, float]] = {}
        quality_flags: list[str] = []
        for momentum in MOMENTA:
            distribution_key = (*key, momentum)
            row = distributions.get(distribution_key)
            if distribution_key in duplicate_distributions:
                reasons.append(f"duplicate nominal {momentum} GeV/c response row")
                continue
            if row is None:
                reasons.append(f"missing nominal {momentum} GeV/c response")
                continue
            if not status_is_success(row, response=True):
                reasons.append(f"invalid nominal {momentum} GeV/c response ({row.get('status', '')})")
                continue
            try:
                peak = as_finite_float(row["peak_PE"])
                peak_error = as_finite_float(row["peak_error_PE"])
                if peak_error <= 0:
                    raise ValueError("non-positive peak uncertainty")
            except ValueError as exc:
                reasons.append(f"invalid nominal {momentum} GeV/c peak ({exc})")
                continue
            peaks[momentum] = (peak, peak_error)
            quality_flag = row.get("quality_flag", "").strip()
            if quality_flag and quality_flag != "good":
                quality_flags.append(f"{momentum} GeV/c: {quality_flag}")

        linearity_row = linearity.get(key)
        if key in duplicate_linearity:
            reasons.append("duplicate nominal linear-fit row")
        elif linearity_row is None:
            reasons.append("missing nominal linear fit")
        elif not status_is_success(linearity_row):
            reasons.append(f"invalid nominal linear fit ({linearity_row.get('status', '')})")
        elif parse_momenta(linearity_row["available_momenta_GeV_c"]) != MOMENTA:
            reasons.append("nominal linear fit does not use all five momenta")

        systematic_row = systematics.get(key)
        if key in duplicate_systematics:
            reasons.append("duplicate threshold-systematic row")
        elif systematic_row is None:
            reasons.append("missing threshold systematic")
        elif systematic_row.get("nominal_status", "").strip() != "success":
            reasons.append("threshold systematic has no successful nominal fit")

        slope = slope_error = intercept = intercept_error = slope_systematic = intercept_systematic = math.nan
        if linearity_row is not None and status_is_success(linearity_row):
            try:
                slope = as_finite_float(linearity_row["slope_PE_per_GeV"])
                slope_error = as_finite_float(linearity_row["slope_error_PE_per_GeV"])
                intercept = as_finite_float(linearity_row["intercept_PE"])
                intercept_error = as_finite_float(linearity_row["intercept_error_PE"])
                if slope_error <= 0 or intercept_error <= 0:
                    raise ValueError("non-positive ODR uncertainty")
            except ValueError as exc:
                reasons.append(f"invalid nominal linear-fit parameters ({exc})")
        if systematic_row is not None and systematic_row.get("nominal_status", "").strip() == "success":
            try:
                slope_systematic = as_finite_float(systematic_row["slope_threshold_systematic_PE_per_GeV"])
                intercept_systematic = as_finite_float(systematic_row["intercept_threshold_systematic_PE"])
                if slope_systematic < 0 or intercept_systematic < 0:
                    raise ValueError("negative threshold systematic")
            except ValueError as exc:
                reasons.append(f"invalid threshold systematic ({exc})")

        included = not reasons and len(peaks) == len(MOMENTA)
        selection_rows.append({
            "apa": apa,
            "endpoint": endpoint,
            "channel": channel,
            "included_in_thesis_products": int(included),
            "reason": "; ".join(reasons),
            "quality_flags_nominal": "; ".join(quality_flags),
            "slope_PE_per_GeV": slope,
            "slope_fit_error_PE_per_GeV": slope_error,
            "slope_threshold_systematic_PE_per_GeV": slope_systematic,
            "intercept_PE": intercept,
            "intercept_fit_error_PE": intercept_error,
            "intercept_threshold_systematic_PE": intercept_systematic,
        })
        if included:
            successful[apa].append({
                "apa": apa,
                "endpoint": endpoint,
                "channel": channel,
                "peaks": peaks,
                "slope": slope,
                "slope_error": slope_error,
                "slope_systematic": slope_systematic,
                "intercept": intercept,
                "intercept_error": intercept_error,
                "intercept_systematic": intercept_systematic,
                "quality_flags": quality_flags,
            })
    return successful, selection_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def latex_number(value: float, precision: int = 1) -> str:
    """Format a finite number consistently for math-mode table cells."""
    if not math.isfinite(value):
        return r"\text{--}"
    return f"{value:.{precision}f}"


def latex_value_error(value: float, error: float, systematic: float | None = None) -> str:
    core = rf"{latex_number(value)} \pm {latex_number(error)}"
    if systematic is not None:
        core += rf" \pm {latex_systematic_number(systematic)}"
    return f"${core}$"


def latex_systematic_number(value: float) -> str:
    """Retain visible non-zero threshold systematics in the fit-result tables."""
    if not math.isfinite(value):
        return r"\text{--}"
    if abs(value) < 0.01:
        return f"{value:.3f}"
    if abs(value) < 1.0:
        return f"{value:.2f}"
    return f"{value:.1f}"


def table_preamble(caption: str, label: str, columns: str, header: str) -> list[str]:
    return [
        "% Requires the booktabs and graphicx packages.",
        r"\begin{table}[p]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]


def table_postamble() -> list[str]:
    return [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ]


def write_peak_table(path: Path, apa: int, records: list[dict]) -> None:
    caption = (
        f"Numerical Langauss peak positions for APA~{apa} channels used in the "
        "channel-by-channel linearity analysis. The nominal selection is used; "
        r"at $1~\mathrm{GeV}/c$ no muon-like selection is applied. The uncertainty "
        "is obtained from the Langauss fit. Only channels with successful response "
        "fits at all beam momenta and finite threshold-selection variations are reported."
    )
    header = (
        r"Endpoint & Channel & $x_{\mathrm{peak}}(1~\mathrm{GeV}/c)$ [PE] & "
        r"$x_{\mathrm{peak}}(2~\mathrm{GeV}/c)$ [PE] & "
        r"$x_{\mathrm{peak}}(3~\mathrm{GeV}/c)$ [PE] & "
        r"$x_{\mathrm{peak}}(5~\mathrm{GeV}/c)$ [PE] & "
        r"$x_{\mathrm{peak}}(7~\mathrm{GeV}/c)$ [PE] \\"
    )
    lines = table_preamble(caption, f"tab:apa{apa}_channel_peak_results", "rrccccc", header)
    if records:
        for record in records:
            peak_cells = " & ".join(latex_value_error(*record["peaks"][momentum]) for momentum in MOMENTA)
            lines.append(f"{record['endpoint']} & {record['channel']} & {peak_cells} " + r"\\")
    else:
        lines.append(r"\multicolumn{7}{c}{No channel satisfies the selection criteria.} \\")
    lines.extend(table_postamble())
    path.write_text("\n".join(lines), encoding="utf-8")


def write_linearity_table(path: Path, apa: int, records: list[dict]) -> None:
    caption = (
        f"Nominal linear-fit parameters for APA~{apa}. The numerical Langauss peak "
        r"is fitted as a function of $K_{\mathrm{eff}}$. Each cell is given as the "
        r"central value $\pm$ ODR fit uncertainty $\pm$ threshold-selection systematic. "
        "Only channels with successful response fits at all beam momenta and finite "
        "threshold-selection variations are reported."
    )
    header = (
        r"Endpoint & Channel & $m$ [PE/GeV] & $q$ [PE] \\"
    )
    lines = table_preamble(caption, f"tab:apa{apa}_channel_linearity_results", "rrcc", header)
    if records:
        for record in records:
            slope = latex_value_error(record["slope"], record["slope_error"], record["slope_systematic"])
            intercept = latex_value_error(record["intercept"], record["intercept_error"], record["intercept_systematic"])
            lines.append(f"{record['endpoint']} & {record['channel']} & {slope} & {intercept} " + r"\\")
    else:
        lines.append(r"\multicolumn{4}{c}{No channel satisfies the selection criteria.} \\")
    lines.extend(table_postamble())
    path.write_text("\n".join(lines), encoding="utf-8")


def add_work_in_progress(axis: plt.Axes) -> None:
    axis.text(
        0.985, 0.985, r"$\bf{ProtoDUNE\!-\!HD}$" "\nWork in Progress",
        transform=axis.transAxes, ha="right", va="top", fontsize=10.5,
        linespacing=0.95, color=COLORS["text"], zorder=10,
    )


def draw_light_yield_map(records: dict[int, list[dict]], output_dir: Path, dpi: int) -> tuple[Path, Path]:
    """Draw one physical-layout map with APA 1 and APA 2 in the reference style."""
    maps = map_channels()
    lookup = {
        (record["apa"], record["endpoint"], record["channel"]): record
        for apa_records in records.values() for record in apa_records
    }
    all_slopes = np.asarray([record["slope"] for apa_records in records.values() for record in apa_records])
    if len(all_slopes) == 0:
        raise ValueError("No fully successful channels are available for the light-yield map")
    vmax = max(float(np.max(all_slopes)), 1.0)
    norm = colors.Normalize(vmin=0.0, vmax=vmax, clip=True)
    colour_map = cm.get_cmap("YlOrRd")

    figure, axis = plt.subplots(figsize=(13.4, 8.2), constrained_layout=True)
    cell_width = 43.0
    cell_height = 17.0
    column_pitch = 50.0
    apa_x_origin = {1: 10.0, 2: 260.0}
    row_centres = {row: 570.0 - 57.0 * row for row in range(10)}

    for apa in (1, 2):
        for (endpoint, channel), (row, column) in maps[apa].items():
            x_left = apa_x_origin[apa] + column * column_pitch
            y_centre = row_centres[row]
            record = lookup.get((apa, endpoint, channel))
            if record is None:
                patch = Rectangle(
                    (x_left, y_centre - 0.5 * cell_height), cell_width, cell_height,
                    facecolor=COLORS["missing"], edgecolor=COLORS["missing_edge"],
                    linewidth=0.8, hatch="//", zorder=2,
                )
                axis.add_patch(patch)
                axis.text(x_left + 0.5 * cell_width, y_centre, "not\navailable", ha="center", va="center", fontsize=4.7, color="#555555", zorder=3)
            else:
                patch = Rectangle(
                    (x_left, y_centre - 0.5 * cell_height), cell_width, cell_height,
                    facecolor=colour_map(norm(record["slope"])), edgecolor="#222222",
                    linewidth=0.9, zorder=2,
                )
                axis.add_patch(patch)
                label = rf"$ {latex_number(record['slope'])} \pm {latex_number(record['slope_error'])} $"
                axis.text(x_left + 0.5 * cell_width, y_centre, label, ha="center", va="center", fontsize=7.2, color=COLORS["text"], zorder=3)
            axis.text(x_left + 0.5 * cell_width, y_centre - 12.5, f"END {endpoint} - CH {channel}", ha="center", va="center", fontsize=5.5, color="#333333", zorder=3)

    axis.text(9.0, 603.5, "APA 1", ha="left", va="bottom", fontsize=11, fontweight="bold")
    axis.text(259.0, 603.5, "APA 2", ha="left", va="bottom", fontsize=11, fontweight="bold")
    add_work_in_progress(axis)

    axis.set_xlim(-12.0, 475.0)
    axis.set_ylim(-4.0, 620.0)
    axis.set_xlabel(r"$z$ direction [cm]", fontsize=13)
    axis.set_ylabel(r"$y$ direction [cm]", fontsize=13)
    axis.tick_params(direction="in", top=True, right=True, labelsize=10)
    axis.grid(linestyle="--", linewidth=0.45, alpha=0.28, zorder=0)

    mapper = cm.ScalarMappable(norm=norm, cmap=colour_map)
    mapper.set_array([])
    colour_bar = figure.colorbar(mapper, ax=axis, pad=0.035, fraction=0.046)
    colour_bar.set_label(r"Effective detected light yield ($m$) [PE/GeV]", fontsize=12)
    colour_bar.ax.tick_params(labelsize=10)

    png_path = output_dir / "apa12_channel_light_yield_map.png"
    pdf_path = output_dir / "apa12_channel_light_yield_map.pdf"
    figure.savefig(png_path, dpi=dpi)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def write_summary(path: Path, records: dict[int, list[dict]], selection_rows: list[dict[str, object]], map_paths: tuple[Path, Path], fit_range: str) -> None:
    included = sum(bool(row["included_in_thesis_products"]) for row in selection_rows)
    excluded = len(selection_rows) - included
    reviewed = sum(bool(row["quality_flags_nominal"]) for row in selection_rows if row["included_in_thesis_products"])
    lines = [
        "THESIS-READY CHANNEL LINEARITY OUTPUTS",
        "",
        "The table and map selection requires:",
        "  * successful nominal Langauss response fits at 1, 2, 3, 5, and 7 GeV/c;",
        "  * a successful nominal ODR fit using all five momenta;",
        "  * finite threshold-selection systematics for slope and intercept.",
        "",
        "No automatic chi-square or quality-flag rejection is applied.",
        "Quality flags remain available in channel_linearity_thesis_channel_selection.csv.",
        f"Fit-range source used: {fit_range}",
        "",
        f"APA 1 included channels: {len(records[1])}",
        f"APA 2 included channels: {len(records[2])}",
        f"Included channels total: {included}",
        f"Excluded candidates: {excluded}",
        f"Included channels with a nominal visual-review flag: {reviewed}",
        "",
        "Map colour: nominal slope m.",
        "Map cell: m +/- ODR fit uncertainty. Threshold-selection systematics are reported in the LaTeX tables.",
        f"Map PNG: {map_paths[0].name}",
        f"Map PDF: {map_paths[1].name}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    here = Path(__file__).resolve().parent
    analysis_dir = here.parent.parent
    default_results_dir = analysis_dir / "output/review/channel_calorimetric_linearity_01"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir, help="Directory containing the three channel-analysis CSV files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for map, tables, and selection summary (default: results directory)")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300)")
    args = parser.parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = (args.output_dir or results_dir).expanduser().resolve()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")

    distribution_rows = read_csv(results_dir / "channel_distribution_fit_results.csv", DISTRIBUTION_REQUIRED)
    linearity_rows = read_csv(results_dir / "channel_linearity_fit_results.csv", LINEARITY_REQUIRED)
    systematic_rows = read_csv(results_dir / "channel_linearity_threshold_systematics.csv", SYSTEMATIC_REQUIRED)
    fit_range = choose_fit_range(linearity_rows, systematic_rows)
    records, selection_rows = collect_records(distribution_rows, linearity_rows, systematic_rows, fit_range)
    if not records[1] and not records[2]:
        raise ValueError("No channel satisfies the complete thesis-selection criteria")

    output_dir.mkdir(parents=True, exist_ok=True)
    selection_fields = [
        "apa", "endpoint", "channel", "included_in_thesis_products", "reason",
        "quality_flags_nominal", "slope_PE_per_GeV", "slope_fit_error_PE_per_GeV",
        "slope_threshold_systematic_PE_per_GeV", "intercept_PE", "intercept_fit_error_PE",
        "intercept_threshold_systematic_PE",
    ]
    write_csv(output_dir / "channel_linearity_thesis_channel_selection.csv", selection_fields, selection_rows)
    write_peak_table(output_dir / "apa1_channel_peak_results.tex", 1, records[1])
    write_peak_table(output_dir / "apa2_channel_peak_results.tex", 2, records[2])
    write_linearity_table(output_dir / "apa1_channel_linearity_results.tex", 1, records[1])
    write_linearity_table(output_dir / "apa2_channel_linearity_results.tex", 2, records[2])
    map_paths = draw_light_yield_map(records, output_dir, args.dpi)
    write_summary(output_dir / "channel_linearity_thesis_summary.txt", records, selection_rows, map_paths, fit_range)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
