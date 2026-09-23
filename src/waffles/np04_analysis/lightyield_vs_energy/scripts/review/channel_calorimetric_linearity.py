#!/usr/bin/env python3
r"""Analisi della linearita' calorimetrica canale per canale per APA 1 e APA 2.

SCOPO
    Seleziona i trigger non-muon-like con la media PE di APA 1 e studia la
    distribuzione di N_PE per ogni canale, APA e momento nominale. Per ogni
    distribuzione esegue un fit Langauss binned con devianza di Poisson. Il
    picco numerico della convoluzione viene adattato in funzione di K_eff con
    ODR, includendo gli errori sia su x sia su y.

    A 1 GeV/c non esiste una soglia di discriminazione: sono usati i trigger
    validi dell'APA corrispondente e il punto e' marcato come ``unselected''.
    Il fit lineare usa tutti i punti validi disponibili, purché siano almeno
    tre, e include il punto a 1 GeV/c quando il suo fit è valido.

    Per valutare il sistematico della selezione il programma ripete l'intera
    analisi per T - sigma_T, T e T + sigma_T. I PDF sono prodotti per tutte le
    configurazioni. La stessa Langauss e' usata per tutti i canali e momenti; un fit di
    qualita' debole viene segnalato nei CSV per l'ispezione.

INPUT
    --input-dir: cartella apa1_vs_apa2 con sottocartelle 1GeV, ..., 7GeV;
                 ogni blocco deve contenere photoelectron_dic_<E>GeV.json.
    --trigger-data: apa12_trigger_data.csv prodotto da
                    prepare_apa12_trigger_data.py.
    --population-fit-results: apa1_population_fit_results.csv prodotto da
                    plot_apa12_trigger_distributions.py.
    --composition: np04_beam_particle_content.csv con le rate H4-VLE.

OUTPUT
    apa1_channel_linearity_threshold_<scenario>.pdf
    apa2_channel_linearity_threshold_<scenario>.pdf
        una pagina per canale, con le cinque distribuzioni e la linearita'.
    selection_thresholds.csv
    channel_distribution_fit_results.csv
    channel_linearity_fit_results.csv
    channel_linearity_threshold_systematics.csv
    channel_availability.csv
    skipped_or_flagged_channels.csv
    report.txt
    manifest.json

ESECUZIONE (dalla cartella scripts/review su LXPlus)
    python channel_calorimetric_linearity.py \
        --input-dir ../../output/apa1_vs_apa2 \
        --trigger-data ../../output/review/apa12_trigger_data_01/apa12_trigger_data.csv \
        --population-fit-results ../../output/review/apa12_trigger_distributions_01/apa1_population_fit_results.csv \
        --composition ../../data/np04_beam_particle_content.csv \
        --output-dir ../../output/review/channel_calorimetric_linearity_01

Le barre x usano sigma_Keff = sqrt(sigma_Keff,p^2 + sigma_mix^2): il primo
termine propaga l'incertezza efficace del 5% sul momento comune e il secondo
rappresenta la dispersione tra specie della miscela prevista. Le rate delle
specie non hanno incertezze fornite dalla simulazione e sono mantenute fisse.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import odr
from scipy.optimize import least_squares, minimize_scalar
from landaupy import landau
from scipy.stats import chi2 as chi2_distribution


MOMENTA = (1, 2, 3, 5, 7)
BIN_WIDTH_SCALE = 0.90
MAX_HISTOGRAM_BINS = 90
MASS_GEV = {
    "e": 0.00051099895,
    "k": 0.493677,
    "p": 0.938272088,
    "pi": 0.13957039,
}
COLORS = {
    "data": "#D9D9D9",
    "edge": "#333333",
    "langauss": "#D55E00",
    "one_gev": "#C33232",
    "fit_all": "#D55E00",
    "point_high": "#0072B2",
    "text": "#222222",
}

DISTRIBUTION_FIELDS = [
    "threshold_scenario", "threshold_sigma_multiplier", "apa", "endpoint",
    "channel", "momentum_GeV_c", "selection_kind", "threshold_nominal_PE",
    "threshold_error_PE", "threshold_applied_PE", "input_triggers_seen",
    "triggers_selected", "channel_values_available", "entries_total",
    "entries_in_fit_range", "entries_below_fit_range", "entries_above_fit_range",
    "model", "status", "message", "fit_minimum_PE", "fit_maximum_PE",
    "bin_width_PE", "histogram_bins", "mpv_PE", "mpv_error_PE", "eta_PE",
    "eta_error_PE", "langauss_sigma_PE", "langauss_sigma_error_PE", "yield",
    "yield_error", "peak_PE", "peak_error_PE", "poisson_deviance", "ndf",
    "chi2_per_ndf", "p_value", "pearson_chi2", "pearson_chi2_per_ndf",
    "r_squared", "langauss_integration_points", "optimizer_success",
    "optimizer_status", "function_evaluations", "jacobian_rank",
    "jacobian_condition_number", "covariance_valid", "covariance_method",
    "fit_drawable", "response_valid", "parameters_near_bounds", "quality_flag",
]
LINEARITY_FIELDS = [
    "threshold_scenario", "threshold_sigma_multiplier", "apa", "endpoint",
    "channel", "fit_range", "response_estimator", "required_momenta_GeV_c",
    "available_momenta_GeV_c", "status", "message", "points",
    "slope_PE_per_GeV", "slope_error_PE_per_GeV", "intercept_PE",
    "intercept_error_PE", "chi2", "ndf", "chi2_per_ndf", "p_value",
    "r_squared", "odr_info", "odr_stopreason",
]
SYSTEMATIC_FIELDS = [
    "apa", "endpoint", "channel", "fit_range", "nominal_status",
    "nominal_slope_PE_per_GeV", "nominal_intercept_PE", "minus_slope_PE_per_GeV",
    "plus_slope_PE_per_GeV", "slope_threshold_systematic_PE_per_GeV",
    "minus_intercept_PE", "plus_intercept_PE", "intercept_threshold_systematic_PE",
]
AVAILABILITY_FIELDS = [
    "threshold_scenario", "threshold_sigma_multiplier", "apa", "endpoint",
    "channel", "momentum_GeV_c", "selection_kind", "threshold_applied_PE",
    "input_triggers_seen", "triggers_selected", "channel_values_available",
    "entries_total", "status", "quality_flag",
]

FLAGGED_FIELDS = [
    "threshold_scenario", "apa", "endpoint", "channel", "momentum_GeV_c",
    "kind", "detail",
]


def exact_integer(value: object) -> int:
    number = Decimal(str(value).strip())
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError("atteso un intero finito")
    return int(number)


def finite_float(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("atteso un numero finito")
    return number


def parse_flag(value: object) -> int:
    text = str(value).strip()
    if text not in {"0", "1"}:
        raise ValueError("atteso flag 0 oppure 1")
    return int(text)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def file_info(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def scenario_name(multiplier: float) -> str:
    if math.isclose(multiplier, 0.0):
        return "nominal"
    sign = "plus" if multiplier > 0 else "minus"
    magnitude = abs(multiplier)
    token = f"{magnitude:g}".replace(".", "p")
    return f"{sign}_{token}sigma"


def clear_owned_outputs(output_dir: Path) -> None:
    names = (
        "selection_thresholds.csv", "channel_distribution_fit_results.csv",
        "channel_linearity_fit_results.csv", "channel_linearity_threshold_systematics.csv",
        "channel_availability.csv", "skipped_or_flagged_channels.csv", "report.txt",
        "manifest.json",
    )
    for name in names:
        path = output_dir / name
        if path.is_file():
            path.unlink()
    for path in output_dir.glob("apa[12]_channel_linearity_threshold_*.pdf"):
        path.unlink()


def load_trigger_contexts(path: Path) -> dict[int, dict[tuple[str, int], dict]]:
    required = {
        "momentum_GeV_c", "block", "trigger_time", "apa1_mean", "apa1_valid",
        "apa2_valid",
    }
    contexts: dict[int, dict[tuple[str, int], dict]] = {momentum: {} for momentum in MOMENTA}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Colonne mancanti in {path}: {missing}")
        for row in reader:
            momentum = exact_integer(row["momentum_GeV_c"])
            if momentum not in contexts:
                continue
            key = (row["block"].strip(), exact_integer(row["trigger_time"]))
            if key in contexts[momentum]:
                raise ValueError(f"Trigger duplicato nel CSV: {momentum} GeV/c, {key}")
            apa1_valid = parse_flag(row["apa1_valid"])
            apa2_valid = parse_flag(row["apa2_valid"])
            mean = finite_float(row["apa1_mean"]) if apa1_valid else math.nan
            contexts[momentum][key] = {
                "apa1_valid": apa1_valid,
                "apa2_valid": apa2_valid,
                "apa1_mean": mean,
            }
    return contexts


def load_thresholds(path: Path, multipliers: list[float]) -> tuple[dict[int, tuple[float, float]], list[dict]]:
    rows_by_momentum: dict[int, dict] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        required = {"momentum_GeV_c", "model", "status", "intersection", "intersection_error"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Colonne mancanti in {path}: {missing}")
        for row in reader:
            momentum = exact_integer(row["momentum_GeV_c"])
            if momentum in MOMENTA:
                rows_by_momentum[momentum] = row
    thresholds: dict[int, tuple[float, float]] = {}
    output_rows: list[dict] = []
    for momentum in MOMENTA:
        if momentum == 1:
            for multiplier in multipliers:
                output_rows.append({
                    "threshold_scenario": scenario_name(multiplier),
                    "threshold_sigma_multiplier": multiplier,
                    "momentum_GeV_c": momentum,
                    "selection_kind": "no_threshold_unselected",
                    "threshold_nominal_PE": "", "threshold_error_PE": "",
                    "threshold_applied_PE": "",
                })
            continue
        row = rows_by_momentum.get(momentum)
        if row is None or row["status"] != "success" or row["model"] != "langauss_plus_gaussian":
            raise ValueError(f"Soglia non valida o assente a {momentum} GeV/c")
        threshold = finite_float(row["intersection"])
        error = finite_float(row["intersection_error"])
        if error <= 0:
            raise ValueError(f"Incertezza della soglia non positiva a {momentum} GeV/c")
        thresholds[momentum] = (threshold, error)
        for multiplier in multipliers:
            output_rows.append({
                "threshold_scenario": scenario_name(multiplier),
                "threshold_sigma_multiplier": multiplier,
                "momentum_GeV_c": momentum,
                "selection_kind": "apa1_mean_greater_than_threshold",
                "threshold_nominal_PE": threshold, "threshold_error_PE": error,
                "threshold_applied_PE": threshold + multiplier * error,
            })
    return thresholds, output_rows


def load_kinetic_energies(path: Path, relative_momentum_error: float) -> dict[int, dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        required = {"Momentum [GeV/c]"} | {f"{name} [Hz]" for name in MASS_GEV}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Colonne mancanti nella composizione: {missing}")
        composition = {exact_integer(row["Momentum [GeV/c]"]): row for row in reader}
    output = {}
    for momentum in MOMENTA:
        if momentum not in composition:
            raise ValueError(f"Composizione assente a {momentum} GeV/c")
        rates = np.asarray([finite_float(composition[momentum][f"{name} [Hz]"]) for name in MASS_GEV])
        if np.any(rates < 0) or rates.sum() <= 0:
            raise ValueError(f"Rate non valide a {momentum} GeV/c")
        masses = np.asarray(list(MASS_GEV.values()))
        weights = rates / rates.sum()
        kinetic = np.sqrt(momentum ** 2 + masses ** 2) - masses
        mean = float(weights @ kinetic)
        mixture_rms = float(math.sqrt(weights @ (kinetic - mean) ** 2))
        derivative = float(weights @ (momentum / np.sqrt(momentum ** 2 + masses ** 2)))
        momentum_error = abs(derivative * momentum * relative_momentum_error)
        output[momentum] = {
            "kinetic_mean_GeV": mean,
            "momentum_error_GeV": momentum_error,
            "mixture_rms_GeV": mixture_rms,
            "effective_spread_GeV": math.hypot(momentum_error, mixture_rms),
        }
    return output


def load_channel_records(input_dir: Path, contexts: dict[int, dict[tuple[str, int], dict]]) -> tuple[dict, dict, list[dict], list[Path]]:
    """Legge i JSON una sola volta e conserva (trigger, N_PE) per canale."""
    records = {
        momentum: {1: defaultdict(list), 2: defaultdict(list)}
        for momentum in MOMENTA
    }
    active_contexts = {momentum: {} for momentum in MOMENTA}
    anomalies: list[dict] = []
    inputs: list[Path] = []
    seen_channel_record: set[tuple[int, int, str, int, int, int]] = set()

    def issue(code: str, momentum: int, block: str, trigger: object = "", apa: object = "", endpoint: object = "", channel: object = "", detail: str = "") -> None:
        anomalies.append({
            "threshold_scenario": "", "apa": apa, "endpoint": endpoint,
            "channel": channel, "momentum_GeV_c": momentum, "kind": code,
            "detail": f"block={block}; trigger={trigger}; {detail}",
        })

    for momentum in MOMENTA:
        energy_dir = input_dir / f"{momentum}GeV"
        if not energy_dir.is_dir():
            raise ValueError(f"Cartella energia assente: {energy_dir}")
        for block_dir in sorted(path for path in energy_dir.iterdir() if path.is_dir()):
            block = block_dir.name
            json_path = block_dir / f"photoelectron_dic_{momentum}GeV.json"
            if not json_path.is_file():
                continue
            if not any(key[0] == block for key in contexts[momentum]):
                continue
            inputs.append(json_path)
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                times = [exact_integer(value) for value in data["trigger time"]]
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, InvalidOperation) as exc:
                raise ValueError(f"JSON non leggibile {json_path}: {exc}") from exc
            duplicate_times = [time for time, count in Counter(times).items() if count > 1]
            if duplicate_times:
                raise ValueError(f"Tempi duplicati nel JSON {json_path}: {duplicate_times[:5]}")
            sections = {}
            for apa in (1, 2):
                try:
                    channel_dictionaries = data[str(apa)]["channel_dic"]
                except (KeyError, TypeError) as exc:
                    raise ValueError(f"Struttura APA {apa} invalida in {json_path}: {exc}") from exc
                if len(channel_dictionaries) != len(times):
                    raise ValueError(f"Lunghezza channel_dic APA {apa} incompatibile in {json_path}")
                sections[apa] = channel_dictionaries
            for index, trigger in enumerate(times):
                key = (block, trigger)
                context = contexts[momentum].get(key)
                if context is None:
                    issue("json_trigger_not_in_trigger_data", momentum, block, trigger)
                    continue
                active_contexts[momentum][key] = context
                for apa in (1, 2):
                    channel_dictionary = sections[apa][index]
                    if not isinstance(channel_dictionary, dict):
                        issue("invalid_channel_dictionary", momentum, block, trigger, apa)
                        continue
                    for endpoint_raw, channels in channel_dictionary.items():
                        try:
                            endpoint = exact_integer(endpoint_raw)
                        except (ValueError, InvalidOperation):
                            issue("invalid_endpoint", momentum, block, trigger, apa, endpoint_raw)
                            continue
                        if not isinstance(channels, dict):
                            issue("invalid_endpoint_dictionary", momentum, block, trigger, apa, endpoint)
                            continue
                        for channel_raw, channel_data in channels.items():
                            try:
                                channel = exact_integer(channel_raw)
                                value = finite_float(channel_data["n_pe"])
                            except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
                                issue("invalid_channel_pe", momentum, block, trigger, apa, endpoint, channel_raw, str(exc))
                                continue
                            duplicate_key = (momentum, apa, block, trigger, endpoint, channel)
                            if duplicate_key in seen_channel_record:
                                issue("duplicate_channel_value", momentum, block, trigger, apa, endpoint, channel)
                                continue
                            seen_channel_record.add(duplicate_key)
                            records[momentum][apa][(endpoint, channel)].append((key, value))
    if not inputs:
        raise ValueError("Nessun JSON incluso: controllare input-dir e trigger-data")
    return records, active_contexts, anomalies, inputs


def selected_trigger_keys(momentum: int, apa: int, contexts: dict, threshold: float | None) -> set[tuple[str, int]]:
    if momentum == 1:
        return {key for key, context in contexts.items() if context[f"apa{apa}_valid"]}
    return {
        key for key, context in contexts.items()
        if context["apa1_valid"] and context["apa1_mean"] > threshold
    }


def poisson_deviance_residuals(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    expected = np.clip(expected, 1.0e-12, None)
    term = np.empty_like(expected, dtype=float)
    positive = observed > 0
    term[positive] = expected[positive] - observed[positive] + observed[positive] * np.log(observed[positive] / expected[positive])
    term[~positive] = expected[~positive]
    return np.sign(observed - expected) * np.sqrt(np.maximum(2.0 * term, 0.0))


def histogram_edges(values: np.ndarray) -> np.ndarray:
    lower, upper = np.percentile(values, [0.5, 99.5])
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        lower, upper = float(np.min(values)), float(np.max(values))
    if upper <= lower:
        lower -= 0.5
        upper += 0.5
    q25, q75 = np.percentile(values, [25, 75])
    iqr = float(q75 - q25)
    width = 2.0 * iqr / len(values) ** (1.0 / 3.0) if iqr > 0 else (upper - lower) / 30.0
    width = max(BIN_WIDTH_SCALE * width, (upper - lower) / MAX_HISTOGRAM_BINS, 0.05)
    bins = int(math.ceil((upper - lower) / width))
    bins = min(MAX_HISTOGRAM_BINS, max(20, bins))
    width = (upper - lower) / bins
    start = math.floor(lower / width) * width
    stop = math.ceil(upper / width) * width
    edges = np.arange(start, stop + 0.5 * width, width)
    if len(edges) < 4:
        edges = np.linspace(lower, upper, 21)
    return edges


def failed_fit(values: np.ndarray, status: str, message: str, edges: np.ndarray | None = None) -> dict:
    observed = None
    entries_in_fit_range = 0
    entries_below_fit_range = 0
    entries_above_fit_range = 0
    bin_width = math.nan
    histogram_bins = 0
    fit_minimum = math.nan
    fit_maximum = math.nan
    if edges is not None and len(edges) >= 2:
        observed, edges = np.histogram(values, bins=edges)
        entries_in_fit_range = int(np.count_nonzero((values >= edges[0]) & (values <= edges[-1])))
        entries_below_fit_range = int(np.count_nonzero(values < edges[0]))
        entries_above_fit_range = int(np.count_nonzero(values > edges[-1]))
        bin_width = float(np.median(np.diff(edges)))
        histogram_bins = len(observed)
        fit_minimum = float(edges[0])
        fit_maximum = float(edges[-1])
    return {
        "status": status, "message": message, "entries_total": len(values),
        "entries_in_fit_range": entries_in_fit_range,
        "entries_below_fit_range": entries_below_fit_range,
        "entries_above_fit_range": entries_above_fit_range,
        "fit_minimum_PE": fit_minimum, "fit_maximum_PE": fit_maximum,
        "bin_width_PE": bin_width, "histogram_bins": histogram_bins,
        "mpv_PE": math.nan, "mpv_error_PE": math.nan, "eta_PE": math.nan,
        "eta_error_PE": math.nan, "langauss_sigma_PE": math.nan,
        "langauss_sigma_error_PE": math.nan, "yield": math.nan,
        "yield_error": math.nan, "peak_PE": math.nan, "peak_error_PE": math.nan,
        "poisson_deviance": math.nan, "ndf": 0, "chi2_per_ndf": math.nan,
        "p_value": math.nan, "pearson_chi2": math.nan,
        "pearson_chi2_per_ndf": math.nan, "r_squared": math.nan,
        "langauss_integration_points": 0, "optimizer_success": 0,
        "optimizer_status": "", "function_evaluations": 0, "jacobian_rank": 0,
        "jacobian_condition_number": math.nan, "covariance_valid": 0,
        "covariance_method": "", "fit_drawable": 0, "response_valid": 0,
        "parameters_near_bounds": "", "quality_flag": "not_fitted",
        "edges": edges, "observed": observed, "expected": None,
        "parameters": None,
    }


def langauss_integration_points(eta: float, sigma: float) -> int:
    if eta <= 0 or sigma <= 0:
        return 0
    return min(50000, max(400, int(math.ceil(120.0 * sigma / eta))))


def stable_langauss_pdf(x: np.ndarray, mpv: float, eta: float, sigma: float) -> np.ndarray:
    x_array = np.atleast_1d(np.asarray(x, dtype=float))
    original_shape = x_array.shape
    flat_x = x_array.reshape(-1)
    n_points = langauss_integration_points(float(eta), float(sigma))
    if n_points <= 0:
        return np.full(original_shape, np.nan, dtype=float)
    step = 10.0 * float(sigma) / n_points
    offsets = -5.0 * float(sigma) + (np.arange(n_points) + 0.5) * step
    gaussian_weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    gaussian_weights /= float(sigma) * math.sqrt(2.0 * math.pi)
    result = np.empty_like(flat_x)
    chunk_size = max(8, min(128, int(4_000_000 / n_points)))
    for first in range(0, len(flat_x), chunk_size):
        last = min(first + chunk_size, len(flat_x))
        convolution_axis = flat_x[None, first:last] + offsets[:, None]
        landau_values = np.asarray(
            landau.pdf(convolution_axis.reshape(-1), float(mpv), float(eta)),
            dtype=float,
        ).reshape(convolution_axis.shape)
        if not np.all(np.isfinite(landau_values)):
            return np.full(original_shape, np.nan, dtype=float)
        result[first:last] = step * np.sum(landau_values * gaussian_weights[:, None], axis=0)
    return result.reshape(original_shape)


def fit_langauss(
    values: np.ndarray,
    minimum_entries: int,
    warm_start: np.ndarray | None = None,
) -> dict:
    if len(values) == 0:
        return failed_fit(values, "insufficient_entries", f"Servono almeno {minimum_entries} trigger")
    try:
        edges = histogram_edges(values)
        if len(values) < minimum_entries:
            return failed_fit(values, "insufficient_entries", f"Servono almeno {minimum_entries} trigger", edges)
        observed, edges = np.histogram(values, bins=edges)
        in_range = (values >= edges[0]) & (values <= edges[-1])
        sample = values[in_range]
        minimum_in_range = max(20, int(math.floor(0.98 * minimum_entries)))
        if len(sample) < minimum_in_range:
            return failed_fit(values, "insufficient_entries_in_fit_range", "Campione centrale insufficiente", edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = float(np.median(np.diff(edges)))
        q25, _, q75 = np.percentile(sample, [25, 50, 75])
        robust_width = max(float((q75 - q25) / 1.349), width)
        mode = float(centers[int(np.argmax(observed))])
        span = float(edges[-1] - edges[0])
        initial = np.asarray([mode, max(0.25 * robust_width, 0.30 * width), max(0.55 * robust_width, 0.45 * width), float(len(sample))])
        lower = np.asarray([edges[0] - 0.25 * span, max(0.05, 0.05 * width), max(0.05, 0.05 * width), max(1.0, 0.20 * len(sample))])
        upper = np.asarray([edges[-1], max(span, width), max(span, width), max(10.0, 10.0 * len(values))])
        initial = np.minimum(np.maximum(initial, lower + 1.0e-8), upper - 1.0e-8)

        def expected_counts(parameters: np.ndarray) -> np.ndarray:
            density = stable_langauss_pdf(centers, *parameters[:3])
            if not np.all(np.isfinite(density)) or np.any(density < 0):
                return np.full_like(centers, np.nan, dtype=float)
            return width * parameters[3] * density

        def objective(parameters: np.ndarray) -> np.ndarray:
            expected = expected_counts(parameters)
            if not np.all(np.isfinite(expected)):
                return np.full(len(observed), 1.0e6, dtype=float)
            return poisson_deviance_residuals(observed, expected)

        def bounded_guess(candidate: np.ndarray) -> np.ndarray:
            return np.minimum(np.maximum(candidate, lower + 1.0e-8), upper - 1.0e-8)

        fallback_starts = []
        for mpv_shift, eta_scale, sigma_scale in ((-0.25, 0.50, 0.75), (0.25, 0.50, 0.75), (0.00, 1.00, 1.25)):
            candidate = initial.copy()
            candidate[0] += mpv_shift * robust_width
            candidate[1] *= eta_scale
            candidate[2] *= sigma_scale
            fallback_starts.append(bounded_guess(candidate))

        primary = initial
        if warm_start is not None:
            candidate = np.asarray(warm_start, dtype=float)
            if candidate.shape == initial.shape and np.all(np.isfinite(candidate)):
                primary = bounded_guess(candidate)
                fallback_starts.insert(0, initial)

        def optimize(guess: np.ndarray, max_evaluations: int):
            try:
                result = least_squares(
                    objective, guess, bounds=(lower, upper), method="trf",
                    x_scale="jac", max_nfev=max_evaluations,
                )
            except (ValueError, FloatingPointError):
                return None
            return result if np.isfinite(result.cost) and np.all(np.isfinite(result.x)) else None

        attempts = []
        primary_result = optimize(primary, 1500)
        if primary_result is not None:
            attempts.append(primary_result)
        if primary_result is None or not primary_result.success:
            for guess in fallback_starts:
                result = optimize(guess, 2500)
                if result is None:
                    continue
                attempts.append(result)
                if result.success:
                    break
        if not attempts:
            return failed_fit(values, "optimizer_failed", "Nessuna inizializzazione Langauss finita", edges)
        converged_attempts = [item for item in attempts if item.success]
        result = min(converged_attempts or attempts, key=lambda item: item.cost)
        parameters = result.x
        expected = expected_counts(parameters)
        if not np.all(np.isfinite(expected)):
            return failed_fit(values, "optimizer_failed", "Valore atteso Langauss non finito", edges)
        fit_drawable = True
        residuals = poisson_deviance_residuals(observed, expected)
        deviance = float(np.sum(residuals ** 2))
        ndf = int(len(observed) - len(parameters))
        rank = int(np.linalg.matrix_rank(result.jac))
        condition_number = math.inf
        covariance = None
        errors = np.full(len(parameters), math.nan)
        covariance_valid = False
        covariance_method = ""
        if ndf > 0 and rank == len(parameters):
            try:
                information = result.jac.T @ result.jac
                condition_number = float(np.linalg.cond(information))
                for method, solver in (("inverse", np.linalg.inv), ("pseudoinverse", np.linalg.pinv)):
                    try:
                        candidate = solver(information)
                    except np.linalg.LinAlgError:
                        continue
                    diagonal = np.diag(candidate)
                    if np.all(np.isfinite(candidate)) and np.all(diagonal >= 0):
                        covariance = candidate
                        covariance_valid = True
                        covariance_method = method
                        errors = np.sqrt(diagonal)
                        break
            except np.linalg.LinAlgError:
                pass

        def peak_coordinate(parameter_values: np.ndarray) -> float:
            peak_result = minimize_scalar(
                lambda coordinate: -float(parameter_values[3] * stable_langauss_pdf(np.asarray([coordinate]), *parameter_values[:3])[0]),
                bounds=(edges[0], edges[-1]), method="bounded",
                options={"xatol": max(1.0e-4, width * 1.0e-4)},
            )
            if not peak_result.success or not math.isfinite(peak_result.fun):
                raise RuntimeError("Ricerca numerica del picco Langauss non riuscita")
            return float(peak_result.x)

        peak = math.nan
        peak_error = math.nan
        try:
            peak = peak_coordinate(parameters)
            if covariance_valid and covariance is not None:
                gradient = np.zeros(len(parameters))
                gradient[0] = 1.0
                for index in (1, 2):
                    value = parameters[index]
                    step = max(abs(float(value)) * 1.0e-4, 1.0e-4)
                    plus = parameters.copy()
                    minus = parameters.copy()
                    plus[index] = min(value + step, upper[index] - 1.0e-8)
                    minus[index] = max(value - step, lower[index] + 1.0e-8)
                    denominator = plus[index] - minus[index]
                    gradient[index] = math.nan if denominator <= 0 else (peak_coordinate(plus) - peak_coordinate(minus)) / denominator
                if np.all(np.isfinite(gradient)):
                    variance = float(gradient @ covariance @ gradient)
                    if variance >= 0 and math.isfinite(variance):
                        peak_error = math.sqrt(variance)
        except (RuntimeError, FloatingPointError, ValueError):
            pass

        pearson = float(np.sum((observed - expected) ** 2 / np.clip(expected, 1.0e-12, None)))
        ss_res = float(np.sum((observed - expected) ** 2))
        ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
        tolerance = 1.0e-3 * (upper - lower)
        names = ("mpv", "eta", "langauss_sigma", "yield")
        near_bounds = [name for name, value, lower_value, upper_value, delta in zip(names, parameters, lower, upper, tolerance) if value - lower_value <= delta or upper_value - value <= delta]
        quality = "good"
        if (not covariance_valid or covariance_method != "inverse" or not math.isfinite(peak_error)
                or near_bounds or not result.success or condition_number > 1.0e10):
            quality = "review_optimizer_or_covariance"
        elif ndf <= 0 or deviance / ndf > 2.0 or (math.isfinite(r_squared) and r_squared < 0.8):
            quality = "review_shape"
        response_valid = bool(result.success and covariance_valid and math.isfinite(peak) and math.isfinite(peak_error) and peak_error > 0)
        status = "success" if response_valid else "response_uncertainty_invalid"
        return {
            "status": status, "message": str(result.message), "entries_total": len(values),
            "entries_in_fit_range": len(sample), "entries_below_fit_range": int(np.count_nonzero(values < edges[0])),
            "entries_above_fit_range": int(np.count_nonzero(values > edges[-1])),
            "fit_minimum_PE": float(edges[0]), "fit_maximum_PE": float(edges[-1]),
            "bin_width_PE": width, "histogram_bins": len(observed),
            "mpv_PE": float(parameters[0]), "mpv_error_PE": float(errors[0]),
            "eta_PE": float(parameters[1]), "eta_error_PE": float(errors[1]),
            "langauss_sigma_PE": float(parameters[2]), "langauss_sigma_error_PE": float(errors[2]),
            "yield": float(parameters[3]), "yield_error": float(errors[3]), "peak_PE": peak, "peak_error_PE": peak_error,
            "poisson_deviance": deviance, "ndf": ndf, "chi2_per_ndf": deviance / ndf if ndf > 0 else math.nan,
            "p_value": float(chi2_distribution.sf(deviance, ndf)) if ndf > 0 else math.nan,
            "pearson_chi2": pearson, "pearson_chi2_per_ndf": pearson / ndf if ndf > 0 else math.nan,
            "r_squared": r_squared, "langauss_integration_points": langauss_integration_points(parameters[1], parameters[2]),
            "optimizer_success": int(result.success), "optimizer_status": result.status,
            "function_evaluations": result.nfev, "jacobian_rank": rank, "jacobian_condition_number": condition_number,
            "covariance_valid": int(covariance_valid), "covariance_method": covariance_method,
            "fit_drawable": int(fit_drawable), "response_valid": int(response_valid),
            "parameters_near_bounds": ";".join(near_bounds),
            "quality_flag": quality, "edges": edges, "observed": observed, "expected": expected, "parameters": parameters,
        }
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        return failed_fit(values, "fit_exception", str(exc), histogram_edges(values) if len(values) else None)

def fit_linearity(points: list[dict], scenario: str, multiplier: float, apa: int, endpoint: int, channel: int, fit_range: str, required: tuple[int, ...]) -> dict:
    base = {
        "threshold_scenario": scenario, "threshold_sigma_multiplier": multiplier,
        "apa": apa, "endpoint": endpoint, "channel": channel, "fit_range": fit_range,
        "response_estimator": "langauss_peak_PE",
        "required_momenta_GeV_c": ";".join(map(str, required)),
        "available_momenta_GeV_c": ";".join(str(point["momentum_GeV_c"]) for point in points),
        "status": "", "message": "", "points": len(points), "slope_PE_per_GeV": math.nan,
        "slope_error_PE_per_GeV": math.nan, "intercept_PE": math.nan,
        "intercept_error_PE": math.nan, "chi2": math.nan, "ndf": 0,
        "chi2_per_ndf": math.nan, "p_value": math.nan, "r_squared": math.nan,
        "odr_info": "", "odr_stopreason": "",
    }
    if len(points) < 3:
        base.update(status="insufficient_valid_momenta", message="Servono almeno tre fit Langauss con picco e incertezza validi")
        return base
    x = np.asarray([point["kinetic_mean_GeV"] for point in points])
    sx = np.asarray([point["effective_spread_GeV"] for point in points])
    y = np.asarray([point["peak_PE"] for point in points])
    sy = np.asarray([point["peak_error_PE"] for point in points])
    if not np.all(np.isfinite(np.concatenate((x, sx, y, sy)))) or np.any(sx <= 0) or np.any(sy <= 0):
        base.update(status="invalid_point_uncertainty", message="Punto o incertezza non finita")
        return base
    try:
        initial = np.polyfit(x, y, 1)
        model = odr.Model(lambda beta, coordinate: beta[0] * coordinate + beta[1])
        result = odr.ODR(odr.RealData(x, y, sx=sx, sy=sy), model, beta0=initial, maxit=1000).run()
        if result.info not in (1, 2, 3, 4):
            base.update(status="odr_failed", message="; ".join(result.stopreason), odr_info=result.info, odr_stopreason="; ".join(result.stopreason))
            return base
        errors = np.asarray(result.sd_beta, dtype=float)
        if not np.all(np.isfinite(errors)) or np.any(errors <= 0):
            errors = np.sqrt(np.diag(result.cov_beta * result.res_var))
        slope, intercept = map(float, result.beta)
        prediction = slope * x + intercept
        chi_square = float(np.sum((y - prediction) ** 2 / (sy ** 2 + (slope * sx) ** 2)))
        ndf = len(points) - 2
        ss_total = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - float(np.sum((y - prediction) ** 2)) / ss_total if ss_total > 0 else math.nan
        base.update(status="success", message="; ".join(result.stopreason), slope_PE_per_GeV=slope, slope_error_PE_per_GeV=float(errors[0]), intercept_PE=intercept, intercept_error_PE=float(errors[1]), chi2=chi_square, ndf=ndf, chi2_per_ndf=chi_square / ndf if ndf > 0 else math.nan, p_value=float(chi2_distribution.sf(chi_square, ndf)) if ndf > 0 else math.nan, r_squared=r_squared, odr_info=result.info, odr_stopreason="; ".join(result.stopreason))
        return base
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        base.update(status="odr_exception", message=str(exc))
        return base

def add_work_in_progress(axis: plt.Axes) -> None:
    axis.text(0.98, 0.98, r"$\bf{ProtoDUNE\!-\!HD}$" "\nWork in Progress", transform=axis.transAxes, ha="right", va="top", fontsize=5.6, linespacing=1.0, color=COLORS["text"], zorder=10)


def format_value_with_error(value: float, error: float, precision: int = 1) -> str:
    if math.isfinite(value) and math.isfinite(error):
        return rf"({value:.{precision}f} $\pm$ {error:.{precision}f})"
    if math.isfinite(value):
        return f"{value:.{precision}f} (uncertainty unavailable)"
    return "not available"


def format_distribution_text(fit: dict) -> str:
    if not fit.get("fit_drawable", 0):
        return "Fit results\nnot available\n" + fit["message"]
    text = (
        "Fit results\n"
        + f"MPV = {format_value_with_error(fit['mpv_PE'], fit['mpv_error_PE'])} PE\n"
        + rf"$\eta$ = {format_value_with_error(fit['eta_PE'], fit['eta_error_PE'])} PE" + "\n"
        + rf"$\sigma_{{\rm LG}}$ = {format_value_with_error(fit['langauss_sigma_PE'], fit['langauss_sigma_error_PE'])} PE" + "\n"
        + rf"$N_{{\rm LG}}$ = {format_value_with_error(fit['yield'], fit['yield_error'], 0)}" + "\n"
        + rf"$x_{{\rm peak}}$ = {format_value_with_error(fit['peak_PE'], fit['peak_error_PE'])} PE" + "\n"
        + rf"$\chi^2/\mathrm{{ndf}}$ = {fit['poisson_deviance']:.1f}/{fit['ndf']} = {fit['chi2_per_ndf']:.2f}"
    )
    if fit["quality_flag"] != "good":
        text += "\nReview fit quality"
    return text


def draw_distribution(axis: plt.Axes, fit: dict, momentum: int) -> None:
    axis.set_title(f"{momentum} GeV/c", fontsize=10, loc="left")
    if fit["edges"] is None:
        axis.text(0.5, 0.50, f"{fit['entries_total']} triggers\n{fit['status']}", transform=axis.transAxes, ha="center", va="center", fontsize=8)
    else:
        axis.stairs(fit["observed"], fit["edges"], fill=True, color=COLORS["data"], edgecolor=COLORS["edge"], linewidth=0.9, label=f"Data @ {momentum} GeV/c ({fit['entries_total']} triggers)")
        if fit.get("fit_drawable", 0):
            centers = 0.5 * (fit["edges"][:-1] + fit["edges"][1:])
            label = "Langauss fit" if fit["status"] == "success" else "Langauss fit (review)"
            axis.plot(centers, fit["expected"], color=COLORS["langauss"], linewidth=1.8, label=label)
        summary = format_distribution_text(fit)
        axis.text(0.97, 0.04, summary, transform=axis.transAxes, ha="right", va="bottom", fontsize=6.15, linespacing=1.12, bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.94, "boxstyle": "square,pad=0.27"})
        axis.legend(loc="upper left", facecolor="white", framealpha=1, edgecolor="0.7", fontsize=6.4)
        x_span = float(fit["edges"][-1] - fit["edges"][0])
        x_padding = max(0.03 * x_span, 0.05)
        top = max(float(np.max(fit["observed"])), float(np.max(fit["expected"])) if fit.get("fit_drawable", 0) else 0.0, 1.0)
        axis.set_xlim(fit["edges"][0] - x_padding, fit["edges"][-1] + x_padding)
        axis.set_ylim(0.0, 1.13 * top)
    add_work_in_progress(axis)
    axis.set_xlabel(r"$N_{\mathrm{PE}}$ [PE]", fontsize=9)
    axis.set_ylabel("Counts", fontsize=9)
    axis.tick_params(direction="in", top=True, right=True, labelsize=8)
    axis.grid(linestyle="--", linewidth=0.45, alpha=0.35)


def format_fit_text(fit: dict) -> str:
    if fit["status"] != "success":
        return "Fit results\nnot available\n" + fit["message"]
    momenta = fit["available_momenta_GeV_c"].replace(";", ", ")
    return ("Fit results\n" + f"Points: {momenta} GeV/c\n" + rf"$m = ({fit['slope_PE_per_GeV']:.1f} \pm {fit['slope_error_PE_per_GeV']:.1f})$ PE/GeV" "\n" + rf"$q = ({fit['intercept_PE']:.1f} \pm {fit['intercept_error_PE']:.1f})$ PE" "\n" + rf"$\chi^2/\mathrm{{ndf}} = {fit['chi2']:.2f}/{fit['ndf']} = {fit['chi2_per_ndf']:.2f}$" "\n" + rf"$R^2 = {fit['r_squared']:.3f}$")


def draw_linearity(axis: plt.Axes, point_rows: dict[int, dict], linearity_fit: dict) -> None:
    usable = [point_rows[momentum] for momentum in MOMENTA if point_rows[momentum].get("response_valid", 0)]
    if not usable:
        axis.text(0.5, 0.5, "No valid Langauss fits", transform=axis.transAxes, ha="center", va="center", fontsize=8)
        add_work_in_progress(axis)
        return
    x = np.asarray([row["kinetic_mean_GeV"] for row in usable])
    sx = np.asarray([row["effective_spread_GeV"] for row in usable])
    y = np.asarray([row["peak_PE"] for row in usable])
    sy = np.asarray([row["peak_error_PE"] for row in usable])
    x_low, x_high = float(np.min(x - sx)), float(np.max(x + sx))
    y_low, y_high = float(np.min(y - sy)), float(np.max(y + sy))
    x_span = max(x_high - x_low, 0.1)
    y_span = max(y_high - y_low, 1.0)
    axis.set_xlim(max(0.0, x_low - 0.05 * x_span), x_high + 0.05 * x_span)
    axis.set_ylim(max(0.0, y_low - 0.08 * y_span), y_high + 0.15 * y_span)
    if linearity_fit["status"] == "success":
        limits = axis.get_xlim()
        grid = np.linspace(limits[0], limits[1], 200)
        axis.plot(grid, linearity_fit["slope_PE_per_GeV"] * grid + linearity_fit["intercept_PE"], color=COLORS["fit_all"], linewidth=2.0, label="Linear fit")
    for momentum, row in point_rows.items():
        if not row.get("response_valid", 0):
            continue
        axis.errorbar(row["kinetic_mean_GeV"], row["peak_PE"], xerr=row["effective_spread_GeV"], yerr=row["peak_error_PE"], fmt="o", color=COLORS["point_high"], ecolor=COLORS["point_high"], capsize=2.5, markersize=5.5, label="Langauss peak")
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), loc="upper left", facecolor="white", framealpha=1, edgecolor="0.7", fontsize=6.8)
    axis.text(0.98, 0.04, format_fit_text(linearity_fit), transform=axis.transAxes, ha="right", va="bottom", fontsize=6.6, linespacing=1.12, bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.95, "boxstyle": "square,pad=0.35"})
    add_work_in_progress(axis)
    axis.set_xlabel(r"$K_{\mathrm{eff}}$ [GeV]", fontsize=9)
    axis.set_ylabel(r"$x_{\mathrm{peak}}$ [PE]", fontsize=9)
    axis.set_title("Channel linearity", fontsize=10, loc="left")
    axis.tick_params(direction="in", top=True, right=True, labelsize=8)
    axis.grid(linestyle="--", linewidth=0.45, alpha=0.35)

def write_pdfs(output_dir: Path, scenarios: list[tuple[str, float]], channels: dict[int, list[tuple[int, int]]], distribution_objects: dict, linearity_rows: dict) -> None:
    for scenario, multiplier in scenarios:
        for apa in (1, 2):
            pdf_path = output_dir / f"apa{apa}_channel_linearity_threshold_{scenario}.pdf"
            with PdfPages(pdf_path) as pdf:
                for endpoint, channel in channels[apa]:
                    figure, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))
                    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.075, top=0.935, wspace=0.28, hspace=0.33)
                    figure.text(0.065, 0.975, f"APA {apa} — Endpoint {endpoint}, channel {channel}", ha="left", va="top", fontsize=12)
                    point_rows = {}
                    for panel, momentum in zip(axes.flat[:5], MOMENTA):
                        fit = distribution_objects[(scenario, apa, endpoint, channel, momentum)]
                        point_rows[momentum] = fit
                        draw_distribution(panel, fit, momentum)
                    line_fit = linearity_rows[(scenario, apa, endpoint, channel, "1_to_7_GeV_c")]
                    draw_linearity(axes.flat[5], point_rows, line_fit)
                    pdf.savefig(figure)
                    plt.close(figure)

def main() -> int:
    here = Path(__file__).resolve().parent
    analysis_dir = here.parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=analysis_dir / "output/apa1_vs_apa2")
    parser.add_argument("--trigger-data", type=Path, default=analysis_dir / "output/review/apa12_trigger_data_01/apa12_trigger_data.csv")
    parser.add_argument("--population-fit-results", type=Path, default=analysis_dir / "output/review/apa12_trigger_distributions_01/apa1_population_fit_results.csv")
    parser.add_argument("--composition", type=Path, default=analysis_dir / "data/np04_beam_particle_content.csv")
    parser.add_argument("--output-dir", type=Path, default=analysis_dir / "output/review/channel_calorimetric_linearity_01")
    parser.add_argument("--minimum-entries", type=int, default=150)
    parser.add_argument("--relative-momentum-error", type=float, default=0.05)
    parser.add_argument("--threshold-sigma-multipliers", nargs="+", type=float, default=[-1.0, 0.0, 1.0])
    args = parser.parse_args()
    for attribute in ("input_dir", "trigger_data", "population_fit_results", "composition", "output_dir"):
        setattr(args, attribute, getattr(args, attribute).expanduser().resolve())
    if args.minimum_entries < 10:
        parser.error("--minimum-entries deve essere almeno 10")
    if args.relative_momentum_error < 0:
        parser.error("--relative-momentum-error deve essere non negativo")
    if not args.threshold_sigma_multipliers:
        parser.error("Specificare almeno una variazione della soglia")
    if len(set(args.threshold_sigma_multipliers)) != len(args.threshold_sigma_multipliers):
        parser.error("Le variazioni della soglia devono essere distinte")
    for path in (args.input_dir, args.trigger_data, args.population_fit_results, args.composition):
        if not path.exists():
            parser.error(f"Input inesistente: {path}")
    if args.output_dir == args.input_dir:
        parser.error("La cartella output non puo' coincidere con input-dir")

    contexts = load_trigger_contexts(args.trigger_data)
    thresholds, threshold_rows = load_thresholds(args.population_fit_results, args.threshold_sigma_multipliers)
    kinetic = load_kinetic_energies(args.composition, args.relative_momentum_error)
    raw_records, active_contexts, flagged_rows, json_inputs = load_channel_records(args.input_dir, contexts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clear_owned_outputs(args.output_dir)

    requested_scenarios = [(scenario_name(multiplier), multiplier) for multiplier in args.threshold_sigma_multipliers]
    scenarios = [item for item in requested_scenarios if math.isclose(item[1], 0.0)]
    scenarios.extend(item for item in requested_scenarios if not math.isclose(item[1], 0.0))
    channels = {
        apa: sorted({channel_key for momentum in MOMENTA for channel_key in raw_records[momentum][apa]})
        for apa in (1, 2)
    }
    if not channels[1] and not channels[2]:
        raise ValueError("Nessun canale con dati validi nei JSON inclusi")

    distribution_rows: list[dict] = []
    distribution_objects: dict[tuple, dict] = {}
    linearity_rows: dict[tuple, dict] = {}
    selected_key_cache: dict[tuple, set] = {}
    one_gev_fit_cache: dict[tuple[int, int, int], dict] = {}
    nominal_parameters: dict[tuple[int, int, int, int], np.ndarray] = {}

    for scenario, multiplier in scenarios:
        for momentum in MOMENTA:
            threshold = None if momentum == 1 else thresholds[momentum][0] + multiplier * thresholds[momentum][1]
            for apa in (1, 2):
                selected_key_cache[(scenario, momentum, apa)] = selected_trigger_keys(momentum, apa, active_contexts[momentum], threshold)
        for apa in (1, 2):
            for endpoint, channel in channels[apa]:
                point_lookup: dict[int, dict] = {}
                for momentum in MOMENTA:
                    available = raw_records[momentum][apa].get((endpoint, channel), [])
                    selected = selected_key_cache[(scenario, momentum, apa)]
                    values = np.asarray([value for key, value in available if key in selected], dtype=float)
                    cache_key = (apa, endpoint, channel)
                    if momentum == 1 and cache_key in one_gev_fit_cache:
                        fit = deepcopy(one_gev_fit_cache[cache_key])
                    else:
                        warm_start = nominal_parameters.get((apa, endpoint, channel, momentum)) if scenario != "nominal" else None
                        fit = fit_langauss(values, args.minimum_entries, warm_start)
                        if momentum == 1:
                            one_gev_fit_cache[cache_key] = deepcopy(fit)
                    if scenario == "nominal" and fit.get("parameters") is not None:
                        nominal_parameters[(apa, endpoint, channel, momentum)] = np.asarray(fit["parameters"], dtype=float).copy()
                    fit.update({
                        "model": "langauss_binned_poisson_deviance",
                        "threshold_scenario": scenario,
                        "threshold_sigma_multiplier": multiplier,
                        "apa": apa, "endpoint": endpoint, "channel": channel,
                        "momentum_GeV_c": momentum,
                        "selection_kind": "unselected_apa_local_valid" if momentum == 1 else "apa1_mean_greater_than_threshold",
                        "threshold_nominal_PE": math.nan if momentum == 1 else thresholds[momentum][0],
                        "threshold_error_PE": math.nan if momentum == 1 else thresholds[momentum][1],
                        "threshold_applied_PE": math.nan if momentum == 1 else thresholds[momentum][0] + multiplier * thresholds[momentum][1],
                        "input_triggers_seen": len(active_contexts[momentum]),
                        "triggers_selected": len(selected),
                        "channel_values_available": len(available),
                        "kinetic_mean_GeV": kinetic[momentum]["kinetic_mean_GeV"],
                        "effective_spread_GeV": kinetic[momentum]["effective_spread_GeV"],
                    })
                    distribution_objects[(scenario, apa, endpoint, channel, momentum)] = fit
                    distribution_rows.append({field: fit.get(field, "") for field in DISTRIBUTION_FIELDS})
                    if fit["status"] != "success":
                        flagged_rows.append({
                            "threshold_scenario": scenario, "apa": apa, "endpoint": endpoint,
                            "channel": channel, "momentum_GeV_c": momentum,
                            "kind": fit["status"], "detail": fit["message"],
                        })
                    elif fit["quality_flag"] != "good":
                        flagged_rows.append({
                            "threshold_scenario": scenario, "apa": apa, "endpoint": endpoint,
                            "channel": channel, "momentum_GeV_c": momentum,
                            "kind": fit["quality_flag"], "detail": "Langauss fit completed; inspect PDF before physics use.",
                        })
                    point_lookup[momentum] = fit
                range_name = "1_to_7_GeV_c"
                required = MOMENTA
                points = [point_lookup[momentum] for momentum in required if point_lookup[momentum].get("response_valid", 0)]
                line_fit = fit_linearity(points, scenario, multiplier, apa, endpoint, channel, range_name, required)
                linearity_rows[(scenario, apa, endpoint, channel, range_name)] = line_fit
                if line_fit["status"] != "success":
                    flagged_rows.append({
                        "threshold_scenario": scenario, "apa": apa, "endpoint": endpoint,
                        "channel": channel, "momentum_GeV_c": "", "kind": line_fit["status"],
                        "detail": f"{range_name}: {line_fit['message']}",
                    })

    linearity_output = [linearity_rows[key] for key in sorted(linearity_rows)]
    systematics: list[dict] = []
    nominal = "nominal"
    minus_scenario = scenario_name(-1.0) if -1.0 in args.threshold_sigma_multipliers else None
    plus_scenario = scenario_name(1.0) if 1.0 in args.threshold_sigma_multipliers else None
    for apa in (1, 2):
        for endpoint, channel in channels[apa]:
            for fit_range in ("1_to_7_GeV_c",):
                central = linearity_rows[(nominal, apa, endpoint, channel, fit_range)] if nominal in {name for name, _ in scenarios} else None
                minus = linearity_rows.get((minus_scenario, apa, endpoint, channel, fit_range)) if minus_scenario else None
                plus = linearity_rows.get((plus_scenario, apa, endpoint, channel, fit_range)) if plus_scenario else None
                row = {
                    "apa": apa, "endpoint": endpoint, "channel": channel, "fit_range": fit_range,
                    "nominal_status": central["status"] if central else "nominal_not_requested",
                    "nominal_slope_PE_per_GeV": central["slope_PE_per_GeV"] if central else math.nan,
                    "nominal_intercept_PE": central["intercept_PE"] if central else math.nan,
                    "minus_slope_PE_per_GeV": minus["slope_PE_per_GeV"] if minus and minus["status"] == "success" else math.nan,
                    "plus_slope_PE_per_GeV": plus["slope_PE_per_GeV"] if plus and plus["status"] == "success" else math.nan,
                    "slope_threshold_systematic_PE_per_GeV": math.nan,
                    "minus_intercept_PE": minus["intercept_PE"] if minus and minus["status"] == "success" else math.nan,
                    "plus_intercept_PE": plus["intercept_PE"] if plus and plus["status"] == "success" else math.nan,
                    "intercept_threshold_systematic_PE": math.nan,
                }
                if central and central["status"] == "success":
                    slope_differences = [abs(row[name] - central["slope_PE_per_GeV"]) for name in ("minus_slope_PE_per_GeV", "plus_slope_PE_per_GeV") if math.isfinite(row[name])]
                    intercept_differences = [abs(row[name] - central["intercept_PE"]) for name in ("minus_intercept_PE", "plus_intercept_PE") if math.isfinite(row[name])]
                    row["slope_threshold_systematic_PE_per_GeV"] = max(slope_differences) if slope_differences else math.nan
                    row["intercept_threshold_systematic_PE"] = max(intercept_differences) if intercept_differences else math.nan
                systematics.append(row)

    write_csv(args.output_dir / "selection_thresholds.csv", list(threshold_rows[0]), threshold_rows)
    availability_rows = [
        {field: row[field] for field in AVAILABILITY_FIELDS}
        for row in distribution_rows
    ]
    write_csv(args.output_dir / "channel_distribution_fit_results.csv", DISTRIBUTION_FIELDS, distribution_rows)
    write_csv(args.output_dir / "channel_availability.csv", AVAILABILITY_FIELDS, availability_rows)
    write_csv(args.output_dir / "channel_linearity_fit_results.csv", LINEARITY_FIELDS, linearity_output)
    write_csv(args.output_dir / "channel_linearity_threshold_systematics.csv", SYSTEMATIC_FIELDS, systematics)
    write_csv(args.output_dir / "skipped_or_flagged_channels.csv", FLAGGED_FIELDS, flagged_rows)
    write_pdfs(args.output_dir, scenarios, channels, distribution_objects, linearity_rows)

    successful_distribution = sum(row["status"] == "success" for row in distribution_rows)
    successful_linearity = sum(row["status"] == "success" for row in linearity_output)
    report = [
        "ANALISI DI LINEARITA' CALORIMETRICA CANALE PER CANALE",
        f"Input JSON: {args.input_dir}",
        f"Trigger data: {args.trigger_data}",
        f"Fit popolazioni: {args.population_fit_results}",
        f"Output: {args.output_dir}",
        f"Momenti [GeV/c]: {list(MOMENTA)}",
        f"Minimo trigger per fit Langauss: {args.minimum_entries}",
        f"Incertezza relativa efficace sul momento: {args.relative_momentum_error:g}",
        f"Scenari soglia: {', '.join(name for name, _ in scenarios)}",
        "",
        "SELEZIONE",
        "A 2, 3, 5 e 7 GeV/c: apa1_mean > T, con T = intersezione Langauss--Gaussiana.",
        "A 1 GeV/c: nessuna soglia; ogni APA usa i propri trigger validi.",
        "Per APA 2 a 2--7 GeV/c il tag del trigger e' sempre definito da APA 1.",
        "",
        "FIT",
        "Le distribuzioni sono adattate con una Langauss binned e devianza di Poisson.",
        "Il range del fit e' P0.5--P99.5 della distribuzione; i valori esterni sono contati nel CSV.",
        "Una qualita' marcata review non cambia automaticamente il modello: il PDF va ispezionato.",
        "I JSON storici conservano una sola osservazione per endpoint-canale e trigger.",
        "Se esistono waveform duplicate dello stesso canale, occorre rigenerare gli input con una lista per canale.",
        "I fit lineari usano tutti i punti validi disponibili (almeno tre), il picco Langauss e ODR con sigma_Keff = sqrt(sigma_Keff,p^2 + sigma_mix^2).",
        f"Canali trovati: APA1={len(channels[1])}; APA2={len(channels[2])}.",
        f"Fit di distribuzione riusciti: {successful_distribution}/{len(distribution_rows)}.",
        f"Fit lineari riusciti: {successful_linearity}/{len(linearity_output)}.",
        f"Righe da controllare: {len(flagged_rows)}.",
        "",
        "SISTEMATICO DELLA SOGLIA",
        "Per ogni canale il CSV dedicato riporta l'inviluppo max(|risultato +/-1sigma - nominale|).",
        "Il punto a 1 GeV/c non varia con la soglia perche' non e' selezionato dalla soglia APA 1.",
    ]
    (args.output_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version,
        "configuration": {
            "input_dir": str(args.input_dir), "trigger_data": str(args.trigger_data),
            "population_fit_results": str(args.population_fit_results),
            "composition": str(args.composition), "output_dir": str(args.output_dir),
            "minimum_entries": args.minimum_entries,
            "relative_momentum_error": args.relative_momentum_error,
            "threshold_sigma_multipliers": args.threshold_sigma_multipliers,
            "fit_model": "langauss_binned_poisson_deviance",
        },
        "inputs": [file_info(path) for path in [args.trigger_data, args.population_fit_results, args.composition, *json_inputs]],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, csv.Error, json.JSONDecodeError, InvalidOperation) as error:
        print(f"Errore: {error}", file=sys.stderr)
        raise SystemExit(2)
