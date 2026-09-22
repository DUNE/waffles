#!/usr/bin/env python3
r"""Analisi della linearita' calorimetrica canale per canale per APA 1 e APA 2.

SCOPO
    Seleziona i trigger non-muon-like con la media PE di APA 1 e studia la
    distribuzione di N_PE per ogni canale, APA e momento nominale. Per ogni
    distribuzione esegue un fit Gaussiano binned con devianza di Poisson.
    La media del fit viene quindi adattata in funzione di K_eff con ODR,
    includendo gli errori sia su x sia su y.

    A 1 GeV/c non esiste una soglia di discriminazione: sono usati i trigger
    validi dell'APA corrispondente e il punto e' marcato come ``unselected''.
    I fit lineari 2--7 GeV/c sono il risultato principale; quelli 1--7 GeV/c
    sono un controllo di sensibilita'.

    Per valutare il sistematico della selezione il programma ripete l'intera
    analisi per T - sigma_T, T e T + sigma_T. I PDF sono prodotti per tutte le
    configurazioni. La scelta del modello resta una Gaussiana per tutti i
    canali e momenti: un fit di qualita' debole viene segnalato, ma non viene
    automaticamente sostituito da una Langauss.

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
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import odr
from scipy.optimize import least_squares
from scipy.special import ndtr
from scipy.stats import chi2 as chi2_distribution


MOMENTA = (1, 2, 3, 5, 7)
MASS_GEV = {
    "e": 0.00051099895,
    "k": 0.493677,
    "p": 0.938272088,
    "pi": 0.13957039,
}
COLORS = {
    "data": "#D9D9D9",
    "edge": "#333333",
    "gaussian": "#0072B2",
    "one_gev": "#C33232",
    "fit_all": "#D55E00",
    "fit_high": "#173F6B",
    "text": "#222222",
}

DISTRIBUTION_FIELDS = [
    "threshold_scenario", "threshold_sigma_multiplier", "apa", "endpoint",
    "channel", "momentum_GeV_c", "selection_kind", "threshold_nominal_PE",
    "threshold_error_PE", "threshold_applied_PE", "input_triggers_seen",
    "triggers_selected", "channel_values_available", "entries_total",
    "entries_in_fit_range", "entries_below_fit_range", "entries_above_fit_range",
    "model", "status", "message", "fit_minimum_PE", "fit_maximum_PE",
    "bin_width_PE", "histogram_bins", "mu_PE", "mu_error_PE", "sigma_PE",
    "sigma_error_PE", "yield", "yield_error", "poisson_deviance", "ndf",
    "chi2_per_ndf", "p_value", "pearson_chi2", "pearson_chi2_per_ndf",
    "r_squared", "optimizer_success", "optimizer_status", "function_evaluations",
    "jacobian_rank", "covariance_valid", "parameters_near_bounds", "quality_flag",
]
LINEARITY_FIELDS = [
    "threshold_scenario", "threshold_sigma_multiplier", "apa", "endpoint",
    "channel", "fit_range", "required_momenta_GeV_c", "available_momenta_GeV_c",
    "status", "message", "points", "slope_PE_per_GeV",
    "slope_error_PE_per_GeV", "intercept_PE", "intercept_error_PE", "chi2",
    "ndf", "chi2_per_ndf", "p_value", "r_squared", "odr_info", "odr_stopreason",
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
        "skipped_or_flagged_channels.csv", "report.txt", "manifest.json",
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
    width = max(width, (upper - lower) / 80.0, 0.05)
    bins = int(math.ceil((upper - lower) / width))
    bins = min(80, max(20, bins))
    width = (upper - lower) / bins
    start = math.floor(lower / width) * width
    stop = math.ceil(upper / width) * width
    edges = np.arange(start, stop + 0.5 * width, width)
    if len(edges) < 4:
        edges = np.linspace(lower, upper, 21)
    return edges


def failed_fit(values: np.ndarray, status: str, message: str) -> dict:
    return {
        "status": status, "message": message, "entries_total": len(values),
        "entries_in_fit_range": 0, "entries_below_fit_range": 0,
        "entries_above_fit_range": 0, "fit_minimum_PE": math.nan,
        "fit_maximum_PE": math.nan, "bin_width_PE": math.nan, "histogram_bins": 0,
        "mu_PE": math.nan, "mu_error_PE": math.nan, "sigma_PE": math.nan,
        "sigma_error_PE": math.nan, "yield": math.nan, "yield_error": math.nan,
        "poisson_deviance": math.nan, "ndf": 0, "chi2_per_ndf": math.nan,
        "p_value": math.nan, "pearson_chi2": math.nan,
        "pearson_chi2_per_ndf": math.nan, "r_squared": math.nan,
        "optimizer_success": 0, "optimizer_status": "", "function_evaluations": 0,
        "jacobian_rank": 0, "covariance_valid": 0, "parameters_near_bounds": "",
        "quality_flag": "not_fitted", "edges": None, "observed": None,
        "expected": None,
    }


def fit_gaussian(values: np.ndarray, minimum_entries: int) -> dict:
    if len(values) < minimum_entries:
        return failed_fit(values, "insufficient_entries", f"Servono almeno {minimum_entries} valori")
    try:
        edges = histogram_edges(values)
        observed, edges = np.histogram(values, bins=edges)
        in_range = (values >= edges[0]) & (values <= edges[-1])
        sample = values[in_range]
        if len(sample) < minimum_entries:
            return failed_fit(values, "insufficient_entries_in_fit_range", "Campione centrale insufficiente")
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = float(np.median(np.diff(edges)))
        q25, median, q75 = np.percentile(sample, [25, 50, 75])
        robust_sigma = max(float((q75 - q25) / 1.349), width * 0.75)
        fraction_initial = ndtr((edges[-1] - median) / robust_sigma) - ndtr((edges[0] - median) / robust_sigma)
        initial = np.asarray([median, robust_sigma, len(sample) / max(fraction_initial, 0.05)])
        span = float(edges[-1] - edges[0])
        lower_bounds = np.asarray([edges[0], width * 0.2, len(sample) * 0.20])
        upper_bounds = np.asarray([edges[-1], span * 2.0, len(values) * 10.0])
        initial = np.minimum(np.maximum(initial, lower_bounds + 1.0e-8), upper_bounds - 1.0e-8)

        def expected_counts(parameters: np.ndarray) -> np.ndarray:
            mu, sigma, yield_value = parameters
            probabilities = ndtr((edges[1:] - mu) / sigma) - ndtr((edges[:-1] - mu) / sigma)
            return yield_value * np.clip(probabilities, 1.0e-15, None)

        def objective(parameters: np.ndarray) -> np.ndarray:
            return poisson_deviance_residuals(observed, expected_counts(parameters))

        starts = [initial]
        for mean_shift in (-0.25, 0.25):
            for sigma_scale in (0.7, 1.4):
                candidate = initial.copy()
                candidate[0] += mean_shift * robust_sigma
                candidate[1] *= sigma_scale
                starts.append(np.minimum(np.maximum(candidate, lower_bounds + 1.0e-8), upper_bounds - 1.0e-8))
        attempts = []
        for start in starts:
            result = least_squares(objective, start, bounds=(lower_bounds, upper_bounds), method="trf", x_scale="jac", max_nfev=10000)
            if np.isfinite(result.cost):
                attempts.append(result)
        if not attempts:
            return failed_fit(values, "optimizer_failed", "Nessuna inizializzazione finita")
        result = min(attempts, key=lambda candidate: candidate.cost)
        parameters = result.x
        expected = expected_counts(parameters)
        residuals = poisson_deviance_residuals(observed, expected)
        deviance = float(np.sum(residuals ** 2))
        ndf = int(len(observed) - len(parameters))
        rank = int(np.linalg.matrix_rank(result.jac))
        covariance_valid = bool(result.success and ndf > 0 and rank == len(parameters))
        covariance = None
        errors = np.full(3, math.nan)
        if covariance_valid:
            try:
                covariance = np.linalg.inv(result.jac.T @ result.jac)
                diagonal = np.diag(covariance)
                covariance_valid = bool(np.all(np.isfinite(covariance)) and np.all(diagonal >= 0))
                if covariance_valid:
                    errors = np.sqrt(diagonal)
                else:
                    covariance = None
            except np.linalg.LinAlgError:
                covariance_valid = False
        pearson = float(np.sum((observed - expected) ** 2 / np.clip(expected, 1.0e-12, None)))
        ss_res = float(np.sum((observed - expected) ** 2))
        ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
        tolerance = 1.0e-3 * (upper_bounds - lower_bounds)
        names = ("mu", "sigma", "yield")
        near_bounds = [name for name, value, lower, upper, delta in zip(names, parameters, lower_bounds, upper_bounds, tolerance) if value - lower <= delta or upper - value <= delta]
        quality = "good"
        if not covariance_valid or near_bounds or not result.success:
            quality = "review_optimizer_or_covariance"
        elif ndf <= 0 or deviance / ndf > 2.0 or (math.isfinite(r_squared) and r_squared < 0.8):
            quality = "review_shape"
        return {
            "status": "success" if result.success and covariance_valid else "optimizer_or_covariance_invalid",
            "message": str(result.message), "entries_total": len(values),
            "entries_in_fit_range": len(sample), "entries_below_fit_range": int(np.count_nonzero(values < edges[0])),
            "entries_above_fit_range": int(np.count_nonzero(values > edges[-1])),
            "fit_minimum_PE": float(edges[0]), "fit_maximum_PE": float(edges[-1]),
            "bin_width_PE": width, "histogram_bins": len(observed), "mu_PE": float(parameters[0]),
            "mu_error_PE": float(errors[0]), "sigma_PE": float(parameters[1]),
            "sigma_error_PE": float(errors[1]), "yield": float(parameters[2]),
            "yield_error": float(errors[2]), "poisson_deviance": deviance, "ndf": ndf,
            "chi2_per_ndf": deviance / ndf if ndf > 0 else math.nan,
            "p_value": float(chi2_distribution.sf(deviance, ndf)) if ndf > 0 else math.nan,
            "pearson_chi2": pearson, "pearson_chi2_per_ndf": pearson / ndf if ndf > 0 else math.nan,
            "r_squared": r_squared, "optimizer_success": int(result.success),
            "optimizer_status": result.status, "function_evaluations": result.nfev,
            "jacobian_rank": rank, "covariance_valid": int(covariance_valid),
            "parameters_near_bounds": ";".join(near_bounds), "quality_flag": quality,
            "edges": edges, "observed": observed, "expected": expected,
        }
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        return failed_fit(values, "fit_exception", str(exc))


def fit_linearity(points: list[dict], scenario: str, multiplier: float, apa: int, endpoint: int, channel: int, fit_range: str, required: tuple[int, ...]) -> dict:
    base = {
        "threshold_scenario": scenario, "threshold_sigma_multiplier": multiplier,
        "apa": apa, "endpoint": endpoint, "channel": channel, "fit_range": fit_range,
        "required_momenta_GeV_c": ";".join(map(str, required)),
        "available_momenta_GeV_c": ";".join(str(point["momentum_GeV_c"]) for point in points),
        "status": "", "message": "", "points": len(points), "slope_PE_per_GeV": math.nan,
        "slope_error_PE_per_GeV": math.nan, "intercept_PE": math.nan,
        "intercept_error_PE": math.nan, "chi2": math.nan, "ndf": 0,
        "chi2_per_ndf": math.nan, "p_value": math.nan, "r_squared": math.nan,
        "odr_info": "", "odr_stopreason": "",
    }
    if len(points) != len(required):
        base.update(status="insufficient_valid_momenta", message="Uno o piu' fit Gaussiani non sono validi")
        return base
    x = np.asarray([point["kinetic_mean_GeV"] for point in points])
    sx = np.asarray([point["effective_spread_GeV"] for point in points])
    y = np.asarray([point["mu_PE"] for point in points])
    sy = np.asarray([point["mu_error_PE"] for point in points])
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
        errors = np.sqrt(np.diag(result.cov_beta))
        slope, intercept = map(float, result.beta)
        prediction = slope * x + intercept
        chi_square = float(np.sum((y - prediction) ** 2 / (sy ** 2 + (slope * sx) ** 2)))
        ndf = len(points) - 2
        ss_total = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - float(np.sum((y - prediction) ** 2)) / ss_total if ss_total > 0 else math.nan
        base.update(
            status="success", message="; ".join(result.stopreason), slope_PE_per_GeV=slope,
            slope_error_PE_per_GeV=float(errors[0]), intercept_PE=intercept,
            intercept_error_PE=float(errors[1]), chi2=chi_square, ndf=ndf,
            chi2_per_ndf=chi_square / ndf if ndf > 0 else math.nan,
            p_value=float(chi2_distribution.sf(chi_square, ndf)) if ndf > 0 else math.nan,
            r_squared=r_squared, odr_info=result.info, odr_stopreason="; ".join(result.stopreason),
        )
        return base
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        base.update(status="odr_exception", message=str(exc))
        return base


def draw_distribution(axis: plt.Axes, fit: dict, momentum: int, selection_text: str) -> None:
    axis.set_title(f"{momentum} GeV/c", fontsize=10, loc="left")
    if fit["edges"] is None:
        axis.text(0.5, 0.55, f"{fit['entries_total']} values\n{fit['status']}", transform=axis.transAxes, ha="center", va="center", fontsize=9)
    else:
        axis.stairs(fit["observed"], fit["edges"], fill=True, color=COLORS["data"], edgecolor=COLORS["edge"], linewidth=0.9)
        if fit["status"] == "success":
            grid = np.linspace(fit["edges"][0], fit["edges"][-1], 500)
            density = fit["yield"] * np.exp(-0.5 * ((grid - fit["mu_PE"]) / fit["sigma_PE"]) ** 2) / (fit["sigma_PE"] * math.sqrt(2.0 * math.pi))
            axis.plot(grid, density * fit["bin_width_PE"], color=COLORS["gaussian"], linewidth=1.8)
            summary = (
                rf"$\mu = ({fit['mu_PE']:.1f} \pm {fit['mu_error_PE']:.1f})$ PE" "\n"
                rf"$\sigma = ({fit['sigma_PE']:.1f} \pm {fit['sigma_error_PE']:.1f})$ PE" "\n"
                rf"$\chi^2/\mathrm{{ndf}} = {fit['poisson_deviance']:.1f}/{fit['ndf']} = {fit['chi2_per_ndf']:.2f}$"
            )
            axis.text(0.97, 0.96, summary, transform=axis.transAxes, ha="right", va="top", fontsize=7.2, linespacing=1.2, bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.92, "boxstyle": "square,pad=0.28"})
        if fit["entries_below_fit_range"] or fit["entries_above_fit_range"]:
            axis.text(0.03, 0.05, f"{fit['entries_below_fit_range'] + fit['entries_above_fit_range']} values outside fit range", transform=axis.transAxes, ha="left", va="bottom", fontsize=6.5)
    axis.text(0.03, 0.96, selection_text, transform=axis.transAxes, ha="left", va="top", fontsize=6.6, color="0.30")
    axis.set_xlabel(r"$N_{\mathrm{PE}}$ [PE]", fontsize=9)
    axis.set_ylabel("Trigger counts", fontsize=9)
    axis.tick_params(direction="in", top=True, right=True, labelsize=8)
    axis.grid(linestyle="--", linewidth=0.45, alpha=0.35)


def format_fit_text(fit: dict) -> str:
    if fit["status"] != "success":
        return f"{fit['fit_range']}\n{fit['status']}"
    return (
        rf"$m = ({fit['slope_PE_per_GeV']:.1f} \pm {fit['slope_error_PE_per_GeV']:.1f})$ PE/GeV" "\n"
        rf"$q = ({fit['intercept_PE']:.1f} \pm {fit['intercept_error_PE']:.1f})$ PE" "\n"
        rf"$\chi^2/\mathrm{{ndf}} = {fit['chi2']:.2f}/{fit['ndf']} = {fit['chi2_per_ndf']:.2f}$" "\n"
        rf"$R^2 = {fit['r_squared']:.3f}$"
    )


def draw_linearity(axis: plt.Axes, point_rows: dict[int, dict], linearity_all: dict, linearity_high: dict, scenario: str) -> None:
    usable = [point_rows[momentum] for momentum in MOMENTA if point_rows[momentum]["status"] == "success"]
    if not usable:
        axis.text(0.5, 0.5, "No valid Gaussian fits", transform=axis.transAxes, ha="center", va="center")
        return
    x = np.asarray([row["kinetic_mean_GeV"] for row in usable])
    y = np.asarray([row["mu_PE"] for row in usable])
    sx = np.asarray([row["effective_spread_GeV"] for row in usable])
    sy = np.asarray([row["mu_error_PE"] for row in usable])
    if linearity_all["status"] == "success":
        grid = np.linspace(max(0.0, x.min() - 0.3), x.max() + 0.3, 200)
        axis.plot(grid, linearity_all["slope_PE_per_GeV"] * grid + linearity_all["intercept_PE"], color=COLORS["fit_all"], linewidth=2.0, label="Linear fit: 1–7 GeV/c")
    if linearity_high["status"] == "success":
        grid = np.linspace(max(0.0, x.min() - 0.3), x.max() + 0.3, 200)
        axis.plot(grid, linearity_high["slope_PE_per_GeV"] * grid + linearity_high["intercept_PE"], color=COLORS["fit_high"], linewidth=2.0, linestyle="--", label="Linear fit: 2–7 GeV/c")
    for momentum, row in point_rows.items():
        if row["status"] != "success":
            continue
        is_one = momentum == 1
        color = COLORS["one_gev"] if is_one else COLORS["gaussian"]
        marker = "D" if is_one else "o"
        label = "Gaussian mean (1 GeV/c, unselected)" if is_one else "Gaussian mean (2–7 GeV/c)"
        axis.errorbar(row["kinetic_mean_GeV"], row["mu_PE"], xerr=row["effective_spread_GeV"], yerr=row["mu_error_PE"], fmt=marker, color=color, ecolor=color, capsize=2.5, markersize=5.5, label=label)
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), loc="upper left", facecolor="white", framealpha=1, edgecolor="0.7", fontsize=6.8)
    summary = "Fit including 1 GeV/c\n" + format_fit_text(linearity_all) + "\n\nFit excluding 1 GeV/c\n" + format_fit_text(linearity_high)
    axis.text(0.98, 0.04, summary, transform=axis.transAxes, ha="right", va="bottom", fontsize=6.6, linespacing=1.12, bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.95, "boxstyle": "square,pad=0.35"})
    axis.set_xlabel(r"$K_{\mathrm{eff}}$ [GeV]", fontsize=9)
    axis.set_ylabel(r"$\mu$ [PE]", fontsize=9)
    axis.set_title(f"Channel linearity — {scenario}", fontsize=10, loc="left")
    axis.tick_params(direction="in", top=True, right=True, labelsize=8)
    axis.grid(linestyle="--", linewidth=0.45, alpha=0.35)


def write_pdfs(output_dir: Path, scenarios: list[tuple[str, float]], channels: dict[int, list[tuple[int, int]]], distribution_objects: dict, linearity_rows: dict) -> None:
    for scenario, multiplier in scenarios:
        for apa in (1, 2):
            pdf_path = output_dir / f"apa{apa}_channel_linearity_threshold_{scenario}.pdf"
            with PdfPages(pdf_path) as pdf:
                for endpoint, channel in channels[apa]:
                    figure, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))
                    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.075, top=0.91, wspace=0.28, hspace=0.33)
                    figure.text(0.065, 0.965, f"APA {apa} — Endpoint {endpoint}, channel {channel}", ha="left", va="top", fontsize=14)
                    figure.text(0.985, 0.965, r"$\bf{ProtoDUNE\!-!HD}$ Work in Progress", ha="right", va="top", fontsize=11)
                    point_rows = {}
                    for panel, momentum in zip(axes.flat[:5], MOMENTA):
                        fit = distribution_objects[(scenario, apa, endpoint, channel, momentum)]
                        point_rows[momentum] = fit
                        selected = fit["triggers_selected"]
                        if momentum == 1:
                            selection_text = f"unselected; {selected} valid triggers"
                        else:
                            selection_text = f"APA 1 mean > {fit['threshold_applied_PE']:.1f} PE; {selected} triggers"
                        draw_distribution(panel, fit, momentum, selection_text)
                    all_fit = linearity_rows[(scenario, apa, endpoint, channel, "including_1_GeV_c")]
                    high_fit = linearity_rows[(scenario, apa, endpoint, channel, "excluding_1_GeV_c")]
                    draw_linearity(axes.flat[5], point_rows, all_fit, high_fit, scenario)
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
    parser.add_argument("--minimum-entries", type=int, default=50)
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

    scenarios = [(scenario_name(multiplier), multiplier) for multiplier in args.threshold_sigma_multipliers]
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
                    fit = fit_gaussian(values, args.minimum_entries)
                    fit.update({
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
                            "kind": fit["quality_flag"], "detail": "Gaussian fit completed; inspect PDF before physics use.",
                        })
                    point_lookup[momentum] = fit
                for range_name, required in (("including_1_GeV_c", MOMENTA), ("excluding_1_GeV_c", (2, 3, 5, 7))):
                    points = [point_lookup[momentum] for momentum in required if point_lookup[momentum]["status"] == "success"]
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
            for fit_range in ("including_1_GeV_c", "excluding_1_GeV_c"):
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
        f"Minimo valori per fit Gaussiano: {args.minimum_entries}",
        f"Incertezza relativa efficace sul momento: {args.relative_momentum_error:g}",
        f"Scenari soglia: {', '.join(name for name, _ in scenarios)}",
        "",
        "SELEZIONE",
        "A 2, 3, 5 e 7 GeV/c: apa1_mean > T, con T = intersezione Langauss--Gaussiana.",
        "A 1 GeV/c: nessuna soglia; ogni APA usa i propri trigger validi.",
        "Per APA 2 a 2--7 GeV/c il tag del trigger e' sempre definito da APA 1.",
        "",
        "FIT",
        "Le distribuzioni sono adattate con una Gaussiana binned e devianza di Poisson.",
        "Il range del fit e' P0.5--P99.5 della distribuzione; i valori esterni sono contati nel CSV.",
        "Una qualita' marcata review non cambia automaticamente il modello: il PDF va ispezionato.",
        "I JSON storici conservano una sola osservazione per endpoint-canale e trigger.",
        "Se esistono waveform duplicate dello stesso canale, occorre rigenerare gli input con una lista per canale.",
        "I fit lineari usano ODR con sigma_Keff = sqrt(sigma_Keff,p^2 + sigma_mix^2).",
        f"Canali trovati: APA1={len(channels[1])}; APA2={len(channels[2])}.",
        f"Fit di distribuzione riusciti: {successful_distribution}/{len(distribution_rows)}.",
        f"Fit lineari riusciti: {successful_linearity}/{len(linearity_output)}.",
        f"Righe da controllare: {len(flagged_rows)}.",
        "",
        "SISTEMATICO DELLA SOGLIA",
        "Per ogni canale il CSV dedicato riporta l'inviluppo max(|risultato +/-1sigma - nominale|).",
        "Il punto a 1 GeV/c non varia con la soglia perche' e' unselected.",
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
            "fit_model": "gaussian_binned_poisson_deviance",
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
