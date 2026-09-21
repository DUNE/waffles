#!/usr/bin/env python3
r"""Crea figure diagnostiche APA 1–APA 2 e adatta le popolazioni di APA 1.

SCOPO
    Per ogni momento nominale disegna la distribuzione di apa1_mean, la
    distribuzione di apa2_mean e lo scatter plot trigger per trigger tra le due
    medie. Per 2, 3, 5 e 7 GeV/c adatta la distribuzione di APA 1 con la somma
    di una Langauss (popolazione muonica) e una Gaussiana (popolazione non
    muonica). La soglia candidata è l'intersezione delle due componenti tra il
    massimo della Langauss e la media della Gaussiana. A 1 GeV/c esegue un fit
    con una sola Langauss nell'intervallo 10--150 PE, senza interpretarlo come
    separazione di popolazioni e senza ricavare una soglia. Non applica le
    soglie agli eventi e non rimuove outlier.

INPUT
    --input-file: apa12_trigger_data.csv prodotto da
    prepare_apa12_trigger_data.py.
    --momenta: momenti da rappresentare (default: 1 2 3 5 7 GeV/c).
    --output-dir: cartella per risultati e figure. Può essere riutilizzata:
    vengono sovrascritti soltanto gli output appartenenti a questo script.
    --skip-fits: crea soltanto le figure diagnostiche, senza eseguire i fit.
    Per l'istogramma di ciascuna APA sono usate tutte le righe con la relativa
    media valida. Per lo scatter sono usate soltanto le righe both_apa_valid=1.

OUTPUT
    apa1_hist_<momento>GeV.png: distribuzione di APA 1;
    apa2_hist_<momento>GeV.png: distribuzione di APA 2;
    apa12_pe_distribution_<momento>GeV.png: scatter trigger per trigger;
    apa1_population_fit_<momento>GeV.png: fit Langauss di APA 1 a 1 GeV/c e
    fit Langauss + Gaussiana a 2, 3, 5 e 7 GeV/c;
    apa1_population_fit_results.csv: parametri, incertezze statistiche locali,
    qualità del fit e intersezione delle componenti;
    plot_summary.csv: conteggi e statistiche descrittive;
    histogram_bins.json: bordi esatti dei bin;
    extreme_events.csv: provenienza e valori degli eventi oltre almeno un
    limite superiore mostrato nelle figure;
    anomalies.csv: incoerenze del dataset di input;
    report.txt: descrizione dei risultati e dei limiti;
    manifest.json: configurazione, versioni e SHA-256 dell'input.
    I bin diagnostici sono determinati con la regola di Freedman–Diaconis. I
    fit usano larghezze fissate per momento e tutti i bin, inclusi quelli vuoti.
    A 1 GeV/c il range è 10--150 PE; per gli altri momenti è l'intervallo
    robusto P0.5--P99.5 riportato negli output.
    Le vecchie figure PNG/PDF prodotte da questo script vengono rimosse prima
    di scrivere i nuovi risultati; eventuali altri file non vengono toccati.
    Codice di uscita: 0 = figure create; 2 = errore, figure non create.

ESECUZIONE (dalla cartella scripts/review su LXPlus)
    python plot_apa12_trigger_distributions.py \
        --input-file ../../output/review/apa12_trigger_data_01/apa12_trigger_data.csv \
        --output-dir ../../output/review/apa12_trigger_distributions_01

INTERPRETAZIONE
    apa1_mean e apa2_mean sono medie PE per canale contribuente, non somme
    sull'intera APA. Il coefficiente di Pearson è soltanto descrittivo e non è
    usato per selezionare eventi. Le incertezze del fit sono statistiche locali
    e condizionate al modello, al binning e all'intervallo scelti. APA 2 non
    viene adattata con il modello misto perché il self-trigger e il numero
    variabile di canali contribuenti ne modificano la distribuzione.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_FIELDS = {
    "momentum_GeV_c",
    "block",
    "source_file",
    "source_csv_row",
    "trigger_time",
    "apa1_mean",
    "apa1_std",
    "apa1_n_events",
    "apa1_valid",
    "apa2_mean",
    "apa2_std",
    "apa2_n_events",
    "apa2_valid",
    "both_apa_valid",
    "validity_category",
}

SUMMARY_FIELDS = [
    "momentum_GeV_c",
    "input_rows",
    "apa1_entries",
    "apa2_entries",
    "scatter_entries",
    "apa1_minimum",
    "apa1_p01",
    "apa1_p05",
    "apa1_median",
    "apa1_mean",
    "apa1_population_std",
    "apa1_p95",
    "apa1_p99",
    "apa1_maximum",
    "apa1_histogram_bins",
    "apa1_display_maximum",
    "apa1_entries_above_display_maximum",
    "apa2_minimum",
    "apa2_p01",
    "apa2_p05",
    "apa2_median",
    "apa2_mean",
    "apa2_population_std",
    "apa2_p95",
    "apa2_p99",
    "apa2_maximum",
    "apa2_histogram_bins",
    "apa2_display_maximum",
    "apa2_entries_above_display_maximum",
    "pearson_r",
]

ANOMALY_FIELDS = [
    "severity",
    "code",
    "csv_row",
    "momentum_GeV_c",
    "trigger_time",
    "detail",
]

EXTREME_FIELDS = [
    "momentum_GeV_c",
    "trigger_time",
    "block",
    "source_file",
    "source_csv_row",
    "validity_category",
    "apa1_mean",
    "apa1_std",
    "apa1_n_events",
    "apa2_mean",
    "apa2_std",
    "apa2_n_events",
    "apa1_display_maximum",
    "apa2_display_maximum",
    "outside_apa1_display",
    "outside_apa2_display",
]

VALID_CATEGORIES = {"both_valid", "apa1_only", "apa2_only", "neither_valid"}

# Palette Okabe--Ito: leggibile, sobria e distinguibile anche con deficit
# comuni della visione dei colori.
COLORS = {
    "apa1": "#D55E00",
    "apa2": "#0072B2",
    "scatter": "#56B4E9",
    "langauss": "#D55E00",
    "gaussian": "#0072B2",
    "intersection": "#CC79A7",
    "total": "#222222",
    "data": "#D9D9D9",
}

# Le larghezze e gli anchor riprendono la scala osservata nel programma storico.
# L'anchor identifica le componenti durante il fit (MPV < anchor < media
# gaussiana), ma non è la soglia finale: questa viene ricavata dall'intersezione.
FIT_CONFIGURATION = {
    2: {"bin_width": 2.0, "component_anchor": 110.0},
    3: {"bin_width": 4.0, "component_anchor": 150.0},
    5: {"bin_width": 5.0, "component_anchor": 190.0},
    7: {"bin_width": 8.0, "component_anchor": 200.0},
}

FIT_FIELDS = [
    "momentum_GeV_c", "model", "status", "message", "entries_total",
    "entries_in_fit_range", "entries_below_fit_range", "entries_above_fit_range",
    "fit_minimum", "fit_maximum", "bin_width", "histogram_bins",
    "component_anchor", "mpv", "mpv_error", "eta", "eta_error",
    "langauss_sigma", "langauss_sigma_error", "langauss_yield",
    "langauss_yield_error", "gaussian_mean", "gaussian_mean_error",
    "gaussian_sigma", "gaussian_sigma_error", "gaussian_yield",
    "gaussian_yield_error", "langauss_peak", "langauss_peak_error",
    "intersection", "intersection_error", "poisson_deviance", "ndf",
    "deviance_per_ndf", "deviance_p_value", "pearson_chi2",
    "pearson_chi2_per_ndf", "r_squared", "langauss_integration_points",
    "optimizer_success", "optimizer_status",
    "optimizer_attempts", "function_evaluations", "jacobian_rank",
    "covariance_valid", "parameters_near_bounds",
]


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clear_owned_outputs(output_dir):
    """Rimuove soltanto file prodotti da questo script in esecuzioni precedenti."""
    static_names = (
        "plot_summary.csv",
        "apa1_population_fit_results.csv",
        "histogram_bins.json",
        "extreme_events.csv",
        "anomalies.csv",
        "report.txt",
        "manifest.json",
    )
    for name in static_names:
        path = output_dir / name
        if path.is_file():
            path.unlink()
    for momentum in (1, 2, 3, 5, 7):
        for stem in (
            f"apa1_hist_{momentum}GeV",
            f"apa2_hist_{momentum}GeV",
            f"apa12_pe_distribution_{momentum}GeV",
            f"apa1_population_fit_{momentum}GeV",
            f"diagnostic_{momentum}GeV",
        ):
            for suffix in (".png", ".pdf"):
                path = output_dir / f"{stem}{suffix}"
                if path.is_file():
                    path.unlink()


def parse_flag(raw):
    if raw not in ("0", "1"):
        raise ValueError("atteso 0 oppure 1")
    return int(raw)


def parse_optional_finite(raw, is_valid):
    if is_valid:
        if not raw.strip():
            raise ValueError("valore dichiarato valido ma campo vuoto")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("valore dichiarato valido ma non finito")
        return value
    if raw.strip():
        raise ValueError("valore dichiarato non valido ma campo non vuoto")
    return None


def parse_nonnegative_integer(raw):
    value = int(raw)
    if value < 0 or str(value) != raw.strip():
        raise ValueError("atteso un intero non negativo in forma canonica")
    return value


def load_rows(path, requested_momenta):
    rows = []
    anomalies = []
    seen = set()

    def issue(code, csv_row, momentum="", trigger="", detail=""):
        anomalies.append({
            "severity": "error",
            "code": code,
            "csv_row": csv_row,
            "momentum_GeV_c": momentum,
            "trigger_time": trigger,
            "detail": detail,
        })

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        headers = reader.fieldnames or []
        missing = sorted(REQUIRED_FIELDS - set(headers))
        if missing or len(headers) != len(set(headers)):
            issue(
                "invalid_columns",
                1,
                detail=(
                    f"Colonne mancanti: {missing}; "
                    f"intestazioni duplicate: {len(headers) != len(set(headers))}."
                ),
            )
            return rows, anomalies

        try:
            for record in reader:
                csv_row = reader.line_num
                if None in record or any(value is None for value in record.values()):
                    issue("malformed_row", csv_row,
                          detail="Numero di campi diverso dall'intestazione.")
                    continue
                try:
                    momentum = int(record["momentum_GeV_c"])
                    trigger = int(record["trigger_time"])
                    apa1_valid = parse_flag(record["apa1_valid"])
                    apa2_valid = parse_flag(record["apa2_valid"])
                    both_valid = parse_flag(record["both_apa_valid"])
                    apa1_mean = parse_optional_finite(record["apa1_mean"], apa1_valid)
                    apa2_mean = parse_optional_finite(record["apa2_mean"], apa2_valid)
                    apa1_std = parse_optional_finite(record["apa1_std"], apa1_valid)
                    apa2_std = parse_optional_finite(record["apa2_std"], apa2_valid)
                    apa1_n_events = parse_nonnegative_integer(record["apa1_n_events"])
                    apa2_n_events = parse_nonnegative_integer(record["apa2_n_events"])
                    source_csv_row = parse_nonnegative_integer(record["source_csv_row"])
                except (ValueError, OverflowError) as exc:
                    issue("invalid_value", csv_row, record.get("momentum_GeV_c", ""),
                          record.get("trigger_time", ""), str(exc))
                    continue

                if momentum not in (1, 2, 3, 5, 7) or trigger < 0:
                    issue("invalid_identity", csv_row, momentum, trigger,
                          "Momento non previsto oppure trigger_time negativo.")
                    continue
                if (
                    bool(apa1_n_events) != bool(apa1_valid)
                    or bool(apa2_n_events) != bool(apa2_valid)
                ):
                    issue(
                        "inconsistent_valid_count",
                        csv_row,
                        momentum,
                        trigger,
                        "n_events deve essere positivo esattamente quando la media è valida.",
                    )
                    continue
                expected_both = int(apa1_valid and apa2_valid)
                if both_valid != expected_both:
                    issue("inconsistent_both_valid", csv_row, momentum, trigger,
                          f"Salvato={both_valid}, atteso={expected_both}.")
                    continue
                if apa1_valid and apa2_valid:
                    expected_category = "both_valid"
                elif apa1_valid:
                    expected_category = "apa1_only"
                elif apa2_valid:
                    expected_category = "apa2_only"
                else:
                    expected_category = "neither_valid"
                category = record["validity_category"]
                if category not in VALID_CATEGORIES or category != expected_category:
                    issue("inconsistent_validity_category", csv_row, momentum, trigger,
                          f"Salvata={category!r}, attesa={expected_category!r}.")
                    continue
                identity = (momentum, trigger)
                if identity in seen:
                    issue("repeated_trigger", csv_row, momentum, trigger,
                          "Identità (momento, trigger_time) ripetuta.")
                    continue
                seen.add(identity)
                if momentum in requested_momenta:
                    rows.append({
                        "momentum": momentum,
                        "trigger": trigger,
                        "block": record["block"],
                        "source_file": record["source_file"],
                        "source_csv_row": source_csv_row,
                        "validity_category": category,
                        "apa1_mean": apa1_mean,
                        "apa1_std": apa1_std,
                        "apa1_n_events": apa1_n_events,
                        "apa2_mean": apa2_mean,
                        "apa2_std": apa2_std,
                        "apa2_n_events": apa2_n_events,
                        "apa1_valid": apa1_valid,
                        "apa2_valid": apa2_valid,
                        "both_valid": both_valid,
                    })
        except csv.Error as exc:
            issue("csv_parse_error", reader.line_num, detail=str(exc))
    return rows, anomalies


def descriptive(values, np):
    percentiles = np.percentile(values, [1, 5, 50, 95, 99])
    return {
        "minimum": float(np.min(values)),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "median": float(percentiles[2]),
        "mean": float(np.mean(values)),
        "population_std": float(np.std(values, ddof=0)),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
        "maximum": float(np.max(values)),
    }


def histogram_edges(values, np):
    """Freedman–Diaconis con fallback esplicito per campioni degeneri."""
    edges = np.histogram_bin_edges(values, bins="fd")
    if len(edges) >= 2 and edges[-1] > edges[0]:
        return edges
    center = float(values[0])
    half_width = max(abs(center) * 0.01, 0.5)
    return np.array([center - half_width, center + half_width], dtype=float)


def display_maximum(values, np):
    """Limite grafico robusto; non modifica né seleziona i dati analizzati."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        far_out_fence = q3 + 5.0 * iqr
        candidate = max(float(np.percentile(values, 99)), float(far_out_fence))
    else:
        candidate = float(np.max(values))
    maximum = float(np.max(values))
    if maximum <= candidate:
        return maximum * 1.03 if maximum > 0 else 1.0, 0
    limit = candidate * 1.03
    return limit, int(np.count_nonzero(values > limit))


def collect_extreme_events(rows, display_max1, display_max2):
    """Seleziona per il solo riepilogo gli eventi oltre i limiti grafici."""
    output = []
    for row in rows:
        outside1 = bool(
            row["apa1_valid"] and row["apa1_mean"] > display_max1
        )
        outside2 = bool(
            row["apa2_valid"] and row["apa2_mean"] > display_max2
        )
        if not (outside1 or outside2):
            continue
        output.append({
            "momentum_GeV_c": row["momentum"],
            "trigger_time": row["trigger"],
            "block": row["block"],
            "source_file": row["source_file"],
            "source_csv_row": row["source_csv_row"],
            "validity_category": row["validity_category"],
            "apa1_mean": "" if row["apa1_mean"] is None else row["apa1_mean"],
            "apa1_std": "" if row["apa1_std"] is None else row["apa1_std"],
            "apa1_n_events": row["apa1_n_events"],
            "apa2_mean": "" if row["apa2_mean"] is None else row["apa2_mean"],
            "apa2_std": "" if row["apa2_std"] is None else row["apa2_std"],
            "apa2_n_events": row["apa2_n_events"],
            "apa1_display_maximum": display_max1,
            "apa2_display_maximum": display_max2,
            "outside_apa1_display": int(outside1),
            "outside_apa2_display": int(outside2),
        })
    return output


def poisson_deviance_residuals(observed, expected, np):
    """Residui con segno della devianza di Poisson, definiti anche per N=0."""
    expected = np.clip(expected, 1.0e-12, None)
    term = np.empty_like(expected)
    positive = observed > 0
    term[positive] = (
        expected[positive]
        - observed[positive]
        + observed[positive] * np.log(observed[positive] / expected[positive])
    )
    term[~positive] = expected[~positive]
    return np.sign(observed - expected) * np.sqrt(np.maximum(2.0 * term, 0.0))


def langauss_integration_points(eta, sigma):
    """Passi sufficienti a risolvere la scala Landau nella convoluzione."""
    if eta <= 0 or sigma <= 0:
        return 0
    return min(50000, max(400, int(math.ceil(120.0 * sigma / eta))))


def stable_langauss_pdf(x, mpv, eta, sigma, np, landau):
    """Convoluzione Landau--Gauss a midpoint, valutata a blocchi.

    landaupy usa circa 3 punti per eta quando sigma >> eta; questa versione ne
    usa almeno 12 ed evita le oscillazioni numeriche osservate a 3 GeV/c.
    """
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
    chunk_size = 64
    for start in range(0, len(flat_x), chunk_size):
        stop = min(start + chunk_size, len(flat_x))
        convolution_axis = flat_x[None, start:stop] + offsets[:, None]
        landau_values = np.asarray(
            landau.pdf(
                convolution_axis.reshape(-1),
                float(mpv),
                float(eta),
            ),
            dtype=float,
        ).reshape(convolution_axis.shape)
        result[start:stop] = step * np.sum(
            landau_values * gaussian_weights[:, None], axis=0
        )
    return result.reshape(original_shape)


def fit_apa1_langauss(
    values, np, landau, least_squares, minimize_scalar, chi2_distribution,
):
    """Fit binned Poisson con una sola Langauss per il campione a 1 GeV/c."""
    momentum = 1
    width = 2.0
    fit_minimum = 10.0
    fit_maximum = 150.0
    edges = np.arange(fit_minimum, fit_maximum + width, width, dtype=float)
    observed, edges = np.histogram(values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    in_range = (values >= edges[0]) & (values <= edges[-1])
    sample = values[in_range]
    if len(sample) < 20:
        raise RuntimeError(
            f"Campione insufficiente nel range 10--150 PE: {len(sample)} eventi."
        )

    q25, median, q75 = np.percentile(sample, [25, 50, 75])
    robust_sigma = max(float((q75 - q25) / 1.349), width)
    initial = np.asarray([
        float(median), 5.0, max(2.0, robust_sigma / 2.0), float(len(sample))
    ])
    lower_bounds = np.asarray([30.0, 0.1, 0.2, 0.01])
    upper_bounds = np.asarray([100.0, 30.0, 35.0, float(len(values) * 10)])
    initial = np.minimum(
        np.maximum(initial, lower_bounds + 1.0e-6), upper_bounds - 1.0e-6
    )

    def langauss_density(parameters, x):
        mpv, eta, sigma_lg, yield_lg = parameters
        return yield_lg * stable_langauss_pdf(
            x, mpv, eta, sigma_lg, np, landau
        )

    def expected_counts(parameters):
        return width * langauss_density(parameters, centers)

    def objective(parameters):
        return poisson_deviance_residuals(observed, expected_counts(parameters), np)

    starts = [initial]
    for eta_start in (2.0, 5.0, 10.0):
        for sigma_scale in (0.7, 1.0, 1.4):
            candidate = initial.copy()
            candidate[1] = eta_start
            candidate[2] = initial[2] * sigma_scale
            starts.append(np.minimum(
                np.maximum(candidate, lower_bounds + 1.0e-6),
                upper_bounds - 1.0e-6,
            ))
    attempts = []
    for start in starts:
        try:
            candidate_result = least_squares(
                objective,
                start,
                bounds=(lower_bounds, upper_bounds),
                method="trf",
                x_scale="jac",
                max_nfev=30000,
            )
            if np.isfinite(candidate_result.cost):
                attempts.append(candidate_result)
        except (ValueError, FloatingPointError):
            continue
    if not attempts:
        raise RuntimeError(
            "Nessuna inizializzazione del fit Langauss ha prodotto un risultato finito."
        )

    result = min(attempts, key=lambda candidate_result: candidate_result.cost)
    parameters = result.x
    expected = expected_counts(parameters)
    residuals = poisson_deviance_residuals(observed, expected, np)
    deviance = float(np.sum(residuals ** 2))
    ndf = int(len(observed) - len(parameters))
    jacobian_rank = int(np.linalg.matrix_rank(result.jac))
    covariance_valid = bool(
        result.success and ndf > 0 and jacobian_rank == len(parameters)
    )
    parameter_errors = np.full(len(parameters), np.nan)
    if covariance_valid:
        try:
            covariance = np.linalg.inv(result.jac.T @ result.jac)
            diagonal = np.diag(covariance)
            covariance_valid = bool(
                np.all(np.isfinite(covariance)) and np.all(diagonal >= 0)
            )
            if covariance_valid:
                parameter_errors = np.sqrt(diagonal)
        except np.linalg.LinAlgError:
            covariance_valid = False

    pearson_chi2 = float(np.sum(
        (observed - expected) ** 2 / np.clip(expected, 1.0e-12, None)
    ))
    sum_squared_residuals = float(np.sum((observed - expected) ** 2))
    total_sum_squares = float(np.sum((observed - np.mean(observed)) ** 2))
    r_squared = (
        1.0 - sum_squared_residuals / total_sum_squares
        if total_sum_squares > 0 else math.nan
    )
    def peak_coordinate(parameter_values):
        peak_result = minimize_scalar(
            lambda coordinate: -float(
                langauss_density(parameter_values, np.asarray([coordinate]))[0]
            ),
            bounds=(edges[0], edges[-1]),
            method="bounded",
            options={"xatol": 1.0e-5},
        )
        if not peak_result.success:
            raise RuntimeError("Ricerca numerica del picco Langauss non riuscita.")
        return float(peak_result.x)

    langauss_peak = peak_coordinate(parameters)
    peak_error = math.nan
    if covariance_valid:
        gradient = np.zeros(len(parameters))
        for index, value in enumerate(parameters):
            step = max(abs(float(value)) * 1.0e-4, 1.0e-4)
            plus = parameters.copy()
            minus = parameters.copy()
            plus[index] = min(value + step, upper_bounds[index] - 1.0e-8)
            minus[index] = max(value - step, lower_bounds[index] + 1.0e-8)
            denominator = plus[index] - minus[index]
            if denominator <= 0:
                gradient[index] = math.nan
                continue
            gradient[index] = (
                peak_coordinate(plus) - peak_coordinate(minus)
            ) / denominator
        if np.all(np.isfinite(gradient)):
            variance = float(gradient @ covariance @ gradient)
            if variance >= 0:
                peak_error = math.sqrt(variance)

    names = ("mpv", "eta", "langauss_sigma", "langauss_yield")
    bound_tolerance = 1.0e-3 * (upper_bounds - lower_bounds)
    near_bounds = [
        name
        for name, value, lower, upper, tolerance in zip(
            names, parameters, lower_bounds, upper_bounds, bound_tolerance
        )
        if value - lower <= tolerance or upper - value <= tolerance
    ]
    row = {
        "momentum_GeV_c": momentum,
        "model": "langauss",
        "status": "success" if result.success else "optimizer_failed",
        "message": result.message,
        "entries_total": len(values),
        "entries_in_fit_range": int(np.count_nonzero(in_range)),
        "entries_below_fit_range": int(np.count_nonzero(values < edges[0])),
        "entries_above_fit_range": int(np.count_nonzero(values > edges[-1])),
        "fit_minimum": float(edges[0]),
        "fit_maximum": float(edges[-1]),
        "bin_width": width,
        "histogram_bins": len(observed),
        "component_anchor": "",
        "poisson_deviance": deviance,
        "ndf": ndf,
        "deviance_per_ndf": deviance / ndf if ndf > 0 else math.nan,
        "deviance_p_value": (
            float(chi2_distribution.sf(deviance, ndf)) if ndf > 0 else math.nan
        ),
        "pearson_chi2": pearson_chi2,
        "pearson_chi2_per_ndf": pearson_chi2 / ndf if ndf > 0 else math.nan,
        "r_squared": r_squared,
        "langauss_integration_points": langauss_integration_points(
            parameters[1], parameters[2]
        ),
        "optimizer_success": int(result.success),
        "optimizer_status": result.status,
        "optimizer_attempts": len(attempts),
        "function_evaluations": result.nfev,
        "jacobian_rank": jacobian_rank,
        "covariance_valid": int(covariance_valid),
        "parameters_near_bounds": ";".join(near_bounds),
    }
    for name, value, error in zip(names, parameters, parameter_errors):
        row[name] = float(value)
        row[f"{name}_error"] = float(error)
    row["langauss_peak"] = langauss_peak
    row["langauss_peak_error"] = peak_error

    return {
        "row": row,
        "edges": edges,
        "centers": centers,
        "observed": observed,
        "expected": expected,
        "residuals": residuals,
        "parameters": parameters,
        "langauss_density": langauss_density,
    }


def fit_apa1_population(
    values, momentum, np, landau, least_squares, minimize_scalar, brentq,
    chi2_distribution,
):
    """Fit binned Poisson Langauss + Gaussiana per un singolo momento."""
    configuration = FIT_CONFIGURATION[momentum]
    width = configuration["bin_width"]
    anchor = configuration["component_anchor"]

    lower_quantile, upper_quantile = np.percentile(values, [0.5, 99.5])
    fit_minimum = max(0.0, math.floor(lower_quantile / width) * width)
    fit_maximum = math.ceil(upper_quantile / width) * width
    if fit_maximum <= fit_minimum or not fit_minimum < anchor < fit_maximum:
        raise RuntimeError(
            f"Intervallo [{fit_minimum}, {fit_maximum}] incompatibile con anchor={anchor}."
        )

    edges = np.arange(fit_minimum, fit_maximum + 0.5 * width, width, dtype=float)
    if edges[-1] < fit_maximum:
        edges = np.append(edges, fit_maximum)
    observed, edges = np.histogram(values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    in_range = (values >= edges[0]) & (values <= edges[-1])
    low_sample = values[in_range & (values < anchor)]
    high_sample = values[in_range & (values >= anchor)]
    if len(low_sample) < 20 or len(high_sample) < 20:
        raise RuntimeError(
            f"Campioni iniziali insufficienti: basso={len(low_sample)}, alto={len(high_sample)}."
        )

    low_counts, low_edges = np.histogram(low_sample, bins=edges[edges <= anchor])
    if not len(low_counts):
        raise RuntimeError("Nessun bin disponibile per inizializzare la Langauss.")
    low_centers = 0.5 * (low_edges[:-1] + low_edges[1:])
    mpv_initial = float(low_centers[int(np.argmax(low_counts))])
    high_q25, high_q50, high_q75 = np.percentile(high_sample, [25, 50, 75])
    low_q25, _, low_q75 = np.percentile(low_sample, [25, 50, 75])
    robust_low_sigma = max(float((low_q75 - low_q25) / 1.349), width)
    robust_high_sigma = max(float((high_q75 - high_q25) / 1.349), width)

    initial = np.asarray([
        min(max(mpv_initial, fit_minimum + width), anchor - width),
        3.0,
        max(width, robust_low_sigma / 2.0),
        float(len(low_sample)),
        min(max(float(high_q50), anchor + width), fit_maximum - width),
        robust_high_sigma,
        float(len(high_sample)),
    ])
    lower_bounds = np.asarray([
        fit_minimum,
        0.1,
        0.2,
        0.01,
        anchor,
        max(0.5, width / 4.0),
        0.01,
    ])
    upper_bounds = np.asarray([
        anchor,
        max(30.0, width * 5.0),
        max(anchor - fit_minimum, width * 4.0),
        float(len(values) * 10),
        fit_maximum,
        max(fit_maximum - fit_minimum, width * 4.0),
        float(len(values) * 10),
    ])
    initial = np.minimum(np.maximum(initial, lower_bounds + 1.0e-6), upper_bounds - 1.0e-6)

    def components(parameters, x):
        mpv, eta, sigma_lg, yield_lg, mean_g, sigma_g, yield_g = parameters
        langauss_density = stable_langauss_pdf(
            x, mpv, eta, sigma_lg, np, landau
        )
        langauss_density = np.nan_to_num(
            langauss_density, nan=0.0, posinf=0.0, neginf=0.0
        )
        gaussian_density = np.exp(-0.5 * ((x - mean_g) / sigma_g) ** 2)
        gaussian_density /= sigma_g * math.sqrt(2.0 * math.pi)
        return yield_lg * langauss_density, yield_g * gaussian_density

    def expected_counts(parameters):
        langauss_density, gaussian_density = components(parameters, centers)
        return width * (langauss_density + gaussian_density)

    def objective(parameters):
        return poisson_deviance_residuals(observed, expected_counts(parameters), np)

    alternative_starts = [initial]
    for eta_start in (1.0, 6.0):
        candidate = initial.copy()
        candidate[1] = eta_start
        alternative_starts.append(candidate)
    candidate = initial.copy()
    candidate[4] = float(np.mean(high_sample))
    alternative_starts.append(candidate)
    for scale in (0.65, 1.50):
        candidate = initial.copy()
        candidate[5] *= scale
        alternative_starts.append(candidate)
    alternative_starts = [
        np.minimum(
            np.maximum(candidate, lower_bounds + 1.0e-6),
            upper_bounds - 1.0e-6,
        )
        for candidate in alternative_starts
    ]
    attempts = []
    for start in alternative_starts:
        try:
            candidate_result = least_squares(
                objective,
                start,
                bounds=(lower_bounds, upper_bounds),
                method="trf",
                x_scale="jac",
                max_nfev=30000,
            )
            if np.isfinite(candidate_result.cost):
                attempts.append(candidate_result)
        except (ValueError, FloatingPointError):
            continue
    if not attempts:
        raise RuntimeError(
            "Nessuna inizializzazione del fit ha prodotto un risultato finito."
        )
    result = min(attempts, key=lambda candidate_result: candidate_result.cost)
    parameters = result.x
    expected = expected_counts(parameters)
    residuals = poisson_deviance_residuals(observed, expected, np)
    deviance = float(np.sum(residuals ** 2))
    ndf = int(len(observed) - len(parameters))
    jacobian_rank = int(np.linalg.matrix_rank(result.jac))
    covariance_valid = bool(
        result.success and ndf > 0 and jacobian_rank == len(parameters)
    )
    covariance = None
    parameter_errors = np.full(len(parameters), np.nan)
    if covariance_valid:
        try:
            covariance = np.linalg.inv(result.jac.T @ result.jac)
            diagonal = np.diag(covariance)
            covariance_valid = bool(np.all(np.isfinite(covariance)) and np.all(diagonal >= 0))
            if covariance_valid:
                parameter_errors = np.sqrt(diagonal)
            else:
                covariance = None
        except np.linalg.LinAlgError:
            covariance_valid = False

    pearson_chi2 = float(np.sum((observed - expected) ** 2 / np.clip(expected, 1.0e-12, None)))
    sum_squared_residuals = float(np.sum((observed - expected) ** 2))
    total_sum_squares = float(np.sum((observed - np.mean(observed)) ** 2))
    r_squared = (
        1.0 - sum_squared_residuals / total_sum_squares
        if total_sum_squares > 0 else math.nan
    )

    def derived_quantities(parameter_values):
        peak_result = minimize_scalar(
            lambda coordinate: -float(
                components(parameter_values, np.asarray([coordinate]))[0][0]
            ),
            bounds=(edges[0], min(anchor, edges[-1])),
            method="bounded",
            options={"xatol": 1.0e-5},
        )
        if not peak_result.success:
            raise RuntimeError("Ricerca numerica del picco Langauss non riuscita.")
        langauss_peak = float(peak_result.x)
        gaussian_mean = float(parameter_values[4])
        candidate_x = np.linspace(langauss_peak, gaussian_mean, 401)
        langauss_density, gaussian_density = components(parameter_values, candidate_x)
        difference = langauss_density - gaussian_density
        if len(candidate_x) < 2 or not np.all(np.isfinite(difference)):
            raise RuntimeError("Intervallo vuoto tra picco Langauss e media Gaussiana.")
        crossings = np.flatnonzero(difference[:-1] * difference[1:] <= 0)
        if not len(crossings):
            raise RuntimeError("Le componenti non si intersecano tra i rispettivi picchi.")
        index = int(crossings[0])
        x0, x1 = candidate_x[index], candidate_x[index + 1]
        if difference[index] == 0:
            intersection = float(x0)
        else:
            def component_difference(coordinate):
                values_lg, values_g = components(
                    parameter_values, np.asarray([coordinate])
                )
                return float(values_lg[0] - values_g[0])

            intersection = float(brentq(
                component_difference,
                x0,
                x1,
                xtol=1.0e-7,
            ))
        return langauss_peak, intersection

    langauss_peak, intersection = derived_quantities(parameters)
    derived_errors = np.full(2, np.nan)
    if covariance_valid:
        gradient = np.zeros((2, len(parameters)))
        for index, value in enumerate(parameters):
            step = max(abs(float(value)) * 1.0e-4, 1.0e-4)
            plus = parameters.copy()
            minus = parameters.copy()
            plus[index] = min(value + step, upper_bounds[index] - 1.0e-8)
            minus[index] = max(value - step, lower_bounds[index] + 1.0e-8)
            denominator = plus[index] - minus[index]
            if denominator <= 0:
                continue
            try:
                gradient[:, index] = (
                    np.asarray(derived_quantities(plus))
                    - np.asarray(derived_quantities(minus))
                ) / denominator
            except RuntimeError:
                gradient[:, index] = np.nan
        if np.all(np.isfinite(gradient)):
            derived_variance = np.diag(gradient @ covariance @ gradient.T)
            if np.all(derived_variance >= 0):
                derived_errors = np.sqrt(derived_variance)

    row = {
        "momentum_GeV_c": momentum,
        "model": "langauss_plus_gaussian",
        "status": "success" if result.success else "optimizer_failed",
        "message": result.message,
        "entries_total": len(values),
        "entries_in_fit_range": int(np.count_nonzero(in_range)),
        "entries_below_fit_range": int(np.count_nonzero(values < edges[0])),
        "entries_above_fit_range": int(np.count_nonzero(values > edges[-1])),
        "fit_minimum": float(edges[0]),
        "fit_maximum": float(edges[-1]),
        "bin_width": width,
        "histogram_bins": len(observed),
        "component_anchor": anchor,
        "poisson_deviance": deviance,
        "ndf": ndf,
        "deviance_per_ndf": deviance / ndf if ndf > 0 else math.nan,
        "deviance_p_value": float(chi2_distribution.sf(deviance, ndf)) if ndf > 0 else math.nan,
        "pearson_chi2": pearson_chi2,
        "pearson_chi2_per_ndf": pearson_chi2 / ndf if ndf > 0 else math.nan,
        "r_squared": r_squared,
        "langauss_integration_points": langauss_integration_points(
            parameters[1], parameters[2]
        ),
        "optimizer_success": int(result.success),
        "optimizer_status": result.status,
        "optimizer_attempts": len(attempts),
        "function_evaluations": result.nfev,
        "jacobian_rank": jacobian_rank,
        "covariance_valid": int(covariance_valid),
    }
    names = (
        "mpv", "eta", "langauss_sigma", "langauss_yield",
        "gaussian_mean", "gaussian_sigma", "gaussian_yield",
    )
    for name, value, error in zip(names, parameters, parameter_errors):
        row[name] = float(value)
        row[f"{name}_error"] = float(error)
    bound_tolerance = 1.0e-3 * (upper_bounds - lower_bounds)
    near_bounds = [
        name
        for name, value, lower, upper, tolerance in zip(
            names, parameters, lower_bounds, upper_bounds, bound_tolerance
        )
        if value - lower <= tolerance or upper - value <= tolerance
    ]
    row["parameters_near_bounds"] = ";".join(near_bounds)
    row["langauss_peak"] = langauss_peak
    row["langauss_peak_error"] = float(derived_errors[0])
    row["intersection"] = intersection
    row["intersection_error"] = float(derived_errors[1])

    return {
        "row": row,
        "edges": edges,
        "centers": centers,
        "observed": observed,
        "expected": expected,
        "residuals": residuals,
        "parameters": parameters,
        "components": components,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--momenta", nargs="+", type=int, choices=(1, 2, 3, 5, 7),
        default=[1, 2, 3, 5, 7]
    )
    parser.add_argument(
        "--skip-fits",
        action="store_true",
        help="Crea le sole figure diagnostiche senza eseguire i fit di APA 1.",
    )
    args = parser.parse_args()
    args.input_file = args.input_file.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.momenta = sorted(set(args.momenta))

    if not args.input_file.is_file():
        parser.error(f"File input inesistente: {args.input_file}")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        parser.error(f"Il percorso output esiste ma non è una cartella: {args.output_dir}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        from matplotlib.offsetbox import AnchoredText
        import numpy as np
    except ImportError as exc:
        parser.error(f"Dipendenza mancante nell'ambiente Python: {exc}")

    fit_momenta = [
        momentum for momentum in args.momenta
        if momentum in FIT_CONFIGURATION and not args.skip_fits
    ]
    fit_one_gev = 1 in args.momenta and not args.skip_fits
    landau = least_squares = minimize_scalar = brentq = chi2_distribution = None
    if fit_momenta or fit_one_gev:
        try:
            from scipy.optimize import least_squares
            from scipy.stats import chi2 as chi2_distribution
        except ImportError as exc:
            parser.error(
                "Dipendenza necessaria per i fit mancante nell'ambiente Python: "
                f"{exc}. Usare --skip-fits soltanto per rigenerare le diagnostiche."
            )
    if fit_momenta or fit_one_gev:
        try:
            from landaupy import landau
            from scipy.optimize import brentq, minimize_scalar
        except ImportError as exc:
            parser.error(
                "Dipendenza necessaria per i fit Langauss mancante nell'ambiente "
                f"Python: {exc}."
            )

    rows, anomalies = load_rows(args.input_file, set(args.momenta))
    summaries = []
    bins_manifest = {}
    extreme_events = []
    errors = sum(item["severity"] == "error" for item in anomalies)
    prepared = {}
    fit_results = {}
    fit_rows = []

    if not errors:
        for momentum in args.momenta:
            selected = [row for row in rows if row["momentum"] == momentum]
            apa1 = np.asarray(
                [row["apa1_mean"] for row in selected if row["apa1_valid"]],
                dtype=float,
            )
            apa2 = np.asarray(
                [row["apa2_mean"] for row in selected if row["apa2_valid"]],
                dtype=float,
            )
            paired = [row for row in selected if row["both_valid"]]
            paired_apa1 = np.asarray([row["apa1_mean"] for row in paired], dtype=float)
            paired_apa2 = np.asarray([row["apa2_mean"] for row in paired], dtype=float)

            if not selected or not len(apa1) or not len(apa2) or not len(paired):
                anomalies.append({
                    "severity": "error",
                    "code": "empty_requested_sample",
                    "csv_row": "",
                    "momentum_GeV_c": momentum,
                    "trigger_time": "",
                    "detail": (
                        f"Righe={len(selected)}, APA1={len(apa1)}, "
                        f"APA2={len(apa2)}, coppie={len(paired)}."
                    ),
                })
                continue

            edges1 = histogram_edges(apa1, np)
            edges2 = histogram_edges(apa2, np)
            display_max1, above_display1 = display_maximum(apa1, np)
            display_max2, above_display2 = display_maximum(apa2, np)
            stats1 = descriptive(apa1, np)
            stats2 = descriptive(apa2, np)
            extreme_events.extend(
                collect_extreme_events(selected, display_max1, display_max2)
            )
            if len(paired) > 1 and np.std(paired_apa1) > 0 and np.std(paired_apa2) > 0:
                pearson = float(np.corrcoef(paired_apa1, paired_apa2)[0, 1])
            else:
                pearson = math.nan

            summary = {
                "momentum_GeV_c": momentum,
                "input_rows": len(selected),
                "apa1_entries": len(apa1),
                "apa2_entries": len(apa2),
                "scatter_entries": len(paired),
                "apa1_histogram_bins": len(edges1) - 1,
                "apa2_histogram_bins": len(edges2) - 1,
                "apa1_display_maximum": display_max1,
                "apa1_entries_above_display_maximum": above_display1,
                "apa2_display_maximum": display_max2,
                "apa2_entries_above_display_maximum": above_display2,
                "pearson_r": pearson,
            }
            for key, value in stats1.items():
                summary[f"apa1_{key}"] = value
            for key, value in stats2.items():
                summary[f"apa2_{key}"] = value
            summaries.append(summary)
            bins_manifest[str(momentum)] = {
                "apa1": [float(value) for value in edges1],
                "apa2": [float(value) for value in edges2],
            }
            prepared[momentum] = (
                apa1, apa2, paired_apa1, paired_apa2, edges1, edges2, pearson,
                display_max1, display_max2, above_display1, above_display2,
            )

        requested_fits = ([1] if fit_one_gev else []) + fit_momenta
        for momentum in requested_fits:
            if momentum not in prepared:
                continue
            try:
                if momentum == 1:
                    result = fit_apa1_langauss(
                        prepared[momentum][0], np, landau, least_squares,
                        minimize_scalar, chi2_distribution,
                    )
                else:
                    result = fit_apa1_population(
                        prepared[momentum][0], momentum, np, landau,
                        least_squares, minimize_scalar, brentq,
                        chi2_distribution,
                    )
                fit_rows.append(result["row"])
                if result["row"]["optimizer_success"]:
                    fit_results[momentum] = result
                else:
                    anomalies.append({
                        "severity": "warning",
                        "code": "fit_optimizer_failed",
                        "csv_row": "",
                        "momentum_GeV_c": momentum,
                        "trigger_time": "",
                        "detail": result["row"]["message"],
                    })
                if not result["row"]["covariance_valid"]:
                    anomalies.append({
                        "severity": "warning",
                        "code": "fit_covariance_unavailable",
                        "csv_row": "",
                        "momentum_GeV_c": momentum,
                        "trigger_time": "",
                        "detail": "Matrice di covarianza non invertibile o non definita positiva.",
                    })
                if result["row"]["parameters_near_bounds"]:
                    anomalies.append({
                        "severity": "warning",
                        "code": "fit_parameter_near_bound",
                        "csv_row": "",
                        "momentum_GeV_c": momentum,
                        "trigger_time": "",
                        "detail": (
                            "Parametri entro lo 0.1% del range ammesso: "
                            f"{result['row']['parameters_near_bounds']}."
                        ),
                    })
                if result["row"]["deviance_p_value"] < 0.01:
                    anomalies.append({
                        "severity": "warning",
                        "code": "low_fit_p_value",
                        "csv_row": "",
                        "momentum_GeV_c": momentum,
                        "trigger_time": "",
                        "detail": (
                            "p-value asintotico della devianza di Poisson "
                            f"={result['row']['deviance_p_value']:.4g} < 0.01."
                        ),
                    })
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                fit_rows.append({
                    "momentum_GeV_c": momentum,
                    "model": (
                        "langauss" if momentum == 1
                        else "langauss_plus_gaussian"
                    ),
                    "status": "failed",
                    "message": str(exc),
                    "entries_total": len(prepared[momentum][0]),
                })
                anomalies.append({
                    "severity": "warning",
                    "code": "apa1_population_fit_failed",
                    "csv_row": "",
                    "momentum_GeV_c": momentum,
                    "trigger_time": "",
                    "detail": str(exc),
                })

    errors = sum(item["severity"] == "error" for item in anomalies)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clear_owned_outputs(args.output_dir)
    write_csv(args.output_dir / "anomalies.csv", ANOMALY_FIELDS, anomalies)

    if not errors:
        plt.rcParams.update({
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.linewidth": 1.0,
        })
        def add_preliminary_label(axis):
            axis.text(
                0.98,
                0.96,
                r"$\bf{ProtoDUNE\!-\!HD}$ Preliminary",
                transform=axis.transAxes,
                fontsize=11,
                ha="right",
                va="top",
            )

        def save_figure(fig, stem):
            fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

        def finish_axis(axis, grid_axis="y"):
            axis.tick_params(direction="in", which="both", top=True, right=True)
            axis.grid(
                True, axis=grid_axis, linestyle="--", linewidth=0.5, alpha=0.35
            )

        def format_estimate(value, error, digits=2, unit=None):
            if math.isfinite(value) and math.isfinite(error):
                estimate = f"{value:.{digits}f} ± {error:.{digits}f}"
            elif math.isfinite(value):
                estimate = f"{value:.{digits}f}"
            else:
                return "not available"
            return f"({estimate}) {unit}" if unit else estimate

        for momentum in args.momenta:
            (
                apa1, apa2, paired1, paired2, edges1, edges2, pearson,
                display_max1, display_max2, above_display1, above_display2,
            ) = prepared[momentum]

            fig, axis = plt.subplots(figsize=(9, 5.2))
            axis.hist(
                apa1,
                bins=edges1,
                color=COLORS["apa1"],
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                histtype="stepfilled",
                label=f"{len(apa1)} triggers",
            )
            axis.set_xlim(0, display_max1)
            axis.set_xlabel(r"$\langle N_{\mathrm{PE}} \rangle_{\mathrm{APA\,1}}$")
            axis.set_ylabel("Triggers / bin")
            axis.set_title(
                rf"Photoelectron distribution — APA 1 — $p_{{\rm beam}}={momentum}$ GeV/$c$"
            )
            axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=9, min_n_ticks=5))
            axis.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, integer=True))
            finish_axis(axis)
            axis.legend(loc="upper left", frameon=False)
            add_preliminary_label(axis)
            if above_display1:
                axis.text(
                    0.98, 0.86,
                    f"{above_display1} triggers above displayed range",
                    transform=axis.transAxes, ha="right", va="top", fontsize=9,
                )
            fig.tight_layout()
            save_figure(fig, args.output_dir / f"apa1_hist_{momentum}GeV")

            fig, axis = plt.subplots(figsize=(9, 5.2))
            axis.hist(
                apa2,
                bins=edges2,
                color=COLORS["apa2"],
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                histtype="stepfilled",
                label=f"{len(apa2)} triggers",
            )
            axis.set_xlim(0, display_max2)
            axis.set_xlabel(r"$\langle N_{\mathrm{PE}} \rangle_{\mathrm{APA\,2}}$")
            axis.set_ylabel("Triggers / bin")
            axis.set_title(
                rf"Photoelectron distribution — APA 2 — $p_{{\rm beam}}={momentum}$ GeV/$c$"
            )
            axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=9, min_n_ticks=5))
            axis.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, integer=True))
            finish_axis(axis)
            axis.legend(loc="upper left", frameon=False)
            add_preliminary_label(axis)
            if above_display2:
                axis.text(
                    0.98, 0.86,
                    f"{above_display2} triggers above displayed range",
                    transform=axis.transAxes, ha="right", va="top", fontsize=9,
                )
            fig.tight_layout()
            save_figure(fig, args.output_dir / f"apa2_hist_{momentum}GeV")

            fig, axis = plt.subplots(figsize=(8, 6))
            axis.scatter(
                paired1,
                paired2,
                c=COLORS["scatter"],
                s=8,
                alpha=0.4,
                edgecolors="none",
                rasterized=True,
                label=f"{len(paired1)} triggers",
            )
            axis.set_xlim(0, display_max1)
            axis.set_ylim(0, display_max2)
            axis.set_xlabel(r"$\langle N_{\mathrm{PE}} \rangle_{\mathrm{APA\,1}}$")
            axis.set_ylabel(r"$\langle N_{\mathrm{PE}} \rangle_{\mathrm{APA\,2}}$")
            axis.set_title(
                rf"APA 1–APA 2 photoelectron response — $p_{{\rm beam}}={momentum}$ GeV/$c$"
            )
            axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, min_n_ticks=5))
            axis.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, min_n_ticks=5))
            finish_axis(axis, grid_axis="both")
            axis.legend(loc="upper left", frameon=False)
            axis.text(
                0.98,
                0.84,
                f"Pearson $r$ = {pearson:.3f}",
                transform=axis.transAxes,
                ha="right",
                va="top",
            )
            paired_above_display = int(np.count_nonzero(
                (paired1 > display_max1) | (paired2 > display_max2)
            ))
            if paired_above_display:
                axis.text(
                    0.98, 0.76,
                    f"{paired_above_display} triggers outside displayed range",
                    transform=axis.transAxes, ha="right", va="top", fontsize=9,
                )
            add_preliminary_label(axis)
            fig.tight_layout()
            save_figure(fig, args.output_dir / f"apa12_pe_distribution_{momentum}GeV")

            if momentum in fit_results:
                fit_result = fit_results[momentum]
                fit_row = fit_result["row"]
                fit_edges = fit_result["edges"]
                fit_observed = fit_result["observed"]
                fit_parameters = fit_result["parameters"]
                dense_x = np.linspace(fit_edges[0], fit_edges[-1], 3000)
                fit_width = fit_row["bin_width"]
                if fit_row["model"] == "langauss":
                    langauss_counts = fit_width * fit_result["langauss_density"](
                        fit_parameters, dense_x
                    )
                    total_counts = langauss_counts
                else:
                    langauss_density, gaussian_density = fit_result["components"](
                        fit_parameters, dense_x
                    )
                    langauss_counts = fit_width * langauss_density
                    gaussian_counts = fit_width * gaussian_density
                    total_counts = langauss_counts + gaussian_counts

                # Mostra anche i dati a sinistra del range adattato; il fit
                # continua a usare esclusivamente i bin originali fit_edges.
                plot_minimum = max(
                    0.0,
                    math.floor(
                        (fit_edges[0] - 0.12 * (fit_edges[-1] - fit_edges[0]))
                        / fit_width
                    ) * fit_width,
                )
                plot_edges = np.arange(
                    plot_minimum, fit_edges[-1] + 0.5 * fit_width,
                    fit_width, dtype=float,
                )
                plot_observed, _ = np.histogram(apa1, bins=plot_edges)
                fig, axis = plt.subplots(figsize=(8, 5))
                axis.stairs(
                    plot_observed,
                    plot_edges,
                    fill=True,
                    color=COLORS["data"],
                    edgecolor=COLORS["total"],
                    linewidth=0.8,
                    label=f"Data ({int(np.sum(plot_observed))} triggers shown)",
                )
                if fit_row["model"] == "langauss":
                    axis.plot(
                        dense_x, langauss_counts, color=COLORS["total"],
                        linewidth=2.0, label="Langauss fit",
                    )
                else:
                    axis.plot(
                        dense_x, total_counts, color=COLORS["total"],
                        linewidth=1.8, label="Langauss + Gaussian",
                    )
                    axis.plot(
                        dense_x, langauss_counts, color=COLORS["langauss"],
                        linestyle="--", linewidth=1.5,
                        label="Langauss (muon-like population)",
                    )
                    axis.plot(
                        dense_x, gaussian_counts, color=COLORS["gaussian"],
                        linestyle="--", linewidth=1.5,
                        label="Gaussian (non-muon-like population)",
                    )
                    axis.axvline(
                        fit_row["intersection"], color=COLORS["intersection"],
                        linestyle=":", linewidth=1.8,
                        label="Intersection = " + format_estimate(
                            fit_row["intersection"],
                            fit_row["intersection_error"], unit="PE",
                        ),
                    )
                axis.set_ylabel("Trigger counts")
                title = (
                    "APA 1 Langauss fit" if fit_row["model"] == "langauss"
                    else "APA 1 population fit"
                )
                axis.set_title(rf"{title} — $p_{{\rm beam}}={momentum}$ GeV/$c$")
                axis.yaxis.set_major_locator(ticker.MaxNLocator(nbins=7, integer=True))
                axis.set_xlabel(r"$\langle N_{\mathrm{PE}} \rangle_{\mathrm{APA\,1}}$")
                axis.set_xlim(plot_edges[0], plot_edges[-1])
                axis.set_ylim(0, max(plot_observed) * 1.22)
                finish_axis(axis, grid_axis="both")
                legend = axis.legend(
                    loc="upper left", frameon=True, facecolor="white",
                    framealpha=1.0, edgecolor="0.75", fontsize=8.3,
                )
                legend.set_zorder(5)

                quality_text = (
                    rf"$\chi^2/\mathrm{{ndf}}$ = "
                    f"{fit_row['pearson_chi2']:.1f}/{fit_row['ndf']}"
                    f" = {fit_row['pearson_chi2_per_ndf']:.2f}"
                    "\n" + rf"$R^2$ = {fit_row['r_squared']:.3f}"
                )
                langauss_text = (
                        "MPV = " + format_estimate(
                            fit_row["mpv"], fit_row["mpv_error"], unit="PE"
                        ) + "\n"
                        + r"$\eta$ = " + format_estimate(
                            fit_row["eta"], fit_row["eta_error"], unit="PE"
                        ) + "\n"
                        + r"$\sigma_{\rm LG}$ = " + format_estimate(
                            fit_row["langauss_sigma"],
                            fit_row["langauss_sigma_error"], unit="PE",
                        ) + "\n"
                        + r"$N_{\rm LG}$ = " + format_estimate(
                            fit_row["langauss_yield"],
                            fit_row["langauss_yield_error"], digits=0,
                        ) + "\n"
                        + r"$x_{\rm peak}$ = " + format_estimate(
                            fit_row["langauss_peak"],
                            fit_row["langauss_peak_error"], unit="PE",
                        )
                )
                if fit_row["model"] == "langauss":
                    info_text = "Langauss:\n" + langauss_text + "\n\n" + quality_text
                else:
                    gaussian_text = (
                        r"$\mu$ = " + format_estimate(
                            fit_row["gaussian_mean"],
                            fit_row["gaussian_mean_error"], unit="PE",
                        ) + "\n"
                        + r"$\sigma_{\rm G}$ = " + format_estimate(
                            fit_row["gaussian_sigma"],
                            fit_row["gaussian_sigma_error"], unit="PE",
                        ) + "\n"
                        + r"$N_{\rm G}$ = " + format_estimate(
                            fit_row["gaussian_yield"],
                            fit_row["gaussian_yield_error"], digits=0,
                        )
                    )
                    info_text = (
                        "Langauss:\n" + langauss_text + "\n\nGaussian:\n"
                        + gaussian_text + "\n\n" + quality_text
                        + "\n\nIntersection = " + format_estimate(
                            fit_row["intersection"],
                            fit_row["intersection_error"], unit="PE",
                        )
                    )
                info_box = AnchoredText(
                    info_text, loc="upper right", frameon=True,
                    prop={"size": 7.7}, borderpad=0.5,
                )
                info_box.patch.set_facecolor("white")
                info_box.patch.set_alpha(0.94)
                info_box.patch.set_edgecolor("0.65")
                axis.add_artist(info_box)
                axis.text(
                    0.52, 0.97, r"$\bf{ProtoDUNE\!-\!HD}$ Preliminary",
                    transform=axis.transAxes, fontsize=9,
                    ha="center", va="top",
                )
                fig.tight_layout()
                save_figure(
                    fig, args.output_dir / f"apa1_population_fit_{momentum}GeV"
                )

        write_csv(args.output_dir / "plot_summary.csv", SUMMARY_FIELDS, summaries)
        write_csv(
            args.output_dir / "apa1_population_fit_results.csv",
            FIT_FIELDS,
            fit_rows,
        )
        extreme_events.sort(
            key=lambda row: (row["momentum_GeV_c"], row["trigger_time"])
        )
        write_csv(
            args.output_dir / "extreme_events.csv", EXTREME_FIELDS, extreme_events
        )
        (args.output_dir / "histogram_bins.json").write_text(
            json.dumps({
                "method": "numpy.histogram_bin_edges with bins='fd'",
                "meaning": "Diagnostic binning only; not a fit range.",
                "edges": bins_manifest,
            }, indent=2) + "\n",
            encoding="utf-8",
        )

    input_bytes = args.input_file.read_bytes()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "input_file": str(args.input_file),
            "output_dir": str(args.output_dir),
            "momenta": args.momenta,
            "histogram_binning": "Freedman-Diaconis, independently for each APA and momentum",
            "display_range": (
                "0 to max(99th percentile, Q3 + 5*IQR), with 3% upper margin; "
                "the full sample remains in all calculations"
            ),
            "extreme_events_definition": (
                "Valid APA mean above the corresponding displayed upper limit; "
                "events are reported but not removed from calculations."
            ),
            "cuts": "none",
            "fits": (
                "skipped by command-line option" if args.skip_fits else
                "APA 1 at 1 GeV/c: single normalized Langauss in 10--150 PE; "
                "APA 1 at 2, 3, 5 and 7 GeV/c when requested: normalized "
                "Langauss + Gaussian. All fits include empty histogram bins and "
                "use signed Poisson-deviance residuals"
            ),
            "fit_range": (
                "1 GeV/c: fixed 10--150 PE interval, following the historical "
                "analysis; 2, 3, 5 and 7 GeV/c: bin-aligned interval spanning "
                "P0.5 to P99.5. Entries outside are reported and retained in the "
                "source dataset"
            ),
            "fit_bin_widths_PE": {
                "1": 2.0,
                **{
                    str(momentum): configuration["bin_width"]
                    for momentum, configuration in FIT_CONFIGURATION.items()
                },
            },
            "fit_component_anchors_PE": {
                str(momentum): configuration["component_anchor"]
                for momentum, configuration in FIT_CONFIGURATION.items()
            },
            "fit_component_anchor_meaning": (
                "Value used to initialize and identify the lower Langauss and upper "
                "Gaussian components; it is not the selected threshold"
            ),
            "threshold_definition": (
                "First Langauss-Gaussian intersection between the numerical "
                "Langauss peak and the fitted Gaussian mean"
            ),
            "langauss_evaluation": (
                "Explicit midpoint convolution of landaupy.landau.pdf with a "
                "Gaussian over +/-5 sigma; at least 12 integration points per "
                "fitted Landau eta, evaluated in memory-bounded chunks"
            ),
            "goodness_of_fit": (
                "Poisson deviance/ndf with asymptotic p-value and Pearson "
                "chi-square/ndf; R-squared is reported as descriptive only"
            ),
            "figure_format": "PNG only",
            "output_policy": (
                "The output directory may be reused; only files owned by this script "
                "are replaced. Legacy PDF outputs with the same stems are removed."
            ),
        },
        "input": {
            "bytes": len(input_bytes),
            "sha256": hashlib.sha256(input_bytes).hexdigest(),
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "scipy_version": (
            __import__("scipy").__version__
            if fit_momenta or fit_one_gev else None
        ),
        "figures_created": not errors,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    if args.skip_fits:
        fit_description = "Fit APA 1: non eseguiti (--skip-fits)."
    elif fit_momenta or fit_one_gev:
        descriptions = []
        if fit_one_gev:
            descriptions.append("Langauss singola a 1 GeV/c nel range 10--150 PE")
        if fit_momenta:
            descriptions.append(
                f"Langauss + Gaussiana ai momenti {fit_momenta} GeV/c"
            )
        fit_description = (
            "Fit APA 1: " + "; ".join(descriptions)
            + "; funzione obiettivo basata sulla devianza di Poisson."
        )
    else:
        fit_description = "Fit APA 1: nessun momento adatto richiesto."

    lines = [
        "FIGURE DIAGNOSTICHE E FIT APA 1–APA 2",
        f"Input: {args.input_file}",
        f"Output: {args.output_dir}",
        f"Momenti nominali (GeV/c): {args.momenta}",
        "Binning: Freedman–Diaconis, separato per APA e momento.",
        "Tagli applicati agli eventi: nessuno.",
        fit_description,
        "Fit misto APA 2: non eseguito.",
        "",
        "RISULTATI PER MOMENTO",
    ]
    for summary in summaries:
        lines.append(
            f"{summary['momentum_GeV_c']} GeV/c: "
            f"APA1={summary['apa1_entries']}, APA2={summary['apa2_entries']}, "
            f"scatter={summary['scatter_entries']}, "
            f"Pearson r={summary['pearson_r']:.4f}; "
            f"bin APA1={summary['apa1_histogram_bins']}, "
            f"bin APA2={summary['apa2_histogram_bins']}."
        )
    if not args.skip_fits:
        lines += ["", "RISULTATI FIT APA 1"]
        for row in fit_rows:
            momentum = row["momentum_GeV_c"]
            if row["status"] != "success":
                lines.append(
                    f"{momentum} GeV/c: FIT NON RIUSCITO — {row['message']}"
                )
                continue
            if row["model"] == "langauss":
                lines.append(
                    f"{momentum} GeV/c: Langauss singola; picco="
                    f"{row['langauss_peak']:.2f} ± "
                    f"{row['langauss_peak_error']:.2f} PE, "
                    f"MPV={row['mpv']:.2f} ± {row['mpv_error']:.2f} PE, "
                    f"eta={row['eta']:.2f} ± {row['eta_error']:.2f} PE, "
                    f"sigma={row['langauss_sigma']:.2f} ± "
                    f"{row['langauss_sigma_error']:.2f} PE; "
                    f"D/ndf={row['deviance_per_ndf']:.3f}, "
                    f"p={row['deviance_p_value']:.4g}; "
                    f"Pearson chi2/ndf={row['pearson_chi2_per_ndf']:.3f}, "
                    f"R2={row['r_squared']:.4f}; "
                    f"range=[{row['fit_minimum']:.1f}, "
                    f"{row['fit_maximum']:.1f}] PE, "
                    f"bin={row['bin_width']:.1f} PE."
                )
                continue
            intersection_error = row["intersection_error"]
            error_text = (
                f" ± {intersection_error:.2f}" if math.isfinite(intersection_error)
                else " (incertezza non disponibile)"
            )
            lines.append(
                f"{momentum} GeV/c: intersezione={row['intersection']:.2f}"
                f"{error_text} PE; media Gaussiana={row['gaussian_mean']:.2f} "
                f"± {row['gaussian_mean_error']:.2f} PE; "
                f"D/ndf={row['deviance_per_ndf']:.3f}, "
                f"p={row['deviance_p_value']:.4g}; "
                f"Pearson chi2/ndf={row['pearson_chi2_per_ndf']:.3f}, "
                f"R2={row['r_squared']:.4f}; "
                f"range=[{row['fit_minimum']:.1f}, {row['fit_maximum']:.1f}] PE, "
                f"bin={row['bin_width']:.1f} PE, "
                f"integrazione Langauss={row['langauss_integration_points']} punti."
            )
    lines += ["", "ANOMALIE PER TIPO"]
    counts = Counter(item["code"] for item in anomalies)
    if counts:
        lines.extend(f"{code}: {count}" for code, count in sorted(counts.items()))
    else:
        lines.append("Nessuna anomalia rilevata.")
    if not errors:
        lines += [
            "",
            "EVENTI OLTRE L'INTERVALLO MOSTRATO",
            f"Eventi distinti elencati in extreme_events.csv: {len(extreme_events)}.",
            "Sono segnalazioni diagnostiche: nessun evento è escluso dai calcoli.",
        ]
    lines += [
        "",
        "COME LEGGERE LE FIGURE",
        "Gli istogrammi usano tutti i valori validi della rispettiva APA.",
        "Lo scatter usa soltanto trigger con entrambe le medie valide.",
        "Tutti i dati sono conservati nelle statistiche; nessun outlier viene rimosso.",
        "Il limite superiore mostrato è max(P99, Q3 + 5 IQR), con margine del 3%.",
        "Gli eventi oltre il limite sono conservati e il loro numero è annotato nella figura.",
        "Il coefficiente di Pearson è descrittivo e non definisce una selezione.",
        "Il binning diagnostico non stabilisce il binning o l'intervallo del fit.",
        "Il fit usa anche i bin vuoti e minimizza residui della devianza di Poisson.",
        "A 1 GeV/c il range di fit è 10--150 PE; agli altri momenti copre P0.5--P99.5.",
        "I punti fuori dal range di fit non sono cancellati dal dataset.",
        "La soglia candidata è l'intersezione tra Langauss e Gaussiana compresa tra i picchi.",
        "Le incertezze sono statistiche locali e condizionate a modello, binning e range.",
        "Il p-value della devianza è una valutazione asintotica della qualità del fit.",
        "A 1 GeV/c il fit è una singola Langauss e non definisce una soglia.",
        "A 1 GeV/c non viene assunta la presenza di due popolazioni separabili.",
        "APA 2 non viene adattata: la risposta self-trigger dipende dai canali contribuenti.",
        "",
        ("Figure create." if not errors else "Figure NON create a causa degli errori sopra elencati."),
    ]
    report = "\n".join(lines) + "\n"
    (args.output_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 2 if errors else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OSError as exc:
        print(f"Errore di accesso ai file: {exc}", file=sys.stderr)
        sys.exit(2)
