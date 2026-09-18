#!/usr/bin/env python3
r"""Crea figure diagnostiche APA 1–APA 2 senza fit o selezioni.

SCOPO
    Per ogni momento nominale disegna la distribuzione di apa1_mean, la
    distribuzione di apa2_mean e lo scatter plot trigger per trigger tra le due
    medie. Le figure servono a osservare le popolazioni prima di scegliere un
    modello o un intervallo di fit. Non applica soglie, tagli muonici, fit o
    rimozione di outlier.

INPUT
    --input-file: apa12_trigger_data.csv prodotto da
    prepare_apa12_trigger_data.py.
    --momenta: momenti da rappresentare (default: 1 2 3 5 7 GeV/c).
    --output-dir: nuova cartella per risultati e figure; non viene sovrascritta.
    Per l'istogramma di ciascuna APA sono usate tutte le righe con la relativa
    media valida. Per lo scatter sono usate soltanto le righe both_apa_valid=1.

OUTPUT
    apa1_hist_<momento>GeV.png e .pdf: distribuzione di APA 1;
    apa2_hist_<momento>GeV.png e .pdf: distribuzione di APA 2;
    apa12_pe_distribution_<momento>GeV.png e .pdf: scatter trigger per trigger;
    plot_summary.csv: conteggi e statistiche descrittive;
    histogram_bins.json: bordi esatti dei bin;
    anomalies.csv: incoerenze del dataset di input;
    report.txt: descrizione dei risultati e dei limiti;
    manifest.json: configurazione, versioni e SHA-256 dell'input.
    I bin sono determinati separatamente con la regola di Freedman–Diaconis e
    non devono essere interpretati come intervalli di fit.
    Codice di uscita: 0 = figure create; 2 = errore, figure non create.

ESECUZIONE (dalla cartella scripts/review su LXPlus)
    python plot_apa12_trigger_distributions.py \
        --input-file ../../output/review/apa12_trigger_data_01/apa12_trigger_data.csv \
        --output-dir ../../output/review/apa12_trigger_distributions_01

INTERPRETAZIONE
    apa1_mean e apa2_mean sono medie PE per canale contribuente, non somme
    sull'intera APA. Il coefficiente di Pearson è soltanto descrittivo e non è
    usato per selezionare eventi. Il campione a 1 GeV/c viene rappresentato ma
    non viene forzata una separazione tra popolazioni.
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
    "apa1_valid",
    "apa2_mean",
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

VALID_CATEGORIES = {"both_valid", "apa1_only", "apa2_only", "neither_valid"}


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
                except (ValueError, OverflowError) as exc:
                    issue("invalid_value", csv_row, record.get("momentum_GeV_c", ""),
                          record.get("trigger_time", ""), str(exc))
                    continue

                if momentum not in (1, 2, 3, 5, 7) or trigger < 0:
                    issue("invalid_identity", csv_row, momentum, trigger,
                          "Momento non previsto oppure trigger_time negativo.")
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
                        "apa1_mean": apa1_mean,
                        "apa2_mean": apa2_mean,
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
    args = parser.parse_args()
    args.input_file = args.input_file.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.momenta = sorted(set(args.momenta))

    if not args.input_file.is_file():
        parser.error(f"File input inesistente: {args.input_file}")
    if args.output_dir.exists():
        parser.error(f"La cartella output esiste già: {args.output_dir}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError as exc:
        parser.error(f"Dipendenza mancante nell'ambiente Python: {exc}")

    rows, anomalies = load_rows(args.input_file, set(args.momenta))
    summaries = []
    bins_manifest = {}
    errors = sum(item["severity"] == "error" for item in anomalies)
    prepared = {}

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
            stats1 = descriptive(apa1, np)
            stats2 = descriptive(apa2, np)
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
            prepared[momentum] = (apa1, apa2, paired_apa1, paired_apa2,
                                  edges1, edges2, pearson)

    errors = sum(item["severity"] == "error" for item in anomalies)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "anomalies.csv", ANOMALY_FIELDS, anomalies)

    if not errors:
        plt.rcParams.update({
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
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
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)

        for momentum in args.momenta:
            apa1, apa2, paired1, paired2, edges1, edges2, pearson = prepared[momentum]

            fig, axis = plt.subplots(figsize=(10, 5))
            axis.hist(
                apa1,
                bins=edges1,
                color="tomato",
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                label=f"{len(apa1)} triggers",
            )
            axis.set_xlabel(r"$\langle N_{\mathrm{PE}} \rangle$ on APA 1")
            axis.set_ylabel("Counts")
            axis.set_title(
                rf"Photoelectron distribution — APA 1 — $p_{{\rm beam}}={momentum}$ GeV/$c$"
            )
            axis.xaxis.set_major_locator(ticker.MultipleLocator(50))
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            axis.legend(loc="upper left")
            add_preliminary_label(axis)
            fig.tight_layout()
            save_figure(fig, args.output_dir / f"apa1_hist_{momentum}GeV")

            fig, axis = plt.subplots(figsize=(10, 5))
            axis.hist(
                apa2,
                bins=edges2,
                color="dodgerblue",
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                label=f"{len(apa2)} triggers",
            )
            axis.set_xlabel(r"$\langle N_{\mathrm{PE}} \rangle$ on APA 2")
            axis.set_ylabel("Counts")
            axis.set_title(
                rf"Photoelectron distribution — APA 2 — $p_{{\rm beam}}={momentum}$ GeV/$c$"
            )
            axis.xaxis.set_major_locator(ticker.MultipleLocator(50))
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            axis.legend(loc="upper left")
            add_preliminary_label(axis)
            fig.tight_layout()
            save_figure(fig, args.output_dir / f"apa2_hist_{momentum}GeV")

            fig, axis = plt.subplots(figsize=(12, 4))
            axis.scatter(
                paired1,
                paired2,
                c="dodgerblue",
                s=7,
                alpha=0.7,
                edgecolors="none",
                rasterized=True,
                label=f"{len(paired1)} triggers",
            )
            axis.set_xlabel(r"$\langle N_{\mathrm{PE}} \rangle$ on APA 1")
            axis.set_ylabel(r"$\langle N_{\mathrm{PE}} \rangle$ on APA 2")
            axis.set_title(
                rf"APA 1–APA 2 photoelectron response — $p_{{\rm beam}}={momentum}$ GeV/$c$"
            )
            axis.xaxis.set_major_locator(ticker.MultipleLocator(50))
            axis.yaxis.set_major_locator(ticker.MultipleLocator(50))
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            axis.legend(loc="upper left")
            axis.text(
                0.98,
                0.84,
                f"Pearson $r$ = {pearson:.3f}",
                transform=axis.transAxes,
                ha="right",
                va="top",
            )
            add_preliminary_label(axis)
            fig.tight_layout()
            save_figure(fig, args.output_dir / f"apa12_pe_distribution_{momentum}GeV")

        write_csv(args.output_dir / "plot_summary.csv", SUMMARY_FIELDS, summaries)
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
            "cuts": "none",
            "fits": "none",
        },
        "input": {
            "bytes": len(input_bytes),
            "sha256": hashlib.sha256(input_bytes).hexdigest(),
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "figures_created": not errors,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "FIGURE DIAGNOSTICHE APA 1–APA 2",
        f"Input: {args.input_file}",
        f"Output: {args.output_dir}",
        f"Momenti nominali (GeV/c): {args.momenta}",
        "Binning: Freedman–Diaconis, separato per APA e momento.",
        "Tagli: nessuno. Fit e soglie: nessuno.",
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
    lines += ["", "ANOMALIE PER TIPO"]
    counts = Counter(item["code"] for item in anomalies)
    if counts:
        lines.extend(f"{code}: {count}" for code, count in sorted(counts.items()))
    else:
        lines.append("Nessuna anomalia rilevata.")
    lines += [
        "",
        "COME LEGGERE LE FIGURE",
        "Gli istogrammi usano tutti i valori validi della rispettiva APA.",
        "Lo scatter usa soltanto trigger con entrambe le medie valide.",
        "Gli assi mostrano l'intero intervallo dei dati: non sono rimossi outlier.",
        "Il coefficiente di Pearson è descrittivo e non definisce una selezione.",
        "Il binning diagnostico non stabilisce il binning o l'intervallo del fit.",
        "A 1 GeV/c non viene assunta la presenza di due popolazioni separabili.",
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
