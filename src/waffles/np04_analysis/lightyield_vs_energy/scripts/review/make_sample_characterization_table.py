#!/usr/bin/env python3
r"""Crea una tabella semplice di caratterizzazione dei campioni PDS.

SCOPO
    Analizza lo stesso blocco (default: 0_to_10) per ciascun momento nominale
    e calcola cinque quantità descrittive:
      1. numero medio di canali distinti con waveform APA 1 per trigger FS-ST;
      2. numero medio di canali distinti con waveform APA 2 per trigger FS-ST;
      3. media, tra i trigger validi, di apa1_mean;
      4. media, tra i trigger validi, di apa2_mean;
      5. percentuale di identità evento FS senza alcuna waveform ST.
    Per le prime quattro quantità salva deviazione standard della popolazione
    (ddof=0), minimo e massimo. Conta separatamente eventuali waveform duplicate
    dello stesso canale. Non modifica gli input e non applica tagli muone/non muone.

INPUT
    --input-dir: cartella apa1_vs_apa2 contenente
    <momento>GeV/<blocco>/FS_wfset.pkl, ST_wfset.pkl e
    photoelectron_dataframe_<momento>GeV.csv.
    --momenta: momenti da analizzare (default: 1 2 3 5 7).
    --block: sottocartella usata per tutti i momenti (default: 0_to_10).
    --output-dir: nuova cartella per i risultati; non viene sovrascritta.
    --analysis-label e --delta-ticks devono corrispondere al vecchio notebook;
    i default sono finding_peaks e 100 tick.

OUTPUT
    sample_characterization.csv: valori numerici completi e denominatori;
    sample_characterization_table.md: tabella compatta per la revisione;
    sample_characterization_table.tex: tabella LaTeX, non inserita nella tesi;
    anomalies.csv: incoerenze o associazioni non riproducibili;
    report.txt: definizioni, risultati e limiti;
    manifest.json: configurazione e SHA-256 degli input.
    Codice di uscita: 0 = nessuna anomalia; 1 = anomalie da esaminare;
    2 = errore di esecuzione.

ESECUZIONE (dalla cartella scripts/review su LXPlus)
    python make_sample_characterization_table.py \
        --input-dir ../../output/apa1_vs_apa2 \
        --block 0_to_10 \
        --output-dir ../../output/review/sample_characterization_0_to_10_01

INTERPRETAZIONE
    Le medie di waveform e PE sono calcolate sulle righe del CSV, cioè sul
    campione con associazione FS-ST. La percentuale senza APA 2 usa invece come
    denominatore tutte le identità evento presenti nel pickle FS selezionato.
    I risultati descrivono il blocco scelto e non vengono automaticamente
    presentati come rappresentativi dell'intero campione a quella energia.
"""

import argparse
import bisect
import csv
import gc
import hashlib
import json
import math
import pickle
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[5]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


OUTPUT_FIELDS = [
    "momentum_GeV_c", "block", "paired_csv_triggers", "fs_event_identities",
    "st_event_identities", "shared_fs_st_event_identities",
    "fs_event_identities_without_st", "fs_without_st_percent",
    "apa1_waveforms_per_paired_trigger_mean",
    "apa1_waveforms_per_paired_trigger_population_std",
    "apa1_waveforms_per_paired_trigger_minimum",
    "apa1_waveforms_per_paired_trigger_maximum",
    "apa2_waveforms_per_paired_trigger_mean",
    "apa2_waveforms_per_paired_trigger_population_std",
    "apa2_waveforms_per_paired_trigger_minimum",
    "apa2_waveforms_per_paired_trigger_maximum",
    "apa1_unique_channels_per_paired_trigger_mean",
    "apa1_unique_channels_per_paired_trigger_population_std",
    "apa1_unique_channels_per_paired_trigger_minimum",
    "apa1_unique_channels_per_paired_trigger_maximum",
    "apa2_unique_channels_per_paired_trigger_mean",
    "apa2_unique_channels_per_paired_trigger_population_std",
    "apa2_unique_channels_per_paired_trigger_minimum",
    "apa2_unique_channels_per_paired_trigger_maximum",
    "apa1_triggers_with_duplicate_channel_waveforms",
    "apa2_triggers_with_duplicate_channel_waveforms",
    "apa1_duplicate_channel_waveforms",
    "apa2_duplicate_channel_waveforms",
    "apa1_mean_pe_valid_triggers", "apa1_mean_pe_across_triggers",
    "apa1_mean_pe_across_triggers_population_std",
    "apa1_mean_pe_across_triggers_minimum",
    "apa1_mean_pe_across_triggers_maximum",
    "apa2_mean_pe_valid_triggers", "apa2_mean_pe_across_triggers",
    "apa2_mean_pe_across_triggers_population_std",
    "apa2_mean_pe_across_triggers_minimum",
    "apa2_mean_pe_across_triggers_maximum",
]
ANOMALY_FIELDS = [
    "severity", "code", "momentum_GeV_c", "csv_row", "trigger_time", "detail",
]


def exact_integer(value):
    number = Decimal(str(value).strip())
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError("È richiesto un intero finito.")
    return int(number)


def optional_finite_float(value):
    text = str(value).strip()
    if not text:
        return None
    number = float(text)
    return number if math.isfinite(number) else None


def file_info(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_pickle(path):
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not hasattr(value, "waveforms"):
        raise TypeError(f"{path} non contiene un oggetto con waveforms.")
    return value


def collect_entries(wfset, label, mode, momentum, anomaly):
    entries = []
    identities = set()
    for index, waveform in enumerate(wfset.waveforms):
        identity = (
            exact_integer(waveform.run_number),
            exact_integer(waveform.record_number),
            exact_integer(waveform.daq_window_timestamp),
        )
        identities.add(identity)
        try:
            peaks = waveform.analyses[label].result["beam_peak_absolute_time"]
        except (AttributeError, KeyError, TypeError) as exc:
            anomaly("error", "missing_peak_analysis", momentum,
                    detail=f"{mode}, waveform {index}: {exc}")
            continue
        if len(peaks) != 1:
            anomaly("warning", "unexpected_beam_peak_multiplicity", momentum,
                    detail=f"{mode}, waveform {index}: {len(peaks)} picchi.")
        for peak in peaks:
            entries.append((exact_integer(peak), index, waveform))
    entries.sort(key=lambda item: item[0])
    return entries, identities


def select_waveforms(entries, times, trigger, delta):
    left = bisect.bisect_right(times, trigger - delta)
    right = bisect.bisect_left(times, trigger + delta)
    return {index: waveform for _, index, waveform in entries[left:right]}


def describe(values):
    return statistics.mean(values), statistics.pstdev(values), min(values), max(values)


def rounding_decimals(std, significant_digits=2):
    """Cifre decimali necessarie per al massimo due cifre significative in std."""
    if std == 0:
        return 2
    exponent = math.floor(math.log10(abs(std)))
    return significant_digits - 1 - exponent


def rounded_text(value, decimals):
    rounded = round(value, decimals)
    if decimals > 0:
        return f"{rounded:.{decimals}f}"
    return f"{rounded:.0f}"


def formatted_range(mean, std, minimum, maximum, integer_range=False):
    decimals = rounding_decimals(std)
    minimum_text = f"{minimum:.0f}" if integer_range else rounded_text(minimum, decimals)
    maximum_text = f"{maximum:.0f}" if integer_range else rounded_text(maximum, decimals)
    return (
        f"{rounded_text(mean, decimals)} ± {rounded_text(std, decimals)} "
        f"[{minimum_text}–{maximum_text}]"
    )


def latex_formatted_range(mean, std, minimum, maximum, integer_range=False):
    """Stessa formattazione, usando comandi siunitx e intervallo [min,max]."""
    decimals = rounding_decimals(std)
    minimum_text = f"{minimum:.0f}" if integer_range else rounded_text(minimum, decimals)
    maximum_text = f"{maximum:.0f}" if integer_range else rounded_text(maximum, decimals)
    return (
        rf"$\num{{{rounded_text(mean, decimals)}}}\pm"
        rf"\num{{{rounded_text(std, decimals)}}}\,["
        rf"\num{{{minimum_text}}},\num{{{maximum_text}}}]$"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--momenta", nargs="+", type=int, choices=(1, 2, 3, 5, 7), default=[1, 2, 3, 5, 7])
    parser.add_argument("--block", default="0_to_10")
    parser.add_argument("--analysis-label", default="finding_peaks")
    parser.add_argument("--delta-ticks", type=int, default=100)
    args = parser.parse_args()
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.momenta = sorted(set(args.momenta))
    if not args.input_dir.is_dir():
        parser.error(f"Cartella input inesistente: {args.input_dir}")
    if args.output_dir.exists():
        parser.error(f"La cartella output esiste già: {args.output_dir}")
    if not re.fullmatch(r"\d+_to_\d+", args.block):
        parser.error("--block deve avere forma <start>_to_<stop>, per esempio 0_to_10.")
    start, stop = map(int, args.block.split("_to_"))
    if start >= stop:
        parser.error("L'intervallo indicato da --block deve essere crescente.")
    if args.delta_ticks <= 0:
        parser.error("--delta-ticks deve essere positivo.")

    anomalies = []

    def anomaly(severity, code, momentum, csv_row="", trigger_time="", detail=""):
        anomalies.append(dict(zip(ANOMALY_FIELDS, [
            severity, code, momentum, csv_row, trigger_time, detail,
        ])))

    results = []
    inputs = []
    for momentum in args.momenta:
        block_dir = args.input_dir / f"{momentum}GeV" / args.block
        csv_path = block_dir / f"photoelectron_dataframe_{momentum}GeV.csv"
        fs_path = block_dir / "FS_wfset.pkl"
        st_path = block_dir / "ST_wfset.pkl"
        for path in (csv_path, fs_path, st_path):
            if not path.is_file():
                parser.error(f"Input richiesto assente: {path}")
            inputs.append(file_info(path))

        print(f"{momentum} GeV/c: caricamento FS...", flush=True)
        fs_wfset = load_pickle(fs_path)
        fs_entries, fs_identities = collect_entries(
            fs_wfset, args.analysis_label, "FS_APA1", momentum, anomaly)
        del fs_wfset
        gc.collect()
        print(f"{momentum} GeV/c: caricamento ST...", flush=True)
        st_wfset = load_pickle(st_path)
        st_entries, st_identities = collect_entries(
            st_wfset, args.analysis_label, "ST_APA2", momentum, anomaly)
        del st_wfset
        gc.collect()
        fs_times = [entry[0] for entry in fs_entries]
        st_times = [entry[0] for entry in st_entries]

        csv_rows = []
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, strict=True)
            required = {"trigger_time", "apa1_mean", "apa2_mean"}
            missing = sorted(required - set(reader.fieldnames or []))
            if missing:
                raise ValueError(f"Colonne mancanti in {csv_path}: {missing}")
            for row in reader:
                trigger = exact_integer(row["trigger_time"])
                csv_rows.append((reader.line_num, trigger, row))
        duplicate_times = {value for value, count in Counter(t for _, t, _ in csv_rows).items() if count > 1}
        for csv_row, trigger, _ in csv_rows:
            if trigger in duplicate_times:
                anomaly("error", "duplicate_csv_trigger_time", momentum, csv_row, trigger,
                        "Il tempo compare più volte nel CSV del blocco.")

        fs_counts, st_counts = [], []
        fs_channel_counts, st_channel_counts = [], []
        fs_duplicate_triggers = st_duplicate_triggers = 0
        fs_duplicate_waveforms = st_duplicate_waveforms = 0
        apa1_pe, apa2_pe = [], []
        for csv_row, trigger, row in csv_rows:
            fs = select_waveforms(fs_entries, fs_times, trigger, args.delta_ticks)
            st = select_waveforms(st_entries, st_times, trigger, args.delta_ticks)
            fs_counts.append(len(fs))
            st_counts.append(len(st))
            fs_channels = {(wf.endpoint, wf.channel) for wf in fs.values()}
            st_channels = {(wf.endpoint, wf.channel) for wf in st.values()}
            fs_channel_counts.append(len(fs_channels))
            st_channel_counts.append(len(st_channels))
            fs_duplicates = len(fs) - len(fs_channels)
            st_duplicates = len(st) - len(st_channels)
            fs_duplicate_waveforms += fs_duplicates
            st_duplicate_waveforms += st_duplicates
            fs_duplicate_triggers += int(fs_duplicates > 0)
            st_duplicate_triggers += int(st_duplicates > 0)
            if not fs:
                anomaly("error", "missing_fs_waveforms", momentum, csv_row, trigger,
                        "Nessuna waveform FS nella finestra temporale.")
            if not st:
                anomaly("error", "missing_st_waveforms", momentum, csv_row, trigger,
                        "Nessuna waveform ST nella finestra temporale.")
            fs_full = {(wf.run_number, wf.record_number, wf.daq_window_timestamp) for wf in fs.values()}
            st_full = {(wf.run_number, wf.record_number, wf.daq_window_timestamp) for wf in st.values()}
            if fs and st and not (fs_full & st_full):
                anomaly("error", "no_shared_event_identity", momentum, csv_row, trigger,
                        "FS e ST non condividono (run, record, timestamp DAQ).")
            value = optional_finite_float(row["apa1_mean"])
            if value is not None:
                apa1_pe.append(value)
            value = optional_finite_float(row["apa2_mean"])
            if value is not None:
                apa2_pe.append(value)

        if not csv_rows or not fs_counts or not st_counts or not apa1_pe or not apa2_pe:
            raise ValueError(f"Campione vuoto o privo di medie PE valide per {momentum} GeV/c.")
        fs_mean, fs_std, fs_min, fs_max = describe(fs_counts)
        st_mean, st_std, st_min, st_max = describe(st_counts)
        fs_ch_mean, fs_ch_std, fs_ch_min, fs_ch_max = describe(fs_channel_counts)
        st_ch_mean, st_ch_std, st_ch_min, st_ch_max = describe(st_channel_counts)
        apa1_mean, apa1_std, apa1_min, apa1_max = describe(apa1_pe)
        apa2_mean, apa2_std, apa2_min, apa2_max = describe(apa2_pe)
        shared_identities = fs_identities & st_identities
        fs_without_st = fs_identities - st_identities
        missing_percent = 100 * len(fs_without_st) / len(fs_identities) if fs_identities else math.nan
        if fs_duplicate_waveforms:
            anomaly("warning", "duplicate_apa1_channel_waveforms", momentum,
                    detail=f"{fs_duplicate_waveforms} waveform duplicate in {fs_duplicate_triggers} trigger associati.")
        if st_duplicate_waveforms:
            anomaly("warning", "duplicate_apa2_channel_waveforms", momentum,
                    detail=f"{st_duplicate_waveforms} waveform duplicate in {st_duplicate_triggers} trigger associati.")
        if st_identities - fs_identities:
            anomaly("warning", "st_event_identity_without_fs", momentum,
                    detail=f"Identità ST non presenti in FS: {len(st_identities-fs_identities)}.")
        results.append({
            "momentum_GeV_c": momentum, "block": args.block,
            "paired_csv_triggers": len(csv_rows),
            "fs_event_identities": len(fs_identities),
            "st_event_identities": len(st_identities),
            "shared_fs_st_event_identities": len(shared_identities),
            "fs_event_identities_without_st": len(fs_without_st),
            "fs_without_st_percent": missing_percent,
            "apa1_waveforms_per_paired_trigger_mean": fs_mean,
            "apa1_waveforms_per_paired_trigger_population_std": fs_std,
            "apa1_waveforms_per_paired_trigger_minimum": fs_min,
            "apa1_waveforms_per_paired_trigger_maximum": fs_max,
            "apa2_waveforms_per_paired_trigger_mean": st_mean,
            "apa2_waveforms_per_paired_trigger_population_std": st_std,
            "apa2_waveforms_per_paired_trigger_minimum": st_min,
            "apa2_waveforms_per_paired_trigger_maximum": st_max,
            "apa1_unique_channels_per_paired_trigger_mean": fs_ch_mean,
            "apa1_unique_channels_per_paired_trigger_population_std": fs_ch_std,
            "apa1_unique_channels_per_paired_trigger_minimum": fs_ch_min,
            "apa1_unique_channels_per_paired_trigger_maximum": fs_ch_max,
            "apa2_unique_channels_per_paired_trigger_mean": st_ch_mean,
            "apa2_unique_channels_per_paired_trigger_population_std": st_ch_std,
            "apa2_unique_channels_per_paired_trigger_minimum": st_ch_min,
            "apa2_unique_channels_per_paired_trigger_maximum": st_ch_max,
            "apa1_triggers_with_duplicate_channel_waveforms": fs_duplicate_triggers,
            "apa2_triggers_with_duplicate_channel_waveforms": st_duplicate_triggers,
            "apa1_duplicate_channel_waveforms": fs_duplicate_waveforms,
            "apa2_duplicate_channel_waveforms": st_duplicate_waveforms,
            "apa1_mean_pe_valid_triggers": len(apa1_pe),
            "apa1_mean_pe_across_triggers": apa1_mean,
            "apa1_mean_pe_across_triggers_population_std": apa1_std,
            "apa1_mean_pe_across_triggers_minimum": apa1_min,
            "apa1_mean_pe_across_triggers_maximum": apa1_max,
            "apa2_mean_pe_valid_triggers": len(apa2_pe),
            "apa2_mean_pe_across_triggers": apa2_mean,
            "apa2_mean_pe_across_triggers_population_std": apa2_std,
            "apa2_mean_pe_across_triggers_minimum": apa2_min,
            "apa2_mean_pe_across_triggers_maximum": apa2_max,
        })
        del fs_entries, st_entries
        gc.collect()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "sample_characterization.csv", OUTPUT_FIELDS, results)
    write_csv(args.output_dir / "anomalies.csv", ANOMALY_FIELDS, anomalies)

    header = [
        "Momento", "Canali con WF APA 1 / trigger", "Canali ST APA 2 / trigger",
        "Media PE/canale APA 1", "Media PE/canale APA 2", "FS senza APA 2",
    ]
    md_lines = [
        "| " + " | ".join(header) + " |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    if args.block == "0_to_10":
        latex_subsample = "the first ten input files"
    else:
        latex_block = args.block.replace("_", r"\_")
        latex_subsample = rf"the \texttt{{{latex_block}}} input-file block"

    tex_lines = [
        r"% Required packages: booktabs, multirow, makecell, graphicx, siunitx, glossaries",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{%",
        rf"Characterization of the \gls{{pds}} subsample obtained from {latex_subsample} at each beam \gls{{momentum}}.",
        r"For triggers with matched \gls{apa}~1 \gls{fs} and \gls{apa}~2 \gls{st} information, the table reports the number of distinct channels with an associated \gls{waveform} and the mean \gls{pe} response per contributing channel.",
        r"The \gls{apa}~1-only fraction is the percentage of \gls{fs} event identities without an associated \gls{apa}~2 \gls{st} \gls{waveform}.",
        r"Values are reported as $\mu\pm\sigma\,[\min,\max]$, where $\mu$ and $\sigma$ are the mean and population standard deviation across triggers.",
        r"The \gls{pe} averages include only channels with an available template and a valid fit; noisy channels without a reliable template are excluded.",
        r"No particle-species selection is applied.",
        r"}",
        r"\label{tab:pds_fs_st_sample_characterization}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{c c c c c c}",
        r"\toprule",
        r"\multirow{2}{*}{Beam \gls{momentum} [\si{\GeV/c}]} &",
        r"\multicolumn{3}{c}{\gls{apa}~1} &",
        r"\multicolumn{2}{c}{\gls{apa}~2} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
        r"& \makecell{Channels with a\\\gls{waveform} per trigger}",
        r"& \makecell{Mean \gls{pe}\\per channel}",
        r"& \makecell{\gls{apa}~1-only\\triggers [\si{\percent}]}",
        r"& \makecell{Channels with a\\\gls{waveform} per trigger}",
        r"& \makecell{Mean \gls{pe}\\per channel} \\",
        r"\midrule",
    ]
    for row in results:
        cells = [
            f"{row['momentum_GeV_c']} GeV/c",
            formatted_range(
                row["apa1_unique_channels_per_paired_trigger_mean"],
                row["apa1_unique_channels_per_paired_trigger_population_std"],
                row["apa1_unique_channels_per_paired_trigger_minimum"],
                row["apa1_unique_channels_per_paired_trigger_maximum"], integer_range=True),
            formatted_range(
                row["apa2_unique_channels_per_paired_trigger_mean"],
                row["apa2_unique_channels_per_paired_trigger_population_std"],
                row["apa2_unique_channels_per_paired_trigger_minimum"],
                row["apa2_unique_channels_per_paired_trigger_maximum"], integer_range=True),
            formatted_range(
                row["apa1_mean_pe_across_triggers"],
                row["apa1_mean_pe_across_triggers_population_std"],
                row["apa1_mean_pe_across_triggers_minimum"],
                row["apa1_mean_pe_across_triggers_maximum"]),
            formatted_range(
                row["apa2_mean_pe_across_triggers"],
                row["apa2_mean_pe_across_triggers_population_std"],
                row["apa2_mean_pe_across_triggers_minimum"],
                row["apa2_mean_pe_across_triggers_maximum"]),
            f"{row['fs_without_st_percent']:.1f}%",
        ]
        md_lines.append("| " + " | ".join(cells) + " |")
        tex_cells = [
            rf"\num{{{row['momentum_GeV_c']}}}",
            latex_formatted_range(
                row["apa1_unique_channels_per_paired_trigger_mean"],
                row["apa1_unique_channels_per_paired_trigger_population_std"],
                row["apa1_unique_channels_per_paired_trigger_minimum"],
                row["apa1_unique_channels_per_paired_trigger_maximum"], integer_range=True),
            latex_formatted_range(
                row["apa1_mean_pe_across_triggers"],
                row["apa1_mean_pe_across_triggers_population_std"],
                row["apa1_mean_pe_across_triggers_minimum"],
                row["apa1_mean_pe_across_triggers_maximum"]),
            rf"\num{{{row['fs_without_st_percent']:.1f}}}",
            latex_formatted_range(
                row["apa2_unique_channels_per_paired_trigger_mean"],
                row["apa2_unique_channels_per_paired_trigger_population_std"],
                row["apa2_unique_channels_per_paired_trigger_minimum"],
                row["apa2_unique_channels_per_paired_trigger_maximum"], integer_range=True),
            latex_formatted_range(
                row["apa2_mean_pe_across_triggers"],
                row["apa2_mean_pe_across_triggers_population_std"],
                row["apa2_mean_pe_across_triggers_minimum"],
                row["apa2_mean_pe_across_triggers_maximum"]),
        ]
        tex_lines.append(" & ".join(tex_cells) + " \\\\")
    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    note = (
        f"Blocco analizzato per ogni momento: {args.block}. Le prime quattro quantità sono "
        "calcolate sui trigger associati FS–ST; la percentuale usa tutte le identità evento FS. "
        "La dispersione riportata è la deviazione standard della popolazione (ddof=0)."
    )
    (args.output_dir / "sample_characterization_table.md").write_text(
        "\n".join(md_lines) + "\n\n" + note + "\n", encoding="utf-8")
    (args.output_dir / "sample_characterization_table.tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="utf-8")

    created = datetime.now(timezone.utc)
    manifest = {
        "created_utc": created.isoformat(),
        "configuration": {
            "input_dir": str(args.input_dir), "output_dir": str(args.output_dir),
            "momenta": args.momenta, "block": args.block,
            "analysis_label": args.analysis_label, "delta_ticks": args.delta_ticks,
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version, "inputs": inputs,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    codes = Counter(row["code"] for row in anomalies)
    lines = [
        "TABELLA DI CARATTERIZZAZIONE DEI CAMPIONI PDS",
        f"Blocco usato per ogni momento: {args.block}.",
        "", *md_lines, "", note, "", "ANOMALIE PER TIPO",
    ]
    lines += [f"{code}: {count}" for code, count in sorted(codes.items())]
    if not anomalies:
        lines.append("Nessuna anomalia rilevata dai controlli implementati.")
    lines += [
        "", "LIMITI",
        "Le colonne dei canali contano coppie endpoint-canale distinte; eventuali waveform duplicate sono registrate soltanto nel CSV completo.",
        "La media PE/canale è la media tra trigger di apa1_mean o apa2_mean; non è la somma dei PE dell'APA.",
        "Le medie PE usano soltanto canali dotati di template e con fit valido; i canali rumorosi esclusi dai template non sono trattati come PE nulli.",
        "I trigger con media PE NaN sono esclusi soltanto dalla relativa media PE.",
        "Il campione contiene la miscela di particelle del fascio e non rappresenta la sola popolazione non muonica.",
        "Un singolo blocco è un sottocampione: la stabilità temporale deve essere verificata prima di attribuire i valori all'intera energia.",
    ]
    report = "\n".join(lines) + "\n"
    (args.output_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 1 if anomalies else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, TypeError, KeyError, pickle.UnpicklingError,
            EOFError, AttributeError, ImportError, csv.Error, InvalidOperation) as exc:
        print(f"Errore: {exc}", file=sys.stderr)
        sys.exit(2)
