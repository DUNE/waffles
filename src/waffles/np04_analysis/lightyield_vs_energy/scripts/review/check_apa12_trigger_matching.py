#!/usr/bin/env python3
r"""Controllo dell'associazione temporale tra APA 1 e APA 2 in un blocco.

SCOPO
    Confronta ogni trigger candidato del CSV con le waveform full-streaming
    (APA 1) e self-trigger (APA 2) già salvate nei due pickle. Ricostruisce la
    stessa selezione temporale del vecchio notebook (distanza strettamente
    minore di --delta-ticks), confronta run, record e timestamp DAQ e segnala
    waveform assegnate a più righe. Non ricalcola i fotoelettroni, non esegue
    fit e non modifica né i CSV né i pickle.

INPUT
    --block-dir: una cartella come ../../output/apa1_vs_apa2/1GeV/0_to_10,
    contenente FS_wfset.pkl, ST_wfset.pkl e photoelectron_dataframe_1GeV.csv.
    --output-dir: una nuova cartella per i risultati; non viene sovrascritta.
    --analysis-label: etichetta dell'analisi dei picchi (default: finding_peaks).
    --delta-ticks: semilarghezza esclusiva usata dal vecchio notebook
    (default: 100 tick, quindi abs(tempo_picco - trigger_time) < 100).
    Il nome del CSV viene ricavato dalla cartella del momento, per esempio
    1GeV -> photoelectron_dataframe_1GeV.csv.

OUTPUT
    trigger_matching.csv: una riga per trigger del CSV, con numero di waveform,
    metadati distinti e indicatori di accordo tra FS e ST;
    anomalies.csv: casi mancanti, ambigui o non riproducibili;
    waveform_summary.csv: riepilogo dei due pickle;
    report.txt: risultati e spiegazione delle categorie;
    manifest.json: configurazione e SHA-256 dei tre file letti.
    Codice di uscita: 0 = nessuna anomalia; 1 = anomalie da esaminare;
    2 = errore di esecuzione. Un codice 0 non prova da solo che record o
    timestamp DAQ debbano avere la stessa definizione nei due flussi.

ESECUZIONE (dalla cartella scripts/review su LXPlus)
    python check_apa12_trigger_matching.py \
        --block-dir ../../output/apa1_vs_apa2/1GeV/0_to_10 \
        --output-dir ../../output/review/check_trigger_matching_1GeV_0_to_10_01

    Il programma carica circa la somma delle dimensioni dei due pickle, più
    l'overhead Python delle waveform. Per questo primo controllo usare un solo
    blocco alla volta.
"""

import argparse
import bisect
import csv
import hashlib
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


# Permette di importare waffles dalla repository anche eseguendo il programma
# direttamente dalla cartella scripts/review.
SRC_DIR = Path(__file__).resolve().parents[5]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


MATCH_FIELDS = [
    "csv_row", "trigger_time", "apa1_mean", "apa2_mean", "apa1_n_events",
    "apa2_n_events", "fs_waveforms", "st_waveforms", "fs_distinct_peak_times",
    "st_distinct_peak_times", "minimum_fs_st_peak_distance_ticks",
    "fs_run_numbers", "st_run_numbers", "shared_run_number",
    "fs_record_numbers", "st_record_numbers", "shared_record_number",
    "fs_run_record_pairs", "st_run_record_pairs", "shared_run_record_pair",
    "fs_daq_timestamps", "st_daq_timestamps", "shared_daq_timestamp",
    "minimum_daq_timestamp_distance_ticks", "fs_full_identities",
    "st_full_identities", "shared_full_identity", "fs_duplicate_channel_waveforms",
    "st_duplicate_channel_waveforms", "matching_category",
]
ANOMALY_FIELDS = [
    "severity", "code", "csv_row", "trigger_time", "mode", "waveform_index",
    "detail",
]
WAVEFORM_SUMMARY_FIELDS = [
    "mode", "pickle", "waveforms", "waveforms_with_analysis",
    "waveforms_with_one_beam_peak", "waveforms_with_zero_beam_peaks",
    "waveforms_with_multiple_beam_peaks", "beam_peaks", "run_numbers",
    "record_numbers", "daq_timestamps", "endpoints", "channels",
]


def exact_integer(raw):
    value = Decimal(str(raw).strip())
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError("È richiesto un intero finito.")
    return int(value)


def compact(values):
    return ";".join(str(value) for value in sorted(values))


def compact_pairs(values):
    return ";".join(":".join(str(part) for part in value) for value in sorted(values))


def minimum_distance(first, second):
    if not first or not second:
        return ""
    a, b = sorted(first), sorted(second)
    i = j = 0
    best = None
    while i < len(a) and j < len(b):
        distance = abs(a[i] - b[j])
        best = distance if best is None else min(best, distance)
        if a[i] < b[j]:
            i += 1
        else:
            j += 1
    return best


def trigger_time_searching(values, delta):
    """Riproduce trigger_time_searching del vecchio utils.py."""
    values = sorted(values)
    if not values:
        return []
    filtered = [values[0]]
    for value in values[1:]:
        if abs(value - filtered[-1]) > delta:
            filtered.append(value)
    return filtered


def file_info(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load_pickle(path):
    with path.open("rb") as handle:
        result = pickle.load(handle)
    if not hasattr(result, "waveforms"):
        raise TypeError(f"{path} non contiene un oggetto con l'attributo waveforms.")
    return result


def collect_waveforms(wfset, mode, label, anomaly):
    entries = []
    counts = Counter()
    metadata = defaultdict(set)
    for index, waveform in enumerate(wfset.waveforms):
        metadata["run_numbers"].add(waveform.run_number)
        metadata["record_numbers"].add(waveform.record_number)
        metadata["daq_timestamps"].add(waveform.daq_window_timestamp)
        metadata["endpoints"].add(waveform.endpoint)
        metadata["channels"].add(waveform.channel)
        try:
            result = waveform.analyses[label].result
            peak_times = result["beam_peak_absolute_time"]
            counts["with_analysis"] += 1
        except (AttributeError, KeyError, TypeError) as exc:
            anomaly("error", "missing_peak_analysis", mode=mode, waveform_index=index,
                    detail=f"Analisi {label!r} o beam_peak_absolute_time assente: {exc}")
            continue
        try:
            peaks = [exact_integer(value) for value in peak_times]
        except (TypeError, ValueError, InvalidOperation) as exc:
            anomaly("error", "invalid_beam_peak_time", mode=mode, waveform_index=index,
                    detail=str(exc))
            continue
        counts["beam_peaks"] += len(peaks)
        if len(peaks) == 0:
            counts["zero"] += 1
            anomaly("warning", "zero_beam_peaks", mode=mode, waveform_index=index,
                    detail="La waveform salvata non contiene picchi di fascio.")
        elif len(peaks) == 1:
            counts["one"] += 1
        else:
            counts["multiple"] += 1
            anomaly("warning", "multiple_beam_peaks", mode=mode, waveform_index=index,
                    detail=f"La waveform contiene {len(peaks)} picchi di fascio.")
        for peak in peaks:
            entries.append((peak, index, waveform))
    entries.sort(key=lambda item: item[0])
    summary = {
        "mode": mode,
        "waveforms": len(wfset.waveforms),
        "waveforms_with_analysis": counts["with_analysis"],
        "waveforms_with_one_beam_peak": counts["one"],
        "waveforms_with_zero_beam_peaks": counts["zero"],
        "waveforms_with_multiple_beam_peaks": counts["multiple"],
        "beam_peaks": counts["beam_peaks"],
        **{key: compact(values) for key, values in metadata.items()},
    }
    return entries, summary


def selected_entries(entries, times, trigger_time, delta):
    # Il vecchio timestamp_filter usa la disuguaglianza stretta abs(diff) < delta.
    left = bisect.bisect_right(times, trigger_time - delta)
    right = bisect.bisect_left(times, trigger_time + delta)
    return entries[left:right]


def describe_selection(entries):
    waveforms = {}
    peaks = set()
    for peak, index, waveform in entries:
        peaks.add(peak)
        waveforms[index] = waveform
    values = list(waveforms.values())
    run_numbers = {wf.run_number for wf in values}
    record_numbers = {wf.record_number for wf in values}
    run_record = {(wf.run_number, wf.record_number) for wf in values}
    daq = {wf.daq_window_timestamp for wf in values}
    full = {(wf.run_number, wf.record_number, wf.daq_window_timestamp) for wf in values}
    channel_keys = [(wf.run_number, wf.record_number, wf.endpoint, wf.channel) for wf in values]
    duplicate_channels = sum(count - 1 for count in Counter(channel_keys).values() if count > 1)
    return {
        "waveform_indices": set(waveforms), "waveforms": len(values), "peak_times": peaks,
        "run_numbers": run_numbers, "record_numbers": record_numbers,
        "run_record": run_record, "daq": daq, "full": full,
        "duplicate_channels": duplicate_channels,
    }


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--block-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-label", default="finding_peaks")
    parser.add_argument("--delta-ticks", type=int, default=100)
    args = parser.parse_args()
    args.block_dir = args.block_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.delta_ticks <= 0:
        parser.error("--delta-ticks deve essere positivo.")
    if not args.block_dir.is_dir():
        parser.error(f"Cartella del blocco inesistente: {args.block_dir}")
    if args.output_dir.exists():
        parser.error(f"La cartella output esiste già: {args.output_dir}")
    energy_folder = args.block_dir.parent.name
    if not re.fullmatch(r"(?:1|2|3|5|7)GeV", energy_folder):
        parser.error(f"Impossibile ricavare il momento dalla cartella {energy_folder!r}.")
    csv_path = args.block_dir / f"photoelectron_dataframe_{energy_folder}.csv"
    fs_path = args.block_dir / "FS_wfset.pkl"
    st_path = args.block_dir / "ST_wfset.pkl"
    for path in (csv_path, fs_path, st_path):
        if not path.is_file():
            parser.error(f"Input richiesto assente: {path}")

    anomalies = []

    def anomaly(severity, code, csv_row="", trigger_time="", mode="",
                waveform_index="", detail=""):
        anomalies.append(dict(zip(ANOMALY_FIELDS, [
            severity, code, csv_row, trigger_time, mode, waveform_index, detail,
        ])))

    print(f"Caricamento {fs_path.name}...", flush=True)
    fs_wfset = load_pickle(fs_path)
    print(f"Caricamento {st_path.name}...", flush=True)
    st_wfset = load_pickle(st_path)
    fs_entries, fs_summary = collect_waveforms(
        fs_wfset, "FS_APA1", args.analysis_label, anomaly)
    st_entries, st_summary = collect_waveforms(
        st_wfset, "ST_APA2", args.analysis_label, anomaly)
    fs_summary["pickle"], st_summary["pickle"] = str(fs_path), str(st_path)
    fs_times = [entry[0] for entry in fs_entries]
    st_times = [entry[0] for entry in st_entries]

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        required = {"trigger_time", "apa1_mean", "apa2_mean", "apa1_n_events", "apa2_n_events"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Colonne mancanti nel CSV: {missing}")
        csv_rows = []
        for row in reader:
            try:
                trigger = exact_integer(row["trigger_time"])
            except (ValueError, InvalidOperation) as exc:
                anomaly("error", "invalid_csv_trigger_time", reader.line_num,
                        row.get("trigger_time", ""), detail=str(exc))
                continue
            csv_rows.append((reader.line_num, trigger, row))

    csv_times = [trigger for _, trigger, _ in csv_rows]
    duplicate_csv_times = {value for value, count in Counter(csv_times).items() if count > 1}
    for csv_row, trigger, _ in csv_rows:
        if trigger in duplicate_csv_times:
            anomaly("warning", "duplicate_csv_trigger_time", csv_row, trigger,
                    detail="Il tempo candidato compare più volte nel CSV.")

    rebuilt = trigger_time_searching(fs_times + st_times, args.delta_ticks)
    rebuilt_set, csv_set = set(rebuilt), set(csv_times)
    for trigger in sorted(csv_set - rebuilt_set):
        anomaly("warning", "csv_trigger_not_rebuilt", trigger_time=trigger,
                detail="Il tempo del CSV non compare nella lista ricostruita dai pickle.")
    for trigger in sorted(rebuilt_set - csv_set):
        fs_count = len(describe_selection(selected_entries(
            fs_entries, fs_times, trigger, args.delta_ticks))["waveform_indices"])
        st_count = len(describe_selection(selected_entries(
            st_entries, st_times, trigger, args.delta_ticks))["waveform_indices"])
        anomaly("warning", "rebuilt_trigger_missing_from_csv", trigger_time=trigger,
                detail=f"Candidato ricostruito assente dal CSV: FS={fs_count} waveform, ST={st_count} waveform.")

    assignments = {"FS_APA1": defaultdict(list), "ST_APA2": defaultdict(list)}
    output_rows = []
    for csv_row, trigger, original in csv_rows:
        fs = describe_selection(selected_entries(fs_entries, fs_times, trigger, args.delta_ticks))
        st = describe_selection(selected_entries(st_entries, st_times, trigger, args.delta_ticks))
        for index in fs["waveform_indices"]:
            assignments["FS_APA1"][index].append((csv_row, trigger))
        for index in st["waveform_indices"]:
            assignments["ST_APA2"][index].append((csv_row, trigger))
        if not fs["waveforms"]:
            anomaly("error", "missing_fs_waveforms", csv_row, trigger, "FS_APA1",
                    detail="Nessuna waveform FS soddisfa la finestra temporale.")
        if not st["waveforms"]:
            anomaly("error", "missing_st_waveforms", csv_row, trigger, "ST_APA2",
                    detail="Nessuna waveform ST soddisfa la finestra temporale.")
        if fs["duplicate_channels"]:
            anomaly("warning", "duplicate_fs_channel_waveforms", csv_row, trigger, "FS_APA1",
                    detail=f"{fs['duplicate_channels']} waveform oltre la prima condividono run, record, endpoint e canale.")
        if st["duplicate_channels"]:
            anomaly("warning", "duplicate_st_channel_waveforms", csv_row, trigger, "ST_APA2",
                    detail=f"{st['duplicate_channels']} waveform oltre la prima condividono run, record, endpoint e canale.")
        if len(fs["run_record"]) > 1:
            anomaly("warning", "multiple_fs_run_record_pairs", csv_row, trigger, "FS_APA1",
                    detail=f"La finestra contiene {len(fs['run_record'])} coppie (run, record) FS distinte.")
        if len(st["run_record"]) > 1:
            anomaly("warning", "multiple_st_run_record_pairs", csv_row, trigger, "ST_APA2",
                    detail=f"La finestra contiene {len(st['run_record'])} coppie (run, record) ST distinte.")

        shared_run = fs["run_numbers"] & st["run_numbers"]
        shared_record = fs["record_numbers"] & st["record_numbers"]
        shared_pair = fs["run_record"] & st["run_record"]
        shared_daq = fs["daq"] & st["daq"]
        shared_full = fs["full"] & st["full"]
        if not fs["waveforms"] or not st["waveforms"]:
            category = "missing_one_mode"
        elif shared_full:
            category = "shared_run_record_daq"
        elif shared_pair and shared_daq:
            category = "shared_run_record_and_daq_separately"
        elif shared_pair:
            category = "shared_run_record"
        elif shared_daq:
            category = "shared_daq_timestamp"
        elif shared_record:
            category = "shared_record_only"
        elif shared_run:
            category = "shared_run_only"
        else:
            category = "time_only"
        output_rows.append({
            "csv_row": csv_row, "trigger_time": trigger,
            "apa1_mean": original["apa1_mean"], "apa2_mean": original["apa2_mean"],
            "apa1_n_events": original["apa1_n_events"], "apa2_n_events": original["apa2_n_events"],
            "fs_waveforms": fs["waveforms"], "st_waveforms": st["waveforms"],
            "fs_distinct_peak_times": len(fs["peak_times"]),
            "st_distinct_peak_times": len(st["peak_times"]),
            "minimum_fs_st_peak_distance_ticks": minimum_distance(fs["peak_times"], st["peak_times"]),
            "fs_run_numbers": compact(fs["run_numbers"]), "st_run_numbers": compact(st["run_numbers"]),
            "shared_run_number": bool(shared_run),
            "fs_record_numbers": compact(fs["record_numbers"]),
            "st_record_numbers": compact(st["record_numbers"]),
            "shared_record_number": bool(shared_record),
            "fs_run_record_pairs": compact_pairs(fs["run_record"]),
            "st_run_record_pairs": compact_pairs(st["run_record"]),
            "shared_run_record_pair": bool(shared_pair),
            "fs_daq_timestamps": compact(fs["daq"]), "st_daq_timestamps": compact(st["daq"]),
            "shared_daq_timestamp": bool(shared_daq),
            "minimum_daq_timestamp_distance_ticks": minimum_distance(fs["daq"], st["daq"]),
            "fs_full_identities": compact_pairs(fs["full"]),
            "st_full_identities": compact_pairs(st["full"]),
            "shared_full_identity": bool(shared_full),
            "fs_duplicate_channel_waveforms": fs["duplicate_channels"],
            "st_duplicate_channel_waveforms": st["duplicate_channels"],
            "matching_category": category,
        })

    for mode, mapping in assignments.items():
        for index, assigned in mapping.items():
            if len(assigned) > 1:
                anomaly("warning", "waveform_assigned_to_multiple_csv_triggers",
                        mode=mode, waveform_index=index,
                        detail="Assegnazioni: " + ", ".join(
                            f"riga {row}, tempo {trigger}" for row, trigger in assigned))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "trigger_matching.csv", MATCH_FIELDS, output_rows)
    write_csv(args.output_dir / "anomalies.csv", ANOMALY_FIELDS, anomalies)
    write_csv(args.output_dir / "waveform_summary.csv", WAVEFORM_SUMMARY_FIELDS,
              [fs_summary, st_summary])
    categories = Counter(row["matching_category"] for row in output_rows)
    codes = Counter(row["code"] for row in anomalies)
    created = datetime.now(timezone.utc)
    manifest = {
        "created_utc": created.isoformat(),
        "configuration": {
            "block_dir": str(args.block_dir), "output_dir": str(args.output_dir),
            "analysis_label": args.analysis_label, "delta_ticks": args.delta_ticks,
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version,
        "inputs": [file_info(path) for path in (csv_path, fs_path, st_path)],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "CONTROLLO ASSOCIAZIONE TRIGGER APA 1–APA 2",
        f"Blocco: {args.block_dir}", f"Output: {args.output_dir}",
        f"Finestra: abs(tempo_picco - trigger_time) < {args.delta_ticks} tick.",
        f"Righe CSV analizzate: {len(csv_rows)}.",
        f"Candidati ricostruiti dai pickle: {len(rebuilt)}.",
        f"Waveform FS: {fs_summary['waveforms']}; waveform ST: {st_summary['waveforms']}.",
        "", "CATEGORIE DI ASSOCIAZIONE",
    ]
    lines += [f"{key}: {value}" for key, value in sorted(categories.items())]
    lines += ["", "ANOMALIE PER TIPO"]
    lines += [f"{key}: {value}" for key, value in sorted(codes.items())]
    if not anomalies:
        lines.append("Nessuna anomalia rilevata dai controlli implementati.")
    lines += [
        "", "INTERPRETAZIONE",
        "shared_run_record_daq: almeno una identità (run, record, timestamp DAQ) è comune.",
        "shared_run_record: run e record sono comuni, ma non la stessa identità completa.",
        "shared_run_record_and_daq_separately: coppia (run, record) e timestamp DAQ sono comuni, ma su waveform diverse.",
        "shared_daq_timestamp: è comune soltanto almeno un timestamp DAQ.",
        "shared_record_only: è comune soltanto almeno un numero di record.",
        "shared_run_only: è comune soltanto almeno un run; l'abbinamento resta temporale.",
        "time_only: nessuno dei metadati confrontati è comune; esaminare la semantica dei flussi.",
        "Le categorie descrivono i metadati osservati e non applicano esclusioni ai dati.",
        "I conteggi apa1/apa2_n_events rappresentano PE validi, non necessariamente tutte le waveform.",
    ]
    report = "\n".join(lines) + "\n"
    (args.output_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 1 if anomalies else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError,
            TypeError, ValueError, csv.Error) as exc:
        print(f"Errore: {exc}", file=sys.stderr)
        sys.exit(2)
