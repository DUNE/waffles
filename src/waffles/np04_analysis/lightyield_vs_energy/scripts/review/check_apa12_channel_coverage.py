#!/usr/bin/env python3
r"""Controllo dei canali che contribuiscono alle medie PE di APA 1 e APA 2.

SCOPO
    Confronta il JSON per canale prodotto dal vecchio notebook con il CSV dei
    fotoelettroni e con trigger_matching.csv. Per ogni trigger e APA misura:
    waveform selezionate, valori PE validi, valori mancanti, frazione valida e
    identità dei canali che contribuiscono alla media. Non ricalcola i PE, non
    applica tagli fisici e non modifica gli input.

INPUT
    --block-dir: cartella come ../../output/apa1_vs_apa2/1GeV/0_to_10,
    contenente photoelectron_dic_1GeV.json e
    photoelectron_dataframe_1GeV.csv.
    --matching-csv: trigger_matching.csv prodotto da
    check_apa12_trigger_matching.py per lo stesso blocco.
    --output-dir: nuova cartella per i risultati; non viene sovrascritta.

OUTPUT
    trigger_channel_coverage.csv: due righe per trigger, una per APA;
    channel_summary.csv: frequenza con cui ogni canale fornisce un PE valido;
    block_summary.csv: una riga con statistiche descrittive adatte a una futura
    tabella comparativa per energia;
    anomalies.csv: incoerenze tra JSON, CSV, matching e mappa dei canali;
    report.txt: sintesi leggibile e limiti dell'interpretazione;
    manifest.json: configurazione e SHA-256 dei tre file letti.
    Codice di uscita: 0 = nessuna anomalia; 1 = anomalie da esaminare;
    2 = errore di esecuzione.

ESECUZIONE (dalla cartella scripts/review su LXPlus)
    python check_apa12_channel_coverage.py \
        --block-dir ../../output/apa1_vs_apa2/1GeV/0_to_10 \
        --matching-csv ../../output/review/check_trigger_matching_1GeV_0_to_10_01/trigger_matching.csv \
        --output-dir ../../output/review/check_channel_coverage_1GeV_0_to_10_01

LIMITI
    Il JSON conserva una sola voce per coppia endpoint-canale. Se nello stesso
    trigger esistessero waveform duplicate dello stesso canale, il vecchio
    notebook avrebbe sovrascritto le voci precedenti. Il programma segnala la
    differenza tra n_events e numero di canali nel JSON; il controllo delle
    waveform duplicate resta responsabilità di check_apa12_trigger_matching.py.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


TRIGGER_FIELDS = [
    "csv_row", "trigger_time", "apa", "acquisition_mode", "selected_waveforms",
    "valid_pe_values", "missing_or_failed_pe", "valid_pe_fraction",
    "channels_with_valid_pe", "distinct_endpoints", "channel_ids",
    "mean_pe", "std_pe", "minimum_r_squared", "median_r_squared",
    "maximum_r_squared", "coverage_status",
]
CHANNEL_FIELDS = [
    "apa", "acquisition_mode", "endpoint", "channel", "in_expected_apa_map",
    "triggers_with_valid_pe", "fraction_of_csv_triggers_with_valid_pe",
    "r_squared_values", "minimum_r_squared", "median_r_squared",
    "mean_r_squared", "maximum_r_squared",
]
ANOMALY_FIELDS = [
    "severity", "code", "csv_row", "trigger_time", "apa", "endpoint",
    "channel", "detail",
]

# Copia esplicita delle coppie endpoint-canale definite in
# waffles/np04_data/ProtoDUNE_HD_APA_maps.py al momento della revisione.
# Serve soltanto a verificare l'identità dei canali letti dal JSON.
EXPECTED_APA_CHANNELS = {
    1: (
        {(104, channel) for channel in (*range(0, 8), *range(10, 18))}
        | {(105, channel) for channel in (*range(0, 8), 10, 12, 15, 17, 21, 23, 24, 26)}
        | {(107, channel) for channel in (0, 2, 5, 7, 10, 12, 15, 17)}
    ),
    2: {(109, channel) for decade in (0, 10, 20, 30, 40) for channel in range(decade, decade + 8)},
}


def exact_integer(value):
    number = Decimal(str(value).strip())
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError("È richiesto un intero finito.")
    return int(number)


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("È richiesto un numero finito.")
    return number


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


def read_csv_by_trigger(path, required):
    rows = {}
    duplicates = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        missing = sorted(set(required) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Colonne mancanti in {path}: {missing}")
        for row in reader:
            trigger = exact_integer(row["trigger_time"])
            row["_physical_row"] = reader.line_num
            if trigger in rows:
                duplicates.add(trigger)
            else:
                rows[trigger] = row
    if duplicates:
        raise ValueError(f"Tempi trigger duplicati in {path}: {sorted(duplicates)[:10]}")
    return rows


def expected_channels(apa):
    return set(EXPECTED_APA_CHANNELS[apa])


def describe(values, prefix):
    values = list(values)
    if not values:
        return {
            f"{prefix}_mean": "", f"{prefix}_population_std": "",
            f"{prefix}_median": "", f"{prefix}_minimum": "",
            f"{prefix}_maximum": "",
        }
    return {
        f"{prefix}_mean": statistics.mean(values),
        f"{prefix}_population_std": statistics.pstdev(values),
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_minimum": min(values),
        f"{prefix}_maximum": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--block-dir", type=Path, required=True)
    parser.add_argument("--matching-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.block_dir = args.block_dir.expanduser().resolve()
    args.matching_csv = args.matching_csv.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.block_dir.is_dir():
        parser.error(f"Cartella del blocco inesistente: {args.block_dir}")
    if not args.matching_csv.is_file():
        parser.error(f"CSV del matching inesistente: {args.matching_csv}")
    if args.output_dir.exists():
        parser.error(f"La cartella output esiste già: {args.output_dir}")
    energy_folder = args.block_dir.parent.name
    if not re.fullmatch(r"(?:1|2|3|5|7)GeV", energy_folder):
        parser.error(f"Impossibile ricavare il momento da {energy_folder!r}.")
    momentum = int(energy_folder[:-3])
    json_path = args.block_dir / f"photoelectron_dic_{energy_folder}.json"
    dataframe_path = args.block_dir / f"photoelectron_dataframe_{energy_folder}.csv"
    for path in (json_path, dataframe_path):
        if not path.is_file():
            parser.error(f"Input richiesto assente: {path}")

    dataframe = read_csv_by_trigger(dataframe_path, [
        "trigger_time", "apa1_mean", "apa1_std", "apa1_n_events",
        "apa2_mean", "apa2_std", "apa2_n_events",
    ])
    matching = read_csv_by_trigger(args.matching_csv, [
        "trigger_time", "csv_row", "fs_waveforms", "st_waveforms",
        "matching_category", "shared_full_identity",
    ])
    with json_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    anomalies = []

    def anomaly(severity, code, csv_row="", trigger_time="", apa="",
                endpoint="", channel="", detail=""):
        anomalies.append(dict(zip(ANOMALY_FIELDS, [
            severity, code, csv_row, trigger_time, apa, endpoint, channel, detail,
        ])))

    try:
        json_times = [exact_integer(value) for value in data["trigger time"]]
    except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
        raise ValueError(f"Campo JSON 'trigger time' non valido: {exc}") from exc
    duplicate_json_times = [value for value, count in Counter(json_times).items() if count > 1]
    if duplicate_json_times:
        raise ValueError(f"Tempi duplicati nel JSON: {duplicate_json_times[:10]}")
    if set(json_times) != set(dataframe):
        anomaly("error", "json_csv_trigger_set_mismatch",
                detail=f"Solo JSON: {len(set(json_times)-set(dataframe))}; solo CSV: {len(set(dataframe)-set(json_times))}.")
    if set(dataframe) != set(matching):
        anomaly("error", "csv_matching_trigger_set_mismatch",
                detail=f"Solo CSV dati: {len(set(dataframe)-set(matching))}; solo matching: {len(set(matching)-set(dataframe))}.")

    expected = {1: expected_channels(1), 2: expected_channels(2)}
    modes = {1: ("FS_APA1", "fs_waveforms"), 2: ("ST_APA2", "st_waveforms")}
    json_arrays = {}
    for apa in (1, 2):
        try:
            section = data[str(apa)]
            arrays = {key: section[key] for key in ("channel_dic", "list", "mean", "std", "n_events")}
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Struttura JSON APA {apa} non valida: {exc}") from exc
        lengths = {key: len(value) for key, value in arrays.items()}
        if any(length != len(json_times) for length in lengths.values()):
            raise ValueError(f"Lunghezze JSON APA {apa} incoerenti: trigger={len(json_times)}, {lengths}")
        json_arrays[apa] = arrays

    trigger_rows = []
    channel_occurrences = {1: defaultdict(int), 2: defaultdict(int)}
    channel_r2 = {1: defaultdict(list), 2: defaultdict(list)}
    per_apa_valid = {1: [], 2: []}
    per_apa_fraction = {1: [], 2: []}
    per_mode_waveforms = {1: [], 2: []}
    zero_valid = Counter()

    for index, trigger in enumerate(json_times):
        csv_data = dataframe.get(trigger)
        match = matching.get(trigger)
        if csv_data is None or match is None:
            continue
        csv_row = csv_data["_physical_row"]
        if match["matching_category"] != "shared_run_record_daq":
            anomaly("warning", "matching_not_full_identity", csv_row, trigger,
                    detail=f"Categoria: {match['matching_category']}.")
        for apa in (1, 2):
            mode, waveform_field = modes[apa]
            arrays = json_arrays[apa]
            selected = exact_integer(match[waveform_field])
            valid = exact_integer(arrays["n_events"][index])
            csv_valid = exact_integer(csv_data[f"apa{apa}_n_events"])
            if valid != csv_valid:
                anomaly("error", "json_csv_n_events_mismatch", csv_row, trigger, apa,
                        detail=f"JSON={valid}, CSV={csv_valid}.")
            pe_list = arrays["list"][index]
            if not isinstance(pe_list, list) or len(pe_list) != valid:
                anomaly("error", "json_pe_list_count_mismatch", csv_row, trigger, apa,
                        detail=f"n_events={valid}, lunghezza lista={len(pe_list) if isinstance(pe_list, list) else 'non-lista'}.")
            channel_dict = arrays["channel_dic"][index]
            if not isinstance(channel_dict, dict):
                anomaly("error", "invalid_channel_dictionary", csv_row, trigger, apa,
                        detail="channel_dic non è un dizionario.")
                channel_dict = {}
            channel_ids = []
            r_squared = []
            for endpoint_raw, channels in channel_dict.items():
                try:
                    endpoint = exact_integer(endpoint_raw)
                except (ValueError, InvalidOperation):
                    anomaly("error", "invalid_endpoint", csv_row, trigger, apa,
                            endpoint=endpoint_raw, detail="Endpoint non intero.")
                    continue
                if not isinstance(channels, dict):
                    anomaly("error", "invalid_endpoint_dictionary", csv_row, trigger, apa,
                            endpoint=endpoint, detail="Il contenuto dell'endpoint non è un dizionario.")
                    continue
                for channel_raw, result in channels.items():
                    try:
                        channel = exact_integer(channel_raw)
                    except (ValueError, InvalidOperation):
                        anomaly("error", "invalid_channel", csv_row, trigger, apa,
                                endpoint=endpoint, channel=channel_raw, detail="Canale non intero.")
                        continue
                    key = (endpoint, channel)
                    channel_ids.append(key)
                    channel_occurrences[apa][key] += 1
                    if key not in expected[apa]:
                        anomaly("error", "channel_outside_apa_map", csv_row, trigger, apa,
                                endpoint, channel, "Il canale non appartiene alla mappa dell'APA.")
                    if isinstance(result, dict) and result.get("r_squared") is not None:
                        try:
                            r2 = finite_float(result["r_squared"])
                        except (TypeError, ValueError):
                            anomaly("error", "invalid_r_squared", csv_row, trigger, apa,
                                    endpoint, channel, f"Valore: {result.get('r_squared')!r}.")
                        else:
                            r_squared.append(r2)
                            channel_r2[apa][key].append(r2)
            if len(channel_ids) != valid:
                anomaly("warning", "channel_count_n_events_mismatch", csv_row, trigger, apa,
                        detail=f"Canali nel JSON={len(channel_ids)}, n_events={valid}; possibile sovrascrittura o struttura incompleta.")
            if valid > selected:
                anomaly("error", "valid_pe_exceeds_selected_waveforms", csv_row, trigger, apa,
                        detail=f"PE validi={valid}, waveform selezionate={selected}.")
            missing = selected - valid
            fraction = valid / selected if selected else ""
            status = "none" if valid == 0 else ("complete" if valid == selected else "partial")
            if valid == 0:
                zero_valid[apa] += 1
            per_mode_waveforms[apa].append(selected)
            per_apa_valid[apa].append(valid)
            if selected:
                per_apa_fraction[apa].append(valid / selected)
            trigger_rows.append({
                "csv_row": csv_row, "trigger_time": trigger, "apa": apa,
                "acquisition_mode": mode, "selected_waveforms": selected,
                "valid_pe_values": valid, "missing_or_failed_pe": missing,
                "valid_pe_fraction": fraction,
                "channels_with_valid_pe": len(channel_ids),
                "distinct_endpoints": len({endpoint for endpoint, _ in channel_ids}),
                "channel_ids": ";".join(f"{endpoint}:{channel}" for endpoint, channel in sorted(channel_ids)),
                "mean_pe": arrays["mean"][index], "std_pe": arrays["std"][index],
                "minimum_r_squared": min(r_squared) if r_squared else "",
                "median_r_squared": statistics.median(r_squared) if r_squared else "",
                "maximum_r_squared": max(r_squared) if r_squared else "",
                "coverage_status": status,
            })

    channel_rows = []
    for apa in (1, 2):
        mode, _ = modes[apa]
        all_channels = expected[apa] | set(channel_occurrences[apa])
        for endpoint, channel in sorted(all_channels):
            occurrences = channel_occurrences[apa][(endpoint, channel)]
            r2_values = channel_r2[apa][(endpoint, channel)]
            channel_rows.append({
                "apa": apa, "acquisition_mode": mode, "endpoint": endpoint,
                "channel": channel, "in_expected_apa_map": (endpoint, channel) in expected[apa],
                "triggers_with_valid_pe": occurrences,
                "fraction_of_csv_triggers_with_valid_pe": occurrences / len(dataframe) if dataframe else "",
                "r_squared_values": len(r2_values),
                "minimum_r_squared": min(r2_values) if r2_values else "",
                "median_r_squared": statistics.median(r2_values) if r2_values else "",
                "mean_r_squared": statistics.mean(r2_values) if r2_values else "",
                "maximum_r_squared": max(r2_values) if r2_values else "",
            })

    category_counts = Counter(row["matching_category"] for row in matching.values())
    block_summary = {
        "momentum_GeV_c": momentum, "block": args.block_dir.name,
        "csv_triggers": len(dataframe), "json_triggers": len(json_times),
        "matching_triggers": len(matching),
        "triggers_with_shared_full_identity": category_counts["shared_run_record_daq"],
        "apa1_triggers_with_zero_valid_pe": zero_valid[1],
        "apa2_triggers_with_zero_valid_pe": zero_valid[2],
        "apa1_expected_channels": len(expected[1]),
        "apa2_expected_channels": len(expected[2]),
        "apa1_channels_observed_with_valid_pe": len(channel_occurrences[1]),
        "apa2_channels_observed_with_valid_pe": len(channel_occurrences[2]),
    }
    block_summary.update(describe(per_mode_waveforms[1], "apa1_selected_waveforms_per_trigger"))
    block_summary.update(describe(per_mode_waveforms[2], "apa2_selected_waveforms_per_trigger"))
    block_summary.update(describe(per_apa_valid[1], "apa1_valid_pe_per_trigger"))
    block_summary.update(describe(per_apa_valid[2], "apa2_valid_pe_per_trigger"))
    block_summary.update(describe(per_apa_fraction[1], "apa1_valid_pe_fraction_per_trigger"))
    block_summary.update(describe(per_apa_fraction[2], "apa2_valid_pe_fraction_per_trigger"))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "trigger_channel_coverage.csv", TRIGGER_FIELDS, trigger_rows)
    write_csv(args.output_dir / "channel_summary.csv", CHANNEL_FIELDS, channel_rows)
    write_csv(args.output_dir / "block_summary.csv", list(block_summary), [block_summary])
    write_csv(args.output_dir / "anomalies.csv", ANOMALY_FIELDS, anomalies)
    created = datetime.now(timezone.utc)
    manifest = {
        "created_utc": created.isoformat(),
        "configuration": {
            "block_dir": str(args.block_dir), "matching_csv": str(args.matching_csv),
            "output_dir": str(args.output_dir),
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version,
        "inputs": [file_info(path) for path in (json_path, dataframe_path, args.matching_csv)],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    codes = Counter(row["code"] for row in anomalies)
    lines = [
        "CONTROLLO COPERTURA CANALI APA 1–APA 2",
        f"Blocco: {args.block_dir}", f"Output: {args.output_dir}",
        f"Trigger nel CSV: {len(dataframe)}; nel JSON: {len(json_times)}; nel matching: {len(matching)}.",
        "", "APA 1 — FULL-STREAMING",
        f"Waveform selezionate per trigger: media={block_summary['apa1_selected_waveforms_per_trigger_mean']:.3f}, "
        f"mediana={block_summary['apa1_selected_waveforms_per_trigger_median']}, "
        f"intervallo=[{block_summary['apa1_selected_waveforms_per_trigger_minimum']}, {block_summary['apa1_selected_waveforms_per_trigger_maximum']}].",
        f"PE validi per trigger: media={block_summary['apa1_valid_pe_per_trigger_mean']:.3f}, "
        f"mediana={block_summary['apa1_valid_pe_per_trigger_median']}, "
        f"intervallo=[{block_summary['apa1_valid_pe_per_trigger_minimum']}, {block_summary['apa1_valid_pe_per_trigger_maximum']}].",
        f"Frazione valida media={block_summary['apa1_valid_pe_fraction_per_trigger_mean']:.4f}; "
        f"trigger senza PE validi={zero_valid[1]}.",
        "", "APA 2 — SELF-TRIGGER",
        f"Waveform selezionate per trigger: media={block_summary['apa2_selected_waveforms_per_trigger_mean']:.3f}, "
        f"mediana={block_summary['apa2_selected_waveforms_per_trigger_median']}, "
        f"intervallo=[{block_summary['apa2_selected_waveforms_per_trigger_minimum']}, {block_summary['apa2_selected_waveforms_per_trigger_maximum']}].",
        f"PE validi per trigger: media={block_summary['apa2_valid_pe_per_trigger_mean']:.3f}, "
        f"mediana={block_summary['apa2_valid_pe_per_trigger_median']}, "
        f"intervallo=[{block_summary['apa2_valid_pe_per_trigger_minimum']}, {block_summary['apa2_valid_pe_per_trigger_maximum']}].",
        f"Frazione valida media={block_summary['apa2_valid_pe_fraction_per_trigger_mean']:.4f}; "
        f"trigger senza PE validi={zero_valid[2]}.",
        "", "ANOMALIE PER TIPO",
    ]
    lines += [f"{code}: {count}" for code, count in sorted(codes.items())]
    if not anomalies:
        lines.append("Nessuna anomalia rilevata dai controlli implementati.")
    lines += [
        "", "INTERPRETAZIONE",
        "selected_waveforms conta le waveform associate temporalmente al trigger.",
        "valid_pe_values conta le waveform che hanno prodotto un PE non NaN nel vecchio notebook.",
        "Per APA 2, la frazione è condizionata ai canali che hanno fatto self-trigger; non è l'efficienza sui 40 canali dell'APA.",
        "Le statistiche descrivono soltanto i trigger già presenti nel CSV, cioè la coincidenza FS–ST.",
        "channel_summary.csv misura la stabilità dell'insieme di canali usato nelle medie.",
        "La deviazione standard del riepilogo è quella della popolazione analizzata (ddof=0).",
    ]
    report = "\n".join(lines) + "\n"
    (args.output_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 1 if anomalies else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError,
            csv.Error, InvalidOperation) as exc:
        print(f"Errore: {exc}", file=sys.stderr)
        sys.exit(2)
