#!/usr/bin/env python3
r"""Prepara il dataset APA 1–APA 2 a livello di trigger per la revisione.

SCOPO
    Unisce in modo riproducibile i CSV prodotti per blocchi dal vecchio notebook,
    mantenendo la provenienza di ogni riga e un solo record per trigger e momento.
    Verifica schema, liste PE, medie, deviazioni standard e duplicati prima di
    pubblicare il dataset. Non produce figure, non esegue fit e non applica la
    selezione muonica/non muonica.

INPUT
    --input-dir: cartella apa1_vs_apa2 contenente
    <momento>GeV/<start>_to_<stop>/photoelectron_dataframe_<momento>GeV.csv.
    --momenta: momenti nominali da includere (default: 1 2 3 5 7 GeV/c).
    --output-dir: nuova cartella per gli output; non viene mai sovrascritta.
    Il blocco 2GeV/0_to_1 è escluso perché le sue 28 righe sono già presenti
    in 2GeV/0_to_10. L'esclusione è registrata negli output.

OUTPUT
    apa12_trigger_data.csv: dataset ordinato per momento e trigger;
    summary.csv: conteggi per momento e categoria di validità;
    anomalies.csv: problemi strutturali o numerici rilevati;
    report.txt: descrizione leggibile dei risultati e delle esclusioni;
    manifest.json: configurazione e SHA-256 degli input.
    Se esistono errori, il dataset principale non viene creato, per evitare di
    usare accidentalmente dati incompleti. Gli altri file diagnostici vengono
    comunque salvati. Codice di uscita: 0 = dataset creato senza anomalie;
    1 = dataset creato con avvisi; 2 = errori, dataset non creato.

ESECUZIONE (dalla cartella scripts/review su LXPlus)
    python prepare_apa12_trigger_data.py \
        --input-dir ../../output/apa1_vs_apa2 \
        --output-dir ../../output/review/apa12_trigger_data_01

INTERPRETAZIONE
    apa1_mean e apa2_mean sono le medie dei valori PE validi salvati dal vecchio
    notebook per quel trigger; non sono somme sull'intera APA. Una media mancante
    resta nel dataset come campo vuoto ed è descritta da validity_category.
    I CSV non contengono run e record DAQ: l'identità fisica delle associazioni
    è stata verificata separatamente sui pickle, non viene ricostruita qui.
"""

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


REQUIRED_FIELDS = ["trigger_time"] + [
    f"apa{apa}_{field}"
    for apa in (1, 2)
    for field in ("list", "mean", "std", "n_events")
]

DATA_FIELDS = [
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
]

SUMMARY_FIELDS = [
    "momentum_GeV_c",
    "included_blocks",
    "rows",
    "distinct_trigger_times",
    "both_valid",
    "apa1_only",
    "apa2_only",
    "neither_valid",
    "apa1_valid_total",
    "apa2_valid_total",
    "errors",
    "warnings",
]

ANOMALY_FIELDS = [
    "severity",
    "code",
    "momentum_GeV_c",
    "block",
    "source_file",
    "source_csv_row",
    "trigger_time",
    "apa",
    "related_file",
    "related_csv_row",
    "detail",
]

DEFAULT_EXCLUSIONS = {
    "2GeV/0_to_1": (
        "Blocco ridondante: le 28 righe sono identiche a righe già presenti "
        "in 2GeV/0_to_10."
    ),
}


def parse_number_list(raw):
    """Legge una lista numerica, inclusi np.float64(...), senza usare eval."""
    tree = ast.parse(raw, mode="eval").body
    if not isinstance(tree, ast.List):
        raise ValueError("Il valore non è una lista [...].")

    def number(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id.lower() in ("nan", "inf", "infinity"):
            return float(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            sign = -1 if isinstance(node.op, ast.USub) else 1
            return sign * number(node.operand)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("np", "numpy")
            and node.func.attr == "float64"
            and len(node.args) == 1
            and not node.keywords
        ):
            return number(node.args[0])
        raise ValueError("La lista contiene un elemento non numerico.")

    return [number(node) for node in tree.elts]


def exact_nonnegative_integer(raw):
    """Conserva esattamente timestamp grandi e conteggi interi."""
    value = Decimal(raw.strip())
    if not value.is_finite() or value != value.to_integral_value() or value < 0:
        raise ValueError("È richiesto un intero non negativo.")
    return int(value)


def numeric(raw):
    """Converte un campo numerico; un campo vuoto rappresenta NaN."""
    return float(raw) if raw.strip() else math.nan


def output_number(value):
    """Scrive i valori finiti con precisione riproducibile e i mancanti vuoti."""
    return format(value, ".17g") if math.isfinite(value) else ""


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def discover_blocks(input_dir, momentum, exclusions, anomalies):
    """Restituisce i blocchi inclusi in ordine numerico."""
    momentum_dir = input_dir / f"{momentum}GeV"
    if not momentum_dir.is_dir():
        anomalies.append({
            "severity": "error",
            "code": "missing_momentum_folder",
            "momentum_GeV_c": momentum,
            "block": "",
            "source_file": str(momentum_dir),
            "source_csv_row": "",
            "trigger_time": "",
            "apa": "",
            "related_file": "",
            "related_csv_row": "",
            "detail": "Cartella del momento assente.",
        })
        return []

    blocks = []
    for directory in momentum_dir.iterdir():
        if not directory.is_dir() or "_to_" not in directory.name:
            continue
        relative = f"{momentum}GeV/{directory.name}"
        if relative in exclusions:
            continue
        match = re.fullmatch(r"(\d+)_to_(\d+)", directory.name)
        if not match or int(match[1]) >= int(match[2]):
            anomalies.append({
                "severity": "error",
                "code": "invalid_block_name",
                "momentum_GeV_c": momentum,
                "block": directory.name,
                "source_file": str(directory),
                "source_csv_row": "",
                "trigger_time": "",
                "apa": "",
                "related_file": "",
                "related_csv_row": "",
                "detail": "Atteso <start>_to_<stop> con start < stop.",
            })
            continue
        blocks.append((int(match[1]), int(match[2]), directory))
    return [item[2] for item in sorted(blocks)]


def prepare(args):
    rows = []
    summaries = []
    anomalies = []
    inputs = []
    exclusions = dict(DEFAULT_EXCLUSIONS)

    def issue(severity, code, momentum, block="", source="", csv_row="",
              trigger="", apa="", related_file="", related_row="", detail=""):
        anomalies.append(dict(zip(ANOMALY_FIELDS, [
            severity,
            code,
            momentum,
            block,
            str(source),
            csv_row,
            str(trigger),
            apa,
            str(related_file),
            related_row,
            detail,
        ])))

    for momentum in args.momenta:
        blocks = discover_blocks(args.input_dir, momentum, exclusions, anomalies)
        seen_triggers = {}
        momentum_rows = []

        if not blocks:
            issue(
                "error",
                "no_included_blocks",
                momentum,
                source=args.input_dir / f"{momentum}GeV",
                detail="Nessun blocco valido disponibile dopo le esclusioni.",
            )

        for block_dir in blocks:
            block = block_dir.name
            source = block_dir / f"photoelectron_dataframe_{momentum}GeV.csv"
            if not source.is_file():
                issue("error", "missing_csv", momentum, block, source,
                      detail="CSV atteso assente.")
                continue

            try:
                raw_bytes = source.read_bytes()
                text = raw_bytes.decode("utf-8-sig")
            except (OSError, UnicodeError) as exc:
                issue("error", "unreadable_csv", momentum, block, source,
                      detail=str(exc))
                continue

            inputs.append({
                "momentum_GeV_c": momentum,
                "block": block,
                "path": str(source),
                "bytes": len(raw_bytes),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            })

            reader = csv.DictReader(io.StringIO(text), strict=True)
            headers = reader.fieldnames or []
            missing = sorted(set(REQUIRED_FIELDS) - set(headers))
            duplicated_headers = len(headers) != len(set(headers))
            if missing or duplicated_headers:
                issue(
                    "error",
                    "invalid_columns",
                    momentum,
                    block,
                    source,
                    csv_row=1,
                    detail=(
                        f"Colonne mancanti: {missing}; "
                        f"intestazioni duplicate: {duplicated_headers}."
                    ),
                )
                continue

            try:
                for record in reader:
                    csv_row = reader.line_num
                    raw_trigger = record.get("trigger_time") or ""
                    if None in record or any(value is None for value in record.values()):
                        issue("error", "malformed_row", momentum, block, source,
                              csv_row, raw_trigger,
                              detail="Numero di campi diverso dall'intestazione.")
                        continue

                    try:
                        trigger = exact_nonnegative_integer(raw_trigger)
                    except (ValueError, InvalidOperation) as exc:
                        issue("error", "invalid_trigger_time", momentum, block,
                              source, csv_row, raw_trigger, detail=str(exc))
                        continue

                    if trigger in seen_triggers:
                        previous_file, previous_row = seen_triggers[trigger]
                        issue(
                            "error",
                            "repeated_trigger_time",
                            momentum,
                            block,
                            source,
                            csv_row,
                            trigger,
                            related_file=previous_file,
                            related_row=previous_row,
                            detail=(
                                "Timestamp ripetuto nei blocchi inclusi. Il dataset "
                                "univoco non può essere pubblicato."
                            ),
                        )
                        continue
                    seen_triggers[trigger] = (source, csv_row)

                    apa_data = {}
                    row_is_valid = True
                    for apa in (1, 2):
                        prefix = f"apa{apa}_"
                        try:
                            values = parse_number_list(record[prefix + "list"])
                            mean = numeric(record[prefix + "mean"])
                            std = numeric(record[prefix + "std"])
                            count = exact_nonnegative_integer(record[prefix + "n_events"])
                        except (
                            ValueError,
                            SyntaxError,
                            InvalidOperation,
                            OverflowError,
                            RecursionError,
                        ) as exc:
                            issue("error", "invalid_apa_fields", momentum, block,
                                  source, csv_row, trigger, apa, detail=str(exc))
                            row_is_valid = False
                            continue

                        if count != len(values):
                            issue(
                                "error",
                                "count_mismatch",
                                momentum,
                                block,
                                source,
                                csv_row,
                                trigger,
                                apa,
                                detail=f"n_events={count}, lunghezza lista={len(values)}.",
                            )
                            row_is_valid = False
                            continue

                        if not values:
                            if not (math.isnan(mean) and math.isnan(std)):
                                issue(
                                    "error",
                                    "empty_list_statistics",
                                    momentum,
                                    block,
                                    source,
                                    csv_row,
                                    trigger,
                                    apa,
                                    detail="Lista vuota, ma media o deviazione standard non è NaN.",
                                )
                                row_is_valid = False
                                continue
                            apa_data[apa] = {"mean": math.nan, "std": math.nan,
                                             "count": 0, "valid": False}
                            continue

                        if not all(math.isfinite(value) for value in values):
                            issue("error", "nonfinite_pe_list", momentum, block,
                                  source, csv_row, trigger, apa,
                                  detail="La lista contiene NaN o infinito.")
                            row_is_valid = False
                            continue

                        expected_mean = statistics.mean(values)
                        expected_std = statistics.pstdev(values)
                        if (
                            not math.isfinite(mean)
                            or not math.isclose(mean, expected_mean,
                                                rel_tol=args.rtol, abs_tol=args.atol)
                        ):
                            issue(
                                "error",
                                "mean_mismatch",
                                momentum,
                                block,
                                source,
                                csv_row,
                                trigger,
                                apa,
                                detail=f"Salvata={mean!r}, ricalcolata={expected_mean!r}.",
                            )
                            row_is_valid = False
                        if (
                            not math.isfinite(std)
                            or not math.isclose(std, expected_std,
                                                rel_tol=args.rtol, abs_tol=args.atol)
                        ):
                            issue(
                                "error",
                                "std_mismatch",
                                momentum,
                                block,
                                source,
                                csv_row,
                                trigger,
                                apa,
                                detail=f"Salvata={std!r}, ricalcolata={expected_std!r}; ddof=0.",
                            )
                            row_is_valid = False
                        if any(value < 0 for value in values):
                            issue(
                                "warning",
                                "negative_pe",
                                momentum,
                                block,
                                source,
                                csv_row,
                                trigger,
                                apa,
                                detail="Lista con almeno un valore PE negativo; dato conservato.",
                            )
                        apa_data[apa] = {
                            "mean": mean,
                            "std": std,
                            "count": count,
                            "valid": math.isfinite(mean),
                        }

                    if not row_is_valid or len(apa_data) != 2:
                        continue

                    apa1_valid = apa_data[1]["valid"]
                    apa2_valid = apa_data[2]["valid"]
                    if apa1_valid and apa2_valid:
                        category = "both_valid"
                    elif apa1_valid:
                        category = "apa1_only"
                    elif apa2_valid:
                        category = "apa2_only"
                    else:
                        category = "neither_valid"

                    momentum_rows.append({
                        "momentum_GeV_c": momentum,
                        "block": block,
                        "source_file": str(source),
                        "source_csv_row": csv_row,
                        "trigger_time": trigger,
                        "apa1_mean": output_number(apa_data[1]["mean"]),
                        "apa1_std": output_number(apa_data[1]["std"]),
                        "apa1_n_events": apa_data[1]["count"],
                        "apa1_valid": int(apa1_valid),
                        "apa2_mean": output_number(apa_data[2]["mean"]),
                        "apa2_std": output_number(apa_data[2]["std"]),
                        "apa2_n_events": apa_data[2]["count"],
                        "apa2_valid": int(apa2_valid),
                        "both_apa_valid": int(apa1_valid and apa2_valid),
                        "validity_category": category,
                    })
            except csv.Error as exc:
                issue("error", "csv_parse_error", momentum, block, source,
                      reader.line_num, detail=str(exc))

        momentum_rows.sort(key=lambda row: row["trigger_time"])
        rows.extend(momentum_rows)
        categories = Counter(row["validity_category"] for row in momentum_rows)
        momentum_anomalies = [
            anomaly for anomaly in anomalies
            if anomaly["momentum_GeV_c"] == momentum
        ]
        summaries.append({
            "momentum_GeV_c": momentum,
            "included_blocks": len(blocks),
            "rows": len(momentum_rows),
            "distinct_trigger_times": len({row["trigger_time"] for row in momentum_rows}),
            "both_valid": categories["both_valid"],
            "apa1_only": categories["apa1_only"],
            "apa2_only": categories["apa2_only"],
            "neither_valid": categories["neither_valid"],
            "apa1_valid_total": sum(row["apa1_valid"] for row in momentum_rows),
            "apa2_valid_total": sum(row["apa2_valid"] for row in momentum_rows),
            "errors": sum(item["severity"] == "error" for item in momentum_anomalies),
            "warnings": sum(item["severity"] == "warning" for item in momentum_anomalies),
        })

    rows.sort(key=lambda row: (row["momentum_GeV_c"], row["trigger_time"]))
    exclusion_records = []
    for relative, reason in exclusions.items():
        path = args.input_dir / relative
        exclusion_records.append({
            "block": relative,
            "path": str(path),
            "present": path.is_dir(),
            "reason": reason,
        })
    return rows, summaries, anomalies, inputs, exclusion_records


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--momenta",
        nargs="+",
        type=int,
        choices=(1, 2, 3, 5, 7),
        default=[1, 2, 3, 5, 7],
    )
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-9)
    args = parser.parse_args()
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.momenta = sorted(set(args.momenta))

    if not args.input_dir.is_dir():
        parser.error(f"Cartella input inesistente: {args.input_dir}")
    if args.output_dir.exists():
        parser.error(f"La cartella output esiste già: {args.output_dir}")
    if any(not math.isfinite(value) or value < 0 for value in (args.rtol, args.atol)):
        parser.error("rtol e atol devono essere finiti e non negativi.")

    created = datetime.now(timezone.utc)
    rows, summaries, anomalies, inputs, exclusions = prepare(args)
    errors = sum(item["severity"] == "error" for item in anomalies)
    warnings = sum(item["severity"] == "warning" for item in anomalies)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "summary.csv", SUMMARY_FIELDS, summaries)
    write_csv(args.output_dir / "anomalies.csv", ANOMALY_FIELDS, anomalies)
    if not errors:
        write_csv(args.output_dir / "apa12_trigger_data.csv", DATA_FIELDS, rows)

    manifest = {
        "created_utc": created.isoformat(),
        "configuration": {
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "momenta": args.momenta,
            "rtol": args.rtol,
            "atol": args.atol,
            "standard_deviation_ddof": 0,
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version,
        "excluded_blocks": exclusions,
        "inputs": inputs,
        "dataset_created": not errors,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "PREPARAZIONE DATASET APA 1–APA 2",
        f"Input: {args.input_dir}",
        f"Output: {args.output_dir}",
        f"Momenti nominali (GeV/c): {args.momenta}",
        f"Tolleranze: rtol={args.rtol}, atol={args.atol}; std con ddof=0.",
        "",
        "BLOCCHI ESCLUSI",
    ]
    for item in exclusions:
        presence = "presente" if item["present"] else "non presente"
        lines.append(f"{item['block']} ({presence}) — {item['reason']}")
    lines += ["", "RISULTATI PER MOMENTO"]
    for summary in summaries:
        lines.append(
            f"{summary['momentum_GeV_c']} GeV/c: {summary['rows']} righe; "
            f"both_valid={summary['both_valid']}, "
            f"apa1_only={summary['apa1_only']}, "
            f"apa2_only={summary['apa2_only']}, "
            f"neither_valid={summary['neither_valid']}; "
            f"{summary['errors']} errori, {summary['warnings']} avvisi."
        )
    lines += ["", "ANOMALIE PER TIPO"]
    counts = Counter(item["code"] for item in anomalies)
    if counts:
        lines.extend(f"{code}: {count}" for code, count in sorted(counts.items()))
    else:
        lines.append("Nessuna anomalia rilevata.")
    lines += [
        "",
        "OUTPUT PRINCIPALE",
        (
            "apa12_trigger_data.csv creato."
            if not errors
            else "apa12_trigger_data.csv NON creato a causa degli errori sopra elencati."
        ),
        "Le righe sono ordinate per momento e trigger_time.",
        "I campi apa1_mean e apa2_mean mancanti sono vuoti, non sostituiti con zero.",
        "validity_category distingue both_valid, apa1_only, apa2_only e neither_valid.",
        "",
        "LIMITI",
        "Il dataset contiene soltanto i trigger già associati nei CSV storici.",
        "Non recupera trigger FS privi di una waveform ST, perché non erano stati salvati nei CSV.",
        "Non applica selezioni di particella e non determina soglie.",
        "L'identità DAQ dell'associazione APA 1–APA 2 è stata controllata separatamente sui pickle.",
    ]
    report = "\n".join(lines) + "\n"
    (args.output_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)

    if errors:
        return 2
    return 1 if warnings else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OSError as exc:
        print(f"Errore di accesso ai file: {exc}", file=sys.stderr)
        sys.exit(2)
