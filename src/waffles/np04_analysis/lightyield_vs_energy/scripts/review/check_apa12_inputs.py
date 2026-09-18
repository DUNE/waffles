#!/usr/bin/env python3
r"""Controllo degli input APA 1–APA 2, senza modificare i dati originali.

SCOPO
    Inventaria i CSV per blocco, verifica colonne e statistiche delle liste di
    fotoelettroni e segnala tempi ripetuti/vicini e blocchi sovrapposti.
    Non elimina righe, non esegue fit e non seleziona muoni o non muoni.
    I soli CSV NON permettono di certificare l'identità del trigger DAQ,
    il numero di canali distinti o la completezza dei dati FS/ST.

INPUT
    --input-dir: cartella apa1_vs_apa2 contenente, per ciascun momento p,
    <p>GeV/<start>_to_<stop>/photoelectron_dataframe_<p>GeV.csv.
    Colonne richieste: trigger_time e, per apa1/apa2, list, mean, std, n_events.
    Le liste possono contenere numeri semplici o np.float64(...)/numpy.float64(...).
    [] e liste con nan/inf vengono gestite, anche dentro questi involucri.
    Gli involucri sono letti come testo: non viene eseguito codice del CSV.
    files_read.txt è facoltativo: viene controllato se presente, ma non è
    considerato una prova che tutti i file siano stati letti correttamente.
    --momenta: momenti nominali in GeV/c (default: 1 2 3 5 7).
    --exclude-block BLOCCO MOTIVO: esclude esplicitamente una cartella relativa
    a --input-dir, ad esempio 2GeV/0_to_1, con una motivazione obbligatoria.
    L'opzione è ripetibile; la cartella deve esistere e appartenere ai momenti
    richiesti. Senza questa opzione non viene escluso nessun blocco.
    Non occorrono pandas, scipy, waffles o i dati grezzi: solo Python >= 3.9.

OUTPUT
    In una NUOVA cartella --output-dir (mai sovrascritta):
      summary.csv: conteggi e anomalie per blocco e per momento;
      anomalies.csv: dettagli, file sorgente e riga CSV (intestazione = 1);
      report.txt: spiegazione dei risultati, impostazioni e limiti;
      manifest.json: percorsi, SHA-256 degli input letti e configurazione.
    Rapporto e manifest registrano i blocchi esclusi e le motivazioni;
    summary.csv e anomalies.csv riguardano soltanto i blocchi inclusi.
    Nessuna figura: questo primo passo controlla soltanto gli input.
    Senza --output-dir crea output/review/check_apa12_inputs_<data UTC>.
    Codice di uscita: 0 = nessuna anomalia rilevata; 1 = anomalie da esaminare;
    2 = errore di esecuzione. Anche il codice 0 NON certifica i trigger DAQ.

ESECUZIONE (dalla cartella scripts/review su LXPlus)
    python check_apa12_inputs.py \
        --input-dir ../../output/apa1_vs_apa2 \
        --output-dir ../../output/review/check_inputs_01

    Per una prima prova su un solo momento aggiungere: --momenta 1
    Per escludere il blocco ridondante a 2 GeV/c aggiungere:
        --momenta 2 --exclude-block 2GeV/0_to_1 "28 righe identiche gia presenti in 0_to_10"
    std viene ricalcolata con ddof=0; confronti numerici: rtol=1e-9, atol=1e-9.
    start/stop sono interpretati come intervalli [start, stop), come nel vecchio
    lettore. Tempi vicini: distanza <= 200 tick, solo segnalazione diagnostica.
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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


REQUIRED = ["trigger_time"] + [
    f"apa{apa}_{field}" for apa in (1, 2)
    for field in ("list", "mean", "std", "n_events")
]
SUMMARY_FIELDS = [
    "scope", "momentum_GeV_c", "block", "source_file", "status", "rows",
    "valid_trigger_times", "distinct_trigger_times", "rows_repeated_trigger_time",
    "rows_with_two_finite_means", "apa1_empty_lists", "apa2_empty_lists",
    "errors", "warnings",
]
ANOMALY_FIELDS = [
    "severity", "code", "momentum_GeV_c", "block", "source_file", "csv_row",
    "trigger_time", "apa", "related_file", "related_csv_row", "detail",
]


def parse_number_list(raw):
    """Accetta solo liste numeriche, senza eval né esecuzione di codice."""
    tree = ast.parse(raw, mode="eval").body
    if not isinstance(tree, ast.List):
        raise ValueError("Il valore non è una lista [...].")

    def number(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id.lower() in ("nan", "inf", "infinity"):
            return float(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return (-1 if isinstance(node.op, ast.USub) else 1) * number(node.operand)
        # Rappresentazione presente nei CSV storici: leggiamo solo l'argomento
        # numerico di float64, senza importare NumPy o chiamare la funzione.
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("np", "numpy")
                and node.func.attr == "float64"
                and len(node.args) == 1 and not node.keywords):
            return number(node.args[0])
        raise ValueError("La lista contiene un elemento non numerico.")

    return [number(node) for node in tree.elts]


def exact_integer(raw):
    # Evita arrotondamenti dei timestamp grandi tramite float.
    value = Decimal(raw.strip())
    if not value.is_finite() or value != value.to_integral_value() or value < 0:
        raise ValueError("È richiesto un intero non negativo.")
    return int(value)


def numeric(raw):
    return float(raw) if raw.strip() else math.nan


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def check_inputs(args):
    anomalies, summaries, manifest = [], [], []
    excluded_blocks = {block for block, reason in args.exclude_block}

    def issue(severity, code, p, block="", source="", row="", trigger="", apa="",
              related_file="", related_row="", detail=""):
        anomalies.append(dict(zip(ANOMALY_FIELDS, [
            severity, code, p, block, str(source), row, str(trigger), apa,
            str(related_file), related_row, detail,
        ])))

    def read_text(path, p, block):
        try:
            data = path.read_bytes()
            manifest.append({"momentum_GeV_c": p, "path": str(path),
                             "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            return data.decode("utf-8-sig")
        except (OSError, UnicodeError) as exc:
            issue("error", "unreadable_file", p, block, path, detail=str(exc))
            return None

    for p in args.momenta:
        folder = args.input_dir / f"{p}GeV"
        seen_times, seen_files = {}, {}
        time_counts = Counter()
        blocks, block_summaries = [], []
        if not folder.is_dir():
            issue("error", "missing_momentum_folder", p, source=folder,
                  detail="Cartella del momento assente.")
        else:
            for sub in sorted(folder.iterdir()):
                if not sub.is_dir() or "_to_" not in sub.name:
                    continue
                if f"{p}GeV/{sub.name}" in excluded_blocks:
                    continue
                match = re.fullmatch(r"(\d+)_to_(\d+)", sub.name)
                if not match:
                    issue("error", "invalid_block_name", p, sub.name, sub,
                          detail="Atteso <start>_to_<stop> con indici interi.")
                    continue
                start, stop = map(int, match.groups())
                if start >= stop:
                    issue("error", "invalid_block_range", p, sub.name, sub,
                          detail="Intervallo vuoto o invertito.")
                for old_start, old_stop, old_sub in blocks:
                    if max(start, old_start) < min(stop, old_stop):
                        issue("warning", "overlapping_blocks", p, sub.name, sub,
                              related_file=old_sub, detail="Gli intervalli [start, stop) si sovrappongono.")
                blocks.append((start, stop, sub))

        for start, stop, sub in blocks:
            path = sub / f"photoelectron_dataframe_{p}GeV.csv"
            summary = {key: 0 for key in SUMMARY_FIELDS}
            summary.update(scope="block", momentum_GeV_c=p, block=sub.name,
                           source_file=str(path), status="checked")
            block_summaries.append(summary)
            local_times = Counter()
            list_path = sub / "files_read.txt"
            if not list_path.is_file():
                issue("warning", "missing_files_read", p, sub.name, list_path,
                      detail="Provenienza dei file non disponibile in questo blocco.")
            else:
                content = read_text(list_path, p, sub.name)
                if content is not None:
                    entries = [(i, line.strip()) for i, line in enumerate(content.splitlines(), 1)
                               if line.strip()]
                    if not entries:
                        issue("warning", "empty_files_read", p, sub.name, list_path,
                              detail="Lista dei file vuota.")
                    for line_no, entry in entries:
                        if entry in seen_files:
                            old_path, old_line = seen_files[entry]
                            issue("warning", "repeated_source_entry", p, sub.name, list_path,
                                  line_no, related_file=old_path, related_row=old_line,
                                  detail=f"Voce ripetuta nella lista dei file: {entry}")
                        else:
                            seen_files[entry] = (list_path, line_no)

            if not path.is_file():
                summary["status"] = "missing"
                issue("error", "missing_csv", p, sub.name, path, detail="CSV atteso assente.")
                continue
            content = read_text(path, p, sub.name)
            if content is None:
                summary["status"] = "unreadable"
                continue
            reader = csv.DictReader(io.StringIO(content), strict=True)
            try:
                headers = reader.fieldnames or []
                missing = sorted(set(REQUIRED) - set(headers))
                if missing or len(headers) != len(set(headers)):
                    summary["status"] = "invalid_schema"
                    issue("error", "invalid_columns", p, sub.name, path, 1,
                          detail=f"Colonne mancanti: {missing}; intestazioni duplicate: {len(headers) != len(set(headers))}.")
                    continue
                for record in reader:
                    row_no = reader.line_num
                    summary["rows"] += 1
                    raw_time = record.get("trigger_time") or ""
                    if None in record or any(value is None for value in record.values()):
                        issue("error", "malformed_row", p, sub.name, path, row_no, raw_time,
                              detail="Numero di campi diverso dall'intestazione.")
                        continue
                    try:
                        timestamp = exact_integer(raw_time)
                    except (ValueError, InvalidOperation):
                        issue("error", "invalid_trigger_time", p, sub.name, path, row_no, raw_time,
                              detail="Timestamp assente, non finito, negativo o non intero.")
                    else:
                        summary["valid_trigger_times"] += 1
                        local_times[timestamp] += 1
                        time_counts[timestamp] += 1
                        signature = tuple(record[key].strip() for key in REQUIRED if key != "trigger_time")
                        if timestamp in seen_times:
                            old_path, old_row, old_signature = seen_times[timestamp]
                            issue("warning", "repeated_trigger_time", p, sub.name, path, row_no, timestamp,
                                  related_file=old_path, related_row=old_row,
                                  detail="Stesso tempo candidato; " +
                                  ("campi APA testualmente identici." if signature == old_signature
                                   else "campi APA differenti: verificare i contributi.") +
                                  " Non prova lo stesso trigger DAQ: il run manca nei CSV.")
                        else:
                            seen_times[timestamp] = (path, row_no, signature)

                    finite_means = 0
                    for apa in (1, 2):
                        prefix = f"apa{apa}_"
                        context = (p, sub.name, path, row_no, raw_time, apa)
                        try:
                            mean, std = (numeric(record[prefix + key]) for key in ("mean", "std"))
                            finite_means += int(math.isfinite(mean))
                            count = exact_integer(record[prefix + "n_events"])
                            values = parse_number_list(record[prefix + "list"])
                        except (ValueError, SyntaxError, InvalidOperation, OverflowError, RecursionError) as exc:
                            issue("error", "invalid_apa_fields", *context, detail=str(exc))
                            continue
                        if count != len(values):
                            issue("error", "count_mismatch", *context,
                                  detail=f"n_events={count}, lunghezza lista={len(values)}.")
                        if not values:
                            summary[prefix + "empty_lists"] += 1
                            issue("warning", "empty_pe_list", *context,
                                  detail="Nessun contributo PE: media non utilizzabile.")
                            if not (math.isnan(mean) and math.isnan(std)):
                                issue("error", "empty_list_statistics", *context,
                                      detail="Per lista vuota si attendono mean e std NaN/vuoti.")
                            continue
                        if not all(math.isfinite(value) for value in values):
                            issue("error", "nonfinite_pe_list", *context,
                                  detail="La lista contiene NaN o inf; statistiche non verificate.")
                            continue
                        if any(value < 0 for value in values):
                            issue("warning", "negative_pe", *context,
                                  detail="Lista con PE negativi: da esaminare, nessuna esclusione automatica.")
                        try:
                            expected = {"mean": statistics.mean(values), "std": statistics.pstdev(values)}
                        except (OverflowError, ValueError) as exc:
                            issue("error", "statistics_failure", *context, detail=str(exc))
                            continue
                        for key, observed in (("mean", mean), ("std", std)):
                            if not math.isfinite(observed) or not math.isclose(
                                    observed, expected[key], rel_tol=args.rtol, abs_tol=args.atol):
                                issue("error", f"{key}_mismatch", *context,
                                      detail=f"Salvato={observed!r}, ricalcolato={expected[key]!r} (std: ddof=0).")
                    summary["rows_with_two_finite_means"] += int(finite_means == 2)
            except csv.Error as exc:
                summary["status"] = "invalid_csv"
                issue("error", "csv_parse_error", p, sub.name, path, reader.line_num,
                      detail=f"Lettura interrotta; conteggi parziali: {exc}")
            summary["distinct_trigger_times"] = len(local_times)
            summary["rows_repeated_trigger_time"] = sum(n for n in local_times.values() if n > 1)

        ordered = sorted(seen_times)
        for previous, current in zip(ordered, ordered[1:]):
            if current - previous <= args.near_ticks:
                path, row_no, _ = seen_times[current]
                old_path, old_row, _ = seen_times[previous]
                issue("warning", "near_trigger_times", p, path.parent.name, path, row_no, current,
                      related_file=old_path, related_row=old_row,
                      detail=f"Tempi candidati distinti distanti {current - previous} tick; verificare sulle waveform.")
        if not blocks:
            issue("error", "no_blocks", p, source=folder,
                  detail="Nessun blocco <start>_to_<stop> disponibile dopo le eventuali esclusioni.")
        for summary in block_summaries:
            relevant = [a for a in anomalies if a["momentum_GeV_c"] == p and a["block"] == summary["block"]]
            for severity in ("error", "warning"):
                summary[severity + "s"] = sum(a["severity"] == severity for a in relevant)
            if summary["rows"] == 0 and summary["status"] == "checked":
                summary["status"] = "empty"
                issue("warning", "empty_csv", p, summary["block"], summary["source_file"],
                      detail="CSV senza righe dati.")
                summary["warnings"] += 1
        summaries.extend(block_summaries)
        total = {key: sum(s[key] for s in block_summaries) for key in SUMMARY_FIELDS[5:]}
        total.update(scope="momentum", momentum_GeV_c=p, block="ALL", source_file="",
                     status="see_report", distinct_trigger_times=len(time_counts),
                     rows_repeated_trigger_time=sum(n for n in time_counts.values() if n > 1))
        for severity in ("error", "warning"):
            total[severity + "s"] = sum(a["severity"] == severity and a["momentum_GeV_c"] == p for a in anomalies)
        summaries.append(total)
    return summaries, anomalies, manifest


def main():
    base = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=base / "output/apa1_vs_apa2")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--momenta", nargs="+", type=int, choices=(1, 2, 3, 5, 7), default=[1, 2, 3, 5, 7])
    parser.add_argument("--exclude-block", nargs=2, action="append", default=[],
                        metavar=("BLOCCO", "MOTIVO"),
                        help="Cartella relativa a input-dir e motivo, ad esempio 2GeV/0_to_1 'Blocco ridondante'. Ripetibile.")
    parser.add_argument("--near-ticks", type=int, default=200, help="Distanza diagnostica, non una regola di associazione.")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-9)
    args = parser.parse_args()
    if args.near_ticks < 0 or any(not math.isfinite(v) or v < 0 for v in (args.rtol, args.atol)):
        parser.error("Tolleranze e near-ticks devono essere finiti e non negativi.")
    args.momenta = sorted(set(args.momenta))
    args.input_dir = args.input_dir.expanduser().resolve()
    created = datetime.now(timezone.utc)
    output = (args.output_dir or base / "output/review" /
              ("check_apa12_inputs_" + created.strftime("%Y%m%dT%H%M%S_%fZ"))).expanduser().resolve()
    if not args.input_dir.is_dir():
        parser.error(f"Cartella input inesistente: {args.input_dir}")
    if output.exists():
        parser.error(f"La cartella output esiste già; scegliere un nuovo nome: {output}")
    exclusions = []
    seen_exclusions = set()
    for block, reason in args.exclude_block:
        match = re.fullmatch(r"(1|2|3|5|7)GeV/(\d+)_to_(\d+)", block)
        if not match or int(match[2]) >= int(match[3]):
            parser.error(f"Blocco da escludere non valido: {block}. Atteso, ad esempio, 2GeV/0_to_1.")
        if int(match[1]) not in args.momenta:
            parser.error(f"Il blocco {block} non appartiene ai momenti richiesti.")
        if block in seen_exclusions:
            parser.error(f"Esclusione ripetuta: {block}.")
        if not reason.strip():
            parser.error(f"Specificare una motivazione non vuota per {block}.")
        path = args.input_dir / block
        if not path.is_dir():
            parser.error(f"La cartella da escludere non esiste: {path}")
        seen_exclusions.add(block)
        exclusions.append({"block": block, "path": str(path), "reason": reason.strip()})
    summaries, anomalies, manifest = check_inputs(args)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "summary.csv", SUMMARY_FIELDS, summaries)
    write_csv(output / "anomalies.csv", ANOMALY_FIELDS, anomalies)
    config = {**vars(args), "input_dir": str(args.input_dir), "output_dir": str(output)}
    (output / "manifest.json").write_text(json.dumps({
        "created_utc": created.isoformat(), "configuration": config,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version, "inputs": manifest, "excluded_blocks": exclusions,
    }, indent=2) + "\n", encoding="utf-8")
    lines = [
        "CONTROLLO INPUT APA 1–APA 2", f"Input: {args.input_dir}", f"Output: {output}",
        f"Momenti nominali (GeV/c): {args.momenta}",
        f"Tolleranze statistiche: rtol={args.rtol}, atol={args.atol}; ddof=0.",
        f"Tempi vicini: <= {args.near_ticks} tick; confrontate coppie consecutive di tempi distinti.",
        "", "BLOCCHI ESCLUSI ESPLICITAMENTE",
    ]
    if exclusions:
        for item in exclusions:
            lines.append(f"{item['block']} — {item['reason']}\n  Percorso: {item['path']}")
        lines.append("I blocchi esclusi non sono letti né conteggiati; i file originali restano invariati.")
    else:
        lines.append("Nessuno.")
    lines += ["", "RISULTATI PER MOMENTO"]
    for s in summaries:
        if s["scope"] == "momentum":
            lines.append(f"{s['momentum_GeV_c']} GeV/c: {s['rows']} righe, "
                         f"{s['distinct_trigger_times']} tempi distinti, "
                         f"{s['rows_repeated_trigger_time']} righe con tempo ripetuto; "
                         f"{s['errors']} errori, {s['warnings']} avvisi.")
    lines += ["", "ANOMALIE PER TIPO"]
    lines += [f"{code}: {count}" for code, count in sorted(Counter(a["code"] for a in anomalies).items())]
    if not anomalies:
        lines.append("Nessuna anomalia rilevata dai controlli implementati.")
    lines += [
        "", "COME LEGGERE I RISULTATI",
        "error = input assente/non leggibile oppure incoerenza di schema o di valori.",
        "warning = situazione da esaminare; non implica necessariamente dati errati.",
        "Gli errori/avvisi contano segnalazioni, non trigger: una riga può averne diverse.",
        "rows_repeated_trigger_time conta tutte le righe dei gruppi ripetuti, inclusa la prima.",
        "Nei blocchi i tempi ripetuti sono contati internamente; nei totali anche tra blocchi.",
        "csv_row è il numero di riga fisica finale del record CSV; intestazione alla riga 1.",
        "rows_with_two_finite_means non implica che le statistiche siano corrette.",
        "Le cartelle senza '_to_' (ad esempio apa12_study) non sono input e sono ignorate.",
        "", "LIMITI E CONTROLLI ANCORA NECESSARI SU LXPLUS",
        "I CSV non contengono run/record DAQ: tempi uguali sono solo duplicati candidati.",
        "Verificare run_number, record_number e daq_window_timestamp nei pickle FS/ST.",
        "Verificare waveform assegnate a più candidati, trigger di una sola APA e canali ripetuti.",
        "Verificare copertura dei canali, template e selezioni prima del salvataggio dei CSV.",
        "files_read.txt può descrivere solo FS e includere caricamenti falliti.",
        "Intervalli sovrapposti e voci ripetute non certificano duplicati fisici.",
        "I trigger già scartati a monte non sono recuperabili dai soli CSV.",
        "Non sono state eliminate righe, modificati input, prodotti fit o definite soglie.",
        "Anche in assenza di anomalie, l'associazione fisica APA 1–APA 2 resta da verificare.",
    ]
    report = "\n".join(lines) + "\n"
    (output / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 1 if anomalies else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OSError as exc:
        print(f"Errore di accesso ai file: {exc}", file=sys.stderr)
        sys.exit(2)
