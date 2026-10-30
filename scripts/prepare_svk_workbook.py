#!/usr/bin/env python3
"""Prepare the supplied Svk settlement workbook, without changing the source.

This is data preparation only: it neither activates study configs nor runs a
backtest. Source times are fixed Swedish standard time, verified against eSett
UTC observations in winter and summer. Values are interval MWh; one-hour
intervals become average MW. Missing cells remain missing, including in sums.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

import xlrd

UTC = timezone.utc
SOURCE_TZ = timezone(timedelta(hours=1), "Swedish standard time")
# Exact header meanings, rather than positional guesses. Losses are withdrawals.
FIELDS = {
    ("Timmätt förbr", "exkl. avk.last"): ("metered_ordinary", -1),
    ("Timmätt förbr", ">50 MW"): ("metered_industry_gt50", -1),
    ("Avkopplingsbar", "last"): ("interruptible", -1),
    ("Energilager", "förbrukning"): ("storage_consumption", -1),
    ("Timmätta", "förluster"): ("metered_losses", -1),
    ("Ospec.", "produktion"): ("other_generation", 1),
    ("Vattenkraft", "produktion"): ("hydro", 1),
    ("Vindkraft", "landbaserad"): ("wind_onshore", 1),
    ("Vindkraft", "havsbaserad"): ("wind_offshore", 1),
    ("Värmekraft", "produktion"): ("thermal", 1),
    ("Solkraft", "produktion"): ("solar", 1),
    ("Energilager", "produktion"): ("storage_generation", 1),
    ("Schablonleverans", "förbrukning"): ("profiled_consumption", -1),
    ("Schablonleverans", "förluster"): ("profiled_losses", -1),
}
METERED = ("metered_ordinary", "metered_industry_gt50", "interruptible",
           "storage_consumption", "metered_losses")
PROFILED = ("profiled_consumption", "profiled_losses")
ROOT = Path(__file__).resolve().parents[1]


def parse_time(value, datemode=0):
    if isinstance(value, (int, float)):
        dt = xlrd.xldate_as_datetime(value, datemode)
        kind = "excel_date"
    elif isinstance(value, str):
        dt = datetime.strptime(value.strip(), "%d.%m.%Y %H:%M")
        kind = "text_date"
    else:
        raise ValueError("unrecognized source timestamp")
    if dt.minute or dt.second or dt.microsecond:
        raise ValueError("source timestamp is not on an hour boundary")
    return dt.replace(tzinfo=SOURCE_TZ).astimezone(UTC), kind


def numeric(value):
    if value is None or value == "" or value == "NA":
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"invalid source measurement: {value!r}")
    return Decimal(str(value))


def total(values):
    return None if any(v is None for v in values) else sum(values, Decimal(0))


def derive(values):
    result = dict(values)
    result["metered_total"] = total([values[k] for k in METERED])
    result["profiled_total"] = total([values[k] for k in PROFILED])
    result["load"] = total([values[k] for k in METERED + PROFILED])
    result["wind"] = total([values["wind_onshore"], values["wind_offshore"]])
    return result


def read_source(path):
    book = xlrd.open_workbook(path)
    if book.sheet_names() != ["generis_temp"]:
        raise ValueError("unexpected workbook sheets; review the source schema")
    sheet = book.sheet_by_index(0)
    columns = {}
    for c in range(1, sheet.ncols):
        if str(sheet.cell_value(2, c)).strip() != "SE2":
            continue
        key = tuple(str(sheet.cell_value(r, c)).strip() for r in (0, 1))
        if key not in FIELDS or key in columns:
            raise ValueError(f"unknown or duplicated SE2 header {key!r}")
        if str(sheet.cell_value(3, c)).strip() != "MWh":
            raise ValueError("source unit must be MWh")
        columns[key] = c
    if set(columns) != set(FIELDS):
        raise ValueError("missing expected SE2 consumption/generation columns")
    records = []
    counts = {"excel_date": 0, "text_date": 0, "blank_rows_ignored": 0}
    for r in range(5, sheet.nrows):
        raw_time = sheet.cell_value(r, 0)
        if raw_time == "":
            if any(v != "" for v in sheet.row_values(r)):
                raise ValueError(f"row {r + 1}: measurements without a timestamp")
            counts["blank_rows_ignored"] += 1
            continue
        ts, kind = parse_time(raw_time, book.datemode)
        if records and ts - records[-1]["time"] != timedelta(hours=1):
            raise ValueError(f"row {r + 1}: duplicate, unordered or missing source hour")
        values = {}
        for key, c in columns.items():
            if sheet.cell_type(r, c) == xlrd.XL_CELL_ERROR:
                raise ValueError(f"spreadsheet error at row {r + 1}, column {c + 1}")
            value = numeric(sheet.cell_value(r, c))
            name, sign = FIELDS[key]
            if sign == -1 and value is not None and value > 0:
                raise ValueError(f"unexpected consumption sign at row {r + 1}")
            values[name] = None if value is None else value * sign
        counts[kind] += 1
        records.append({"time": ts, "source_row": r + 1, "values": derive(values)})
    if not records:
        raise ValueError("no observations")
    return records, counts, {
        FIELDS[key][0]: {"column": xlrd.formula.colname(c),
                       "source_header": list(key), "source_sign_multiplier": FIELDS[key][1]}
        for key, c in columns.items()
    }


def reconcile(records, evidence_dir):
    """Frozen independent public observations verify both time and aggregation."""
    lookup = {r["time"]: r["values"] for r in records}
    details = []
    epochs = set()
    for path in sorted(evidence_dir.glob("*-*.json")):
        if path.name == "sources.json":
            continue
        source = json.loads(path.read_text())
        if not isinstance(source, list):
            continue
        for record in source:
            if record["mba"] != "SE2":
                raise ValueError("reconciliation evidence is for a different bidding zone")
            ts = datetime.fromisoformat(record["timestampUTC"].replace("Z", "+00:00"))
            if ts not in lookup:
                continue
            values = lookup[ts]
            checks = ({"wind": "wind_onshore", "windOffshore": "wind_offshore",
                       "solar": "solar", "hydro": "hydro"}
                      if "production" in path.name else
                      {"metered": "metered_total", "profiled": "profiled_total", "total": "load"})
            sign = 1 if "production" in path.name else -1
            # Winter totals match to numerical precision. July's independently
            # published consumption snapshot differs by up to 0.003468 MW.
            # Keep source values, record every difference, and allow 0.01 MW
            # for this cross-snapshot reconciliation (not forecast scoring).
            tolerance = Decimal("0.000001") if sign == 1 else Decimal("0.01")
            for external, internal in checks.items():
                expected = Decimal(str(record[external])) * sign
                actual = values[internal]
                if actual is None or abs(actual - expected) > tolerance:
                    raise ValueError(f"eSett mismatch: {path.name} {ts} {internal}: {actual} != {expected}")
                details.append({"timestamp_utc": ts.isoformat(), "field": internal,
                                "absolute_error_mw": str(abs(actual - expected)),
                                "tolerance_mw": str(tolerance), "evidence": path.name})
            epochs.add(ts.month)
    if not details or 1 not in epochs or not ({6, 7} & epochs):
        raise ValueError("need matching winter and summer evidence before UTC exports")
    return details


def text(value):
    return "NA" if value is None else format(value, "f")


def prepare(source, output, evidence_dir):
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    records, counts, columns = read_source(source)
    matches = reconcile(records, evidence_dir)
    if output.exists():
        raise ValueError("output already exists; choose a fresh directory to preserve provenance")
    output.mkdir(parents=True)
    for field in ("load", "wind", "solar"):
        with (output / f"{field}.csv").open("w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["timestamp", "duration_seconds", "value"])
            for r in records:
                writer.writerow([r["time"].isoformat().replace("+00:00", "Z"), 3600,
                                 text(r["values"][field])])
    fields = list(records[0]["values"])
    with (output / "components.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["timestamp_utc", "source_excel_row"] + [k + "_mw" for k in fields])
        for r in records:
            writer.writerow([r["time"].isoformat().replace("+00:00", "Z"), r["source_row"]] +
                            [text(r["values"][k]) for k in fields])
    evidence = {}
    for p in sorted(evidence_dir.glob("*.json")):
        dest = output / "evidence" / p.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(p.read_bytes())
        evidence[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {
        "source": str(source.resolve()), "source_sha256": source_hash,
        "source_sheet": "generis_temp", "zone": "SE2", "rows": len(records),
        "timestamp_cell_types": counts,
        "source_timezone": "fixed UTC+01:00 (Swedish standard time), not Europe/Stockholm DST",
        "timezone_validation": "independently matched production and consumption against eSett timestampUTC in winter and summer",
        "first_interval_utc": records[0]["time"].isoformat(),
        "end_exclusive_utc": (records[-1]["time"] + timedelta(hours=1)).isoformat(),
        "interval_seconds": 3600, "source_unit": "MWh per interval", "output_unit": "average MW",
        "conversion": "MW = MWh / (duration_seconds / 3600); here all intervals are exactly one hour",
        "columns": columns, "load_components": list(METERED + PROFILED),
        "load_definition": "positive registered SE2 withdrawal including metered and profiled network losses, industry, interruptible load and storage charging; not adjusted for hidden self-generation",
        "wind_components": ["wind_onshore", "wind_offshore"],
        "missing_policy": "NA; never zero-fill; a missing component makes its aggregate missing",
        "precision": "source measurements retained as decimal values; fixed-point rounding deferred to the existing Rust importer",
        "missing_by_field": {k: sum(r["values"][k] is None for r in records) for k in fields},
        "evidence_sha256": evidence, "independent_checks": matches,
        "reconciliation_note": "July consumption snapshots are not bit-identical; workbook values are preserved. Maximum observed difference is recorded below; the 0.01 MW comparison tolerance does not alter exported values or forecast metrics.",
        "maximum_reconciliation_error_mw": str(max(Decimal(m["absolute_error_mw"]) for m in matches)),
        "status": "prepared_and_reconciled_not_activated_in_pipeline",
        "vintage_status": "retrospective settlement observations; historical publication vintages not established",
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(output.glob("*.csv"))},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise ValueError("source workbook changed during preparation")
    print(json.dumps({"output": str(output), "rows": len(records),
                      "checks": len(matches), "timestamp_types": counts,
                      "first_utc": manifest["first_interval_utc"],
                      "end_exclusive_utc": manifest["end_exclusive_utc"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/prepared/svk_2026")
    parser.add_argument("--evidence", type=Path, default=ROOT / "tests/fixtures/svk")
    args = parser.parse_args()
    prepare(args.source, args.output, args.evidence)
