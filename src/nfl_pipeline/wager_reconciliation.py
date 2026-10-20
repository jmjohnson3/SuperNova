"""Import actual NFL receipts and costs into a local append-only journal.

This records manual execution; it does not place bets, approve cash, or reinterpret
paper recommendations as wagers. Use --template for the JSON import contract.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOURNAL = ROOT / "data" / "nfl_wager_reconciliation"
VERSION = 1
KINDS = {"wager_placed", "wager_settled", "execution_attempt", "cost", "cost_coverage"}
COMMON = {"event_id", "kind"}
REFERENCE = {"source_kind", "recommendation_id", "observation_id"}
FIELDS = {
    "wager_placed": COMMON | REFERENCE | {"wager_id", "book", "ticket_id", "receipt_reference",
        "bet_description", "currency", "stake", "executed_price", "placed_at_utc", "recommended_price"},
    "wager_settled": COMMON | {"wager_id", "result", "cash_return", "settled_at_utc", "receipt_reference"},
    "execution_attempt": COMMON | REFERENCE | {"book", "status", "observed_at_utc", "reason",
        "observed_price", "requested_stake", "available_stake", "currency"},
    "cost": COMMON | {"cost_id", "currency", "amount", "incurred_at_utc", "category", "description",
        "receipt_reference"},
    "cost_coverage": COMMON | {"currency", "start_date", "end_date", "attestation"},
}


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _time(value, name):
    try:
        when = datetime.fromisoformat(_text(value, name).replace("Z", "+00:00"))
        if when.tzinfo is None or when.utcoffset() is None:
            raise ValueError()
        return when.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} requires an ISO timestamp with a timezone") from exc


def _money(value, name, *, positive=False):
    try:
        if isinstance(value, bool):
            raise ValueError()
        number = Decimal(str(value))
        if not number.is_finite() or number < 0 or (positive and number <= 0) or number != number.quantize(Decimal(".01")):
            raise ValueError()
        return format(number, ".2f")
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{name} requires a finite {'positive' if positive else 'nonnegative'} amount in cents") from exc


def _price(value, name):
    try:
        if isinstance(value, bool):
            raise ValueError()
        number = Decimal(str(value))
        if not number.is_finite() or abs(number) < 100 or number != int(number):
            raise ValueError()
        return int(number)
    except (InvalidOperation, ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} requires integer American odds (<= -100 or >= 100)") from exc


def normalize_event(value):
    if not isinstance(value, dict) or value.get("kind") not in KINDS:
        raise ValueError("Each event needs a supported kind")
    event = dict(value)
    extra = set(event) - FIELDS[event["kind"]]
    if extra:
        raise ValueError(f"Unknown fields for {event['kind']}: {sorted(extra)}")
    event["event_id"] = _text(event.get("event_id"), "event_id")
    required = {
        "wager_placed": ["wager_id", "book", "ticket_id", "receipt_reference", "bet_description", "source_kind", "recommendation_id", "currency"],
        "wager_settled": ["wager_id", "result", "receipt_reference"],
        "execution_attempt": ["book", "source_kind", "recommendation_id", "status", "reason", "currency"],
        "cost": ["cost_id", "currency", "category", "description", "receipt_reference"],
        "cost_coverage": ["currency", "attestation"],
    }[event["kind"]]
    for key in required:
        event[key] = _text(event.get(key), key)
    if "source_kind" in event and event["source_kind"] not in {"cash", "sharp", "external"}:
        raise ValueError("source_kind must be cash, sharp, or external; cash uses ledger_id, sharp uses alert_id")
    if "book" in event:
        event["book"] = event["book"].lower()
    if "currency" in event:
        event["currency"] = event["currency"].upper()
        if len(event["currency"]) != 3 or not event["currency"].isascii() or not event["currency"].isalpha():
            raise ValueError("currency must be a three-letter currency code")
    for key in ("placed_at_utc", "settled_at_utc", "observed_at_utc", "incurred_at_utc"):
        required_time = {"wager_placed": "placed_at_utc", "wager_settled": "settled_at_utc",
                         "execution_attempt": "observed_at_utc", "cost": "incurred_at_utc"}.get(event["kind"])
        if key == required_time:
            event[key] = _time(event.get(key), key)
    for key in ("stake", "cash_return", "amount", "requested_stake", "available_stake"):
        required_money = {"wager_placed": "stake", "wager_settled": "cash_return", "cost": "amount"}.get(event["kind"])
        if key in event or key == required_money:
            event[key] = _money(event.get(key), key, positive=key in {"stake", "amount"})
    for key in ("executed_price", "recommended_price", "observed_price"):
        if key in event or (key == "executed_price" and event["kind"] == "wager_placed"):
            event[key] = _price(event.get(key), key)
    if "observation_id" in event:
        event["observation_id"] = _text(event["observation_id"], "observation_id")
    if event["kind"] == "wager_settled" and event["result"] not in {"win", "loss", "push", "void", "cashout"}:
        raise ValueError("result must be win, loss, push, void, or cashout")
    if event["kind"] == "execution_attempt" and event["status"] not in {"available", "declined", "not_placed"}:
        raise ValueError("Attempt status must be available, declined, or not_placed")
    if event["kind"] == "cost_coverage":
        for key in ("start_date", "end_date"):
            event[key] = date.fromisoformat(_text(event.get(key), key)).isoformat()
        if event["end_date"] < event["start_date"]:
            raise ValueError("Cost coverage end precedes start")
    return event


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(events):
    return hashlib.sha256(_canonical(events).encode("utf-8")).hexdigest()


def _validated_union(existing, incoming):
    events = {}; placements = {}; settlements = {}; tickets = {}; costs = {}
    for raw in [*existing, *incoming]:
        event = normalize_event(raw)
        eid = event["event_id"]
        if eid in events:
            if events[eid] != event:
                raise ValueError(f"Conflicting event_id: {eid}; prior events cannot be overwritten")
            continue
        events[eid] = event
        kind = event["kind"]
        if kind == "wager_placed":
            wid = event["wager_id"]; ticket = (event["book"], event["ticket_id"])
            if wid in placements or ticket in tickets:
                raise ValueError(f"Duplicate wager or sportsbook ticket: {wid}; reuse its original event_id")
            placements[wid] = event; tickets[ticket] = wid
        elif kind == "wager_settled":
            wid = event["wager_id"]
            if wid in settlements:
                raise ValueError(f"Settlement already recorded for {wid}; conflicting receipts need review")
            settlements[wid] = event
        elif kind == "cost":
            if event["cost_id"] in costs:
                raise ValueError(f"Duplicate cost_id: {event['cost_id']}")
            costs[event["cost_id"]] = event
    for wid, settlement in settlements.items():
        if wid not in placements:
            raise ValueError(f"Settlement has no actual placement: {wid}")
        if settlement["settled_at_utc"] < placements[wid]["placed_at_utc"]:
            raise ValueError(f"Settlement precedes placement: {wid}")
    prior_ids = {e["event_id"] for e in existing}
    added = [e for eid, e in events.items() if eid not in prior_ids]
    return list(events.values()), added


def read_events(journal=DEFAULT_JOURNAL):
    events = []
    for path in sorted((Path(journal) / "batches").glob("*.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(batch, dict) or batch.get("schema_version") != VERSION
                or not isinstance(batch.get("events"), list)
                or batch.get("sha256") != _digest(batch.get("events")) or path.stem != batch.get("sha256")):
            raise ValueError(f"Journal batch failed integrity verification: {path.name}")
        events.extend(batch["events"])
    return _validated_union([], events)[0]


def parse_document(document):
    if (not isinstance(document, dict) or type(document.get("schema_version")) is not int
            or document.get("schema_version") != VERSION or set(document) != {"schema_version", "events"}):
        raise ValueError('Import requires exactly {"schema_version": 1, "events": [...]}')
    if not isinstance(document["events"], list):
        raise ValueError("events must be a list")
    return [normalize_event(e) for e in document["events"]]


def import_document(document, journal=DEFAULT_JOURNAL, *, dry_run=False):
    incoming = parse_document(document)
    journal = Path(journal)
    if dry_run:
        existing = read_events(journal)
        _, added = _validated_union(existing, incoming)
        return dict(valid=True, added=len(added), duplicate_events=len(incoming)-len(added), written=False)
    journal.mkdir(parents=True, exist_ok=True)
    lock = journal / ".import.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError("Another import holds .import.lock; investigate an interrupted import before removing it") from exc
    temp = None
    try:
        with handle:
            handle.write(str(os.getpid()))
        existing = read_events(journal)
        _, added = _validated_union(existing, incoming)
        if not added:
            return dict(valid=True, added=0, duplicate_events=len(incoming), written=False)
        batch_hash = _digest(added)
        batch = dict(schema_version=VERSION, recorded_at_utc=datetime.now(timezone.utc).isoformat(), sha256=batch_hash, events=added)
        directory = journal / "batches"
        directory.mkdir(exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".import-", suffix=".tmp", dir=directory)
        temp = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(batch, indent=2, ensure_ascii=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        destination = directory / f"{batch_hash}.json"
        if destination.exists():
            raise ValueError("Journal batch already exists unexpectedly")
        os.rename(temp, destination)
        temp = None
        return dict(valid=True, added=len(added), duplicate_events=len(incoming)-len(added), written=True, batch_sha256=batch_hash)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
        lock.unlink()


def _implied(price):
    return -price / (100-price) if price < 0 else 100 / (100+price)


def _payout(price):
    return Decimal(price)/100 if price > 0 else Decimal(100)/-price


def _covered(coverage, start, end):
    cursor = start.toordinal()
    for row in sorted(coverage, key=lambda r: r["start_date"]):
        left = date.fromisoformat(row["start_date"]).toordinal()
        right = date.fromisoformat(row["end_date"]).toordinal()
        if left > cursor:
            break
        cursor = max(cursor, right+1)
    return cursor > end.toordinal()


def summarize(events, *, start_date=None, end_date=None):
    """Cash-basis settled P&L by UTC date; open stakes and currencies stay separate."""
    events = _validated_union([], events)[0]
    end = date.fromisoformat(str(end_date)) if end_date else datetime.now(timezone.utc).date()
    dated = [e[k][:10] for e in events for k in ("placed_at_utc", "settled_at_utc", "incurred_at_utc", "observed_at_utc") if k in e]
    start = date.fromisoformat(str(start_date)) if start_date else date.fromisoformat(min(dated)) if dated else end
    if end < start:
        raise ValueError("Report end date precedes start date")
    inside = lambda stamp: str(start) <= stamp[:10] <= str(end)
    placements = {e["wager_id"]: e for e in events if e["kind"] == "wager_placed" and e["placed_at_utc"][:10] <= str(end)}
    settled = {e["wager_id"]: e for e in events if e["kind"] == "wager_settled" and e["settled_at_utc"][:10] <= str(end)}
    costs = [e for e in events if e["kind"] == "cost" and inside(e["incurred_at_utc"])]
    attempts = [e for e in events if e["kind"] == "execution_attempt" and inside(e["observed_at_utc"])]
    coverage = [e for e in events if e["kind"] == "cost_coverage"]
    currencies = sorted({e["currency"] for e in [*placements.values(), *costs, *coverage, *attempts]})
    summaries = {}; rows = []; warnings = []
    if not placements:
        warnings.append("No actual wager receipts imported through this report date; actual execution remains unknown")
    for currency in currencies:
        group = [p for p in placements.values() if p["currency"] == currency]
        done = [p for p in group if p["wager_id"] in settled and inside(settled[p["wager_id"]]["settled_at_utc"])]
        opened = [p for p in group if p["wager_id"] not in settled]
        stake = sum((Decimal(p["stake"]) for p in done), Decimal(0))
        cash_return = sum((Decimal(settled[p["wager_id"]]["cash_return"]) for p in done), Decimal(0))
        gross = cash_return-stake
        cost = sum((Decimal(c["amount"]) for c in costs if c["currency"] == currency), Decimal(0))
        complete = _covered([c for c in coverage if c["currency"] == currency], start, end)
        summaries[currency] = dict(placed_in_period=sum(inside(p["placed_at_utc"]) for p in group),
            settled_wagers=len(done), record=dict(Counter(settled[p["wager_id"]]["result"] for p in done)),
            settled_stake=format(stake, ".2f"), cash_return=format(cash_return, ".2f"), gross_profit=format(gross, ".2f"),
            gross_roi=float(gross/stake) if stake else None, recorded_costs=format(cost, ".2f"),
            cost_coverage_complete=complete, net_after_recorded_costs=format(gross-cost, ".2f"),
            net_profit=format(gross-cost, ".2f") if complete else None,
            open_wagers=len(opened), open_stake=format(sum((Decimal(p["stake"]) for p in opened), Decimal(0)), ".2f"))
        if not complete:
            warnings.append(f"{currency}: complete operating costs are unknown; net_profit is null")
    for wid, placement in placements.items():
        settlement = settled.get(wid)
        if not inside(placement["placed_at_utc"]) and not (settlement and inside(settlement["settled_at_utc"])):
            continue
        recommended = placement.get("recommended_price"); executed = placement["executed_price"]
        row = dict(placement, result=settlement["result"] if settlement else "open",
            cash_return=settlement["cash_return"] if settlement else None,
            profit=format(Decimal(settlement["cash_return"])-Decimal(placement["stake"]), ".2f") if settlement else None,
            implied_probability_slippage=_implied(executed)-_implied(recommended) if recommended is not None else None,
            profit_per_unit_change=float(_payout(executed)-_payout(recommended)) if recommended is not None else None)
        rows.append(row)
        if settlement and settlement["result"] != "cashout":
            expected = Decimal(placement["stake"]) * (1+_payout(executed)) if settlement["result"] == "win" else Decimal(0) if settlement["result"] == "loss" else Decimal(placement["stake"])
            if abs(expected-Decimal(settlement["cash_return"])) > Decimal(".01"):
                warnings.append(f"{wid}: receipt return differs from nominal {settlement['result']} payout; actual receipt amount retained")
    return dict(schema_version=VERSION, built_at=datetime.now(timezone.utc).isoformat(), start_date=str(start), end_date=str(end),
        accounting="settlements and costs by UTC date; receipt return includes returned stake; no currency conversion",
        receipt_wagers_through_end=len(placements), currencies=summaries, wagers=rows,
        attempts=dict(Counter(e["status"] for e in attempts)),
        attempt_observations=attempts, warnings=warnings,
        limitations=["Receipt imports are user-confirmed records, not independently verified sportsbook transactions.",
            "No record means unknown execution; an unconfirmed recommendation is never treated as a loss or wager.",
            "Positive implied_probability_slippage means a worse executed price. Missing recommended odds remain unknown.",
            "Paper and legacy database confirmations are excluded from receipt P&L; database cross-checks are reported separately.",
            "Net profit requires an explicit complete-cost attestation covering this reporting period."],
        cash_authorized=False, wagers_placed_by_this_tool=False)


def legacy_confirmations():
    """Read existing confirmed states without creating schemas or trusting suggested stakes."""
    import psycopg2
    import psycopg2.extras
    from nfl_pipeline.db import PG_DSN
    output = []; unavailable = []
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET LOCAL statement_timeout='30s'")
            for kind, table in (("cash", "bets.nfl_cash_execution_events"), ("sharp", "bets.nfl_sharp_alerts")):
                cur.execute("SELECT to_regclass(%s) AS relation", (table,))
                if cur.fetchone()["relation"] is None:
                    unavailable.append(table); continue
                if kind == "cash":
                    cur.execute("SELECT DISTINCT ON (ledger_id) ledger_id,state,payload FROM bets.nfl_cash_execution_events ORDER BY ledger_id,event_id DESC")
                    for row in cur.fetchall():
                        p = row["payload"]
                        if row["state"] not in {"confirmed", "settled", "paused"} or not p.get("execution_confirmed_at"):
                            continue
                        output.append(dict(source_kind=kind, recommendation_id=str(row["ledger_id"]),
                            book=(p.get("forecast") or {}).get("book"), stake=p.get("stake"), executed_price=p.get("executed_price"),
                            confirmed_at=p.get("execution_confirmed_at"), recorded_result=p.get("result"),
                            timestamp_basis="confirmation_time_not_sportsbook_receipt"))
                else:
                    cur.execute("SELECT alert_id,book,stake,taken_price,placed_at FROM bets.nfl_sharp_alerts WHERE placed_at IS NOT NULL")
                    for row in cur.fetchall():
                        output.append(dict(source_kind=kind, recommendation_id=str(row["alert_id"]), book=row["book"],
                            stake=str(row["stake"]) if row["stake"] is not None else None, executed_price=row["taken_price"],
                            confirmed_at=row["placed_at"].isoformat(), timestamp_basis="legacy_confirmation_time_price_may_have_been_assumed"))
    return dict(records=output, unavailable_tables=unavailable)


def cross_check(events, legacy):
    placements = [e for e in events if e["kind"] == "wager_placed"]
    rows = []
    for old in legacy.get("records", []):
        matches = [p for p in placements if (p["source_kind"], p["recommendation_id"]) == (old["source_kind"], old["recommendation_id"])]
        discrepancies = []
        if len(matches) == 1:
            receipt = matches[0]
            for field in ("stake", "executed_price"):
                if old.get(field) is not None and Decimal(str(old[field])) != Decimal(str(receipt[field])):
                    discrepancies.append(field)
            if old.get("book") and old["book"].lower() != receipt["book"]:
                discrepancies.append("book")
        rows.append(dict(old, receipt_wager_ids=[p["wager_id"] for p in matches], discrepancies=discrepancies,
            reconciliation="missing_receipt" if not matches else "multiple_receipts_review" if len(matches)>1 else "mismatch_review" if discrepancies else "matched"))
    return dict(records=rows, counts=dict(Counter(r["reconciliation"] for r in rows)),
        unavailable_tables=legacy.get("unavailable_tables", []), included_in_receipt_pnl=False)


def template():
    """Examples are descriptions only; the importable document deliberately has no events."""
    return dict(import_document={"schema_version": VERSION, "events": []},
        instructions="Fill your own receipts. Import only import_document, not this help wrapper. Reuse event_id for retries; changed IDs do not permit duplicate tickets.",
        required_fields={kind: sorted(fields-{"observation_id", "recommended_price", "observed_price", "requested_stake", "available_stake"}) for kind, fields in FIELDS.items()},
        field_help={"wager_placed": "Actual cash stake, explicit executed American price, actual placed_at_utc, sportsbook ticket_id and receipt_reference. source_kind cash links ledger_id; sharp links alert_id; external links your own tracking ID. bet_description identifies the exact selection. Bonus bets require separate accounting and are not supported.",
            "wager_settled": "Link wager_id; cash_return is actual money returned INCLUDING returned stake, not profit. Supply settled_at_utc, result win/loss/push/void/cashout, receipt_reference.",
            "execution_attempt": "status available means quote observed, declined means bookmaker rejected execution, not_placed means no wager submitted. Always provide reason and observed_at_utc. Optional observation_id links immutable sharp quote evidence.",
            "cost": "amount is the actual NFL-allocated operating cost; category can be data, hosting, tooling, fees or other. Use stable cost_id and receipt_reference. Do not import total account deposits or stakes as operating costs.",
            "cost_coverage": "Explicitly attest all NFL costs for currency and inclusive UTC start_date/end_date have been supplied, including a true zero-cost period. This enables complete net_profit for covered periods."},
        optional_fields=["recommended_price", "observation_id", "observed_price", "requested_stake", "available_stake"],
        automatically_places_bets=False)


def write_report(document, path):
    """Replace a derived report atomically; this never writes the receipt journal."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".wager-report-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(document, indent=2, default=str) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--template", action="store_true")
    action.add_argument("--validate", type=Path, metavar="JSON")
    action.add_argument("--import-json", type=Path, metavar="JSON")
    action.add_argument("--report", action="store_true")
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--output", type=Path, help="Atomically write the derived report JSON, without adding receipt records")
    parser.add_argument("--include-db", action="store_true", help="Read-only cross-check; no legacy or paper rows are added to receipt P&L")
    args = parser.parse_args(argv)
    if (args.include_db or args.start_date or args.end_date or args.output) and not args.report:
        parser.error("--include-db, --start-date, --end-date, and --output require --report")
    if args.output and (args.output.resolve().is_relative_to((args.journal/"batches").resolve())
                        or args.output.resolve() == (args.journal/".import.lock").resolve()):
        parser.error("Report output cannot replace immutable journal batches or the import lock")
    try:
        if args.template:
            result = template()
        elif args.validate or args.import_json:
            path = args.validate or args.import_json
            result = import_document(json.loads(path.read_text(encoding="utf-8-sig")), args.journal, dry_run=bool(args.validate))
        else:
            events = read_events(args.journal)
            result = summarize(events, start_date=args.start_date, end_date=args.end_date)
            if args.include_db:
                result["legacy_cross_check"] = cross_check(events, legacy_confirmations())
            if args.output:
                write_report(result, args.output)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Reconciliation failed: {exc}\n")


if __name__ == "__main__":
    main()
