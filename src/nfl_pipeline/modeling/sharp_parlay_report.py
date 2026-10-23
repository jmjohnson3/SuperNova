"""Grade the research parlays (sharp_parlays) at the sharp close and on results.

Each leg is graded exactly as the alert report grades it. The parlay's EV at the close is the
product of the legs' closing fair probabilities times the combined price, minus one; it is known only
when every leg has a close. Settlement follows the books: any losing leg loses, a pushed leg drops out,
and the rest pays the product of the remaining prices. Research only: nothing here is a bet.

    python -m nfl_pipeline.modeling.sharp_parlay_report
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling import sharp_alert_report as alerts
from nfl_pipeline.sharp_parlays import american_price, decimal_price

ROOT = Path(__file__).resolve().parents[3]


def grade_parlay(legs: list[dict]) -> dict:
    """legs: dicts with bet_price, ev_at_sharp_close (or None) and result ('win'/'loss'/'push'/None)."""
    decimal = 1.0
    fair_close = 1.0
    for leg in legs:
        decimal *= decimal_price(leg["bet_price"])
        close = leg.get("ev_at_sharp_close")
        fair_close = None if close is None or fair_close is None else fair_close * (float(close) + 1.0) / decimal_price(leg["bet_price"])
    results = [leg.get("result") for leg in legs]
    if any(r is None for r in results):
        outcome, units = None, None
    elif "loss" in results:
        outcome, units = "loss", -1.0
    else:
        paid = 1.0
        for leg in legs:
            if leg["result"] == "win":
                paid *= decimal_price(leg["bet_price"])
        outcome, units = ("push" if paid == 1.0 else "win"), paid - 1.0
    return dict(decimal_price=decimal, ev_at_sharp_close=None if fair_close is None else fair_close * decimal - 1.0,
                result=outcome, units=units)


def build(conn) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT to_regclass('bets.nfl_sharp_parlays') AS t")
        if cur.fetchone()["t"] is None:
            return dict(parlays=[], summary=dict(parlays=0))
        cur.execute("SELECT * FROM bets.nfl_sharp_parlays ORDER BY et_day")
        parlays = [dict(r) for r in cur.fetchall()]
    frame = alerts.grade(alerts.load(conn))
    ledger = frame[frame.get("evidence_kind", pd.Series("", index=frame.index)).eq("alert_ledger")] if not frame.empty else frame
    by_id = {str(r["alert_id"]): r for r in ledger.to_dict("records")} if not ledger.empty else {}
    rows = []
    for parlay in parlays:
        legs = []
        for alert_id in parlay["leg_alert_ids"]:
            graded = by_id.get(str(alert_id)) or {}
            close = graded.get("ev_at_sharp_close")
            legs.append(dict(alert_id=int(alert_id), bet_price=graded.get("bet_price") or next(
                l["fd_price"] for l in parlay["legs"] if int(l["alert_id"]) == int(alert_id)),
                ev_at_sharp_close=None if close is None or pd.isna(close) else float(close),
                result=graded.get("result") if isinstance(graded.get("result"), str) else None))
        rows.append(dict(parlay_id=parlay["parlay_id"], et_day=str(parlay["et_day"]), book=parlay["book"],
                         ev_when_seen=float(parlay["ev"]), price=american_price(float(parlay["decimal_price"])),
                         posted=parlay["posted"], legs=legs, **grade_parlay(legs)))
    graded_close = [r["ev_at_sharp_close"] for r in rows if r["ev_at_sharp_close"] is not None]
    settled = [r for r in rows if r["units"] is not None]
    summary = dict(parlays=len(rows), with_close=len(graded_close),
                   mean_ev_when_seen=sum(r["ev_when_seen"] for r in rows) / len(rows) if rows else None,
                   mean_ev_at_close=sum(graded_close) / len(graded_close) if graded_close else None,
                   settled=len(settled), wins=sum(r["result"] == "win" for r in settled),
                   units=sum(r["units"] for r in settled) if settled else None)
    return dict(parlays=rows, summary=summary)


def markdown(doc: dict) -> str:
    s = doc["summary"]
    pct = lambda v: "-" if v is None else f"{v:+.1%}"
    lines = ["# NFL Research Parlays", "",
             "Two unstaked +3% sharp-gap legs from different games at one book, one per day, chosen before kickoff. "
             "Never a bet; graded to decide whether such parlays deserve a stake.", "",
             f"Parlays: {s['parlays']}; graded at the close: {s.get('with_close', 0)} (mean EV when seen "
             f"{pct(s.get('mean_ev_when_seen'))}, at the close {pct(s.get('mean_ev_at_close'))}); settled {s.get('settled', 0)}, "
             f"won {s.get('wins', 0)}, flat units {'-' if s.get('units') is None else f'{s['units']:+.2f}'}.", "",
             "| Day | Book | Price | EV when seen | EV at close | Result |", "|---|---|---:|---:|---:|---|"]
    for r in doc["parlays"]:
        lines.append(f"| {r['et_day']} | {r['book']} | {r['price']:+d} | {pct(r['ev_when_seen'])} | "
                     f"{pct(r['ev_at_sharp_close'])} | {r['result'] or 'pending'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        doc = build(conn)
    doc["built_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(ROOT / "reports" / "nfl_sharp_parlays_latest.json", doc)
    (ROOT / "reports" / "nfl_sharp_parlays_latest.md").write_text(markdown(doc), encoding="utf-8")
    print(json.dumps(doc["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
