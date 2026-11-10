"""Is a DraftKings price still there after the alert? One provider re-quote per delivered NBA paper alert.

DraftKings refuses scripted requests to its own site (HTTP 403 from its bot protection), so this cannot
read the book directly the way the FanDuel observers do. Instead, about a minute after an alert is
delivered, it asks The Odds API for that one market at DraftKings only (1 credit) and records what the
book showed once its quote had refreshed after the alert. One attempt per alert, including failures.

    python -m sharp_pipeline.draftkings_checks [--report-only]

This is the provider's view of DraftKings, not the logged-in screen: it cannot show limits or bet
acceptance. Nothing is staked and no alert is sent. Separate from the frozen NBA experiment and from
the manual check ledger; it only reads their alerts.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras
import requests

from nfl_pipeline import sharp_math
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.markets import normalize_name

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "reports" / "sharp_multi_state.json"
BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"
BOOK = "draftkings"
MARKET_KEY = {"points": "player_points", "rebounds": "player_rebounds", "assists": "player_assists", "threes": "player_threes"}
WAIT_SECONDS = 60          # DraftKings quotes at the provider are 25s old at the median, 63s at the 90th percentile
MAX_DELAY_SECONDS = 300    # an alert older than this is never checked
DAILY_CREDITS = 40         # override with DK_CHECK_DAILY_CREDITS
CREDIT_FLOOR = 1000        # never spend below this balance; override with DK_CHECK_CREDIT_FLOOR
_ET = ZoneInfo("America/New_York")
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bets.draftkings_price_checks (
  alert_id BIGINT PRIMARY KEY REFERENCES bets.nba_paper_price_alerts(alert_id),
  attempted_at TIMESTAMPTZ NOT NULL, observed_at TIMESTAMPTZ, quote_updated_at TIMESTAMPTZ,
  status TEXT NOT NULL CHECK (status IN ('available','changed_line','not_listed','unknown')),
  reason TEXT NOT NULL, line NUMERIC, price INTEGER, credits INTEGER NOT NULL DEFAULT 0,
  source TEXT NOT NULL DEFAULT 'odds_api_requote', payload JSONB NOT NULL);
"""


def unknown(reason: str) -> dict:
    return dict(status="unknown", reason=reason, line=None, price=None, quote_updated_at=None)


def read_quote(alert: dict, event: dict, observed: datetime) -> dict:
    """What DraftKings showed for the alerted player, side and line in a re-quote taken after the alert."""
    start = safe_time(event.get("commence_time")) or alert["commence_time_utc"]
    if observed >= min(start, alert["commence_time_utc"]):
        return unknown("game_started")
    book = next((b for b in event.get("bookmakers", []) if b.get("key") == BOOK), None)
    market = next((m for m in (book or {}).get("markets", []) if m.get("key") == MARKET_KEY[alert["market"]]), None)
    if market is None:
        return dict(unknown("market_absent"), status="not_listed")
    updated = safe_time(market.get("last_update") or book.get("last_update"))
    if updated is None or updated > observed:
        return unknown("quote_time_unverified")
    if updated <= alert["sent_at"]:
        return unknown("provider_not_refreshed")  # the same quote the alert was built from proves nothing
    side = alert["side"].lower()
    offers = [(float(o["point"]), int(o["price"])) for o in market.get("outcomes", [])
              if normalize_name(o.get("description") or "") == alert["player_norm"]
              and str(o.get("name") or "").lower() == side and o.get("point") is not None
              and sharp_math.implied(o.get("price")) is not None]
    if not offers:
        return dict(status="not_listed", reason="player_absent", line=None, price=None, quote_updated_at=updated)
    line = float(alert["line"])
    same = sorted({price for point, price in offers if point == line})
    if len(same) > 1:
        return dict(unknown("conflicting_quotes"), quote_updated_at=updated)
    if same:
        better = sharp_math.implied(same[0]) <= sharp_math.implied(int(alert["price"])) + 1e-12
        return dict(status="available", reason="same_or_better_price" if better else "worse_price",
                    line=line, price=same[0], quote_updated_at=updated)
    point, price = min(offers, key=lambda o: abs(o[0] - line))
    return dict(status="changed_line", reason="line_moved", line=point, price=price, quote_updated_at=updated)


def pending(conn, now: datetime) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT a.alert_id, a.sent_at, d.event_id, d.commence_time_utc, d.market, d.player, d.player_norm,
                              d.side, d.line, d.price, d.fair_probability, d.ev, d.decided_at
                       FROM bets.nba_paper_price_alerts a JOIN bets.sharp_multi_decisions d USING (decision_id)
                       WHERE a.book=%s AND d.book=%s AND a.delivery_status='sent' AND a.sent_at >= %s
                         AND d.commence_time_utc > %s AND d.market = ANY(%s)
                         AND NOT EXISTS (SELECT 1 FROM bets.draftkings_price_checks c WHERE c.alert_id=a.alert_id)
                       ORDER BY a.sent_at, a.alert_id""",
                    (BOOK, BOOK, now - timedelta(seconds=MAX_DELAY_SECONDS), now, list(MARKET_KEY)))
        return [dict(r) for r in cur.fetchall()]


def spent_today(conn, now: datetime) -> int:
    midnight = datetime.combine(now.astimezone(_ET).date(), datetime.min.time(), tzinfo=_ET)
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(SUM(credits),0) FROM bets.draftkings_price_checks WHERE attempted_at >= %s", (midnight,))
        return int(cur.fetchone()[0])


def balance() -> int | None:
    try:
        return int(json.loads(STATE.read_text(encoding="utf-8")).get("credits_remaining"))
    except (OSError, ValueError, TypeError):
        return None


def save(conn, alert: dict, attempted: datetime, observed: datetime | None, result: dict, credits: int) -> None:
    payload = dict(alert={k: str(v) for k, v in alert.items()}, seconds_after_alert=(
        (observed - alert["sent_at"]).total_seconds() if observed else None),
        ev_at_alert=float(alert["ev"]),
        ev_at_check=(sharp_math.ev(float(alert["fair_probability"]), result["price"])
                     if result["status"] == "available" else None))
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO bets.draftkings_price_checks
                         (alert_id, attempted_at, observed_at, quote_updated_at, status, reason, line, price, credits, payload)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                    (alert["alert_id"], attempted, observed, result.get("quote_updated_at"), result["status"],
                     result["reason"], result["line"], result["price"], credits, psycopg2.extras.Json(payload)))
    conn.commit()


def run(conn, *, key: str, session=requests, clock=lambda: datetime.now(timezone.utc), sleep=time.sleep,
        credits_remaining: int | None = None) -> dict:
    alerts = pending(conn, clock())
    out = dict(status="ok", pending=len(alerts), checked=0, credits=0, results={})
    if not alerts:
        return dict(out, status="nothing_pending")
    daily = int(os.getenv("DK_CHECK_DAILY_CREDITS") or DAILY_CREDITS)
    floor = int(os.getenv("DK_CHECK_CREDIT_FLOOR") or CREDIT_FLOOR)
    spent = spent_today(conn, clock())
    remaining = credits_remaining if credits_remaining is not None else balance()
    # Let the provider refresh its DraftKings quote after the newest alert before asking.
    wait = (max(a["sent_at"] for a in alerts) + timedelta(seconds=WAIT_SECONDS) - clock()).total_seconds()
    if wait > 0:
        sleep(min(wait, WAIT_SECONDS))
    groups: dict[tuple, list[dict]] = {}
    for alert in alerts:
        groups.setdefault((alert["event_id"], alert["market"]), []).append(alert)
    for (event_id, market), members in groups.items():
        attempted, observed, event, used, failure = clock(), None, None, 0, None
        if not key:
            failure = "api_key_missing"
        elif spent + 1 > daily:
            failure = "daily_credit_budget"
        elif remaining is None or remaining - 1 < floor:
            failure = "credit_floor"
        else:
            try:
                response = session.get(f"{BASE}/events/{event_id}/odds", timeout=20, params={
                    "apiKey": key, "bookmakers": BOOK, "markets": MARKET_KEY[market],
                    "oddsFormat": "american", "dateFormat": "iso"})
                observed = clock()
                used = int(response.headers.get("x-requests-last") or 1)
                remaining = int(response.headers.get("x-requests-remaining") or remaining - used)
                spent += used
                out["credits"] += used
                if response.status_code != 200:
                    failure = f"http_{response.status_code}"
                else:
                    event = response.json()
                    if not isinstance(event, dict):
                        failure = "response_format_unverified"
            except (requests.RequestException, ValueError) as exc:
                failure = "network_" + type(exc).__name__
        for i, alert in enumerate(members):
            if failure:
                result = unknown(failure)
            elif (observed - alert["sent_at"]).total_seconds() > MAX_DELAY_SECONDS:
                result = unknown("check_window_missed")
            else:
                result = read_quote(alert, event, observed)
            save(conn, alert, attempted, observed, result, used if i == 0 else 0)
            out["checked"] += 1
            out["results"][f"{result['status']}/{result['reason']}"] = out["results"].get(f"{result['status']}/{result['reason']}", 0) + 1
    return out


def report(conn) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT c.*, d.player, d.market, d.side, d.line AS alert_line, d.price AS alert_price
                       FROM bets.draftkings_price_checks c JOIN bets.nba_paper_price_alerts a USING (alert_id)
                       JOIN bets.sharp_multi_decisions d USING (decision_id) ORDER BY c.attempted_at, c.alert_id""")
        checks = [dict(r) for r in cur.fetchall()]
    answered = [c for c in checks if c["status"] != "unknown"]
    held = [c for c in answered if c["status"] == "available" and c["reason"] == "same_or_better_price"]
    mean = lambda values: (sum(values) / len(values)) if values else None
    doc = dict(built_at=datetime.now(timezone.utc).isoformat(), source="odds_api_requote", book=BOOK, attempts=len(checks),
               answered=len(answered), outcomes=dict(Counter(f"{c['status']}/{c['reason']}" for c in checks)),
               held_same_or_better=len(held), held_share=(len(held) / len(answered)) if answered else None,
               median_seconds_after_alert=(sorted(c["payload"]["seconds_after_alert"] for c in answered)[len(answered) // 2]
                                           if answered else None),
               # A price that is gone is worth nothing; a worse price is valued at what it became.
               mean_ev_at_alert=mean([c["payload"]["ev_at_alert"] for c in answered]),
               mean_ev_obtainable=mean([c["payload"]["ev_at_check"] or 0.0 for c in answered]),
               credits=sum(c["credits"] for c in checks),
               note="Provider re-quote of DraftKings about a minute after the alert. Not the logged-in screen: "
                    "limits, bet acceptance and profit remain unverified. Nothing is staked.")
    atomic_json(ROOT / "reports" / "draftkings_price_checks_latest.json", doc)
    pct = lambda v: "-" if v is None else f"{v:+.2%}"
    lines = ["# DraftKings price checks (NBA paper alerts)", "", doc["note"], "",
             f"Attempts: {doc['attempts']}; answered: {doc['answered']}; same or better price still there: "
             f"{doc['held_same_or_better']}" + (f" ({doc['held_share']:.0%})" if answered else "") + ".",
             f"EV when alerted {pct(doc['mean_ev_at_alert'])}; EV at the price found (gone = 0) {pct(doc['mean_ev_obtainable'])}. "
             f"Credits spent: {doc['credits']}.", "",
             "| Outcome | Count |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(doc["outcomes"].items())]
    lines += ["", "| Alert | Player | Bet | Alerted | Found | Outcome | Seconds after alert |", "|---|---|---|---|---|---|---:|"]
    for c in checks[-40:]:
        seconds = c["payload"].get("seconds_after_alert")
        found = "-" if c["price"] is None else f"{float(c['line']):g} / {int(c['price']):+d}"
        lines.append(f"| #{c['alert_id']} | {c['player']} | {c['side']} {float(c['alert_line']):g} {c['market']} | "
                     f"{int(c['alert_price']):+d} | {found} | {c['status']} / {c['reason']} | "
                     f"{'-' if seconds is None else round(seconds)} |")
    (ROOT / "reports" / "draftkings_price_checks_latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {k: v for k, v in doc.items() if k != "note"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-only", action="store_true", help="rebuild the report without checking anything")
    args = ap.parse_args()
    from nfl_pipeline.crawler_oddsapi import OddsCrawlerConfig
    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('bets.nba_paper_price_alerts')")
            if cur.fetchone()[0] is None:
                print(json.dumps(dict(status="no_alert_table")))
                return
            cur.execute("SELECT pg_try_advisory_lock(86731551)")
            if not cur.fetchone()[0]:
                print(json.dumps(dict(status="busy")))
                return
            cur.execute(SCHEMA_SQL)
        conn.commit()
        result = dict(status="report_only") if args.report_only else run(conn, key=OddsCrawlerConfig().oddsapi_key)
        print(json.dumps(dict(result, report=report(conn)), indent=2, default=str))


if __name__ == "__main__":
    main()
