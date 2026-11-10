"""Grade the multi-sport research decisions at Pinnacle's close, and post the morning digest.

A decision is the first +3% gap seen 15-45 minutes before the start for an event, market and player
(the best book first), exactly as the NFL strategy counts them. Its EV at the close is its price valued
at Pinnacle's no-vig probability for the same line in the last stored quote before the start (at most
20 minutes old). Missing closes stay unknown and count against coverage.

NCAAF retains the legacy 50-close/seven-day review check. NBA's separate frozen, book-specific
experiment requires 21 days and timely manual price checks; see nba_experiment. Neither gate
authorizes betting.

    python -m sharp_pipeline.report              # write reports/sharp_multi_latest.{md,json}
    python -m sharp_pipeline.report --digest     # also post yesterday's digest if not yet posted
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras
import requests

from nfl_pipeline import sharp_math
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from sharp_pipeline import outcomes, scan

ROOT = Path(__file__).resolve().parents[2]
MAX_CLOSE_AGE_MINUTES = 22  # game lines are polled about every 20 minutes
REVIEW = dict(min_decisions=50, min_days=7, min_mean_close_ev=0.015, min_close_coverage=0.90)
SPORT_LABEL = {"nba": "NBA", "ncaaf": "NCAAF"}


def payload_kind(event: dict) -> str:
    """Props and game lines arrive in separate polls; a decision is graded only by its own kind."""
    for book in event.get("bookmakers") or []:
        for market in book.get("markets") or []:
            return "props" if str(market.get("key", "")).startswith("player_") else "lines"
    return "lines"


def row_kind(row: dict) -> str:
    return "props" if row["market"] in scan.PROP_STAT.values() else "lines"


def load(conn) -> tuple[pd.DataFrame, dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT to_regclass('bets.sharp_multi_decisions') AS t")
        if cur.fetchone()["t"] is None:
            return pd.DataFrame(), {}
        cur.execute("SELECT * FROM bets.sharp_multi_decisions WHERE commence_time_utc <= now() ORDER BY decided_at")
        decisions = pd.DataFrame([dict(r) for r in cur.fetchall()])
        if decisions.empty:
            return decisions, {}
        cur.execute("""SELECT sport, event_id, observed_at_utc, commence_time_utc, payload FROM bets.sharp_multi_quotes
                       WHERE (sport, event_id) IN (SELECT DISTINCT sport, event_id FROM bets.sharp_multi_decisions)
                         AND observed_at_utc < commence_time_utc
                         AND observed_at_utc >= commence_time_utc - interval '%s minutes'""" % MAX_CLOSE_AGE_MINUTES)
        closes: dict[tuple, tuple] = {}
        for r in cur.fetchall():
            key = (r["sport"], r["event_id"], payload_kind(r["payload"]))
            if key not in closes or r["observed_at_utc"] > closes[key][0]:
                closes[key] = (r["observed_at_utc"], r["payload"])
    return decisions, closes


def close_evidence(row: dict, closes: dict) -> dict:
    """Keep the observed quote and conversion method beside every closing estimate."""
    found = closes.get((row["sport"], row["event_id"], row_kind(row)))
    if not found:
        return {}
    observed, event = found
    sharp = next((b for b in event.get("bookmakers", []) if b.get("key") == scan.SHARP_BOOK), None)
    if sharp is None:
        return {}
    quotes = scan.pairs(dict(sharp, _home=event.get("home_team"), _away=event.get("away_team")), safe_time(observed))
    line = float(row["line"])
    first_side = row["side"] in ("over", "home")
    key_line = line if (first_side or row["market"] not in scan.TEAM_MARKETS) else -line
    found = scan.reference(quotes, (row["market"], row["player_norm"] or "game", key_line), row["sport"])
    if found is None:
        return {}
    fair_a = found[1]
    probability = fair_a if first_side else 1 - fair_a
    return dict(method=found[2], observed_at=str(observed), source_updated_at=str(found[0]["updated"]),
                probability=probability, ev=sharp_math.ev(probability, int(row["price"])))


def close_ev(row: dict, closes: dict) -> float | None:
    evidence = close_evidence(row, closes)
    return evidence.get("ev") if evidence.get("method") == "exact" else None


def decisions_frame(raw: pd.DataFrame, closes: dict, *, per_book: bool = False) -> pd.DataFrame:
    if raw.empty:
        return raw
    df = raw.copy()
    df["ev"] = df.ev.astype(float)
    df = df.sort_values(["decided_at", "ev"], ascending=[True, False])
    keys = ["sport", "event_id", "market", "player_norm", "tier"]
    if per_book:
        keys += ['book', 'experiment_id', 'experiment_sha256']
    df = df.drop_duplicates(keys, keep="first")
    evidence = [close_evidence(r, closes) for r in df.to_dict("records")]
    df["close_method"] = [e.get("method") for e in evidence]
    df["close_probability"] = [e.get("probability") if e.get("method") == 'exact' else None for e in evidence]
    df["close_observed_at"] = [e.get("observed_at") for e in evidence]
    df["close_source_updated_at"] = [e.get("source_updated_at") for e in evidence]
    df["ev_at_close"] = [e.get("ev") if e.get("method") == "exact" else None for e in evidence]
    df["inferred_ev_at_close"] = [e.get("ev") if e.get("method") == "shifted" else None for e in evidence]
    df["day"] = pd.to_datetime(df.commence_time_utc, utc=True).dt.tz_convert("America/New_York").dt.date
    return df


def stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(decisions=0)
    exact = df.get("close_method", pd.Series(None, index=df.index, dtype=object)).eq("exact")
    close = pd.to_numeric(df.loc[exact, "ev_at_close"], errors="coerce").dropna()
    inferred = pd.to_numeric(df.get("inferred_ev_at_close", pd.Series(dtype=float)), errors="coerce").dropna()
    return dict(decisions=int(len(df)), days=int(df.day.nunique()), ev_when_seen=float(df.ev.mean()),
                graded=int(len(close)), close_coverage=float(len(close) / len(df)),
                ev_at_close=float(close.mean()) if len(close) else None,
                still_positive=float((close > 0).mean()) if len(close) else None,
                estimated_closes=int(len(inferred)),
                estimated_close_ev=float(inferred.mean()) if len(inferred) else None)


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    for sport in scan.SPORTS:
        part = df[df.sport.eq(sport)] if not df.empty else df
        decided = part[part.tier.eq("decision")] if not part.empty else part
        overall = stats(decided)
        ready = (overall.get("graded", 0) >= REVIEW["min_decisions"] and overall.get("days", 0) >= REVIEW["min_days"]
                 and (overall.get("ev_at_close") or -1) >= REVIEW["min_mean_close_ev"]
                 and (overall.get("close_coverage") or 0) >= REVIEW["min_close_coverage"])
        out[sport] = dict(overall=overall, status="ready_for_stake_review" if ready and sport != 'nba' else "research",
                          by_market={m: stats(g) for m, g in decided.groupby("market")} if not decided.empty else {},
                          by_book={b: stats(g) for b, g in decided.groupby("book")} if not decided.empty else {},
                          logged=stats(part[part.tier.eq("logged")]) if not part.empty else dict(decisions=0),
                          record=outcomes.record(decided) if not decided.empty else dict(settled=0),
                          shadow=outcomes.shadow_split(decided[decided.market.isin(list(outcomes.MARKET_STAT))])
                          if sport == "nba" and not decided.empty else {})
    return out


def _pct(v, signed=True):
    return "-" if v is None else (f"{v:+.2%}" if signed else f"{v:.0%}")


def record_text(rec: dict) -> str:
    text = f"{rec.get('wins', 0)}-{rec.get('losses', 0)}-{rec.get('pushes', 0)}"
    if rec.get("settled"):
        text += f", ROI {_pct(rec.get('roi'))} at $1 flat ({rec.get('units', 0):+.2f}u)"
    extra = [f"{rec[k]} {k}" for k in ("voids", "pending") if rec.get(k)]
    return text + (f" ({', '.join(extra)})" if extra else "")


def markdown(summary: dict) -> str:
    lines = ["# Sharp-Gap Research: NBA and NCAAF", "",
             "NBA totals below are historical/context results. Its frozen prospective experiment and book-specific "
             "execution review are in `nba_price_checks_latest.md`; these pooled counts cannot authorize NBA stakes.", "",
             "Research only: nothing is staked. Same test as the NFL strategy: +3% or better against Pinnacle's no-vig "
             "price at the exact line, seen 15-45 minutes before the start, graded at Pinnacle's last quote before the start. "
             f"NCAAF stake review needs {REVIEW['min_decisions']}+ graded decisions over {REVIEW['min_days']}+ days, "
             f"+{REVIEW['min_mean_close_ev']:.1%} or better at the exact-line close and {REVIEW['min_close_coverage']:.0%} exact-close coverage. "
             "Estimated line conversions are reported separately and cannot satisfy review requirements.", ""]
    for sport, s in summary.items():
        o = s["overall"]
        lines += [f"## {SPORT_LABEL[sport]}: {s['status']}", "",
                  f"Decisions {o['decisions']} over {o.get('days', 0)} day(s); EV when seen {_pct(o.get('ev_when_seen'))}; "
                  f"at the close {_pct(o.get('ev_at_close'))} ({_pct(o.get('still_positive'), False)} still +EV); "
                  f"exact-close coverage {_pct(o.get('close_coverage'), False)}. "
                  f"Estimated closes: {o.get('estimated_closes', 0)} (EV {_pct(o.get('estimated_close_ev'))}; excluded from review).", ""]
        rec = s.get("record") or {}
        if rec.get("settled") or rec.get("voids") or rec.get("pending"):
            lines += [f"Results (secondary to EV at the close): {record_text(rec)}.", ""]
        if s.get("shadow"):
            lines += ["Model shadow filter (NBA props; recorded beside each decision, never changes one):", "",
                      "| Model vs Pinnacle | Decisions | EV at close | Record | ROI |", "|---|---:|---:|---:|---:|"]
            for label, sh in s["shadow"].items():
                lines.append(f"| {label} | {sh['decisions']} | {_pct(sh.get('ev_at_close'))} | "
                             f"{sh.get('wins', 0)}-{sh.get('losses', 0)}-{sh.get('pushes', 0)} | {_pct(sh.get('roi'))} |")
            lines.append("")
        for title, key in (("Market", "by_market"), ("Book", "by_book")):
            if s[key]:
                lines += [f"| {title} | Decisions | EV when seen | EV at close | Still +EV | Coverage |", "|---|---:|---:|---:|---:|---:|"]
                for name, st in sorted(s[key].items()):
                    lines.append(f"| {name} | {st['decisions']} | {_pct(st.get('ev_when_seen'))} | {_pct(st.get('ev_at_close'))} | "
                                 f"{_pct(st.get('still_positive'), False)} | {_pct(st.get('close_coverage'), False)} |")
                lines.append("")
    return "\n".join(lines) + "\n"


def describe(row: dict) -> str:
    who = row["player"] or f"{row['away_team']} @ {row['home_team']}"
    market = {"h2h": "moneyline", "spread": "spread", "total": "total", "spread_h1": "1st-half spread",
              "total_h1": "1st-half total", "team_total": "team total"}.get(row["market"], row["market"])
    if row["market"] in ("h2h",) + scan.TEAM_MARKETS:  # team sides
        side = row["home_team"] if row["side"] == "home" else row["away_team"]
        line = f" {float(row['line']):+g}" if row["market"] in scan.TEAM_MARKETS else ""
    else:  # over/under on a total or a player stat
        side, line = row["side"], f" {float(row['line']):g}"
    return f"{who}: {side}{line} {market} at {row['book']} {int(row['price']):+d}"


def digest_text(df: pd.DataFrame, sport: str, day: date, scanned: int = 0) -> str | None:
    """The day's research summary. A scanned day with no decisions still posts, so silence never looks like
    a broken scanner (Friday's NCAAF games had none and the digest said nothing)."""
    day_rows = df[df.sport.eq(sport) & df.day.eq(day)] if not df.empty else df
    part = day_rows[day_rows.tier.eq("decision")] if not day_rows.empty else day_rows
    if part.empty:
        if not scanned:
            return None
        best = float(day_rows.ev.max()) if not day_rows.empty else None
        return (f"**{SPORT_LABEL[sport]} sharp-gap research - {day:%a %b %d}** (research only, nothing staked)\n"
                f"{scanned} games scanned against Pinnacle; no gap reached +3% inside the 15-45 minute window. "
                f"{len(day_rows)} smaller gaps logged" + (f" (best {best:+.1%})." if best is not None else ".")
                + " These markets were priced close to Pinnacle all day.")
    s = stats(part)
    top = part.sort_values("ev", ascending=False).head(6).to_dict("records")
    lines = [f"**{SPORT_LABEL[sport]} sharp-gap research - {day:%a %b %d}** (research only, nothing staked)",
             f"{s['decisions']} decisions at +3% or better vs Pinnacle, seen 15-45 min before the start. "
             f"EV when seen {_pct(s['ev_when_seen'])}; at the close {_pct(s.get('ev_at_close'))} "
             f"({_pct(s.get('still_positive'), False)} still +EV, {_pct(s.get('close_coverage'), False)} exact-close coverage). "
             f"Estimated closes: {s.get('estimated_closes', 0)}, excluded from review."]
    if "result" in part:
        lines.append(f"Results: {record_text(outcomes.record(part))}.")
    for r in top:
        close = "-" if r["ev_at_close"] is None or pd.isna(r["ev_at_close"]) else f"{r['ev_at_close']:+.1%}"
        result = f", {r['result']}" if r.get("result") else ""
        lines.append(f"- {describe(r)}: {r['ev']:+.1%} seen, {close} at close{result}")
    return "\n".join(lines)


def _webhook(sport: str) -> str:
    if sport == 'nba':
        from sharp_pipeline.paper_alerts import webhook
        return webhook()
    from supernovabets_config import _saved_windows_env
    for name in (f"{sport.upper()}_DISCORD_WEBHOOK_URL", "MLB_DISCORD_WEBHOOK_URL"):
        value = os.getenv(name) or _saved_windows_env(name)
        if value:
            return value
    return ""


def post(text: str, sport: str) -> None:
    url = _webhook(sport)
    if not url:
        raise RuntimeError("No Discord webhook configured")
    response = requests.post(url, params={"wait": "true"}, json={"content": text[:1950], "allowed_mentions": {"parse": []}}, timeout=20)
    if response.status_code not in (200, 204):
        raise RuntimeError(f"Discord post failed (HTTP {response.status_code})")


def run(*, digest: bool = False, now: datetime | None = None, poster=post) -> dict:
    now = now or datetime.now(timezone.utc)
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        raw, closes = load(conn)
    df = decisions_frame(raw, closes)
    if not df.empty:
        with psycopg2.connect(PG_DSN) as conn:
            conn.set_session(readonly=True)
            df = outcomes.attach(conn, df)
    summary = summarize(df)
    from sharp_pipeline import nba_experiment
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        nba_review = nba_experiment.write_report(conn, raw, closes, now)
    atomic_json(ROOT / "reports" / "sharp_multi_latest.json", dict(built_at=now.isoformat(), summary=summary))
    (ROOT / "reports" / "sharp_multi_latest.md").write_text(markdown(summary), encoding="utf-8")
    posted = []
    scanned = {}
    if digest:
        with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('bets.sharp_multi_quotes')")
            if cur.fetchone()[0] is not None:
                cur.execute("""SELECT sport, (commence_time_utc AT TIME ZONE 'America/New_York')::date, COUNT(DISTINCT event_id)
                               FROM bets.sharp_multi_quotes GROUP BY 1, 2""")
                scanned = {(sport, day): int(n) for sport, day, n in cur.fetchall()}
    # Post after 9 AM Eastern so the previous day's late West Coast and college games are all in.
    if digest and now.astimezone(scan._ET).hour >= 9:
        state = scan._load_state()
        yesterday = now.astimezone(scan._ET).date() - timedelta(days=1)
        for sport in scan.SPORTS:
            if (state.get("digest_posted") or {}).get(sport) == str(yesterday):
                continue
            text = digest_text(df, sport, yesterday, scanned.get((sport, yesterday), 0))
            if sport == 'nba' and text:
                text = nba_experiment.digest(nba_review)
            if text:
                poster(text, sport)
                posted.append(sport)
            state.setdefault("digest_posted", {})[sport] = str(yesterday)
        atomic_json(scan.STATE, state)
    return dict(summary={k: v["status"] for k, v in summary.items()}, digests_posted=posted,
                nba_execution_review={k: v['status'] for k, v in nba_review['by_book'].items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--digest", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run(digest=args.digest), indent=2, default=str))


if __name__ == "__main__":
    main()
