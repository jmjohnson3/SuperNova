"""Grade the multi-sport research decisions at Pinnacle's close, and post the morning digest.

A decision is the first +3% gap seen 15-45 minutes before the start for an event, market and player
(the best book first), exactly as the NFL strategy counts them. Its EV at the close is its price valued
at Pinnacle's no-vig probability for the same line in the last stored quote before the start (at most
20 minutes old). Missing closes stay unknown and count against coverage.

Per sport, research ends only when the evidence supports a stake review: at least 50 graded decisions
across 7+ days, mean EV at the close of +1.5% or better and close coverage of 90% or more. That turns
nothing on; staking a sport is a separate, explicit decision, as it was for NFL.

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
from sharp_pipeline import scan

ROOT = Path(__file__).resolve().parents[2]
MAX_CLOSE_AGE_MINUTES = 22  # game lines are polled about every 20 minutes
REVIEW = dict(min_decisions=50, min_days=7, min_mean_close_ev=0.015, min_close_coverage=0.90)
SPORT_LABEL = {"nba": "NBA", "ncaaf": "NCAAF"}


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
            key = (r["sport"], r["event_id"])
            if key not in closes or r["observed_at_utc"] > closes[key][0]:
                closes[key] = (r["observed_at_utc"], r["payload"])
    return decisions, closes


def close_ev(row: dict, closes: dict) -> float | None:
    found = closes.get((row["sport"], row["event_id"]))
    if not found:
        return None
    observed, event = found
    sharp = next((b for b in event.get("bookmakers", []) if b.get("key") == scan.SHARP_BOOK), None)
    if sharp is None:
        return None
    quotes = scan.pairs(dict(sharp, _home=event.get("home_team"), _away=event.get("away_team")), safe_time(observed))
    line = float(row["line"])
    first_side = row["side"] in ("over", "home")
    key_line = line if (first_side or row["market"] not in scan.TEAM_MARKETS) else -line
    found = scan.reference(quotes, (row["market"], row["player_norm"] or "game", key_line), row["sport"])
    if found is None:
        return None
    fair_a = found[1]
    return sharp_math.ev(fair_a if first_side else 1 - fair_a, int(row["price"]))


def decisions_frame(raw: pd.DataFrame, closes: dict) -> pd.DataFrame:
    if raw.empty:
        return raw
    df = raw.copy()
    df["ev"] = df.ev.astype(float)
    df = df.sort_values(["decided_at", "ev"], ascending=[True, False])
    df = df.drop_duplicates(["sport", "event_id", "market", "player_norm", "tier"], keep="first")
    df["ev_at_close"] = [close_ev(r, closes) for r in df.to_dict("records")]
    df["day"] = pd.to_datetime(df.commence_time_utc, utc=True).dt.tz_convert("America/New_York").dt.date
    return df


def stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(decisions=0)
    close = pd.to_numeric(df.ev_at_close, errors="coerce").dropna()
    return dict(decisions=int(len(df)), days=int(df.day.nunique()), ev_when_seen=float(df.ev.mean()),
                graded=int(len(close)), close_coverage=float(len(close) / len(df)),
                ev_at_close=float(close.mean()) if len(close) else None,
                still_positive=float((close > 0).mean()) if len(close) else None)


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    for sport in scan.SPORTS:
        part = df[df.sport.eq(sport)] if not df.empty else df
        decided = part[part.tier.eq("decision")] if not part.empty else part
        overall = stats(decided)
        ready = (overall.get("graded", 0) >= REVIEW["min_decisions"] and overall.get("days", 0) >= REVIEW["min_days"]
                 and (overall.get("ev_at_close") or -1) >= REVIEW["min_mean_close_ev"]
                 and (overall.get("close_coverage") or 0) >= REVIEW["min_close_coverage"])
        out[sport] = dict(overall=overall, status="ready_for_stake_review" if ready else "research",
                          by_market={m: stats(g) for m, g in decided.groupby("market")} if not decided.empty else {},
                          by_book={b: stats(g) for b, g in decided.groupby("book")} if not decided.empty else {},
                          logged=stats(part[part.tier.eq("logged")]) if not part.empty else dict(decisions=0))
    return out


def _pct(v, signed=True):
    return "-" if v is None else (f"{v:+.2%}" if signed else f"{v:.0%}")


def markdown(summary: dict) -> str:
    lines = ["# Sharp-Gap Research: NBA and NCAAF", "",
             "Research only: nothing is staked. Same test as the NFL strategy: +3% or better against Pinnacle's no-vig "
             "price at the exact line, seen 15-45 minutes before the start, graded at Pinnacle's last quote before the start. "
             f"Stake review needs {REVIEW['min_decisions']}+ graded decisions over {REVIEW['min_days']}+ days, "
             f"+{REVIEW['min_mean_close_ev']:.1%} or better at the close and {REVIEW['min_close_coverage']:.0%} close coverage.", ""]
    for sport, s in summary.items():
        o = s["overall"]
        lines += [f"## {SPORT_LABEL[sport]}: {s['status']}", "",
                  f"Decisions {o['decisions']} over {o.get('days', 0)} day(s); EV when seen {_pct(o.get('ev_when_seen'))}; "
                  f"at the close {_pct(o.get('ev_at_close'))} ({_pct(o.get('still_positive'), False)} still +EV); "
                  f"close coverage {_pct(o.get('close_coverage'), False)}.", ""]
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
             f"({_pct(s.get('still_positive'), False)} still +EV, {_pct(s.get('close_coverage'), False)} graded)."]
    for r in top:
        close = "-" if r["ev_at_close"] is None or pd.isna(r["ev_at_close"]) else f"{r['ev_at_close']:+.1%}"
        lines.append(f"- {describe(r)}: {r['ev']:+.1%} seen, {close} at close")
    return "\n".join(lines)


def _webhook(sport: str) -> str:
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
    summary = summarize(df)
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
            if text:
                poster(text, sport)
                posted.append(sport)
            state.setdefault("digest_posted", {})[sport] = str(yesterday)
        atomic_json(scan.STATE, state)
    return dict(summary={k: v["status"] for k, v in summary.items()}, digests_posted=posted)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--digest", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run(digest=args.digest), indent=2, default=str))


if __name__ == "__main__":
    main()
