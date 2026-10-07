"""Grade sharp-edge alerts by closing-line value: the go/no-go evidence for betting them.

For each alert whose game has started:
- FanDuel CLV: FanDuel's last no-vig price before kickoff, moved to the alert's line, vs its no-vig price
  when alerted. FanDuel moves props by line at a fixed price, so a same-line lookup misses most moves.
- EV at the sharp close: the alerted FanDuel price valued at the sharp book's last fair line before
  kickoff (moved to FanDuel's line when close). Positive = the alert really was +EV.
- Result (win/loss/push) from final stats, reported last; a few weeks of results is mostly noise.

Pass bar to enable SHARP_EDGE_BETS_ENABLED: >= 100 graded alerts over >= 3 NFL weeks, >= 55% beat
FanDuel's close, and mean EV at the sharp close >= +1%.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline import sharp_math
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.sharp_watch import PING_WINDOW_BY_STAT, PING_WINDOW_MINUTES, SHARP_BOOKS

ROOT = Path(__file__).resolve().parents[3]
PASS = dict(min_alerts=100, min_weeks=3, min_beat_close=0.55, min_sharp_close_ev=0.01)


def load(conn) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='120s'")
        cur.execute("SELECT to_regclass('bets.nfl_sharp_alerts')")
        if cur.fetchone()["to_regclass"] is None:
            return pd.DataFrame()
        cur.execute("""
            SELECT a.*, g.season, g.week, g.status, g.home_score, g.away_score,
              fd.close_price, fd.fd_close_line, fd.fd_close_over, fd.fd_close_under,
              ft.fd_then_over, ft.fd_then_under,
              sc.sharp_close_book, sc.sharp_close_line, sc.sharp_close_over, sc.sharp_close_under,
              gl.actual
            FROM bets.nfl_sharp_alerts a
            JOIN raw.nfl_games g ON g.game_id = a.game_id
            LEFT JOIN LATERAL (  -- FanDuel's last two-sided quote before kickoff, any line (latest snapshot, nearest line)
              SELECT * FROM (
              SELECT q.line AS fd_close_line, q.over_price AS fd_close_over, q.under_price AS fd_close_under, q.fetched_at_utc
              FROM odds.nfl_player_prop_lines q
              WHERE a.player_name_norm IS NOT NULL AND q.bookmaker_key='fanduel' AND q.player_name_norm=a.player_name_norm
                AND q.stat=a.stat AND q.fetched_at_utc < a.commence_time_utc
                AND q.commence_time_utc BETWEEN a.commence_time_utc - interval '30 minutes' AND a.commence_time_utc + interval '30 minutes'
                AND q.over_price IS NOT NULL AND q.under_price IS NOT NULL
              UNION ALL
              SELECT q.total_points, q.total_over_price, q.total_under_price, q.fetched_at_utc
              FROM odds.nfl_game_lines q
              WHERE a.stat='total' AND q.bookmaker_key='fanduel' AND q.home_team_abbr=a.home_team
                AND q.away_team_abbr=a.away_team AND q.fetched_at_utc < a.commence_time_utc
                AND q.total_over_price IS NOT NULL AND q.total_under_price IS NOT NULL
              ) u
              ORDER BY date_trunc('minute', u.fetched_at_utc) DESC, abs(u.fd_close_line - a.fd_line) LIMIT 1) fd0 ON TRUE
            LEFT JOIN LATERAL (SELECT fd0.fd_close_line, fd0.fd_close_over, fd0.fd_close_under,
                CASE WHEN fd0.fd_close_line = a.fd_line
                  THEN CASE WHEN a.side='over' THEN fd0.fd_close_over ELSE fd0.fd_close_under END END AS close_price) fd ON TRUE
            LEFT JOIN LATERAL (  -- FanDuel's two-sided quote at the alert's line when the alert was logged
              SELECT * FROM (
              SELECT q.over_price AS fd_then_over, q.under_price AS fd_then_under, q.fetched_at_utc
              FROM odds.nfl_player_prop_lines q
              WHERE a.player_name_norm IS NOT NULL AND q.bookmaker_key='fanduel' AND q.player_name_norm=a.player_name_norm
                AND q.stat=a.stat AND q.line=a.fd_line AND q.fetched_at_utc <= a.alerted_at + interval '5 minutes'
                AND q.commence_time_utc BETWEEN a.commence_time_utc - interval '30 minutes' AND a.commence_time_utc + interval '30 minutes'
                AND q.over_price IS NOT NULL AND q.under_price IS NOT NULL
              UNION ALL
              SELECT q.total_over_price, q.total_under_price, q.fetched_at_utc
              FROM odds.nfl_game_lines q
              WHERE a.stat='total' AND q.bookmaker_key='fanduel' AND q.total_points=a.fd_line AND q.home_team_abbr=a.home_team
                AND q.away_team_abbr=a.away_team AND q.fetched_at_utc <= a.alerted_at + interval '5 minutes'
                AND q.total_over_price IS NOT NULL AND q.total_under_price IS NOT NULL
              ) u ORDER BY u.fetched_at_utc DESC LIMIT 1) ft ON TRUE
            LEFT JOIN LATERAL (
              SELECT * FROM (
              SELECT q.bookmaker_key AS sharp_close_book, q.line AS sharp_close_line,
                     q.over_price AS sharp_close_over, q.under_price AS sharp_close_under, q.fetched_at_utc
              FROM odds.nfl_player_prop_lines q
              WHERE a.player_name_norm IS NOT NULL AND q.bookmaker_key = ANY(%(sharp)s) AND q.player_name_norm=a.player_name_norm
                AND q.stat=a.stat AND q.fetched_at_utc < a.commence_time_utc
                AND q.over_price IS NOT NULL AND q.under_price IS NOT NULL
                AND q.commence_time_utc BETWEEN a.commence_time_utc - interval '30 minutes' AND a.commence_time_utc + interval '30 minutes'
              UNION ALL
              SELECT q.bookmaker_key, q.total_points, q.total_over_price, q.total_under_price, q.fetched_at_utc
              FROM odds.nfl_game_lines q
              WHERE a.stat='total' AND q.bookmaker_key = ANY(%(sharp)s) AND q.home_team_abbr=a.home_team
                AND q.away_team_abbr=a.away_team AND q.fetched_at_utc < a.commence_time_utc
                AND q.total_over_price IS NOT NULL AND q.total_under_price IS NOT NULL
              ) u  -- latest sharp snapshot, then the line nearest FanDuel's
              ORDER BY date_trunc('minute', u.fetched_at_utc) DESC, abs(u.sharp_close_line - a.fd_line) LIMIT 1) sc ON TRUE
            LEFT JOIN LATERAL (
              SELECT CASE a.stat WHEN 'receiving_yards' THEN l.receiving_yards WHEN 'rushing_yards' THEN l.rushing_yards
                       WHEN 'passing_yards' THEN l.passing_yards WHEN 'receptions' THEN l.receptions
                       WHEN 'passing_tds' THEN l.passing_tds END::float AS actual
              FROM raw.nfl_rosters r JOIN raw.nfl_player_gamelogs l
                ON l.player_id=r.player_id AND l.game_id=a.game_id AND l.team_abbr=r.team_abbr
              WHERE a.player_name_norm IS NOT NULL AND r.season=g.season AND r.player_name_norm=a.player_name_norm
                AND r.team_abbr IN (a.home_team, a.away_team)
              LIMIT 1) gl ON TRUE
            WHERE a.commence_time_utc <= now()""", {"sharp": list(SHARP_BOOKS)})
        return pd.DataFrame([dict(r) for r in cur.fetchall()])


def _side(p_over, side):
    return None if p_over is None else (p_over if side == "over" else 1.0 - p_over)


def fd_clv(r) -> float | None:
    """FanDuel's no-vig probability for the alerted side at the alert's line: close minus then."""
    if pd.isna(r.fd_close_line) or pd.isna(r.fd_then_over):
        return None
    close_over = sharp_math.shift_probability(r.stat, float(r.fd_close_line),
                                              sharp_math.no_vig_over(r.fd_close_over, r.fd_close_under), float(r.fd_line))
    then_over = sharp_math.no_vig_over(r.fd_then_over, r.fd_then_under)
    close, then = _side(close_over, r.side), _side(then_over, r.side)
    return None if close is None or then is None else close - then


def grade(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for col in ("fd_line", "fd_price", "close_price", "sharp_close_line", "home_score", "away_score", "actual",
                "fd_close_line", "fd_close_over", "fd_close_under", "fd_then_over", "fd_then_under"):
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["clv"] = [fd_clv(r) for r in df.itertuples()]
    # A placed bet is graded at the price actually taken; everything else at the alerted price.
    taken = pd.to_numeric(df.get("taken_price"), errors="coerce") if "taken_price" in df else pd.Series(np.nan, index=df.index)
    df["bet_price"] = taken.fillna(df.fd_price)
    df["slippage"] = [None if pd.isna(t) else sharp_math.implied(p) - sharp_math.implied(t)
                      for t, p in zip(taken, df.fd_price)]
    ev_close = []
    for r in df.itertuples():
        fair_over = None
        if pd.notna(r.sharp_close_line):
            fair_over = sharp_math.over_probability_at(r.stat, float(r.sharp_close_line),
                                                       sharp_math.no_vig_over(r.sharp_close_over, r.sharp_close_under), float(r.fd_line))
        fair = None if fair_over is None else (fair_over if r.side == "over" else 1 - fair_over)
        ev_close.append(None if fair is None else sharp_math.ev(fair, r.bet_price))
    df["ev_at_sharp_close"] = ev_close
    actual = np.where(df.stat == "total", df.home_score + df.away_score, df.actual)
    diff = (actual - df.fd_line) * np.where(df.side == "over", 1, -1)
    df["result"] = np.where(pd.isna(actual) | (df.status != "final"), None, np.where(diff > 0, "win", np.where(diff < 0, "loss", "push")))
    payout = np.where(df.bet_price > 0, df.bet_price / 100, 100 / df.bet_price.abs())
    df["units"] = np.where(df.result == "win", payout, np.where(df.result == "loss", -1.0, np.where(df.result == "push", 0.0, np.nan)))
    return df


def _stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(alerts=0, weeks=0)
    clv = pd.to_numeric(df.clv, errors="coerce").dropna()
    evc = pd.to_numeric(df.ev_at_sharp_close, errors="coerce").dropna()
    weeks = df[["season", "week"]].drop_duplicates().shape[0]
    return dict(alerts=int(len(df)), weeks=int(weeks), with_fd_close=int(len(clv)), beat_fd_close=float((clv > 1e-9).mean()) if len(clv) else None,
                mean_fd_clv=float(clv.mean()) if len(clv) else None, with_sharp_close=int(len(evc)),
                mean_ev_at_sharp_close=float(evc.mean()) if len(evc) else None,
                positive_at_sharp_close=float((evc > 0).mean()) if len(evc) else None,
                settled=int(df.result.notna().sum()), roi=float(pd.to_numeric(df.units, errors="coerce").dropna().mean()) if df.result.notna().any() else None,
                mean_alert_ev=float(pd.to_numeric(df.ev, errors="coerce").mean()),
                staked=float(_stake(df).sum()),
                mean_slippage=float(pd.to_numeric(df.get("slippage"), errors="coerce").dropna().mean())
                if "slippage" in df and pd.to_numeric(df["slippage"], errors="coerce").notna().any() else None,
                # Dollars won or lost on settled rows. sharp_watch reads this for its sticky loss pause.
                realized_profit=float((pd.to_numeric(df.units, errors="coerce") * _stake(df)).dropna().sum()))


def _stake(df: pd.DataFrame) -> pd.Series:
    if "stake" not in df:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df["stake"], errors="coerce").fillna(0.0)


def _tier(df: pd.DataFrame) -> pd.Series:
    """Tier as the strategy is now defined: a >= +3% gap seen more than PING_WINDOW_MINUTES before kickoff is
    'early' (pings sent before the window existed are relabelled here; stored rows are not rewritten)."""
    tier = df["tier"].fillna("alert") if "tier" in df else pd.Series("alert", index=df.index)
    if {"commence_time_utc", "alerted_at"} <= set(df.columns):
        lead = (pd.to_datetime(df.commence_time_utc, utc=True) - pd.to_datetime(df.alerted_at, utc=True)).dt.total_seconds() / 60
        window = df.stat.map(PING_WINDOW_BY_STAT).fillna(PING_WINDOW_MINUTES) if "stat" in df else PING_WINDOW_MINUTES
        tier = tier.where(~((tier == "alert") & (lead > window)), "early")
    return tier


def summarize(df: pd.DataFrame) -> dict:
    """Pass/fail is judged on pinged alerts (EV >= +3%). Logged gaps (+1% to +3%, no ping) are reported
    beside them: they show sooner whether FanDuel-vs-sharp gaps predict the close at all."""
    if df.empty:
        return dict(alerts=0, verdict="no graded alerts yet", logged=dict(alerts=0, weeks=0))
    tier = _tier(df)
    out = _stats(df[tier == "alert"])
    out["logged"] = _stats(df[tier == "logged"])
    out["early"] = _stats(df[tier == "early"])
    # Bets actually placed, confirmed by hand with sharp_bets --confirm. This is the real money record.
    placed = df[df.placed_at.notna()] if "placed_at" in df else df.iloc[:0]
    out["placed"] = _stats(placed)
    books = df["book"].fillna("fanduel") if "book" in df else pd.Series("fanduel", index=df.index)
    out["by_book"] = {str(book): _stats(g[_tier(g) == "alert"]) for book, g in df.groupby(books)}
    weeks = out.get("weeks", 0)
    checks = dict(enough_alerts=out["alerts"] >= PASS["min_alerts"], enough_weeks=weeks >= PASS["min_weeks"],
                  beats_fd_close=(out.get("beat_fd_close") or 0) >= PASS["min_beat_close"],
                  positive_at_sharp_close=(out.get("mean_ev_at_sharp_close") or -1) >= PASS["min_sharp_close_ev"])
    out["checks"] = checks
    out["verdict"] = ("PASS: the alerted prices hold up; raising SHARP_WATCH_STAKE is justified" if all(checks.values())
                      else "no graded alerts yet" if not out["alerts"]
                      else "keep collecting" if not (checks["enough_alerts"] and checks["enough_weeks"])
                      else "FAIL: the alerted prices do not hold up at the close; do not bet them")
    return out


def markdown(summary: dict, df: pd.DataFrame) -> str:
    pct = lambda v: "-" if v is None else f"{v:.1%}"
    signed = lambda v: "-" if v is None else f"{v:+.2%}"
    a, e, g = summary, summary.get("early") or {}, summary.get("logged") or {}
    row = lambda label, key, fmt, bar: f"| {label} | {fmt(a.get(key))} | {fmt(e.get(key))} | {fmt(g.get(key))} | {bar} |"
    lines = ["# NFL Sharp-Edge Alerts", "", f"Verdict: **{summary['verdict']}**", "",
             f"Alerts graded: {a.get('alerts', 0)} over {a.get('weeks', 0)} week(s) "
             f"(pass bar: {PASS['min_alerts']} over {PASS['min_weeks']}). Early gaps graded: {e.get('alerts', 0)}. "
             f"Logged gaps graded: {g.get('alerts', 0)}.", "",
             f"| Evidence (most reliable first) | Alerts (>= +3%, last {PING_WINDOW_MINUTES} min, pinged) | Early (>= +3%, earlier) "
             "| Logged (+1% to +3%) | Pass bar (alerts) |",
             "|---|---:|---:|---:|---:|",
             row("Mean EV at the sharp close", "mean_ev_at_sharp_close", signed, f">= {PASS['min_sharp_close_ev']:+.0%}"),
             row("Share positive at the sharp close", "positive_at_sharp_close", pct, "-"),
             row("Beat FanDuel's close (no-vig, at the alert's line)", "beat_fd_close", pct, f">= {PASS['min_beat_close']:.0%}"),
             row("Mean FanDuel CLV (no-vig prob)", "mean_fd_clv", signed, "> 0"),
             row("EV claimed when seen", "mean_alert_ev", signed, "-"),
             row("Flat ROI", "roi", signed, "noise for weeks"),
             row("Settled", "settled", lambda v: "-" if v is None else str(v), "-")]
    placed = summary.get("placed") or {}
    if placed.get("alerts"):
        slip = placed.get("mean_slippage")
        lines += ["", f"**Placed bets: {placed['alerts']} for ${placed.get('staked', 0):.2f} staked; "
                      f"realized ${placed.get('realized_profit', 0):+.2f} on {placed.get('settled', 0)} settled "
                      f"(EV at sharp close {signed(placed.get('mean_ev_at_sharp_close'))}"
                      + (f", slippage {signed(slip)} vs the alerted price" if slip is not None else "") + ").**"]
    by_book = summary.get("by_book") or {}
    if len(by_book) > 1:
        lines += ["", "| Book | Alerts | EV at sharp close | Beat close |", "|---|---:|---:|---:|"]
        for book, st in sorted(by_book.items()):
            lines.append(f"| {book} | {st.get('alerts', 0)} | {signed(st.get('mean_ev_at_sharp_close'))} | {pct(st.get('beat_fd_close'))} |")
    if not df.empty:
        tier = _tier(df)
        lines += ["", "| Week | Tier | Count | Beat FD close | Mean EV at sharp close |", "|---|---|---:|---:|---:|"]
        for (season, week, t), grp in df.assign(tier=tier).groupby(["season", "week", "tier"]):
            c = pd.to_numeric(grp.clv, errors="coerce").dropna(); e = pd.to_numeric(grp.ev_at_sharp_close, errors="coerce").dropna()
            lines.append(f"| {season}-{week} | {t} | {len(grp)} | {pct((c > 1e-9).mean() if len(c) else None)} | {signed(e.mean() if len(e) else None)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        df = grade(load(conn))
    summary = summarize(df)
    doc = dict(built_at=datetime.now(timezone.utc).isoformat(), pass_bar=PASS, summary=summary)
    atomic_json(ROOT / "reports" / "nfl_sharp_alerts_latest.json", doc)
    (ROOT / "reports" / "nfl_sharp_alerts_latest.md").write_text(markdown(summary, df), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
