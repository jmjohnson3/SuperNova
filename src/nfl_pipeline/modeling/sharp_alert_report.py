"""Grade sharp-edge alerts by closing-line value: the go/no-go evidence for betting them.

For each alert whose game has started:
- FanDuel CLV: FanDuel's last price for the same player/stat/line/side before kickoff vs the alerted
  price (positive implied-probability move = beat the close).
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
from nfl_pipeline.sharp_watch import SHARP_BOOKS

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
              fd.close_price,
              sc.sharp_close_book, sc.sharp_close_line, sc.sharp_close_over, sc.sharp_close_under,
              gl.actual
            FROM bets.nfl_sharp_alerts a
            JOIN raw.nfl_games g ON g.game_id = a.game_id
            LEFT JOIN LATERAL (
              SELECT CASE WHEN a.side='over' THEN q.over_price ELSE q.under_price END AS close_price, q.fetched_at_utc
              FROM odds.nfl_player_prop_lines q
              WHERE a.player_name_norm IS NOT NULL AND q.bookmaker_key='fanduel' AND q.player_name_norm=a.player_name_norm
                AND q.stat=a.stat AND q.line=a.fd_line AND q.fetched_at_utc < a.commence_time_utc
                AND q.commence_time_utc BETWEEN a.commence_time_utc - interval '30 minutes' AND a.commence_time_utc + interval '30 minutes'
                AND CASE WHEN a.side='over' THEN q.over_price ELSE q.under_price END IS NOT NULL
              UNION ALL
              SELECT CASE WHEN a.side='over' THEN q.total_over_price ELSE q.total_under_price END, q.fetched_at_utc
              FROM odds.nfl_game_lines q
              WHERE a.stat='total' AND q.bookmaker_key='fanduel' AND q.total_points=a.fd_line
                AND q.home_team_abbr=a.home_team AND q.away_team_abbr=a.away_team AND q.fetched_at_utc < a.commence_time_utc
                AND CASE WHEN a.side='over' THEN q.total_over_price ELSE q.total_under_price END IS NOT NULL
              ORDER BY fetched_at_utc DESC LIMIT 1) fd ON TRUE
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


def grade(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for col in ("fd_line", "fd_price", "close_price", "sharp_close_line", "home_score", "away_score", "actual"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["clv"] = [None if pd.isna(c) else sharp_math.implied(c) - sharp_math.implied(p) for c, p in zip(df.close_price, df.fd_price)]
    ev_close = []
    for r in df.itertuples():
        fair_over = None
        if pd.notna(r.sharp_close_line):
            fair_over = sharp_math.over_probability_at(r.stat, float(r.sharp_close_line),
                                                       sharp_math.no_vig_over(r.sharp_close_over, r.sharp_close_under), float(r.fd_line))
        fair = None if fair_over is None else (fair_over if r.side == "over" else 1 - fair_over)
        ev_close.append(None if fair is None else sharp_math.ev(fair, r.fd_price))
    df["ev_at_sharp_close"] = ev_close
    actual = np.where(df.stat == "total", df.home_score + df.away_score, df.actual)
    diff = (actual - df.fd_line) * np.where(df.side == "over", 1, -1)
    df["result"] = np.where(pd.isna(actual) | (df.status != "final"), None, np.where(diff > 0, "win", np.where(diff < 0, "loss", "push")))
    payout = np.where(df.fd_price > 0, df.fd_price / 100, 100 / df.fd_price.abs())
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
                mean_alert_ev=float(pd.to_numeric(df.ev, errors="coerce").mean()))


def _tier(df: pd.DataFrame) -> pd.Series:
    return df["tier"].fillna("alert") if "tier" in df else pd.Series("alert", index=df.index)


def summarize(df: pd.DataFrame) -> dict:
    """Pass/fail is judged on pinged alerts (EV >= +3%). Logged gaps (+1% to +3%, no ping) are reported
    beside them: they show sooner whether FanDuel-vs-sharp gaps predict the close at all."""
    if df.empty:
        return dict(alerts=0, verdict="no graded alerts yet", logged=dict(alerts=0, weeks=0))
    tier = _tier(df)
    out = _stats(df[tier == "alert"])
    out["logged"] = _stats(df[tier == "logged"])
    weeks = out.get("weeks", 0)
    checks = dict(enough_alerts=out["alerts"] >= PASS["min_alerts"], enough_weeks=weeks >= PASS["min_weeks"],
                  beats_fd_close=(out.get("beat_fd_close") or 0) >= PASS["min_beat_close"],
                  positive_at_sharp_close=(out.get("mean_ev_at_sharp_close") or -1) >= PASS["min_sharp_close_ev"])
    out["checks"] = checks
    out["verdict"] = ("PASS: enable SHARP_EDGE_BETS_ENABLED with fixed small stakes" if all(checks.values())
                      else "no graded alerts yet" if not out["alerts"]
                      else "keep collecting" if not (checks["enough_alerts"] and checks["enough_weeks"])
                      else "FAIL: the alerted prices do not hold up at the close; do not bet them")
    return out


def markdown(summary: dict, df: pd.DataFrame) -> str:
    pct = lambda v: "-" if v is None else f"{v:.1%}"
    signed = lambda v: "-" if v is None else f"{v:+.2%}"
    a, g = summary, summary.get("logged") or {}
    lines = ["# NFL Sharp-Edge Alerts", "", f"Verdict: **{summary['verdict']}**", "",
             f"Alerts graded: {a.get('alerts', 0)} over {a.get('weeks', 0)} week(s) "
             f"(pass bar: {PASS['min_alerts']} over {PASS['min_weeks']}). Logged gaps graded: {g.get('alerts', 0)}.", "",
             "| Evidence (most reliable first) | Alerts (>= +3%, pinged) | Logged (+1% to +3%) | Pass bar (alerts) |",
             "|---|---:|---:|---:|",
             f"| Mean EV at the sharp close | {signed(a.get('mean_ev_at_sharp_close'))} | {signed(g.get('mean_ev_at_sharp_close'))} | >= {PASS['min_sharp_close_ev']:+.0%} |",
             f"| Share positive at the sharp close | {pct(a.get('positive_at_sharp_close'))} | {pct(g.get('positive_at_sharp_close'))} | - |",
             f"| Beat FanDuel's close | {pct(a.get('beat_fd_close'))} | {pct(g.get('beat_fd_close'))} | >= {PASS['min_beat_close']:.0%} |",
             f"| Mean FanDuel CLV (implied prob) | {signed(a.get('mean_fd_clv'))} | {signed(g.get('mean_fd_clv'))} | > 0 |",
             f"| EV claimed when seen | {signed(a.get('mean_alert_ev'))} | {signed(g.get('mean_alert_ev'))} | - |",
             f"| Flat ROI (settled: {a.get('settled', 0)} / {g.get('settled', 0)}) | {signed(a.get('roi'))} | {signed(g.get('roi'))} | noise for weeks |"]
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
