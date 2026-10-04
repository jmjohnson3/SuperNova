"""Backtest the sharp-gap idea on past seasons with The Odds API's historical snapshots.

Question: when FanDuel's price sits off Pinnacle's no-vig fair price shortly before kickoff, does the
gap hold up? For sampled games it pulls FanDuel + Pinnacle/exchanges at T-90, T-45, T-15 and at kickoff
(T-2), prices every FanDuel side against the sharp fair line (nearby lines converted, as in sharp_watch),
and grades each side by:
- EV at the sharp close: the side's FanDuel price valued at the sharp book's kickoff snapshot (most reliable),
- FanDuel CLV: FanDuel's kickoff no-vig price moved to the bet's line vs its no-vig price then
  (FanDuel moves props by line at a fixed -112/-112, so a same-line comparison misses most moves),
- the result and flat ROI (noisy).

Historical calls cost ~10 credits per market per snapshot. Every paid response is stored in
raw.nfl_api_responses (endpoint nfl_player_props_history, which no parser reads) and reused on rerun.
The run stops before the remaining balance would drop below --min-remaining.

  python -m nfl_pipeline.modeling.sharp_gap_backtest --season 2025 --games 30 --markets player_reception_yds
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import requests
from scipy.stats import norm

from nfl_pipeline import sharp_math
from nfl_pipeline.crawler_oddsapi import OddsCrawlerConfig, _save_payload
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.markets import STAT_BY_MARKET, normalize_name, normalize_team
from nfl_pipeline.sharp_watch import BOOKMAKERS, SHARP_BOOKS, _pairs, find_edges

ROOT = Path(__file__).resolve().parents[3]
BASE = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl"
ENDPOINT = "nfl_player_props_history"
PROVIDER = "oddsapi_history"
SNAPSHOTS = (90, 45, 15)   # minutes before kickoff when a gap could be acted on
CLOSE_MINUTES = 2          # "close": last snapshot before kickoff
THRESHOLDS = (0.01, 0.03)


class Budget(Exception):
    pass


class Client:
    def __init__(self, conn, key: str, min_remaining: int, session=requests):
        self.conn, self.key, self.min_remaining, self.session = conn, key, min_remaining, session
        self.remaining: int | None = None
        self.spent = 0
        self.cached = 0

    def get(self, path: str, params: dict, *, cost: int, as_of) -> dict:
        cache_url = f"{BASE}/{path}?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        with self.conn.cursor() as cur:
            cur.execute("SELECT payload FROM raw.nfl_api_responses WHERE provider=%s AND endpoint=%s AND url=%s "
                        "ORDER BY fetched_at_utc DESC LIMIT 1", (PROVIDER, ENDPOINT, cache_url))
            row = cur.fetchone()
        if row:
            self.cached += 1
            return row[0]
        if self.remaining is not None and self.remaining - cost < self.min_remaining:
            raise Budget(f"stopping: {self.remaining} credits left, floor {self.min_remaining}")
        for attempt in range(4):  # transient 5xx/429: back off and retry (unanswered calls are not charged)
            r = self.session.get(f"{BASE}/{path}", params=dict(params, apiKey=self.key), timeout=30)
            if r.status_code < 500 and r.status_code != 429:
                break
            time.sleep(2 ** (attempt + 1))
        r.raise_for_status()
        self.remaining = int(r.headers.get("x-requests-remaining") or 0)
        self.spent += int(r.headers.get("x-requests-last") or cost)
        payload = r.json()
        _save_payload(self.conn, endpoint=ENDPOINT, snapshot_role="history", as_of_date=as_of,
                      url=cache_url, payload=payload, provider=PROVIDER)
        self.conn.commit()
        return payload


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sample_games(conn, season: int, n: int, seed: int = 7) -> list[dict]:
    """Sunday regular-season games spread across weeks (deterministic)."""
    with conn.cursor() as cur:
        cur.execute("""SELECT game_id, week, game_date_et, start_ts_utc, home_team_abbr, away_team_abbr, home_score, away_score
            FROM raw.nfl_games WHERE season=%s AND season_type='REG' AND status='final'
              AND extract(dow FROM game_date_et)=0 AND week BETWEEN 2 AND 17 ORDER BY start_ts_utc, game_id""", (season,))
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    by_week = defaultdict(list)
    for r in rows:
        by_week[r["week"]].append(r)
    rng = random.Random(seed)
    for games in by_week.values():
        rng.shuffle(games)
    picked, i = [], 0
    while len(picked) < n and any(by_week.values()):  # round-robin across weeks
        week = sorted(by_week)[i % len(by_week)]
        if by_week[week]:
            picked.append(by_week[week].pop())
        i += 1
    return sorted(picked, key=lambda g: g["start_ts_utc"])


def find_events(client: Client, games: list[dict]) -> dict[str, str]:
    """game_id -> Odds API event id, one events call per kickoff date."""
    out = {}
    for day in sorted({g["game_date_et"] for g in games}):
        first = min(g["start_ts_utc"] for g in games if g["game_date_et"] == day)
        payload = client.get("events", {"date": _iso(first - timedelta(hours=3))}, cost=1, as_of=day)
        idx = {(normalize_team(e["away_team"]), normalize_team(e["home_team"])): e["id"] for e in payload.get("data", [])}
        for g in games:
            if g["game_date_et"] == day and (g["away_team_abbr"], g["home_team_abbr"]) in idx:
                out[g["game_id"]] = idx[(g["away_team_abbr"], g["home_team_abbr"])]
    return out


def fetch_snapshots(client: Client, game: dict, event_id: str, market: str) -> dict[int, dict]:
    snaps = {}
    for minutes in (*SNAPSHOTS, CLOSE_MINUTES):
        params = {"date": _iso(game["start_ts_utc"] - timedelta(minutes=minutes)), "markets": market,
                  "bookmakers": ",".join(BOOKMAKERS), "oddsFormat": "american"}
        payload = client.get(f"events/{event_id}/odds", params, cost=10, as_of=game["game_date_et"])
        snaps[minutes] = payload.get("data") or {}
    return snaps


MAX_CLOSE_SHIFT = {"receiving_yards": 10.0, "rushing_yards": 10.0, "passing_yards": 20.0, "receptions": 1.0, "total": 3.0}


def _shift(stat: str, from_line: float, p_over: float | None, to_line: float) -> float | None:
    """P(over to_line) from a no-vig P(over from_line). FanDuel moves props by line, not price, so the
    close must be compared at the bet's line. Wider than sharp_math's gap limit (it measures movement)."""
    model = sharp_math.LINE_MODEL.get(stat)
    if model is None or p_over is None or not 0.0 < p_over < 1.0:
        return None
    if abs(to_line - from_line) < 1e-9:
        return p_over
    if abs(to_line - from_line) > MAX_CLOSE_SHIFT.get(stat, 0.0) or model["max_gap"] == 0.0:
        return None
    mean = from_line + model["sigma"] * norm.ppf(p_over)
    return float(1.0 - norm.cdf(to_line, loc=mean, scale=model["sigma"]))


def _player_pairs(snapshot: dict, book: str, stat: str, who: str) -> list[tuple[float, float]]:
    """[(line, no-vig P(over))] for one player (or 'game') at one book."""
    for b in snapshot.get("bookmakers", []):
        if b["key"] == book:
            return [(line, sharp_math.no_vig_over(q["over"], q["under"])) for (s_, w, line), q in _pairs(b).items()
                    if s_ == stat and w == who and q.get("over") is not None and q.get("under") is not None]
    return []


def _side(p_over: float | None, side: str) -> float | None:
    return None if p_over is None else (p_over if side == "over" else 1.0 - p_over)


def sides(game: dict, snaps: dict[int, dict]) -> list[dict]:
    """Every FanDuel side that has a sharp fair price, at each actionable snapshot, graded at the close."""
    close = snaps.get(CLOSE_MINUTES) or {}
    out = []
    for minutes in SNAPSHOTS:
        snap = snaps.get(minutes) or {}
        for e in find_edges(snap, min_ev=-1.0):
            who = e["player_norm"] or "game"
            # Sharp close moved to the bet's line (nearest sharp line first, preferred book order).
            fair_close = None
            for book in SHARP_BOOKS:
                cands = sorted(_player_pairs(close, book, e["stat"], who), key=lambda lp: abs(lp[0] - e["line"]))
                if cands:
                    fair_close = sharp_math.over_probability_at(e["stat"], cands[0][0], cands[0][1], e["line"])
                    if fair_close is not None:
                        break
            # FanDuel no-vig at the bet's line, then and at the close (usually a moved line at the same price).
            fd_then = dict(_player_pairs(snap, "fanduel", e["stat"], who)).get(e["line"])
            fd_close_pairs = _player_pairs(close, "fanduel", e["stat"], who)
            fd_close = None
            if fd_close_pairs:
                line, p = min(fd_close_pairs, key=lambda lp: abs(lp[0] - e["line"]))
                fd_close = _shift(e["stat"], line, p, e["line"])
                fd_close_line = line
            then_p, close_p = _side(fd_then, e["side"]), _side(fd_close, e["side"])
            out.append(dict(game_id=game["game_id"], week=game["week"], minutes=minutes, stat=e["stat"],
                            player=e["player"], player_norm=e["player_norm"], side=e["side"], line=e["line"],
                            price=e["price"], ev=e["ev"], line_gap=e["line_gap"], sharp_book=e["sharp_book"],
                            fd_close_line=fd_close_line if fd_close_pairs else None,
                            ev_at_sharp_close=sharp_math.ev(_side(fair_close, e["side"]), e["price"]) if fair_close is not None else None,
                            fd_clv=None if then_p is None or close_p is None else close_p - then_p))
    return out


def attach_results(conn, df: pd.DataFrame, season: int) -> pd.DataFrame:
    if df.empty:
        return df
    with conn.cursor() as cur:
        cur.execute("""SELECT DISTINCT ON (l.game_id, r.player_name_norm) l.game_id, r.player_name_norm,
                l.receiving_yards, l.rushing_yards, l.passing_yards, l.receptions
            FROM raw.nfl_player_gamelogs l JOIN raw.nfl_rosters r ON r.player_id=l.player_id AND r.season=l.season
            WHERE l.season=%s AND l.game_id = ANY(%s)""", (season, sorted(df.game_id.unique())))
        stats = {(g, n): dict(receiving_yards=a, rushing_yards=b, passing_yards=c, receptions=d) for g, n, a, b, c, d in cur.fetchall()}
    strip = lambda s: " ".join(w for w in s.split() if w not in {"jr", "sr", "ii", "iii", "iv", "v"})
    stats.update({(g, strip(n)): v for (g, n), v in list(stats.items())})
    actual = []
    for r in df.itertuples():
        row = stats.get((r.game_id, r.player_norm)) or stats.get((r.game_id, strip(r.player_norm or "")))
        actual.append(None if row is None or row.get(r.stat) is None else float(row[r.stat]))
    df = df.assign(actual=actual)
    diff = (df.actual - df.line) * np.where(df.side == "over", 1, -1)
    df["result"] = np.where(df.actual.isna(), None, np.where(diff > 0, "win", np.where(diff < 0, "loss", "push")))
    payout = np.where(df.price > 0, df.price / 100, 100 / df.price.abs())
    df["units"] = np.where(df.result == "win", payout, np.where(df.result == "loss", -1.0, np.where(df.result == "push", 0.0, np.nan)))
    return df


def summarize(df: pd.DataFrame) -> list[dict]:
    rows = []
    if df.empty:
        return rows
    for (stat, minutes), g in df.groupby(["stat", "minutes"]):
        for label, sub in (("all sides", g), *((f">= {t:+.0%}", g[g.ev >= t]) for t in THRESHOLDS)):
            evc = pd.to_numeric(sub.ev_at_sharp_close, errors="coerce").dropna()
            clv = pd.to_numeric(sub.fd_clv, errors="coerce").dropna()
            units = pd.to_numeric(sub.units, errors="coerce").dropna()
            rows.append(dict(stat=stat, minutes=int(minutes), bucket=label, n=int(len(sub)),
                             share=float(len(sub) / len(g)) if len(g) else None,
                             mean_ev=float(sub.ev.mean()) if len(sub) else None,
                             mean_ev_at_sharp_close=float(evc.mean()) if len(evc) else None,
                             positive_at_sharp_close=float((evc > 0).mean()) if len(evc) else None,
                             beat_fd_close=float((clv > 1e-9).mean()) if len(clv) else None,
                             fd_close_moved=float((clv.abs() > 1e-9).mean()) if len(clv) else None,
                             settled=int(len(units)), roi=float(units.mean()) if len(units) else None,
                             games=int(sub.game_id.nunique())))
    return rows


def markdown(meta: dict, rows: list[dict]) -> str:
    pct = lambda v: "-" if v is None else f"{v:.1%}"
    sgn = lambda v: "-" if v is None else f"{v:+.2%}"
    lines = ["# NFL Sharp-Gap Backtest", "",
             f"Season {meta['season']}, {meta['games']} sampled Sunday games, markets {', '.join(meta['markets'])}. "
             f"Credits spent this run: {meta['credits_spent']} (cached responses reused: {meta['cached']}); "
             f"remaining: {meta['credits_remaining']}.", "",
             "Each FanDuel side with a sharp fair price, at each snapshot. EV is against the sharp no-vig line "
             "at that time; 'EV at sharp close' values the same price at the sharp kickoff snapshot.", "",
             "| Stat | T- | Bucket | Sides | Share | EV when seen | EV at sharp close | +ve at sharp close | Beat FD close | FD close moved | ROI (settled) |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['stat']} | {r['minutes']} | {r['bucket']} | {r['n']} | {pct(r['share'])} | {sgn(r['mean_ev'])} | "
                     f"{sgn(r['mean_ev_at_sharp_close'])} | {pct(r['positive_at_sharp_close'])} | {pct(r['beat_fd_close'])} | "
                     f"{pct(r['fd_close_moved'])} | {sgn(r['roi'])} ({r['settled']}) |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--markets", default="player_reception_yds")
    ap.add_argument("--min-remaining", type=int, default=15000, help="never let the balance fall below this")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    key = OddsCrawlerConfig().oddsapi_key
    conn = psycopg2.connect(PG_DSN)
    client = Client(conn, key, args.min_remaining)
    games = sample_games(conn, args.season, args.games, args.seed)
    all_sides, stopped, failed = [], None, []
    try:
        events = find_events(client, games)
        for game in games:
            if game["game_id"] not in events:
                continue
            for market in markets:
                try:
                    all_sides += sides(game, fetch_snapshots(client, game, events[game["game_id"]], market))
                except requests.HTTPError as exc:  # skip this game; saved snapshots are reused on rerun
                    failed.append(f"{game['game_id']}/{market}: {exc.response.status_code if exc.response is not None else '?'}")
    except Budget as exc:
        stopped = str(exc)
    df = attach_results(conn, pd.DataFrame(all_sides), args.season)
    rows = summarize(df)
    meta = dict(season=args.season, games=int(df.game_id.nunique()) if not df.empty else 0, sampled=len(games),
                markets=markets, credits_spent=client.spent, cached=client.cached, credits_remaining=client.remaining,
                stopped=stopped, failed=failed, built_at=datetime.now(timezone.utc).isoformat())
    atomic_json(ROOT / "reports" / "nfl_sharp_gap_backtest_latest.json", dict(meta=meta, summary=rows))
    (ROOT / "reports" / "nfl_sharp_gap_backtest_latest.md").write_text(markdown(meta, rows), encoding="utf-8")
    df.to_csv(ROOT / "reports" / "nfl_sharp_gap_backtest_sides.csv", index=False)
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
