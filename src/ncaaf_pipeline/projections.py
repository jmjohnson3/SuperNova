"""Daily NCAAF projections: score, spread, total and team totals for the day's FBS games, posted to #ncaa.

Research context, not bets: the backtest (docs/ncaaf_game_models.md) shows these ratings do not beat
the closing line. Each run refreshes this season's games and lines from CFBD (2 calls), refits the
ratings on completed games only, stores every projection in ncaaf.projections (so they can be graded),
and posts the slate once per day.

    python -m ncaaf_pipeline.projections [--date 2026-10-24] [--no-refresh] [--no-post] [--force]
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from ncaaf_pipeline import ingest, ratings

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "reports" / "ncaaf_projections_state.json"
ET = ZoneInfo("America/New_York")
MT = ZoneInfo("America/Denver")
MIN_GAMES = 6          # a team needs this many games in the last two seasons to be projected
BOOK_ORDER = ("DraftKings", "ESPN Bet", "Bovada")
DISCORD_LIMIT = 1900

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ncaaf.projections (
  game_id BIGINT NOT NULL, run_date DATE NOT NULL, generated_at_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
  season INTEGER, week INTEGER, home_team TEXT, away_team TEXT, start_date TIMESTAMPTZ,
  pred_home NUMERIC, pred_away NUMERIC, pred_margin NUMERIC, pred_total NUMERIC,
  market_provider TEXT, market_spread NUMERIC, market_total NUMERIC,
  PRIMARY KEY (game_id, run_date));
"""


def upcoming(conn, day: date) -> pd.DataFrame:
    """FBS games kicking off on `day` (Eastern), with the first available book's current line."""
    games = pd.read_sql("""SELECT game_id, season, week, season_type, start_date, neutral_site, home_team, away_team
                           FROM ncaaf.games
                           WHERE NOT completed AND start_date > now()
                             AND (start_date AT TIME ZONE 'America/New_York')::date = %(day)s
                           ORDER BY start_date, game_id""", conn, params=dict(day=day))
    lines = pd.read_sql("SELECT game_id, provider, spread, total FROM ncaaf.lines WHERE game_id = ANY(%(ids)s)",
                        conn, params=dict(ids=[int(g) for g in games.game_id]))
    market = {}
    for game_id, block in lines.groupby("game_id"):
        for book in BOOK_ORDER + tuple(sorted(set(block.provider) - set(BOOK_ORDER))):
            row = block[block.provider.eq(book) & block.spread.notna()]
            if len(row):
                r = row.iloc[0]
                market[game_id] = (book, float(r.spread), None if pd.isna(r.total) else float(r.total))
                break
    games["market_provider"] = games.game_id.map(lambda g: market.get(g, (None,))[0])
    games["market_spread"] = games.game_id.map(lambda g: market.get(g, (None, None))[1])
    games["market_total"] = games.game_id.map(lambda g: market.get(g, (None, None, None))[2])
    return games


def project(history: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return games
    season = int(games.season.iloc[0])
    week = 17.0 if games.season_type.iloc[0] == "postseason" else float(games.week.iloc[0])
    recent = history[history.season >= season - 1]
    counts = pd.concat([recent.home_team, recent.away_team]).value_counts()
    out = ratings.predict_block(history[history.season >= season - 2], games, season, week)
    out["projected"] = out.known & out.home_team.map(counts).ge(MIN_GAMES) & out.away_team.map(counts).ge(MIN_GAMES)
    return out


def save(conn, df: pd.DataFrame, day: date) -> None:
    rows = [(int(r.game_id), day, int(r.season), None if pd.isna(r.week) else int(r.week), r.home_team, r.away_team,
             r.start_date, round(float(r.pred_home), 2), round(float(r.pred_away), 2), round(float(r.pred_margin), 2),
             round(float(r.pred_total), 2), r.market_provider,
             None if pd.isna(r.market_spread) else float(r.market_spread),
             None if pd.isna(r.market_total) else float(r.market_total))
            for r in df[df.projected].itertuples()]
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        if rows:
            psycopg2.extras.execute_values(cur, """
                INSERT INTO ncaaf.projections (game_id, run_date, season, week, home_team, away_team, start_date,
                    pred_home, pred_away, pred_margin, pred_total, market_provider, market_spread, market_total)
                VALUES %s ON CONFLICT (game_id, run_date) DO UPDATE SET generated_at_utc = now(),
                    pred_home = EXCLUDED.pred_home, pred_away = EXCLUDED.pred_away, pred_margin = EXCLUDED.pred_margin,
                    pred_total = EXCLUDED.pred_total, market_provider = EXCLUDED.market_provider,
                    market_spread = EXCLUDED.market_spread, market_total = EXCLUDED.market_total""", rows)
    conn.commit()


def _spread_text(home: str, away: str, home_spread: float) -> str:
    """Home spread (negative = home favoured) as 'Favourite -x'."""
    if abs(home_spread) < 0.25:
        return "PK"
    return f"{home} {home_spread:+.1f}" if home_spread < 0 else f"{away} {-home_spread:+.1f}"


def game_line(r) -> str:
    kickoff = pd.Timestamp(r.start_date).tz_convert(MT).strftime("%I:%M %p").lstrip("0")
    at = "vs" if r.neutral_site else "@"
    model = f"{_spread_text(r.home_team, r.away_team, -r.pred_margin)}, total {r.pred_total:.1f}"
    if pd.notna(r.market_spread):
        total = "" if pd.isna(r.market_total) else f", {r.market_total:.1f}"
        market = f"{_spread_text(r.home_team, r.away_team, r.market_spread)}{total}"
    else:
        market = "no line yet"
    return (f"`{kickoff:>8}` **{r.away_team} {r.pred_away:.0f} {at} {r.home_team} {r.pred_home:.0f}**\n"
            f"      model {model} | market {market}")


def messages(df: pd.DataFrame, day: date) -> list[str]:
    shown = df[df.projected]
    skipped = int((~df.projected).sum())
    header = (f"**NCAAF projections, {day.strftime('%a %b')} {day.day}** ({len(shown)} games, times MT)\n"
              "_Ratings model context, not bets: it has not beaten the closing line in testing. "
              "Bets come only from the sharp-gap alerts._")
    if skipped:
        header += f"\n_{skipped} game(s) skipped: not enough FBS history for a team._"
    out, current = [], header
    for r in shown.itertuples():
        line = game_line(r)
        if len(current) + len(line) + 1 > DISCORD_LIMIT:
            out.append(current)
            current = line
        else:
            current += "\n" + line
    out.append(current)
    return out


def _state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def run(day: date, *, refresh: bool = True, post: bool = True, force: bool = False, poster=None) -> dict:
    state = _state()
    if post and not force and state.get("posted") == str(day):
        return dict(day=str(day), status="already_posted")
    refresh_error = None
    if refresh:
        try:
            ingest.ingest([day.year], endpoints=("games", "lines"),
                          season_types=("postseason",) if day.month == 1 else ("regular", "postseason") if day.month == 12 else ("regular",))
        except Exception as exc:  # CFBD has 502'd after its retries (25 Oct): project from stored data instead
            refresh_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            print(f"CFBD refresh failed, using stored games and lines: {refresh_error}", flush=True)
    with psycopg2.connect(PG_DSN) as conn:
        history = ratings.load(conn)
        games = upcoming(conn, day)
        df = project(history, games)
        if not df.empty:
            save(conn, df, day)
    result = dict(day=str(day), games=int(len(df)), projected=int(df.projected.sum()) if len(df) else 0,
                  refresh_error=refresh_error)
    if df.empty or not result["projected"]:
        return dict(result, status="no_games")
    texts = messages(df, day)
    if post:
        if poster is None:
            from sharp_pipeline.report import post as discord_post
            poster = lambda text: discord_post(text, "ncaaf")
        for text in texts:
            poster(text)
        state["posted"] = str(day)
        STATE.write_text(json.dumps(state), encoding="utf-8")
    else:
        print("\n\n".join(texts))
    return dict(result, status="posted" if post else "printed", messages=len(texts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", type=date.fromisoformat, default=None)
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--no-post", action="store_true")
    ap.add_argument("--force", action="store_true", help="post even if today's slate was already posted")
    args = ap.parse_args()
    day = args.date or datetime.now(timezone.utc).astimezone(ET).date()
    print(run(day, refresh=not args.no_refresh, post=not args.no_post, force=args.force))


if __name__ == "__main__":
    main()
