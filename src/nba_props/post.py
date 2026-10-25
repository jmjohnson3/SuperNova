"""Post the day's NBA projections to #nba once, about three hours before the first tip.

Per game: projected score (team totals), model spread and total beside DraftKings' current line and
Pinnacle's (one bulk Odds API call, 3 credits), who is out, and the top players' points, rebounds,
assists and threes. Uses the latest stored run, so a late scratch caught by the news watch is in it.
Research context, not bets: the models do not beat the closing line on their own; bets come from the
sharp-gap scanner.

    python -m nba_props.post [--force] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg2
import requests

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "reports" / "nba_props_post_state.json"
ET, MT = ZoneInfo("America/New_York"), ZoneInfo("America/Denver")
POST_HOURS_BEFORE = 3.0
TOP_PLAYERS = 4
LIMIT = 1900


def latest(conn, day) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_sql("""SELECT * FROM nbaprops.game_projections WHERE game_date = %(d)s AND run_at_utc =
                           (SELECT max(run_at_utc) FROM nbaprops.game_projections WHERE game_date = %(d)s)""",
                        conn, params=dict(d=day))
    props = pd.read_sql("""SELECT * FROM nbaprops.projections WHERE game_date = %(d)s AND run_at_utc =
                           (SELECT max(run_at_utc) FROM nbaprops.projections WHERE game_date = %(d)s)""",
                        conn, params=dict(d=day))
    return games, props


def market_lines(session=requests) -> dict:
    """{(home abbr, away abbr): {book: (home spread, total)}} from one bulk call."""
    from nba_api.stats.static import teams
    from nfl_pipeline.crawler_oddsapi import OddsCrawlerConfig
    abbr = {t["full_name"]: t["abbreviation"] for t in teams.get_teams()}
    key = OddsCrawlerConfig().oddsapi_key
    if not key:
        return {}
    r = session.get("https://api.the-odds-api.com/v4/sports/basketball_nba/odds", timeout=30,
                    params=dict(apiKey=key, bookmakers="draftkings,pinnacle", markets="spreads,totals", oddsFormat="american"))
    if not r.ok:
        return {}
    out = {}
    for e in r.json():
        h, a = abbr.get(e["home_team"]), abbr.get(e["away_team"])
        for b in e.get("bookmakers", []):
            spread = total = None
            for m in b.get("markets", []):
                for o in m.get("outcomes", []):
                    if m["key"] == "spreads" and o["name"] == e["home_team"]:
                        spread = float(o["point"])
                    if m["key"] == "totals" and o["name"] == "Over":
                        total = float(o["point"])
            out.setdefault((h, a), {})[b["key"]] = (spread, total)
    return out


def _spread(home: str, away: str, home_spread: float | None) -> str:
    if home_spread is None or pd.isna(home_spread):
        return "-"
    if abs(home_spread) < 0.25:
        return "PK"
    return f"{home} {home_spread:+.1f}" if home_spread < 0 else f"{away} {-home_spread:+.1f}"


def game_block(g, props: pd.DataFrame, lines: dict) -> str:
    tip = pd.Timestamp(g.tip_utc).tz_convert(MT).strftime("%I:%M %p").lstrip("0")
    head = f"`{tip:>8}` **{g.away_team} {g.pred_away:.0f} @ {g.home_team} {g.pred_home:.0f}**"
    model = f"model {_spread(g.home_team, g.away_team, -float(g.pred_margin))}, total {float(g.pred_total):.1f}"
    mk = []
    for book, label in (("draftkings", "DK"), ("pinnacle", "Pinn")):
        s, t = (lines.get((g.home_team, g.away_team)) or {}).get(book, (None, None))
        if s is not None or t is not None:
            mk.append(f"{label} {_spread(g.home_team, g.away_team, s)}, {t if t is not None else '-'}")
    lines_out = [head, f"      {model}" + (f" | {' | '.join(mk)}" if mk else "")]
    outs = "; ".join(f"{t}: {n}" for t, n in ((g.away_team, g.away_out), (g.home_team, g.home_out)) if n)
    if outs:
        lines_out.append(f"      out: {outs}")
    wide = props.pivot_table(index=["player_name", "team"], columns="stat", values="mu").reset_index()
    for team in (g.away_team, g.home_team):
        top = wide[wide.team.eq(team)].sort_values("pts", ascending=False).head(TOP_PLAYERS)
        cells = [f"{r.player_name.split(' ', 1)[-1]} {r.pts:.1f}p {r.reb:.1f}r {r.ast:.1f}a {r.fg3m:.1f}x3" for r in top.itertuples()]
        if cells:
            lines_out.append(f"      {team}: " + " · ".join(cells))
    return "\n".join(lines_out)


def messages(games: pd.DataFrame, props: pd.DataFrame, lines: dict, day) -> list[str]:
    run = pd.Timestamp(games.run_at_utc.max()).tz_convert(MT).strftime("%I:%M %p").lstrip("0")
    header = (f"**NBA projections, {day.strftime('%a %b')} {day.day}** ({len(games)} games, times MT, as of {run})\n"
              "_Model context, not bets: it has not beaten the closing line in testing. Bets come only from the "
              "sharp-gap alerts. p/r/a/x3 = points, rebounds, assists, threes._")
    out, cur = [], header
    for g in games.sort_values("tip_utc").itertuples():
        block = game_block(g, props[props.game_id.eq(g.game_id)], lines)
        if len(cur) + len(block) + 2 > LIMIT:
            out.append(cur)
            cur = block
        else:
            cur += "\n" + block
    out.append(cur)
    return out


def _state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def run(now: datetime | None = None, force: bool = False, dry_run: bool = False, poster=None) -> dict:
    now = now or datetime.now(timezone.utc)
    day = now.astimezone(ET).date()
    state = _state()
    if not force and state.get("posted") == str(day):
        return dict(status="already_posted")
    with psycopg2.connect(PG_DSN) as conn:
        games, props = latest(conn, day)
    if games.empty:
        return dict(status="no_projections")
    first_tip = pd.to_datetime(games.tip_utc, utc=True).min()
    if not force and first_tip - pd.Timestamp(now) > timedelta(hours=POST_HOURS_BEFORE):
        return dict(status="too_early", first_tip=str(first_tip))
    if not force and first_tip <= pd.Timestamp(now):
        return dict(status="slate_started")
    texts = messages(games, props, market_lines() if not dry_run else {}, day)
    if dry_run:
        print("\n\n".join(texts))
        return dict(status="dry_run", messages=len(texts))
    if poster is None:
        from sharp_pipeline.report import post as discord_post
        poster = lambda text: discord_post(text, "nba")
    for text in texts:
        poster(text)
    atomic_json(STATE, dict(posted=str(day), at=now.isoformat()))
    return dict(status="posted", messages=len(texts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(run(force=args.force, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
