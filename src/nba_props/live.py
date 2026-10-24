"""Live NBA prop projections for the day's games, kept as a shadow filter beside the sharp-gap decisions.

Each run refreshes this season's game logs (nba_api), refits the walk-forward model's four LightGBM models
on every finished game, and projects every player who appeared for a team in its last 21 days, for each of
that team's games today (stats.nba.com ScoreboardV3). Projections are appended to nbaprops.projections with
the run time; a sharp decision is compared with the latest projection made before it (see
sharp_pipeline.report). They are research: the model does not beat DraftKings' pre-game line
(docs/nba_props_model.md), and lineup news after the run is not in it.

    python -m nba_props.live [--date 2026-10-24] [--no-refresh]
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nba_props import ingest, model

ET = ZoneInfo("America/New_York")
POOL_DAYS = 21
MARKET_STAT = {"points": "pts", "rebounds": "reb", "assists": "ast", "threes": "fg3m"}
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nbaprops.projections (
  run_at_utc TIMESTAMPTZ NOT NULL, game_date DATE NOT NULL, game_id TEXT NOT NULL, tip_utc TIMESTAMPTZ,
  player_id BIGINT NOT NULL, player_name TEXT NOT NULL, name_key TEXT NOT NULL, team TEXT NOT NULL,
  opponent TEXT NOT NULL, stat TEXT NOT NULL, mu NUMERIC NOT NULL, disp_a NUMERIC NOT NULL, disp_b NUMERIC NOT NULL,
  PRIMARY KEY (run_at_utc, player_id, game_id, stat));
CREATE INDEX IF NOT EXISTS projections_lookup ON nbaprops.projections (game_date, name_key, stat, run_at_utc);
"""


def schedule(day: date) -> pd.DataFrame:
    from nba_api.stats.endpoints import scoreboardv3
    games = scoreboardv3.ScoreboardV3(game_date=day.isoformat(), timeout=60).get_dict()["scoreboard"]["games"]
    return pd.DataFrame([dict(game_id=g["gameId"], home=g["homeTeam"]["teamTricode"], away=g["awayTeam"]["teamTricode"],
                              tip_utc=pd.Timestamp(g["gameTimeUTC"])) for g in games])


def scheduled_rows(games: pd.DataFrame, sched: pd.DataFrame, day: date) -> pd.DataFrame:
    """One unplayed row per pool player per game today: the player's latest team, seen in the last POOL_DAYS."""
    recent = games[games.game_date >= pd.Timestamp(day - timedelta(days=POOL_DAYS))]
    latest = recent.sort_values("game_date").drop_duplicates("player_id", keep="last")
    season = ingest.season_label(ingest.current_start_year(day))
    rows = []
    for g in sched.itertuples():
        for team, opp, home in ((g.home, g.away, True), (g.away, g.home, False)):
            for p in latest[latest.team.eq(team)].itertuples():
                rows.append(dict(game_id=g.game_id, player_id=p.player_id, season=season,
                                 season_type="Playoffs" if g.game_id.startswith("004") else "Regular Season",
                                 game_date=pd.Timestamp(day), player_name=p.player_name, team=team, opponent=opp,
                                 home=home))
    return pd.DataFrame(rows)


def project(day: date, now: datetime | None = None) -> pd.DataFrame:
    import lightgbm as lgb
    now = now or datetime.now(timezone.utc)
    sched = schedule(day)
    if sched.empty:
        return pd.DataFrame()
    with psycopg2.connect(PG_DSN) as conn:
        games = model.load_games(conn)
    games = games[games.game_date < pd.Timestamp(day)]
    upcoming = scheduled_rows(games, sched, day)
    f = model.features(pd.concat([games, upcoming], ignore_index=True))
    cols = model.feature_columns(f)
    train = f[(f.game_date < pd.Timestamp(day)) & (f.minutes > 0) & (f.games_before >= 3)]
    live = f[f.game_date.eq(pd.Timestamp(day)) & f.minutes.isna()].copy()
    out = []
    for s in model.STATS:
        fit = lgb.LGBMRegressor(**model.LGB_PARAMS).fit(train[cols], train[s])
        tail = train.iloc[-60000:]
        a, b = model.dispersion(tail[s].to_numpy(), fit.predict(tail[cols]))
        out.append(live[["game_id", "player_id", "player_name", "team", "opponent"]].assign(
            stat=s, mu=fit.predict(live[cols]), disp_a=a, disp_b=b))
    proj = pd.concat(out, ignore_index=True).merge(sched[["game_id", "tip_utc"]], on="game_id")
    proj["name_key"] = proj.player_name.map(model.match_name)
    proj["game_date"] = day
    proj["run_at_utc"] = now
    return proj


def save(proj: pd.DataFrame) -> int:
    cols = ["run_at_utc", "game_date", "game_id", "tip_utc", "player_id", "player_name", "name_key", "team",
            "opponent", "stat", "mu", "disp_a", "disp_b"]
    rows = [tuple(r) for r in proj[cols].astype(object).itertuples(index=False)]
    with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        if rows:
            psycopg2.extras.execute_values(cur, f"INSERT INTO nbaprops.projections ({', '.join(cols)}) VALUES %s "
                                                "ON CONFLICT DO NOTHING", rows, page_size=5000)
    return len(rows)


def side_probability(proj_row: dict, market: str, side: str, line: float) -> float | None:
    """Model probability for the decision's side, from a stored projection."""
    p = float(model.p_over(np.array([float(proj_row["mu"])]), np.array([line]),
                           float(proj_row["disp_a"]), float(proj_row["disp_b"]))[0])
    return p if side == "over" else 1.0 - p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", type=date.fromisoformat, default=None)
    ap.add_argument("--no-refresh", action="store_true")
    args = ap.parse_args()
    day = args.date or datetime.now(timezone.utc).astimezone(ET).date()
    if not args.no_refresh:
        ingest.ingest([ingest.current_start_year(day)])
    proj = project(day)
    n = save(proj) if len(proj) else 0
    print(dict(day=str(day), games=int(proj.game_id.nunique()) if len(proj) else 0, rows=n))


if __name__ == "__main__":
    main()
