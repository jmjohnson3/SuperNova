"""Live NBA projections: player props, spreads, totals and team totals, refreshed when lineup news breaks.

A full run (task \\SuperNovaBets\\NBA-Props-Live, 6:45 AM, 11:00 AM and 3:30 PM MT) refreshes this season's game
logs, captures nba.com lineups and ESPN's injury report, fits the prop models (four LightGBM, one per stat)
and the spread/total model on every finished game, saves the fits for the day, and stores projections for
every rotation player and game today. A news run (`--news`, task NBA-News-Watch every 10 minutes) only
captures lineups and injuries; when anything changed for a team playing today (a starter, an inactive,
an injury status) it re-projects with the day's saved fits, so a late scratch moves his teammates'
projections within minutes.

Who is out: lineup Inactive, or Out on ESPN's injury report. Their own projections are not stored; their
minutes, usage and points move to the team's "missing rotation" inputs (nba_props.model.add_availability).

Rows go to nbaprops.projections and nbaprops.game_projections with the run time, so a sharp decision is
compared with the latest projection made before it. Research only: the props model does not beat
DraftKings' pre-game line on its own (docs/nba_props_model.md).

    python -m nba_props.live [--date 2026-10-24] [--no-refresh]
    python -m nba_props.live --news
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nba_props import availability, games as game_model, ingest, model

ROOT = Path(__file__).resolve().parents[2]
FITS = ROOT / "src" / "nba_props" / "models"          # gitignored: the day's fitted models
STATE = ROOT / "reports" / "nba_props_live_state.json"
ET = ZoneInfo("America/New_York")
POOL_DAYS = 21
MODEL_VERSION = "nba-props-v2-availability"
MARKET_STAT = {"points": "pts", "rebounds": "reb", "assists": "ast", "threes": "fg3m"}
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nbaprops.projections (
  run_at_utc TIMESTAMPTZ NOT NULL, game_date DATE NOT NULL, game_id TEXT NOT NULL, tip_utc TIMESTAMPTZ,
  player_id BIGINT NOT NULL, player_name TEXT NOT NULL, name_key TEXT NOT NULL, team TEXT NOT NULL,
  opponent TEXT NOT NULL, stat TEXT NOT NULL, mu NUMERIC NOT NULL, disp_a NUMERIC NOT NULL, disp_b NUMERIC NOT NULL,
  PRIMARY KEY (run_at_utc, player_id, game_id, stat));
ALTER TABLE nbaprops.projections ADD COLUMN IF NOT EXISTS starter BOOLEAN;
ALTER TABLE nbaprops.projections ADD COLUMN IF NOT EXISTS model_version TEXT;
ALTER TABLE nbaprops.projections ADD COLUMN IF NOT EXISTS trigger TEXT;
CREATE INDEX IF NOT EXISTS projections_lookup ON nbaprops.projections (game_date, name_key, stat, run_at_utc);
CREATE TABLE IF NOT EXISTS nbaprops.game_projections (
  run_at_utc TIMESTAMPTZ NOT NULL, game_date DATE NOT NULL, game_id TEXT NOT NULL, tip_utc TIMESTAMPTZ,
  home_team TEXT NOT NULL, away_team TEXT NOT NULL, pred_home NUMERIC, pred_away NUMERIC, pred_margin NUMERIC,
  pred_total NUMERIC, home_out TEXT, away_out TEXT, model_version TEXT, trigger TEXT,
  PRIMARY KEY (run_at_utc, game_id));
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


def latest_injuries(conn, before: datetime) -> pd.DataFrame:
    return pd.read_sql("""SELECT team, name_key, status FROM nbaprops.injury_snapshots
                          WHERE captured_at_utc = (SELECT max(captured_at_utc) FROM nbaprops.injury_snapshots
                                                   WHERE captured_at_utc <= %(t)s)""", conn, params=dict(t=before))


def declared_out(upcoming: pd.DataFrame, injuries: pd.DataFrame) -> set:
    out = set(zip(injuries[injuries.status.eq("Out")].team, injuries[injuries.status.eq("Out")].name_key))
    keys = upcoming.player_name.map(model.match_name)
    return {(g, p) for g, p, t, k in zip(upcoming.game_id, upcoming.player_id, upcoming.team, keys) if (t, k) in out}


def _fit_path(day: date) -> Path:
    return FITS / f"fits_{day.isoformat()}.joblib"


def fit_models(train: pd.DataFrame, cols: list[str], game_rows: pd.DataFrame) -> dict:
    import lightgbm as lgb
    fits = dict(cols=cols, props={}, version=MODEL_VERSION)
    for s in model.STATS:
        m = lgb.LGBMRegressor(**model.LGB_PARAMS).fit(train[cols], train[s])
        tail = train.iloc[-60000:]
        fits["props"][s] = (m, *model.dispersion(tail[s].to_numpy(), m.predict(tail[cols])))
    fits["games"] = game_model.stage2_fit(game_rows[game_rows.home_points.notna()])
    return fits


def project(day: date, now: datetime | None = None, refit: bool = True, trigger: str = "scheduled") -> dict:
    now = now or datetime.now(timezone.utc)
    sched = schedule(day)
    if sched.empty:
        return dict(day=str(day), games=0)
    with psycopg2.connect(PG_DSN) as conn:
        players = model.load_games(conn)
        lineups = model.final_lineups(conn)
        injuries = latest_injuries(conn, now)
    players = players[players.game_date < pd.Timestamp(day)]
    upcoming = scheduled_rows(players, sched, day)
    outs = declared_out(upcoming, injuries)
    f = model.features(pd.concat([players, upcoming], ignore_index=True), lineups, outs)
    cols = model.feature_columns(f)
    live = f[f.game_date.eq(pd.Timestamp(day)) & f.minutes.isna()].copy()
    status = lineups.set_index(["game_id", "player_id"]).roster_status.to_dict()
    live["is_out"] = [(g, p) in outs or status.get((g, p)) == "Inactive" for g, p in zip(live.game_id, live.player_id)]
    # team games for the spread/total model (unplayed ones carry today's missing rotation)
    g = game_model.team_games(pd.concat([players, upcoming], ignore_index=True),
                              f[["game_id", "team", "out_minutes", "out_usage", "out_pts"]])
    path = _fit_path(day)
    if refit or not path.exists():
        train = f[(f.game_date < pd.Timestamp(day)) & (f.minutes > 0) & (f.games_before >= 3)]
        rated = game_model.stage1(g[g.game_date >= pd.Timestamp(day) - pd.Timedelta(days=400)],
                                  pd.Timestamp(day) - pd.Timedelta(days=200))
        fits = fit_models(train, cols, rated)
        FITS.mkdir(parents=True, exist_ok=True)
        joblib.dump(fits, path)
    else:
        fits = joblib.load(path)
    # props
    play = live[~live.is_out]
    rows = []
    for s, (m, a, b) in fits["props"].items():
        rows.append(play[["game_id", "player_id", "player_name", "team", "opponent", "starter"]].assign(
            stat=s, mu=m.predict(play[fits["cols"]]), disp_a=a, disp_b=b))
    proj = pd.concat(rows, ignore_index=True).merge(sched[["game_id", "tip_utc"]], on="game_id")
    proj["name_key"] = proj.player_name.map(model.match_name)
    proj = proj.assign(game_date=day, run_at_utc=now, model_version=fits["version"], trigger=trigger)
    # games: today's ratings from finished games, then the saved stage 2
    today = game_model.stage1(g[g.game_date >= pd.Timestamp(day) - pd.Timedelta(days=400)], pd.Timestamp(day))
    today = game_model.predict_stage2(fits["games"], today[today.game_date.eq(pd.Timestamp(day))])
    names = live[live.is_out].groupby("game_id").apply(
        lambda x: {t: ", ".join(sorted(x[x.team.eq(t)].player_name)) for t in x.team.unique()})
    gp = today.merge(sched.rename(columns={"home": "home_team", "away": "away_team"}), on=["game_id", "home_team", "away_team"])
    gp["home_out"] = [(names.get(i) or {}).get(h, "") for i, h in zip(gp.game_id, gp.home_team)]
    gp["away_out"] = [(names.get(i) or {}).get(a, "") for i, a in zip(gp.game_id, gp.away_team)]
    gp = gp.assign(run_at_utc=now, model_version=fits["version"], trigger=trigger)
    save(proj, gp)
    return dict(day=str(day), games=int(len(sched)), players=int(play.player_id.nunique()), out=int(live.is_out.sum()),
                trigger=trigger, refit=bool(refit or not path.exists()))


def save(proj: pd.DataFrame, gp: pd.DataFrame) -> None:
    pcols = ["run_at_utc", "game_date", "game_id", "tip_utc", "player_id", "player_name", "name_key", "team",
             "opponent", "stat", "mu", "disp_a", "disp_b", "starter", "model_version", "trigger"]
    gcols = ["run_at_utc", "game_date", "game_id", "tip_utc", "home_team", "away_team", "pred_home", "pred_away",
             "pred_margin", "pred_total", "home_out", "away_out", "model_version", "trigger"]
    prow = proj[pcols].assign(starter=proj.starter.map({1.0: True, 0.0: False})).astype(object)
    prow = prow.where(prow.notna(), None)
    with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        psycopg2.extras.execute_values(cur, f"INSERT INTO nbaprops.projections ({', '.join(pcols)}) VALUES %s ON CONFLICT DO NOTHING",
                                       [tuple(r) for r in prow.itertuples(index=False)], page_size=5000)
        if len(gp):
            g = gp.assign(game_date=gp.game_date.dt.date)[gcols].astype(object)
            psycopg2.extras.execute_values(cur, f"INSERT INTO nbaprops.game_projections ({', '.join(gcols)}) VALUES %s ON CONFLICT DO NOTHING",
                                           [tuple(r) for r in g.where(g.notna(), None).itertuples(index=False)])


def side_probability(proj_row: dict, market: str, side: str, line: float) -> float | None:
    """Model probability for the decision's side, from a stored projection."""
    p = float(model.p_over(np.array([float(proj_row["mu"])]), np.array([line]),
                           float(proj_row["disp_a"]), float(proj_row["disp_b"]))[0])
    return p if side == "over" else 1.0 - p


def _state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def news(now: datetime | None = None) -> dict:
    """Capture lineups and injuries; re-project when anything changed for a team still to play today."""
    now = now or datetime.now(timezone.utc)
    day = now.astimezone(ET).date()
    sched = schedule(day)
    pending = sched[sched.tip_utc > pd.Timestamp(now)] if len(sched) else sched
    if pending.empty:
        return dict(day=str(day), status="no_games_pending")
    lineups, injuries = availability.fetch_lineups(day), availability.fetch_injuries()
    with psycopg2.connect(PG_DSN) as conn:
        availability.save(conn, lineups, injuries, now)
    teams = set(pending.home) | set(pending.away)
    digest = availability.news_digest(lineups, injuries, teams)
    state = _state()
    if state.get("day") == str(day) and state.get("digest") == digest:
        return dict(day=str(day), status="no_change")
    first = state.get("day") != str(day)
    result = project(day, now, refit=False, trigger="news") if _fit_path(day).exists() else dict(status="no_fits_yet")
    atomic_json(STATE, dict(day=str(day), digest=digest, updated_at=now.isoformat()))
    return dict(result, status="baseline" if first else "reprojected")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", type=date.fromisoformat, default=None)
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--news", action="store_true", help="capture lineups/injuries; re-project only if they changed")
    args = ap.parse_args()
    if args.news:
        print(news())
        return
    now = datetime.now(timezone.utc)
    day = args.date or now.astimezone(ET).date()
    if not args.no_refresh:
        ingest.ingest([ingest.current_start_year(day)])
        print(availability.capture(now))
    print(project(day, now))


if __name__ == "__main__":
    main()
