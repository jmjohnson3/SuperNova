"""NBA spreads, totals and team totals: team ratings plus who is missing, tested against DraftKings.

Stage 1 (as of each day, earlier games only): decayed ridge ratings, points scored = mean + offense[team]
+ defense[opponent] + home court, fitted twice (light shrinkage for the margin, heavy for the total, as
for NCAAF), so spread, total and both team totals agree. Weight halves every HALF_LIFE_DAYS; the previous
season counts as SEASON_GAP_DAYS older.

Stage 2: ridge on the stage-1 margin/total plus each side's missing rotation (minutes, usage, points of
players out, from nba_props.model.add_availability), rest and back-to-backs, refit monthly on earlier games.

The test (reports/nba_games_model_latest.md): DraftKings spread and total from odds.nba_game_lines, the last
snapshot before tip (about a third of the 01:00 UTC backfill snapshots are after tip, i.e. live, and are
excluded). Model MAE vs the line's, ATS / over-under record when the model disagrees by k points, and the
model's weight in a regression of the result on line and model.

    python -m nba_props.games [--test-from 2023-11-01]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import sparse
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import atomic_json
from nba_props import model as props

ROOT = Path(__file__).resolve().parents[2]
HALF_LIFE_DAYS = 45.0
SEASON_GAP_DAYS = 120.0
ALPHA_MARGIN = 2.0
ALPHA_TOTAL = 20.0
EDGES = (1, 2, 3, 4, 5, 7)
STAGE2 = ["r_margin", "r_total", "home_out_minutes", "home_out_usage", "home_out_pts", "away_out_minutes",
          "away_out_usage", "away_out_pts", "home_rest", "away_rest", "home_b2b", "away_b2b"]


def team_games(players: pd.DataFrame, avail: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per game: home/away team, points (NaN when unplayed), rest and missing rotation."""
    tg = players.groupby(["game_id", "game_date", "season", "team", "opponent", "home"], as_index=False).pts.sum(min_count=1)
    tg = tg.sort_values(["team", "game_date"])
    tg["rest"] = tg.groupby("team").game_date.diff().dt.days.clip(upper=5)
    home, away = tg[tg.home.astype(bool)], tg[~tg.home.astype(bool)]
    g = home.merge(away[["game_id", "team", "pts", "rest"]], on="game_id", suffixes=("_h", "_a"))
    g = g.rename(columns={"team_h": "home_team", "team_a": "away_team", "pts_h": "home_points", "pts_a": "away_points",
                          "rest_h": "home_rest", "rest_a": "away_rest"}).drop(columns=["opponent", "home"])
    g["home_b2b"], g["away_b2b"] = g.home_rest.eq(1).astype(float), g.away_rest.eq(1).astype(float)
    if avail is not None:
        a = avail.drop_duplicates(["game_id", "team"]).set_index(["game_id", "team"])
        for side in ("home", "away"):
            for c in ("out_minutes", "out_usage", "out_pts"):
                g[f"{side}_{c}"] = [a[c].get((gid, t), 0.0) for gid, t in zip(g.game_id, g[f"{side}_team"])]
    return g.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def fit(history: pd.DataFrame, day: pd.Timestamp, alpha: float) -> dict:
    teams = sorted(set(history.home_team) | set(history.away_team))
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(history), len(teams)
    rows = np.arange(2 * n)
    att = np.concatenate([history.home_team.map(idx), history.away_team.map(idx)])
    dfn = np.concatenate([history.away_team.map(idx), history.home_team.map(idx)])
    X = sparse.hstack([sparse.csr_matrix((np.ones(2 * n), (rows, att)), shape=(2 * n, k)),
                       sparse.csr_matrix((np.ones(2 * n), (rows, dfn)), shape=(2 * n, k)),
                       sparse.csr_matrix(np.concatenate([np.ones(n), -np.ones(n)]).reshape(-1, 1) / 2)]).tocsr()
    y = np.concatenate([history.home_points.to_numpy(float), history.away_points.to_numpy(float)])
    age = (day - history.game_date).dt.days.to_numpy(float)
    age = age + np.where(history.season.to_numpy() != history.season.iloc[-1], SEASON_GAP_DAYS, 0.0)
    w = np.tile(0.5 ** (age / HALF_LIFE_DAYS), 2)
    m = Ridge(alpha=alpha).fit(X, y, sample_weight=w)
    return dict(idx=idx, off=m.coef_[:k], de=m.coef_[k:2 * k], hca=m.coef_[-1], mean=m.intercept_)


def rate(r: dict, home: pd.Series, away: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    get = lambda arr, teams: np.array([arr[r["idx"][t]] if t in r["idx"] else 0.0 for t in teams])
    h = r["mean"] + get(r["off"], home) + get(r["de"], away) + r["hca"] / 2
    a = r["mean"] + get(r["off"], away) + get(r["de"], home) - r["hca"] / 2
    return h, a


def stage1(g: pd.DataFrame, first_day: pd.Timestamp) -> pd.DataFrame:
    """Ratings as of each game day, from games before it (two seasons back)."""
    parts = []
    for day, block in g[g.game_date >= first_day].groupby("game_date"):
        hist = g[(g.game_date < day) & g.home_points.notna()]
        hist = hist[hist.season.isin(sorted(hist.season.unique())[-2:])]
        if len(hist) < 300:
            continue
        hm, am = rate(fit(hist, day, ALPHA_MARGIN), block.home_team, block.away_team)
        ht, at = rate(fit(hist, day, ALPHA_TOTAL), block.home_team, block.away_team)
        parts.append(block.assign(r_margin=hm - am, r_total=ht + at))
    return pd.concat(parts, ignore_index=True)


def stage2_fit(train: pd.DataFrame):
    cols = [c for c in STAGE2 if c in train]
    mm = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 21))).fit(
        train[cols].fillna(0), train.home_points - train.away_points)
    tm = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 21))).fit(
        train[cols].fillna(0), train.home_points + train.away_points)
    return cols, mm, tm


def predict_stage2(models, frame: pd.DataFrame) -> pd.DataFrame:
    cols, mm, tm = models
    out = frame.copy()
    out["pred_margin"] = mm.predict(frame[cols].fillna(0))
    out["pred_total"] = tm.predict(frame[cols].fillna(0))
    out["pred_home"] = (out.pred_total + out.pred_margin) / 2
    out["pred_away"] = (out.pred_total - out.pred_margin) / 2
    return out


def walk_forward(r: pd.DataFrame, test_from: str) -> pd.DataFrame:
    played = r[r.home_points.notna()]
    parts = []
    for month in pd.period_range(pd.Timestamp(test_from), played.game_date.max(), freq="M"):
        train = played[played.game_date < month.start_time]
        test = played[(played.game_date >= month.start_time) & (played.game_date <= month.end_time)]
        if test.empty or len(train) < 500:
            continue
        parts.append(predict_stage2(stage2_fit(train), test))
    return pd.concat(parts, ignore_index=True)


# --- market test -----------------------------------------------------------------------------------
def load_lines(conn) -> pd.DataFrame:
    from nba_api.stats.static import teams
    abbr = {t["full_name"]: t["abbreviation"] for t in teams.get_teams()}
    abbr["LA Clippers"] = "LAC"
    lines = pd.read_sql("""
        SELECT home_team, away_team, commence_time_utc AS commence, spread_home_points AS spread, total_points AS total,
               spread_home_price, spread_away_price, total_over_price, total_under_price,
               CASE WHEN fetched_at_utc - as_of_date::timestamptz > interval '3 days'
                    THEN (as_of_date::timestamp + interval '1 hour') AT TIME ZONE 'UTC' ELSE fetched_at_utc END AS snapshot
        FROM odds.nba_game_lines WHERE bookmaker_key = 'draftkings' AND spread_home_points IS NOT NULL""", conn)
    lines["commence"] = pd.to_datetime(lines.commence, utc=True)
    lines["snapshot"] = pd.to_datetime(lines.snapshot, utc=True)
    lines = lines[(lines.snapshot < lines.commence) & (lines.commence - lines.snapshot <= pd.Timedelta(hours=30))]
    lines["game_date"] = lines.commence.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    lines["home_team"], lines["away_team"] = lines.home_team.map(abbr), lines.away_team.map(abbr)
    return lines.sort_values("snapshot").drop_duplicates(["game_date", "home_team", "away_team"], keep="last")


def ats(d: pd.DataFrame, kind: str) -> list[dict]:
    if kind == "spread":
        edge, result = d.pred_margin + d.spread, (d.home_points - d.away_points) + d.spread
    else:
        edge, result = d.pred_total - d.total, (d.home_points + d.away_points) - d.total
    rows = []
    for k in EDGES:
        sel = edge.abs().ge(k) & result.ne(0)
        hit = (np.sign(edge[sel]) == np.sign(result[sel]))
        rows.append(dict(edge=k, picks=int(sel.sum()), hit_rate=float(hit.mean()) if sel.any() else None,
                         by_season={str(s): round(float(hit[d.season[sel] == s].mean()), 3)
                                    for s in sorted(d.season[sel].unique())}))
    return rows


def blend(d: pd.DataFrame, pred: str, market: pd.Series, actual: pd.Series) -> dict:
    X = np.column_stack([np.ones(len(d)), market.to_numpy(float), d[pred].to_numpy(float)])
    coef, *_ = np.linalg.lstsq(X, actual.to_numpy(float), rcond=None)
    resid = actual.to_numpy(float) - X @ coef
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=3))
    return dict(market_weight=float(coef[1]), model_weight=float(coef[2]), model_se=float(se[2]))


def evaluate(pred: pd.DataFrame, lines: pd.DataFrame) -> dict:
    d = pred.merge(lines[["game_date", "home_team", "away_team", "spread", "total"]], on=["game_date", "home_team", "away_team"])
    d[["spread", "total"]] = d[["spread", "total"]].astype(float)
    margin, total = d.home_points - d.away_points, d.home_points + d.away_points
    return dict(games=int(len(d)), seasons=sorted(d.season.unique().tolist()),
                margin_mae=dict(model=float((d.pred_margin - margin).abs().mean()), line=float((-d.spread - margin).abs().mean()),
                                ratings_only=float((d.r_margin - margin).abs().mean())),
                total_mae=dict(model=float((d.pred_total - total).abs().mean()), line=float((d.total - total).abs().mean()),
                               ratings_only=float((d.r_total - total).abs().mean())),
                spread_blend=blend(d, "pred_margin", -d.spread, margin), total_blend=blend(d, "pred_total", d.total, total),
                spread_picks=ats(d, "spread"), total_picks=ats(d, "total"))


def markdown(ev: dict) -> str:
    pct = lambda v: "-" if v is None else f"{v:.1%}"
    L = ["# NBA Spread/Total Model vs DraftKings (pre-game lines)", "",
         f"Walk-forward; {ev['games']} games with a DraftKings line taken before tip (seasons {', '.join(ev['seasons'])}).", "",
         "| | Model MAE | Ratings only | DraftKings line MAE |", "|---|---:|---:|---:|",
         f"| Margin | {ev['margin_mae']['model']:.2f} | {ev['margin_mae']['ratings_only']:.2f} | {ev['margin_mae']['line']:.2f} |",
         f"| Total | {ev['total_mae']['model']:.2f} | {ev['total_mae']['ratings_only']:.2f} | {ev['total_mae']['line']:.2f} |", "",
         f"Model weight in a regression of the result on line and model: spread {ev['spread_blend']['model_weight']:+.2f} "
         f"(se {ev['spread_blend']['model_se']:.2f}), total {ev['total_blend']['model_weight']:+.2f} (se {ev['total_blend']['model_se']:.2f}).", ""]
    for name, key in (("Spread", "spread_picks"), ("Total", "total_picks")):
        L += [f"## {name}: model disagrees with DraftKings by k points", "",
              "| k | Picks | Hit rate (52.4% breaks even) | By season |", "|---:|---:|---:|---|"]
        L += [f"| {r['edge']} | {r['picks']} | {pct(r['hit_rate'])} | "
              + ", ".join(f"{s}: {v:.0%}" for s, v in r["by_season"].items()) + " |" for r in ev[key]]
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-from", default="2023-11-01")
    args = ap.parse_args()
    with psycopg2.connect(PG_DSN) as conn:
        players, lineups, lines = props.load_games(conn), props.final_lineups(conn), load_lines(conn)
    f = props.features(players, lineups)
    g = team_games(players, f[["game_id", "team", "out_minutes", "out_usage", "out_pts"]])
    r = stage1(g, pd.Timestamp("2022-10-01"))
    pred = walk_forward(r, args.test_from)
    ev = evaluate(pred, lines)
    atomic_json(ROOT / "reports" / "nba_games_model_latest.json", clean(ev))
    text = markdown(ev)
    (ROOT / "reports" / "nba_games_model_latest.md").write_bytes(text.encode())
    print(text)


if __name__ == "__main__":
    main()
