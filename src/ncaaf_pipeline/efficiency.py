"""Opponent-adjusted efficiency ratings and a market-anchored model for spreads, totals and team totals.

Stage 1 (as of each week, earlier games only): for every per-play metric below, a decayed ridge fit
    value(team's offense in a game) = mean + offense[team] + defense[opponent] + home field
gives each team an adjusted offense and defense rating. Points ratings come from ncaaf_pipeline.ratings.

Stage 2: the closing line is the anchor. A ridge regression predicts the gap between the result and the
close (home margin + close spread; total - close total) from the two teams' ratings, trained only on
earlier seasons. A pick is the side the predicted gap points to when it reaches k points; its record
against the closing line is the evidence (52.4% breaks even at -110).

    python -m ncaaf_pipeline.efficiency [--test-seasons 2020-2025]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import atomic_json
from ncaaf_pipeline import ratings as points

ROOT = Path(__file__).resolve().parents[2]
METRICS = {
    "ppa": "(offense->>'ppa')::float",
    "success": "(offense->>'successRate')::float",
    "explosive": "(offense->>'explosiveness')::float",
    "rush_ppa": "(offense->'rushingPlays'->>'ppa')::float",
    "pass_ppa": "(offense->'passingPlays'->>'ppa')::float",
    "plays": "(offense->>'plays')::float",
    "stuff": "(offense->>'stuffRate')::float",
}
METRIC_ALPHA = 3.0
FIRST_SEASON = 2016
THRESHOLDS = (0.5, 1, 1.5, 2, 3, 4)


def load_stats(conn) -> pd.DataFrame:
    cols = ", ".join(f"{sql} AS {name}" for name, sql in METRICS.items())
    return pd.read_sql(f"SELECT game_id, team, opponent, {cols} FROM ncaaf.team_game_stats", conn)


def _metric_history(history: pd.DataFrame, stats: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Recast team-game metric rows into the games layout ratings.fit expects (home/away 'points')."""
    s = stats[["game_id", "team", metric]].dropna()
    h = history.merge(s.rename(columns={"team": "home_team", metric: "home_points"}), on=["game_id", "home_team"],
                      suffixes=("_x", "")).drop(columns=["home_points_x"], errors="ignore")
    h = h.merge(s.rename(columns={"team": "away_team", metric: "away_points"}), on=["game_id", "away_team"],
                suffixes=("_x", "")).drop(columns=["away_points_x"], errors="ignore")
    return h


def stage1(games: pd.DataFrame, stats: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Walk-forward points predictions plus adjusted offense/defense per metric for both teams."""
    parts = []
    stat_games = {m: _metric_history(games, stats, m) for m in METRICS}
    for season in seasons:
        for order, block in games[games.season.eq(season)].groupby("order"):
            history = games[(games.order < order) & (games.season >= season - 2)]
            if len(history) < 200:
                continue
            week = 17.0 if block.season_type.iloc[0] == "postseason" else float(block.week.iloc[0])
            out = points.predict_block(history, block, season, week)
            for m, sg in stat_games.items():
                mh = sg[(sg.order < order) & (sg.season >= season - 2)]
                r = points.fit(mh, season, week, METRIC_ALPHA)
                idx, off, de = r["index"], r["offense"], r["defense"]
                look = lambda teams, arr: np.array([arr[idx[t]] if t in idx else 0.0 for t in teams])
                out[f"home_off_{m}"] = look(block.home_team, off)
                out[f"home_def_{m}"] = look(block.home_team, de)
                out[f"away_off_{m}"] = look(block.away_team, off)
                out[f"away_def_{m}"] = look(block.away_team, de)
            parts.append(out)
    out = pd.concat(parts, ignore_index=True)
    out["margin"] = out.home_points - out.away_points
    out["total"] = out.home_points + out.away_points
    return out


def features(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Spread features are home-minus-away matchups; total features are sums. Market and points model included."""
    spread, total = {}, {}
    for m in METRICS:
        # home offense against away defense, minus away offense against home defense
        spread[f"{m}_matchup"] = (df[f"home_off_{m}"] + df[f"away_def_{m}"]) - (df[f"away_off_{m}"] + df[f"home_def_{m}"])
        total[f"{m}_sum"] = df[f"home_off_{m}"] + df[f"away_def_{m}"] + df[f"away_off_{m}"] + df[f"home_def_{m}"]
    spread["market_margin"] = -df.close_spread
    spread["points_vs_market"] = df.pred_margin + df.close_spread
    spread["neutral"] = df.neutral_site.astype(float)
    total["market_total"] = df.close_total
    total["points_vs_market"] = df.pred_total - df.close_total
    return dict(spread=pd.DataFrame(spread), total=pd.DataFrame(total))


def stage2(df: pd.DataFrame, test_seasons: list[int]) -> pd.DataFrame:
    """Season-by-season: train on every earlier season, predict the gap to the close for this one."""
    d = df[df.known & df.close_spread.notna() & df.close_total.notna()].copy()
    d["spread_gap"] = d.margin + d.close_spread
    d["total_gap"] = d.total - d.close_total
    X = features(d)
    d["pred_spread_gap"] = np.nan
    d["pred_total_gap"] = np.nan
    for season in test_seasons:
        train, test = d.season.lt(season), d.season.eq(season)
        for kind in ("spread", "total"):
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 21)))
            model.fit(X[kind][train], d.loc[train, f"{kind}_gap"])
            d.loc[test, f"pred_{kind}_gap"] = model.predict(X[kind][test])
    return d[d.season.isin(test_seasons)]


def picks(d: pd.DataFrame, kind: str) -> list[dict]:
    rows = []
    gap, pred = d[f"{kind}_gap"], d[f"pred_{kind}_gap"]
    for k in THRESHOLDS:
        sel = pred.abs().ge(k) & gap.ne(0)
        wins = int((np.sign(pred[sel]) == np.sign(gap[sel])).sum())
        n = int(sel.sum())
        by_season = {int(s): round(float((np.sign(pred[sel & d.season.eq(s)]) == np.sign(gap[sel & d.season.eq(s)])).mean()), 3)
                     for s in sorted(d.season.unique()) if (sel & d.season.eq(s)).sum()}
        rows.append(dict(threshold=k, picks=n, hit_rate=wins / n if n else None, by_season=by_season))
    return rows


def opener_moves(df: pd.DataFrame, test_seasons: list[int]) -> dict:
    """Does the model predict where the opener moves by the close? Bets at the opener only pay if it does
    by more than the vig, so report the move it captures (CLV, points) and its record against the opener."""
    d = df[df.known & df.open_spread.notna() & df.close_spread.notna() & df.open_total.notna() & df.close_total.notna()].copy()
    X = features(d.assign(close_spread=d.open_spread, close_total=d.open_total))  # same features, anchored to the opener
    out = {}
    for kind in ("spread", "total"):
        move = -(d.close_spread - d.open_spread) if kind == "spread" else d.close_total - d.open_total
        result = d.margin + d.open_spread if kind == "spread" else d.total - d.open_total
        pred = pd.Series(np.nan, index=d.index)
        for season in test_seasons:
            train, test = d.season.lt(season), d.season.eq(season)
            if train.sum() < 500 or not test.any():
                continue
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 21))).fit(X[kind][train], move[train])
            pred[test] = model.predict(X[kind][test])
        t = pred.notna()
        rows = []
        for k in (0.25, 0.5, 1.0, 1.5):
            sel = t & pred.abs().ge(k)
            clv = np.sign(pred[sel]) * move[sel]
            decided = sel & result.ne(0)
            rows.append(dict(threshold=k, picks=int(sel.sum()), avg_clv_points=float(clv.mean()) if sel.any() else None,
                             beat_close=float((clv > 0).mean()) if sel.any() else None,
                             win_vs_open=float((np.sign(pred[decided]) == np.sign(result[decided])).mean()) if decided.any() else None))
        out[kind] = dict(games=int(t.sum()), seasons=sorted(int(s) for s in d.season[t].unique()),
                         correlation=float(np.corrcoef(pred[t], move[t])[0, 1]), rows=rows)
    return out


def evaluate(d: pd.DataFrame) -> dict:
    out = dict(games=int(len(d)), seasons=sorted(int(s) for s in d.season.unique()))
    for kind in ("spread", "total"):
        gap, pred = d[f"{kind}_gap"], d[f"pred_{kind}_gap"]
        out[kind] = dict(
            close_mae=float(gap.abs().mean()), model_mae=float((gap - pred).abs().mean()),
            correlation=float(np.corrcoef(pred, gap)[0, 1]), pred_sd=float(pred.std()),
            picks=picks(d, kind))
    return out


def markdown(ev: dict) -> str:
    lines = ["# NCAAF Efficiency Model vs the Closing Line", "",
             f"Out of sample, seasons {ev['seasons'][0]}-{ev['seasons'][-1]}, {ev['games']} games. Each season's "
             "model is trained only on earlier seasons; every rating is as of the week before the game.", "",
             "| | Close MAE | Close + model MAE | Corr(pred gap, actual gap) | Pred gap SD |", "|---|---:|---:|---:|---:|"]
    for kind in ("spread", "total"):
        e = ev[kind]
        lines.append(f"| {kind.title()} | {e['close_mae']:.3f} | {e['model_mae']:.3f} | {e['correlation']:+.3f} | {e['pred_sd']:.2f} |")
    for kind in ("spread", "total"):
        lines += ["", f"## {kind.title()} picks against the closing line", "",
                  "| Predicted gap | Picks | Hit rate | By season |", "|---:|---:|---:|---|"]
        for r in ev[kind]["picks"]:
            hr = "-" if r["hit_rate"] is None else f"{r['hit_rate']:.1%}"
            seasons = ", ".join(f"{s}: {v:.0%}" for s, v in r["by_season"].items())
            lines.append(f"| >= {r['threshold']} | {r['picks']} | {hr} | {seasons} |")
    if "opener" in ev:
        pct = lambda v: "-" if v is None else f"{v:.1%}"
        for kind, o in ev["opener"].items():
            lines += ["", f"## {kind.title()}: predicting the opener-to-close move ({o['seasons'][0]}-{o['seasons'][-1]}, "
                      f"{o['games']} games, correlation {o['correlation']:+.3f})", "",
                      "| Predicted move | Picks | Avg CLV (pts) | Beat close | Win vs opener |", "|---:|---:|---:|---:|---:|"]
            for r in o["rows"]:
                clv = "-" if r["avg_clv_points"] is None else f"{r['avg_clv_points']:+.2f}"
                lines.append(f"| >= {r['threshold']} | {r['picks']} | {clv} | {pct(r['beat_close'])} | {pct(r['win_vs_open'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-seasons", default="2020-2025")
    args = ap.parse_args()
    start, end = map(int, args.test_seasons.split("-"))
    with psycopg2.connect(PG_DSN) as conn:
        games, stats = points.load(conn), load_stats(conn)
    cache = ROOT / "reports" / "ncaaf_efficiency_stage1.csv.gz"
    df = stage1(games, stats, list(range(FIRST_SEASON, end + 1)))
    df.to_csv(cache, index=False)
    d = stage2(df, list(range(start, end + 1)))
    ev = evaluate(d)
    ev["opener"] = opener_moves(df, [2023, 2024, 2025])
    atomic_json(ROOT / "reports" / "ncaaf_efficiency_latest.json", clean(ev))
    text = markdown(ev)
    (ROOT / "reports" / "ncaaf_efficiency_latest.md").write_bytes(text.encode())
    print(text)


if __name__ == "__main__":
    main()
