"""Opponent-adjusted team ratings for FBS, and a walk-forward test against the closing line.

Each team-game is one row: points scored = average + offense[team] + defense[opponent] + home field.
A ridge regression fits the ratings from earlier games only, weighting recent games more (weight halves
every HALF_LIFE_WEEKS; last season counts as SEASON_GAP_WEEKS older). Margins want little shrinkage and
totals want a lot (tuned on 2016-2019 only), so the model is fitted twice: the margin fit gives the spread,
the total fit gives the total, and the team totals are (total +/- margin) / 2, so all of them agree.

The test refits every week of 2016-2025 and predicts that week's games, then compares with the market:
    - closing spread and total (consensus before 2022, else the median of the books CFBD reports)
    - the opener, where books report one (2022 on)
    - picks: when the model differs from the line by at least k points, how often its side covers
      the closing line (52.4% breaks even at -110)
    - blend: regress the result on market and model; a model weight above zero is information the
      market does not already have.

    python -m ncaaf_pipeline.ratings [--seasons 2016-2025]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import sparse
from sklearn.linear_model import Ridge

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import atomic_json

ROOT = Path(__file__).resolve().parents[2]
HALF_LIFE_WEEKS = 16.0
SEASON_GAP_WEEKS = 12.0
ALPHA_MARGIN = 0.1
ALPHA_TOTAL = 6.0
TUNING_SEASONS = (2016, 2019)   # settings above were chosen on these; judge on later seasons
EDGES = (1, 2, 3, 4, 5, 7)


def load(conn) -> pd.DataFrame:
    import warnings
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    games = pd.read_sql("""SELECT game_id, season, week, season_type, start_date, neutral_site,
                                  home_team, away_team, home_points, away_points, home_division, away_division
                           FROM ncaaf.games WHERE completed AND home_points IS NOT NULL AND away_points IS NOT NULL""", conn)
    lines = pd.read_sql("SELECT game_id, provider, spread, spread_open, total, total_open FROM ncaaf.lines", conn)
    consensus = lines[lines.provider.eq("consensus")].set_index("game_id")
    books = lines[~lines.provider.isin(["consensus", "teamrankings", "numberfire"])].groupby("game_id")
    market = pd.DataFrame(dict(
        close_spread=consensus.spread.combine_first(books.spread.median()),
        close_total=consensus.total.combine_first(books.total.median()),
        open_spread=books.spread_open.median(), open_total=books.total_open.median()))
    games = games.merge(market, left_on="game_id", right_index=True, how="left")
    games["order"] = games.season * 100 + np.where(games.season_type.eq("postseason"), 30, games.week.fillna(0))
    return games.sort_values(["order", "game_id"]).reset_index(drop=True)


def _age_weeks(history: pd.DataFrame, season: int, week: float) -> np.ndarray:
    hist_week = np.where(history.season_type.eq("postseason"), 17, history.week.fillna(0))
    return (season - history.season.to_numpy()) * (17 + SEASON_GAP_WEEKS) + (week - hist_week)


def fit(history: pd.DataFrame, season: int, week: float, alpha: float) -> dict:
    """Ratings from games before (season, week)."""
    teams = sorted(set(history.home_team) | set(history.away_team))
    index = {t: i for i, t in enumerate(teams)}
    n, k = len(history), len(teams)
    rows = np.arange(2 * n)
    attack = np.concatenate([history.home_team.map(index), history.away_team.map(index)])
    defend = np.concatenate([history.away_team.map(index), history.home_team.map(index)])
    home = np.concatenate([np.where(history.neutral_site, 0.0, 1.0), np.where(history.neutral_site, 0.0, -1.0)])
    X = sparse.hstack([sparse.csr_matrix((np.ones(2 * n), (rows, attack)), shape=(2 * n, k)),
                       sparse.csr_matrix((np.ones(2 * n), (rows, defend)), shape=(2 * n, k)),
                       sparse.csr_matrix(home.reshape(-1, 1) / 2.0)]).tocsr()
    y = np.concatenate([history.home_points.to_numpy(float), history.away_points.to_numpy(float)])
    age = np.tile(_age_weeks(history, season, week), 2)
    weight = 0.5 ** (np.clip(age, 0, None) / HALF_LIFE_WEEKS)
    model = Ridge(alpha=alpha, fit_intercept=True).fit(X, y, sample_weight=weight)
    coef = model.coef_
    return dict(index=index, offense=coef[:k], defense=coef[k:2 * k], hfa=coef[-1], mean=model.intercept_)


def predict(ratings: dict, games: pd.DataFrame) -> pd.DataFrame:
    index, off, de = ratings["index"], ratings["offense"], ratings["defense"]
    def score(team, opp, home_sign):
        return (ratings["mean"] + np.array([off[index[t]] if t in index else 0.0 for t in team])
                + np.array([de[index[o]] if o in index else 0.0 for o in opp]) + home_sign * ratings["hfa"] / 2.0)
    sign = np.where(games.neutral_site, 0.0, 1.0)
    out = games.copy()
    out["pred_home"] = score(games.home_team, games.away_team, sign)
    out["pred_away"] = score(games.away_team, games.home_team, -sign)
    out["known"] = games.home_team.isin(index) & games.away_team.isin(index)
    return out


def predict_block(history: pd.DataFrame, block: pd.DataFrame, season: int, week: float) -> pd.DataFrame:
    margin = predict(fit(history, season, week, ALPHA_MARGIN), block)
    total = predict(fit(history, season, week, ALPHA_TOTAL), block)
    out = margin.drop(columns=["pred_home", "pred_away"])
    out["pred_margin"] = margin.pred_home - margin.pred_away          # home minus away
    out["pred_total"] = total.pred_home + total.pred_away
    out["pred_home"] = (out.pred_total + out.pred_margin) / 2
    out["pred_away"] = (out.pred_total - out.pred_margin) / 2
    return out


def walk_forward(games: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    parts = []
    for season in seasons:
        season_games = games[games.season.eq(season)]
        for order, block in season_games.groupby("order"):
            history = games[games.order < order]
            history = history[history.season >= season - 2]
            if len(history) < 200:
                continue
            week = 17.0 if block.season_type.iloc[0] == "postseason" else float(block.week.iloc[0])
            parts.append(predict_block(history, block, season, week))
    out = pd.concat(parts, ignore_index=True)
    out["margin"] = out.home_points - out.away_points
    out["total"] = out.home_points + out.away_points
    return out


def ats(df: pd.DataFrame, pred: str, line: str, actual: str, kind: str) -> list[dict]:
    """Pick the model's side when it differs from the line by >= k; hit rate against that line.

    Spreads: the line is the home spread (negative = home favoured), so home covers when margin + line > 0
    and the model likes home when pred_margin + line > 0. Totals: over when actual > line.
    """
    rows = []
    d = df.dropna(subset=[pred, line])
    if kind == "spread":
        edge, result = d[pred] + d[line], d[actual] + d[line]
    else:
        edge, result = d[pred] - d[line], d[actual] - d[line]
    for k in EDGES:
        pick = d[edge.abs() >= k]
        e, r = edge[pick.index], result[pick.index]
        decided = r != 0
        wins = int(((np.sign(e) == np.sign(r)) & decided).sum())
        n = int(decided.sum())
        rows.append(dict(edge=k, picks=n, hit_rate=wins / n if n else None, per_season=n / max(1, d.season.nunique())))
    return rows


def blend_weight(df: pd.DataFrame, pred: str, market: str, actual: str) -> dict:
    """OLS of actual on market and model (no intercept): the model's weight beyond the market."""
    d = df.dropna(subset=[pred, market, actual])
    X = np.column_stack([d[market].to_numpy(float), d[pred].to_numpy(float)])
    coef, *_ = np.linalg.lstsq(X, d[actual].to_numpy(float), rcond=None)
    resid = d[actual].to_numpy(float) - X @ coef
    cov = np.linalg.inv(X.T @ X) * resid.var(ddof=2)
    return dict(rows=int(len(d)), market_weight=float(coef[0]), model_weight=float(coef[1]),
                model_weight_se=float(np.sqrt(cov[1, 1])))


def evaluate(df: pd.DataFrame) -> dict:
    d = df[df.known]
    spread_market = -d.close_spread  # market's expected home margin
    out = dict(games=int(len(d)), seasons=sorted(int(s) for s in d.season.unique()))
    has = d.close_spread.notna()
    out["margin_mae"] = dict(model=float((d.pred_margin - d.margin)[has].abs().mean()),
                             close=float((spread_market - d.margin)[has].abs().mean()), games=int(has.sum()))
    has_t = d.close_total.notna()
    out["total_mae"] = dict(model=float((d.pred_total - d.total)[has_t].abs().mean()),
                            close=float((d.close_total - d.total)[has_t].abs().mean()), games=int(has_t.sum()))
    out["spread_vs_close"] = ats(d, "pred_margin", "close_spread", "margin", "spread")
    out["total_vs_close"] = ats(d, "pred_total", "close_total", "total", "total")
    out["spread_vs_open"] = ats(d, "pred_margin", "open_spread", "margin", "spread")
    out["total_vs_open"] = ats(d, "pred_total", "open_total", "total", "total")
    out["spread_blend"] = blend_weight(d.assign(m=-d.close_spread), "pred_margin", "m", "margin")
    out["total_blend"] = blend_weight(d, "pred_total", "close_total", "total")
    opened = d.dropna(subset=["open_spread", "close_spread"])
    if len(opened):
        model_vs_open = opened.pred_margin + opened.open_spread       # >0: model likes home vs the opener
        move = -(opened.close_spread - opened.open_spread)             # >0: market moved toward home
        out["spread_line_move_agreement"] = dict(games=int(len(opened)), correlation=float(np.corrcoef(model_vs_open, move)[0, 1]))
    return out


def markdown(ev: dict, title: str) -> str:
    pct = lambda v: "-" if v is None else f"{v:.1%}"
    lines = [f"# {title}", "", f"Walk-forward, seasons {ev['seasons'][0]}-{ev['seasons'][-1]}, {ev['games']} games "
             "(each week predicted from earlier games only).", "",
             "| | Model MAE | Closing line MAE | Games |", "|---|---:|---:|---:|",
             f"| Margin | {ev['margin_mae']['model']:.2f} | {ev['margin_mae']['close']:.2f} | {ev['margin_mae']['games']} |",
             f"| Total | {ev['total_mae']['model']:.2f} | {ev['total_mae']['close']:.2f} | {ev['total_mae']['games']} |", "",
             f"Blend (actual on market and model): spread model weight {ev['spread_blend']['model_weight']:+.3f} "
             f"(se {ev['spread_blend']['model_weight_se']:.3f}); total model weight {ev['total_blend']['model_weight']:+.3f} "
             f"(se {ev['total_blend']['model_weight_se']:.3f}). A weight clearly above zero is information the market lacks.", ""]
    if "spread_line_move_agreement" in ev:
        m = ev["spread_line_move_agreement"]
        lines += [f"Line movement: correlation between the model's disagreement with the opener and the opener-to-close move "
                  f"= {m['correlation']:+.3f} on {m['games']} games (positive = the market moves toward the model).", ""]
    for name, key in (("Spread vs closing line", "spread_vs_close"), ("Total vs closing line", "total_vs_close"),
                      ("Spread vs opener", "spread_vs_open"), ("Total vs opener", "total_vs_open")):
        lines += [f"## {name}", "", "| Model differs by | Picks | Per season | Hit rate (52.4% breaks even) |", "|---:|---:|---:|---:|"]
        lines += [f"| >= {r['edge']} pts | {r['picks']} | {r['per_season']:.0f} | {pct(r['hit_rate'])} |" for r in ev[key]]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2016-2025")
    args = ap.parse_args()
    start, end = map(int, args.seasons.split("-"))
    with psycopg2.connect(PG_DSN) as conn:
        games = load(conn)
    df = walk_forward(games, list(range(start, end + 1)))
    ev = evaluate(df)
    atomic_json(ROOT / "reports" / "ncaaf_ratings_latest.json", clean(ev))
    text = markdown(ev, "NCAAF Ratings Model vs the Market")
    (ROOT / "reports" / "ncaaf_ratings_latest.md").write_text(text, encoding="utf-8")
    df.to_csv(ROOT / "reports" / "ncaaf_ratings_predictions.csv", index=False)
    print(text)


if __name__ == "__main__":
    main()
