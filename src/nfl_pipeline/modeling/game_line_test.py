"""NFL spreads and totals against the closing line, season by season. Research only.

features.nfl_game_training_features holds every game since 1999 with lagged team scoring, pace, efficiency,
rest and injury scores, and the closing spread and total. Two models per market, refitted before each test
season on all earlier seasons only:

    direct     the home margin (or total points) from the team features alone, never the line
    residual   what the result did beyond the line, from the same features plus the line; added back to the line

Each is judged where it matters for a bet: its average miss beside the line's, how much weight it earns next
to the line in a regression of the result, and the record of its picks when it disagrees with the line by a
set amount, priced at -110 on both sides. Nothing is saved for scoring and nothing live changes.

    python -m nfl_pipeline.modeling.game_line_test [--first-season 2010]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports" / "nfl_game_line_test_latest.md"
MARKETS = {"spread": dict(actual="home_margin", line="line_margin", label="Home margin"),
           "total": dict(actual="total_points_actual", line="market_total", label="Total points")}
NOT_FEATURES = {"game_id", "season", "week", "season_type", "game_date_et", "start_ts_utc", "home_team_abbr", "away_team_abbr", "home_score",
                "away_score", "home_margin", "total_points_actual", "market_spread_home", "market_total", "line_margin", "roof", "surface",
                "created_at_utc", "updated_at_utc"}
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=8, min_data_in_leaf=40, feature_fraction=0.7, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7, num_threads=4)
ROUNDS = 300
THRESHOLDS = {"direct": (2.0, 3.0, 4.0, 6.0), "residual": (0.5, 1.0, 1.5, 2.0)}
WIN = 100.0 / 110.0             # profit on a winning bet at -110
BREAK_EVEN = 110.0 / 210.0


def load(conn) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM features.nfl_game_training_features WHERE home_margin IS NOT NULL AND market_spread_home IS NOT NULL "
                     "AND market_total IS NOT NULL ORDER BY game_date_et, game_id", conn)
    for c in df.columns:
        if c not in ("game_id", "season_type", "game_date_et", "start_ts_utc", "home_team_abbr", "away_team_abbr", "roof", "surface",
                     "created_at_utc", "updated_at_utc"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return with_line_margin(df)


def with_line_margin(df: pd.DataFrame) -> pd.DataFrame:
    """The home margin the closing spread expects, whichever way the stored spread is signed."""
    sign = 1.0 if np.corrcoef(df.market_spread_home, df.home_margin)[0, 1] > 0 else -1.0
    return df.assign(line_margin=sign * df.market_spread_home)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NOT_FEATURES and pd.api.types.is_numeric_dtype(df[c])]


def walk_forward(df: pd.DataFrame, first_season: int, fitter=None) -> pd.DataFrame:
    """Direct and residual forecasts for every season from first_season, each fitted on earlier seasons only."""
    fitter = fitter or (lambda X, y: lgb.train(PARAMS, lgb.Dataset(X, y), num_boost_round=ROUNDS))
    cols = feature_columns(df)
    parts = []
    for season in sorted(s for s in df.season.unique() if s >= first_season):
        train, test = df[df.season < season], df[df.season == season].copy()
        if train.empty or test.empty:
            continue
        for market, spec in MARKETS.items():
            with_line = cols + [spec["line"]]
            test[f"{market}_direct"] = fitter(train[cols], train[spec["actual"]]).predict(test[cols])
            beyond = fitter(train[with_line], train[spec["actual"]] - train[spec["line"]]).predict(test[with_line])
            test[f"{market}_residual"] = test[spec["line"]] + beyond
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


def weight_beside_line(y: np.ndarray, line: np.ndarray, model: np.ndarray) -> dict:
    """Result minus line regressed on model minus line: 0 means the model adds nothing, 1 means use the model."""
    x, r = model - line, y - line
    slope = float((x * r).sum() / (x * x).sum())
    se = float(np.sqrt(((r - slope * x) ** 2).sum() / (len(x) - 1) / (x * x).sum()))
    return dict(weight=slope, se=se)


def picks(rows: pd.DataFrame, actual: str, line: str, model: str, threshold: float) -> dict:
    """Bets where the model is at least `threshold` from the line, on the model's side, at -110."""
    gap = rows[model] - rows[line]
    chosen = rows[gap.abs() >= threshold]
    side = np.sign(gap[chosen.index])
    margin = (chosen[actual] - chosen[line]) * side
    wins, losses = int((margin > 0).sum()), int((margin < 0).sum())
    out = dict(threshold=threshold, picks=int(len(chosen)), wins=wins, losses=losses, pushes=int((margin == 0).sum()),
               rate=None, roi=None, low=None, high=None, seasons=int(chosen.season.nunique()), winning_seasons=0)
    if wins + losses:
        rate = wins / (wins + losses)
        half = 1.96 * np.sqrt(rate * (1 - rate) / (wins + losses))
        out.update(rate=rate, low=rate - half, high=rate + half, roi=(wins * WIN - losses) / len(chosen))
        by = pd.DataFrame(dict(season=chosen.season, won=margin > 0, decided=margin != 0)).query("decided").groupby("season").won.mean()
        out["winning_seasons"] = int((by > BREAK_EVEN).sum())
    return out


def evaluate(oof: pd.DataFrame) -> dict:
    out = {}
    for market, spec in MARKETS.items():
        y, line = oof[spec["actual"]].to_numpy(float), oof[spec["line"]].to_numpy(float)
        entry = dict(games=int(len(oof)), line_mae=float(np.abs(y - line).mean()), models={})
        for kind in ("direct", "residual"):
            model = oof[f"{market}_{kind}"].to_numpy(float)
            entry["models"][kind] = dict(mae=float(np.abs(y - model).mean()), **weight_beside_line(y, line, model),
                                         picks=[picks(oof, spec["actual"], spec["line"], f"{market}_{kind}", t) for t in THRESHOLDS[kind]],
                                         by_season={int(s): dict(games=int(len(g)), line=float((g[spec["actual"]] - g[spec["line"]]).abs().mean()),
                                                                 model=float((g[spec["actual"]] - g[f"{market}_{kind}"]).abs().mean()))
                                                    for s, g in oof.groupby("season")})
        out[market] = entry
    return out


def markdown(doc: dict) -> str:
    L = ["# NFL spreads and totals against the closing line", "",
         f"Built {doc['built_at']}. Seasons {doc['first_season']} to {doc['last_season']}, each forecast from a model fitted on earlier seasons "
         "only (history from 1999). Research only; nothing live changes.", "",
         f"Picks are priced at -110 on both sides, so a pick needs {BREAK_EVEN:.1%} to break even. Weight is how much of the model's "
         "disagreement with the line shows up in results: 0 means the line already has it."]
    for market, e in doc["markets"].items():
        L += ["", f"## {MARKETS[market]['label']} ({e['games']} games)", "", "| Forecast | Average miss | Weight beside the line (se) |", "|---|---:|---:|",
              f"| Closing line | {e['line_mae']:.3f} | - |"]
        for kind, m in e["models"].items():
            L.append(f"| {kind.capitalize()} model | {m['mae']:.3f} | {m['weight']:+.2f} ({m['se']:.2f}) |")
        L += ["", "| Model | Disagrees by | Picks | Record | Win rate (95%) | Return at -110 | Seasons above break-even |", "|---|---:|---:|---:|---:|---:|---:|"]
        for kind, m in e["models"].items():
            for p in m["picks"]:
                rate = "-" if p["rate"] is None else f"{p['rate']:.1%} ({p['low']:.1%} to {p['high']:.1%})"
                roi = "-" if p["roi"] is None else f"{p['roi']:+.1%}"
                L.append(f"| {kind.capitalize()} | {p['threshold']:g}+ | {p['picks']} | {p['wins']}-{p['losses']}-{p['pushes']} | {rate} | {roi} | "
                         f"{p['winning_seasons']} of {p['seasons']} |")
    L += ["", "The closing line is the last price before kickoff; a bet placed earlier would face a different number."]
    return "\n".join(L) + "\n"


def run(first_season: int, log=print) -> dict:
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        df = load(conn)
    log(f"  {len(df)} games with a closing line, {int(df.season.min())} to {int(df.season.max())}")
    oof = walk_forward(df, first_season)
    doc = dict(built_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), first_season=first_season, last_season=int(oof.season.max()),
               features=len(feature_columns(df)), markets=evaluate(oof), research_only=True, production_changed=False)
    atomic_json(REPORT.with_suffix(".json"), doc)
    REPORT.write_text(markdown(doc), encoding="utf-8")
    log(f"  wrote {REPORT}")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--first-season", type=int, default=2010)
    run(ap.parse_args().first_season)


if __name__ == "__main__":
    main()
