"""Fit how far NFL spread/total probabilities may move away from FanDuel's no-vig price.

Same method and gate as fit_market_calibration (props): logit(p') = logit(market) + w *
(logit(model) - logit(market)), w chosen on log loss with leave-one-week-out folds, and a market
gets a non-zero w only if it beat the market in every fold. Each game counts once per market (the
pipeline re-scores games several times before kickoff). The no-vig price is rebuilt from the exact
FanDuel quote each forecast locked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import MODEL_ROOT, atomic_json

ROOT = Path(__file__).resolve().parents[3]
PATH = MODEL_ROOT / "game_bets" / "market_calibration.json"
CONTRACT = "nfl-game-market-calibration-v1"
# A fitted trust must beat the market by a real margin in every fold, not by noise.
MIN_LOG_LOSS_GAIN = 0.002
TRUST_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def _implied(price):
    price = pd.to_numeric(price, errors="coerce")
    return np.where(price > 0, 100 / (price + 100), -price / (-price + 100))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _blend(p, market, w):
    return 1.0 / (1.0 + np.exp(-(_logit(market) + w * (_logit(p) - _logit(market)))))


def _log_loss(p, y, weight):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=weight))


def load(conn) -> pd.DataFrame:
    return pd.read_sql("""
        SELECT p.id, p.market, p.side, p.week, p.game_id, p.probability::float AS model_p, r.result,
               CASE WHEN p.market='spread' AND p.side='home' THEN q.spread_home_price
                    WHEN p.market='spread' THEN q.spread_away_price
                    WHEN p.side='over' THEN q.total_over_price ELSE q.total_under_price END AS own_price,
               CASE WHEN p.market='spread' AND p.side='home' THEN q.spread_away_price
                    WHEN p.market='spread' THEN q.spread_home_price
                    WHEN p.side='over' THEN q.total_under_price ELSE q.total_over_price END AS other_price
        FROM bets.nfl_game_predictions p
        JOIN bets.nfl_game_prediction_results r ON r.prediction_id = p.id AND r.result IN ('win', 'loss')
        JOIN raw.nfl_games g ON g.game_id = p.game_id
        JOIN LATERAL (
            -- The exact locked quote when recorded; otherwise (pre-contract forecasts) the latest
            -- FanDuel quote for the same game and line that existed when the forecast was made.
            SELECT * FROM odds.nfl_game_lines q
            WHERE q.bookmaker_key = 'fanduel'
              AND q.home_team_abbr = p.home_team_abbr AND q.away_team_abbr = p.away_team_abbr
              AND q.fetched_at_utc <= p.created_at_utc AND q.fetched_at_utc < g.start_ts_utc
              AND ((p.market = 'spread' AND (CASE WHEN p.side = 'home' THEN q.spread_home_points ELSE q.spread_away_points END) = p.line)
                   OR (p.market = 'total' AND q.total_points = p.line))
            ORDER BY (q.fetched_at_utc = (p.forecast_payload->>'quote_fetched_at_utc')::timestamptz) DESC NULLS LAST,
                     q.fetched_at_utc DESC
            LIMIT 1) q ON TRUE
        WHERE p.integrity_version = 'nfl-asof-v2' AND p.probability IS NOT NULL""", conn)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    own, other = _implied(df.own_price), _implied(df.other_price)
    df = df.assign(market_p=own / (own + other), won=(df.result == "win").astype(float))
    df = df[np.isfinite(df.market_p) & df.model_p.between(0, 1, inclusive="neither")].copy()
    df["weight"] = 1.0 / df.groupby(["market", "game_id"]).id.transform("size")
    return df


def fit_market(d: pd.DataFrame) -> dict:
    def best(x):
        scores = {w: _log_loss(_blend(x.model_p, x.market_p, w), x.won, x.weight) for w in TRUST_GRID}
        return min(scores, key=scores.get)

    folds = []
    for test in sorted(d.week.unique()):
        train, held = d[d.week != test], d[d.week == test]
        if train.empty:
            continue
        w = best(train)
        folds.append(dict(test_week=int(test), probability_trust=float(w), games=int(held.game_id.nunique()),
                          log_loss=_log_loss(_blend(held.model_p, held.market_p, w), held.won, held.weight),
                          log_loss_market=_log_loss(held.market_p, held.won, held.weight),
                          log_loss_uncalibrated=_log_loss(held.model_p, held.won, held.weight)))
    fitted = float(best(d))
    passed = bool(folds) and all(f["log_loss"] < f["log_loss_market"] - MIN_LOG_LOSS_GAIN for f in folds)
    return dict(probability_trust=fitted if passed else 0.0,
                evidence=dict(fitted_probability_trust=fitted, market_gate_passed=passed, games=int(d.game_id.nunique()),
                              forecasts=int(len(d)), weeks=[int(w) for w in sorted(d.week.unique())], leave_one_week_out=folds,
                              in_sample_log_loss_market=_log_loss(d.market_p, d.won, d.weight),
                              in_sample_log_loss_uncalibrated=_log_loss(d.model_p, d.won, d.weight)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Install parameters for live game scoring")
    args = parser.parse_args()
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        data = prepare(load(conn))
    markets = {m: fit_market(g) for m, g in data.groupby("market")}
    doc = dict(contract=CONTRACT, version="game-market-calibration-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
               fitted_at=datetime.now(timezone.utc).isoformat(), markets=markets)
    atomic_json(ROOT / "reports" / "nfl_game_market_calibration_latest.json", doc)
    for m, rec in markets.items():
        ev = rec["evidence"]
        print(f"{m:7s} w={rec['probability_trust']:.2f} (fitted {ev['fitted_probability_trust']:.2f}, gate {ev['market_gate_passed']}) "
              f"games={ev['games']} LL raw={ev['in_sample_log_loss_uncalibrated']:.4f} market={ev['in_sample_log_loss_market']:.4f}  "
              + "  ".join(f"wk{f['test_week']}: w={f['probability_trust']:.2f} LL {f['log_loss']:.4f} vs mkt {f['log_loss_market']:.4f} vs raw {f['log_loss_uncalibrated']:.4f}"
                          for f in ev["leave_one_week_out"]))
    if args.write:
        atomic_json(PATH, doc)
        print("installed", PATH)


if __name__ == "__main__":
    main()
