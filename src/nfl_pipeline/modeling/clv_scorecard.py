"""CLV-first scorecard for NFL prop forecasts.

Win/loss over a few weeks is noisy and repeated lines on one player are correlated, so changes
are judged in this order: valid-close coverage, closing-line value (beat rate and average implied
probability movement), probability accuracy against FanDuel's no-vig price, and ROI last.
Every metric gives each player-game equal weight. Rows are split by stat and by the scoring
calibration version that produced them, so a new version is compared on its own forecasts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.betting_preferences import SHARP_EDGE_MIN_EV
from nfl_pipeline.integrity import atomic_json

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = "nfl-clv-scorecard-v1"


def load(conn) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='120s'")
        cur.execute("""
            SELECT p.id, p.stat, p.week, p.game_id, p.player_id, p.side, p.price, p.probability::float AS probability,
                   p.market_no_vig_probability::float AS market_probability,
                   COALESCE(p.forecast_payload->'probability_trace'->>'market_calibration_version', 'pre-calibration')
                     || CASE WHEN (p.forecast_payload->>'sharp_ev')::float >= %(sharp_min_ev)s THEN ' | sharp-edge'
                             WHEN p.forecast_payload->>'sharp_book' IS NOT NULL THEN ' | sharp-priced' ELSE '' END AS version,
                   c.clv_status, c.clv_prob_delta::float AS clv,
                   r.result,
                   EXISTS (SELECT 1 FROM bets.nfl_bet_ledger l WHERE l.source_kind='prop' AND l.prediction_id=p.id) AS selected
            FROM bets.nfl_player_prop_predictions p
            LEFT JOIN bets.nfl_prediction_clv c ON c.source_kind='prop' AND c.prediction_id=p.id
            LEFT JOIN bets.nfl_player_prop_prediction_results r ON r.prediction_id=p.id
            WHERE p.integrity_version='nfl-asof-v2' AND p.line IS NOT NULL AND p.side IN ('over','under')""",
                    {"sharp_min_ev": SHARP_EDGE_MIN_EV})
        return pd.DataFrame([dict(r) for r in cur.fetchall()])


def _weighted(values: pd.Series, weights: pd.Series) -> float | None:
    mask = values.notna()
    if not mask.any():
        return None
    return float(np.average(values[mask].astype(float), weights=weights[mask]))


def score(group: pd.DataFrame) -> dict:
    g = group.copy()
    g["w"] = 1.0 / g.groupby(["game_id", "player_id"]).id.transform("size")
    closed = g[g.clv_status.notna()]
    valid = g[g.clv_status == "valid_close"]
    settled = g[g.result.isin(["win", "loss"])].copy()
    out = dict(forecasts=int(len(g)), player_games=int(g.groupby(["game_id", "player_id"]).ngroups))
    out["valid_close_rate"] = _weighted((closed.clv_status == "valid_close").astype(float), closed.w) if len(closed) else None
    out["clv_beat_rate"] = _weighted((valid.clv > 1e-9).astype(float), valid.w)
    out["clv_worse_rate"] = _weighted((valid.clv < -1e-9).astype(float), valid.w)
    out["avg_clv_prob"] = _weighted(valid.clv, valid.w)
    out["valid_close_n"] = int(len(valid))
    if len(settled):
        y = (settled.result == "win").astype(float)
        both = settled.probability.notna() & settled.market_probability.notna()
        out["settled_n"] = int(len(settled))
        out["brier_model"] = _weighted((settled.probability - y)[both] ** 2, settled.w[both])
        out["brier_market"] = _weighted((settled.market_probability - y)[both] ** 2, settled.w[both])
        price = settled.price.astype(float)
        units = np.where(y == 1, np.where(price > 0, price / 100, 100 / price.abs()), -1.0)
        out["roi_flat"] = _weighted(pd.Series(units, index=settled.index), settled.w)
    return out


def build(df: pd.DataFrame) -> dict:
    sections = {}
    for (stat, version), g in df.groupby(["stat", "version"]):
        sections.setdefault(stat, {})[version] = dict(all_offers=score(g),
                                                       selected=score(g[g.selected]) if g.selected.any() else None)
    return dict(contract=CONTRACT, built_at=datetime.now(timezone.utc).isoformat(), stats=sections)


def _fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{v:+.1%}" if pct == "signed" else (f"{v:.1%}" if pct else f"{v:.4f}")


def markdown(doc: dict) -> str:
    lines = ["# NFL CLV Scorecard", "", f"Built {doc['built_at']}. Judge changes top to bottom: close coverage, CLV, "
             "probability accuracy vs FanDuel no-vig, then ROI. Player-games weighted equally.", "",
             "| Stat | Version | Set | Player-games | Valid close | Beat close | Worse | Avg CLV (prob) | Brier model | Brier market | ROI |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for stat, versions in sorted(doc["stats"].items()):
        for version, sets in sorted(versions.items()):
            for name in ("all_offers", "selected"):
                s = sets.get(name)
                if not s:
                    continue
                lines.append(f"| {stat} | {version} | {name} | {s['player_games']} | {_fmt(s['valid_close_rate'], True)} | "
                             f"{_fmt(s['clv_beat_rate'], True)} | {_fmt(s['clv_worse_rate'], True)} | "
                             f"{_fmt(s['avg_clv_prob'], 'signed')} | {_fmt(s.get('brier_model'))} | {_fmt(s.get('brier_market'))} | "
                             f"{_fmt(s.get('roi_flat'), 'signed')} |")
    return "\n".join(lines) + "\n"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        doc = build(load(conn))
    atomic_json(ROOT / "reports" / "nfl_clv_scorecard_latest.json", doc)
    (ROOT / "reports" / "nfl_clv_scorecard_latest.md").write_text(markdown(doc), encoding="utf-8")
    print(markdown(doc))


if __name__ == "__main__":
    main()
