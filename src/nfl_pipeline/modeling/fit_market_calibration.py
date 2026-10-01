"""Fit market-anchoring parameters for prop scoring from settled, replayable forecasts.

Weeks 1-3 of 2026 showed the model's disagreement with FanDuel lines carried no
information, forecast ranges were too narrow, and probabilities were overconfident.
This fits, per stat:

* projection_trust a: projection' = line + a * (model projection - line), min MAE vs actual.
* range_scale k: widen the distribution about its median until p10-p90 covers ~80%.
* probability_trust w: logit(p') = logit(market) + w * (logit(p) - logit(market)), min log loss.

Captured live scoring arguments are re-scored through the *current* _candidate_from_offer, so
the fit describes what the new code will do. Each player-game has equal weight. Parameters are
chosen on a grid with leave-one-week-out evaluation reported; the installed values are fit on all
weeks. Receptions (no captured history yet) are fitted from FanDuel lines recovered from raw
SportsGameOdds payloads and the receptions projection used live.

Market gate: a stat keeps a fitted probability_trust only if the calibrated forecast beat the
FanDuel no-vig price on log loss in every leave-one-week-out fold; otherwise it is set to 0
(price at the market) until later weeks show otherwise.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from scipy.stats import poisson

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling import predict_player_props as props

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = "nfl-market-calibration-v1"
TRUST_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
SCALE_GRID = np.round(np.arange(1.0, 3.0001, 0.05), 2)
TARGET_COVERAGE = 0.80
STAT_COLUMN = {"passing_yards": "passing_yards", "rushing_yards": "rushing_yards",
               "receiving_yards": "receiving_yards", "passing_tds": "passing_tds", "receptions": "receptions"}


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _blend(p, market, w):
    return 1.0 / (1.0 + np.exp(-(_logit(market) + w * (_logit(p) - _logit(market)))))


def _log_loss(p, y, weight):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=weight))


def load_captured(conn) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='180s'")
        cur.execute("""
            SELECT p.id, p.stat, p.game_id, p.player_id, g.week, p.line::float AS line,
                   p.forecast_payload->'scoring_replay' AS captured,
                   CASE p.stat WHEN 'passing_yards' THEN l.passing_yards WHEN 'rushing_yards' THEN l.rushing_yards
                     WHEN 'receiving_yards' THEN l.receiving_yards WHEN 'passing_tds' THEN l.passing_tds
                     WHEN 'receptions' THEN l.receptions END::float AS actual
            FROM bets.nfl_player_prop_predictions p
            JOIN raw.nfl_games g ON g.game_id = p.game_id AND g.status = 'final'
            JOIN raw.nfl_player_gamelogs l ON l.game_id = p.game_id AND l.player_id = p.player_id AND l.team_abbr = p.team_abbr
            WHERE p.integrity_version = 'nfl-asof-v2' AND p.line IS NOT NULL
              AND p.forecast_payload ? 'scoring_replay'
              AND NOT COALESCE((p.forecast_payload->'scoring_replay'->>'exact_overlay_present')::boolean, false)
              AND (COALESCE(l.offense_snaps, 0) > 0
                   OR COALESCE(l.pass_attempts, 0) + COALESCE(l.carries, 0) + COALESCE(l.targets, 0) > 0)
              AND p.stat IN ('passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'receptions')""")
        rows = [dict(r) for r in cur.fetchall()]
    df = pd.DataFrame(rows)
    return df[df.actual.notna()].reset_index(drop=True)


def rescore(df: pd.DataFrame, params: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Score each captured offer through the current code with probability_trust=1 (blend is applied analytically)."""
    out = []
    for rec in df.itertuples():
        cap = rec.captured
        stat = cap["stat"]
        p = dict(params.get(stat) or {}, probability_trust=1.0)
        metrics = dict(cap["metrics"], market_calibration={"version": "fit", "stats": {stat: p}})
        cand = props._candidate_from_offer(
            cap["row"], stat, cap["projection"], cap["baseline"], metrics, cap["offer"], cap["distribution"],
            min_ev=cap["min_ev"], probability_calibration_by_key={tuple(r["key"]): r["value"] for r in cap["calibration"]},
            clv_guard_by_key={tuple(r["key"]): r["value"] for r in cap["clv_guards"]}, model_artifact={})
        market_over = props._no_vig_side_probability(cap["offer"].get("over_price"), cap["offer"].get("under_price"), "over")
        out.append(dict(id=rec.id, stat=stat, week=rec.week, key=f"{rec.game_id}|{rec.player_id}", line=rec.line,
                        actual=rec.actual, model_projection=float(cap["projection"]), projection=cand["projection"],
                        p10=cand["projection_p10"], p90=cand["projection_p90"],
                        p_over=cand["conditional_over_probability"], market_over=market_over))
    r = pd.DataFrame(out)
    r = r[r.market_over.notna() & (r.actual != r.line)].copy()  # pushes excluded from binary fit
    r["over"] = (r.actual > r.line).astype(float)
    r["weight"] = 1.0 / r.groupby("key").key.transform("size")
    return r


def _one_per_player_game(r: pd.DataFrame) -> pd.DataFrame:
    # Projection/range quality is a player-game property; use the line closest to the model projection.
    r = r.assign(dist=(r.line - r.model_projection).abs())
    return r.sort_values("dist").groupby("key").head(1)


def fit_projection_trust(r):
    g = _one_per_player_game(r)
    scores = {a: float(((g.line + a * (g.model_projection - g.line)) - g.actual).abs().mean()) for a in TRUST_GRID}
    return min(scores, key=scores.get), scores


def fit_range_scale(stat, captured, a, weeks):
    sub = captured[(captured.stat == stat) & captured.week.isin(weeks)]
    base = rescore(sub, {stat: {"projection_trust": a, "range_scale": 1.0}})
    g = _one_per_player_game(base)
    if g.empty or stat.endswith("_tds"):
        return 1.0, None
    p50 = (g.p10 + g.p90) / 2  # p10/p90 are symmetric about the median for these summaries
    for k in SCALE_GRID:
        lo, hi = (p50 + k * (g.p10 - p50)).clip(lower=0), p50 + k * (g.p90 - p50)
        coverage = float(((g.actual >= lo) & (g.actual <= hi)).mean())
        if coverage >= TARGET_COVERAGE:
            return float(k), coverage
    return float(SCALE_GRID[-1]), coverage


def fit_probability_trust(r):
    scores = {w: _log_loss(_blend(r.p_over, r.market_over, w), r.over, r.weight) for w in TRUST_GRID}
    return min(scores, key=scores.get), scores


def summarize(r, w):
    p = _blend(r.p_over, r.market_over, w)
    wt = r.weight
    return dict(n_offers=int(len(r)), player_games=int(r.key.nunique()),
                log_loss=_log_loss(p, r.over, wt), log_loss_market=_log_loss(r.market_over, r.over, wt),
                brier=float(np.average((p - r.over) ** 2, weights=wt)),
                brier_market=float(np.average((r.market_over - r.over) ** 2, weights=wt)))


def fit_captured_stat(stat, captured):
    weeks = sorted(captured.loc[captured.stat == stat, "week"].unique())
    folds = []
    for test in weeks if len(weeks) > 1 else []:
        train = [w for w in weeks if w != test]
        tr0 = rescore(captured[(captured.stat == stat) & captured.week.isin(train)], {})
        a, _ = fit_projection_trust(tr0)
        k, _ = fit_range_scale(stat, captured, a, train)
        tr = rescore(captured[(captured.stat == stat) & captured.week.isin(train)], {stat: {"projection_trust": a, "range_scale": k}})
        w, _ = fit_probability_trust(tr)
        te = rescore(captured[(captured.stat == stat) & (captured.week == test)], {stat: {"projection_trust": a, "range_scale": k}})
        te_raw = rescore(captured[(captured.stat == stat) & (captured.week == test)], {})
        folds.append(dict(test_week=int(test), projection_trust=a, range_scale=k, probability_trust=w,
                          calibrated=summarize(te, w), uncalibrated=summarize(te_raw, 1.0)))
    all0 = rescore(captured[captured.stat == stat], {})
    a, a_scores = fit_projection_trust(all0)
    k, coverage = fit_range_scale(stat, captured, a, weeks)
    full = rescore(captured[captured.stat == stat], {stat: {"projection_trust": a, "range_scale": k}})
    w, _ = fit_probability_trust(full)
    fitted_w = w
    beats = bool(folds) and all(f["calibrated"]["log_loss"] < f["calibrated"]["log_loss_market"] - 1e-6 for f in folds)
    if not beats:
        w = 0.0
    g = _one_per_player_game(all0)
    return dict(projection_trust=float(a), range_scale=float(k), probability_trust=float(w),
                evidence=dict(weeks=[int(x) for x in weeks], fit_coverage_p10_p90=coverage,
                              fitted_probability_trust=float(fitted_w), market_gate_passed=beats,
                              mae_model=float((g.model_projection - g.actual).abs().mean()),
                              mae_line=float((g.line - g.actual).abs().mean()),
                              mae_anchored=a_scores[a], in_sample=summarize(full, w), leave_one_week_out=folds))


def fit_receptions(path: Path):
    if not path.exists():
        return None
    d = pd.read_csv(path)
    d = d[d.actual != d.line].copy()
    d["over"] = (d.actual > d.line).astype(float)
    d["key"] = d.game_id.astype(str) + "|" + d.player_id.astype(str)
    d["weight"] = 1.0 / d.groupby("key").key.transform("size")
    games = d.game_id.str.split("_").str[1].astype(int)
    d["week"] = games

    def scored(sub, a):
        proj = sub.line + a * (sub.proj - sub.line)
        return 1 - poisson.cdf(np.floor(sub.line), proj.clip(lower=0.05))

    def fit(sub):
        a_scores = {a: float(((sub.line + a * (sub.proj - sub.line)) - sub.actual).abs().mean()) for a in TRUST_GRID}
        a = min(a_scores, key=a_scores.get)
        p = scored(sub, a)
        w_scores = {w: _log_loss(_blend(p, sub.mkt_over, w), sub.over, sub.weight) for w in TRUST_GRID}
        return a, min(w_scores, key=w_scores.get)

    folds = []
    for test in sorted(d.week.unique()):
        tr, te = d[d.week != test], d[d.week == test]
        if tr.empty:
            continue
        a, w = fit(tr)
        p = _blend(scored(te, a), te.mkt_over, w)
        folds.append(dict(test_week=int(test), projection_trust=a, probability_trust=w,
                          log_loss=_log_loss(p, te.over, te.weight), log_loss_market=_log_loss(te.mkt_over, te.over, te.weight),
                          log_loss_uncalibrated=_log_loss(scored(te, 1.0), te.over, te.weight)))
    a, w = fit(d)
    fitted_w = w
    beats = bool(folds) and all(f["log_loss"] < f["log_loss_market"] - 1e-6 for f in folds)
    if not beats:
        w = 0.0
    return dict(projection_trust=float(a), range_scale=1.0, probability_trust=float(w),
                evidence=dict(source="FanDuel receptions lines recovered from raw SportsGameOdds payloads",
                              fitted_probability_trust=float(fitted_w), market_gate_passed=beats,
                              n_lines=int(len(d)), player_games=int(d.key.nunique()), leave_one_week_out=folds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receptions-lines", type=Path, help="CSV of settled FanDuel receptions lines with the live receptions projection")
    parser.add_argument("--write", action="store_true", help="Install parameters for live scoring")
    args = parser.parse_args()
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        captured = load_captured(conn)
    stats = {stat: fit_captured_stat(stat, captured) for stat in sorted(captured.stat.unique())}
    if args.receptions_lines:
        rec = fit_receptions(args.receptions_lines)
        if rec:
            stats["receptions"] = rec
    doc = dict(contract=CONTRACT, version="market-calibration-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
               fitted_at=datetime.now(timezone.utc).isoformat(), target_coverage=TARGET_COVERAGE,
               method=__doc__.strip().splitlines()[0], stats=stats)
    report = ROOT / "reports" / "nfl_market_calibration_latest.json"
    atomic_json(report, doc)
    for stat, rec in stats.items():
        ev = rec["evidence"]
        print(f"{stat:16s} a={rec['projection_trust']:.2f} k={rec['range_scale']:.2f} w={rec['probability_trust']:.2f}  "
              + "  ".join(f"wk{f['test_week']}: a={f['projection_trust']:.2f} w={f['probability_trust']:.2f} "
                          + (f"LL {f['calibrated']['log_loss']:.4f} vs mkt {f['calibrated']['log_loss_market']:.4f} vs raw {f['uncalibrated']['log_loss']:.4f}"
                             if 'calibrated' in f else f"LL {f['log_loss']:.4f} vs mkt {f['log_loss_market']:.4f} vs raw {f['log_loss_uncalibrated']:.4f}")
                          for f in ev.get("leave_one_week_out", [])))
    if args.write:
        atomic_json(props.MARKET_CALIBRATION_PATH, doc)
        print("installed", props.MARKET_CALIBRATION_PATH)
    print("report", report)


if __name__ == "__main__":
    main()
