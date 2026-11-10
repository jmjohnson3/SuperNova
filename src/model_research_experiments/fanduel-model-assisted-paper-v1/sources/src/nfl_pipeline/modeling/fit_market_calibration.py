"""Fit offline, version-specific calibration using expanding chronological folds.

Replay original archived scoring code and locked inputs. Never use later weeks
or labels unavailable before a test lock. The full scoring path, fixed original
side, and true paired market are evaluated together. No automatic installation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
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
from nfl_pipeline.modeling.chronological_calibration import forward_folds, equal_player_game_weights
from nfl_pipeline.modeling.scoring_capture import replay

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = "nfl-market-calibration-v2"
# A fitted trust must beat the market by a real margin in every fold, not by noise.
MIN_LOG_LOSS_GAIN = 0.002
TRUST_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
SCALE_GRID = np.round(np.arange(1.0, 3.0001, 0.05), 2)
TARGET_COVERAGE = 0.80
STAT_COLUMN = {"passing_yards": "passing_yards", "rushing_yards": "rushing_yards",
               "receiving_yards": "receiving_yards", "passing_tds": "passing_tds", "receptions": "receptions"}


def load_calibration_records(stat):
    """Bound expensive payload joins by slate instead of one all-history query."""
    from nfl_pipeline.modeling.live_scoring_replay import load_rows
    with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
        conn.set_session(readonly=True)
        cur.execute("SET LOCAL statement_timeout='30s'")
        cur.execute("""SELECT DISTINCT game_date_et FROM bets.nfl_player_prop_predictions
            WHERE integrity_version='nfl-asof-v2' AND stat=%s ORDER BY game_date_et""",(stat,))
        dates=[r[0] for r in cur.fetchall()]
    records=[]
    for day in dates:
        batch=load_rows(day,stat)
        records.extend(batch)
        print('Calibration input',day,'rows',len(batch),flush=True)
    return records


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _blend(p, market, w):
    return 1.0 / (1.0 + np.exp(-(_logit(market) + w * (_logit(p) - _logit(market)))))


def _log_loss(p, y, weight):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=weight))


def prepare_captured(records, preserve_ids=(), require_market_layer=True):
    from nfl_pipeline.modeling.live_scoring_replay import validate_lock
    from nfl_pipeline.modeling.prospective_market import locked_market
    from nfl_pipeline.modeling.scoring_versions import archived_candidate
    from nfl_pipeline.context_contract import safe_time
    from nfl_pipeline.modeling.scoring_equivalence import verified_aliases, cohort_for
    aliases = verified_aliases()
    rows, excluded, seen = [], Counter(), set()
    for r in sorted(records, key=lambda r: (str(r['created_at_utc']), int(r['id']))):
        p = r.get('forecast_payload') or {}
        cap = p.get('scoring_replay') or {}
        error = validate_lock(r)
        if not error and (not cap or cap.get('exact_overlay_present')):
            error = 'missing_replay_or_exact_overlay_artifact'
        if not error and (not cap.get('release_id') or cap['release_id'] != p.get('model_version')):
            error = 'missing_or_mismatched_model_version'
        if not error and locked_market(r)['market_probability'] is None:
            error = 'missing_true_paired_market'
        if not error:
            quote = safe_time(cap['offer'].get('fetched_at_utc'))
            lock = safe_time(r['created_at_utc'])
            if not quote or not 0 <= (lock-quote).total_seconds() <= 1200:
                error = 'stale_original_quote'
        if not error:
            try:
                if require_market_layer and '_market_calibration_params' not in archived_candidate(cap.get('scoring_fingerprint')).__globals__:
                    excluded['archived_code_has_no_market_calibration_layer'] += 1
                    continue
                original = replay(cap)
                if original['side'] != p['side'] or abs(original['probability']-float(p['probability'])) > 1e-10:
                    error = 'original_scoring_replay_mismatch'
            except (ValueError, KeyError) as exc:
                error = str(exc)
        if error:
            excluded[error] += 1
            continue
        key = (r['game_id'], r['player_id'], r['stat'], p['book'], p['line'], p['side'],
               p['model_version'], cap['scoring_fingerprint'])
        if key in seen and int(r['id']) not in preserve_ids:
            excluded['later_revision_same_decision'] += 1
            continue
        seen.add(key)
        rows.append(dict(id=r['id'],stat=r['stat'],game_id=r['game_id'],player_id=r['player_id'],
            season=r['season'],week=r['week'],key=f"{r['game_id']}|{r['player_id']}",
            locked_at=r['created_at_utc'],label_available_at=r.get('label_available_at'),
            model_version=p['model_version'],
            scoring_version=cohort_for(cap['scoring_fingerprint'], r['id'], aliases),
            source_scoring_version=cap['scoring_fingerprint'],variant='production_curve',
            captured=cap,line=float(p['line']),actual=float(r['actual']) if r.get('actual') is not None and r.get('status')=='final' else np.nan,side=p['side'],
            original_probability=float(p['probability'])))
    return pd.DataFrame(rows), dict(excluded)


def rescore(df: pd.DataFrame, params: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Replay all adjustments; return over probability even if replay selects under."""
    out = []
    for rec in df.itertuples():
        cap = rec.captured
        stat = cap["stat"]
        p = dict(projection_trust=1., range_scale=1., probability_trust=1.)
        p.update(params.get(stat) or {})
        revised = deepcopy(cap)
        revised['metrics']['market_calibration'] = {'version': CONTRACT, 'stats': {stat: p}}
        cand = replay(revised)
        market_over = props._no_vig_side_probability(cap["offer"].get("over_price"), cap["offer"].get("under_price"), "over")
        trace = cand.get('probability_trace') or {}
        sharp = cap['offer'].get('sharp_reference') or {}
        sharp_over = props._no_vig_side_probability(sharp.get('over_price'), sharp.get('under_price'), 'over')
        out.append(dict(id=rec.id, stat=stat, season=rec.season, week=rec.week, key=rec.key, line=rec.line,
                        actual=rec.actual, model_projection=float(cap["projection"]), projection=cand["projection"],
                        p10=cand["projection_p10"], p50=cand['projection_p50'], p90=cand["projection_p90"],
                        p_over=cand['probability'] if cand['side']=='over' else 1-cand['probability'],
                        market_over=market_over, blend_reference=sharp_over if sharp_over is not None else market_over,
                        boundary=trace.get('raw_over') in (0., 1.), trace=trace))
    r = pd.DataFrame(out)
    if r.empty:
        return r
    r = r[r.market_over.notna()].copy()
    r["over"] = (r.actual > r.line).astype(float)
    r['is_push'] = r.actual == r.line
    r["weight"] = equal_player_game_weights(r)
    return r


def _one_per_player_game(r: pd.DataFrame) -> pd.DataFrame:
    # Projection/range quality is a player-game property; use the line closest to the model projection.
    r = r.assign(dist=(r.line - r.model_projection).abs())
    return r.sort_values("dist").groupby("key").head(1)


def fit_projection_trust(r):
    g = _one_per_player_game(r)
    scores = {a: float(((g.line + a * (g.model_projection - g.line)) - g.actual).abs().mean()) for a in TRUST_GRID}
    return min(scores, key=scores.get), scores


def range_scale_from_summary(g):
    """Use the actual median; a skewed interval's midpoint is not its median."""
    if g.empty:
        return 1., None
    for k in SCALE_GRID:
        lo = (g.p50 + k * (g.p10-g.p50)).clip(lower=0)
        hi = (g.p50 + k * (g.p90-g.p50)).clip(lower=0)
        coverage = float(((g.actual >= lo) & (g.actual <= hi)).mean())
        if coverage >= TARGET_COVERAGE:
            return float(k), coverage
    return float(SCALE_GRID[-1]), coverage


def fit_range_scale(stat, captured, a, weeks):
    sub = captured[(captured.stat == stat) & captured.week.isin(weeks)]
    base = rescore(sub, {stat: {"projection_trust": a, "range_scale": 1.0}})
    g = _one_per_player_game(base)
    if g.empty or stat.endswith("_tds"):
        return 1.0, None
    return range_scale_from_summary(g)


def fit_probability_trust(r):
    r = r.loc[~r.is_push & r.actual.notna()].copy()
    r['weight'] = equal_player_game_weights(r)
    scores = {w: _log_loss(np.where(r.boundary, r.p_over, _blend(r.p_over, r.blend_reference, w)),
        r.over, r.weight) for w in TRUST_GRID}
    return min(scores, key=scores.get), scores


def summarize(r, w):
    from nfl_pipeline.modeling.evaluation import probability_metrics, projection_metrics
    g = _one_per_player_game(r)
    r = r.loc[~r.is_push & r.actual.notna()].copy()
    if r.empty:
        return dict(n_offers=0, player_games=0)
    r['weight'] = equal_player_game_weights(r)
    p = np.where(r.boundary, r.p_over, _blend(r.p_over, r.blend_reference, w))
    wt = r.weight
    settled_g=g.loc[g.actual.notna()]
    return dict(n_offers=int(len(r)), player_games=int(r.key.nunique()),
                log_loss=_log_loss(p, r.over, wt), log_loss_market=_log_loss(r.market_over, r.over, wt),
                brier=float(np.average((p - r.over) ** 2, weights=wt)),
                brier_market=float(np.average((r.market_over - r.over) ** 2, weights=wt)),
                probability=probability_metrics(p, r.over, wt),
                market=probability_metrics(r.market_over, r.over, wt),
                intervals=dict(projection_metrics(g.actual,g.projection,g.p10,g.p90),
                    below_p10=float((settled_g.actual<settled_g.p10).mean()) if len(settled_g) else None,
                    above_p90=float((settled_g.actual>settled_g.p90).mean()) if len(settled_g) else None))


def fit_captured_stat(stat, captured):
    captured = captured.loc[captured.stat == stat].copy()
    cohorts = captured[['model_version','scoring_version','variant']].drop_duplicates()
    if len(cohorts) != 1:
        raise ValueError('Calibration requires one model/scoring/variant cohort')
    weeks = sorted(captured.week.unique())
    folds, scored_rows = [], []
    for training, testing, metadata in forward_folds(captured):
        if training.key.nunique() < 20 or not (training.actual != training.line).any():
            folds.append(dict(metadata, status='insufficient_earlier_available_labels'))
            continue
        tr0 = rescore(training, {})
        a, _ = fit_projection_trust(tr0)
        k, _ = fit_range_scale(stat, training, a, training.week.unique())
        parameters = {stat: dict(projection_trust=a, range_scale=k)}
        tr = rescore(training, parameters)
        w, _ = fit_probability_trust(tr)
        te = rescore(testing, parameters)
        te_raw = rescore(testing, {})
        # Execute the fitted blend through the real archived scoring path too.
        parameters[stat]['probability_trust'] = w
        final = rescore(testing, parameters)
        expected = np.where(te.boundary, te.p_over, _blend(te.p_over, te.blend_reference, w))
        if not np.allclose(expected, final.p_over, atol=1e-10, rtol=0):
            raise ValueError('Analytical blend differs from full scoring replay')
        status='evaluated' if (testing.actual.notna() & testing.actual.ne(testing.line)).any() else 'scored_awaiting_settlement'
        folds.append(dict(metadata, status=status, projection_trust=float(a), range_scale=float(k), probability_trust=float(w),
                          calibrated=summarize(final, 1.), uncalibrated=summarize(te_raw, 1.0)))
        for rec in final.to_dict('records'):
            raw = te_raw.loc[te_raw.id == rec['id']].iloc[0]
            rec.update(raw_probability=float(raw.p_over), pre_market_probability=float(te.loc[te.id == rec['id'],'p_over'].iloc[0]),
                projection_trust=float(a), range_scale=float(k), probability_trust=float(w))
            scored_rows.append(rec)
    captured = captured.loc[captured.actual.notna()].copy()
    all0 = rescore(captured, {})
    if all0.empty or not (~all0.is_push).any():
        return dict(status='no_binary_outcomes',cohort=cohorts.iloc[0].to_dict(),
            evidence=dict(forward_folds=folds,evaluated_folds=0,forward_test_rows=0),
            rows=[],deployment_approved=False)
    a, a_scores = fit_projection_trust(all0)
    k, coverage = fit_range_scale(stat, captured, a, weeks)
    full = rescore(captured, {stat: {"projection_trust": a, "range_scale": k}})
    w, _ = fit_probability_trust(full)
    fitted_w = w
    evaluated = [f for f in folds if f['status']=='evaluated']
    beats = bool(evaluated) and all(f["calibrated"].get("log_loss",float('inf')) < f["calibrated"].get("log_loss_market",0) - MIN_LOG_LOSS_GAIN for f in evaluated)
    if not beats:
        w = 0.0
    g = _one_per_player_game(all0)
    return dict(status='offline_evaluated' if evaluated else 'waiting_for_earlier_available_labels',
                cohort=cohorts.iloc[0].to_dict(), projection_trust=float(a), range_scale=float(k), probability_trust=float(w),
                rows=scored_rows, deployment_approved=False,
                evidence=dict(weeks=[int(x) for x in weeks],
                              season_weeks=sorted(set(zip(captured.season.astype(int),captured.week.astype(int)))),
                              evaluated_folds=len(evaluated), forward_test_rows=len(scored_rows),
                              fit_coverage_p10_p90=coverage,
                              fitted_probability_trust=float(fitted_w), market_gate_passed=beats,
                              mae_model=float((g.model_projection - g.actual).abs().mean()),
                              mae_line=float((g.line - g.actual).abs().mean()),
                              mae_anchored=a_scores[a], in_sample=summarize(full, w), forward_folds=folds))


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
    d['season'] = d.game_id.str.split('_').str[0].astype(int)
    if not {'locked_at','label_available_at'}.issubset(d.columns):
        return dict(status='missing_lock_or_label_provenance', deployment_approved=False)

    def scored(sub, a):
        proj = sub.line + a * (sub.proj - sub.line)
        mean = proj.clip(lower=.05)
        push = np.where(sub.line.mod(1).eq(0), poisson.pmf(sub.line, mean), 0.)
        return (1-poisson.cdf(np.floor(sub.line), mean))/(1-push)

    def fit(sub):
        a_scores = {a: float(((sub.line + a * (sub.proj - sub.line)) - sub.actual).abs().mean()) for a in TRUST_GRID}
        a = min(a_scores, key=a_scores.get)
        p = scored(sub, a)
        w_scores = {w: _log_loss(_blend(p, sub.mkt_over, w), sub.over, sub.weight) for w in TRUST_GRID}
        return a, min(w_scores, key=w_scores.get)

    folds = []
    for tr, te, metadata in forward_folds(d):
        if tr.empty:
            continue
        a, w = fit(tr)
        p = _blend(scored(te, a), te.mkt_over, w)
        folds.append(dict(metadata, projection_trust=a, probability_trust=w,
                          log_loss=_log_loss(p, te.over, te.weight), log_loss_market=_log_loss(te.mkt_over, te.over, te.weight),
                          log_loss_uncalibrated=_log_loss(scored(te, 1.0), te.over, te.weight)))
    a, w = fit(d)
    fitted_w = w
    beats = bool(folds) and all(f["log_loss"] < f["log_loss_market"] - MIN_LOG_LOSS_GAIN for f in folds)
    if not beats:
        w = 0.0
    return dict(projection_trust=float(a), range_scale=1.0, probability_trust=float(w),
                evidence=dict(source="FanDuel receptions lines recovered from raw SportsGameOdds payloads",
                              fitted_probability_trust=float(fitted_w), market_gate_passed=beats,
                              n_lines=int(len(d)), player_games=int(d.key.nunique()), forward_folds=folds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receptions-lines", type=Path, help="CSV of settled FanDuel receptions lines with the live receptions projection")
    parser.add_argument('--stat', choices=tuple(STAT_COLUMN), default='receiving_yards')
    parser.add_argument("--write", action="store_true", help="Unsupported: deployment requires an explicit versioned review")
    args = parser.parse_args()
    if args.write:
        parser.error('Offline v2 calibration cannot overwrite live parameters; review forward/prospective evidence first')
    from nfl_pipeline.modeling.final_probability_validation import attach_label_availability
    from nfl_pipeline.forecast_store import clean
    captured, exclusions = prepare_captured(attach_label_availability(load_calibration_records(args.stat)))
    stats = {}
    if not captured.empty:
        for cohort, group in captured.groupby(['stat','model_version','scoring_version','variant']):
            print('Fitting', '|'.join(cohort), 'rows', len(group), flush=True)
            stats['|'.join(cohort)] = fit_captured_stat(cohort[0], group)
    if args.receptions_lines:
        rec = fit_receptions(args.receptions_lines)
        if rec:
            stats["receptions"] = rec
    doc = dict(contract=CONTRACT, version="market-calibration-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
               fitted_at=datetime.now(timezone.utc).isoformat(), target_coverage=TARGET_COVERAGE,
               method=__doc__.strip().splitlines()[0], cohorts=stats, exclusions=exclusions,
               production_changed=False, deployment_approved=False, betting_approved=False)
    report = ROOT / "reports" / "nfl_market_calibration_v2_latest.json"
    atomic_json(report, clean(doc))
    for stat, rec in stats.items():
        print(stat, rec.get('status'), 'forward rows',len(rec.get('rows', [])), flush=True)
    print("report", report)


if __name__ == "__main__":
    main()
