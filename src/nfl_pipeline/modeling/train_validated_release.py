"""Fixed-candidate expanding evaluation and atomic NFL release publication.

Legacy research trainers remain available, but cannot mutate this release.
No hyperparameter search, feature selection or calibrator is fit on test labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import create_engine, text

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import FEATURE_CONTRACT, MODEL_ROOT, atomic_joblib, atomic_json, assert_publication_allowed
from nfl_pipeline.markets import STAT_SPECS
from nfl_pipeline.modeling.train_game_models import make_game_features, baseline_home_margin, baseline_total_points
from nfl_pipeline.modeling.train_player_stat_models import _make_features
from nfl_pipeline.modeling.release_models import ConservativeStatModel, RareEventStatModel

log = logging.getLogger(__name__)
GROUPS = {
    "passing": ("passing", "pass_attempts", "opp_allowed_passing", "red_zone_pass"),
    "rushing": ("rushing", "carries", "opp_allowed_rushing", "red_zone_carries", "goal_line_carries", "offense_snap", "snap_share"),
    "receiving": ("receiving", "targets", "receptions", "air_yards", "wopr", "routes", "route_", "offense_snap", "snap_share", "opp_allowed_receiving", "red_zone_targets", "goal_line_targets"),
}


def expanding_folds(df, season, width=4):
    weeks = sorted(df.loc[df.season == season, "week"].dropna().unique())
    for i in range(0, len(weeks), width):
        start = weeks[i]
        train = df.loc[(df.season < season) | ((df.season == season) & (df.week < start))]
        test = df.loc[(df.season == season) & df.week.isin(weeks[i:i + width])]
        if len(train) >= 200 and len(test):
            yield train, test


def feature_columns(X, target, game=False):
    if game:
        # Closing odds/observed weather are benchmarks, not morning-lock features.
        return [c for c in X if ("_avg_" in c or c.endswith("_rest_days") or c == "rest_diff")
                and not any(t in c for t in ("market", "implied", "injury"))]
    group = GROUPS[target.split("_")[0]]
    return [c for c in X if c in {"is_home", "rest_days"} or (
        c.startswith(group) and ("_avg_" in c or "_std_" in c)
        and "share_avg" not in c and "team_player" not in c)]


def baseline(df, target, game):
    if game:
        fn = baseline_home_margin if target == "home_margin" else baseline_total_points
        return fn(df).to_numpy(dtype=float)
    return pd.to_numeric(df[f"{target}_avg_5"], errors="coerce").fillna(0).clip(lower=0).to_numpy()


def fit_predict(train, test, target, game=False):
    make = make_game_features if game else _make_features
    X = make(train)
    columns = feature_columns(X, target, game)
    if not columns:
        raise RuntimeError(f"No audited features for {target}")
    X = X[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    columns = [c for c in columns if X[c].notna().any()]
    X = X[columns]
    fills = X.median().fillna(0).to_dict()
    X = X.fillna(fills)
    Y = pd.to_numeric(train[target], errors="raise").to_numpy(dtype=float)
    est = lgb.LGBMRegressor(objective="regression_l1", n_estimators=180, max_depth=4,
                           num_leaves=15, min_child_samples=80, learning_rate=0.035,
                           reg_lambda=10.0, verbosity=-1, random_state=42, n_jobs=2)
    if not game and target.endswith('_tds'):
        clf = lgb.LGBMClassifier(n_estimators=180, max_depth=3, num_leaves=7, min_child_samples=100,
                                learning_rate=0.035, reg_lambda=10, verbosity=-1, random_state=42, n_jobs=2)
        clf.fit(X, (Y > 0).astype(int))
        positive_mean = float(np.mean(Y[Y > 0])) if np.any(Y > 0) else 1.0
        model = RareEventStatModel(clf, f'{target}_avg_5', positive_mean)
    elif game:
        est.fit(X, Y)
        model = est
    else:
        est.fit(X, Y - baseline(train, target, False))
        model = ConservativeStatModel(est, f"{target}_avg_5")
    Xt = make(test).reindex(columns=columns).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(fills)
    pred = model.predict(Xt) if len(test) else np.array([])
    return model, columns, fills, np.asarray(pred, dtype=float)


def evaluate(actual, pred, base, sigma, lines):
    result = {"rows": len(actual), "mae": float(np.mean(abs(pred-actual))),
              "baseline_mae": float(np.mean(abs(base-actual))), "bias": float(np.mean(pred-actual))}
    p = norm.sf(lines, loc=pred, scale=sigma)
    b = norm.sf(lines, loc=base, scale=sigma)
    outcomes = actual > lines
    result.update(brier=float(np.mean((p-outcomes)**2)), baseline_brier=float(np.mean((b-outcomes)**2)))
    return result


def _probability_rows(actual, pred, base, sigma, target, game, positive_mean=None,
                      model_residuals=None, baseline_residuals=None):
    # A common predeclared proxy line, never chosen from this row's actual outcome.
    width = 3 if game else 25 if target == "passing_yards" else 10 if target.endswith("yards") else 1
    lines = np.floor(base / width) * width + (0.5 if not game else 0)
    result = evaluate(actual, pred, base, sigma, lines)
    if target.endswith('yards') and model_residuals is not None and len(model_residuals) >= 40:
        grid = np.quantile(model_residuals, np.linspace(0,1,201))
        base_grid = np.quantile(baseline_residuals, np.linspace(0,1,201))
        p = np.mean(pred[:,None] + grid > lines[:,None],axis=1)
        b = np.mean(base[:,None] + base_grid > lines[:,None],axis=1)
        result.update(brier=float(np.mean((p-(actual>lines))**2)),
                      baseline_brier=float(np.mean((b-(actual>lines))**2)))
    if positive_mean is not None:
        result['brier'] = float(np.mean((np.clip(pred/positive_mean,0,1)-(actual>0))**2))
        result['baseline_brier'] = float(np.mean(((1-np.exp(-base))-(actual>0))**2))
    return result


def fit_target(df, target, season, game=False):
    records, ys, ps, bs = [], [], [], []
    # An earlier expanding fold supplies calibration residuals to each later fold.
    residuals, base_residuals = [], []
    for train, test in expanding_folds(df, season - 1):
        fold_model, _, _, pred = fit_predict(train, test, target, game)
        y = test[target].to_numpy(dtype=float)
        base = baseline(test, target, game)
        sigma = max(1.0, float(np.std(residuals))) if len(residuals) >= 40 else max(1.0, float(train[target].std()))
        rec = _probability_rows(y, pred, base, sigma, target, game, getattr(fold_model,'positive_mean',None),
                                residuals, base_residuals)
        rec["weeks"] = sorted(int(w) for w in test.week.unique())
        records.append(rec)
        residuals.extend((y - pred).tolist())
        base_residuals.extend((y - base).tolist())
        ys.extend(y); ps.extend(pred); bs.extend(base)
        log.info("%s fold %s: model %.3f baseline %.3f", target, rec["weeks"], rec["mae"], rec["baseline_mae"])
    if not records:
        raise RuntimeError(f"Missing expanding evaluation rows for {target}")
    y, p, b = map(np.asarray, (ys, ps, bs))
    sigma = max(1.0, float(np.std(y-p)))
    oof = {"rows":len(y), "mae":float(np.mean(abs(y-p))), "baseline_mae":float(np.mean(abs(y-b))),
           "brier":float(np.average([r['brier'] for r in records],weights=[r['rows'] for r in records])),
           "baseline_brier":float(np.average([r['baseline_brier'] for r in records],weights=[r['rows'] for r in records]))}
    projection_selected = oof['mae'] < oof['baseline_mae'] and sum(r['mae'] < r['baseline_mae'] for r in records) >= 3
    if target.endswith('_tds'):
        projection_selected = oof['brier'] < oof['baseline_brier'] and sum(r['brier'] < r['baseline_brier'] for r in records) >= 3
    distribution_selected = projection_selected and oof['brier'] < oof['baseline_brier']
    train, test = df.loc[df.season < season], df.loc[df.season == season]
    model, cols, fills, pred = fit_predict(train, test, target, game)
    final = _probability_rows(test[target].to_numpy(dtype=float), pred, baseline(test,target,game), sigma, target,game,
                              getattr(model,'positive_mean',None),residuals,base_residuals) if len(test) else {"rows":0}
    # Later data is a veto, not an opportunity to search another configuration.
    passed = projection_selected and (not len(test) or final['mae'] <= final['baseline_mae'])
    if target.endswith('_tds'):
        passed = projection_selected and (not len(test) or final['brier'] <= final['baseline_brier'])
    distribution_passed = distribution_selected and passed and (not len(test) or final['brier'] <= final['baseline_brier'])
    # Refit the already selected specification for subsequent, unplayed games.
    model, cols, fills, _ = fit_predict(df, df.iloc[:0], target, game)
    base_residuals = y-b
    selected_residuals = y-p if passed else base_residuals
    metrics = {"status":"trained", "accepted":bool(passed), "projection_pass":bool(passed),
               "projection_accepted":bool(passed), "distribution_pass":bool(distribution_passed),
               "baseline_column":"rolling_5", "residual_sigma":max(1.0,float(np.std(selected_residuals))),
               "evaluation":"expanding_fixed_candidate", "folds":records, "oof":oof, "temporal_test":final,
               "holdout_rows":int(len(test)), "mae":final.get('mae'), "baseline_mae":final.get('baseline_mae'),
               "min_gain":0.0, "true_offer_proof":False}
    metrics['mae_gain_vs_baseline'] = oof['baseline_mae'] - oof['mae']
    if target.endswith('_tds'):
        metrics.update(td_probability_accepted=bool(passed),rare_event={'positive_mean':model.positive_mean},
                       selection_target='P(any TD), not count MAE')
    distribution = {"kind":"empirical_oof_residual", "accepted_distribution":bool(distribution_passed),
                    "residual_sigma":metrics['residual_sigma'], "residual_quantiles":np.quantile(selected_residuals,np.linspace(0,1,201)).tolist(),
                    "projection_confidence":0.5, "evaluation":"prior_fold_residuals_not_test_labels"}
    return model, cols, fills, metrics, distribution


def train_release(season, publish=False):
    if publish:
        assert_publication_allowed()
    engine = create_engine(PG_DSN)
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT f.* FROM features.nfl_player_game_training_features f JOIN raw.nfl_games g USING(game_id) JOIN raw.nfl_player_gamelogs p ON p.game_id=f.game_id AND p.player_id=f.player_id AND p.team_abbr=f.team_abbr WHERE g.status='final' AND f.n_games_prev_3>=3 AND (COALESCE(p.offense_snaps,0)>0 OR COALESCE(p.pass_attempts,0)+COALESCE(p.carries,0)+COALESCE(p.targets,0)>0) ORDER BY f.season,f.week,f.game_id,f.player_id"), conn)
        games = pd.read_sql(text("SELECT f.* FROM features.nfl_game_training_features f JOIN raw.nfl_games g USING(game_id) WHERE g.status='final' ORDER BY f.season,f.week,f.game_id"), conn)
    missing = set(range(season-4,season)) - set(df.season.astype(int))
    if missing:
        raise RuntimeError(f"Refusing release: missing player seasons {sorted(missing)}")
    df = df.drop_duplicates(['game_id','player_id']).copy()
    release_id = datetime.now(timezone.utc).strftime("nfl-%Y%m%dT%H%M%SZ")
    root = MODEL_ROOT / "releases" / release_id
    artifacts = {}
    for kind, data, targets in [('players',df,[s.stat for s in STAT_SPECS]),('games',games,['home_margin','total_points_actual'])]:
        artifact = {"status":"ready", "version":release_id, "feature_contract":FEATURE_CONTRACT,
                    "apply_workload_adjustment":False, "models":{},"metrics":{},"feature_columns":{},"fill_values":{},"distributions":{}}
        for target in targets:
            spec = next((s for s in STAT_SPECS if s.stat==target),None)
            sub = data.loc[data.position.isin(spec.positions)].copy() if spec else data.copy()
            sub = sub.loc[pd.to_numeric(sub[target],errors='coerce').notna()].copy()
            m,c,f,metrics,dist = fit_target(sub,target,season,kind=='games')
            artifact['models'][target]=m
            artifact['feature_columns'][target]=c
            artifact['fill_values'][target]=f
            artifact['metrics'][target]=metrics
            artifact['distributions'][target]=dist
        atomic_joblib(root/(kind+'.joblib'),artifact)
        artifacts[kind] = artifact
    summary = {"release_id":release_id,"feature_contract":FEATURE_CONTRACT,"status":"validated",
               "rows":{"players":len(df),"games":len(games)},
               "metrics":{k:v['metrics'] for k,v in artifacts.items()},
               "limitations":["Line Brier uses predeclared proxy lines, not executable historical offers.",
                              "2026 outcomes were examined in prior research; subsequent prospective dates remain essential.",
                              "Unavailable as-of historical context is excluded, not fabricated."]}
    atomic_json(root/'report.json',summary)
    atomic_json(MODEL_ROOT.parents[3]/'reports'/'nfl_validated_release_latest.json',summary)
    manifest = {"release_id":release_id,"feature_contract":FEATURE_CONTRACT,
                "sha256":{k:hashlib.sha256((root/(k+'.joblib')).read_bytes()).hexdigest() for k in artifacts}}
    atomic_json(root/'manifest.json',manifest)
    if publish:
        assert_publication_allowed()
        atomic_json(MODEL_ROOT/'active_release.json',manifest)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season',type=int,default=datetime.now().year)
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')
    result=train_release(args.season,args.publish)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
