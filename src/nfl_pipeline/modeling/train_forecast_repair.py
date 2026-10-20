"""Train isolated NFL repairs and retain identical chronological comparators."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mlb_pipeline.atomic_io import atomic_write_text
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, active_release, atomic_joblib, atomic_json
from nfl_pipeline.modeling.challenger_models import numeric, curve_summary
from nfl_pipeline.modeling.component_validation import interval_score
from nfl_pipeline.modeling.evaluation import expanding_week_folds, projection_metrics, probability_metrics, clustered_gain
from nfl_pipeline.modeling.forecast_repair import CONTRACT, VARIANTS, PlayerRepair, SmallTotalRepair
from nfl_pipeline.modeling.role_workload import STATS, validate
from nfl_pipeline.modeling.train_accuracy_challengers import ROOT, load_training, line_frame, game_baseline

LOG = logging.getLogger(__name__)
MARKETS = ('rushing_yards', 'passing_yards', 'total')


def pin_latest(markets=MARKETS):
    from nfl_pipeline.modeling import distribution_policy as dp
    result = {}
    for market in markets:
        report = json.loads((ROOT/f'reports/nfl_forecast_repair_{market}_latest.json').read_text(encoding='utf-8'))
        variant = 'small_total' if market == 'total' else 'joint_calibrated'
        screens = report['decisions'][variant]['historical_screens']
        if not any(screens.values()):
            result[market] = dict(status='no_historical_output_improved'); continue
        descriptor = dp.repair_descriptor(report['artifact'], variant)
        if descriptor['sha256'] != report['artifact_sha256']:
            raise ValueError('Repair artifact does not match evaluated artifact')
        existing = [p for p in dp.load_registry('sources').values()
            if p['key'].rsplit('|', 1)[0] == dp.policy_key(market, descriptor['production_release']).rsplit('|', 1)[0]]
        if any(p['source'] != dp.repair_source(descriptor) for p in existing):
            result[market] = dict(status='existing_source_kept_frozen'); continue
        source = dp.register_source(descriptor)
        result[market] = dict(status='pinned_for_new_locks_only', source=source['source'],
            historical_screens=screens, cash_approved=False, forecast_deployed=False)
    return result


def scores(rows, lines):
    return {v: dict(expected=projection_metrics(g.actual, g['mean'], g.p10, g.p90),
        typical=projection_metrics(g.actual, g['median']),
        below=float((g.actual < g.p10).mean()), above=float((g.actual > g.p90).mean()),
        interval_score=float(np.mean(interval_score(g.actual, g.p10, g.p90))),
        probability=probability_metrics(lines.loc[lines.variant.eq(v), 'probability'],
            lines.loc[lines.variant.eq(v), 'outcome'], lines.loc[lines.variant.eq(v), 'weight']))
        for v, g in rows.groupby('variant')}


def compare(rows, lines, result):
    decisions = {}
    ref = rows.loc[rows.variant.eq('reference')]
    ref_lines = lines.loc[lines.variant.eq('reference')]
    for variant in result:
        if variant == 'reference': continue
        p = ref.merge(rows.loc[rows.variant.eq(variant)], on=['game_id', 'player_id', 'season', 'week'],
            validate='one_to_one', suffixes=('_old', '_new'))
        gains = {}
        for output, column, power in (('expected', 'mean', 2), ('typical', 'median', 1)):
            gains[output] = clustered_gain(p.assign(reference_error=abs(p[column+'_old']-p.actual_old)**power,
                challenger_error=abs(p[column+'_new']-p.actual_new)**power))
        l = ref_lines.merge(lines.loc[lines.variant.eq(variant)], on=['player_game', 'season', 'week', 'line'],
            validate='one_to_one', suffixes=('_old', '_new'))
        l = l.assign(reference_error=(l.probability_old-l.outcome_old)**2,
            challenger_error=(l.probability_new-l.outcome_new)**2)
        gains['probability'] = clustered_gain(l.groupby(['season', 'week', 'player_game'])[
            ['reference_error', 'challenger_error']].mean().reset_index())
        a, b = result[variant], result['reference']
        passed = {k: (g.get('lower_95') or 0) > 0 for k, g in gains.items()}
        passed['expected'] &= abs(a['expected']['bias']) <= abs(b['expected']['bias'])
        passed['probability'] &= (a['probability']['calibration_error'] <= b['probability']['calibration_error']
            and abs(a['expected']['coverage_80']-.8) <= abs(b['expected']['coverage_80']-.8)
            and a['interval_score'] <= b['interval_score'])
        decisions[variant] = dict(historical_screens=passed, weekly_gains=gains,
            forecast_deployment_approved=False, cash_approved=False)
    return decisions


def evaluate(frame, market, seasons):
    rows, lines, folds = [], [], []
    variants = ('reference', 'small_total') if market == 'total' else VARIANTS
    target = 'total_points_actual' if market == 'total' else market
    for name, train, test in expanding_week_folds(frame, seasons):
        LOG.info('%s %s: %d training / %d later rows', market, name, len(train), len(test))
        model = SmallTotalRepair().fit(train) if market == 'total' else PlayerRepair().fit(train, market)
        folds.append(dict(fold=name, blocks=model.blocks, train_end=model.training_end,
            test_start=str(test.game_date_et.min()), test_end=str(test.game_date_et.max()),
            rate_strength=None if market == 'total' else model.rate.strength,
            cdf_validation=None if market == 'total' else model.cdf_validation,
            count_weight=None if market == 'total' else model.workload.alpha,
            total_strength=model.strength if market == 'total' else None))
        for variant in variants:
            v, w = model.curve(test, variant)
            summary = curve_summary(v, w)
            r = test[['game_id', 'player_id', 'season', 'week', 'game_date_et']].reset_index(drop=True)
            r = r.assign(variant=variant, actual=numeric(test, target).to_numpy(), **summary)
            if market != 'total':
                r['position'] = test.position.to_numpy()
                r['actual_opportunity'] = numeric(test, STATS[market]).to_numpy()
                r['prior_opportunity'] = model.workload.baseline(test)
                r['predicted_opportunity'] = (lambda x: (x[0]*x[1]).sum(axis=1))(model.workload.count_curve(test, True, True))
                r['efficiency'] = model.rate.predict(test)
                l = line_frame(test, market, v, w)
            else:
                threshold = np.floor(game_baseline(test, target))+.5
                l = r[['season', 'week']].assign(player_game=r.game_id.astype(str), line=threshold,
                    probability=(w*(v > threshold[:, None])).sum(axis=1), outcome=(r.actual > threshold).astype(float), weight=1.)
            rows.append(r); lines.append(l.assign(variant=variant))
    if not rows: raise ValueError('No later evaluation folds')
    return pd.concat(rows, ignore_index=True), pd.concat(lines, ignore_index=True), folds


def run(cache, through='2026-09-17', seasons=(2025, 2026), markets=MARKETS):
    before = active_release()
    cache = Path(cache)
    if cache.exists():
        data = joblib.load(cache)
    else:
        LOG.info('Loading finalized player-games and games once (no schema changes)')
        players, games = load_training()
        data = dict(players=players, games=games, captured_at=datetime.now(timezone.utc).isoformat())
        atomic_joblib(cache, data)
    result = {}
    for market in markets:
        frame = data['games' if market == 'total' else 'players'].copy()
        frame = frame.loc[pd.to_datetime(frame.game_date_et).le(pd.Timestamp(through))].copy()
        if market == 'total':
            frame['player_id'] = 'game'
            frame = frame.loc[numeric(frame, 'total_points_actual').notna()]
        else:
            positions = ('QB',) if market == 'passing_yards' else ('RB', 'QB')
            frame = frame.loc[frame.position.isin(positions)].copy()
            n, y = numeric(frame, STATS[market]), numeric(frame, market)
            frame = frame.loc[np.isfinite(n) & np.isfinite(y) & n.ge(0) & ~(n.eq(0) & y.ne(0))].copy()
            validate(frame, market)
        if frame.duplicated(['game_id', 'player_id']).any():
            raise ValueError('Duplicate forecast training outcomes')
        rows, lines, folds = evaluate(frame, market, seasons)
        measurements = scores(rows, lines)
        decisions = compare(rows, lines, measurements)
        LOG.info('%s: final isolated fit', market)
        model = SmallTotalRepair().fit(frame) if market == 'total' else PlayerRepair().fit(frame, market)
        run_id = 'forecast-repair-'+market+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        path = MODEL_ROOT/'forecast_repair'/run_id
        artifact = dict(contract=CONTRACT, market=market, model=model, training_end=model.training_end,
            production_release=before['release_id'], run_id=run_id, output_decisions=decisions)
        atomic_joblib(path/'model.joblib', artifact)
        atomic_joblib(path/'oof.joblib', dict(rows=rows, lines=lines))
        report = clean(dict(run_id=run_id, market=market, artifact=str(path/'model.joblib'),
            artifact_sha256=hashlib.sha256((path/'model.joblib').read_bytes()).hexdigest(),
            production_release=before['release_id'], training_end=model.training_end, scores=measurements,
            decisions=decisions, folds=folds, blocks=model.blocks,
            feature_support=model.support if market == 'total' else model.workload.support,
            by_season={str(s): scores(g, lines.loc[lines.season.eq(s)]) for s, g in rows.groupby('season')},
            production_changed=False, cash_approved=False,
            limitations=['Historical development only; these seasons have already been examined.',
                'One player-game or game per outcome; whole-week chronology and clustered uncertainty.',
                'Proxy lines are not real offered lines. Reference is a chronological recipe, not hindsight live production.',
                'Receiving pinned policy remains unchanged. No odds, ledgers, locks or Discord posts are rewritten.',
                'Models, rate tuning, residual tails, CDF fitting and CDF gating use separate chronological blocks.',
                'No current injuries or actual routes are invented; unavailable inputs remain missing.',
                'Deployment requires later captured same-offer validation, independently of cash approval.']))
        atomic_json(path/'report.json', report)
        atomic_json(ROOT/f'reports/nfl_forecast_repair_{market}_latest.json', report)
        content = ['# NFL Isolated Forecast Repair', '', market, '',
            '| Variant | Mean RMSE | Bias | Median MAE | Brier | Calibration | Below / above 80% interval |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name, m in measurements.items():
            content.append(f"| {name} | {m['expected']['rmse']:.3f} | {m['expected']['bias']:+.3f} | "
                f"{m['typical']['mae']:.3f} | {m['probability']['brier']:.4f} | {m['probability']['calibration_error']:.2%} | {m['below']:.1%} / {m['above']:.1%} |")
        content += ['', '## Decisions', '', *[k+': '+json.dumps(v['historical_screens']) for k, v in decisions.items()],
            '', *report['limitations']]
        atomic_write_text(ROOT/f'reports/nfl_forecast_repair_{market}_latest.md', '\n'.join(content)+'\n')
        result[market] = dict(artifact=str(path/'model.joblib'), scores=measurements, decisions=decisions)
        if active_release() != before: raise RuntimeError('Production release changed during offline work')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=MODEL_ROOT/'forecast_repair'/'training-cache.joblib')
    parser.add_argument('--through', default='2026-09-17')
    parser.add_argument('--seasons', type=int, nargs='+', default=[2025, 2026])
    parser.add_argument('--markets', choices=MARKETS, nargs='+', default=list(MARKETS))
    parser.add_argument('--pin-latest', action='store_true', help='Capture evaluated sources on future locks; no training or deployment')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.pin_latest:
        print(json.dumps(pin_latest(args.markets), indent=2)); return
    result = run(args.cache, args.through, args.seasons, args.markets)
    print(json.dumps({k: dict(artifact=v['artifact'], screens={name: d['historical_screens']
        for name, d in v['decisions'].items()}) for k, v in result.items()}))


if __name__ == '__main__': main()
