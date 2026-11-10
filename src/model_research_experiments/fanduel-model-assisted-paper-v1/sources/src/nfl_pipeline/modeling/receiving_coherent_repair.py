"""Isolated receiving downside/CDF challengers with chronological acceptance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, active_release, atomic_joblib, atomic_json
from nfl_pipeline.modeling.challenger_models import FittedHead, numeric, player_features, curve_summary, POSITIONS
from nfl_pipeline.modeling.evaluation import expanding_week_folds
from nfl_pipeline.modeling.model_benchmark import fit_uncertainty, raw_curve, curve_metrics, summarize
from nfl_pipeline.modeling.sample_aware_splits import sample_aware_partitions, describe_blocks
from nfl_pipeline.modeling.tail_calibration import TailPreservingCalibration
from nfl_pipeline.modeling.train_accuracy_challengers import ReferenceRecipe, line_frame, ROOT

STAT = 'receiving_yards'
LOG = logging.getLogger(__name__)


def downside_labels(frame, values, weights):
    """Prior-target volume is a historical proxy, not a reconstructed live target forecast."""
    summary = curve_summary(values, weights)
    y = numeric(frame, STAT).to_numpy()
    targets = numeric(frame, 'targets').to_numpy()
    prior = numeric(frame, 'targets_avg_5').to_numpy()
    eligible = np.isfinite(targets) & np.isfinite(prior) & (prior >= 4)
    low = eligible & (y < summary['p10'])
    volume = low & (targets < .5*prior)
    labels = np.where(volume, 1, np.where(low, 2, 0))
    return labels, eligible


class ReceivingDownside:
    """Replace a fraction of the lower-tail mixture only for established target roles."""
    def fit(self, frame, values, weights):
        labels, eligible = downside_labels(frame, values, weights)
        self.enabled = bool(eligible.sum() >= 100 and (labels > 0).sum() >= 30)
        self.counts = dict(eligible=int(eligible.sum()), low_volume=int((labels == 1).sum()),
                           efficiency_residual=int((labels == 2).sum()))
        if not self.enabled:
            return self
        X = player_features(frame, STAT)
        self.head = FittedHead(classifier=True, params={'n_estimators': 70, 'num_leaves': 5,
            'max_depth': 2, 'reg_lambda': 30}).fit(X.loc[eligible], labels[eligible])
        self.prior = np.bincount(labels[eligible], minlength=3)/eligible.sum()
        center = np.maximum(5., curve_summary(values, weights)['median'])
        ratios = numeric(frame, STAT).to_numpy()/center
        self.ratios = {}
        pooled = ratios[labels > 0]
        for cause in (1, 2):
            observed = ratios[labels == cause]
            own = np.quantile(observed if len(observed) else pooled, np.linspace(.01, .99, 51))
            parent = np.quantile(pooled, np.linspace(.01, .99, 51))
            self.ratios[cause] = (len(observed)*own+40*parent)/(len(observed)+40)
        return self

    def transform(self, frame, values, weights):
        if not self.enabled:
            return values.copy(), weights.copy()
        summary = curve_summary(values, weights)
        known = numeric(frame, 'targets_avg_5').ge(4).to_numpy()
        mass = .8*self.head.probabilities(player_features(frame, STAT), 3)+.2*self.prior
        normal = weights*(values >= summary['p10'][:, None])
        normal /= normal.sum(axis=1, keepdims=True)
        center = np.maximum(5., summary['median'])
        # Keep 75% of the reference distribution. No universal width/mean correction.
        alpha = .25*known.astype(float)
        parts = [values, values]
        probs = [weights*(1-alpha[:, None]), normal*alpha[:, None]*mass[:, 0, None]]
        for cause in (1, 2):
            parts.append(center[:, None]*self.ratios[cause][None, :])
            probs.append(np.broadcast_to(alpha[:, None]*mass[:, cause, None]/51, (len(frame), 51)))
        return np.concatenate(parts, axis=1), np.concatenate(probs, axis=1)


def coherent_after_adjustments(values, weights, lines, adjusted_over, boundary=.1):
    """Last-step survival mapping for one book/player/lock. Never touches past locks.

    All actual offered-line adjustments enter together; the output is a single CDF.
    Identity tails keep interval anchors. Prices at unobserved lines are not invented.
    """
    values = np.asarray(values, float); weights = np.asarray(weights, float)
    lines = np.asarray(lines, float); adjusted_over = np.asarray(adjusted_over, float)
    if values.ndim != 1 or values.shape != weights.shape or lines.shape != adjusted_over.shape:
        raise ValueError('Invalid coherence input shapes')
    if (not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights < 0).any()
            or weights.sum() <= 0 or not np.isfinite(lines).all() or not np.isfinite(adjusted_over).all()
            or ((adjusted_over < 0) | (adjusted_over > 1)).any()):
        raise ValueError('Invalid coherence probabilities or distribution')
    weights = weights/weights.sum()
    raw = ((values[None, :] > lines[:, None])*weights).sum(axis=1)
    use = (raw > boundary) & (raw < 1-boundary)
    mapping = TailPreservingCalibration(boundary)
    if use.any():
        iso = IsotonicRegression(y_min=boundary, y_max=1-boundary, out_of_bounds='clip')
        iso.fit(raw[use], np.clip(adjusted_over[use], boundary, 1-boundary))
        mapping.knots = np.r_[boundary, iso.X_thresholds_, 1-boundary]
        mapping.values = np.r_[boundary, iso.y_thresholds_, 1-boundary]
        mapping.strength = .999  # Preserve strict ordering and the 10/90 anchors.
        mapping.enabled = True
    v, w = mapping.transform(values[None, :], weights[None, :])
    return v[0], w[0]


def passes(before, after):
    b = before['expected']['coverage_80']; a = after['expected']['coverage_80']
    return bool(after['probability']['brier'] < before['probability']['brier']
        and after['probability']['calibration_error'] <= before['probability']['calibration_error']
        and abs(a-.8) <= abs(b-.8)+1e-12
        and after['interval_score'] <= before['interval_score']+1e-12)


def fit_bundle(train):
    early, scale, residual, fit, tune, gate = sample_aware_partitions(train, STAT)
    ref = ReferenceRecipe().fit(early, STAT)
    base = dict(kind='conditional', model=ref, uncertainty=fit_uncertainty(ref, scale, residual, STAT))
    downside = ReceivingDownside().fit(fit, *raw_curve(base, fit, STAT))
    weeks = sorted(set(zip(gate.season, gate.week)))
    split = max(1, len(weeks)//2)
    first = set(weeks[:split])
    mask = np.array([(s, w) in first for s, w in zip(gate.season, gate.week)])
    cal_tune, cal_gate = gate.loc[mask], gate.loc[~mask]
    calibrators = {}
    for variant in ('coherent_base', 'coherent_downside'):
        frames = []
        for block in (tune, cal_tune, cal_gate):
            v, w = raw_curve(base, block, STAT)
            if variant == 'coherent_downside':
                v, w = downside.transform(block, v, w)
            frames.append(line_frame(block, STAT, v, w))
        calibrators[variant] = TailPreservingCalibration().fit(*frames)
    bundle = dict(base=base, downside=downside, calibrators=calibrators,
        blocks=describe_blocks((early, scale, residual, fit, tune, gate)), training_end=str(train.game_date_et.max()))
    curves = variants(bundle, cal_gate)
    scores = {name: curve_metrics(cal_gate, STAT, *curve) for name, curve in curves.items()}
    accepted = [name for name in ('coherent_base', 'coherent_downside') if passes(scores['reference'], scores[name])]
    bundle['choice'] = min(accepted, key=lambda name: scores[name]['probability']['brier']) if accepted else 'reference'
    bundle['validation'] = dict(choice=bundle['choice'], scores=scores, downside_fit=downside.counts,
        calibrators={k: v.validation for k, v in calibrators.items()})
    return bundle


def variants(bundle, frame):
    v, w = raw_curve(bundle['base'], frame, STAT)
    dv, dw = bundle['downside'].transform(frame, v, w)
    return dict(reference=(v, w), downside_candidate=(dv, dw),
        coherent_base=bundle['calibrators']['coherent_base'].transform(v, w),
        coherent_downside=bundle['calibrators']['coherent_downside'].transform(dv, dw))


def run(cache, seasons):
    frozen = active_release()
    started = datetime.now(timezone.utc)
    run_id = 'receiving-coherent-'+started.strftime('%Y%m%dT%H%M%S%fZ')
    root = MODEL_ROOT/'receiving_coherent_repair'/run_id
    data = joblib.load(cache)
    data = data.loc[data.position.isin(POSITIONS[STAT]) & numeric(data, STAT).notna() & numeric(data, 'targets').notna()].copy()
    if data.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Duplicate player-games')
    rows = []; lines = []; folds = []; failures = []
    for fold, train, test in expanding_week_folds(data, seasons):
        LOG.info('%s: %s train / %s test', fold, len(train), len(test))
        bundle = fit_bundle(train)
        curves = variants(bundle, test)
        curves['gated_policy'] = curves[bundle['choice']]
        labels, known = downside_labels(test, *curves['reference'])
        failures.append(dict(fold=fold, established_rows=int(known.sum()), low_volume_misses=int((labels == 1).sum()),
                             efficiency_residual_misses=int((labels == 2).sum())))
        for name, (v, w) in curves.items():
            rows.append(test[['game_id', 'player_id', 'season', 'week', 'game_date_et']].reset_index(drop=True).assign(
                fold=fold, family=name, actual=numeric(test, STAT).to_numpy(), **curve_summary(v, w)))
            lines.append(line_frame(test, STAT, v, w).assign(fold=fold, family=name))
        folds.append(dict(fold=fold, training_end=bundle['training_end'], test_start=str(test.game_date_et.min()),
                          blocks=bundle['blocks'], validation=bundle['validation']))
    if not rows:
        raise ValueError('No test folds')
    r = pd.concat(rows, ignore_index=True); l = pd.concat(lines, ignore_index=True)
    scores = summarize(r, l)
    chosen = scores['gated_policy']; ref = scores['reference']; gain = chosen['brier_gain']
    accepted = bool(gain.get('lower_95') is not None and gain['lower_95'] > 0 and passes(ref, chosen))
    report = dict(run_id=run_id, built_at=datetime.now(timezone.utc).isoformat(),
        status='historical_screen_passed_needs_prospective_full_path' if accepted else 'not_accepted',
        production_release=frozen['release_id'], summary=scores, folds=folds, downside_diagnostic=failures,
        source_cache_sha256=hashlib.sha256(Path(cache).read_bytes()).hexdigest(),
        forecast_deployment_approved=False, betting_approved=False, production_changed=False,
        limitations=['Refitted historical conservative recipe, not actual historical production forecasts.',
            'Proxy lines test probability skill, not offered-line edge or ROI.',
            'Historical low-volume label uses <50% of prior targets; remaining downside is not proof of an efficiency cause.',
            'Final live adjustments and ranking must also pass exact-lock replay and later prospective weeks.',
            'No production pointer, fixed trial selection, or bankroll rule is modified.'])
    final = fit_bundle(data)
    atomic_joblib(root/'model.joblib', dict(bundle=final, run_id=run_id, production_release=frozen['release_id'],
        created_at=report['built_at'], training_end=str(data.game_date_et.max())))
    atomic_joblib(root/'oof.joblib', dict(rows=r, lines=l))
    atomic_json(root/'report.json', clean(report))
    atomic_json(ROOT/'reports/nfl_receiving_coherent_repair_latest.json', clean(report))
    text = ['# NFL Receiving Coherent Repair', '', report['status'], '',
        '| Variant | Brier | Calibration | Coverage 80% | Interval score |', '|---|---:|---:|---:|---:|']
    for name, s in scores.items():
        text.append(f"| {name} | {s['probability']['brier']:.4f} | {s['probability']['calibration_error']:.2%} | {s['expected']['coverage_80']:.1%} | {s['interval_score']:.2f} |")
    text += ['', 'Week-grouped gated Brier gain: '+json.dumps(gain), '', *report['limitations']]
    (ROOT/'reports/nfl_receiving_coherent_repair_latest.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    if active_release() != frozen:
        raise RuntimeError('Release changed during experiment')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--seasons', type=int, nargs='+', default=[2024, 2025, 2026])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    from nfl_pipeline.modeling.receiving_coherent_repair import run as train
    print(json.dumps({k: v for k, v in train(args.cache, args.seasons).items() if k in ('status', 'run_id')}))
