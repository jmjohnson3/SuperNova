"""Leakage-safe final-stage calibration and selected-pick tests, offline only."""
from collections import Counter

import numpy as np
import pandas as pd

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.modeling.evaluation import probability_metrics, projection_metrics
from nfl_pipeline.modeling.challenger_models import curve_summary
from nfl_pipeline.modeling.prospective_market import locked_market
from nfl_pipeline.modeling.scoring_capture import replay
from nfl_pipeline.modeling.tail_calibration import TailPreservingCalibration


def verified_frame(records, frame):
    """Keep pending offers in the ranking pool; never reconstruct legacy inputs."""
    lookup = {int(r['id']): r for r in records}
    rows = []; excluded = Counter()
    for row in frame.to_dict('records'):
        rec = lookup[row['prediction_id']]; payload = rec['forecast_payload']
        captured = payload.get('scoring_replay')
        if not captured:
            excluded['missing_immutable_scoring_inputs'] += 1; continue
        try:
            result = replay(captured)
        except (ValueError, KeyError) as exc:
            excluded['replay_unavailable_'+type(exc).__name__] += 1; continue
        if result['side'] != row['side'] or abs(result['probability']-row['probability']) > 1e-10:
            excluded['original_final_probability_mismatch'] += 1; continue
        market = locked_market(rec)
        if market.get('market_probability') is None:
            excluded['no_verified_true_pair'] += 1; continue
        row['market'] = float(market['market_probability'])
        # The current frozen trial is paper-tier: legacy micro tier labels must
        # not silently erase its pre-cap candidate pool from the experiment.
        row['eligible'] = bool(row.get('research_eligible',row['eligible']))
        trace = result.get('probability_trace') or {}
        for column, key in (('raw','raw_over'), ('context','context_blend_over'), ('heuristic','heuristic_over')):
            value = trace.get(key)
            row[column] = None if value is None else float(value) if row['side']=='over' else 1-float(value)
        row['player_game'] = str(row['game_id'])+'|'+str(row['player_id'])
        rows.append(row)
    return pd.DataFrame(rows), dict(excluded)


def earlier_blocks(frame, test):
    """Fit/tune/gate labels must all have been observed before the first test lock."""
    cutoff = pd.to_datetime(test.locked_at, utc=True).min()
    known = pd.to_datetime(frame.label_available_at, utc=True, errors='coerce')
    locks = pd.to_datetime(frame.locked_at, utc=True, errors='coerce')
    first_week = min(zip(test.season, test.week))
    earlier = pd.Series([k < first_week for k in zip(frame.season, frame.week)], index=frame.index)
    history = frame.loc[earlier & known.notna() & known.lt(cutoff) & locks.lt(cutoff)
                        & frame.outcome.notna() & ~frame.game_id.isin(test.game_id)].copy()
    # Integer lines need a calibrated three-way push distribution, not a binary complement.
    history = history.loc[(history.line % 1 != 0) & history.push_probability.eq(0)]
    keys = sorted(set(zip(history.season, history.week)))
    if len(keys) < 3:
        return None
    history['probability'] = np.where(history.side.eq('over'), history.probability, 1-history.probability)
    history['outcome'] = np.where(history.side.eq('over'), history.outcome, 1-history.outcome)
    history['weight'] = 1/history.groupby('player_game').prediction_id.transform('size')
    membership = pd.Series(list(zip(history.season, history.week)), index=history.index)
    return (history.loc[membership.isin(keys[:-2])], history.loc[membership.map(lambda k:k==keys[-2])],
            history.loc[membership.map(lambda k:k==keys[-1])])


def population(frame, probability='probability'):
    from nfl_pipeline.modeling.receiving_selection_experiment import score
    result = score(frame, probability)
    if frame.empty:
        return result
    settled = frame.loc[frame.outcome.notna()]
    weights = 1/settled.groupby('player_game').prediction_id.transform('size')
    result['market'] = probability_metrics(settled.market, settled.outcome, weights)
    result['stages'] = {c:probability_metrics(settled[c],settled.outcome,weights)
                        for c in ('raw','context','heuristic','probability','calibrated') if c in settled}
    unique = frame.sort_values(['locked_at','prediction_id']).drop_duplicates('player_game')
    prefix = 'calibrated_' if probability=='calibrated' and 'calibrated_p10' in unique else ''
    unique = unique.loc[unique.actual.notna() & unique[prefix+'p10'].notna() & unique[prefix+'p90'].notna()]
    result['interval_metrics'] = projection_metrics(unique.actual, unique[prefix+'projection'], unique[prefix+'p10'], unique[prefix+'p90']) if len(unique) else None
    result['interval_source'] = 'complete_replayed_curve' if 'calibrated_p10' in frame else 'locked_projection_intervals_not_final_cdf'
    return result


def coherence_errors(frame, probability):
    """Only compare identical lock context, not different news/quote batches."""
    reversals = mismatched_same_line = 0
    for _, g in frame.groupby(['game_id','player_id','batch']):
        g = g.loc[(g.line % 1 != 0) & g.push_probability.eq(0)].copy()
        g['over'] = np.where(g.side.eq('over'),g[probability],1-g[probability])
        tails = g.groupby('line').over.agg(['min','max']).sort_index()
        mismatched_same_line += int((tails['max']-tails['min'] > 1e-8).sum())
        reversals += int((tails['max'].iloc[1:].to_numpy()-tails['min'].iloc[:-1].to_numpy() > 1e-8).sum())
    return dict(line_reversals=reversals,same_line_disagreement=mismatched_same_line)


def curve_calibration(frame, calibrator, curves):
    """Price the entire transformed final CDF, not independently adjusted lines."""
    frame=frame.copy(); verified=0
    for prefix in ('','calibrated_'):
        for name in ('projection','p10','p90'):
            frame[prefix+name]=np.nan
    for index,row in frame.iterrows():
        curve=curves.get(int(row.prediction_id))
        if curve is None or row.line % 1 == 0 or row.push_probability != 0:
            continue
        v,w=(np.asarray(a,dtype=float).reshape(1,-1) for a in curve)
        expected=float((w*(v>row.line)).sum())
        if abs((expected if row.side=='over' else 1-expected)-row.probability)>1e-8:
            continue
        cv,cw=calibrator.transform(v,w)
        over=float((cw*(cv>row.line)).sum())
        frame.loc[index,'calibrated']=over if row.side=='over' else 1-over
        for prefix,summary in (('',curve_summary(v,w)),('calibrated_',curve_summary(cv,cw))):
            for name,key in (('projection','mean'),('p10','p10'),('p90','p90')):
                frame.loc[index,prefix+name]=float(summary[key][0])
        verified+=1
    return frame,verified


def chronological(frame, curves=None):
    from nfl_pipeline.modeling.receiving_selection_experiment import ranked
    if frame.empty:
        return dict(status='no_verified_locks', folds=[])
    folds = []
    for cohort, group in frame.groupby(['model_version','scoring_version']):
        for week, test in group.groupby(['season','week'], sort=True):
            test = test.copy(); blocks = earlier_blocks(group, test)
            calibrator = TailPreservingCalibration()
            if blocks is not None:
                calibrator.fit(*blocks)
            valid = (test.line % 1 != 0) & test.push_probability.eq(0)
            over = np.where(test.side.eq('over'), test.probability, 1-test.probability)
            mapped = calibrator.predict(over)
            test['calibrated'] = np.where(valid, np.where(test.side.eq('over'), mapped, 1-mapped), test.probability)
            verified_curves=0
            if curves is not None:
                test,verified_curves=curve_calibration(test,calibrator,curves)
            current = ranked(test)
            conservative = ranked(test, penalty=.5, uncertainty_aware=True)
            revised = ranked(test, penalty=.5, uncertainty_aware=True, probability='calibrated')
            settled = test.loc[test.outcome.notna()]
            weights = 1/settled.groupby('player_game').prediction_id.transform('size')
            folds.append(dict(model_version=cohort[0], scoring_version=cohort[1], season=int(week[0]), week=int(week[1]),
                calibration=calibrator.validation if blocks is not None else {'enabled':False,'reason':'needs_three_earlier_label_available_weeks'},
                training_ids=[] if blocks is None else [b.prediction_id.astype(int).tolist() for b in blocks],
                test_ids=test.prediction_id.astype(int).tolist(),
                stages={c: probability_metrics(settled[c], settled.outcome, weights)
                        for c in ('raw','context','heuristic','probability','calibrated','market')},
                all_offers=population(test), current_selections=population(current),
                conservative_selections=population(conservative),
                calibrated_fixed_selections=population(current,'calibrated'),
                calibrated_reselected=population(revised,'calibrated'),
                exact_micro=population(test.loc[test.exact_micro]),
                fixed_research=population(test.loc[test.get('exact_research',pd.Series(False,index=test.index))]),
                coherence=dict(production=coherence_errors(test,'probability'),calibrated=coherence_errors(test,'calibrated')),
                calibration_curve_rows=verified_curves,
                calibration_coverage_verified=verified_curves==len(test)))
    return dict(status='offline_only', folds=folds, deployment_approved=False, betting_approved=False,
        blockers=['prospective_selected_probability_proof_required']+([] if all(f['calibration_coverage_verified'] for f in folds)
            else ['complete_final_curve_coverage_not_captured']),
        limitations=[
            'One survival calibration per release/scoring cohort; fit, strength selection, acceptance and test use separate chronological weeks.',
            'Label observation times, not scheduled game dates, control training eligibility. Pending picks still consume daily caps.',
            'Calibration follows every archived scoring adjustment. Raw/context/heuristic probabilities are diagnostics, not reconstructed missing inputs.',
            'The central map preserves 10/90 survival anchors and cannot introduce a new line reversal; it cannot repair an already incoherent final curve.',
            'Locked projection intervals are reported separately. They are not evidence of calibrated final-CDF coverage: archived line adjustments do not capture a complete final curve.',
            'Integer/push lines remain unchanged. No calibration or ranking result automatically deploys or authorizes cash.'])


def evaluate(records, frame):
    if frame.empty:
        return dict(status='no_offers', exclusions={})
    verified, excluded = verified_frame(records, frame)
    return dict(chronological(verified), exclusions=excluded)
