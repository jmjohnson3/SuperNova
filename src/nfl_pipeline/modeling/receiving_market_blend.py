"""Version-aware receiving blend and selected-offer comparison, offline only."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling.fit_market_calibration import prepare_captured, fit_captured_stat, load_calibration_records
from nfl_pipeline.modeling.live_scoring_replay import ROOT
from nfl_pipeline.modeling.final_probability_validation import attach_label_availability, load_ledger
from nfl_pipeline.modeling.receiving_selection_experiment import offer_frame, ranked
from nfl_pipeline.modeling.receiving_quantile_validation import eligible_error
from nfl_pipeline.modeling.receiving_quantiles import CONTRACT as QUANTILE_CONTRACT
from nfl_pipeline.modeling.train_receiving_quantiles import curves
from nfl_pipeline.modeling.challenger_models import curve_summary
from nfl_pipeline.modeling.score_accuracy_components import empirical_residuals
from nfl_pipeline.modeling.evaluation import probability_metrics, projection_metrics, clustered_gain
from nfl_pipeline.modeling import receiving_research_trial as trial


def challenger_captures(frame, records, artifact, digest):
    if artifact.get('contract') != QUANTILE_CONTRACT:
        raise ValueError('Wrong receiving challenger contract')
    lookup = {int(r['id']): r for r in records}
    kept, excluded = [], Counter()
    for row in frame.to_dict('records'):
        error = eligible_error(lookup[int(row['id'])], artifact)
        if error:
            excluded[error] += 1
        else:
            kept.append(row)
    if not kept:
        return pd.DataFrame(), dict(excluded)
    features = pd.DataFrame([r['captured']['row'] for r in kept])
    values, weights = curves(artifact['bundle'], features)['calibrated_blend']
    summary = curve_summary(values, weights)
    for i, row in enumerate(kept):
        cap = deepcopy(row['captured'])
        anchor = float(summary['median'][i])
        cap['projection'] = anchor
        cap['distribution'] = dict(kind='empirical_oof_residual',
            residual_quantiles=empirical_residuals(values[i], weights[i], anchor),
            projection_confidence=(cap.get('distribution') or {}).get('projection_confidence', .5))
        row.update(captured=cap, variant='calibrated_blend:'+artifact['run_id']+':'+digest)
    return pd.DataFrame(kept), dict(excluded)


def selected_comparison(rows, offers, fixed_ids):
    """The same stored side/price and preselection universe for every variant."""
    if not rows:
        return dict(status='no_forward_test_rows')
    fit = pd.DataFrame(rows).rename(columns={'id':'prediction_id'})
    fields = ['prediction_id','p_over','raw_probability','p10','p90','projection','is_push']
    fit = fit[fields].rename(columns={'p10':'replayed_p10','p90':'replayed_p90','projection':'replayed_projection', 'is_push':'replayed_push'})
    frame = offers.merge(fit, on='prediction_id', validate='one_to_one')
    frame['blend_probability'] = np.where(frame.side.eq('over'),frame.p_over,1-frame.p_over)
    frame['raw_model_probability'] = np.where(frame.side.eq('over'),frame.raw_probability,1-frame.raw_probability)
    # Probability-independent eligibility. Requiring the old model's EV here
    # would discard precisely the decisions the challenger is meant to test.
    pools = dict(all_offers=frame.loc[frame.eligible], exact_micro=frame.loc[frame.exact_micro],
        fixed_research=frame.loc[frame.prediction_id.isin(fixed_ids)],
        production_ranked=ranked(frame), blend_ranked=ranked(frame, probability='blend_probability'),
        conservative_blend_ranked=ranked(frame, penalty=.5, probability='blend_probability', uncertainty_aware=True))
    results = {}
    for name, pool in pools.items():
        settled = pool.loc[pool.outcome.notna()].copy()
        weight = 1/settled.groupby(['game_id','player_id']).prediction_id.transform('size')
        metrics = {col: probability_metrics(settled[col], settled.outcome, weight)
            for col in ('probability','raw_model_probability','blend_probability','market')}
        unique = pool.sort_values(['batch','prediction_id']).drop_duplicates(['game_id','player_id'])
        errors = settled.assign(reference_error=(settled.market-settled.outcome)**2,
            challenger_error=(settled.blend_probability-settled.outcome)**2)
        gain = clustered_gain(errors.groupby(['season','week','game_id','player_id'])[
            ['reference_error','challenger_error']].mean().reset_index())
        voids = pool.get('is_void', pd.Series(False, index=pool.index)).fillna(False)
        results[name] = dict(selected_rows=len(pool), settled=len(settled),
            pending=int((pool.actual.isna() & ~voids).sum()), voids=int(voids.sum()),
            weeks=len(settled[['season','week']].drop_duplicates()), probabilities=metrics,
            market_brier_gain=gain, ids=pool.prediction_id.astype(int).tolist(),
            replayed_stat_intervals=projection_metrics(unique.actual,unique.replayed_projection,unique.replayed_p10,unique.replayed_p90),
            interval_scope='Replayed stat curve; a per-line market blend is not a certified joint CDF.')
    violations=0
    for _,group in frame.groupby(['game_id','player_id','batch']):
        by_line=group.groupby('line').p_over.mean().sort_index()
        violations+=int((np.diff(by_line)>1e-10).sum())
    results['probability_consistency']=dict(monotonicity_violations=violations,
        approved=False,note='Per-offer blends do not grant whole-CDF calibration approval.')
    return results


def run(artifact_path=None):
    records = attach_label_availability(load_calibration_records('receiving_yards'))
    print('Validating archived scoring inputs',len(records),flush=True)
    ledger=load_ledger()
    registration=trial.load_registration()
    fixed_ids={int(r['forecast_id']) for d in trial.captures(registration) for r in d.get('selected',[])} if registration else set()
    preserve_ids=fixed_ids | {int(r['prediction_id']) for r in ledger}
    common=[r for r in records if ((r.get('forecast_payload') or {}).get('scoring_replay') or {}).get('offer',{}).get('market_key')=='player_reception_yds']
    captured, excluded = prepare_captured(common,preserve_ids)
    excluded['outside_common_line_strategy']=len(records)-len(common)
    offers, offer_exclusions = offer_frame(records, ledger)
    from nfl_pipeline.modeling.scoring_equivalence import verified_aliases, cohort_for
    aliases = verified_aliases()
    if not offers.empty:
        offers['source_scoring_version'] = offers.scoring_version
        offers['scoring_version'] = [cohort_for(r.scoring_version, r.prediction_id, aliases) for r in offers.itertuples()]
    lookup = {int(r['id']):r for r in records}
    if not offers.empty:
        def usable(row):
            from urllib.parse import urlparse
            from nfl_pipeline.context_contract import safe_time
            p=lookup[row.prediction_id]['forecast_payload'];q=(p.get('scoring_replay') or {}).get('offer') or {}
            at=safe_time(q.get('fetched_at_utc'));lock=safe_time(row.locked_at)
            link=urlparse(str(q.get(row.side+'_link') or ''))
            return bool(at and lock and 0 <= (lock-at).total_seconds() <= 1200
                and q.get('market_key')=='player_reception_yds' and link.scheme=='https'
                and (link.hostname=='fanduel.com' or (link.hostname or '').endswith('.fanduel.com')))
        offers['eligible'] = [usable(r) for r in offers.itertuples()]
    variants = [captured]
    artifact_metadata = None
    if artifact_path and not captured.empty:
        artifact=joblib.load(artifact_path);digest=hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        changed, missing=challenger_captures(captured,records,artifact,digest)
        variants.append(changed);excluded.update({'challenger:'+k:v for k,v in missing.items()})
        artifact_metadata=dict(run_id=artifact['run_id'],sha256=digest,training_end=artifact['training_end'])
    cohorts={}
    for frame in variants:
        if frame.empty:continue
        for key,group in frame.groupby(['model_version','scoring_version','variant']):
            print('Forward blend',key,'rows',len(group),flush=True)
            fitted=fit_captured_stat('receiving_yards',group)
            fitted['selected_comparison']=selected_comparison(fitted.pop('rows'),offers,fixed_ids)
            cohorts['|'.join(key)]=fitted
    report=clean(dict(contract='nfl-receiving-market-blend-v1',built_at=datetime.now(timezone.utc).isoformat(),
        cohorts=cohorts,input_rows=len(records),exclusions=excluded,offer_exclusions=offer_exclusions,
        challenger=artifact_metadata,production_changed=False,deployment_approved=False,betting_approved=False,
        limitations=['Retrospective development, not prospective acceptance or cash authorization.',
            'Earlier observed labels only; no current context or replacement of historical locks.',
            'Forecast version, scoring version and challenger checksum remain separate.',
            'Rankings are counterfactual on archived offers, not the complete market or actual wagers.',
            'Fixed research and exact micro IDs are never reselected after settlement.',
            'Whole-distribution coverage and line-level blending are different claims.']))
    atomic_json(ROOT/'reports/nfl_receiving_market_blend_latest.json',report)
    lines=['# Receiving Market Blend: Forward Full-Path Replay','',
        'Production and betting approval unchanged. Results are development evidence.','',
        '## Evidence Status','',
        '| Scoring version / curve | Season-weeks | Later-week rows | Status |',
        '|---|---|---:|---|']
    for c in report['cohorts'].values():
        cohort=c.get('cohort') or {};evidence=c.get('evidence') or {}
        label=cohort.get('scoring_version','unknown')[:12]+' / '+cohort.get('variant','unknown').split(':')[0]
        lines.append(f"| {label} | {evidence.get('season_weeks', [])} | {evidence.get('forward_test_rows', 0)} | {c['status']} |")
    lines += ['', 'No later-week rows means no accepted calibration or selected-pick improvement; in-sample fits are diagnostic only.', '',
        '## Forward Comparisons','',
        '| Cohort / selected pool | Settled | Weeks | Production Brier | Blend Brier | Market Brier |',
        '|---|---:|---:|---:|---:|---:|']
    for name,c in report['cohorts'].items():
        for pool,m in c['selected_comparison'].items():
            if not isinstance(m,dict) or 'probabilities' not in m:continue
            scores=m['probabilities']
            vals=[f"{scores[k]['brier']:.4f}" if scores[k].get('rows') else '-' for k in ('probability','blend_probability','market')]
            lines.append(f"| {name} / {pool} | {m['settled']} | {m['weeks']} | "+' | '.join(vals)+' |')
    lines+=['',*report['limitations'],'','Exclusions: '+json.dumps(excluded)]
    (ROOT/'reports/nfl_receiving_market_blend_latest.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path)
    a=p.parse_args();r=run(a.artifact)
    print(json.dumps(dict(cohorts=len(r['cohorts']),exclusions=r['exclusions'])))
