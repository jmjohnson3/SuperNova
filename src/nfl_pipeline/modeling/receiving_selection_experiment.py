"""Exact-lock receiving scoring/ranking experiments. Never publish or lock a bet."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, atomic_json
from nfl_pipeline.modeling.challenger_models import curve_summary
from nfl_pipeline.modeling.evaluation import probability_metrics, projection_metrics, clustered_gain
from nfl_pipeline.modeling.live_scoring_replay import load_rows, validate_lock, ROOT
from nfl_pipeline.modeling.final_probability_validation import load_ledger, attach_label_availability
from nfl_pipeline.modeling.predict_player_props import _ev_per_unit
from nfl_pipeline.modeling.receiving_coherent_repair import variants, coherent_after_adjustments, STAT
from nfl_pipeline.modeling.score_accuracy_components import full_path


def ranked(frame, penalty=0., probability='probability', cap=5, uncertainty_aware=False):
    """Chronological batches with shared daily/player caps; outcomes are not read."""
    selected = []
    for _, day in frame.groupby(['model_version', 'scoring_version', 'day'], sort=True):
        used = set(); remaining = cap
        for _, batch in day.groupby('batch', sort=True):
            pool = batch.loc[batch.eligible & batch.market.notna()].copy()
            for column in ('edge', 'confidence'):
                if column not in pool:
                    pool[column] = 0.
            uncertainty = pool.get('uncertainty_proxy', pd.Series(1., index=pool.index)).fillna(1.).clip(0, 1)
            discount = penalty * (.5 + .5*uncertainty) if uncertainty_aware else penalty
            pool['score'] = (1-pool.push_probability)*(pool[probability]*(1+pool.payout)-1
                -discount*(pool[probability]-pool.market).abs()*(1+pool.payout))
            pool = pool.loc[pool.score.gt(0)].sort_values(['score', 'edge', 'confidence', 'prediction_id'],
                ascending=[False, False, False, True], kind='stable')
            for index, row in pool.iterrows():
                key = (row.game_id, row.player_id)
                if remaining and key not in used:
                    selected.append(index); used.add(key); remaining -= 1
    return frame.loc[selected].copy()


def score(frame, probability='probability'):
    if frame.empty:
        return dict(rows=0, pending=0)
    settled = frame.loc[frame.outcome.notna()]
    weights = 1/settled.groupby(['model_version', 'scoring_version', 'game_id', 'player_id']).prediction_id.transform('size')
    return dict(probability_metrics(settled[probability], settled.outcome, weights),
        selected_rows=len(frame), pending=int((frame.actual.isna() & ~frame.get('is_void',pd.Series(False,index=frame.index))).sum()),
        voids=int(frame.get('is_void',pd.Series(False,index=frame.index)).sum()), pushes=int(frame.is_push.sum()),
        independent_weeks=len(set(zip(settled.season, settled.week))),
        player_games=len(set(zip(settled.game_id, settled.player_id))),
        mean_disagreement=float((settled[probability]-settled.market).abs().mean()) if len(settled) else None,
        ids=frame.prediction_id.astype(int).tolist())


def offer_frame(records, ledger):
    by_id = {int(l['prediction_id']): l for l in ledger}; rows = []; excluded = Counter()
    for r in records:
        p = r.get('forecast_payload') or {}
        if r['stat'] != STAT or r.get('line') is None:
            continue
        error = validate_lock(r)
        if error:
            excluded[error] += 1; continue
        if p.get('book') != 'fanduel' or p.get('side') not in ('over', 'under'):
            excluded['not_fanduel_receiving_side'] += 1; continue
        market = p.get('market_no_vig_probability')
        if market is None or not np.isfinite(float(market)) or not 0 < float(market) < 1 or _ev_per_unit(p.get('probability'), p.get('price')) is None:
            excluded['missing_probability_price_or_paired_market'] += 1; continue
        price = float(p['price']); actual = r.get('actual'); line = float(r['line'])
        is_push = actual is not None and float(actual) == line
        outcome = None if actual is None or is_push else float(float(actual) > line if p['side'] == 'over' else float(actual) < line)
        l = by_id.get(int(r['id'])); micro = False
        if l:
            when = safe_time(l.get('locked_at_utc'))
            micro = bool(when and safe_time(r['created_at_utc']) <= when < safe_time(r['start_ts_utc'])
                and all(str(l.get('ledger_'+k)) == str(p.get(k)) for k in ('book', 'side', 'model_version'))
                and float(l.get('ledger_line', -999)) == line and float(l.get('ledger_price', 0)) == price)
        low, high = p.get('projection_p10'), p.get('projection_p90')
        width = (float(high)-float(low)) if low is not None and high is not None else np.nan
        projection = float(p.get('projection') or 0)
        quote = (p.get('scoring_replay') or {}).get('offer') or {}
        fetched, locked = safe_time(quote.get('fetched_at_utc')), safe_time(r['created_at_utc'])
        link = urlparse(str(quote.get(p['side']+'_link') or ''))
        fresh = fetched and locked and 0 <= (locked-fetched).total_seconds() <= 1200
        research_eligible = bool(fresh and quote.get('market_key') == 'player_reception_yds'
            and p.get('drift_guard_pass') is True and _ev_per_unit(p['probability'],price) > 0
            and link.scheme == 'https' and (link.hostname=='fanduel.com' or (link.hostname or '').endswith('.fanduel.com')))
        rows.append(dict(prediction_id=int(r['id']), game_id=r['game_id'], player_id=r['player_id'],
            model_version=p['model_version'], scoring_version=(p.get('scoring_replay') or {}).get('scoring_fingerprint', 'legacy_unknown'),
            day=str(p['game_date_et']), batch=str(p['prediction_context_cutoff_utc']),
            season=r['season'], week=r['week'], line=line, side=p['side'], actual=actual, outcome=outcome,
            is_push=is_push, is_void=r.get('graded_result')=='void_nonparticipant', probability=float(p['probability']), market=float(market),
            locked_at=r['created_at_utc'], label_available_at=r.get('label_available_at'),
            projection=projection, p10=low, p90=high,
            uncertainty_proxy=min(1., width/(2*max(10., abs(projection)))) if np.isfinite(width) and width >= 0 else 1.,
            research_eligible=research_eligible,
            edge=float(p.get('model_market_edge') or 0), confidence=float(p.get('projection_confidence') or 0),
            payout=price/100 if price > 0 else 100/-price, push_probability=float(p.get('push_probability') or 0),
            eligible=bool(p.get('drift_guard_pass') and (p.get('tier') == 'micro_projection'
                or 'micro_daily_cap_research_only' in str(p.get('reasons', '')))), exact_micro=micro))
    return pd.DataFrame(rows), dict(excluded)


def evaluate_ranking(frame):
    if frame.empty:
        return dict(status='no_eligible_locks')
    current = ranked(frame)
    conservative = ranked(frame, penalty=.5)
    uncertainty = ranked(frame, penalty=.5, uncertainty_aware=True)
    micro = frame.loc[frame.exact_micro]
    return dict(status='descriptive_only', current=score(current), conservative=score(conservative),
        uncertainty_aware=score(uncertainty), exact_micro=score(micro),
        chronological_weeks=[dict(model_version=k[0], scoring_version=k[1], season=int(k[2]), week=int(k[3]),
            all_offers=score(g), current=score(ranked(g)),
            uncertainty_aware=score(ranked(g, penalty=.5, uncertainty_aware=True)))
            for k,g in frame.groupby(['model_version','scoring_version','season','week'])],
        by_scoring_cohort={str(k): dict(current=score(ranked(g)), conservative=score(ranked(g, penalty=.5)),
             exact_micro=score(g.loc[g.exact_micro])) for k, g in frame.groupby(['model_version', 'scoring_version'])},
        approved=False, penalty=.5, limitations=[
            'Fixed disagreement penalty; no tuning on these results. Missing market evidence cannot rank.',
            'Both policies use the same stored pre-cap eligibility and chronological quote batches, never future offers.',
            'At most five picks per release/scoring cohort/day and one per player-game; this is an experiment, not a new wager cap.',
            'Push and pending candidates remain in selection; binary metrics exclude them, not the selection algorithm.',
            'The conservative score changes ranking/abstention, not the reported win probability.',
            'Uncertainty-aware discount is fixed before evaluation: half to full disagreement penalty, scaled by locked interval width relative to projection. Missing uncertainty receives the full penalty.',
            'Historical reconstruction is separate from exact ledger micro selections. Simulated records are not cash wagers.'])


def evaluate_full_path(records, artifact, frame):
    bundle = artifact['bundle']; lookup = {int(r['id']): r for r in records}; excluded = Counter(); groups = {}
    for row in frame.itertuples():
        r = lookup[row.prediction_id]; p = r['forecast_payload']
        if not p.get('scoring_replay') or not p.get('forecast_features'):
            excluded['missing_immutable_scoring_inputs'] += 1; continue
        if str(artifact['training_end']) >= safe_time(p['prediction_context_cutoff_utc']).date().isoformat():
            excluded['training_not_before_lock_date'] += 1; continue
        if p['model_version'] != artifact['production_release']:
            excluded['wrong_production_cohort'] += 1; continue
        if float(p['line']).is_integer():
            excluded['integer_line_requires_separate_push_curve_validation'] += 1; continue
        key = (row.model_version, row.scoring_version, row.game_id, row.player_id, row.batch)
        groups.setdefault(key, []).append(r)
    output = []; curve_rows = []; final_curves = {}
    for group in groups.values():
        p = group[0]['forecast_payload']
        features = pd.DataFrame([p['forecast_features']])
        # One coherent curve per player/book/lock, never a different model per line.
        curves = variants(bundle, features)
        for family in ('coherent_base', 'coherent_downside'):
            v, w = curves[family]; s = curve_summary(v, w)
            kept = []; adjusted = []; traces = []
            for r in group:
                q = r['forecast_payload']
                if q['forecast_features'] != p['forecast_features']:
                    excluded['inconsistent_context_within_curve'] += 1; continue
                try:
                    replayed = full_path(q, v[0], w[0], float(s['mean'][0]))
                except ValueError as exc:
                    excluded[str(exc)] += 1; continue
                kept.append(r); adjusted.append(replayed['calibrated_over_probability']); traces.append(replayed['probability_trace'])
            if not kept:
                continue
            lines = [float(r['line']) for r in kept]
            cv, cw = coherent_after_adjustments(v[0], w[0], lines, adjusted)
            coherent = ((cv[None, :] > np.asarray(lines)[:, None])*cw).sum(axis=1)
            cs = curve_summary(cv[None, :], cw[None, :])
            if kept[0].get('actual') is not None:
                curve_rows.append(dict(family=family, model_version=p['model_version'], scoring_version=group[0]['forecast_payload']['scoring_replay']['scoring_fingerprint'],
                    game_id=kept[0]['game_id'], player_id=kept[0]['player_id'], batch=p['prediction_context_cutoff_utc'],
                    actual=float(kept[0]['actual']), mean=float(cs['mean'][0]), p10=float(cs['p10'][0]), p90=float(cs['p90'][0]),
                    production=float(p['projection']), production_p10=p.get('projection_p10'), production_p90=p.get('projection_p90')))
            for i, r in enumerate(kept):
                side = r['forecast_payload']['side']
                flip = lambda x: float(x) if side == 'over' else 1-float(x)
                trace = traces[i]
                output.append(dict(prediction_id=int(r['id']), family=family, raw=flip(trace['raw_over']),
                    heuristic=flip(trace['heuristic_over']), context_blend=None if trace.get('context_blend_over') is None else flip(trace['context_blend_over']),
                    after_all_live_adjustments=flip(adjusted[i]), coherent_probability=flip(coherent[i])))
                final_curves.setdefault(family,{})[int(r['id'])] = (cv,cw)
    if not output:
        return dict(status='no_replayable_rows', exclusions=dict(excluded), deployment_approved=False)
    combined = frame.merge(pd.DataFrame(output), on='prediction_id', validate='one_to_many')
    summary = {}
    for family, g in combined.groupby('family'):
        base = ranked(g)
        chosen = ranked(g, penalty=.5, probability='coherent_probability')
        settled = g.loc[g.outcome.notna()].copy()
        weights = 1/settled.groupby(['game_id', 'player_id']).prediction_id.transform('size')
        stages = {col: probability_metrics(settled[col], settled.outcome, weights) for col in
                  ('probability', 'raw', 'context_blend', 'heuristic', 'after_all_live_adjustments', 'coherent_probability', 'market')}
        by_cohort = {}
        for cohort, part in settled.groupby(['model_version', 'scoring_version']):
            pw = 1/part.groupby(['game_id', 'player_id']).prediction_id.transform('size')
            by_cohort['|'.join(cohort)] = {col: probability_metrics(part[col], part.outcome, pw)
                for col in ('probability', 'after_all_live_adjustments', 'coherent_probability', 'market')}
        err = settled.assign(reference_error=(settled.probability-settled.outcome)**2,
                             challenger_error=(settled.coherent_probability-settled.outcome)**2)
        gain = clustered_gain(err.groupby(['season', 'week', 'game_id', 'player_id'])[['reference_error', 'challenger_error']].mean().reset_index())
        c = pd.DataFrame(curve_rows)
        c = c.loc[c.family.eq(family)].sort_values('batch').drop_duplicates(['game_id', 'player_id', 'scoring_version'])
        intervals = dict(challenger=projection_metrics(c.actual, c['mean'], c.p10, c.p90),
            production=projection_metrics(c.actual, c.production, c.production_p10, c.production_p90)) if len(c) else {}
        from nfl_pipeline.modeling.receiving_later_week_validation import chronological
        calibrated_frame = g.copy()
        calibrated_frame['probability'] = calibrated_frame.coherent_probability
        calibrated_frame['context'] = calibrated_frame.context_blend
        calibrated_frame['player_game'] = calibrated_frame.game_id.astype(str)+'|'+calibrated_frame.player_id.astype(str)
        calibrated_frame['eligible'] = calibrated_frame.get('research_eligible',calibrated_frame.eligible)
        summary[family] = dict(stages=stages, by_scoring_cohort=by_cohort, clustered_brier_gain=gain, intervals=intervals,
            later_week_curve_calibration=chronological(calibrated_frame,final_curves[family]),
            fixed_current_selections=dict(production=score(base), challenger=score(base, 'coherent_probability')),
            conservative_selections=score(chosen, 'coherent_probability'),
            exact_micro=dict(production=score(g.loc[g.exact_micro]), challenger=score(g.loc[g.exact_micro], 'coherent_probability')))
    return clean(dict(status='retrospective_exact_input_experiment', summary=summary, rows=output, exclusions=dict(excluded),
        deployment_approved=False, betting_approved=False, limitations=[
            'This artifact was built after historical locks: results are retrospective, never prospective credit.',
            'Each eligible counterfactual uses genuine original scoring inputs, archived adjustment code, and the original side/price.',
            'Final coherence follows all live adjustments; intermediate and final probabilities are separately scored.',
            'The last coherence map preserves 10/90 survival anchors. Missing historical inputs remain excluded.',
            'Historical row dates before or equal to training_end cannot enter evaluation.',
            'Acceptance needs later prospective weeks and selected-pick proof; no automatic deployment or cash approval.']))


def run(model=None):
    records = attach_label_availability(load_rows(stat=STAT)); frame, excluded = offer_frame(records, load_ledger())
    from nfl_pipeline.modeling import receiving_research_trial as trial
    registration = trial.load_registration()
    fixed_ids = {int(r['forecast_id']) for d in trial.captures(registration) for r in d['selected']} if registration else set()
    if len(frame):
        frame['exact_research'] = frame.prediction_id.isin(fixed_ids)
    ranking = evaluate_ranking(frame)
    full = evaluate_full_path(records, joblib.load(model), frame) if model and len(frame) else {'status': 'no_model_or_locks'}
    from nfl_pipeline.modeling.receiving_later_week_validation import evaluate
    later = evaluate(records, frame)
    report = clean(dict(built_at=datetime.now(timezone.utc).isoformat(), ranking=ranking, full_path=full,
        later_week_validation=later, exclusions=excluded, production_changed=False, deployment_approved=False, betting_approved=False))
    atomic_json(ROOT/'reports/nfl_receiving_selection_experiment_latest.json', report)
    if model:
        atomic_json(ROOT/'reports/nfl_receiving_full_path_experiment_latest.json', report)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    atomic_json(MODEL_ROOT/'receiving_selection_experiment'/stamp/'report.json', report)
    text = ['# Receiving Selection Experiment', '', 'Retrospective diagnostic. Production/trial unchanged. No betting approval.', '',
            '## Fixed Ranking Comparison', '', json.dumps({k: ranking.get(k) for k in ('current', 'conservative', 'exact_micro')}, indent=2), '',
            '## Later-Week Validation', '', json.dumps(later, indent=2), '',
            '## Full Scoring Path', '', full['status']]
    for name, s in full.get('summary', {}).items():
        text += ['', name, 'Stages: '+json.dumps(s['stages']), 'Intervals: '+json.dumps(s['intervals']),
                 'Exact micro: '+json.dumps(s['exact_micro'])]
    text += ['', *ranking.get('limitations', []), *full.get('limitations', [])]
    (ROOT/'reports/nfl_receiving_selection_experiment_latest.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    if model:
        (ROOT/'reports/nfl_receiving_full_path_experiment_latest.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--model', type=Path)
    args = parser.parse_args(); result = run(args.model)
    print(json.dumps(dict(status=result['full_path']['status'], ranking=result['ranking'].get('current'))))
