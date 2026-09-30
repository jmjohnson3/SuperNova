"""Immutable benchmark shadows on real offers and exact original micro picks."""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, active_release, atomic_json
from nfl_pipeline.modeling.challenger_models import curve_summary, player_features
from nfl_pipeline.modeling.evaluation import clustered_gain, probability_metrics, projection_metrics
from nfl_pipeline.modeling.final_probability_validation import load_ledger
from nfl_pipeline.modeling.live_scoring_replay import load_rows, validate_lock, ROOT
from nfl_pipeline.modeling.model_benchmark import CONTRACT, SUPPORTED_CONTRACTS, named_curve
from nfl_pipeline.modeling.score_accuracy_components import full_path
from nfl_pipeline.modeling.receiver_role_model import context_inputs
from nfl_pipeline.modeling.workload_depth_model import historical_features
from nfl_pipeline.modeling.prospective_market import locked_market, market_comparison
from nfl_pipeline.modeling import receiving_research_trial

STORE = MODEL_ROOT/'model_benchmark'


def load_artifact(*, latest=False):
    pointer = STORE/'latest.json' if latest else STORE/'active_trial.json'
    if not pointer.exists() and not latest:
        raise ValueError('No pinned benchmark trial; use --pin-current before collecting evidence')
    if not pointer.exists():
        return None, None
    manifest = json.loads(pointer.read_text())
    path = STORE/manifest['run_id']/'models.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Benchmark artifact checksum mismatch')
    artifact = joblib.load(path)
    if artifact['contract'] not in SUPPORTED_CONTRACTS or artifact['production_release'] != active_release()['release_id']:
        raise ValueError('Benchmark contract or frozen production mismatch')
    return artifact, manifest


def pin_current():
    artifact, manifest = load_artifact(latest=True)
    if artifact is None:
        raise ValueError('No benchmark artifact to pin')
    target = STORE/'active_trial.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('A different trial is pinned; finish its review before starting another')
    atomic_json(target, manifest)
    return dict(status='trial_pinned', run_id=artifact['run_id'], production_changed=False, betting_approved=False)


def ledger_matches(ledger, record, scored_at):
    p = record['forecast_payload']
    when = safe_time(ledger.get('locked_at_utc'))
    created = safe_time(record.get('created_at_utc'))
    if not when or not created or not created <= when <= scored_at:
        return False
    if int(ledger['prediction_id']) != int(record['id']):
        return False
    if any(str(ledger.get('ledger_'+k)) != str(p.get(k)) for k in ('book', 'side', 'model_version')):
        return False
    try:
        return float(ledger['ledger_line']) == float(p['line']) and int(ledger['ledger_price']) == int(p['price'])
    except (ValueError, TypeError, KeyError):
        return False


def prospective_error(record, artifact, now):
    error = validate_lock(record)
    if error:
        return error
    p = record['forecast_payload']
    cutoff = safe_time(p.get('prediction_context_cutoff_utc'))
    made = safe_time(artifact.get('created_at'))
    start = safe_time(record.get('start_ts_utc'))
    created = safe_time(record.get('created_at_utc'))
    if not made or not made <= cutoff <= created <= now < start:
        return 'artifact_or_scoring_not_prospective_at_original_lock'
    if date.fromisoformat(artifact['training_end']) >= start.date():
        return 'training_includes_forecast_date'
    if p.get('model_version') != artifact['production_release']:
        return 'production_release_mismatch'
    if p.get('line') is None or p.get('side') not in ('over', 'under'):
        return 'not_an_offered_line'
    if not p.get('forecast_features') or not p.get('scoring_replay'):
        return 'missing_exact_lock_inputs'
    return None


def scoring_frame(records, artifact):
    requests = []
    for r in records:
        p = r['forecast_payload']
        requests.append(dict(p['forecast_features'], game_id=r['game_id'], player_id=str(r['player_id']),
            game_date_et=date.fromisoformat(str(p['game_date_et'])[:10]),
            team_abbr=p['team_abbr'], position=p['position'], context_evidence=p.get('context_evidence'),
            prediction_context_cutoff_utc=p['prediction_context_cutoff_utc']))
    frame = pd.DataFrame(requests)
    unique = frame.drop_duplicates(['game_id', 'player_id'])
    history = artifact['history']
    wd = historical_features(history, unique)
    # All historical outcomes come from the frozen, pre-lock artifact, not a current query.
    extra = pd.concat([unique[['game_id', 'player_id']], wd], axis=1)
    frame = frame.drop(columns=[c for c in frame if c.startswith('wd_')])
    frame = frame.merge(extra, on=['game_id', 'player_id'], how='left', validate='many_to_one')
    # Offers for one player can lock at different times. Historical skill is shared,
    # but each offer keeps its own timestamped availability/depth evidence.
    contexts = pd.DataFrame([{'wd_'+k[3:]: v for k, v in context_inputs(r).items()}
                             for r in requests], index=frame.index)
    for column in contexts:
        frame[column] = contexts[column]
    return frame


def score_records(records, ledger, artifact, now):
    issues = Counter(); output = []
    by_prediction = {}
    for l in ledger:
        by_prediction.setdefault(int(l['prediction_id']), []).append(l)
    for stat, suite in artifact['models'].items():
        selected = []
        for r in records:
            if r['stat'] != stat:
                continue
            error = prospective_error(r, artifact, now)
            if error:
                issues[error] += 1
            else:
                selected.append(r)
        if not selected:
            continue
        frame = scoring_frame(selected, artifact)
        choices = suite['choices']
        v, w = named_curve(suite, frame, stat, choices['probability'] or 'reference')
        mean_curve = named_curve(suite, frame, stat, choices['expected'])
        median_curve = named_curve(suite, frame, stat, choices['typical'])
        point = curve_summary(*mean_curve)['mean']
        median = curve_summary(*median_curve)['median']
        curves = [('selected_policy', v, w, point, median)]
        # Isolate uncertainty from changing a conservative live point forecast.
        current = np.array([float(r['forecast_payload']['projection']) for r in selected])
        u = suite['candidates']['reference_conditional']['uncertainty']
        cv, cw = u.mixture(player_features(frame, stat), current, np.ones((len(frame), 1)))
        curves.append(('production_point_conditional', cv, cw, current, curve_summary(cv, cw)['median']))
        names = {'stable_ensemble': 'ensemble', 'stable_ensemble_calibrated': 'ensemble+tail_cal',
                 'qb_tail': 'qb_tail', 'qb_tail_calibrated': 'qb_tail+tail_cal'}
        for variant in suite.get('prospective_variants', []):
            cv, cw = named_curve(suite, frame, stat, names[variant])
            s = curve_summary(cv, cw)
            curves.append((variant, cv, cw, s['mean'], s['median']))
        for variant, values, weights, centers, medians in curves:
            summary = curve_summary(values, weights)
            for i, r in enumerate(selected):
                p = r['forecast_payload']
                try:
                    replayed = full_path(p, values[i], weights[i], centers[i])
                except ValueError as exc:
                    issues[str(exc)] += 1; continue
                micros = [l for l in by_prediction.get(int(r['id']), []) if ledger_matches(l, r, now)]
                over = replayed['calibrated_over_probability']
                output.append(dict(forecast_id=int(r['id']), variant=variant, stat=stat,
                    production_release=p['model_version'], source_context_cutoff=p['prediction_context_cutoff_utc'],
                    source_book=p['book'], source_line=p['line'], source_price=p['price'], source_side=p['side'],
                    ledger_ids=[l['ledger_id'] for l in micros], choices=choices,
                    expected_yards=float(centers[i]), median_yards=float(medians[i]),
                    distribution_mean=float(summary['mean'][i]),
                    raw_p10=float(summary['p10'][i]), raw_p90=float(summary['p90'][i]),
                    same_side_probability=float(over if p['side'] == 'over' else 1-over),
                    production_probability=float(p['probability']), **locked_market(r), **replayed))
    return output, dict(issues)


def score(day):
    artifact, manifest = load_artifact()
    if artifact is None:
        return {'status': 'waiting_for_benchmark'}
    from nfl_pipeline.game_scope import filter_records
    records = filter_records(load_rows(day)); ledger = load_ledger()
    # Do not replace an earlier shadow after results or a different scoring revision.
    existing = set()
    for path in (STORE/artifact['run_id']/'prospective'/str(day)).glob('*.json'):
        for row in json.loads(path.read_text()).get('rows', []):
            existing.add((int(row['forecast_id']), row['variant']))
    pending = [r for r in records if not all((int(r['id']), v) in existing
               for v in ('selected_policy', 'production_point_conditional',
                         *artifact['models'].get(r['stat'], {}).get('prospective_variants', [])))]
    rows, issues = score_records(pending, ledger, artifact, datetime.now(timezone.utc))
    at = datetime.now(timezone.utc)
    starts = {int(r['id']): safe_time(r['start_ts_utc']) for r in records}
    rows = [r for r in rows if (r['forecast_id'], r['variant']) not in existing and at < starts[r['forecast_id']]]
    doc = clean(dict(contract=artifact['contract'], run_id=artifact['run_id'], scored_at=at.isoformat(),
                     artifact_created_at=artifact['created_at'], training_end=artifact['training_end'],
                     source_manifest=manifest, rows=rows, exclusions=issues))
    atomic_json(STORE/artifact['run_id']/'prospective'/str(day)/(at.strftime('%Y%m%dT%H%M%S%fZ')+'.json'), doc)
    research_capture = receiving_research_trial.capture(records, doc)
    eligible = [r for r in records if r['stat'] in artifact['models'] and prospective_error(r, artifact, at) is None]
    expected = len(eligible)
    captured = existing | {(r['forecast_id'], r['variant']) for r in rows}
    missing_pairs = [(int(r['id']), v) for r in eligible for v in
        ('selected_policy', 'production_point_conditional', *artifact['models'][r['stat']].get('prospective_variants', []))
        if (int(r['id']), v) not in captured]
    missing = sorted({p[0] for p in missing_pairs})
    capture_errors = {k: v for k, v in issues.items() if k in {
        'missing_exact_lock_inputs', 'production_replay_mismatch',
        'scoring_code_version_unavailable', 'exact_overlay_artifact_not_captured'}}
    return dict(status='capture_incomplete' if missing or capture_errors else 'prospective_only', rows=len(rows),
        run_id=artifact['run_id'], eligible_original_locks=expected, missing_forecast_ids=missing,
        missing_variants=missing_pairs,
        capture_errors=capture_errors,
        receiving_research_trial=research_capture,
        exclusions=issues, production_changed=False, betting_approved=False)


def original_offer_matches(rec, row, doc):
    p = rec.get('forecast_payload') or {}
    at = safe_time(doc.get('scored_at'))
    stub = dict(created_at=doc.get('artifact_created_at'), training_end=doc.get('training_end'),
                production_release=row.get('production_release'))
    return (doc.get('contract') in SUPPORTED_CONTRACTS and at is not None
            and prospective_error(rec, stub, at) is None
            and row.get('source_context_cutoff') == p.get('prediction_context_cutoff_utc')
            and all(str(row.get('source_'+k)) == str(p.get(k)) for k in ('book', 'side', 'line', 'price'))
            and abs(float(row['production_probability'])-float(p['probability'])) <= 1e-10)


def validate_documents(records, ledger, documents):
    lookup = {int(r['id']): r for r in records}
    ledgers = {str(l['ledger_id']): l for l in ledger}
    issues = Counter(); rows = []; seen = set(); pending = 0
    for doc in sorted(documents, key=lambda d: d.get('scored_at', '')):
        at = safe_time(doc.get('scored_at'))
        for row in doc.get('rows', []):
            rec = lookup.get(int(row['forecast_id']))
            if not rec:
                issues['unknown_forecast'] += 1; continue
            p = rec.get('forecast_payload') or {}
            if not original_offer_matches(rec, row, doc):
                issues['invalid_original_offer_or_timing'] += 1; continue
            key = (doc['run_id'], row['variant'], int(rec['id']))
            if key in seen:
                continue
            seen.add(key)
            if rec.get('graded_result')=='void_nonparticipant':
                issues['void_nonparticipant_not_binary'] += 1; continue
            if rec.get('actual') is None:
                pending += 1; continue
            actual = float(rec['actual']); line = float(p['line'])
            if actual == line:
                issues['push_not_binary'] += 1; continue
            if rec.get('status') != 'final':
                issues['unfinalized_result'] += 1; continue
            micro_ids = [k for k in row.get('ledger_ids', []) if str(k) in ledgers
                         and ledger_matches(ledgers[str(k)], rec, at)]
            result = dict(row, run_id=doc['run_id'], game_id=rec['game_id'], player_id=rec['player_id'],
                scoring_version=(p.get('scoring_replay') or {}).get('scoring_fingerprint', 'legacy_unknown'),
                season=rec['season'], week=rec['week'], created_at=str(rec['created_at_utc']), scored_at=doc['scored_at'],
                actual=actual, production_yards=float(p['projection']),
                production_p10=p.get('projection_p10'), production_p90=p.get('projection_p90'),
                outcome=float(actual > line if p['side'] == 'over' else actual < line), micro_ids=micro_ids)
            # Older shadow documents can use their immutable lock quote as a
            # baseline; this does not create a retrospective challenger forecast.
            result.update(locked_market(rec))
            rows.append(result)
    return rows, dict(issues), pending


def offered_metrics(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    if 'scoring_version' not in frame:
        frame['scoring_version'] = 'legacy_unknown'
    frame['scoring_version'] = frame['scoring_version'].fillna('legacy_unknown')
    results = {}
    for key, all_rows in frame.groupby(['run_id', 'production_release', 'stat', 'variant', 'scoring_version']):
        all_rows = all_rows.sort_values('created_at')
        scopes = dict(real_offers=all_rows.drop_duplicates(['game_id', 'player_id', 'source_book', 'source_line', 'source_side']),
                      exact_micro=all_rows.loc[all_rows.micro_ids.map(bool)].drop_duplicates('forecast_id'))
        result = {}
        for scope, g in scopes.items():
            if g.empty:
                result[scope] = {'rows': 0, 'status': 'waiting_for_exact_prospective_matches'}; continue
            g = g.copy()
            g['weight'] = 1/g.groupby(['game_id', 'player_id']).outcome.transform('size')
            raw_over = g.raw_over_probability.to_numpy()
            raw_side = np.where(g.source_side.eq('over'), raw_over, 1-raw_over)
            g['reference_error'] = (g.production_probability-g.outcome)**2
            g['challenger_error'] = (g.same_side_probability-g.outcome)**2
            grouped = g.groupby(['season', 'week', 'game_id', 'player_id'])[['reference_error', 'challenger_error']].mean().reset_index()
            pg = g.drop_duplicates(['game_id', 'player_id'])
            gain = clustered_gain(grouped)
            def trace_probability(row, stage):
                trace = row.get('probability_trace') or {}
                value = trace.get(stage)
                if value is None:
                    return np.nan
                source_side = trace.get('side', row['candidate_side']) if stage.endswith('_side') else 'over'
                return float(value) if source_side == row['source_side'] else 1-float(value)
            stages = {stage: probability_metrics(g.apply(lambda r: trace_probability(r, stage), axis=1),
                                                  g.outcome, g.weight)
                      for stage in ('raw_over', 'heuristic_over', 'post_exact_side', 'final_side')}
            result[scope] = dict(rows=len(g), unique_player_games=len(pg),
                weeks=len(g[['season', 'week']].drop_duplicates()),
                production=probability_metrics(g.production_probability, g.outcome, g.weight),
                raw=probability_metrics(raw_side, g.outcome, g.weight),
                final=probability_metrics(g.same_side_probability, g.outcome, g.weight),
                matched_market=market_comparison(g),
                scoring_stages=stages,
                final_brier_gain=gain,
                expected=projection_metrics(pg.actual, pg.expected_yards),
                typical=projection_metrics(pg.actual, pg.median_yards),
                production_point=projection_metrics(pg.actual, pg.production_yards),
                live_curve=projection_metrics(pg.actual, pg.distribution_mean, pg.live_p10, pg.live_p90),
                production_curve=projection_metrics(pg.actual, pg.production_yards,
                    pd.to_numeric(pg.get('production_p10'), errors='coerce'),
                    pd.to_numeric(pg.get('production_p90'), errors='coerce')),
                hypothetical_side_flips=int((g.candidate_side != g.source_side).sum()),
                confirmation='insufficient_prospective_evidence' if len(grouped[['season', 'week']].drop_duplicates()) < 3
                    else 'positive_grouped_brier' if (gain.get('lower_95') or -1) > 0 else 'improvement_not_confirmed')
        results['|'.join(key)] = result
    return results


def deployment_review(scopes):
    """A component review never grants betting eligibility or changes production."""
    blockers = []
    for scope in ('real_offers',):
        m = scopes.get(scope) or {}
        if not m.get('rows'):
            blockers.append(scope+':missing_matches'); continue
        if m.get('weeks', 0) < 3:
            blockers.append(scope+':insufficient_independent_weeks')
        gain = m.get('final_brier_gain') or {}
        if gain.get('lower_95') is None or gain['lower_95'] <= 0:
            blockers.append(scope+':brier_improvement_unconfirmed')
        before, after = m['production'], m['final']
        if after.get('calibration_error') is None or after['calibration_error'] > before['calibration_error']:
            blockers.append(scope+':calibration_not_improved')
        coverage = m.get('live_curve', {}).get('coverage_80')
        reference = m.get('production_curve', {}).get('coverage_80')
        if (coverage is None or reference is None or not .75 <= coverage <= .85
                or abs(coverage-.8) > abs(reference-.8) + .005):
            blockers.append(scope+':distribution_coverage_not_confirmed')
    # A forecast component may improve while the corresponding betting strategy
    # still fails against the market. Keep those decisions separate.
    market_blockers = []
    for scope in ('real_offers', 'exact_micro'):
        m = (scopes.get(scope) or {}).get('matched_market') or {}
        if not m.get('rows'):
            market_blockers.append(scope+':missing_true_pair_matches'); continue
        if m.get('weeks', 0) < 3:
            market_blockers.append(scope+':insufficient_independent_market_weeks')
        gain = m.get('challenger_brier_gain_vs_market') or {}
        if gain.get('lower_95') is None or gain['lower_95'] <= 0:
            market_blockers.append(scope+':market_brier_advantage_unconfirmed')
        if m['challenger']['calibration_error'] >= m['market']['calibration_error']:
            market_blockers.append(scope+':market_calibration_advantage_unconfirmed')
    return dict(status='ready_for_component_review' if not blockers else 'collecting_evidence',
                blockers=blockers, deployment_approved=False, betting_approved=False,
                market_evidence_blockers=market_blockers, automatic_promotion=False)


def pinned_capture_summary(records, documents, manifest):
    if not manifest:
        return {'status': 'not_pinned'}
    lookup = {int(r['id']): r for r in records}
    variants = set()
    for d in documents:
        if d.get('run_id') != manifest['run_id'] or d.get('source_manifest') != manifest:
            continue
        for r in d.get('rows', []):
            rec = lookup.get(int(r['forecast_id']))
            stub = dict(created_at=d.get('artifact_created_at'), training_end=d.get('training_end'),
                        production_release=r.get('production_release'))
            if rec and prospective_error(rec, stub, safe_time(d['scored_at'])) is None:
                variants.add((int(r['forecast_id']), r['variant']))
    ids = {i for i, _ in variants}
    settled = {i for i in ids if lookup[i].get('actual') is not None and lookup[i].get('status') == 'final'}
    weeks = {(lookup[i]['season'], lookup[i]['week']) for i in settled}
    return dict(run_id=manifest['run_id'], captured_variant_rows=len(variants),
        original_forecasts=len(ids), finalized_original_forecasts=len(settled),
        pending_original_forecasts=len(ids-settled), independent_settled_weeks=len(weeks),
        status='collecting_evidence', automatic_promotion=False)


def matched_variant_comparisons(rows):
    """Same original forecast IDs, never one variant's preferred subset of picks."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    result = {}
    for (run_id, release, stat), group in frame.groupby(['run_id', 'production_release', 'stat']):
        for reference, challenger in (('selected_policy', 'stable_ensemble'),
                ('selected_policy', 'stable_ensemble_calibrated'),
                ('stable_ensemble', 'qb_tail'), ('stable_ensemble_calibrated', 'qb_tail_calibrated')):
            ref = group.loc[group.variant.eq(reference)].drop_duplicates('forecast_id')
            new = group.loc[group.variant.eq(challenger)].drop_duplicates('forecast_id')
            if ref.empty or new.empty:
                continue
            paired = new.merge(ref[['forecast_id', 'same_side_probability', 'micro_ids']], on='forecast_id',
                               suffixes=('', '_reference'), validate='one_to_one')
            paired = paired.sort_values('created_at')
            scopes = {'real_offers': paired.drop_duplicates(['game_id', 'player_id', 'source_book', 'source_line', 'source_side']),
                      'exact_micro': paired.loc[[bool(set(a) & set(b)) for a, b in
                                                 zip(paired.micro_ids, paired.micro_ids_reference)]]}
            for scope, g in scopes.items():
                if g.empty:
                    continue
                w = 1/g.groupby(['game_id', 'player_id']).outcome.transform('size')
                g = g.assign(reference_error=(g.same_side_probability_reference-g.outcome)**2,
                             challenger_error=(g.same_side_probability-g.outcome)**2)
                clustered = g.groupby(['season', 'week', 'game_id', 'player_id'])[['reference_error', 'challenger_error']].mean().reset_index()
                key = '|'.join((run_id, release, stat, challenger+'_vs_'+reference, scope))
                result[key] = dict(rows=len(g), unique_player_games=len(clustered),
                    reference=probability_metrics(g.same_side_probability_reference, g.outcome, w),
                    challenger=probability_metrics(g.same_side_probability, g.outcome, w),
                    brier_gain=clustered_gain(clustered), automatic_promotion=False)
    return result


def report():
    documents = [json.loads(p.read_text()) for p in STORE.glob('*/prospective/*/*.json')]
    records = load_rows(); ledger = load_ledger()
    rows, exclusions, pending = validate_documents(records, ledger, documents)
    results = offered_metrics(rows)
    pin = json.loads((STORE/'active_trial.json').read_text()) if (STORE/'active_trial.json').exists() else None
    research_trial = receiving_research_trial.report(records, rows, offered_metrics)
    atomic_json(ROOT/'reports'/'nfl_receiving_research_trial_latest.json', research_trial)
    doc = clean(dict(built_at=datetime.now(timezone.utc).isoformat(), cohorts=results,
        pending=pending, exclusions=exclusions, original_micro_ledger_rows=len(ledger),
        matched_variants=matched_variant_comparisons(rows),
        receiving_research_trial=research_trial,
        deployment_reviews={name: deployment_review(scopes) for name, scopes in results.items()},
        active_trial=pin, pinned_release_evidence=pinned_capture_summary(records, documents, pin),
        status='evaluated' if rows else 'waiting_for_results' if pending else 'waiting_for_new_benchmark_pregame_forecasts',
        automatic_promotion=False, limitations=[
            'New artifacts cannot manufacture predictions for older micro locks.',
            'All comparisons retain the original selected side, book, line and price.',
            'Multiple books/lines share one outcome and receive inverse player-game weights.',
            'Micro ledger rows may be simulated recommendations, not confirmed placed wagers.']))
    atomic_json(ROOT/'reports'/'nfl_benchmark_offers_latest.json', doc)
    text = ['# NFL Benchmark Offered-Line Validation', '', 'Status: '+doc['status'],
            f'Original micro ledger rows: {len(ledger)}; pending shadow rows: {pending}',
            'Pinned release evidence: '+str(doc['pinned_release_evidence']),
            'Production unchanged. Exact original micro IDs only; no after-the-fact selected picks.', '',
            '| Cohort / scope | Rows | Weeks | Production Brier | Challenger final Brier | Status |',
            '|---|---:|---:|---:|---:|---|']
    for name, scopes in results.items():
        name = name.replace('|', ' / ')
        for scope, metrics in scopes.items():
            if not metrics['rows']:
                text.append(f'| {name} / {scope} | 0 | - | - | - | waiting |')
            else:
                text.append(f"| {name} / {scope} | {metrics['rows']} | {metrics['weeks']} | "
                            f"{metrics['production']['brier']:.4f} | {metrics['final']['brier']:.4f} | {metrics['confirmation']} |")
    text += ['', '## Matched Challenger Comparisons', '',
             '| Cohort / comparison / scope | Matched rows | Reference Brier | Challenger Brier |',
             '|---|---:|---:|---:|']
    for name, m in doc['matched_variants'].items():
        name = name.replace('|', ' / ')
        text.append(f"| {name} | {m['rows']} | {m['reference']['brier']:.4f} | {m['challenger']['brier']:.4f} |")
    text += ['', '## Identical-Offer Market Comparison', '',
             'Only captured true same-book pairs. Missing market evidence remains unknown.', '',
             '| Cohort / scope | Paired / eligible | Production Brier | Challenger Brier | Market Brier |',
             '|---|---:|---:|---:|---:|']
    for name, scopes in results.items():
        name = name.replace('|', ' / ')
        for scope, metrics in scopes.items():
            m = metrics.get('matched_market') or {}
            if m.get('rows'):
                text.append(f"| {name} / {scope} | {m['rows']} / {m['eligible_rows']} | "
                    f"{m['production']['brier']:.4f} | {m['challenger']['brier']:.4f} | {m['market']['brier']:.4f} |")
            else:
                text.append(f"| {name} / {scope} | 0 / {metrics['rows']} | - | - | unknown |")
    trial_text = ['# FanDuel Receiving Research Trial', '', 'Status: '+research_trial['status'],
        'Research selections only. No automatic deployment or betting approval.',
        'Three independent weeks is a review floor, not an approval rule.', '']
    if research_trial.get('registration'):
        c = research_trial['registration']['config']
        trial_text += [f"Registered: {c['registered_at']}", f"Pinned benchmark: {c['run_id']}",
            f"Variant: {c['variant']}; book: {c['book']}; common receiving yards only.",
            f"Selected: {research_trial['selected']}; accounting: {research_trial['accounting']}",
            'No pre-registration forecasts are counted as trial selections.', '', '## Review Blockers',
            *['- '+b for b in research_trial['blockers']]]
    (ROOT/'reports'/'nfl_receiving_research_trial_latest.md').write_text('\n'.join(trial_text)+'\n', encoding='utf-8')
    text += ['', *doc['limitations']]
    text += ['', '## Component Review (Not Betting Approval)', '']
    for name, review in doc['deployment_reviews'].items():
        text.append(f"- {name}: {review['status']}; {', '.join(review['blockers'])}")
    (ROOT/'reports'/'nfl_benchmark_offers_latest.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--date', type=date.fromisoformat)
    p.add_argument('--report', action='store_true')
    p.add_argument('--pin-current', action='store_true')
    p.add_argument('--register-receiving-trial', action='store_true')
    args = p.parse_args()
    if not args.report and not args.date and not args.pin_current and not args.register_receiving_trial:
        p.error('--date, --report, --pin-current or --register-receiving-trial is required')
    result = (receiving_research_trial.register(*load_artifact()) if args.register_receiving_trial else
              pin_current() if args.pin_current else report() if args.report else score(args.date))
    print(json.dumps({k: v for k, v in result.items() if k != 'cohorts'}, indent=2))
    if result.get('status') == 'capture_incomplete':
        raise SystemExit(1)
