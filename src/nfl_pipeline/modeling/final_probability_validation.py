"""Evaluate exact locked micro selections and final-stage calibration, offline only."""
import json
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, atomic_json, atomic_joblib
from nfl_pipeline.modeling.evaluation import probability_metrics, clustered_gain
from nfl_pipeline.modeling.live_scoring_replay import load_rows, validate_lock, prospective_report, ROOT
from nfl_pipeline.modeling.scoring_capture import replay
from nfl_pipeline.modeling.selected_pick_diagnostics import (
    locked_workload, selected_report, workload_report, monotonicity_violations, write_diagnostics,
)


def load_ledger():
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='60s'")
        cur.execute("""SELECT ledger_id,prediction_id,locked_at_utc,game_date_et,tier,stake,result,profit,
                       execution_status,book AS ledger_book,side AS ledger_side,line AS ledger_line,
                       price AS ledger_price,model_version AS ledger_model_version FROM bets.nfl_bet_ledger
                       WHERE source_kind='prop' AND tier='micro_projection' ORDER BY locked_at_utc,ledger_id""")
        return [dict(r) for r in cur.fetchall()]


def attach_label_availability(records):
    """Preserve first observed exact labels; a game date alone is not result provenance."""
    evidence_root = MODEL_ROOT / 'final_probability_validation' / 'label_evidence'
    known = {}
    def remember(prediction_id, actual, when):
        at = safe_time(when)
        if at is None or actual is None:
            return
        key = (int(prediction_id), float(actual))
        if key not in known or at < known[key]:
            known[key] = at
    for path in sorted(evidence_root.glob('*.json')):
        doc = json.loads(path.read_text(encoding='utf-8'))
        if doc.get('contract') != 'nfl-observed-final-labels-v1':
            continue
        for row in doc.get('rows', []):
            remember(row['prediction_id'], row['actual'], doc.get('observed_at'))
    with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout='60s'")
        cur.execute("""SELECT prediction_id,actual_stat,updated_at_utc FROM bets.nfl_player_prop_prediction_results
                       WHERE result IN ('win','loss','push') AND actual_stat IS NOT NULL""")
        for prediction_id, actual, updated in cur.fetchall():
            # First graded time could belong to a different/pending result. The
            # current value is proven by its latest update or an immutable prior observation.
            remember(prediction_id, actual, updated)
    now = datetime.now(timezone.utc)
    observed = []
    for row in records:
        if row.get('actual') is not None and row.get('status') == 'final':
            remember(row['id'], row['actual'], now)
            observed.append(dict(prediction_id=int(row['id']), actual=float(row['actual'])))
            row['label_available_at'] = known[(int(row['id']), float(row['actual']))]
    atomic_json(evidence_root / (now.strftime('%Y%m%dT%H%M%S%fZ')+'.json'),
                dict(contract='nfl-observed-final-labels-v1', observed_at=now.isoformat(), rows=observed))
    return records


def profit_multiple(price):
    p = float(price)
    if not np.isfinite(p) or p == 0:
        raise ValueError('Invalid American price')
    return p / 100 if p > 0 else 100 / -p


def decision_frame(records):
    rows = []; excluded = Counter()
    for r in records:
        error = validate_lock(r)
        if error:
            excluded[error] += 1; continue
        p = r.get('forecast_payload') or {}
        if r.get('line') is None or r.get('actual') is None or r.get('side') not in ('over', 'under'):
            continue
        if float(r['actual']) == float(r['line']):
            excluded['push'] += 1; continue
        status = 'missing_lock_time_scoring_inputs'
        if p.get('scoring_replay'):
            try:
                result = replay(p['scoring_replay'])
                status = 'matched' if result['side'] == p['side'] and abs(result['probability'] - p['probability']) < 1e-10 else 'replay_mismatch'
            except ValueError as exc:
                status = str(exc)
        try:
            multiple = profit_multiple(p.get('price'))
        except (TypeError, ValueError):
            excluded['invalid_price'] += 1; continue
        line = float(r['line']); probability = float(p['probability'])
        side = r['side']; over = float(r['actual']) > line
        trace = p.get('probability_trace') or {}
        raw = trace.get('raw_over', p.get('raw_over_probability'))
        heuristic = trace.get('heuristic_over')
        flip = lambda x: np.nan if x is None else float(x) if side == 'over' else 1 - float(x)
        rows.append(dict(prediction_id=int(r['id']), game_id=r['game_id'], player_id=r['player_id'],
            player_game=str(r['game_id']) + '|' + str(r['player_id']), player=p.get('player_name'),
            stat=r['stat'], side=side, line=line, book=r['book'], price=p['price'], payout=multiple,
            model_version=p['model_version'], season=r['season'], week=r['week'],
            scoring_version=(p.get('scoring_replay') or {}).get('scoring_fingerprint', 'legacy_unknown'),
            kickoff=r['start_ts_utc'],
            label_available_at=r.get('label_available_at'),
            day=p.get('game_date_et') or str(r['start_ts_utc'].date()), locked_at=r['created_at_utc'],
            position=p.get('position') or 'unknown',
            workload_bucket=locked_workload(p, r['stat'])['workload_bucket'],
            line_bucket=str(int(line // (50 if r['stat'] == 'passing_yards' else 20))),
            gap_bucket=str(int(abs(float(p.get('projection', line)) - line) // 15)),
            probability=probability, outcome=float(over if side == 'over' else not over),
            raw=flip(raw), context_blend=flip(trace.get('context_blend_over')),
            heuristic=flip(heuristic), post_exact=trace.get('post_exact_side'),
            market=p.get('market_no_vig_probability'), replay_status=status,
            push_probability=float(p.get('push_probability') or 0),
            ev=(1-float(p.get('push_probability') or 0)) * (probability * (1 + multiple) - 1),
            eligible=(p.get('tier') == 'micro_projection' or 'micro_daily_cap_research_only' in str(p.get('reasons', '')))
                     and p.get('drift_guard_pass') is True,
            confidence=p.get('projection_confidence') or 0,
            edge=p.get('model_market_edge') or 0))
    return pd.DataFrame(rows).drop_duplicates('prediction_id') if rows else pd.DataFrame(), dict(excluded)


def unique_decisions(frame):
    cohort = ['scoring_version'] if 'scoring_version' in frame else []
    out = frame.sort_values(['locked_at', 'prediction_id']).drop_duplicates(
        ['model_version', *cohort, 'game_id', 'player_id', 'stat', 'side', 'line', 'book']).copy()
    out['weight'] = 1 / out.groupby(['model_version', *cohort, 'player_game', 'stat']).probability.transform('size')
    return out


def select_top_five(frame, probability='probability'):
    """Fixed, previously approved pool only. Outcome never participates in selection."""
    g = frame.loc[frame.eligible].copy()
    g['rank_ev'] = (1-g.get('push_probability', 0)) * (g[probability] * (1 + g.payout) - 1)
    g = g.loc[g.rank_ev.gt(0)].sort_values(['rank_ev', 'edge', 'confidence', 'prediction_id'],
                                         ascending=[False, False, False, True], kind='stable')
    g = g.drop_duplicates(['model_version', 'day', 'game_id', 'player_id'])
    return g.groupby(['model_version', 'day'], sort=False).head(5)


class FinalStageCalibrator:
    """Regularized chosen-side calibration, not a mean or stat-distribution model."""
    GROUPS = ('stat', 'side', 'position', 'workload_bucket', 'line_bucket', 'book', 'gap_bucket')

    def features(self, frame, fit=False):
        X = pd.get_dummies(frame[list(self.GROUPS)].fillna('unknown').astype(str), dtype=float)
        # Logit is the principal signal; sparse group offsets shrink strongly toward it.
        X['logit'] = logit(np.clip(frame.probability.to_numpy(), .001, .999))
        if fit:
            self.columns = X.columns.tolist()
        return X.reindex(columns=self.columns, fill_value=0.)

    def fit(self, frame):
        if frame.player_game.nunique() < 100 or len(frame[['season', 'week']].drop_duplicates()) < 2:
            raise ValueError('insufficient_independent_training_history')
        if frame.outcome.nunique() < 2:
            raise ValueError('single_class_training_history')
        self.head = LogisticRegression(C=.15, max_iter=500).fit(
            self.features(frame, True), frame.outcome, sample_weight=frame.weight)
        return self

    def predict(self, frame):
        return self.head.predict_proba(self.features(frame))[:, 1]


def scores(frame):
    if frame.empty:
        return {'rows': 0}
    p = probability_metrics(frame.probability, frame.outcome, frame.get('weight'))
    return dict(p, wins=int(frame.outcome.sum()), losses=int((1-frame.outcome).sum()),
                mean_probability=float(frame.probability.mean()), unique_player_games=frame.player_game.nunique(),
                weeks=len(frame[['season', 'week']].drop_duplicates()))


def calibration_experiment(frame):
    results = {}; artifacts = {}
    cohort = ['model_version', 'scoring_version'] if 'scoring_version' in frame else 'model_version'
    for release, g in frame.groupby(cohort):
        if isinstance(release, tuple):
            release = '|'.join(release)
        g = unique_decisions(g)
        g['label_available_at'] = pd.to_datetime(g.get('label_available_at'), utc=True, errors='coerce')
        weeks = sorted(set(zip(g.season, g.week)))
        parts = []; folds = []
        for week in weeks[2:]:
            test = g.loc[(g.season == week[0]) & (g.week == week[1])].copy()
            train = g.loc[g.day.astype(str).lt(test.day.astype(str).min()) & ~g.game_id.isin(test.game_id)
                          & g.label_available_at.le(pd.to_datetime(test.locked_at, utc=True).min())].copy()
            try:
                model = FinalStageCalibrator().fit(train)
            except ValueError:
                continue
            test['candidate_probability'] = model.predict(test)
            parts.append(test)
            folds.append(dict(week=list(week), training_rows=len(train), test_rows=len(test),
                              training_last_day=str(train.day.max()), test_first_day=str(test.day.min())))
        if not parts:
            results[release] = dict(status='insufficient_later_week_evidence', weeks=len(weeks),
                unique_player_games=g.player_game.nunique(), enabled=False,
                reason='Need at least two training weeks before a later-week test; no in-sample calibrator is enabled.')
            continue
        oof = pd.concat(parts, ignore_index=True)
        all_before = probability_metrics(oof.probability, oof.outcome, oof.weight)
        all_after = probability_metrics(oof.candidate_probability, oof.outcome, oof.weight)
        selected = select_top_five(oof)
        reselected = select_top_five(oof, 'candidate_probability')
        fixed_before = probability_metrics(selected.probability, selected.outcome)
        fixed_after = probability_metrics(selected.candidate_probability, selected.outcome)
        reranked = probability_metrics(reselected.candidate_probability, reselected.outcome)
        paired = oof.assign(reference_error=(oof.probability-oof.outcome)**2,
                            challenger_error=(oof.candidate_probability-oof.outcome)**2)
        paired = paired.groupby(['season', 'week', 'player_game'])[['reference_error', 'challenger_error']].mean().reset_index()
        gain = clustered_gain(paired)
        enabled = bool(len(folds) >= 3 and gain['lower_95'] is not None and gain['lower_95'] > 0
            and len(selected) >= 15 and fixed_after['brier'] < fixed_before['brier']
            and all_after['calibration_error'] <= all_before['calibration_error']
            and reranked.get('brier', 1) <= fixed_before['brier'])
        results[release] = dict(status='evaluated', enabled=enabled, all_before=all_before, all_after=all_after,
            fixed_top_five_before=fixed_before, fixed_top_five_after=fixed_after, reranked_top_five=reranked,
            clustered_brier_gain=gain, folds=folds, production_enabled=False)
        if enabled:
            artifacts[release] = FinalStageCalibrator().fit(g)
    return results, artifacts


def build_report(records, ledger):
    frame, excluded = decision_frame(records)
    if frame.empty:
        return {'status': 'no_settled_locks', 'exclusions': excluded}, {}
    unique = unique_decisions(frame)
    ledger_frame = pd.DataFrame(ledger)
    micro = (ledger_frame.merge(frame, on='prediction_id', how='inner', validate='many_to_one')
             if not ledger_frame.empty else frame.iloc[:0].copy())
    if not micro.empty:
        micro = micro.drop_duplicates('prediction_id').copy()
        valid = []
        for row in micro.to_dict('records'):
            when = safe_time(row.get('locked_at_utc'))
            ok = bool(when and safe_time(row['locked_at']) <= when < safe_time(row['kickoff']))
            ok = ok and all(str(row.get('ledger_' + k)) == str(row[k]) for k in ('book', 'side', 'model_version'))
            try:
                ok = ok and float(row['ledger_line']) == float(row['line']) and float(row['ledger_price']) == float(row['price'])
            except (ValueError, TypeError, KeyError):
                ok = False
            valid.append(ok)
        excluded['ledger_identity_or_lock_mismatch'] = len(micro)-sum(valid)
        micro = micro.loc[valid].copy()
        micro['weight'] = 1.
    matched = unique.loc[unique.replay_status.eq('matched')].copy()
    stages = {}
    for stat, group in matched.groupby('stat'):
        paired = group.dropna(subset=['raw', 'heuristic', 'post_exact', 'probability'])
        stages[stat] = {col: probability_metrics(paired[col], paired.outcome, paired.weight)
                       for col in ('raw', 'heuristic', 'post_exact', 'probability')}
        if 'context_blend' in paired:
            detailed = paired.dropna(subset=['context_blend'])
            stages[stat]['detailed_matched_subset'] = {
                col: probability_metrics(detailed[col], detailed.outcome, detailed.weight)
                for col in ('raw', 'context_blend', 'heuristic', 'post_exact', 'probability')}
    calibration, artifacts = calibration_experiment(matched)
    report = dict(built_at=datetime.now(timezone.utc).isoformat(), status='evaluated',
        settled_unique_offers=len(unique), exact_replay_status=unique.replay_status.value_counts().to_dict(),
        matched_stage_comparisons=stages, all_locked=scores(unique), actual_locked_micro=scores(micro),
        scoring_cohorts={'|'.join(key): scores(g) for key, g in unique.groupby(['model_version', 'scoring_version'])}
            if 'scoring_version' in unique else {},
        ledger_coverage=dict(total_micro_prediction_ids=len({int(r['prediction_id']) for r in ledger}),
            scored_micro_prediction_ids=len(micro),
            unscored_prediction_ids=sorted({int(r['prediction_id']) for r in ledger} - set(micro.prediction_id))),
        micro_by_date={str(k): scores(g) for k, g in micro.groupby('day')},
        micro_rows=micro[[c for c in ('prediction_id', 'player', 'stat', 'side', 'line', 'price', 'probability',
            'outcome', 'replay_status', 'execution_status') if c in micro]].to_dict('records'),
        top_five_fixed_approved_pool=scores(select_top_five(unique)),
        calibration_experiment=calibration, exclusions=excluded,
        selected_pick_diagnostic=selected_report(unique, micro),
        limitations=[
            'Actual micro uses exact ledger prediction IDs, never later revised forecasts.',
            'Missing historical replay inputs remain missing; stored final probabilities can still be scored.',
            'Counterfactual top five uses earliest offers from a fixed approved pool, not an exact historical batch/ledger replay.',
            'Week-grouped validation keeps repeated books/lines from inflating training evidence.',
            'A training label must be verifiably available before the earliest test lock; earlier game dates alone are insufficient.',
            'A simulated ledger entry is not evidence that money was wagered.',
            'Challenger probabilities are not automatically installed in live scoring.'], production_changed=False)
    return clean(report), artifacts


def run():
    records = attach_label_availability(load_rows())
    ledger = load_ledger()
    report, artifacts = build_report(records, ledger)
    workload = workload_report(records)
    consistency = monotonicity_violations(records)
    write_diagnostics(report.get('selected_pick_diagnostic', {'populations': {}}), workload, consistency)
    report['diagnostic_reports'] = ['nfl_selected_pick_diagnostic_latest', 'nfl_workload_uncertainty_diagnostic_latest',
                                    'nfl_probability_consistency_latest']
    from nfl_pipeline.modeling.micro_reconciliation import load_entries, reconcile, write_report
    reconciliation = reconcile(load_entries(), [r['prediction_id'] for r in report.get('micro_rows', [])])
    report['micro_reconciliation'] = reconciliation
    if reconciliation['evaluable_but_unscored']:
        report['status'] = 'incomplete_micro_evaluation'
    write_report(reconciliation)
    prospective = prospective_report(records)
    report['uncertainty_only_real_offer_cohorts'] = {key: value for key, value in prospective.get('cohorts', {}).items()
                                                   if 'receiving_conditional' in key}
    report['uncertainty_micro_id_coverage'] = uncertainty_micro_coverage(records, ledger)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = MODEL_ROOT / 'final_probability_validation' / stamp
    atomic_json(path / 'report.json', report)
    if artifacts:
        atomic_joblib(path / 'calibrators.joblib', artifacts)
    atomic_json(ROOT / 'reports' / 'nfl_final_probability_validation_latest.json', report)
    lines = ['# NFL Final Probability Validation', '', 'Production unchanged. No new real-money approval.', '',
             '## Actual Locked Micro', '', json.dumps(report.get('actual_locked_micro', {})), '',
             'Ledger coverage: ' + json.dumps(report.get('ledger_coverage', {})), '',
             '## Every Micro Entry Accounted For', '', json.dumps(reconciliation['categories']), '',
             'Evaluable but unscored: '+json.dumps(reconciliation['evaluable_but_unscored']), '',
             'Legacy/void rows remain in nfl_micro_reconciliation_latest.md, not in verified probability metrics.', '',
             '## Exact Replay Coverage', '', json.dumps(report.get('exact_replay_status', {})), '',
             '## Calibration Experiment', '', json.dumps(report.get('calibration_experiment', {}), indent=2), '',
             '## Uncertainty-Only Real Offers', '', '| Cohort | Rows | Production Brier | Challenger Brier | Coverage 80% | Weeks |',
             '|---|---:|---:|---:|---:|---:|']
    for key, r in report['uncertainty_only_real_offer_cohorts'].items():
        a = r.get('challenger_probability', {}); b = r.get('production_probability', {})
        if a.get('rows'):
            lines.append(f"| {key} | {a['rows']} | {b['brier']:.4f} | {a['brier']:.4f} | {r['projection'].get('coverage_80', 0):.1%} | {r['weeks']} |")
    lines += ['', 'Exact micro-ID uncertainty coverage: ' + json.dumps(report['uncertainty_micro_id_coverage']), '',
              *['- ' + x for x in report.get('limitations', [])]]
    (ROOT / 'reports' / 'nfl_final_probability_validation_latest.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


def uncertainty_micro_coverage(records, ledger):
    lookup = {int(r['id']): r for r in records}
    ids = {int(r['prediction_id']) for r in ledger}
    found = set()
    for path in (MODEL_ROOT / 'challengers').glob('*receiving_conditional*/prospective/*/*.json'):
        doc = json.loads(path.read_text(encoding='utf-8'))
        when = safe_time(doc.get('scored_at'))
        for row in doc.get('rows', []):
            i = int(row['forecast_id']); original = lookup.get(i)
            if i not in ids or not original or not when:
                continue
            p = original.get('forecast_payload') or {}
            if (not validate_lock(original) and when < safe_time(original['start_ts_utc'])
                and row.get('production_release') == p.get('model_version')
                and row.get('scoring_stage') == 'complete_live_path'
                and safe_time(row.get('source_context_cutoff')) == safe_time(p.get('prediction_context_cutoff_utc'))
                and doc.get('training_end') and str(doc['training_end']) < str(original['start_ts_utc'].date())):
                found.add(i)
    return dict(micro_prediction_ids=len(ids), exact_pregame_uncertainty_ids=len(found),
                missing_ids=sorted(ids-found), later_revisions_used=False)


if __name__ == '__main__':
    r = run()
    print(json.dumps({k: r.get(k) for k in ('actual_locked_micro', 'calibration_experiment', 'uncertainty_micro_id_coverage')}, indent=2))
    if r.get('status') == 'incomplete_micro_evaluation':
        raise SystemExit(1)
