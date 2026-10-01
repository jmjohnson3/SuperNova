"""Audit stored lock-time probabilities, with exact replay when inputs exist."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import MODEL_ROOT, atomic_json
from nfl_pipeline.context_contract import safe_time, validate_evidence
from nfl_pipeline.modeling.evaluation import probability_metrics, projection_metrics, clustered_gain
from nfl_pipeline.modeling.scoring_capture import replay

ROOT=Path(__file__).resolve().parents[3]


def load_rows(day=None, stat=None):
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='90s'")
        cur.execute("""
            SELECT p.id,p.game_id,p.player_id,p.stat,p.side,p.line,p.book,p.created_at_utc,p.is_current,
                   g.season,g.week,g.status,g.start_ts_utc,p.forecast_payload,
                   o.fetched_at_utc AS offer_fetched_at,
                   l.game_id AS result_game_id, l.offense_snaps AS actual_offense_snaps,
                   graded.result AS graded_result,
                   CASE WHEN g.status='final' THEN l.targets END AS actual_targets,
                   CASE WHEN g.status='final' THEN l.carries END AS actual_carries,
                   CASE WHEN g.status='final' THEN l.pass_attempts END AS actual_pass_attempts,
                   CASE WHEN g.status='final' AND (COALESCE(l.offense_snaps,0)>0 OR
                     COALESCE(l.pass_attempts,0)+COALESCE(l.carries,0)+COALESCE(l.targets,0)>0)
                   THEN CASE p.stat WHEN 'passing_yards' THEN l.passing_yards WHEN 'rushing_yards' THEN l.rushing_yards
                     WHEN 'receiving_yards' THEN l.receiving_yards WHEN 'passing_tds' THEN l.passing_tds
                     WHEN 'rushing_tds' THEN l.rushing_tds WHEN 'receiving_tds' THEN l.receiving_tds
                     WHEN 'receptions' THEN l.receptions END END AS actual
            FROM bets.nfl_player_prop_predictions p JOIN raw.nfl_games g USING(game_id)
            LEFT JOIN raw.nfl_player_gamelogs l ON l.game_id=p.game_id AND l.player_id=p.player_id AND l.team_abbr=p.team_abbr
            LEFT JOIN odds.nfl_player_prop_lines o ON o.id=p.offer_id
            LEFT JOIN bets.nfl_player_prop_prediction_results graded ON graded.prediction_id=p.id
            WHERE p.integrity_version='nfl-asof-v2' AND (%s IS NULL OR p.game_date_et=%s)
              AND (%s IS NULL OR p.stat=%s)
            ORDER BY p.created_at_utc,p.id
        """,(day,day,stat,stat))
        return [dict(r) for r in cur.fetchall()]


def validate_lock(record):
    payload=record.get('forecast_payload') or {}
    lock=safe_time(record.get('created_at_utc')); start=safe_time(record.get('start_ts_utc'))
    cutoff=safe_time(payload.get('prediction_context_cutoff_utc'))
    if not lock or not start or lock>=start: return 'lock_not_before_start'
    if not cutoff or cutoff>lock: return 'invalid_context_cutoff'
    if payload.get('line') is not None:
        offer=safe_time(record.get('offer_fetched_at'))
        if not offer or offer>cutoff: return 'invalid_lock_offer_timing'
    context=payload.get('context_evidence') or {}
    if context and validate_evidence(context,cutoff) is None: return 'invalid_or_future_context'
    return None


def build_report(records):
    exclusions=Counter(); replay_status=Counter(); decisions=[]; projections=[]; seen=set(); seen_projection=set()
    for rec in records:
        error=validate_lock(rec)
        if error:
            exclusions[error]+=1; continue
        p=rec.get('forecast_payload') or {}
        proj_key=(rec['game_id'],rec['player_id'],rec['stat'],p.get('model_version'))
        if rec.get('actual') is not None and proj_key not in seen_projection:
            seen_projection.add(proj_key)
            projections.append({'stat':rec['stat'],'model_version':p.get('model_version'),'actual':float(rec['actual']),
                'projection':float(p['projection']),'p10':p.get('projection_p10'),'p90':p.get('projection_p90')})
        if p.get('line') is None: continue
        key=(*proj_key,rec['side'],str(rec['line']),rec['book'])
        if key in seen:
            exclusions['later_revision_same_decision']+=1; continue
        seen.add(key)
        captured=p.get('scoring_replay')
        status='missing_lock_time_scoring_inputs'
        if captured:
            try:
                output=replay(captured)
                status='matched' if output['side']==p['side'] and abs(output['probability']-p['probability'])<1e-10 else 'replay_mismatch'
            except ValueError as exc:
                status=str(exc)
        replay_status[status]+=1
        actual=rec.get('actual')
        if actual is None: continue
        if float(actual)==float(rec['line']):
            exclusions['push_not_binary_target']+=1; continue
        outcome=float(float(actual)>float(rec['line']))
        trace=p.get('probability_trace') or {}
        side=p['side']
        raw=trace.get('raw_over',p.get('raw_over_probability'))
        heuristic=trace.get('heuristic_over')
        decisions.append({'game_id':rec['game_id'],'player_id':rec['player_id'],'stat':rec['stat'],
            'season':rec['season'],'week':rec['week'],
            'model_version':p.get('model_version'),'outcome':outcome if side=='over' else 1-outcome,
            'raw':raw if side=='over' or raw is None else 1-raw,
            'heuristic':heuristic if side=='over' or heuristic is None else 1-heuristic,
            'post_exact':trace.get('post_exact_side'),'final':p['probability'],
            'market':p.get('market_no_vig_probability'),'replay_status':status})
    frame=pd.DataFrame(decisions); projected=pd.DataFrame(projections)
    metrics={}; projection_scores={}
    if not frame.empty:
        frame['weight']=1/frame.groupby(['model_version','game_id','player_id','stat']).outcome.transform('size')
        for (stat,version),group in frame.groupby(['stat','model_version']):
            metrics[f'{version}|{stat}']={stage:probability_metrics(pd.to_numeric(group[stage],errors='coerce'),group.outcome,group.weight)
                for stage in ('raw','heuristic','post_exact','final','market')}
            paired=group.dropna(subset=['raw','final']).copy()
            if len(paired):
                paired['reference_error']=(paired.raw-paired.outcome)**2
                paired['challenger_error']=(paired.final-paired.outcome)**2
                grouped=paired.groupby(['season','week','game_id','player_id'])[['reference_error','challenger_error']].mean().reset_index()
                metrics[f'{version}|{stat}']['final_vs_raw_clustered_brier_gain']=clustered_gain(grouped)
    if not projected.empty:
        for (stat,version),group in projected.groupby(['stat','model_version']):
            projection_scores[f'{version}|{stat}']=projection_metrics(group.actual,group.projection,
                pd.to_numeric(group.p10,errors='coerce'),pd.to_numeric(group.p90,errors='coerce'))
    return {'built_at':datetime.now(timezone.utc).isoformat(),'status':'evaluated' if decisions else 'waiting_for_settled_versioned_forecasts',
            'records_scanned':len(records),'unique_offer_decisions':len(seen),'settled_offer_decisions':len(decisions),
            'replay_status':dict(replay_status),'exclusions':dict(exclusions),'probability_stages':metrics,'projection_scores':projection_scores,
            'limitations':['Missing old scoring inputs are never replaced with current context or calibration.',
                'Stored-probability evaluation is separate from exact reproducibility; inspect replay_status.',
                'Multiple books/lines share one player-game outcome and receive inverse multiplicity weights.',
                'Intervals without locked bounds cannot establish interval coverage.',
                'Locked p10/p90 describe the stat forecast, not a distribution inferred from independently adjusted book probabilities.']}


def prospective_report(records,root=None):
    """Score immutable shadow files against the exact production forecast they used."""
    root=root or MODEL_ROOT/'challengers'
    lookup={int(r['id']):r for r in records}
    seen=set(); rows=[]; exclusions=Counter(); pending=0; pending_unique=set(); pending_by_cohort=Counter()
    for path in sorted(root.glob('*/prospective/*/*.json')):
        data=json.loads(path.read_text(encoding='utf-8'))
        scored_at=safe_time(data.get('scored_at'))
        for shadow in data.get('rows',[]):
            rec=lookup.get(int(shadow['forecast_id']))
            if not rec: continue
            key=(shadow['challenger_run'],rec['game_id'],rec['player_id'],rec['stat'])
            payload=rec.get('forecast_payload') or {}
            start=safe_time(rec['start_ts_utc'])
            training_end=data.get('training_end')
            if (validate_lock(rec) or not scored_at or not start or scored_at>=start
                or not training_end or date.fromisoformat(training_end)>=start.date()
                or shadow['production_release']!=payload.get('model_version')
                or safe_time(shadow.get('source_context_cutoff'))!=safe_time(payload.get('prediction_context_cutoff_utc'))):
                exclusions['unverified_prospective_timing_or_cohort']+=1; continue
            if key in seen: continue
            seen.add(key)
            if rec['actual'] is None:
                pending+=1
                pending_unique.add((rec['game_id'],rec['player_id'],rec['stat']))
                pending_by_cohort['|'.join((shadow['challenger_run'],shadow['production_release'],rec['stat']))]+=1
                continue
            actual=float(rec['actual'])
            item={'challenger_run':shadow['challenger_run'],'production_release':shadow['production_release'],
                'stat':rec['stat'],'season':rec['season'],'week':rec['week'],'game_id':rec['game_id'],
                'actual':actual,'production':float(payload['projection']),'challenger':shadow['projection_mean'],
                'median':shadow.get('projection_median',np.nan),
                'distribution_mean':shadow.get('distribution_mean',np.nan),
                'scoring_stage':shadow.get('scoring_stage','raw_challenger'),
                'p10':shadow['p10'],'p90':shadow['p90'],
                'reference_error':abs(float(payload['projection'])-actual),
                'challenger_error':abs(shadow['projection_mean']-actual)}
            if shadow.get('locked_line') is not None and actual!=float(shadow['locked_line']):
                item.update(outcome=float(actual>float(shadow['locked_line'])),
                    challenger_probability=shadow.get('calibrated_over_probability',shadow.get('raw_over_probability')),
                    production_probability=payload['probability'] if payload['side']=='over' else 1-payload['probability'])
            rows.append(item)
    cohorts={}; frame=pd.DataFrame(rows)
    if not frame.empty:
        for key,g in frame.groupby(['challenger_run','production_release','stat']):
            result={'projection':projection_metrics(g.actual,g.challenger,g.p10,g.p90),
                'production':projection_metrics(g.actual,g.production),'clustered_mae_gain':clustered_gain(g),
                'median':projection_metrics(g.actual,g['median']),
                'distribution_mean':projection_metrics(g.actual,g.distribution_mean),
                'clustered_squared_error_gain':clustered_gain(g.assign(
                    reference_error=(g.production-g.actual)**2,challenger_error=(g.challenger-g.actual)**2)),
                'weeks':len(g[['season','week']].drop_duplicates())}
            medians=g.loc[g['median'].notna()]
            if len(medians):
                result['clustered_median_mae_gain']=clustered_gain(medians.assign(
                    reference_error=abs(medians.production-medians.actual),challenger_error=abs(medians['median']-medians.actual)))
            if 'outcome' in g:
                result['challenger_probability']=probability_metrics(g.challenger_probability,g.outcome)
                result['production_probability']=probability_metrics(g.production_probability,g.outcome)
                paired=g.dropna(subset=['challenger_probability','production_probability','outcome'])
                result['clustered_brier_gain']=clustered_gain(paired.assign(
                    reference_error=(paired.production_probability-paired.outcome)**2,
                    challenger_error=(paired.challenger_probability-paired.outcome)**2))
                result['probability_scoring_stages']=paired.scoring_stage.value_counts().to_dict()
            cohorts['|'.join(key)]=result
    return {'status':'evaluated' if rows else 'waiting_for_settled_prospective_challengers',
        'pending':pending,'unique_pending_player_games':len(pending_unique),'pending_by_cohort':dict(pending_by_cohort),
        'settled_player_games':len(rows),'cohorts':cohorts,'exclusions':dict(exclusions),
        'automatic_promotion':False}


def run(day=None):
    records=load_rows(day)
    report=build_report(records)
    report['prospective_challengers']=prospective_report(records)
    suffix=str(day) if day else 'all'
    atomic_json(ROOT/'reports'/f'nfl_live_scoring_replay_{suffix}.json',report)
    atomic_json(ROOT/'reports'/'nfl_live_scoring_replay_latest.json',report)
    lines=['# NFL Live Scoring Replay','',f"Status: {report['status']}",
        f"Unique offer decisions: {report['unique_offer_decisions']}; settled: {report['settled_offer_decisions']}",
        '', '## Replay Coverage','',*[f'- {k}: {v}' for k,v in report['replay_status'].items()],
        '', '## Probability Stages','', '| Cohort | Stage | Rows | Brier | Calibration error |','|---|---|---:|---:|---:|']
    for key,stages in report['probability_stages'].items():
        for stage,metric in stages.items():
            if metric.get('rows',0):
                lines.append(f"| {key} | {stage} | {metric['rows']} | {metric['brier']:.4f} | {metric['calibration_error']:.4f} |")
    lines+=['',*['- '+s for s in report['limitations']]]
    lines+=['','## Prospective Challengers','',f"Status: {report['prospective_challengers']['status']}",
        f"Settled cohort rows: {report['prospective_challengers']['settled_player_games']}; unique pending player-games: {report['prospective_challengers']['unique_pending_player_games']}",
        'Different challenger runs are compared separately, not pooled as independent outcomes.']
    (ROOT/'reports'/'nfl_live_scoring_replay_latest.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--date')
    args=parser.parse_args(); r=run(date.fromisoformat(args.date) if args.date else None)
    print(json.dumps({k:v for k,v in r.items() if k not in {'probability_stages','projection_scores'}},indent=2))


if __name__=='__main__': main()
