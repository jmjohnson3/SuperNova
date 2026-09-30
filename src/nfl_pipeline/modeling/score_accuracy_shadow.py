"""Prospective challenger forecasts from immutable live features, never betting rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date,datetime,timezone

import joblib
import pandas as pd
import psycopg2

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT,atomic_json,active_release
from nfl_pipeline.context_contract import validate_evidence
from nfl_pipeline.modeling.challenger_models import OPPORTUNITY,player_features,curve_summary,curve_over,numeric
from nfl_pipeline.modeling.train_accuracy_challengers import mixture_curve


def run(day):
    pointer=MODEL_ROOT/'challengers'/'latest.json'
    if not pointer.exists(): return {'status':'waiting_for_challenger_artifact','rows':0}
    manifest=json.loads(pointer.read_text(encoding='utf-8'))
    root=MODEL_ROOT/'challengers'/manifest['run_id']
    path=root/'models.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']: raise RuntimeError('Challenger checksum mismatch')
    artifact=joblib.load(path)
    if artifact['production_release']!=active_release()['release_id']: raise RuntimeError('Challenger belongs to a different frozen release')
    if date.fromisoformat(artifact['training_end'])>=day: raise RuntimeError('Challenger trained on requested prediction date')
    with psycopg2.connect(PG_DSN) as conn,conn.cursor() as cur:
        cur.execute("""SELECT p.id,p.forecast_payload FROM bets.nfl_player_prop_predictions p
            JOIN raw.nfl_games g USING(game_id) WHERE p.is_current AND p.game_date_et=%s
            AND g.start_ts_utc>NOW() AND p.created_at_utc<g.start_ts_utc AND p.integrity_version='nfl-asof-v2'
            ORDER BY p.created_at_utc,p.id""",(day,))
        records=cur.fetchall()
    rows=[]; seen=set(); missing=0; by_stat={}
    for forecast_id,p in records:
        stat=p['stat']; key=(p['game_id'],p['player_id'],stat)
        if stat not in artifact['models'] or key in seen: continue
        features=p.get('forecast_features') or (p.get('scoring_replay') or {}).get('row')
        if not features: missing+=1; continue
        if p.get('model_version')!=artifact['production_release']:
            missing+=1; continue
        if validate_evidence(p.get('context_evidence'),p.get('prediction_context_cutoff_utc')) is None:
            missing+=1; continue
        seen.add(key)
        features=dict(features,context_evidence=p.get('context_evidence'))
        by_stat.setdefault(stat,[]).append((forecast_id,p,features))
    for stat,items in by_stat.items():
        # Batch by model: hundreds of one-row LightGBM calls exceed live timeouts.
        frame=pd.DataFrame([features for _,_,features in items]); bundle=artifact['models'][stat]
        values,weights,comp=mixture_curve(bundle['model'],bundle['uncertainty'],frame,stat)
        summary=curve_summary(values,weights)
        confidence=bundle['uncertainty'].confidence(player_features(frame,stat))
        expected=(comp['weights']*comp['opportunities']).sum(axis=1)
        for idx,(forecast_id,p,_) in enumerate(items):
            row={'forecast_id':forecast_id,'game_id':p['game_id'],'player_id':p['player_id'],'stat':stat,
                 'production_release':p['model_version'],'production_projection':p['projection'],
                 'challenger_run':manifest['run_id'],'source_context_cutoff':p['prediction_context_cutoff_utc'],
                 'projection_mean':float(summary['mean'][idx]),'projection_median':float(summary['median'][idx]),
                 'p10':float(summary['p10'][idx]),'p90':float(summary['p90'][idx]),
                 'confidence':float(confidence[idx]),
                 'low_workload_probability':float(comp['weights'][idx,0]),'spike_workload_probability':float(comp['weights'][idx,2]),
                 'expected_opportunity':float(expected[idx]),'betting_eligible':False}
            if p.get('line') is not None:
                row['locked_line']=p['line']; row['raw_over_probability']=float(curve_over(values[idx:idx+1],weights[idx:idx+1],[p['line']])[0])
                op=float(numeric(frame,f'{OPPORTUNITY[stat]}_avg_5').fillna(0).iloc[idx])
                calibration=pd.DataFrame([{'probability':row['raw_over_probability'],'position':p['position'],
                    'workload_bucket':'low' if op<4 else 'normal' if op<10 else 'high',
                    'line_bucket':str(int(float(p['line'])//(50 if stat=='passing_yards' else 20))),
                    'book':p['book']}])
                row['calibrated_over_probability']=float(bundle['calibrator'].predict(calibration)[0])
            rows.append(row)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report=clean({'status':'shadow_only','date':str(day),'rows':rows,'missing_lock_features':missing,
                  'training_end':artifact['training_end'],
                  'scored_at':datetime.now(timezone.utc).isoformat(),'source_manifest':manifest})
    atomic_json(root/'prospective'/str(day)/(stamp+'.json'),report)
    return {'status':'shadow_only','run_id':manifest['run_id'],'rows':len(rows),'missing_lock_features':missing}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--date',required=True)
    args=parser.parse_args(); print(json.dumps(run(date.fromisoformat(args.date)),indent=2))


if __name__=='__main__': main()
