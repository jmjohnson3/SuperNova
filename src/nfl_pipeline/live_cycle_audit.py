"""Audit scheduled FanDuel pregame publication through close and settlement."""
import argparse
from datetime import date, datetime, timezone
import json

import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling.live_scoring_replay import ROOT, load_rows, validate_lock

EFFECTIVE_DATE=date(2026,9,24)


def publication_matches(publication,lookup,start):
    when=safe_time(publication.get('sent_at'))
    if not when or not when<start:
        return False
    manifests=[m for m in publication.get('forecast_manifest',[]) if m.get('kind')=='prop']
    if not manifests:
        return False
    verified = 0
    for m in manifests:
        r=lookup.get(int(m['forecast_id']));p=(r or {}).get('forecast_payload') or {}
        # Projection-only research rows are not offers. Verify their identity,
        # but do not demand a nonexistent price or treat them as betting proof.
        if r and all(p.get(k) is None and m.get(k) is None for k in ('book','side','line','price')):
            if str(r['game_id']) != str(publication.get('game_id')) or m.get('model_version') != p.get('model_version'):
                return False
            continue
        if not r or str(r['game_id'])!=str(publication.get('game_id')) or validate_lock(r):
            return False
        if any(str(m.get(k))!=str(p.get(k)) for k in ('book','side','model_version')):
            return False
        try:
            if any(float(m[k])!=float(p[k]) for k in ('line','price')):
                return False
        except (ValueError,KeyError,TypeError):
            return False
        quote=safe_time(r.get('offer_fetched_at'));lock=safe_time(r.get('created_at_utc'))
        if not quote or not lock or not quote<=lock<=when or (when-quote).total_seconds()>1200:
            return False
        verified += 1
    return verified > 0


def build(day,games,records,checkpoint,daily_runs,now):
    lookup={int(r['id']):r for r in records}
    capture=checkpoint.get('capture') or {}
    eligible=set(capture.get('eligible_forecast_ids',[]));paired=set(capture.get('captured_forecast_ids',[]))
    observed={r['forecast_id']:r for r in (checkpoint.get('close',{}).get('all_eligible',{}).get('observations',[]))}
    game_rows=[]
    for g in games:
        start=safe_time(g['start_ts_utc']);remaining=(start-now).total_seconds()/60
        ids={i for i in eligible if i in lookup and lookup[i]['game_id']==g['game_id']}
        # Successful sends are recorded individually. A later close/audit failure
        # must not erase an already delivered, independently verifiable manifest.
        publications=[p for d in daily_runs if d.get('pregame')
            for p in d.get('publications',[]) if str(p.get('game_id'))==str(g['game_id'])]
        matched=any(publication_matches(p,lookup,start) for p in publications)
        blockers=[]
        due=remaining<=20
        if due and not matched:
            blockers.append('game_aware_discord_manifest_missing_or_invalid')
        if due and ids-paired:
            blockers.append('eligible_receiving_challenger_capture_missing')
        if due and not ids:
            blockers.append('no_eligible_receiving_trial_offers')
        closed=[i for i in ids if observed.get(i,{}).get('valid_exact_capture')]
        coverage=len(closed)/len(ids) if remaining<=0 and ids else None
        if coverage is not None and coverage<.9:
            blockers.append('valid_exact_close_coverage_below_90_percent')
        unresolved=[i for i in ids if g['status']=='final' and lookup[i].get('actual') is None
                    and lookup[i].get('graded_result')!='void_nonparticipant']
        if unresolved:
            blockers.append('final_result_or_participation_unresolved')
        if g['status']=='final' and (checkpoint.get('settlement') or {}).get('missing_validated_capture',0):
            blockers.append('settled_offers_missing_validated_evaluation')
        phase=('waiting_for_pregame_window' if remaining>90 else 'pregame' if remaining>0
               else 'awaiting_settlement' if g['status']!='final' else 'settled')
        game_rows.append(dict(game_id=g['game_id'],kickoff=start.isoformat(),phase=phase,
            eligible_forecasts=len(ids),challenger_captures=len(ids&paired),
            discord_exact_prop_manifest_verified=matched,valid_close_coverage=coverage,
            unknown_closes=len(ids)-len(closed) if remaining<=0 else None,
            unresolved_forecast_ids=unresolved,blockers=blockers))
    blockers=[f"{g['game_id']}:{b}" for g in game_rows for b in g['blockers']]
    settled=bool(game_rows) and all(g['phase']=='settled' for g in game_rows)
    return dict(day=str(day),built_at=now.isoformat(),effective_date=str(EFFECTIVE_DATE),
        status='pre_contract_history' if day<EFFECTIVE_DATE else 'needs_attention' if blockers else 'no_games' if not games else 'complete' if settled else 'pending',
        games=game_rows,blockers=blockers,production_changed=False,betting_approved=False,
        note='Future close windows are pending, never counted as failed capture. Completion is operational, not cash approval.')


def report(day):
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='30s'")
        cur.execute('SELECT game_id,start_ts_utc,status FROM raw.nfl_games WHERE game_date_et=%s ORDER BY start_ts_utc',(day,))
        games=[dict(r) for r in cur.fetchall()]
    checkpoint_path=ROOT/'reports'/f'nfl_receiving_trial_checkpoint_{day}.json'
    checkpoint=json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    runs=[]
    for path in (ROOT/'reports').glob('nfl_daily_run_pregame_*.json'):
        d=json.loads(path.read_text())
        if d.get('date')==str(day):runs.append(d)
    doc=build(day,games,load_rows(day),checkpoint,runs,datetime.now(timezone.utc))
    for suffix in (str(day),'latest'):
        atomic_json(ROOT/'reports'/f'nfl_live_cycle_{suffix}.json',doc)
        lines=['# NFL Live Cycle', '',f"Date: {day}; status: {doc['status']}",doc['note'],'',
            '| Game | Phase | Eligible / challenger | Discord verified | Valid closes | Blockers |',
            '|---|---|---|---|---|---|']
        for g in doc['games']:
            coverage='pending' if g['valid_close_coverage'] is None else f"{g['valid_close_coverage']:.1%}"
            lines.append(f"| {g['game_id']} | {g['phase']} | {g['eligible_forecasts']} / {g['challenger_captures']} | {g['discord_exact_prop_manifest_verified']} | {coverage} | {', '.join(g['blockers']) or 'none'} |")
        (ROOT/'reports'/f'nfl_live_cycle_{suffix}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return doc


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--date',type=date.fromisoformat,required=True)
    result=report(p.parse_args().date);print(json.dumps(result,default=str))
    if result['status']=='needs_attention':raise SystemExit(1)
