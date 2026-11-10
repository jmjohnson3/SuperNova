"""Independent, timestamped NBA research inputs. Never alters frozen player tables."""
from __future__ import annotations
from collections import Counter
from datetime import date,datetime,timedelta,timezone
from pathlib import Path
import gzip
import hashlib
import json

import numpy as np
import pandas as pd
import psycopg2.extras

from nfl_pipeline.context_contract import safe_time,number
from nfl_pipeline.integrity import atomic_json

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/nba_props/research'
STATS=('pts','reb','ast','fg3m','fg3a','fga','fgm','fta','ftm','tov','oreb')
SCHEMA="""
CREATE SCHEMA IF NOT EXISTS nbaprops;
CREATE TABLE IF NOT EXISTS nbaprops.research_games (
 snapshot_id text PRIMARY KEY, game_id text NOT NULL, game_date date NOT NULL,
 observed_at timestamptz NOT NULL, payload jsonb NOT NULL);
CREATE INDEX IF NOT EXISTS research_games_asof ON nbaprops.research_games(game_id,observed_at);
CREATE TABLE IF NOT EXISTS nbaprops.research_context (
 id bigserial PRIMARY KEY,game_id text NOT NULL,player_id bigint NOT NULL,team text NOT NULL,
 observed_at timestamptz NOT NULL,recorded_at timestamptz NOT NULL DEFAULT now(),
 minutes_cap numeric,source text NOT NULL,note text NOT NULL,
 CHECK(minutes_cap>0 AND minutes_cap<=48));
CREATE OR REPLACE FUNCTION nbaprops.research_immutable() RETURNS trigger AS $$
 BEGIN RAISE EXCEPTION 'Research evidence is append-only'; END; $$ LANGUAGE plpgsql;
DO $$ DECLARE t text; BEGIN
 FOREACH t IN ARRAY ARRAY['research_games','research_context'] LOOP
 IF NOT EXISTS(SELECT 1 FROM pg_trigger WHERE tgrelid=('nbaprops.'||t)::regclass AND tgname='research_immutable') THEN
 EXECUTE format('CREATE TRIGGER research_immutable BEFORE UPDATE OR DELETE ON nbaprops.%I FOR EACH ROW EXECUTE FUNCTION nbaprops.research_immutable()',t);
 END IF; END LOOP; END $$;
"""


def now():return datetime.now(timezone.utc)
def clean(value):return json.loads(json.dumps(value,default=str,allow_nan=False))
def digest(value):return hashlib.sha256(json.dumps(clean(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def rows(conn,query,params=None):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query,params)
        return [dict(r) for r in cur.fetchall()]


def init(conn):
    with conn.cursor() as cur:cur.execute(SCHEMA)
    conn.commit()


def normalize(frame):
    records=[]
    for raw in frame.to_dict('records'):
        r={k.lower():v for k,v in raw.items()}
        r['game_id']=str(r['game_id']).zfill(10)
        r['game_date']=pd.Timestamp(r['game_date']).date().isoformat()
        matchup=str(r.get('matchup',''))
        r['home']=' vs. ' in matchup
        parts=matchup.replace(' vs. ',' @ ').split(' @ ')
        r['team']=r.get('team_abbreviation');r['opponent']=parts[-1] if len(parts)==2 else None
        r['minutes']=number(r.get('min'))
        for stat in STATS:r[stat]=number(r.get(stat))
        records.append({k:r.get(k) for k in ('game_id','game_date','player_id','player_name','team','opponent','home','minutes','wl',*STATS)})
    return records


def reconcile(players,teams):
    """Require two agreeing final team logs and complete, non-conflicting player totals."""
    valid=[];excluded=Counter()
    pg={};tg={}
    for p in players:pg.setdefault(p['game_id'],[]).append(p)
    for t in teams:tg.setdefault(t['game_id'],[]).append(t)
    for gid,group in pg.items():
        group=list({digest(p):p for p in group}.values())
        team=list({digest(t):t for t in tg.get(gid,[])}.values())
        reason=None
        if len({p.get('player_id') for p in group})!=len(group):reason='conflicting_player_keys'
        elif len(team)!=2 or len({t['team'] for t in team})!=2:reason='two_final_team_logs_missing'
        elif (sum(bool(t['home']) for t in team)!=1 or any(t.get('wl') not in ('W','L') for t in team)
              or {t.get('wl') for t in team}!={'W','L'} or len({t['game_date'] for t in team})!=1
              or any(t['opponent'] not in {u['team'] for u in team if u!=t} for t in team)
              or any(t['pts'] is None or t['pts']<=0 for t in team)
              or (team[0]['pts']==team[1]['pts'])
              or any((t['wl']=='W')!=(t['pts']>next(u['pts'] for u in team if u!=t)) for t in team)):
            reason='final_identity_unverified'
        if reason:excluded[reason]+=1;continue
        for t in team:
            part=[p for p in group if p['team']==t['team'] and p['opponent']==t['opponent'] and p['game_date']==t['game_date']]
            if not part or any(p['minutes'] is None or not 0<=p['minutes']<=70 for p in part):reason='participation_unverified';break
            if any(any(p[s] is None or p[s]<0 or p[s]!=int(p[s]) for s in STATS) for p in part):reason='stat_missing_or_invalid';break
            if any(p['fg3m']>p['fg3a'] or p['fg3a']>p['fga'] or p['fgm']>p['fga'] or p['ftm']>p['fta']
                   or p['pts']!=2*p['fgm']+p['fg3m']+p['ftm'] for p in part):reason='shot_totals_inconsistent';break
            if any(t[s] is None or abs(sum(p[s] for p in part)-t[s])>.01 for s in ('pts','fgm','fga','fg3m','fg3a','ftm','fta','ast')):
                reason='player_totals_disagree_with_team';break
            if t['minutes'] is None or not 239<=t['minutes']<=350 or abs(sum(p['minutes'] for p in part)-t['minutes'])>2:
                reason='incomplete_game_minutes';break
        if reason:excluded[reason]+=1;continue
        if len(group)!=sum(sum(p['team']==t['team'] for p in group) for t in team):excluded['unexpected_team']+=1;continue
        valid.append(dict(game_id=gid,game_date=team[0]['game_date'],players=group,teams=team,
                          final_verified=True,participation='positive minutes only; absent rows remain unknown'))
    return valid,dict(excluded)


def fetch_season(year,kind):
    from nba_api.stats.endpoints import playergamelogs,leaguegamelog
    season=f'{year}-{str(year+1)[-2:]}'
    start=now()
    p=playergamelogs.PlayerGameLogs(season_nullable=season,season_type_nullable=kind,timeout=30).get_data_frames()[0]
    t=leaguegamelog.LeagueGameLog(season=season,season_type_all_star=kind,player_or_team_abbreviation='T',timeout=30).get_data_frames()[0]
    at=now();raw=clean(dict(season=season,season_type=kind,requested_at=start,observed_at=at,
        source='stats.nba.com PlayerGameLogs + LeagueGameLog',
        players=p.astype(object).where(p.notna(),None).to_dict('records'),
        teams=t.astype(object).where(t.notna(),None).to_dict('records')))
    folder=DATA/'raw';folder.mkdir(parents=True,exist_ok=True)
    target=folder/(at.strftime('%Y%m%dT%H%M%S%fZ')+f'_{year}_{kind.replace(" ","_")}.json.gz')
    with gzip.open(target,'wt',encoding='utf-8') as f:json.dump(raw,f,allow_nan=False)
    valid,bad=reconcile(normalize(p),normalize(t))
    return valid,bad,dict(path=str(target.relative_to(ROOT)),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),observed_at=at)


def ingest(conn,years):
    init(conn);summary=[]
    for year in years:
        for kind in ('Regular Season','Playoffs'):
            if year>=now().year and kind=='Playoffs':continue
            try:games,bad,receipt=fetch_season(year,kind)
            except Exception as exc:
                summary.append(dict(year=year,kind=kind,error=type(exc).__name__,detail=str(exc)[:180]));print(summary[-1],flush=True);continue
            values=[]
            for g in games:
                payload=clean(dict(g,receipt=receipt,season=year,season_type=kind))
                values.append((digest(payload),g['game_id'],g['game_date'],receipt['observed_at'],psycopg2.extras.Json(payload)))
            with conn.cursor() as cur:
                if values:psycopg2.extras.execute_values(cur,'INSERT INTO nbaprops.research_games VALUES %s ON CONFLICT DO NOTHING',values,page_size=500)
            conn.commit()
            summary.append(dict(year=year,kind=kind,verified_games=len(games),excluded=bad,receipt=clean(receipt)))
            print({k:v for k,v in summary[-1].items() if k!='receipt'},flush=True)
    atomic_json(ROOT/'reports/nba_research_ingestion_latest.json',dict(built_at=now(),runs=summary))
    return summary


def history(conn,asof):
    ids=rows(conn,"""SELECT DISTINCT ON(game_id) snapshot_id FROM nbaprops.research_games
        WHERE observed_at<=%s ORDER BY game_id,observed_at DESC,snapshot_id""",(asof,))
    cache=DATA/'cache'/('history_v1_'+digest(sorted(r['snapshot_id'] for r in ids))+'.pkl')
    if cache.exists():return pd.read_pickle(cache)
    found=rows(conn,"""SELECT DISTINCT ON(game_id) snapshot_id,observed_at,payload FROM nbaprops.research_games
        WHERE observed_at<=%s ORDER BY game_id,observed_at DESC,snapshot_id""",(asof,))
    players=[];teams=[]
    for r in found:
        p=r['payload']
        for target,key in ((players,'players'),(teams,'teams')):
            fields=('game_id','game_date','player_id','player_name','team','opponent','home','minutes','wl',*STATS)
            target.extend(dict({k:item.get(k) for k in fields},source_snapshot=r['snapshot_id'],source_observed_at=r['observed_at']) for item in p[key])
    result=(pd.DataFrame(players),pd.DataFrame(teams))
    cache.parent.mkdir(parents=True,exist_ok=True)
    temporary=cache.with_suffix('.tmp');pd.to_pickle(result,temporary);temporary.replace(cache)
    return result


def context(conn,game_id,team,player_id,cutoff):
    """Only observations AND publication times known at the forecast cutoff may affect it."""
    lineup=rows(conn,"""SELECT * FROM nbaprops.lineup_snapshots WHERE game_id=%s AND team=%s AND player_id=%s
        AND captured_at_utc<=%s AND captured_at_utc>=%s-interval '6 hours' ORDER BY captured_at_utc DESC LIMIT 1""",
        (game_id,team,player_id,cutoff,cutoff))
    out=dict(starter=None,availability='unknown',minutes_cap=None,lineup_observed_at=None,injury_observed_at=None)
    if lineup:
        l=lineup[0];published=safe_time(l.get('source_timestamp'))
        if not published or published<=cutoff:
            roster=str(l.get('roster_status') or '').lower()
            out.update(starter=bool(l['starter']) if str(l.get('lineup_status','')).lower()=='confirmed' else None,
                       availability='out' if roster=='inactive' else 'active' if roster=='active' else 'unknown',
                       lineup_observed_at=l['captured_at_utc'])
    from nba_props.model import match_name
    names=rows(conn,'SELECT player_name FROM nbaprops.player_games WHERE player_id=%s ORDER BY game_date DESC LIMIT 1',(player_id,))
    if names:
        injury=rows(conn,"""SELECT * FROM nbaprops.injury_snapshots WHERE team=%s AND name_key=%s AND captured_at_utc<=%s
            AND captured_at_utc>=%s-interval '6 hours'
            AND captured_at_utc=(SELECT max(captured_at_utc) FROM nbaprops.injury_snapshots WHERE captured_at_utc<=%s)
            ORDER BY captured_at_utc DESC LIMIT 1""",
            (team,match_name(names[0]['player_name']),cutoff,cutoff,cutoff))
        if injury:
            i=injury[0];published=safe_time(i.get('reported_at'))
            if not published or published<=cutoff:
                status=str(i['status']).lower()
                out.update(availability='out' if status=='out' else 'uncertain' if status in ('questionable','doubtful','day-to-day') else out['availability'],injury_observed_at=i['captured_at_utc'])
    caps=rows(conn,"""SELECT * FROM nbaprops.research_context WHERE game_id=%s AND player_id=%s AND team=%s
        AND observed_at<=%s AND recorded_at<=%s ORDER BY observed_at DESC,id DESC LIMIT 1""",(game_id,player_id,team,cutoff,cutoff))
    if caps:out.update(minutes_cap=float(caps[0]['minutes_cap']),cap_source=caps[0]['source'],cap_observed_at=caps[0]['observed_at'])
    return clean(out)


def audit(conn):
    init(conn)
    result=dict(built_at=now(),historical_limit='Results retrieved now cannot prove their historical availability or revision state. Historical tests are reconstructed development evidence; prospective forecasts enforce observed_at <= cutoff.',
        missing_participation='never inferred from missing player rows',historical_lineups='not used as contemporaneous context; only timestamped prospective context is consumed',
        frozen_inputs_changed=False,betting_authorized=False)
    result['labels']=rows(conn,'SELECT count(*) snapshots,count(DISTINCT game_id) games,min(game_date),max(game_date) FROM nbaprops.research_games')[0]
    result['lineups']=rows(conn,"""SELECT count(*) rows,
        count(*) FILTER(WHERE (captured_at_utc AT TIME ZONE 'America/New_York')::time='23:59:00'::time) end_of_day_backfill_rows,
        min(captured_at_utc),max(captured_at_utc) FROM nbaprops.lineup_snapshots""")[0]
    result['rules']=dict(result_lag_days=2,no_current_game_labels_in_features=True,three_point_attempts='fg3a in independent verified research snapshots',
        unavailable_historical_context='unknown, never reconstructed from current-game participation')
    atomic_json(ROOT/'reports/nba_research_audit_latest.json',clean(result))
    return clean(result)
