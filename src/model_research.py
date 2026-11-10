"""Prospective FanDuel model-assisted decisions; never grants wagering authority.

The existing price experiments are untouched. A paired price-only control and a
model-assisted selection are saved on the same first qualifying offer. Forecasts
must exist before observation and capture must happen live, before the game.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
import math
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras

from nfl_pipeline import experiment_registry as registry, sharp_math
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / 'src/model_research_experiments'
NAME = 'fanduel-model-assisted-paper-v1'
NBA_STATS = {'points':'pts', 'rebounds':'reb', 'assists':'ast', 'threes':'fg3m'}
NFL_STATS = {'receiving_yards', 'rushing_yards', 'passing_yards', 'receptions', 'passing_tds'}
CONFIG = dict(book='fanduel', reference='pinnacle', min_price_ev=.03,
    window_minutes=[15,45], max_capture_delay_seconds=300, max_forecast_age_hours=12,
    nba_tip_tolerance_minutes=15,
    lines='half-integer only; no fabricated push probability',
    selection='first exact price qualifier per sport/game/market/player; highest EV within observation',
    props='independent model probability must imply positive EV at this exact offer',
    games='projected margin or total must favor the offered side; no invented game probability',
    nba_refit_policy='fixed source recipe; daily fits on earlier games, saved forecast version and values',
    missing_forecast='veto; never backfilled', monthly_data_cost_usd='30.00', comparison_stake_usd='1.00',
    nba_review=dict(min_days=21,min_games=20,min_checked_closes=50,min_exact_coverage=.9,
                    min_check_coverage=.9,min_available=.8,min_alert_close_ev=.015),
    nfl_review=dict(min_days=21,min_games=20,min_checked_closes=100,min_exact_coverage=.9,
                    min_check_coverage=.9,min_available=.8,min_alert_close_ev=.015),
    review='positive game/day clustered lower bounds; positive paired improvement; review only',
    automatic_betting_authorization=False)

SCHEMA = """
CREATE TABLE IF NOT EXISTS bets.model_research_decisions (
 id BIGSERIAL PRIMARY KEY, experiment_id TEXT NOT NULL, experiment_sha256 TEXT NOT NULL,
 sport TEXT NOT NULL, event_id TEXT NOT NULL, market TEXT NOT NULL, player_norm TEXT NOT NULL,
 captured_at TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL,
 UNIQUE(experiment_id,sport,event_id,market,player_norm));
CREATE TABLE IF NOT EXISTS bets.model_research_delivery (
 id BIGINT PRIMARY KEY REFERENCES bets.model_research_decisions(id),
 delivery_status TEXT NOT NULL, attempted_at TIMESTAMPTZ, sent_at TIMESTAMPTZ,
 message_id TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS bets.model_research_checks (
 id BIGINT PRIMARY KEY REFERENCES bets.model_research_decisions(id),
 checked_at TIMESTAMPTZ NOT NULL, recorded_at TIMESTAMPTZ NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('available','changed_line','disappeared','suspended')),
 displayed_line NUMERIC, displayed_price INTEGER, note TEXT,
 CHECK((status IN ('available','changed_line') AND displayed_line IS NOT NULL
        AND displayed_price IS NOT NULL AND abs(displayed_price)>=100)
    OR (status IN ('disappeared','suspended') AND displayed_line IS NULL AND displayed_price IS NULL)));
CREATE OR REPLACE FUNCTION bets.model_research_immutable() RETURNS trigger AS $$
 BEGIN RAISE EXCEPTION 'Model research evidence is append-only'; END; $$ LANGUAGE plpgsql;
DO $$ BEGIN
 IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgrelid='bets.model_research_decisions'::regclass
                AND tgname='model_research_decision_immutable') THEN
 CREATE TRIGGER model_research_decision_immutable BEFORE UPDATE OR DELETE ON bets.model_research_decisions
 FOR EACH ROW EXECUTE FUNCTION bets.model_research_immutable(); END IF;
 IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgrelid='bets.model_research_checks'::regclass
                AND tgname='model_research_check_immutable') THEN
 CREATE TRIGGER model_research_check_immutable BEFORE UPDATE OR DELETE ON bets.model_research_checks
 FOR EACH ROW EXECUTE FUNCTION bets.model_research_immutable(); END IF;
END $$;
"""


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,(date,datetime)):return value.isoformat()
    if isinstance(value,Decimal):value=float(value)
    if hasattr(value,'item'):value=value.item()
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def dependencies():
    from sharp_pipeline import nba_experiment
    from nfl_pipeline.modeling import receiving_paper_trial
    from nfl_pipeline import sharp_watch
    nba = nba_experiment.active()
    receiving = registry.check_experiment(receiving_paper_trial.NAME)
    receiving_paper_trial._check_candidate(receiving)
    nfl = registry.check_experiment(sharp_watch.EXPERIMENT_NAME)
    return dict(nba=nba, nfl=nfl, receiving=receiving)


def configuration(docs):
    return dict(CONFIG, parents={k:dict(experiment_id=d['experiment_id'],sha256=d['sha256']) for k,d in docs.items()})


def register():
    docs = dependencies()
    modules = ['model_research','model_research_report','model_research_checks','model_research_results',
               'nba_props.live','nba_props.model','nba_props.games','nba_props.availability','nba_props.ingest',
               'nba_props','sharp_pipeline','nfl_pipeline']
    files = {}
    absent = {}
    for d in docs.values():
        for label,e in d['files'].items():
            (absent if e.get('absent') else files)[e['path']] = ROOT / e['path']
    for m in modules:
        p = ROOT/'src'/Path(*m.split('.')).with_suffix('.py')
        if not p.exists(): p = ROOT/'src'/Path(*m.split('.'))/'__init__.py'
        files[m] = p
    files['active_nfl_release']=ROOT/'src/nfl_pipeline/modeling/models/active_release.json'
    return registry.register_experiment(NAME,configuration(docs),files,root=ROOT,store=STORE,absent_paths=absent)


def active():
    doc = registry.check_experiment(NAME,root=ROOT,store=STORE)
    if doc['config'] != configuration(dependencies()): raise ValueError('Model research dependency/configuration drift')
    return doc


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError,ValueError): return None


def eligible(row, registered_at, now):
    at, start = safe_time(row.get('decided_at')), safe_time(row.get('commence_time_utc'))
    line = number(row.get('line'))
    value = number(row.get('ev'))
    return bool(at and start and at >= safe_time(registered_at) and at <= now < start
        and (now-at).total_seconds() <= CONFIG['max_capture_delay_seconds']
        and 15 <= (start-at).total_seconds()/60 <= 45
        and row.get('book')=='fanduel' and row.get('method')=='exact'
        and line is not None and abs(line-math.floor(line)-.5)<1e-9
        and value is not None and value >= CONFIG['min_price_ev']
        and sharp_math.implied(row.get('price')) is not None)


def assess(row, forecast):
    """A veto changes the model strategy's decision, never the paired price control."""
    out = dict(selected=False,reason='missing_forecast',model_probability=None,model_ev=None,
               directional_margin=None,forecast=clean(forecast) if forecast else None)
    if not forecast: return out
    at, made = safe_time(row['decided_at']), safe_time(forecast.get('run_at_utc'))
    if not made or not 0 <= (at-made).total_seconds() <= CONFIG['max_forecast_age_hours']*3600:
        return dict(out,reason='forecast_after_offer_or_stale')
    if safe_time(forecast.get('tip_utc')) and at>=safe_time(forecast['tip_utc']):
        return dict(out,reason='past_model_game_start')
    if forecast.get('identity_verified') is not True or not forecast.get('model_version'):
        return dict(out,reason='forecast_identity_or_version_missing')
    market, side, line = row['market'], row['side'], float(row['line'])
    if market in {'spread','total'}:
        if market=='spread':
            mean = number(forecast.get('pred_margin'))
            gap = None if mean is None else (mean if side=='home' else -mean)+line
        else:
            mean = number(forecast.get('pred_total'))
            gap = None if mean is None else (mean-line)*(1 if side=='over' else -1)
        return dict(out,selected=gap is not None and gap>0,reason='model_supports_side' if gap is not None and gap>0
                    else 'model_opposes_side' if gap is not None else 'missing_game_projection', directional_margin=gap)
    if forecast.get('available') is not True:
        return dict(out,reason='participation_not_supported')
    if row['sport']=='nba':
        from nba_props.model import p_over
        mu,a,b = (number(forecast.get(k)) for k in ('mu','disp_a','disp_b'))
        if mu is None or mu<0 or a is None or a<=0 or b is None or b<0:
            return dict(out,reason='invalid_model_distribution')
        over = float(p_over(np.array([mu]),np.array([line]),a,b)[0])
    else:
        if number(forecast.get('line')) != line:
            return dict(out,reason='model_probability_wrong_line')
        over = number(forecast.get('raw_over_probability'))
    if over is None or not 0<over<1:
        return dict(out,reason='independent_model_probability_missing')
    p = over if side=='over' else 1-over
    ev = sharp_math.ev(p,row['price'])
    return dict(out,selected=ev>0,reason='model_positive_ev' if ev>0 else 'model_nonpositive_ev',model_probability=p,model_ev=ev)


def rows(conn, sql, params=()):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql,params)
        return [dict(r) for r in cur.fetchall()]


def nba_team(name):
    from nba_api.stats.static import teams
    names = {t['full_name']:t['abbreviation'] for t in teams.get_teams()}
    names['LA Clippers']='LAC'
    return names.get(name)


def forecast_for(conn,row):
    at = safe_time(row['decided_at'])
    start = safe_time(row['commence_time_utc'])
    earliest = at-timedelta(hours=CONFIG['max_forecast_age_hours'])
    if row['sport']=='nba':
        from nba_props.model import match_name
        home,away=nba_team(row['home_team']),nba_team(row['away_team'])
        if not home or not away: return None
        games = rows(conn,"""SELECT * FROM nbaprops.game_projections WHERE run_at_utc BETWEEN %s AND %s
            AND home_team=%s AND away_team=%s AND tip_utc BETWEEN %s AND %s ORDER BY run_at_utc DESC""",
            (earliest,at,home,away,start-timedelta(minutes=15),start+timedelta(minutes=15)))
        if not games or len({g['game_id'] for g in games})!=1: return None
        from nba_props.live import MODEL_VERSION
        if games[0]['model_version']!=MODEL_VERSION:return None
        if row['market'] in {'spread','total'}: return dict(games[0],identity_verified=True)
        found = rows(conn,"""SELECT * FROM nbaprops.projections WHERE run_at_utc BETWEEN %s AND %s
            AND game_id=%s AND tip_utc=%s AND name_key=%s AND stat=%s
            AND ((team=%s AND opponent=%s) OR (team=%s AND opponent=%s)) ORDER BY run_at_utc DESC,player_id""",
            (earliest,at,games[0]['game_id'],games[0]['tip_utc'],match_name(row['player']),NBA_STATS[row['market']],home,away,away,home))
        if not found or len({r['player_id'] for r in found})!=1 or found[0]['model_version']!=MODEL_VERSION: return None
        f=found[0]
        lineup=rows(conn,"""SELECT roster_status,captured_at_utc FROM nbaprops.lineup_snapshots
            WHERE game_id=%s AND player_id=%s AND captured_at_utc<=%s ORDER BY captured_at_utc DESC LIMIT 1""",
            (f['game_id'],f['player_id'],at))
        injury=rows(conn,"""SELECT status,captured_at_utc FROM nbaprops.injury_snapshots
            WHERE team=%s AND name_key=%s AND captured_at_utc=(SELECT max(captured_at_utc)
            FROM nbaprops.injury_snapshots WHERE captured_at_utc<=%s)""",(f['team'],f['name_key'],at))
        available=bool(lineup and str(lineup[0]['roster_status']).lower()=='active'
            and (at-lineup[0]['captured_at_utc']).total_seconds()<=1200
            and not any(str(i['status']).lower() in {'out','doubtful','inactive'} for i in injury))
        return dict(f,identity_verified=True,available=available,lineup=lineup,injury=injury)
    game = rows(conn,"SELECT * FROM raw.nfl_games WHERE game_id=%s AND start_ts_utc=%s",(row['game_id'],start))
    if not game: return None
    if row['market'] in {'spread','total'}:
        found=rows(conn,"""SELECT id,created_at_utc AS run_at_utc, model_version,
            predicted_home_margin AS pred_margin,predicted_total_points AS pred_total
            FROM bets.nfl_game_predictions WHERE game_id=%s AND created_at_utc BETWEEN %s AND %s
            ORDER BY created_at_utc DESC,id DESC LIMIT 1""",(row['game_id'],earliest,at))
    else:
        found=rows(conn,"""SELECT id,created_at_utc AS run_at_utc,model_version,line,raw_over_probability,
            projection,roster_status,injury_report_status,player_id FROM bets.nfl_player_prop_predictions
            WHERE game_id=%s AND offer_player_name_norm=%s AND stat=%s AND book='fanduel'
            AND created_at_utc BETWEEN %s AND %s ORDER BY created_at_utc DESC,id DESC""",
            (row['game_id'],row['player_norm'],row['market'],earliest,at))
        if not found or len({r['player_id'] for r in found})!=1: return None
    if not found: return None
    f=found[0]
    return dict(f,identity_verified=True,available=f.get('roster_status')=='ACT'
                and str(f.get('injury_report_status') or '').lower() not in {'out','doubtful','inactive'})


def candidates(conn,sport,doc,now):
    after=max(safe_time(doc['registered_at']),now-timedelta(seconds=300))
    if sport=='nba':
        return rows(conn,"""SELECT d.*, a.alert_id AS parent_check_id FROM bets.sharp_multi_decisions d
            LEFT JOIN bets.nba_paper_price_alerts a USING(decision_id)
            WHERE d.sport='nba' AND d.book='fanduel' AND d.tier='decision' AND d.method='exact'
            AND d.decided_at BETWEEN %s AND %s AND d.commence_time_utc>%s
            AND d.experiment_id=%s AND d.experiment_sha256=%s
            AND d.market IN ('points','rebounds','assists','threes','spread','total')
            ORDER BY d.decided_at,d.ev DESC,d.decision_id""",(after,now,now,
            doc['config']['parents']['nba']['experiment_id'],doc['config']['parents']['nba']['sha256']))
    from nfl_pipeline import sharp_watch
    records=rows(conn,"""SELECT payload FROM bets.nfl_sharp_observations
        WHERE observed_at_utc BETWEEN %s AND %s ORDER BY observed_at_utc,observation_id""",(after,now))
    result=[]
    for record in records:
        p=record['payload']; trace=p.get('quote_provenance') or {}
        if (p.get('experiment_sha256')!=doc['config']['parents']['nfl']['sha256']
                or p.get('observation_id')!=registry._digest({k:v for k,v in p.items() if k!='observation_id'})
                or p.get('stat') not in NFL_STATS|{'total','spread'} or p.get('sharp_book')!='pinnacle'): continue
        at=safe_time(trace.get('observed_at_utc'))
        eq,rq=trace.get('execution_quote') or {},trace.get('reference_quote') or {}
        if not at or sharp_watch.quote_error(eq,at) or sharp_watch.quote_error(rq,at): continue
        if abs((sharp_watch.source_time(eq)-sharp_watch.source_time(rq)).total_seconds())>120: continue
        game=rows(conn,"SELECT home_team_abbr,away_team_abbr FROM raw.nfl_games WHERE game_id=%s",(p['game_id'],))
        if not game: continue
        result.append(dict(p,sport='nfl',market=p['stat'],player_norm=p.get('player_norm') or '',
            decided_at=at,method=trace.get('reference_method'),parent_check_id=None,
            home_team=game[0]['home_team_abbr'],away_team=game[0]['away_team_abbr'],
            execution_updated_at=sharp_watch.source_time(eq),reference_updated_at=sharp_watch.source_time(rq)))
    result.extend(nfl_spread_candidates(conn,after,now))
    return sorted(result,key=lambda r:(r['decided_at'],-r['ev']))


def spread_pair(quoted,book,home,away,home_line,at):
    """Pair both sides from the SAME immutable observation; retain their source clocks."""
    from nfl_pipeline import sharp_watch
    sides=[]
    for team,line in ((home,home_line),(away,-home_line)):
        found=[p for p in quoted if p.get('book')==book and p.get('player_name_norm')==team
               and number(p.get('line'))==line]
        if len(found)!=1:return None
        p=found[0];q=p['quote']
        if p.get('observation_id')!=registry._digest({k:v for k,v in p.items() if k!='observation_id'}):return None
        if sharp_math.implied(q.get('price')) is None:return None
        # quote_error expects a two-way quote; validate both actual side timestamps independently.
        if sharp_watch.quote_error(dict(q,over=q['price'],under=q['price']),at):return None
        sides.append(p)
    stamps=[sharp_watch.source_time(p['quote']) for p in sides]
    if abs((stamps[0]-stamps[1]).total_seconds())>120:return None
    return dict(home_price=sides[0]['quote']['price'],away_price=sides[1]['quote']['price'],
                updated=min(stamps),sources=sides)


def nfl_spread_candidates(conn,after,now):
    records=rows(conn,"""SELECT payload FROM bets.nfl_sharp_quote_observations
        WHERE observed_at_utc BETWEEN %s AND %s AND payload->>'stat'='spread'""",(after,now))
    groups={}
    for r in records:
        p=r['payload'];groups.setdefault((p['game_id'],p['event_id'],p['observed_at_utc'],p['commence_time_utc']),[]).append(p)
    result=[]
    for (gid,eid,stamp,kick),quotes in groups.items():
        games=rows(conn,"SELECT home_team_abbr,away_team_abbr FROM raw.nfl_games WHERE game_id=%s AND start_ts_utc=%s",(gid,kick))
        if len(games)!=1:continue
        home,away=games[0]['home_team_abbr'],games[0]['away_team_abbr'];at=safe_time(stamp)
        for line in {float(p['line']) for p in quotes if p['book']=='fanduel' and p['player_name_norm']==home}:
            fd=spread_pair(quotes,'fanduel',home,away,line,at);ref=spread_pair(quotes,'pinnacle',home,away,line,at)
            if not fd or not ref or abs((fd['updated']-ref['updated']).total_seconds())>120:continue
            p=sharp_math.no_vig_over(ref['home_price'],ref['away_price'])
            for side,fair,threshold in (('home',p,line),('away',1-p,-line)):
                price=fd[side+'_price'];ev=sharp_math.ev(fair,price)
                if ev<CONFIG['min_price_ev']:continue
                result.append(dict(sport='nfl',game_id=gid,event_id=eid,market='spread',player=None,player_norm='',
                    home_team=home,away_team=away,side=side,line=threshold,price=price,book='fanduel',
                    fair_probability=fair,ev=ev,method='exact',decided_at=at,commence_time_utc=kick,
                    parent_check_id=None,execution_updated_at=fd['updated'],reference_updated_at=ref['updated'],
                    spread_source_quotes=fd['sources']+ref['sources']))
    return result


def quote_times(conn,row):
    if row['sport']=='nfl': return row
    from sharp_pipeline import scan
    found=rows(conn,'SELECT payload FROM bets.sharp_multi_quotes WHERE observation_id=%s',(row['observation_id'],))
    if not found: return None
    event=found[0]['payload']; at=safe_time(row['decided_at'])
    if (event.get('id')!=row['event_id'] or event.get('home_team')!=row['home_team']
        or event.get('away_team')!=row['away_team']
        or safe_time(event.get('commence_time'))!=safe_time(row['commence_time_utc'])):return None
    keyline=float(row['line']) if row['side']!='away' else -float(row['line'])
    key=(row['market'],row['player_norm'] or 'game',keyline)
    quotes={b['key']:scan.pairs(dict(b,_home=event['home_team'],_away=event['away_team']),at).get(key)
            for b in event.get('bookmakers',[]) if b['key'] in ('fanduel','pinnacle')}
    if not quotes.get('fanduel') or not quotes.get('pinnacle'): return None
    side='a' if row['side'] in ('over','home') else 'b'
    if quotes['fanduel'][side]!=row['price']: return None
    if abs((quotes['fanduel']['updated']-quotes['pinnacle']['updated']).total_seconds())>120: return None
    p=sharp_math.no_vig_over(quotes['pinnacle']['a'],quotes['pinnacle']['b'])
    fair=p if side=='a' else 1-p
    if abs(fair-float(row['fair_probability']))>1e-9: return None
    return dict(row,execution_updated_at=quotes['fanduel']['updated'],reference_updated_at=quotes['pinnacle']['updated'])


def save(conn,doc,row,decision,now):
    payload=clean(dict(offer=row,decision=decision,captured_at=now.isoformat(),price_control_selected=True,
                       betting_authorized=False))
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO bets.model_research_decisions
            (experiment_id,experiment_sha256,sport,event_id,market,player_norm,captured_at,payload)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING id""",
            (NAME,doc['sha256'],row['sport'],row['event_id'],row['market'],row.get('player_norm') or '',now,
             psycopg2.extras.Json(payload)))
        r=cur.fetchone()
    return r[0] if r else None


def capture(conn,sport,doc,now):
    stored=[]
    for row in candidates(conn,sport,doc,now):
        if not eligible(row,doc['registered_at'],now): continue
        row=quote_times(conn,row)
        if row is None: continue
        decision=assess(row,forecast_for(conn,row))
        # Query time is part of live capture. Never save after the five-minute window or tip.
        captured=datetime.now(timezone.utc)
        if not eligible(row,doc['registered_at'],captured): continue
        model_tip=safe_time((decision.get('forecast') or {}).get('tip_utc'))
        if model_tip and captured>=model_tip:continue
        identity=save(conn,doc,row,decision,captured)
        if identity: stored.append(dict(id=identity,selected=decision['selected'],reason=decision['reason']))
    conn.commit()
    return stored


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['register','verify','capture','report','results'])
    ap.add_argument('--sport',choices=['nba','nfl'],default='nba')
    ap.add_argument('--alerts',action='store_true',help='Send real, fresh paper comparisons/check requests only')
    args=ap.parse_args(argv)
    doc=register() if args.command=='register' else active()
    result=dict(experiment_id=NAME,sha256=doc['sha256'],betting_authorized=False)
    if args.command in {'capture','report','results'}:
        with psycopg2.connect(PG_DSN,connect_timeout=10) as conn:
            from model_research_results import SCHEMA as RESULT_SCHEMA
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout='30s'");cur.execute(SCHEMA);cur.execute(RESULT_SCHEMA)
            conn.commit()
            if args.command=='results':
                from model_research_results import refresh
                result['results']=refresh(conn,doc)
            if args.command=='capture':
                result['captures']=capture(conn,args.sport,doc,datetime.now(timezone.utc))
                if args.alerts:
                    from model_research_checks import dispatch
                    result['delivery']=dispatch(conn,doc,args.sport)
            from model_research_report import build_report
            result['report']=build_report(conn,doc)
    print(json.dumps(clean(result),indent=2,allow_nan=False))
    if result.get('results',{}).get('errors'):return 1
    return 0


if __name__=='__main__': raise SystemExit(main())
