"""Read-only, deduplicated prediction scorecard. No wagers, API calls or rescoring models.

python -m prediction_ledger [--since 2026-09-01] [--through 2026-11-05]
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from contextlib import closing
from datetime import date,datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg2

from nba_props import research_data as data
from nba_props.model import match_name
from model_research import nba_team
from nfl_pipeline.context_contract import safe_time,number
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.markets import normalize_name
from sharp_pipeline.outcomes import match_school

ROOT=Path(__file__).resolve().parents[1]
ET=ZoneInfo('America/New_York')
STATS={'points':'pts','rebounds':'reb','assists':'ast','threes':'fg3m'}
LABELS={'win':'Right','loss':'Wrong','push':'Push','void':'Void','pending':'Pending',
        'no_line':'No line','excluded':'Excluded'}


def team_code(name):
    from nba_api.stats.static import teams
    codes={t['abbreviation'] for t in teams.get_teams()}
    return name if name in codes else nba_team(name)


def first_decisions(records):
    """First saved decision per strategy/version/book/player-game/market, never best result."""
    groups={}
    for row in sorted(records,key=lambda r:(r['forecast_at'] or '',str(r['id']))):
        key=tuple(row.get(k) for k in ('cohort','version','sport','book','event_id','player_key','market'))
        if row['result']=='excluded':key+=('excluded',row['id'])
        if key not in groups:groups[key]=dict(row,repeated_observations=0)
        else:groups[key]['repeated_observations']+=1
    return list(groups.values())


def grade(market,side,line,actual):
    if actual is None:return 'pending'
    if side is None or line is None:return 'no_line'
    if market in ('spread','winner'):
        if side not in ('home','away'):return 'excluded'
        delta=(actual if side=='home' else -actual)+line
    else:
        if side not in ('over','under'):return 'excluded'
        delta=(actual-line)*(1 if side=='over' else -1)
    return 'win' if delta>0 else 'loss' if delta<0 else 'push'


def paper_profit(result,price):
    if price is None or abs(price)<100:return None
    return (price/100 if price>0 else 100/abs(price)) if result=='win' else -1. if result=='loss' else 0. if result in ('push','void') else None


def summary(records):
    counts=Counter(r['result'] for r in records);binary=counts['win']+counts['loss']
    errors=[r['prediction']-r['actual'] for r in records if r['result']!='excluded' and r['prediction'] is not None and r['actual'] is not None]
    return dict(rows=len(records),**{k:counts[k] for k in LABELS},
        accuracy=counts['win']/binary if binary else None,accuracy_denominator=binary,
        repeated_observations=sum(r.get('repeated_observations',0) for r in records),
        paper_units=sum(r['paper_units'] for r in records if r['paper_units'] is not None),
        priced_settlements=sum(r['paper_units'] is not None for r in records),
        mae=sum(abs(e) for e in errors)/len(errors) if errors else None,
        bias=sum(errors)/len(errors) if errors else None,error_rows=len(errors))


class Results:
    """Official final results only. Missing participation/statistics never imply a zero."""
    def __init__(self,conn,since,through):
        self.nba={};self.nba_by_match=defaultdict(list);self.nfl={};self.nfl_stats=defaultdict(list)
        self.ncaa=[];self.health=Counter()
        for row in data.rows(conn,"""SELECT DISTINCT ON(game_id) snapshot_id,payload FROM nbaprops.research_games
            WHERE game_date BETWEEN %s AND %s ORDER BY game_id,observed_at DESC,snapshot_id""",(since,through)):
            g=row['payload']
            if row['snapshot_id']!=data.digest(g) or g.get('final_verified') is not True:
                self.health['invalid_nba_result_snapshots']+=1;continue
            self.nba[g['game_id']]=g
        # Official cached recaps fill gaps without requesting data or changing source tables.
        from nba_props.paper_recap import scoreboard,boxscore
        for path in sorted((ROOT/'data/nba_props/paper_recaps').glob('*_official.json')):
            day=date.fromisoformat(path.name[:10])
            if not since<=day<=through:continue
            games,_=scoreboard(day)
            for game in games:
                if game['game_id'] in self.nba:continue
                players,_=boxscore(game)
                if not players:continue
                self.nba[game['game_id']]=dict(game_id=game['game_id'],game_date=game['game_date'],players=players,
                    teams=[dict(team=game[s+'_team'],home=s=='home',pts=game[s+'_score']) for s in ('home','away')])
        for g in self.nba.values():
            h=next(t for t in g['teams'] if t['home']);a=next(t for t in g['teams'] if not t['home'])
            self.nba_by_match[(g['game_date'],h['team'],a['team'])].append(g)
        for g in data.rows(conn,"SELECT game_id,start_ts_utc,home_team_abbr,away_team_abbr,home_score,away_score,status FROM raw.nfl_games WHERE (start_ts_utc AT TIME ZONE 'America/New_York')::date BETWEEN %s AND %s",(since,through)):
            self.nfl[g['game_id']]=g
        for r in data.rows(conn,"SELECT game_id,player_name,stat,actual_stat,result FROM bets.nfl_player_prop_prediction_results WHERE game_date_et BETWEEN %s AND %s",(since,through)):
            self.nfl_stats[(r['game_id'],normalize_name(r['player_name']),r['stat'])].append(r)
        self.ncaa=data.rows(conn,"SELECT game_id,start_date,home_team,away_team,home_points,away_points,completed FROM ncaaf.games WHERE (start_date AT TIME ZONE 'America/New_York')::date BETWEEN %s AND %s",(since,through))

    def actual(self,o):
        sport,market=o['sport'],o['market'];tip=safe_time(o['commence_time_utc'])
        if not tip:return None,'pending','missing_game_time'
        if sport=='nba':
            found=self.nba_by_match.get((tip.astimezone(ET).date().isoformat(),team_code(o['home_team']),team_code(o['away_team'])),[])
            if len(found)!=1:return None,'pending','official_final_missing_or_ambiguous'
            g=found[0]
            if o.get('game_id') and str(o['game_id'])!=g['game_id']:return None,'pending','game_identity_mismatch'
            if market not in ('spread','total','margin','winner'):
                players=[p for p in g['players'] if (str(p.get('player_id'))==str(o['player_id']) if o.get('player_id') is not None else match_name(p['player_name'])==match_name(o.get('player') or ''))]
                if len(players)!=1 or (number(players[0].get('minutes')) or 0)<=0:return None,'pending','participation_not_verified'
                actual=number(players[0].get(STATS.get(market,market)))
                if actual is None or actual<0:return None,'pending','stat_not_verified'
                return actual,None,'verified_nba_final_and_participation'
            home=number(next(t['pts'] for t in g['teams'] if t['home']));away=number(next(t['pts'] for t in g['teams'] if not t['home']))
        elif sport=='nfl':
            g=self.nfl.get(o.get('game_id'))
            if not g or g['status']!='final':return None,'pending','official_final_missing'
            if abs((safe_time(g['start_ts_utc'])-tip).total_seconds())>900:return None,'pending','game_identity_mismatch'
            if market not in ('spread','total','margin','winner'):
                found=self.nfl_stats.get((o['game_id'],normalize_name(o.get('player') or ''),market),[])
                values={number(r['actual_stat']) for r in found if r['result'] in ('win','loss','push') and number(r['actual_stat']) is not None}
                void=any(r['result']=='void_nonparticipant' for r in found)
                if void and values:return None,'pending','conflicting_participation_results'
                if len(values)==1:return values.pop(),None,'verified_nfl_participation_grade'
                if void:return None,'void','verified_nonparticipant'
                return None,'pending','participation_or_stat_not_verified'
            home,away=number(g['home_score']),number(g['away_score'])
        else:
            candidates=[g for g in self.ncaa if abs((safe_time(g['start_date'])-tip).total_seconds())<=900]
            home_name=match_school(o['home_team'],[g['home_team'] for g in candidates]);away_name=match_school(o['away_team'],[g['away_team'] for g in candidates])
            found=[g for g in candidates if g['home_team']==home_name and g['away_team']==away_name]
            if len(found)!=1 or not found[0]['completed']:return None,'pending','official_final_missing_or_ambiguous'
            home,away=number(found[0]['home_points']),number(found[0]['away_points'])
        if home is None or away is None:return None,'pending','final_scores_missing'
        if market in ('spread','margin','winner'):return home-away,None,'official_final_scores'
        if market=='total':return home+away,None,'official_final_scores'
        return None,'pending','unsupported_result_market'


def entry(results,cohort,identity,o,version='',prediction=None,probability=None,reason=None):
    tip=safe_time(o.get('commence_time_utc'));at=safe_time(o.get('decided_at'))
    actual,status,evidence=results.actual(o)
    line=number(o.get('line'));side=o.get('side');price=number(o.get('price'))
    if not tip or not at or at>=tip:status='excluded';evidence='not_saved_pregame'
    if reason:status='excluded';evidence=reason
    result=status or grade(o['market'],side,line,actual)
    return dict(id=str(identity),cohort=cohort,version=version or 'unversioned',sport=o['sport'].upper(),
        date=tip.astimezone(ET).date().isoformat() if tip else '',event_id=str(o.get('game_id') or o['event_id']),
        matchup=f"{o['away_team']} @ {o['home_team']}",home_team=o['home_team'],away_team=o['away_team'],player=o.get('player') or '',
        player_key=str(o.get('player_id') or normalize_name(o.get('player') or '')),
        market=o['market'],book=o.get('book') or 'no quoted price',side=side,line=line,price=price,
        prediction=number(prediction),probability=number(probability),actual=actual,result=result,reason=evidence,
        forecast_at=at.isoformat() if at else None,game_at=tip.isoformat() if tip else None,
        paper_units=paper_profit(result,price),source_status=o.get('source_status','saved; delivery not inferred'))


def load_rows(conn,since,through,results):
    output=[]
    def add(*args,**kw):output.append(entry(results,*args,**kw))
    for r in data.rows(conn,"""SELECT d.*,a.alert_id,a.sent_at FROM bets.nba_paper_price_alerts a
        JOIN bets.sharp_multi_decisions d USING(decision_id) WHERE a.delivery_status='sent'
        AND (d.commence_time_utc AT TIME ZONE 'America/New_York')::date BETWEEN %s AND %s""",(since,through)):
        r['source_status']='delivered paper alert';reason=None
        if not r['decided_at']<=r['sent_at']<r['commence_time_utc']:reason='not_delivered_pregame'
        add('Delivered price alerts',f"NBA #{r['alert_id']}",r,r['experiment_id'],reason=reason)
    for r in data.rows(conn,"""SELECT * FROM bets.sharp_multi_decisions WHERE tier='decision'
        AND (sport<>'nba' OR market NOT IN ('points','rebounds','assists','threes'))
        AND (commence_time_utc AT TIME ZONE 'America/New_York')::date BETWEEN %s AND %s""",(since,through)):
        add('Sharp-price game selections',f"shared:{r['decision_id']}",r,r['experiment_id'])
    for r in data.rows(conn,"""SELECT * FROM bets.nfl_sharp_alerts WHERE posted
        AND (commence_time_utc AT TIME ZONE 'America/New_York')::date BETWEEN %s AND %s""",(since,through)):
        o=dict(r,sport='nfl',player=r['player_name'],market=r['stat'],line=r['fd_line'],price=r['fd_price'],decided_at=r['alerted_at'],source_status='posted paper alert; not a wager receipt')
        add('Delivered price alerts',f"NFL sharp #{r['alert_id']}",o,'NFL sharp alerts (historical versions)')
    for r in data.rows(conn,"""SELECT l.*,COALESCE(p.game_id,g.game_id) game_id,p.player_name,
        COALESCE(p.created_at_utc,g.created_at_utc) forecast_time,p.projection,
        g.predicted_home_margin,g.predicted_total_points,n.start_ts_utc,n.home_team_abbr,n.away_team_abbr
        FROM bets.nfl_bet_ledger l LEFT JOIN bets.nfl_player_prop_predictions p ON l.source_kind='prop' AND p.id=l.prediction_id
        LEFT JOIN bets.nfl_game_predictions g ON l.source_kind='game' AND g.id=l.prediction_id
        LEFT JOIN raw.nfl_games n ON n.game_id=COALESCE(p.game_id,g.game_id)
        WHERE l.game_date_et BETWEEN %s AND %s""",(since,through)):
        market=r['stat'] if r['source_kind']=='prop' else r['market']
        pred=r['projection'] if r['source_kind']=='prop' else r['predicted_home_margin'] if market=='spread' else r['predicted_total_points']
        o=dict(r,sport='nfl',event_id=r['game_id'],player=r['player_name'],market=market,
            home_team=r['home_team_abbr'],away_team=r['away_team_abbr'],decided_at=r['locked_at_utc'],commence_time_utc=r['start_ts_utc'])
        reason='legacy_stale_card_record' if str(r['result']).startswith('voided_') else None
        add('Locked model picks / '+r['tier'],f"NFL lock #{r['ledger_id']}",o,r['model_version'],prediction=pred,reason=reason)
    for table,kind in (('bets.model_research_decisions','assisted'),('bets.edge_market_decisions','edge')):
        found=data.rows(conn,f"""SELECT * FROM {table} WHERE (payload->'offer'->>'commence_time_utc')::timestamptz
            >=%s AND (payload->'offer'->>'commence_time_utc')::timestamptz<%s""",
            (datetime.combine(since,datetime.min.time(),ET),datetime.combine(through+timedelta(days=1),datetime.min.time(),ET)))
        for r in found:
            o=r['payload']['offer'];d=r['payload']['decision'];f=d.get('forecast') or (d.get('native') or {}).get('forecast') or {}
            pred=f.get('pred_margin') if o['market']=='spread' else f.get('pred_total') if o['market']=='total' else f.get('mu',f.get('projection'))
            if kind=='assisted':
                if d['selected']:add('Model-assisted paper selections',f"assisted:{r['id']}",o,r['experiment_id'],prediction=pred,probability=d.get('model_probability'))
                continue
            for label,s in d['selection'].items():
                if not s.get('selected'):continue
                variant=dict(o,**{k:s[k] for k in ('side','line','price')})
                model=(d.get('workload') or {}).get('mu') if 'rotation' in label else pred if label!='consensus' else None
                add('Consensus trial / '+label,f"edge:{r['id']}:{label}",variant,r['experiment_sha256'][:12],prediction=model,probability=s['probability'])
            # A diagnostic directional pick uses the saved point prediction on the same offer.
            # It is not a retrospective claim that the system recommended a wager.
            if number(pred) is not None:
                delta=number(pred)+float(o['line']) if o['market']=='spread' else number(pred)-float(o['line'])
                side=('home' if delta>0 else 'away') if o['market']=='spread' else ('over' if delta>0 else 'under')
                v=dict(o,side=side if delta else None,line=-float(o['line']) if side=='away' else o['line'],price=None,
                    source_status='derived from saved pre-offer forecast; not a betting instruction')
                add('Model direction on paired lines',f"edge:{r['id']}:direction",v,r['experiment_sha256'][:12],prediction=pred)
    # Exact stat forecasts have no W/L label without an offered line.
    forecasts=data.rows(conn,"""SELECT DISTINCT ON(game_id,player_id,stat) * FROM nbaprops.projections
        WHERE game_date BETWEEN %s AND %s AND run_at_utc<tip_utc
        ORDER BY game_id,player_id,stat,run_at_utc DESC""",(since,through))
    games=data.rows(conn,"""SELECT DISTINCT ON(game_id) * FROM nbaprops.game_projections
        WHERE game_date BETWEEN %s AND %s AND run_at_utc<tip_utc ORDER BY game_id,run_at_utc DESC""",(since,through))
    game_map={g['game_id']:g for g in games}
    for p in forecasts:
        g=game_map.get(p['game_id'])
        if not g:continue
        o=dict(p,sport='nba',event_id=p['game_id'],player=p['player_name'],market=next((k for k,v in STATS.items() if v==p['stat']),p['stat']),
            home_team=g['home_team'],away_team=g['away_team'],commence_time_utc=p['tip_utc'],decided_at=p['run_at_utc'],source_status='last saved pregame stat forecast')
        add('Stat forecast accuracy',f"NBA forecast:{p['game_id']}:{p['player_id']}:{p['stat']}",o,p['model_version'],prediction=p['mu'])
    for g in games:
        margin=number(g['pred_margin'])
        o=dict(g,sport='nba',event_id=g['game_id'],market='winner',side='home' if margin>0 else 'away' if margin<0 else None,
            line=0,decided_at=g['run_at_utc'],commence_time_utc=g['tip_utc'],source_status='winner derived from last saved pregame score; not a wager')
        add('Game winner forecasts',f"NBA winner:{g['game_id']}",o,g['model_version'],prediction=margin)
    for g in data.rows(conn,"""SELECT DISTINCT ON(game_id) * FROM ncaaf.projections
        WHERE (start_date AT TIME ZONE 'America/New_York')::date BETWEEN %s AND %s
        AND generated_at_utc<start_date ORDER BY game_id,generated_at_utc DESC""",(since,through)):
        for market in ('spread','total'):
            pred=number(g['pred_margin' if market=='spread' else 'pred_total']);line=number(g['market_'+market])
            delta=None if pred is None or line is None else pred+line if market=='spread' else pred-line
            side=None if delta is None or delta==0 else ('home' if delta>0 else 'away') if market=='spread' else ('over' if delta>0 else 'under')
            o=dict(g,sport='ncaaf',event_id=g['game_id'],market=market,side=side,line=-line if side=='away' else line,
                book=g['market_provider'],commence_time_utc=g['start_date'],decided_at=g['generated_at_utc'])
            add('Game model vs saved line',f"NCAA forecast:{g['game_id']}:{market}",o,'saved NCAA projection',prediction=pred)
    return first_decisions(output)


def report(conn,since,through):
    results=Results(conn,since,through);records=load_rows(conn,since,through,results)
    records=[r for r in records if str(since)<=r['date']<=str(through)]
    records=sorted(records,key=lambda r:(r['date'],r['forecast_at'] or '',r['id']),reverse=True)
    groups=defaultdict(list)
    for r in records:groups[(r['cohort'],r['sport'],r['book'],r['market'],r['version'])].append(r)
    summaries=[dict(cohort=k[0],sport=k[1],book=k[2],market=k[3],version=k[4],**summary(v)) for k,v in sorted(groups.items())]
    return dict(built_at=datetime.now(timezone.utc).isoformat(),since=str(since),through=str(through),
        records=records,summaries=summaries,source_health=dict(results.health),
        notes=['Right/wrong means the recorded side at its exact line; pushes/voids/pending/excluded do not enter hit rate.',
            'Strategy alternatives overlap; there is no pooled grand-total record. Sportsbooks and model versions stay separate.',
            'Latest pregame stat forecasts without lines use prediction error, not win/loss.',
            'All dollar/unit figures are hypothetical $1 paper stakes, not actual wagers or net profit.',
            'Dates are game dates in Eastern time. This ledger reads existing verified results; missing evidence remains pending.',
            'Coverage: active NBA, NFL and NCAA pipelines. Legacy MLB forecasts are not included.'],
        betting_authorized=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def write_reports(doc,folder):
    folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'predictions_ledger_latest.json',doc)
    template=(ROOT/'src/prediction_ledger.html').read_text(encoding='utf-8')
    encoded=json.dumps(doc,ensure_ascii=True,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    html=template.replace('LEDGER_DATA_JSON',encoded)
    temp=folder/'predictions_ledger_latest.html.tmp';temp.write_text(html,encoding='utf-8');temp.replace(folder/'predictions_ledger_latest.html')
    lines=['# Predictions ledger','',f"Updated {doc['built_at']}. Games {doc['since']} through {doc['through']}.",'',
        '[Open the filterable ledger](predictions_ledger_latest.html)','',*['- '+n for n in doc['notes']],'',
        '| View | Sport | Book | Market | Version | Right | Wrong | Push | Void | Pending | Excluded | No line | Hit rate | MAE |',
        '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in doc['summaries']:
        cells=[s[k] for k in ('cohort','sport','book','market','version','win','loss','push','void','pending','excluded','no_line')]
        cells += [f"{s['accuracy']:.1%}" if s['accuracy'] is not None else '—',f"{s['mae']:.2f}" if s['mae'] is not None else '—']
        lines.append('| '+' | '.join(str(c).replace('|','\\|') for c in cells)+' |')
    (folder/'predictions_ledger_latest.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--since',type=date.fromisoformat,default=date(2026,9,1))
    ap.add_argument('--through',type=date.fromisoformat,default=datetime.now(ET).date())
    args=ap.parse_args()
    if args.since>args.through:ap.error('--since must not be after --through')
    with closing(psycopg2.connect(PG_DSN,connect_timeout=10)) as conn:
        conn.set_session(readonly=True,isolation_level='REPEATABLE READ')
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout='90s'")
            cur.execute('SELECT pg_try_advisory_lock(86731445)')
            if not cur.fetchone()[0]:print('Ledger refresh already running');return
        doc=report(conn,args.since,args.through)
        write_reports(doc,ROOT/'reports')
    print(json.dumps(dict(status='ok',rows=len(doc['records']),groups=len(doc['summaries']),
        since=doc['since'],through=doc['through'],report='reports/predictions_ledger_latest.html')))


if __name__=='__main__':main()
