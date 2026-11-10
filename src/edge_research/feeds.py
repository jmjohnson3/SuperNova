"""Read existing observed quotes and forecasts. No new paid API requests."""
from __future__ import annotations
from collections import defaultdict
from datetime import timedelta,timezone
from zoneinfo import ZoneInfo

from scipy import stats

import model_research as parent
from model_research_report import verified_result
from nba_props import research_data as data,research_trial as frozen,research_evaluation as evaluation
from nfl_pipeline.context_contract import safe_time,number
from nfl_pipeline import experiment_registry as registry
from sharp_pipeline.outcomes import match_school
from edge_research import market

SUPPORTED=('nba','nfl','ncaaf')
NFL_MARKETS={v:k for k,v in market.MARKETS.items() if k.startswith('player_') and v not in evaluation.STAT_MAP}


def snapshots(conn,sport,after,before):
    if sport in ('nba','ncaaf'):
        records=data.rows(conn,"""SELECT observation_id,observed_at_utc,payload FROM bets.sharp_multi_quotes
            WHERE sport=%s AND observed_at_utc BETWEEN %s AND %s ORDER BY observed_at_utc,observation_id""",(sport,after,before))
        from sharp_pipeline.scan import _digest
        return [dict(id=r['observation_id'],at=r['observed_at_utc'],event=r['payload']) for r in records
            if r['observation_id']==_digest(dict(sport=sport,event_id=r['payload'].get('id'),
                observed_at_utc=r['observed_at_utc'].astimezone(timezone.utc).isoformat(),event=r['payload']))]
    if sport!='nfl':raise ValueError('No verified live quote adapter for '+sport)
    records=data.rows(conn,"""SELECT payload FROM bets.nfl_sharp_quote_observations
        WHERE observed_at_utc BETWEEN %s AND %s ORDER BY observed_at_utc,observation_id""",(after,before))
    groups=defaultdict(list)
    for r in records:
        p=r['payload']
        if p.get('observation_id')!=registry._digest({k:v for k,v in p.items() if k!='observation_id'}):continue
        groups[(p['game_id'],p['event_id'],p['observed_at_utc'],p['commence_time_utc'])].append(p)
    result=[]
    for (gid,eid,at,tip),parts in groups.items():
        games=data.rows(conn,'SELECT home_team_abbr,away_team_abbr FROM raw.nfl_games WHERE game_id=%s AND start_ts_utc=%s',(gid,tip))
        if len(games)!=1:continue
        game=games[0];books=defaultdict(list)
        for p in parts:
            if p.get('validation_error'):continue
            q=p['quote'];stat=p['stat'];line=number(p.get('line'));who=p.get('player_name_norm')
            key='spreads' if stat=='spread' else 'totals' if stat=='total' else NFL_MARKETS.get(stat)
            if not key or line is None:continue
            if stat=='spread':
                if who not in (game['home_team_abbr'],game['away_team_abbr']):continue
                outcomes=[dict(name=who,price=q.get('price'),point=line)]
            else:
                outcomes=[dict(name=side,price=q.get(side.lower()),point=line,description=who) for side in ('Over','Under')]
            books[p['book']].append(dict(key=key,last_update=q.get('market_updated_at'),outcomes=outcomes,
                                         _book_updated=q.get('book_updated_at')))
        # Individual NFL records carry their original book and market clocks.
        event=dict(id=eid,game_id=gid,commence_time=tip,home_team=game['home_team_abbr'],away_team=game['away_team_abbr'],bookmakers=[])
        for book,markets in books.items():
            for m in markets:event['bookmakers'].append(dict(key=book,last_update=m.pop('_book_updated'),markets=[m]))
        result.append(dict(id=registry._digest([p['observation_id'] for p in parts]),at=safe_time(at),event=event))
    return sorted(result,key=lambda r:(r['at'],r['id']))


def canonical_offer(pair):
    """Shape expected by existing immutable forecast and result lookup helpers."""
    return dict(pair,side='home' if pair['market']=='spread' else 'over',price=pair['price_a'],
        fair_probability=(pair.get('consensus') or {}).get('probability'),method='exact')


def native_forecast(conn,pair,artifact):
    o=canonical_offer(pair);sport=o['sport']
    if sport in ('nba','nfl'):
        if sport=='nfl' and o['market'] not in parent.NFL_STATS|{'spread','total'}:return dict(probability=None,reason='unsupported_model_market')
        f=parent.forecast_for(conn,o)
        if not f:return dict(probability=None,forecast=None,reason='no_prior_native_forecast')
        assessed=parent.assess(o,f)
        p=assessed.get('model_probability')
        if o['market'] in ('spread','total'):
            stat='margin' if o['market']=='spread' else 'total'
            err=artifact.get('game_error',{}).get(sport,{}).get(stat)
            if err and err.get('sd',0)>0:
                mu=number(f.get('pred_'+stat));threshold=-float(o['line']) if stat=='margin' else float(o['line'])
                if mu is not None:p=float(stats.norm.sf(threshold,mu-err.get('bias',0),err['sd']))
        return dict(probability=p,forecast=f,context_verified=f.get('available') is True if o['market'] not in ('spread','total') else True,
            reason=assessed['reason'])
    at=safe_time(o['decided_at']);tip=safe_time(o['commence_time_utc'])
    found=data.rows(conn,"""SELECT * FROM ncaaf.projections WHERE generated_at_utc<=%s
        AND generated_at_utc>=%s-interval '24 hours' AND start_date BETWEEN %s-interval '15 minutes' AND %s+interval '15 minutes'
        ORDER BY generated_at_utc DESC""",(at,at,tip,tip))
    home=match_school(o['home_team'],[r['home_team'] for r in found]);away=match_school(o['away_team'],[r['away_team'] for r in found])
    matches=[r for r in found if r['home_team']==home and r['away_team']==away]
    if not matches or len({r['game_id'] for r in matches})!=1:return dict(probability=None,forecast=None,reason='no_unambiguous_prior_native_forecast')
    f=matches[0];stat='margin' if o['market']=='spread' else 'total';error=artifact.get('game_error',{}).get(sport,{}).get(stat)
    p=None
    if error and error.get('sd',0)>0:
        p=float(stats.norm.sf(-o['line'] if stat=='margin' else o['line'],float(f['pred_'+stat])-error.get('bias',0),error['sd']))
    return dict(probability=p,forecast=f,context_verified=True,reason='game_error_unfitted' if p is None else 'prior_game_forecast')


def nba_result_index(conn,asof):
    records=data.rows(conn,"""SELECT DISTINCT ON(game_id) payload FROM nbaprops.research_games
        WHERE observed_at<=%s ORDER BY game_id,observed_at DESC,snapshot_id""",(asof,))
    result=defaultdict(list)
    for row in records:
        g=row['payload'];home=next(t['team'] for t in g['teams'] if t['home']);away=next(t['team'] for t in g['teams'] if not t['home'])
        result[(g['game_date'],home,away)].append(g)
    return result


def outcome(conn,pair,forecast,index=None):
    """Positive, verified participation required for prop grading. Missing remains pending."""
    o=canonical_offer(pair);sport=o['sport'];day=safe_time(o['commence_time_utc']).astimezone(ZoneInfo('America/New_York')).date()
    if sport=='nba':
        from nba_props.model import match_name
        found=(index or {}).get((day.isoformat(),parent.nba_team(o['home_team']),parent.nba_team(o['away_team'])),[])
        if len(found)!=1:return dict(outcome=None,actual=None,result_status='pending_final_identity')
        g=found[0]
        if o['market'] in evaluation.STAT_MAP:
            people=[p for p in g['players'] if match_name(p['player_name'])==match_name(o['player']) and p['minutes'] is not None and p['minutes']>0]
            if len(people)!=1:return dict(outcome=None,actual=None,result_status='pending_participation')
            actual=number(people[0].get(evaluation.STAT_MAP[o['market']]))
        else:
            home=next(t['pts'] for t in g['teams'] if t['home']);away=next(t['pts'] for t in g['teams'] if not t['home'])
            actual=home-away if o['market']=='spread' else home+away
        if actual is None:return dict(outcome=None,actual=None,result_status='pending_stat')
        delta=actual+float(o['line']) if o['market']=='spread' else actual-float(o['line'])
        return dict(outcome=float(delta>0),actual=actual,result_status='settled')
    if sport=='nfl':
        result=verified_result(conn,o,forecast)
        actual=None
        if result in ('win','loss'):
            if o['market'] in ('spread','total'):
                found=data.rows(conn,"SELECT home_score,away_score FROM raw.nfl_games WHERE game_id=%s AND status='final'",(o['game_id'],))
                if len(found)==1:
                    g=found[0];actual=g['home_score']-g['away_score'] if o['market']=='spread' else g['home_score']+g['away_score']
            elif forecast and forecast.get('id'):
                found=data.rows(conn,'SELECT actual_stat FROM bets.nfl_player_prop_prediction_results WHERE prediction_id=%s',(forecast['id'],))
                if len(found)==1:actual=number(found[0]['actual_stat'])
        return dict(outcome=1. if result=='win' else 0. if result=='loss' else None,actual=actual,result_status=result or 'pending_official_result')
    games=data.rows(conn,"""SELECT * FROM ncaaf.games WHERE completed AND start_date BETWEEN %s-interval '15 minutes' AND %s+interval '15 minutes'""",
        (o['commence_time_utc'],o['commence_time_utc']))
    home=match_school(o['home_team'],[r['home_team'] for r in games]);away=match_school(o['away_team'],[r['away_team'] for r in games])
    matches=[g for g in games if g['home_team']==home and g['away_team']==away]
    if len(matches)!=1 or any(matches[0].get(k) is None for k in ('home_points','away_points')):
        return dict(outcome=None,actual=None,result_status='pending_official_result')
    g=matches[0];actual=g['home_points']-g['away_points'] if o['market']=='spread' else g['home_points']+g['away_points']
    return dict(outcome=float(actual+o['line']>0) if o['market']=='spread' else float(actual>o['line']),actual=actual,result_status='settled')
