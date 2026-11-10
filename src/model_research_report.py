"""Paired prospective comparisons, execution diagnostics and cost-aware paper economics."""
from __future__ import annotations
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import math
import numpy as np
import pandas as pd
from model_research import (ROOT,NAME,CONFIG,NBA_STATS,rows,number,nba_team,clean,safe_time,atomic_json)
from nfl_pipeline import sharp_math
from sharp_pipeline.nba_experiment import check_evidence,clustered_interval


def exact_close(conn,offer):
    start=safe_time(offer['commence_time_utc']); observed=safe_time(offer['decided_at'])
    if offer['sport']=='nba':
        from sharp_pipeline import report
        found=rows(conn,"""SELECT observed_at_utc,payload FROM bets.sharp_multi_quotes WHERE sport='nba'
            AND event_id=%s AND observed_at_utc>=%s AND observed_at_utc<%s
            AND observed_at_utc>=%s-interval '22 minutes' ORDER BY observed_at_utc DESC""",
            (offer['event_id'],observed,start,start))
        kind=report.row_kind(offer)
        # Match the existing policy: last observation of this market family, never an inferred conversion.
        for r in found:
            if report.payload_kind(r['payload'])!=kind:continue
            evidence=report.close_evidence(offer,{('nba',offer['event_id'],kind):(r['observed_at_utc'],r['payload'])})
            return evidence if evidence.get('method')=='exact' else {}
        return {}
    from nfl_pipeline import sharp_watch
    from model_research import spread_pair,registry
    found=rows(conn,"""SELECT payload FROM bets.nfl_sharp_quote_observations WHERE game_id=%s
        AND observed_at_utc>=%s AND observed_at_utc<%s AND observed_at_utc>=%s-interval '22 minutes'
        ORDER BY observed_at_utc DESC""",(offer['game_id'],observed,start,start))
    if offer['market']=='spread':
        groups={}
        for r in found:
            p=r['payload']
            if p.get('event_id')==offer['event_id'] and p.get('stat')=='spread' and safe_time(p.get('commence_time_utc'))==start:
                groups.setdefault(p['observed_at_utc'],[]).append(p)
        home_line=float(offer['line'])*(1 if offer['side']=='home' else -1)
        for at,quotes in sorted(groups.items(),reverse=True):
            pair=spread_pair(quotes,'pinnacle',offer['home_team'],offer['away_team'],home_line,safe_time(at))
            if not pair:continue
            p=sharp_math.no_vig_over(pair['home_price'],pair['away_price'])
            probability=p if offer['side']=='home' else 1-p
            return dict(method='exact',probability=probability,ev=sharp_math.ev(probability,offer['price']))
        return {}
    for r in found:
        p=r['payload'];q=p.get('quote') or {}; at=safe_time(p.get('observed_at_utc'))
        if (p.get('book')!='pinnacle' or p.get('event_id')!=offer['event_id']
            or p.get('observation_id')!=registry._digest({k:v for k,v in p.items() if k!='observation_id'})
            or p.get('stat')!=offer['market'] or (p.get('player_name_norm') or '')!=offer['player_norm']
            or number(p.get('line'))!=float(offer['line']) or safe_time(p.get('commence_time_utc'))!=start
            or sharp_watch.quote_error(q,at)): continue
        fair=sharp_math.no_vig_over(q['over'],q['under'])
        probability=fair if offer['side']=='over' else 1-fair
        return dict(method='exact',probability=probability,ev=sharp_math.ev(probability,offer['price']))
    return {}


def verified_result(conn,offer,forecast):
    """Official final game and verified player participation are required; absence stays pending."""
    from sharp_pipeline.outcomes import grade
    if offer['sport']=='nba':
        from nba_props.model import match_name
        from nba_props.ingest import conflicted_games
        from model_research_results import lookup
        home,away=nba_team(offer['home_team']),nba_team(offer['away_team'])
        game=lookup(conn,offer)
        if not game:return None
        actual=None
        if offer['market'] in NBA_STATS:
            found=rows(conn,"""SELECT * FROM nbaprops.player_games WHERE game_id=%s
                AND ((team=%s AND opponent=%s) OR (team=%s AND opponent=%s))""",
                (game['game_id'],home,away,away,home))
            if any(abs(sum(float(p['pts'] or 0) for p in found if p['team']==team)-game[side+'_score'])>.01
                   for team,side in ((home,'home'),(away,'away'))):return None
            bad=set(conflicted_games(conn))
            found=[p for p in found if p['game_id'] not in bad and match_name(p['player_name'])==match_name(offer['player'])]
            if len(found)!=1 or number(found[0]['minutes']) is None or float(found[0]['minutes'])<=0:return None
            actual=number(found[0][NBA_STATS[offer['market']]])
        return grade(offer,actual,float(game['home_score']),float(game['away_score']))
    games=rows(conn,"SELECT * FROM raw.nfl_games WHERE game_id=%s AND status='final'",(offer['game_id'],))
    if not games:return None
    if offer['market'] in {'spread','total'}:
        game=games[0]
        hp,ap=number(game.get('home_score')),number(game.get('away_score'))
        return grade(offer,None,hp,ap)
    if not forecast or not forecast.get('id'):return None
    found=rows(conn,"SELECT actual_stat,result FROM bets.nfl_player_prop_prediction_results WHERE prediction_id=%s",(forecast['id'],))
    if not found or found[0]['result'] not in {'win','loss','push'}:return None
    actual=number(found[0]['actual_stat'])
    if actual is None:return None
    diff=(actual-float(offer['line']))*(1 if offer['side']=='over' else -1)
    return 'win' if diff>0 else 'loss' if diff<0 else 'push'


def delay(end,start):
    a,b=safe_time(end),safe_time(start)
    return (a-b).total_seconds() if a and b else None


def diagnostics(frame):
    def median(key):
        s=pd.to_numeric(frame.get(key,pd.Series(dtype=float)),errors='coerce').dropna()
        return float(s.median()) if len(s) else None
    return dict(median_execution_quote_age_seconds=median('execution_age'),
        median_reference_quote_age_seconds=median('reference_age'),
        median_observation_to_delivery_seconds=median('delivery_delay'),
        median_delivery_to_check_seconds=median('check_delay_seconds'),
        median_check_to_record_seconds=median('recording_lag_seconds'),
        median_reference_revision_ev=median('reference_revision_ev'),
        median_price_slippage_ev=median('price_slippage_ev'),
        missing_checks=int(frame.checked_at.isna().sum()),
        late_checks=int((frame.checked_at.notna() & ~frame.timely_check).sum()),
        disappeared=int(frame.check_status.eq('disappeared').sum()),
        changed_line=int(frame.check_status.eq('changed_line').sum()),
        note='Timing and EV changes diagnose losses; no causal claim from a small sample.')


def strategy_summary(frame,sport):
    rule=CONFIG[sport+'_review']
    if frame.empty:return dict(decisions=0,status='collecting',betting_authorized=False)
    n=len(frame); exact=frame.close_method.eq('exact'); timely=frame.timely_check
    available=frame.exact_offer_available; priced=frame.checked_price_ev.notna()
    span=(max(frame.day)-min(frame.day)).days+1
    intervals={label:clustered_interval(frame,value,cluster) for label,value,cluster in
        [('alert_game','close_ev','event_id'),('alert_day','close_ev','day'),
         ('checked_game','checked_price_ev','event_id'),('checked_day','checked_price_ev','day')]}
    mean=pd.to_numeric(frame.close_ev,errors='coerce').mean()
    gates=dict(days=span>=rule['min_days'],games=frame.event_id.nunique()>=rule['min_games'],
        checked_closes=int(priced.sum())>=rule['min_checked_closes'],exact_closes=exact.mean()>=rule['min_exact_coverage'],
        check_coverage=timely.mean()>=rule['min_check_coverage'],
        availability=bool(timely.any() and available.sum()/timely.sum()>=rule['min_available']),
        checked_close_coverage=bool(available.any() and priced.sum()/available.sum()>=.9),
        active_dates=frame.day.nunique()>=7 if sport=='nba' else frame.day.nunique()>=3,
        weeks=len({(d-min(frame.day)).days//7 for d in frame.day})>=3,
        alert_close_edge=bool(pd.notna(mean) and mean>=rule['min_alert_close_ev']))
    for k,v in intervals.items():gates[k]=v['lower_95'] is not None and v['lower_95']>0
    settled=frame.result.isin(['win','loss','push'])
    return clean(dict(decisions=n,games=int(frame.event_id.nunique()),span_days=span,
        exact_close_coverage=float(exact.mean()),timely_check_coverage=float(timely.mean()),
        checked_prices=int(priced.sum()),mean_alert_close_ev=mean,
        mean_checked_price_ev=pd.to_numeric(frame.checked_price_ev,errors='coerce').mean(),
        intervals=intervals,gates={k:bool(v) for k,v in gates.items()},
        status='review_only' if all(gates.values()) else 'collecting',betting_authorized=False,
        settled=int(settled.sum()),paper_units=float(frame.loc[settled,'paper_profit'].sum()),
        pending_results=int((~settled).sum())))


def calibration(frame):
    binary=frame[frame.result.isin(['win','loss']) & frame.model_probability.notna()].copy()
    if binary.empty:return dict(settled_common_offers=0,brier_gain=None,bins=[])
    binary['outcome']=binary.result.eq('win').astype(float)
    binary['gain']=(binary.fair_probability-binary.outcome)**2-(binary.model_probability-binary.outcome)**2
    return dict(settled_common_offers=len(binary),brier_gain=float(binary.gain.mean()),
        gain_by_game=clustered_interval(binary,'gain','event_id'),gain_by_day=clustered_interval(binary,'gain','day'),
        bins=[dict(lower=lo,upper=lo+.1,n=len(g),model_mean=float(g.model_probability.mean()),
                   reference_mean=float(g.fair_probability.mean()),win_rate=float(g.outcome.mean()))
              for lo in np.arange(0,1,.1) if len(g:=binary[(binary.model_probability>=lo)&(binary.model_probability<lo+.1)])])


def economics(frame,collection_days=None):
    """Use $1 comparison stakes and count $30 once. Unknown checks never become fills."""
    empty=dict(monthly_cost_usd='30.00',comparison_stake_usd='1.00',projected_net_usd=None,
               basis='paper checks valued at reference close; not actual or guaranteed profit')
    if frame.empty:return dict(empty,status='no_prospective_evidence')
    span=collection_days or (max(frame.day)-min(frame.day)).days+1
    selected=frame[frame.selected]
    known=selected.opportunity_ev.notna()
    if span<21 or selected.empty or not known.all():
        return dict(empty,status='insufficient_span_or_unknown_execution',observed_days=span,
                    selected=len(selected),unknown_execution=int((~known).sum()))
    # Include unavailable checked offers as zero opportunities and all event dates,
    # not just the dates of successful price checks.
    frame=frame.copy();frame['model_opportunity']=np.where(frame.selected,frame.opportunity_ev,0.)
    per_day=frame.groupby('day').model_opportunity.sum()
    rate=float(per_day.sum()/span*30)
    intervals={label:clustered_interval(frame,'model_opportunity',cluster)
               for label,cluster in (('game','event_id'),('day','day'))}
    lows=[v['lower_95'] for v in intervals.values()]
    conservative_net=min(lows)*len(frame)/span*30-30 if all(v is not None for v in lows) else None
    return dict(empty,status='scenario_only',observed_days=span,selected=len(selected),
        observed_opportunity_units=float(per_day.sum()),projected_monthly_gross_usd=rate,
        projected_net_usd=rate-30,conservative_monthly_net_usd=conservative_net,
        clears_cost_review=conservative_net is not None and conservative_net>0,
        assumes='Same future opportunity rate and reference accuracy; no larger stakes; includes missed availability as zero.',
        uncertainty_by_game=intervals['game'],uncertainty_by_day=intervals['day'])


def summarize(frame,collection_days=None):
    groups={}
    if not frame.empty:
        for (sport,family),part in frame.groupby(['sport','family']):
            paired=part.copy()
            paired['model_minus_control']=np.where(paired.selected,0.,-pd.to_numeric(paired.opportunity_ev,errors='coerce'))
            groups[sport+'_'+family]=dict(price_control=strategy_summary(part,sport),
                model_assisted=strategy_summary(part[part.selected],sport),
                veto_reasons=part.reason.value_counts().to_dict(),diagnostics=diagnostics(part),
                paired_improvement_by_game=clustered_interval(paired,'model_minus_control','event_id'),
                paired_improvement_by_day=clustered_interval(paired,'model_minus_control','day'),
                probability_calibration=calibration(part),selected_calibration=calibration(part[part.selected]))
            g=groups[sport+'_'+family]
            g['paired_execution_coverage']=float(pd.to_numeric(part.opportunity_ev,errors='coerce').notna().mean())
            improvement=g['paired_execution_coverage']>=.9 and all(g[k]['lower_95'] is not None and g[k]['lower_95']>0
                for k in ('paired_improvement_by_game','paired_improvement_by_day'))
            cal=g['probability_calibration']
            probability_ok=family=='games' or all(cal.get(k,{}).get('lower_95') is not None
                and cal[k]['lower_95']>0 for k in ('gain_by_game','gain_by_day'))
            g['model_adds_value_review']=improvement and probability_ok and g['model_assisted']['status']=='review_only'
    return clean(dict(groups=groups,economics=economics(frame,collection_days),betting_authorized=False,
        note='Paired control is this experiment\'s identical offers, not a replacement for older frozen evidence. Pinnacle is a benchmark, not known truth.'))


def build_report(conn,doc):
    saved=rows(conn,"""SELECT d.*,m.delivery_status,m.sent_at,c.checked_at,c.recorded_at,c.status AS check_status,
        c.displayed_line,c.displayed_price FROM bets.model_research_decisions d
        LEFT JOIN bets.model_research_delivery m USING(id) LEFT JOIN bets.model_research_checks c USING(id)
        WHERE experiment_id=%s AND experiment_sha256=%s ORDER BY d.id""",(NAME,doc['sha256']))
    current=datetime.now(timezone.utc);data=[];upcoming=[]
    for s in saved:
        offer=s['payload']['offer'];decision=s['payload']['decision']
        if safe_time(offer['commence_time_utc'])>current:
            upcoming.append(dict(id=s['id'],offer=offer,decision=decision));continue
        parent=offer.get('parent_check_id')
        if parent:
            found=rows(conn,"""SELECT a.delivery_status,a.sent_at,c.checked_at,c.recorded_at,c.status AS check_status,
                c.displayed_line,c.displayed_price FROM bets.nba_paper_price_alerts a
                LEFT JOIN bets.nba_paper_price_checks c USING(alert_id) WHERE a.alert_id=%s""",(parent,))
            if found:s.update(found[0])
        close=exact_close(conn,offer)
        row=dict(offer,**{k:s.get(k) for k in ('delivery_status','sent_at','checked_at','recorded_at','check_status','displayed_line','displayed_price')},
            close_probability=close.get('probability'),close_method=close.get('method'),evidence_source='manual_live')
        check=check_evidence(row)
        result=verified_result(conn,offer,decision.get('forecast'))
        payout=100/abs(offer['price']) if offer['price']<0 else offer['price']/100
        reference_ev=close.get('ev')
        data.append(dict(row,**check,selected=decision['selected'],reason=decision['reason'],
            family='games' if offer['market'] in {'spread','total'} else 'props',
            day=safe_time(offer['commence_time_utc']).astimezone(ZoneInfo('America/New_York')).date(),model_probability=decision.get('model_probability'),
            close_ev=reference_ev,result=result,paper_profit=payout if result=='win' else -1 if result=='loss' else 0 if result=='push' else None,
            execution_age=delay(offer['decided_at'],offer.get('execution_updated_at')),
            reference_age=delay(offer['decided_at'],offer.get('reference_updated_at')),
            delivery_delay=delay(s.get('sent_at'),offer['decided_at']),
            reference_revision_ev=reference_ev-float(offer['ev']) if reference_ev is not None else None,
            price_slippage_ev=check['checked_price_ev']-reference_ev if check['checked_price_ev'] is not None else None))
    frame=pd.DataFrame(data)
    collection_days=max(1,(current-safe_time(doc['registered_at'])).total_seconds()/86400)
    result=dict(built_at=current.isoformat(),experiment_id=NAME,sha256=doc['sha256'],registered_at=doc['registered_at'],
                captured=len(saved),upcoming=upcoming,**summarize(frame,collection_days))
    atomic_json(ROOT/'reports/model_research_latest.json',result)
    text=['# FanDuel model-assisted paper decisions','',f"Experiment: {NAME}",
          f"Captured: {len(saved)}; awaiting game start: {len(upcoming)}.",
          'Player probabilities select or veto props; game projections select or veto spreads/totals. No wagering authority.','',
          '| Sport / family | Price control | Model selected | Selected checked prices | Selected exact closes | Model adds value review |',
          '|---|---:|---:|---:|---:|---|']
    for key,g in result['groups'].items():
        m=g['model_assisted']
        text.append(f"| {key} | {g['price_control']['decisions']} | {m['decisions']} | {m.get('checked_prices',0)} | {m.get('exact_close_coverage','unknown')} | {g['model_adds_value_review']} |")
    text+=['',f"Economics: {result['economics']['status']}; monthly data cost $30, comparison stake $1.",
           f"Projected monthly net: {result['economics']['projected_net_usd'] if result['economics']['projected_net_usd'] is not None else 'unknown'}.",
           'These are paper scenarios, not actual profit. Unknown availability remains unknown.','',
           '## Upcoming comparisons','']
    for r in upcoming:
        o,d=r['offer'],r['decision'];f=d.get('forecast') or {}
        text.append(f"- M{r['id']}: {o['sport']} {o.get('player') or o['event_id']} {o['side']} {o['line']} {o['market']} {o['price']:+d}: {'selected' if d['selected'] else 'veto'} ({d['reason']}); forecast {f.get('run_at_utc','missing')}.")
    text+=['','Full quote-to-alert/check delays, slippage, paired intervals and calibration are in the JSON report.',
           'Existing frozen price-only and receiving experiments are unchanged.']
    (ROOT/'reports/model_research_latest.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return result
