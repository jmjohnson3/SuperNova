"""Paired outcomes and observed-price evidence; automated quotes never become fills."""
from __future__ import annotations
from collections import Counter
from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from edge_research import market,feeds,trial
from nba_props import research_data as data,research_trial as frozen
from nfl_pipeline.context_contract import safe_time,number
from nfl_pipeline import sharp_math
from nfl_pipeline.integrity import atomic_json
from sharp_pipeline.nba_experiment import check_evidence,clustered_interval


def exact_close(conn,o,schedule,cache):
    tip=safe_time(schedule['tip']);key=(o['sport'],tip)
    if key not in cache:cache[key]=feeds.snapshots(conn,o['sport'],tip-timedelta(minutes=22),tip-timedelta(microseconds=1))
    packets=[p for p in cache[key] if p['event']['id']==o['event_id'] and p['at']>=safe_time(o['decided_at'])]
    target=(o['market'],o['player_norm'],float(o['line']))
    for p in reversed(packets):
        if (p['event'].get('home_team')!=o['home_team'] or p['event'].get('away_team')!=o['away_team']
            or safe_time(p['event'].get('commence_time'))!=safe_time(o['commence_time_utc'])):continue
        pairs=market.event_pairs(p['event'],p['at'])
        if not any(k[1]==o['market'] for k in pairs):continue
        ref=market.consensus(pairs,target)
        # Never reach back past a newer observation of the same market to rescue a moved line.
        if ref:return dict(probability=ref['probability'],method='observed_exact_line',observed_at=p['at'],sources=ref['sources'])
        return dict(probability=None,method='missing_exact_consensus',observed_at=p['at'])
    return dict(probability=None,method='missing_exact_consensus')


def execution(conn,o,selected,close):
    """Only the exact original alert can supply a check; a later quote resets no clock."""
    if not selected.get('selected'):return {}
    offer=dict(o,**{k:selected[k] for k in ('side','line','price','probability')})
    found=[]
    if o['sport']=='nba' and o['market'] in feeds.evaluation.STAT_MAP:
        found=data.rows(conn,"""SELECT a.alert_id,a.delivery_status,a.sent_at,c.checked_at,c.recorded_at,c.status check_status,
            c.displayed_line,c.displayed_price,c.evidence_source,l.payload feed FROM bets.nba_paper_price_alerts a
            JOIN bets.sharp_multi_decisions d USING(decision_id)
            LEFT JOIN bets.nba_paper_price_checks c USING(alert_id) LEFT JOIN bets.fanduel_live_checks l USING(alert_id)
            WHERE d.sport=%s AND d.event_id=%s AND d.market=%s AND d.player_norm=%s AND d.book='fanduel'
            AND d.decided_at=%s AND d.side=%s AND d.line=%s AND d.price=%s""",
            (o['sport'],o['event_id'],o['market'],o['player_norm'],o['decided_at'],selected['side'],selected['line'],selected['price']))
    else:
        found=data.rows(conn,"""SELECT m.id,m.delivery_status,m.sent_at,c.checked_at,c.recorded_at,c.status check_status,
            c.displayed_line,c.displayed_price,'manual_live' evidence_source FROM bets.model_research_decisions d
            JOIN bets.model_research_delivery m USING(id) LEFT JOIN bets.model_research_checks c USING(id)
            WHERE d.sport=%s AND d.event_id=%s AND d.market=%s AND d.player_norm=%s
            AND d.payload->'offer'->>'book'='fanduel' AND (d.payload->'offer'->>'decided_at')::timestamptz=%s
            AND d.payload->'offer'->>'side'=%s AND (d.payload->'offer'->>'line')::numeric=%s
            AND (d.payload->'offer'->>'price')::numeric=%s""",
            (o['sport'],o['event_id'],o['market'],o['player_norm'],o['decided_at'],selected['side'],selected['line'],selected['price']))
    r=found[0] if len(found)==1 else {}
    p=close.get('probability');p=p if p is None or selected['side'] in ('over','home') else 1-p
    offer.update(r,close_method='exact' if p is not None and close.get('method')=='observed_exact_line' else None,close_probability=p)
    out=dict(check_evidence(offer),manual_status=r.get('check_status'),alert_id=r.get('alert_id') or r.get('id'),
        displayed_line=r.get('displayed_line'),displayed_price=r.get('displayed_price'),sent_at=r.get('sent_at'),
        checked_at=r.get('checked_at'),recorded_at=r.get('recorded_at'),feed_status='unknown',feed_checked_price_ev=None,
        feed_model_ev=None,feed_timely=False,feed_reason='no_matching_original_alert_check')
    feed=r.get('feed') or {};a=feed.get('alert') or {}
    if feed:
        same=all(a.get(k)==offer.get(k) for k in ('event_id','market','side','book'))
        same=same and number(a.get('line'))==float(selected['line']) and number(a.get('price'))==float(selected['price'])
        same=same and safe_time(a.get('decided_at'))==safe_time(o['decided_at'])
        timely=bool(same and feed.get('timely') and feed.get('source')=='automated_fanduel_public_feed')
        out.update(feed_status=feed.get('status','unknown'),feed_timely=timely,feed_reason=feed.get('reason'),
            feed_observed_at=feed.get('observed_at'),feed_displayed_line=feed.get('line'),feed_displayed_price=feed.get('price'))
        exact=timely and feed.get('status')=='available' and number(feed.get('line'))==float(selected['line']) and sharp_math.implied(feed.get('price')) is not None
        if exact:
            out['feed_model_ev']=sharp_math.ev(selected['probability'],feed['price'])
            if p is not None:out['feed_checked_price_ev']=sharp_math.ev(p,feed['price'])
    return out


def economics(selected,span):
    """Illustrative $1 volume at checked prices; never an actual-profit claim."""
    values=pd.to_numeric(selected.opportunity_ev,errors='coerce')
    coverage=float(values.notna().mean()) if len(values) else 0.
    out=dict(monthly_cost_usd=30,comparison_stake_usd=1,priced_opportunity_coverage=coverage,
        projected_monthly_gross=None,projected_monthly_net=None,actual_profit=None,
        reason='Need 21 days and 90% of selected opportunities checked and priced; unknowns are not zero.')
    if span<21 or coverage<.9:return out
    monthly_count=len(selected)*30/span
    intervals={k:clustered_interval(selected,'opportunity_ev',c) for k,c in (('game','event_id'),('day','day'))}
    out.update(projected_monthly_decisions=monthly_count,mean_opportunity_ev=float(values.mean()),
        projected_monthly_gross=float(values.mean()*monthly_count),
        projected_monthly_net=float(values.mean()*monthly_count-30),
        monthly_net_bounds={k:{b:None if v[b] is None else float(v[b]*monthly_count-30)
            for b in ('lower_95','upper_95')} for k,v in intervals.items()},
        reason='Illustrative continuation of observed volume and checked-price reference-close EV at $1. No stake increase; not realized profit.')
    return out


def summary(frame):
    if frame.empty:return dict(decisions=0,status='collecting',betting_authorized=False)
    n=len(frame);span=(max(frame.day)-min(frame.day)).days+1
    settled=frame.outcome.notna();exact=frame.close_probability.notna()
    result=dict(decisions=n,games=int(frame.event_id.nunique()),days=int(frame.day.nunique()),span_days=span,
        settled=int(settled.sum()),pending=int((~settled).sum()),exact_close_coverage=float(exact.mean()),variants={},status='collecting',betting_authorized=False)
    for label in ('native','rotation'):
        if label+'_error' in frame:
            error=pd.to_numeric(frame[label+'_error'],errors='coerce').dropna()
            result[label+'_stat_accuracy']=dict(rows=len(error),mae=float(error.abs().mean()) if len(error) else None,
                rmse=float(np.sqrt((error**2).mean())) if len(error) else None)
    for variant in ('consensus','production','rotation_mean','rotation_mixture','blend_production','blend_rotation'):
        p=pd.to_numeric(frame[variant],errors='coerce');common=frame[settled&p.notna()].copy()
        item=dict(probability_rows=len(common),missing_probabilities=int(p.isna().sum()))
        if len(common):
            prob=p.loc[common.index];y=common.outcome
            common['gain']=(common.consensus-y)**2-(prob-y)**2
            item.update(brier=float(((prob-y)**2).mean()),consensus_brier=float(((common.consensus-y)**2).mean()),
                paired_gain={k:clustered_interval(common,'gain',c) for k,c in (('game','event_id'),('day','day'))},
                bins=[dict(lower=float(lo),rows=int(mask.sum()),predicted=float(prob[mask].mean()),observed=float(y[mask].mean()))
                    for lo in np.arange(0,1,.1) if (mask:=(prob>=lo)&(prob<lo+.1)).any()])
        selected=frame[frame[variant+'_selected']].copy();item['selected']=len(selected)
        if len(selected):
            for key in ('timely_check','exact_offer_available','checked_price_ev','opportunity_ev','feed_checked_price_ev','feed_timely','feed_status'):
                selected[key]=[r.get(key) for r in selected[variant+'_execution']]
            checked=pd.to_numeric(selected.checked_price_ev,errors='coerce').notna()
            timely=selected.timely_check.fillna(False).astype(bool);available=selected.exact_offer_available.fillna(False).astype(bool)
            selected['close_ev']=selected[variant+'_close_ev']
            intervals={k:clustered_interval(selected,value,c) for k,value,c in (
                ('checked_game','checked_price_ev','event_id'),('checked_day','checked_price_ev','day'),
                ('quoted_game','close_ev','event_id'),('quoted_day','close_ev','day'))}
            gates=dict(span=span>=21,games=selected.event_id.nunique()>=20,active_days=selected.day.nunique()>=7,
                weeks=len({(d-min(frame.day)).days//7 for d in selected.day})>=3,
                checked_exact_closes=int(checked.sum())>=50,exact_close_coverage=selected.close_probability.notna().mean()>=.9,
                manual_check_coverage=timely.mean()>=.9,availability=bool(timely.any() and available.sum()/timely.sum()>=.8),
                checked_close_coverage=bool(available.any() and checked.sum()/available.sum()>=.9),
                positive_execution_bounds=all(v['lower_95'] is not None and v['lower_95']>0 for v in intervals.values()))
            if variant!='consensus':gates['improves_consensus']=all(v['lower_95'] is not None and v['lower_95']>0 for v in item.get('paired_gain',{}).values()) and bool(item.get('paired_gain'))
            selected_settled=selected[selected.outcome.notna()&selected[variant].notna()]
            paired=selected_settled.assign(gain=(selected_settled.consensus-selected_settled.outcome)**2
                -(selected_settled[variant]-selected_settled.outcome)**2)
            selected_gain={k:clustered_interval(paired,'gain',c) for k,c in (('game','event_id'),('day','day'))}
            if variant!='consensus':
                gates['improves_consensus_on_selected']=all(v['lower_95'] is not None and v['lower_95']>0 for v in selected_gain.values())
            item.update(manual_checked_closes=int(checked.sum()),manual_check_coverage=float(timely.mean()),
                public_feed_checked_closes=int(pd.to_numeric(selected.feed_checked_price_ev,errors='coerce').notna().sum()),
                public_feed_statuses=dict(Counter(selected.feed_status)),intervals=intervals,gates={k:bool(v) for k,v in gates.items()},
                review_only=all(gates.values()),economics=economics(selected,span),selected_paired_gain=selected_gain,
                paper_profit=float(pd.to_numeric(selected[variant+'_paper_profit'],errors='coerce').sum()),
                selected_brier=float(((selected_settled[variant]-selected_settled.outcome)**2).mean()) if len(selected_settled) else None)
        result['variants'][variant]=item
    return result


def build(conn,doc):
    saved=data.rows(conn,'SELECT id,payload FROM bets.edge_market_decisions WHERE experiment_sha256=%s ORDER BY id',(doc['sha256'],))
    index=feeds.nba_result_index(conn,data.now()) if any(r['payload']['offer']['sport']=='nba' for r in saved) else {}
    records=[];cache={}
    for r in saved:
        payload=r['payload'];o=payload['offer'];d=payload['decision']
        actual=feeds.outcome(conn,o,d['native'].get('forecast'),index)
        close=exact_close(conn,o,payload['schedule'],cache) if safe_time(payload['schedule']['tip'])<=data.now() else {}
        row=dict(id=r['id'],sport=o['sport'],market=o['market'],event_id=o['event_id'],player=o['player'],
            day=safe_time(o['commence_time_utc']).astimezone(ZoneInfo('America/New_York')).date(),
            close_probability=close.get('probability'),close_method=close.get('method'),**actual,**d['probabilities'])
        f=d['native'].get('forecast') or {};w=d.get('workload') or {}
        mean=number(f.get('pred_margin' if o['market']=='spread' else 'pred_total')) if o['market'] in ('spread','total') else number(f.get('mu',f.get('projection')))
        row['native_error']=mean-actual['actual'] if mean is not None and actual['actual'] is not None else None
        row['rotation_error']=w['mu']-actual['actual'] if w.get('mu') is not None and actual['actual'] is not None else None
        for label,s in d['selection'].items():
            row[label+'_selected']=s.get('selected',False);row[label+'_execution']=execution(conn,o,s,close)
            cp=close.get('probability');is_a=s.get('side') in ('home','over')
            row[label+'_close_ev']=sharp_math.ev(cp if is_a else 1-cp,s['price']) if cp is not None and s.get('selected') else None
            y=actual['outcome'];profit=None
            if y is not None and s.get('selected'):
                win=bool(y) if is_a else not bool(y);profit=(s['price']/100 if s['price']>0 else 100/abs(s['price'])) if win else -1.
            row[label+'_paper_profit']=profit
        records.append(row)
    frame=pd.DataFrame(records);groups={}
    if len(frame):
        for (sport,stat),g in frame.groupby(['sport','market']):groups[sport+':'+stat]=summary(g)
    observation_counts=data.rows(conn,"""SELECT sport,count(*) observations,count(DISTINCT event_id) games,
        count(*) FILTER(WHERE payload->>'reference_status'<>'ok') missing_consensus
        FROM bets.edge_market_observations WHERE experiment_sha256=%s GROUP BY sport""",(doc['sha256'],))
    result=frozen.clean(dict(built_at=data.now(),experiment_id=trial.NAME,sha256=doc['sha256'],registered_at=doc['registered_at'],
        groups=groups,observations=observation_counts,records=records,betting_authorized=False,automatic_promotion=False,
        execution_note='Public-feed observations do not establish logged-in availability or accepted wagers. Missing manual checks remain unknown.',
        economics=dict(monthly_cost_usd=30,comparison_stake_usd=1,actual_profit=None,projected_monthly_net=None,
            reason='Actual wagers are reconciled separately; no net-profit claim from unknown execution.'),
        unsupported=dict(mlb='Legacy stored lines do not retain both book/market clocks; no live adapter activated.')))
    atomic_json(trial.ROOT/'reports/market_rotation_trial_latest.json',result)
    text=['# Market consensus and joint rotation trial','',f"Registered: {doc['registered_at']}. Captured pairs: {len(records)}.",
        'FanDuel excluded from reference consensus. Separate sport/market evidence; no pooled cross-sport success claim.','',
        '| Group | Pairs | Games | Settled | Exact-close coverage |','|---|---:|---:|---:|---:|']
    for key,g in groups.items():text.append(f"| {key} | {g['decisions']} | {g['games']} | {g['settled']} | {g['exact_close_coverage']:.1%} |")
    text+=['',result['execution_note'],'All variants remain research; review gates never authorize stakes.']
    (trial.ROOT/'reports/market_rotation_trial_latest.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return result


def availability_audit(conn):
    """Price survival and slippage for the existing NBA direct checks, without editing them."""
    from model_research_report import exact_close as original_close
    found=data.rows(conn,'SELECT alert_id,status,reason,payload FROM bets.fanduel_live_checks ORDER BY id')
    records=[]
    for r in found:
        p=r['payload'];a=p['alert'];close=original_close(conn,dict(a,sport='nba'))
        cp=close.get('probability') if close.get('method')=='exact' else None
        actual_price=p.get('price') if p.get('timely') and r['status']=='available' and number(p.get('line'))==number(a['line']) else None
        checked_ev=sharp_math.ev(cp,actual_price) if cp is not None and sharp_math.implied(actual_price) is not None else None
        alert_ev=sharp_math.ev(cp,a['price']) if cp is not None else None
        at=safe_time(p.get('observed_at'));sent=safe_time(a.get('sent_at'))
        records.append(dict(alert_id=r['alert_id'],event_id=a['event_id'],player=a['player'],market=a['market'],side=a['side'],
            line=a['line'],alert_price=a['price'],displayed_line=p.get('line'),displayed_price=p.get('price'),
            alert_at=a.get('sent_at'),observed_at=p.get('observed_at'),seconds_after_alert=(at-sent).total_seconds() if at and sent else None,
            status=r['status'],reason=r['reason'],timely=bool(p.get('timely')),reference_close_probability=cp,
            alert_close_ev=alert_ev,feed_close_ev=checked_ev,price_slippage_ev=checked_ev-alert_ev if checked_ev is not None and alert_ev is not None else None))
    usable=[r for r in records if r['feed_close_ev'] is not None]
    result=frozen.clean(dict(built_at=data.now(),source='automated_fanduel_public_feed',attempts=len(records),statuses=dict(Counter(r['status'] for r in records)),
        paired_exact_closes=len(usable),mean_feed_close_ev=float(np.mean([r['feed_close_ev'] for r in usable])) if usable else None,
        records=records,manual_checks_added=0,betting_authorized=False,
        limitations='Public-feed quotes only. Not account-specific offers, manual checks, accepted wagers or actual profit.'))
    atomic_json(trial.ROOT/'reports/price_survival_audit_latest.json',result)
    return result
