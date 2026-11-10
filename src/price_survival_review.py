"""Read-only price survival, paired model diagnostics and market review.

Uses immutable current-feed receipts and saved exact closes. No API requests,
historical substitutions, strategy promotion, cash authorization or ledger writes.
"""
from collections import Counter
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import hashlib
from statistics import median
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg2

import automatic_price_checks as observer
import native_close_audit
from nba_props import fanduel_live as legacy
from model_research import ROOT, active as model_active, clean, safe_time, CONFIG
from model_research_report import exact_close, verified_result
from nfl_pipeline.context_contract import number
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.sharp_math import ev
from sharp_pipeline.nba_experiment import clustered_interval

ET=ZoneInfo('America/New_York')
DENVER=ZoneInfo('America/Denver')


def delta(end,start):
    end,start=safe_time(end),safe_time(start)
    return (end-start).total_seconds() if end and start else None


def profit_per_win(price):
    p=number(price)
    return p/100 if p is not None and p>=100 else 100/abs(p) if p is not None and p<=-100 else None


def matches(a,b):
    return (all(str(a.get(k) or '')==str(b.get(k) or '') for k in ('sport','book','event_id','market','player_norm','side'))
            and number(a.get('line'))==number(b.get('line')) and number(a.get('price'))==number(b.get('price')))


def close_clock_check(alert,close):
    """Audit against the earlier schedule, without rewriting a frozen close selection."""
    native,provider=safe_time(alert.get('native_tip')),safe_time(alert.get('commence_time_utc'))
    if close.get('method')!='exact':return False,'no_exact_close'
    if not native or not provider:return False,'native_cutoff_unverified'
    at=safe_time(close.get('observed_at'));updated=safe_time(close.get('source_updated_at'))
    if at:
        if at>=min(native,provider) or (updated and updated>=min(native,provider)):
            return False,'close_not_before_earliest_schedule'
        return True,'observed_before_earliest_schedule'
    # The pinned NFL reader queries only pregame observations. Its concise return
    # omits the observation time; that query cutoff is safe only when both clocks agree.
    if alert['sport']=='nfl' and native==provider:
        return True,'registered_nfl_reader_enforces_identical_cutoff'
    return False,'close_observation_time_unverified'


def evaluate(record,model,close,result,now,*,audited_close=None):
    """Unknown observations never become losses, disappearances, zero EV or manual checks."""
    a,p=record['alert'],record.get('payload') or {}
    observed=safe_time(p.get('observed_at'))
    checked_delay=delta(observed,a.get('sent_at'));recording_delay=delta(record.get('recorded_at'),observed)
    starts=[safe_time(a[k]) for k in ('native_tip','commence_time_utc') if safe_time(a.get(k))]
    receipts=p.get('receipts') or []
    receipt_errors=[]
    if len(receipts)!=2:receipt_errors.append('two_current_receipts_required')
    for receipt in receipts:
        reason=legacy.receipt_reason(receipt,observed or safe_time(receipt.get('received_at')) or now)
        if reason:receipt_errors.append(reason)
        requested=safe_time(receipt.get('requested_at'))
        if not requested or not safe_time(a.get('sent_at')) or requested<safe_time(a['sent_at']):
            receipt_errors.append('receipt_before_alert_or_time_missing')
    timely=bool(p.get('timely') and observed and starts and safe_time(record.get('recorded_at'))
        and safe_time(record['recorded_at'])<min(starts) and checked_delay is not None and recording_delay is not None
        and 0<=checked_delay<=300 and 0<=recording_delay<=300)
    original_payout=profit_per_win(a.get('price'));payout=profit_per_win(p.get('price'))
    available=bool(timely and not receipt_errors and p.get('status')=='available'
        and number(p.get('line'))==number(a.get('line')) and payout is not None)
    unavailable=bool(timely and not receipt_errors and p.get('status') in ('not_listed','changed_line','suspended'))
    change=(payout-original_payout) if available and original_payout is not None else None
    price_change='worse' if change is not None and change<-1e-9 else 'better' if change is not None and change>1e-9 else 'same' if change is not None else 'unknown'
    # Half-integer lines avoid inferring unrecorded push probabilities.
    line=number(a.get('line'));no_push=line is not None and abs(line%1-.5)<1e-9
    reported_ok,reported_reason=close_clock_check(a,close)
    audited=close if audited_close is None else audited_close
    close_p=number(audited.get('probability'))
    clock_ok,clock_reason=close_clock_check(a,audited)
    has_close=bool(clock_ok and close_p is not None and 0<=close_p<=1)
    initial_p=number(a.get('fair_probability'))
    initial_p=initial_p if initial_p is not None and 0<=initial_p<=1 else None
    checked_ev=ev(close_p,p['price']) if available and has_close and no_push else None
    initial_checked=ev(initial_p,p['price']) if available and initial_p is not None and no_push else None
    original_ev=ev(close_p,a['price']) if has_close and original_payout is not None and no_push else None
    opportunity=checked_ev if available else 0.0 if unavailable else None
    settled=result in ('win','loss','push')
    profit=(payout if result=='win' else -1. if result=='loss' else 0.) if settled and available else None
    model_ok=bool(model and matches(a,model['payload']['offer']) and safe_time(model['captured_at'])
        and safe_time(record.get('attempted_at')) and safe_time(model['captured_at'])<=safe_time(record['attempted_at']))
    selected=bool(model['payload']['decision']['selected']) if model_ok else None
    mp=number(model['payload']['decision'].get('model_probability')) if model_ok else None
    y=1. if result=='win' else 0. if result=='loss' else None
    model_brier=(mp-y)**2 if y is not None and mp is not None and 0<=mp<=1 else None
    reference_brier=(initial_p-y)**2 if y is not None and initial_p is not None else None
    incremental=(0.0 if selected else -opportunity) if model_ok and opportunity is not None else None
    return clean(dict(alert_key=record['alert_key'],sport=a['sport'],book=a['book'],market=a['market'],
        player=a.get('player'),event_id=a['event_id'],day=safe_time(a['commence_time_utc']).astimezone(ET).date(),
        started=safe_time(a['commence_time_utc'])<=now,status=p.get('status','unknown'),reason=p.get('reason','attempt_incomplete'),
        side=a['side'],original_line=a['line'],original_price=a['price'],observed_line=p.get('line'),observed_price=p.get('price'),
        observed_at=observed,delivery_seconds=delta(a.get('sent_at'),a.get('decided_at')),
        queue_seconds=delta(record.get('attempted_at'),a.get('sent_at')),check_delay_seconds=checked_delay,
        recording_seconds=recording_delay,receipt_errors=sorted(set(receipt_errors)),timely=timely,
        available=available,unavailable=unavailable,price_change=price_change,
        exact_close=has_close,reported_exact_close=close.get('method')=='exact',close_clock_reason=clock_reason,
        reported_close_passed_audit=reported_ok,reported_close_clock_reason=reported_reason,
        recovered_from_saved_quote=bool(has_close and not reported_ok),
        close_observed_at=audited.get('observed_at'),priceable_without_push_assumption=no_push,
        original_price_close_ev=original_ev,checked_price_close_ev=checked_ev,
        checked_price_initial_ev=initial_checked,slippage_ev=checked_ev-original_ev if checked_ev is not None and original_ev is not None else None,
        opportunity_ev=opportunity,paired_model=model_ok,model_selected=selected,incremental_ev=incremental,
        verified_result=result,paper_profit_at_observed_price=profit,
        incremental_paper_profit=(0.0 if selected else -profit) if model_ok and profit is not None else None,
        model_brier=model_brier,reference_brier=reference_brier,
        brier_improvement=reference_brier-model_brier if model_brier is not None and reference_brier is not None else None,
        account_availability_verified=False,betting_authorized=False))


def interval(frame,value,cluster):
    if frame.empty:return dict(observations=0,clusters=0,mean=None,lower_95=None,upper_95=None)
    return clustered_interval(frame,value,cluster)


def distribution(records,key):
    values=sorted(float(r[key]) for r in records if number(r.get(key)) is not None and r[key]>=0)
    return dict(n=len(values),median=median(values) if values else None,
                p90=values[max(0, int(.9*(len(values)-1)))] if values else None)


def summarize(records,sport):
    frame=pd.DataFrame(records)
    completed=frame[frame.started] if not frame.empty else frame
    pair=completed[completed.paired_model] if not completed.empty else completed
    usable=pair[pair.checked_price_close_ev.notna()] if not pair.empty else pair
    selected=usable[usable.model_selected.eq(True)] if not usable.empty else usable
    span=(pd.to_datetime(completed.day).max()-pd.to_datetime(completed.day).min()).days+1 if not completed.empty else 0
    game=interval(pair,'incremental_ev','event_id');day=interval(pair,'incremental_ev','day')
    brier=interval(usable,'brier_improvement','event_id')
    required=CONFIG[sport+'_review']
    counts=bool(span>=required['min_days'] and not pair.empty and pair.event_id.nunique()>=required['min_games']
                and len(usable)>=required['min_checked_closes'] and len(selected)>=required['min_checked_closes'])
    coverage=bool(not completed.empty and completed.exact_close.mean()>=.9 and completed.available.mean()>=.9
                  and len(pair)/len(completed)>=.9)
    positive=all(ci['lower_95'] is not None and ci['lower_95']>0 for ci in (game,day))
    negative=all(ci['upper_95'] is not None and ci['upper_95']<0 for ci in (game,day))
    props=all(r['market'] not in ('spread','total') for r in records)
    accuracy=not props or (brier['lower_95'] is not None and brier['lower_95']>0)
    verdict=('public_evidence_supports_manual_review' if counts and coverage and positive and accuracy else
             'public_evidence_against_filter_review' if counts and coverage and negative else 'insufficient_public_evidence')
    exact_prices=[r for r in records if r['checked_price_close_ev'] is not None]
    return clean(dict(attempts=len(records),completed_attempts=len(completed),span_days=span,
        statuses=dict(Counter(r['status'] for r in records)),reasons=dict(Counter(r['reason'] for r in records)),
        receipt_error_counts=dict(Counter(reason for r in records for reason in r['receipt_errors'])),
        timely=sum(r['timely'] for r in records),available=sum(r['available'] for r in records),
        price_changes=dict(Counter(r['price_change'] for r in records)),
        unavailable=sum(r['unavailable'] for r in records),
        exact_closes=sum(r['exact_close'] for r in records),checked_prices_with_exact_closes=len(exact_prices),
        close_clock_reasons=dict(Counter(r['close_clock_reason'] for r in records)),
        excluded_reported_exact_closes=sum(r['reported_exact_close'] and not r['reported_close_passed_audit'] for r in records),
        recovered_saved_exact_closes=sum(r['recovered_from_saved_quote'] for r in records),
        positive_at_checked_price=sum(r['checked_price_close_ev']>0 for r in exact_prices),
        delays={k:distribution(records,k) for k in ('delivery_seconds','queue_seconds','check_delay_seconds','recording_seconds')},
        pairs=len(pair),unpaired_completed=len(completed)-len(pair),model_selected=len(pair[pair.model_selected.eq(True)]) if not pair.empty else 0,
        price_control_ev_by_game=interval(usable,'checked_price_close_ev','event_id'),
        model_selected_ev_by_game=interval(selected,'checked_price_close_ev','event_id'),
        model_incremental_ev_by_game=game,model_incremental_ev_by_day=day,
        brier_improvement_by_game=brier,model_incremental_paper_profit_by_game=interval(pair,'incremental_paper_profit','event_id'),
        model_selected_checked_closes=len(selected),model_value_status=verdict,
        observed_subset_only=True,automatic_filter_promotion=False,betting_authorized=False,
        required_manual_gate_unchanged=True))


def models_by_alert(conn,doc):
    found=observer.rows(conn,'SELECT id,captured_at,payload FROM bets.model_research_decisions WHERE experiment_sha256=%s',(doc['sha256'],))
    result={}
    for r in found:
        parent=r['payload']['offer'].get('parent_check_id');key=f'#{parent}' if parent else f"M{r['id']}"
        if key in result:raise ValueError('Ambiguous model comparison')
        result[key]=r
    return result


def cohort_records(conn,doc,*,old=False):
    if old:
        found=observer.rows(conn,"""SELECT c.alert_id,c.attempted_at,c.recorded_at,c.payload,
            to_jsonb(d) AS offer,a.sent_at FROM bets.fanduel_live_checks c
            JOIN bets.nba_paper_price_alerts a USING(alert_id)
            JOIN bets.sharp_multi_decisions d USING(decision_id) WHERE c.observer_sha256=%s ORDER BY c.id""",(doc['sha256'],))
        return [dict(r,alert_key=f"#{r['alert_id']}",alert=dict(r['offer'],sent_at=r['sent_at'],
                    native_tip=r['payload']['alert'].get('native_tip'))) for r in found]
    return observer.rows(conn,"""SELECT a.alert_key,a.attempted_at,a.alert,o.recorded_at,o.payload
        FROM bets.automatic_paper_attempts a LEFT JOIN bets.automatic_paper_observations o ON o.attempt_id=a.id
        WHERE a.observer_sha256=%s ORDER BY a.id""",(doc['sha256'],))


def market_reviews(conn):
    from sharp_pipeline import nba_experiment, report
    doc=nba_experiment.active();raw,closes=report.load(conn)
    alerts=pd.DataFrame(observer.rows(conn,"""SELECT a.decision_id,a.delivery_status,a.sent_at,c.checked_at,c.recorded_at,
        c.status AS check_status,c.displayed_line,c.displayed_price,c.evidence_source FROM bets.nba_paper_price_alerts a
        LEFT JOIN bets.nba_paper_price_checks c USING(alert_id) WHERE a.experiment_sha256=%s""",(doc['sha256'],)))
    reviews=[]
    if not raw.empty and not alerts.empty:
        chosen=raw[raw.decision_id.isin(alerts.decision_id) & raw.experiment_sha256.eq(doc['sha256'])
                   & (pd.to_datetime(raw.decided_at,utc=True)>=pd.Timestamp(doc['registered_at']))]
        if chosen.empty:return []
        frame=report.decisions_frame(chosen,closes,per_book=True).merge(alerts,on='decision_id',validate='one_to_one')
        games=observer.rows(conn,"""SELECT DISTINCT game_id,home_team,away_team,tip_utc FROM nbaprops.game_projections
            WHERE tip_utc>=%s AND tip_utc<=%s""",
            (frame.commence_time_utc.min()-timedelta(hours=1),frame.commence_time_utc.max()+timedelta(hours=1)))
        frame['source_had_exact_close']=frame.close_method.eq('exact')
        frame['source_close_rejected']=False
        frame['recovered_saved_close']=False
        frame['native_close_audit']='unverified'
        for index,row in frame.iterrows():
            a=row.to_dict()
            matched=[g for g in games if all(legacy.team_code(a[k]) and legacy.team_code(a[k])==legacy.team_code(g[k])
                for k in ('home_team','away_team')) and abs((safe_time(a['commence_time_utc'])-g['tip_utc']).total_seconds())<=900]
            a['native_tip']=min(g['tip_utc'] for g in matched) if len({g['game_id'] for g in matched})==1 else None
            valid,reason=close_clock_check(a,dict(method=a['close_method'],observed_at=a['close_observed_at'],
                                                  source_updated_at=a['close_source_updated_at']))
            frame.loc[index,'native_close_audit']=reason
            if not valid:
                frame.loc[index,'source_close_rejected']=bool(a['close_method']=='exact')
                evidence=native_close_audit.load(conn,a,a['native_tip'])
                recovered=bool(evidence.get('method')=='exact')
                frame.loc[index,'recovered_saved_close']=recovered
                frame.loc[index,'close_method']='exact' if recovered else 'unverified_cutoff'
                frame.loc[index,'close_probability']=evidence.get('probability')
                frame.loc[index,'ev_at_close']=evidence.get('ev')
        for (book,market),group in frame.groupby(['book','market']):
            m=nba_experiment.book_summary(group)
            reviews.append(dict(sport='nba',book=book,market=market,decisions=m['decisions'],
                exact_closes=m.get('with_exact_close',0),checked_prices=m.get('priced_checks',0),
                originally_reported_exact_closes=int(group.source_had_exact_close.sum()),
                original_closes_rejected=int(group.source_close_rejected.sum()),
                recovered_saved_closes=int(group.recovered_saved_close.sum()),
                native_close_audit=dict(Counter(group.native_close_audit)),
                failed_gates=m['reasons'],status=m['status'],source_sha256=doc['sha256'],
                expansion_authorized=False,betting_authorized=False))
    return reviews


def upcoming(conn,now):
    nba=observer.rows(conn,"""SELECT game_id,min(tip_utc) AS tip, min(home_team) AS home,min(away_team) AS away
        FROM nbaprops.game_projections WHERE tip_utc>%s AND tip_utc<%s GROUP BY game_id ORDER BY tip""",(now,now+timedelta(days=2)))
    nfl=observer.rows(conn,"""SELECT game_id,start_ts_utc AS tip,home_team_abbr AS home,away_team_abbr AS away
        FROM raw.nfl_games WHERE start_ts_utc>%s AND start_ts_utc<%s ORDER BY tip""",(now,now+timedelta(days=4)))
    return clean([dict(r,sport=sport,scan_from=r['tip']-timedelta(minutes=60),
                       decision_from=r['tip']-timedelta(minutes=45),decision_until=r['tip']-timedelta(minutes=15))
                  for sport,games in [('nba',nba),('nfl',nfl)] for r in games])


def run():
    now=datetime.now(timezone.utc);current=observer.active();previous=legacy.active();model_doc=model_active()
    with closing(psycopg2.connect(PG_DSN,connect_timeout=10)) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute('SELECT pg_try_advisory_lock(86731494)')
            if not cur.fetchone()[0]:return dict(status='review_busy')
        models=models_by_alert(conn,model_doc);cohorts={}
        for doc,old in [(previous,True),(current,False)]:
            evaluated=[]
            for r in cohort_records(conn,doc,old=old):
                a=r['alert'];model=models.get(r['alert_key']);started=safe_time(a['commence_time_utc'])<=now
                close=exact_close(conn,a) if started else {}
                audited=native_close_audit.load(conn,a,a.get('native_tip')) if started else {}
                forecast=model['payload']['decision'].get('forecast') if model else None
                result=verified_result(conn,a,forecast) if started else None
                evaluated.append(evaluate(r,model,close,result,now,audited_close=audited))
            groups={}
            for sport in ['nba','nfl']:
                markets=sorted({r['market'] for r in evaluated if r['sport']==sport})
                for market in markets:
                    groups[sport+':'+market]=summarize([r for r in evaluated if r['sport']==sport and r['market']==market],sport)
            cohorts[doc['experiment_id']]=dict(sha256=doc['sha256'],registered_at=doc['registered_at'],
                groups=groups,records=evaluated,attempts=len(evaluated))
        pending={sport:observer.pending(conn,current,sport,now) for sport in ['nba','nfl']}
        backlog={s:dict(unattempted=len(rs),expired=sum(delta(now,r['sent_at'])>300 for r in rs)) for s,rs in pending.items()}
        expansion=market_reviews(conn);schedule=upcoming(conn,now)
    result=clean(dict(built_at=now,cohorts=cohorts,backlog=backlog,upcoming=schedule,
        audit_reader=native_close_audit.METHOD,
        audit_sources_sha256={name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()
                              for name in ('price_survival_review.py','native_close_audit.py')},
        frozen_model_sha256=model_doc['sha256'],
        market_execution_reviews=expansion,observer_cohorts_pooled=False,betting_authorized=False,
        strategy_changes_authorized=False,account_availability_verified=False,
        notes=['Main-market-only v2 and original v1 observations are evaluated separately.',
            'Estimated closes, unknown checks, late checks and integer-line push assumptions cannot create checked-price EV.',
            'An original close after the earlier native/provider schedule cutoff is excluded here. Frozen report counts may still include it; those counts alone cannot establish pregame close coverage.',
            'A replacement is used only when an exact quote was already saved before the earlier cutoff. Recovered closes are identified separately; absent exact quotes remain missing. No historical API requests or synthetic conversions are made.',
            'Unavailable main offers contribute zero opportunity only when verified by a timely complete response; unknowns stay unknown.',
            'Paired diagnostics require the same original offer and a model decision captured before the automatic attempt.',
            'Confidence bounds describe observed subsets, not missing prices. Public evidence cannot clear the manual execution gate.',
            'NBA market reviews apply existing review thresholds separately by book and market; none grants expansion or cash approval.']))
    atomic_json(ROOT/'reports/price_survival_review_latest.json',result)
    lines=['# Price survival and model contribution','',f"Updated {now.isoformat()}.",
        'Paper research only. No filter promotion, sportsbook expansion or wagering authorization.','',
        '| Observer / market | Attempts | Available | Worse prices | Main offer unavailable | Checked exact closes | Positive at close | Model-value status |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for name,c in cohorts.items():
        if not c['groups']:lines.append(f'| {name} — waiting for new alerts | 0 | 0 | 0 | 0 | 0 | 0 | collecting |')
        for market,g in c['groups'].items():
            lines.append(f"| {name} / {market} | {g['attempts']} | {g['available']} | {g['price_changes'].get('worse',0)} | "
                f"{g['unavailable']} | {g['checked_prices_with_exact_closes']} | {g['positive_at_checked_price']} | {g['model_value_status']} |")
    lines+=['','## Delays, failures and paired comparisons','']
    for name,c in cohorts.items():
        for market,g in c['groups'].items():
            ci=g['model_incremental_ev_by_game']
            lines.append(f"- {name} / {market}: reasons {g['reasons']}; median delivery-to-check "
                f"{g['delays']['check_delay_seconds']['median']} seconds; {g['pairs']} paired offers, "
                f"{g['model_selected']} model selections. Incremental close-valued EV game interval: "
                f"{ci['lower_95']} to {ci['upper_95']} (unknown bounds remain unknown).")
            lines.append(f"  Close-clock audit: {g['close_clock_reasons']}; {g['excluded_reported_exact_closes']} originally reported exact closes excluded.")
            lines.append(f"  Separately recovered exact closes from existing pregame quotes: {g['recovered_saved_exact_closes']}.")
    lines+=['','## Market-specific execution review','',
        '| Sport | Book | Market | Decisions | Audited exact closes | Checked prices | Review |','|---|---|---|---:|---:|---:|---|']
    for r in expansion:lines.append(f"| {r['sport']} | {r['book']} | {r['market']} | {r['decisions']} | {r['exact_closes']} | {r['checked_prices']} | {r['status']} |")
    lines+=['','NFL comparisons remain separate; no additional sportsbook or market is enabled.','',
        f"Unattempted current-observer alerts: {backlog}. Expired checks cannot be recovered from historical snapshots.",'',
        '## Upcoming collection windows','']
    for r in schedule:
        scan=safe_time(r['scan_from']).astimezone(DENVER);tip=safe_time(r['tip']).astimezone(DENVER)
        lines.append(f"- {r['sport'].upper()} {r['away']} @ {r['home']}: scan {scan:%b %d %I:%M %p %Z}; start {tip:%I:%M %p %Z}.")
    lines+=['','## Interpretation','']+['- '+s for s in result['notes']]
    (ROOT/'reports/price_survival_review_latest.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(status='updated',attempts={k:v['attempts'] for k,v in cohorts.items()},backlog=backlog,
                report='reports/price_survival_review_latest.md',betting_authorized=False)


if __name__=='__main__':print(json.dumps(run()))
