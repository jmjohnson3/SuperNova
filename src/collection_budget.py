"""Read-only collection capacity and small-stake economics audit.

Uses cached free event metadata and official NBA schedules. Never buys odds,
changes polling, changes staking, or treats quoted offers as checked prices.
"""
from collections import Counter
import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
import json
import os
from zoneinfo import ZoneInfo

import psycopg2

from model_research import ROOT, rows, clean, safe_time, atomic_json
from nfl_pipeline.db import PG_DSN
from sharp_pipeline import scan

ET=ZoneInfo('America/New_York')


def economics(daily,monthly_cost,stake=1,*,period_start=None,period_end=None,cohort=None):
    cost,stake=Decimal(str(monthly_cost)),Decimal(str(stake))
    if not cost.is_finite() or not stake.is_finite() or cost<0 or stake<=0:
        raise ValueError('Cost must be finite/nonnegative and stake finite/positive')
    scenarios=[]
    if (period_start is None) != (period_end is None):
        raise ValueError('Supply both collection-period boundaries')
    if period_start is not None:
        start=datetime.fromisoformat(str(period_start)).date()
        end=datetime.fromisoformat(str(period_end)).date()
        if end<start:raise ValueError('Collection period is reversed')
        daily=[r for r in daily if start<=datetime.fromisoformat(str(r['day'])).date()<=end]
        days=(end-start).days+1
        offers=sum(int(r['distinct_offers']) for r in daily)
        monthly=Decimal(offers)/Decimal(days)*30
    elif daily:
        dates=[datetime.fromisoformat(str(r['day'])).date() for r in daily]
        days=(max(dates)-min(dates)).days+1
        offers=sum(int(r['distinct_offers']) for r in daily)
        monthly=Decimal(offers)/Decimal(days)*30
    else:days=offers=0;monthly=None
    for edge in (Decimal('.01'),Decimal('.03'),Decimal('.05')):
        count=int((cost/(stake*edge)).to_integral_value(rounding=ROUND_CEILING))
        gross=monthly*stake*edge if monthly is not None else None
        scenarios.append(dict(assumed_true_edge=str(edge),required_monthly_bets=count,
            maximum_monthly_cost_if_all_offers_available=str(gross) if gross is not None else None,
            gross_if_all_observed_opportunities_available=str(gross) if gross is not None else None,
            net_if_all_observed_opportunities_available=str(gross-cost) if gross is not None else None))
    return dict(book='fanduel',sport='nba',cohort=cohort,period_start=period_start,period_end=period_end,
        observed_offers=offers,calendar_days=days,
        assumed_monthly_opportunities=str(monthly) if monthly is not None else None,
        monthly_cost=str(cost),stake=str(stake),scenarios=scenarios,
        edge_needed_at_observed_rate=str(cost/(monthly*stake)) if monthly else None,
        interpretation=('Conditional arithmetic for the specified frozen cohort, including zero-offer days. ' if cohort else
        'Conditional arithmetic using historical offer counts across rule versions. ')+
        'Availability, true edge, future volume and net profit are unknown. NFL contribution is not assumed.')


def forecast(events,official,now,reset_day=3):
    """Conservative poll demand; five-minute NFL and ten-minute shared scheduler cases."""
    start=now.astimezone(ET).date();reset=scan.reset_date(start,reset_day)
    stops={};counts={}
    for sport,p in events['sports'].items():
        upcoming=[e for e in p['events'] if now<safe_time(e['commence_time'])
                  and safe_time(e['commence_time']).astimezone(ET).date()<reset]
        stops[sport]=[safe_time(e['commence_time']) for e in upcoming]
        counts[sport]=Counter(t.astimezone(ET).date().isoformat() for t in stops[sport])
    provider_days=set(counts.get('basketball_nba',{}))
    for day in official['days']:
        if day['status']!='ok' or day['day'] in provider_days:continue
        for game in day.get('games',[]):
            tip=safe_time(game['tip_utc'])
            if now<tip and tip.astimezone(ET).date()<reset:
                stops['basketball_nba'].append(tip)
                counts['basketball_nba'][day['day']]+=1
    # Bulk line requests are shared by every game in that sport, never charged per game.
    line_polls=Counter();last={};at=now.replace(second=0,microsecond=0)
    at+=timedelta(minutes=(5-at.minute)%10)
    if at<now:at+=timedelta(minutes=10)
    end=datetime.combine(reset,datetime.min.time(),ET).astimezone(timezone.utc)
    while at<end:
        for sport in ('basketball_nba','americanfootball_ncaaf'):
            if any(0<(tip-at).total_seconds()<=3600 for tip in stops.get(sport,[])):
                if sport not in last or (at-last[sport]).total_seconds()>=scan.LINES_SPACING*60:
                    line_polls[sport]+=1;last[sport]=at
        at+=timedelta(minutes=10)
    nba=sum(counts['basketball_nba'].values());nfl=sum(counts['americanfootball_nfl'].values())
    spread_polls=Counter();at=now.replace(second=0,microsecond=0)
    while at<end:
        if any(0<(tip-at).total_seconds()<=3600 for tip in stops.get('americanfootball_nfl',[])):
            spread_polls[at.astimezone(ET).date().isoformat()]+=1
        at+=timedelta(minutes=5)
    costs=dict(nba_prop_credit_ceiling=nba*5*4,nfl_sharp_credit_ceiling=nfl*12*4,
               nfl_spread_credit_ceiling=sum(min(48,n) for n in spread_polls.values()),
               shared_bulk_line_credits=sum(line_polls.values())*3)
    remaining=int(events['sports']['basketball_nba']['headers']['x-requests-remaining'])
    # These bounds cover the frozen watchers. Other crawlers share this account.
    age=(now-safe_time(events['observed_at'])).total_seconds()
    return dict(configured_reset_date=str(reset),remaining_at_snapshot=remaining,
        snapshot_at=events['observed_at'],counts_by_day={k:dict(sorted(v.items())) for k,v in counts.items()},
        balance_snapshot_age_seconds=age,balance_is_recent=0<=age<=1800,
        line_polls=dict(line_polls),watcher_credit_estimates=costs,
        after_watchers_and_150_floor=remaining-sum(costs.values())-150,
        caveat='Projected watcher demand for listed schedules only, not a complete billing-cycle forecast. '
        'Includes the separate NFL spread collector; automatic FanDuel feed checks consume zero Odds API credits. '
        'Other crawlers, retries, schedule changes and manual API usage are not included. '
        'The reset date is local configuration, not provider-confirmed billing metadata.')


def refresh_events():
    """Only the provider's free current event-list endpoints; never requests odds."""
    import requests
    from supernovabets_config import _saved_windows_env
    key=_saved_windows_env('ODDS_API_KEY') or os.getenv('ODDS_API_KEY')
    if not key:raise ValueError('Odds API key is not configured')
    data=dict(observed_at=datetime.now(timezone.utc).isoformat(),sports={})
    for sport in ('basketball_nba','americanfootball_nfl','americanfootball_ncaaf'):
        try:
            r=requests.get(f'https://api.the-odds-api.com/v4/sports/{sport}/events',
                           params={'apiKey':key},timeout=15)
        except requests.RequestException as exc:
            raise RuntimeError('Free event request failed: '+type(exc).__name__) from None
        if r.status_code!=200:raise RuntimeError(f'Free event request returned HTTP {r.status_code}')
        body=r.json()
        if not isinstance(body,list):raise ValueError('Unexpected event-list response')
        data['sports'][sport]=dict(events=body,headers={k:r.headers.get(k) for k in
            ('x-requests-remaining','x-requests-used','x-requests-last')})
    atomic_json(ROOT/'data/collection_budget/events_latest.json',data)


def run(*,quiet=False):
    events=json.loads((ROOT/'data/collection_budget/events_latest.json').read_text())
    official=json.loads((ROOT/'data/collection_budget/nba_upcoming_schedule.json').read_text())
    budget=json.loads((ROOT/'data/sports_finance/reports/monthly_budget.json').read_text())
    from sharp_pipeline.nba_experiment import active
    doc=active();now=datetime.now(timezone.utc)
    start=safe_time(doc['registered_at']).astimezone(ET).date()
    end=now.astimezone(ET).date()-timedelta(days=1)
    with psycopg2.connect(PG_DSN,connect_timeout=10) as conn:
        conn.set_session(readonly=True)
        daily=rows(conn,"""WITH offers AS (SELECT DISTINCT ON(event_id,market,player_norm)
            event_id,market,player_norm,commence_time_utc,decided_at FROM bets.sharp_multi_decisions
            WHERE sport='nba' AND book='fanduel' AND tier='decision' AND method='exact'
            AND experiment_id=%s AND experiment_sha256=%s AND decided_at>=%s
            AND commence_time_utc<now() ORDER BY event_id,market,player_norm,decided_at,ev DESC)
            SELECT (commence_time_utc AT TIME ZONE 'America/New_York')::date AS day,
            count(*) AS distinct_offers,count(DISTINCT event_id) AS games FROM offers GROUP BY 1 ORDER BY 1""",
            (doc['experiment_id'],doc['sha256'],doc['registered_at']))
        cur=rows(conn,"""SELECT count(*) AS n FROM bets.nba_paper_price_checks c
            JOIN bets.nba_paper_price_alerts a USING(alert_id) WHERE a.book='fanduel'
            AND a.experiment_id=%s AND a.experiment_sha256=%s""",(doc['experiment_id'],doc['sha256']))
    from supernovabets_config import _saved_windows_env
    reset_day=int(os.getenv('NFL_ODDS_API_RESET_DAY') or _saved_windows_env('NFL_ODDS_API_RESET_DAY') or 1)
    result=clean(dict(built_at=now,collection=forecast(events,official,now,reset_day),
        economics=economics(daily,budget['known_monthly_subscription'],period_start=str(start),period_end=str(end),
                            cohort=doc['sha256']),daily_offers=daily,
        recorded_price_checks=cur[0]['n'],betting_authorized=False,
        provider_documentation='https://the-odds-api.com/liveapi/guides/v4/'))
    atomic_json(ROOT/'reports/collection_economics_latest.json',result)
    c,e=result['collection'],result['economics']
    lines=['# Collection budget and opportunity economics','',
        f"Provider balance: {c['remaining_at_snapshot']} credits at {c['snapshot_at']}.",
        f"Balance snapshot within 30 minutes: {c['balance_is_recent']}.",
        f"Configured next reset: {c['configured_reset_date']} (not independently verified with billing).",
        f"Known-schedule watcher estimates: {sum(c['watcher_credit_estimates'].values())} credits; "
        f"{c['after_watchers_and_150_floor']} left after those estimates and a 150-credit floor.",
        c['caveat'],'',
        f"Frozen FanDuel NBA opportunities ({e['period_start']} through {e['period_end']}): "
        f"{e['observed_offers']} over {e['calendar_days']} calendar days, including days with none; "
        f"{format(Decimal(e['assumed_monthly_opportunities']), '.1f') if e['assumed_monthly_opportunities'] is not None else 'unknown'} "
        'over 30 days at that same rate.',
        f"Manual NBA checks recorded: {result['recorded_price_checks']}. Availability remains unknown.",'',
        '| Assumed true edge | $1 bets needed/month for $30 | Gross at observed rate | Net after $30 |',
        '|---|---:|---:|---:|']
    for s in e['scenarios']:
        gross=s['gross_if_all_observed_opportunities_available'];net=s['net_if_all_observed_opportunities_available']
        lines.append(f"| {float(s['assumed_true_edge']):.0%} | {s['required_monthly_bets']} | "
                     f"{('$'+format(float(gross),'.2f')) if gross is not None else 'unknown'} | "
                     f"{('$'+format(float(net),'.2f')) if net is not None else 'unknown'} |")
    lines+=['',e['interpretation'],
        'These are $1 planning scenarios, not a recommendation to increase stakes or an estimate of actual profit.',
        'Source: [The Odds API quota documentation](https://the-odds-api.com/liveapi/guides/v4/). '
        'The event-list requests used to obtain quota headers cost zero credits.']
    (ROOT/'reports/collection_economics_latest.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if not quiet:print(json.dumps(result,indent=2,allow_nan=False))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--refresh-events',action='store_true')
    if ap.parse_args().refresh_events:refresh_events()
    run()
