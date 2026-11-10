"""Offline 500-credit collection assessment. Never requests odds or changes a task.

This is a capacity design, not an activated or registered betting experiment.
"""
from datetime import datetime, timedelta, timezone
from math import ceil
import json
from contextlib import closing
from zoneinfo import ZoneInfo

from model_research import ROOT, clean, safe_time
from nfl_pipeline.integrity import atomic_json
from nba_props.fanduel_live import team_code

ET=ZoneInfo('America/New_York')
DENVER=ZoneInfo('America/Denver')
PROPOSAL=dict(sport='basketball_nba',markets=['player_points'],
    books=['fanduel','pinnacle','draftkings','betmgm'],max_games_per_day=2,
    max_games_per_cycle=62,polls_before_tip_minutes=[40,20,5],max_retries_per_poll=1,
    monthly_credits=500,reserved_credits=100,
    game_selection='earliest scheduled tips, then event ID; independent of odds, predictions and results')


def capacity(config):
    markets=len(set(config['markets']));books=len(set(config['books']))
    if not markets or not books:raise ValueError('Markets and books must be nonempty')
    for k in ('max_games_per_cycle','max_games_per_day','monthly_credits'):
        if not isinstance(config[k],int) or isinstance(config[k],bool) or config[k]<=0:
            raise ValueError('Positive integer required: '+k)
    for k in ('max_retries_per_poll','reserved_credits'):
        if not isinstance(config[k],int) or isinstance(config[k],bool) or config[k]<0:
            raise ValueError('Nonnegative integer required: '+k)
    slots=config['polls_before_tip_minutes']
    if not slots or any(not isinstance(x,int) or x<=0 for x in slots) or len(set(slots))!=len(slots):
        raise ValueError('Polling slots must be distinct positive minutes before tip')
    cost=markets*ceil(books/10)
    requests=config['max_games_per_cycle']*len(slots)
    primary=requests*cost;retry=primary*config['max_retries_per_poll']
    maximum=primary+retry+config['reserved_credits']
    return dict(cost_per_nonempty_request=cost,primary_requests=requests,
        primary_credits=primary,maximum_retry_credits=retry,reserved_credits=config['reserved_credits'],
        maximum_cycle_credits=maximum,headroom=config['monthly_credits']-maximum,
        fits_arithmetic=maximum<=config['monthly_credits'],activated=False,
        assumption='One current event-odds endpoint, explicit books, no historical/scores/market-discovery calls. '
        'Every attempt is reserved at full cost before HTTP, including timeouts. No unbounded retries.')


def proposed_schedule(events,now,config=PROPOSAL,*,native_games=(),require_native=False,cycle_end=None):
    """Display a possible future schedule from the cached event list; no API calls."""
    unique={};skipped={};cycle_end=cycle_end or now+timedelta(days=31)
    for e in events:
        tip=safe_time(e.get('commence_time'))
        if not e.get('id') or not tip or not now<tip<cycle_end:continue
        key=str(e['id'])
        if key in unique and safe_time(unique[key]['commence_time'])!=tip:
            raise ValueError('Conflicting cached event times; schedule cannot be planned')
        unique[key]=e
    selected=[];counts={}
    validated=[]
    for e in unique.values():
        provider=safe_time(e['commence_time'])
        native=[g for g in native_games if all(team_code(e.get(k)) and team_code(e.get(k))==team_code(g.get(k))
                for k in ('home_team','away_team')) and safe_time(g.get('tip'))
                and abs((safe_time(g['tip'])-provider).total_seconds())<=900]
        if require_native and len({g['game_id'] for g in native})!=1:
            skipped[e['id']]='native_schedule_missing_or_ambiguous';continue
        tip=min([provider]+[safe_time(g['tip']) for g in native])
        if not now<tip<cycle_end:continue
        validated.append(dict(e,tip=tip,native_verified=bool(native)))
    for e in sorted(validated,key=lambda e:(e['tip'],str(e['id']))):
        tip=e['tip'];day=tip.astimezone(ET).date().isoformat()
        if counts.get(day,0)>=config['max_games_per_day'] or len(selected)>=config['max_games_per_cycle']:continue
        counts[day]=counts.get(day,0)+1
        selected.append(dict(event_id=e['id'],away_team=e.get('away_team'),home_team=e.get('home_team'),tip_utc=tip,
            provider_tip_utc=e['commence_time'],native_schedule_verified=e['native_verified'],
            poll_utc=[tip-timedelta(minutes=m) for m in config['polls_before_tip_minutes']],
            complete_window_still_ahead=now<tip-timedelta(minutes=max(config['polls_before_tip_minutes']))))
    return clean(dict(cached_upcoming_games=len(unique),selected_games=selected,skipped=skipped,
                      omitted_games=len(unique)-len(selected),activated=False))


def switch_assessment(cap,*,frozen_collection_active=True,books_verified=False,reset_verified=False,
                      all_other_paid_consumers_disabled=False,prospective_registration=False):
    facts=dict(within_500_credits=cap['fits_arithmetic'],frozen_collection_completed=not frozen_collection_active,
        required_books_verified_on_free_plan=books_verified,provider_credit_reset_verified=reset_verified,
        other_paid_consumers_controlled=all_other_paid_consumers_disabled,
        new_prospective_protocol_registered=prospective_registration)
    return dict(checks=facts,blockers=[k for k,v in facts.items() if not v],
        ready_for_switch_review=all(facts.values()),subscription_changed=False,collectors_changed=False,
        betting_authorized=False)


def current_usage(now):
    from sharp_pipeline.scan import reset_date
    from supernovabets_config import _saved_windows_env
    reset_day=int(_saved_windows_env('NFL_ODDS_API_RESET_DAY') or 1)
    end=reset_date(now.astimezone(ET).date(),reset_day)
    # Reset configuration is a planning boundary, not a billing attestation.
    before=end.replace(day=1)-timedelta(days=1)
    start=before.replace(day=min(reset_day,before.day))
    counters={}
    for file,key,label in [('sharp_multi_state.json','spent','shared_watcher'),
                           ('nfl_sharp_watch_state.json','spent','nfl_watcher'),
                           ('nfl_spread_collection_state.json','daily_calls','nfl_spreads')]:
        path=ROOT/'reports'/file
        try:
            d=json.loads(path.read_text());counts=d.get(key)
            if not isinstance(counts,dict):raise ValueError('Missing counter')
            selected={k:int(v) for k,v in counts.items() if str(start)<=k<str(end)}
            counters[label]=dict(recorded=sum(selected.values()),daily=selected)
        except (OSError,ValueError,TypeError):counters[label]=dict(recorded=None,error='counter_unavailable')
    # Explicitly a partial lower bound. Other crawlers share the provider account.
    known=sum(r['recorded'] for r in counters.values() if r['recorded'] is not None)
    return dict(configured_cycle_start=str(start),configured_next_reset=str(end),
        provider_reset_confirmed=False,counters=counters,known_watcher_usage=known,
        all_account_usage_known=False,
        caveat='Counters include only these watchers; missing counters, other crawlers and manual requests are not zero. '
        'NFL spread counter counts attempts at the one-credit maximum; this may exceed actual charged credits.')


def run(now=None):
    now=now or datetime.now(timezone.utc)
    cap=capacity(PROPOSAL)
    cache=json.loads((ROOT/'data/collection_budget/events_latest.json').read_text())
    events=cache['sports']['basketball_nba']['events']
    usage=current_usage(now)
    import psycopg2
    from nfl_pipeline.db import PG_DSN
    from automatic_price_checks import rows
    cycle_end=datetime.fromisoformat(usage['configured_next_reset']).replace(tzinfo=ET)
    with closing(psycopg2.connect(PG_DSN,connect_timeout=10)) as conn:
        conn.set_session(readonly=True)
        games=rows(conn,"""SELECT game_id,home_team,away_team,min(tip_utc) AS tip FROM nbaprops.game_projections
            WHERE tip_utc>%s AND tip_utc<%s GROUP BY game_id,home_team,away_team""",(now,cycle_end))
    result=clean(dict(built_at=now,status='assessment_only',proposal=PROPOSAL,capacity=cap,
        schedule=proposed_schedule(events,now,native_games=games,require_native=True,cycle_end=cycle_end),
        schedule_observed_at=cache.get('observed_at'),
        current_usage=usage,switch_review=switch_assessment(cap),
        tradeoffs=['Only NBA points would receive paid odds collection in this proposed design.',
            'Points is a capacity example, not a finding that this market is more profitable.',
            'NFL/NCAAF odds and other NBA prop markets would not fit unchanged alongside this schedule.',
            'Fewer polls may miss brief edges and exact closes; neither coverage nor 50 qualifying observations is guaranteed.',
            'The free plan says most bookmakers; required reference-book availability is unverified.',
            'Original experiments keep their evidence. This new sampling schedule needs its own prospective registration.'],
        pricing_source='https://the-odds-api.com/',quota_source='https://the-odds-api.com/liveapi/guides/v4/',
        betting_authorized=False))
    atomic_json(ROOT/'reports/low_credit_assessment_latest.json',result)
    lines=['# 500-credit collection assessment','',
        'Assessment only. No subscription, scheduled task, live polling rule or frozen registration was changed.','',
        '| Item | Maximum credits per cycle |','|---|---:|',
        f"| 62 games × 3 polls × 1 market × one group of books | {cap['primary_credits']} |",
        f"| One additional charged attempt per poll | {cap['maximum_retry_credits']} |",
        f"| Reserved credits | {cap['reserved_credits']} |",
        f"| Total | {cap['maximum_cycle_credits']} / 500 |",'',
        'Proposed collection: NBA points, at most two games/day and 62 games/provider cycle. '
        'Poll at T−40, T−20 and T−5. Pick games by schedule before seeing odds or results. '
        'Use the earlier matched native/provider schedule; omit unverified games. '
        'Hard request limits and a verified credit-cycle boundary would be required before activation.','',
        f"Recorded selected-watcher usage in the configured cycle: {result['current_usage']['known_watcher_usage']} credits/attempt budget; "
        'this is not a complete account audit.',result['current_usage']['caveat'],'',
        '## What changes and what remains unverified','']+['- '+s for s in result['tradeoffs']]
    lines+=['','## Example from the current cached schedule','',
        f"Cache timestamp: {cache.get('observed_at')}. This is a proposal, not the live scanner schedule."]
    for g in result['schedule']['selected_games']:
        times=', '.join(safe_time(t).astimezone(DENVER).strftime('%b %d %I:%M %p %Z') for t in g['poll_utc'])
        lines.append(f"- {g['away_team']} @ {g['home_team']}: {times}.")
    lines+=['','Switch-review blockers: '+', '.join(result['switch_review']['blockers'])+'.','',
        'Provider: [pricing](https://the-odds-api.com/) and '
        '[quota calculation](https://the-odds-api.com/liveapi/guides/v4/). '
        'The 500-credit free plan excludes historical odds; current event-list requests are free.']
    (ROOT/'reports/low_credit_assessment_latest.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    r=run();print(json.dumps(dict(status=r['status'],capacity=r['capacity'],switch_review=r['switch_review'])))
