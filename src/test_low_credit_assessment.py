from copy import deepcopy
from datetime import datetime,timezone

import pytest

from low_credit_assessment import PROPOSAL,capacity,proposed_schedule,switch_assessment


def test_includes_full_cost_retries_and_reserve():
    c=capacity(PROPOSAL)
    assert (c['primary_credits'],c['maximum_retry_credits'],c['maximum_cycle_credits'],c['headroom'])==(186,186,472,28)
    assert c['fits_arithmetic'] and not c['activated']
    assert not switch_assessment(c)['ready_for_switch_review']


def test_an_extra_market_or_book_group_does_not_fit():
    config=deepcopy(PROPOSAL);config['markets'].append('player_rebounds')
    assert not capacity(config)['fits_arithmetic']
    config=deepcopy(PROPOSAL);config['books']=[str(n) for n in range(11)]
    assert capacity(config)['cost_per_nonempty_request']==2 and not capacity(config)['fits_arithmetic']


@pytest.mark.parametrize('key,value',[('max_retries_per_poll',-1),('max_games_per_cycle',0),('max_games_per_day',True),
    ('polls_before_tip_minutes',[40,40,5]),('polls_before_tip_minutes',[40,20,0]),('markets',[])])
def test_invalid_budget_assumptions_rejected(key,value):
    cfg=dict(PROPOSAL,**{key:value})
    with pytest.raises(ValueError):capacity(cfg)


def test_selection_ignores_price_model_and_results_and_deduplicates_events():
    now=datetime(2026,11,6,19,tzinfo=timezone.utc)
    games=[dict(id=k,commence_time='2026-11-07T00:00:00Z',price=1000 if k=='z' else -110) for k in ['z','b','a','a']]
    out=proposed_schedule(games,now)
    assert [r['event_id'] for r in out['selected_games']]==['a','b']
    assert out['omitted_games']==1 and not out['activated']
    games[-1]['commence_time']='2026-11-07T01:00:00Z'
    with pytest.raises(ValueError):proposed_schedule(games,now)


def test_no_calendar_assumption_can_exceed_hard_cycle_cap():
    now=datetime(2026,11,6,19,tzinfo=timezone.utc)
    games=[dict(id=str(n),commence_time=f'2026-12-{n+1:02}T00:00:00Z') for n in range(20)]
    assert len(proposed_schedule(games,now,dict(PROPOSAL,max_games_per_cycle=3))['selected_games'])==3


def test_close_plan_uses_earlier_native_tip_and_omits_later_billing_cycles():
    now=datetime(2026,11,6,19,tzinfo=timezone.utc)
    games=[dict(id='a',home_team='Atlanta Hawks',away_team='Washington Wizards',commence_time='2026-11-07T00:10:00Z'),
           dict(id='later',home_team='Atlanta Hawks',away_team='Washington Wizards',commence_time='2026-12-25T17:00:00Z')]
    native=[dict(game_id='nba-a',home_team='ATL',away_team='WAS',tip='2026-11-07T00:00:00Z')]
    result=proposed_schedule(games,now,native_games=native,require_native=True)
    assert len(result['selected_games'])==1
    assert result['selected_games'][0]['poll_utc'][-1]=='2026-11-06T23:55:00+00:00'
    assert proposed_schedule(games,now,require_native=True)['selected_games']==[]
