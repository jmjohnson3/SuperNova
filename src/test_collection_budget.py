from datetime import datetime, timezone
from decimal import Decimal

import pytest

from collection_budget import economics,forecast


def test_cost_scenarios_do_not_infer_available_prices_or_guaranteed_edge():
    out=economics([dict(day='2026-10-23',distinct_offers=20),dict(day='2026-10-28',distinct_offers=27)],30)
    middle=out['scenarios'][1]
    assert out['calendar_days']==6
    assert Decimal(out['assumed_monthly_opportunities'])==235
    assert middle['required_monthly_bets']==1000
    assert Decimal(middle['net_if_all_observed_opportunities_available'])==Decimal('-22.95')


def test_no_offers_leaves_monthly_net_unknown():
    result=economics([],30)
    assert result['assumed_monthly_opportunities'] is None
    assert all(r['net_if_all_observed_opportunities_available'] is None for r in result['scenarios'])


def test_registered_period_includes_quiet_days_and_excludes_other_dates():
    out=economics([dict(day='2026-10-27',distinct_offers=100),dict(day='2026-10-29',distinct_offers=5)],30,
                  period_start='2026-10-29',period_end='2026-11-05',cohort='frozen-hash')
    assert out['calendar_days']==8 and out['observed_offers']==5
    assert Decimal(out['assumed_monthly_opportunities'])==Decimal('18.75')
    assert out['cohort']=='frozen-hash'
    assert Decimal(out['scenarios'][1]['maximum_monthly_cost_if_all_offers_available'])==Decimal('.5625')


def test_zero_offer_registered_period_is_zero_volume_not_missing_period():
    out=economics([],30,period_start='2026-11-01',period_end='2026-11-05',cohort='frozen')
    assert out['calendar_days']==5 and Decimal(out['assumed_monthly_opportunities'])==0
    assert Decimal(out['scenarios'][1]['net_if_all_observed_opportunities_available'])==-30
    assert out['edge_needed_at_observed_rate'] is None


def test_nfl_spreads_are_counted_once_per_bulk_poll_not_per_game():
    now=datetime(2026,11,6,17,tzinfo=timezone.utc)
    sports={key:dict(events=[],headers={'x-requests-remaining':'1000'}) for key in
            ('basketball_nba','americanfootball_nfl','americanfootball_ncaaf')}
    sports['americanfootball_nfl']['events']=[dict(commence_time='2026-11-06T18:00:00Z')]*2
    result=forecast(dict(observed_at=now.isoformat(),sports=sports),dict(days=[]),now)
    assert result['watcher_credit_estimates']['nfl_spread_credit_ceiling']==12
    assert result['watcher_credit_estimates']['nfl_sharp_credit_ceiling']==96


@pytest.mark.parametrize('cost,stake',[(-1,1),(30,0),(float('nan'),1),(30,float('inf'))])
def test_invalid_cost_inputs_are_rejected(cost,stake):
    with pytest.raises(ValueError):economics([],cost,stake)


def test_bulk_line_cost_is_shared_and_past_events_are_not_new_demand():
    now=datetime(2026,10,29,17,tzinfo=timezone.utc)
    games=[dict(id='a',commence_time='2026-10-30T00:10:00Z'),dict(id='b',commence_time='2026-10-30T00:10:00Z'),
           dict(id='past',commence_time='2026-10-29T00:10:00Z')]
    sports={key:dict(events=[],headers={'x-requests-remaining':'1000'}) for key in
            ('basketball_nba','americanfootball_nfl','americanfootball_ncaaf')}
    sports['basketball_nba']['events']=games
    data=dict(observed_at=now.isoformat(),sports=sports)
    result=forecast(data,dict(days=[]),now)
    assert result['counts_by_day']['basketball_nba']=={'2026-10-29':2}
    assert result['watcher_credit_estimates']['nba_prop_credit_ceiling']==40
    assert result['line_polls']['basketball_nba']==3
    assert result['watcher_credit_estimates']['shared_bulk_line_credits']==9
    assert result['after_watchers_and_150_floor']==801
    assert result['balance_is_recent'] is True


def test_old_balance_is_explicitly_not_recent():
    sports={key:dict(events=[],headers={'x-requests-remaining':'1000'}) for key in
            ('basketball_nba','americanfootball_nfl','americanfootball_ncaaf')}
    result=forecast(dict(observed_at='2026-10-29T15:00:00Z',sports=sports),dict(days=[]),
                    datetime(2026,10,29,17,tzinfo=timezone.utc))
    assert result['balance_is_recent'] is False
