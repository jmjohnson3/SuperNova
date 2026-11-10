from copy import deepcopy
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from edge_research import market,rotation,rotation_study
from nba_props import research_models as base
from nba_props.test_research import Fake,histories
from sharp_pipeline.test_sharp_multi import NOW,event,bookmaker,lines


def test_consensus_excludes_fanduel_and_never_converts_lines():
    e=event([bookmaker('fanduel',lines(home_spread=160,away_spread=-190)),
        bookmaker('pinnacle',lines()),bookmaker('draftkings',lines()),bookmaker('betmgm',lines(spread_home=-7.5))])
    offers=market.offers(e,'nba',NOW)
    spread=next(o for o in offers if o['market']=='spread')
    assert spread['consensus']['probability']==pytest.approx(.5)
    assert {q['book'] for q in spread['consensus']['sources']}=={'pinnacle','draftkings'}
    assert market.selection(spread,.5)['selected']
    e['bookmakers']=[e['bookmakers'][0],e['bookmakers'][1]]
    assert all(o['consensus'] is None for o in market.offers(e,'nba',NOW))


def test_market_clock_future_quotes_and_conflicts_fail_closed():
    books=[bookmaker(b,lines()) for b in ('fanduel','draftkings','pinnacle')]
    books[1]['markets'][0]['last_update']=(NOW-timedelta(hours=1)).isoformat()
    books[1]['markets'][1]['last_update']=(NOW-timedelta(hours=1)).isoformat()
    e=event(books);spread=next(o for o in market.offers(e,'nba',NOW) if o['market']=='spread')
    assert spread['consensus'] is None
    assert market.source_stamp(NOW.isoformat(),'bad clock',NOW) is None
    assert market.source_stamp(NOW.isoformat(),(NOW+timedelta(seconds=1)).isoformat(),NOW) is None
    conflict=deepcopy(books[0]['markets'][1]);conflict['outcomes'][0]['price']=123
    e['bookmakers'][0]['markets'].append(conflict)
    assert not any(o['market']=='spread' for o in market.offers(e,'nba',NOW))


def test_reference_skew_is_not_hidden_by_fresh_execution_quote():
    e=event([bookmaker('fanduel',lines()),bookmaker('draftkings',lines(),NOW-timedelta(seconds=121)),bookmaker('pinnacle',lines())])
    for m in e['bookmakers'][1]['markets']:m['last_update']=(NOW-timedelta(seconds=121)).isoformat()
    assert all(o['consensus'] is None for o in market.offers(e,'nba',NOW))


def test_first_line_and_side_selection_are_deterministic():
    e=event([bookmaker('fanduel',lines(home_spread=150,away_spread=-180)),bookmaker('draftkings',lines()),bookmaker('pinnacle',lines())])
    o=next(o for o in market.offers(e,'nba',NOW) if o['market']=='spread')
    assert market.selection(o,.8)['side']=='home'
    away=market.selection(o,.1)
    assert away['side']=='away' and away['line']==6.5
    assert not market.selection(o,.8,eligible=False)['selected']


def test_model_weight_can_be_zero_and_does_not_pool_small_samples():
    rows=[dict(event_id=str(i),day=pd.Timestamp('2025-01-01')+pd.Timedelta(days=i//4),reference=.5,
        model=.9 if i%2 else .1,outcome=i%2) for i in range(120)]
    good=pd.DataFrame(rows)
    assert market.fit_weight(good)['weight']>0
    bad=good.assign(model=1-good.model)
    assert market.fit_weight(bad)['weight']==0
    assert market.fit_weight(good.iloc[:20])['status']=='insufficient_earlier_evidence'
    assert market.blend(.55,.99,0)==pytest.approx(.55)


def team():
    p,t=histories();ps,ts,budgets=rotation_study.states(p,t)
    target=p[p.game_id=='10'].copy();f=rotation_study.features(target,ps,ts,budgets)
    # Ten synthetic players for the same rotation, with known pregame availability.
    f['team']='BOS';f['confirmed_starter']=np.nan;f['minutes_cap']=np.nan;f['availability']='active'
    fit=dict(minutes=Fake(),rates={s:Fake(True) for s in ('reb','ast','fg3a')})
    return fit,f,dict(fga=90.,fg3a=35.,fta=20.,ast=24.,reb=43.)


def test_joint_minutes_caps_absences_and_opportunities_are_conserved():
    fit,f,b=team();f.loc[0,'availability']='out';f.loc[1,'minutes_cap']=18
    result=rotation.scenarios(fit,f,b,[.9,.08,.02],'game')
    v=result['values'];w=result['weights']
    assert w.sum()==pytest.approx(1)
    assert np.all(v['minutes'][0]==0) and np.all(v['minutes'][1]<=18+1e-8)
    budgets=np.tile([240,265,290],rotation.SPEC['rotations'])
    assert np.allclose(v['minutes'].sum(axis=0),budgets)
    assert np.allclose(v['fga'].sum(axis=0),90*budgets/240)
    assert np.all(v['fg3a']<=v['fga']+1e-8)
    assert np.all(result['role'].sum(axis=0)==5)
    assert result['selection_eligible']
    second=rotation.scenarios(fit,f,b,[.9,.08,.02],'game')
    assert np.array_equal(v['minutes'],second['values']['minutes'])


def test_unknown_teammates_cannot_become_verified_and_capacity_is_not_invented():
    fit,f,b=team();f.loc[0,'availability']='uncertain'
    assert not rotation.scenarios(fit,f,b,[1.,0.,0.],'g')['selection_eligible']
    f['minutes_cap']=10
    with pytest.raises(ValueError):rotation.scenarios(fit,f,b,[1.,0.,0.],'g')


def test_workload_mixture_differs_from_pricing_the_average_and_remains_monotone():
    d=dict(a=1.,b=0.);means=np.array([1.,9.]);weights=np.array([.5,.5])
    mixture=rotation.mixture_over(means,weights,4.5,d)
    collapsed=float(base.over(5.,4.5,d))
    assert abs(mixture-collapsed)>.05
    prices=[rotation.mixture_over(means,weights,line,d) for line in np.arange(.5,15,.5)]
    assert np.all(np.diff(prices)<=0)


def test_historical_roster_does_not_use_current_game_participation():
    p,t=histories();p['game_date']=pd.to_datetime(p.game_date)+pd.DateOffset(years=1)
    t['game_date']=pd.to_datetime(t.game_date)+pd.DateOffset(years=1)
    before=rotation_study.historical_pool(p,t)
    altered=p[~((p.game_id=='10')&(p.player_id==10))].copy()
    after=rotation_study.historical_pool(altered,t)
    pd.testing.assert_frame_equal(before[before.game_id=='10'].reset_index(drop=True),after[after.game_id=='10'].reset_index(drop=True))
