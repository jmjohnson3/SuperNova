import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nba_props import count_distribution_research as research


def test_overdispersed_counts_keep_integer_support_and_target_moments():
    mu, a = 10., 3.
    size = mu/(a-1)
    probability = size/(size+mu)
    assert stats.nbinom.mean(size,probability) == pytest.approx(mu)
    assert stats.nbinom.var(size,probability) == pytest.approx(a*mu)
    assert research.count_over(mu,10.5,a,0) == pytest.approx(stats.nbinom.sf(10,size,probability))
    values=research.count_over(mu,np.arange(0.5,30,1),a,0)
    assert np.all(np.diff(values)<0)


def test_poisson_limit_zero_mean_and_probability_bounds():
    assert research.count_over(10,10.5,1,0) == pytest.approx(stats.poisson.sf(10,10))
    assert research.count_over(10,10.5,.7,0) == pytest.approx(stats.poisson.sf(10,10))
    assert research.count_over(0,.5,3,0) == 0
    probabilities=research.count_over([1,20,50],[.5,19.5,49.5],[2,3,4],[.1,.02,0])
    assert np.all((probabilities>=0)&(probabilities<=1))


@pytest.mark.parametrize('mu,a,b',[(-1,2,0),(np.nan,2,0),(1,0,0),(1,2,-.1)])
def test_invalid_distribution_is_rejected(mu,a,b):
    with pytest.raises(ValueError):research.count_over(mu,2.5,a,b)


def fixtures():
    p=dict(game_id='0022500001',player_id=1,player_name='Test Player',game_date='2025-11-01')
    for s in research.SPEC['markets']:
        p.update({s:10,'pred_'+s:12,'disp_a_'+s:2,'disp_b_'+s:0})
    q=dict(quote_id=1,event_id='event',home_team='Boston Celtics',away_team='New York Knicks',
        player_name='Test Player',stat='points',line=9.5,over_price=-110,under_price=-110,
        commence='2025-11-02T00:00:00Z',fetched_at_utc='2025-11-01T23:30:00Z',
        updated_at_utc='2025-11-01T23:29:00Z')
    identity=dict(game_id='0022500001',player_id=1,team='BOS',opponent='NYK',home=True)
    return pd.DataFrame([p]),pd.DataFrame([q]),pd.DataFrame([identity])


def test_real_offer_matches_player_game_and_keeps_original_quote_id():
    frame=research.matched_rows(*fixtures())
    assert len(frame)==1 and frame.iloc[0].quote_id==1
    assert frame.iloc[0].game_id=='0022500001'


@pytest.mark.parametrize('issue',['postgame','stale','future_update','wrong_game','integer_line','conflicting_quote'])
def test_invalid_market_evidence_is_excluded(issue):
    p,q,i=fixtures()
    if issue=='postgame':q['fetched_at_utc']='2025-11-02T01:00:00Z'
    elif issue=='stale':q['fetched_at_utc']=q['updated_at_utc']='2025-10-30T23:00:00Z'
    elif issue=='future_update':q['updated_at_utc']='2025-11-01T23:40:00Z'
    elif issue=='wrong_game':q['home_team']='Phoenix Suns'
    elif issue=='integer_line':q['line']=10
    else:q=pd.concat([q,q.assign(quote_id=2,over_price=120)],ignore_index=True)
    assert research.matched_rows(p,q,i).empty


def test_selection_uses_executable_odds_and_single_side():
    frame=pd.DataFrame(dict(game_id=['g','g'],game_date=[pd.Timestamp('2025-11-01')]*2,
        over_price=[-110,-110],under_price=[-110,-110],won_over=[True,True],p_count=[.8,.2]))
    result=research.selection(frame,'p_count')
    assert result['selections']==2
    assert result['paper_profit']==pytest.approx(100/110-1)
    assert result['roi_by_game']['clusters']==1
    assert result['roi_by_game']['lower_95'] is None


def test_no_review_pass_from_one_game():
    frame=research.matched_rows(*fixtures())
    result=research.evaluate(frame)
    assert result['decision']=='do_not_promote'
    assert result['betting_authorized'] is False
