import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling.receiving_uncertainty_repair import ReceivingTailRepair, improvement
from nfl_pipeline.modeling.challenger_models import curve_over,curve_summary


def sample():
    n=500
    f=pd.DataFrame(dict(game_id=[f'g{i}' for i in range(n)],player_id='p',position='WR',
        targets_avg_5=6.,receiving_yards=np.linspace(0,120,n)))
    v=np.tile(np.linspace(20,80,101),(n,1));w=np.full(v.shape,1/101)
    return f,v,w


def test_tail_transform_keeps_coherent_probabilities_and_median():
    f,v,w=sample();m=ReceivingTailRepair().fit(f,v,w)
    # No live change before explicit acceptance.
    np.testing.assert_allclose(m.transform(f,v,w)[0],v)
    new,mass=m.transform(f,v,w,1.)
    assert (np.diff(new,axis=1)>=0).all()
    np.testing.assert_allclose(mass.sum(axis=1),1.)
    np.testing.assert_allclose(curve_summary(new,mass)['median'],curve_summary(v,w)['median'])
    probabilities=np.array([curve_over(new,mass,np.full(len(f),line)) for line in (10,30,50,80,110)])
    assert (np.diff(probabilities,axis=0)<=0).all() and ((0<=probabilities)&(probabilities<=1)).all()
    assert np.mean(new[:,-1])>np.mean(v[:,-1])


def test_small_role_groups_shrink_to_global_and_duplicate_games_rejected():
    f,v,w=sample();f.loc[:5,'position']='RB'
    m=ReceivingTailRepair().fit(f,v,w)
    assert 'RB|normal' not in m.by_group
    f.loc[1,'game_id']=f.loc[0,'game_id']
    with pytest.raises(ValueError,match='unique player-games'):
        ReceivingTailRepair().fit(f,v,w)


def test_widening_intervals_alone_does_not_pass():
    before=dict(expected={'coverage_80':.72},probability={'brier':.23,'calibration_error':.03},interval_score=100)
    after=dict(expected={'coverage_80':.80},probability={'brier':.24,'calibration_error':.02},interval_score=99)
    assert not improvement(before,after)
    after['probability']['brier']=.22
    assert improvement(before,after)
    after['interval_score']=110
    assert not improvement(before,after)
