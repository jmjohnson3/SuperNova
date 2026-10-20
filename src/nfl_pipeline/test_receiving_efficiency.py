from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling.receiving_efficiency import ReceivingEfficiency, forecast, frozen_opportunity
from nfl_pipeline.modeling.train_receiving_efficiency import decisions


def frame():
    i=np.arange(200)
    targets=(i%9+1).astype(float)
    return pd.DataFrame(dict(game_id=i.astype(str),player_id='p',position=np.where(i%2,'WR','RB'),
        targets=targets,receiving_yards=targets*7+(i%3),n_games_prev_10=10,
        targets_avg_10=5.,targets_avg_5=5.,targets_avg_3=5.,
        receiving_yards_avg_10=35.,receiving_yards_avg_5=35.,receiving_yards_avg_3=35.,
        receptions_avg_10=3.,receptions_avg_5=3.,receptions_avg_3=3.,
        receiving_air_yards_avg_10=45.,receiving_air_yards_avg_5=45.,receiving_air_yards_avg_3=45.,
        receiving_yards_after_catch_avg_10=12.,receiving_yards_after_catch_avg_5=12.,receiving_yards_after_catch_avg_3=12.))


def test_rate_features_do_not_read_current_results_or_market():
    data=frame(); model=ReceivingEfficiency().fit(data)
    expected=model.predict(data)
    changed=data.copy()
    for key in ('targets','receiving_yards','receptions','receiving_air_yards','offense_snaps',
                'line','price','probability','actual','team_actual_pass_attempts','injury_status'):
        changed[key]=9999
    np.testing.assert_array_equal(model.features(data),model.features(changed))
    np.testing.assert_array_equal(expected,model.predict(changed))


def test_unknown_player_history_uses_prior_without_claiming_observed_zero():
    data=frame(); model=ReceivingEfficiency().fit(data)
    data.loc[0,'receiving_yards_avg_10']=np.nan
    values,known=model.prior(data)
    assert values[0]==model.position_rates['RB']
    assert not known[0]
    assert model.features(data).loc[0,'ypt_10__missing']==1


def test_low_exposure_player_is_shrunk_more_strongly():
    data=frame(); model=ReceivingEfficiency().fit(data)
    data.loc[:1,'receiving_yards_avg_10']=100.
    data.loc[:1,'position']='WR'
    data.loc[0,'n_games_prev_10']=1
    data.loc[1,'n_games_prev_10']=10
    rate,_=model.prior(data)
    assert model.position_rates['WR'] < rate[0] < rate[1] < 20.


def test_training_rejects_duplicate_player_games_and_unknown_exposure():
    data=frame()
    with pytest.raises(ValueError,match='unique player-games'):
        ReceivingEfficiency().fit(pd.concat([data,data.iloc[:1]]))
    data.targets=np.nan
    with pytest.raises(ValueError,match='positive target exposure'):
        ReceivingEfficiency().fit(data)


class Reference:
    def predict(self,data):
        return np.full(len(data),30.)


def test_rate_repair_preserves_targets_and_distribution_shape():
    data=frame().iloc[:5].copy(); model=ReceivingEfficiency().fit(frame())
    bundle=dict(reference=Reference(),model=model,reference_errors=np.array([-20.,0.,30.]),
        residual_weight=.5,alpha=.5)
    c=forecast(bundle,data)
    np.testing.assert_array_equal(c['targets'],data.targets_avg_5)
    np.testing.assert_allclose(np.diff(c['candidate'],axis=1),np.diff(c['reference'],axis=1))
    assert np.isfinite(c['candidate']).all()
    np.testing.assert_allclose(c['weights'].sum(axis=1),1.)
    over=[((c['candidate']>line)*c['weights']).sum(axis=1) for line in (10.5,30.5,60.5)]
    assert np.all(np.diff(over,axis=0)<=0)
    off=deepcopy(bundle); off['alpha']=0.
    nochange=forecast(off,data)
    np.testing.assert_array_equal(nochange['candidate'],nochange['reference'])
    data.loc[0,'targets_avg_5']=np.nan
    with pytest.raises(ValueError,match='frozen target'):
        frozen_opportunity(data)


def test_point_gain_is_not_vetoed_by_probability_or_cash():
    rows=[]; lines=[]
    for variant in ('reference','rolling_baseline','rate_repair'):
        for i in range(30):
            actual=float(20+i%3); center=actual+(0 if variant=='rate_repair' else 3)
            rows.append(dict(variant=variant,game_id=str(i),player_id='p',season=2025,week=i//5,
                actual=actual,mean=center,median=center,p10=center-10,p90=center+10))
            lines.append(dict(variant=variant,player_game=str(i),season=2025,week=i//5,line=20.5,
                outcome=float(i%2),probability=.99 if variant=='rate_repair' else .5,weight=1.))
    _,_,result=decisions(pd.DataFrame(rows),pd.DataFrame(lines))
    assert result['expected_mean']['historical_screen_passed']
    assert result['median']['historical_screen_passed']
    assert not result['probability']['historical_screen_passed']
    assert not result['expected_mean']['deployment_approved']
    assert not result['expected_mean']['betting_approved']
