from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling.chronological_calibration import forward_folds, equal_player_game_weights
from nfl_pipeline.modeling import fit_market_calibration as fit


def rows():
    return pd.DataFrame([
        dict(season=2025,week=18,key='a',locked_at='2026-01-04T18:00Z',label_available_at='2026-01-05T12:00Z'),
        dict(season=2026,week=1,key='b',locked_at='2026-09-09T18:00Z',label_available_at='2026-09-10T12:00Z'),
        dict(season=2026,week=2,key='c',locked_at='2026-09-17T18:00Z',label_available_at='2026-09-18T12:00Z'),
    ])


def test_folds_are_season_aware_forward_only():
    folds=list(forward_folds(rows()))
    assert len(folds)==2
    assert folds[0][0].key.tolist()==['a'] and folds[0][1].key.tolist()==['b']
    assert folds[1][0].key.tolist()==['a','b'] and folds[1][1].key.tolist()==['c']
    for train,test,_ in folds:
        assert set(train.key).isdisjoint(test.key)
        assert pd.to_datetime(train.label_available_at,utc=True).max()<pd.to_datetime(test.locked_at,utc=True).min()


def test_unknown_late_and_overlapping_labels_are_purged():
    data=rows();data.loc[0,'label_available_at']=None
    data.loc[1,'label_available_at']='2026-10-01T00:00Z'
    assert all(train.empty for train,_,_ in forward_folds(data))
    data=rows();data.loc[2,'key']='a'
    assert list(forward_folds(data))[-1][0].key.tolist()==['b']


def test_each_player_game_has_one_unit_of_weight():
    data=pd.DataFrame({'key':['a','a','a','b']})
    w=equal_player_game_weights(data)
    assert np.allclose(w.groupby(data.key).sum(),1.)


def test_missing_actual_cannot_become_a_training_loss():
    data=rows();data['actual']=[np.nan, 40., 50.]
    folds=list(forward_folds(data))
    assert folds[0][0].empty and folds[1][0].key.tolist()==['b']


def test_asymmetric_range_uses_actual_median():
    data=pd.DataFrame(dict(p10=[0.],p50=[10.],p90=[40.],actual=[55.]))
    scale,coverage=fit.range_scale_from_summary(data)
    assert scale==1.5 and coverage==1.


def test_cohorts_cannot_be_pooled():
    data=pd.DataFrame(dict(stat=['receiving_yards']*2,model_version=['v1','v2'],
        scoring_version=['code']*2,variant=['production']*2))
    with pytest.raises(ValueError,match='one model'):
        fit.fit_captured_stat('receiving_yards',data)


def test_offline_calibrator_cannot_overwrite_production(monkeypatch):
    monkeypatch.setattr('sys.argv',['fit_market_calibration','--write'])
    with pytest.raises(SystemExit) as error:
        fit.main()
    assert error.value.code==2


def test_rescore_replays_archived_code_keeps_median_pushes_and_original_inputs(monkeypatch):
    cap=dict(stat='receiving_yards',projection=40.,metrics={'old':'keep'},
        offer=dict(over_price=-110,under_price=-110),scoring_fingerprint='original')
    original=deepcopy(cap)
    calls=[]
    def fake(revised):
        calls.append(revised)
        return dict(side='under',probability=.7,projection=40.,projection_p10=0.,projection_p50=25.,
            projection_p90=90.,probability_trace={'raw_over':.4})
    monkeypatch.setattr(fit,'replay',fake)
    data=pd.DataFrame([dict(id=1,captured=cap,season=2026,week=2,key='g|p',line=40.,actual=40.)])
    result=fit.rescore(data,{'receiving_yards':{'probability_trust':.25}})
    assert result.iloc[0].p_over==pytest.approx(.3)
    assert result.iloc[0].p50==25. and result.iloc[0].is_push
    assert calls[0]['scoring_fingerprint']=='original' and cap==original
    assert calls[0]['metrics']['market_calibration']['stats']['receiving_yards']['probability_trust']==.25


def test_receptions_without_label_provenance_cannot_fit(tmp_path):
    p=tmp_path/'lines.csv'
    pd.DataFrame([dict(game_id='2026_01_A_B',player_id='p',actual=5,line=4.5)]).to_csv(p,index=False)
    assert fit.fit_receptions(p)['status']=='missing_lock_or_label_provenance'


def test_full_path_fold_parameters_do_not_use_later_outcomes(monkeypatch):
    def replay(cap):
        parameters=cap['metrics']['market_calibration']['stats']['receiving_yards']
        projection=30.5+parameters['projection_trust']*(cap['projection']-30.5)
        scale=parameters['range_scale']
        probability=float(fit._blend(.65,.5,parameters['probability_trust']))
        return dict(side='over',probability=probability,projection=projection,
            projection_p10=max(0.,projection-20*scale),projection_p50=projection,
            projection_p90=projection+40*scale,probability_trace={'raw_over':.65})
    monkeypatch.setattr(fit,'replay',replay)
    records=[]
    for week in (1,2):
        for player in range(21):
            records.append(dict(id=week*100+player,stat='receiving_yards',season=2026,week=week,
                key=f'{week}|{player}',locked_at=f'2026-09-{week*7:02}T18:00Z',
                label_available_at=f'2026-09-{week*7+1:02}T12:00Z',model_version='release',
                scoring_version='code',variant='production_curve',line=30.5,
                actual=45. if player<14 else 15.,side='over',
                captured=dict(stat='receiving_yards',projection=40.,metrics={},
                    offer=dict(over_price=-110,under_price=-110))))
    frame=pd.DataFrame(records)
    first=fit.fit_captured_stat('receiving_yards',frame)
    frame.loc[frame.week.eq(2),'actual']=100.
    second=fit.fit_captured_stat('receiving_yards',frame)
    assert first['status']=='offline_evaluated' and len(first['rows'])==21
    for column in ('p_over','projection_trust','range_scale','probability_trust'):
        assert [r[column] for r in first['rows']]==[r[column] for r in second['rows']]
    assert first['evidence']['forward_folds'][0]['train_periods']==[(2026,1)]
    assert not first['deployment_approved']
