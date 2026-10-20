import pandas as pd
import pytest

from nfl_pipeline.modeling.receiving_market_blend import selected_comparison, challenger_captures


def offers():
    return pd.DataFrame([dict(prediction_id=i,game_id='g',player_id=str(i),model_version='v',scoring_version='c',
        season=2026,week=4,day='2026-10-04',batch='2026-10-04T15:00Z',side='over',line=40.5,
        probability=.6,market=.5,payout=1.,push_probability=0.,actual=actual,outcome=outcome,
        eligible=True,exact_micro=(i==1),edge=1.,confidence=.5,uncertainty_proxy=.5)
        for i,actual,outcome in [(1,20.,0.),(2,60.,1.),(3,None,None)]])


def replay_rows():
    return [dict(id=i,p_over=p,raw_probability=p,p10=0.,p90=80.,projection=40.,is_push=False)
        for i,p in [(1,.55),(2,.75),(3,.9)]]


def test_comparison_preserves_fixed_ids_and_ranks_pending_without_outcome_access():
    frame=offers()
    report=selected_comparison(replay_rows(),frame,{2})
    assert report['fixed_research']['ids']==[2]
    assert report['exact_micro']['ids']==[1]
    assert report['blend_ranked']['ids'][0]==3
    assert report['blend_ranked']['pending']==1
    assert report['all_offers']['settled']==2
    frame['actual']=999.;frame['outcome']=1.
    changed=selected_comparison(replay_rows(),frame,{2})
    assert report['blend_ranked']['ids']==changed['blend_ranked']['ids']
    assert report['conservative_blend_ranked']['ids']==changed['conservative_blend_ranked']['ids']


def test_invalid_artifact_cannot_be_reused_as_a_receiving_challenger():
    with pytest.raises(ValueError,match='contract'):
        challenger_captures(pd.DataFrame(),[],{'contract':'wrong'},'hash')


def test_no_forward_rows_does_not_fabricate_evidence():
    assert selected_comparison([],offers(),set())['status']=='no_forward_test_rows'
