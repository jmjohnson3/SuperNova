import pandas as pd
from nfl_pipeline.repair_receiving_results import verified_receiving, verified_inactive


def fixture():
    row=dict(game_id='g',player_id='p',team_abbr='DEN',pfr_ids=['pfr'],existing_yards=None)
    snaps=pd.DataFrame([dict(game_id='g',pfr_player_id='pfr',team='DEN',offense_snaps=27)])
    pbp=pd.DataFrame([
        dict(game_id='g',play_id=1,desc='pass',posteam='DEN',play_type='pass',pass_attempt=1,
             complete_pass=1,receiver_player_id='other',receiving_yards=10.,lateral_receiver_player_id=None),
        dict(game_id='g',play_id=2,desc='END GAME',posteam=None,play_type=None,pass_attempt=0,
             complete_pass=0,receiver_player_id=None,receiving_yards=None,lateral_receiver_player_id=None)])
    pbp['sack']=0;pbp['two_point_attempt']=0
    return row,snaps,pbp,{('g','DEN'):(1,10)}


def test_zero_requires_positive_snaps_complete_pbp_and_reconciled_team_totals():
    args=fixture();proof,error=verified_receiving(*args)
    assert error is None and proof['receiving_yards']==0 and proof['offense_snaps']==27
    for snaps in (args[1].iloc[:0],args[1].assign(offense_snaps=0),args[1].assign(offense_snaps=None)):
        assert verified_receiving(args[0],snaps,args[2],args[3])[0] is None
    assert verified_receiving(args[0],args[1],args[2].iloc[:1],args[3])[0] is None
    assert verified_receiving(*args[:3],{('g','DEN'):(2,20)})[0] is None


def test_ambiguous_identity_laterals_and_incomplete_stats_stay_unresolved():
    row,snap,pbp,total=fixture()
    assert verified_receiving(dict(row,pfr_ids=['one','two']),snap,pbp,total)[0] is None
    assert verified_receiving(row,snap,pbp.assign(lateral_receiver_player_id='x'),total)[0] is None
    assert verified_receiving(row,snap,pbp.assign(receiving_yards=None),total)[0] is None


def test_existing_conflicting_result_is_never_overwritten():
    row,snap,pbp,total=fixture()
    assert verified_receiving(dict(row,existing_yards=10),snap,pbp,total)[1]=='existing_stat_conflict'
    pbp.loc[0,'receiver_player_id']='p'
    proof,error=verified_receiving(dict(row,existing_yards=10),snap,pbp,total)
    assert error is None and proof['receiving_yards']==10 and proof['targets']==1


def test_sacks_and_two_point_plays_are_not_official_passing_attempts():
    row,snap,pbp,total=fixture()
    sack=pbp.iloc[[0]].assign(play_id=1.1,sack=1,complete_pass=0,receiving_yards=None)
    conversion=pbp.iloc[[0]].assign(play_id=1.2,two_point_attempt=1,receiver_player_id='p')
    proof,error=verified_receiving(row,snap,pd.concat([pbp,sack,conversion]),total)
    assert error is None and proof['receiving_yards']==0


def test_inactive_needs_exact_week_status_and_published_participation():
    row,snap,pbp,_=fixture()
    snap['defense_snaps']=0;snap['st_snaps']=0
    other=snap.assign(pfr_player_id='another')
    assert not verified_inactive(row,other,pbp)
    assert verified_inactive(dict(row,roster_statuses=['INA']),other,pbp)
    assert not verified_inactive(dict(row,roster_statuses=['INA','ACT']),other,pbp)
    assert not verified_inactive(dict(row,roster_statuses=['INA']),snap,pbp)
    assert not verified_inactive(dict(row,roster_statuses=['INA']),other.iloc[:0],pbp)
