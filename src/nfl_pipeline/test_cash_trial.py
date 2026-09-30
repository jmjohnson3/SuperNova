from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json

import pytest

from nfl_pipeline import cash_trial_policy as policy
from nfl_pipeline import cash_execution as execution
from nfl_pipeline import cash_readiness as readiness
from nfl_pipeline import cash_publication

NOW = datetime(2026,9,27,16,tzinfo=timezone.utc)


def sample(n=50):
    rows = []
    for i in range(n):
        win = i % 10 < 8
        rows.append(dict(forecast_id=i, decision_key=str(i), week_key=f'2026:{i//10}',
            player_id=str(i%10), team=str(i%5), result='win' if win else 'loss', outcome=float(win),
            probability=.8, production_probability=.5, market_probability=.5, profit=1 if win else -1,
            covered_80=i%10<8, clv=.02, stale_close=False, locked_at=f'{i:03d}'))
    return rows


def reservation(**overrides):
    return dict(dict(decision_key='one', day='2026-09-27',week_key='2026:3',stake=1.,
                     expires_at=(NOW+timedelta(minutes=10)).isoformat(),state='published'), **overrides)


def test_readiness_does_not_confuse_minimum_sample_with_proof():
    rows=sample()
    yes=policy.evaluate(rows, rows, 50)
    assert yes['status']=='cash_trial_eligible'
    assert yes['metrics']['alpha_per_test'] < .0011
    for r in rows:
        r.update(probability=.95, profit=-1, clv=None)
    no=policy.evaluate(rows, rows, 50)
    assert no['status']=='research'
    assert 'positive_roi_unconfirmed' in no['blockers']
    assert 'selected_calibration_above_five_percent' in no['blockers']
    assert no['metrics']['average_clv'] is None


def test_pushes_return_stake_but_are_not_binary_losses():
    rows=sample();rows[0].update(result='push', outcome=None, profit=0)
    report=policy.evaluate(rows,rows,50)
    assert report['metrics']['decisions']==50 and report['metrics']['binary']==49
    assert report['metrics']['roi']==pytest.approx(.58)


def test_duplicate_rows_concentration_and_missing_inputs_fail():
    rows=sample()
    for r in rows:
        r.update(decision_key='same', week_key='one', player_id='one', team=None, covered_80=None)
    report=policy.evaluate(rows,rows,50,unresolved=1)
    assert 'duplicate_selected_decision' in report['blockers']
    assert 'independent_weeks_below_three' in report['blockers']
    assert 'team_concentration' in report['blockers']
    assert 'distribution_coverage_not_confirmed' in report['blockers']


def test_unchanged_close_counts_separately_from_unknown():
    rows=sample();rows[0]['clv']=0;rows[1]['clv']=None
    m=policy.evaluate(rows, rows, 50)['metrics']
    assert m['unchanged_closes']==1 and m['valid_close_coverage']==.98
    assert m['clv_beat_rate']==pytest.approx(48/49)


def test_global_budget_covers_both_games_and_props_and_unknown_execution():
    candidate=reservation(decision_key='next')
    assert policy.budget_error([reservation(decision_key=str(i)) for i in range(5)],candidate,NOW)=='daily_limit'
    week=[reservation(decision_key=str(i),day=str(i),state='confirmed') for i in range(20)]
    assert policy.budget_error(week,candidate,NOW)=='weekly_stake_limit'
    assert policy.budget_error([reservation(expires_at=(NOW-timedelta(seconds=1)).isoformat())],candidate,NOW)=='execution_reconciliation_required'
    assert policy.budget_error([reservation(state='settled',profit=-30)],candidate,NOW)=='cumulative_loss_limit'
    assert policy.budget_error([reservation(state='paused',profit=10)],candidate,NOW)=='cash_trial_paused'
    assert policy.budget_error([reservation(state='confirmed',expires_at=(NOW-timedelta(days=2)).isoformat())],candidate,NOW)=='confirmed_result_reconciliation_required'
    assert policy.budget_error([reservation(state='not_placed')],candidate,NOW) is None
    assert policy.budget_error([reservation(state='not_placed')],reservation(),NOW)=='decision_already_reserved'


def test_execution_requires_explicit_confirmation_and_grades_actual_price():
    row=reservation(forecast={'probability':.6})
    with pytest.raises(ValueError):
        execution.transition(row,'settled',NOW,result='win')
    filled=execution.transition(row,'confirmed',NOW,price=120)
    settled=execution.transition(filled,'settled',NOW,result='win')
    assert settled['profit']==1.2 and not filled['execution_deviation']
    late=execution.transition(row,'confirmed',NOW+timedelta(hours=1),price=-200)
    assert late['execution_deviation']
    with pytest.raises(ValueError):
        execution.transition(filled,'not_placed',NOW)


def test_legacy_micro_and_bankroll_labels_are_not_authority():
    for tier in ('bankroll','micro_projection','locked_micro'):
        row=cash_publication.research_rows([{'tier':tier}])[0]
        assert row['tier']=='paper' and not row['cash_eligible']


def test_checkpoint_is_immutable_and_does_not_reapprove_after_peeking(tmp_path):
    registration={'sha256':'registered'}; rows=sample()
    first=readiness.review(registration,rows,rows,0,NOW,store=tmp_path)
    assert first['status']=='cash_trial_eligible'
    content=(tmp_path/'reviews'/'50.json').read_bytes()
    rows[0]['profit']=-999
    second=readiness.review(registration,rows,rows,0,NOW+timedelta(days=1),store=tmp_path)
    assert (tmp_path/'reviews'/'50.json').read_bytes()==content
    assert second['status']=='paused'
    assert readiness.review(registration,sample(),sample(),0,NOW,store=tmp_path)['status']=='paused'


def test_register_cannot_repin_under_existing_policy(tmp_path):
    cfg=dict(trial_sha256='trial',release='release',scoring_version='score',variant='ensemble',artifact_sha256='artifact')
    doc=policy.register(cfg,store=tmp_path)
    assert policy.load_registration(tmp_path)==doc
    with pytest.raises(ValueError):
        policy.register(dict(cfg,scoring_version='new'),store=tmp_path)


def test_pending_selected_rows_prevent_consuming_checkpoint(tmp_path):
    result=readiness.review({'sha256':'x'},sample(),sample(),1,NOW,store=tmp_path)
    assert result['status']=='research' and result['next_checkpoint']==50
    assert not (tmp_path/'reviews'/'50.json').exists()


def test_pending_future_games_do_not_pause_a_passing_strategy(tmp_path):
    rows=sample();registration={'sha256':'x'}
    assert readiness.review(registration,rows,rows,0,NOW,store=tmp_path)['status']=='cash_trial_eligible'
    still=readiness.review(registration,rows,rows,0,NOW,store=tmp_path,pending=10,completed_weeks=set())
    assert still['status']=='cash_trial_eligible'


def test_candidate_checks_same_probability_for_ev_and_price():
    from nfl_pipeline.offer_selection import CONTRACT
    row=dict(book='fanduel',link='https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=2',
        execution_contract=CONTRACT,quote_fetched_at_utc=(NOW-timedelta(minutes=1)).isoformat(),
        prediction_context_cutoff_utc=NOW.isoformat(),start_ts_utc=(NOW+timedelta(hours=1)).isoformat(),
        probability=.6,push_probability=0,price=-150,line=45.5,side='over',drift_guard_pass=True,market_probability=.5)
    assert execution.candidate_error(row,NOW)=='nonpositive_current_ev'
    row['price']=-149
    assert execution.candidate_error(row,NOW) is None
    row['quote_fetched_at_utc']=(NOW-timedelta(hours=2)).isoformat()
    assert execution.candidate_error(row,NOW)=='stale_quote_refresh_required'
