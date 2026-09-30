from copy import deepcopy
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.context_contract import context_evidence
from nfl_pipeline.modeling.component_validation import WorkloadTailCalibration, output_gates, calibrate_curve
from nfl_pipeline.modeling.challenger_models import curve_summary, curve_over
from nfl_pipeline.modeling.score_accuracy_components import (
    empirical_residuals, full_path, score_records, snapshot_before,
)
from nfl_pipeline.modeling.scoring_capture import capture
from nfl_pipeline.modeling.team_workload import (
    COUNTS, YARDS, TeamAllocationModel, prepare_history, roster_features, team_features,
)
from nfl_pipeline.modeling.train_accuracy_components import prospective_outputs
from nfl_pipeline.test_accuracy_challengers import locked_record, scoring_fixture


def contract_rows():
    rows = pd.DataFrame({
        'actual': [10.] * 200, 'reference': [12.] * 200,
        'challenger': [110.] * 200, 'median': [110.] * 200,
        'p10': [9.] * 160 + [11.] * 40, 'p90': [11.] * 200,
        'season': [2025] * 200, 'week': np.repeat(np.arange(1, 11), 20),
    })
    lines = pd.DataFrame({
        'player_game': [str(i) for i in range(200)], 'season': rows.season,
        'week': rows.week, 'outcome': [0, 1] * 100, 'weight': 1.,
        'reference_probability': .9, 'probability': .5,
    })
    return rows, lines


def test_probability_gain_does_not_require_better_point_error():
    rows, lines = contract_rows()
    gates = output_gates(rows, lines, 'probability')
    assert gates['probability']['pass']
    assert not gates['expected_mean']['pass']
    assert not gates['median']['pass']
    assert not gates['bankroll_approval']


def test_mean_rmse_can_pass_while_median_mae_fails():
    rows, lines = contract_rows()
    rows['reference'] = 10 + np.tile([0.] * 9 + [50.], 20)
    rows['challenger'] = rows['median'] = 10 + np.tile([-10., 10.], 100)
    gates = output_gates(rows, lines, 'probability')
    assert gates['expected_mean']['pass']
    assert not gates['median']['pass']
    assert gates['expected_mean']['metrics']['mae'] > gates['expected_mean']['reference']['mae']


def test_preserved_point_never_authorizes_mean_or_median_replacement():
    rows, lines = contract_rows()
    rows['challenger'] = rows['median'] = rows.actual
    gates = output_gates(rows, lines, 'probability', point_preserved=True)
    assert gates['probability']['pass']
    assert not gates['expected_mean']['pass'] and not gates['median']['pass']


def test_duplicate_lines_do_not_inflate_clustered_probability_evidence():
    rows, lines = contract_rows()
    first = output_gates(rows, lines, 'probability')
    duplicated = pd.concat([lines] * 20, ignore_index=True)
    duplicated['weight'] /= 20
    second = output_gates(rows, duplicated, 'probability')
    assert first['probability']['clustered_brier_gain'] == pytest.approx(second['probability']['clustered_brier_gain'])
    one_week = output_gates(rows.assign(week=1), duplicated.assign(week=1), 'probability')
    assert not one_week['probability']['pass']


def tail_model():
    model = WorkloadTailCalibration()
    model.global_factors = np.array([2., 1.5])
    model.by_workload = {}
    model.enabled = False
    return model


def test_tail_disabled_is_identity_and_enabled_preserves_mass_and_median():
    values = np.array([[10., 20., 30., 40., 50.]])
    weights = np.full((1, 5), .2)
    tail = tail_model()
    unchanged, _ = tail.transform(values, weights, [25])
    np.testing.assert_array_equal(unchanged, values)
    tail.enabled = True
    repaired, mass = tail.transform(values, weights, [25])
    np.testing.assert_array_equal(mass, weights)
    assert np.all(np.diff(repaired[0]) > 0)
    assert curve_summary(repaired, mass)['median'][0] == 30
    assert set(tail.groups([np.nan, 10, 25, 40])) == {'unknown', 'low', 'normal', 'high'}


def test_tail_cannot_pass_by_widening_interval_only():
    actual = np.array([0.] * 70 + [2.] * 10 + [3.] * 20)
    before = {'p10': np.full(100, -1.), 'p90': np.full(100, 1.)}
    after = {'p10': np.full(100, -100.), 'p90': np.full(100, 2.5)}
    tail = tail_model().validate(actual, before, after, .25, .2)
    assert tail.validation['coverage_after'] == .8
    assert tail.validation['interval_score_after'] > tail.validation['interval_score_before']
    assert not tail.enabled


def test_tail_validation_veto_is_limited_to_probabilities_not_proven_point_outputs():
    tail = tail_model()
    tail.validation = {'enabled': False}
    gates = {kind: {'pass': True} for kind in ('expected_mean', 'median', 'probability')}
    assert prospective_outputs('qb_tail', gates, {'bundle': {'tail': tail}}) == ['expected_mean', 'median']
    assert gates['prospective_blocker'] == 'tail_probability_disabled_latest_calibration'


class ScaleCalibration:
    enabled = True

    def predict(self, frame):
        return .8 * frame.probability.to_numpy()


def test_line_calibration_is_carried_into_valid_replay_curve():
    values = np.array([[0., 10., 20., 30., 40.]])
    weights = np.full((1, 5), .2)
    frame = pd.DataFrame({'position': ['QB'], 'pass_attempts_avg_5': [30]})
    calibrated, mass = calibrate_curve(ScaleCalibration(), frame, 'passing_yards', values, weights,
        books=['draftkings'], extra_lines=[[15.5]])
    assert np.all(mass >= 0)
    assert mass.sum() == pytest.approx(1)
    assert curve_over(calibrated, mass, [15.5])[0] == pytest.approx(.8 * .6)
    assert np.all(np.diff(calibrated[0]) >= 0)
    empirical = np.array(empirical_residuals(calibrated[0], mass[0], 20)) + 20
    assert abs(np.mean(empirical > 15.5) - .8 * .6) < 1 / 2001
    disabled = ScaleCalibration()
    disabled.enabled = False
    raw, raw_mass = calibrate_curve(disabled, frame, 'passing_yards', values, weights)
    np.testing.assert_array_equal(raw, values)
    np.testing.assert_array_equal(raw_mass, weights)


def raw_history():
    rows = []
    for week in range(1, 7):
        for player, position, attempts, carries, targets in (
            ('qb', 'QB', 40, 2, 0), ('wr', 'WR', 0, 0, 20), ('rb', 'RB', 0, 18, 15),
        ):
            rows.append({
                'game_id': f'g{week}', 'player_id': player, 'team_abbr': 'A', 'opponent_abbr': 'B',
                'position': position, 'season': 2025, 'week': week,
                'game_date_et': date(2025, 9, 1) + timedelta(weeks=week),
                'pass_attempts': attempts, 'carries': carries, 'targets': targets,
                'passing_yards': attempts * 7, 'rushing_yards': carries * 4, 'receiving_yards': targets * 8,
            })
    return pd.DataFrame(rows)


def test_roster_membership_and_features_do_not_use_current_outcomes():
    raw = raw_history()
    before_teams, before_players = prepare_history(raw)
    changed = raw.copy()
    changed.loc[changed.game_id.eq('g4'), list(COUNTS) + list(YARDS)] *= 10
    changed.loc[changed.game_id.eq('g4') & changed.player_id.eq('wr'), 'player_id'] = 'newcomer'
    after_teams, after_players = prepare_history(changed)
    mask = before_teams.game_id.eq('g4')
    pd.testing.assert_frame_equal(team_features(before_teams.loc[mask]), team_features(after_teams.loc[mask]))
    before = before_players.loc[before_players.game_id.eq('g4')].reset_index(drop=True)
    after = after_players.loc[after_players.game_id.eq('g4')].reset_index(drop=True)
    assert 'newcomer' not in set(after.player_id)
    pd.testing.assert_frame_equal(roster_features(before), roster_features(after))
    assert after_teams.loc[mask, 'unallocated_targets'].iloc[0] > 0
    assert roster_features(before).injury_out.isna().all()
    assert roster_features(before).injury_out_missing.eq(1).all()


def test_same_day_results_never_enter_team_history():
    raw = raw_history()
    raw.loc[raw.game_id.eq('g4'), 'game_date_et'] = raw.loc[raw.game_id.eq('g3'), 'game_date_et'].iloc[0]
    teams, _ = prepare_history(raw)
    left = team_features(teams.loc[teams.game_id.eq('g3')]).reset_index(drop=True)
    right = team_features(teams.loc[teams.game_id.eq('g4')]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_team_feature_whitelists_exclude_actual_labels():
    teams, players = prepare_history(raw_history())
    assert not any(c.startswith(('actual_', 'unallocated_')) for c in team_features(teams))
    assert not any(c.startswith('actual_') for c in roster_features(players))


class ConstantHead:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


@pytest.mark.parametrize('stat,op', [('receiving_yards', 'targets'), ('rushing_yards', 'carries'), ('passing_yards', 'pass_attempts')])
def test_allocation_respects_full_team_budget_and_explicit_out(stat, op):
    teams, roster = prepare_history(raw_history())
    teams = teams.loc[teams.game_id.eq('g6')]
    roster = roster.loc[roster.game_id.eq('g6')]
    model = TeamAllocationModel()
    model.stat, model.op = stat, op
    model.volume = {'pass_attempts': ConstantHead(40), 'carries': ConstantHead(20)}
    model.target_fraction, model.reserve = .9, .1
    model.share, model.rate, model.league_rate = ConstantHead(.3), ConstantHead(7), 7
    predicted = model.predict(teams, roster)
    assert predicted.expected_opportunity.sum() == pytest.approx(predicted.team_budget.iloc[0] * .9)
    assert predicted.expected_opportunity.sum() <= (40 if op != 'carries' else 20)
    offered = predicted.loc[predicted.player_id.eq('wr')]
    assert offered.expected_opportunity.iloc[0] < predicted.team_budget.iloc[0]
    roster = roster.copy()
    roster.loc[roster.player_id.eq('wr'), 'injury_out'] = 1.
    out = model.predict(teams, roster)
    assert out.loc[out.player_id.eq('wr'), 'expected_opportunity'].iloc[0] == 0
    assert out.expected_opportunity.sum() <= out.team_budget.iloc[0]
    with pytest.raises(ValueError, match='prior team volume'):
        model.predict(teams.assign(prior_pass_attempts_5=np.nan), roster)


def test_snapshot_selection_requires_known_prelock_context():
    cutoff = datetime(2026, 9, 20, 12, tzinfo=timezone.utc)
    valid = {'source_cutoff': '2026-09-20T11:00:00Z', 'captured_at': '2026-09-20T11:01:00Z',
             'contract': 'nfl-team-context-v1'}
    future = dict(valid, captured_at='2026-09-20T12:01:00Z')
    backwards = dict(valid, source_cutoff='2026-09-20T13:00:00Z')
    assert snapshot_before([future, backwards], cutoff) is None
    assert snapshot_before([future, valid, backwards], cutoff) == valid
    assert snapshot_before([dict(valid, contract='unknown')], cutoff) is None


def component_record():
    record = locked_record()
    payload = record['forecast_payload']
    row, offer, metrics, distribution = scoring_fixture()
    payload['scoring_replay'] = capture(row, 'receiving_yards', 55, 50, metrics, offer,
        distribution, {}, {}, .02, {'version': 'frozen'})
    cutoff = datetime(2099, 9, 20, 12, tzinfo=timezone.utc)
    payload['context_evidence'] = context_evidence({}, cutoff)
    payload['forecast_features'] = row
    return record


def test_full_path_replays_final_probability_without_mutating_lock():
    payload = component_record()['forecast_payload']
    original = deepcopy(payload)
    result = full_path(payload, np.array([25., 45., 55., 65., 85.]), np.full(5, .2), 55.)
    assert result['candidate_probability'] == payload['probability']
    assert result['candidate_side'] == payload['side']
    assert result['scoring_stage'] == 'complete_live_path'
    expected_over = payload['probability'] if payload['side'] == 'over' else 1 - payload['probability']
    assert result['calibrated_over_probability'] == expected_over
    assert result['probability_trace']['final_side'] == result['candidate_probability']
    assert payload == original
    payload['probability'] += .01
    with pytest.raises(ValueError, match='production_replay_mismatch'):
        full_path(payload, np.array([55.]), np.ones(1), 55.)


def test_missing_scoring_arguments_cannot_claim_complete_replay():
    payload = component_record()['forecast_payload']
    payload.pop('scoring_replay')
    with pytest.raises(ValueError, match='missing_lock_time_scoring_inputs'):
        full_path(payload, np.array([55.]), np.ones(1), 55.)


def test_weighted_cdf_quadrature_preserves_probabilities():
    values = np.array([0., 10., 100.])
    masses = np.array([.05, .85, .10])
    empirical = np.asarray(empirical_residuals(values, masses, 30.)) + 30.
    assert abs(np.mean(empirical > 50) - .1) < 1 / 2001
    assert abs(np.mean(empirical > 5) - .95) < 1 / 2001


class FixedUncertainty:
    def mixture(self, features, center, weights):
        return np.asarray(center)[:, None] + np.array([-30., -10., 0., 10., 30.]), np.full((len(center), 5), .2)


def test_only_accepted_outputs_score_and_receiving_point_is_preserved():
    record = component_record()
    payload = record['forecast_payload']
    artifact = {'production_release': payload['model_version'], 'run_id': 'component-test', 'models': {
        'receiving_conditional': {'stat': 'receiving_yards', 'enabled_outputs': ['probability'], 'uncertainty': FixedUncertainty()},
        'team_receiving_yards': {'stat': 'receiving_yards', 'enabled_outputs': []},
    }}
    results, excluded = score_records([(1, payload), (2, payload)], artifact)
    assert not excluded and set(results) == {'receiving_conditional'}
    assert len(results['receiving_conditional']) == 1
    result = results['receiving_conditional'][0]
    assert result['projection_mean'] == payload['projection']
    assert result['point_forecast_preserved'] and not result['betting_eligible']
    assert result['scoring_stage'] == 'complete_live_path'
    payload['context_evidence']['injury_observed_at'] = '2099-09-20T13:00:00Z'
    results, excluded = score_records([(1, payload)], artifact)
    assert not results and excluded['missing_or_invalid_lock_inputs'] == 1


def test_qb_point_only_approval_cannot_price_a_bet(monkeypatch):
    from nfl_pipeline.modeling import score_accuracy_components as scorer
    payload = component_record()['forecast_payload']
    payload['stat'] = 'passing_yards'
    payload['forecast_features'] = dict(payload['forecast_features'], position='QB', pass_attempts_avg_5=30)
    calibration = ScaleCalibration()
    calibration.enabled = False
    bundle = {'model': None, 'uncertainty': None, 'tail': tail_model(), 'calibrator': calibration}
    def curve(*args):
        return np.array([[0., 20., 40., 60., 200.]]), np.full((1, 5), .2), {
            'weights': np.ones((1, 1)), 'opportunities': np.full((1, 1), 30.)}
    monkeypatch.setattr(scorer, 'mixture_curve', curve)
    artifact = {'production_release': payload['model_version'], 'run_id': 'point-only', 'models': {
        'qb_tail': {'stat': 'passing_yards', 'enabled_outputs': ['expected_mean', 'median'], 'bundle': bundle},
    }}
    results, excluded = scorer.score_records([(1, payload)], artifact)
    result = results['qb_tail'][0]
    assert not excluded and not result['betting_eligible']
    assert result['projection_mean'] == pytest.approx(64)
    assert result['projection_median'] == pytest.approx(40)
    assert not result['point_forecast_preserved']
    assert 'candidate_probability' not in result and 'locked_line' not in result


def test_prospective_report_separates_mean_median_and_full_path_probabilities(tmp_path):
    from nfl_pipeline.integrity import atomic_json
    from nfl_pipeline.modeling.live_scoring_replay import prospective_report
    record = component_record()
    payload = record['forecast_payload']
    shadow = {'forecast_id': 1, 'challenger_run': 'component-test', 'production_release': payload['model_version'],
        'source_context_cutoff': payload['prediction_context_cutoff_utc'], 'projection_mean': 55.,
        'projection_median': 50., 'distribution_mean': 58., 'p10': 25., 'p90': 90.,
        'locked_line': 45.5, 'calibrated_over_probability': .6, 'scoring_stage': 'complete_live_path'}
    atomic_json(tmp_path / 'test' / 'prospective' / '2099-09-20' / 'one.json', {
        'rows': [shadow], 'scored_at': '2099-09-20T13:00:00Z', 'training_end': '2099-09-19',
    })
    report = prospective_report([record], tmp_path)
    cohort = next(iter(report['cohorts'].values()))
    assert cohort['projection']['rmse'] == 5
    assert cohort['median']['mae'] == 10
    assert cohort['distribution_mean']['bias'] == -2
    assert cohort['probability_scoring_stages'] == {'complete_live_path': 1}
    assert cohort['clustered_brier_gain']['lower_95'] is None
    assert not report['automatic_promotion']
