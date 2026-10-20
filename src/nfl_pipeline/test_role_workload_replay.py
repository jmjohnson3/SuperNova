from copy import deepcopy
import numpy as np
import pandas as pd

from nfl_pipeline.modeling import validate_role_workload as validation
from nfl_pipeline.test_role_workload import fitted, sample


def test_full_path_only_changes_raw_center_and_distribution(monkeypatch):
    captured = dict(projection=42., offer={'line': 40.5, 'over_price': -110, 'under_price': -110},
        distribution={'kind': 'old', 'projection_confidence': .37}, metrics={'market_calibration': {'v': 1}},
        calibration=[{'a': 1}], clv_guards=[{'b': 2}], scoring_fingerprint='old', row={'targets_avg_5': 6.})
    before = deepcopy(captured)
    seen = []
    monkeypatch.setattr(validation, 'replay', lambda x: seen.append(x) or {'probability': .55})
    validation.candidate_for_curve(captured, np.array([0., 20., 60.]), np.array([.2, .4, .4]))
    revised = seen[0]
    assert captured == before
    for key in ('offer', 'metrics', 'calibration', 'clv_guards', 'scoring_fingerprint', 'row'):
        assert revised[key] == captured[key]
    assert revised['distribution']['projection_confidence'] == .37
    assert revised['projection'] == 20.


def test_archived_ranking_includes_other_stats_and_never_reads_results(monkeypatch):
    seen = []
    def cap(candidates, limit):
        seen.append(deepcopy(candidates))
        top = max(candidates, key=lambda c: c['ev'])
        for candidate in candidates:
            candidate['tier'] = 'micro_projection' if candidate is top else 'paper'
    class Archive:
        __globals__ = {'_apply_daily_micro_cap': cap}
    monkeypatch.setattr(validation, 'archived_candidate', lambda _: Archive())
    rows = [dict(id=i, day='2026-10-01', batch='batch', source_scoring_version='archived',
        stat='receiving_yards' if i == 1 else 'rushing_yards', actual=999. if i == 1 else 0.) for i in (1, 2)]
    candidates = {1: {'ev': .1, 'tier': 'micro_projection'}, 2: {'ev': .2, 'tier': 'micro_projection'}}
    selected, exclusions = validation.selected_ids(rows, candidates)
    assert selected == {2} and not exclusions
    assert len(seen[0]) == 2 and all('actual' not in c for c in seen[0])
    assert candidates[1]['tier'] == 'micro_projection'
    selected, exclusions = validation.selected_ids(rows, {1: candidates[1]})
    assert not selected and exclusions == {'incomplete_replay_batch': 1}


def test_replay_requires_prior_training_and_original_features():
    artifact = {'production_release': 'release', 'training_end': '2026-09-17', 'stat': 'receiving_yards'}
    row = dict(model_version='release', payload={'prediction_context_cutoff_utc': '2026-09-24T20:00:00Z'},
        line=40.5, captured={'row': {'targets_avg_5': 5., 'receiving_yards_avg_5': 40.}})
    assert validation.eligibility(row, artifact) is None
    assert validation.eligibility(row, dict(artifact, training_end='2026-09-24')) == 'training_not_before_lock'
    row['captured']['row']['targets_avg_5'] = None
    assert validation.eligibility(row, artifact) == 'missing_original_player_history'


def test_same_selected_side_is_graded_when_challenger_switches(monkeypatch):
    model = fitted()
    features = sample('2026-09-24').iloc[0].to_dict()
    features['receiving_yards_avg_5'] = 30.
    payload = dict(book='fanduel', side='over', probability=.6, game_date_et='2026-09-24',
        prediction_context_cutoff_utc='2026-09-24T20:00:00Z')
    cap = dict(row=features, projection=30., metrics={}, distribution={'kind': 'empirical_oof_residual', 'residual_quantiles': [-10., 10.]},
        scoring_fingerprint='old', offer={'line': 40.5})
    prepared = dict(id=1, stat='receiving_yards', game_id='g', player_id='p', season=2026, week=3,
        model_version='release', scoring_version='old', source_scoring_version='old', captured=cap,
        line=40.5, side='over', actual=20., locked_at='2026-09-24T20:00:00Z')
    monkeypatch.setattr(validation, 'prepare_captured', lambda *a, **k: (pd.DataFrame([prepared]), {}))
    from nfl_pipeline.modeling import validate_distribution_policy
    monkeypatch.setattr(validate_distribution_policy, 'candidate_at_lock',
                        lambda *a: {'side': 'over', 'probability': .6, 'tier': 'micro_projection'})
    monkeypatch.setattr(validation, 'candidate_for_curve', lambda *a: {'side': 'under', 'probability': .7, 'price': -110})
    monkeypatch.setattr(validation, 'locked_market', lambda _: {'market_probability': .45})
    monkeypatch.setattr(validation, 'selected_ids', lambda *a: ({1}, {}))
    class Archive:
        __globals__ = {'_side_prob': None}
    monkeypatch.setattr(validation, 'archived_candidate', lambda _: Archive())
    record = dict(id=1, forecast_payload=payload, actual_targets=3.)
    artifact = dict(contract=validation.CONTRACT, production_release='release', training_end='2026-09-17',
        stat='receiving_yards', bundle={'model': model})
    frame, excluded, _ = validation.score_records([record], [{'prediction_id': 1}], {1}, artifact)
    assert not excluded and len(frame) == 1
    row = frame.iloc[0]
    assert row.outcome == 0.
    assert row.combined_selected_outcome == 1.
    assert row.combined_selected_market == .55
    assert abs(row.combined_final-.3) < 1e-12
    assert row.combined_selected_probability == .7
    assert row.exact_micro and row.fixed_research


def test_selected_pool_is_distinct_from_offer_average():
    rows = []
    for i in range(6):
        row = dict(id=i, game_id=str(i), player_id='p', season=2026, week=i//3+3, batch='batch',
            model_version='release', scoring_version='scoring', actual=10., actual_opportunity=3., outcome=0.,
            production=.5, market=.5, production_point=20., production_mean=20., production_p10=0., production_p90=40.,
            role='higher', exact_micro=i == 0, fixed_research=i == 0, production_ranked=i == 0)
        for v in validation.VARIANTS:
            row.update({v+'_raw': .8 if i == 0 else .1, v+'_final': .8 if i == 0 else .1,
                v+'_ranked': i == 0, v+'_selected_probability': .8, v+'_selected_market': .5,
                v+'_selected_outcome': 0., v+'_mean': 15., v+'_median': 12., v+'_p10': 0., v+'_p90': 30., v+'_opportunity': 3.})
        rows.append(row)
    report = validation.evaluate(pd.DataFrame(rows))['cohorts']['release|scoring']
    assert report['all_offers']['combined_final']['brier'] < .25
    assert report['exact_micro']['combined_final']['brier'] > .25
    assert report['combined_ranked']['selected_ids'] == [0]
