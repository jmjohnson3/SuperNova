from copy import deepcopy

import numpy as np
import pandas as pd

from nfl_pipeline.import_usage_context import _advanced_usage_rows, advanced_usage_health
from nfl_pipeline.modeling.yardage_lock_validation import workload_decomposition, revised_score
from nfl_pipeline.modeling.receiving_selection_experiment import ranked


def test_advanced_aliases_preserve_measured_zero_and_skip_missing_primary():
    frame = pd.DataFrame([dict(season=2026, week=3, game_id='g', team='DEN', player_id='p',
        routes_run=np.nan, routes=20., targets=5., receiving_yards=60.,
        targets_per_route_run=0., yards_per_route_run=0., first_read_targets=0.)])
    rows = _advanced_usage_rows(frame, {})
    assert rows[0][5] == 20
    assert rows[0][12:15] == (0., 0., 0.)
    assert advanced_usage_health(frame, rows)['matched_nonmissing']['first_read_targets'] == 1


def test_source_without_routes_cannot_claim_measured_routes():
    frame = pd.DataFrame([dict(season=2026, week=3, game_id='g', team='DEN', player_id='p',
        targets=5., receiving_yards=60., route_participation_proxy=.8)])
    rows = _advanced_usage_rows(frame, {})
    assert rows[0][5] is None
    health = advanced_usage_health(frame, rows)
    assert not health['capabilities']['routes_run']
    assert health['matched_nonmissing']['routes_run'] == 0


def test_decomposition_reconciles_and_does_not_invent_missing_opportunity():
    record = dict(stat='rushing_yards', actual_opportunity=17., actual=49.,
        forecast_payload=dict(projection=61.1, scoring_replay=dict(row=dict(rb_projected_carries_v3=11.1))))
    d = workload_decomposition(record)
    assert d['workload_yards'] > 0 and d['efficiency_yards'] < 0
    assert np.isclose(d['workload_yards'] + d['efficiency_yards'], 49.-61.1)
    record['actual_opportunity'] = None
    assert workload_decomposition(record)['status'] == 'missing_decomposition_inputs'


def test_uncertainty_replay_preserves_original_lock(monkeypatch):
    import nfl_pipeline.modeling.yardage_lock_validation as m
    calls = []
    def replay(captured):
        calls.append(deepcopy(captured))
        return dict(side='over', probability=.6, raw_over_probability=.65)
    monkeypatch.setattr(m, 'replay', replay)
    class Uncertainty:
        def mixture(self, X, centers, weights):
            return np.array([[10., 20., 30.]]), np.full((1, 3), 1/3)
        def confidence(self, X):
            return np.array([.73])
    p = dict(side='over', probability=.6, stat='rushing_yards', line=20.5, projection=25.,
        scoring_replay=dict(row={'position': 'RB'}, distribution={'original': True}))
    before = deepcopy(p)
    r = revised_score(p, Uncertainty())
    assert p == before and r['confidence'] == .73
    assert calls[-1]['distribution']['projection_confidence'] == .73
    assert calls[-1]['distribution']['residual_quantiles'] == [-15., -5., 5.]


def test_ranking_never_reads_settlement_or_fills_pending_slots_later():
    frame = pd.DataFrame([dict(model_version='v', scoring_version='s', day='d', batch='1',
        game_id='g', player_id=str(i), prediction_id=i, eligible=True, market=.5,
        probability=.7-i*.01, payout=1., push_probability=0., actual=None, outcome=None)
        for i in range(8)])
    picked = ranked(frame).prediction_id.tolist()
    frame['outcome'] = np.arange(8) % 2
    frame['actual'] = np.arange(8)*20
    assert picked == ranked(frame).prediction_id.tolist()
    assert len(picked) == 5


def test_markdown_separates_paired_replay_from_full_population(tmp_path):
    from nfl_pipeline.modeling.yardage_lock_validation import write_markdown
    population = dict(rows=8, binary_rows=8, stages={'probability': dict(rows=8, brier=.24)},
        paired_comparison={'probability': dict(rows=3, brier=.21),
            'market': dict(rows=3, brier=.25), 'challenger_probability': dict(rows=3, brier=.23)},
        record={'win': 5, 'loss': 3})
    report = dict(limitations=['Diagnostic only'], stats={'receiving_yards': dict(
        all_displayed=population, context=dict(rows=8, unknown_injury=2,
            routes={'observed_prior_games': 4, 'proxy_only': 4}),
        decomposition={'workload': 3, 'efficiency': 5})},
        fixed_research=population, ranking={'ev': population})
    path = tmp_path / 'audit.md'
    write_markdown(report, path)
    text = path.read_text()
    assert '| receiving_yards | 8 | - | 0.2400 | - |' in text
    assert '| receiving_yards | 3 | 0.2100 | 0.2300 | 0.2500 |' in text
    assert '| receiving_yards | 8 | 2 | 4 | 3 | 5 | 0 |' in text
    assert 'not actual placed bets' in text
