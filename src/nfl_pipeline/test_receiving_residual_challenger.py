from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling import benchmark_offers
from nfl_pipeline.modeling.benchmark_offers import independent_line_probability
from nfl_pipeline.modeling.challenger_models import curve_summary
from nfl_pipeline.modeling import receiving_residual_challenger as module


def settled_rows(inverted=False):
    rows = []
    start = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    forecast_id = 1
    for week in range(1, 5):
        lock = start + timedelta(weeks=week-1)
        for player in range(20):
            outcome = float(player % 2 == 0)
            independent = .7 if outcome else .3
            if inverted:
                independent = 1-independent
            rows.append(dict(
                forecast_id=forecast_id,
                run_id='benchmark',
                production_release='release',
                scoring_version='scoring',
                variant=module.VARIANT,
                stat='receiving_yards',
                source_book='fanduel',
                source_side='over',
                source_price=-110,
                market_probability=.5,
                market_evidence='true_same_book_at_lock',
                independent_same_side_probability=independent,
                independent_push_probability=0.,
                outcome=outcome,
                season=2026,
                week=week,
                game_id=f'g{week}-{player//4}',
                player_id=f'p{player}',
                created_at=lock.isoformat(),
                label_available_at=(lock+timedelta(days=1)).isoformat(),
                raw_p10=10.,
                raw_p90=90.,
                distribution_mean=50.,
            ))
            forecast_id += 1
    return rows


def test_independent_curve_probability_is_not_market_anchored():
    over, push = independent_line_probability(
        np.array([10., 40., 70.]), np.array([.2, .3, .5]), 40.)
    assert push == pytest.approx(.3)
    assert over == pytest.approx(.5/.7)


def test_verified_replay_cache_avoids_rebuilding_historical_curves(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_offers, 'STORE', tmp_path)
    values = np.array([[10., 40., 70.]])
    weights = np.array([[.2, .3, .5]])
    summary = curve_summary(values, weights)
    row = dict(
        forecast_id=1,
        run_id='benchmark',
        production_release='release',
        variant='stable_ensemble_calibrated',
        stat='receiving_yards',
        source_side='over',
        source_line=40.5,
        distribution_mean=float(summary['mean'][0]),
        raw_p10=float(summary['p10'][0]),
        raw_p90=float(summary['p90'][0]),
    )
    record = {'id': 1, 'forecast_payload': {'scoring_replay': {'scoring_fingerprint': 's'}}}
    artifact = {'run_id': 'benchmark', 'production_release': 'release',
                'models': {'receiving_yards': object()}}
    monkeypatch.setattr(benchmark_offers, 'scoring_frame', lambda records, artifact: pd.DataFrame([{}]))
    monkeypatch.setattr(benchmark_offers, 'named_curve', lambda *args: (values, weights))
    first, status = benchmark_offers.reconstruct_independent_probabilities(
        [record], [row], artifact, 'sha')
    assert status['verified_replays'] == 1
    assert first[0]['independent_same_side_probability'] == pytest.approx(.5)

    monkeypatch.setattr(benchmark_offers, 'named_curve', lambda *args: (_ for _ in ()).throw(AssertionError('replayed')))
    second, status = benchmark_offers.reconstruct_independent_probabilities(
        [record], [row], artifact, 'sha')
    assert status['cached_verified_replays'] == 1
    assert second[0]['independent_same_side_probability'] == first[0]['independent_same_side_probability']


def test_forward_residual_fit_supports_real_signal_and_rejects_inversion(monkeypatch):
    monkeypatch.setattr(module, 'verified_aliases', lambda: {})
    good = module.fit(settled_rows())
    policy = next(iter(good['policies'].values()))
    assert policy['point_trust'] > 0
    assert policy['conservative_trust'] > 0
    assert policy['conservative_supported']
    assert good['forward_folds'][-1]['conservative_blend']['brier'] < good['forward_folds'][-1]['market']['brier']

    bad = module.fit(settled_rows(inverted=True))
    policy = next(iter(bad['policies'].values()))
    assert policy['point_trust'] == 0
    assert policy['conservative_trust'] == 0
    assert not policy['conservative_supported']


def test_apply_requires_artifact_and_labels_strictly_before_lock(monkeypatch):
    monkeypatch.setattr(module, 'verified_aliases', lambda: {})
    artifact = module.fit(settled_rows())
    artifact['created_at'] = '2026-10-01T00:00:00Z'
    row = dict(
        forecast_id=999,
        run_id='benchmark',
        production_release='release',
        source_scoring_version='scoring',
        variant=module.VARIANT,
        stat='receiving_yards',
        source_book='fanduel',
        source_side='over',
        source_price=110,
        source_context_cutoff='2026-10-10T12:00:00Z',
        market_probability=.5,
        market_evidence='true_same_book_at_lock',
        independent_same_side_probability=.70,
        independent_push_probability=0.,
        raw_p10=10.,
        raw_p90=90.,
        distribution_mean=50.,
    )
    scored = module.apply_document_rows([{'id': 999}], [row], artifact)[0]['residual_signal']
    assert scored['status'] == 'eligible_paper_signal'
    assert scored['conservative_probability'] > scored['market_probability']
    assert scored['conservative_ev'] > 0

    artifact['created_at'] = '2026-10-11T00:00:00Z'
    blocked = module.apply_document_rows([{'id': 999}], [row], artifact)[0]['residual_signal']
    assert blocked['status'] == 'residual_policy_not_available_at_lock'


def test_selector_is_outcome_blind_and_five_is_a_cap_not_a_quota():
    rows = []
    for index in range(8):
        rows.append(dict(
            forecast_id=index,
            run_id='r',
            production_release='v',
            scoring_cohort='s',
            day='2026-10-18',
            game_id=f'g{index//2}',
            player_id=f'p{index}',
            source_price=100,
            independent_push_probability=0.,
            market_probability=.5,
            conservative_probability=.60-index*.01,
            conservative_supported=True,
            outcome=float(index % 2),
        ))
    frame = pd.DataFrame(rows)
    selected = module.select_frame(frame)
    assert len(selected) == 5
    ids = selected.forecast_id.tolist()
    frame['outcome'] = 1-frame.outcome
    assert module.select_frame(frame).forecast_id.tolist() == ids

    frame['conservative_probability'] = .49
    assert module.select_frame(frame).empty
