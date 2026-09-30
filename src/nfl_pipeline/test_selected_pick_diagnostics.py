from datetime import datetime, timezone

import pandas as pd
import pytest

from nfl_pipeline.modeling.selected_pick_diagnostics import locked_workload, selected_report, workload_report
from nfl_pipeline.test_final_probability_validation import frame


def test_selection_uses_exact_ids_and_matching_release_dates():
    offers = frame(6).assign(market=.5)
    offers.loc[5, 'day'] = '2026-09-27'
    micro = offers.iloc[[1, 3]].copy()
    result = selected_report(offers, micro)['populations']
    assert result['all_eligible']['summary']['rows'] == 6
    assert result['eligible_on_micro_release_dates']['summary']['rows'] == 5
    assert result['exact_locked_micro']['summary']['rows'] == 2
    assert result['exact_locked_micro']['summary']['mean_probability'] == pytest.approx(.55)


def test_eligible_and_micro_are_not_all_paper_rows():
    offers = frame(4).assign(market=None)
    offers.loc[0, 'eligible'] = False
    result = selected_report(offers, offers.iloc[:0])['populations']
    assert result['all_eligible']['summary']['rows'] == 3
    assert result['eligible_on_micro_release_dates']['summary']['rows'] == 0
    assert 'paired_market_comparison' not in result['all_eligible']['summary']


def test_missing_workload_injury_not_filled_by_zero_or_prior():
    p = {'forecast_features': {'targets_avg_5': 6, 'injury_report_status': None}}
    row = locked_workload(p, 'receiving_yards')
    assert row['workload_bucket'] == 'unknown'
    assert row['projected_opportunity'] is None
    assert row['role_proxy'] == 'prior_targets_ge_4'
    assert row['injury_status'] == 'unknown'
    p['forecast_features']['receiver_projected_targets_v3'] = 0
    assert locked_workload(p, 'receiving_yards')['projected_opportunity'] == 0


def record(i=1):
    return dict(id=i, game_id='g', player_id='p', stat='receiving_yards', season=2026, week=2,
        line=40.5, actual=10., actual_targets=2., offer_fetched_at='2026-09-20T15:55:00Z',
        created_at_utc=datetime(2026, 9, 20, 16, tzinfo=timezone.utc), start_ts_utc='2026-09-20T17:00:00Z',
        forecast_payload=dict(model_version='r', projection=50., projection_p10=20., projection_p90=80.,
            line=40.5, prediction_context_cutoff_utc='2026-09-20T16:00:00Z',
            forecast_features={'receiver_projected_targets_v3': 5., 'targets_avg_5': 6.}))


def test_workload_decomposition_dedup_and_downside():
    result = workload_report([record(), record(2)])
    assert len(result['rows']) == 1
    row = result['rows'][0]
    assert row['interval'] == 'below'
    assert row['opportunity_error_yards'] == 30
    assert row['efficiency_residual_yards'] == 10
    assert row['opportunity_error_yards'] + row['efficiency_residual_yards'] == row['projection']-row['actual']
    assert row['dominant_component'] == 'opportunity'


def test_workload_does_not_invent_missing_intervals_or_projections():
    r = record(); r['forecast_payload'].pop('forecast_features'); r['forecast_payload'].pop('projection_p10')
    row = workload_report([r])['rows'][0]
    assert row['interval'] == 'unknown'
    assert row['opportunity_error_yards'] is None
    assert row['dominant_component'] == 'unknown'


def test_invalid_locks_and_pending_not_used_for_workload():
    a = record(); a['actual'] = None
    b = record(2); b['forecast_payload']['prediction_context_cutoff_utc'] = '2026-09-20T18:00:00Z'
    result = workload_report([a, b])
    assert result['rows'] == []
    assert sum(result['exclusions'].values()) == 2


def test_duplicate_offers_do_not_inflate_probability_weight():
    one = frame(2).assign(market=.5)
    many = pd.concat([one, *[one.iloc[:1].assign(prediction_id=10+i) for i in range(10)]])
    a = selected_report(one, one.iloc[:0])['populations']['all_eligible']['summary']
    b = selected_report(many, many.iloc[:0])['populations']['all_eligible']['summary']
    assert a['brier'] == pytest.approx(b['brier'])
    assert a['mean_probability'] == pytest.approx(b['mean_probability'])
