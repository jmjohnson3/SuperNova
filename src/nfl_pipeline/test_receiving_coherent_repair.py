import numpy as np
import pandas as pd
import pytest
import copy
from datetime import datetime, timezone

from nfl_pipeline.modeling.challenger_models import curve_summary
from nfl_pipeline.modeling.receiving_coherent_repair import (
    ReceivingDownside, coherent_after_adjustments, downside_labels, passes,
)
from nfl_pipeline.modeling.receiving_selection_experiment import ranked, score
from nfl_pipeline.modeling import receiving_selection_experiment as experiment


def test_final_coherence_repairs_reversal_and_preserves_interval_anchors():
    v = np.arange(101, dtype=float); w = np.ones(101)/101
    cv, cw = coherent_after_adjustments(v, w, [30.5, 40.5, 65.5], [.50, .65, .30])
    tails = [cw[cv > line].sum() for line in np.arange(-.5, 102, .5)]
    assert np.all(np.diff(tails) <= 1e-12)
    assert (cw >= 0).all() and cw.sum() == pytest.approx(1)
    before = curve_summary(v[None, :], w[None, :]); after = curve_summary(cv[None, :], cw[None, :])
    assert after['p10'] == before['p10']
    assert after['p90'] == before['p90']
    assert cw[cv > 30.5].sum() >= cw[cv > 40.5].sum()


def test_coherent_curve_not_changed_by_offer_order_or_duplicate_line():
    v = np.arange(100.); w = np.ones(100)/100
    a, b = coherent_after_adjustments(v, w, [30.5, 60.5], [.6, .4])
    c, d = coherent_after_adjustments(v, w, [60.5, 30.5], [.4, .6])
    assert np.allclose(a, c) and np.allclose(b, d)
    _, masses = coherent_after_adjustments(v, w, [30.5, 30.5], [.4, .7])
    assert masses.sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        coherent_after_adjustments(v, w, [40], [1.4])


def test_no_offered_adjustments_identity():
    v = np.arange(101.); w = np.ones(101)/101
    cv, cw = coherent_after_adjustments(v, w, [], [])
    assert np.allclose(cv, v) and np.allclose(cw, w)


def test_downside_causes_distinguish_volume_and_efficiency_unknown():
    frame = pd.DataFrame({'targets_avg_5': [6, 6, np.nan, 2], 'targets': [1, 6, 0, 0],
                          'receiving_yards': [2, 3, 0, 0]})
    v = np.tile(np.arange(20, 81), (4, 1)); w = np.ones_like(v)/61
    labels, eligible = downside_labels(frame, v, w)
    assert labels.tolist() == [1, 2, 0, 0]
    assert eligible.tolist() == [True, True, False, False]


def test_downside_leaves_low_role_distribution_alone():
    n = 180
    frame = pd.DataFrame({'game_id': [str(i) for i in range(n)], 'player_id': 'p', 'position': 'WR',
        'targets_avg_5': 6., 'targets': [1]*40+[6]*140, 'receiving_yards': [5]*40+[10]*40+[60]*100})
    v = np.tile(np.arange(20, 81, dtype=float), (n, 1)); w = np.ones_like(v)/61
    model = ReceivingDownside().fit(frame, v, w)
    assert model.enabled
    unknown = frame.iloc[:1].assign(targets_avg_5=np.nan)
    cv, cw = model.transform(unknown, v[:1], w[:1])
    a = curve_summary(v[:1], w[:1]); b = curve_summary(cv, cw)
    assert b['mean'] == pytest.approx(a['mean'])
    assert b['p10'] == pytest.approx(a['p10'])
    assert cw.sum() == pytest.approx(1)


def test_gate_rejects_brier_gain_with_worse_coverage_or_calibration():
    base = dict(expected={'coverage_80': .8}, probability={'brier': .23, 'calibration_error': .02}, interval_score=80)
    candidate = dict(expected={'coverage_80': .72}, probability={'brier': .22, 'calibration_error': .01}, interval_score=79)
    assert not passes(base, candidate)
    candidate['expected']['coverage_80'] = .8
    assert passes(base, candidate)
    candidate['probability']['calibration_error'] = .03
    assert not passes(base, candidate)


def ranking_frame():
    return pd.DataFrame([dict(prediction_id=i, model_version='r', scoring_version='a', day='2026-09-20',
        game_id='g', player_id=str(i), batch='2026-09-20T16:00:00Z', probability=.7-i*.01,
        market=.5, payout=1., push_probability=0., eligible=True, actual=10., outcome=1.,
        is_push=False, season=2026, week=2, exact_micro=i == 0) for i in range(8)])


def test_ranking_is_outcome_blind_and_never_selects_future_offer_early():
    frame = ranking_frame()
    frame.loc[7, ['batch', 'probability']] = ['2026-09-20T17:00:00Z', .99]
    current = ranked(frame)
    assert current.prediction_id.tolist() == [0, 1, 2, 3, 4]
    frame['outcome'] = 0.; frame['actual'] = 0.
    assert ranked(frame).prediction_id.tolist() == current.prediction_id.tolist()
    assert ranked(frame, penalty=.5).prediction_id.tolist() == current.prediction_id.tolist()


def test_pending_and_pushes_are_selected_but_not_binary_losses():
    frame = ranking_frame().iloc[:2].copy()
    frame.loc[0, ['actual', 'outcome']] = [np.nan, np.nan]
    frame.loc[1, ['is_push', 'outcome']] = [True, np.nan]
    result = score(ranked(frame))
    assert result['selected_rows'] == 2 and result['rows'] == 0
    assert result['pending'] == 1 and result['pushes'] == 1


def test_ranker_deduplicates_player_and_separates_scoring_releases():
    frame = ranking_frame()
    frame.loc[1, 'player_id'] = '0'
    assert ranked(frame).prediction_id.tolist() == [0, 2, 3, 4, 5]
    other = frame.assign(scoring_version='b', prediction_id=lambda f: f.prediction_id+20)
    both = ranked(pd.concat([frame, other], ignore_index=True))
    assert len(both) == 10


def test_conservative_rank_does_not_change_probability_or_require_wins():
    frame = ranking_frame().iloc[:2].copy()
    frame.loc[0, ['probability', 'market']] = [.7, .4]
    frame.loc[1, ['probability', 'market']] = [.65, .6]
    assert ranked(frame, cap=1).prediction_id.tolist() == [0]
    assert ranked(frame, penalty=.5, cap=1).prediction_id.tolist() == [1]
    assert frame.loc[0, 'probability'] == .7


def test_complete_scoring_path_is_replayed_without_modifying_original_locks(monkeypatch):
    from nfl_pipeline.test_accuracy_challengers import scoring_fixture
    from nfl_pipeline.modeling.predict_player_props import _candidate_from_offer
    from nfl_pipeline.modeling.scoring_capture import capture
    row, offer, metrics, distribution = scoring_fixture()
    offer['bookmaker_key'] = 'fanduel'
    records = []
    for i, line in enumerate([30.5, 60.5]):
        o = dict(offer, line=line)
        p = _candidate_from_offer(row, STAT := 'receiving_yards', 55, 50, metrics, o, distribution)
        p.update(prediction_context_cutoff_utc='2099-09-20T16:00:00Z', forecast_features=copy.deepcopy(row),
                 scoring_replay=capture(row, STAT, 55, 50, metrics, o, distribution, {}, {}, .02, {'version': 'frozen'}))
        records.append(dict(id=i, stat=STAT, line=line, side=p['side'], book='fanduel', game_id='g', player_id='p',
            season=2026, week=2, actual=55., created_at_utc=datetime(2099, 9, 20, 16, tzinfo=timezone.utc),
            start_ts_utc=row['start_ts_utc'], offer_fetched_at='2099-09-20T15:59:00Z', forecast_payload=p))
    before = copy.deepcopy(records)
    frame, excluded = experiment.offer_frame(records, [])
    assert not excluded
    v = np.arange(101.)[None, :]; w = np.ones_like(v)/101
    monkeypatch.setattr(experiment, 'variants', lambda *args: {'coherent_base': (v, w), 'coherent_downside': (v, w)})
    result = experiment.evaluate_full_path(records, {'bundle': {}, 'training_end': '2099-09-10', 'production_release': 'frozen'}, frame)
    assert len(result['rows']) == 4
    assert result['summary']['coherent_base']['stages']['coherent_probability']['rows'] == 2
    assert records == before
    assert result['deployment_approved'] is False
    # A model trained after the original lock cannot replay a next-week game.
    records[0]['forecast_payload']['prediction_context_cutoff_utc'] = '2099-09-09T16:00:00Z'
    result = experiment.evaluate_full_path(records[:1], {'bundle': {}, 'training_end': '2099-09-10', 'production_release': 'frozen'}, frame.iloc[:1])
    assert result['exclusions']['training_not_before_lock_date'] == 1
