from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling import final_probability_validation as audit


def frame(n=12):
    return pd.DataFrame([dict(prediction_id=i, game_id=f'g{i}', player_id=f'p{i}', player_game=f'g{i}|p{i}',
        player=f'p{i}', stat='receiving_yards', side='over', line=40.5, book='fanduel', price=-110,
        payout=100/110, model_version='release', season=2026, week=2, day='2026-09-20',
        locked_at=datetime(2026, 9, 18, tzinfo=timezone.utc), probability=.51 + .02*i,
        kickoff=datetime(2026, 9, 20, 17, tzinfo=timezone.utc),
        label_available_at=datetime(2026, 9, 17, tzinfo=timezone.utc),
        outcome=float(i%2), eligible=True, confidence=.5, edge=.02, position='WR',
        workload_bucket='normal', line_bucket='2', gap_bucket='1', weight=1., replay_status='matched',
        raw=.6, heuristic=.55, post_exact=.53) for i in range(n)])


def test_top_five_selection_does_not_look_at_results():
    f = frame()
    a = audit.select_top_five(f).prediction_id.tolist()
    f['outcome'] = 1-f.outcome
    assert audit.select_top_five(f).prediction_id.tolist() == a
    assert len(a) == 5 and a == [11, 10, 9, 8, 7]


def test_nonpositive_ev_or_unapproved_is_not_selected():
    f = frame(); f['probability'] = .4
    assert audit.select_top_five(f).empty
    f['probability'] = .9; f['eligible'] = False
    assert audit.select_top_five(f).empty


def test_repeated_offers_share_one_player_game_weight():
    f = frame(1)
    copies = pd.concat([f.assign(prediction_id=i, line=20.5+i) for i in range(8)])
    u = audit.unique_decisions(copies)
    assert u.weight.sum() == pytest.approx(1.)


def test_scoring_code_cohorts_are_not_collapsed_or_calibrated_together():
    a = frame(1).assign(scoring_version='before_fix')
    b = a.assign(prediction_id=99, scoring_version='after_fix')
    combined = pd.concat([a, b])
    assert len(audit.unique_decisions(combined)) == 2
    reports, models = audit.calibration_experiment(combined)
    assert set(reports) == {'release|before_fix', 'release|after_fix'}
    assert not models


def test_no_calibrator_fitted_on_one_week():
    result, models = audit.calibration_experiment(frame())
    assert result['release']['enabled'] is False
    assert not models
    with pytest.raises(ValueError, match='independent'):
        audit.FinalStageCalibrator().fit(frame())


def test_calibration_outer_week_never_in_training(monkeypatch):
    f = pd.concat([frame(120).assign(week=w, day=f'2026-09-{w*7:02d}', game_id=lambda x: x.game_id+f'w{w}',
         player_game=lambda x: x.player_game+f'w{w}', prediction_id=lambda x: x.prediction_id+w*1000) for w in (1, 2, 3)])
    calls = []
    class Model:
        def fit(self, train):
            self.last = train.week.max(); calls.append(self.last)
            return self
        def predict(self, test):
            assert self.last < test.week.min()
            return np.full(len(test), .5)
    monkeypatch.setattr(audit, 'FinalStageCalibrator', Model)
    r, _ = audit.calibration_experiment(f)
    assert calls == [2]
    assert not r['release']['enabled']  # One holdout week is insufficient.


def test_actual_ledger_uses_exact_prediction_id_not_later_revision(monkeypatch):
    f = frame(1)
    f.loc[0, 'prediction_id'] = 11
    newer = f.assign(prediction_id=12, probability=.95, outcome=0.,
                     locked_at=datetime(2026, 9, 19, tzinfo=timezone.utc))
    monkeypatch.setattr(audit, 'decision_frame', lambda _: (pd.concat([f, newer]), {}))
    ledger = [dict(prediction_id=11, execution_status='simulated',
        locked_at_utc=datetime(2026, 9, 18, tzinfo=timezone.utc),
        ledger_book='fanduel', ledger_side='over', ledger_model_version='release', ledger_line=40.5, ledger_price=-110)]
    report, _ = audit.build_report([], ledger)
    assert report['actual_locked_micro']['mean_probability'] == pytest.approx(.51)
    assert report['micro_rows'][0]['prediction_id'] == 11


def test_invalid_american_odds():
    for p in (0, np.nan, np.inf):
        with pytest.raises(ValueError):
            audit.profit_multiple(p)
    assert audit.profit_multiple(-114) == pytest.approx(100/114)


def test_late_result_cannot_train_an_earlier_lock(monkeypatch):
    f = pd.concat([frame(120).assign(week=w, day=f'2026-09-{w*7:02d}', game_id=lambda x: x.game_id+f'w{w}',
        player_game=lambda x: x.player_game+f'w{w}', prediction_id=lambda x: x.prediction_id+w*1000) for w in (1, 2, 3)])
    f.loc[f.week.eq(2), 'label_available_at'] = datetime(2026, 9, 22, tzinfo=timezone.utc)
    calls = []
    class Model:
        def fit(self, train):
            calls.append(set(train.week))
            raise ValueError('Only one usable prior week remains')
    monkeypatch.setattr(audit, 'FinalStageCalibrator', Model)
    result, artifacts = audit.calibration_experiment(f)
    assert calls == [{1}]
    assert not artifacts and not result['release']['enabled']
