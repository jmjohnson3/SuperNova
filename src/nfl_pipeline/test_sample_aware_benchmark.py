from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling.sample_aware_splits import (
    BlockRequirement, sample_aware_partitions, regular_mask, describe_blocks)
from nfl_pipeline.modeling.qb_tail_distribution import QBTailDistribution, qb_features
from nfl_pipeline.modeling.challenger_models import curve_summary


def calendar():
    records = []; index = 0
    for season in (2023, 2024, 2025):
        for week in range(1, 23):
            for player in range(6 if week <= 18 else 1):
                records.append(dict(game_id=f'{season}-{week}-{player}', player_id=str(player),
                    season=season, week=week, game_date_et=date(2023, 1, 1)+timedelta(days=7*index)))
            index += 1
    return pd.DataFrame(records)


def test_playoff_blocks_expand_until_real_samples_and_weeks_exist():
    frame = calendar()
    needs = (BlockRequirement(20, 3, 18),)*5
    parts = sample_aware_partitions(frame, 'passing_yards', needs, min_model_rows=30, min_model_weeks=5)
    assert sum(map(len, parts)) == len(frame)
    for block in describe_blocks(parts)[1:]:
        assert block['rows'] >= 20
        assert block['weeks'] >= 3
        assert block['regular_rows'] >= 18
    assert describe_blocks(parts)[-1]['postseason_rows'] == 4
    for earlier, later in zip(parts, parts[1:]):
        assert earlier.game_date_et.max() < later.game_date_et.min()
    changed = frame.assign(passing_yards=np.random.default_rng(8).normal(size=len(frame)))
    again = sample_aware_partitions(changed, 'passing_yards', needs, 30, 5)
    assert [set(x.game_id) for x in parts] == [set(x.game_id) for x in again]


def test_duplicate_offers_cannot_satisfy_sample_thresholds():
    frame = calendar()
    with pytest.raises(ValueError, match='unique player-games'):
        sample_aware_partitions(pd.concat([frame, frame]), 'passing_yards')
    with pytest.raises(ValueError, match='Insufficient'):
        sample_aware_partitions(frame.iloc[-12:], 'passing_yards')


def test_a_game_cannot_span_training_and_validation_dates():
    frame = calendar()
    frame.loc[frame.index[-1], 'game_id'] = frame.game_id.iloc[0]
    frame.loc[frame.index[-1], 'player_id'] = 'other_participant'
    with pytest.raises(ValueError, match='overlap'):
        sample_aware_partitions(frame, 'passing_yards', (BlockRequirement(20, 3, 18),)*5, 30, 5)


def test_playoff_boundary_depends_on_season():
    f = pd.DataFrame({'season': [2020, 2021, 2025], 'week': [18, 18, 19]})
    assert regular_mask(f).tolist() == [False, True, False]


class ConstantHead:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


def test_qb_asymmetric_tail_transform_is_coherent_and_preserves_median():
    model = QBTailDistribution()
    model.heads = {.1: ConstantHead(-140), .5: ConstantHead(0), .9: ConstantHead(100)}
    model.strength = .5
    frame = pd.DataFrame({'passing_yards_avg_5': [250.], 'season': [2025], 'week': [10]})
    values = np.linspace(150, 350, 101)[None, :]; weights = np.full(values.shape, 1/101)
    v, w = model.transform(frame, values, weights)
    before, after = curve_summary(values, weights), curve_summary(v, w)
    assert after['median'][0] == before['median'][0]
    assert after['p10'][0] < before['p10'][0]
    assert after['p90'][0] > before['p90'][0]
    assert np.all(np.diff(v) > 0)
    np.testing.assert_array_equal(weights, w)
    np.testing.assert_array_equal(model.transform(frame, values, weights, 0)[0], values)
    with pytest.raises(ValueError):
        model.transform(frame, values, weights, 2)


def test_actual_attempts_and_yards_do_not_enter_qb_tail_features():
    frame = pd.DataFrame(dict(passing_yards_avg_5=[200, 230], pass_attempts_avg_5=[25, 33],
                              season=[2024, 2024], week=[1, 2], passing_yards=[100, 500], pass_attempts=[5, 60]))
    x = qb_features(frame)
    pd.testing.assert_frame_equal(x, qb_features(frame.assign(passing_yards=999, pass_attempts=999)))
    assert 'passing_yards' not in x and 'pass_attempts' not in x
