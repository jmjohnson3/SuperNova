import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling import volume_model as vm


def test_no_same_game_outcome_is_a_candidate_feature():
    features = set(vm.candidate_features())
    assert not features & vm.OUTCOMES and not features & vm.IDENTIFIERS
    assert {"targets_avg_5", "carries_avg_10", "team_implied_points"} <= features


def test_leak_guard_drops_a_feature_that_tracks_the_outcome():
    rng = np.random.default_rng(1)
    y = rng.poisson(5, 1000).astype(float)
    frame = pd.DataFrame(dict(targets=y, leaky=y + rng.normal(0, 0.05, 1000), honest=y + rng.normal(0, 4, 1000)))
    kept, dropped = vm.leak_guard(frame, "targets", ["leaky", "honest"])
    assert kept == ["honest"] and "leaky" in dropped


def test_walk_forward_blocks_train_only_on_earlier_games():
    frame = pd.DataFrame([dict(season=s, week=w) for s in (2024, 2025) for w in range(1, 19)])
    seen = list(vm.blocks(frame, [2025]))
    assert [(season, span) for season, span, _, _ in seen] == [(2025, (1, 4)), (2025, (5, 8)), (2025, (9, 12)),
                                                              (2025, (13, 16)), (2025, (17, 18))]
    for season, (start, _), train, test in seen:
        trained = frame[train]
        assert not ((trained.season == season) & (trained.week >= start)).any()
        assert not (train & test).any()


def test_evaluation_compares_on_identical_rows_and_requires_every_check():
    rng = np.random.default_rng(2)
    rows = []
    for week in range(1, 13):
        for i in range(60):
            truth = rng.poisson(6)
            rows.append(dict(season=2025, week=week, position="WR", targets=truth, model=truth + rng.normal(0, 1),
                             targets_avg_3=truth + rng.normal(0, 2.5), targets_avg_5=truth + rng.normal(0, 2),
                             targets_avg_10=truth + rng.normal(0, 2), receiver_projected_targets_v3=np.nan))
    result = vm.evaluate(pd.DataFrame(rows), "targets")
    assert "receiver_projected_targets_v3" not in result["scores"]  # a baseline that is mostly missing is skipped
    assert result["volume_accepted"] and result["weekly_mae_difference_vs_best"]["upper"] < 0
    worse = pd.DataFrame(rows).assign(model=lambda d: d.targets + rng.normal(0, 3, len(d)))
    assert not vm.evaluate(worse, "targets")["volume_accepted"]


def test_volume_times_efficiency_shrinks_toward_the_position_rate():
    df = pd.DataFrame(dict(model=[10.0, 10.0], position=["WR", "WR"], n_games_prev_10=[10, 1],
                           receiving_yards_avg_10=[80.0, 200.0], targets_avg_10=[10.0, 10.0]))
    built = vm.volume_times_efficiency(df, "receiving_yards_avg_10", "targets_avg_10", 30.0)
    position_rate = (800 + 200) / (100 + 10)
    assert built.iloc[0] == pytest.approx(10 * (800 + 30 * position_rate) / (100 + 30))
    assert built.iloc[1] == pytest.approx(10 * (200 + 30 * position_rate) / (10 + 30))  # one big game barely moves it
