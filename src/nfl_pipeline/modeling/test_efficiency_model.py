"""Efficiency challenger: the shrunk baseline rate, scoring with the volume known, and the report (no database)."""
import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling import efficiency_model as em


def test_the_baseline_rate_is_the_players_own_rate_shrunk_toward_his_position():
    df = pd.DataFrame(dict(position=["WR", "WR", "WR"], n_games_prev_10=[10.0, 2.0, 0.0], receiving_yards_avg_10=[60.0, 100.0, np.nan],
                           targets_avg_10=[6.0, 5.0, np.nan]))
    rate = em.baseline_rate(df, "receiving_yards_avg_10", "targets_avg_10", 30.0)
    position = (600 + 200) / (60 + 10)
    assert rate[2] == pytest.approx(position)                                             # no history: the position's rate
    assert rate[0] == pytest.approx((600 + 30 * position) / (60 + 30))
    assert position < rate[0] + 2 and rate[1] > rate[0]                                   # 20 yards a target on 10 targets is pulled most of the way back
    assert rate[1] == pytest.approx((200 + 30 * position) / (10 + 30))


def rows() -> pd.DataFrame:
    weeks = [w for w in range(1, 7) for _ in range(4)]
    y = np.tile([40.0, 60.0, 20.0, 80.0], 6)
    return pd.DataFrame(dict(season=2025, week=weeks, position=["WR", "TE"] * 12, y=y, baseline=y + 10.0, model=y + 4.0,
                             **{"+ opponent": y - 3.0, "+ opponent and weather": y + 6.0}))


def test_estimates_are_scored_on_the_same_rows_against_the_baseline_by_week():
    table = em.evaluate(rows(), ["baseline", *em.VARIANTS])
    everything = table["All"]
    assert everything["rows"] == 24 and everything["mae"] == pytest.approx({"baseline": 10.0, "model": 4.0, "+ opponent": 3.0, "+ opponent and weather": 6.0})
    assert everything["weekly"]["+ opponent"]["mean"] == pytest.approx(-7.0) and everything["weekly"]["+ opponent"]["weeks"] == 6
    assert set(table) == {"All", "Season 2025", "WR", "TE"} and table["WR"]["rows"] == 12


def test_the_report_names_the_best_model_and_the_live_rows():
    table = em.evaluate(rows(), ["baseline", *em.VARIANTS])
    live = dict(rows=50, production=20.0, baseline=21.0, model=20.5, **{"+ opponent": 20.4, "+ opponent and weather": 20.6},
                with_line=dict(rows=40, line=19.0, production=20.2, baseline=21.1, model=20.6, **{"+ opponent": 20.5, "+ opponent and weather": 20.7}))
    text = em.markdown(dict(built_at="now", seasons=[2025, 2026], efficiency={"receiving_yards": table}, live={"receiving_yards": live}))
    assert "## receiving_yards with the volume known" in text and "+ opponent: -7.000" in text
    assert "| receiving_yards, with a line | 40 | 19.00 | 20.20 | 21.10 | 20.60 | 20.50 | 20.70 |" in text
