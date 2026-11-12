"""The NBA prop challenger's inputs and scoring (no database, no real model fits)."""
import numpy as np
import pandas as pd
import pytest

from nba_props import challenger


class Mean:
    """Stands in for LightGBM: forecasts the training average of the target."""
    def __init__(self, X, y):
        self.value = float(np.mean(y))

    def predict(self, X):
        return np.full(len(X), self.value)


def frame() -> pd.DataFrame:
    days = pd.date_range("2024-01-01", periods=120, freq="D")
    rows = []
    for n, day in enumerate(days):
        for player, team, opponent, minutes in ((1, "AAA", "BBB", 30.0 + n % 5), (2, "BBB", "AAA", 20.0), (3, "AAA", "BBB", 0.0)):
            rows.append(dict(game_id=f"g{n}", player_id=player, player_name=f"Player {player}", team=team, opponent=opponent,
                             game_date=day, season="2023-24", minutes=minutes, games_before=n, min_ewm3=minutes or 10.0,
                             min_ewm10=minutes or 10.0, starter=1.0 if player == 1 else 0.0, out_count=2 if n % 10 == 0 else 0,
                             pts=minutes / 2, reb=minutes / 6, ast=minutes / 10, fg3m=1.0, x=float(n),
                             **{f"{s}_rate_ewm10": 0.5 for s in challenger.STATS}, **{f"opp_allow_{s}_pm": 1.1 for s in challenger.STATS}))
    return pd.DataFrame(rows)


def test_the_game_line_is_seen_from_each_teams_side():
    lines = pd.DataFrame([dict(game_date=pd.Timestamp("2024-01-01"), home_team="AAA", away_team="BBB", spread=-6.5, total=220.5)])
    out = challenger.add_game_line(frame(), lines)
    home, away = out[(out.game_id == "g0") & (out.team == "AAA")].iloc[0], out[(out.game_id == "g0") & (out.team == "BBB")].iloc[0]
    assert (home.mkt_spread, away.mkt_spread) == (-6.5, 6.5) and home.mkt_total == away.mkt_total == 220.5
    assert home.mkt_team_total == pytest.approx(113.5) and home.mkt_opp_total == pytest.approx(107.0)        # the favourite scores more
    assert away.mkt_team_total == pytest.approx(107.0) and home.mkt_abs_spread == away.mkt_abs_spread == 6.5
    assert out[out.game_id == "g1"].mkt_total.isna().all() and len(out) == len(frame())                       # no line: left empty


def test_minutes_are_forecast_only_from_earlier_games():
    f = frame()
    seen = []

    def fitter(X, y):
        seen.append((X.x.max(), len(X)))
        return Mean(X, y)
    f.loc[f.minutes > 0, "games_before"] = 5                                   # every row that played is eligible
    minutes = challenger.minutes_walk_forward(f, ["x"], start="2024-03-01", fitter=lambda X, y: Mean(X, y), min_rows=50)
    assert minutes[f.game_date < "2024-03-01"].isna().all() and minutes[f.minutes == 0].isna().all()
    assert minutes[(f.game_date >= "2024-03-01") & (f.minutes > 0)].isna().sum() == 0
    challenger.minutes_walk_forward(f, ["x"], start="2024-03-01", fitter=fitter, min_rows=50)
    assert [day for day, _ in seen] == [59.0, 90.0]                            # March is fitted through Feb 29, April through Mar 31
    assert challenger.minutes_walk_forward(f, ["x"], start="2024-03-01", fitter=fitter).isna().all()             # too little history


def test_minutes_features_apply_the_rates_to_the_forecast_minutes():
    f = frame()
    out = challenger.add_minutes(f, pd.Series(np.where(f.minutes > 0, 24.0, np.nan), index=f.index))
    row = out[out.player_id == 1].iloc[0]
    assert row.pred_min == 24.0 and row.pred_min_ratio == pytest.approx(24.0 / row.min_ewm3)
    assert row.pts_at_pred_min == pytest.approx(24.0 * 0.5 * 1.1) and set(challenger.MINUTES_FEATURES) <= set(out.columns)
    assert out[out.player_id == 3].pred_min.isna().all()


def test_the_walk_forward_keeps_both_spreads_and_scores_every_test_month():
    f = frame()
    f.loc[f.minutes > 0, "games_before"] = 5
    log = []
    preds = challenger.walk_forward(f, ["x"], "2024-03-01", fitter=lambda X, y: Mean(X, y), log=log.append)
    assert len(log) == 2 and preds.game_date.min() == pd.Timestamp("2024-03-01") and (preds.minutes > 0).all()
    for s in challenger.STATS:
        assert {f"pred_{s}", f"disp_a_{s}", f"honest_a_{s}", f"honest_b_{s}"} <= set(preds.columns)
    assert (preds.honest_a_pts == preds.disp_a_pts).all()                      # too few earlier forecasts: the training-fit spread
    swapped = challenger.honest(preds.assign(honest_a_pts=9.0))
    assert (swapped.disp_a_pts == 9.0).all() and (preds.disp_a_pts != 9.0).all()
    rng = np.random.default_rng(1)
    mean = rng.uniform(4, 25, 30000)
    history = pd.DataFrame(dict(pts=rng.poisson(mean).astype(float), pred_pts=mean))
    a, b = challenger.honest_dispersion(history, "pts", (5.0, 5.0))
    assert 0.9 < a < 1.1 and b < 0.01 and challenger.honest_dispersion(history.head(100), "pts", (5.0, 5.0)) == (5.0, 5.0)


def test_the_paired_comparison_says_which_way_and_how_sure():
    rng = np.random.default_rng(3)
    n = 4000
    base = pd.DataFrame(dict(game_date=pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 100, n), unit="D"),
                             name_key=[f"p{k}" for k in range(n)], stat="pts", actual=rng.normal(15, 6, n)))
    base["pred"] = base.actual + rng.normal(0, 5, n)
    base["over"] = rng.random(n) < 0.5
    base["p_model"] = 0.5
    better = base.assign(pred=base.actual + (base.pred - base.actual) * 0.8, p_model=np.where(base.over, 0.6, 0.4))
    out = challenger.paired(base, better)["pts"]
    assert out["rows"] == n and out["mae"] < 0 and out["mae_low"] < out["mae"] < out["mae_high"] < 0
    assert out["ll"] < 0 and out["ll_high"] < 0
    same = challenger.paired(base, base)["pts"]
    assert same["mae"] == 0 and same["ll"] == 0
    assert list(challenger.log_loss(np.array([0.5, 1.0]), np.array([True, False]))) == pytest.approx([np.log(2), -np.log(1e-6)])


def test_minutes_accuracy_and_the_report():
    f = frame()
    f.loc[f.minutes > 0, "games_before"] = 5
    f = challenger.add_minutes(f, pd.Series(np.where(f.minutes > 0, f.minutes + 1.0, np.nan), index=f.index))
    acc = challenger.minutes_accuracy(f, "2024-03-01")
    assert acc["model"] == pytest.approx(1.0) and acc["last_3_average"] == 0 and acc["starters"]["rows"] < acc["rows"]
    assert acc["two_or_more_out"]["rows"] > 0
    score = dict(rows=10, model_mae=5.0, line_mae=4.9, blend_model_weight=0.2, blend_model_se=0.05, logloss_model=0.70,
                 logloss_market=0.69, picks_4pct=dict(picks=4, hit_rate=0.5, roi=-0.04))
    entry = dict(all_games_mae={s: 1.0 for s in challenger.STATS}, training_fit_spread={s: score for s in challenger.LINE_STATS},
                 honest_spread={s: score for s in challenger.LINE_STATS})
    change = dict(rows=10, mae=-0.02, mae_low=-0.03, mae_high=-0.01, ll=-0.001, ll_low=-0.002, ll_high=0.0)
    report = dict(built_at="now", test_from="2024-03-01", game_line_coverage=0.6, lines_with_game_line=0.9,
                  variants=dict(baseline=entry, both=dict(entry, minutes=acc, against_baseline={s: change for s in challenger.LINE_STATS})))
    text = challenger.markdown(report)
    assert "| Points | 10 | 4.900 | 5.000 | 5.000 |" in text and "-0.020 (-0.030 to -0.010)" in text
    assert "Minutes model with the game line" in text and "90% of the scored lines (60% of all test" in text and "+ game line" not in text
