from datetime import date

import numpy as np
import pandas as pd
import pytest

from nfl_pipeline import role_group


def frame(values, members=None):
    df = pd.DataFrame(dict(game_id="g", team_abbr="PHI", targets_avg_5=values))
    return df, pd.Series(members if members is not None else [True] * len(values), index=df.index)


def test_members_make_the_team_total_and_non_members_are_scored_against_it():
    df, member = frame([8.0, 4.0, 0.0, 6.0], [True, True, True, False])  # the last one is a cold start / debut
    out = role_group.role_features(df, ["targets"], member)
    assert out.team_player_targets_avg5_sum.tolist() == [12.0] * 4
    assert out.targets_share_avg_5.tolist() == pytest.approx([8 / 12, 4 / 12, 0.0, 6 / 18])
    assert out.targets_role_rank.tolist() == [1.0, 2.0, 3.0, 2.0]


def test_expected_teammates_without_a_row_count_in_total_and_rank():
    df, member = frame([5.0, 3.0])
    out = role_group.role_features(df, ["targets"], member, extra={("g", "PHI"): {"targets": [7.0]}})
    assert out.team_player_targets_avg5_sum.tolist() == [15.0, 15.0]
    assert out.targets_role_rank.tolist() == [2.0, 3.0]


def test_recency_uses_the_teams_fourth_most_recent_game():
    dates = [pd.Timestamp(d) for d in ("2026-09-07", "2026-09-14", "2026-09-21", "2026-09-28", "2026-10-05")]
    cut = role_group.recency_cutoffs({"PHI": dates})
    assert cut[("PHI", dates[4])] == dates[0] and cut[("PHI", dates[2])] == dates[0] and cut[("PHI", dates[0])] is None
    last = pd.Series([dates[0], pd.Timestamp("2026-08-01"), pd.NaT])
    assert role_group.is_recent(last, pd.Series([dates[0]] * 3)).tolist() == [True, False, False]
    assert role_group.is_recent(pd.Series([pd.Timestamp("2025-12-28")]), pd.Series([None])).tolist() == [True]


def test_ruled_out_reads_the_injury_report():
    status = pd.Series(["Out", "Doubtful", "Questionable", None, "Injured Reserve"])
    assert role_group.is_ruled_out(status).tolist() == [True, True, False, False, True]


def test_live_and_training_build_the_same_shares_for_the_same_pregame_facts():
    """The live snapshot path and the training path agree when given the same players and history."""
    from nfl_pipeline.modeling import predict_player_props as props
    from nfl_pipeline import features
    days = [pd.Timestamp(d) for d in ("2026-09-07", "2026-09-14", "2026-09-21", "2026-09-28")]
    history = pd.DataFrame([dict(team_abbr="PHI", game_date_et=d, game_id=f"w{i}") for i, d in enumerate(days)])
    today = pd.Timestamp("2026-10-05")
    live = pd.DataFrame(dict(game_id="g5", team_abbr="PHI", game_date_et=today,
                             rest_days=[7, 7, 35, np.nan], cold_start=[False, False, False, True],
                             roster_is_active=True, injury_report_status=[None, "Questionable", None, None],
                             **{f"{s}_avg_5": [6.0, 3.0, 5.0, 4.0] for s in features.ROLE_SIGNAL_STATS}))
    out = props._add_snapshot_role_features(live, history)
    # members: the two recent players; the long-absent returnee and the cold start count zero
    assert out.team_player_targets_avg5_sum.tolist() == [9.0] * 4
    assert out.targets_share_avg_5.iloc[0] == pytest.approx(6 / 9)
