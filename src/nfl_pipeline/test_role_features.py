import pandas as pd
import pytest

from nfl_pipeline.features import ROLE_SIGNAL_STATS, _add_usage_role_features


def frame(include_b_week3=False):
    rows = []
    for week in (1, 2, 3):
        for player, carries in (("A", 10.0), ("B", 30.0)):
            if player == "B" and week == 3 and not include_b_week3:
                continue  # B was a regular but did not play week 3
            row = dict(game_id=f"g{week}", team_abbr="T", season=2024, week=week, player_id=player)
            for stat in ROLE_SIGNAL_STATS:
                row[stat] = carries if stat == "carries" else 0.0
                row[f"{stat}_avg_5"] = carries if week > 1 else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_expected_teammate_who_sits_counts_in_pregame_share_and_rank():
    out = _add_usage_role_features(frame())
    a3 = out[(out.player_id == "A") & (out.week == 3)].iloc[0]
    # Pregame, B (30 carries/game) was expected: A's share is 10/40, not 10/10, and A ranks second.
    assert a3["carries_share_avg_5"] == pytest.approx(10 / 40)
    assert a3["carries_role_rank"] == 2


def test_teammate_ruled_out_pregame_is_excluded():
    out = _add_usage_role_features(frame(), {(2024, 3, "T", "B")})
    a3 = out[(out.player_id == "A") & (out.week == 3)].iloc[0]
    assert a3["carries_share_avg_5"] == pytest.approx(1.0) and a3["carries_role_rank"] == 1


def test_players_who_played_are_unchanged_when_nobody_is_missing():
    out = _add_usage_role_features(frame(include_b_week3=True))
    b3 = out[(out.player_id == "B") & (out.week == 3)].iloc[0]
    assert b3["carries_share_avg_5"] == pytest.approx(30 / 40) and b3["carries_role_rank"] == 1
