"""The new minutes inputs: who is newly out or back, the player's own lay-off, and tonight's role (no database)."""
import numpy as np
import pandas as pd
import pytest

from nba_props import challenger, minutes_info, model

NO_FILES = pd.DataFrame(columns=["game_id", "player_id", "team", "starter", "roster_status"])


def games(star_out=(10, 11, 12, 20), other_out=(), n=30) -> pd.DataFrame:
    """AAA: a star (34 minutes), his backup (14, or 28 when the star sits) and a regular (24). BBB: two regulars.
    A player who sits has no box-score row."""
    rows = []
    for k in range(n):
        sits = k in star_out
        for player, team, minutes in ((1, "AAA", 34.0), (2, "AAA", 28.0 if sits else 14.0), (3, "AAA", 24.0), (4, "BBB", 30.0), (5, "BBB", 30.0)):
            if (player == 1 and sits) or (player == 3 and k in other_out):
                continue
            rows.append(dict(game_id=f"g{k:02d}", player_id=player, player_name=f"Player {player}", season="2023-24",
                             season_type="Regular Season", game_date=pd.Timestamp("2024-01-01") + pd.Timedelta(days=k), team=team,
                             opponent="BBB" if team == "AAA" else "AAA", home=team == "AAA", minutes=minutes, pts=minutes / 2,
                             reb=minutes / 6, ast=minutes / 10, fg3m=1.0, fga=minutes / 3, fta=2.0, tov=1.0, oreb=1.0, stl=1.0,
                             blk=0.0, plus_minus=0.0))
    return pd.DataFrame(rows)


def at(frame: pd.DataFrame, game: int, player: int) -> pd.Series:
    return frame[(frame.game_id == f"g{game:02d}") & (frame.player_id == player)].iloc[0]


def test_rotation_players_are_out_when_they_do_not_play_or_the_lineup_file_says_so():
    df = games()
    grid = minutes_info.candidates(df, NO_FILES)
    assert at(grid, 10, 1).out == 1 and at(grid, 10, 1).pm == pytest.approx(34.0) and at(grid, 13, 1).out == 0
    assert at(grid, 10, 2).out == 0 and grid[grid.game_id == "g00"].empty                    # nobody has a record before the first game
    file = pd.DataFrame([dict(game_id="g21", player_id=1, team="AAA", starter=False, roster_status="Inactive"),
                         dict(game_id="g21", player_id=2, team="AAA", starter=True, roster_status="Active")])
    filed = minutes_info.candidates(df, file)
    listed = filed[(filed.game_id == "g21") & (filed.team == "AAA")]
    assert dict(zip(listed.player_id, listed.out)) == {1: 1.0, 2: 0.0}                         # declared out although he played; the unlisted regular is left out
    assert at(filed, 20, 1).out == 1                                                           # no file for that game: did not play


def test_absence_inputs_say_what_changed_from_earlier_games_only():
    df = games()
    out = minutes_info.absence_features(df, minutes_info.candidates(df, NO_FILES))
    first = at(out, 10, 2)                                           # the star sits for the first time: no history yet
    assert first.wo_min_adj == 0 and first.absent_new_min == pytest.approx(34.0) and first.absent_back_min == 0
    exposure = pd.Series([0.0] * 9 + [1.0] * 3).ewm(halflife=3).mean().iloc[-1]       # the pair's record starts at the second game
    weight = (3 / 5) * (9 / 11)                                      # three games at 28 against nine at 14
    back = at(out, 13, 2)                                            # he returns
    assert back.wo_min_adj == pytest.approx(-14.0 * weight * exposure) and back.absent_back_min == pytest.approx(34.0 * exposure)
    assert back.absent_new_min == 0 and back.wo_support == pytest.approx(exposure * 3)
    again = at(out, 20, 2)                                           # he sits again: now the history says about +14
    assert 5.0 < again.wo_min_adj < 14.0 and again.wo_pts_rate_adj == pytest.approx(0.0)      # the backup's per-minute scoring did not move
    assert out[out.player_id == 3].wo_min_adj.abs().max() == pytest.approx(0.0)               # the regular's minutes never moved
    assert 0 < again.absent_new_sim < again.absent_new_min                                     # weighted by how alike the two players are
    changed = df.copy()
    changed.loc[(changed.game_id == "g20") & (changed.player_id == 2), "minutes"] = 40.0       # tonight's result must not reach tonight's inputs
    after = minutes_info.absence_features(changed, minutes_info.candidates(changed, NO_FILES))
    assert at(after, 20, 2).wo_min_adj == pytest.approx(again.wo_min_adj)
    assert at(after, 21, 2).wo_min_adj != pytest.approx(at(out, 21, 2).wo_min_adj)


def test_return_inputs_follow_a_lay_off_for_ten_games():
    out = minutes_info.return_features(games(star_out=(), other_out=range(5, 10)))
    back = at(out, 10, 3)
    assert (back.days_off, back.team_games_missed, back.games_since_return, back.absence_len) == (6, 5, 0, 5)
    assert back.min_pre_absence == pytest.approx(24.0) and back.return_gap == pytest.approx(0.0)
    assert at(out, 19, 3).games_since_return == 9 and at(out, 19, 3).min_pre_absence == pytest.approx(24.0)
    later = at(out, 20, 3)
    assert later.games_since_return == 10 and later.absence_len == 0 and np.isnan(later.min_pre_absence)
    steady = out[out.player_id == 2]
    assert (steady.games_since_return == 10).all() and steady.team_games_missed.iloc[1:].eq(0).all() and np.isnan(steady.team_games_missed.iloc[0])


def test_role_inputs_compare_tonights_lineup_with_earlier_starts_and_bench_games():
    days = pd.date_range("2024-01-01", periods=21, freq="D")
    starts = [1.0] * 10 + [0.0] * 10 + [1.0]
    f = pd.DataFrame(dict(game_id=[f"g{k:02d}" for k in range(21)], player_id=1, team="AAA", game_date=days,
                          minutes=[30.0] * 10 + [15.0] * 10 + [31.0], starter=starts, min_ewm3=[np.nan] + [30.0] * 10 + [15.0] * 10,
                          starter_rate=[np.nan] + [1.0] * 10 + [0.1] * 10))
    out = minutes_info.role_features(f)
    tonight = at(out, 20, 1)
    assert tonight.min_as_starter == pytest.approx(30.0) and tonight.min_as_bench == pytest.approx(15.0)
    assert tonight.min_in_role == pytest.approx(30.0) and tonight.role_min_gap == pytest.approx(15.0)     # a start again: +15 on his recent minutes
    assert (tonight.starter_flip, tonight.games_in_role, tonight.team_new_starters) == (1, 10, 1)
    moved = at(out, 10, 1)                                           # first game off the bench: no bench history yet
    assert moved.starter_flip == -1 and np.isnan(moved.min_in_role) and moved.games_in_role == 0
    unknown = minutes_info.role_features(f.assign(starter=np.nan))
    assert unknown.min_in_role.isna().all() and unknown.team_new_starters.isna().all()                    # no lineup file: nothing invented


def test_the_inputs_join_the_pinned_frame_one_row_per_player_game():
    df = games()
    lineups = df.assign(starter=df.minutes >= 24, roster_status="Active")[["game_id", "player_id", "team", "starter", "roster_status"]]
    sat = pd.DataFrame([dict(game_id=f"g{k:02d}", player_id=1, team="AAA", starter=False, roster_status="Inactive") for k in (10, 11, 12, 20)])
    lineups = pd.concat([lineups, sat], ignore_index=True)
    f = model.features(df, lineups)
    out = minutes_info.add(f, df, lineups)
    assert len(out) == len(f) and set(minutes_info.FEATURES) <= set(out.columns) and not out[minutes_info.GROUPS["absence"]].isna().any().any()
    assert set(minutes_info.FEATURES).isdisjoint(model.feature_columns(f))
    promoted = at(out, 20, 2)                                        # the backup starts when the star sits
    assert promoted.starter_flip == 1 and promoted.role_min_gap > 5 and promoted.wo_min_adj > 5 and promoted.out_count == 1
    marks = challenger.situations(out)
    assert marks["Role changed tonight"].sum() == 4 and marks["30+ rotation minutes newly out"].any() and marks["All"].all()
