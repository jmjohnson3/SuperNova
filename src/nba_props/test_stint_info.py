"""Rotation stints: the stored rows, one game's floor time by kind, and the inputs built from earlier games (no database)."""
import numpy as np
import pandas as pd
import pytest

from nba_props import minutes_info, model, stint_info, stints
from nba_props.test_minutes_info import at, games

M = 600  # tenths of a second in a minute


def one_game(scores=True) -> pd.DataFrame:
    """Team 100: a starter (0-24, 30-44), his backup (24-30, 44-48) and a regular (0-36, 36-48). Team 200: three players.
    Team 100 is 2 up after 24 minutes, 20 up after 36, 24 up after 44 and wins by 23."""
    rows = [(1, 100, 0, 24, 2), (1, 100, 30, 44, 22), (2, 100, 24, 30, 0), (2, 100, 44, 48, -1), (3, 100, 0, 36, 20), (3, 100, 36, 48, 3),
            (4, 200, 0, 48, -23), (5, 200, 0, 44, -24), (6, 200, 44, 48, 1)]
    return pd.DataFrame([dict(game_id="g1", player_id=p, team_id=t, in_time=a * M, out_time=b * M, pt_diff=float(d) if scores else np.nan)
                         for p, t, a, b, d in rows])


def test_stored_rows_keep_one_stint_per_entry_time():
    feed = pd.DataFrame([dict(GAME_ID="g1", PERSON_ID=7, TEAM_ID=100, IN_TIME_REAL=0.0, OUT_TIME_REAL=3000.0, PLAYER_PTS=4, PT_DIFF=-2.0, USG_PCT=0.2),
                         dict(GAME_ID="g1", PERSON_ID=7, TEAM_ID=100, IN_TIME_REAL=0.0, OUT_TIME_REAL=4750.0, PLAYER_PTS=6, PT_DIFF=-7.0, USG_PCT=0.1),
                         dict(GAME_ID="g1", PERSON_ID=7, TEAM_ID=100, IN_TIME_REAL=9000.0, OUT_TIME_REAL=np.nan, PLAYER_PTS=0, PT_DIFF=0.0, USG_PCT=0.0),
                         dict(GAME_ID="g1", PERSON_ID=8, TEAM_ID=200, IN_TIME_REAL=600.0, OUT_TIME_REAL=1200.0, PLAYER_PTS=np.nan, PT_DIFF=np.nan, USG_PCT=np.nan)])
    assert stints.to_rows([feed.iloc[:3], feed.iloc[3:]]) == [("g1", 7, 100, 0, 4750, 6, -7.0, 0.1), ("g1", 8, 200, 600, 1200, None, None, None)]
    assert stints.to_rows([feed.iloc[0:0]]) == []


def test_one_game_gives_floor_time_by_kind_and_time_shared_with_each_teammate():
    players, pairs = stint_info.summarise(one_game())
    star, backup = players[players.player_id == 1].iloc[0], players[players.player_id == 2].iloc[0]
    assert (star.sec, star.q4_sec, star.garbage_sec, star.first_in, star.last_out, star.n_stints, star.longest) == (38 * M, 8 * M, 8 * M, 0, 44 * M, 2, 24 * M)
    assert (backup.sec, backup.q4_sec, backup.garbage_sec, backup.first_in) == (10 * M, 4 * M, 4 * M, 24 * M)
    assert (star.game_len, star.garbage_len, star.close_len) == (48 * M, 12 * M, 0)          # 20 up entering the fourth: all of it is garbage time
    shared = pairs.set_index(["player_id", "mate"]).shared
    assert shared[(1, 2)] == 0 and shared[(2, 1)] == 0 and shared[(1, 3)] == 38 * M and (1, 4) not in shared.index    # teammates only
    row = pairs[(pairs.player_id == 2) & (pairs.mate == 1)].iloc[0]
    assert (row.sec, row.mate_sec, row.game_len) == (10 * M, 38 * M, 48 * M)
    level, _ = stint_info.summarise(one_game(scores=False))                                    # no score known: treated as level throughout
    assert level.garbage_len.eq(0).all() and level.close_len.eq(4 * M).all() and level[level.player_id == 2].close_sec.iloc[0] == 4 * M


def test_games_whose_stints_do_not_match_the_box_score_are_left_out():
    players = pd.DataFrame(dict(game_id=["a", "a", "b", "b"], player_id=[1, 2, 1, 2], sec=[30.0 * M, 20 * M, 30 * M, 11 * M]))
    pairs = pd.DataFrame(dict(game_id=["a", "b"], player_id=[1, 1], mate=[2, 2]))
    box = pd.DataFrame(dict(game_id=["a", "a", "b", "b"], player_id=[1, 2, 1, 2], minutes=[30.0, 20.2, 30.0, 20.0]))
    kept, kept_pairs, dropped = stint_info.verified(players, pairs, box)
    assert set(kept.game_id) == {"a"} and set(kept_pairs.game_id) == {"a"} and dropped == 1


def test_stint_inputs_come_from_earlier_games_and_skip_games_without_stints():
    days = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame(dict(game_id=[f"g{k}" for k in range(6)], player_id=1, game_date=days, minutes=30.0))
    players = pd.DataFrame([dict(game_id=f"g{k}", player_id=1, sec=30.0 * M, q4_sec=6.0 * M, garbage_sec=0.0, close_sec=2.0 * M, first_in=0.0,
                                 last_out=(34 if k == 3 else 44) * M, n_stints=3.0, longest=12.0 * M, game_len=48.0 * M, garbage_len=12.0 * M,
                                 close_len=4.0 * M) for k in range(6) if k != 2])
    out = stint_info.stint_features(df, players).set_index("game_id")
    assert out.loc["g0"].drop("player_id").isna().all()                                       # nothing earlier
    first = out.loc["g1"]                                                                     # 30 of 36 competitive minutes: a 40-minute pace
    assert first.comp_min_ewm3 == pytest.approx(40.0) and first.garbage_lift3 == pytest.approx(10.0) and first.garbage_share_ewm10 == 0
    assert first.q4_share_ewm10 == pytest.approx(0.2) and first.close_share_ewm10 == pytest.approx(0.5)
    assert (first.first_in_ewm5, first.stints_ewm5, first.longest_ewm5) == (0, 3, 12)
    assert out.loc["g3"].comp_min_ewm3 == pytest.approx(40.0) and np.isnan(out.loc["g3"].last_out_gap)   # the game before had no stints
    assert out.loc["g4"].last_out_gap == pytest.approx(-10.0) and out.loc["g5"].last_out_gap > 0          # he left ten minutes early in g3
    assert out.loc["g4"].last_comp_gap == pytest.approx(0.0)
    short = stint_info.stint_features(df, players.assign(garbage_len=30.0 * M))               # under 24 competitive minutes: no pace figure
    assert short.comp_min_ewm3.isna().all()


def test_substitute_inputs_weight_the_change_by_whose_bench_time_the_player_fills():
    days = pd.date_range("2024-01-01", periods=8, freq="D")
    df = pd.DataFrame(dict(game_id=[f"g{k}" for k in range(8)], game_date=days))
    record = pd.DataFrame([dict(game_id=f"g{k}", player_id=p, mate=1, shared=shared * M, sec=sec * M, mate_sec=38.0 * M, game_len=48.0 * M)
                           for k in range(5) for p, shared, sec in ((2, 0.0, 10.0), (3, 38.0, 48.0))])
    change = lambda game, player, sign: dict(game_id=f"g{game}", player_id=player, mate=1, game_date=days[game], pm=34.0, change=float(sign),
                                             newly=float(sign > 0), back=float(sign < 0))
    absence = pd.DataFrame([change(5, 2, 1), change(5, 3, 1), change(6, 2, -1), change(6, 9, 1), change(7, 2, 0)])
    out = stint_info.substitute_features(df, absence, record).set_index(["game_id", "player_id"])
    backup = out.loc[("g5", 2)]                                       # always on when the starter sits, never with him; five shared games
    assert backup.sub_new == pytest.approx(34.0 * 5 / 8) and backup.sub_back == 0 and backup.sub_support == 5
    assert out.loc[("g5", 3)].sub_new == pytest.approx(0.0)                                   # plays the whole game either way
    assert out.loc[("g6", 2)].sub_back == pytest.approx(34.0 * 5 / 8) and out.loc[("g6", 9)].sub_new == 0     # no shared record: nothing claimed
    assert ("g7", 2) not in out.index                                                         # nothing changed that night
    same_day = pd.concat([record, pd.DataFrame([dict(game_id="g5", player_id=2, mate=1, shared=10.0 * M, sec=10.0 * M, mate_sec=38.0 * M, game_len=48.0 * M)])])
    again = stint_info.substitute_features(df, absence, same_day).set_index(["game_id", "player_id"])
    assert again.loc[("g5", 2)].sub_new == pytest.approx(backup.sub_new)                      # tonight's stints never reach tonight's inputs
    assert again.loc[("g6", 2)].sub_back < backup.sub_new


def test_the_stint_inputs_join_the_frame_one_row_per_player_game():
    df = games()
    lineups = df.assign(starter=df.minutes >= 24, roster_status="Active")[["game_id", "player_id", "team", "starter", "roster_status"]]
    sat = pd.DataFrame([dict(game_id=f"g{k:02d}", player_id=1, team="AAA", starter=False, roster_status="Inactive") for k in (10, 11, 12, 20)])
    lineups = pd.concat([lineups, sat], ignore_index=True)
    f = minutes_info.add(model.features(df, lineups), df, lineups)
    covered = df[df.game_id >= "g05"]                                 # stints stored from the sixth game on: the backup plays the star's bench minutes
    stored = pd.DataFrame([dict(game_id=r.game_id, player_id=r.player_id, team_id=100 if r.team == "AAA" else 200,
                                in_time=int((48 - r.minutes) * M) if r.player_id == 2 else 0,
                                out_time=48 * M if r.player_id == 2 else int(r.minutes * M), pt_diff=0.0) for r in covered.itertuples()])
    out = stint_info.add(f, df, lineups, stored)
    assert len(out) == len(f) and set(stint_info.FEATURES) <= set(out.columns) and set(stint_info.FEATURES).isdisjoint(f.columns)
    assert out[out.game_id <= "g05"][stint_info.FEATURES].isna().all().all()                  # before any earlier game has stints: empty
    late = out[out.game_id > "g06"]
    assert not late[stint_info.GROUPS["substitutes"]].isna().any().any() and late.comp_min_ewm3.notna().all()
    promoted = at(out, 20, 2)                                         # the star sits again: his backup fills his bench time
    assert promoted.sub_new > 5 and at(out, 20, 3).sub_new < 0      # the regular shares the floor with the star: not his substitute
    empty = stint_info.add(f, df, lineups, stored.iloc[0:0])
    assert len(empty) == len(f) and empty[stint_info.FEATURES].isna().all().all()
