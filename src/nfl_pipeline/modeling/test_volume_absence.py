"""Teammate-absence volume inputs: built from earlier games and the injury report only (no database)."""
import pandas as pd
import pytest

from nfl_pipeline.modeling import volume_absence as va


def season() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Team AAA, weeks 1-6. The lead receiver (1) plays weeks 1-3, is ruled out weeks 4-5 and returns in week 6.
    His understudy (2) gets 3 targets with him and 9 without; the tight end (3) always gets 5; QB 9 plays throughout."""
    rows = []
    for week in range(1, 7):
        star = week not in (4, 5)
        players = [(2, "WR", 3.0 if star else 9.0, 0.0), (3, "TE", 5.0, 0.0), (9, "QB", 0.0, 4.0)] + ([(1, "WR", 10.0, 0.0)] if star else [])
        rows += [dict(season=2025, week=week, game_id=f"g{week}", game_date_et=pd.Timestamp("2025-09-07") + pd.Timedelta(weeks=week - 1),
                      team_abbr="AAA", player_id=p, position=pos, targets=t, carries=c) for p, pos, t, c in players]
    injuries = pd.DataFrame([dict(season=2025, week=4, team_abbr="AAA", player_id=1, report_status="Out"),
                             dict(season=2025, week=5, team_abbr="AAA", player_id=1, report_status="Doubtful"),
                             dict(season=2025, week=6, team_abbr="AAA", player_id=1, report_status="Questionable"),
                             dict(season=2025, week=3, team_abbr="AAA", player_id=3, report_status=None)])
    return pd.DataFrame(rows), injuries


def at(features: pd.DataFrame, week: int, player: int) -> pd.Series:
    return features[(features.game_id == f"g{week}") & (features.player_id == player)].iloc[0]


def test_recent_teammates_carry_their_volume_and_the_injury_report():
    frame, injuries = season()
    mates = va.teammates(frame, injuries).set_index(["game_id", "mate"])
    assert "g1" not in mates.index.get_level_values(0)                                      # nobody has played yet
    star4, star5, star6 = mates.loc[("g4", 1)], mates.loc[("g5", 1)], mates.loc[("g6", 1)]
    assert star4.out and star4.recent_targets == 10 and star4.missed == 0 and star4.new == 1 and star4.back == 0
    assert star5.out and star5.missed == pytest.approx(1 / 3) and star5.new == pytest.approx(2 / 3)
    assert not star6.out and star6.back == pytest.approx(2 / 3) and star6.new == 0          # on the report, not ruled out, missed two of three
    assert not mates.loc[("g3", 3)].out and mates.loc[("g3", 3)].back == 0                  # listed without a tag, missed nothing


def test_absence_inputs_use_only_earlier_games_and_the_report():
    frame, injuries = season()
    f = va.absence_features(frame, injuries)
    assert len(f) == len(frame) and set(va.FEATURES) <= set(f.columns)
    assert (f[f.game_id.isin(["g1", "g2", "g3"])][va.FEATURES] == 0).all().all()            # nobody out
    first = at(f, 4, 2)                                                                     # the star's first game out: no history without him yet
    assert first.vacated_targets == 10 and first.vacated_new_targets == 10 and first.vacated_position_targets == 10
    assert first.without_adj_targets == 0 and first.teammates_out == 1 and first.quarterback_out == 0
    assert at(f, 4, 3).vacated_position_targets == 0                                        # the tight end is not at his position
    second = at(f, 5, 2)                                                                    # one game without (9), two on record with (3)
    assert second.vacated_new_targets == pytest.approx(10 * 2 / 3)
    assert second.without_adj_targets == pytest.approx((9 - 3) * (2 / 4) * (1 / 3) * (2 / 3))
    back = at(f, 6, 2)                                                                      # he returns: the understudy's extra targets go
    assert back.vacated_targets == 0 and back.returning_targets == pytest.approx(10 * 2 / 3)
    assert back.without_adj_targets == pytest.approx(-(9 - 3) * (2 / 4) * (2 / 4) * (2 / 3))
    assert at(f, 6, 3).without_adj_targets == 0                                             # the tight end's targets never moved
    changed = frame.copy()
    changed.loc[(changed.game_id == "g5") & (changed.player_id == 2), "targets"] = 20.0     # this game's result
    assert at(va.absence_features(changed, injuries), 5, 2)[va.FEATURES].equals(second[va.FEATURES])


def test_a_quarterback_ruled_out_is_flagged_and_the_report_reads():
    frame, injuries = season()
    injuries = pd.concat([injuries, pd.DataFrame([dict(season=2025, week=3, team_abbr="AAA", player_id=9, report_status="Out")])])
    f = va.absence_features(frame, injuries)
    assert at(f, 3, 2).quarterback_out == 1 and at(f, 3, 2).vacated_carries == 4 and at(f, 2, 2).quarterback_out == 0
    oof = frame.merge(f, on=["game_id", "player_id"]).assign(model=4.0)
    table = va.compare(oof, oof.assign(model=5.0), "targets")
    assert table["All"]["rows"] == len(frame) and "8+ targets newly vacated" in table and table["WR"]["rows"] == 10
    doc = dict(built_at="now", seasons=[2025], volume={"targets": table}, yardage={"receiving_yards": dict(rows=5, production_mae=20.0, base_mae=21.0, absence_mae=20.5)},
               coverage=dict(with_someone_out=0.3, newly_vacated=0.1))
    text = va.markdown(doc)
    assert "## targets" in text and "| receiving_yards | 5 | 20.00 | 21.00 | 20.50 |" in text
