"""Packets: names match across sources, nothing after the cutoff is counted, team statistics add up, the text reads."""
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from nfl_ai_report import packets, team_stats


def test_names_match_between_the_injury_report_and_the_game_logs():
    pairs = [("A.J. Brown", "A.Brown"), ("Amon-Ra St. Brown", "A.St. Brown"), ("Marvin Harrison Jr.", "M.Harrison"),
             ("Kenneth Walker III", "K.Walker"), ("Jaxon Smith-Njigba", "J.Smith-Njigba"), ("Ja'Marr Chase", "J.Chase")]
    for full, short in pairs:
        assert packets.name_key(full) == packets.name_key(short) != ""
    assert packets.name_key("A.St. Brown") != packets.name_key("A.Brown") and packets.name_key(None) == ""


def test_the_report_time_is_11_eastern_or_90_minutes_before_an_early_kickoff():
    assert packets.report_time(date(2026, 11, 15)) == datetime(2026, 11, 15, 16, 0, tzinfo=timezone.utc)
    london = pd.Timestamp("2026-11-15 14:30", tz="UTC")
    assert packets.report_time(date(2026, 11, 15), london) == datetime(2026, 11, 15, 13, 0, tzinfo=timezone.utc)


def games():
    rows = [("g1", 1, "2026-09-13 17:00", "DET", "NE", 27, 20, 3.0), ("g2", 2, "2026-09-20 17:00", "NE", "DET", 24, 24, -1.5),
            ("g3", 3, "2026-09-27 17:00", "DET", "NE", None, None, 2.5)]
    g = pd.DataFrame(rows, columns=["game_id", "week", "start_ts_utc", "home", "away", "home_score", "away_score", "spread_line"])
    return g.assign(start_ts_utc=pd.to_datetime(g.start_ts_utc, utc=True))


def test_records_count_only_finished_games_before_the_cutoff():
    g = games()
    det = packets.record(g, "DET", datetime(2026, 9, 28, tzinfo=timezone.utc))
    assert det["record"] == "1-0-1" and det["ats"] == "1-1"           # won by 7 laying 3; tied laying 1.5 on the road
    assert det["record"] == "1-0-1" and det["last_results"] == ["W +7", "T +0"]
    early = packets.record(g, "NE", datetime(2026, 9, 14, tzinfo=timezone.utc))
    assert early["record"] == "0-1" and early["ats"] == "0-1" and early["point_diff_per_game"] == -7
    assert packets.record(g, "NE", datetime(2026, 9, 1, tzinfo=timezone.utc))["point_diff_per_game"] is None


def play(game, week, off, deff, **kw):
    base = dict(game_id=game, season_type="REG", week=week, game_date=f"2026-09-{12 + 7 * week}", home_team="DET", away_team="NE",
                posteam=off, defteam=deff, play_type="run", yards_gained=0, rush_attempt=0, pass_attempt=0, sack=0, qb_kneel=0, qb_spike=0,
                epa=0.0, success=0, third_down_converted=0, third_down_failed=0, interception=0, fumble_lost=0, penalty=0, penalty_team=None,
                penalty_yards=0, punt_attempt=0, kick_distance=None, return_yards=None, touchback=0, fixed_drive=1, fixed_drive_result="Punt",
                drive_inside20=0, drive_time_of_possession="2:30")
    return {**base, **kw}


def test_team_statistics_per_game_and_before_a_date():
    pbp = pd.DataFrame([
        play("g1", 0, "DET", "NE", yards_gained=10, rush_attempt=1, epa=0.5, success=1, third_down_converted=1),
        play("g1", 0, "DET", "NE", play_type="pass", yards_gained=20, pass_attempt=1, epa=1.0, success=1, third_down_failed=1, interception=1),
        play("g1", 0, "DET", "NE", play_type="run", qb_kneel=1, yards_gained=-1),                                        # not counted
        play("g1", 0, "NE", "DET", yards_gained=3, rush_attempt=1, fixed_drive=2, drive_inside20=1, fixed_drive_result="Touchdown",
             drive_time_of_possession="4:00"),
        play("g1", 0, "NE", "DET", play_type="punt", punt_attempt=1, kick_distance=50, return_yards=10, fixed_drive=3),
        play("g1", 0, "NE", "DET", play_type="no_play", penalty=1, penalty_team="DET", penalty_yards=15, fixed_drive=3),
        play("g2", 1, "DET", "NE", yards_gained=99, rush_attempt=1)])                                                     # after the cutoff
    rows = team_stats.game_rows(pbp)
    det = team_stats.before(rows, "DET", date(2026, 9, 15))
    assert det["games"] == 1 and det["yards_per_game"] == 30 and det["rush_yards_per_game"] == 10 and det["pass_yards_per_game"] == 20
    assert det["third_down_pct"] == 0.5 and det["giveaways"] == 1 and det["takeaways"] == 0 and det["turnover_margin"] == -1
    assert det["penalty_yards_per_game"] == 15 and det["rush_yards_allowed_per_game"] == 3 and det["red_zone_td_pct_allowed"] == 1.0
    ne = team_stats.before(rows, "NE", date(2026, 9, 15))
    assert ne["net_punt_avg"] == 40 and ne["takeaways"] == 1 and ne["time_of_possession_per_game"] == "6:30"
    assert team_stats.before(rows, "DET", date(2026, 9, 30))["games"] == 2 and team_stats.before(rows, "DET", date(2026, 9, 1)) == {"games": 0}


def test_the_text_shows_lines_stats_injuries_and_props():
    side = lambda team: {"team": team, "record": "5-4", "ats": "3-5-1", "point_diff_per_game": 0.4, "last_results": ["W +3"],
                         "season": {"games": 9, "third_down_pct": 0.415, "third_down_pct_allowed": 0.38, "turnover_margin": -11, "giveaways": 19,
                                    "takeaways": 8, "net_punt_avg": 41.3},
                         "last_3": {"games": 3, "third_down_pct": 0.4}, "league_rank": {"turnover_margin": 31},
                         "injuries": [{"player": "A.J. Brown", "pos": "WR", "status": "Questionable", "injury": "Knee",
                                       "usage": {"snap_share": 0.8, "targets": 7.0, "carries": 0.0}}]}
    line = {"seen": "Sun 08:30 ET", "home_spread": -2.5, "home_price": -115, "away_price": -105, "total": 50.5, "over_price": -120, "under_price": 100}
    packet = {"season": 2026, "week": 10, "slate": "2026-11-15", "report_time_utc": "2026-11-15T16:00:00+00:00", "injury_snapshots": True,
              "games": [{"matchup": "NE @ DET", "kickoff_et": "Sun 15 Nov 01:00 PM", "venue": {"roof": "dome", "surface": None, "weather_at_kickoff": None},
                         "lines": {"fanduel": {"earliest": {**line, "home_spread": -1.5, "total": 49.5}, "latest": line, "snapshots": 4}},
                         "teams": {"away": side("NE"), "home": side("DET")},
                         "props_fanduel": [{"player": "A.Brown", "team": "NE", "pos": "WR", "stat": "receiving_yards", "line": 50.5, "side_priced": "over",
                                            "price": -113, "book_no_vig_chance": 0.5, "model_projection": 55.2}]}]}
    text = packets.to_text(packet)
    assert "Report time: 11:00 AM ET" in text and "| fanduel | -2.5 (-115) was -1.5 | 50.5 (-120/100) was 49.5 |" in text
    assert "| 3rd-down %: offense / defense | 0.415 / 0.38 |" in text and "| Turnover margin (giveaways/takeaways) | -11 (19/8) |" in text
    assert "| Rank: turnover margin (1 = best of 32) | 31 | 31 |" in text and "A.J. Brown (WR) Questionable, Knee [snap share 0.8, 7 targets" in text
    assert "| A.Brown (NE WR) | receiving_yards | 50.5 | over -113 | 0.5 | 55.2 |" in text and "Venue: dome\n" in text
