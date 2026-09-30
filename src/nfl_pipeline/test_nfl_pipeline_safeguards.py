from __future__ import annotations

from pathlib import Path

import pandas as pd

from nfl_pipeline.markets import ODDS_API_GAME_MARKETS, ODDS_API_MARKETS, SPEC_BY_STAT, STAT_BY_MARKET, normalize_team
from nfl_pipeline.crawler_oddsapi import (
    OddsCrawlerConfig,
    _canonical_events_from_sgo_payload,
    _manual_game_rows,
    _manual_prop_rows,
    _provider_names,
)
from nfl_pipeline.import_usage_context import _pbp_player_usage_rows
from nfl_pipeline.parse_oddsapi import _match_team_outcome, _outcome_side_and_player, _rows_from_game_odds_payload

ROOT = Path(__file__).resolve().parents[2]


def test_nfl_market_specs_cover_requested_player_props():
    expected_stats = {
        "passing_yards",
        "rushing_yards",
        "passing_tds",
        "rushing_tds",
        "receiving_yards",
        "receiving_tds",
    }
    assert expected_stats <= set(SPEC_BY_STAT)
    assert SPEC_BY_STAT["passing_yards"].positions == ("QB",)
    assert "QB" in SPEC_BY_STAT["rushing_yards"].positions
    assert "RB" in SPEC_BY_STAT["rushing_yards"].positions
    assert SPEC_BY_STAT["passing_tds"].count_like
    assert SPEC_BY_STAT["rushing_tds"].count_like
    assert SPEC_BY_STAT["receiving_tds"].count_like
    assert STAT_BY_MARKET["player_pass_yds"] == "passing_yards"
    assert STAT_BY_MARKET["player_passing_yards"] == "passing_yards"
    assert STAT_BY_MARKET["player_rush_yds"] == "rushing_yards"
    assert STAT_BY_MARKET["player_reception_yds"] == "receiving_yards"
    assert "player_passing_yards" not in ODDS_API_MARKETS
    assert "player_receiving_yards" not in ODDS_API_MARKETS
    assert ODDS_API_GAME_MARKETS == "spreads,totals"


def test_nfl_team_and_odds_outcome_normalization():
    assert normalize_team("Kansas City Chiefs") == "KC"
    assert normalize_team("WSH") == "WAS"
    assert normalize_team("JAC") == "JAX"
    assert normalize_team("Los Angeles Rams") == "LA"
    assert normalize_team("LAR") == "LA"
    assert _match_team_outcome("Denver Broncos", "Denver Broncos", "Las Vegas Raiders") == "home"
    assert _outcome_side_and_player({"name": "Over", "description": "Patrick Mahomes"}) == (
        "over",
        "Patrick Mahomes",
    )
    assert _outcome_side_and_player({"name": "Bijan Robinson", "description": "Under"}) == (
        "under",
        "Bijan Robinson",
    )


def test_nfl_game_odds_parser_extracts_spread_total_rows():
    rows = _rows_from_game_odds_payload(
        snapshot_role="close",
        as_of_date=__import__("datetime").date(2026, 9, 10),
        fetched_at_utc=__import__("datetime").datetime(2026, 9, 10, 14, 0),
        payload=[{
            "id": "evt_1",
            "commence_time": "2026-09-10T20:20:00Z",
            "home_team": "Denver Broncos",
            "away_team": "Las Vegas Raiders",
            "bookmakers": [{
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {"key": "spreads", "outcomes": [
                        {"name": "Denver Broncos", "point": -2.5, "price": -110, "link": "https://fanduel.com/home"},
                        {"name": "Las Vegas Raiders", "point": 2.5, "price": -110, "link": "https://fanduel.com/away"},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "point": 44.5, "price": -105, "link": "https://fanduel.com/over"},
                        {"name": "Under", "point": 44.5, "price": -115, "link": "https://fanduel.com/under"},
                    ]},
                ],
            }],
        }],
    )
    assert len(rows) == 1
    row = rows[0]
    assert row[3] == "close"
    assert row[11] == "LV"
    assert row[12] == -2.5
    assert row[16] == 44.5


def test_nfl_odds_provider_fallback_helpers_normalize_sgo_and_manual_rows():
    assert _provider_names("sgo,the-rundown,the-odds-api,manual") == [
        "sportsgameodds",
        "therundown",
        "oddsapi",
        "manual_csv",
    ]

    sgo_events = _canonical_events_from_sgo_payload({
        "data": [{
            "eventID": "evt_1",
            "startTime": "2026-09-15T00:15:00Z",
            "homeTeamName": "Kansas City Chiefs",
            "awayTeamName": "Denver Broncos",
            "statEntities": [{"statEntityID": "p1", "name": "Patrick Mahomes"}],
            "odds": {
                "passing-yards-p1-game-ou-over": {
                    "byBookmaker": {"fanduel": {"americanOdds": -112, "overUnder": 260.5, "link": "fd-over"}}
                },
                "passing-yards-p1-game-ou-under": {
                    "byBookmaker": {"fanduel": {"americanOdds": -108, "overUnder": 260.5, "link": "fd-under"}}
                },
                "points-total-game-ou-over": {
                    "byBookmaker": {"draftkings": {"americanOdds": -110, "overUnder": 47.5}}
                },
                "points-total-game-ou-under": {
                    "byBookmaker": {"draftkings": {"americanOdds": -110, "overUnder": 47.5}}
                },
            },
        }]
    })
    assert len(sgo_events) == 1
    fanduel = next(book for book in sgo_events[0]["bookmakers"] if book["key"] == "fanduel")
    fd_market = next(market for market in fanduel["markets"] if market["key"] == "player_pass_yds")
    assert {outcome["name"] for outcome in fd_market["outcomes"]} == {"Over", "Under"}
    assert fd_market["outcomes"][0]["description"] == "Patrick Mahomes"
    draftkings = next(book for book in sgo_events[0]["bookmakers"] if book["key"] == "draftkings")
    assert any(market["key"] == "totals" for market in draftkings["markets"])

    cfg = OddsCrawlerConfig(snapshot_role="lock")
    day = __import__("datetime").date(2026, 9, 14)
    prop_rows = _manual_prop_rows([
        {
            "book": "DK",
            "event_id": "2026_01_DEN_KC",
            "commence_time_utc": "2026-09-15T00:15:00Z",
            "home_team": "Kansas City Chiefs",
            "away_team": "Denver Broncos",
            "player_name": "Patrick Mahomes",
            "stat": "passing_yards",
            "line": "260.5",
            "over_price": "-115",
            "under_price": "-105",
        }
    ], cfg, day)
    assert len(prop_rows) == 1
    assert prop_rows[0][0] == "manual_csv"
    assert prop_rows[0][6] == "draftkings"
    assert prop_rows[0][13] == "passing_yards"
    assert prop_rows[0][15] == -115

    game_rows = _manual_game_rows([
        {
            "book": "fanduel",
            "event_id": "2026_01_DEN_KC",
            "home_team": "Kansas City Chiefs",
            "away_team": "Denver Broncos",
            "spread_home_points": "-4.5",
            "spread_home_price": "-110",
            "spread_away_points": "4.5",
            "spread_away_price": "-110",
            "total_points": "47.5",
            "total_over_price": "-105",
            "total_under_price": "-115",
        }
    ], cfg, day)
    assert len(game_rows) == 1
    assert game_rows[0][6] == "fanduel"
    assert game_rows[0][10] == "KC"
    assert game_rows[0][11] == "DEN"


def test_pbp_usage_hydrator_builds_current_target_air_yards_and_red_zone_rows():
    rows = _pbp_player_usage_rows(pd.DataFrame([
        {
            "season": 2026,
            "week": 1,
            "game_id": "2026_01_DEN_LV",
            "posteam": "DEN",
            "defteam": "LV",
            "yardline_100": 12,
            "pass_attempt": 1,
            "rush_attempt": 0,
            "complete_pass": 1,
            "pass_touchdown": 1,
            "rush_touchdown": 0,
            "passer_player_id": "qb1",
            "passer_player_name": "QB One",
            "receiver_player_id": "wr1",
            "receiver_player_name": "WR One",
            "td_player_id": "wr1",
            "air_yards": 12,
            "yards_after_catch": 3,
            "passing_yards": 15,
            "receiving_yards": 15,
            "rushing_yards": 0,
        },
        {
            "season": 2026,
            "week": 1,
            "game_id": "2026_01_DEN_LV",
            "posteam": "DEN",
            "defteam": "LV",
            "yardline_100": 40,
            "pass_attempt": 1,
            "rush_attempt": 0,
            "complete_pass": 0,
            "pass_touchdown": 0,
            "rush_touchdown": 0,
            "passer_player_id": "qb1",
            "passer_player_name": "QB One",
            "receiver_player_id": "wr2",
            "receiver_player_name": "WR Two",
            "td_player_id": None,
            "air_yards": 8,
            "yards_after_catch": 0,
            "passing_yards": 0,
            "receiving_yards": 0,
            "rushing_yards": 0,
        },
        {
            "season": 2026,
            "week": 1,
            "game_id": "2026_01_DEN_LV",
            "posteam": "DEN",
            "defteam": "LV",
            "yardline_100": 4,
            "pass_attempt": 0,
            "rush_attempt": 1,
            "complete_pass": 0,
            "pass_touchdown": 0,
            "rush_touchdown": 1,
            "rusher_player_id": "rb1",
            "rusher_player_name": "RB One",
            "rushing_yards": 4,
        },
    ]))
    by_player = {row[3]: row for row in rows}

    wr1 = by_player["wr1"]
    rb1 = by_player["rb1"]
    qb1 = by_player["qb1"]
    assert wr1[14] == 1.0
    assert wr1[17] == 0.5
    assert wr1[18] == 0.6
    assert wr1[22] == 0.0
    assert wr1[23] == 1.0
    assert wr1[31] == 0.0
    assert wr1[32] == 1.0
    assert rb1[13] == 1.0
    assert rb1[22] == 1.0
    assert rb1[30] == 1.0
    assert qb1[16] == 2.0
    assert qb1[25] == 1.0


# Runtime contracts are tested in test_integrity.py, not by source substrings.
