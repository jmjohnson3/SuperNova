"""Sports and player-prop markets, by their keys at The Odds API. A book only returns the markets it offers,
and a key the API rejects is skipped by the client, so listing a market here never guarantees it exists."""
from __future__ import annotations

SPORT_KEYS = {"nfl": "americanfootball_nfl", "nba": "basketball_nba", "mlb": "baseball_mlb"}
SPORT_BY_KEY = {v: k for k, v in SPORT_KEYS.items()}

MARKET_LABELS = {
    # NFL
    "player_anytime_td": "Anytime touchdown", "player_tds_over": "Touchdowns (2+ and up)",
    "player_pass_tds": "Passing touchdowns", "player_pass_yds": "Passing yards",
    "player_rush_yds": "Rushing yards", "player_reception_yds": "Receiving yards", "player_receptions": "Receptions",
    "player_rush_yds_alternate": "Rushing yards (alternate)", "player_reception_yds_alternate": "Receiving yards (alternate)",
    "player_pass_yds_alternate": "Passing yards (alternate)", "player_pass_tds_alternate": "Passing touchdowns (alternate)",
    # NBA
    "player_points": "Points", "player_rebounds": "Rebounds", "player_assists": "Assists", "player_threes": "Three-pointers",
    "player_points_rebounds_assists": "Points + rebounds + assists",
    "player_points_alternate": "Points (alternate)", "player_rebounds_alternate": "Rebounds (alternate)",
    "player_assists_alternate": "Assists (alternate)", "player_threes_alternate": "Three-pointers (alternate)",
    # MLB
    "batter_home_runs": "Home runs", "batter_total_bases": "Total bases", "batter_hits": "Hits",
    "pitcher_strikeouts": "Pitcher strikeouts", "pitcher_strikeouts_alternate": "Pitcher strikeouts (alternate)",
    "batter_hits_alternate": "Hits (alternate)", "batter_total_bases_alternate": "Total bases (alternate)",
    "batter_home_runs_alternate": "Home runs (alternate)",
}

CATALOG = {
    "nfl": ["player_anytime_td", "player_tds_over", "player_pass_tds", "player_pass_yds", "player_rush_yds",
            "player_reception_yds", "player_receptions", "player_rush_yds_alternate", "player_reception_yds_alternate",
            "player_pass_yds_alternate", "player_pass_tds_alternate"],
    "nba": ["player_points", "player_rebounds", "player_assists", "player_threes", "player_points_rebounds_assists",
            "player_points_alternate", "player_rebounds_alternate", "player_assists_alternate", "player_threes_alternate"],
    "mlb": ["batter_home_runs", "batter_total_bases", "batter_hits", "pitcher_strikeouts", "pitcher_strikeouts_alternate",
            "batter_hits_alternate", "batter_total_bases_alternate", "batter_home_runs_alternate"],
}

# Each market costs one credit per game per scan, so the defaults are short. Add more in SNIPER_MARKETS.
DEFAULT_MARKETS = {
    "nfl": ["player_anytime_td", "player_pass_tds", "player_rush_yds", "player_reception_yds"],
    "nba": ["player_points", "player_rebounds", "player_assists", "player_threes"],
    "mlb": ["batter_home_runs", "batter_total_bases", "pitcher_strikeouts"],
}

BOOK_LABELS = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "espnbet": "theScore Bet",
               "fanatics": "Fanatics", "pinnacle": "Pinnacle", "williamhill_us": "Caesars", "betrivers": "BetRivers"}


def market_label(key: str) -> str:
    return MARKET_LABELS.get(key) or key.replace("player_", "").replace("batter_", "").replace("pitcher_", "").replace("_", " ").capitalize()


def book_label(key: str) -> str:
    return BOOK_LABELS.get(key) or key.replace("_", " ").title()
