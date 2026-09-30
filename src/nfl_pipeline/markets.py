"""Shared NFL player prop market/stat definitions."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class NflStatSpec:
    stat: str
    label: str
    market_keys: tuple[str, ...]
    positions: tuple[str, ...]
    count_like: bool = False
    min_projection: float = 0.0


STAT_SPECS: tuple[NflStatSpec, ...] = (
    NflStatSpec(
        stat="passing_yards",
        label="QB Passing Yards",
        market_keys=("player_pass_yds",),
        positions=("QB",),
        min_projection=60.0,
    ),
    NflStatSpec(
        stat="rushing_yards",
        label="Rushing Yards",
        market_keys=("player_rush_yds",),
        positions=("QB", "RB"),
        min_projection=5.0,
    ),
    NflStatSpec(
        stat="passing_tds",
        label="QB Passing TDs",
        market_keys=("player_pass_tds",),
        positions=("QB",),
        count_like=True,
    ),
    NflStatSpec(
        stat="rushing_tds",
        label="Rushing TDs",
        market_keys=("player_rush_tds",),
        positions=("RB",),
        count_like=True,
    ),
    NflStatSpec(
        stat="receiving_yards",
        label="Receiving Yards",
        market_keys=("player_reception_yds",),
        positions=("RB", "WR", "TE"),
        min_projection=5.0,
    ),
    NflStatSpec(
        stat="receiving_tds",
        label="Receiving TDs",
        market_keys=("player_reception_tds",),
        positions=("WR", "TE"),
        count_like=True,
    ),
)

LEGACY_STAT_BY_MARKET: dict[str, str] = {
    "player_passing_yards": "passing_yards",
    "player_rushing_yards": "rushing_yards",
    "player_passing_touchdowns": "passing_tds",
    "player_rushing_touchdowns": "rushing_tds",
    "player_receiving_yards": "receiving_yards",
    "player_receiving_touchdowns": "receiving_tds",
}

STAT_BY_MARKET: dict[str, str] = {
    market_key: spec.stat
    for spec in STAT_SPECS
    for market_key in spec.market_keys
}
STAT_BY_MARKET.update(LEGACY_STAT_BY_MARKET)

SPEC_BY_STAT: dict[str, NflStatSpec] = {spec.stat: spec for spec in STAT_SPECS}

ODDS_API_MARKETS: str = ",".join(
    dict.fromkeys(market for spec in STAT_SPECS for market in spec.market_keys)
)
ODDS_API_GAME_MARKETS: str = "spreads,totals"

TEAM_NAME_TO_ABBR: dict[str, str] = {
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "dallas cowboys": "DAL",
    "denver broncos": "DEN",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "los angeles chargers": "LAC",
    "los angeles rams": "LA",
    "miami dolphins": "MIA",
    "minnesota vikings": "MIN",
    "new england patriots": "NE",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "new york jets": "NYJ",
    "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN",
    "washington commanders": "WAS",
}


def normalize_name(name: str | None) -> str:
    text = unicodedata.normalize("NFKD", name or "")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def normalize_team(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    upper = text.upper()
    if 2 <= len(upper) <= 4 and upper.replace("WSH", "WAS").isalpha():
        if upper == "WSH":
            return "WAS"
        if upper == "JAC":
            return "JAX"
        if upper == "LAR":
            return "LA"
        return upper
    return TEAM_NAME_TO_ABBR.get(normalize_name(text), upper)


def role_for_position(position: str | None) -> str:
    pos = str(position or "").upper()
    if pos == "QB":
        return "QB"
    if pos == "RB":
        return "RB"
    if pos in {"WR", "TE"}:
        return "WR/TE"
    return pos or "UNKNOWN"


def stats_for_position(position: str | None) -> tuple[NflStatSpec, ...]:
    pos = str(position or "").upper()
    return tuple(spec for spec in STAT_SPECS if pos in spec.positions)


def stat_allowed_for_position(stat: str, position: str | None) -> bool:
    spec = SPEC_BY_STAT.get(stat)
    return bool(spec and str(position or "").upper() in spec.positions)


def all_market_keys() -> tuple[str, ...]:
    return tuple(market for spec in STAT_SPECS for market in spec.market_keys)


def canonical_market_map(markets: Iterable[str]) -> dict[str, str]:
    return {str(market): STAT_BY_MARKET[str(market)] for market in markets if str(market) in STAT_BY_MARKET}
