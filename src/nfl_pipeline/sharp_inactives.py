"""Game-day player statuses for the sharp strategy's shadow filter (pinned by the v8 registration).

The frozen forecast the filter reads is made near T-120 (run_pregame); inactives come out about T-90,
when books are slowest to move. Each scan reads ESPN's NFL injury feed (free, minutes fresh; cached for
CACHE_SECONDS) and records, beside every decision:

- the bet player's own ESPN status (an Out player's prop is void at the book, so no money is at risk,
  but it marks a stale price);
- teammates now Out (or IR / suspended) whose forecast did not already have them out, and the bet
  player's projection with their projected volume moved to the remaining teammates in proportion to
  their own projections (same stat), so the filter can see whether the news flips the model's side.

Recorded only: nothing here stakes, skips or sizes a bet (sharp_strategy.SHADOW_FILTER).
"""
from __future__ import annotations

import time
from typing import Any

import requests

from nfl_pipeline.markets import normalize_name

URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
CACHE_SECONDS = 240
TEAM_FIX = {"LAR": "LA", "WSH": "WAS"}
GONE = ("out", "injured reserve", "suspension")
_cache: dict[str, Any] = {}


def statuses(session=requests, now: float | None = None) -> dict[tuple[str, str], str]:
    """{(team, player_name_norm): status lowercased}; {} when the feed is unavailable."""
    now = time.time() if now is None else now
    if _cache.get("at") and now - _cache["at"] < CACHE_SECONDS:
        return _cache["data"]
    data: dict[tuple[str, str], str] = {}
    try:
        payload = session.get(URL, timeout=20).json()
        for team in payload.get("injuries", []):
            for item in team.get("injuries", []):
                athlete = item.get("athlete") or {}
                abbr = (athlete.get("team") or {}).get("abbreviation") or ""
                name = normalize_name(athlete.get("displayName") or "")
                if abbr and name:
                    data[(TEAM_FIX.get(abbr, abbr), name)] = str(item.get("status") or "").strip().lower()
    except Exception:
        data = {}
    _cache.update(at=now, data=data)
    return data


TEAM_SQL = """SELECT COALESCE(offer_player_name_norm, ''), player_name, team_abbr, projection, injury_report_status
    FROM bets.nfl_player_prop_predictions WHERE game_id=%s AND stat=%s AND is_current"""


def _forecast_name(offer_norm: str, player_name: str) -> str:
    return offer_norm or normalize_name(player_name or "")


def teammate_shift(rows: list[tuple], player_norm: str, status: dict[tuple[str, str], str]) -> dict[str, Any]:
    """The bet player's projection with newly-out teammates' projected volume redistributed.

    rows: (offer_player_name_norm, player_name, team_abbr, projection, injury_report_status) for every
    current forecast of this game and stat.
    """
    table = [(_forecast_name(r[0], r[1]), r[2], float(r[3]) if r[3] is not None else None, str(r[4] or "").lower())
             for r in rows]
    me = next((t for t in table if t[0] == player_norm), None)
    if me is None or me[2] is None:
        return dict(teammates_out=[], adjusted_projection=None)
    team = [t for t in table if t[1] == me[1] and t[2] is not None]
    newly_out = [t for t in team if t[0] != player_norm and status.get((t[1], t[0])) in GONE
                 and t[3] not in ("out", "ir", "inactive", "doubtful")]
    out_names = {t[0] for t in newly_out}
    remaining = [t for t in team if t[0] not in out_names and status.get((t[1], t[0])) not in GONE]
    pool = sum(max(t[2], 0.0) for t in remaining)
    moved = sum(max(t[2], 0.0) for t in newly_out)
    adjusted = me[2] + (moved * max(me[2], 0.0) / pool if pool > 0 else 0.0)
    return dict(teammates_out=sorted(out_names), adjusted_projection=round(adjusted, 2))


def context(cur, edge: dict, game: dict, session=requests) -> dict[str, Any]:
    """ESPN status of the bet player and the teammate shift; empty values when nothing is known."""
    empty = dict(espn_status=None, teammates_out=[], adjusted_projection=None)
    if not edge.get("player_norm"):
        return empty
    status = statuses(session)
    if not status:
        return empty
    cur.execute(TEAM_SQL, (game["game_id"], edge["stat"]))
    rows = cur.fetchall()
    team = next((r[2] for r in rows if _forecast_name(r[0], r[1]) == edge["player_norm"]), None)
    own = status.get((team, edge["player_norm"])) if team else None
    return dict(espn_status=own, **teammate_shift(rows, edge["player_norm"], status))
