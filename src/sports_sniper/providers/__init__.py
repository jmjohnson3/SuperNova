"""Where prices come from. Every provider yields Snapshots: one Odds API event as it looked at one moment."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Snapshot:
    sport: str            # nfl | nba | mlb
    observed_at: datetime
    event: dict           # an event in The Odds API's shape, with its bookmakers
