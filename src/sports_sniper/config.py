"""Settings, read from the environment or a local .env file. Secrets are never printed or logged."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SNIPER_", env_file=str(ROOT / ".env"), extra="ignore")

    # --- data sources ---
    odds_api_key: SecretStr | None = Field(default=None, validation_alias=AliasChoices("SNIPER_ODDS_API_KEY", "ODDS_API_KEY"))
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"
    database_url: str = f"sqlite:///{(ROOT / 'data' / 'sniper' / 'sniper.db').as_posix()}"
    saved_quotes_dsn: str | None = None      # the existing pipeline's Postgres; None uses that pipeline's own setting

    # --- what to scan ---
    sports: list[str] = ["nfl", "nba"]
    bookmakers: list[str] = ["draftkings", "fanduel", "betmgm", "espnbet", "fanatics", "pinnacle"]  # 10 or fewer cost one region
    sharp_books: list[str] = ["pinnacle"]    # used as references, never evaluated as places to bet
    bet_books: list[str] = ["fanduel", "draftkings", "espnbet"]   # books to flag bets at; the rest are references only. Empty = every book
    markets: dict[str, list[str]] = {}       # per sport; empty uses markets.DEFAULT_MARKETS
    lookahead_hours: float = 24.0            # live scans: games starting within this many hours
    saved_lookback_hours: float = 36.0       # saved snapshots: how far back to read
    results_dsn: str | None = None           # final statistics for settlement; None uses the existing pipeline's

    # --- odds API budget ---
    credit_floor: int = 2000                 # never spend below this many remaining credits
    max_credits_per_scan: int = 60
    request_timeout_seconds: float = 15.0
    max_retries: int = 3
    cache_seconds: float = 60.0

    # --- pricing ---
    devig_method: str = "power"
    reference_weights: dict[str, float] = {"pinnacle": 3.0}   # other books weigh 1
    min_reference_books: int = 1
    uncertainty_floor: float = 0.015         # smallest standard error ever assumed for a consensus probability
    soft_only_multiplier: float = 2.0        # the floor is this much larger when no sharp book is a reference
    outlier_min_premium: float = 0.10        # one-sided longshots: flag a price this far above the other books' middle
    confidence_z: float = 1.0                # the conservative estimate is this many standard errors below
    max_quote_age_seconds: int = 300
    max_book_skew_seconds: int = 180         # a reference must be this close in time to the price it judges
    max_plausible_ev: float = 0.15           # a bigger edge is treated as a stale or mismatched price, not a gift

    # --- what to show ---
    min_conservative_ev: float = 0.0
    min_ev: float = 0.04                     # smallest estimated edge shown and paper-traded (smaller ones have not held up)
    longshot_min_american: int = 200
    max_snapshot_age_minutes: float = 20.0   # older prices are shown as history, not as available
    line_move_threshold: float = 0.02        # change in break-even probability worth flagging
    paper_stake: float = 1.0
    link_state: str = "co"                   # two-letter state for sportsbook links that need one (BetMGM)

    # --- paper trading ---
    model_view: bool = True             # add the prediction model's calibrated chance to flagged bets (display and records only)
    auto_paper: bool = True                  # enter every flagged bet that clears the bar as a paper bet
    auto_paper_evidence: list[str] = ["strong", "moderate"]
    replay_after_minutes: float = 60.0       # a bet entered this long after its price was seen is a replay
    closing_window_minutes: float = 22.0     # the close is the last look this close to the start

    def markets_for(self, sport: str) -> list[str]:
        from sports_sniper.markets import DEFAULT_MARKETS
        return list(self.markets.get(sport) or DEFAULT_MARKETS.get(sport, []))

    def api_key(self) -> str | None:
        """The key from the environment or .env, else the one the existing pipelines saved on this PC."""
        if self.odds_api_key is not None:
            return self.odds_api_key.get_secret_value()
        try:
            from supernovabets_config import _saved_windows_env
            return _saved_windows_env("ODDS_API_KEY") or None
        except Exception:
            return None


@lru_cache
def get_settings() -> Settings:
    return Settings()
