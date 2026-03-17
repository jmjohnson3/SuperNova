# src/nba_pipeline/refresh_prop_links.py
"""Re-crawl today's prop lines and re-parse into odds.nba_player_prop_lines.

Run mid-pipeline (after training, before alt-line scan) so FanDuel betslip
deeplinks that weren't available at the early-morning crawl are captured.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg2

from .crawler_oddsapi import OddsCrawlerConfig, _fetch_prop_lines_for_day
from .parse_oddsapi import parse_prop_odds, parse_prop_odds_alt

log = logging.getLogger("nba_pipeline.refresh_prop_links")
_ET = ZoneInfo("America/New_York")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = OddsCrawlerConfig()
    today_et = datetime.now(_ET).date()

    log.info("Refreshing prop lines for %s (force re-crawl)…", today_et)

    with psycopg2.connect(cfg.pg_dsn) as conn:
        credits = _fetch_prop_lines_for_day(cfg, conn, today_et, force=True)
        if credits is not None:
            log.info("Prop lines re-crawl complete. credits_remaining=%d", credits)
        else:
            log.info("Prop lines re-crawl complete (credits unknown).")

    # Re-parse from yesterday so today's upserted payloads are included
    parse_prop_odds(since_date=today_et - timedelta(days=1))
    parse_prop_odds_alt(since_date=today_et - timedelta(days=1))
    log.info("Prop lines refresh done.")


if __name__ == "__main__":
    main()
