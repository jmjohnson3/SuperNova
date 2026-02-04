# src/nba_pipeline/parse_oddsapi.py
import json
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Optional

import psycopg2
import psycopg2.extras

log = logging.getLogger("nba_pipeline.parse_oddsapi")

UPSERT_SQL = """
INSERT INTO odds.nba_game_lines (
  provider, as_of_date, fetched_at_utc,
  event_id, commence_time_utc,
  bookmaker_key, bookmaker_title,
  home_team, away_team,
  spread_home_points, spread_home_price,
  spread_away_points, spread_away_price,
  total_points, total_over_price, total_under_price,
  updated_at_utc
)
VALUES %s
ON CONFLICT (provider, fetched_at_utc, event_id, bookmaker_key)
DO UPDATE SET
  commence_time_utc      = EXCLUDED.commence_time_utc,
  bookmaker_title        = EXCLUDED.bookmaker_title,
  home_team              = EXCLUDED.home_team,
  away_team              = EXCLUDED.away_team,
  spread_home_points     = EXCLUDED.spread_home_points,
  spread_home_price      = EXCLUDED.spread_home_price,
  spread_away_points     = EXCLUDED.spread_away_points,
  spread_away_price      = EXCLUDED.spread_away_price,
  total_points           = EXCLUDED.total_points,
  total_over_price       = EXCLUDED.total_over_price,
  total_under_price      = EXCLUDED.total_under_price,
  updated_at_utc         = EXCLUDED.updated_at_utc
;
"""

SQL_LOAD = """
SELECT as_of_date, fetched_at_utc, payload
FROM raw.api_responses
WHERE provider = 'oddsapi'
  AND endpoint = 'nba_odds'
  AND (%(as_of_date)s IS NULL OR as_of_date = %(as_of_date)s)
ORDER BY fetched_at_utc;
"""


@dataclass(frozen=True)
class ParseConfig:
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"
    as_of_date: Optional[date] = None  # set to parse a specific ET date bucket


def _get_market(book: dict, key: str) -> Optional[dict]:
    for m in book.get("markets", []) or []:
        if m.get("key") == key:
            return m
    return None


def _find_outcome(market: dict, name: str) -> Optional[dict]:
    for o in market.get("outcomes", []) or []:
        if o.get("name") == name:
            return o
    return None


def _to_num(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def iter_rows(as_of_date: date, fetched_at_utc, events: list[dict]) -> Iterable[tuple]:
    provider = "oddsapi"
    for ev in events:
        event_id = ev.get("id")
        commence_time_utc = ev.get("commence_time")
        home_team = ev.get("home_team")
        away_team = ev.get("away_team")

        for book in ev.get("bookmakers", []) or []:
            bookmaker_key = book.get("key")
            bookmaker_title = book.get("title")

            spreads = _get_market(book, "spreads")
            totals = _get_market(book, "totals")

            spread_home_points = spread_home_price = None
            spread_away_points = spread_away_price = None
            total_points = total_over_price = total_under_price = None

            if spreads:
                oh = _find_outcome(spreads, home_team)
                oa = _find_outcome(spreads, away_team)
                if oh:
                    spread_home_points = _to_num(oh.get("point"))
                    spread_home_price = _to_int(oh.get("price"))
                if oa:
                    spread_away_points = _to_num(oa.get("point"))
                    spread_away_price = _to_int(oa.get("price"))

            if totals:
                o_over = _find_outcome(totals, "Over")
                o_under = _find_outcome(totals, "Under")
                # totals market point should match for both outcomes
                if o_over:
                    total_points = _to_num(o_over.get("point"))
                    total_over_price = _to_int(o_over.get("price"))
                if o_under:
                    if total_points is None:
                        total_points = _to_num(o_under.get("point"))
                    total_under_price = _to_int(o_under.get("price"))

            yield (
                provider,
                as_of_date,
                fetched_at_utc,
                event_id,
                commence_time_utc,
                bookmaker_key,
                bookmaker_title,
                home_team,
                away_team,
                spread_home_points,
                spread_home_price,
                spread_away_points,
                spread_away_price,
                total_points,
                total_over_price,
                total_under_price,
            )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = ParseConfig()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(SQL_LOAD, {"as_of_date": cfg.as_of_date})
            snaps = cur.fetchall()

        if not snaps:
            log.warning("No oddsapi snapshots found (as_of_date=%s).", cfg.as_of_date)
            return

        total_rows = 0
        with conn.cursor() as cur:
            for s in snaps:
                as_of_date = s["as_of_date"]
                fetched_at_utc = s["fetched_at_utc"]
                payload = s["payload"]

                # payload may already be dict/list depending on driver
                if isinstance(payload, str):
                    payload_obj = json.loads(payload)
                else:
                    payload_obj = payload

                if not isinstance(payload_obj, list):
                    log.warning("Unexpected payload type=%s; skipping", type(payload_obj))
                    continue

                rows = list(iter_rows(as_of_date, fetched_at_utc, payload_obj))
                if not rows:
                    continue

                psycopg2.extras.execute_values(
                    cur,
                    UPSERT_SQL,
                    [
                        (*r, fetched_at_utc)  # add updated_at_utc at end
                        for r in rows
                    ],
                    page_size=500,
                )
                total_rows += len(rows)

        conn.commit()
        log.info("Upserted %d rows into odds.nba_game_lines", total_rows)


if __name__ == "__main__":
    main()
