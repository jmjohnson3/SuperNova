"""Lock approved NFL predictions into the betting ledger."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.schema import ensure_schema
from nfl_pipeline.betting_preferences import EXECUTION_BOOK, FANDUEL_LINK_PATTERN, execution_link
from nfl_pipeline.offer_selection import CONTRACT as EXECUTION_CONTRACT, MAX_QUOTE_AGE_MINUTES
from nfl_pipeline.game_scope import game_ids

log = logging.getLogger("nfl_pipeline.lock_ledger")


@dataclass(frozen=True)
class LedgerConfig:
    pg_dsn: str = PG_DSN
    game_date: date | None = None
    tier: str = "micro"
    max_rows: int = 5
    stake: float = 1.0


APPROVED_TIERS = {"paper", "micro", "micro_projection", "starter", "bankroll"}


def _insert_rows(conn, rows: list[tuple]) -> int:
    inserted = 0
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(hashtext('nfl_ledger_lock'))")
        for row in rows:
            kind, prediction_id, day, tier = row[:4]
            if tier != 'paper' and (row[5] != EXECUTION_BOOK or not execution_link(row[11])):
                continue
            table = {"prop": "nfl_player_prop_predictions", "game": "nfl_game_predictions"}[kind]
            same_player = "old.player_id IS NOT DISTINCT FROM new.player_id AND old.stat=new.stat" if kind == "prop" else "old.market=new.market"
            tier_filter = 'l.tier=%s' if tier == 'paper' else "l.tier<>'paper'"
            exact_offer = 'AND old.book=new.book AND old.side=new.side AND old.line=new.line' if tier == 'paper' else ''
            if kind == 'prop' and tier != 'paper':
                same_player = 'old.player_id IS NOT DISTINCT FROM new.player_id'
            params = (prediction_id, kind, day) + ((tier,) if tier == 'paper' else ())
            cur.execute(f"""
                SELECT EXISTS (
                  SELECT 1 FROM bets.nfl_bet_ledger l
                  JOIN bets.{table} old ON old.id=l.prediction_id
                  JOIN bets.{table} new ON new.id=%s
                  WHERE l.source_kind=%s AND l.game_date_et=%s AND {tier_filter}
                    AND old.game_id=new.game_id AND {same_player}
                    {exact_offer}
                )
            """, params)
            if cur.fetchone()[0]:
                continue
            if tier != "paper":
                cur.execute("SELECT count(*) FROM bets.nfl_bet_ledger WHERE game_date_et=%s AND tier=%s", (day, tier))
                if cur.fetchone()[0] >= 5:
                    continue
            cur.execute("""
                INSERT INTO bets.nfl_bet_ledger (
                    source_kind,prediction_id,game_date_et,tier,stake,book,
                    market,stat,side,line,price,link,model_version,prediction_key)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT DO NOTHING
            """, row)
            inserted += max(0, cur.rowcount)
    conn.commit()
    return inserted


def sync_stale_ledger_rows(conn, cfg: LedgerConfig) -> int:
    """Compatibility entry point: display changes never void locked history."""
    return 0


def lock_game_predictions(conn, cfg: LedgerConfig) -> int:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                id, game_date_et, tier, book, market, NULL::text AS stat,
                side, line, price, link, model_version, prediction_key,
                ev
            FROM bets.nfl_game_predictions
            WHERE tier = %(tier)s
              AND is_current AND integrity_version = 'nfl-asof-v2'
              AND EXISTS (SELECT 1 FROM raw.nfl_games g WHERE g.game_id=bets.nfl_game_predictions.game_id AND g.start_ts_utc>NOW())
              AND (%(game_date)s IS NULL OR game_date_et = %(game_date)s)
              AND (%(game_ids)s IS NULL OR game_id=ANY(%(game_ids)s))
              AND price IS NOT NULL
              AND line IS NOT NULL
              AND (%(tier)s = 'paper' OR (book = %(execution_book)s AND LOWER(link) ~ %(execution_link_pattern)s))
              AND (%(tier)s = 'paper' OR (forecast_payload->>'execution_contract' = %(execution_contract)s
                AND (forecast_payload->>'quote_fetched_at_utc')::timestamptz
                    BETWEEN NOW() - %(max_quote_age)s * interval '1 minute' AND created_at_utc))
            ORDER BY COALESCE(ev, -999) DESC, id
            LIMIT %(max_rows)s
            """,
            {"tier": cfg.tier, "game_date": cfg.game_date, "max_rows": cfg.max_rows,
             "execution_book": EXECUTION_BOOK, "execution_link_pattern": FANDUEL_LINK_PATTERN,
             "execution_contract": EXECUTION_CONTRACT, "max_quote_age": MAX_QUOTE_AGE_MINUTES, "game_ids": game_ids()},
        )
        rows = cur.fetchall()
    tuples = [
        (
            "game", row["id"], row["game_date_et"], row["tier"], cfg.stake,
            row["book"], row["market"], row["stat"], row["side"], row["line"],
            row["price"], row["link"], row["model_version"], row["prediction_key"],
        )
        for row in rows
    ]
    return _insert_rows(conn, tuples)


def lock_prop_predictions(conn, cfg: LedgerConfig) -> int:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                id, game_date_et, tier, book, NULL::text AS market, stat,
                side, line, price, link, model_version, prediction_key,
                ev
            FROM bets.nfl_player_prop_predictions
            WHERE tier = %(tier)s
              AND is_current AND integrity_version = 'nfl-asof-v2'
              AND (%(game_date)s IS NULL OR game_date_et = %(game_date)s)
              AND (%(game_ids)s IS NULL OR game_id=ANY(%(game_ids)s))
              AND price IS NOT NULL
              AND line IS NOT NULL
              AND side IN ('over', 'under')
              AND (%(tier)s = 'paper' OR (book = %(execution_book)s AND LOWER(link) ~ %(execution_link_pattern)s))
              AND (%(tier)s = 'paper' OR (forecast_payload->>'execution_contract' = %(execution_contract)s
                AND (forecast_payload->>'quote_fetched_at_utc')::timestamptz
                    BETWEEN NOW() - %(max_quote_age)s * interval '1 minute' AND created_at_utc))
              AND NOT EXISTS (
                  SELECT 1
                  FROM raw.nfl_games g
                  WHERE g.game_id = bets.nfl_player_prop_predictions.game_id
                    AND g.start_ts_utc <= NOW()
              )
            ORDER BY COALESCE(ev, -999) DESC, id
            LIMIT %(max_rows)s
            """,
            {"tier": cfg.tier, "game_date": cfg.game_date, "max_rows": cfg.max_rows,
             "execution_book": EXECUTION_BOOK, "execution_link_pattern": FANDUEL_LINK_PATTERN,
             "execution_contract": EXECUTION_CONTRACT, "max_quote_age": MAX_QUOTE_AGE_MINUTES, "game_ids": game_ids()},
        )
        rows = cur.fetchall()
    tuples = [
        (
            "prop", row["id"], row["game_date_et"], row["tier"], cfg.stake,
            row["book"], row["market"], row["stat"], row["side"], row["line"],
            row["price"], row["link"], row["model_version"], row["prediction_key"],
        )
        for row in rows
    ]
    return _insert_rows(conn, tuples)


def refresh_ledger_results(conn, cfg: LedgerConfig) -> int:
    updated = 0
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE bets.nfl_bet_ledger l
            SET
                result = g.result,
                profit = CASE WHEN g.result='win' THEN l.stake * CASE WHEN l.price>0 THEN l.price/100.0 ELSE 100.0/NULLIF(ABS(l.price),0) END WHEN g.result='loss' THEN -l.stake WHEN g.result='push' THEN 0 ELSE NULL END,
                clv_status = c.clv_status,
                updated_at_utc = NOW()
            FROM bets.nfl_game_prediction_results g
            LEFT JOIN bets.nfl_prediction_clv c
              ON c.source_kind = 'game'
             AND c.prediction_id = g.prediction_id
            WHERE l.source_kind = 'game'
              AND l.prediction_id = g.prediction_id
              AND (%(game_date)s IS NULL OR l.game_date_et = %(game_date)s)
              AND COALESCE(l.result, '') NOT LIKE 'voided_%%'
              AND (l.result, l.profit, l.clv_status) IS DISTINCT FROM (
                  g.result,
                  CASE WHEN g.result='win' THEN l.stake * CASE WHEN l.price>0 THEN l.price/100.0 ELSE 100.0/NULLIF(ABS(l.price),0) END WHEN g.result='loss' THEN -l.stake WHEN g.result='push' THEN 0 ELSE NULL END,
                  c.clv_status)
            """,
            {"game_date": cfg.game_date},
        )
        updated += max(0, cur.rowcount)
        cur.execute(
            """
            UPDATE bets.nfl_bet_ledger l
            SET
                result = p.result,
                profit = CASE WHEN p.result='win' THEN l.stake * CASE WHEN l.price>0 THEN l.price/100.0 ELSE 100.0/NULLIF(ABS(l.price),0) END WHEN p.result='loss' THEN -l.stake WHEN p.result='push' THEN 0 ELSE NULL END,
                clv_status = c.clv_status,
                updated_at_utc = NOW()
            FROM bets.nfl_player_prop_prediction_results p
            LEFT JOIN bets.nfl_prediction_clv c
              ON c.source_kind = 'prop'
             AND c.prediction_id = p.prediction_id
            WHERE l.source_kind = 'prop'
              AND l.prediction_id = p.prediction_id
              AND (%(game_date)s IS NULL OR l.game_date_et = %(game_date)s)
              AND COALESCE(l.result, '') NOT LIKE 'voided_%%'
              AND (l.result, l.profit, l.clv_status) IS DISTINCT FROM (
                  p.result,
                  CASE WHEN p.result='win' THEN l.stake * CASE WHEN l.price>0 THEN l.price/100.0 ELSE 100.0/NULLIF(ABS(l.price),0) END WHEN p.result='loss' THEN -l.stake WHEN p.result='push' THEN 0 ELSE NULL END,
                  c.clv_status)
            """,
            {"game_date": cfg.game_date},
        )
        updated += max(0, cur.rowcount)
    conn.commit()
    return updated


def lock_ledger(cfg: LedgerConfig) -> dict[str, Any]:
    if cfg.tier not in APPROVED_TIERS:
        raise ValueError(f"Tier {cfg.tier!r} is not an approved NFL ledger tier")
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_schema(conn)
        stale_voided = 0  # A later display card cannot cancel a locked decision.
        game_locked = lock_game_predictions(conn, cfg)
        prop_locked = lock_prop_predictions(conn, cfg)
        refreshed = refresh_ledger_results(conn, cfg)
    return {
        "status": "ok",
        "tier": cfg.tier,
        "game_locked": game_locked,
        "prop_locked": prop_locked,
        "stale_ledger_rows_voided": stale_voided,
        "ledger_results_refreshed": refreshed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock approved NFL predictions into the ledger")
    parser.add_argument("--date", default=None, help="ET date YYYY-MM-DD")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--tier", default="micro", choices=sorted(APPROVED_TIERS))
    parser.add_argument("--max-rows", type=int, default=5)
    parser.add_argument("--stake", type=float, default=1.0)
    parser.add_argument("--refresh-only", action="store_true", help="Refresh existing results; never insert bets or run schema DDL")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = LedgerConfig(
        pg_dsn=args.pg_dsn,
        game_date=date.fromisoformat(args.date) if args.date else None,
        tier=args.tier,
        max_rows=args.max_rows,
        stake=args.stake,
    )
    if args.refresh_only:
        with psycopg2.connect(cfg.pg_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout='90s'")
                cur.execute("SET lock_timeout='5s'")
            result = dict(status='ok', ledger_results_refreshed=refresh_ledger_results(conn, cfg), new_bets_locked=0)
    else:
        result = lock_ledger(cfg)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
