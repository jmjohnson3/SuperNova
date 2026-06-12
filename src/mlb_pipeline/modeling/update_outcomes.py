# src/mlb_pipeline/modeling/update_outcomes.py
"""
Fill actual scores and ATS results for completed MLB games in bets.mlb_game_predictions,
and actual stats for bets.mlb_prop_predictions.

Run after games complete (nightly):
    python -m mlb_pipeline.modeling.update_outcomes
"""
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from .bankroll_ledger import grade_bankroll_ledger
from .game_line_clv import game_line_clv, resolve_valid_game_close
from .model_pick_ledger import grade_model_pick_ledger
from .prop_offer_snapshots import (
    ensure_prop_offer_snapshot_schema,
    resolve_valid_prop_close,
)

log = logging.getLogger("mlb_pipeline.modeling.update_outcomes")

_ET = ZoneInfo("America/New_York")
PG_DSN = "postgresql://josh:password@localhost:5432/nba"

def _american_to_prob(price: float) -> float:
    """Convert American odds price to raw implied probability (no vig removal)."""
    if price >= 100:
        return 100.0 / (price + 100.0)
    else:
        return abs(price) / (abs(price) + 100.0)


def _price_clv(entry_price: float, closing_price: float) -> float:
    """Price CLV in probability percentage points.

    Positive = we got a better price than the closing line (beat the close).
    entry_price and closing_price should be for the SAME side we bet.
    """
    return round((_american_to_prob(closing_price) - _american_to_prob(entry_price)) * 100, 2)


# Stat column mapping: prop stat name → mlb_player_gamelogs column
_STAT_COL = {
    "pitcher_strikeouts": "strikeouts_pitcher",
    "batter_hits":        "hits",
    "batter_home_runs":   "home_runs",
    "batter_total_bases": "total_bases",
    "batter_walks":       "walks_batter",
}


def _game_close_input(row: dict, market: str) -> dict:
    out = dict(row)
    if market == "run_line":
        side = row.get("run_line_bet_side")
        home_line = _clean_float(row.get("market_run_line"))
        out.update({
            "market": "run_line",
            "side": side,
            "market_line": home_line,
            "bet_line": home_line if side == "home" else (-home_line if home_line is not None else None),
            "market_price": row.get("market_rl_price"),
        })
    else:
        out.update({
            "market": "total",
            "side": row.get("total_bet_side"),
            "market_line": row.get("market_total"),
            "bet_line": row.get("market_total"),
            "market_price": row.get("market_total_price"),
        })
    return out


def _blank_game_close() -> dict:
    """Return the CLV shape for a market that was not selected as a bet."""
    return {
        "valid": None,
        "status": None,
        "unknown_reason": None,
        "closing_line": None,
        "closing_price": None,
        "entry_line": None,
        "entry_price": None,
    }


def _resolve_game_close_for_bet(conn, row: dict, market: str) -> dict:
    side_key = "run_line_bet_side" if market == "run_line" else "total_bet_side"
    if not row.get(side_key):
        return _blank_game_close()
    return resolve_valid_game_close(conn, _game_close_input(row, market))


def update_game_outcomes(conn) -> int:
    """Fill actuals and ATS results for completed MLB games.

    For each row in bets.mlb_game_predictions where actual_home_score IS NULL
    and the game has a final score in raw.mlb_games, compute:
      - actual_home_score, actual_away_score, actual_run_diff, actual_total
      - run_line_covered  (did the bet side win vs the run line?)
      - total_covered     (did the bet side win vs the total?)
      - closing_run_line, closing_total  (latest fetched line from odds.mlb_game_lines)
      - clv_run_line, clv_total          (Closing Line Value — positive = beat the close)

    Returns number of rows updated.
    """
    today = datetime.now(_ET).date()

    with conn.cursor() as cur:
        cur.execute(
            """
            ALTER TABLE bets.mlb_game_predictions
                ADD COLUMN IF NOT EXISTS clv_rl_valid BOOLEAN,
                ADD COLUMN IF NOT EXISTS clv_rl_status TEXT,
                ADD COLUMN IF NOT EXISTS clv_rl_unknown_reason TEXT,
                ADD COLUMN IF NOT EXISTS clv_total_valid BOOLEAN,
                ADD COLUMN IF NOT EXISTS clv_total_status TEXT,
                ADD COLUMN IF NOT EXISTS clv_total_unknown_reason TEXT;
            """
        )
    conn.commit()

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT game_slug, game_date_et,
                   home_team_abbr, away_team_abbr,
                   market_run_line, market_total,
                   run_line_bet_side, total_bet_side,
                   pred_run_diff,
                   market_rl_price, market_total_price,
                   predicted_at_utc, created_at_utc
            FROM bets.mlb_game_predictions
            WHERE actual_home_score IS NULL
              AND game_date_et <= %s
        """, (today,))
        pending = cur.fetchall()

    if not pending:
        log.info("No pending MLB game predictions to update.")
        return 0

    slugs = [r["game_slug"] for r in pending]

    # Pull final scores
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT game_slug, home_score, away_score
            FROM raw.mlb_games
            WHERE game_slug = ANY(%s)
              AND status = 'final'
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
        """, (slugs,))
        finals = {r["game_slug"]: r for r in cur.fetchall()}

    if not finals:
        log.info("No final MLB games found for %d pending predictions.", len(pending))
        return 0

    updated = 0
    with conn.cursor() as cur:
        for row in pending:
            slug = row["game_slug"]
            if slug not in finals:
                continue

            g = finals[slug]
            home_score = float(g["home_score"])
            away_score = float(g["away_score"])
            actual_run_diff = home_score - away_score
            actual_total    = home_score + away_score

            # ── Run line ATS ────────────────────────────────────────────────
            # market_run_line follows Odds API sign convention:
            #   negative = home is FAVORITE (e.g. -1.5 means home gives 1.5 runs)
            #   positive = home is UNDERDOG (e.g. +1.5 means home gets 1.5 runs)
            # Home covers when: actual_run_diff > -market_run_line
            run_line_covered = None
            if row["market_run_line"] is not None and row["run_line_bet_side"] is not None:
                mkt_rl = float(row["market_run_line"])
                threshold = -mkt_rl
                if abs(actual_run_diff - threshold) > 1e-9:
                    home_covered = actual_run_diff > threshold
                    if row["run_line_bet_side"] == "home":
                        run_line_covered = home_covered
                    else:
                        run_line_covered = not home_covered

            # ── Total ────────────────────────────────────────────────────────
            total_covered = None
            if row["market_total"] is not None and row["total_bet_side"] is not None:
                mkt_t = float(row["market_total"])
                if abs(actual_total - mkt_t) > 1e-9:
                    if row["total_bet_side"] == "over":
                        total_covered = actual_total > mkt_t
                    else:
                        total_covered = actual_total < mkt_t

            # ── CLV ──────────────────────────────────────────────────────────
            # Positive CLV = we beat the closing line (sharp signal).
            # home bet: clv = market_run_line - closing_run_line
            # away bet: clv = closing_run_line - market_run_line
            # over bet: clv = closing_total - market_total
            # under bet: clv = market_total - closing_total
            rl_close = _resolve_game_close_for_bet(conn, dict(row), "run_line")
            total_close = _resolve_game_close_for_bet(conn, dict(row), "total")

            closing_rl = closing_rl_price = clv_rl = clv_rl_price = None
            if rl_close["valid"]:
                closing_rl = _clean_float(rl_close.get("closing_line"))
                closing_rl_price = _clean_float(rl_close.get("closing_price"))
                clv_rl = game_line_clv(
                    "run_line",
                    row.get("run_line_bet_side"),
                    rl_close.get("entry_line"),
                    closing_rl,
                )
                if (
                    rl_close.get("entry_line") == closing_rl
                    and rl_close.get("entry_price") is not None
                    and closing_rl_price is not None
                ):
                    clv_rl_price = _price_clv(rl_close.get("entry_price"), closing_rl_price)

            closing_tot = closing_total_price = clv_tot = clv_total_price = None
            if total_close["valid"]:
                closing_tot = _clean_float(total_close.get("closing_line"))
                closing_total_price = _clean_float(total_close.get("closing_price"))
                clv_tot = game_line_clv(
                    "total",
                    row.get("total_bet_side"),
                    total_close.get("entry_line"),
                    closing_tot,
                )
                if (
                    total_close.get("entry_line") == closing_tot
                    and total_close.get("entry_price") is not None
                    and closing_total_price is not None
                ):
                    clv_total_price = _price_clv(total_close.get("entry_price"), closing_total_price)

            cur.execute("""
                UPDATE bets.mlb_game_predictions
                SET actual_home_score  = %s,
                    actual_away_score  = %s,
                    actual_run_diff    = %s,
                    actual_total       = %s,
                    run_line_covered   = %s,
                    total_covered      = %s,
                    closing_run_line   = %s,
                    closing_total      = %s,
                    clv_run_line       = %s,
                    clv_total          = %s,
                    closing_rl_price   = %s,
                    closing_total_price= %s,
                    clv_rl_price       = %s,
                    clv_total_price    = %s,
                    clv_rl_valid       = %s,
                    clv_rl_status      = %s,
                    clv_rl_unknown_reason = %s,
                    clv_total_valid    = %s,
                    clv_total_status   = %s,
                    clv_total_unknown_reason = %s,
                    updated_at_utc     = NOW()
                WHERE game_slug = %s
            """, (
                home_score, away_score, actual_run_diff, actual_total,
                run_line_covered, total_covered,
                closing_rl, closing_tot,
                clv_rl, clv_tot,
                closing_rl_price, closing_total_price,
                clv_rl_price, clv_total_price,
                rl_close["valid"], rl_close["status"], rl_close["unknown_reason"],
                total_close["valid"], total_close["status"], total_close["unknown_reason"],
                row["game_slug"],
            ))
            updated += 1

    conn.commit()
    log.info("update_game_outcomes: updated %d rows", updated)
    return updated


def backfill_clv(conn) -> int:
    """Recompute CLV for already-graded rows where clv_run_line IS NULL
    but a closing line is now available (e.g. evening crawl ran after grading).

    Returns number of rows updated.
    """
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE bets.mlb_game_predictions
            SET closing_run_line = NULL,
                clv_run_line = NULL,
                closing_rl_price = NULL,
                clv_rl_price = NULL,
                clv_rl_valid = NULL,
                clv_rl_status = NULL,
                clv_rl_unknown_reason = NULL
            WHERE run_line_bet_side IS NULL
              AND (closing_run_line IS NOT NULL OR clv_run_line IS NOT NULL
                   OR closing_rl_price IS NOT NULL OR clv_rl_price IS NOT NULL
                   OR clv_rl_valid IS NOT NULL OR clv_rl_status IS NOT NULL
                   OR clv_rl_unknown_reason IS NOT NULL)
        """)
        cur.execute("""
            UPDATE bets.mlb_game_predictions
            SET closing_total = NULL,
                clv_total = NULL,
                closing_total_price = NULL,
                clv_total_price = NULL,
                clv_total_valid = NULL,
                clv_total_status = NULL,
                clv_total_unknown_reason = NULL
            WHERE total_bet_side IS NULL
              AND (closing_total IS NOT NULL OR clv_total IS NOT NULL
                   OR closing_total_price IS NOT NULL OR clv_total_price IS NOT NULL
                   OR clv_total_valid IS NOT NULL OR clv_total_status IS NOT NULL
                   OR clv_total_unknown_reason IS NOT NULL)
        """)
    conn.commit()

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT game_slug, game_date_et, home_team_abbr, away_team_abbr,
                   market_run_line, market_total,
                   run_line_bet_side, total_bet_side,
                   market_rl_price, market_total_price,
                   predicted_at_utc, created_at_utc
            FROM bets.mlb_game_predictions
            WHERE actual_home_score IS NOT NULL
              AND (
                    (run_line_bet_side IS NOT NULL AND clv_rl_status IS NULL)
                 OR (total_bet_side IS NOT NULL AND clv_total_status IS NULL)
              )
        """)
        rows = cur.fetchall()

    if not rows:
        log.info("backfill_clv: nothing to update.")
        return 0

    updated = 0
    with conn.cursor() as cur:
        for row in rows:
            rl_close = _resolve_game_close_for_bet(conn, dict(row), "run_line")
            total_close = _resolve_game_close_for_bet(conn, dict(row), "total")

            closing_rl = closing_rl_price = clv_rl = clv_rl_price = None
            if rl_close["valid"]:
                closing_rl = _clean_float(rl_close.get("closing_line"))
                closing_rl_price = _clean_float(rl_close.get("closing_price"))
                clv_rl = game_line_clv("run_line", row.get("run_line_bet_side"), rl_close.get("entry_line"), closing_rl)
                if (
                    rl_close.get("entry_line") == closing_rl
                    and rl_close.get("entry_price") is not None
                    and closing_rl_price is not None
                ):
                    clv_rl_price = _price_clv(rl_close.get("entry_price"), closing_rl_price)

            closing_tot = closing_total_price = clv_tot = clv_total_price = None
            if total_close["valid"]:
                closing_tot = _clean_float(total_close.get("closing_line"))
                closing_total_price = _clean_float(total_close.get("closing_price"))
                clv_tot = game_line_clv("total", row.get("total_bet_side"), total_close.get("entry_line"), closing_tot)
                if (
                    total_close.get("entry_line") == closing_tot
                    and total_close.get("entry_price") is not None
                    and closing_total_price is not None
                ):
                    clv_total_price = _price_clv(total_close.get("entry_price"), closing_total_price)

            cur.execute("""
                UPDATE bets.mlb_game_predictions
                SET closing_run_line  = COALESCE(closing_run_line, %s),
                    closing_total     = COALESCE(closing_total, %s),
                    clv_run_line      = %s,
                    clv_total         = %s,
                    closing_rl_price  = COALESCE(closing_rl_price, %s),
                    closing_total_price = COALESCE(closing_total_price, %s),
                    clv_rl_price      = %s,
                    clv_total_price   = %s,
                    clv_rl_valid      = %s,
                    clv_rl_status     = %s,
                    clv_rl_unknown_reason = %s,
                    clv_total_valid   = %s,
                    clv_total_status  = %s,
                    clv_total_unknown_reason = %s,
                    updated_at_utc    = NOW()
                WHERE game_slug = %s
            """, (closing_rl, closing_tot, clv_rl, clv_tot,
                  closing_rl_price, closing_total_price,
                  clv_rl_price, clv_total_price,
                  rl_close["valid"], rl_close["status"], rl_close["unknown_reason"],
                  total_close["valid"], total_close["status"], total_close["unknown_reason"],
                  row["game_slug"]))
            updated += 1

    conn.commit()
    log.info("backfill_clv: updated %d rows", updated)
    return updated


def update_prop_outcomes(conn) -> int:
    """Fill actual stat values and over_hit flag for completed MLB prop predictions.

    Pulls actuals from raw.mlb_player_gamelogs for games marked 'final'.
    Supported stats: pitcher_strikeouts, batter_hits, batter_home_runs,
                     batter_total_bases, batter_walks.

    Returns number of rows updated.
    """
    today = datetime.now(_ET).date()

    with conn.cursor() as cur:
        cur.execute(
            """
            ALTER TABLE bets.mlb_prop_predictions
                ADD COLUMN IF NOT EXISTS closing_line NUMERIC,
                ADD COLUMN IF NOT EXISTS closing_price NUMERIC,
                ADD COLUMN IF NOT EXISTS clv_line NUMERIC,
                ADD COLUMN IF NOT EXISTS clv_price NUMERIC,
                ADD COLUMN IF NOT EXISTS beat_clv_line BOOLEAN,
                ADD COLUMN IF NOT EXISTS beat_clv_price BOOLEAN,
                ADD COLUMN IF NOT EXISTS closing_source_row_id BIGINT,
                ADD COLUMN IF NOT EXISTS closing_snapshot_id BIGINT,
                ADD COLUMN IF NOT EXISTS closing_fetched_at_utc TIMESTAMPTZ,
                ADD COLUMN IF NOT EXISTS clv_match_method TEXT,
                ADD COLUMN IF NOT EXISTS clv_valid BOOLEAN,
                ADD COLUMN IF NOT EXISTS clv_status TEXT,
                ADD COLUMN IF NOT EXISTS clv_unknown_reason TEXT,
                ADD COLUMN IF NOT EXISTS lock_snapshot_id BIGINT,
                ADD COLUMN IF NOT EXISTS locked_at_utc TIMESTAMPTZ;
            """
        )
    conn.commit()
    ensure_prop_offer_snapshot_schema(conn)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, game_date_et, game_slug, player_id, player_name, stat,
                   book_line, bet_side, bookmaker_key, bet_price,
                   prediction_key, prop_offer_id, prop_offer_source_row_id,
                   lock_snapshot_id, locked_at_utc
            FROM bets.mlb_prop_predictions
            WHERE game_date_et <= %s
              AND (actual_value IS NULL OR clv_status IS NULL)
        """, (today,))
        pending = cur.fetchall()

    if not pending:
        log.info("No pending MLB prop predictions to update.")
        return 0

    slugs      = list({r["game_slug"]  for r in pending})
    player_ids = list({int(r["player_id"]) for r in pending})

    # Pull all relevant gamelogs from finished games in one query
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT gl.game_slug, gl.player_id,
                   gl.strikeouts_pitcher,
                   gl.hits,
                   gl.home_runs,
                   gl.total_bases,
                   gl.walks_batter
            FROM raw.mlb_player_gamelogs gl
            JOIN raw.mlb_games g ON g.game_slug = gl.game_slug
            WHERE gl.game_slug = ANY(%s)
              AND gl.player_id = ANY(%s)
              AND g.status = 'final'
        """, (slugs, player_ids))
        # Key: (game_slug, player_id)
        actuals = {
            (r["game_slug"], int(r["player_id"])): r
            for r in cur.fetchall()
        }

    updated = 0
    with conn.cursor() as cur:
        for row in pending:
            key = (row["game_slug"], int(row["player_id"]))
            if key not in actuals:
                continue

            a = actuals[key]
            stat = row["stat"]
            gl_col = _STAT_COL.get(stat)
            if gl_col is None:
                log.warning("Unknown stat '%s' for prop id=%s — skipping", stat, row["id"])
                continue

            raw_val = a[gl_col]
            if raw_val is None:
                continue

            actual_value = float(raw_val)
            over_hit = None
            if row["book_line"] is not None:
                book_line = float(row["book_line"])
                if abs(actual_value - book_line) > 1e-9:
                    over_hit = actual_value > book_line
            closing_line = closing_price = clv_line = clv_price = None
            closing_source_row_id = closing_snapshot_id = None
            closing_fetched_at_utc = clv_match_method = None
            clv_valid = False
            clv_status = "unknown"
            clv_unknown_reason = "no_valid_close_snapshot"
            beat_clv_line = beat_clv_price = None
            if row["book_line"] is not None and row.get("bet_side") in {"over", "under"}:
                close = resolve_valid_prop_close(conn, dict(row))
                clv_valid = bool(close["valid"])
                clv_status = close["status"]
                clv_unknown_reason = close["unknown_reason"]
                clv_match_method = close.get("match_method")
                if clv_valid:
                    closing_line = _clean_float(close.get("line"))
                    closing_price = _clean_float(
                        close.get("over_price") if row["bet_side"] == "over" else close.get("under_price")
                    )
                    book_line = _clean_float(row.get("book_line"))
                    if closing_line is not None and book_line is not None:
                        clv_line = (
                            round(closing_line - book_line, 2)
                            if row["bet_side"] == "over"
                            else round(book_line - closing_line, 2)
                        )
                        beat_clv_line = clv_line > 0
                    entry_price = _clean_float(
                        close.get("lock_over_price")
                        if row["bet_side"] == "over"
                        else close.get("lock_under_price")
                    )
                    if entry_price is None:
                        entry_price = _clean_float(row.get("bet_price"))
                    if entry_price is not None and closing_price is not None:
                        clv_price = _price_clv(entry_price, closing_price)
                        beat_clv_price = clv_price > 0
                    closing_source_row_id = close.get("source_row_id")
                    closing_snapshot_id = close.get("snapshot_id")
                    closing_fetched_at_utc = close.get("fetched_at_utc")

            cur.execute("""
                UPDATE bets.mlb_prop_predictions
                SET actual_value = %s,
                    over_hit     = %s,
                    closing_line = %s,
                    closing_price = %s,
                    clv_line = %s,
                    clv_price = %s,
                    beat_clv_line = %s,
                    beat_clv_price = %s,
                    closing_source_row_id = %s,
                    closing_snapshot_id = %s,
                    closing_fetched_at_utc = %s,
                    clv_match_method = %s,
                    clv_valid = %s,
                    clv_status = %s,
                    clv_unknown_reason = %s
                WHERE id = %s
            """, (
                actual_value,
                over_hit,
                closing_line,
                closing_price,
                clv_line,
                clv_price,
                beat_clv_line,
                beat_clv_price,
                closing_source_row_id,
                closing_snapshot_id,
                closing_fetched_at_utc,
                clv_match_method,
                clv_valid,
                clv_status,
                clv_unknown_reason,
                row["id"],
            ))
            updated += 1

    conn.commit()
    log.info("update_prop_outcomes: updated %d prop rows", updated)
    return updated


def _clean_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def print_running_record(conn) -> None:
    """Print a one-line MLB ATS/total record summary with CLV stats."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE run_line_covered IS NOT NULL)        AS rl_n,
                SUM(CASE WHEN run_line_covered = TRUE THEN 1 ELSE 0 END)    AS rl_wins,
                COUNT(*) FILTER (WHERE total_covered IS NOT NULL)           AS tot_n,
                SUM(CASE WHEN total_covered = TRUE THEN 1 ELSE 0 END)       AS tot_wins,
                COUNT(*) FILTER (WHERE clv_run_line IS NOT NULL)            AS clv_rl_n,
                SUM(CASE WHEN clv_run_line > 0 THEN 1 ELSE 0 END)
                    FILTER (WHERE clv_run_line IS NOT NULL)                  AS clv_rl_beat,
                AVG(clv_run_line) FILTER (WHERE clv_run_line IS NOT NULL)   AS avg_clv_rl,
                COUNT(*) FILTER (WHERE clv_total IS NOT NULL)               AS clv_tot_n,
                AVG(clv_total)    FILTER (WHERE clv_total IS NOT NULL)      AS avg_clv_tot,
                COUNT(*) FILTER (WHERE clv_rl_price IS NOT NULL)            AS price_clv_rl_n,
                AVG(clv_rl_price) FILTER (WHERE clv_rl_price IS NOT NULL)   AS avg_price_clv_rl
            FROM bets.mlb_game_predictions
            WHERE run_line_bet_side IS NOT NULL OR total_bet_side IS NOT NULL
        """)
        row = cur.fetchone()

    if not row:
        return

    rl_n     = int(row["rl_n"]   or 0)
    rl_w     = int(row["rl_wins"] or 0)
    tot_n    = int(row["tot_n"]  or 0)
    tot_w    = int(row["tot_wins"] or 0)
    clv_rl_n = int(row["clv_rl_n"] or 0)
    clv_rl_b = int(row["clv_rl_beat"] or 0)
    avg_clv_rl  = float(row["avg_clv_rl"]  or 0.0)
    clv_tot_n = int(row["clv_tot_n"] or 0)
    avg_clv_tot = float(row["avg_clv_tot"] or 0.0)
    price_clv_rl_n  = int(row["price_clv_rl_n"] or 0)
    avg_price_clv_rl = float(row["avg_price_clv_rl"] or 0.0)

    rl_pct  = (rl_w / rl_n * 100)   if rl_n  > 0 else 0.0
    tot_pct = (tot_w / tot_n * 100) if tot_n > 0 else 0.0
    rl_roi  = ((rl_w * 100  - (rl_n  - rl_w)  * 110) / (rl_n  * 110) * 100) if rl_n  > 0 else 0.0
    tot_roi = ((tot_w * 100 - (tot_n - tot_w) * 110) / (tot_n * 110) * 100) if tot_n > 0 else 0.0
    clv_rl_pct = (clv_rl_b / clv_rl_n * 100) if clv_rl_n > 0 else 0.0

    print(
        f"MLB Run Line: {rl_w}-{rl_n - rl_w} ({rl_pct:.1f}%) ROI: {rl_roi:+.1f}%"
        f" | Total: {tot_w}-{tot_n - tot_w} ({tot_pct:.1f}%) ROI: {tot_roi:+.1f}%"
    )
    if clv_rl_n > 0:
        print(
            f"MLB CLV Run Line: beat close {clv_rl_b}/{clv_rl_n} ({clv_rl_pct:.0f}%)"
            f" avg CLV={avg_clv_rl:+.2f} runs"
            + (f" | CLV Total avg={avg_clv_tot:+.2f} runs" if clv_tot_n > 0 else "")
        )
    if price_clv_rl_n > 0:
        print(f"MLB Price CLV Run Line: {price_clv_rl_n} bets  avg={avg_price_clv_rl:+.2f}%")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    with psycopg2.connect(PG_DSN) as conn:
        n_games = update_game_outcomes(conn)
        log.info("Updated %d MLB game outcome rows", n_games)

        n_clv = backfill_clv(conn)
        if n_clv:
            log.info("Backfilled CLV for %d historical rows", n_clv)

        n_props = update_prop_outcomes(conn)
        log.info("Updated %d MLB prop outcome rows", n_props)

        n_ledger = grade_bankroll_ledger(conn)
        log.info("Graded %d MLB bankroll ledger rows", n_ledger)

        n_model_ledger = grade_model_pick_ledger(conn)
        log.info("Graded %d MLB model-pick ledger rows", n_model_ledger)

        print_running_record(conn)


if __name__ == "__main__":
    main()
