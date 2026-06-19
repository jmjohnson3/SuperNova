"""Build side-level MLB prop market training examples.

Each row represents one offered side for one replayed model prediction:
player/date/market/line/book/side/price.  This table is the common source for
direct side classifiers, betting-layer models, CLV models, recalibration, and
bucket reopen decisions.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

from .prop_replay import (
    american_to_prob,
    ensure_prop_replay_schema,
    ev_per_unit,
    no_vig_probs,
)
from .prop_offer_links import ensure_prop_offer_links_schema
from .side_recalibration import price_bucket, prop_line_bucket, prop_line_surface

log = logging.getLogger("mlb_pipeline.modeling.prop_market_training")

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MARKETS = ("pitcher_strikeouts", "batter_hits", "batter_total_bases", "batter_home_runs")
_SQL_DIR = Path(__file__).resolve().parents[3] / "sql"
_GAME_FEATURES_READY = False
_GAME_FEATURES_LOCK_KEY = "mlb_game_training_features_ddl"
_MARKET_TRAINING_SCHEMA_READY = False


@dataclass(frozen=True)
class PropMarketTrainingConfig:
    pg_dsn: str = _PG_DSN
    lookback_days: int = 365
    date_from: date | None = None
    date_to: date | None = None
    run_ids: tuple[str, ...] = ()
    include_pending: bool = False
    require_lock: bool = True
    replace: bool = True
    ensure_schema: bool = False


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _kelly_from_price(p_win: Any, price: Any) -> float | None:
    p = _clean_float(p_win)
    pr = _clean_float(price)
    if p is None or pr is None or pr == 0:
        return None
    b = pr / 100.0 if pr > 0 else 100.0 / abs(pr)
    if b <= 0:
        return None
    k = (b * p - (1.0 - p)) / b
    return max(0.0, k)


def _profit_units(won: bool | None, push: bool, price: Any) -> float | None:
    if push:
        return 0.0
    if won is None:
        return None
    pr = _clean_float(price)
    if pr is None or pr == 0:
        return None
    if not won:
        return -1.0
    return round(pr / 100.0 if pr > 0 else 100.0 / abs(pr), 4)


def _pair_quality(paired_price_source: Any, paired_price: Any) -> str:
    source = str(paired_price_source or "").strip().lower()
    if paired_price is None:
        return "one_sided"
    if source in {"same_book_exact_line", "same_book_exact_line_fallback", "prediction_same_book"}:
        return "same_book"
    if source in {"cross_book_exact_line", "cross_book_exact_line_fallback", "prediction_cross_book"}:
        return "cross_book"
    if source == "synthetic_fanduel_over_only_complement":
        return "synthetic"
    if "same_book" in source:
        return "same_book"
    if "cross_book" in source:
        return "cross_book"
    if "synthetic" in source:
        return "synthetic"
    return "unknown"


def ensure_prop_market_training_schema(conn) -> None:
    global _MARKET_TRAINING_SCHEMA_READY
    if _MARKET_TRAINING_SCHEMA_READY:
        return
    if _table_exists(conn, "features", "mlb_prop_market_training_examples") and _market_training_has_required_columns(conn):
        _MARKET_TRAINING_SCHEMA_READY = True
        return
    with conn.cursor() as cur:
        cur.execute("SET LOCAL lock_timeout = '2s'")
        cur.execute("SET LOCAL statement_timeout = '20s'")
        cur.execute(
            """
            CREATE SCHEMA IF NOT EXISTS features;
            CREATE TABLE IF NOT EXISTS features.mlb_prop_market_training_examples (
                id BIGSERIAL PRIMARY KEY,
                source TEXT NOT NULL DEFAULT 'replay',
                run_id TEXT NOT NULL,
                replay_id BIGINT NOT NULL,
                source_pred_id INTEGER,
                prediction_key TEXT,
                prop_offer_id BIGINT,
                prop_offer_source_row_id INTEGER,
                lock_snapshot_id BIGINT,
                game_date_et DATE NOT NULL,
                game_slug TEXT NOT NULL,
                player_id BIGINT NOT NULL,
                player_name TEXT,
                player_name_norm TEXT,
                team_abbr TEXT,
                market TEXT NOT NULL,
                side TEXT NOT NULL,
                bookmaker_key TEXT,
                market_line NUMERIC,
                market_price NUMERIC,
                paired_price NUMERIC,
                paired_bookmaker_key TEXT,
                paired_price_source TEXT,
                pair_quality TEXT,
                same_book_pair_flag NUMERIC,
                cross_book_pair_flag NUMERIC,
                synthetic_pair_flag NUMERIC,
                clean_market_pair_flag NUMERIC,
                true_pair_flag NUMERIC,
                minutes_to_first_pitch_at_lock NUMERIC,
                lock_price_age_minutes NUMERIC,
                raw_market_prob NUMERIC,
                no_vig_market_prob NUMERIC,
                market_prob_side NUMERIC,
                market_prob_source TEXT,
                price_bucket TEXT,
                line_bucket TEXT,
                line_surface TEXT,
                model_family TEXT,
                edge_type TEXT,
                pred_value NUMERIC,
                pred_count NUMERIC,
                model_prob_over NUMERIC,
                model_prob_side NUMERIC,
                count_edge_side NUMERIC,
                prob_edge_vs_market NUMERIC,
                confirmed_batting_order NUMERIC,
                confirmed_lineup_source TEXT,
                projected_pa NUMERIC,
                pa_games INTEGER,
                projected_ip NUMERIC,
                projected_bf NUMERIC,
                projected_pitch_count NUMERIC,
                pitcher_starts INTEGER,
                is_home NUMERIC,
                opponent_abbr TEXT,
                opp_sp_id BIGINT,
                opp_sp_hand TEXT,
                opp_sp_hand_l NUMERIC,
                opp_sp_k_pct_10 NUMERIC,
                opp_sp_bb_pct NUMERIC,
                opp_sp_xwoba NUMERIC,
                opp_sp_hard_hit_pct NUMERIC,
                opp_sp_whiff_pct NUMERIC,
                opp_bp_era_10 NUMERIC,
                opp_bp_whip_10 NUMERIC,
                opp_bp_k9_10 NUMERIC,
                opp_bp_ip_last_3 NUMERIC,
                opp_bp_ip_last_7 NUMERIC,
                opp_team_k_pct_10 NUMERIC,
                opp_team_avg_10 NUMERIC,
                opp_team_obp_10 NUMERIC,
                opp_team_slg_10 NUMERIC,
                batter_vs_hand_hits_avg_10 NUMERIC,
                batter_vs_hand_tb_avg_10 NUMERIC,
                batter_vs_hand_hr_avg_10 NUMERIC,
                batter_vs_hand_iso_avg_10 NUMERIC,
                batter_vs_hand_k_rate_10 NUMERIC,
                batter_vs_hand_games_10 NUMERIC,
                batter_vs_rp_ba_30 NUMERIC,
                batter_vs_rp_slg_30 NUMERIC,
                batter_vs_rp_hr_rate_30 NUMERIC,
                batter_vs_rp_k_rate_30 NUMERIC,
                pinch_hit_risk NUMERIC,
                team_implied_runs NUMERIC,
                opponent_implied_runs NUMERIC,
                game_total_line NUMERIC,
                actual_pa NUMERIC,
                actual_bf NUMERIC,
                actual_ip NUMERIC,
                actual_pitch_count_proxy NUMERIC,
                low_pa_flag NUMERIC,
                ev NUMERIC,
                kelly_fraction NUMERIC,
                actual_value NUMERIC,
                over_hit BOOLEAN,
                won BOOLEAN,
                push BOOLEAN,
                profit_units NUMERIC,
                closing_line NUMERIC,
                closing_price NUMERIC,
                clv_line NUMERIC,
                clv_price NUMERIC,
                closing_source_row_id BIGINT,
                closing_snapshot_id BIGINT,
                closing_fetched_at_utc TIMESTAMPTZ,
                clv_match_method TEXT,
                clv_valid BOOLEAN,
                clv_status TEXT,
                clv_unknown_reason TEXT,
                beat_clv_line BOOLEAN,
                beat_clv_price BOOLEAN,
                result_status TEXT NOT NULL DEFAULT 'pending',
                source_created_at TIMESTAMPTZ,
                example_updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (run_id, replay_id, side)
            );
            ALTER TABLE features.mlb_prop_market_training_examples
                ADD COLUMN IF NOT EXISTS prediction_key TEXT,
                ADD COLUMN IF NOT EXISTS prop_offer_id BIGINT,
                ADD COLUMN IF NOT EXISTS prop_offer_source_row_id INTEGER,
                ADD COLUMN IF NOT EXISTS lock_snapshot_id BIGINT,
                ADD COLUMN IF NOT EXISTS line_surface TEXT,
                ADD COLUMN IF NOT EXISTS paired_bookmaker_key TEXT,
                ADD COLUMN IF NOT EXISTS paired_price_source TEXT,
                ADD COLUMN IF NOT EXISTS pair_quality TEXT,
                ADD COLUMN IF NOT EXISTS same_book_pair_flag NUMERIC,
                ADD COLUMN IF NOT EXISTS cross_book_pair_flag NUMERIC,
                ADD COLUMN IF NOT EXISTS synthetic_pair_flag NUMERIC,
                ADD COLUMN IF NOT EXISTS clean_market_pair_flag NUMERIC,
                ADD COLUMN IF NOT EXISTS true_pair_flag NUMERIC,
                ADD COLUMN IF NOT EXISTS minutes_to_first_pitch_at_lock NUMERIC,
                ADD COLUMN IF NOT EXISTS lock_price_age_minutes NUMERIC,
                ADD COLUMN IF NOT EXISTS market_prob_source TEXT,
                ADD COLUMN IF NOT EXISTS closing_source_row_id BIGINT,
                ADD COLUMN IF NOT EXISTS closing_snapshot_id BIGINT,
                ADD COLUMN IF NOT EXISTS closing_fetched_at_utc TIMESTAMPTZ,
                ADD COLUMN IF NOT EXISTS clv_match_method TEXT,
                ADD COLUMN IF NOT EXISTS clv_valid BOOLEAN,
                ADD COLUMN IF NOT EXISTS clv_status TEXT,
                ADD COLUMN IF NOT EXISTS clv_unknown_reason TEXT,
                ADD COLUMN IF NOT EXISTS confirmed_batting_order NUMERIC,
                ADD COLUMN IF NOT EXISTS confirmed_lineup_source TEXT,
                ADD COLUMN IF NOT EXISTS projected_pa NUMERIC,
                ADD COLUMN IF NOT EXISTS pa_games INTEGER,
                ADD COLUMN IF NOT EXISTS projected_ip NUMERIC,
                ADD COLUMN IF NOT EXISTS projected_bf NUMERIC,
                ADD COLUMN IF NOT EXISTS projected_pitch_count NUMERIC,
                ADD COLUMN IF NOT EXISTS pitcher_starts INTEGER,
                ADD COLUMN IF NOT EXISTS is_home NUMERIC,
                ADD COLUMN IF NOT EXISTS opponent_abbr TEXT,
                ADD COLUMN IF NOT EXISTS opp_sp_id BIGINT,
                ADD COLUMN IF NOT EXISTS opp_sp_hand TEXT,
                ADD COLUMN IF NOT EXISTS opp_sp_hand_l NUMERIC,
                ADD COLUMN IF NOT EXISTS opp_sp_k_pct_10 NUMERIC,
                ADD COLUMN IF NOT EXISTS opp_sp_bb_pct NUMERIC,
                ADD COLUMN IF NOT EXISTS opp_sp_xwoba NUMERIC,
                ADD COLUMN IF NOT EXISTS opp_sp_hard_hit_pct NUMERIC,
                ADD COLUMN IF NOT EXISTS opp_sp_whiff_pct NUMERIC,
                ADD COLUMN IF NOT EXISTS opp_bp_era_10 NUMERIC,
                ADD COLUMN IF NOT EXISTS opp_bp_whip_10 NUMERIC,
                ADD COLUMN IF NOT EXISTS opp_bp_k9_10 NUMERIC,
                ADD COLUMN IF NOT EXISTS opp_bp_ip_last_3 NUMERIC,
                ADD COLUMN IF NOT EXISTS opp_bp_ip_last_7 NUMERIC,
                ADD COLUMN IF NOT EXISTS opp_team_k_pct_10 NUMERIC,
                ADD COLUMN IF NOT EXISTS opp_team_avg_10 NUMERIC,
                ADD COLUMN IF NOT EXISTS opp_team_obp_10 NUMERIC,
                ADD COLUMN IF NOT EXISTS opp_team_slg_10 NUMERIC,
                ADD COLUMN IF NOT EXISTS batter_vs_hand_hits_avg_10 NUMERIC,
                ADD COLUMN IF NOT EXISTS batter_vs_hand_tb_avg_10 NUMERIC,
                ADD COLUMN IF NOT EXISTS batter_vs_hand_hr_avg_10 NUMERIC,
                ADD COLUMN IF NOT EXISTS batter_vs_hand_iso_avg_10 NUMERIC,
                ADD COLUMN IF NOT EXISTS batter_vs_hand_k_rate_10 NUMERIC,
                ADD COLUMN IF NOT EXISTS batter_vs_hand_games_10 NUMERIC,
                ADD COLUMN IF NOT EXISTS batter_vs_rp_ba_30 NUMERIC,
                ADD COLUMN IF NOT EXISTS batter_vs_rp_slg_30 NUMERIC,
                ADD COLUMN IF NOT EXISTS batter_vs_rp_hr_rate_30 NUMERIC,
                ADD COLUMN IF NOT EXISTS batter_vs_rp_k_rate_30 NUMERIC,
                ADD COLUMN IF NOT EXISTS pinch_hit_risk NUMERIC,
                ADD COLUMN IF NOT EXISTS team_implied_runs NUMERIC,
                ADD COLUMN IF NOT EXISTS opponent_implied_runs NUMERIC,
                ADD COLUMN IF NOT EXISTS game_total_line NUMERIC,
                ADD COLUMN IF NOT EXISTS actual_pa NUMERIC,
                ADD COLUMN IF NOT EXISTS actual_bf NUMERIC,
                ADD COLUMN IF NOT EXISTS actual_ip NUMERIC,
                ADD COLUMN IF NOT EXISTS actual_pitch_count_proxy NUMERIC,
                ADD COLUMN IF NOT EXISTS low_pa_flag NUMERIC;
            CREATE INDEX IF NOT EXISTS idx_mlb_prop_market_training_date
                ON features.mlb_prop_market_training_examples (game_date_et);
            CREATE INDEX IF NOT EXISTS idx_mlb_prop_market_training_bucket
                ON features.mlb_prop_market_training_examples
                (market, side, line_surface, line_bucket, price_bucket, model_family);
            CREATE INDEX IF NOT EXISTS idx_mlb_prop_market_training_status
                ON features.mlb_prop_market_training_examples (result_status, game_date_et);
            CREATE INDEX IF NOT EXISTS idx_mlb_prop_market_training_offer
                ON features.mlb_prop_market_training_examples (prop_offer_id);
            """
        )
    conn.commit()
    _MARKET_TRAINING_SCHEMA_READY = True


def _table_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
              SELECT 1 FROM information_schema.tables
              WHERE table_schema = %s AND table_name = %s
            )
            """,
            (schema, table),
        )
        return bool(cur.fetchone()[0])


def _ensure_lineup_name_columns(conn) -> None:
    if not _table_exists(conn, "raw", "mlb_lineups"):
        return
    with conn.cursor() as cur:
        cur.execute("SET LOCAL lock_timeout = '2s'")
        cur.execute("SET LOCAL statement_timeout = '20s'")
        cur.execute(
            """
            ALTER TABLE raw.mlb_lineups
                ADD COLUMN IF NOT EXISTS player_name TEXT,
                ADD COLUMN IF NOT EXISTS player_name_norm TEXT;
            CREATE INDEX IF NOT EXISTS idx_mlb_lineups_name_norm
                ON raw.mlb_lineups (game_slug, team_abbr, player_name_norm);
            """
        )
    conn.commit()


def _ensure_boxscore_player_stats_compat(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SET LOCAL lock_timeout = '2s'")
        cur.execute("SET LOCAL statement_timeout = '20s'")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS raw.mlb_boxscore_player_stats (
                game_slug TEXT NOT NULL,
                player_id INTEGER NOT NULL,
                season TEXT,
                team_abbr TEXT,
                team_id INTEGER,
                is_home BOOLEAN,
                first_name TEXT,
                last_name TEXT,
                primary_position TEXT,
                batting_order INTEGER,
                stats JSONB,
                source_fetched_at_utc TIMESTAMPTZ,
                updated_at_utc TIMESTAMPTZ,
                PRIMARY KEY (game_slug, player_id)
            );
            CREATE INDEX IF NOT EXISTS idx_mlb_boxscore_player_stats_game
                ON raw.mlb_boxscore_player_stats (game_slug, team_abbr);
            """
        )
    conn.commit()


def _market_training_has_required_columns(conn) -> bool:
    required = {
        "prediction_key", "prop_offer_id", "prop_offer_source_row_id",
        "lock_snapshot_id", "market", "side", "market_line", "market_price",
        "paired_price", "paired_bookmaker_key", "paired_price_source", "pair_quality",
        "same_book_pair_flag", "cross_book_pair_flag", "synthetic_pair_flag",
        "clean_market_pair_flag", "true_pair_flag", "minutes_to_first_pitch_at_lock",
        "lock_price_age_minutes",
        "no_vig_market_prob", "market_prob_side", "market_prob_source",
        "line_surface", "projected_pa", "projected_bf",
        "projected_pitch_count", "actual_pa", "actual_bf",
        "actual_pitch_count_proxy", "clv_valid", "beat_clv_price",
        "result_status",
    }
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'features'
              AND table_name = 'mlb_prop_market_training_examples'
            """
        )
        existing = {str(row[0]) for row in cur.fetchall()}
    return required.issubset(existing)


def _table_has_columns(conn, schema: str, table: str, columns: set[str]) -> bool:
    if not _table_exists(conn, schema, table):
        return False
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name = %s
            """,
            (schema, table),
        )
        existing = {str(row[0]) for row in cur.fetchall()}
    return columns.issubset(existing)


def prepare_prop_market_training_dependencies(conn) -> None:
    """Create/upgrade tables and compatibility views used by the trainer.

    This is intentionally opt-in for scheduler safety.  Live pregame/close
    jobs should not perform DDL while trying to publish or capture odds.
    """
    ensure_prop_replay_schema(conn)
    ensure_prop_market_training_schema(conn)
    ensure_game_training_features(conn)
    ensure_prop_offer_links_schema(conn)
    _ensure_lineup_name_columns(conn)
    _ensure_boxscore_player_stats_compat(conn)


def verify_prop_market_training_dependencies(conn) -> list[str]:
    """Return missing dependencies without creating or altering anything."""
    missing: list[str] = []
    if not _table_exists(conn, "features", "mlb_prop_market_training_examples"):
        missing.append("features.mlb_prop_market_training_examples")
    elif not _market_training_has_required_columns(conn):
        missing.append("features.mlb_prop_market_training_examples required columns")
    if not _table_exists(conn, "bets", "mlb_prop_prediction_replay"):
        missing.append("bets.mlb_prop_prediction_replay")
    if not _table_exists(conn, "features", "mlb_prop_offer_links"):
        missing.append("features.mlb_prop_offer_links")
    if not _relation_exists(conn, "features.mlb_game_training_features"):
        missing.append("features.mlb_game_training_features")
    if not _table_exists(conn, "raw", "mlb_games"):
        missing.append("raw.mlb_games")
    if not _table_exists(conn, "raw", "mlb_player_gamelogs"):
        missing.append("raw.mlb_player_gamelogs")
    if not _table_exists(conn, "raw", "mlb_boxscore_player_stats"):
        missing.append("raw.mlb_boxscore_player_stats")
    if not _table_has_columns(conn, "raw", "mlb_lineups", {"player_name_norm"}):
        missing.append("raw.mlb_lineups.player_name_norm")
    return missing


def _relation_exists(conn, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (name,))
        return cur.fetchone()[0] is not None


def _create_minimal_game_training_features(conn) -> None:
    """Create a compatibility view when the full MLB006 view cannot be applied.

    The prop market training table can still build without these context
    features; the affected feature values will be NULL instead of crashing the
    daily pre-game/report path.
    """
    numeric_cols = [
        "home_sp_pitch_hand_l",
        "away_sp_pitch_hand_l",
        "home_sp_k_pct_10",
        "away_sp_k_pct_10",
        "home_sp_sc_bb_pct",
        "away_sp_sc_bb_pct",
        "home_sp_sc_xwoba",
        "away_sp_sc_xwoba",
        "home_sp_sc_hard_hit_pct",
        "away_sp_sc_hard_hit_pct",
        "home_sp_sc_whiff_pct",
        "away_sp_sc_whiff_pct",
        "home_bp_era_10",
        "away_bp_era_10",
        "home_bp_whip_10",
        "away_bp_whip_10",
        "home_bp_k9_10",
        "away_bp_k9_10",
        "home_bullpen_ip_last_3",
        "away_bullpen_ip_last_3",
        "home_bullpen_ip_last_7",
        "away_bullpen_ip_last_7",
        "home_k_pct_avg_10",
        "away_k_pct_avg_10",
        "home_avg_avg_10",
        "away_avg_avg_10",
        "home_obp_avg_10",
        "away_obp_avg_10",
        "home_slg_avg_10",
        "away_slg_avg_10",
        "home_implied_runs",
        "away_implied_runs",
        "total_line",
    ]
    select_cols = ",\n                ".join(
        f"NULL::numeric AS {col}" for col in numeric_cols
    )
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE SCHEMA IF NOT EXISTS features;
            CREATE OR REPLACE VIEW features.mlb_game_training_features AS
            SELECT
                g.game_slug,
                {select_cols}
            FROM raw.mlb_games g;
            """
        )
    conn.commit()


def ensure_game_training_features(conn) -> None:
    global _GAME_FEATURES_READY
    if _GAME_FEATURES_READY:
        return
    if _relation_exists(conn, "features.mlb_game_training_features"):
        _GAME_FEATURES_READY = True
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (_GAME_FEATURES_LOCK_KEY,))
            if _relation_exists(conn, "features.mlb_game_training_features"):
                conn.commit()
                _GAME_FEATURES_READY = True
                return
            cur.execute((_SQL_DIR / "MLB006_mlb_game_features.sql").read_text(encoding="utf-8"))
        conn.commit()
        _GAME_FEATURES_READY = True
        log.info("Created features.mlb_game_training_features from MLB006_mlb_game_features.sql")
    except Exception:
        conn.rollback()
        log.exception(
            "Could not apply MLB006_mlb_game_features.sql; creating minimal "
            "features.mlb_game_training_features compatibility view"
        )
        _create_minimal_game_training_features(conn)
        _GAME_FEATURES_READY = True


def _load_replay_rows(conn, cfg: PropMarketTrainingConfig) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)
    date_from = cfg.date_from or cutoff
    date_to = cfg.date_to
    filters = [
        "r.game_date_et >= %s",
        "r.stat = ANY(%s)",
        "r.model_prob_over IS NOT NULL",
        "r.market_line IS NOT NULL",
    ]
    params: list[Any] = [date_from, list(_MARKETS)]
    if date_to is not None:
        filters.append("r.game_date_et <= %s")
        params.append(date_to)
    if cfg.run_ids:
        filters.append("r.run_id = ANY(%s)")
        params.append(list(cfg.run_ids))
    if cfg.require_lock:
        filters.append("r.lock_snapshot_id IS NOT NULL")
    if not cfg.include_pending:
        filters.append("r.actual_value IS NOT NULL")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT
                r.*,
                CASE
                    WHEN r.side = 'over' THEN COALESCE(r.over_price::float, selected_offer.price::float, r.market_price::float)
                    WHEN r.side = 'under' THEN COALESCE(
                        r.over_price::float,
                        pair_same.price::float,
                        pair_fallback_same.price::float,
                        pair_cross.price::float,
                        pair_fallback_cross.price::float,
                        synthetic_pair.price
                    )
                    ELSE r.over_price::float
                END AS resolved_over_price,
                CASE
                    WHEN r.side = 'under' THEN COALESCE(r.under_price::float, selected_offer.price::float, r.market_price::float)
                    WHEN r.side = 'over' THEN COALESCE(
                        r.under_price::float,
                        pair_same.price::float,
                        pair_fallback_same.price::float,
                        pair_cross.price::float,
                        pair_fallback_cross.price::float,
                        synthetic_pair.price
                    )
                    ELSE r.under_price::float
                END AS resolved_under_price,
                CASE
                    WHEN r.side = 'over' AND r.under_price IS NOT NULL THEN LOWER(r.bookmaker_key)
                    WHEN r.side = 'under' AND r.over_price IS NOT NULL THEN LOWER(r.bookmaker_key)
                    WHEN pair_same.price IS NOT NULL THEN LOWER(pair_same.bookmaker_key)
                    WHEN pair_fallback_same.price IS NOT NULL THEN LOWER(pair_fallback_same.bookmaker_key)
                    WHEN pair_cross.price IS NOT NULL THEN LOWER(pair_cross.bookmaker_key)
                    WHEN pair_fallback_cross.price IS NOT NULL THEN LOWER(pair_fallback_cross.bookmaker_key)
                    WHEN synthetic_pair.price IS NOT NULL THEN LOWER(synthetic_pair.bookmaker_key)
                    ELSE NULL
                END AS paired_bookmaker_key_resolved,
                CASE
                    WHEN r.side = 'over' AND r.under_price IS NOT NULL THEN 'prediction_same_book'
                    WHEN r.side = 'under' AND r.over_price IS NOT NULL THEN 'prediction_same_book'
                    WHEN pair_same.price IS NOT NULL THEN 'same_book_exact_line'
                    WHEN pair_fallback_same.price IS NOT NULL THEN 'same_book_exact_line_fallback'
                    WHEN pair_cross.price IS NOT NULL THEN 'cross_book_exact_line'
                    WHEN pair_fallback_cross.price IS NOT NULL THEN 'cross_book_exact_line_fallback'
                    WHEN synthetic_pair.price IS NOT NULL THEN 'synthetic_fanduel_over_only_complement'
                    ELSE NULL
                END AS paired_price_source_resolved,
                EXTRACT(EPOCH FROM (
                    g.start_ts_utc - COALESCE(r.locked_at_utc, r.source_created_at, r.run_started_at_utc)
                )) / 60.0 AS minutes_to_first_pitch_at_lock,
                EXTRACT(EPOCH FROM (
                    COALESCE(r.locked_at_utc, r.source_created_at, r.run_started_at_utc) - selected_offer.fetched_at_utc
                )) / 60.0 AS lock_price_age_minutes,
                COALESCE(bps.batting_order, lu.batting_order, lu_name.batting_order) AS confirmed_batting_order,
                CASE
                    WHEN bps.batting_order IS NOT NULL THEN 'boxscore_actual'
                    WHEN lu.batting_order IS NOT NULL THEN lu.lineup_source
                    WHEN lu_name.batting_order IS NOT NULL THEN lu_name.lineup_source
                    ELSE NULL
                END AS confirmed_lineup_source,
                bat_opp.projected_pa,
                bat_opp.pa_games,
                pit_opp.projected_ip,
                pit_opp.projected_bf,
                pit_opp.projected_pitch_count,
                pit_opp.pitcher_starts,
                CASE
                    WHEN r.team_abbr = g.home_team_abbr THEN 1.0
                    WHEN r.team_abbr = g.away_team_abbr THEN 0.0
                    ELSE NULL
                END AS is_home,
                CASE
                    WHEN r.team_abbr = g.home_team_abbr THEN g.away_team_abbr
                    WHEN r.team_abbr = g.away_team_abbr THEN g.home_team_abbr
                    ELSE NULL
                END AS opponent_abbr,
                CASE
                    WHEN r.team_abbr = g.home_team_abbr THEN g.away_sp_id
                    WHEN r.team_abbr = g.away_team_abbr THEN g.home_sp_id
                    ELSE NULL
                END AS opp_sp_id,
                CASE
                    WHEN r.team_abbr = g.home_team_abbr THEN gf.away_sp_pitch_hand_l
                    WHEN r.team_abbr = g.away_team_abbr THEN gf.home_sp_pitch_hand_l
                    ELSE NULL
                END AS opp_sp_hand_l,
                CASE
                    WHEN r.team_abbr = g.home_team_abbr AND gf.away_sp_pitch_hand_l = 1 THEN 'L'
                    WHEN r.team_abbr = g.home_team_abbr AND gf.away_sp_pitch_hand_l = 0 THEN 'R'
                    WHEN r.team_abbr = g.away_team_abbr AND gf.home_sp_pitch_hand_l = 1 THEN 'L'
                    WHEN r.team_abbr = g.away_team_abbr AND gf.home_sp_pitch_hand_l = 0 THEN 'R'
                    ELSE bvh.opp_sp_hand
                END AS opp_sp_hand,
                CASE WHEN r.team_abbr = g.home_team_abbr THEN gf.away_sp_k_pct_10 ELSE gf.home_sp_k_pct_10 END AS opp_sp_k_pct_10,
                CASE WHEN r.team_abbr = g.home_team_abbr THEN gf.away_sp_sc_bb_pct ELSE gf.home_sp_sc_bb_pct END AS opp_sp_bb_pct,
                CASE WHEN r.team_abbr = g.home_team_abbr THEN gf.away_sp_sc_xwoba ELSE gf.home_sp_sc_xwoba END AS opp_sp_xwoba,
                CASE WHEN r.team_abbr = g.home_team_abbr THEN gf.away_sp_sc_hard_hit_pct ELSE gf.home_sp_sc_hard_hit_pct END AS opp_sp_hard_hit_pct,
                CASE WHEN r.team_abbr = g.home_team_abbr THEN gf.away_sp_sc_whiff_pct ELSE gf.home_sp_sc_whiff_pct END AS opp_sp_whiff_pct,
                CASE WHEN r.team_abbr = g.home_team_abbr THEN gf.away_bp_era_10 ELSE gf.home_bp_era_10 END AS opp_bp_era_10,
                CASE WHEN r.team_abbr = g.home_team_abbr THEN gf.away_bp_whip_10 ELSE gf.home_bp_whip_10 END AS opp_bp_whip_10,
                CASE WHEN r.team_abbr = g.home_team_abbr THEN gf.away_bp_k9_10 ELSE gf.home_bp_k9_10 END AS opp_bp_k9_10,
                CASE WHEN r.team_abbr = g.home_team_abbr THEN gf.away_bullpen_ip_last_3 ELSE gf.home_bullpen_ip_last_3 END AS opp_bp_ip_last_3,
                CASE WHEN r.team_abbr = g.home_team_abbr THEN gf.away_bullpen_ip_last_7 ELSE gf.home_bullpen_ip_last_7 END AS opp_bp_ip_last_7,
                CASE WHEN r.team_abbr = g.home_team_abbr THEN gf.away_k_pct_avg_10 ELSE gf.home_k_pct_avg_10 END AS opp_team_k_pct_10,
                CASE WHEN r.team_abbr = g.home_team_abbr THEN gf.away_avg_avg_10 ELSE gf.home_avg_avg_10 END AS opp_team_avg_10,
                CASE WHEN r.team_abbr = g.home_team_abbr THEN gf.away_obp_avg_10 ELSE gf.home_obp_avg_10 END AS opp_team_obp_10,
                CASE WHEN r.team_abbr = g.home_team_abbr THEN gf.away_slg_avg_10 ELSE gf.home_slg_avg_10 END AS opp_team_slg_10,
                CASE WHEN bvh.opp_sp_hand = 'L' THEN bvh.hits_avg_10_vs_lhp ELSE bvh.hits_avg_10_vs_rhp END AS batter_vs_hand_hits_avg_10,
                CASE WHEN bvh.opp_sp_hand = 'L' THEN bvh.tb_avg_10_vs_lhp ELSE bvh.tb_avg_10_vs_rhp END AS batter_vs_hand_tb_avg_10,
                CASE WHEN bvh.opp_sp_hand = 'L' THEN bvh.hr_avg_10_vs_lhp ELSE bvh.hr_avg_10_vs_rhp END AS batter_vs_hand_hr_avg_10,
                CASE WHEN bvh.opp_sp_hand = 'L' THEN bvh.iso_avg_10_vs_lhp ELSE bvh.iso_avg_10_vs_rhp END AS batter_vs_hand_iso_avg_10,
                CASE WHEN bvh.opp_sp_hand = 'L' THEN bvh.k_rate_avg_10_vs_lhp ELSE bvh.k_rate_avg_10_vs_rhp END AS batter_vs_hand_k_rate_10,
                CASE WHEN bvh.opp_sp_hand = 'L' THEN bvh.n_games_vs_lhp_10 ELSE bvh.n_games_vs_rhp_10 END AS batter_vs_hand_games_10,
                bvr.bvr_ba_30 AS batter_vs_rp_ba_30,
                bvr.bvr_slg_30 AS batter_vs_rp_slg_30,
                bvr.bvr_hr_rate_30 AS batter_vs_rp_hr_rate_30,
                bvr.bvr_k_rate_30 AS batter_vs_rp_k_rate_30,
                CASE WHEN r.team_abbr = g.home_team_abbr THEN gf.home_implied_runs ELSE gf.away_implied_runs END AS team_implied_runs,
                CASE WHEN r.team_abbr = g.home_team_abbr THEN gf.away_implied_runs ELSE gf.home_implied_runs END AS opponent_implied_runs,
                gf.total_line AS game_total_line,
                CASE
                    WHEN r.stat IN ('batter_hits', 'batter_total_bases', 'batter_home_runs') THEN
                        COALESCE(
                            NULLIF(bps.stats #>> '{{batting,plateAppearances}}', '')::float,
                            GREATEST(COALESCE(actual_gl.at_bats, 0) + COALESCE(actual_gl.walks_batter, 0), 0)::float
                        )
                    ELSE NULL
                END AS actual_pa,
                CASE
                    WHEN r.stat = 'pitcher_strikeouts' THEN
                        GREATEST(
                            ROUND(COALESCE(actual_gl.innings_pitched, 0) * 3)
                            + COALESCE(actual_gl.hits_allowed, 0)
                            + COALESCE(actual_gl.walks_allowed, 0),
                            0
                        )
                    ELSE NULL
                END AS actual_bf,
                CASE WHEN r.stat = 'pitcher_strikeouts' THEN actual_gl.innings_pitched ELSE NULL END AS actual_ip,
                CASE
                    WHEN r.stat = 'pitcher_strikeouts' THEN
                        GREATEST(
                            ROUND(COALESCE(actual_gl.innings_pitched, 0) * 3)
                            + COALESCE(actual_gl.hits_allowed, 0)
                            + COALESCE(actual_gl.walks_allowed, 0),
                            0
                        ) * 3.85
                    ELSE NULL
                END AS actual_pitch_count_proxy,
                CASE
                    WHEN r.stat IN ('batter_hits', 'batter_total_bases', 'batter_home_runs')
                         AND actual_gl.player_id IS NOT NULL THEN
                        CASE
                            WHEN COALESCE(
                                NULLIF(bps.stats #>> '{{batting,plateAppearances}}', '')::float,
                                GREATEST(COALESCE(actual_gl.at_bats, 0) + COALESCE(actual_gl.walks_batter, 0), 0)::float
                            ) <= 2 THEN 1.0
                            ELSE 0.0
                        END
                    ELSE NULL
                END AS low_pa_flag,
                CASE
                    WHEN r.stat = 'pitcher_strikeouts' THEN NULL
                    WHEN COALESCE(bps.batting_order, lu.batting_order, lu_name.batting_order) IS NULL THEN 0.35
                    WHEN COALESCE(bps.batting_order, lu.batting_order, lu_name.batting_order) >= 8 THEN 0.25
                    WHEN COALESCE(bps.batting_order, lu.batting_order, lu_name.batting_order) >= 6 THEN 0.15
                    ELSE 0.05
                END AS pinch_hit_risk
            FROM bets.mlb_prop_prediction_replay r
            LEFT JOIN raw.mlb_games g
              ON g.game_slug = r.game_slug
            LEFT JOIN LATERAL (
                SELECT o.*
                FROM features.mlb_prop_offer_links o
                WHERE o.as_of_date = r.game_date_et
                  AND o.player_name_norm = r.player_name_norm
                  AND o.stat = r.stat
                  AND o.side = r.side
                  AND ABS(o.line::float - r.market_line::float) <= 1e-9
                  AND LOWER(o.bookmaker_key) = LOWER(COALESCE(r.bookmaker_key, o.bookmaker_key))
                  AND (r.prop_offer_id IS NULL OR o.id = r.prop_offer_id)
                  AND (
                        o.id = r.prop_offer_id
                     OR
                        g.game_slug IS NULL
                     OR (
                            UPPER(o.home_team) IN (UPPER(g.home_team_abbr), UPPER(g.away_team_abbr))
                        AND UPPER(o.away_team) IN (UPPER(g.home_team_abbr), UPPER(g.away_team_abbr))
                     )
                  )
                ORDER BY
                    CASE WHEN o.id = r.prop_offer_id THEN 0 ELSE 1 END,
                    o.updated_at_utc DESC NULLS LAST,
                    o.fetched_at_utc DESC NULLS LAST,
                    o.id DESC
                LIMIT 1
            ) selected_offer ON TRUE
            LEFT JOIN LATERAL (
                SELECT o.*
                FROM features.mlb_prop_offer_links o
                WHERE selected_offer.id IS NOT NULL
                  AND o.as_of_date = selected_offer.as_of_date
                  AND o.event_id = selected_offer.event_id
                  AND o.player_name_norm = selected_offer.player_name_norm
                  AND o.stat = selected_offer.stat
                  AND o.bookmaker_key = selected_offer.bookmaker_key
                  AND ABS(o.line::float - selected_offer.line::float) <= 1e-9
                  AND o.side = CASE WHEN selected_offer.side = 'over' THEN 'under' ELSE 'over' END
                  AND o.price IS NOT NULL
                ORDER BY
                    o.updated_at_utc DESC NULLS LAST,
                    o.fetched_at_utc DESC NULLS LAST,
                    o.id DESC
                LIMIT 1
            ) pair_same ON TRUE
            LEFT JOIN LATERAL (
                SELECT o.*
                FROM features.mlb_prop_offer_links o
                WHERE selected_offer.id IS NOT NULL
                  AND o.as_of_date = selected_offer.as_of_date
                  AND o.event_id = selected_offer.event_id
                  AND o.player_name_norm = selected_offer.player_name_norm
                  AND o.stat = selected_offer.stat
                  AND o.bookmaker_key <> selected_offer.bookmaker_key
                  AND ABS(o.line::float - selected_offer.line::float) <= 1e-9
                  AND o.side = CASE WHEN selected_offer.side = 'over' THEN 'under' ELSE 'over' END
                  AND o.price IS NOT NULL
                ORDER BY
                    CASE LOWER(o.bookmaker_key)
                        WHEN 'draftkings' THEN 0
                        WHEN 'fanduel' THEN 1
                        ELSE 2
                    END,
                    o.updated_at_utc DESC NULLS LAST,
                    o.fetched_at_utc DESC NULLS LAST,
                    o.id DESC
                LIMIT 1
            ) pair_cross ON TRUE
            LEFT JOIN LATERAL (
                SELECT o.*
                FROM features.mlb_prop_offer_links o
                WHERE (selected_offer.id IS NULL OR pair_same.id IS NULL)
                  AND o.as_of_date = r.game_date_et
                  AND o.player_name_norm = r.player_name_norm
                  AND o.stat = r.stat
                  AND LOWER(o.bookmaker_key) = LOWER(COALESCE(r.bookmaker_key, selected_offer.bookmaker_key))
                  AND ABS(o.line::float - r.market_line::float) <= 1e-9
                  AND o.side = CASE WHEN r.side = 'over' THEN 'under' ELSE 'over' END
                  AND o.price IS NOT NULL
                ORDER BY
                    o.updated_at_utc DESC NULLS LAST,
                    o.fetched_at_utc DESC NULLS LAST,
                    o.id DESC
                LIMIT 1
            ) pair_fallback_same ON TRUE
            LEFT JOIN LATERAL (
                SELECT o.*
                FROM features.mlb_prop_offer_links o
                WHERE (selected_offer.id IS NULL OR pair_cross.id IS NULL)
                  AND pair_fallback_same.id IS NULL
                  AND o.as_of_date = r.game_date_et
                  AND o.player_name_norm = r.player_name_norm
                  AND o.stat = r.stat
                  AND LOWER(o.bookmaker_key) <> LOWER(COALESCE(r.bookmaker_key, selected_offer.bookmaker_key, ''))
                  AND ABS(o.line::float - r.market_line::float) <= 1e-9
                  AND o.side = CASE WHEN r.side = 'over' THEN 'under' ELSE 'over' END
                  AND o.price IS NOT NULL
                ORDER BY
                    CASE LOWER(o.bookmaker_key)
                        WHEN 'draftkings' THEN 0
                        WHEN 'fanduel' THEN 1
                        ELSE 2
                    END,
                    o.updated_at_utc DESC NULLS LAST,
                    o.fetched_at_utc DESC NULLS LAST,
                    o.id DESC
                LIMIT 1
            ) pair_fallback_cross ON TRUE
            LEFT JOIN LATERAL (
                SELECT
                    CASE
                        WHEN p.complement_prob > 0.5 THEN
                            ROUND(-100.0 * p.complement_prob / NULLIF(1.0 - p.complement_prob, 0.0))::float
                        ELSE
                            ROUND(100.0 * (1.0 - p.complement_prob) / NULLIF(p.complement_prob, 0.0))::float
                    END AS price,
                    LOWER(COALESCE(selected_offer.bookmaker_key, r.bookmaker_key)) AS bookmaker_key
                FROM (
                    SELECT
                        1.0 - CASE
                            WHEN src.side_price > 0 THEN 100.0 / (src.side_price + 100.0)
                            ELSE ABS(src.side_price) / (ABS(src.side_price) + 100.0)
                        END AS complement_prob
                    FROM (
                        SELECT COALESCE(
                            selected_offer.price::float,
                            r.market_price::float,
                            CASE WHEN r.side = 'over' THEN r.over_price::float ELSE r.under_price::float END
                        ) AS side_price
                    ) src
                    WHERE src.side_price IS NOT NULL
                      AND src.side_price <> 0
                ) p
                WHERE pair_same.id IS NULL
                  AND pair_fallback_same.id IS NULL
                  AND pair_cross.id IS NULL
                  AND pair_fallback_cross.id IS NULL
                  AND LOWER(COALESCE(selected_offer.bookmaker_key, r.bookmaker_key)) = 'fanduel'
                  AND r.stat IN ('batter_hits', 'batter_total_bases', 'batter_home_runs')
                  AND p.complement_prob > 0.0
                  AND p.complement_prob < 1.0
                LIMIT 1
            ) synthetic_pair ON TRUE
            LEFT JOIN features.mlb_game_training_features gf
              ON gf.game_slug = r.game_slug
            LEFT JOIN raw.mlb_lineups lu
              ON lu.game_slug = r.game_slug
             AND lu.team_abbr = r.team_abbr
             AND lu.player_id = r.player_id
            LEFT JOIN raw.mlb_lineups lu_name
              ON lu_name.game_slug = r.game_slug
             AND lu_name.team_abbr = r.team_abbr
             AND lu_name.player_name_norm = r.player_name_norm
            LEFT JOIN raw.mlb_boxscore_player_stats bps
              ON bps.game_slug = r.game_slug
             AND bps.player_id = r.player_id
            LEFT JOIN features.mlb_batting_vs_hand bvh
              ON bvh.game_slug = r.game_slug
             AND bvh.team_abbr = r.team_abbr
             AND bvh.player_id = r.player_id
            LEFT JOIN features.mlb_batter_vs_rp bvr
              ON bvr.game_slug = r.game_slug
             AND bvr.batter_id = r.player_id
            LEFT JOIN raw.mlb_player_gamelogs actual_gl
              ON actual_gl.game_slug = r.game_slug
             AND actual_gl.player_id = r.player_id
            LEFT JOIN LATERAL (
                SELECT
                    AVG(pa_est)::float AS projected_pa,
                    COUNT(*)::int AS pa_games
                FROM (
                    SELECT GREATEST(COALESCE(gl.at_bats, 0) + COALESCE(gl.walks_batter, 0), 0) AS pa_est
                    FROM raw.mlb_player_gamelogs gl
                    JOIN raw.mlb_games g ON g.game_slug = gl.game_slug
                    WHERE gl.player_id = r.player_id
                      AND gl.team_abbr = r.team_abbr
                      AND g.status = 'final'
                      AND g.game_date_et < r.game_date_et
                      AND (COALESCE(gl.at_bats, 0) + COALESCE(gl.walks_batter, 0)) > 0
                    ORDER BY g.game_date_et DESC, gl.game_slug DESC
                    LIMIT 10
                ) recent_pa
            ) bat_opp ON r.stat IN ('batter_hits', 'batter_total_bases', 'batter_home_runs')
            LEFT JOIN LATERAL (
                SELECT
                    AVG(innings_pitched)::float AS projected_ip,
                    AVG(bf_est)::float AS projected_bf,
                    AVG(bf_est * 3.85)::float AS projected_pitch_count,
                    COUNT(*)::int AS pitcher_starts
                FROM (
                    SELECT
                        gl.innings_pitched,
                        GREATEST(
                            ROUND(COALESCE(gl.innings_pitched, 0) * 3)
                            + COALESCE(gl.hits_allowed, 0)
                            + COALESCE(gl.walks_allowed, 0),
                            1
                        ) AS bf_est
                    FROM raw.mlb_player_gamelogs gl
                    JOIN raw.mlb_games g ON g.game_slug = gl.game_slug
                    WHERE gl.player_id = r.player_id
                      AND g.status = 'final'
                      AND g.game_date_et < r.game_date_et
                      AND gl.is_starter IS TRUE
                      AND gl.innings_pitched >= 1.0
                    ORDER BY g.game_date_et DESC, gl.game_slug DESC
                    LIMIT 5
                ) recent_starts
            ) pit_opp ON r.stat = 'pitcher_strikeouts'
            WHERE {' AND '.join(filters)}
            ORDER BY r.game_date_et, r.game_slug, r.stat, r.player_id, r.run_id, r.id
            """,
            params,
        )
        return [dict(r) for r in cur.fetchall()]


def _example_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    p_over = _clean_float(row.get("model_prob_over"))
    line = _clean_float(row.get("market_line"))
    if p_over is None or line is None:
        return []
    over_price = _clean_float(row.get("resolved_over_price"))
    if over_price is None:
        over_price = _clean_float(row.get("over_price"))
    under_price = _clean_float(row.get("resolved_under_price"))
    if under_price is None:
        under_price = _clean_float(row.get("under_price"))
    nv_over, nv_under = no_vig_probs(over_price, under_price)
    actual = _clean_float(row.get("actual_value"))
    push = bool(actual is not None and abs(actual - line) <= 1e-9)
    over_hit = row.get("over_hit")
    if over_hit is None and actual is not None:
        over_hit = actual > line
    market = str(row.get("stat") or "")
    lb = row.get("line_bucket") or prop_line_bucket(market, line)
    if lb == "unknown":
        lb = prop_line_bucket(market, line)
    pred_count = _clean_float(row.get("pred_count"))
    selected_side = (row.get("side") or "").lower()
    out: list[dict[str, Any]] = []
    selected_side = (row.get("side") or "").lower()
    sides = (selected_side,) if selected_side in {"over", "under"} else ()
    for side in sides:
        price = over_price if side == "over" else under_price
        paired = under_price if side == "over" else over_price
        raw_mkt = american_to_prob(price)
        no_vig = nv_over if side == "over" else nv_under
        market_prob = no_vig if no_vig is not None else raw_mkt
        paired_price_source = row.get("paired_price_source_resolved") if paired is not None else None
        paired_bookmaker_key = row.get("paired_bookmaker_key_resolved") if paired is not None else None
        if paired_price_source == "synthetic_fanduel_over_only_complement":
            market_prob_source = "synthetic_fanduel_over_only"
        elif no_vig is not None and paired_price_source in {"cross_book_exact_line", "cross_book_exact_line_fallback"}:
            market_prob_source = "no_vig_cross_book_exact_line"
        elif no_vig is not None:
            market_prob_source = "no_vig_same_book"
        else:
            market_prob_source = "raw_implied"
        pair_quality = _pair_quality(paired_price_source, paired)
        same_book_pair_flag = 1.0 if pair_quality == "same_book" else 0.0
        cross_book_pair_flag = 1.0 if pair_quality == "cross_book" else 0.0
        synthetic_pair_flag = 1.0 if pair_quality == "synthetic" else 0.0
        true_pair_flag = 1.0 if pair_quality in {"same_book", "cross_book"} else 0.0
        clean_market_pair_flag = (
            1.0
            if true_pair_flag
            and market_prob_source not in {"raw_implied", "synthetic_fanduel_over_only"}
            else 0.0
        )
        p_side = p_over if side == "over" else 1.0 - p_over
        target_won = None
        if over_hit is not None and not push:
            target_won = bool(over_hit) if side == "over" else not bool(over_hit)
        count_edge = None
        if pred_count is not None:
            count_edge = pred_count - line if side == "over" else line - pred_count
        prob_edge = p_side - market_prob if market_prob is not None else None

        clv_valid = bool(row.get("clv_valid")) and selected_side == side
        closing_line = _clean_float(row.get("closing_line")) if clv_valid else None
        closing_price = _clean_float(row.get("closing_price")) if clv_valid else None
        closing_source_row_id = row.get("closing_source_row_id") if clv_valid else None
        closing_snapshot_id = row.get("closing_snapshot_id") if clv_valid else None
        closing_fetched_at_utc = row.get("closing_fetched_at_utc") if clv_valid else None
        clv_match_method = row.get("clv_match_method")
        clv_line = _clean_float(row.get("clv_line")) if clv_valid else None
        clv_p = _clean_float(row.get("clv_price")) if clv_valid else None
        clv_status = row.get("clv_status") or ("valid_movement" if clv_valid else "unknown")
        clv_unknown_reason = (
            None
            if clv_valid
            else row.get("clv_unknown_reason") or "not_regraded_with_snapshot_rules"
        )
        ev = ev_per_unit(p_side, price)
        out.append({
            "source": "replay",
            "run_id": row.get("run_id"),
            "replay_id": row.get("id"),
            "source_pred_id": row.get("source_pred_id"),
            "prediction_key": row.get("prediction_key"),
            "prop_offer_id": row.get("prop_offer_id"),
            "prop_offer_source_row_id": row.get("prop_offer_source_row_id"),
            "lock_snapshot_id": row.get("lock_snapshot_id"),
            "game_date_et": row.get("game_date_et"),
            "game_slug": row.get("game_slug"),
            "player_id": row.get("player_id"),
            "player_name": row.get("player_name"),
            "player_name_norm": row.get("player_name_norm"),
            "team_abbr": row.get("team_abbr"),
            "market": market,
            "side": side,
            "bookmaker_key": row.get("bookmaker_key"),
            "market_line": line,
            "market_price": price,
            "paired_price": paired,
            "paired_bookmaker_key": paired_bookmaker_key,
            "paired_price_source": paired_price_source,
            "pair_quality": pair_quality,
            "same_book_pair_flag": same_book_pair_flag,
            "cross_book_pair_flag": cross_book_pair_flag,
            "synthetic_pair_flag": synthetic_pair_flag,
            "clean_market_pair_flag": clean_market_pair_flag,
            "true_pair_flag": true_pair_flag,
            "minutes_to_first_pitch_at_lock": _clean_float(row.get("minutes_to_first_pitch_at_lock")),
            "lock_price_age_minutes": _clean_float(row.get("lock_price_age_minutes")),
            "raw_market_prob": raw_mkt,
            "no_vig_market_prob": no_vig,
            "market_prob_side": market_prob,
            "market_prob_source": market_prob_source,
            "price_bucket": price_bucket(price),
            "line_bucket": lb,
            "line_surface": prop_line_surface(market, side, line),
            "model_family": row.get("model_family") or "unknown",
            "edge_type": row.get("edge_type"),
            "pred_value": _clean_float(row.get("pred_value")),
            "pred_count": pred_count,
            "model_prob_over": p_over,
            "model_prob_side": p_side,
            "count_edge_side": count_edge,
            "prob_edge_vs_market": prob_edge,
            "confirmed_batting_order": _clean_float(row.get("confirmed_batting_order")),
            "confirmed_lineup_source": row.get("confirmed_lineup_source"),
            "projected_pa": _clean_float(row.get("projected_pa")),
            "pa_games": row.get("pa_games"),
            "projected_ip": _clean_float(row.get("projected_ip")),
            "projected_bf": _clean_float(row.get("projected_bf")),
            "projected_pitch_count": _clean_float(row.get("projected_pitch_count")),
            "pitcher_starts": row.get("pitcher_starts"),
            "is_home": _clean_float(row.get("is_home")),
            "opponent_abbr": row.get("opponent_abbr"),
            "opp_sp_id": row.get("opp_sp_id"),
            "opp_sp_hand": row.get("opp_sp_hand"),
            "opp_sp_hand_l": _clean_float(row.get("opp_sp_hand_l")),
            "opp_sp_k_pct_10": _clean_float(row.get("opp_sp_k_pct_10")),
            "opp_sp_bb_pct": _clean_float(row.get("opp_sp_bb_pct")),
            "opp_sp_xwoba": _clean_float(row.get("opp_sp_xwoba")),
            "opp_sp_hard_hit_pct": _clean_float(row.get("opp_sp_hard_hit_pct")),
            "opp_sp_whiff_pct": _clean_float(row.get("opp_sp_whiff_pct")),
            "opp_bp_era_10": _clean_float(row.get("opp_bp_era_10")),
            "opp_bp_whip_10": _clean_float(row.get("opp_bp_whip_10")),
            "opp_bp_k9_10": _clean_float(row.get("opp_bp_k9_10")),
            "opp_bp_ip_last_3": _clean_float(row.get("opp_bp_ip_last_3")),
            "opp_bp_ip_last_7": _clean_float(row.get("opp_bp_ip_last_7")),
            "opp_team_k_pct_10": _clean_float(row.get("opp_team_k_pct_10")),
            "opp_team_avg_10": _clean_float(row.get("opp_team_avg_10")),
            "opp_team_obp_10": _clean_float(row.get("opp_team_obp_10")),
            "opp_team_slg_10": _clean_float(row.get("opp_team_slg_10")),
            "batter_vs_hand_hits_avg_10": _clean_float(row.get("batter_vs_hand_hits_avg_10")),
            "batter_vs_hand_tb_avg_10": _clean_float(row.get("batter_vs_hand_tb_avg_10")),
            "batter_vs_hand_hr_avg_10": _clean_float(row.get("batter_vs_hand_hr_avg_10")),
            "batter_vs_hand_iso_avg_10": _clean_float(row.get("batter_vs_hand_iso_avg_10")),
            "batter_vs_hand_k_rate_10": _clean_float(row.get("batter_vs_hand_k_rate_10")),
            "batter_vs_hand_games_10": _clean_float(row.get("batter_vs_hand_games_10")),
            "batter_vs_rp_ba_30": _clean_float(row.get("batter_vs_rp_ba_30")),
            "batter_vs_rp_slg_30": _clean_float(row.get("batter_vs_rp_slg_30")),
            "batter_vs_rp_hr_rate_30": _clean_float(row.get("batter_vs_rp_hr_rate_30")),
            "batter_vs_rp_k_rate_30": _clean_float(row.get("batter_vs_rp_k_rate_30")),
            "pinch_hit_risk": _clean_float(row.get("pinch_hit_risk")),
            "team_implied_runs": _clean_float(row.get("team_implied_runs")),
            "opponent_implied_runs": _clean_float(row.get("opponent_implied_runs")),
            "game_total_line": _clean_float(row.get("game_total_line")),
            "actual_pa": _clean_float(row.get("actual_pa")),
            "actual_bf": _clean_float(row.get("actual_bf")),
            "actual_ip": _clean_float(row.get("actual_ip")),
            "actual_pitch_count_proxy": _clean_float(row.get("actual_pitch_count_proxy")),
            "low_pa_flag": _clean_float(row.get("low_pa_flag")),
            "ev": ev,
            "kelly_fraction": _kelly_from_price(p_side, price),
            "actual_value": actual,
            "over_hit": over_hit,
            "won": target_won,
            "push": push if actual is not None else None,
            "profit_units": _profit_units(target_won, push, price),
            "closing_line": closing_line,
            "closing_price": closing_price,
            "clv_line": clv_line,
            "clv_price": clv_p,
            "closing_source_row_id": closing_source_row_id,
            "closing_snapshot_id": closing_snapshot_id,
            "closing_fetched_at_utc": closing_fetched_at_utc,
            "clv_match_method": clv_match_method,
            "clv_valid": clv_valid,
            "clv_status": clv_status,
            "clv_unknown_reason": clv_unknown_reason,
            "beat_clv_line": None if clv_line is None else clv_line > 0,
            "beat_clv_price": None if clv_p is None else clv_p > 0,
            "result_status": "graded" if actual is not None else "pending",
            "source_created_at": row.get("source_created_at"),
        })
    return out


_INSERT_SQL = """
INSERT INTO features.mlb_prop_market_training_examples (
    source, run_id, replay_id, source_pred_id, prediction_key,
    prop_offer_id, prop_offer_source_row_id, lock_snapshot_id, game_date_et, game_slug,
    player_id, player_name, player_name_norm, team_abbr, market, side,
    bookmaker_key, market_line, market_price, paired_price, paired_bookmaker_key,
    paired_price_source, pair_quality, same_book_pair_flag, cross_book_pair_flag,
    synthetic_pair_flag, clean_market_pair_flag, true_pair_flag,
    minutes_to_first_pitch_at_lock, lock_price_age_minutes,
    raw_market_prob, no_vig_market_prob, market_prob_side,
    market_prob_source, price_bucket, line_bucket, line_surface,
    model_family, edge_type, pred_value, pred_count, model_prob_over,
    model_prob_side, count_edge_side, prob_edge_vs_market, confirmed_batting_order,
    confirmed_lineup_source, projected_pa, pa_games, projected_ip, projected_bf,
    projected_pitch_count, pitcher_starts, is_home, opponent_abbr, opp_sp_id,
    opp_sp_hand, opp_sp_hand_l, opp_sp_k_pct_10, opp_sp_bb_pct, opp_sp_xwoba,
    opp_sp_hard_hit_pct, opp_sp_whiff_pct, opp_bp_era_10, opp_bp_whip_10,
    opp_bp_k9_10, opp_bp_ip_last_3, opp_bp_ip_last_7, opp_team_k_pct_10,
    opp_team_avg_10, opp_team_obp_10, opp_team_slg_10,
    batter_vs_hand_hits_avg_10, batter_vs_hand_tb_avg_10,
    batter_vs_hand_hr_avg_10, batter_vs_hand_iso_avg_10,
    batter_vs_hand_k_rate_10, batter_vs_hand_games_10,
    batter_vs_rp_ba_30, batter_vs_rp_slg_30,
    batter_vs_rp_hr_rate_30, batter_vs_rp_k_rate_30,
    pinch_hit_risk, team_implied_runs, opponent_implied_runs, game_total_line,
    actual_pa, actual_bf, actual_ip, actual_pitch_count_proxy, low_pa_flag,
    ev, kelly_fraction,
    actual_value, over_hit, won, push, profit_units, closing_line,
    closing_price, clv_line, clv_price, closing_source_row_id, closing_snapshot_id,
    closing_fetched_at_utc, clv_match_method, clv_valid, clv_status,
    clv_unknown_reason, beat_clv_line, beat_clv_price,
    result_status, source_created_at, example_updated_at
) VALUES (
    %(source)s, %(run_id)s, %(replay_id)s, %(source_pred_id)s, %(prediction_key)s,
    %(prop_offer_id)s, %(prop_offer_source_row_id)s, %(lock_snapshot_id)s, %(game_date_et)s, %(game_slug)s,
    %(player_id)s, %(player_name)s, %(player_name_norm)s, %(team_abbr)s, %(market)s, %(side)s,
    %(bookmaker_key)s, %(market_line)s, %(market_price)s, %(paired_price)s, %(paired_bookmaker_key)s,
    %(paired_price_source)s, %(pair_quality)s, %(same_book_pair_flag)s, %(cross_book_pair_flag)s,
    %(synthetic_pair_flag)s, %(clean_market_pair_flag)s, %(true_pair_flag)s,
    %(minutes_to_first_pitch_at_lock)s, %(lock_price_age_minutes)s,
    %(raw_market_prob)s, %(no_vig_market_prob)s, %(market_prob_side)s,
    %(market_prob_source)s, %(price_bucket)s, %(line_bucket)s, %(line_surface)s,
    %(model_family)s, %(edge_type)s, %(pred_value)s, %(pred_count)s, %(model_prob_over)s,
    %(model_prob_side)s, %(count_edge_side)s, %(prob_edge_vs_market)s, %(confirmed_batting_order)s,
    %(confirmed_lineup_source)s, %(projected_pa)s, %(pa_games)s, %(projected_ip)s, %(projected_bf)s,
    %(projected_pitch_count)s, %(pitcher_starts)s, %(is_home)s, %(opponent_abbr)s, %(opp_sp_id)s,
    %(opp_sp_hand)s, %(opp_sp_hand_l)s, %(opp_sp_k_pct_10)s, %(opp_sp_bb_pct)s, %(opp_sp_xwoba)s,
    %(opp_sp_hard_hit_pct)s, %(opp_sp_whiff_pct)s, %(opp_bp_era_10)s, %(opp_bp_whip_10)s,
    %(opp_bp_k9_10)s, %(opp_bp_ip_last_3)s, %(opp_bp_ip_last_7)s, %(opp_team_k_pct_10)s,
    %(opp_team_avg_10)s, %(opp_team_obp_10)s, %(opp_team_slg_10)s,
    %(batter_vs_hand_hits_avg_10)s, %(batter_vs_hand_tb_avg_10)s,
    %(batter_vs_hand_hr_avg_10)s, %(batter_vs_hand_iso_avg_10)s,
    %(batter_vs_hand_k_rate_10)s, %(batter_vs_hand_games_10)s,
    %(batter_vs_rp_ba_30)s, %(batter_vs_rp_slg_30)s,
    %(batter_vs_rp_hr_rate_30)s, %(batter_vs_rp_k_rate_30)s,
    %(pinch_hit_risk)s, %(team_implied_runs)s, %(opponent_implied_runs)s, %(game_total_line)s,
    %(actual_pa)s, %(actual_bf)s, %(actual_ip)s, %(actual_pitch_count_proxy)s, %(low_pa_flag)s,
    %(ev)s, %(kelly_fraction)s,
    %(actual_value)s, %(over_hit)s, %(won)s, %(push)s, %(profit_units)s, %(closing_line)s,
    %(closing_price)s, %(clv_line)s, %(clv_price)s, %(closing_source_row_id)s,
    %(closing_snapshot_id)s, %(closing_fetched_at_utc)s, %(clv_match_method)s,
    %(clv_valid)s, %(clv_status)s, %(clv_unknown_reason)s, %(beat_clv_line)s, %(beat_clv_price)s,
    %(result_status)s, %(source_created_at)s, now()
)
ON CONFLICT (run_id, replay_id, side) DO UPDATE SET
    bookmaker_key = EXCLUDED.bookmaker_key,
    market_price = EXCLUDED.market_price,
    paired_price = EXCLUDED.paired_price,
    paired_bookmaker_key = EXCLUDED.paired_bookmaker_key,
    paired_price_source = EXCLUDED.paired_price_source,
    pair_quality = EXCLUDED.pair_quality,
    same_book_pair_flag = EXCLUDED.same_book_pair_flag,
    cross_book_pair_flag = EXCLUDED.cross_book_pair_flag,
    synthetic_pair_flag = EXCLUDED.synthetic_pair_flag,
    clean_market_pair_flag = EXCLUDED.clean_market_pair_flag,
    true_pair_flag = EXCLUDED.true_pair_flag,
    minutes_to_first_pitch_at_lock = EXCLUDED.minutes_to_first_pitch_at_lock,
    lock_price_age_minutes = EXCLUDED.lock_price_age_minutes,
    raw_market_prob = EXCLUDED.raw_market_prob,
    no_vig_market_prob = EXCLUDED.no_vig_market_prob,
    market_prob_side = EXCLUDED.market_prob_side,
    market_prob_source = EXCLUDED.market_prob_source,
    price_bucket = EXCLUDED.price_bucket,
    line_bucket = EXCLUDED.line_bucket,
    line_surface = EXCLUDED.line_surface,
    model_family = EXCLUDED.model_family,
    edge_type = EXCLUDED.edge_type,
    model_prob_side = EXCLUDED.model_prob_side,
    count_edge_side = EXCLUDED.count_edge_side,
    prob_edge_vs_market = EXCLUDED.prob_edge_vs_market,
    confirmed_batting_order = EXCLUDED.confirmed_batting_order,
    confirmed_lineup_source = EXCLUDED.confirmed_lineup_source,
    projected_pa = EXCLUDED.projected_pa,
    pa_games = EXCLUDED.pa_games,
    projected_ip = EXCLUDED.projected_ip,
    projected_bf = EXCLUDED.projected_bf,
    projected_pitch_count = EXCLUDED.projected_pitch_count,
    pitcher_starts = EXCLUDED.pitcher_starts,
    is_home = EXCLUDED.is_home,
    opponent_abbr = EXCLUDED.opponent_abbr,
    opp_sp_id = EXCLUDED.opp_sp_id,
    opp_sp_hand = EXCLUDED.opp_sp_hand,
    opp_sp_hand_l = EXCLUDED.opp_sp_hand_l,
    opp_sp_k_pct_10 = EXCLUDED.opp_sp_k_pct_10,
    opp_sp_bb_pct = EXCLUDED.opp_sp_bb_pct,
    opp_sp_xwoba = EXCLUDED.opp_sp_xwoba,
    opp_sp_hard_hit_pct = EXCLUDED.opp_sp_hard_hit_pct,
    opp_sp_whiff_pct = EXCLUDED.opp_sp_whiff_pct,
    opp_bp_era_10 = EXCLUDED.opp_bp_era_10,
    opp_bp_whip_10 = EXCLUDED.opp_bp_whip_10,
    opp_bp_k9_10 = EXCLUDED.opp_bp_k9_10,
    opp_bp_ip_last_3 = EXCLUDED.opp_bp_ip_last_3,
    opp_bp_ip_last_7 = EXCLUDED.opp_bp_ip_last_7,
    opp_team_k_pct_10 = EXCLUDED.opp_team_k_pct_10,
    opp_team_avg_10 = EXCLUDED.opp_team_avg_10,
    opp_team_obp_10 = EXCLUDED.opp_team_obp_10,
    opp_team_slg_10 = EXCLUDED.opp_team_slg_10,
    batter_vs_hand_hits_avg_10 = EXCLUDED.batter_vs_hand_hits_avg_10,
    batter_vs_hand_tb_avg_10 = EXCLUDED.batter_vs_hand_tb_avg_10,
    batter_vs_hand_hr_avg_10 = EXCLUDED.batter_vs_hand_hr_avg_10,
    batter_vs_hand_iso_avg_10 = EXCLUDED.batter_vs_hand_iso_avg_10,
    batter_vs_hand_k_rate_10 = EXCLUDED.batter_vs_hand_k_rate_10,
    batter_vs_hand_games_10 = EXCLUDED.batter_vs_hand_games_10,
    batter_vs_rp_ba_30 = EXCLUDED.batter_vs_rp_ba_30,
    batter_vs_rp_slg_30 = EXCLUDED.batter_vs_rp_slg_30,
    batter_vs_rp_hr_rate_30 = EXCLUDED.batter_vs_rp_hr_rate_30,
    batter_vs_rp_k_rate_30 = EXCLUDED.batter_vs_rp_k_rate_30,
    pinch_hit_risk = EXCLUDED.pinch_hit_risk,
    team_implied_runs = EXCLUDED.team_implied_runs,
    opponent_implied_runs = EXCLUDED.opponent_implied_runs,
    game_total_line = EXCLUDED.game_total_line,
    actual_pa = EXCLUDED.actual_pa,
    actual_bf = EXCLUDED.actual_bf,
    actual_ip = EXCLUDED.actual_ip,
    actual_pitch_count_proxy = EXCLUDED.actual_pitch_count_proxy,
    low_pa_flag = EXCLUDED.low_pa_flag,
    ev = EXCLUDED.ev,
    kelly_fraction = EXCLUDED.kelly_fraction,
    actual_value = EXCLUDED.actual_value,
    over_hit = EXCLUDED.over_hit,
    won = EXCLUDED.won,
    push = EXCLUDED.push,
    profit_units = EXCLUDED.profit_units,
    closing_line = EXCLUDED.closing_line,
    closing_price = EXCLUDED.closing_price,
    clv_line = EXCLUDED.clv_line,
    clv_price = EXCLUDED.clv_price,
    closing_source_row_id = EXCLUDED.closing_source_row_id,
    closing_snapshot_id = EXCLUDED.closing_snapshot_id,
    closing_fetched_at_utc = EXCLUDED.closing_fetched_at_utc,
    clv_match_method = EXCLUDED.clv_match_method,
    clv_valid = EXCLUDED.clv_valid,
    clv_status = EXCLUDED.clv_status,
    clv_unknown_reason = EXCLUDED.clv_unknown_reason,
    beat_clv_line = EXCLUDED.beat_clv_line,
    beat_clv_price = EXCLUDED.beat_clv_price,
    result_status = EXCLUDED.result_status,
    example_updated_at = now()
"""


def _delete_existing(conn, cfg: PropMarketTrainingConfig) -> int:
    filters = ["TRUE"]
    params: list[Any] = []
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)
    date_from = cfg.date_from or cutoff
    filters.append("game_date_et >= %s")
    params.append(date_from)
    if cfg.date_to is not None:
        filters.append("game_date_et <= %s")
        params.append(cfg.date_to)
    if cfg.run_ids:
        filters.append("run_id = ANY(%s)")
        params.append(list(cfg.run_ids))
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM features.mlb_prop_market_training_examples WHERE {' AND '.join(filters)}",
            params,
        )
        deleted = cur.rowcount
    conn.commit()
    return deleted


def refresh_prop_market_training_examples(cfg: PropMarketTrainingConfig) -> dict[str, int]:
    with psycopg2.connect(cfg.pg_dsn) as conn:
        if cfg.ensure_schema:
            prepare_prop_market_training_dependencies(conn)
        else:
            missing = verify_prop_market_training_dependencies(conn)
            if missing:
                raise RuntimeError(
                    "Prop market training dependencies are not ready: "
                    + ", ".join(missing)
                    + ". Run `python -m mlb_pipeline.modeling.build_prop_market_training_table --ensure-schema` "
                    + "from a non-live maintenance window."
                )
        deleted = _delete_existing(conn, cfg) if cfg.replace else 0
        if not _table_exists(conn, "bets", "mlb_prop_prediction_replay"):
            return {"deleted": deleted, "replay_rows": 0, "examples": 0}
        replay_rows = _load_replay_rows(conn, cfg)
        examples: list[dict[str, Any]] = []
        for row in replay_rows:
            examples.extend(_example_rows(row))
        if examples:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, _INSERT_SQL, examples, page_size=500)
            conn.commit()
    return {"deleted": deleted, "replay_rows": len(replay_rows), "examples": len(examples)}
