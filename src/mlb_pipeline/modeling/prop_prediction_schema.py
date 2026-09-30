"""Schema SQL and runtime checks for MLB prop predictions."""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("mlb_pipeline.modeling.prop_prediction_schema")
_SQL_DIR = Path(__file__).resolve().parents[3] / "sql"

_ENSURE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bets.mlb_prop_predictions (
    id               SERIAL PRIMARY KEY,
    game_date_et     DATE        NOT NULL,
    game_slug        TEXT        NOT NULL,
    player_id        BIGINT      NOT NULL,
    player_name      TEXT,
    team_abbr        TEXT,
    stat             TEXT        NOT NULL,
    prediction_key   TEXT,
    prop_offer_id    BIGINT,
    prop_offer_source_row_id INTEGER,
    pred_value       NUMERIC,
    book_line        NUMERIC,
    edge             NUMERIC,
    kelly_fraction   NUMERIC,
    actual_value     NUMERIC,
    over_hit         BOOLEAN,
    closing_line     NUMERIC,
    closing_price    NUMERIC,
    clv_line         NUMERIC,
    clv_price        NUMERIC,
    beat_clv_line    BOOLEAN,
    beat_clv_price   BOOLEAN,
    opportunity_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    run_id           TEXT,
    is_active        BOOLEAN NOT NULL DEFAULT TRUE,
    superseded_at    TIMESTAMPTZ,
    stale_reason     TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (game_date_et, game_slug, player_id, stat)
);

ALTER TABLE bets.mlb_prop_predictions
    DROP CONSTRAINT IF EXISTS mlb_prop_predictions_game_date_et_game_slug_player_id_stat_key,
    DROP CONSTRAINT IF EXISTS mlb_prop_predictions_game_slug_player_id_stat_key;

ALTER TABLE bets.mlb_prop_predictions
    ADD COLUMN IF NOT EXISTS prediction_key   TEXT,
    ADD COLUMN IF NOT EXISTS prop_offer_id    BIGINT,
    ADD COLUMN IF NOT EXISTS prop_offer_source_row_id INTEGER,
    ADD COLUMN IF NOT EXISTS closing_line     NUMERIC,
    ADD COLUMN IF NOT EXISTS closing_price    NUMERIC,
    ADD COLUMN IF NOT EXISTS clv_line         NUMERIC,
    ADD COLUMN IF NOT EXISTS clv_price        NUMERIC,
    ADD COLUMN IF NOT EXISTS beat_clv_line    BOOLEAN,
    ADD COLUMN IF NOT EXISTS beat_clv_price   BOOLEAN,
    ADD COLUMN IF NOT EXISTS pred_count      NUMERIC,
    ADD COLUMN IF NOT EXISTS pred_prob_over  NUMERIC,
    ADD COLUMN IF NOT EXISTS edge_type       TEXT,
    ADD COLUMN IF NOT EXISTS model_family    TEXT,
    ADD COLUMN IF NOT EXISTS bet_side        TEXT,
    ADD COLUMN IF NOT EXISTS line_bucket     TEXT,
    ADD COLUMN IF NOT EXISTS over_price      NUMERIC,
    ADD COLUMN IF NOT EXISTS under_price     NUMERIC,
    ADD COLUMN IF NOT EXISTS bet_price       NUMERIC,
    ADD COLUMN IF NOT EXISTS minimum_acceptable_price NUMERIC,
    ADD COLUMN IF NOT EXISTS breakeven_prob  NUMERIC,
    ADD COLUMN IF NOT EXISTS ev              NUMERIC,
    ADD COLUMN IF NOT EXISTS bookmaker_key   TEXT,
    ADD COLUMN IF NOT EXISTS bet_link        TEXT,
    ADD COLUMN IF NOT EXISTS bankroll_tier   TEXT,
    ADD COLUMN IF NOT EXISTS bankroll_candidate BOOLEAN,
    ADD COLUMN IF NOT EXISTS bankroll_reasons TEXT,
    ADD COLUMN IF NOT EXISTS opportunity_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS stake_pct       NUMERIC,
    ADD COLUMN IF NOT EXISTS stake_usd       NUMERIC,
    ADD COLUMN IF NOT EXISTS lock_snapshot_id BIGINT,
    ADD COLUMN IF NOT EXISTS locked_at_utc   TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS closing_snapshot_id BIGINT,
    ADD COLUMN IF NOT EXISTS closing_source_row_id BIGINT,
    ADD COLUMN IF NOT EXISTS closing_fetched_at_utc TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS clv_match_method TEXT,
    ADD COLUMN IF NOT EXISTS clv_valid BOOLEAN,
    ADD COLUMN IF NOT EXISTS clv_status TEXT,
    ADD COLUMN IF NOT EXISTS clv_unknown_reason TEXT,
    ADD COLUMN IF NOT EXISTS run_id          TEXT,
    ADD COLUMN IF NOT EXISTS is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS superseded_at   TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS stale_reason    TEXT,
    ADD COLUMN IF NOT EXISTS updated_at      TIMESTAMPTZ DEFAULT NOW();

CREATE UNIQUE INDEX IF NOT EXISTS uq_mlb_prop_predictions_prediction_key
    ON bets.mlb_prop_predictions (prediction_key);
CREATE INDEX IF NOT EXISTS idx_mlb_prop_predictions_offer
    ON bets.mlb_prop_predictions (prop_offer_id);
CREATE INDEX IF NOT EXISTS idx_mlb_prop_predictions_date_market
    ON bets.mlb_prop_predictions (game_date_et, stat, bet_side);
"""

_UPSERT_SQL = """
INSERT INTO bets.mlb_prop_predictions
    (game_date_et, game_slug, player_id, player_name, team_abbr, stat,
     prediction_key, prop_offer_id, prop_offer_source_row_id,
     pred_value, pred_count, pred_prob_over, book_line, edge, edge_type,
     model_family, bet_side, line_bucket, over_price, under_price, bet_price,
     minimum_acceptable_price, breakeven_prob, ev, bookmaker_key, bet_link, kelly_fraction,
     bankroll_tier, bankroll_candidate, bankroll_reasons, opportunity_context, stake_pct, stake_usd,
     run_id, is_active, stale_reason)
VALUES
    (%(game_date_et)s, %(game_slug)s, %(player_id)s, %(player_name)s, %(team_abbr)s,
     %(stat)s, %(prediction_key)s, %(prop_offer_id)s, %(prop_offer_source_row_id)s,
     %(pred_value)s, %(pred_count)s, %(pred_prob_over)s, %(book_line)s,
     %(edge)s, %(edge_type)s, %(model_family)s, %(bet_side)s, %(line_bucket)s,
     %(over_price)s, %(under_price)s, %(bet_price)s, %(minimum_acceptable_price)s, %(breakeven_prob)s,
     %(ev)s, %(bookmaker_key)s, %(bet_link)s, %(kelly_fraction)s,
     %(bankroll_tier)s, %(bankroll_candidate)s, %(bankroll_reasons)s, %(opportunity_context)s, %(stake_pct)s, %(stake_usd)s,
     %(run_id)s, TRUE, NULL)
ON CONFLICT (prediction_key) DO UPDATE SET
    player_name     = EXCLUDED.player_name,
    team_abbr       = EXCLUDED.team_abbr,
    prop_offer_id   = EXCLUDED.prop_offer_id,
    prop_offer_source_row_id = EXCLUDED.prop_offer_source_row_id,
    pred_value      = EXCLUDED.pred_value,
    pred_count      = EXCLUDED.pred_count,
    pred_prob_over  = EXCLUDED.pred_prob_over,
    book_line       = EXCLUDED.book_line,
    edge            = EXCLUDED.edge,
    edge_type       = EXCLUDED.edge_type,
    model_family    = EXCLUDED.model_family,
    bet_side        = EXCLUDED.bet_side,
    line_bucket     = EXCLUDED.line_bucket,
    over_price      = EXCLUDED.over_price,
    under_price     = EXCLUDED.under_price,
    bet_price       = EXCLUDED.bet_price,
    minimum_acceptable_price = EXCLUDED.minimum_acceptable_price,
    breakeven_prob  = EXCLUDED.breakeven_prob,
    ev              = EXCLUDED.ev,
    bookmaker_key   = EXCLUDED.bookmaker_key,
    bet_link        = EXCLUDED.bet_link,
    kelly_fraction  = EXCLUDED.kelly_fraction,
    bankroll_tier   = EXCLUDED.bankroll_tier,
    bankroll_candidate = EXCLUDED.bankroll_candidate,
    bankroll_reasons = EXCLUDED.bankroll_reasons,
    opportunity_context = EXCLUDED.opportunity_context,
    stake_pct       = EXCLUDED.stake_pct,
    stake_usd       = EXCLUDED.stake_usd,
    run_id          = EXCLUDED.run_id,
    is_active       = TRUE,
    stale_reason    = NULL,
    superseded_at   = NULL,
    lock_snapshot_id = NULL,
    locked_at_utc   = NULL,
    actual_value    = NULL,
    over_hit        = NULL,
    closing_line    = NULL,
    closing_price   = NULL,
    clv_line        = NULL,
    clv_price       = NULL,
    beat_clv_line   = NULL,
    beat_clv_price  = NULL,
    closing_source_row_id = NULL,
    closing_snapshot_id = NULL,
    closing_fetched_at_utc = NULL,
    clv_match_method = NULL,
    clv_valid       = NULL,
    clv_status      = NULL,
    clv_unknown_reason = NULL,
    updated_at      = NOW()
"""


_PROP_PREDICTION_RUNTIME_COLUMNS = frozenset({
    "game_date_et", "game_slug", "player_id", "player_name", "team_abbr", "stat",
    "prediction_key", "prop_offer_id", "prop_offer_source_row_id",
    "pred_value", "pred_count", "pred_prob_over", "book_line", "edge", "edge_type",
    "model_family", "bet_side", "line_bucket", "over_price", "under_price",
    "bet_price", "minimum_acceptable_price", "breakeven_prob", "ev",
    "bookmaker_key", "bet_link", "kelly_fraction", "bankroll_tier",
    "bankroll_candidate", "bankroll_reasons", "opportunity_context", "stake_pct", "stake_usd",
    "run_id", "is_active", "stale_reason", "superseded_at", "updated_at",
    "lock_snapshot_id", "locked_at_utc", "closing_snapshot_id",
    "closing_source_row_id", "closing_fetched_at_utc", "clv_match_method",
    "clv_valid", "clv_status", "clv_unknown_reason",
})


def _table_columns(conn, schema: str, table: str) -> set[str]:
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
        return {str(row[0]) for row in cur.fetchall()}


def _verify_prediction_runtime_dependencies(conn) -> list[str]:
    missing: list[str] = []
    if not _regclass_exists(conn, "bets.mlb_prop_predictions"):
        missing.append("bets.mlb_prop_predictions")
    else:
        missing_cols = sorted(
            _PROP_PREDICTION_RUNTIME_COLUMNS
            - _table_columns(conn, "bets", "mlb_prop_predictions")
        )
        if missing_cols:
            missing.append(
                "bets.mlb_prop_predictions columns: "
                + ", ".join(missing_cols[:12])
                + ("..." if len(missing_cols) > 12 else "")
            )
        if not _regclass_exists(conn, "bets.uq_mlb_prop_predictions_prediction_key"):
            missing.append("bets.uq_mlb_prop_predictions_prediction_key")

    if not _regclass_exists(conn, "features.mlb_lineup_quality_mat"):
        missing.append("features.mlb_lineup_quality_mat")
    return missing


def _ensure_schema(conn, *, allow_ddl: bool = True) -> None:
    if allow_ddl:
        with conn.cursor() as cur:
            cur.execute(_ENSURE_TABLE_SQL)
        conn.commit()
        _ensure_lineup_quality_dependency(conn)
        return

    missing = _verify_prediction_runtime_dependencies(conn)
    if missing:
        raise RuntimeError(
            "Prediction runtime dependencies are not ready: "
            + "; ".join(missing)
            + ". Run a maintenance/bootstrap job with --allow-schema-ddl before live prediction."
        )


def _regclass_exists(conn, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (name,))
        return bool(cur.fetchone()[0])


def _ensure_player_batting_rolling_mat(conn) -> None:
    if _regclass_exists(conn, "features.mlb_player_batting_rolling_mat"):
        return
    if not _regclass_exists(conn, "features.mlb_player_batting_rolling"):
        log.warning("features.mlb_player_batting_rolling missing; lineup quality will use empty fallback")
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_player_batting_rolling_mat AS
            SELECT * FROM features.mlb_player_batting_rolling
            WITH DATA;

            CREATE UNIQUE INDEX IF NOT EXISTS idx_mlb_batting_player_mat_pk
                ON features.mlb_player_batting_rolling_mat (game_slug, player_id);
            CREATE INDEX IF NOT EXISTS idx_mlb_batting_player_mat_player_date
                ON features.mlb_player_batting_rolling_mat (player_id, game_date_et DESC, game_slug DESC);
            """
        )
    conn.commit()
    log.info("Created compatibility matview features.mlb_player_batting_rolling_mat")


def _create_empty_lineup_quality_mat(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE SCHEMA IF NOT EXISTS features;
            DROP MATERIALIZED VIEW IF EXISTS features.mlb_lineup_quality_mat;
            CREATE MATERIALIZED VIEW features.mlb_lineup_quality_mat AS
            SELECT
                NULL::text AS game_slug,
                NULL::text AS team_abbr,
                NULL::boolean AS is_home,
                NULL::numeric AS lineup_avg_avg_10,
                NULL::numeric AS lineup_slg_avg_10,
                NULL::numeric AS lineup_iso_avg_10,
                NULL::numeric AS top4_slg_avg_10,
                NULL::double precision AS lineup_data_completeness,
                NULL::double precision AS lineup_xwoba_avg,
                NULL::double precision AS lineup_xslg_avg,
                NULL::double precision AS lineup_barrel_avg,
                NULL::double precision AS lineup_hard_hit_avg,
                NULL::numeric AS lineup_k_pct_std,
                NULL::numeric AS lineup_k_pct_cv,
                NULL::numeric AS pct_lhb
            WHERE false;
            CREATE UNIQUE INDEX IF NOT EXISTS mlb_lineup_quality_mat_pk
                ON features.mlb_lineup_quality_mat (game_slug, team_abbr);
            """
        )
    conn.commit()
    log.warning("Created empty fallback features.mlb_lineup_quality_mat; lineup quality fields will be median-imputed")


def _ensure_lineup_quality_dependency(conn) -> None:
    if _regclass_exists(conn, "features.mlb_lineup_quality_mat"):
        return
    try:
        _ensure_player_batting_rolling_mat(conn)
        lineup_sql = _SQL_DIR / "MLB011_mlb_lineup_quality.sql"
        mat_sql = _SQL_DIR / "MLB011b_mlb_lineup_quality_mat.sql"
        if not lineup_sql.exists() or not mat_sql.exists():
            raise FileNotFoundError("MLB011 lineup quality SQL files are missing")
        with conn.cursor() as cur:
            cur.execute(lineup_sql.read_text(encoding="utf-8"))
            cur.execute(mat_sql.read_text(encoding="utf-8"))
        conn.commit()
        log.info("Created features.mlb_lineup_quality_mat for player prop prediction")
    except Exception:
        conn.rollback()
        log.exception("Could not create full lineup quality matview; using empty fallback")
        _create_empty_lineup_quality_mat(conn)


