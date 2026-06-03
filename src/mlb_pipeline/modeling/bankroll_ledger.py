"""Locked bankroll ledger for MLB real-money candidates.

Prediction tables are allowed to be overwritten by reruns. This ledger is not:
the first observed bankroll candidate for a deterministic pick key is inserted
once and later graded in place.
"""
from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from datetime import datetime, timezone
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg2.extras

log = logging.getLogger("mlb_pipeline.modeling.bankroll_ledger")

_ET = ZoneInfo("America/New_York")

_STAT_COL = {
    "pitcher_strikeouts": "strikeouts_pitcher",
    "batter_hits": "hits",
    "batter_home_runs": "home_runs",
    "batter_total_bases": "total_bases",
    "batter_walks": "walks_batter",
}


def _normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().replace(".", "").split())


def _clean_float(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_int(value) -> int | None:
    val = _clean_float(value)
    return int(val) if val is not None else None


def _json(data: dict[str, Any] | None) -> psycopg2.extras.Json:
    return psycopg2.extras.Json(data or {}, dumps=lambda obj: json.dumps(obj, default=str))


def _pick_key(*parts: Any) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def _american_profit_mult(price) -> float | None:
    price = _clean_float(price)
    if price is None or price == 0:
        return None
    if price > 0:
        return price / 100.0
    return 100.0 / abs(price)


def _profit_units(won: bool | None, push: bool, price) -> float | None:
    if push:
        return 0.0
    if won is None:
        return None
    mult = _american_profit_mult(price)
    if mult is None:
        return None
    return mult if won else -1.0


def _ev_per_unit(p_win, price) -> float | None:
    p = _clean_float(p_win)
    mult = _american_profit_mult(price)
    if p is None or mult is None:
        return None
    return round(p * mult - (1.0 - p), 4)


def _american_to_prob(price: float) -> float:
    return 100.0 / (price + 100.0) if price >= 100 else abs(price) / (abs(price) + 100.0)


def _price_clv(entry_price, closing_price) -> float | None:
    entry = _clean_float(entry_price)
    close = _clean_float(closing_price)
    if entry is None or close is None:
        return None
    return round((_american_to_prob(close) - _american_to_prob(entry)) * 100.0, 2)


def _run_line_side_label(home_market_line, side: str | None) -> str | None:
    line = _clean_float(home_market_line)
    if line is None or side not in {"home", "away"}:
        return None
    side_line = line if side == "home" else -line
    return f"{side_line:+.1f}"


def _cfg_thresholds(cfg) -> dict[str, Any]:
    if cfg is None:
        return {}
    keys = [
        "min_edge_run_line",
        "min_edge_total",
        "max_run_line_lay_price",
        "max_away_dog_run_line_lay_price",
        "max_total_lay_price",
        "threshold_strikeouts",
        "threshold_strikeouts_over",
        "threshold_strikeouts_under",
        "threshold_hits",
        "threshold_total_bases",
        "threshold_total_bases_over",
        "threshold_total_bases_under",
        "threshold_home_runs_over",
        "threshold_home_runs_under",
        "threshold_clf",
        "min_ev",
        "bankroll_max_stake_pct",
        "bankroll_max_daily_exposure_pct",
        "bankroll_max_lay_price",
    ]
    return {key: getattr(cfg, key) for key in keys if hasattr(cfg, key)}


def ensure_bankroll_ledger_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE SCHEMA IF NOT EXISTS bets;
            CREATE TABLE IF NOT EXISTS bets.mlb_bankroll_ledger (
                id BIGSERIAL PRIMARY KEY,
                pick_key TEXT NOT NULL UNIQUE,
                inserted_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                sport TEXT NOT NULL DEFAULT 'mlb',
                source TEXT NOT NULL,
                game_date_et DATE NOT NULL,
                game_slug TEXT NOT NULL,
                market TEXT NOT NULL,
                stat TEXT,
                side TEXT NOT NULL,
                label TEXT,
                team_abbr TEXT,
                opponent_abbr TEXT,
                home_team_abbr TEXT,
                away_team_abbr TEXT,
                player_id BIGINT,
                player_name TEXT,
                player_name_norm TEXT,
                bookmaker_key TEXT,
                market_line NUMERIC,
                bet_line NUMERIC,
                market_price NUMERIC,
                link TEXT,
                pred_value NUMERIC,
                pred_count NUMERIC,
                model_prob NUMERIC,
                edge NUMERIC,
                edge_type TEXT,
                ev NUMERIC,
                kelly_fraction NUMERIC,
                bankroll_tier TEXT,
                bankroll_reasons TEXT,
                stake_pct NUMERIC,
                model_meta JSONB NOT NULL DEFAULT '{}'::jsonb,
                thresholds JSONB NOT NULL DEFAULT '{}'::jsonb,
                result_status TEXT NOT NULL DEFAULT 'pending',
                won BOOLEAN,
                push BOOLEAN NOT NULL DEFAULT FALSE,
                profit_units NUMERIC,
                actual_value NUMERIC,
                actual_home_score NUMERIC,
                actual_away_score NUMERIC,
                actual_run_diff NUMERIC,
                actual_total NUMERIC,
                over_hit BOOLEAN,
                closing_line NUMERIC,
                closing_price NUMERIC,
                clv_line NUMERIC,
                clv_price NUMERIC,
                graded_at_utc TIMESTAMPTZ,
                grade_source TEXT
            );
            ALTER TABLE bets.mlb_bankroll_ledger
                ADD COLUMN IF NOT EXISTS model_meta JSONB NOT NULL DEFAULT '{}'::jsonb,
                ADD COLUMN IF NOT EXISTS thresholds JSONB NOT NULL DEFAULT '{}'::jsonb,
                ADD COLUMN IF NOT EXISTS bookmaker_key TEXT,
                ADD COLUMN IF NOT EXISTS bet_line NUMERIC,
                ADD COLUMN IF NOT EXISTS player_name_norm TEXT,
                ADD COLUMN IF NOT EXISTS push BOOLEAN NOT NULL DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS clv_price NUMERIC;
            UPDATE bets.mlb_bankroll_ledger
            SET bet_line = CASE
                WHEN market = 'run_line' AND side = 'away' THEN -market_line
                ELSE market_line
            END
            WHERE bet_line IS NULL
              AND market_line IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_mlb_bankroll_ledger_date
                ON bets.mlb_bankroll_ledger (game_date_et);
            CREATE INDEX IF NOT EXISTS idx_mlb_bankroll_ledger_pending
                ON bets.mlb_bankroll_ledger (result_status, game_date_et);
            """
        )


_INSERT_SQL = """
INSERT INTO bets.mlb_bankroll_ledger (
    pick_key, source, game_date_et, game_slug, market, stat, side, label,
    team_abbr, opponent_abbr, home_team_abbr, away_team_abbr,
    player_id, player_name, player_name_norm, bookmaker_key,
    market_line, bet_line, market_price, link, pred_value, pred_count, model_prob,
    edge, edge_type, ev, kelly_fraction, bankroll_tier, bankroll_reasons,
    stake_pct, model_meta, thresholds
) VALUES (
    %(pick_key)s, %(source)s, %(game_date_et)s, %(game_slug)s, %(market)s, %(stat)s,
    %(side)s, %(label)s, %(team_abbr)s, %(opponent_abbr)s, %(home_team_abbr)s,
    %(away_team_abbr)s, %(player_id)s, %(player_name)s, %(player_name_norm)s,
    %(bookmaker_key)s, %(market_line)s, %(bet_line)s, %(market_price)s, %(link)s,
    %(pred_value)s, %(pred_count)s, %(model_prob)s, %(edge)s, %(edge_type)s, %(ev)s,
    %(kelly_fraction)s, %(bankroll_tier)s, %(bankroll_reasons)s, %(stake_pct)s,
    %(model_meta)s, %(thresholds)s
) ON CONFLICT (pick_key) DO NOTHING
"""


def insert_ledger_rows(conn, rows: Iterable[dict[str, Any]]) -> int:
    payload = list(rows)
    if not payload:
        return 0
    ensure_bankroll_ledger_schema(conn)
    for row in payload:
        row["model_meta"] = _json(row.get("model_meta"))
        row["thresholds"] = _json(row.get("thresholds"))
    with conn.cursor() as cur:
        inserted = 0
        for row in payload:
            cur.execute(_INSERT_SQL + " RETURNING id", row)
            if cur.fetchone() is not None:
                inserted += 1
    conn.commit()
    return inserted


def void_negative_ev_game_bankroll_candidates(conn, cfg=None) -> int:
    """Void pending game bankroll rows that do not clear price-adjusted EV."""
    ensure_bankroll_ledger_schema(conn)
    min_ev = _clean_float(getattr(cfg, "min_game_ev", None)) if cfg is not None else 0.02
    min_ev = 0.02 if min_ev is None else min_ev
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, model_prob, market_price, bankroll_reasons
            FROM bets.mlb_bankroll_ledger
            WHERE source = 'game'
              AND result_status = 'pending'
              AND model_prob IS NOT NULL
              AND market_price IS NOT NULL
            """
        )
        fills = []
        voids = []
        for row in cur.fetchall():
            ev = _ev_per_unit(row.get("model_prob"), row.get("market_price"))
            if ev is None:
                continue
            fills.append((ev, row["id"]))
            if ev >= min_ev:
                continue
            reason = "negative_price_ev" if ev < 0 else "below_min_price_ev"
            existing = row.get("bankroll_reasons") or ""
            parts = [p.strip() for p in existing.split(";") if p.strip()]
            if reason not in parts:
                parts.append(reason)
            voids.append((ev, "; ".join(parts), row["id"]))
        for ev, row_id in fills:
            cur.execute(
                """
                UPDATE bets.mlb_bankroll_ledger
                SET ev = COALESCE(ev, %s)
                WHERE id = %s
                """,
                (ev, row_id),
            )
        for ev, reasons, row_id in voids:
            cur.execute(
                """
                UPDATE bets.mlb_bankroll_ledger
                SET result_status = 'voided',
                    ev = %s,
                    bankroll_reasons = %s,
                    profit_units = 0,
                    graded_at_utc = NOW(),
                    grade_source = 'price_ev_reconciliation'
                WHERE id = %s
                  AND result_status = 'pending'
                """,
                (ev, reasons, row_id),
            )
    conn.commit()
    return len(voids)


def sync_pending_game_bankroll_metadata(conn, rows: Iterable[dict[str, Any]]) -> int:
    payload = list(rows)
    if not payload:
        return 0
    ensure_bankroll_ledger_schema(conn)
    updated = 0
    with conn.cursor() as cur:
        for row in payload:
            cur.execute(
                """
                UPDATE bets.mlb_bankroll_ledger
                SET model_prob = %s,
                    ev = %s,
                    kelly_fraction = %s,
                    bankroll_tier = %s,
                    bankroll_reasons = %s,
                    stake_pct = %s,
                    model_meta = model_meta || %s::jsonb
                WHERE pick_key = %s
                  AND source = 'game'
                  AND result_status = 'pending'
                """,
                (
                    row.get("model_prob"),
                    row.get("ev"),
                    row.get("kelly_fraction"),
                    row.get("bankroll_tier"),
                    row.get("bankroll_reasons"),
                    row.get("stake_pct"),
                    json.dumps({
                        "bankroll_label_synced_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
                    }),
                    row.get("pick_key"),
                ),
            )
            updated += cur.rowcount
    conn.commit()
    return updated


def insert_game_bankroll_ledger(conn, rows: list[dict[str, Any]], *, fd_links=None, cfg=None) -> int:
    thresholds = _cfg_thresholds(cfg)
    out: list[dict[str, Any]] = []
    for row in rows:
        game_date = row.get("game_date_et")
        slug = row.get("game_slug")
        home = row.get("home_team_abbr")
        away = row.get("away_team_abbr")
        fd = (fd_links or {}).get((home, away))

        if row.get("bankroll_candidate_rl"):
            side = row.get("run_line_bet_side")
            team = home if side == "home" else away
            opp = away if side == "home" else home
            link = None
            if fd is not None:
                link = getattr(fd, "spread_home_link", None) if side == "home" else getattr(fd, "spread_away_link", None)
            line_label = _run_line_side_label(row.get("market_run_line"), side)
            home_line = _clean_float(row.get("market_run_line"))
            bet_line = home_line if side == "home" else (-home_line if home_line is not None else None)
            model_prob = _clean_float(row.get("win_prob_rl"))
            market_price = _clean_float(row.get("market_rl_price"))
            out.append({
                "pick_key": _pick_key("mlb", "game", game_date, slug, "run_line", side),
                "source": "game",
                "game_date_et": game_date,
                "game_slug": slug,
                "market": "run_line",
                "stat": None,
                "side": side,
                "label": f"{team} {line_label}" if line_label else f"{team} run line",
                "team_abbr": team,
                "opponent_abbr": opp,
                "home_team_abbr": home,
                "away_team_abbr": away,
                "player_id": None,
                "player_name": None,
                "player_name_norm": None,
                "bookmaker_key": "fanduel" if link else None,
                "market_line": home_line,
                "bet_line": bet_line,
                "market_price": market_price,
                "link": link,
                "pred_value": _clean_float(row.get("pred_run_diff")),
                "pred_count": None,
                "model_prob": model_prob,
                "edge": _clean_float(row.get("edge_run_line")),
                "edge_type": "run_line",
                "ev": _ev_per_unit(model_prob, market_price),
                "kelly_fraction": _clean_float(row.get("kelly_fraction_rl")),
                "bankroll_tier": row.get("bankroll_tier_rl"),
                "bankroll_reasons": row.get("bankroll_reasons_rl"),
                "stake_pct": _clean_float(row.get("stake_pct_rl")),
                "model_meta": {
                    "season": row.get("season"),
                    "used_residual_model": row.get("used_residual_model"),
                    "sigma_q_rl": row.get("sigma_q_rl"),
                    "p_home_cover_clf": row.get("p_home_cover_clf"),
                },
                "thresholds": thresholds,
            })

        if row.get("bankroll_candidate_total"):
            side = row.get("total_bet_side")
            link = None
            if fd is not None:
                link = getattr(fd, "total_over_link", None) if side == "over" else getattr(fd, "total_under_link", None)
            model_prob = _clean_float(row.get("win_prob_total"))
            market_price = _clean_float(row.get("market_total_price"))
            out.append({
                "pick_key": _pick_key("mlb", "game", game_date, slug, "total", side),
                "source": "game",
                "game_date_et": game_date,
                "game_slug": slug,
                "market": "total",
                "stat": None,
                "side": side,
                "label": f"{side.upper()} {row.get('market_total')}",
                "team_abbr": None,
                "opponent_abbr": None,
                "home_team_abbr": home,
                "away_team_abbr": away,
                "player_id": None,
                "player_name": None,
                "player_name_norm": None,
                "bookmaker_key": "fanduel" if link else None,
                "market_line": _clean_float(row.get("market_total")),
                "bet_line": _clean_float(row.get("market_total")),
                "market_price": market_price,
                "link": link,
                "pred_value": _clean_float(row.get("pred_total")),
                "pred_count": None,
                "model_prob": model_prob,
                "edge": _clean_float(row.get("edge_total")),
                "edge_type": "total",
                "ev": _ev_per_unit(model_prob, market_price),
                "kelly_fraction": _clean_float(row.get("kelly_fraction_total")),
                "bankroll_tier": row.get("bankroll_tier_total"),
                "bankroll_reasons": row.get("bankroll_reasons_total"),
                "stake_pct": _clean_float(row.get("stake_pct_total")),
                "model_meta": {
                    "season": row.get("season"),
                    "used_residual_model": row.get("used_residual_model"),
                    "sigma_q_total": row.get("sigma_q_total"),
                    "p_total_over_clf": row.get("p_total_over_clf"),
                },
                "thresholds": thresholds,
            })
    inserted = insert_ledger_rows(conn, out)
    synced = sync_pending_game_bankroll_metadata(conn, out)
    if synced:
        log.info("Synced %d pending MLB game bankroll rows", synced)
    voided = void_negative_ev_game_bankroll_candidates(conn, cfg=cfg)
    if voided:
        log.info("Voided %d pending MLB game bankroll rows below price-adjusted EV minimum", voided)
    return inserted


def insert_prop_bankroll_ledger(conn, rows: list[dict[str, Any]], *, prop_lines=None, cfg=None) -> int:
    thresholds = _cfg_thresholds(cfg)
    out: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("bankroll_candidate"):
            continue
        side = (row.get("bet_side") or "").lower()
        if side not in {"over", "under"}:
            continue
        name = row.get("player_name") or ""
        stat = row.get("stat")
        norm = _normalize_name(name)
        ld = (prop_lines or {}).get((norm, stat), {})
        link = ld.get("over_link") if side == "over" else ld.get("under_link")
        bookmaker = row.get("bookmaker_key")
        if link and "fanduel.com" in link:
            bookmaker = "fanduel"
        elif side == "under" and ld.get("under_link_book"):
            bookmaker = ld.get("under_link_book")
        p_over = _clean_float(row.get("pred_prob_over"))
        model_prob = p_over if side == "over" else (1.0 - p_over if p_over is not None else None)
        line = _clean_float(row.get("book_line"))
        out.append({
            "pick_key": _pick_key("mlb", "prop", row.get("game_date_et"), row.get("game_slug"), row.get("player_id"), stat, side),
            "source": "prop",
            "game_date_et": row.get("game_date_et"),
            "game_slug": row.get("game_slug"),
            "market": stat,
            "stat": stat,
            "side": side,
            "label": f"{name} {stat} {side} {line}",
            "team_abbr": row.get("team_abbr"),
            "opponent_abbr": None,
            "home_team_abbr": None,
            "away_team_abbr": None,
            "player_id": _clean_int(row.get("player_id")),
            "player_name": name,
            "player_name_norm": norm,
            "bookmaker_key": bookmaker,
            "market_line": line,
            "bet_line": line,
            "market_price": _clean_float(row.get("bet_price")),
            "link": link,
            "pred_value": _clean_float(row.get("pred_value")),
            "pred_count": _clean_float(row.get("pred_count")),
            "model_prob": model_prob,
            "edge": _clean_float(row.get("edge")),
            "edge_type": row.get("edge_type"),
            "ev": _clean_float(row.get("ev")),
            "kelly_fraction": _clean_float(row.get("kelly_fraction")),
            "bankroll_tier": row.get("bankroll_tier"),
            "bankroll_reasons": row.get("bankroll_reasons"),
            "stake_pct": _clean_float(row.get("stake_pct")),
            "model_meta": {
                "model_family": row.get("model_family"),
                "line_bucket": row.get("line_bucket"),
                "bookmaker_key": row.get("bookmaker_key"),
            },
            "thresholds": thresholds,
        })
    return insert_ledger_rows(conn, out)


def grade_bankroll_ledger(conn) -> int:
    ensure_bankroll_ledger_schema(conn)
    updated = _grade_game_ledger(conn) + _grade_prop_ledger(conn)
    conn.commit()
    return updated


def _load_game_closes(conn, dates: list) -> dict[tuple, dict]:
    if not dates:
        return {}
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (home_team, away_team, as_of_date)
                home_team, away_team, as_of_date,
                spread_home_points AS closing_run_line,
                total_points AS closing_total,
                spread_home_price AS closing_rl_home_price,
                spread_away_price AS closing_rl_away_price,
                total_over_price AS closing_total_over_price,
                total_under_price AS closing_total_under_price
            FROM odds.mlb_game_lines
            WHERE as_of_date = ANY(%s::date[])
              AND (spread_home_points IS NOT NULL OR total_points IS NOT NULL)
            ORDER BY
                home_team, away_team, as_of_date,
                CASE WHEN bookmaker_key = 'fanduel' THEN 0 ELSE 1 END,
                fetched_at_utc DESC
            """,
            (dates,),
        )
        return {(r["home_team"], r["away_team"], r["as_of_date"]): r for r in cur.fetchall()}


def _grade_game_ledger(conn) -> int:
    today = datetime.now(_ET).date()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT *
            FROM bets.mlb_bankroll_ledger
            WHERE source = 'game'
              AND result_status = 'pending'
              AND game_date_et <= %s
            """,
            (today,),
        )
        rows = cur.fetchall()
    if not rows:
        return 0

    slugs = list({r["game_slug"] for r in rows})
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT game_slug, home_score, away_score
            FROM raw.mlb_games
            WHERE game_slug = ANY(%s)
              AND status = 'final'
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
            """,
            (slugs,),
        )
        finals = {r["game_slug"]: r for r in cur.fetchall()}

    closes = _load_game_closes(conn, list({r["game_date_et"] for r in rows}))
    updated = 0
    with conn.cursor() as cur:
        for row in rows:
            final = finals.get(row["game_slug"])
            if not final:
                continue
            home_score = float(final["home_score"])
            away_score = float(final["away_score"])
            actual_run_diff = home_score - away_score
            actual_total = home_score + away_score
            market_line = _clean_float(row["market_line"])
            side = row["side"]
            won = None
            push = False
            over_hit = None

            if row["market"] == "run_line" and market_line is not None:
                threshold = -market_line
                push = abs(actual_run_diff - threshold) <= 1e-9
                home_covered = actual_run_diff > threshold
                won = None if push else (home_covered if side == "home" else not home_covered)
            elif row["market"] == "total" and market_line is not None:
                push = abs(actual_total - market_line) <= 1e-9
                over_hit = actual_total > market_line
                won = None if push else (over_hit if side == "over" else not over_hit)

            close = closes.get((row["home_team_abbr"], row["away_team_abbr"], row["game_date_et"]))
            closing_line = closing_price = clv_line = clv_price = None
            if close:
                if row["market"] == "run_line":
                    closing_line = _clean_float(close.get("closing_run_line"))
                    if side == "home":
                        closing_price = _clean_float(close.get("closing_rl_home_price"))
                        if closing_line is not None and market_line is not None:
                            clv_line = round(market_line - closing_line, 2)
                    else:
                        closing_price = _clean_float(close.get("closing_rl_away_price"))
                        if closing_line is not None and market_line is not None:
                            clv_line = round(closing_line - market_line, 2)
                elif row["market"] == "total":
                    closing_line = _clean_float(close.get("closing_total"))
                    if side == "over":
                        closing_price = _clean_float(close.get("closing_total_over_price"))
                        if closing_line is not None and market_line is not None:
                            clv_line = round(closing_line - market_line, 2)
                    else:
                        closing_price = _clean_float(close.get("closing_total_under_price"))
                        if closing_line is not None and market_line is not None:
                            clv_line = round(market_line - closing_line, 2)
                clv_price = _price_clv(row["market_price"], closing_price)

            profit = _profit_units(won, push, row["market_price"])
            cur.execute(
                """
                UPDATE bets.mlb_bankroll_ledger
                SET result_status = 'graded',
                    won = %s,
                    push = %s,
                    profit_units = %s,
                    actual_home_score = %s,
                    actual_away_score = %s,
                    actual_run_diff = %s,
                    actual_total = %s,
                    over_hit = %s,
                    closing_line = %s,
                    closing_price = %s,
                    clv_line = %s,
                    clv_price = %s,
                    graded_at_utc = NOW(),
                    grade_source = 'raw.mlb_games'
                WHERE id = %s
                """,
                (
                    won,
                    push,
                    profit,
                    home_score,
                    away_score,
                    actual_run_diff,
                    actual_total,
                    over_hit,
                    closing_line,
                    closing_price,
                    clv_line,
                    clv_price,
                    row["id"],
                ),
            )
            updated += 1
    log.info("Graded %d game bankroll ledger rows", updated)
    return updated


def _prop_close(conn, row: dict) -> tuple[float | None, float | None, float | None, float | None]:
    norm = row.get("player_name_norm") or _normalize_name(row.get("player_name") or "")
    stat = row.get("stat") or row.get("market")
    if not norm or not stat:
        return None, None, None, None
    bookmaker = row.get("bookmaker_key")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if bookmaker:
            cur.execute(
                """
                SELECT line, over_price, under_price
                FROM odds.mlb_player_prop_lines
                WHERE as_of_date = %s
                  AND player_name_norm = %s
                  AND stat = %s
                  AND bookmaker_key = %s
                ORDER BY fetched_at_utc DESC
                LIMIT 1
                """,
                (row["game_date_et"], norm, stat, bookmaker),
            )
            close = cur.fetchone()
            if close:
                closing_line = _clean_float(close.get("line"))
                closing_price = _clean_float(close.get("over_price") if row["side"] == "over" else close.get("under_price"))
                return closing_line, closing_price, None, None
        cur.execute(
            """
            SELECT line, over_price, under_price
            FROM odds.mlb_player_prop_lines
            WHERE as_of_date = %s
              AND player_name_norm = %s
              AND stat = %s
            ORDER BY
              CASE WHEN bookmaker_key = 'fanduel' THEN 0 ELSE 1 END,
              fetched_at_utc DESC
            LIMIT 1
            """,
            (row["game_date_et"], norm, stat),
        )
        close = cur.fetchone()
    if not close:
        return None, None, None, None
    closing_line = _clean_float(close.get("line"))
    closing_price = _clean_float(close.get("over_price") if row["side"] == "over" else close.get("under_price"))
    return closing_line, closing_price, None, None


def _grade_prop_ledger(conn) -> int:
    today = datetime.now(_ET).date()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT *
            FROM bets.mlb_bankroll_ledger
            WHERE source = 'prop'
              AND result_status = 'pending'
              AND game_date_et <= %s
            """,
            (today,),
        )
        rows = cur.fetchall()
    if not rows:
        return 0

    slugs = list({r["game_slug"] for r in rows})
    player_ids = list({int(r["player_id"]) for r in rows if r["player_id"] is not None})
    if not slugs or not player_ids:
        return 0
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
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
            """,
            (slugs, player_ids),
        )
        actuals = {(r["game_slug"], int(r["player_id"])): r for r in cur.fetchall()}

    updated = 0
    with conn.cursor() as cur:
        for row in rows:
            key = (row["game_slug"], int(row["player_id"])) if row["player_id"] is not None else None
            actual = actuals.get(key)
            if not actual:
                continue
            stat = row["stat"] or row["market"]
            gl_col = _STAT_COL.get(stat)
            if not gl_col:
                continue
            actual_value = _clean_float(actual.get(gl_col))
            market_line = _clean_float(row["market_line"])
            if actual_value is None or market_line is None:
                continue
            push = abs(actual_value - market_line) <= 1e-9
            over_hit = actual_value > market_line
            won = None if push else (over_hit if row["side"] == "over" else not over_hit)
            closing_line, closing_price, _, _ = _prop_close(conn, row)
            clv_line = None
            if closing_line is not None:
                clv_line = round(closing_line - market_line, 2) if row["side"] == "over" else round(market_line - closing_line, 2)
            clv_price = _price_clv(row["market_price"], closing_price)
            profit = _profit_units(won, push, row["market_price"])
            cur.execute(
                """
                UPDATE bets.mlb_bankroll_ledger
                SET result_status = 'graded',
                    won = %s,
                    push = %s,
                    profit_units = %s,
                    actual_value = %s,
                    over_hit = %s,
                    closing_line = %s,
                    closing_price = %s,
                    clv_line = %s,
                    clv_price = %s,
                    graded_at_utc = NOW(),
                    grade_source = 'raw.mlb_player_gamelogs'
                WHERE id = %s
                """,
                (
                    won,
                    push,
                    profit,
                    actual_value,
                    over_hit,
                    closing_line,
                    closing_price,
                    clv_line,
                    clv_price,
                    row["id"],
                ),
            )
            updated += 1
    log.info("Graded %d prop bankroll ledger rows", updated)
    return updated
