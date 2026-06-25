"""Build historical side-level MLB prop market examples from odds + outcomes.

Replay rows only cover dates where we locked model predictions.  This table is
the deeper market-history base: one row per player/date/market/book/line/side
with the offered price and final outcome.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

import pandas as pd
import psycopg2
import psycopg2.extras

from .prop_replay import american_to_prob, ev_per_unit, no_vig_probs
from .side_recalibration import price_bucket, prop_line_bucket, prop_line_surface

from mlb_pipeline.db import PG_DSN as _PG_DSN
_MARKETS = ("pitcher_strikeouts", "batter_hits", "batter_total_bases", "batter_home_runs")
_BOOKMAKERS = ("fanduel", "draftkings")


@dataclass(frozen=True)
class PropMarketHistoryConfig:
    pg_dsn: str = _PG_DSN
    lookback_days: int = 540
    date_from: date | None = None
    date_to: date | None = None
    markets: tuple[str, ...] = _MARKETS
    bookmakers: tuple[str, ...] = _BOOKMAKERS
    replace: bool = True


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _norm_name(first: Any, last: Any) -> str:
    name = f"{first or ''} {last or ''}".strip()
    if not name:
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", errors="ignore").decode("ascii")
    return re.sub(r"[^a-z ]", "", ascii_name.lower()).strip()


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


def ensure_prop_market_history_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE SCHEMA IF NOT EXISTS features;
            CREATE TABLE IF NOT EXISTS features.mlb_prop_market_history_examples (
                id BIGSERIAL PRIMARY KEY,
                source TEXT NOT NULL DEFAULT 'historical_odds',
                game_date_et DATE NOT NULL,
                game_slug TEXT NOT NULL,
                player_id BIGINT NOT NULL,
                player_name TEXT,
                player_name_norm TEXT NOT NULL,
                team_abbr TEXT,
                market TEXT NOT NULL,
                side TEXT NOT NULL,
                bookmaker_key TEXT NOT NULL,
                market_line NUMERIC NOT NULL,
                market_price NUMERIC,
                paired_price NUMERIC,
                raw_market_prob NUMERIC,
                no_vig_market_prob NUMERIC,
                market_prob_side NUMERIC,
                price_bucket TEXT,
                line_bucket TEXT,
                line_surface TEXT,
                actual_value NUMERIC,
                over_hit BOOLEAN,
                won BOOLEAN,
                push BOOLEAN,
                profit_units NUMERIC,
                fetched_at_utc TIMESTAMPTZ,
                example_updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (game_date_et, player_id, market, side, bookmaker_key, market_line)
            );
            ALTER TABLE features.mlb_prop_market_history_examples
                ADD COLUMN IF NOT EXISTS line_surface TEXT;
            CREATE INDEX IF NOT EXISTS idx_mlb_prop_market_history_date
                ON features.mlb_prop_market_history_examples (game_date_et);
            CREATE INDEX IF NOT EXISTS idx_mlb_prop_market_history_bucket
                ON features.mlb_prop_market_history_examples
                (market, side, line_surface, line_bucket, price_bucket, bookmaker_key);
            """
        )
    conn.commit()


def _date_window(cfg: PropMarketHistoryConfig) -> tuple[date, date | None]:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)
    return cfg.date_from or cutoff, cfg.date_to


def _load_player_names(conn) -> tuple[dict[int, str], dict[int, str]]:
    rows = pd.read_sql(
        """
        SELECT DISTINCT player_id, first_name, last_name
        FROM raw.mlb_boxscore_player_stats
        WHERE first_name IS NOT NULL AND last_name IS NOT NULL
        """,
        conn,
    )
    norm: dict[int, str] = {}
    full: dict[int, str] = {}
    for _, r in rows.iterrows():
        pid = int(r["player_id"])
        first = str(r["first_name"] or "").strip()
        last = str(r["last_name"] or "").strip()
        norm_name = _norm_name(first, last)
        if norm_name:
            norm[pid] = norm_name
            full[pid] = f"{first} {last}".strip()
    return norm, full


def _load_lines(conn, cfg: PropMarketHistoryConfig) -> pd.DataFrame:
    date_from, date_to = _date_window(cfg)
    filters = [
        "as_of_date >= %(date_from)s",
        "stat = ANY(%(markets)s)",
        "bookmaker_key = ANY(%(bookmakers)s)",
        "line IS NOT NULL",
    ]
    params: dict[str, Any] = {
        "date_from": date_from,
        "markets": list(cfg.markets),
        "bookmakers": list(cfg.bookmakers),
    }
    if date_to is not None:
        filters.append("as_of_date <= %(date_to)s")
        params["date_to"] = date_to
    sql = f"""
        SELECT DISTINCT ON (as_of_date, player_name_norm, stat, bookmaker_key, line)
            as_of_date,
            player_name,
            player_name_norm,
            stat AS market,
            bookmaker_key,
            line::float AS market_line,
            over_price::float AS over_price,
            under_price::float AS under_price,
            fetched_at_utc
        FROM odds.mlb_player_prop_lines
        WHERE {' AND '.join(filters)}
        ORDER BY
            as_of_date, player_name_norm, stat, bookmaker_key, line,
            fetched_at_utc DESC NULLS LAST
    """
    df = pd.read_sql(sql, conn, params=params)
    if df.empty:
        return df
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    df["market_line"] = pd.to_numeric(df["market_line"], errors="coerce")
    df["over_price"] = pd.to_numeric(df["over_price"], errors="coerce")
    df["under_price"] = pd.to_numeric(df["under_price"], errors="coerce")
    return df.dropna(subset=["market_line", "player_name_norm", "market"])


def _load_base_training(conn, cfg: PropMarketHistoryConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_from, date_to = _date_window(cfg)
    filters = ["gl.game_date_et >= %(date_from)s", "g.status = 'final'"]
    params: dict[str, Any] = {"date_from": date_from}
    if date_to is not None:
        filters.append("gl.game_date_et <= %(date_to)s")
        params["date_to"] = date_to
    sql = f"""
        SELECT
            gl.season,
            gl.game_slug,
            gl.game_date_et,
            gl.player_id,
            gl.team_abbr,
            gl.innings_pitched::float AS innings_pitched,
            gl.strikeouts_pitcher AS strikeouts,
            gl.at_bats,
            gl.hits,
            gl.total_bases,
            gl.home_runs
        FROM raw.mlb_player_gamelogs gl
        JOIN raw.mlb_games g
          ON g.game_slug = gl.game_slug
        WHERE {' AND '.join(filters)}
          AND (
                gl.strikeouts_pitcher IS NOT NULL
             OR gl.hits IS NOT NULL
             OR gl.total_bases IS NOT NULL
             OR gl.home_runs IS NOT NULL
          )
    """
    base = pd.read_sql(sql, conn, params=params)
    if base.empty:
        return pd.DataFrame(), pd.DataFrame()
    base["game_date_et"] = pd.to_datetime(base["game_date_et"]).dt.date
    pitcher = base[
        base["strikeouts"].notna()
        & (pd.to_numeric(base["innings_pitched"], errors="coerce").fillna(0.0) >= 1.0)
    ].copy()
    batter = base[
        base["hits"].notna()
        & base["total_bases"].notna()
        & base["home_runs"].notna()
        & (pd.to_numeric(base["at_bats"], errors="coerce").fillna(0.0) >= 0.0)
    ].copy()
    return pitcher, batter


def _base_market_rows(
    base: pd.DataFrame,
    *,
    market: str,
    target_col: str,
    norm_names: dict[int, str],
    full_names: dict[int, str],
) -> pd.DataFrame:
    if base.empty or target_col not in base.columns:
        return pd.DataFrame()
    out = base[["game_date_et", "game_slug", "player_id", "team_abbr", target_col]].copy()
    out = out.rename(columns={target_col: "actual_value"})
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce")
    out = out.dropna(subset=["player_id", "actual_value"])
    out["player_id"] = out["player_id"].astype(int)
    out["player_name_norm"] = out["player_id"].map(norm_names)
    out["player_name"] = out["player_id"].map(full_names)
    out["market"] = market
    return out.dropna(subset=["player_name_norm"])


def _expand_sides(row: pd.Series) -> list[dict[str, Any]]:
    line = _clean_float(row.get("market_line"))
    actual = _clean_float(row.get("actual_value"))
    if line is None or actual is None:
        return []
    market = str(row.get("market") or "")
    over_price = _clean_float(row.get("over_price"))
    under_price = _clean_float(row.get("under_price"))
    nv_over, nv_under = no_vig_probs(over_price, under_price)
    push = abs(actual - line) <= 1e-9
    over_hit = actual > line
    out: list[dict[str, Any]] = []
    for side in ("over", "under"):
        price = over_price if side == "over" else under_price
        paired = under_price if side == "over" else over_price
        raw_prob = american_to_prob(price)
        no_vig = nv_over if side == "over" else nv_under
        market_prob = no_vig if no_vig is not None else raw_prob
        won = None if push else (over_hit if side == "over" else not over_hit)
        out.append({
            "source": "historical_odds",
            "game_date_et": row.get("game_date_et"),
            "game_slug": row.get("game_slug"),
            "player_id": int(row.get("player_id")),
            "player_name": row.get("player_name"),
            "player_name_norm": row.get("player_name_norm"),
            "team_abbr": row.get("team_abbr"),
            "market": market,
            "side": side,
            "bookmaker_key": row.get("bookmaker_key"),
            "market_line": line,
            "market_price": price,
            "paired_price": paired,
            "raw_market_prob": raw_prob,
            "no_vig_market_prob": no_vig,
            "market_prob_side": market_prob,
            "price_bucket": price_bucket(price),
            "line_bucket": prop_line_bucket(market, line),
            "line_surface": prop_line_surface(market, side, line),
            "actual_value": actual,
            "over_hit": over_hit,
            "won": won,
            "push": push,
            "profit_units": _profit_units(won, push, price),
            "fetched_at_utc": row.get("fetched_at_utc"),
        })
    return out


_INSERT_SQL = """
INSERT INTO features.mlb_prop_market_history_examples (
    source, game_date_et, game_slug, player_id, player_name, player_name_norm,
    team_abbr, market, side, bookmaker_key, market_line, market_price,
    paired_price, raw_market_prob, no_vig_market_prob, market_prob_side,
    price_bucket, line_bucket, line_surface, actual_value, over_hit, won, push,
    profit_units, fetched_at_utc, example_updated_at
) VALUES (
    %(source)s, %(game_date_et)s, %(game_slug)s, %(player_id)s, %(player_name)s, %(player_name_norm)s,
    %(team_abbr)s, %(market)s, %(side)s, %(bookmaker_key)s, %(market_line)s, %(market_price)s,
    %(paired_price)s, %(raw_market_prob)s, %(no_vig_market_prob)s, %(market_prob_side)s,
    %(price_bucket)s, %(line_bucket)s, %(line_surface)s, %(actual_value)s, %(over_hit)s, %(won)s, %(push)s,
    %(profit_units)s, %(fetched_at_utc)s, now()
)
ON CONFLICT (game_date_et, player_id, market, side, bookmaker_key, market_line) DO UPDATE SET
    player_name = EXCLUDED.player_name,
    player_name_norm = EXCLUDED.player_name_norm,
    team_abbr = EXCLUDED.team_abbr,
    market_price = EXCLUDED.market_price,
    paired_price = EXCLUDED.paired_price,
    raw_market_prob = EXCLUDED.raw_market_prob,
    no_vig_market_prob = EXCLUDED.no_vig_market_prob,
    market_prob_side = EXCLUDED.market_prob_side,
    price_bucket = EXCLUDED.price_bucket,
    line_bucket = EXCLUDED.line_bucket,
    line_surface = EXCLUDED.line_surface,
    actual_value = EXCLUDED.actual_value,
    over_hit = EXCLUDED.over_hit,
    won = EXCLUDED.won,
    push = EXCLUDED.push,
    profit_units = EXCLUDED.profit_units,
    fetched_at_utc = EXCLUDED.fetched_at_utc,
    example_updated_at = now()
"""


def _delete_existing(conn, cfg: PropMarketHistoryConfig) -> int:
    date_from, date_to = _date_window(cfg)
    filters = ["game_date_et >= %s", "market = ANY(%s)", "bookmaker_key = ANY(%s)"]
    params: list[Any] = [date_from, list(cfg.markets), list(cfg.bookmakers)]
    if date_to is not None:
        filters.append("game_date_et <= %s")
        params.append(date_to)
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM features.mlb_prop_market_history_examples WHERE {' AND '.join(filters)}",
            params,
        )
        deleted = cur.rowcount
    conn.commit()
    return deleted


def refresh_prop_market_history_examples(cfg: PropMarketHistoryConfig) -> dict[str, int]:
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_prop_market_history_schema(conn)
        deleted = _delete_existing(conn, cfg) if cfg.replace else 0
        lines = _load_lines(conn, cfg)
        if lines.empty:
            return {"deleted": deleted, "base_rows": 0, "line_rows": 0, "examples": 0}
        norm_names, full_names = _load_player_names(conn)
        pitcher, batter = _load_base_training(conn, cfg)
        bases = []
        if "pitcher_strikeouts" in cfg.markets:
            bases.append(_base_market_rows(
                pitcher,
                market="pitcher_strikeouts",
                target_col="strikeouts",
                norm_names=norm_names,
                full_names=full_names,
            ))
        target_map = {
            "batter_hits": "hits",
            "batter_total_bases": "total_bases",
            "batter_home_runs": "home_runs",
        }
        for market, target_col in target_map.items():
            if market in cfg.markets:
                bases.append(_base_market_rows(
                    batter,
                    market=market,
                    target_col=target_col,
                    norm_names=norm_names,
                    full_names=full_names,
                ))
        base = pd.concat([b for b in bases if not b.empty], ignore_index=True) if bases else pd.DataFrame()
        if base.empty:
            return {"deleted": deleted, "base_rows": 0, "line_rows": int(len(lines)), "examples": 0}
        base["_date"] = pd.to_datetime(base["game_date_et"]).dt.date
        lines = lines.rename(columns={"as_of_date": "_date"})
        merged = base.merge(
            lines,
            on=["_date", "player_name_norm", "market"],
            how="inner",
            suffixes=("", "_line"),
        )
        examples: list[dict[str, Any]] = []
        if not merged.empty:
            for _, row in merged.iterrows():
                examples.extend(_expand_sides(row))
        if examples:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, _INSERT_SQL, examples, page_size=1000)
            conn.commit()
    return {
        "deleted": deleted,
        "base_rows": int(len(base)),
        "line_rows": int(len(lines)),
        "matched_rows": int(len(merged)) if "merged" in locals() else 0,
        "examples": int(len(examples)) if "examples" in locals() else 0,
    }
