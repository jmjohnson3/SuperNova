"""Normalized MLB prop offer/link utilities.

The raw odds table stores one row per player/stat/line/book with separate
over/under columns.  This module normalizes that into one row per offered side
so candidate selection, Discord links, and ledgers can all reason about the
same book/price/link surface.
"""
from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

import pandas as pd
import psycopg2.extras

log = logging.getLogger("mlb_pipeline.modeling.prop_offer_links")

DEFAULT_BOOKMAKERS = ("fanduel", "draftkings")
OFFER_TABLE = "features.mlb_prop_offer_links"


@dataclass(frozen=True)
class PropOfferBuildResult:
    table: str
    rows_upserted: int
    game_date: str | None
    lookback_days: int | None
    bookmakers: tuple[str, ...]


def _normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "")
    ascii_str = s.encode("ascii", "ignore").decode("ascii")
    ascii_str = re.sub(r"[^a-z0-9\s]", "", ascii_str.lower())
    return re.sub(r"\s+", " ", ascii_str).strip()


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def ensure_prop_offer_links_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE SCHEMA IF NOT EXISTS features;
            CREATE TABLE IF NOT EXISTS features.mlb_prop_offer_links (
                id BIGSERIAL PRIMARY KEY,
                source_row_id INTEGER,
                as_of_date DATE NOT NULL,
                fetched_at_utc TIMESTAMPTZ,
                updated_at_utc TIMESTAMPTZ,
                event_id TEXT,
                commence_time_utc TEXT,
                bookmaker_key TEXT NOT NULL,
                home_team TEXT,
                away_team TEXT,
                player_name TEXT,
                player_name_norm TEXT NOT NULL,
                stat TEXT NOT NULL,
                side TEXT NOT NULL CHECK (side IN ('over', 'under')),
                line NUMERIC NOT NULL,
                price INTEGER,
                link TEXT,
                is_linkable BOOLEAN NOT NULL DEFAULT FALSE,
                created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                refreshed_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT uq_mlb_prop_offer_links_event_offer
                    UNIQUE (as_of_date, event_id, player_name_norm, stat, bookmaker_key, line, side)
            );
            ALTER TABLE features.mlb_prop_offer_links
                DROP CONSTRAINT IF EXISTS mlb_prop_offer_links_as_of_date_player_name_norm_stat_bookm_key;
            CREATE UNIQUE INDEX IF NOT EXISTS uq_mlb_prop_offer_links_event_offer
                ON features.mlb_prop_offer_links
                (as_of_date, event_id, player_name_norm, stat, bookmaker_key, line, side);
            CREATE INDEX IF NOT EXISTS idx_mlb_prop_offer_links_date_stat
                ON features.mlb_prop_offer_links (as_of_date, stat, side);
            CREATE INDEX IF NOT EXISTS idx_mlb_prop_offer_links_player
                ON features.mlb_prop_offer_links (as_of_date, player_name_norm, stat);
            CREATE INDEX IF NOT EXISTS idx_mlb_prop_offer_links_book
                ON features.mlb_prop_offer_links (bookmaker_key, as_of_date);
            """
        )
    conn.commit()


def _source_filter_sql(*, game_date: date | None, lookback_days: int | None) -> tuple[str, dict[str, Any]]:
    filters = [
        "bookmaker_key = ANY(%(bookmakers)s)",
        "player_name_norm IS NOT NULL",
        "stat IS NOT NULL",
        "line IS NOT NULL",
    ]
    params: dict[str, Any] = {}
    if game_date is not None:
        filters.append("as_of_date = %(game_date)s")
        params["game_date"] = game_date
    elif lookback_days is not None:
        filters.append("as_of_date >= CURRENT_DATE - (%(lookback_days)s::int * INTERVAL '1 day')")
        params["lookback_days"] = int(lookback_days)
    return " AND ".join(filters), params


def refresh_prop_offer_links(
    conn,
    *,
    game_date: date | None = None,
    lookback_days: int | None = 14,
    bookmakers: Iterable[str] = DEFAULT_BOOKMAKERS,
) -> PropOfferBuildResult:
    ensure_prop_offer_links_schema(conn)
    books = tuple(str(b).lower() for b in bookmakers)
    where_sql, params = _source_filter_sql(game_date=game_date, lookback_days=lookback_days)
    params["bookmakers"] = list(books)
    sql = f"""
    WITH source_rows AS (
        SELECT
            id AS source_row_id,
            as_of_date,
            fetched_at_utc,
            updated_at_utc,
            COALESCE(
                event_id,
                CASE
                    WHEN home_team IS NOT NULL AND away_team IS NOT NULL THEN
                        CONCAT_WS('|', 'fallback_event', as_of_date::text, LOWER(home_team), LOWER(away_team), COALESCE(commence_time_utc::text, ''))
                    ELSE CONCAT('source_row:', id::text)
                END
            ) AS event_id,
            commence_time_utc,
            LOWER(bookmaker_key) AS bookmaker_key,
            home_team,
            away_team,
            player_name,
            player_name_norm,
            stat,
            'over'::text AS side,
            line,
            over_price AS price,
            over_link AS link,
            (over_link IS NOT NULL AND over_link <> '') AS is_linkable
        FROM odds.mlb_player_prop_lines
        WHERE {where_sql}
        UNION ALL
        SELECT
            id AS source_row_id,
            as_of_date,
            fetched_at_utc,
            updated_at_utc,
            COALESCE(
                event_id,
                CASE
                    WHEN home_team IS NOT NULL AND away_team IS NOT NULL THEN
                        CONCAT_WS('|', 'fallback_event', as_of_date::text, LOWER(home_team), LOWER(away_team), COALESCE(commence_time_utc::text, ''))
                    ELSE CONCAT('source_row:', id::text)
                END
            ) AS event_id,
            commence_time_utc,
            LOWER(bookmaker_key) AS bookmaker_key,
            home_team,
            away_team,
            player_name,
            player_name_norm,
            stat,
            'under'::text AS side,
            line,
            under_price AS price,
            under_link AS link,
            (under_link IS NOT NULL AND under_link <> '') AS is_linkable
        FROM odds.mlb_player_prop_lines
        WHERE {where_sql}
    ),
    ranked AS (
        SELECT DISTINCT ON (as_of_date, event_id, player_name_norm, stat, bookmaker_key, line, side)
            *
        FROM source_rows
        WHERE price IS NOT NULL OR is_linkable
        ORDER BY
            as_of_date, event_id, player_name_norm, stat, bookmaker_key, line, side,
            is_linkable DESC,
            updated_at_utc DESC NULLS LAST,
            fetched_at_utc DESC NULLS LAST,
            source_row_id DESC
    )
    INSERT INTO features.mlb_prop_offer_links (
        source_row_id, as_of_date, fetched_at_utc, updated_at_utc, event_id,
        commence_time_utc, bookmaker_key, home_team, away_team, player_name,
        player_name_norm, stat, side, line, price, link, is_linkable,
        refreshed_at_utc
    )
    SELECT
        source_row_id, as_of_date, fetched_at_utc, updated_at_utc, event_id,
        commence_time_utc, bookmaker_key, home_team, away_team, player_name,
        player_name_norm, stat, side, line, price, link, is_linkable,
        NOW()
    FROM ranked
    ON CONFLICT (as_of_date, event_id, player_name_norm, stat, bookmaker_key, line, side)
    DO UPDATE SET
        source_row_id = EXCLUDED.source_row_id,
        fetched_at_utc = EXCLUDED.fetched_at_utc,
        updated_at_utc = EXCLUDED.updated_at_utc,
        event_id = EXCLUDED.event_id,
        commence_time_utc = EXCLUDED.commence_time_utc,
        home_team = EXCLUDED.home_team,
        away_team = EXCLUDED.away_team,
        player_name = EXCLUDED.player_name,
        price = EXCLUDED.price,
        link = EXCLUDED.link,
        is_linkable = EXCLUDED.is_linkable,
        refreshed_at_utc = NOW()
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = int(cur.rowcount or 0)
    conn.commit()
    return PropOfferBuildResult(
        table=OFFER_TABLE,
        rows_upserted=rows,
        game_date=str(game_date) if game_date else None,
        lookback_days=lookback_days,
        bookmakers=books,
    )


def load_prop_offer_rows(
    conn,
    game_date: date,
    *,
    bookmakers: Iterable[str] = DEFAULT_BOOKMAKERS,
    refresh_if_empty: bool = True,
) -> list[dict[str, Any]]:
    ensure_prop_offer_links_schema(conn)
    books = tuple(str(b).lower() for b in bookmakers)
    params = {"game_date": game_date, "bookmakers": list(books)}
    sql = """
        SELECT
            o.id, o.source_row_id, o.as_of_date, o.bookmaker_key, o.player_name, o.player_name_norm, o.stat,
            o.side, o.line::float AS line, o.price::float AS price, o.link,
            o.is_linkable, o.fetched_at_utc, o.updated_at_utc, o.event_id, o.commence_time_utc,
            o.home_team, o.away_team,
            opening.open_price::float AS open_price,
            opening.open_line::float AS open_line,
            opening.open_exact_line,
            opening.open_snapshot_at_utc
        FROM features.mlb_prop_offer_links o
        LEFT JOIN LATERAL (
            SELECT
                CASE WHEN o.side = 'over' THEN s.over_price ELSE s.under_price END AS open_price,
                s.line AS open_line,
                (s.line = o.line) AS open_exact_line,
                s.snapshot_at_utc AS open_snapshot_at_utc
            FROM odds.mlb_player_prop_line_snapshots s
            WHERE s.snapshot_role = 'open'
              AND s.as_of_date = o.as_of_date
              AND COALESCE(s.event_id, '') = COALESCE(o.event_id, '')
              AND s.player_name_norm = o.player_name_norm
              AND s.stat = o.stat
              AND LOWER(s.bookmaker_key) = LOWER(o.bookmaker_key)
              AND s.snapshot_at_utc <= COALESCE(o.fetched_at_utc, o.updated_at_utc, NOW())
            ORDER BY CASE WHEN s.line = o.line THEN 0 ELSE 1 END, s.snapshot_at_utc DESC, s.id DESC
            LIMIT 1
        ) opening ON TRUE
        WHERE o.as_of_date = %(game_date)s
          AND o.bookmaker_key = ANY(%(bookmakers)s)
        ORDER BY o.player_name_norm, o.stat, o.line, o.side, o.bookmaker_key
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = [dict(row) for row in cur.fetchall()]
    if rows or not refresh_if_empty:
        return rows
    refresh_prop_offer_links(conn, game_date=game_date, lookback_days=None, bookmakers=books)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def _best_offer(
    offers: list[dict[str, Any]],
    *,
    side: str,
    line: float,
    preferred_books: tuple[str, ...],
) -> dict[str, Any] | None:
    side_offers = [
        o for o in offers
        if o.get("side") == side and _clean_float(o.get("line")) is not None
    ]
    if not side_offers:
        return None
    exact = [o for o in side_offers if abs(float(o["line"]) - float(line)) <= 1e-9]
    candidates = exact or sorted(side_offers, key=lambda o: abs(float(o["line"]) - float(line)))[:4]
    book_rank = {book: idx for idx, book in enumerate(preferred_books)}

    def _score(o: dict[str, Any]) -> tuple[int, int, float, int]:
        book = str(o.get("bookmaker_key") or "")
        price = _clean_float(o.get("price"))
        return (
            -book_rank.get(book, 99),
            1 if o.get("link") else 0,
            price if price is not None else -99999.0,
            1 if o.get("is_linkable") else 0,
        )

    return max(candidates, key=_score)


def _best_same_book_pair(
    offers: list[dict[str, Any]],
    *,
    line: float,
    preferred_books: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return a coherent over/under pair from one book when one exists."""
    book_rank = {book: idx for idx, book in enumerate(preferred_books)}
    books = sorted({
        str(o.get("bookmaker_key") or "").lower()
        for o in offers
        if _clean_float(o.get("line")) is not None
        and abs(float(o["line"]) - float(line)) <= 1e-9
    })
    pairs: list[tuple[tuple[int, int, int, float], dict[str, Any], dict[str, Any]]] = []
    for book in books:
        book_offers = [o for o in offers if str(o.get("bookmaker_key") or "").lower() == book]
        over = _best_offer(book_offers, side="over", line=line, preferred_books=(book,))
        under = _best_offer(book_offers, side="under", line=line, preferred_books=(book,))
        if not over or not under:
            continue
        if _clean_float(over.get("price")) is None or _clean_float(under.get("price")) is None:
            continue
        # Sort ascending: preferred book first, then linked pairs, then fresher ids.
        score = (
            book_rank.get(book, 99),
            -int(bool(over.get("link")) and bool(under.get("link"))),
            -int(bool(over.get("link")) or bool(under.get("link"))),
            -float(max(int(over.get("id") or 0), int(under.get("id") or 0))),
        )
        pairs.append((score, over, under))
    if not pairs:
        return None
    pairs.sort(key=lambda item: item[0])
    return pairs[0][1], pairs[0][2]


def filter_prop_offers_for_game(
    offers: list[dict[str, Any]],
    *,
    team_abbr: Any,
    opponent_abbr: Any,
    start_ts_utc: Any,
) -> list[dict[str, Any]]:
    """Keep only the event-level offers belonging to one predicted game."""
    team = str(team_abbr or "").upper()
    opponent = str(opponent_abbr or "").upper()
    if not team or not opponent:
        return []
    matchup = [
        dict(offer) for offer in offers
        if {str(offer.get("home_team") or "").upper(), str(offer.get("away_team") or "").upper()}
        == {team, opponent}
    ]
    if not matchup:
        return []
    event_ids = sorted({str(offer.get("event_id")) for offer in matchup if offer.get("event_id")})
    if len(event_ids) <= 1:
        return matchup

    start = pd.to_datetime(start_ts_utc, utc=True, errors="coerce")
    if pd.isna(start):
        return []
    event_distance: dict[str, float] = {}
    for offer in matchup:
        event_id = str(offer.get("event_id") or "")
        commence = pd.to_datetime(offer.get("commence_time_utc"), utc=True, errors="coerce")
        if not event_id or pd.isna(commence):
            continue
        distance = abs((commence - start).total_seconds())
        event_distance[event_id] = min(event_distance.get(event_id, float("inf")), distance)
    if not event_distance:
        return []
    nearest_event = min(event_distance, key=lambda event_id: (event_distance[event_id], event_id))
    return [offer for offer in matchup if str(offer.get("event_id") or "") == nearest_event]


def _canonical_line(key: tuple[str, str], offers: list[dict[str, Any]]) -> float | None:
    stat = key[1]
    lines = sorted({float(o["line"]) for o in offers if _clean_float(o.get("line")) is not None})
    if not lines:
        return None
    preferred = {
        "batter_hits": 0.5,
        "batter_total_bases": 1.5,
        "batter_home_runs": 0.5,
    }.get(stat)
    if preferred is not None and any(abs(line - preferred) <= 1e-9 for line in lines):
        return preferred

    dk_lines = sorted({
        float(o["line"]) for o in offers
        if o.get("bookmaker_key") == "draftkings" and _clean_float(o.get("line")) is not None
    })
    if dk_lines:
        return dk_lines[0]

    over_offers = [o for o in offers if o.get("side") == "over" and _clean_float(o.get("price")) is not None]
    if over_offers:
        best = min(over_offers, key=lambda o: abs(float(o["price"])))
        return float(best["line"])
    return lines[0]


def build_prop_line_map(offer_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in offer_rows:
        stat = _clean_text(row.get("stat"))
        name_norm_raw = _clean_text(row.get("player_name_norm")) or str(row.get("player_name") or "")
        name_norm = _normalize_name(name_norm_raw)
        if not stat or not name_norm:
            continue
        line = _clean_float(row.get("line"))
        if line is None:
            continue
        clean = dict(row)
        clean["id"] = row.get("id")
        clean["source_row_id"] = row.get("source_row_id")
        clean["player_name_norm"] = name_norm
        clean["stat"] = stat
        clean["side"] = str(row.get("side") or "").lower()
        clean["bookmaker_key"] = str(row.get("bookmaker_key") or "").lower()
        clean["line"] = line
        clean["price"] = _clean_float(row.get("price"))
        clean["link"] = _clean_text(row.get("link"))
        grouped.setdefault((name_norm, stat), []).append(clean)

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, offers in grouped.items():
        line = _canonical_line(key, offers)
        if line is None:
            continue
        paired = _best_same_book_pair(
            offers,
            line=line,
            preferred_books=("fanduel", "draftkings"),
        )
        if paired:
            over, under = paired
            pairing_mode = "same_book"
            # When DK provides the same-book pair (FD lacks an under for TB/HR),
            # prefer FD's over_link for betting while keeping DK prices for no-vig probability.
            if (over or {}).get("bookmaker_key") == "draftkings":
                fd_over = _best_offer(
                    [o for o in offers if str(o.get("bookmaker_key") or "").lower() == "fanduel"],
                    side="over", line=line, preferred_books=("fanduel",),
                )
                if fd_over and fd_over.get("link"):
                    over = dict(over)
                    over["link"] = fd_over["link"]
                    over["link_bookmaker_key"] = "fanduel"
        else:
            over = _best_offer(offers, side="over", line=line, preferred_books=("fanduel", "draftkings"))
            under = _best_offer(offers, side="under", line=line, preferred_books=("draftkings", "fanduel"))
            pairing_mode = "mixed_book_or_one_sided"
        if not over and not under:
            continue
        result[key] = {
            "line": line,
            "bookmaker_key": (over or under or {}).get("bookmaker_key"),
            "over_price": (over or {}).get("price"),
            "under_price": (under or {}).get("price"),
            "over_link": (over or {}).get("link"),
            "under_link": (under or {}).get("link"),
            "over_bookmaker_key": (over or {}).get("link_bookmaker_key") or (over or {}).get("bookmaker_key"),
            "under_bookmaker_key": (under or {}).get("bookmaker_key"),
            "under_link_book": (under or {}).get("bookmaker_key"),
            "over_offer_id": (over or {}).get("id"),
            "under_offer_id": (under or {}).get("id"),
            "over_offer_source_row_id": (over or {}).get("source_row_id"),
            "under_offer_source_row_id": (under or {}).get("source_row_id"),
            "pairing_mode": pairing_mode,
            "offers": sorted(
                offers,
                key=lambda o: (
                    float(o.get("line") or 0.0),
                    str(o.get("side") or ""),
                    str(o.get("bookmaker_key") or ""),
                    int(o.get("id") or 0),
                ),
            ),
            "offer_source": OFFER_TABLE,
        }
    return result


def build_prop_line_map_for_date(
    conn,
    game_date: date,
    *,
    bookmakers: Iterable[str] = DEFAULT_BOOKMAKERS,
) -> dict[tuple[str, str], dict[str, Any]]:
    refresh_prop_offer_links(
        conn,
        game_date=game_date,
        lookback_days=None,
        bookmakers=bookmakers,
    )
    offers = load_prop_offer_rows(
        conn,
        game_date,
        bookmakers=bookmakers,
        refresh_if_empty=False,
    )
    line_map = build_prop_line_map(offers)
    if not line_map:
        log.warning("No normalized prop offers available for %s", game_date)
    return line_map
