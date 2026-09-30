"""External-model pick ledger and agreement matching for MLB props.

This stores picks from tools such as Edge Terminal, Outlier, Rithmm, or any
CSV/manual export, then attaches same-side agreement metadata to our locked
prop predictions. The external feed is evidence for $1 micro testing only; it
does not reopen bankroll buckets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPORT_DIR = _REPO_ROOT / "reports"
_DEFAULT_IMPORT_DIR = _REPO_ROOT / "data" / "external_picks" / "mlb"
_SCHEMA_READY = False

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "platform": ("platform", "source", "model", "app", "service"),
    "external_pick_id": ("external_pick_id", "pick_id", "id", "alert_id"),
    "game_date_et": ("game_date_et", "game_date", "date", "event_date"),
    "observed_at_utc": ("observed_at_utc", "observed_at", "timestamp", "created_at", "lock_time"),
    "game_slug": ("game_slug", "event_slug", "game_id", "event_id"),
    "player_name": ("player_name", "player", "athlete", "name"),
    "team_abbr": ("team_abbr", "team", "player_team"),
    "market": ("market", "stat", "prop", "prop_type"),
    "side": ("side", "bet_side", "pick", "selection", "direction"),
    "bookmaker_key": ("bookmaker_key", "book", "sportsbook", "bookmaker"),
    "market_line": ("market_line", "line", "book_line", "prop_line"),
    "market_price": ("market_price", "price", "odds", "american_odds"),
    "external_probability": ("external_probability", "probability", "prob", "win_probability", "win_prob"),
    "external_ev": ("external_ev", "ev", "edge", "expected_value"),
    "external_grade": ("external_grade", "grade", "rating"),
    "external_confidence": ("external_confidence", "confidence", "stars", "score"),
    "is_value": ("is_value", "recommended", "is_pick", "value", "bettable"),
}

_MARKET_ALIASES = {
    "hits": "batter_hits",
    "hit": "batter_hits",
    "batter_hits": "batter_hits",
    "total_bases": "batter_total_bases",
    "total bases": "batter_total_bases",
    "tb": "batter_total_bases",
    "batter_total_bases": "batter_total_bases",
    "home_runs": "batter_home_runs",
    "home runs": "batter_home_runs",
    "homeruns": "batter_home_runs",
    "hr": "batter_home_runs",
    "batter_home_runs": "batter_home_runs",
    "strikeouts": "pitcher_strikeouts",
    "pitcher_strikeouts": "pitcher_strikeouts",
    "ks": "pitcher_strikeouts",
    "k": "pitcher_strikeouts",
}

_BOOK_ALIASES = {
    "dk": "draftkings",
    "draft kings": "draftkings",
    "draftkings": "draftkings",
    "fd": "fanduel",
    "fan duel": "fanduel",
    "fanduel": "fanduel",
}


def normalize_name(name: Any) -> str:
    if not name:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_name = nfkd.encode("ascii", errors="ignore").decode("ascii")
    cleaned = re.sub(r"[^a-z0-9\s]", "", ascii_name.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _text(value: Any) -> str:
    return str(value or "").strip()


def _key_text(value: Any) -> str:
    return _text(value).lower().replace("-", "_").replace(" ", "_")


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().replace("%", "")
        if not value:
            return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _clean_bool(value: Any, *, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "pick", "recommended", "value", "a", "b"}:
        return True
    if text in {"0", "false", "no", "n", "off", "avoid", "pass"}:
        return False
    return default


def _date_from_any(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    text = _text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _slug(value: Any) -> str:
    text = _text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _platform_from_path(path: Path, fallback: str | None = None) -> str | None:
    if fallback:
        return fallback
    parent = _slug(path.parent.name)
    if parent and parent not in {"mlb", "external_picks", "data", "imports", "exports"}:
        return parent
    stem = _slug(path.stem)
    stem = re.sub(r"(?:^|_)20\d{2}_\d{2}_\d{2}(?:_|$)", "_", stem)
    stem = re.sub(r"(?:^|_)(mlb|picks|props|export|external|value)(?:_|$)", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or None


def _env_csv_paths() -> list[Path]:
    raw = os.getenv("MLB_EXTERNAL_PICKS_CSVS") or os.getenv("MLB_EXTERNAL_PICKS_CSV") or ""
    paths: list[Path] = []
    for part in re.split(r"[;,]", raw):
        text = part.strip().strip('"')
        if text:
            paths.append(Path(text))
    return paths


def discover_external_pick_files(
    *,
    explicit_paths: Iterable[Path] | None = None,
    import_dir: Path | None = _DEFAULT_IMPORT_DIR,
    pattern: str = "*.csv",
) -> list[Path]:
    """Return de-duplicated CSV files from explicit paths plus the import dir."""
    found: list[Path] = []
    for path in explicit_paths or []:
        if path:
            found.append(Path(path))
    if import_dir is not None:
        root = Path(import_dir)
        root.mkdir(parents=True, exist_ok=True)
        if root.exists():
            iterator = root.rglob(pattern) if "**" not in pattern else root.glob(pattern)
            found.extend(path for path in iterator if path.is_file())
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in found:
        key = str(path.resolve()).lower() if path.exists() else str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _normalize_market(value: Any) -> str | None:
    key = _key_text(value)
    if not key:
        return None
    key_space = key.replace("_", " ")
    return _MARKET_ALIASES.get(key) or _MARKET_ALIASES.get(key_space) or key


def _normalize_side(value: Any) -> str | None:
    text = _key_text(value)
    if not text:
        return None
    if text.startswith("over") or text in {"o", "ov"}:
        return "over"
    if text.startswith("under") or text in {"u", "un"}:
        return "under"
    return None


def _normalize_book(value: Any) -> str | None:
    text = _text(value).lower().replace("_", " ")
    if not text:
        return None
    return _BOOK_ALIASES.get(text) or text.replace(" ", "")


def _grade_rank(value: Any) -> int:
    grade = _text(value).upper()
    if not grade:
        return 999
    first = grade[0]
    return {"A": 1, "B": 2, "C": 3, "D": 4, "F": 5}.get(first, 999)


def _line_equal(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    l_val = _clean_float(left)
    r_val = _clean_float(right)
    return l_val is not None and r_val is not None and abs(l_val - r_val) <= tolerance


def _stable_external_key(row: Mapping[str, Any]) -> str:
    raw = "|".join(
        _text(row.get(key)).lower()
        for key in (
            "platform",
            "external_pick_id",
            "game_date_et",
            "game_slug",
            "player_name_norm",
            "market",
            "side",
            "bookmaker_key",
            "market_line",
            "observed_at_utc",
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _table_exists(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('bets.mlb_external_pick_ledger')")
        return cur.fetchone()[0] is not None


def ensure_external_pick_ledger_schema(conn) -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with conn.cursor() as cur:
        cur.execute("SET LOCAL lock_timeout = '2s'")
        cur.execute("SET LOCAL statement_timeout = '30s'")
        cur.execute(
            """
            CREATE SCHEMA IF NOT EXISTS bets;
            CREATE TABLE IF NOT EXISTS bets.mlb_external_pick_ledger (
                id BIGSERIAL PRIMARY KEY,
                external_key TEXT NOT NULL UNIQUE,
                inserted_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                observed_at_utc TIMESTAMPTZ,
                sport TEXT NOT NULL DEFAULT 'mlb',
                platform TEXT NOT NULL,
                external_pick_id TEXT,
                game_date_et DATE NOT NULL,
                game_slug TEXT,
                player_id BIGINT,
                player_name TEXT,
                player_name_norm TEXT,
                team_abbr TEXT,
                market TEXT NOT NULL,
                side TEXT NOT NULL CHECK (side IN ('over', 'under')),
                bookmaker_key TEXT,
                market_line NUMERIC,
                market_price NUMERIC,
                external_probability NUMERIC,
                external_ev NUMERIC,
                external_grade TEXT,
                external_confidence NUMERIC,
                is_value BOOLEAN NOT NULL DEFAULT TRUE,
                source_file TEXT,
                raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb
            );
            ALTER TABLE bets.mlb_external_pick_ledger
                ADD COLUMN IF NOT EXISTS external_key TEXT,
                ADD COLUMN IF NOT EXISTS observed_at_utc TIMESTAMPTZ,
                ADD COLUMN IF NOT EXISTS sport TEXT NOT NULL DEFAULT 'mlb',
                ADD COLUMN IF NOT EXISTS platform TEXT,
                ADD COLUMN IF NOT EXISTS external_pick_id TEXT,
                ADD COLUMN IF NOT EXISTS game_date_et DATE,
                ADD COLUMN IF NOT EXISTS game_slug TEXT,
                ADD COLUMN IF NOT EXISTS player_id BIGINT,
                ADD COLUMN IF NOT EXISTS player_name TEXT,
                ADD COLUMN IF NOT EXISTS player_name_norm TEXT,
                ADD COLUMN IF NOT EXISTS team_abbr TEXT,
                ADD COLUMN IF NOT EXISTS market TEXT,
                ADD COLUMN IF NOT EXISTS side TEXT,
                ADD COLUMN IF NOT EXISTS bookmaker_key TEXT,
                ADD COLUMN IF NOT EXISTS market_line NUMERIC,
                ADD COLUMN IF NOT EXISTS market_price NUMERIC,
                ADD COLUMN IF NOT EXISTS external_probability NUMERIC,
                ADD COLUMN IF NOT EXISTS external_ev NUMERIC,
                ADD COLUMN IF NOT EXISTS external_grade TEXT,
                ADD COLUMN IF NOT EXISTS external_confidence NUMERIC,
                ADD COLUMN IF NOT EXISTS is_value BOOLEAN NOT NULL DEFAULT TRUE,
                ADD COLUMN IF NOT EXISTS source_file TEXT,
                ADD COLUMN IF NOT EXISTS raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb;
            CREATE INDEX IF NOT EXISTS idx_mlb_external_pick_ledger_date
                ON bets.mlb_external_pick_ledger (game_date_et);
            CREATE INDEX IF NOT EXISTS idx_mlb_external_pick_ledger_match
                ON bets.mlb_external_pick_ledger
                (game_date_et, player_name_norm, market, side, bookmaker_key, market_line);
            """
        )
    _SCHEMA_READY = True


def _lookup(raw: Mapping[str, Any], canonical: str) -> Any:
    lowered = {_key_text(key): value for key, value in raw.items()}
    for alias in _FIELD_ALIASES.get(canonical, (canonical,)):
        key = _key_text(alias)
        if key in lowered:
            return lowered[key]
    return None


def normalize_external_pick(raw: Mapping[str, Any], *, platform: str | None = None, source_file: str | None = None) -> dict[str, Any] | None:
    market = _normalize_market(_lookup(raw, "market"))
    side = _normalize_side(_lookup(raw, "side"))
    game_date = _lookup(raw, "game_date_et")
    player_name = _text(_lookup(raw, "player_name"))
    if not market or side not in {"over", "under"} or not game_date:
        return None
    probability = _clean_float(_lookup(raw, "external_probability"))
    if probability is not None and probability > 1.0:
        probability /= 100.0
    row = {
        "platform": _text(platform or _lookup(raw, "platform") or "external"),
        "external_pick_id": _text(_lookup(raw, "external_pick_id")) or None,
        "game_date_et": game_date,
        "observed_at_utc": _lookup(raw, "observed_at_utc") or None,
        "game_slug": _text(_lookup(raw, "game_slug")) or None,
        "player_id": None,
        "player_name": player_name or None,
        "player_name_norm": normalize_name(player_name),
        "team_abbr": _text(_lookup(raw, "team_abbr")).upper() or None,
        "market": market,
        "side": side,
        "bookmaker_key": _normalize_book(_lookup(raw, "bookmaker_key")),
        "market_line": _clean_float(_lookup(raw, "market_line")),
        "market_price": _clean_float(_lookup(raw, "market_price")),
        "external_probability": probability,
        "external_ev": _clean_float(_lookup(raw, "external_ev")),
        "external_grade": _text(_lookup(raw, "external_grade")) or None,
        "external_confidence": _clean_float(_lookup(raw, "external_confidence")),
        "is_value": _clean_bool(_lookup(raw, "is_value"), default=True),
        "source_file": source_file,
        "raw_payload": dict(raw),
    }
    row["external_key"] = _stable_external_key(row)
    return row


_INSERT_SQL = """
INSERT INTO bets.mlb_external_pick_ledger (
    external_key, observed_at_utc, sport, platform, external_pick_id,
    game_date_et, game_slug, player_id, player_name, player_name_norm,
    team_abbr, market, side, bookmaker_key, market_line, market_price,
    external_probability, external_ev, external_grade, external_confidence,
    is_value, source_file, raw_payload
) VALUES (
    %(external_key)s, %(observed_at_utc)s, 'mlb', %(platform)s, %(external_pick_id)s,
    %(game_date_et)s, %(game_slug)s, %(player_id)s, %(player_name)s, %(player_name_norm)s,
    %(team_abbr)s, %(market)s, %(side)s, %(bookmaker_key)s, %(market_line)s, %(market_price)s,
    %(external_probability)s, %(external_ev)s, %(external_grade)s, %(external_confidence)s,
    %(is_value)s, %(source_file)s, %(raw_payload)s
) ON CONFLICT (external_key) DO NOTHING
"""


def insert_external_picks(conn, rows: Iterable[Mapping[str, Any]], *, setup_schema: bool = True) -> int:
    payload = [dict(row) for row in rows if row]
    if not payload:
        return 0
    if setup_schema:
        ensure_external_pick_ledger_schema(conn)
    inserted = 0
    with conn.cursor() as cur:
        for row in payload:
            row["raw_payload"] = psycopg2.extras.Json(row.get("raw_payload") or {})
            cur.execute(_INSERT_SQL + " RETURNING id", row)
            if cur.fetchone() is not None:
                inserted += 1
    conn.commit()
    return inserted


def import_external_csv(
    conn,
    path: Path,
    *,
    platform: str | None = None,
    game_date: date | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {
            "status": "missing",
            "source_file": str(path),
            "platform": platform or _platform_from_path(path),
            "normalized_rows": 0,
            "skipped_rows": 0,
            "inserted_rows": 0,
        }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        normalized = [
            normalize_external_pick(
                row,
                platform=_platform_from_path(path, platform),
                source_file=str(path),
            )
            for row in reader
        ]
    rows = [
        row for row in normalized
        if row is not None
        and (game_date is None or _date_from_any(row.get("game_date_et")) == game_date)
    ]
    inserted = insert_external_picks(conn, rows)
    return {
        "status": "ok",
        "source_file": str(path),
        "platform": platform or _platform_from_path(path),
        "normalized_rows": len(rows),
        "skipped_rows": len(normalized) - len(rows),
        "inserted_rows": inserted,
    }


def import_external_sources(
    conn,
    *,
    game_date: date,
    csv_paths: Iterable[Path] | None = None,
    import_dir: Path | None = _DEFAULT_IMPORT_DIR,
    pattern: str = "*.csv",
    platform: str | None = None,
) -> dict[str, Any]:
    """Import all configured external-pick files for one slate date."""
    paths = discover_external_pick_files(
        explicit_paths=[*_env_csv_paths(), *(csv_paths or [])],
        import_dir=import_dir,
        pattern=pattern,
    )
    files: list[dict[str, Any]] = []
    totals = Counter()
    for path in paths:
        result = import_external_csv(conn, path, platform=platform, game_date=game_date)
        files.append(result)
        totals["files_checked"] += 1
        if result.get("status") == "ok":
            totals["files_imported"] += 1
        elif result.get("status") == "missing":
            totals["files_missing"] += 1
        totals["normalized_rows"] += int(result.get("normalized_rows") or 0)
        totals["skipped_rows"] += int(result.get("skipped_rows") or 0)
        totals["inserted_rows"] += int(result.get("inserted_rows") or 0)
    return {
        "status": "ok",
        "game_date": game_date.isoformat(),
        "import_dir": str(import_dir) if import_dir is not None else None,
        "pattern": pattern,
        "files": files,
        "files_checked": int(totals.get("files_checked", 0)),
        "files_imported": int(totals.get("files_imported", 0)),
        "files_missing": int(totals.get("files_missing", 0)),
        "normalized_rows": int(totals.get("normalized_rows", 0)),
        "skipped_rows": int(totals.get("skipped_rows", 0)),
        "inserted_rows": int(totals.get("inserted_rows", 0)),
    }


def _load_external_rows(conn, game_date: date) -> list[dict[str, Any]]:
    if not _table_exists(conn):
        return []
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                id, platform, external_pick_id, game_date_et, game_slug,
                player_id, player_name, player_name_norm, team_abbr,
                market, side, bookmaker_key, market_line::float AS market_line,
                market_price::float AS market_price,
                external_probability::float AS external_probability,
                external_ev::float AS external_ev,
                external_grade, external_confidence::float AS external_confidence,
                is_value, observed_at_utc, source_file
            FROM bets.mlb_external_pick_ledger
            WHERE sport = 'mlb'
              AND game_date_et = %s
              AND COALESCE(is_value, true) IS TRUE
            """,
            (game_date,),
        )
        return [dict(row) for row in cur.fetchall()]


def _row_date(row: Mapping[str, Any], fallback: date | None = None) -> date | None:
    value = row.get("game_date_et") or fallback
    if isinstance(value, date):
        return value
    if value:
        return date.fromisoformat(str(value)[:10])
    return None


def _same_game(pred: Mapping[str, Any], ext: Mapping[str, Any]) -> bool:
    pred_slug = _text(pred.get("game_slug"))
    ext_slug = _text(ext.get("game_slug"))
    if pred_slug and ext_slug and pred_slug != ext_slug:
        return False
    pred_team = _text(pred.get("team_abbr")).upper()
    ext_team = _text(ext.get("team_abbr")).upper()
    return not pred_team or not ext_team or pred_team == ext_team


def _match_level(pred: Mapping[str, Any], ext: Mapping[str, Any]) -> tuple[str | None, float]:
    if normalize_name(pred.get("player_name")) != _text(ext.get("player_name_norm")):
        return None, 0.0
    market = _normalize_market(pred.get("stat") or pred.get("market"))
    if market != _text(ext.get("market")):
        return None, 0.0
    if _normalize_side(pred.get("bet_side") or pred.get("side")) != _text(ext.get("side")):
        return None, 0.0
    if not _same_game(pred, ext):
        return None, 0.0
    pred_book = _normalize_book(pred.get("bookmaker_key"))
    ext_book = _normalize_book(ext.get("bookmaker_key"))
    exact_line = _line_equal(pred.get("book_line") or pred.get("market_line"), ext.get("market_line"))
    if exact_line and pred_book and ext_book and pred_book == ext_book:
        return "exact_line_same_book", 1.0
    if exact_line and (not ext_book or not pred_book):
        return "exact_line_bookless", 0.90
    if exact_line:
        return "exact_line_cross_book", 0.75
    return "player_market_side", 0.35


def _opposite_side_matches(pred: Mapping[str, Any], ext: Mapping[str, Any]) -> bool:
    if normalize_name(pred.get("player_name")) != _text(ext.get("player_name_norm")):
        return False
    market = _normalize_market(pred.get("stat") or pred.get("market"))
    if market != _text(ext.get("market")):
        return False
    pred_side = _normalize_side(pred.get("bet_side") or pred.get("side"))
    ext_side = _text(ext.get("side"))
    if pred_side not in {"over", "under"} or ext_side not in {"over", "under"} or pred_side == ext_side:
        return False
    if not _same_game(pred, ext):
        return False
    return _line_equal(pred.get("book_line") or pred.get("market_line"), ext.get("market_line"))


def external_agreement_for_row(row: Mapping[str, Any], external_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    matches: list[tuple[str, float, Mapping[str, Any]]] = []
    disagreements = 0
    for ext in external_rows:
        level, strength = _match_level(row, ext)
        if level:
            matches.append((level, strength, ext))
        elif _opposite_side_matches(row, ext):
            disagreements += 1
    matches.sort(
        key=lambda item: (
            item[1],
            -_grade_rank(item[2].get("external_grade")),
            _clean_float(item[2].get("external_ev")) or -999.0,
            _clean_float(item[2].get("external_probability")) or -999.0,
        ),
        reverse=True,
    )
    if not matches and not disagreements:
        return {
            "external_agreement_count": 0,
            "external_disagreement_count": 0,
            "external_agreement": False,
            "external_agreement_strength": 0.0,
            "external_match_level": None,
            "external_platforms": [],
            "external_agreements": [],
        }
    platforms = sorted({str(match[2].get("platform")) for match in matches if match[2].get("platform")})
    agreements = [
        {
            "id": match[2].get("id"),
            "platform": match[2].get("platform"),
            "match_level": match[0],
            "strength": match[1],
            "bookmaker_key": match[2].get("bookmaker_key"),
            "line": match[2].get("market_line"),
            "price": match[2].get("market_price"),
            "probability": match[2].get("external_probability"),
            "ev": match[2].get("external_ev"),
            "grade": match[2].get("external_grade"),
            "observed_at_utc": match[2].get("observed_at_utc"),
        }
        for match in matches[:5]
    ]
    best = matches[0] if matches else (None, 0.0, {})
    return {
        "external_agreement_count": len(matches),
        "external_disagreement_count": disagreements,
        "external_agreement": bool(matches),
        "external_agreement_strength": best[1],
        "external_match_level": best[0],
        "external_platforms": platforms,
        "external_best_grade": (best[2] or {}).get("external_grade"),
        "external_max_probability": max(
            (_clean_float(match[2].get("external_probability")) or 0.0)
            for match in matches
        ) if matches else None,
        "external_max_ev": max(
            (_clean_float(match[2].get("external_ev")) or -999.0)
            for match in matches
        ) if matches else None,
        "external_agreements": agreements,
    }


def attach_external_agreement(conn, rows: list[dict[str, Any]], *, game_date: date | None = None) -> None:
    if not rows:
        return
    dates = sorted({d for row in rows if (d := _row_date(row, game_date)) is not None})
    if not dates or not _table_exists(conn):
        return
    external_by_date = {day: _load_external_rows(conn, day) for day in dates}
    for row in rows:
        day = _row_date(row, game_date)
        agreement = external_agreement_for_row(row, external_by_date.get(day, []))
        row.update(agreement)


def build_comparison_payload(conn, game_date: date) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                game_date_et, game_slug, player_name, team_abbr, stat, bet_side,
                bookmaker_key, book_line::float AS book_line,
                bet_price::float AS bet_price,
                pred_prob_over::float AS pred_prob_over,
                ev::float AS ev,
                prediction_key,
                prop_offer_id
            FROM bets.mlb_prop_predictions
            WHERE game_date_et = %s
              AND COALESCE(is_active, true) IS TRUE
              AND bet_side IN ('over','under')
            """,
            (game_date,),
        )
        predictions = [dict(row) for row in cur.fetchall()]
    attach_external_agreement(conn, predictions, game_date=game_date)
    buckets = Counter()
    platform_counts = Counter()
    rows = []
    for row in predictions:
        agreement = bool(row.get("external_agreement"))
        disagreement = int(row.get("external_disagreement_count") or 0) > 0
        if agreement:
            bucket = "both_agree"
        elif disagreement:
            bucket = "external_disagrees"
        else:
            bucket = "our_only_or_no_external"
        buckets[bucket] += 1
        for platform in row.get("external_platforms") or []:
            platform_counts[str(platform)] += 1
        if agreement or disagreement:
            p_over = _clean_float(row.get("pred_prob_over"))
            side = row.get("bet_side")
            model_prob = p_over if side == "over" else (1.0 - p_over if p_over is not None else None)
            rows.append({
                "player_name": row.get("player_name"),
                "team_abbr": row.get("team_abbr"),
                "market": row.get("stat"),
                "side": side,
                "line": row.get("book_line"),
                "bookmaker_key": row.get("bookmaker_key"),
                "model_prob_side": model_prob,
                "ev": row.get("ev"),
                "comparison_bucket": bucket,
                "external_agreement_count": row.get("external_agreement_count"),
                "external_disagreement_count": row.get("external_disagreement_count"),
                "external_platforms": row.get("external_platforms"),
                "external_match_level": row.get("external_match_level"),
                "external_best_grade": row.get("external_best_grade"),
                "external_max_ev": row.get("external_max_ev"),
                "external_max_probability": row.get("external_max_probability"),
            })
    external_rows = _load_external_rows(conn, game_date)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "game_date": game_date.isoformat(),
        "active_prediction_rows": len(predictions),
        "external_value_rows": len(external_rows),
        "comparison_buckets": dict(buckets.most_common()),
        "agreement_platform_counts": dict(platform_counts.most_common()),
        "rows": rows[:250],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    numeric = _clean_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.{digits}f}"


def _render_comparison(payload: Mapping[str, Any]) -> str:
    import_summary = payload.get("import") or {}
    lines = [
        "# MLB External Model Comparison",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Date: {payload.get('game_date')}",
        "",
        "## Auto Import",
        "",
        f"- Import dir: `{import_summary.get('import_dir') or '-'}`",
        f"- Files checked: {import_summary.get('files_checked', 0)}",
        f"- Files imported: {import_summary.get('files_imported', 0)}",
        f"- Rows normalized for date: {import_summary.get('normalized_rows', 0)}",
        f"- Newly inserted rows: {import_summary.get('inserted_rows', 0)}",
        "",
        f"- Active prediction rows: {payload.get('active_prediction_rows', 0)}",
        f"- External value rows: {payload.get('external_value_rows', 0)}",
        "",
        "## Buckets",
        "",
        "| Bucket | Rows |",
        "|---|---:|",
    ]
    for bucket, count in (payload.get("comparison_buckets") or {}).items():
        lines.append(f"| {bucket} | {count} |")
    lines.extend([
        "",
        "## Agreement Rows",
        "",
        "| Player | Market | Side | Line | Book | Ours P | Ours EV | Platforms | Match | Grade | Ext EV |",
        "|---|---|---|---:|---|---:|---:|---|---|---|---:|",
    ])
    rows = [row for row in payload.get("rows") or [] if row.get("comparison_bucket") == "both_agree"]
    rows.sort(key=lambda row: (_clean_float(row.get("ev")) or -999.0), reverse=True)
    for row in rows[:80]:
        platforms = ", ".join(row.get("external_platforms") or [])
        lines.append(
            f"| {row.get('player_name')} | {row.get('market')} | {row.get('side')} | {_fmt(row.get('line'), 1)} | "
            f"{row.get('bookmaker_key') or '-'} | {_fmt(row.get('model_prob_side'))} | {_fmt(row.get('ev'))} | "
            f"{platforms or '-'} | {row.get('external_match_level') or '-'} | {row.get('external_best_grade') or '-'} | "
            f"{_fmt(row.get('external_max_ev'))} |"
        )
    return "\n".join(lines) + "\n"


def write_comparison_outputs(payload: dict[str, Any], *, model_dir: Path = _MODEL_DIR, report_dir: Path = _REPORT_DIR) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = model_dir / "external_model_comparison.json"
    report_path = report_dir / "mlb_external_model_comparison_latest.md"
    atomic_write_json(json_path, payload, default=str)
    atomic_write_text(report_path, _render_comparison(payload))
    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Import and compare external MLB prop picks.")
    parser.add_argument("--date", default=os.getenv("MLB_ET_DATE"), help="Slate date in YYYY-MM-DD ET")
    parser.add_argument(
        "--csv",
        dest="csv_paths",
        action="append",
        type=Path,
        default=[],
        help="Optional CSV export of external value picks. Can be passed more than once.",
    )
    parser.add_argument(
        "--import-dir",
        type=Path,
        default=Path(os.getenv("MLB_EXTERNAL_PICKS_DIR", str(_DEFAULT_IMPORT_DIR))),
        help="Directory scanned for external-pick CSV files.",
    )
    parser.add_argument(
        "--glob",
        default=os.getenv("MLB_EXTERNAL_PICKS_GLOB", "*.csv"),
        help="CSV filename glob inside --import-dir.",
    )
    parser.add_argument(
        "--no-import-dir",
        action="store_true",
        help="Only import explicit --csv/env CSV files; do not scan the import directory.",
    )
    parser.add_argument(
        "--skip-import",
        action="store_true",
        help="Do not import files; only rebuild the comparison report from the DB ledger.",
    )
    parser.add_argument(
        "--platform",
        default=os.getenv("MLB_EXTERNAL_PICKS_PLATFORM"),
        help="Platform/source label for CSV import, e.g. outlier or edge_terminal",
    )
    parser.add_argument("--pg-dsn", default=PG_DSN)
    args = parser.parse_args()

    if args.date:
        game_date = date.fromisoformat(args.date)
    else:
        game_date = datetime.now(ZoneInfo("America/New_York")).date()
    with psycopg2.connect(args.pg_dsn) as conn:
        import_result = None
        if not args.skip_import:
            import_result = import_external_sources(
                conn,
                game_date=game_date,
                csv_paths=args.csv_paths,
                import_dir=None if args.no_import_dir else args.import_dir,
                pattern=args.glob,
                platform=args.platform,
            )
        payload = build_comparison_payload(conn, game_date)
        if import_result is not None:
            payload["import"] = import_result
        json_path, report_path = write_comparison_outputs(payload)
    print(json.dumps({
        "status": "ok",
        "game_date": game_date.isoformat(),
        "comparison_json": str(json_path),
        "comparison_report": str(report_path),
        "external_value_rows": payload.get("external_value_rows", 0),
        "both_agree": (payload.get("comparison_buckets") or {}).get("both_agree", 0),
        "import": import_result,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
