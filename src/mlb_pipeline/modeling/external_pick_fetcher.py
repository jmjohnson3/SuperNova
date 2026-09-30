"""Fetch external MLB pick exports into the local import folder.

This module intentionally avoids scraping private app screens. It supports
direct CSV/JSON/HTML-table feeds that the user is allowed to access, including
Google Sheet CSV export URLs and authenticated vendor endpoints configured via
environment variables or a local JSON config file.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN
from mlb_pipeline.modeling.external_pick_ledger import (
    _date_from_any,
    _slug,
    import_external_sources,
    normalize_external_pick,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_IMPORT_DIR = _REPO_ROOT / "data" / "external_picks" / "mlb"
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "mlb_external_pick_sources.json"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = _REPO_ROOT / "reports"

_SECRET_QUERY_KEYS = {
    "api_key",
    "apikey",
    "key",
    "token",
    "access_token",
    "auth",
    "authorization",
    "password",
    "secret",
}

_CANONICAL_COLUMNS = (
    "platform",
    "external_pick_id",
    "game_date_et",
    "observed_at_utc",
    "game_slug",
    "player_name",
    "team_abbr",
    "market",
    "side",
    "bookmaker_key",
    "market_line",
    "market_price",
    "external_probability",
    "external_ev",
    "external_grade",
    "external_confidence",
    "is_value",
    "source_url",
    "source_name",
)


@dataclass(frozen=True)
class ExternalPickSource:
    platform: str
    url: str
    format: str = "auto"
    name: str | None = None
    method: str = "GET"
    headers: Mapping[str, str] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)
    json_path: str | None = None
    table_index: int = 0
    enabled: bool = True

    @property
    def source_name(self) -> str:
        return self.name or self.platform


def _expand_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return os.path.expandvars(value)


def _expand_mapping(values: Mapping[str, Any] | None) -> dict[str, str]:
    expanded: dict[str, str] = {}
    for key, value in (values or {}).items():
        text = _expand_env(value)
        if text is None:
            continue
        expanded[str(key)] = str(text)
    return expanded


def _platform_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower().split("@")[-1].split(":")[0]
    host = re.sub(r"^www\.", "", host)
    if not host:
        return "external"
    pieces = host.split(".")
    return _slug(pieces[0] if pieces else host) or "external"


def _source_from_mapping(raw: Mapping[str, Any]) -> ExternalPickSource | None:
    if raw.get("enabled") is False:
        return None
    url = str(_expand_env(raw.get("url") or "")).strip()
    if not url:
        return None
    platform = str(raw.get("platform") or _platform_from_url(url)).strip() or "external"
    return ExternalPickSource(
        platform=_slug(platform) or "external",
        url=url,
        format=str(raw.get("format") or "auto").strip().lower(),
        name=str(raw.get("name") or "").strip() or None,
        method=str(raw.get("method") or "GET").strip().upper(),
        headers=_expand_mapping(raw.get("headers") or {}),
        params=_expand_mapping(raw.get("params") or {}),
        json_path=str(raw.get("json_path") or "").strip() or None,
        table_index=int(raw.get("table_index") or 0),
        enabled=bool(raw.get("enabled", True)),
    )


def load_sources_from_config(path: Path | None = _DEFAULT_CONFIG_PATH) -> list[ExternalPickSource]:
    """Load source definitions from JSON.

    Accepted formats:
      {"sources": [{...}, {...}]}
      [{...}, {...}]
    """
    if path is None or not Path(path).exists():
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_sources = payload.get("sources", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_sources, list):
        raise ValueError(f"Expected a list of sources in {path}")
    sources: list[ExternalPickSource] = []
    for raw in raw_sources:
        if not isinstance(raw, Mapping):
            continue
        source = _source_from_mapping(raw)
        if source:
            sources.append(source)
    return sources


def _global_headers_from_env() -> dict[str, str]:
    raw = os.getenv("MLB_EXTERNAL_PICK_FETCH_HEADERS_JSON", "").strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, Mapping):
        raise ValueError("MLB_EXTERNAL_PICK_FETCH_HEADERS_JSON must be a JSON object")
    return _expand_mapping(parsed)


def sources_from_env_text(raw: str | None) -> list[ExternalPickSource]:
    """Parse semicolon/newline-separated env source specs.

    Supported forms:
      platform|format|https://example.com/export.csv
      platform|https://example.com/export.csv
      https://example.com/export.csv
    """
    text = (raw or "").strip()
    if not text:
        return []
    global_headers = _global_headers_from_env()
    sources: list[ExternalPickSource] = []
    for chunk in re.split(r"[;\n]+", text):
        spec = chunk.strip()
        if not spec:
            continue
        parts = [part.strip() for part in spec.split("|")]
        platform = ""
        fmt = "auto"
        url = ""
        if len(parts) >= 3:
            platform, fmt, url = parts[0], parts[1], "|".join(parts[2:])
        elif len(parts) == 2:
            platform, url = parts
        else:
            url = parts[0]
            platform = _platform_from_url(url)
        url = _expand_env(url)
        if not url:
            continue
        sources.append(ExternalPickSource(
            platform=_slug(platform or _platform_from_url(url)) or "external",
            url=str(url),
            format=fmt.lower() or "auto",
            headers=global_headers,
        ))
    return sources


def load_sources(*, config_path: Path | None = _DEFAULT_CONFIG_PATH, cli_sources: Iterable[str] = ()) -> list[ExternalPickSource]:
    env_raw = os.getenv("MLB_EXTERNAL_PICK_SOURCE_URLS") or os.getenv("MLB_EXTERNAL_PICK_FEED_URLS")
    sources = [
        *load_sources_from_config(config_path),
        *sources_from_env_text(env_raw),
        *sources_from_env_text("\n".join(cli_sources)),
    ]
    deduped: list[ExternalPickSource] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        key = (source.platform.lower(), source.url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _redact_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.query:
        return url
    query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() in _SECRET_QUERY_KEYS or any(secret in key.lower() for secret in _SECRET_QUERY_KEYS):
            query.append((key, "***"))
        else:
            query.append((key, value))
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def _guess_format(source: ExternalPickSource, content_type: str, text: str) -> str:
    if source.format and source.format != "auto":
        return source.format
    lowered_type = content_type.lower()
    path = urlparse(source.url).path.lower()
    stripped = text.lstrip()
    if "json" in lowered_type or path.endswith(".json") or stripped.startswith(("{", "[")):
        return "json"
    if "html" in lowered_type or path.endswith((".html", ".htm")) or stripped.startswith("<"):
        return "html"
    return "csv"


def _walk_json_path(payload: Any, path: str | None) -> Any:
    if not path:
        return payload
    current = payload
    cleaned = path.strip()
    if cleaned.startswith("$."):
        cleaned = cleaned[2:]
    elif cleaned.startswith("$"):
        cleaned = cleaned[1:].lstrip(".")
    for part in filter(None, cleaned.split(".")):
        match = re.fullmatch(r"([^\[]+)(?:\[(\d+)\])?", part)
        if not match:
            return None
        key, index_text = match.groups()
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            return None
        if index_text is not None:
            if not isinstance(current, list):
                return None
            idx = int(index_text)
            if idx >= len(current):
                return None
            current = current[idx]
    return current


def _first_list_of_dicts(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        if all(isinstance(row, Mapping) for row in payload):
            return list(payload)
        for item in payload:
            found = _first_list_of_dicts(item)
            if found:
                return found
    if isinstance(payload, Mapping):
        preferred = ("picks", "props", "bets", "data", "results", "items", "value_bets", "events")
        for key in preferred:
            found = _first_list_of_dicts(payload.get(key))
            if found:
                return found
        for value in payload.values():
            found = _first_list_of_dicts(value)
            if found:
                return found
    return []


def _rows_from_response(source: ExternalPickSource, text: str, content_type: str) -> tuple[str, list[Mapping[str, Any]]]:
    fmt = _guess_format(source, content_type, text)
    if fmt == "csv":
        rows = list(csv.DictReader(io.StringIO(text)))
        return fmt, rows
    if fmt == "json":
        payload = json.loads(text)
        selected = _walk_json_path(payload, source.json_path)
        rows = _first_list_of_dicts(selected)
        return fmt, rows
    if fmt in {"html", "html_table", "table"}:
        import pandas as pd

        tables = pd.read_html(io.StringIO(text))
        if not tables:
            return fmt, []
        idx = min(max(source.table_index, 0), len(tables) - 1)
        rows = tables[idx].where(tables[idx].notna(), None).to_dict(orient="records")
        return fmt, rows
    raise ValueError(f"Unsupported external pick format: {fmt}")


def _fetch_text(source: ExternalPickSource, *, timeout_s: int, retries: int) -> tuple[str, str]:
    last_error: Exception | None = None
    headers = dict(source.headers or {})
    headers.setdefault("User-Agent", "SuperNovaBets external-pick fetcher/1.0")
    method = source.method.upper()
    for attempt in range(max(1, retries)):
        try:
            response = requests.request(
                method,
                source.url,
                headers=headers,
                params=dict(source.params or {}),
                timeout=timeout_s,
            )
            response.raise_for_status()
            return response.text, response.headers.get("content-type", "")
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= max(1, retries):
                break
            time.sleep(min(2.0 * (attempt + 1), 8.0))
    assert last_error is not None
    raise RuntimeError(f"External pick fetch failed for {source.source_name}: {last_error}") from last_error


def _canonical_csv_path(import_dir: Path, source: ExternalPickSource, game_date: date) -> Path:
    platform = _slug(source.platform) or "external"
    source_name = _slug(source.source_name) or platform
    suffix = "" if source_name == platform else f"_{source_name}"
    return import_dir / platform / f"{game_date.isoformat()}{suffix}.csv"


def _normalized_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    source: ExternalPickSource,
    game_date: date,
) -> tuple[list[dict[str, Any]], int]:
    normalized: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        enriched = dict(row)
        enriched.setdefault("platform", source.platform)
        pick = normalize_external_pick(enriched, platform=source.platform, source_file=source.url)
        if pick is None:
            skipped += 1
            continue
        if _date_from_any(pick.get("game_date_et")) != game_date:
            skipped += 1
            continue
        pick["source_url"] = source.url
        pick["source_name"] = source.source_name
        normalized.append(pick)
    return normalized, skipped


def write_canonical_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    serializable_rows: list[dict[str, Any]] = []
    for row in rows:
        serializable_rows.append({
            column: row.get(column)
            for column in _CANONICAL_COLUMNS
        })

    def render() -> str:
        handle = io.StringIO()
        writer = csv.DictWriter(handle, fieldnames=list(_CANONICAL_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(serializable_rows)
        return handle.getvalue()

    return atomic_write_text(path, render())


def fetch_external_picks(
    *,
    game_date: date,
    sources: Iterable[ExternalPickSource],
    import_dir: Path = _DEFAULT_IMPORT_DIR,
    timeout_s: int = 30,
    retries: int = 3,
    dry_run: bool = False,
) -> dict[str, Any]:
    import_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "game_date": game_date.isoformat(),
        "import_dir": str(import_dir),
        "dry_run": dry_run,
        "sources": [],
    }
    totals = Counter()
    for source in sources:
        result: dict[str, Any] = {
            "platform": source.platform,
            "name": source.source_name,
            "url": _redact_url(source.url),
            "status": "pending",
            "raw_rows": 0,
            "normalized_rows": 0,
            "skipped_rows": 0,
            "written_csv": None,
        }
        try:
            text, content_type = _fetch_text(source, timeout_s=timeout_s, retries=retries)
            fmt, raw_rows = _rows_from_response(source, text, content_type)
            normalized, skipped = _normalized_rows(raw_rows, source=source, game_date=game_date)
            result.update({
                "status": "ok",
                "format": fmt,
                "raw_rows": len(raw_rows),
                "normalized_rows": len(normalized),
                "skipped_rows": skipped,
            })
            if normalized and not dry_run:
                output = _canonical_csv_path(import_dir, source, game_date)
                write_canonical_csv(output, normalized)
                result["written_csv"] = str(output)
        except Exception as exc:
            result.update({
                "status": "failed",
                "error": str(exc),
            })
        totals["sources"] += 1
        totals[f"status_{result['status']}"] += 1
        totals["raw_rows"] += int(result.get("raw_rows") or 0)
        totals["normalized_rows"] += int(result.get("normalized_rows") or 0)
        totals["skipped_rows"] += int(result.get("skipped_rows") or 0)
        if result.get("written_csv"):
            totals["files_written"] += 1
        payload["sources"].append(result)
    if totals["sources"] == 0:
        payload["status"] = "no_sources_configured"
    payload["summary"] = dict(totals)
    return payload


def _render_report(payload: Mapping[str, Any]) -> str:
    summary = payload.get("summary") or {}
    lines = [
        "# MLB External Pick Fetch",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Date: {payload.get('game_date')}",
        f"Status: {payload.get('status')}",
        f"Import dir: `{payload.get('import_dir')}`",
        "",
        "## Summary",
        "",
        f"- Sources configured: {summary.get('sources', 0)}",
        f"- Sources succeeded: {summary.get('status_ok', 0)}",
        f"- Sources failed: {summary.get('status_failed', 0)}",
        f"- Raw rows: {summary.get('raw_rows', 0)}",
        f"- Normalized rows for date: {summary.get('normalized_rows', 0)}",
        f"- CSV files written: {summary.get('files_written', 0)}",
        "",
        "## Sources",
        "",
        "| Platform | Source | Status | Raw | Normalized | Skipped | Written CSV | Error |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for source in payload.get("sources") or []:
        error = str(source.get("error") or "").replace("|", "\\|")
        written = source.get("written_csv") or "-"
        lines.append(
            f"| {source.get('platform')} | {source.get('name')} | {source.get('status')} | "
            f"{source.get('raw_rows', 0)} | {source.get('normalized_rows', 0)} | "
            f"{source.get('skipped_rows', 0)} | `{written}` | {error or '-'} |"
        )
    lines.extend([
        "",
        "## Configuration",
        "",
        "Use direct export/feed URLs only. Examples:",
        "",
        "```powershell",
        '$env:MLB_EXTERNAL_PICK_SOURCE_URLS="outlier|csv|https://docs.google.com/spreadsheets/d/.../export?format=csv"',
        "```",
        "",
        "or create `config/mlb_external_pick_sources.json` from the docs template.",
    ])
    return "\n".join(lines) + "\n"


def write_fetch_outputs(payload: dict[str, Any], *, model_dir: Path = _MODEL_DIR, report_dir: Path = _REPORT_DIR) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = model_dir / "external_pick_fetch.json"
    report_path = report_dir / "mlb_external_pick_fetch_latest.md"
    atomic_write_json(json_path, payload, default=str)
    atomic_write_text(report_path, _render_report(payload))
    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch external MLB pick feeds into the local CSV import folder.")
    parser.add_argument("--date", default=os.getenv("MLB_ET_DATE"), help="Slate date in YYYY-MM-DD ET")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.getenv("MLB_EXTERNAL_PICK_SOURCES_JSON", str(_DEFAULT_CONFIG_PATH))),
        help="JSON source config. Missing config is allowed.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source spec: platform|format|url, platform|url, or just url. Can be passed more than once.",
    )
    parser.add_argument(
        "--import-dir",
        type=Path,
        default=Path(os.getenv("MLB_EXTERNAL_PICKS_DIR", str(_DEFAULT_IMPORT_DIR))),
    )
    parser.add_argument("--timeout-s", type=int, default=int(os.getenv("MLB_EXTERNAL_PICK_FETCH_TIMEOUT_S", "30")))
    parser.add_argument("--retries", type=int, default=int(os.getenv("MLB_EXTERNAL_PICK_FETCH_RETRIES", "3")))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--import-after-fetch", action="store_true", help="Import written CSVs into bets.mlb_external_pick_ledger.")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    args = parser.parse_args()

    if args.date:
        game_date = date.fromisoformat(args.date)
    else:
        game_date = datetime.now().date()

    sources = load_sources(config_path=args.config, cli_sources=args.source)
    payload = fetch_external_picks(
        game_date=game_date,
        sources=sources,
        import_dir=args.import_dir,
        timeout_s=args.timeout_s,
        retries=args.retries,
        dry_run=args.dry_run,
    )
    if args.import_after_fetch and not args.dry_run:
        import psycopg2

        with psycopg2.connect(args.pg_dsn) as conn:
            payload["import"] = import_external_sources(
                conn,
                game_date=game_date,
                import_dir=args.import_dir,
                pattern="*.csv",
            )
    json_path, report_path = write_fetch_outputs(payload)
    print(json.dumps({
        "status": payload.get("status"),
        "game_date": game_date.isoformat(),
        "sources": (payload.get("summary") or {}).get("sources", 0),
        "normalized_rows": (payload.get("summary") or {}).get("normalized_rows", 0),
        "files_written": (payload.get("summary") or {}).get("files_written", 0),
        "fetch_json": str(json_path),
        "fetch_report": str(report_path),
        "import": payload.get("import"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
