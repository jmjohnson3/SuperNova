import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

import psycopg2
import psycopg2.extras

log = logging.getLogger("nba_pipeline.raw_store")


def _sha256_json(payload: Any) -> str:
    # Stable hash: sort keys + compact separators
    b = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def save_api_response(
    conn: psycopg2.extensions.connection,
    *,
    provider: str,
    endpoint: str,
    url: str,
    payload: Any,
    season: Optional[str] = None,
    game_slug: Optional[str] = None,
    as_of_date: Optional[date] = None,
) -> bool:
    """
    Inserts payload into raw.api_responses with dedupe via (provider, endpoint, payload_sha256).
    Returns True if inserted, False if already existed.
    """
    payload_hash = _sha256_json(payload)

    sql = """
    INSERT INTO raw.api_responses (
      provider, endpoint, season, game_slug, as_of_date, url, fetched_at_utc, payload, payload_sha256
    )
    VALUES (
      %(provider)s, %(endpoint)s, %(season)s, %(game_slug)s, %(as_of_date)s,
      %(url)s, now(), %(payload)s::jsonb, %(payload_sha256)s
    )
    ON CONFLICT (provider, endpoint, payload_sha256) DO NOTHING
    ;
    """

    params = {
        "provider": provider,
        "endpoint": endpoint,
        "season": season,
        "game_slug": game_slug,
        "as_of_date": as_of_date,
        "url": url,
        "payload": json.dumps(payload, ensure_ascii=False),
        "payload_sha256": payload_hash,
    }

    with conn.cursor() as cur:
        cur.execute(sql, params)
        inserted = cur.rowcount == 1

    if inserted:
        log.info("Saved raw payload provider=%s endpoint=%s hash=%s", provider, endpoint, payload_hash[:10])
    else:
        log.info("Skipped duplicate payload provider=%s endpoint=%s hash=%s", provider, endpoint, payload_hash[:10])

    return inserted
