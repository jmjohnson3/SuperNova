# src/nba_pipeline/raw_store.py
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

log = logging.getLogger("nba_pipeline.raw_store")

SQL_INSERT_API_RESPONSE = """
INSERT INTO raw.api_responses (
  provider,
  endpoint,
  season,
  game_slug,
  as_of_date,
  url,
  fetched_at_utc,
  payload,
  payload_sha256
)
VALUES (
  %(provider)s,
  %(endpoint)s,
  %(season)s,
  %(game_slug)s,
  %(as_of_date)s,
  %(url)s,
  %(fetched_at_utc)s,
  %(payload)s::jsonb,
  %(payload_sha256)s
)
ON CONFLICT DO NOTHING;
"""

def _sha256_json(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

def save_api_response(
    conn,
    *,
    provider: str,
    endpoint: str,
    url: str,
    payload: Any,
    season: Optional[str] = None,
    game_slug: Optional[str] = None,
    as_of_date: Optional[object] = None,
    fetched_at_utc: Optional[datetime] = None,
) -> str:
    """
    Insert raw API payload, idempotent.
    Returns payload_sha256 (useful for logging).
    """
    if fetched_at_utc is None:
        fetched_at_utc = datetime.now(timezone.utc)

    payload_sha256 = _sha256_json(payload)

    params = {
        "provider": provider,
        "endpoint": endpoint,
        "season": season,
        "game_slug": game_slug,
        "as_of_date": as_of_date,
        "url": url,
        "fetched_at_utc": fetched_at_utc,
        "payload": json.dumps(payload),
        "payload_sha256": payload_sha256,
    }

    with conn.cursor() as cur:
        cur.execute(SQL_INSERT_API_RESPONSE, params)

    return payload_sha256
