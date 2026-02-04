# src/nba_pipeline/fetcher.py
import logging
import time
from typing import Any, Dict

import requests
from requests.auth import HTTPBasicAuth

log = logging.getLogger("nba_pipeline.fetcher")


class RateLimitedError(RuntimeError):
    pass


class NoContentYetError(RuntimeError):
    pass


class BadPayloadError(RuntimeError):
    pass


def _is_known_empty_payload(url: str, data: Dict[str, Any]) -> bool:
    """
    MySportsFeeds often returns small but valid payloads with empty arrays.
    Treat these as OK.
    """
    if not isinstance(data, dict):
        return False

    # games_by_date
    if url.endswith("/games.json") and "games" in data and isinstance(data["games"], list):
        return True  # empty or not, both valid

    # player_gamelogs
    if "player_gamelogs.json" in url and "gamelogs" in data and isinstance(data["gamelogs"], list):
        return True

    return False


class MySportsFeedsClient:
    def __init__(self, api_key: str, timeout: int = 30, max_retries: int = 3, backoff_seconds: float = 1.5):
        self.auth = HTTPBasicAuth(api_key, "MYSPORTSFEEDS")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def fetch_json(self, url: str) -> Dict[str, Any]:
        for attempt in range(1, self.max_retries + 1):
            try:
                log.info("Fetching URL (attempt %d/%d): %s", attempt, self.max_retries, url)
                resp = requests.get(url, auth=self.auth, timeout=self.timeout, headers={"Accept": "application/json"})

                RETRYABLE = {429, 500, 502, 503, 504}
                NOT_READY = {204, 404}

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code in NOT_READY:
                    # don't retry; just raise a special error the crawler can handle
                    raise NotReadyError(f"HTTP {resp.status_code} for {url}: {resp.text[:200]}")

                if resp.status_code in RETRYABLE:
                    raise RuntimeError(f"HTTP {resp.status_code} for {url}: {resp.text[:200]}")

                raise RuntimeError(f"HTTP {resp.status_code} for {url}: {resp.text[:200]}")

                if resp.status_code == 429:
                    raise RateLimitedError(f"HTTP 429 for {url}")
                if resp.status_code == 204:
                    raise NoContentYetError(f"HTTP 204 for {url}")
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code} for {url}: {resp.text[:500]}")

                data = resp.json()
                # If endpoint returns empty data arrays, don't save; caller will skip.
                if isinstance(data, dict):
                    if "gamelogs" in data and not data.get("gamelogs"):
                        raise NoContentYetError("Empty gamelogs")
                    if "games" in data and not data.get("games"):
                        raise NoContentYetError("Empty games")
                return data

                # error-shaped payloads
                if isinstance(data, dict) and ("error" in data or "errors" in data):
                    raise BadPayloadError(f"API error payload for {url}: {data}")

                # small payloads are OK if they match known shapes (games/gamelogs, etc.)
                if len(resp.content or b"") < 200 and not _is_known_empty_payload(url, data):
                    raise BadPayloadError(
                        f"Suspiciously small payload ({len(resp.content)} bytes) for {url}: {resp.text[:200]}"
                    )

                log.info("Fetched %d bytes from %s", len(resp.content), url)
                return data

            except NoContentYetError:
                raise
            except RateLimitedError as exc:
                log.warning("Fetch failed: %s", exc)
                if attempt >= self.max_retries:
                    log.error("Giving up on %s after %d attempts", url, attempt)
                    raise
                sleep_for = self.backoff_seconds * (2 ** (attempt - 1))
                log.info("Retrying in %.1f seconds...", sleep_for)
                time.sleep(sleep_for)
            except Exception as exc:
                log.warning("Fetch failed: %s", exc)
                if attempt >= self.max_retries:
                    log.error("Giving up on %s after %d attempts", url, attempt)
                    raise
                sleep_for = self.backoff_seconds * (2 ** (attempt - 1))
                log.info("Retrying in %.1f seconds...", sleep_for)
                time.sleep(sleep_for)

        raise RuntimeError("Unreachable fetch_json failure")