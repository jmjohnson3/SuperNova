import logging
import time
from typing import Any, Dict, Optional
import requests
from requests.auth import HTTPBasicAuth

log = logging.getLogger("nba_pipeline.fetcher")


class MySportsFeedsClient:
    """
    Thin HTTP client for MySportsFeeds.
    Responsibility: fetch JSON safely, nothing else.
    """

    BASE_URL = "https://api.mysportsfeeds.com/v2.1/pull"

    def __init__(
        self,
        api_key: str,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
    ):
        self.auth = HTTPBasicAuth(api_key, "MYSPORTSFEEDS")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def fetch_json(self, url: str) -> Dict[str, Any]:
        """
        Fetch a URL and return parsed JSON.
        Raises on non-200 or invalid JSON.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                log.info("Fetching URL (attempt %d/%d): %s", attempt, self.max_retries, url)

                resp = requests.get(
                    url,
                    auth=self.auth,
                    timeout=self.timeout,
                    headers={"Accept": "application/json"},
                )

                if resp.status_code != 200:
                    raise RuntimeError(
                        f"HTTP {resp.status_code} for {url}: {resp.text[:500]}"
                    )

                data = resp.json()
                log.info("Fetched %d bytes from %s", len(resp.content), url)
                return data

            except Exception as exc:
                log.warning("Fetch failed: %s", exc)
                if attempt >= self.max_retries:
                    log.error("Giving up on %s after %d attempts", url, attempt)
                    raise
                sleep_for = self.backoff_seconds ** attempt
                log.info("Retrying in %.1f seconds...", sleep_for)
                time.sleep(sleep_for)

        # Should never get here
        raise RuntimeError("Unreachable fetch_json failure")
