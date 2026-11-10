"""Asynchronous client for The Odds API v4.

Listing events is free. Odds for one event cost one credit per market (for up to ten named bookmakers), so the
client tracks the quota from every response, refuses to go below a credit floor or past a per-scan budget,
caches identical requests briefly, retries timeouts and server errors with exponential backoff, and skips
markets the API rejects. The key is sent as a query parameter and is never logged or put in an error.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)   # httpx logs whole request URLs at INFO; ours carry the key


class OddsApiError(Exception):
    """A request failed for good. The message never contains the key or a full URL."""


class AuthError(OddsApiError):
    pass


class QuotaGuard(OddsApiError):
    """The next request would cross the credit floor or the scan's budget, so it was not sent."""


class UnsupportedMarket(OddsApiError):
    pass


@dataclass
class Quota:
    remaining: int | None = None
    used: int | None = None
    last_cost: int | None = None
    spent: int = 0            # by this client


def _number(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


class OddsApiClient:
    def __init__(self, api_key: str, *, base_url: str = "https://api.the-odds-api.com/v4", timeout: float = 15.0,
                 max_retries: int = 3, backoff: float = 0.5, cache_seconds: float = 60.0, credit_floor: int = 0,
                 max_credits: int | None = None, transport: httpx.AsyncBaseTransport | None = None,
                 sleep=asyncio.sleep, clock=time.monotonic) -> None:
        if not api_key:
            raise AuthError("No Odds API key is configured (set ODDS_API_KEY).")
        self._key = api_key
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)
        self.max_retries, self.backoff, self.cache_seconds = max_retries, backoff, cache_seconds
        self.credit_floor, self.max_credits = credit_floor, max_credits
        self.quota = Quota()
        self.unsupported: set[str] = set()
        self._cache: dict[tuple, tuple[float, Any]] = {}
        self._sleep, self._clock = sleep, clock

    async def __aenter__(self) -> "OddsApiClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    def _guard(self, cost: int) -> None:
        if cost <= 0:
            return
        if self.max_credits is not None and self.quota.spent + cost > self.max_credits:
            raise QuotaGuard(f"scan budget of {self.max_credits} credits reached")
        if self.quota.remaining is not None and self.quota.remaining - cost < self.credit_floor:
            raise QuotaGuard(f"{self.quota.remaining} credits left; the floor is {self.credit_floor}")

    async def _get(self, path: str, params: dict[str, str], cost: int = 0) -> Any:
        key = (path, tuple(sorted(params.items())))
        cached = self._cache.get(key)
        if cached and self._clock() - cached[0] <= self.cache_seconds:
            return cached[1]
        self._guard(cost)
        problem = "no response"
        for attempt in range(self.max_retries + 1):
            if attempt:
                await self._sleep(self.backoff * 2 ** (attempt - 1))
            try:
                response = await self._client.get(path, params={**params, "apiKey": self._key})
            except httpx.TimeoutException:
                problem = "timed out"
                continue
            except httpx.TransportError as error:
                problem = f"connection failed ({type(error).__name__})"
                continue
            remaining, used, last = (_number(response.headers.get(h)) for h in
                                     ("x-requests-remaining", "x-requests-used", "x-requests-last"))
            if remaining is not None:
                self.quota.remaining, self.quota.used, self.quota.last_cost = remaining, used, last
            status = response.status_code
            if status == 200:
                try:
                    data = response.json()
                except ValueError as error:
                    raise OddsApiError(f"{path}: the response was not JSON") from error
                self.quota.spent += last if last is not None else cost
                self._cache[key] = (self._clock(), data)
                return data
            if status in (401, 403):
                raise AuthError("The Odds API rejected the key or the plan does not include this request.")
            if status == 404:
                return None
            if status == 422:
                raise UnsupportedMarket(f"{path}: the API rejected the request (HTTP 422)")
            if status == 429:
                problem = "rate limited"
                wait = _number(response.headers.get("retry-after"))
                if wait and attempt < self.max_retries:
                    await self._sleep(min(wait, 30))
                continue
            if status >= 500:
                problem = f"server error (HTTP {status})"
                continue
            raise OddsApiError(f"{path}: HTTP {status}")
        raise OddsApiError(f"{path}: {problem} after {self.max_retries + 1} attempts")

    async def sports(self) -> list[dict]:
        """In-season sports. Free, and it refreshes the quota figures."""
        data = await self._get("/sports", {})
        return data if isinstance(data, list) else []

    async def events(self, sport_key: str) -> list[dict]:
        """Upcoming and live events for a sport. Free."""
        data = await self._get(f"/sports/{sport_key}/events", {"dateFormat": "iso"})
        return [e for e in data if isinstance(e, dict) and e.get("id")] if isinstance(data, list) else []

    async def event_odds(self, sport_key: str, event_id: str, markets: list[str], bookmakers: list[str]) -> dict | None:
        """One event with the requested markets at the requested books, or None when nothing is available.
        Markets the API rejects are remembered and left out from then on."""
        wanted = [m for m in markets if m not in self.unsupported]
        if not wanted or not bookmakers:
            return None
        cost = len(wanted) * max(1, math.ceil(len(bookmakers) / 10))
        params = {"bookmakers": ",".join(bookmakers), "markets": ",".join(wanted), "oddsFormat": "american",
                  "dateFormat": "iso", "includeLinks": "true"}
        try:
            data = await self._get(f"/sports/{sport_key}/events/{event_id}/odds", params, cost)
        except UnsupportedMarket:
            if len(wanted) == 1:
                self.unsupported.add(wanted[0])
                log.warning("market %s is not supported for %s; skipping it", wanted[0], sport_key)
                return None
            merged: dict | None = None      # find the rejected market by asking for each on its own
            for market in wanted:
                part = await self.event_odds(sport_key, event_id, [market], bookmakers)
                merged = _merge(merged, part)
            return merged
        return data if isinstance(data, dict) and data.get("id") else None


def _merge(base: dict | None, extra: dict | None) -> dict | None:
    """Combine two responses for the same event into one, book by book."""
    if base is None or extra is None:
        return base or extra
    books = {b["key"]: dict(b, markets=list(b.get("markets") or [])) for b in base.get("bookmakers") or []}
    for book in extra.get("bookmakers") or []:
        if book["key"] in books:
            books[book["key"]]["markets"] += list(book.get("markets") or [])
        else:
            books[book["key"]] = book
    return dict(base, bookmakers=list(books.values()))
