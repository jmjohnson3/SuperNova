"""Bounded Discord transport for the frozen sharp-price experiment.

Keep delivery independent of the daily orchestration module so report timeout
changes cannot change the sharp experiment's registered source dependencies.
"""
from __future__ import annotations

import asyncio
import os

import httpx

from supernovabets_config import _saved_windows_env


def _webhook_url() -> str:
    return (
        os.getenv("NFL_DISCORD_WEBHOOK_URL")
        or _saved_windows_env("NFL_DISCORD_WEBHOOK_URL")
        or os.getenv("MLB_DISCORD_WEBHOOK_URL")
        or os.getenv("DISCORD_WEBHOOK_URL")
        or ""
    )


async def _post_payload(payload: dict) -> None:
    webhook = _webhook_url()
    if not webhook:
        raise RuntimeError("NFL Discord webhook is not configured")
    async with httpx.AsyncClient(timeout=20) as client:
        for attempt in range(4):
            try:
                response = await client.post(webhook, params={"wait": "true"}, json=payload)
            except httpx.RequestError:
                raise RuntimeError('Discord delivery was not confirmed; not retrying an uncertain send') from None
            if response.status_code in {200, 204}:
                return
            if response.status_code != 429 or attempt == 3:
                raise RuntimeError(f"Discord post failed (HTTP {response.status_code})")
            retry_after = float(response.json().get('retry_after', 1))
            if not 0 <= retry_after <= 60:
                raise RuntimeError('Discord rate limit exceeds bounded retry window')
            await asyncio.sleep(max(.1, retry_after))
