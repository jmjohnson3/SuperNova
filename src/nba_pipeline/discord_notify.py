import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import aiohttp

log = logging.getLogger("nba_pipeline.discord_notify")

DISCORD_MAX_LEN = 2000  # hard limit for Discord messages


@dataclass(frozen=True)
class DiscordConfig:
    webhook_url: str = "https://discord.com/api/webhooks/1461420766108319858/LenBk50YR1eS1isFMSOzE8gMWgSgBTSYmU4Ac1unf2SOo_kPSGk71afBqbBiQDuUZwD3"
    username: str = "1408438245594763375"
    avatar_url: str | None = None
    timeout_s: int = 20

    def resolved_webhook(self) -> str | None:
        return self.webhook_url or os.getenv("DISCORD_WEBHOOK_URL")


def _chunk_message(text: str, max_len: int = DISCORD_MAX_LEN) -> list[str]:
    """
    Splits on paragraph boundaries first; falls back to hard splits.
    Keeps each chunk <= max_len.
    """
    text = text.strip()
    if len(text) <= max_len:
        return [text]

    parts: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue

        add_len = len(block) + (2 if buf else 0)
        if buf_len + add_len <= max_len:
            if buf:
                buf.append("")  # blank line
            buf.append(block)
            buf_len += add_len
            continue

        if buf:
            parts.append("\n\n".join(buf))
            buf, buf_len = [], 0

        # block alone too big -> hard split
        while len(block) > max_len:
            parts.append(block[:max_len])
            block = block[max_len:]
        if block:
            buf = [block]
            buf_len = len(block)

    if buf:
        parts.append("\n\n".join(buf))

    return parts


async def send_discord_message(text: str, cfg: DiscordConfig | None = None) -> None:
    cfg = cfg or DiscordConfig()
    webhook = cfg.resolved_webhook()
    if not webhook:
        log.warning("DISCORD_WEBHOOK_URL not set; skipping Discord notification.")
        return

    chunks = _chunk_message(text)

    payload_base = {
        "content": None,
        "username": cfg.username,
    }
    if cfg.avatar_url:
        payload_base["avatar_url"] = cfg.avatar_url

    timeout = aiohttp.ClientTimeout(total=cfg.timeout_s)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, chunk in enumerate(chunks, start=1):
            payload = dict(payload_base)
            payload["content"] = chunk

            for attempt in range(4):
                try:
                    async with session.post(webhook, json=payload) as resp:
                        # Discord webhooks return 204 No Content on success
                        if resp.status in (200, 204):
                            break

                        body = await resp.text()
                        # basic backoff on rate-limit / transient errors
                        if resp.status in (429, 500, 502, 503, 504) and attempt < 3:
                            sleep_s = 1.5 * (attempt + 1)
                            log.warning("Discord post failed (%s). Retrying in %.1fs. Body=%s", resp.status, sleep_s, body[:300])
                            await asyncio.sleep(sleep_s)
                            continue

                        raise RuntimeError(f"Discord webhook failed status={resp.status} body={body[:500]}")
                except Exception:
                    if attempt >= 3:
                        raise
                    await asyncio.sleep(1.5 * (attempt + 1))

            # small pause between chunks to be nice to Discord
            if i < len(chunks):
                await asyncio.sleep(0.35)
