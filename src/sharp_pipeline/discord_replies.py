"""Read replies to the NBA paper alerts from Discord, then refresh the reply-trial report.

A Discord webhook can post but cannot read, so reading replies needs a bot token: DISCORD_BOT_TOKEN in
the user environment. The bot needs View Channel, Read Message History and Add Reactions in the alerts
channel, and Message Content Intent switched on (without it Discord hands over replies with no text).

Each run reads the channel forward from where the last run stopped, keeps every human reply to an alert
(a Discord reply, or a message starting with the alert number: "283 y") and marks it: a tick when it
was understood, a clock when it came more than 5 minutes after the alert, a question mark when not
understood. What a reply means is decided by reply_trial, which is pinned; this module only moves
messages and is not, so a Discord change needs no new registration.

    python -m sharp_pipeline.discord_replies [--report-only] [--no-post]

Without a token it skips the reading and still writes the every-alert ledger. One Discord summary a day
after 9 AM Eastern. Nothing is staked and no wager is requested.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras
import requests

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from sharp_pipeline import reply_trial

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "reports" / "nba_reply_trial_state.json"
API = "https://discord.com/api/v10"
HEADERS = {"User-Agent": "DiscordBot (https://localhost, 1.0)"}
FIRST_RUN_LOOKBACK = timedelta(hours=24)
MAX_PAGES = 30
ACK = {"understood": "✅", "late": "⏰", "unknown": "❓"}
NUMBERED = re.compile(r"\s*#?(\d{1,9})\b[:\s]*(.*)$", re.S)
_ET = ZoneInfo("America/New_York")
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bets.nba_alert_replies (
  reply_message_id TEXT PRIMARY KEY, alert_id BIGINT NOT NULL REFERENCES bets.nba_paper_price_alerts(alert_id),
  channel_id TEXT NOT NULL, author_id TEXT, replied_at TIMESTAMPTZ NOT NULL, edited BOOLEAN NOT NULL DEFAULT false,
  imported_at TIMESTAMPTZ NOT NULL DEFAULT now(), raw_text TEXT NOT NULL, understood BOOLEAN NOT NULL,
  source TEXT NOT NULL DEFAULT 'discord_reply');
CREATE INDEX IF NOT EXISTS nba_alert_replies_alert ON bets.nba_alert_replies (alert_id);
CREATE TABLE IF NOT EXISTS bets.nba_alert_reply_cursor (channel_id TEXT PRIMARY KEY, last_message_id TEXT NOT NULL);
"""


class DiscordError(RuntimeError):
    """Carries only a short reason: never a URL, which could hold the webhook or the token."""


def saved(name: str) -> str:
    from supernovabets_config import _saved_windows_env
    return os.getenv(name) or _saved_windows_env(name) or ""


def snowflake(at: datetime) -> str:
    return str((int(at.timestamp() * 1000) - reply_trial.DISCORD_EPOCH_MS) << 22)


def channel_of_webhook(session, webhook_url: str) -> str:
    try:
        response = session.get(webhook_url, headers=HEADERS, timeout=20)
        if response.status_code != 200:
            raise DiscordError(f"webhook_http_{response.status_code}")
        return str(response.json()["channel_id"])
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        raise DiscordError("webhook_" + type(exc).__name__) from None


def call(session, method: str, path: str, token: str, sleep=time.sleep, **kwargs):
    """One Discord request, waiting out a rate limit once."""
    for attempt in (0, 1):
        try:
            response = session.request(method, API + path, headers=dict(HEADERS, Authorization="Bot " + token),
                                       timeout=20, **kwargs)
        except requests.RequestException as exc:
            raise DiscordError("network_" + type(exc).__name__) from None
        if response.status_code != 429 or attempt:
            return response
        try:
            sleep(min(float(response.json().get("retry_after", 1)), 5))
        except (ValueError, TypeError, AttributeError):
            sleep(1)
    return response


def fetch(session, token: str, channel: str, after: str, sleep=time.sleep) -> list[dict]:
    """Every message after `after`, oldest first. Discord pages forward from the given ID."""
    found: dict[str, dict] = {}
    for _ in range(MAX_PAGES):
        response = call(session, "GET", f"/channels/{channel}/messages", token, sleep, params=dict(after=after, limit=100))
        if response.status_code != 200:
            raise DiscordError({401: "bot_token_rejected", 403: "bot_cannot_read_channel",
                                404: "channel_not_found"}.get(response.status_code, f"http_{response.status_code}"))
        batch = response.json()
        if not isinstance(batch, list) or not batch:
            break
        found.update({str(m["id"]): m for m in batch})
        after = str(max(int(m["id"]) for m in batch))
        if len(batch) < 100:
            break
    return [found[k] for k in sorted(found, key=int)]


def match_alert(message: dict, by_message: dict[str, int], alert_ids: set[int]) -> tuple[int | None, str]:
    """(alert, text) for a human reply to an alert, or (None, '') for ordinary chat."""
    author = message.get("author") or {}
    if author.get("bot") or message.get("webhook_id"):
        return None, ""
    text = str(message.get("content") or "")
    referenced = str((message.get("message_reference") or {}).get("message_id") or "")
    if referenced in by_message:
        return by_message[referenced], text
    numbered = NUMBERED.match(text)
    if numbered and int(numbered.group(1)) in alert_ids and numbered.group(2).strip():
        return int(numbered.group(1)), numbered.group(2)
    return None, ""


def run(conn, *, token: str, webhook_url: str, session=requests, now: datetime | None = None, sleep=time.sleep,
        only_author: str = "") -> dict:
    now = now or datetime.now(timezone.utc)
    channel = channel_of_webhook(session, webhook_url)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT last_message_id FROM bets.nba_alert_reply_cursor WHERE channel_id=%s", (channel,))
        row = cur.fetchone()
        cur.execute("""SELECT alert_id, discord_message_id FROM bets.nba_paper_price_alerts
                       WHERE delivery_status='sent' AND discord_message_id IS NOT NULL""")
        alerts = {str(r["discord_message_id"]): int(r["alert_id"]) for r in cur.fetchall()}
    after = row["last_message_id"] if row else snowflake(now - FIRST_RUN_LOOKBACK)
    messages = fetch(session, token, channel, after, sleep)
    out = dict(status="ok", messages=len(messages), replies=0, understood=0, not_understood=0, without_text=0, acknowledged=0)
    for message in messages:
        alert_id, text = match_alert(message, alerts, set(alerts.values()))
        if alert_id is None or (only_author and str((message.get("author") or {}).get("id")) != only_author):
            continue
        replied = safe_time(message.get("timestamp")) or reply_trial.snowflake_time(message["id"])
        understood = reply_trial.parse_reply(text) is not None
        edited = message.get("edited_timestamp") is not None
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO bets.nba_alert_replies
                             (reply_message_id, alert_id, channel_id, author_id, replied_at, edited, raw_text, understood)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (str(message["id"]), alert_id, channel, str((message.get("author") or {}).get("id") or ""),
                         replied, edited, text.strip()[:200], understood))
            new = cur.rowcount == 1
        conn.commit()
        if not new:
            continue
        out["replies"] += 1
        out["understood" if understood else "not_understood"] += 1
        out["without_text"] += not text.strip()
        posted = reply_trial.snowflake_time(next(m for m, a in alerts.items() if a == alert_id))
        late = bool(posted and replied and (replied - posted).total_seconds() > reply_trial.MAX_CHECK_DELAY_SECONDS)
        mark = ACK["unknown"] if not understood or edited else ACK["late"] if late else ACK["understood"]
        try:  # the mark is a courtesy; the reply is already saved
            answer = call(session, "PUT", f"/channels/{channel}/messages/{message['id']}/reactions/{quote(mark)}/@me", token, sleep)
            out["acknowledged"] += answer.status_code == 204
            sleep(0.3)
        except DiscordError:
            pass
    if messages:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO bets.nba_alert_reply_cursor (channel_id, last_message_id) VALUES (%s,%s)
                           ON CONFLICT (channel_id) DO UPDATE SET last_message_id=EXCLUDED.last_message_id""",
                        (channel, str(max(int(m["id"]) for m in messages))))
        conn.commit()
    if out["without_text"]:
        out["hint"] = "replies arrived without text: switch on Message Content Intent for the bot"
    return out


def post_digest(doc: dict, now: datetime, poster=None) -> bool:
    """Once a day after 9 AM Eastern, when the previous night's games are graded."""
    local = now.astimezone(_ET)
    if local.hour < 9:
        return False
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    yesterday = local.date() - timedelta(days=1)
    if state.get("digest_posted") == str(yesterday):
        return False
    text = reply_trial.digest(doc, yesterday)
    if text:
        if poster is None:
            from sharp_pipeline.report import post
            poster = lambda body: post(body, "nba")
        poster(text)
    atomic_json(STATE, dict(state, digest_posted=str(yesterday)))
    return bool(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-only", action="store_true", help="skip Discord; rebuild the report from saved replies")
    ap.add_argument("--no-post", action="store_true", help="do not post the daily summary")
    args = ap.parse_args()
    from sharp_pipeline.paper_alerts import webhook
    now = datetime.now(timezone.utc)
    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('bets.nba_paper_price_alerts')")
            if cur.fetchone()[0] is None:
                print(json.dumps(dict(status="no_alert_table")))
                return
            cur.execute("SELECT pg_try_advisory_lock(86731552)")
            if not cur.fetchone()[0]:
                print(json.dumps(dict(status="busy")))
                return
            cur.execute(SCHEMA_SQL)
        conn.commit()
        token, url = saved("DISCORD_BOT_TOKEN"), webhook()
        if args.report_only:
            imported = dict(status="report_only")
        elif not token:
            imported = dict(status="bot_token_missing")
        elif not url:
            imported = dict(status="nba_webhook_missing")
        else:
            try:
                imported = run(conn, token=token, webhook_url=url, now=now, only_author=saved("NBA_REPLY_USER_ID"))
            except DiscordError as exc:
                conn.rollback()
                imported = dict(status=str(exc))
        doc = reply_trial.write_report(conn, now)
        posted = False
        if not args.no_post:
            try:
                posted = post_digest(doc, now)
            except RuntimeError as exc:
                posted = "failed: " + str(exc)[:80]
    total = doc["assumed"]["total"]
    print(json.dumps(dict(now=now.isoformat(), replies=imported, digest_posted=posted, trial=doc["source_status"],
                          assumed=dict(settled=total["settled"], units=total["units"], roi=total["roi"]),
                          reply_checks={b: m["status"] for b, m in doc["reply_checks"].items()}), default=str))


if __name__ == "__main__":
    main()
