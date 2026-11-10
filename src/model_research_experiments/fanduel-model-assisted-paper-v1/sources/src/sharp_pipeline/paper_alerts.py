"""NBA paper-only check requests, selected prospectively and delivered through an outbox."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import psycopg2.extras
import requests

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bets.nba_paper_price_alerts (
  alert_id BIGSERIAL PRIMARY KEY, decision_id BIGINT NOT NULL UNIQUE REFERENCES bets.sharp_multi_decisions(decision_id),
  experiment_id TEXT NOT NULL, experiment_sha256 TEXT NOT NULL, book TEXT NOT NULL,
  event_id TEXT NOT NULL, market TEXT NOT NULL, player_norm TEXT NOT NULL,
  selected_at TIMESTAMPTZ NOT NULL DEFAULT now(), delivery_status TEXT NOT NULL DEFAULT 'pending',
  attempted_at TIMESTAMPTZ, sent_at TIMESTAMPTZ, discord_message_id TEXT, delivery_error TEXT,
  UNIQUE (experiment_id, book, event_id, market, player_norm),
  CHECK (delivery_status IN ('pending','sending','sent','expired','uncertain')));
CREATE TABLE IF NOT EXISTS bets.nba_paper_price_checks (
  alert_id BIGINT PRIMARY KEY REFERENCES bets.nba_paper_price_alerts(alert_id),
  checked_at TIMESTAMPTZ NOT NULL, recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  status TEXT NOT NULL CHECK (status IN ('available','changed_line','disappeared','suspended')),
  displayed_line NUMERIC, displayed_price INTEGER, note TEXT,
  evidence_source TEXT NOT NULL DEFAULT 'manual_live' CHECK (evidence_source='manual_live'),
  CHECK ((status IN ('available','changed_line') AND displayed_line IS NOT NULL
          AND displayed_price IS NOT NULL AND abs(displayed_price)>=100)
      OR (status IN ('disappeared','suspended') AND displayed_line IS NULL AND displayed_price IS NULL)));
"""


def enqueue(conn, observation_id: str, experiment: dict) -> int:
    """First qualifying exact prop per book/player/game/market; no projection or outcome filter."""
    from sharp_pipeline import nba_experiment
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO bets.nba_paper_price_alerts
          (decision_id, experiment_id, experiment_sha256, book, event_id, market, player_norm)
          SELECT DISTINCT ON (book, event_id, market, player_norm)
            decision_id, experiment_id, experiment_sha256, book, event_id, market, player_norm
          FROM bets.sharp_multi_decisions
          WHERE observation_id=%s AND sport='nba' AND tier='decision' AND method='exact'
            AND market=ANY(%s) AND book=ANY(%s) AND player_norm<>''
            AND experiment_id=%s AND experiment_sha256=%s AND decided_at >= %s
          ORDER BY book, event_id, market, player_norm, ev DESC, decision_id
          ON CONFLICT DO NOTHING""", (observation_id, list(nba_experiment.MARKETS), list(nba_experiment.BOOKS),
                                       experiment['experiment_id'], experiment['sha256'], experiment['registered_at']))
        return cur.rowcount


def webhook() -> str:
    # Paper requests must reach the NBA channel; never fall back to another sport's webhook.
    from supernovabets_config import _saved_windows_env
    return os.getenv('NBA_DISCORD_WEBHOOK_URL') or _saved_windows_env('NBA_DISCORD_WEBHOOK_URL') or ''


def card(row: dict) -> str:
    names = {'fanduel': 'FanDuel', 'draftkings': 'DraftKings', 'espnbet': 'theScore / ESPN Bet'}
    observed = row['decided_at'].astimezone(timezone.utc)
    tip = row['commence_time_utc'].astimezone(timezone.utc)
    text = (f"**PAPER PRICE CHECK — NBA #{row['alert_id']}**\n"
            f"**{row['player']} — {row['side'].upper()} {float(row['line']):g} {row['market']}**\n"
            f"{names.get(row['book'], row['book'])}: **{int(row['price']):+d}**\n"
            f"{row['away_team']} @ {row['home_team']} | Tip <t:{int(tip.timestamp())}:t>\n"
            f"Observed {observed:%Y-%m-%d %H:%M:%S UTC} (<t:{int(observed.timestamp())}:R>)\n"
            "Research only — no wager requested. Check within 5 minutes. Record check time, displayed line/price, "
            "or disappeared/suspended, using this check ID.\n"
            "Record with the local price-check tool or report the details in the Codex chat; Discord replies are not imported.")
    link = row.get('link')
    if isinstance(link, str) and link.startswith('https://') and not any(c in link for c in '\r\n<>'):
        text += f"\nOpen sportsbook: <{link}>"
    return text[:1950]


def send(text: str) -> str:
    url = webhook()
    if not url:
        raise RuntimeError('NBA webhook unavailable')
    response = requests.post(url, params={'wait': 'true'},
                             json={'content': text, 'allowed_mentions': {'parse': []}}, timeout=20)
    if response.status_code != 200:
        raise RuntimeError(f'Discord HTTP {response.status_code}')
    message_id = response.json().get('id')
    if not message_id:
        raise RuntimeError('Discord did not return a message ID')
    return str(message_id)


def dispatch(conn, experiment: dict, *, now: datetime | None = None, poster=send) -> dict:
    from sharp_pipeline import nba_experiment
    fixed_now = now
    sent, uncertain, expired = 0, 0, 0
    if poster is send and not webhook():
        return dict(sent=0, status='nba_webhook_missing')
    while True:
        at = fixed_now or datetime.now(timezone.utc)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""SELECT a.alert_id, d.* FROM bets.nba_paper_price_alerts a
                JOIN bets.sharp_multi_decisions d USING (decision_id)
                WHERE a.experiment_id=%s AND a.experiment_sha256=%s AND a.delivery_status='pending'
                ORDER BY d.decided_at, a.alert_id LIMIT 1 FOR UPDATE OF a SKIP LOCKED""",
                (experiment['experiment_id'], experiment['sha256']))
            row = cur.fetchone()
            if row is None:
                conn.commit()
                break
            row = dict(row)
            age = (at - row['decided_at']).total_seconds()
            if not 0 <= age <= nba_experiment.MAX_ALERT_AGE_SECONDS or at >= row['commence_time_utc']:
                cur.execute("UPDATE bets.nba_paper_price_alerts SET delivery_status='expired' WHERE alert_id=%s", (row['alert_id'],))
                conn.commit()
                expired += 1
                continue
            cur.execute("UPDATE bets.nba_paper_price_alerts SET delivery_status='sending', attempted_at=%s WHERE alert_id=%s",
                        (at, row['alert_id']))
        conn.commit()  # Claim before posting. An interrupted/uncertain send is never blindly retried.
        try:
            message_id = poster(card(row))
            if not message_id:
                raise RuntimeError('No delivery receipt')
        except Exception as exc:
            with conn.cursor() as cur:
                cur.execute("UPDATE bets.nba_paper_price_alerts SET delivery_status='uncertain', delivery_error=%s WHERE alert_id=%s",
                            (type(exc).__name__, row['alert_id']))  # never log a webhook-bearing exception URL
            conn.commit()
            uncertain += 1
            continue
        with conn.cursor() as cur:
            cur.execute("""UPDATE bets.nba_paper_price_alerts SET delivery_status='sent', sent_at=%s,
                           discord_message_id=%s WHERE alert_id=%s""",
                        (fixed_now or datetime.now(timezone.utc), str(message_id), row['alert_id']))
        conn.commit()
        sent += 1
    return dict(sent=sent, uncertain=uncertain, expired=expired, status='ok')
