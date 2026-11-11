"""Alerts: tell a Discord channel when the scheduled pass enters a new paper bet whose price is still live.

Off unless SNIPER_ALERT_WEBHOOK_URL is set. A bet is alerted once, only when it was entered from a price seen in
the last few minutes for a game that has not started (never a replayed one), and at most a few per pass. The
message is information for research: it says the bet is on paper and that the price may already be gone. The
webhook address is a secret: it is never logged or put in an error.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from sports_sniper import db, odds_math
from sports_sniper.config import Settings
from sports_sniper.markets import book_label, market_label
from sports_sniper.opportunities import describe


def line(bet, now: datetime, link_state: str) -> str:
    """One bet as a line of text."""
    minutes = (db.as_utc(bet.commence_time) - now).total_seconds() / 60
    text = (f"**{bet.player} {describe(bet.side, bet.line)} {market_label(bet.market)}** at {book_label(bet.book)} {bet.price:+d}"
            f" (fair {odds_math.fair_american(bet.probability):+d}, est. edge {bet.ev:+.1%}) · {bet.event_name}, starts in {minutes:.0f} min")
    if bet.model_probability is not None:
        text += f" · model {'agrees' if bet.model_probability > 1 / bet.decimal else 'disagrees'}"
    if bet.link:
        text += f" · [open]({str(bet.link).replace('{state}', link_state.lower())})"
    return text


def message(bets: list, now: datetime, link_state: str) -> str:
    head = f"🎯 Sniper: {len(bets)} new paper bet{'s' if len(bets) != 1 else ''} (research only; the price may already have moved)"
    return "\n".join([head] + [f"• {line(b, now, link_state)}" for b in bets])[:1900]


def pending(session: Session, settings: Settings, now: datetime) -> list:
    """New automatic paper bets worth an alert: entered from a fresh price, game still to start, not alerted yet."""
    fresh = now - timedelta(minutes=settings.alert_max_age_minutes)
    rows = session.scalars(select(db.PaperBet).where(
        db.PaperBet.strategy == db.STRATEGY_AUTO, db.PaperBet.alerted_at.is_(None), db.PaperBet.replayed.is_(False),
        db.PaperBet.data_source != "sample", db.PaperBet.status == "open").order_by(db.PaperBet.conservative_ev.desc())).all()
    return [b for b in rows if db.as_utc(b.commence_time) > now and db.as_utc(b.price_observed_at) >= fresh
            and (b.ev or 0) >= settings.alert_min_ev][:settings.alert_max_per_pass]


def send(engine: Engine, settings: Settings, *, now: datetime | None = None, post=None) -> dict:
    """Post one message for the new paper bets. Returns counts; never raises for a delivery problem."""
    url = settings.alert_webhook_url
    if not url:
        return dict(status="off", sent=0)
    now = now or datetime.now(timezone.utc)
    db.init_db(engine)
    with Session(engine) as session, session.begin():
        bets = pending(session, settings, now)
        if not bets:
            return dict(status="nothing new", sent=0)
        try:
            if post is None:
                response = httpx.post(url, json={"content": message(bets, now, settings.link_state)}, timeout=10)
                response.raise_for_status()
            else:
                post(url, message(bets, now, settings.link_state))
        except Exception as error:               # the address must not appear in a log line or an error
            return dict(status=f"failed ({type(error).__name__})", sent=0)
        for bet in bets:
            bet.alerted_at = now
        return dict(status="sent", sent=len(bets))
