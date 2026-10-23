"""Research parlay of the day: the two best unstaked sharp-gap legs, combined. Never a bet.

Each leg is a gap the scanner found at +3% or better against Pinnacle's no-vig price, inside the
stake window, that carries no stake (the daily cap, the per-game cap or a stake elsewhere on the same
player stopped it). Legs from different games are independent, so the parlay's fair probability is the
product of the legs' and its EV compounds: two +4% legs make about +8%. It is leverage on edges we
already have, not a new edge, and it hits far less often; it is posted and graded to find out whether
it is worth staking, which is a separate decision.

Rules: at most one parlay per ET day; both legs at the same book (a parlay cannot span books), from
different games, neither started, each still within its stake window and seen in the last 15 minutes;
ladder rungs (research pricing) are never used. Rows in bets.nfl_sharp_parlays are append-only.
Grading (modeling/sharp_parlay_report) values each leg at Pinnacle's close and settles on results.

Display and research only: this module is not part of the strategy registration.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2.extras

from nfl_pipeline import sharp_math
from nfl_pipeline.sharp_links import BOOK_LABEL, book_link

_ET = ZoneInfo("America/New_York")
MIN_LEG_EV = 0.03
FRESH_MINUTES = 15
START_BUFFER_MINUTES = 10
LEGS = 2

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bets.nfl_sharp_parlays (
  parlay_id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  et_day DATE NOT NULL UNIQUE,
  book TEXT NOT NULL,
  leg_alert_ids BIGINT[] NOT NULL,
  decimal_price NUMERIC NOT NULL,
  fair_probability NUMERIC NOT NULL,
  ev NUMERIC NOT NULL,
  legs JSONB NOT NULL,
  link TEXT,
  posted BOOLEAN NOT NULL DEFAULT FALSE,
  research_only BOOLEAN NOT NULL DEFAULT TRUE
);
"""

CANDIDATES_SQL = """
SELECT alert_id, event_id, game_id, commence_time_utc, home_team, away_team, book, stat, player_name,
       player_name_norm, side, fd_line, fd_price, fd_link, fair_probability, ev, alerted_at
FROM bets.nfl_sharp_alerts
WHERE tier = 'alert' AND stake = 0 AND ev >= %(min_ev)s
  AND alerted_at >= %(fresh)s AND commence_time_utc > %(start_after)s
  AND COALESCE((alert_provenance ->> 'ladder')::boolean, FALSE) = FALSE
ORDER BY ev DESC
"""


def decimal_price(american: float) -> float:
    american = float(american)
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / -american)


def american_price(decimal: float) -> int:
    return int(round((decimal - 1.0) * 100)) if decimal >= 2.0 else int(round(-100.0 / (decimal - 1.0)))


def price(legs: list[dict]) -> dict[str, float]:
    """Combined decimal price, fair probability (independent legs) and EV."""
    decimal, fair = 1.0, 1.0
    for leg in legs:
        decimal *= decimal_price(leg["fd_price"])
        fair *= float(leg["fair_probability"])
    return dict(decimal_price=decimal, fair_probability=fair, ev=fair * decimal - 1.0)


def choose(candidates: list[dict]) -> list[dict] | None:
    """Highest-EV pair at one book from different games; None when no pair qualifies."""
    best = None
    for pair in combinations(candidates, LEGS):
        if len({leg["book"] for leg in pair}) != 1 or len({leg["game_id"] for leg in pair}) != LEGS:
            continue
        value = price(list(pair))["ev"]
        if best is None or value > best[0]:
            best = (value, list(pair))
    return best[1] if best else None


def _game(leg: dict) -> dict:
    return dict(start=leg["commence_time_utc"], home=leg["home_team"], away=leg["away_team"])


def _leg_edge(leg: dict) -> dict:
    return dict(book=leg["book"], stat=leg["stat"], player=leg["player_name"], player_norm=leg["player_name_norm"],
                side=leg["side"], line=float(leg["fd_line"]), price=int(leg["fd_price"]), link=leg["fd_link"])


def parlay_link(legs: list[dict]) -> str | None:
    """One-tap slip for FanDuel (state-resolved); other books have no verified multi-leg format."""
    if legs[0]["book"] != "fanduel":
        return None
    from nfl_pipeline.fanduel_links import parlay_betslip_url
    rows = [dict(_leg_edge(leg), away=leg["away_team"], home=leg["home_team"], market=None) for leg in legs]
    try:
        return parlay_betslip_url([leg["fd_link"] for leg in legs], rows=rows)
    except Exception:
        return None


def format_parlay(legs: list[dict], priced: dict, link: str | None) -> str:
    book = BOOK_LABEL.get(legs[0]["book"], legs[0]["book"].title())
    lines = [f"**RESEARCH PARLAY - not a bet** ({book}, {len(legs)} legs, different games)",
             f"Combined {american_price(priced['decimal_price']):+d} | fair {priced['fair_probability']:.1%} | "
             f"EV {priced['ev']:+.1%} (legs priced against Pinnacle no-vig; graded at the close)"]
    for leg in legs:
        url, kind = book_link(_leg_edge(leg), _game(leg))
        label = "" if not url else f" [leg link](<{url}>)"
        kickoff = int(leg["commence_time_utc"].timestamp())
        lines.append(f"- {leg['player_name']} {leg['side'].upper()} {float(leg['fd_line']):g} "
                     f"{leg['stat'].replace('_', ' ')} ({leg['away_team']} @ {leg['home_team']}, <t:{kickoff}:R>) "
                     f"{int(leg['fd_price']):+d}, EV {float(leg['ev']):+.1%}{label}")
    if link:
        lines.append(f"[Add both to slip](<{link}>)")
    lines.append("Research only: tracked to decide whether parlays of unstaked edges deserve a stake.")
    return "\n".join(lines)


def suggest(conn, now: datetime, post) -> dict[str, Any]:
    """Pick, store and post today's research parlay if none exists yet. Never raises into the scanner."""
    day = now.astimezone(_ET).date()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(SCHEMA_SQL)
        cur.execute("SELECT 1 FROM bets.nfl_sharp_parlays WHERE et_day = %s", (day,))
        if cur.fetchone():
            conn.commit()
            return dict(status="already_suggested_today")
        cur.execute(CANDIDATES_SQL, dict(min_ev=MIN_LEG_EV, fresh=now - timedelta(minutes=FRESH_MINUTES),
                                         start_after=now + timedelta(minutes=START_BUFFER_MINUTES)))
        legs = choose([dict(r) for r in cur.fetchall()])
        if not legs:
            conn.commit()
            return dict(status="no_qualifying_pair")
        priced = price(legs)
        link = parlay_link(legs)
        stored = [{k: (v.isoformat() if hasattr(v, "isoformat") else float(v) if k in ("fd_line", "fair_probability", "ev") else v)
                   for k, v in leg.items()} for leg in legs]
        cur.execute("""INSERT INTO bets.nfl_sharp_parlays (et_day, book, leg_alert_ids, decimal_price, fair_probability, ev, legs, link)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (et_day) DO NOTHING RETURNING parlay_id""",
                    (day, legs[0]["book"], [leg["alert_id"] for leg in legs], priced["decimal_price"],
                     priced["fair_probability"], priced["ev"], psycopg2.extras.Json(stored), link))
        row = cur.fetchone()
        conn.commit()
    if not row:
        return dict(status="already_suggested_today")
    try:
        post(format_parlay(legs, priced, link))
        with conn.cursor() as cur:
            cur.execute("UPDATE bets.nfl_sharp_parlays SET posted = TRUE WHERE parlay_id = %s", (row[0],))
        conn.commit()
    except Exception as exc:  # stored and graded even if Discord is down
        return dict(status="stored_not_posted", parlay_id=row[0], error=type(exc).__name__)
    return dict(status="suggested", parlay_id=row[0], ev=round(priced["ev"], 4), legs=[l["alert_id"] for l in legs])
