"""Cross-book arbitrage alerts between FanDuel and DraftKings. Information only; nothing is staked.

Reads the snapshots the scanners just saved (no Odds API credits): `bets.sharp_multi_quotes` for NBA and
NCAAF, the NFL watcher's payloads in `raw.nfl_api_responses`. An arbitrage is the two sides of one
market at the same line, one side at each book, whose prices together return more than was staked
whichever side wins. Each one is posted to Discord once.

    python -m sharp_pipeline.arb_watch [--dry-run] [--minutes 5]

Both prices have to still be there when the bets are placed. The quotes are the provider's, up to a few
minutes old; a book's own screen is the only proof.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras
import requests

from nfl_pipeline import sharp_math
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.markets import normalize_name

BOOKS = {"fanduel": "FanDuel", "draftkings": "DraftKings"}
MAX_QUOTE_AGE = 300      # seconds between a book's last update and the snapshot
MAX_QUOTE_SKEW = 120     # seconds between the two books' updates
MAX_SNAPSHOT_AGE = 240   # seconds: an older snapshot is recorded but not posted
MIN_RETURN = 0.001       # guaranteed return on the total staked
WEBHOOKS = {"nba": "NBA_DISCORD_WEBHOOK_URL", "ncaaf": "NCAAF_DISCORD_WEBHOOK_URL", "nfl": "NFL_DISCORD_WEBHOOK_URL"}
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bets.cross_book_arbs (
  arb_key TEXT PRIMARY KEY, first_seen TIMESTAMPTZ NOT NULL, sport TEXT NOT NULL, event_id TEXT NOT NULL,
  commence_time_utc TIMESTAMPTZ NOT NULL, guaranteed_return NUMERIC NOT NULL, payload JSONB NOT NULL,
  delivery_status TEXT NOT NULL DEFAULT 'pending', discord_message_id TEXT);
"""


def _updated(book: dict, market: dict, observed: datetime) -> datetime | None:
    stamp = safe_time(market.get("last_update") or book.get("last_update"))
    if stamp is None or stamp > observed or (observed - stamp).total_seconds() > MAX_QUOTE_AGE:
        return None
    return stamp


def quotes(book: dict, event: dict, observed: datetime) -> dict[tuple, dict]:
    """(market, who, line) -> {'a': price, 'b': price, links, labels, updated}. Side a is over / home."""
    out: dict[tuple, dict] = {}
    home, away = event.get("home_team"), event.get("away_team")
    for market in book.get("markets", []):
        key, updated = str(market.get("key") or ""), _updated(book, market, observed)
        if updated is None:
            continue
        for o in market.get("outcomes", []):
            name, point, who = str(o.get("name") or ""), o.get("point"), o.get("description")
            if sharp_math.implied(o.get("price")) is None:
                continue
            if name in (home, away):  # moneyline or spread, keyed on the home team's line
                if key.startswith("h2h"):
                    k, label = (key, "game", 0.0), name
                elif point is not None:
                    k = (key, "game", float(point) if name == home else -float(point))
                    label = f"{name} {float(point):+g}"
                else:
                    continue
                side = "a" if name == home else "b"
            elif name.lower() in ("over", "under") and point is not None:
                k = (key, normalize_name(who) if who else "game", float(point))
                side, label = ("a" if name.lower() == "over" else "b"), f"{name.upper()} {float(point):g}"
            else:
                continue
            rec = out.setdefault(k, dict(updated=updated, player=who))
            rec[side], rec[side + "_link"], rec[side + "_label"] = int(o["price"]), o.get("link"), label
    return {k: v for k, v in out.items() if "a" in v and "b" in v}


def find_arbs(event: dict, observed: datetime, min_return: float = MIN_RETURN) -> list[dict]:
    """Every same-line pair of opposite sides, one at each book, that returns more than the total staked."""
    books = {b.get("key"): quotes(b, event, observed) for b in event.get("bookmakers", []) if b.get("key") in BOOKS}
    if len(books) < 2:
        return []
    first, second = (books[k] for k in BOOKS)
    found = []
    for key in first.keys() & second.keys():
        x, y = first[key], second[key]
        if abs((x["updated"] - y["updated"]).total_seconds()) > MAX_QUOTE_SKEW:
            continue
        for (book_a, qa), (book_b, qb) in (((list(BOOKS)[0], x), (list(BOOKS)[1], y)),
                                           ((list(BOOKS)[1], y), (list(BOOKS)[0], x))):
            cost = sharp_math.implied(qa["a"]) + sharp_math.implied(qb["b"])  # staked per 1 returned
            if cost >= 1 or 1 / cost - 1 < min_return:
                continue
            market, who, line = key
            legs = [dict(book=book_a, label=qa["a_label"], price=qa["a"], link=qa.get("a_link"),
                         share=sharp_math.implied(qa["a"]) / cost, quote_age=(observed - qa["updated"]).total_seconds()),
                    dict(book=book_b, label=qb["b_label"], price=qb["b"], link=qb.get("b_link"),
                         share=sharp_math.implied(qb["b"]) / cost, quote_age=(observed - qb["updated"]).total_seconds())]
            found.append(dict(market=market, who=who, line=line, player=qa.get("player") if who != "game" else None,
                              guaranteed_return=1 / cost - 1, legs=legs))
    return sorted(found, key=lambda a: -a["guaranteed_return"])


def arb_key(sport: str, event_id: str, arb: dict) -> str:
    return "|".join([sport, event_id, arb["market"], arb["who"], f"{arb['line']:g}", arb["legs"][0]["book"]])


def card(sport: str, event: dict, arb: dict, observed: datetime) -> str:
    start = safe_time(event["commence_time"])
    subject = f"{arb['player']} — {arb['market'].replace('player_', '').replace('_', ' ')}" if arb["player"] else arb["market"]
    lines = [f"**ARBITRAGE — {sport.upper()}** FanDuel / DraftKings", f"**{subject}**"]
    for leg in arb["legs"]:
        text = f"{BOOKS[leg['book']]}: {leg['label']} **{leg['price']:+d}** — ${100 * leg['share']:.2f} of every $100"
        link = leg.get("link")
        if isinstance(link, str) and link.startswith("https://") and not any(c in link for c in "\r\n<>"):
            text += f" <{link}>"
        lines.append(text)
    ages = " and ".join(f"{leg['quote_age']:.0f}s" for leg in arb["legs"])
    lines += [f"Returns **{arb['guaranteed_return']:+.2%}** on the total staked whichever side wins, if both fill at these prices.",
              f"{event.get('away_team')} @ {event.get('home_team')} | starts <t:{int(start.timestamp())}:t> | "
              f"quotes {ages} old when seen (<t:{int(observed.timestamp())}:R>)",
              "Information only. Check both prices on the books' own screens first; if either has moved, the arbitrage is gone."]
    return "\n".join(lines)[:1950]


def webhook(sport: str) -> str:
    from supernovabets_config import _saved_windows_env
    for name in (WEBHOOKS.get(sport), "MLB_DISCORD_WEBHOOK_URL"):
        url = name and (os.getenv(name) or _saved_windows_env(name))
        if url:
            return url
    return ""


def send(sport: str, text: str) -> str:
    url = webhook(sport)
    if not url:
        raise RuntimeError("webhook unavailable")
    response = requests.post(url, params={"wait": "true"}, json={"content": text, "allowed_mentions": {"parse": []}}, timeout=20)
    if response.status_code != 200:
        raise RuntimeError(f"Discord HTTP {response.status_code}")
    return str(response.json().get("id") or "")


def snapshots(conn, since: datetime) -> list[tuple[str, datetime, dict]]:
    """(sport, observed, event) for every snapshot saved since `since`."""
    out = []
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('bets.sharp_multi_quotes'), to_regclass('raw.nfl_api_responses')")
        multi, nfl = cur.fetchone()
        if multi:
            cur.execute("SELECT sport, observed_at_utc, payload FROM bets.sharp_multi_quotes WHERE observed_at_utc >= %s", (since,))
            out += [(s, t, p) for s, t, p in cur.fetchall()]
        if nfl:
            cur.execute("""SELECT 'nfl', fetched_at_utc, payload FROM raw.nfl_api_responses
                           WHERE provider='oddsapi_watch' AND endpoint='nfl_player_props' AND fetched_at_utc >= %s""", (since,))
            out += [(s, t, p) for s, t, p in cur.fetchall()]
    return [(s, t, p) for s, t, p in out if isinstance(p, dict) and p.get("bookmakers") and safe_time(p.get("commence_time"))]


def run(conn, *, now: datetime | None = None, minutes: float = 5, dry_run: bool = False, poster=send) -> dict:
    now = now or datetime.now(timezone.utc)
    result = dict(status="ok", snapshots=0, found=0, new=0, sent=0, failed=0, stale=0, arbs=[])
    if not dry_run:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
    for sport, observed, event in sorted(snapshots(conn, now - timedelta(minutes=minutes)), key=lambda s: s[1]):
        result["snapshots"] += 1
        start = safe_time(event["commence_time"])
        if observed >= start:
            continue
        for arb in find_arbs(event, observed):
            result["found"] += 1
            key = arb_key(sport, str(event.get("id")), arb)
            summary = dict(key=key, guaranteed_return=round(arb["guaranteed_return"], 4),
                           legs=[f"{leg['book']} {leg['label']} {leg['price']:+d}" for leg in arb["legs"]])
            if dry_run:
                result["arbs"].append(summary)
                continue
            fresh = (now - observed).total_seconds() <= MAX_SNAPSHOT_AGE and now < start
            with conn.cursor() as cur:  # claim before posting: an arbitrage is never posted twice
                cur.execute("""INSERT INTO bets.cross_book_arbs (arb_key, first_seen, sport, event_id, commence_time_utc,
                                 guaranteed_return, payload, delivery_status) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                               ON CONFLICT DO NOTHING""",
                            (key, observed, sport, str(event.get("id")), start, arb["guaranteed_return"],
                             psycopg2.extras.Json(dict(arb, home_team=event.get("home_team"), away_team=event.get("away_team"))),
                             "sending" if fresh else "stale"))
                claimed = cur.rowcount == 1
            conn.commit()
            if not claimed:
                continue
            result["new"] += 1
            result["arbs"].append(summary)
            if not fresh:
                result["stale"] += 1
                continue
            try:
                message_id, status = poster(sport, card(sport, event, arb, observed)), "sent"
                result["sent"] += 1
            except Exception as exc:  # never log a webhook-bearing exception
                message_id, status = type(exc).__name__, "failed"
                result["failed"] += 1
            with conn.cursor() as cur:
                cur.execute("UPDATE bets.cross_book_arbs SET delivery_status=%s, discord_message_id=%s WHERE arb_key=%s",
                            (status, message_id, key))
            conn.commit()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="list arbitrages without recording or posting them")
    ap.add_argument("--minutes", type=float, default=5, help="look at snapshots saved in the last N minutes")
    args = ap.parse_args()
    with psycopg2.connect(PG_DSN) as conn:
        print(json.dumps(run(conn, minutes=args.minutes, dry_run=args.dry_run), indent=2, default=str))


if __name__ == "__main__":
    main()
