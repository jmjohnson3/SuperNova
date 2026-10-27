"""Confirm which sharp-edge alerts were actually bet, and report the money.

Alerts in bets.nfl_sharp_alerts carry a recommended `stake`. Nothing is ever placed automatically:
you place the bet by hand at the book and confirm it here, which sets `placed_at`. Only confirmed
rows count as real money in modeling/sharp_alert_report (its "Placed bets" line) and in the sticky
loss pause that sharp_watch checks before staking anything new.

  python -m nfl_pipeline.sharp_bets --pending            # staked alerts not yet confirmed
  python -m nfl_pipeline.sharp_bets --confirm 41 42      # mark these placed, at the stored stake
  python -m nfl_pipeline.sharp_bets --confirm 41 --stake 10   # placed at a different size
  python -m nfl_pipeline.sharp_bets --confirm 41 --price -115 # the price you actually got
  python -m nfl_pipeline.sharp_bets --unconfirm 41       # placed it by mistake / did not get the price
  python -m nfl_pipeline.sharp_bets --summary            # staked, realized, open
  python -m nfl_pipeline.sharp_bets --confirm 41 42 --late   # placed before kickoff, confirmed after it

A late confirmation (the game has started) must name every staked alert of those games, so it can only
record "I bet all of them", never pick winners; such rows are marked confirmed_late.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN

PENDING_SQL = """
SELECT alert_id, book, stat, player_name, side, fd_line, fd_price, minimum_price, ev, stake,
       commence_time_utc, home_team, away_team
FROM bets.nfl_sharp_alerts
WHERE stake > 0 AND placed_at IS NULL AND commence_time_utc > now()
ORDER BY commence_time_utc, ev DESC
"""


def pending(conn) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(PENDING_SQL)
        return [dict(r) for r in cur.fetchall()]


def confirm(conn, alert_ids: list[int], stake: float | None, placed: bool = True,
            price: int | None = None) -> list[dict]:
    """Mark alerts placed (or not). Refuses a game that has already started: a bet cannot be
    recorded after the fact, or the closing-line grade would be meaningless.

    `price` is the price actually obtained; left out, the alerted price is assumed. Everything is
    graded on the taken price, so slippage shows up as lost EV instead of hiding.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""UPDATE bets.nfl_sharp_alerts
            SET placed_at = CASE WHEN %(placed)s THEN now() END,
                stake = COALESCE(%(stake)s, stake),
                taken_price = CASE WHEN %(placed)s THEN COALESCE(%(price)s, taken_price, fd_price) END
            WHERE alert_id = ANY(%(ids)s)
              AND (NOT %(placed)s OR commence_time_utc > now())
            RETURNING alert_id, book, player_name, side, fd_line, fd_price, taken_price, stake, placed_at""",
                    {"ids": alert_ids, "stake": stake, "placed": placed, "price": price})
        rows = [dict(r) for r in cur.fetchall()]
    conn.commit()
    return rows


def confirm_late(conn, alert_ids: list[int], price: int | None = None) -> dict:
    """Record bets placed before kickoff but confirmed after it. All-or-nothing per game: every staked
    alert of each named game must be listed, or nothing is written."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("ALTER TABLE bets.nfl_sharp_alerts ADD COLUMN IF NOT EXISTS confirmed_late BOOLEAN")
        cur.execute("""SELECT alert_id, game_id FROM bets.nfl_sharp_alerts
                       WHERE stake > 0 AND game_id IN (SELECT game_id FROM bets.nfl_sharp_alerts WHERE alert_id = ANY(%s))""",
                    (alert_ids,))
        staked = {r["alert_id"]: r["game_id"] for r in cur.fetchall()}
        missing = sorted(set(staked) - set(alert_ids))
        unknown = sorted(set(alert_ids) - set(staked))
        if missing or unknown:
            conn.rollback()
            return dict(confirmed=[], refused="a late confirmation must list every staked alert of its games",
                        missing_from_request=missing, not_staked_or_unknown=unknown)
        cur.execute("""UPDATE bets.nfl_sharp_alerts
            SET placed_at = now(), confirmed_late = TRUE, taken_price = COALESCE(%(price)s, taken_price, fd_price)
            WHERE alert_id = ANY(%(ids)s) AND placed_at IS NULL
            RETURNING alert_id, book, player_name, side, fd_line, fd_price, taken_price, stake, placed_at""",
                    {"ids": alert_ids, "price": price})
        rows = [dict(r) for r in cur.fetchall()]
    conn.commit()
    return dict(confirmed=rows)


def summary(conn) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT count(*) FILTER (WHERE placed_at IS NOT NULL) AS placed,
                              COALESCE(SUM(stake) FILTER (WHERE placed_at IS NOT NULL), 0) AS staked,
                              count(*) FILTER (WHERE placed_at IS NOT NULL AND commence_time_utc > now()) AS open_bets,
                              count(*) FILTER (WHERE stake > 0 AND placed_at IS NULL) AS unconfirmed
                       FROM bets.nfl_sharp_alerts""")
        out = dict(cur.fetchone())
    from nfl_pipeline.sharp_watch import realized_profit
    out["realized_profit"] = realized_profit()  # graded by sharp_alert_report, which must run first
    return {k: (float(v) if v is not None else None) for k, v in out.items()}  # Decimal -> JSON number


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm", nargs="+", type=int, metavar="ALERT_ID")
    ap.add_argument("--unconfirm", nargs="+", type=int, metavar="ALERT_ID")
    ap.add_argument("--stake", type=float, help="actual stake, when it differs from the recommendation")
    ap.add_argument("--price", type=int, help="American price actually obtained (default: the alerted price)")
    ap.add_argument("--pending", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--late", action="store_true",
                    help="with --confirm: the bets were placed before kickoff; list every staked alert of those games")
    args = ap.parse_args()
    with psycopg2.connect(PG_DSN) as conn:
        if args.confirm and args.late:
            print(json.dumps(confirm_late(conn, args.confirm, args.price), indent=2, default=str))
        elif args.confirm:
            rows = confirm(conn, args.confirm, args.stake, placed=True, price=args.price)
            missed = sorted(set(args.confirm) - {r["alert_id"] for r in rows})
            print(json.dumps({"confirmed": rows, "not_updated_already_started_or_unknown": missed}, indent=2, default=str))
        if args.unconfirm:
            print(json.dumps({"unconfirmed": confirm(conn, args.unconfirm, None, placed=False)}, indent=2, default=str))
        if args.pending or not (args.confirm or args.unconfirm or args.summary):
            print(json.dumps({"pending": pending(conn)}, indent=2, default=str))
        if args.summary:
            print(json.dumps({"summary": summary(conn)}, indent=2, default=str))


if __name__ == "__main__":
    main()
