"""Phone-recorded price checks and an every-alert paper ledger for the NBA paper alerts.

Two questions, kept apart:

1. Assumed ledger: what would every delivered alert have returned as a flat $1 bet at the alerted
   price? It needs nothing from anyone, and it assumes the price was on the screen, which is unverified.
2. Reply checks: was the price really there? A reply to the alert in Discord records it:

       y             the alerted line and price are offered        (y -125 when the price differs)
       b 5           I placed $5 on it at the alerted price        (b 5 -125 at another price; b alone is $1)
       m 8.5 -120    the line moved; this is what is offered now
       n             the bet is gone
       s             suspended or locked
   Add "max 20" to a y or b reply to record the largest stake the bet slip would take.

The check time is Discord's own timestamp on the reply, so it cannot be back-dated; an edited reply is
not evidence. A check is timely within 5 minutes of the alert and before the tip; the last timely reply
is the check, so a mistyped one can be corrected inside the window. Coverage and the three-week span are
counted on checking days: game dates with at least one reply for that book.

Registration `nba-reply-check-v1` reads the alerts the frozen `nba-prop-price-check-v1` experiment
selects and never changes them. Research only: a passed review prompts a manual decision and never
authorizes a stake. A "b" reply is the user's own record of their own decision; nothing here places or
requests a wager. Reading Discord and posting live in discord_replies, which is not pinned.

    python -m sharp_pipeline.reply_trial [--register]
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg2.extras

from nfl_pipeline import experiment_registry as registry, sharp_math
from nfl_pipeline.context_contract import safe_time
from sharp_pipeline import nba_experiment

ROOT = Path(__file__).resolve().parents[2]
NAME = "nba-reply-check-v1"
SOURCE = nba_experiment.NAME
BOOKS = nba_experiment.BOOKS
BOOK_LABEL = {"fanduel": "FanDuel", "draftkings": "DraftKings", "espnbet": "theScore Bet"}
MAX_CHECK_DELAY_SECONDS = 300
DEFAULT_STAKE = 1.0
MAX_STAKE = 10_000.0
REVIEW = dict(nba_experiment.REVIEW)
DISCORD_EPOCH_MS = 1420070400000
MODULES = ("sharp_pipeline.reply_trial", "sharp_pipeline.scan", "sharp_pipeline.report", "sharp_pipeline.outcomes",
           "sharp_pipeline.nba_experiment", "sharp_pipeline.paper_alerts", "nfl_pipeline.sharp_math",
           "nfl_pipeline.context_contract", "nfl_pipeline.markets", "nfl_pipeline.integrity",
           "nfl_pipeline.experiment_registry")
VERBS = {"available": {"y", "yes", "ok", "there", "available", "✅", "\U0001F44D"},
         "placed": {"b", "bet", "placed", "took"},
         "changed_line": {"m", "moved", "move", "line"},
         "disappeared": {"n", "no", "gone", "x", "❌"},
         "suspended": {"s", "susp", "suspended", "locked", "lock", "\U0001F512"}}
NUMBER = re.compile(r"([+-]?)\$?(\d+(?:\.\d+)?)$")


def snowflake_time(message_id) -> datetime | None:
    """When Discord created a message: the timestamp is part of its ID."""
    try:
        return datetime.fromtimestamp(((int(message_id) >> 22) + DISCORD_EPOCH_MS) / 1000, timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _price(sign: str, digits: str) -> int | None:
    value = float(digits)
    if value != int(value) or sharp_math.implied(int(value) * (-1 if sign == "-" else 1)) is None:
        return None
    return int(value) * (-1 if sign == "-" else 1)


def parse_reply(text: str) -> dict | None:
    """One reply -> what the screen showed. Anything not understood is None, never a guess."""
    tokens = [t for t in re.split(r"[\s,;]+", (text or "").strip().lower()) if t]
    if not tokens:
        return None
    out = dict(status=None, placed=False, line=None, price=None, stake=None, max_stake=None)
    if "max" in tokens:
        at = tokens.index("max")
        found = NUMBER.match(tokens[at + 1]) if at + 1 < len(tokens) else None
        if not found or found.group(1) or not 0 < float(found.group(2)) <= MAX_STAKE:
            return None
        out["max_stake"] = float(found.group(2))
        del tokens[at:at + 2]
        if not tokens:
            return None
    verb = next((status for status, words in VERBS.items() if tokens[0] in words), None)
    numbers = tokens[1:] if verb else tokens  # a bare price ("-125") means: there, at this price
    parsed = [NUMBER.match(t) for t in numbers]
    if not all(parsed) or (verb is None and len(parsed) != 1):
        return None
    parsed = [(m.group(1), m.group(2)) for m in parsed]
    if verb in ("disappeared", "suspended"):
        return dict(out, status=verb) if not parsed and out["max_stake"] is None else None
    if verb == "changed_line":
        if out["max_stake"] is not None or len(parsed) not in (0, 2):
            return None
        if parsed:
            (line_sign, line), (sign, digits) = parsed
            out["line"], out["price"] = float(line) * (-1 if line_sign == "-" else 1), _price(sign, digits)
            if out["price"] is None:
                return None
        return dict(out, status="changed_line")
    if verb == "placed":
        for sign, digits in parsed:
            if sign:  # a price always carries its sign; an unsigned number is the stake
                if out["price"] is not None or _price(sign, digits) is None:
                    return None
                out["price"] = _price(sign, digits)
            elif out["stake"] is not None or not 0 < float(digits) <= MAX_STAKE:
                return None
            else:
                out["stake"] = float(digits)
        return dict(out, status="available", placed=True, stake=out["stake"] or DEFAULT_STAKE)
    if len(parsed) > 1:
        return None
    if parsed:
        out["price"] = _price(*parsed[0])
        if out["price"] is None:
            return None
    return dict(out, status="available")


def check(alert: dict, replies: list[dict]) -> dict:
    """The alert's check (last timely reply) and the bet reported on it, if any."""
    posted = snowflake_time(alert.get("discord_message_id")) or safe_time(alert.get("sent_at"))
    tip = safe_time(alert.get("commence_time_utc"))
    out = dict(replies=len(replies), readable_replies=0, timely_check=False, check_status=None, displayed_line=None,
               displayed_price=None, check_delay_seconds=None, max_stake=None, bet_stake=None, bet_price=None)
    if not posted or not tip:
        return out
    for reply in sorted(replies, key=lambda r: (r["replied_at"], str(r.get("reply_message_id")))):
        parsed = None if reply.get("edited") else parse_reply(reply.get("raw_text") or "")
        at = safe_time(reply["replied_at"])
        if parsed is None or at is None or at >= tip or at < posted:
            continue
        out["readable_replies"] += 1
        price = parsed["price"] if parsed["price"] is not None else int(alert["price"])
        if parsed["placed"]:
            out.update(bet_stake=parsed["stake"], bet_price=price)
        elif parsed["status"] != "available":  # "b 5" then "n": the bet did not go through after all
            out.update(bet_stake=None, bet_price=None)
        if (at - posted).total_seconds() > MAX_CHECK_DELAY_SECONDS:
            continue
        same_line = parsed["status"] == "available"
        out.update(timely_check=True, check_status=parsed["status"], check_delay_seconds=(at - posted).total_seconds(),
                   displayed_line=float(alert["line"]) if same_line else parsed["line"],
                   displayed_price=price if same_line else parsed["price"], max_stake=parsed["max_stake"])
    return out


def evidence(frame: pd.DataFrame, replies: dict[int, list[dict]]) -> pd.DataFrame:
    """Attach each alert's check, the EV of the checked price at the close and any reported bet."""
    if frame.empty:
        return frame
    frame = frame.reset_index(drop=True)
    rows = []
    for r in frame.to_dict("records"):
        found = check(r, replies.get(int(r["alert_id"]), []))
        available = found["timely_check"] and found["check_status"] == "available"
        exact = r.get("close_method") == "exact" and pd.notna(r.get("close_probability"))
        found["exact_offer_available"] = bool(available)
        found["checked_price_ev"] = (sharp_math.ev(float(r["close_probability"]), found["displayed_price"])
                                     if available and exact else None)
        result, stake = r.get("result"), found["bet_stake"]
        found["bet_profit"] = None
        if stake and result in ("win", "loss", "push"):
            payout = found["bet_price"] / 100 if found["bet_price"] > 0 else 100 / abs(found["bet_price"])
            found["bet_profit"] = stake * {"win": payout, "loss": -1.0, "push": 0.0}[result]
        rows.append(found)
    return pd.concat([frame, pd.DataFrame(rows)], axis=1)


def ledger(frame: pd.DataFrame, profit: str = "profit") -> dict:
    """Flat $1 on every row at its own price. Unsettled rows are counted, never guessed."""
    if frame.empty:
        return dict(alerts=0, settled=0, wins=0, losses=0, pushes=0, pending=0, units=0.0, roi=None,
                    ev_when_alerted=None, ev_at_close=None, roi_by_game=None)
    settled = frame[frame.result.isin(["win", "loss", "push"]) & pd.to_numeric(frame[profit], errors="coerce").notna()]
    close = pd.to_numeric(frame.ev_at_close, errors="coerce").dropna()
    return dict(alerts=int(len(frame)), settled=int(len(settled)),
                wins=int(settled.result.eq("win").sum()), losses=int(settled.result.eq("loss").sum()),
                pushes=int(settled.result.eq("push").sum()), pending=int(len(frame) - len(settled)),
                units=float(settled[profit].astype(float).sum()),
                roi=float(settled[profit].astype(float).mean()) if len(settled) else None,
                ev_when_alerted=float(frame.ev.astype(float).mean()),
                ev_at_close=float(close.mean()) if len(close) else None,
                roi_by_game=nba_experiment.clustered_interval(settled, profit, "event_id") if len(settled) else None)


def book_review(frame: pd.DataFrame) -> dict:
    """Review gates for one book, over the alerts posted since this trial was registered."""
    if frame.empty:
        return dict(alerts=0, status="collecting", reasons=["no alerts since registration"], betting_authorized=False)
    frame = frame.copy()
    close = pd.to_numeric(frame.ev_at_close, errors="coerce").where(frame.close_method.eq("exact"))
    frame["ev_at_close"] = close
    checking_days = set(frame.loc[frame.readable_replies > 0, "day"])
    on_checking_days = frame.day.isin(checking_days)
    # The span is the span of the checking itself: three weeks of alerts with one checked night prove little.
    dates = pd.Series(pd.to_datetime(sorted(checking_days)))
    span = int((dates.max() - dates.min()).days) + 1 if len(dates) else 0
    weeks = int(((dates - dates.min()).dt.days // 7).nunique()) if len(dates) else 0
    timely, available = frame.timely_check.astype(bool), frame.exact_offer_available.astype(bool)
    priced = pd.to_numeric(frame.checked_price_ev, errors="coerce").notna()
    intervals = {label: nba_experiment.clustered_interval(frame, value, cluster) for label, value, cluster in (
        ("close_by_game", "ev_at_close", "event_id"), ("close_by_day", "ev_at_close", "day"),
        ("checked_by_game", "checked_price_ev", "event_id"), ("checked_by_day", "checked_price_ev", "day"))}
    coverage = float(close.notna().mean())
    check_coverage = float(timely[on_checking_days].mean()) if on_checking_days.any() else 0.0
    availability = float(available.sum() / timely.sum()) if timely.any() else 0.0
    checked_close_coverage = float(priced.sum() / available.sum()) if available.any() else 0.0
    gates = dict(decisions=int(close.notna().sum()) >= REVIEW["min_decisions"],
                 collection_span=span >= REVIEW["min_span_days"], active_days=len(checking_days) >= REVIEW["min_active_days"],
                 weeks=weeks >= REVIEW["min_weeks"], games=frame.event_id.nunique() >= REVIEW["min_games"],
                 exact_close_coverage=coverage >= REVIEW["min_close_coverage"],
                 closing_edge=bool(close.notna().any() and close.mean() >= REVIEW["min_mean_close_ev"]),
                 reply_check_coverage=check_coverage >= REVIEW["min_check_coverage"],
                 offer_availability=availability >= REVIEW["min_available_fraction"],
                 checked_prices=int(priced.sum()) >= REVIEW["min_priced_checks"],
                 checked_close_coverage=checked_close_coverage >= REVIEW["min_checked_close_coverage"])
    for label, ci in intervals.items():
        gates[label] = ci["lower_95"] is not None and ci["lower_95"] > REVIEW["min_cluster_lower"]
    limits = pd.to_numeric(frame.max_stake, errors="coerce").dropna()
    return dict(alerts=int(len(frame)), games=int(frame.event_id.nunique()), days=int(frame.day.nunique()),
                span_days=span, weeks=weeks, with_exact_close=int(close.notna().sum()), exact_close_coverage=coverage,
                mean_close_ev=float(close.mean()) if close.notna().any() else None,
                checking_days=len(checking_days), alerts_on_checking_days=int(on_checking_days.sum()),
                timely_checks=int(timely.sum()), check_coverage=check_coverage,
                check_coverage_all_alerts=float(timely.mean()),
                available=int(available.sum()), availability=availability,
                disappeared=int((timely & frame.check_status.eq("disappeared")).sum()),
                suspended=int((timely & frame.check_status.eq("suspended")).sum()),
                changed_line=int((timely & frame.check_status.eq("changed_line")).sum()),
                priced_checks=int(priced.sum()), mean_checked_price_ev=intervals["checked_by_game"]["mean"],
                median_check_delay_seconds=(float(frame.loc[timely, "check_delay_seconds"].median()) if timely.any() else None),
                stake_limits_recorded=int(len(limits)), median_stake_limit=float(limits.median()) if len(limits) else None,
                intervals=intervals, gates={k: bool(v) for k, v in gates.items()},
                status="execution_review_only" if all(gates.values()) else "collecting",
                reasons=[k for k, passed in gates.items() if not passed], betting_authorized=False)


def reported_bets(frame: pd.DataFrame) -> dict:
    bets = frame[pd.to_numeric(frame.bet_stake, errors="coerce").notna()] if not frame.empty else frame
    settled = bets[pd.to_numeric(bets.bet_profit, errors="coerce").notna()] if len(bets) else bets
    staked = float(settled.bet_stake.astype(float).sum()) if len(settled) else 0.0
    return dict(bets=int(len(bets)), settled=int(len(settled)), staked=staked,
                wins=int(settled.result.eq("win").sum()) if len(settled) else 0,
                losses=int(settled.result.eq("loss").sum()) if len(settled) else 0,
                profit=float(settled.bet_profit.astype(float).sum()) if len(settled) else 0.0,
                roi=(float(settled.bet_profit.astype(float).sum()) / staked) if staked else None,
                open_stake=float(bets.bet_stake.astype(float).sum()) - staked if len(bets) else 0.0)


def requoted(frame: pd.DataFrame, checks: dict[int, dict]) -> dict:
    """DraftKings alerts valued at the provider's re-quote a minute later; a price that left is no bet."""
    rows = []
    for r in frame.to_dict("records") if not frame.empty else []:
        found = checks.get(int(r["alert_id"]))
        if found is None or found["status"] == "unknown" or r.get("result") not in ("win", "loss", "push"):
            continue
        profit = 0.0
        if found["status"] == "available":
            payout = found["price"] / 100 if found["price"] > 0 else 100 / abs(found["price"])
            profit = {"win": payout, "loss": -1.0, "push": 0.0}[r["result"]]
        rows.append(dict(bet=found["status"] == "available", profit=profit))
    bets = [r for r in rows if r["bet"]]
    return dict(answered=len(rows), bets=len(bets), units=sum(r["profit"] for r in bets),
                roi=(sum(r["profit"] for r in bets) / len(bets)) if bets else None)


def configuration() -> dict:
    return dict(sport="nba", source_experiment=SOURCE, books=list(BOOKS), review=REVIEW,
                population="delivered paper alerts of the source experiment posted at or after this registration",
                check_evidence="discord reply to the alert; check time is the Discord message timestamp",
                check_selection="last unedited, understood reply within the window and before the tip",
                max_check_delay_seconds=MAX_CHECK_DELAY_SECONDS,
                coverage="timely checks / alerts on game dates with at least one understood reply for the book",
                span="first to last checking day for the book; active days are checking days",
                reported_bet="last understood reply before the tip: b sets it; gone, suspended or moved clears it",
                assumed_ledger="every delivered alert, flat one unit at the alerted price; context, not a gate",
                reported_bets="user-reported stakes; never requested, placed or sized here",
                default_reported_stake=DEFAULT_STAKE, research_only=True, automatic_betting_authorization=False)


def load_registration():
    return registry.load_experiment(NAME, root=ROOT, store=nba_experiment.STORE)


def active():
    doc = registry.check_experiment(NAME, root=ROOT, store=nba_experiment.STORE)
    if doc["config"] != configuration():
        raise ValueError("Reply trial configuration drift")
    return doc


def register():
    files = {m: ROOT / "src" / Path(*m.split(".")).with_suffix(".py") for m in MODULES}
    return registry.register_experiment(NAME, configuration(), files, root=ROOT, store=nba_experiment.STORE)


def load(conn) -> tuple[pd.DataFrame, dict, dict]:
    """Delivered alerts of started games with close and result, their replies, and DraftKings re-quotes."""
    from sharp_pipeline import outcomes, report
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT a.alert_id, a.sent_at, a.discord_message_id, d.*
                       FROM bets.nba_paper_price_alerts a JOIN bets.sharp_multi_decisions d USING (decision_id)
                       WHERE a.delivery_status='sent' AND a.experiment_id=%s AND d.commence_time_utc <= now()
                       ORDER BY a.alert_id""", (SOURCE,))
        raw = pd.DataFrame([dict(r) for r in cur.fetchall()])
        replies, checks, closes = {}, {}, {}
        if raw.empty:
            return raw, replies, checks
        cur.execute("""SELECT sport, event_id, observed_at_utc, payload FROM bets.sharp_multi_quotes
                       WHERE sport='nba' AND event_id = ANY(%s) AND observed_at_utc < commence_time_utc
                         AND observed_at_utc >= commence_time_utc - %s * interval '1 minute'""",
                    (sorted(set(raw.event_id)), report.MAX_CLOSE_AGE_MINUTES))
        for r in cur.fetchall():
            key = (r["sport"], r["event_id"], report.payload_kind(r["payload"]))
            if key not in closes or r["observed_at_utc"] > closes[key][0]:
                closes[key] = (r["observed_at_utc"], r["payload"])
        cur.execute("SELECT to_regclass('bets.nba_alert_replies') AS replies, to_regclass('bets.draftkings_price_checks') AS dk")
        tables = cur.fetchone()
        if tables["replies"]:
            cur.execute("SELECT * FROM bets.nba_alert_replies WHERE alert_id IS NOT NULL")
            for r in cur.fetchall():
                replies.setdefault(int(r["alert_id"]), []).append(dict(r))
        if tables["dk"]:
            cur.execute("SELECT alert_id, status, price FROM bets.draftkings_price_checks")
            checks = {int(r["alert_id"]): dict(r) for r in cur.fetchall()}
    frame = report.decisions_frame(raw, closes, per_book=True)
    # Results as outcomes.attach grades them, without its model shadow (minutes of projection matching).
    players, _, _ = outcomes.nba_actuals(conn, sorted(set(frame.day)))
    results = []
    for r in frame.to_dict("records"):
        stats = players.get((r["day"], outcomes.match_name(r.get("player") or "")))
        actual = stats[outcomes.MARKET_STAT[r["market"]]] if stats is not None and r["market"] in outcomes.MARKET_STAT else None
        results.append(outcomes.grade(r, actual, None, None))
    frame["result"] = results
    frame["profit"] = [outcomes.profit(result, price) for result, price in zip(results, frame.price)]
    return frame, replies, checks


def summarize(frame: pd.DataFrame, replies: dict, checks: dict, registration: dict | None, now: datetime) -> dict:
    frame = evidence(frame, replies)
    started = safe_time((registration or {}).get("registered_at"))
    posted = [snowflake_time(m) or safe_time(s) for m, s in zip(frame.discord_message_id, frame.sent_at)] if not frame.empty else []
    in_trial = frame[[bool(started and p and p >= started) for p in posted]] if not frame.empty else frame
    by_day = []
    if not frame.empty:
        for day, part in sorted(frame.groupby("day"), key=lambda g: g[0])[-10:]:
            by_day.append(dict(day=str(day), **{k: v for k, v in ledger(part).items() if k != "roi_by_game"}))
    unread = sum(1 for rows in replies.values() for r in rows if not r.get("edited") and parse_reply(r.get("raw_text") or "") is None)
    return dict(built_at=now.isoformat(), experiment_id=NAME, registered_at=(registration or {}).get("registered_at"),
                experiment_sha256=(registration or {}).get("sha256"), research_only=True, betting_authorized=False,
                review_requirements=REVIEW,
                assumed=dict(since=str(frame.day.min()) if not frame.empty else None, total=ledger(frame),
                             by_book={b: ledger(frame[frame.book.eq(b)]) for b in BOOKS} if not frame.empty else {},
                             by_day=by_day),
                draftkings_requoted=requoted(frame[frame.book.eq("draftkings")] if not frame.empty else frame, checks),
                reply_checks={b: book_review(in_trial[in_trial.book.eq(b)] if not in_trial.empty else in_trial) for b in BOOKS},
                reported_bets=reported_bets(frame),
                replies=dict(total=sum(len(v) for v in replies.values()), not_understood=unread,
                             edited=sum(1 for rows in replies.values() for r in rows if r.get("edited"))))


def _pct(value) -> str:
    return "-" if value is None else f"{value:+.1%}"


def _record(m: dict) -> str:
    if not m["settled"]:
        return "no results yet"
    return f"{m['wins']}-{m['losses']}" + (f"-{m['pushes']}" if m["pushes"] else "")


def _money(value: float) -> str:
    return f"{'-' if value < 0 else '+'}${abs(value):,.2f}"


def markdown(doc: dict, source_status: str) -> str:
    total = doc["assumed"]["total"]
    lines = ["# NBA alerts: every-alert paper ledger and phone price checks", "",
             f"Trial `{doc['experiment_id']}` - {source_status}. Registered: {doc.get('registered_at')}.",
             "Research only. Nothing here places, requests or authorizes a wager.", "",
             "## If every alert were a $1 bet", "",
             "Assumes the alerted price was on the screen every time, which is not verified. "
             "Results are a secondary test; EV at the close is the primary one.", "",
             "| Book | Alerts | Settled | Record | Result ($1 each) | Per bet | EV when alerted | EV at close |",
             "|---|---:|---:|---|---:|---:|---:|---:|"]
    for label, m in [(BOOK_LABEL[b], doc["assumed"]["by_book"].get(b)) for b in BOOKS] + [("All books", total)]:
        if m and m["alerts"]:
            lines.append(f"| {label} | {m['alerts']} | {m['settled']} | {_record(m)} | {_money(m['units'])} | {_pct(m['roi'])} | "
                         f"{_pct(m['ev_when_alerted'])} | {_pct(m['ev_at_close'])} |")
    ci = total.get("roi_by_game") or {}
    if ci.get("lower_95") is not None:
        lines += ["", f"All books, per bet: 95% range {ci['lower_95']:+.1%} to {ci['upper_95']:+.1%} "
                      f"across {ci['clusters']} games. {total['pending']} alerts have no result yet."]
    if total["settled"]:
        lines += ["", "At other flat stakes: " + "; ".join(f"${s} a bet {_money(total['units'] * s)}" for s in (5, 10, 25)) + "."]
    lines += ["", "| Game date | Alerts | Settled | Record | Result ($1 each) |", "|---|---:|---:|---|---:|"]
    lines += [f"| {d['day']} | {d['alerts']} | {d['settled']} | {_record(d)} | {_money(d['units'])} |" for d in doc["assumed"]["by_day"]]
    dk = doc["draftkings_requoted"]
    lines += ["", "## DraftKings at the price found a minute later", "",
              f"{dk['answered']} settled alerts had an automatic re-quote; {dk['bets']} still had the line."
              + (f" At the re-quoted price: {_money(dk['units'])} ({_pct(dk['roi'])} per bet)." if dk["bets"] else ""),
              "", "## Price checks by Discord reply", "",
              "Each book needs 50 exact closes, 21 days spanning 3 weeks, 7 checking days, 20 games, 90% exact-close "
              "coverage, at least +1.5% mean closing EV, timely replies on 90% of alerts on checking days, 80% of checked "
              "offers available, 50 checked prices with exact closes, and positive 95% lower bounds by game and by day.", "",
              "| Book | Alerts | Checking days | Timely checks | Coverage | There | Gone | Moved | Checked-price EV | Status |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for book in BOOKS:
        m = doc["reply_checks"][book]
        if not m["alerts"]:
            lines.append(f"| {BOOK_LABEL[book]} | 0 | 0 | 0 | - | 0 | 0 | 0 | - | {m['status']} |")
            continue
        lines.append(f"| {BOOK_LABEL[book]} | {m['alerts']} | {m['checking_days']} | {m['timely_checks']} | "
                     f"{m['check_coverage']:.0%} | {m['available']} | {m['disappeared'] + m['suspended']} | "
                     f"{m['changed_line']} | {_pct(m['mean_checked_price_ev'])} | {m['status']} |")
    for book in BOOKS:
        m = doc["reply_checks"][book]
        if m["alerts"]:
            limit = f" Median stake limit recorded: ${m['median_stake_limit']:,.0f}." if m["stake_limits_recorded"] else ""
            lines.append(f"\n{BOOK_LABEL[book]} outstanding: {', '.join(m['reasons']) or 'manual review; still no wagering authorization'}.{limit}")
    bets, replies = doc["reported_bets"], doc["replies"]
    lines += ["", "## Bets you reported placing", "",
              (f"{bets['bets']} reported, {bets['settled']} settled: {bets['wins']}-{bets['losses']}, "
               f"${bets['staked']:,.2f} staked, {_money(bets['profit'])} ({_pct(bets['roi'])}). "
               f"${bets['open_stake']:,.2f} still open." if bets["bets"] else "None reported."),
              "", f"Replies read: {replies['total']}; not understood: {replies['not_understood']}; edited (not evidence): {replies['edited']}.",
              "", "## How to reply", "",
              "Reply to the alert in Discord within 5 minutes: `y` (same line, alerted price or better), "
              "`y -125` (same line, now -125), `b 5` (placed $5), "
              "`m 8.5 -120` (line moved), `n` (gone), `s` (suspended). Add `max 20` for the slip's stake limit. "
              "Send a new reply to correct one; an edited reply does not count."]
    return "\n".join(lines) + "\n"


def digest(doc: dict, day) -> str | None:
    """One short daily message: the assumed ledger first, then what the replies have shown."""
    total = doc["assumed"]["total"]
    if not total["settled"]:
        return None
    yesterday = next((d for d in doc["assumed"]["by_day"] if d["day"] == str(day)), None)
    lines = ["**NBA alerts: if every one were a $1 bet** (assumed; prices not verified on your screen)",
             f"Since {doc['assumed']['since']}: {total['settled']} settled, {_record(total)}, "
             f"**{_money(total['units'])}** ({_pct(total['roi'])} per bet). EV at the close {_pct(total['ev_at_close'])}."]
    if yesterday and yesterday["settled"]:
        lines.append(f"{yesterday['day']}: {yesterday['settled']} bets, {_record(yesterday)}, {_money(yesterday['units'])}.")
    books = [f"{BOOK_LABEL[b]} {_money(m['units'])} on {m['settled']}" for b, m in doc["assumed"]["by_book"].items() if m["settled"]]
    lines.append(" | ".join(books))
    ci = total.get("roi_by_game") or {}
    lines.append(f"At $10 a bet: {_money(total['units'] * 10)}."
                 + (f" Per-bet 95% range {ci['lower_95']:+.0%} to {ci['upper_95']:+.0%}." if ci.get("lower_95") is not None else ""))
    checked = [(BOOK_LABEL[b], m) for b, m in doc["reply_checks"].items() if m.get("timely_checks")]
    if checked:
        lines.append("Your checks: " + "; ".join(
            f"{label} {m['timely_checks']} of {m['alerts_on_checking_days']} answered, {m['available']} there"
            for label, m in checked) + ".")
    else:
        lines.append("No price checks yet. Reply to an alert within 5 minutes: `y` there, `n` gone, "
                     "`m 8.5 -120` moved, `b 5` placed $5.")
    bets = doc["reported_bets"]
    if bets["settled"]:
        lines.append(f"Bets you reported: {bets['settled']} settled, ${bets['staked']:,.2f} staked, {_money(bets['profit'])}.")
    return "\n".join(lines)


def write_report(conn, now: datetime | None = None) -> dict:
    from nfl_pipeline.integrity import atomic_json
    now = now or datetime.now(timezone.utc)
    try:
        registration, source_status = active(), "frozen"
    except (OSError, ValueError):
        try:
            registration = load_registration()
        except (OSError, ValueError):
            registration = None
        source_status = "unregistered_or_drifted"
    frame, replies, checks = load(conn)
    doc = summarize(frame, replies, checks, registration, now)
    doc["source_status"] = source_status
    if source_status != "frozen":
        for metrics in doc["reply_checks"].values():
            metrics["status"] = "blocked_source_drift"
            metrics["reasons"].append("registration must match current source")
    atomic_json(ROOT / "reports/nba_reply_checks_latest.json", doc)
    (ROOT / "reports/nba_reply_checks_latest.md").write_text(markdown(doc, source_status), encoding="utf-8")
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--register", action="store_true")
    args = ap.parse_args()
    doc = register() if args.register else active()
    print(json.dumps({k: doc[k] for k in ("experiment_id", "sha256", "registered_at")}, indent=2))


if __name__ == "__main__":
    main()
