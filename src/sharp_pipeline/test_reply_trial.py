"""Reply grammar, check evidence, ledgers and the Discord import; no network, database or Discord writes."""
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

import pandas as pd
import pytest

from sharp_pipeline import discord_replies as replies, reply_trial as trial

POSTED = datetime(2026, 11, 10, 23, 0, tzinfo=timezone.utc)
TIP = POSTED + timedelta(minutes=30)
REGISTERED = dict(registered_at=(POSTED - timedelta(hours=1)).isoformat(), sha256="abc")


def alert(**changes):
    return dict(dict(alert_id=1, discord_message_id=replies.snowflake(POSTED), sent_at=POSTED, commence_time_utc=TIP,
                     line=22.5, price=105, book="draftkings", event_id="g1", day=POSTED.date(), ev=0.05,
                     ev_at_close=0.04, close_method="exact", close_probability=0.52, result="win", profit=1.05), **changes)


def reply(text, seconds=60, edited=False, n=1):
    return dict(reply_message_id=str(n), replied_at=POSTED + timedelta(seconds=seconds), edited=edited, raw_text=text)


@pytest.mark.parametrize("text,expected", [
    ("y", dict(status="available", placed=False, price=None)),
    ("Yes", dict(status="available")),
    ("y -125", dict(status="available", price=-125)),
    ("+110", dict(status="available", price=110)),
    ("y max 20", dict(status="available", max_stake=20.0)),
    ("b", dict(status="available", placed=True, stake=1.0, price=None)),
    ("bet 5", dict(placed=True, stake=5.0)),
    ("b $5 -125 max 50", dict(placed=True, stake=5.0, price=-125, max_stake=50.0)),
    ("m 8.5 -120", dict(status="changed_line", line=8.5, price=-120)),
    ("moved", dict(status="changed_line", line=None, price=None)),
    ("n", dict(status="disappeared")),
    ("locked", dict(status="suspended")),
])
def test_reply_grammar(text, expected):
    parsed = trial.parse_reply(text)
    assert parsed is not None and {k: parsed[k] for k in expected} == expected


@pytest.mark.parametrize("text", ["", "nice one", "y thanks", "y 5", "n -110", "m 8.5", "b 5 6", "b -125 -130",
                                  "max 20", "y max", "-50", "y -125 -130"])
def test_anything_unclear_is_not_a_check(text):
    assert trial.parse_reply(text) is None


def test_discord_ids_carry_their_own_time():
    assert trial.snowflake_time(replies.snowflake(POSTED)) == POSTED
    assert trial.snowflake_time("not-an-id") is None


def test_check_is_the_last_timely_reply_and_late_or_edited_ones_are_not():
    assert trial.check(alert(), [reply("y")]) | dict(replies=0, readable_replies=0) == dict(
        replies=0, readable_replies=0, timely_check=True, check_status="available", displayed_line=22.5,
        displayed_price=105, check_delay_seconds=60.0, max_stake=None, bet_stake=None, bet_price=None)
    corrected = trial.check(alert(), [reply("y", 30, n=1), reply("n", 90, n=2)])
    assert (corrected["timely_check"], corrected["check_status"], corrected["displayed_price"]) == (True, "disappeared", None)
    assert trial.check(alert(), [reply("y -140")])["displayed_price"] == -140
    moved = trial.check(alert(), [reply("m 21.5 -110")])
    assert (moved["check_status"], moved["displayed_line"], moved["displayed_price"]) == ("changed_line", 21.5, -110)
    for bad in (reply("y", 301), reply("y", edited=True), reply("y", 31 * 60), reply("y", -5), reply("looks fine")):
        assert trial.check(alert(), [bad])["timely_check"] is False


def test_a_late_bet_is_still_a_bet_but_not_a_check():
    found = trial.check(alert(), [reply("b 5 -110", 600)])
    assert (found["timely_check"], found["bet_stake"], found["bet_price"]) == (False, 5.0, -110)
    assert trial.check(alert(), [reply("b 5", 31 * 60)])["bet_stake"] is None  # after the tip proves nothing
    assert trial.check(alert(), [reply("b 5", 30, n=1), reply("n", 90, n=2)])["bet_stake"] is None  # it did not go through
    assert trial.check(alert(), [reply("b 5", 30, n=1), reply("y max 50", 90, n=2)])["bet_stake"] == 5.0


def sample(days=21, per_day=3, checked=lambda day, i: True):
    rows, answers = [], {}
    for day in range(days):
        for i in range(per_day):
            posted = POSTED + timedelta(days=day, minutes=i)
            n = day * per_day + i + 1
            rows.append(alert(alert_id=n, discord_message_id=replies.snowflake(posted), sent_at=posted,
                              commence_time_utc=posted + timedelta(minutes=30), event_id=f"g{day}-{i % 2}",
                              day=posted.date(), result="win" if n % 2 else "loss", profit=1.05 if n % 2 else -1.0))
            if checked(day, i):
                answers[n] = [dict(reply_message_id=str(n), replied_at=posted + timedelta(seconds=45), edited=False, raw_text="y")]
    return pd.DataFrame(rows), answers


def review(frame, answers):
    return trial.book_review(trial.evidence(frame, answers))


def test_complete_evidence_only_prompts_a_manual_review():
    result = review(*sample())
    assert result["status"] == "execution_review_only", result["reasons"]
    assert result["betting_authorized"] is False and result["priced_checks"] == 63
    assert result["mean_checked_price_ev"] == pytest.approx(0.52 * 2.05 - 1)


def test_unchecked_days_do_not_count_but_skipped_alerts_on_a_checking_day_do():
    skipped_days = review(*sample(days=28, checked=lambda day, i: day % 4 != 3))
    assert skipped_days["check_coverage"] == 1.0 and skipped_days["checking_days"] == 21
    assert skipped_days["check_coverage_all_alerts"] == pytest.approx(0.75)
    assert skipped_days["gates"]["reply_check_coverage"]
    one_week = review(*sample(checked=lambda day, i: day < 7))  # three weeks of alerts, one week of checking
    assert (one_week["span_days"], one_week["weeks"]) == (7, 1) and not one_week["gates"]["collection_span"]
    picked = review(*sample(checked=lambda day, i: i == 0))
    assert picked["check_coverage"] == pytest.approx(1 / 3) and not picked["gates"]["reply_check_coverage"]
    assert picked["status"] == "collecting"


def test_gone_offers_fail_availability_and_no_replies_is_collecting():
    frame, answers = sample()
    for rows in answers.values():
        rows[0]["raw_text"] = "n"
    gone = review(frame, answers)
    assert gone["available"] == 0 and gone["disappeared"] == 63 and not gone["gates"]["offer_availability"]
    silent = review(frame, {})
    assert silent["timely_checks"] == 0 and silent["checking_days"] == 0 and silent["status"] == "collecting"


def test_assumed_ledger_reported_bets_and_requotes():
    frame, answers = sample(days=2, per_day=2)  # results: win, loss, win, loss at +105
    answers[1][0]["raw_text"] = "b 10"
    answers[2][0]["raw_text"] = "b 4 -110"
    frame.loc[3, ["result", "profit"]] = [None, None]
    doc = trial.summarize(frame, answers, {1: dict(status="available", price=100), 2: dict(status="changed_line", price=-110),
                                           3: dict(status="unknown", price=None)}, REGISTERED, POSTED)
    total = doc["assumed"]["total"]
    assert (total["alerts"], total["settled"], total["wins"], total["losses"], total["pending"]) == (4, 3, 2, 1, 1)
    assert total["units"] == pytest.approx(1.1) and total["roi"] == pytest.approx(1.1 / 3)
    assert doc["assumed"]["by_book"]["draftkings"]["alerts"] == 4 and doc["assumed"]["by_book"]["fanduel"]["alerts"] == 0
    assert [d["alerts"] for d in doc["assumed"]["by_day"]] == [2, 2]
    bets = doc["reported_bets"]
    assert (bets["bets"], bets["settled"], bets["staked"]) == (2, 2, 14.0) and bets["profit"] == pytest.approx(10.5 - 4)
    assert doc["draftkings_requoted"] == dict(answered=2, bets=1, units=1.0, roi=1.0)
    assert doc["reply_checks"]["draftkings"]["timely_checks"] == 4 and doc["betting_authorized"] is False
    assert "every one were a $1 bet" in trial.digest(doc, POSTED.date())
    assert "| DraftKings | 4 |" in trial.markdown(doc, "frozen")


def test_alerts_before_registration_are_ledger_only():
    frame, answers = sample(days=2, per_day=2)
    late_start = dict(registered_at=(POSTED + timedelta(hours=12)).isoformat(), sha256="abc")
    doc = trial.summarize(frame, answers, {}, late_start, POSTED)
    assert doc["assumed"]["total"]["alerts"] == 4 and doc["reply_checks"]["draftkings"]["alerts"] == 2
    assert trial.summarize(frame, answers, {}, None, POSTED)["reply_checks"]["draftkings"]["alerts"] == 0
    assert trial.summarize(pd.DataFrame(), {}, {}, None, POSTED)["assumed"]["total"]["alerts"] == 0


# --- Discord import ---------------------------------------------------------------------------------
class Response:
    def __init__(self, status, body=None):
        self.status_code, self.body = status, body

    def json(self):
        return self.body


class Discord:
    """Pages forward from `after`, 100 at a time, newest first within a page, as Discord does."""
    def __init__(self, messages):
        self.messages, self.reads, self.marks = messages, 0, []

    def get(self, url, headers=None, timeout=None):
        return Response(200, dict(channel_id="55"))

    def request(self, method, url, headers=None, timeout=None, params=None):
        assert headers["Authorization"] == "Bot token"
        if method == "PUT":
            self.marks.append((url.split("/messages/")[1].split("/")[0], unquote(url.split("/reactions/")[1].split("/")[0])))
            return Response(204)
        self.reads += 1
        later = sorted((m for m in self.messages if int(m["id"]) > int(params["after"])), key=lambda m: int(m["id"]))
        return Response(200, list(reversed(later[:params["limit"]])))


class Cursor:
    def __init__(self, db):
        self.db, self.rows, self.rowcount = db, [], 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        if "INSERT INTO bets.nba_alert_replies" in sql:
            self.rowcount = 0 if params[0] in self.db.saved else 1
            self.db.saved.setdefault(params[0], params)
        elif "INSERT INTO bets.nba_alert_reply_cursor" in sql:
            self.db.position = params[1]
        elif "FROM bets.nba_alert_reply_cursor" in sql:
            self.rows = [dict(last_message_id=self.db.position)] if self.db.position else []
        else:
            self.rows = [dict(alert_id=a, discord_message_id=m) for m, a in self.db.alerts.items()]

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class Database:
    def __init__(self, alerts):
        self.alerts, self.saved, self.position = alerts, {}, None

    def cursor(self, cursor_factory=None):
        return Cursor(self)

    def commit(self):
        pass


def message(seconds, text="", *, reply_to=None, bot=False, edited=False, user="7"):
    at = POSTED + timedelta(seconds=seconds)
    return dict(id=replies.snowflake(at), content=text, timestamp=at.isoformat(), author=dict(id=user, bot=bot),
                edited_timestamp=at.isoformat() if edited else None,
                message_reference=dict(message_id=reply_to) if reply_to else None, webhook_id="9" if bot else None)


def imported(messages, **kwargs):
    first = replies.snowflake(POSTED)
    db, discord = Database({first: 283}), Discord(messages)
    out = replies.run(db, token="token", webhook_url="https://hook", session=discord, now=POSTED + timedelta(minutes=9),
                      sleep=lambda s: None, **kwargs)
    return db, discord, out


def test_only_human_replies_to_alerts_are_kept_and_marked():
    first = replies.snowflake(POSTED)
    db, discord, out = imported([
        message(0, "PAPER PRICE CHECK", bot=True), message(40, "y -125", reply_to=first), message(50, "nice one"),
        message(70, "#283 b 5"), message(80, "what", reply_to=first), message(400, "n", reply_to=first),
        message(90, "y", reply_to=first, bot=True), message(95, "y", reply_to="12345")])
    assert (out["messages"], out["replies"], out["understood"], out["not_understood"]) == (8, 4, 3, 1)
    assert sorted(row[6] for row in db.saved.values()) == ["b 5", "n", "what", "y -125"]
    assert all(row[1] == 283 and row[2] == "55" and row[3] == "7" for row in db.saved.values())
    assert [mark for _, mark in discord.marks] == [replies.ACK["understood"], replies.ACK["understood"],
                                                    replies.ACK["unknown"], replies.ACK["late"]]
    assert db.position == replies.snowflake(POSTED + timedelta(seconds=400))
    again = replies.run(db, token="token", webhook_url="https://hook", session=discord, now=POSTED, sleep=lambda s: None)
    assert (again["messages"], again["replies"]) == (0, 0) and len(discord.marks) == 4


def test_pages_forward_flags_missing_text_and_can_be_limited_to_one_person():
    first = replies.snowflake(POSTED)
    db, discord, out = imported([message(i + 1, "y", reply_to=first) for i in range(150)])
    assert discord.reads == 2 and out["replies"] == 150 and len(db.saved) == 150
    _, _, blank = imported([message(30, "", reply_to=first)])
    assert blank["without_text"] == 1 and "Message Content Intent" in blank["hint"]
    _, _, other = imported([message(30, "y", reply_to=first, user="8")], only_author="7")
    assert other["replies"] == 0


def test_discord_refusals_are_named_without_the_address():
    class Refusing(Discord):
        def request(self, method, url, headers=None, timeout=None, params=None):
            return Response(403)
    with pytest.raises(replies.DiscordError, match="bot_cannot_read_channel"):
        replies.run(Database({}), token="token", webhook_url="https://hook/secret", session=Refusing([]), now=POSTED,
                    sleep=lambda s: None)
