from datetime import timedelta

from sharp_pipeline import draftkings_checks as checks
from sharp_pipeline.test_sharp_multi import NOW, START, bookmaker, event, market

SENT = NOW
LATER = NOW + timedelta(seconds=70)


def alert(**changes):
    return dict(dict(alert_id=1, sent_at=SENT, event_id="e1", commence_time_utc=START, market="points",
                     player="Test Player", player_norm="test player", side="under", line=22.5, price=105,
                     fair_probability=0.52, ev=0.066, decided_at=SENT), **changes)


def requote(over=-135, under=105, point=22.5, updated=None, player="Test Player", book="draftkings"):
    updated = updated or SENT + timedelta(seconds=40)
    return event([bookmaker(book, [market("player_points", [
        dict(name="Over", description=player, point=point, price=over),
        dict(name="Under", description=player, point=point, price=under)], updated)], updated)])


def test_same_line_is_available_and_a_worse_price_is_named():
    assert checks.read_quote(alert(), requote(), LATER) | dict(quote_updated_at=None) == dict(
        status="available", reason="same_or_better_price", line=22.5, price=105, quote_updated_at=None)
    assert checks.read_quote(alert(), requote(under=115), LATER)["reason"] == "same_or_better_price"
    worse = checks.read_quote(alert(), requote(under=-105), LATER)
    assert (worse["status"], worse["reason"], worse["price"]) == ("available", "worse_price", -105)


def test_moved_line_missing_player_and_missing_market():
    moved = checks.read_quote(alert(), requote(point=21.5), LATER)
    assert (moved["status"], moved["line"], moved["price"]) == ("changed_line", 21.5, 105)
    assert checks.read_quote(alert(), requote(player="Someone Else"), LATER)["reason"] == "player_absent"
    gone = checks.read_quote(alert(), requote(book="fanduel"), LATER)
    assert (gone["status"], gone["reason"]) == ("not_listed", "market_absent")


def test_a_quote_from_before_the_alert_or_after_the_start_proves_nothing():
    stale = checks.read_quote(alert(), requote(updated=SENT - timedelta(seconds=5)), LATER)
    assert (stale["status"], stale["reason"]) == ("unknown", "provider_not_refreshed")
    assert checks.read_quote(alert(), requote(), START + timedelta(seconds=1))["reason"] == "game_started"


class Response:
    def __init__(self, body, status=200):
        self.body, self.status_code, self.headers = body, status, {"x-requests-last": "1", "x-requests-remaining": "5000"}

    def json(self):
        return self.body


class Session:
    def __init__(self, body, status=200):
        self.body, self.status, self.calls = body, status, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return Response(self.body, self.status)


def harness(monkeypatch, alerts, spent=0):
    saved = []
    monkeypatch.setattr(checks, "pending", lambda conn, now: alerts)
    monkeypatch.setattr(checks, "spent_today", lambda conn, now: spent)
    monkeypatch.setattr(checks, "save", lambda conn, a, attempted, observed, result, credits: saved.append((a["alert_id"], result, credits)))
    return saved


def test_one_request_per_event_market_after_waiting_for_the_refresh(monkeypatch):
    saved = harness(monkeypatch, [alert(), alert(alert_id=2, side="over", price=-135)])
    session, slept, times = Session(requote()), [], iter([SENT + timedelta(seconds=20)] * 3 + [LATER] * 10)
    out = checks.run(None, key="k", session=session, clock=lambda: next(times), sleep=slept.append, credits_remaining=5000)
    assert len(session.calls) == 1 and session.calls[0][1]["bookmakers"] == "draftkings" and session.calls[0][1]["markets"] == "player_points"
    assert slept == [40.0] and out["credits"] == 1 and out["checked"] == 2
    assert [(i, r["status"], c) for i, r, c in saved] == [(1, "available", 1), (2, "available", 0)]


def test_budget_floor_and_http_failures_are_recorded_without_retrying(monkeypatch):
    clock = lambda: LATER
    saved = harness(monkeypatch, [alert()], spent=checks.DAILY_CREDITS)
    session = Session(requote())
    checks.run(None, key="k", session=session, clock=clock, sleep=lambda s: None, credits_remaining=5000)
    assert session.calls == [] and saved[0][1]["reason"] == "daily_credit_budget"
    saved = harness(monkeypatch, [alert()])
    checks.run(None, key="k", session=session, clock=clock, sleep=lambda s: None, credits_remaining=checks.CREDIT_FLOOR)
    assert session.calls == [] and saved[0][1]["reason"] == "credit_floor"
    saved = harness(monkeypatch, [alert()])
    failing = Session({}, status=429)
    checks.run(None, key="k", session=failing, clock=clock, sleep=lambda s: None, credits_remaining=5000)
    assert len(failing.calls) == 1 and saved[0][1]["reason"] == "http_429"
