"""The Odds API client against a fake transport: no network, no credits."""
import asyncio
import logging

import httpx
import pytest

from sports_sniper.providers.odds_api import AuthError, OddsApiClient, OddsApiError, QuotaGuard

KEY = "SECRET-KEY-123"
QUOTA = {"x-requests-remaining": "9990", "x-requests-used": "10", "x-requests-last": "0"}


def make(handler, **kw):
    calls, sleeps = [], []

    def record(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request, len(calls))

    async def sleep(seconds):
        sleeps.append(seconds)

    return OddsApiClient(KEY, transport=httpx.MockTransport(record), sleep=sleep, **kw), calls, sleeps


def event(markets, book="fanduel"):
    return dict(id="e1", sport_key="basketball_nba", bookmakers=[dict(key=book, markets=[dict(key=m, outcomes=[]) for m in markets])])


def test_events_are_free_and_refresh_the_quota():
    client, calls, _ = make(lambda r, n: httpx.Response(200, json=[dict(id="e1"), "junk", dict(name="no id")], headers=QUOTA))
    assert asyncio.run(client.events("basketball_nba")) == [dict(id="e1")]
    assert calls[0].url.path == "/v4/sports/basketball_nba/events" and calls[0].url.params["apiKey"] == KEY
    assert (client.quota.remaining, client.quota.used, client.quota.spent) == (9990, 10, 0)


def test_event_odds_asks_for_named_markets_and_books_and_counts_the_cost():
    headers = {"x-requests-remaining": "9986", "x-requests-used": "14", "x-requests-last": "4"}
    client, calls, _ = make(lambda r, n: httpx.Response(200, json=event(["player_points"]), headers=headers))
    markets = ["player_points", "player_rebounds", "player_assists", "player_threes"]
    data = asyncio.run(client.event_odds("basketball_nba", "e1", markets, ["fanduel", "draftkings"]))
    params = calls[0].url.params
    assert data["id"] == "e1" and calls[0].url.path == "/v4/sports/basketball_nba/events/e1/odds"
    assert params["markets"] == ",".join(markets) and params["bookmakers"] == "fanduel,draftkings"
    assert params["oddsFormat"] == "american" and "regions" not in params
    assert client.quota.spent == 4 and client.quota.remaining == 9986
    assert asyncio.run(client.event_odds("basketball_nba", "e1", [], ["fanduel"])) is None and len(calls) == 1


def test_timeouts_retry_with_exponential_backoff():
    def handler(request, n):
        if n <= 2:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json=[], headers=QUOTA)
    client, calls, sleeps = make(handler)
    assert asyncio.run(client.events("basketball_nba")) == []
    assert len(calls) == 3 and sleeps == [0.5, 1.0]


def test_server_errors_give_up_after_the_retries_without_leaking_the_key():
    client, calls, sleeps = make(lambda r, n: httpx.Response(503), max_retries=2)
    with pytest.raises(OddsApiError) as error:
        asyncio.run(client.events("basketball_nba"))
    assert len(calls) == 3 and sleeps == [0.5, 1.0]
    assert "HTTP 503" in str(error.value) and KEY not in str(error.value) and "apiKey" not in str(error.value)


def test_rate_limit_waits_as_told_then_retries():
    client, calls, sleeps = make(lambda r, n: httpx.Response(429, headers={"retry-after": "2"}) if n == 1
                                 else httpx.Response(200, json=[], headers=QUOTA))
    assert asyncio.run(client.events("basketball_nba")) == []
    assert sleeps == [2, 0.5]


def test_bad_key_missing_event_and_bad_json():
    client, _, _ = make(lambda r, n: httpx.Response(401, json=dict(message=f"bad key {KEY}")))
    with pytest.raises(AuthError) as error:
        asyncio.run(client.events("basketball_nba"))
    assert KEY not in str(error.value)
    client, _, _ = make(lambda r, n: httpx.Response(404))
    assert asyncio.run(client.event_odds("basketball_nba", "gone", ["player_points"], ["fanduel"])) is None
    client, _, _ = make(lambda r, n: httpx.Response(200, text="<html>"))
    with pytest.raises(OddsApiError):
        asyncio.run(client.events("basketball_nba"))
    with pytest.raises(AuthError):
        OddsApiClient("")


def test_a_rejected_market_is_found_skipped_and_remembered():
    def handler(request, n):
        wanted = request.url.params["markets"].split(",")
        if "player_made_up" in wanted:
            return httpx.Response(422, json=dict(message="Invalid market"))
        return httpx.Response(200, json=event(wanted), headers={"x-requests-remaining": "9000", "x-requests-last": "1"})
    client, calls, _ = make(handler)
    data = asyncio.run(client.event_odds("basketball_nba", "e1", ["player_points", "player_made_up", "player_threes"], ["fanduel"]))
    assert [m["key"] for m in data["bookmakers"][0]["markets"]] == ["player_points", "player_threes"]
    assert client.unsupported == {"player_made_up"} and client.quota.spent == 2      # the rejected requests cost nothing
    asyncio.run(client.event_odds("basketball_nba", "e2", ["player_points", "player_made_up"], ["fanduel"]))
    assert calls[-1].url.params["markets"] == "player_points"


def test_never_spends_below_the_credit_floor_or_past_the_scan_budget():
    low = {"x-requests-remaining": "2001", "x-requests-used": "17999", "x-requests-last": "0"}
    client, calls, _ = make(lambda r, n: httpx.Response(200, json=[], headers=low), credit_floor=2000)
    asyncio.run(client.sports())
    with pytest.raises(QuotaGuard):
        asyncio.run(client.event_odds("basketball_nba", "e1", ["player_points", "player_threes"], ["fanduel"]))
    assert len(calls) == 1                                                            # the paid request was never sent
    spend = {"x-requests-remaining": "9000", "x-requests-last": "2"}
    client, calls, _ = make(lambda r, n: httpx.Response(200, json=event(["a", "b"]), headers=spend), max_credits=3)
    asyncio.run(client.event_odds("basketball_nba", "e1", ["a", "b"], ["fanduel"]))
    with pytest.raises(QuotaGuard):
        asyncio.run(client.event_odds("basketball_nba", "e2", ["a", "b"], ["fanduel"]))
    assert len(calls) == 1 and client.quota.spent == 2


def test_identical_requests_are_served_from_the_cache():
    client, calls, _ = make(lambda r, n: httpx.Response(200, json=event(["player_points"]), headers=QUOTA))

    async def twice():
        await client.event_odds("basketball_nba", "e1", ["player_points"], ["fanduel"])
        await client.event_odds("basketball_nba", "e1", ["player_points"], ["fanduel"])
        await client.event_odds("basketball_nba", "e2", ["player_points"], ["fanduel"])
    asyncio.run(twice())
    assert len(calls) == 2


def test_the_key_never_reaches_the_logs(caplog):
    client, _, _ = make(lambda r, n: httpx.Response(422))
    with caplog.at_level(logging.DEBUG):
        asyncio.run(client.event_odds("basketball_nba", "e1", ["player_made_up"], ["fanduel"]))
    assert "player_made_up" in caplog.text and KEY not in caplog.text
    assert logging.getLogger("httpx").level >= logging.WARNING
