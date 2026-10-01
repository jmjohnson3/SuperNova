from contextlib import nullcontext
from datetime import date, datetime, timezone

import pytest

from nfl_pipeline import sharp_lines

NOW = datetime(2026, 10, 1, 22, 45, tzinfo=timezone.utc)


class Response:
    def __init__(self, payload, remaining):
        self.payload, self.headers, self.ok, self.status_code = payload, {"x-requests-remaining": str(remaining)}, True, 200

    def json(self):
        return self.payload

    def raise_for_status(self):
        pass


class Session:
    def __init__(self, remaining):
        self.remaining, self.odds_calls = remaining, 0

    def get(self, url, params=None, timeout=None):
        if url.endswith("/events"):
            return Response([{"id": "e1", "home_team": "Cleveland Browns", "away_team": "Pittsburgh Steelers"}], self.remaining)
        self.odds_calls += 1
        self.remaining -= len(sharp_lines.MARKETS)
        return Response({"id": "e1", "bookmakers": [{"key": "pinnacle", "markets": []}]}, self.remaining)


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    games = [dict(game_id="2026_04_PIT_CLE", home="CLE", away="PIT", start=NOW)]
    monkeypatch.setattr(sharp_lines, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(sharp_lines, "due_games", lambda day, role, now, captured: [
        g for g in games if f"{g['game_id']}|{role}" not in captured])
    monkeypatch.setattr(sharp_lines.psycopg2, "connect", lambda *a: nullcontext(object()))
    saved = []
    monkeypatch.setattr(sharp_lines, "_save_payload", lambda conn, **kw: saved.append(kw))
    import nfl_pipeline.parse_oddsapi as parser
    monkeypatch.setattr(parser, "parse_props", lambda cfg: {})
    monkeypatch.setattr(sharp_lines.OddsCrawlerConfig, "__init__", lambda self, **kw: object.__setattr__(self, "oddsapi_key", "k") or None)
    return saved


def test_captures_each_game_once_per_role_and_stores_eu_books(isolated):
    session = Session(remaining=400)
    first = sharp_lines.capture(date(2026, 10, 1), "lock", now=NOW, session=session)
    assert first["status"] == "ok" and first["captured"][0]["books"] == ["pinnacle"]
    assert isolated[0]["provider"] == "oddsapi_eu" and isolated[0]["snapshot_role"] == "lock"
    again = sharp_lines.capture(date(2026, 10, 1), "lock", now=NOW, session=session)
    assert again["status"] == "nothing_due" and session.odds_calls == 1
    close = sharp_lines.capture(date(2026, 10, 1), "close", now=NOW, session=session)
    assert close["status"] == "ok" and session.odds_calls == 2  # close is a separate, single capture


def test_credit_floor_stops_spending(isolated):
    session = Session(remaining=sharp_lines.MIN_CREDITS_REMAINING)
    result = sharp_lines.capture(date(2026, 10, 1), "lock", now=NOW, session=session)
    assert session.odds_calls == 0 and result["skipped"] == {"2026_04_PIT_CLE": "credit_floor"}
