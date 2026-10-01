from urllib.parse import parse_qs, urlsplit

import pytest

from nfl_pipeline import fanduel_links, fanduel_state as fs

EVENTS = {"attachments": {"events": {"36104915": {"eventId": 36104915, "name": "Pittsburgh Steelers @ Cleveland Browns"}}}}


def runner(sid, name, line, status="ACTIVE"):
    return dict(selectionId=sid, runnerName=name, handicap=line, runnerStatus=status)


GAME = {"708.1": dict(marketName="Spread", runners=[runner(50208, "Pittsburgh Steelers", -2.5), runner(50209, "Cleveland Browns", 2.5)]),
        "708.2": dict(marketName="Total Points", runners=[runner(7017916, "Over", 37.5), runner(7017917, "Under", 37.5)])}
RECEIVING = dict(GAME, **{
    "708.10": dict(marketName="DK Metcalf - Receiving Yds", runners=[runner(26036402, "DK Metcalf Over", 45.5), runner(26036403, "DK Metcalf Under", 45.5)]),
    "708.11": dict(marketName="DK Metcalf - Alt Receiving Yds", runners=[runner(87740434, "DK Metcalf 15+ Yards", 0)]),
    "708.12": dict(marketName="DK Metcalf - Total Receptions", runners=[runner(26036402, "DK Metcalf Over", 3.5), runner(26036403, "DK Metcalf Under", 3.5)]),
    "708.13": dict(marketName="Harold Fannin Jr. - Receiving Yds", runners=[runner(60595587, "Harold Fannin Jr. Over", 40.5, "SUSPENDED"), runner(60595588, "Harold Fannin Jr. Under", 40.5)]),
})


@pytest.fixture
def resolver(monkeypatch):
    calls = []

    def fetch(path, params):
        calls.append((path, params.get("tab")))
        if path == "content-managed-page":
            return EVENTS
        return {"attachments": {"markets": RECEIVING if params.get("tab") == "receiving-props" else GAME}}

    monkeypatch.setenv("NFL_FANDUEL_STATE", "co")
    r = fs.Resolver("co", fetch)
    monkeypatch.setattr(fs, "_resolver", r)
    r.calls = calls
    return r


def row(**kw):
    return dict(dict(book="fanduel", game_id="2026_04_PIT_CLE", link="https://sportsbook.fanduel.com/addToBetslip?marketId=717.9&selectionId=1"), **kw)


def test_state_ids_replace_provider_ids_by_player_stat_side_and_exact_line(resolver):
    assert fs.resolve_row(row(player_name="D.K. Metcalf", stat="receiving_yards", side="over", line=45.5)) == ("708.10", "26036402")
    assert fs.resolve_row(row(player_name="DK Metcalf", stat="receptions", side="under", line=3.5)) == ("708.12", "26036403")
    assert fs.resolve_row(row(player_name="Harold Fannin", stat="receiving_yards", side="under", line=40.5)) == ("708.13", "60595588")
    assert fs.resolve_row(row(market="total", side="over", line=37.5)) == ("708.2", "7017916")
    assert fs.resolve_row(row(market="spread", side="away", line=-2.5, home_team_abbr="CLE", away_team_abbr="PIT")) == ("708.1", "50208")
    assert resolver.calls.count(("content-managed-page", None)) == 1  # event list and tabs are cached
    assert resolver.calls.count(("event-page", "receiving-props")) == 1


@pytest.mark.parametrize("kw", [
    dict(player_name="DK Metcalf", stat="receiving_yards", side="over", line=44.5),     # line moved
    dict(player_name="Harold Fannin", stat="receiving_yards", side="over", line=40.5),  # suspended
    dict(player_name="DK Metcalf", stat="rushing_tds", side="over", line=0.5),          # unsupported market
    dict(player_name="DK Metcalf", stat="receiving_yards", side="over", line=45.5, game_id="2026_04_BAL_CIN"),  # no event
])
def test_anything_not_exact_is_manual_selection(resolver, kw):
    r = row(**kw)
    assert fs.resolve_row(r) is None
    link = fanduel_links.row_link(dict(r, line=kw["line"]))
    assert "Add to slip" not in link and "manual selection" in link  # never the provider's other-state IDs


def test_rendered_slip_uses_state_ids(resolver):
    r = row(player_name="DK Metcalf", stat="receiving_yards", side="over", line=45.5)
    url = fanduel_links.betslip_for_row(r)
    q = parse_qs(urlsplit(url).query)
    assert q == {"marketId[0]": ["708.10"], "selectionId[0]": ["26036402"]}
    assert "Add to slip" in fanduel_links.row_link(r)
    parlay = fanduel_links.parlay_betslip_url([r["link"]] * 2, [r, row(market="total", side="over", line=37.5)])
    assert parse_qs(urlsplit(parlay).query)["marketId[1]"] == ["708.2"]


def test_lookup_failure_falls_back(monkeypatch):
    def broken(path, params):
        raise TimeoutError
    monkeypatch.setenv("NFL_FANDUEL_STATE", "co")
    monkeypatch.setattr(fs, "_resolver", fs.Resolver("co", broken))
    assert fs.resolve_row(row(player_name="DK Metcalf", stat="receiving_yards", side="over", line=45.5)) is None


def test_unset_state_keeps_provider_links():
    r = row(player_name="DK Metcalf", stat="receiving_yards", side="over", line=45.5)
    assert fs.state() is None
    assert parse_qs(urlsplit(fanduel_links.betslip_for_row(r)).query)["marketId[0]"] == ["717.9"]
