"""The Parlay Lab's legs: one book's prices and links, last-games hit counts, stored projections, one combined link."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sports_sniper import parlay, parlay_legs
from sports_sniper.config import Settings

NOW = datetime.now(timezone.utc).replace(microsecond=0)
TIP = NOW + timedelta(hours=1)
FD = "https://sportsbook.fanduel.com/addToBetslip?marketId=42.{}&selectionId={}"
DK = "https://sportsbook.draftkings.com/?outcomes=0QA{}%23{}_13L42648Q1Q20"
SCORE = "https://sportsbook.thescore.bet/?market_selection_id[0]={}&odds_numerator[0]=2&odds_denominator[0]=1"


def price_rows():
    rows = []

    def add(event, player, market, line, book, over, under, link=None):
        for n, (side, price) in enumerate((("over", over), ("under", under))):
            rows.append(dict(sport="nba", event_id=event, event_name=f"Away {event} @ Home {event}", commence_time=pd.Timestamp(TIP),
                             book=book, market=market, player=player, player_key=player.lower(), line=line, side=side, price=price,
                             last_update=pd.Timestamp(NOW), observed_at=pd.Timestamp(NOW),
                             link=None if link is None else link.format(len(rows), 100 + len(rows) + n)))
    for event, player in (("e1", "Al Star"), ("e1", "Bo Mate"), ("e2", "Cy Away")):
        add(event, player, "player_points", 20.5, "pinnacle", -105, -115)
        add(event, player, "player_points", 20.5, "fanduel", 110, -140, FD)
        add(event, player, "player_points", 20.5, "draftkings", -102, -122, DK)
    add("e1", "Al Star", "player_assists", 5.5, "pinnacle", -110, -110)
    add("e1", "Al Star", "player_assists", 5.5, "fanduel", -120, -106, FD)
    add("e1", "No Ref", "player_points", 9.5, "fanduel", -110, -110, FD)             # no sharp reference: not offered as a leg
    return pd.DataFrame(rows)


def test_legs_carry_one_books_price_and_link_and_the_market_chance():
    legs = parlay_legs.book_legs(price_rows(), Settings(), "fanduel")
    assert len(legs) == 8 and {leg["book"] for leg in legs} == {"fanduel"} and "No Ref" not in {leg["player"] for leg in legs}
    over, under = (next(leg for leg in legs if leg["player"] == "Al Star" and leg["market"] == "player_points" and leg["side"] == side)
                   for side in ("over", "under"))
    assert (over["price"], under["price"]) == (110, -140) and over["link"] != under["link"] and over["link"].startswith("https://sportsbook.fanduel")
    assert over["probability"] + under["probability"] == pytest.approx(1) and 0.45 < over["probability"] < 0.5    # Pinnacle's -105 / -115
    assert len(parlay_legs.book_legs(price_rows(), Settings(), "draftkings")) == 6                # it has no assists line here
    assert parlay_legs.book_legs(price_rows(), Settings(), "espnbet") == [] and parlay_legs.book_legs(price_rows().iloc[0:0], Settings(), "fanduel") == []
    assert parlay_legs.multiplied_decimal([over, under]) == pytest.approx(2.1 * (1 + 100 / 140))
    assert parlay_legs.american(3.0) == 200


def test_hit_counts_use_the_most_recent_games_with_a_value():
    games = pd.DataFrame(dict(day=pd.date_range("2026-10-20", periods=12), player_points=[30, 5, 5, 25, 25, 5, 25, 25, 25, 20.5, None, 25]))
    assert parlay_legs.hit_count(games, "player_points", 20.5, "over", 5) == (4, 5)              # the missing game is skipped, 20.5 is not over
    assert parlay_legs.hit_count(games, "player_points", 20.5, "over", 10) == (6, 10)
    assert parlay_legs.hit_count(games, "player_points", 20.5, "under", 5) == (0, 5)             # a tie wins for neither side
    assert parlay_legs.hit_count(games, "player_points_alternate", 4.5, "over", 3) == (3, 3)
    assert parlay_legs.hit_count(games.head(2), "player_points", 20.5, "over", 10) == (1, 2)     # fewer games than asked for
    assert parlay_legs.hit_count(games, "player_assists", 5.5, "over", 5) is None and parlay_legs.hit_count(None, "player_points", 1.5, "over", 5) is None
    history = {"nba": games.assign(player_key="al star")}
    assert list(parlay_legs.by_player(history)) == [("nba", "al star")]


def test_one_link_opens_every_leg_or_there_is_none():
    fanduel = parlay_legs.parlay_link("fanduel", [FD.format(1, 11), FD.format(2, 22), FD.format(3, 33)])
    assert fanduel.count("marketId") == 3 and "selectionId%5B2%5D=33" in fanduel and "marketId%5B0%5D=42.1" in fanduel
    assert parlay_legs.parlay_link("fanduel", [FD.format(1, 11), None]) is None                   # a leg with no link: no parlay link
    assert parlay_legs.parlay_link("fanduel", [FD.format(1, 11)]) is None
    assert parlay_legs.parlay_link("fanduel", [FD.format(1, 11), FD.format(1, 12)]) is None       # both sides of one market cannot combine
    draftkings = parlay_legs.parlay_link("draftkings", [DK.format(1, 2), DK.format(3, 4)])
    assert draftkings == "https://sportsbook.draftkings.com/?outcomes=0QA1%232_13L42648Q1Q20+0QA3%234_13L42648Q1Q20"
    assert parlay_legs.parlay_link("draftkings", [DK.format(1, 2), DK.format(1, 2)]) is None
    assert parlay_legs.parlay_link("draftkings", [DK.format(1, 2), "https://sportsbook.draftkings.com/event/5"]) is None
    score = parlay_legs.parlay_link("espnbet", [SCORE.format("aa"), SCORE.format("bb")])
    assert "market_selection_id[0]=aa" in score and "market_selection_id[1]=bb" in score and "odds_denominator[1]=1" in score
    assert parlay_legs.parlay_link("betmgm", ["https://sports.co.betmgm.com/x?options=1", "https://sports.co.betmgm.com/x?options=2"]) is None
    assert parlay_legs.parlay_link("fanduel", ["javascript:alert(1)", FD.format(1, 11)]) is None


def test_a_legs_projection_is_worked_out_at_its_own_line():
    pytest.importorskip("nba_props.model")
    props = pd.DataFrame([dict(sport="nba", player_key="al star", market="player_points", projection=22.0, line=21.5, model_over=0.5,
                               disp_a=1.0, disp_b=0.0),
                          dict(sport="nfl", player_key="cy away", market="player_receptions", projection=4.8, line=4.5, model_over=0.53,
                               disp_a=None, disp_b=None)])
    leg = dict(sport="nba", player_key="al star", market="player_points", line=20.5, side="over")
    projected, over = parlay_legs.projection_for(props, leg)
    low = parlay_legs.projection_for(props, dict(leg, line=30.5))[1]
    assert projected == 22.0 and 0.5 < over < 0.7 and low < 0.1                                   # the leg's line, not the stored one
    assert parlay_legs.projection_for(props, dict(leg, side="under"))[1] == pytest.approx(1 - over)
    nfl = dict(sport="nfl", player_key="cy away", market="player_receptions", line=4.5, side="under")
    assert parlay_legs.projection_for(props, nfl) == (4.8, pytest.approx(0.47))
    assert parlay_legs.projection_for(props, dict(nfl, line=5.5)) == (4.8, None)                  # no stored chance at another line
    assert parlay_legs.projection_for(props, dict(leg, player_key="nobody")) == (None, None)
    assert parlay_legs.projection_for(pd.DataFrame(), leg) == (None, None) and parlay_legs.projection_for(None, leg) == (None, None)


def test_a_parlay_can_have_many_legs_across_games_and_sports():
    table = {"nba": {"same|player_assists|player_points": dict(r=0.4, n=5000)}}
    legs = [dict(leg, team="T") for leg in parlay_legs.book_legs(price_rows(), Settings(), "fanduel") if leg["side"] == "over"]
    assert len(legs) == 4
    out = parlay.evaluate(legs, table, "nba", parlay_legs.american(parlay_legs.multiplied_decimal(legs)))
    linked = [n for n in out["links"] if n["correlation"]]
    assert len(out["links"]) == 6 and len(linked) == 1 and out["joint"] > out["independent"]     # only the same-player pair is tied
    apart = [leg for leg in legs if leg["market"] == "player_points" and leg["player"] != "Bo Mate"]
    other = parlay.evaluate([apart[0], dict(apart[1], sport="nfl")], table, "nba")
    assert other["joint"] == other["independent"] and "different games" in other["links"][0]["why"]
    many = [dict(sport="nba", event_id=f"g{n}", player_key=f"p{n}", market="player_points", side="over", probability=0.5, team="T")
            for n in range(parlay_legs.MAX_LEGS)]
    assert parlay.evaluate(many, table, "nba")["independent"] == pytest.approx(0.5 ** 12)
