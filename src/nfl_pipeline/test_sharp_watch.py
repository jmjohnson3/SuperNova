from datetime import datetime, timedelta, timezone

import pytest

from nfl_pipeline import sharp_watch as w

NOW = datetime(2026, 10, 4, 15, 0, tzinfo=timezone.utc)


def book(key, outcomes, market="player_reception_yds"):
    return {"key": key, "markets": [{"key": market, "outcomes": outcomes}]}


def pair(player, line, over, under, link=None):
    return [dict(name="Over", description=player, point=line, price=over, link=link),
            dict(name="Under", description=player, point=line, price=under, link=link)]


def test_alerts_when_fanduel_is_off_the_sharp_line_and_not_when_it_is_fair():
    payload = {"bookmakers": [book("fanduel", pair("DK Metcalf", 45.5, -110, -110, "https://sportsbook.fanduel.com/x")
                                              + pair("Fair Guy", 30.5, -110, -110)),
                              book("pinnacle", pair("DK Metcalf", 45.5, -140, +120) + pair("Fair Guy", 30.5, -105, -115))]}
    edges = w.find_edges(payload)
    assert [(e["player"], e["side"]) for e in edges] == [("DK Metcalf", "over")]
    e = edges[0]
    assert e["sharp_book"] == "pinnacle" and e["line_gap"] == 0 and e["ev"] > 0.05
    assert w.sharp_math.ev(e["fair_probability"], e["minimum_price"]) >= 0.01


def test_nearby_sharp_line_is_converted_and_large_gaps_are_ignored():
    payload = {"bookmakers": [book("fanduel", pair("Near", 44.5, -110, -110) + pair("Far", 40.5, -110, -110)),
                              book("pinnacle", pair("Near", 46.5, -125, +105) + pair("Far", 46.5, -150, +130))]}
    edges = w.find_edges(payload)
    near = [e for e in edges if e["player"] == "Near"]
    assert near and near[0]["side"] == "over" and near[0]["line_gap"] == -2.0  # FD is 2 yards lower: over is cheap
    assert not [e for e in edges if e["player"] == "Far"]  # 6-yard gap is too far to trust


def test_totals_use_the_same_logic():
    totals = lambda line, o, u: [dict(name="Over", point=line, price=o), dict(name="Under", point=line, price=u)]
    payload = {"bookmakers": [book("fanduel", totals(44.5, -110, -110), "totals"),
                              book("pinnacle", totals(44.5, +115, -135), "totals")]}
    edges = w.find_edges(payload)
    assert [(e["stat"], e["side"]) for e in edges] == [("total", "under")]


def game(gid, minutes):
    return dict(game_id=gid, start=NOW + timedelta(minutes=minutes), home="CLE", away="PIT", day=NOW.date())


def test_poll_schedule_budget_and_floor():
    # Only the last 4 hours are polled at all: a gap seen earlier decays to nothing by the close.
    games = [game("last_hour", 30), game("window", 200), game("early", 600), game("far", 3000)]
    due, skipped = w.plan_polls(games, {}, NOW, remaining=5000, cost=1, floor=150)
    assert [g["game_id"] for g in due] == ["last_hour", "window"]
    recent = {"last_poll": {"last_hour": (NOW - timedelta(minutes=3)).isoformat(),
                            "window": (NOW - timedelta(minutes=5)).isoformat()}}
    due, _ = w.plan_polls(games, recent, NOW, remaining=5000, cost=1, floor=150)
    assert due == []  # 5-minute cadence inside the last hour, 10-minute from T-4h
    due, _ = w.plan_polls(games, {"last_poll": {"last_hour": (NOW - timedelta(minutes=6)).isoformat()}},
                          NOW, remaining=5000, cost=1, floor=150)
    assert [g["game_id"] for g in due] == ["last_hour", "window"]  # 6 min > the 5-minute cadence
    due, skipped = w.plan_polls(games, {}, NOW, remaining=150, cost=1, floor=150)
    assert due == [] and skipped == {"last_hour": "credit_floor", "window": "credit_floor"}
    tight = w.daily_allowance(200, NOW.date(), 150)  # 50 spare credits over the rest of October
    assert tight == pytest.approx(50 / 28)
    due, skipped = w.plan_polls(games, {}, NOW, remaining=200, cost=1, floor=150)
    assert len(due) == 1 and skipped == {"window": "daily_allowance"}  # soonest game gets the budget


def test_tight_budget_is_saved_for_the_final_window():
    # Tonight's case: ~11 checks/day, one game. Early checks may only spend what the last 90 min won't need.
    start = NOW + timedelta(minutes=180)
    g = [dict(game_id="tnf", start=start, home="CLE", away="PIT", day=NOW.date())]
    remaining, spent, polled = 496, 3, []  # 3 early polls already spent today (allowance ~11.3)
    t = NOW
    while t < start:
        state = {"spent": {str(t.astimezone(w._ET).date()): spent},
                 "last_poll": {"tnf": polled[-1].isoformat()} if polled else {}}
        due, _ = w.plan_polls(g, state, t, remaining, cost=1, floor=150)
        if due:
            polled.append(t); spent += 1; remaining -= 1
        t += timedelta(minutes=10)
    in_window = [p for p in polled if (start - p).total_seconds() <= 90 * 60]
    assert len(in_window) == 9 and len(polled) == 9  # no early polls; every slot of the final 90 minutes
    # The floor still binds inside the window.
    due, skipped = w.plan_polls(g, {}, start - timedelta(minutes=30), remaining=150, cost=1, floor=150)
    assert due == [] and skipped == {"tnf": "credit_floor"}
    # With plenty of credits nothing is held back.
    due, skipped = w.plan_polls(g, {}, NOW, remaining=20000, cost=1, floor=150)
    assert [x["game_id"] for x in due] == ["tnf"] and skipped == {}


def test_small_gaps_are_logged_but_not_alerted():
    # Pinnacle -122/+102 -> fair over ~52.6%; FanDuel over -108 -> EV ~ +1.3%: logged, not pinged
    payload = {"bookmakers": [book("fanduel", pair("Small", 45.5, -108, -112)),
                              book("pinnacle", pair("Small", 45.5, -122, +102))]}
    logged = w.find_edges(payload, min_ev=w.LOG_MIN_EV)
    assert [(e["player"], e["side"]) for e in logged] == [("Small", "over")]
    assert w.LOG_MIN_EV <= logged[0]["ev"] < w.MIN_EV
    assert w.find_edges(payload) == []  # below the Discord threshold


def test_budget_follows_games_until_the_plan_resets():
    from datetime import date
    assert w.cycle_end(date(2026, 10, 3), 3) == date(2026, 11, 3)
    assert w.cycle_end(date(2026, 10, 2), 3) == date(2026, 10, 3)
    assert w.cycle_end(date(2026, 1, 31), 31) == date(2026, 2, 28)  # clamped to the month's length
    by_day = {date(2026, 10, 4): 14, date(2026, 10, 5): 1, date(2026, 10, 8): 1, date(2026, 11, 8): 14}
    assert w.game_share(date(2026, 10, 4), by_day, 3) == pytest.approx(14 / 16)  # Nov 8 is next cycle
    assert w.game_share(date(2026, 10, 6), by_day, 3) == 0.0  # no games today: nothing early is spent
    assert w.game_share(date(2026, 10, 4), {}, 3) is None
    # Sunday with 14 games gets the bulk of the spare credits, so early checks are not starved.
    games = [dict(game_id=f"g{i}", start=NOW + timedelta(minutes=200), home="CLE", away="PIT", day=NOW.date()) for i in range(14)]
    due, skipped = w.plan_polls(games, {}, NOW, remaining=20000, cost=4, floor=150, share=14 / 62)
    assert len(due) == 14 and skipped == {}
    sunday_need = 14 * 44 * 4  # 14 games x 44 checks x 4 markets
    assert w.daily_allowance(20000, NOW.date(), 150) < sunday_need <= (20000 - 150) * 14 / 62  # calendar split starves Sunday


def test_pings_only_inside_the_backtested_window():
    assert w.alert_tier(0.05, 60) == "alert" and w.alert_tier(0.05, 180) == "alert"
    assert w.alert_tier(0.05, 181) == "early"  # still logged, upgraded and pinged if it holds into the window
    assert w.alert_tier(0.02, 30) == "logged"


def test_every_bettable_book_is_priced_against_the_sharp_line():
    # Pinnacle fair over ~58%. DraftKings is generous on the over, FanDuel on the under.
    payload = {"bookmakers": [book("fanduel", pair("DK Metcalf", 45.5, -200, +170)),
                              book("draftkings", pair("DK Metcalf", 45.5, +120, -140)),
                              book("pinnacle", pair("DK Metcalf", 45.5, -140, +120))]}
    edges = w.find_edges(payload)
    assert {(e["book"], e["side"]) for e in edges} == {("draftkings", "over"), ("fanduel", "under")}
    assert edges[0]["ev"] >= edges[-1]["ev"]  # still ranked by EV across books
    # A book we cannot bet at is never alerted on, even when it is the most generous price.
    assert w.find_edges(payload, bet_books=("fanduel",)) == [e for e in edges if e["book"] == "fanduel"]


def test_other_books_keep_only_their_own_verified_links():
    game = dict(start=NOW + timedelta(minutes=30), home="CLE", away="PIT")
    edge = dict(book="draftkings", stat="receiving_yards", player="X", player_norm="x", side="over",
                line=45.5, price=120, sharp_book="pinnacle", sharp_line=45.5, line_gap=0.0,
                fair_probability=0.58, ev=0.05, minimum_price=110,
                link="https://sportsbook.draftkings.com/event/123")
    assert w.book_link(edge, game) == ("https://sportsbook.draftkings.com/event/123", False)
    assert w.book_link(dict(edge, link="https://evil.example.com/x"), game) == (None, False)
    assert w.book_link(dict(edge, link="http://sportsbook.draftkings.com/x"), game) == (None, False)
    text = w.format_alert(edge, game, stake=5.0)
    assert "bet $5 at DraftKings" in text and "DraftKings +120" in text and "manual selection" in text


def test_stake_respects_caps_and_the_loss_pause(monkeypatch):
    monkeypatch.setattr(w.prefs, "SHARP_WATCH_STAKE", 5.0)
    monkeypatch.setattr(w.prefs, "SHARP_WATCH_MAX_STAKE_PER_DAY", 10.0)
    monkeypatch.setattr(w.prefs, "SHARP_WATCH_MAX_STAKE_PER_WEEK", 100.0)
    budget = dict(today=0.0, week=0.0, staked=0.0, paused=False)
    assert w.stake_for(None, "alert", NOW, budget) == 5.0
    assert w.stake_for(None, "logged", NOW, budget) == 0.0  # only pinged alerts carry money
    assert w.stake_for(None, "early", NOW, budget) == 0.0
    budget["staked"] = 10.0  # daily cap reached within this run
    assert w.stake_for(None, "alert", NOW, budget) == 0.0
    assert w.stake_for(None, "alert", NOW, dict(budget, staked=0.0, week=100.0)) == 0.0  # weekly cap
    assert w.stake_for(None, "alert", NOW, dict(budget, staked=0.0, paused=True)) == 0.0  # sticky pause


def test_ping_window_is_per_market_because_counts_decay_faster():
    # Receptions sit on integer lines; books correct them within the hour, so a 2025 +3% gap seen at
    # T-90 is worth -0.02% at the close while the same receiving-yards gap is worth +2.70%.
    assert w.alert_tier(0.05, 150, "receiving_yards") == "alert"
    assert w.alert_tier(0.05, 150, "receptions") == "early"
    assert w.alert_tier(0.05, 40, "receptions") == "alert"
    assert w.alert_tier(0.05, 150, None) == "alert"  # unknown market falls back to the default window
    assert w.alert_tier(0.02, 10, "receptions") == "logged"  # threshold still rules


def test_only_backtested_markets_carry_money():
    budget = dict(today=0.0, week=0.0, staked=0.0, paused=False)
    assert w.stake_for(None, "alert", NOW, budget, "receiving_yards") > 0
    assert w.stake_for(None, "alert", NOW, budget, "receptions") > 0
    assert w.stake_for(None, "alert", NOW, budget, "total") == 0.0  # scanned and graded, not staked
    assert w.stake_for(None, "alert", NOW, budget, "passing_yards") == 0.0


def test_anytime_td_is_normalised_onto_the_over_under_shape():
    # Yes/No with no point, priced as over/under a notional 0.5 line.
    td = lambda p, yes, no: [dict(name="Yes", description=p, price=yes), dict(name="No", description=p, price=no)]
    payload = {"bookmakers": [book("fanduel", td("Scorer", 650, -1100), "player_anytime_td"),
                              book("pinnacle", td("Scorer", 600, -950), "player_anytime_td")]}
    priced = w.find_edges(payload, min_ev=-1.0)  # ranked by EV, so the favourite side leads
    assert {(e["stat"], e["side"], e["line"]) for e in priced} == {("anytime_td", "over", 0.5), ("anytime_td", "under", 0.5)}
    # Proportional de-vig would call this yes-side a far better price than it is.
    fair = w.sharp_math.no_vig_over(600, -950)
    assert fair < w.sharp_math.no_vig_over(600, -950, method="proportional") - 0.01


def test_markets_are_only_fetched_while_they_could_still_be_bet():
    wanted = ["player_reception_yds", "player_receptions", "alternate_totals"]
    assert w.markets_for(200, wanted) == []                       # before any window opens
    assert "player_receptions" not in w.markets_for(150, wanted)   # counts decay fast; not yet worth paying for
    assert w.markets_for(30, wanted) == wanted                     # inside every window


def test_model_view_records_which_side_our_projection_takes():
    class FakeCur:
        def __init__(self, projection): self.projection = projection
        def execute(self, *a, **k): pass
        def fetchone(self): return None if self.projection is None else (self.projection,)
    game = dict(game_id="2026_05_TB_DAL")
    edge = dict(player_norm="emeka egbuka", stat="receptions", line=2.5, side="over")
    assert w.model_view(FakeCur(2.67), edge, game) == (2.67, True)      # projection above the line
    assert w.model_view(FakeCur(2.10), edge, game) == (2.10, False)     # below: model takes the under
    assert w.model_view(FakeCur(2.10), dict(edge, side="under"), game) == (2.10, True)
    assert w.model_view(FakeCur(None), edge, game) == (None, None)      # no frozen forecast for this bet
    assert w.model_view(FakeCur(2.67), dict(edge, player_norm=None), game) == (None, None)  # game totals
