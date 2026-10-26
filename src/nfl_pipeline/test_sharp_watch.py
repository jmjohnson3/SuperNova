from datetime import datetime, timedelta, timezone

import pytest

from nfl_pipeline import sharp_watch as w

NOW = datetime(2026, 10, 4, 15, 0, tzinfo=timezone.utc)


def book(key, outcomes, market="player_reception_yds"):
    return {"key": key, "last_update": NOW.isoformat(),
            "markets": [{"key": market, "last_update": NOW.isoformat(), "outcomes": outcomes}]}


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW
    monkeypatch.setattr(w, 'datetime', Clock)


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
    # Only the final hour is polled: it holds the stake window and the closing snapshot.
    games = [game("last_hour", 30), game("window", 55), game("early", 200), game("far", 3000)]
    due, skipped = w.plan_polls(games, {}, NOW, remaining=5000, cost=1, floor=150)
    assert [g["game_id"] for g in due] == ["last_hour", "window"]
    recent = {"last_poll": {"last_hour": (NOW - timedelta(minutes=3)).isoformat(),
                            "window": (NOW - timedelta(minutes=4)).isoformat()}}
    due, _ = w.plan_polls(games, recent, NOW, remaining=5000, cost=1, floor=150)
    assert due == []  # 5-minute cadence
    due, _ = w.plan_polls(games, {"last_poll": {"last_hour": (NOW - timedelta(minutes=6)).isoformat()}},
                          NOW, remaining=5000, cost=1, floor=150)
    assert [g["game_id"] for g in due] == ["last_hour", "window"]  # 6 min > the 5-minute cadence
    due, skipped = w.plan_polls(games, {}, NOW, remaining=150, cost=1, floor=150)
    assert due == [] and skipped == {"last_hour": "credit_floor", "window": "credit_floor"}
    assert w.daily_allowance(200, NOW.date(), 150) == pytest.approx(50 / 28)
    # Every poll is inside the final window, so only the floor binds, never the daily allowance.
    due, skipped = w.plan_polls(games, {}, NOW, remaining=200, cost=1, floor=150)
    assert len(due) == 2 and skipped == {}


def test_every_five_minutes_through_kickoff_so_each_decision_has_a_close():
    start = NOW + timedelta(minutes=180)
    g = [dict(game_id="tnf", start=start, home="CLE", away="PIT", day=NOW.date())]
    polled, t = [], NOW
    while t < start:
        due, _ = w.plan_polls(g, {"last_poll": {"tnf": polled[-1].isoformat()}} if polled else {}, t,
                              remaining=496, cost=1, floor=150)
        if due:
            polled.append(t)
        t += timedelta(minutes=5)
    minutes = [(start - p).total_seconds() / 60 for p in polled]
    assert minutes[0] == 60 and minutes[-1] == 5 and len(polled) == 12
    # The floor still binds.
    due, skipped = w.plan_polls(g, {}, start - timedelta(minutes=30), remaining=150, cost=1, floor=150)
    assert due == [] and skipped == {"tnf": "credit_floor"}


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
    games = [dict(game_id=f"g{i}", start=NOW + timedelta(minutes=40), home="CLE", away="PIT", day=NOW.date()) for i in range(14)]
    due, skipped = w.plan_polls(games, {}, NOW, remaining=20000, cost=4, floor=150, share=14 / 62)
    assert len(due) == 14 and skipped == {}


def test_only_gaps_seen_15_to_45_minutes_out_are_alerts():
    assert w.alert_tier(0.05, 45) == "alert" and w.alert_tier(0.05, 30) == "alert" and w.alert_tier(0.05, 15) == "alert"
    assert w.alert_tier(0.05, 46) == "early"  # still stored; upgraded and staked if it holds into the window
    assert w.alert_tier(0.05, 14) == "late"   # too close to kickoff to place by hand
    assert w.alert_tier(0.02, 30) == "logged"
    for stat in ("receiving_yards", "rushing_yards", "receptions", None):  # one window for every market
        assert w.alert_tier(0.05, 30, stat) == "alert" and w.alert_tier(0.05, 60, stat) == "early"


def test_every_bettable_book_is_priced_against_the_sharp_line():
    # Pinnacle fair over ~58%. DraftKings is generous on the over, FanDuel on the under.
    payload = {"bookmakers": [book("fanduel", pair("DK Metcalf", 45.5, -200, +170)),
                              book("draftkings", pair("DK Metcalf", 45.5, +120, -140)),
                              book("pinnacle", pair("DK Metcalf", 45.5, -140, +120))]}
    edges = w.find_edges(payload, bet_books=("fanduel", "draftkings"))
    assert {(e["book"], e["side"]) for e in edges} == {("draftkings", "over"), ("fanduel", "under")}
    assert edges[0]["ev"] >= edges[-1]["ev"]  # still ranked by EV across books
    # A book we cannot bet at is never alerted on, even when it is the most generous price.
    assert w.find_edges(payload, bet_books=("fanduel",)) == [e for e in edges if e["book"] == "fanduel"]


def test_default_scan_bets_fanduel_espn_bet_and_draftkings():
    payload = {"bookmakers": [book("fanduel", pair("DK Metcalf", 45.5, -200, +170)),
                              book("espnbet", pair("DK Metcalf", 45.5, +125, -145)),
                              book("draftkings", pair("DK Metcalf", 45.5, +120, -140)),
                              book("betmgm", pair("DK Metcalf", 45.5, +150, -190)),
                              book("pinnacle", pair("DK Metcalf", 45.5, -140, +120))]}
    assert {edge["book"] for edge in w.find_edges(payload)} == {"fanduel", "espnbet", "draftkings"}
    assert len(w.BOOKMAKERS) <= 10 and "williamhill_us" not in w.BOOKMAKERS  # one region per market


def test_other_books_keep_only_their_own_verified_links():
    game = dict(start=NOW + timedelta(minutes=30), home="CLE", away="PIT")
    edge = dict(book="draftkings", stat="receiving_yards", player="X", player_norm="x", side="over",
                line=45.5, price=120, sharp_book="pinnacle", sharp_line=45.5, line_gap=0.0,
                fair_probability=0.58, ev=0.05, minimum_price=110,
                link="https://sportsbook.draftkings.com/event/123")
    # a plain event page names no bet; the ?outcomes= form carries the exact selection
    assert w.book_link(edge, game) == ("https://sportsbook.draftkings.com/event/123", "page")
    withsel = dict(edge, link="https://sportsbook.draftkings.com/?outcomes=0QA37612%232309657")
    assert w.book_link(withsel, game)[1] == "selection"
    assert w.book_link(dict(edge, link="https://evil.example.com/?outcomes=1"), game) == (None, None)
    assert w.book_link(dict(edge, link="http://sportsbook.draftkings.com/x"), game) == (None, None)
    assert "find it manually" in w.format_alert(edge, game, stake=5.0)
    text = w.format_alert(withsel, game, stake=5.0)
    assert "bet $5 at DraftKings" in text and "DraftKings +120" in text and "Open this bet on DraftKings" in text


def budget(**changes):
    return dict(dict(today=0, week=0.0, staked=0.0, staked_count=0, staked_keys=set(), per_game={},
                     paused=False), **changes)


def test_stake_respects_caps_and_the_pauses():
    assert w.strategy.STAKE == 1.0 and w.strategy.MAX_BETS_PER_DAY == 10
    assert w.strategy.MAX_STAKE_PER_WEEK == 40.0 and w.strategy.LOSS_PAUSE == -30.0
    assert w.stake_for(None, "alert", NOW, budget()) == 1.0
    for tier in ("logged", "early", "late"):  # only alerts inside the stake window carry money
        assert w.stake_for(None, tier, NOW, budget()) == 0.0
    assert w.stake_for(None, "alert", NOW, budget(today=10)) == 0.0              # ten bets today already
    assert w.stake_for(None, "alert", NOW, budget(today=9, staked_count=1)) == 0.0  # ...counting this run
    assert w.stake_for(None, "alert", NOW, budget(week=39.5)) == 0.0              # weekly cap
    assert w.stake_for(None, "alert", NOW, budget(paused=True)) == 0.0            # loss or readiness pause


def test_one_stake_per_player_game_stat_across_books_and_two_per_game():
    b = budget()
    first = ("g1", "receiving_yards", "a")
    assert w.stake_for(None, "alert", NOW, b, "receiving_yards", first) == 1.0
    w.commit_stake(b, 1.0, first)
    # The same player and stat at another book, or at a later price, is graded but not staked again.
    assert w.stake_for(None, "alert", NOW, b, "receiving_yards", first) == 0.0
    second = ("g1", "rushing_yards", "b")
    assert w.stake_for(None, "alert", NOW, b, "rushing_yards", second) == 1.0
    w.commit_stake(b, 1.0, second)
    assert w.stake_for(None, "alert", NOW, b, "receptions", ("g1", "receptions", "c")) == 0.0  # game cap
    assert w.stake_for(None, "alert", NOW, b, "receptions", ("g2", "receptions", "c")) == 1.0
    assert b["staked"] == 2.0 and b["staked_count"] == 2


def test_only_backtested_markets_carry_money():
    assert w.stake_for(None, "alert", NOW, budget(), "receiving_yards") > 0
    assert w.stake_for(None, "alert", NOW, budget(), "receptions") > 0
    assert w.stake_for(None, "alert", NOW, budget(), "rushing_yards") > 0
    assert w.stake_for(None, "alert", NOW, budget(), "total") > 0  # game totals carry stakes since v7
    assert w.stake_for(None, "alert", NOW, budget(), "passing_yards") == 0.0  # scanned and graded, not staked


def test_disabled_strategy_is_always_research_only(monkeypatch):
    monkeypatch.setattr(w.strategy, "BETS_ENABLED", False)
    assert w.stake_for(None, "alert", NOW, budget(), "receiving_yards") == 0.0


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


def test_markets_are_only_fetched_in_the_final_hour():
    wanted = ["player_reception_yds", "player_receptions", "alternate_totals"]
    assert w.markets_for(61, wanted) == []       # before the poll horizon
    assert w.markets_for(60, wanted) == wanted   # a little before the stake window opens
    assert w.markets_for(3, wanted) == wanted    # through kickoff, for the close
    assert w.markets_for(0, wanted) == []


class FakeCur:
    def __init__(self, projection, **fields):
        self.row = None if projection is None else tuple(
            dict(dict(projection=projection, line=None, over_probability=None, under_probability=None,
                      injury_report_status=None, roster_status="ACT", limited_usage_risk=0.1), **fields)[k]
            for k in w.FORECAST_FIELDS)
    def execute(self, *a, **k): pass
    def fetchone(self): return self.row


def test_model_view_records_which_side_our_projection_takes():
    game = dict(game_id="2026_05_TB_DAL")
    edge = dict(player_norm="emeka egbuka", stat="receptions", line=2.5, side="over")
    assert w.model_view(FakeCur(2.67), edge, game) == (2.67, True)      # projection above the line
    assert w.model_view(FakeCur(2.10), edge, game) == (2.10, False)     # below: model takes the under
    assert w.model_view(FakeCur(2.10), dict(edge, side="under"), game) == (2.10, True)
    assert w.model_view(FakeCur(None), edge, game) == (None, None)      # no frozen forecast for this bet
    assert w.model_view(FakeCur(2.67), dict(edge, player_norm=None), game) == (None, None)  # game totals


def test_shadow_filter_flags_playing_time_and_role_disagreement():
    over = dict(side="over", line=45.5, fair_probability=0.56)
    fine = dict(projection=50.0, line=45.5, over_probability=0.55, under_probability=0.45,
                injury_report_status=None, roster_status="ACT", limited_usage_risk=0.1)
    assert w.shadow_filter(over, fine) == ("pass", [])
    assert w.shadow_filter(over, None) == ("no_forecast", [])
    assert w.shadow_filter(over, dict(fine, injury_report_status="Questionable")) == ("flag", ["over_with_questionable_tag"])
    assert w.shadow_filter(dict(over, side="under"), dict(fine, injury_report_status="Questionable"))[0] == "pass"
    assert w.shadow_filter(over, dict(fine, injury_report_status="Out"))[1][0] == "injury_out"
    assert w.shadow_filter(over, dict(fine, limited_usage_risk=0.45)) == ("flag", ["over_with_limited_usage_risk"])
    assert w.shadow_filter(over, dict(fine, roster_status="RES"))[1] == ["roster_res"]
    # A strong side disagreement only counts at the same line; across lines the model is not comparable.
    assert w.shadow_filter(over, dict(fine, over_probability=0.40)) == ("flag", ["model_strongly_prefers_other_side"])
    assert w.shadow_filter(over, dict(fine, over_probability=0.40, line=47.5)) == ("pass", [])
    assert w.shadow_filter(over, dict(fine, over_probability=0.45)) == ("pass", [])  # 11 points is not strong


def test_model_context_carries_the_filter_verdict(monkeypatch):
    monkeypatch.setattr(w.sharp_inactives, "statuses", lambda session=None: {})
    game = dict(game_id="g")
    edge = dict(player_norm="x", stat="receiving_yards", line=45.5, side="over", fair_probability=0.56)
    context = w.model_context(FakeCur(40.0, injury_report_status="Questionable"), edge, game)
    assert context == dict(projection=40.0, agrees=False, filter="flag", filter_reasons=["over_with_questionable_tag"],
                           inactives=dict(espn_status=None, teammates_out=[], adjusted_projection=None))
    assert w.model_context(FakeCur(None), edge, game)["filter"] == "no_forecast"


@pytest.mark.parametrize('book_index,stamp', [(0, None), (1, None), (0, NOW-timedelta(seconds=301)),
    (1, NOW-timedelta(seconds=301)), (0, NOW+timedelta(seconds=1)), (1, NOW+timedelta(seconds=1))])
def test_invalid_source_timestamps_never_create_edges(book_index, stamp):
    payload = {'bookmakers': [book('fanduel', pair('X', 45.5, -110, -110)),
                             book('pinnacle', pair('X', 45.5, -140, 120))]}
    changed = payload['bookmakers'][book_index]
    changed['last_update'] = changed['markets'][0]['last_update'] = stamp.isoformat() if stamp else None
    assert w.find_edges(payload, now=NOW) == []


def test_two_individually_fresh_quotes_must_also_be_time_aligned():
    payload = {'bookmakers': [book('fanduel', pair('X', 45.5, -110, -110)),
                             book('pinnacle', pair('X', 45.5, -140, 120))]}
    payload['bookmakers'][1]['markets'][0]['last_update'] = (NOW-timedelta(seconds=121)).isoformat()
    assert w.find_edges(payload, now=NOW) == []


def test_exchange_without_cost_and_liquidity_proof_is_not_a_reference():
    payload = {'bookmakers': [book('fanduel', pair('X', 45.5, -110, -110)),
                             book('betfair_ex_eu', pair('X', 45.5, -140, 120))]}
    assert w.find_edges(payload, now=NOW) == []


def test_observations_capture_fair_quotes_and_preserve_original_payload():
    payload = {'id':'event', 'bookmakers': [book('fanduel', pair('X', 45.5, -110, -110)),
                                           book('pinnacle', pair('X', 45.5, -110, -110))]}
    observations = w.quote_observations(payload, game('game', 30), NOW)
    assert len(observations) == 2 and not w.find_edges(payload, now=NOW)
    assert observations == w.quote_observations(payload, game('game', 30), NOW)
    changed = w.quote_observations(payload, game('game', 30), NOW+timedelta(seconds=1))
    assert observations[0]['observation_id'] != changed[0]['observation_id']
    assert observations[0]['quote']['market_updated_at'] == NOW.isoformat()


def test_unregistered_scan_exits_before_reading_games_or_network(monkeypatch):
    from nfl_pipeline import experiment_registry
    monkeypatch.setattr(experiment_registry, 'load_experiment', lambda name: None)
    monkeypatch.setattr(w, '_games', lambda now: pytest.fail('unregistered scan must not start'))
    monkeypatch.setattr(w, '_has_games_soon', lambda now: False)
    assert w.run(now=NOW)['status'] == 'research_not_registered'


def test_source_drift_exits_before_reading_games_or_network(monkeypatch):
    # Drift must refuse as a status row like every other refusal here. Raising took
    # the whole watcher down on 19 October over a column the scanner never reads.
    from nfl_pipeline import experiment_registry

    def drifted(name, **kwargs):
        raise ValueError('Frozen experiment changed; restore the pinned version')

    monkeypatch.setattr(experiment_registry, 'load_experiment', lambda name: {'config': {}})
    monkeypatch.setattr(experiment_registry, 'check_experiment', drifted)
    monkeypatch.setattr(w, '_games', lambda now: pytest.fail('drifted scan must not start'))
    monkeypatch.setattr(w, '_has_games_soon', lambda now: False)
    result = w.run(now=NOW)
    assert result['status'] == 'experiment_source_drift'
    assert result['experiment_id'] == w.EXPERIMENT_NAME


def test_a_refusal_with_games_close_is_announced_once_a_day(monkeypatch, tmp_path):
    # 16-21 October: every run refused and nothing said so.
    from nfl_pipeline import experiment_registry
    monkeypatch.setattr(experiment_registry, 'load_experiment', lambda name: {'config': {'scanner': {'markets': ['a']}}})
    monkeypatch.setattr(experiment_registry, 'check_experiment', lambda name, **k: {'config': {'scanner': {'markets': ['a']}}})
    monkeypatch.setattr(w, 'STATE', tmp_path / 'state.json')
    monkeypatch.setattr(w, '_has_games_soon', lambda now: True)
    sent = []
    first = w.run(now=NOW, post=sent.append)
    assert first['status'] == 'experiment_configuration_drift' and 'markets' in first['detail'] and first['warned']
    assert w.run(now=NOW + timedelta(minutes=5), post=sent.append).get('warned') is None
    assert len(sent) == 1 and 'not running' in sent[0]
    w.run(now=NOW + timedelta(days=1), post=sent.append)
    assert len(sent) == 2
    monkeypatch.setattr(w, '_has_games_soon', lambda now: False)  # nothing to lose: stay quiet
    w.run(now=NOW + timedelta(days=2), post=sent.append)
    assert len(sent) == 2


def test_registration_pins_only_the_strategy_modules():
    files = w.pinned_files()
    assert all(path.is_file() for path in files.values())
    assert set(files) == set(w.PINNED_MODULES) | {"alt_line_model"}  # the ladder pricing model is an input
    assert not any(name in module for module in files for name in
                   ("predict_player_props", "features", "receiving_paper_trial", "cash_", "betting_preferences"))
    config = w.normalized_config()
    assert config == w.normalized_config() and config["stake_window_minutes"] == [15, 45]
    assert config["bet_books"] == ["fanduel", "espnbet", "draftkings"]


def test_the_api_moving_kickoff_a_few_minutes_does_not_drop_the_closing_snapshot():
    # 11 October: PHI-JAX was scheduled 13:30Z; the last two polls reported 13:35Z and were discarded.
    assert w.same_kickoff("2026-10-11T13:35:00Z", datetime(2026, 10, 11, 13, 30, tzinfo=timezone.utc))
    assert w.same_kickoff("2026-10-11T13:30:00Z", datetime(2026, 10, 11, 13, 30, tzinfo=timezone.utc))
    assert not w.same_kickoff("2026-10-11T17:00:00Z", datetime(2026, 10, 11, 13, 30, tzinfo=timezone.utc))
    assert not w.same_kickoff(None, datetime(2026, 10, 11, 13, 30, tzinfo=timezone.utc))


def test_thescore_bet_links_are_accepted_as_espn_bet_and_foreign_hosts_are_not():
    # Since the rebrand every Odds API `espnbet` link is on sportsbook.thescore.bet (all 22,282 in October).
    game = dict(start=NOW + timedelta(minutes=30), home="CHI", away="NE")
    edge = dict(book="espnbet", stat="receiving_yards", player="X", player_norm="x", side="over", line=45.5,
                price=120, sharp_book="pinnacle", sharp_line=45.5, line_gap=0.0, fair_probability=0.58, ev=0.05,
                minimum_price=110,
                link="https://sportsbook.thescore.bet/?market_selection_id[0]=da7f2abf&odds_numerator[0]=2&odds_denominator[0]=1")
    assert w.book_link(edge, game) == (edge["link"], "selection")
    assert "Open this bet on theScore Bet (ESPN Bet)" in w.format_alert(edge, game, stake=1.0)
    assert w.book_link(dict(edge, link="https://thescore.bet.evil.example/?market_selection_id[0]=1"), game) == (None, None)


def test_a_fanduel_fallback_with_another_states_ids_is_labelled_manual(monkeypatch):
    from nfl_pipeline import fanduel_links, fanduel_state
    monkeypatch.setattr(fanduel_state, "state", lambda: "co")
    monkeypatch.setattr(fanduel_links, "betslip_for_row", lambda row: None)  # line moved / not in Colorado's feed
    game = dict(start=NOW + timedelta(minutes=30), home="CHI", away="NE")
    edge = dict(book="fanduel", stat="receptions", player="X", player_norm="x", side="over", line=3.5, price=120,
                sharp_book="pinnacle", sharp_line=3.5, line_gap=0.0, fair_probability=0.58, ev=0.05, minimum_price=110,
                link="https://sportsbook.fanduel.com/addToBetslip?marketId=42.6156&selectionId=15290705")
    url, kind = w.book_link(edge, game)
    assert kind == "page" and "find it manually" in w.format_alert(edge, game, stake=1.0)
    monkeypatch.setattr(fanduel_links, "betslip_for_row", lambda row: "https://account.sportsbook.fanduel.com/x")
    assert w.book_link(edge, game)[1] == "betslip"


def ladder_book(key, player, rungs):
    outcomes = [dict(name="Over", description=player, point=line, price=price) for line, price in rungs]
    return {"key": key, "last_update": NOW.isoformat(),
            "markets": [{"key": "player_reception_yds_alternate", "last_update": NOW.isoformat(), "outcomes": outcomes}]}


def test_an_over_only_ladder_rung_is_priced_from_pinnacles_main_line_as_research():
    payload = {"bookmakers": [ladder_book("draftkings", "X", [(60.5, 400), (80.5, 900)]),
                              book("pinnacle", pair("X", 45.5, -110, -110))]}
    edges = w.find_edges(payload, bet_books=("draftkings",), now=NOW)
    rung = next(e for e in edges if e["line"] == 60.5)
    fair = w.sharp_math.ladder_over_probability(w.LADDER_MODEL, "receiving_yards", 45.5, 0.5, 60.5)
    assert rung["ladder"] and rung["side"] == "over" and rung["fair_probability"] == pytest.approx(fair)
    assert rung["quote_provenance"]["reference_method"] == "ladder_model"
    assert rung["ev"] == pytest.approx(w.sharp_math.ev(fair, 400))
    # No Pinnacle main line for the player: the rung cannot be valued and is not an edge.
    assert w.find_edges({"bookmakers": [ladder_book("draftkings", "X", [(60.5, 400)])]}, bet_books=("draftkings",), now=NOW) == []


def test_a_ladder_rung_never_overwrites_the_main_line_at_the_same_number():
    both = {"key": "fanduel", "last_update": NOW.isoformat(), "markets": [
        {"key": "player_reception_yds_alternate", "last_update": NOW.isoformat(),
         "outcomes": [dict(name="Over", description="X", point=45.5, price=150)]},
        {"key": "player_reception_yds", "last_update": NOW.isoformat(), "outcomes": pair("X", 45.5, -110, -110)}]}
    quote = w._pairs(both)[("receiving_yards", "x", 45.5)]
    assert quote["over"] == -110 and quote["under"] == -110 and not quote["ladder"]


def test_ladders_are_not_polled_in_v8_and_totals_carry_stakes():
    assert "total" in w.strategy.STAKED_STATS
    assert not set(w.LADDER_MARKETS) & set(w.markets())   # v8: no edge found in ladders; not scanned
    assert w.normalized_config()["ladder_scan"] is False
    assert "alt_line_model" in w.pinned_files()
    config = w.normalized_config()
    assert config["ladder_markets"] == w.LADDER_MARKETS and config["ladder_model_rows"] > 1000



def test_daily_bets_are_shared_across_kickoff_windows():
    u = lambda h, m: datetime(2026, 10, 25, h, m, tzinfo=timezone.utc)
    night = u(0, 20) + timedelta(days=1)
    kicks = [u(13, 30)] + [u(17, 0)] * 7 + [u(20, 5)] + [u(20, 25)] * 2 + [night]
    assert [n for _, n in w.strategy.window_allocation(kicks, 5)] == [1, 2, 1, 1]
    assert [w.strategy.cumulative_cap(kicks, k, 5) for k in (u(13, 30), u(17, 0), u(20, 25), night)] == [1, 3, 4, 5]
    assert [n for _, n in w.strategy.window_allocation(kicks)] == [1, 6, 2, 1]   # v9: 10 a day
    # one game: everything goes to its window; more windows than bets: the busiest windows get one each
    assert w.strategy.window_allocation([night], 5) == [(night, 5)]
    many = [u(13, 0) + timedelta(hours=i) for i in range(7)]
    assert sum(n for _, n in w.strategy.window_allocation(many, 5)) == 5


def test_stake_for_respects_the_window_cap_and_rolls_unused_bets_forward(monkeypatch):
    monkeypatch.setattr(w.strategy, "MAX_BETS_PER_DAY", 5)   # the v8 cap; the rule scales with it
    u = lambda h, m: datetime(2026, 10, 25, h, m, tzinfo=timezone.utc)
    kicks = [u(13, 30)] + [u(17, 0)] * 7 + [u(20, 25)] * 2 + [u(0, 20) + timedelta(days=1)]
    budget = dict(today=0, week=0.0, staked=0.0, staked_count=0, staked_keys=set(), per_game={}, paused=False,
                  kickoffs=kicks)
    assert w.stake_for(None, "alert", NOW, budget, "receptions", ("a", "receptions", "p1"), u(13, 30)) == 1.0
    w.commit_stake(budget, 1.0, ("a", "receptions", "p1"))
    assert w.stake_for(None, "alert", NOW, budget, "receptions", ("a", "receptions", "p2"), u(13, 30)) == 0.0
    # the 1 PM window opens with 3 for the day so far
    for i in range(2):
        key = (f"g{i}", "receptions", "p")
        assert w.stake_for(None, "alert", NOW, budget, "receptions", key, u(17, 0)) == 1.0
        w.commit_stake(budget, 1.0, key)
    assert w.stake_for(None, "alert", NOW, budget, "receptions", ("g9", "receptions", "p"), u(17, 0)) == 0.0
    # a quiet early window's unused bet would have rolled forward
    fresh = dict(budget, today=0, staked=0.0, staked_count=0, staked_keys=set(), per_game={})
    for i in range(3):
        key = (f"h{i}", "receptions", "p")
        assert w.stake_for(None, "alert", NOW, fresh, "receptions", key, u(17, 0)) == 1.0
        w.commit_stake(fresh, 1.0, key)


def test_teammate_out_after_the_forecast_moves_volume_and_can_flip_the_side():
    rows = [("a", "A", "PHI", 80.0, ""), ("b", "B", "PHI", 40.0, ""), ("c", "C", "PHI", 20.0, ""),
            ("d", "D", "DAL", 60.0, ""), ("e", "E", "PHI", 30.0, "out")]
    status = {("PHI", "a"): "out", ("PHI", "e"): "out", ("PHI", "b"): "active"}
    shift = w.sharp_inactives.teammate_shift(rows, "b", status)
    assert shift["teammates_out"] == ["a"]                       # e was already out in the forecast
    assert shift["adjusted_projection"] == pytest.approx(40.0 + 80.0 * 40.0 / 60.0, abs=0.01)
    assert w.sharp_inactives.teammate_shift(rows, "zz", status)["adjusted_projection"] is None


def test_espn_out_and_a_flipped_side_are_recorded_as_shadow_flags(monkeypatch):
    monkeypatch.setattr(w.sharp_inactives, "context", lambda cur, edge, game: dict(
        espn_status=None, teammates_out=["a"], adjusted_projection=52.0))
    edge = dict(player_norm="b", stat="receiving_yards", line=45.5, side="under", fair_probability=0.52)
    context = w.model_context(FakeCur(40.0), edge, dict(game_id="g"))
    assert "teammate_out_flips_model_side" in context["filter_reasons"] and context["filter"] == "flag"
    monkeypatch.setattr(w.sharp_inactives, "context", lambda cur, edge, game: dict(
        espn_status="out", teammates_out=[], adjusted_projection=None))
    assert "espn_out" in w.model_context(FakeCur(40.0), edge, dict(game_id="g"))["filter_reasons"]
