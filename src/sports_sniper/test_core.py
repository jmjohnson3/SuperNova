"""Price math, normalization, consensus, opportunity ranking and line movement."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sports_sniper import db, odds_math
from sports_sniper.config import Settings
from sports_sniper.consensus import consensus_for
from sports_sniper.movement import moves
from sports_sniper.normalize import Quote, normalize_event, player_key
from sports_sniper.opportunities import current, detect
from sports_sniper.providers import sample

NOW = datetime(2026, 11, 10, 23, 0, tzinfo=timezone.utc)
SETTINGS = Settings(_env_file=None)


# --- odds math -------------------------------------------------------------------------------------
def test_american_and_decimal_convert_both_ways():
    assert odds_math.american_to_decimal(150) == pytest.approx(2.5)
    assert odds_math.american_to_decimal(-200) == pytest.approx(1.5)
    assert odds_math.decimal_to_american(2.5) == 150 and odds_math.decimal_to_american(1.5) == -200
    for price in (-10000, -250, -110, 100, 105, 340, 5000):
        assert odds_math.decimal_to_american(odds_math.american_to_decimal(price)) == price
    for bad in (50, 0, float("nan")):
        with pytest.raises(ValueError):
            odds_math.american_to_decimal(bad)
    with pytest.raises(ValueError):
        odds_math.decimal_to_american(1.0)


def test_expected_roi_matches_the_worked_examples():
    assert odds_math.implied_probability(700) == pytest.approx(0.125)
    assert odds_math.expected_roi(0.18, odds_math.american_to_decimal(700)) == pytest.approx(0.44)
    assert odds_math.expected_roi(0.20, odds_math.american_to_decimal(500)) == pytest.approx(0.20)
    assert odds_math.expected_roi(0.25, odds_math.american_to_decimal(350)) == pytest.approx(0.125)
    assert odds_math.expected_value(0.18, 8.0, stake=5) == pytest.approx(2.2)
    assert odds_math.expected_roi(0.5, odds_math.american_to_decimal(-110)) < 0


def test_no_vig_removes_the_margin_and_power_takes_more_from_the_longshot():
    for method in odds_math.DEVIG_METHODS:
        assert odds_math.no_vig([-110, -110], method) == pytest.approx([0.5, 0.5])
        assert sum(odds_math.no_vig([-330, 260], method)) == pytest.approx(1.0)
        assert sum(odds_math.no_vig([150, 200, 250], method)) == pytest.approx(1.0)
    assert odds_math.no_vig([260, -330], "power")[0] < odds_math.no_vig([260, -330], "multiplicative")[0]
    assert odds_math.no_vig([260, -330], "multiplicative")[0] < odds_math.implied_probability(260)


def test_no_vig_refuses_a_one_sided_market_and_unknown_methods():
    with pytest.raises(ValueError):
        odds_math.no_vig([240])
    with pytest.raises(ValueError):
        odds_math.no_vig([-110, -110], "magic")


def test_power_no_vig_agrees_with_the_existing_pipeline():
    sharp_math = pytest.importorskip("nfl_pipeline.sharp_math")
    for over, under in ((-120, 100), (260, -330), (-105, -115), (140, -170)):
        assert odds_math.no_vig([over, under], "power")[0] == pytest.approx(sharp_math.no_vig_over(over, under), abs=1e-6)


def test_fair_odds_and_conservative_probability():
    assert odds_math.fair_american(0.5) == 100 and odds_math.fair_american(0.25) == 300 and odds_math.fair_american(0.8) == -400
    assert odds_math.conservative_probability(0.20, 0.03) == pytest.approx(0.17)
    assert odds_math.conservative_probability(0.20, 0.03, z=2) == pytest.approx(0.14)
    assert odds_math.conservative_probability(0.01, 0.05) == 0.0
    with pytest.raises(ValueError):
        odds_math.fair_decimal(0)


def test_interval_and_calibration_helpers():
    assert odds_math.wilson_interval(0, 0) == (0.0, 1.0)
    low, high = odds_math.wilson_interval(50, 100)
    assert low < 0.5 < high and high - low == pytest.approx(0.19, abs=0.01)
    assert odds_math.brier_score([1.0, 0.0], [1, 0]) == 0 and odds_math.brier_score([0.5, 0.5], [1, 0]) == 0.25
    rows = odds_math.calibration_table([0.12, 0.18, 0.81, 0.89], [0, 1, 1, 1], bins=5)
    assert [(r["n"], r["observed"]) for r in rows] == [(2, 0.5), (2, 1.0)] and rows[0]["predicted"] == pytest.approx(0.15)


# --- normalization ---------------------------------------------------------------------------------
def snapshots(now=NOW):
    return sample.load(now)


def quotes(now=NOW, current_only=True, max_age=300):
    out = []
    for snap in snapshots(now):
        if current_only and snap.observed_at != now:
            continue
        out += normalize_event(snap.event, snap.sport, snap.observed_at, max_age_seconds=max_age)[0]
    return out


def test_every_time_is_utc_whatever_zone_it_arrives_in():
    from datetime import timezone as tz
    from sports_sniper.normalize import parse_time
    pacific = tz(timedelta(hours=-8))
    local = datetime(2026, 11, 10, 15, 0, tzinfo=pacific)
    assert parse_time(local) == NOW and parse_time(local).utcoffset() == timedelta(0)
    assert parse_time("2026-11-10T15:00:00-08:00").isoformat() == "2026-11-10T23:00:00+00:00"
    assert parse_time("2026-11-10T23:00:00Z") == NOW and parse_time(datetime(2026, 11, 10, 23, 0)) == NOW
    assert parse_time("yesterday") is None and parse_time(None) is None and parse_time(12) is None
    nba = next(s for s in snapshots() if s.sport == "nba" and s.observed_at == NOW)
    found, _ = normalize_event(nba.event, "nba", local, max_age_seconds=300)
    assert found and {q.observed_at.utcoffset() for q in found} == {timedelta(0)} and found[0].observed_at == NOW


def test_one_key_per_player_however_a_book_spells_him():
    assert player_key("Nikola Jokić") == player_key("NIKOLA JOKIC") == "nikola jokic"
    assert player_key("Michael Porter Jr.") == player_key("Michael Porter")
    assert player_key("A.J. Brown") == player_key("AJ Brown")
    assert player_key("Karl-Anthony Towns") == player_key("Karl Anthony Towns")


def test_sample_event_becomes_clean_quotes():
    nba = next(s for s in snapshots() if s.sport == "nba" and s.observed_at == NOW)
    found, counts = normalize_event(nba.event, "nba", NOW, max_age_seconds=300)
    assert counts["kept"] == len(found) == sum(len(m["outcomes"]) for b in nba.event["bookmakers"] for m in b["markets"])
    q = next(q for q in found if q.book == "pinnacle" and q.market == "player_points" and q.side == "over")
    assert (q.player, q.line, q.price, q.event_name) == ("Sample Player A", 24.5, -120, "Example Town Owls @ Sample City Comets")
    assert q.decimal == pytest.approx(1 + 100 / 120) and q.commence_time == NOW + timedelta(minutes=35)
    nfl = next(s for s in snapshots() if s.sport == "nfl" and s.observed_at == NOW)
    td = [q for q in normalize_event(nfl.event, "nfl", NOW)[0] if q.market == "player_anytime_td"]
    assert {q.side for q in td} == {"yes"} and {q.line for q in td} == {None} and len(td) == 4


def test_bad_and_stale_data_is_dropped_and_counted_never_raised():
    nba = next(s for s in snapshots() if s.sport == "nba" and s.observed_at == NOW).event
    assert normalize_event(nba, "nba", NOW, max_age_seconds=10)[1]["stale"] > 0
    assert normalize_event(nba, "nba", NOW - timedelta(hours=1))[1]["no_timestamp"] > 0     # prices dated after the scan
    assert normalize_event({}, "nba", NOW) == ([], {"invalid_event": 1})
    assert normalize_event({"id": "x", "commence_time": "not a time"}, "nba", NOW)[0] == []
    stamp = (NOW - timedelta(seconds=5)).isoformat()
    outcomes = [dict(name="Over", description="P One", price="abc", point=1.5), dict(name="Over", description="P One", price=50, point=1.5),
                dict(name="Over", description="P One", price=120), dict(name="Over", description="P One", price=120, point=1.5),
                dict(name="Over", description="P One", price=125, point=1.5), dict(name="Home Team", price=-110, point=-3.5)]
    event = dict(id="e", commence_time=(NOW + timedelta(hours=1)).isoformat(), home_team="H", away_team="A", bookmakers=[
        dict(key="fanduel", markets=[dict(key="player_points", last_update=stamp, outcomes=outcomes)]), dict(markets=[]), "junk"])
    found, counts = normalize_event(event, "nba", NOW)
    assert [q.price for q in found] == [125]                                                # the later duplicate wins
    assert (counts["invalid_price"], counts["duplicate"], counts["not_player_market"], counts["invalid_book"]) == (3, 1, 1, 2)
    wrapped, _ = normalize_event(dict(timestamp="t", data=event), "nba", NOW)
    assert len(wrapped) == 1


# --- consensus -------------------------------------------------------------------------------------
def q(book, side, price, line=24.5, market="player_points", player="Test Player", minutes=30, age=40, observed=NOW):
    return Quote(sport="nba", event_id="e1", event_name="A @ B", commence_time=observed + timedelta(minutes=minutes), book=book,
                 market=market, player=player, player_key=player_key(player), line=line, side=side, price=price,
                 decimal=odds_math.american_to_decimal(price), last_update=observed - timedelta(seconds=age), observed_at=observed)


def pair(book, over, under, **kw):
    return {book: {"over": q(book, "over", over, **kw), "under": q(book, "under", under, **kw)}}


def test_consensus_never_includes_the_book_being_judged():
    books = {**pair("draftkings", 105, -125), **pair("pinnacle", -120, 100)}
    ref = consensus_for(books["draftkings"]["over"], books)
    assert ref.books == ("pinnacle",) and ref.has_sharp
    assert ref.probability == pytest.approx(odds_math.no_vig([-120, 100], "power")[0])
    assert ref.standard_error == pytest.approx(0.015)
    under = consensus_for(books["draftkings"]["under"], books)
    assert ref.probability + under.probability == pytest.approx(1.0)


def test_no_probability_without_another_two_sided_book_close_in_time():
    alone = pair("draftkings", 105, -125)
    assert consensus_for(alone["draftkings"]["over"], alone) is None
    one_sided = {**alone, "fanduel": {"over": q("fanduel", "over", -110)}}
    assert consensus_for(alone["draftkings"]["over"], one_sided) is None
    old = {**alone, **pair("pinnacle", -120, 100, age=600)}
    assert consensus_for(alone["draftkings"]["over"], old, max_skew_seconds=180) is None


def test_weights_disagreement_and_retail_only_references_set_the_uncertainty():
    books = {**pair("draftkings", 105, -125), **pair("pinnacle", -120, 100), **pair("fanduel", 100, -120)}
    p_pin, p_fd = odds_math.no_vig([-120, 100])[0], odds_math.no_vig([100, -120])[0]
    ref = consensus_for(books["draftkings"]["over"], books, weights={"pinnacle": 3.0})
    assert ref.probability == pytest.approx((3 * p_pin + p_fd) / 4) and ref.n_books == 2
    assert ref.standard_error > 0.015 and ref.spread > 0                 # the two references disagree by nine points
    soft = {**pair("draftkings", 105, -125), **pair("fanduel", 100, -120)}
    retail = consensus_for(soft["draftkings"]["over"], soft)
    assert not retail.has_sharp and retail.standard_error == pytest.approx(0.03)


# --- opportunities ---------------------------------------------------------------------------------
def find(found, player, side, book):
    return next((o for o in found if o.player == player and o.side == side and o.book == book), None)


def test_sample_scan_ranks_supported_edges_and_labels_the_evidence():
    found = detect(quotes(), SETTINGS)
    best = find(found, "Sample Player H", "over", "draftkings")
    assert best.status == "priced" and best.source == "market_consensus" and best.evidence == "strong" and best.is_longshot
    assert best.ev == pytest.approx(0.105, abs=0.002) and 0 < best.conservative_ev < best.ev
    assert best.reference_books == ("fanduel", "pinnacle") and best.fair_price < best.price
    assert best.ev == pytest.approx(odds_math.expected_roi(best.probability, best.decimal))
    assert "DraftKings pays +340" in best.explanation and "margin removed" in best.explanation
    retail = find(found, "Sample Player D", "over", "draftkings")
    assert retail.evidence == "speculative" and not retail.has_sharp and "retail books only" in retail.explanation
    assert find(found, "Sample Player C", "over", "draftkings") is None            # fairly priced everywhere
    assert not any(o.book == "pinnacle" for o in found)                            # the sharp book is a reference only
    priced = [o for o in found if o.status == "priced"]
    assert [o.conservative_ev for o in priced] == sorted((o.conservative_ev for o in priced), reverse=True)
    assert found[len(priced):] and all(o.status == "insufficient_data" for o in found[len(priced):])
    assert all(o.ev > 0 for o in priced)


def test_one_sided_longshots_get_no_probability_only_a_price_comparison():
    found = detect(quotes(), SETTINGS)
    td = find(found, "Sample Player E", "yes", "draftkings")
    assert td.status == "insufficient_data" and td.evidence == "insufficient"
    assert td.probability is None and td.ev is None and td.conservative_ev is None and td.fair_price is None
    assert td.premium_vs_median == pytest.approx(3.4 / 2.75 - 1) and td.books_quoting == 4 and td.is_best_price
    assert "insufficient data" in td.explanation and "No book prices the other side" in td.explanation
    assert find(found, "Sample Player E", "yes", "fanduel") is None                # not the best price: not flagged
    alt = [o for o in found if o.market == "player_points_alternate"]
    assert [(o.book, o.line, o.status) for o in alt] == [("draftkings", 34.5, "insufficient_data")]


def test_different_lines_are_never_treated_as_the_same_bet():
    books = [q("draftkings", "over", 150, line=24.5), q("draftkings", "under", -180, line=24.5),
             q("pinnacle", "over", -120, line=25.5), q("pinnacle", "under", 100, line=25.5)]
    assert detect(books, SETTINGS) == []


def test_an_edge_too_big_to_believe_is_speculative():
    books = [q("draftkings", "over", 160), q("draftkings", "under", -200), q("pinnacle", "over", -120), q("pinnacle", "under", 100),
             q("fanduel", "over", -118), q("fanduel", "under", -104)]
    over = find(detect(books, SETTINGS), "Test Player", "over", "draftkings")
    assert over.ev > 0.15 and over.conservative_ev > 0 and over.has_sharp and over.evidence == "speculative"
    assert "stale price" in over.explanation
    relaxed = Settings(_env_file=None, max_plausible_ev=0.60)
    assert find(detect(books, relaxed), "Test Player", "over", "draftkings").evidence == "strong"


def test_started_games_and_thin_references_are_left_out():
    live = [q("draftkings", "over", 150, minutes=-5), q("draftkings", "under", -180, minutes=-5),
            q("pinnacle", "over", -120, minutes=-5), q("pinnacle", "under", 100, minutes=-5)]
    assert detect(live, SETTINGS) == []
    strict = Settings(_env_file=None, min_reference_books=4)
    assert not [o for o in detect(quotes(), strict) if o.status == "priced"]


def test_best_price_flag_and_latest_snapshot_per_bet():
    found = detect(quotes(current_only=False), SETTINGS)
    both = [o for o in found if o.player == "Sample Player H" and o.side == "over" and o.book == "draftkings"]
    assert len(both) == 2 and len({o.observed_at for o in both}) == 2      # one per snapshot, never mixed together
    newest = current(found, {"sample-nba-001": NOW, "sample-nfl-001": NOW})
    assert len(newest) < len(found) and {o.observed_at for o in newest} == {NOW}
    assert current(found, {"sample-nba-001": NOW + timedelta(minutes=5)}) == []        # a newer look flagged nothing
    assert len([o for o in newest if o.player == "Sample Player H" and o.side == "over"]) == 1
    assert find(newest, "Sample Player H", "over", "draftkings").observed_at == NOW
    g = find(newest, "Sample Player G", "over", "fanduel")
    assert g.is_best_price and g.books_quoting == 2 and g.price == 165


# --- line movement ---------------------------------------------------------------------------------
def test_line_movement_compares_first_and_latest_stored_price():
    frame = pd.DataFrame(db.quote_rows(quotes(current_only=False), scan_id=1))
    changed = moves(frame)
    assert len(changed) == 2
    a = changed[changed.player == "Sample Player A"].iloc[0]
    assert (a.book, a.side, a.first_price, a.last_price, a.observations) == ("draftkings", "over", -110, 105, 2)
    assert a.break_even_change == pytest.approx(1 / 2.05 - 110 / 210) and a.break_even_change < 0
    g = changed[changed.player == "Sample Player G"].iloc[0]
    assert (g.first_price, g.last_price, g.best_now, g.best_book) == (150, 165, 165, "fanduel")
    assert len(moves(frame, threshold=0.03)) == 1 and moves(frame.iloc[:0]).empty


# --- settings --------------------------------------------------------------------------------------
def test_settings_read_the_environment_and_keep_the_key_secret(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "super-secret-key")
    monkeypatch.setenv("SNIPER_MARKETS", '{"nba": ["player_points"]}')
    monkeypatch.setenv("SNIPER_CREDIT_FLOOR", "5000")
    settings = Settings(_env_file=None)
    assert settings.api_key() == "super-secret-key" and "super-secret-key" not in repr(settings)
    assert settings.markets_for("nba") == ["player_points"] and settings.credit_floor == 5000
    assert settings.markets_for("nfl") == ["player_anytime_td", "player_pass_tds", "player_rush_yds", "player_reception_yds"]
    assert settings.markets_for("cricket") == []
