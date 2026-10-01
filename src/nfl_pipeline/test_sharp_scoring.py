import json
from datetime import datetime, timedelta, timezone

import pytest

from nfl_pipeline.modeling import predict_player_props as props
from nfl_pipeline.modeling.scoring_capture import capture, replay
from nfl_pipeline.test_accuracy_challengers import scoring_fixture

NOW = datetime(2026, 10, 1, 22, 45, tzinfo=timezone.utc)


def market(trust):
    return {"version": "t", "stats": {"receiving_yards": {"projection_trust": 0.0, "range_scale": 1.0, "probability_trust": trust}}}


def setup(sharp_over, sharp_under, trust=0.0, enabled=False):
    row, offer, metrics, distribution = scoring_fixture()
    offer = dict(offer, bookmaker_key="fanduel", over_price=-110, under_price=-110,
                 sharp_reference=dict(bookmaker_key="pinnacle", line=offer["line"], over_price=sharp_over,
                                      under_price=sharp_under, fetched_at_utc=str(NOW)))
    metrics = dict(metrics, market_calibration=market(trust), bet_stats=["receiving_yards"], sharp_edge_bets_enabled=enabled)
    return row, offer, metrics, distribution


def test_prices_fanduel_at_the_sharp_no_vig_line_and_takes_the_off_market_side():
    row, offer, metrics, distribution = setup(sharp_over=-140, sharp_under=+120)  # sharp says over ~57%
    cand = props._candidate_from_offer(row, "receiving_yards", 30, 50, metrics, offer, distribution)
    sharp_over = (140 / 240) / (140 / 240 + 100 / 220)
    assert cand["side"] == "over" and cand["probability"] == pytest.approx(sharp_over)
    assert cand["sharp_book"] == "pinnacle" and cand["sharp_over_probability"] == pytest.approx(sharp_over)
    assert cand["sharp_ev"] == pytest.approx(cand["ev"]) and cand["ev"] > 0.03  # FanDuel -110 is off-market
    assert cand["market_no_vig_probability"] == pytest.approx(0.5)


def test_sharp_edges_stay_research_until_enabled_and_replay_exactly():
    row, offer, metrics, distribution = setup(sharp_over=+130, sharp_under=-150)  # sharp says under
    cand = props._candidate_from_offer(row, "receiving_yards", 80, 50, metrics, offer, distribution)
    assert cand["side"] == "under" and cand["tier"] == "paper"
    captured = capture(row, "receiving_yards", 80, 50, metrics, offer, distribution, {}, {}, .02, {"version": "frozen"})
    restored = replay(json.loads(json.dumps(captured, default=str)))
    assert restored["probability"] == cand["probability"] and restored["side"] == cand["side"]


def test_without_a_sharp_quote_scoring_is_unchanged():
    row, offer, metrics, distribution = setup(sharp_over=-140, sharp_under=+120)
    offer.pop("sharp_reference")
    cand = props._candidate_from_offer(row, "receiving_yards", 60, 50, metrics, offer, distribution)
    assert cand["sharp_book"] is None and cand["sharp_ev"] is None
    assert cand["probability"] == pytest.approx(cand["market_no_vig_probability"])  # trust 0 -> FanDuel no-vig


def test_reference_index_matches_exact_line_fresh_two_sided_sharp_quotes_only():
    start = NOW + timedelta(hours=1)
    base = dict(player_name_norm="dk metcalf", stat="receiving_yards", over_price=-110, under_price=-110, commence_time_utc=start)
    offers = [dict(base, bookmaker_key="pinnacle", line=45.5, fetched_at_utc=NOW - timedelta(minutes=10)),
              dict(base, bookmaker_key="betfair_ex_eu", line=45.5, fetched_at_utc=NOW - timedelta(minutes=5)),
              dict(base, bookmaker_key="pinnacle", line=44.5, fetched_at_utc=NOW - timedelta(hours=5)),  # too old
              dict(base, bookmaker_key="draftkings", line=44.5, fetched_at_utc=NOW)]                     # not sharp
    index = props._sharp_reference_index(offers, NOW)
    fd = dict(base, bookmaker_key="fanduel", line=45.5)
    assert props._sharp_reference_for(index, fd)["bookmaker_key"] == "pinnacle"  # preferred over exchange
    assert props._sharp_reference_for(index, dict(fd, line=44.5)) is None


def test_discord_lists_sharp_edges_in_their_own_research_section():
    import json as _json
    from nfl_pipeline.discord_matchups import SHARP_HEADING, build_bundle
    from nfl_pipeline.test_discord_matchups import prop, schedule, NOW as CARD_NOW, DAY
    edge = prop(player='EdgeGuy', sharp_ev=0.06, sharp_book='pinnacle', sharp_line=45.5, sharp_over_probability=0.58, side='over')
    plain = prop(player='PlainGuy')
    text = _json.dumps(build_bundle(DAY, schedule(), [], [edge, plain], ['frozen'], now=CARD_NOW))
    assert SHARP_HEADING in text and 'vs Pinnacle 45.5: fair 58%, EV +6.0%' in text
    assert text.count('EdgeGuy') >= 1 and 'PlainGuy' in text
