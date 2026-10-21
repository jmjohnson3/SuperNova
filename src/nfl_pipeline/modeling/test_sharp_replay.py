from datetime import datetime, timedelta, timezone

from nfl_pipeline.modeling import sharp_replay as r

KICK = datetime(2026, 10, 11, 17, tzinfo=timezone.utc)


def snapshot(minutes_before, line, over, under):
    at = KICK - timedelta(minutes=minutes_before)
    stamp = at.isoformat()
    outcomes = [dict(name="Over", description="Player", point=line, price=over),
                dict(name="Under", description="Player", point=line, price=under)]
    return at, {"bookmakers": [{"key": "pinnacle", "last_update": stamp,
                                "markets": [{"key": "player_reception_yds", "last_update": stamp, "outcomes": outcomes}]}]}


def decision(**kw):
    return dict(dict(stat="receiving_yards", player_norm="player", line=45.5, side="over", price=-110,
                     kickoff=KICK, decided_at=KICK - timedelta(minutes=30)), **kw)


def test_sharp_close_is_the_last_fresh_quote_and_moves_a_changed_line():
    snaps = [snapshot(25, 45.5, -150, 130), snapshot(5, 45.5, -130, 110)]
    ev, method = r.sharp_close(decision(), snaps)
    assert method == "exact" and ev > 0
    moved, method = r.sharp_close(decision(), [snapshot(5, 47.5, -110, -110)])
    assert method == "shifted" and moved > 0  # Pinnacle moved up two yards: the 45.5 over got better
    assert r.sharp_close(decision(), [snapshot(25, 45.5, -150, 130)]) == (None, None)  # too old to be a close
    assert r.sharp_close(decision(stat="receptions", line=4.5),
                         [snapshot(5, 5.5, -110, -110)]) == (None, None)  # counts are never moved


def test_payload_matches_the_game_when_the_api_moves_kickoff():
    games = {("JAX", "PHI", KICK): dict(game_id="2026_05_PHI_JAX")}
    payload = dict(home_team="Jacksonville Jaguars", away_team="Philadelphia Eagles",
                   commence_time=(KICK + timedelta(minutes=5)).isoformat())
    assert r.match_game(payload, games)["game_id"] == "2026_05_PHI_JAX"
    assert r.match_game(dict(payload, commence_time=(KICK + timedelta(hours=3)).isoformat()), games) is None
