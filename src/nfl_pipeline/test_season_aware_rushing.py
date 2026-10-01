import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling import predict_player_props as props


def games(player, team, season, weeks, yards, carries, pos="RB"):
    return [dict(player_id=player, team_abbr=team, season=season, week=w, game_id=f"{season}_{w:02d}_{team}",
                 season_type="REG", position=pos, rushing_yards=yards, carries=carries) for w in weeks]


def snap(*rows):
    return pd.DataFrame([dict(player_id=p, team_abbr=t, season=2026, position=pos) for p, t, pos in rows])


def test_cold_start_player_is_left_to_the_model():
    history = pd.DataFrame(games("vet", "AAA", 2025, range(1, 18), 60.0, 15.0))
    out = props._season_aware_rushing_estimate(history, snap(("rookie", "AAA", "RB")))
    assert np.isnan(out[0])


def test_week_one_uses_full_prior_season_not_last_five_games():
    # Strong first half, collapsed role late last season: last-5 says 10, the season said 45.
    history = pd.DataFrame(games("rb", "AAA", 2025, range(1, 13), 60.0, 15.0) + games("rb", "AAA", 2025, range(13, 18), 10.0, 3.0))
    out = props._season_aware_rushing_estimate(history, snap(("rb", "AAA", "RB")))
    assert out[0] == pytest.approx((12 * 60 + 5 * 10) / 17)


def test_current_season_role_outweighs_stale_prior_and_team_change_discounts_it():
    prior = games("rb", "OLD", 2025, range(1, 18), 90.0, 20.0)
    now = games("rb", "NEW", 2026, (1, 2, 3), 20.0, 5.0) + games("lead", "NEW", 2026, (1, 2, 3), 80.0, 20.0)
    history = pd.DataFrame(prior + now)
    moved = props._season_aware_rushing_estimate(history, snap(("rb", "NEW", "RB")))[0]
    stayed = props._season_aware_rushing_estimate(history.assign(team_abbr=history.team_abbr.replace("OLD", "NEW")),
                                                  snap(("rb", "NEW", "RB")))[0]
    assert 20.0 < moved < stayed < 90.0  # team change trusts the 90-yard prior less
