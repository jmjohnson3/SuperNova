import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

from nba_props import ingest, model


def test_names_match_across_books_and_stats_nba():
    assert model.match_name("Jaren Jackson Jr.") == model.match_name("Jaren Jackson Jr") == "jaren jackson"
    assert model.match_name("Nikola Jokić") == "nikola jokic"
    assert model.match_name("Shai Gilgeous-Alexander") == "shai gilgeousalexander"


def test_p_over_is_poisson_at_unit_dispersion_and_wider_with_overdispersion():
    mu, line = np.array([20.0]), np.array([22.5])
    assert model.p_over(mu, line, 1.0, 0.0)[0] == pytest.approx(st.poisson.sf(22, 20))
    wide = model.p_over(mu, line, 1.0, 0.05)[0]
    assert wide > model.p_over(mu, line, 1.0, 0.0)[0]          # more mass in the tails above the line
    assert model.p_over(np.array([30.0]), line, 1.0, 0.05)[0] > 0.5


def test_dispersion_recovers_a_negative_binomial():
    rng = np.random.default_rng(0)
    mu = rng.uniform(2, 30, 40000)
    r = 1 / 0.05
    y = rng.negative_binomial(r, r / (r + mu))
    a, b = model.dispersion(y.astype(float), mu)
    assert a == pytest.approx(1.0, abs=0.25) and b == pytest.approx(0.05, abs=0.01)


def test_ingest_rows_split_matchups_into_team_opponent_and_home():
    df = pd.DataFrame(dict(GAME_ID=["0022600001", "0022600001"], PLAYER_ID=[1, 2], GAME_DATE=["2026-10-20T00:00:00"] * 2,
                           PLAYER_NAME=["A", "B"], TEAM_ABBREVIATION=["BOS", "NYK"], MATCHUP=["BOS vs. NYK", "NYK @ BOS"],
                           MIN=[30.5, 20.0], PTS=[20, 10], REB=[5, 4], AST=[3, 2], FG3M=[2, 1], FGA=[15, 8], FTA=[4, 2],
                           TOV=[2, 1], OREB=[1, 0], STL=[1, 0], BLK=[0, 1], PLUS_MINUS=[5.0, -5.0]))
    rows = ingest.to_rows(df, "2026-27", "Regular Season")
    cols = dict(zip(ingest.COLUMNS, rows[1]))
    assert (cols["team"], cols["opponent"], cols["home"]) == ("NYK", "BOS", False)
    assert ingest.season_label(2026) == "2026-27"
    assert ingest.current_start_year(pd.Timestamp("2027-03-01").date()) == 2026
