"""Shared test setup: no test reads the real game schedule or stored projections, and nothing the dashboard
cached in one test is served to the next."""
import pytest
import streamlit as st

import pandas as pd

from sports_sniper import model_compare, nba_candidate, parlay_legs, predictions
from sports_sniper.providers import schedule


@pytest.fixture(autouse=True)
def no_real_schedule(monkeypatch):
    monkeypatch.setattr(schedule, "upcoming", lambda dsn, now, hours=18.0, connect=None: [])
    monkeypatch.setattr(model_compare, "CANDIDATE", None)            # no stored projections are read unless a test supplies them
    monkeypatch.setattr(nba_candidate, "diagnostics", lambda dsn, days, connect=None: {})
    empty_games = predictions.with_gaps(pd.DataFrame(columns=predictions.GAME_COLUMNS))
    monkeypatch.setattr(predictions, "games", lambda dsn, now, connect=None, names=None: empty_games)          # no stored forecasts either
    monkeypatch.setattr(predictions, "props", lambda dsn, now, prices=None, connect=None: pd.DataFrame(columns=predictions.PROP_COLUMNS + ["gap"]))
    monkeypatch.setattr(predictions, "ncaaf_record", lambda dsn, connect=None: None)
    monkeypatch.setattr(parlay_legs, "load_recent", lambda dsn, today, connect=None: {})                 # nor recent game logs
    st.cache_data.clear()
