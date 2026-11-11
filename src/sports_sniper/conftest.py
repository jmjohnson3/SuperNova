"""Shared test setup: no test reads the real game schedule or stored projections, and nothing the dashboard
cached in one test is served to the next."""
import pytest
import streamlit as st

from sports_sniper import model_compare, nba_candidate
from sports_sniper.providers import schedule


@pytest.fixture(autouse=True)
def no_real_schedule(monkeypatch):
    monkeypatch.setattr(schedule, "upcoming", lambda dsn, now, hours=18.0, connect=None: [])
    monkeypatch.setattr(model_compare, "CANDIDATE", None)            # no stored projections are read unless a test supplies them
    monkeypatch.setattr(nba_candidate, "diagnostics", lambda dsn, days, connect=None: {})
    st.cache_data.clear()
