"""Shared test setup: the dashboard never reads the real game schedule during tests, and nothing it cached in
one test is served to the next."""
import pytest
import streamlit as st

from sports_sniper.providers import schedule


@pytest.fixture(autouse=True)
def no_real_schedule(monkeypatch):
    monkeypatch.setattr(schedule, "upcoming", lambda dsn, now, hours=18.0, connect=None: [])
    st.cache_data.clear()
