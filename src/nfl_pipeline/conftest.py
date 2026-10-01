import pytest


@pytest.fixture(autouse=True)
def _no_live_fanduel_state(monkeypatch):
    """Tests never reach FanDuel's state feed, even where NFL_FANDUEL_STATE is set on the machine."""
    monkeypatch.delenv("NFL_FANDUEL_STATE", raising=False)
