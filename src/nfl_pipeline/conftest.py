import pytest


@pytest.fixture(autouse=True)
def _no_live_fanduel_state(monkeypatch):
    """Tests never reach FanDuel's state feed, even where NFL_FANDUEL_STATE is set on the machine."""
    monkeypatch.delenv("NFL_FANDUEL_STATE", raising=False)


@pytest.fixture(autouse=True)
def _no_live_discord(monkeypatch):
    """Tests never post to the real channel. The webhook is also read from the saved Windows
    environment, so clearing os.environ alone is not enough; a send raises instead."""
    for name in ("NFL_DISCORD_WEBHOOK_URL", "MLB_DISCORD_WEBHOOK_URL", "DISCORD_WEBHOOK_URL"):
        monkeypatch.delenv(name, raising=False)
    from nfl_pipeline import sharp_delivery
    monkeypatch.setattr(sharp_delivery, "_webhook_url", lambda: "")
