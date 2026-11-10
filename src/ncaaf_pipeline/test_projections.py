from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ncaaf_pipeline import projections


@pytest.mark.parametrize("day,season", [(date(2027, 1, 2), 2026), (date(2026, 12, 28), 2026), (date(2026, 10, 28), 2026)])
def test_refresh_uses_football_season(monkeypatch, day, season):
    fetch = MagicMock()
    monkeypatch.setattr(projections.ingest, "ingest", fetch)
    monkeypatch.setattr(projections.psycopg2, "connect", MagicMock())
    monkeypatch.setattr(projections.ratings, "load", lambda conn: pd.DataFrame())
    monkeypatch.setattr(projections, "upcoming", lambda conn, day: pd.DataFrame())
    projections.run(day, post=False)
    assert fetch.call_args.args == ([season],)
    assert "postseason" in fetch.call_args.kwargs["season_types"] if day.month in (1, 12) else True
