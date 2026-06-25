"""Central database DSN for the MLB pipeline.

All mlb_pipeline modules import PG_DSN from here instead of hardcoding
the connection string.  Override for a different database with the
PG_DSN environment variable:

    PG_DSN=postgresql://user:pass@host:5432/db python -m mlb_pipeline.run_daily
"""
from __future__ import annotations

import os

PG_DSN: str = os.getenv("PG_DSN", "postgresql://josh:password@localhost:5432/nba")
