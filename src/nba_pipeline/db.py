"""Central database DSN for the NBA pipeline."""
from __future__ import annotations

from supernovabets_config import pg_dsn

PG_DSN: str = pg_dsn()
