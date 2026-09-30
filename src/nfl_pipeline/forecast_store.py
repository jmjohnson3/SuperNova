"""Append-only forecast revisions; current-card state is not betting history."""
from datetime import datetime, timezone
import json
import math

import psycopg2.extras

from nfl_pipeline.integrity import FEATURE_CONTRACT, decision_key
from nfl_pipeline.offer_selection import CONTRACT as EXECUTION_CONTRACT, forecast_quote_error
from nfl_pipeline.game_scope import game_ids


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_forecasts(conn, table, rows, fields):
    if table not in {"nfl_player_prop_predictions", "nfl_game_predictions"}:
        raise ValueError("Unknown NFL forecast table")
    if not rows:
        return 0
    scope = game_ids()
    if scope is not None and any(str(r.get('game_id')) not in scope for r in rows):
        raise ValueError('Refusing to persist forecasts outside the requested game scope')
    now = datetime.now(timezone.utc)
    for row in rows:
        if row.get('execution_contract') == EXECUTION_CONTRACT:
            error = forecast_quote_error(row, now)
            if error:
                raise ValueError('Refusing to lock NFL forecast: '+error)
    with conn.cursor() as cur:
        dates = sorted({r["game_date_et"] for r in rows})
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("nfl_forecast:" + table,))
        cur.execute("SELECT game_id FROM raw.nfl_games WHERE start_ts_utc > %s", (now,))
        upcoming = {r[0] for r in cur.fetchall()}
        valid = [r for r in rows if r.get("game_id") in upcoming]
        if len(valid) != len(rows):
            raise RuntimeError("Refusing to persist an NFL pregame forecast after kickoff or without a known start")
        if scope is None:
            cur.execute(f"UPDATE bets.{table} SET is_current=FALSE WHERE game_date_et=ANY(%s) AND is_current", (dates,))
        else:
            cur.execute(f"UPDATE bets.{table} SET is_current=FALSE WHERE game_date_et=ANY(%s) AND game_id=ANY(%s) AND is_current", (dates, scope))
        extra = ["created_at_utc", "is_current", "integrity_version", "forecast_payload"]
        tuples = []
        for row in valid:
            row.update(clean(row))
            if table == 'nfl_player_prop_predictions':
                from nfl_pipeline.forecast_outputs import for_lock
                row['forecast_outputs'] = for_lock(row)
            row["prediction_key"] = decision_key(row, at=now)
            payload = json.loads(json.dumps(row, default=str, allow_nan=False))
            tuples.append(tuple(row.get(f) for f in fields) + (now, True, FEATURE_CONTRACT, psycopg2.extras.Json(payload)))
        psycopg2.extras.execute_values(cur, f"INSERT INTO bets.{table} ({','.join(fields + extra)}) VALUES %s ON CONFLICT DO NOTHING", tuples)
    conn.commit()
    return len(valid)
