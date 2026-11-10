"""Explicit game scope shared by pregame subprocesses; never inferred from weekday."""
import json
import os
import re

ENV = 'NFL_RUN_GAME_IDS'


def validate_ids(values):
    if not isinstance(values, list) or not values or any(
            not isinstance(v, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', v) for v in values):
        raise ValueError('NFL game scope must be a nonempty list of game IDs')
    return sorted(set(values))


def game_ids():
    value = os.getenv(ENV)
    return validate_ids(json.loads(value)) if value is not None else None


def filter_frame(frame):
    ids = game_ids()
    if ids is None or frame.empty:
        return frame
    return frame.loc[frame.game_id.astype(str).isin(ids)].copy()


def filter_records(records):
    ids = game_ids()
    return records if ids is None else [r for r in records if str(r['game_id']) in ids]
