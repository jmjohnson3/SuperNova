"""Refresh and verify actual lock-date FanDuel quotes immediately before scoring."""
import argparse
from datetime import date, datetime, timezone
import json

import psycopg2

from nfl_pipeline.crawler_oddsapi import OddsCrawlerConfig, fetch_for_date
from nfl_pipeline.parse_oddsapi import ParseConfig, parse_props
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.offer_selection import MAX_QUOTE_AGE_MINUTES
from nfl_pipeline.modeling.live_scoring_replay import ROOT
from nfl_pipeline.game_scope import game_ids
from nfl_pipeline.markets import normalize_team
from nfl_pipeline.context_contract import safe_time


def upcoming_games(day):
    with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout='30s'")
        scope = game_ids()
        cur.execute("SELECT count(*) FROM raw.nfl_games WHERE game_date_et=%s AND start_ts_utc>NOW() AND (%s IS NULL OR game_id=ANY(%s))", (day,scope,scope))
        return cur.fetchone()[0]


def quote_health(day):
    with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout='30s'")
        result = {}
        scope = game_ids()
        identities = None
        if scope is not None:
            cur.execute("SELECT home_team_abbr,away_team_abbr,start_ts_utc FROM raw.nfl_games WHERE game_date_et=%s AND game_id=ANY(%s) AND start_ts_utc>NOW()", (day,scope))
            identities = {(normalize_team(h),normalize_team(a),safe_time(t)) for h,a,t in cur.fetchall()}
        for kind, table in (('props', 'nfl_player_prop_lines'), ('games', 'nfl_game_lines')):
            if identities is not None:
                cur.execute(f"""SELECT home_team,away_team,commence_time_utc,fetched_at_utc
                    FROM odds.{table} WHERE as_of_date=%s AND bookmaker_key='fanduel'
                    AND snapshot_role IN ('open','lock','live')
                    AND fetched_at_utc<=NOW() AND commence_time_utc>NOW()""",(day,))
                quotes = [safe_time(f) for h,a,t,f in cur.fetchall() if (normalize_team(h),normalize_team(a),safe_time(t)) in identities]
                now = datetime.now(timezone.utc)
                result[kind] = dict(observed_rows=len(quotes),fresh_rows=sum(
                    0 <= (now-q).total_seconds() <= MAX_QUOTE_AGE_MINUTES*60 for q in quotes),
                    latest_quote=max(quotes).isoformat() if quotes else None)
                continue
            cur.execute(f"""SELECT count(*),count(*) FILTER (WHERE fetched_at_utc >= NOW()-%s*interval '1 minute'),
                           max(fetched_at_utc) FROM odds.{table}
                           WHERE as_of_date=%s AND bookmaker_key='fanduel'
                             AND snapshot_role IN ('open','lock','live')
                             AND fetched_at_utc<=NOW() AND commence_time_utc>NOW()""", (MAX_QUOTE_AGE_MINUTES, day))
            total, fresh, latest = cur.fetchone()
            result[kind] = dict(observed_rows=total, fresh_rows=fresh, latest_quote=str(latest) if latest else None)
        return result


def refresh(day):
    at = datetime.now(timezone.utc)
    count = upcoming_games(day)
    if not count:
        return dict(status='no_upcoming_games', day=str(day), games=0, api_called=False,
                    built_at=at.isoformat(), ready_for_scoring=False)
    result = fetch_for_date(OddsCrawlerConfig(snapshot_role='lock'), day)
    parse_props(ParseConfig(as_of_date=day))
    health = quote_health(day)
    stale = [k for k, h in health.items() if h['observed_rows'] and not h['fresh_rows']]
    fresh = sum(h['fresh_rows'] for h in health.values())
    status = 'fresh_quotes_ready' if fresh and not stale else 'partial_fresh_quotes' if fresh else 'no_fresh_quotes'
    return dict(status=status, day=str(day), built_at=at.isoformat(), games=count, api_called=True,
        provider_status=result.get('status'), provider=result.get('provider'), quote_health=health, game_ids=game_ids(),
        stale_kinds=stale, max_quote_age_minutes=MAX_QUOTE_AGE_MINUTES,
        ready_for_scoring=bool(fresh),
        note='Only fresh exact offers may be locked; missing props remain projection-only. No timestamp is rewritten.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', type=date.fromisoformat, required=True)
    result = refresh(parser.parse_args().date)
    atomic_json(ROOT/'reports'/'nfl_lock_quote_health_latest.json', result)
    print(json.dumps(result, indent=2))
    if result['status'] == 'no_fresh_quotes':
        raise SystemExit(1)
