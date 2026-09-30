"""One bounded, read-only SGO alternate-line probe. Never creates CLV evidence."""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json

import psycopg2
import requests

from nfl_pipeline.crawler_oddsapi import (
    OddsCrawlerConfig, _canonical_events_from_sgo_payload, _et_day_window_utc,
    _to_z, _sgo_odd_parts, _sgo_market_key,
)
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.markets import normalize_name, SPEC_BY_STAT
from nfl_pipeline.modeling.live_scoring_replay import ROOT


def summarize(payload):
    counts = Counter(); groups = Counter()
    for event in payload.get('data', []):
        for odd_id, odd in (event.get('odds') or {}).items():
            stat, _, period, kind, side = _sgo_odd_parts(odd_id)
            if period != 'game' or kind != 'ou' or not _sgo_market_key(stat):
                continue
            for book, quote in (odd.get('byBookmaker') or {}).items():
                if book not in ('fanduel', 'draftkings'):
                    continue
                counts['main_side_quotes'] += 1
                for alt in quote.get('altLines', []) or []:
                    status = 'available' if alt.get('available') is True else 'inactive_or_unknown'
                    counts['alternate_'+status] += 1
                    groups['|'.join((book, stat, side, status))] += 1
    canonical = _canonical_events_from_sgo_payload(payload, include_alts=True)
    normalized = []
    for event in canonical:
        for book in event['bookmakers']:
            for market in book['markets']:
                if market['key'] in ('spreads', 'totals'):
                    continue
                for outcome in market['outcomes']:
                    normalized.append(dict(event_id=event['id'], book=book['key'], market=market['key'],
                        player=normalize_name(outcome['description']), side=outcome['name'].lower(),
                        line=outcome['point'], alternate=outcome['is_alt_line']))
    return dict(counts=dict(counts), by_book_market_side=dict(groups),
                normalized_player_side_quotes=len(normalized),
                normalized_alternate_side_quotes=sum(r['alternate'] for r in normalized)), normalized


def probe(day):
    cfg = OddsCrawlerConfig()
    result = dict(built_at=datetime.now(timezone.utc).isoformat(), game_date=str(day),
        provider='sportsgameodds', max_api_requests=1, valid_clv_evidence=False,
        production_changed=False, close_coverage_target=.90,
        source='https://sportsgameodds.com/docs/info/v1-to-v2',
        limitation='A current probe cannot repair a missed historical close or prove all moved lines stay published.')
    if not cfg.sports_game_odds_key:
        return dict(result, status='missing_provider_key')
    start, end = _et_day_window_utc(day)
    params = dict(leagueID='NFL', bookmakerID='fanduel,draftkings', oddsAvailable='true',
        started='false', includeAltLines='true', includeOpposingOdds='true',
        startsAfter=_to_z(start), startsBefore=_to_z(end), limit=1)
    try:
        response = requests.get('https://api.sportsgameodds.com/v2/events', params=params,
            headers={'x-api-key': cfg.sports_game_odds_key}, timeout=cfg.timeout_s)
    except requests.RequestException as exc:
        return dict(result, status='provider_request_failed', error_type=type(exc).__name__)
    if not response.ok:
        return dict(result, status='provider_rejected', http_status=response.status_code)
    payload = response.json()
    summary, normalized = summarize(payload)
    result.update(summary, events=len(payload.get('data', [])),
                  status='alternate_quotes_observed' if summary['normalized_alternate_side_quotes'] else 'no_available_alternates_in_probe')
    # Match original offered decisions for this probed event only; never write
    # these pre-window observations into the actual close/CLV tables.
    event_ids = [r['eventID'] for r in payload.get('data', []) if r.get('eventID')]
    with psycopg2.connect(cfg.pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout='30s'")
        cur.execute("""SELECT DISTINCT p.id,o.event_id,o.bookmaker_key,o.player_name_norm,
                       o.stat,p.side,o.line::float FROM bets.nfl_player_prop_predictions p
                       JOIN odds.nfl_player_prop_lines o ON o.id=p.offer_id
                       WHERE p.game_date_et=%s AND o.provider='sportsgameodds'
                         AND o.event_id=ANY(%s) AND p.side IN ('over','under')""", (day, event_ids))
        locks = cur.fetchall()
    quotes = {(r['event_id'],r['book'],r['player'],r['market'],r['side'],r['line']): r for r in normalized}
    matched = []; missing = []
    for ident, event, book, player, stat, side, line in locks:
        market = SPEC_BY_STAT[stat].market_keys[0] if stat in SPEC_BY_STAT else None
        quote = quotes.get((event,book,player,market,side,line))
        (matched if quote else missing).append(ident)
    result.update(original_lock_rows_for_probed_events=len(locks), exact_current_matches=len(matched),
                  missing_current_forecast_ids=missing)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True, type=date.fromisoformat)
    result = probe(parser.parse_args().date)
    atomic_json(ROOT/'reports'/'nfl_alt_close_provider_probe_latest.json', result)
    print(json.dumps(result, indent=2))
