"""Explain frozen receiving-paper close gaps without changing evidence or rules.

Reads saved quotes and original provider responses only; makes no API requests.
python -m nfl_pipeline.receiving_close_audit --date 2026-10-29
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras

from nfl_pipeline import experiment_registry
from nfl_pipeline.context_contract import number, safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.markets import normalize_name
from nfl_pipeline.modeling import receiving_paper_trial as trial
from nfl_pipeline.sharp_math import implied

ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ('provider', 'event_id', 'bookmaker_key', 'player_name_norm', 'stat', 'market_key')


def raw_sgo_evidence(row, snapshots, max_age_minutes=20):
    """Retain unavailable exact quotes as diagnostic facts, never valid closes."""
    from nfl_pipeline.crawler_oddsapi import _sgo_odd_parts, _sgo_player_lookup, _sgo_market_key
    if row['quote']['provider'] != 'sportsgameodds':
        return []
    start = safe_time(row['start_ts_utc'])
    cutoff = max(safe_time(row['captured_at']), start-timedelta(minutes=max_age_minutes))
    found = []
    for snapshot in snapshots:
        fetched = safe_time(snapshot['fetched_at_utc'])
        if not fetched or not cutoff <= fetched < start:
            continue
        for event in snapshot['payload'].get('data', []):
            if str(event.get('eventID')) != str(row['quote']['event_id']):
                continue
            names = _sgo_player_lookup(event)
            for key, odd in (event.get('odds') or {}).items():
                if not isinstance(odd, dict):
                    continue
                stat, entity, period, kind, side = _sgo_odd_parts(odd.get('oddID') or key)
                name = odd.get('playerName') or odd.get('statEntityName') or names.get(str(entity), '')
                if (period != 'game' or kind != 'ou' or side not in {'over', 'under'}
                        or _sgo_market_key(stat) != row['quote']['market_key']
                        or normalize_name(name) != row['quote']['player_name_norm']):
                    continue
                book = (odd.get('byBookmaker') or {}).get(row['book']) or {}
                quotes = [(book, False)] + [(q, True) for q in book.get('altLines', []) or []]
                for quote, alternate in quotes:
                    if not isinstance(quote, dict) or number(quote.get('overUnder')) != row['line']:
                        continue
                    found.append(dict(raw_response_id=snapshot['id'], payload_sha256=snapshot['payload_sha256'],
                        fetched_at=fetched, side=side, line=row['line'], alternate=alternate,
                        available=quote.get('available'), price=number(quote.get('odds')),
                        source_updated_at=quote.get('lastUpdatedAt'),
                        event_started=odd.get('started'), event_ended=odd.get('ended')))
    return clean(found)


def diagnose(row, quotes, snapshots, now, max_age_minutes=20):
    original = row['quote']
    start = safe_time(row['start_ts_utc'])
    cutoff = max(safe_time(row['captured_at']), start-timedelta(minutes=max_age_minutes))
    same = [q for q in quotes if all(str(q.get(k)) == str(original.get(k)) for k in IDENTITY)
            and safe_time(q.get('commence_time_utc')) == start and q.get('snapshot_role') == 'close']
    window = [q for q in same if safe_time(q.get('fetched_at_utc'))
              and cutoff <= safe_time(q['fetched_at_utc']) < start]
    exact = [q for q in window if number(q.get('line')) == row['line']]
    verdict = trial.exact_close(row, quotes, now, max_age_minutes)
    side = row['side'] + '_price'
    reason = ('awaiting_kickoff' if now < start else 'valid_exact_close' if verdict['valid']
              else 'selected_side_price_missing' if exact and not any(implied(q.get(side)) for q in exact)
              else 'opposing_price_or_pair_invalid' if exact
              else 'exact_line_not_observed' if window
              else 'no_matching_market_in_close_window')
    raw = raw_sgo_evidence(row, snapshots, max_age_minutes)
    selected_raw = [q for q in raw if q['side'] == row['side']]
    unavailable = bool(selected_raw) and all(q['available'] is False for q in selected_raw)
    return clean(dict(decision_id=row['decision_id'], forecast_id=row['forecast_id'],
        player=row['player_name'], side=row['side'], line=row['line'], captured_at=row['captured_at'],
        kickoff=start, valid=verdict['valid'], reason=reason, frozen_verdict=verdict['reason'],
        snapshot_times=sorted({safe_time(q['fetched_at_utc']).isoformat() for q in window}),
        observed_lines=sorted({float(q['line']) for q in window}),
        paired_lines=sorted({float(q['line']) for q in window
                            if implied(q.get('over_price')) and implied(q.get('under_price'))}),
        exact_quote_rows=[{k:q.get(k) for k in ('id','line','over_price','under_price','fetched_at_utc')} for q in exact],
        raw_exact_quotes=raw, provider_marks_selected_side_unavailable=unavailable,
        live_sportsbook_availability='unknown'))


def report(day):
    manifest = experiment_registry.check_experiment(trial.NAME)
    trial._check_candidate(manifest)
    selected = [r for d in trial.captures(manifest) for r in d['rows'] if r['day'] == str(day) and r['selected']]
    limit = manifest['config']['review']['max_close_age_minutes']
    now = datetime.now(timezone.utc)
    audited = []
    with psycopg2.connect(PG_DSN, connect_timeout=10) as conn:
        conn.set_session(readonly=True, isolation_level='REPEATABLE READ')
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET LOCAL statement_timeout='30s'")
            snapshots = []
            if selected:
                cur.execute("""SELECT id,fetched_at_utc,payload,payload_sha256 FROM raw.nfl_api_responses
                    WHERE as_of_date=%s AND provider='sportsgameodds'
                    AND endpoint='nfl_sportsgameodds_events_raw' AND snapshot_role='close'
                    AND fetched_at_utc >= %s AND fetched_at_utc < %s ORDER BY fetched_at_utc""",
                    (day, min(safe_time(r['start_ts_utc']) for r in selected)-timedelta(minutes=limit),
                     max(safe_time(r['start_ts_utc']) for r in selected)))
                snapshots = [dict(r) for r in cur.fetchall()]
            for row in selected:
                cur.execute("""SELECT id,provider,event_id,bookmaker_key,player_name_norm,stat,market_key,
                    line::float,over_price,under_price,commence_time_utc,fetched_at_utc,snapshot_role
                    FROM odds.nfl_player_prop_lines WHERE as_of_date=%s AND bookmaker_key=%s
                    AND player_name_norm=%s AND stat=%s ORDER BY fetched_at_utc,id""",
                    (day,row['book'],row['quote']['player_name_norm'],row['stat']))
                audited.append(diagnose(row, [dict(q) for q in cur.fetchall()], snapshots, now, limit))
    matured = [r for r in audited if safe_time(r['kickoff']) <= now]
    doc = clean(dict(day=day,built_at=now,registration_sha256=manifest['sha256'],
        selected=len(audited),matured=len(matured),valid=sum(r['valid'] for r in matured),
        reasons=dict(Counter(r['reason'] for r in matured)),rows=audited,
        frozen_rules_changed=False,evidence_repaired=False,api_requests=0,betting_authorized=False,
        note='Diagnostic only. Inactive quotes, one-sided markets and other lines do not repair exact-close evidence. '
             'Provider availability flags are not a live sportsbook check.'))
    lines = [f'# Receiving paper close audit: {day}', '', doc['note'], '',
             f"Valid frozen closes: {doc['valid']}/{len(matured)} matured selections.", '',
             '| Player | Selected line | Finding | Raw provider marks selected side unavailable |',
             '|---|---|---|---|']
    for r in audited:
        lines.append(f"| {r['player']} | {r['side']} {r['line']} | {r['reason']} | {r['provider_marks_selected_side_unavailable']} |")
    for r in audited:
        lines.extend(['', f"Forecast {r['forecast_id']}: paired lines {r['paired_lines']}; close observations {r['snapshot_times']}.", ''])
    lines += ['Original raw response IDs, hashes, prices, availability flags and source update times are in the JSON.',
              'No stored quote, decision, registration, close evidence or wager was changed.']
    for suffix in (str(day),'latest'):
        atomic_json(ROOT/'reports'/f'nfl_receiving_close_audit_{suffix}.json',doc)
        (ROOT/'reports'/f'nfl_receiving_close_audit_{suffix}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date',type=date.fromisoformat,required=True)
    result = report(parser.parse_args().date)
    print({k:result[k] for k in ('day','selected','matured','valid','reasons','evidence_repaired')})
