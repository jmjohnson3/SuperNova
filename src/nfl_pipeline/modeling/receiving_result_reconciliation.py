"""Account for missing receiving results; no inferred zeros, voids or winners."""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

import psycopg2
import psycopg2.extras

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import MODEL_ROOT, atomic_json


ROOT = Path(__file__).resolve().parents[3]
REGISTRATION = MODEL_ROOT / 'receiving_research_trial' / 'registration.json'


def load_registration_config():
    doc = json.loads(REGISTRATION.read_text(encoding='utf-8'))
    encoded = json.dumps(doc['config'], sort_keys=True, separators=(',', ':')).encode()
    if doc.get('sha256') != hashlib.sha256(encoded).hexdigest():
        raise ValueError('Research trial registration checksum mismatch')
    return doc['config']


def eligible_error(row, config):
    """Lightweight reconciliation eligibility for already-audited v2 locks.

    This deliberately checks immutable lock and exact normalized offer columns,
    not the large stored scoring payload.  It never replays or reconstructs a
    probability; the full lock/context audit remains owned by the live scorer.
    """
    p = row.get('forecast_payload') or {}
    q = (p.get('scoring_replay') or {}).get('offer') or {}
    if row.get('stat') != config['stat'] or row.get('book') != config['book']:
        return 'outside_market'
    if row.get('model_version') != config['production_release']:
        return 'different_production_release'
    lock, start = safe_time(row.get('created_at_utc')), safe_time(row.get('start_ts_utc'))
    fetched = safe_time(row.get('offer_fetched_at'))
    if not lock or not start or lock >= start:
        return 'lock_not_before_start'
    if not fetched or fetched > lock:
        return 'invalid_lock_offer_timing'
    if lock < safe_time(config['registered_at']):
        return 'before_registration'
    if q.get('market_key') not in config['market_keys']:
        return 'not_common_line'
    try:
        if (
            not q.get('event_id')
            or str(q.get('offer_id')) != str(row.get('offer_id'))
            or q.get('bookmaker_key') != row.get('book')
            or q.get('stat') != row.get('stat')
            or q.get('player_name_norm') != row.get('offer_player_name_norm')
            or float(q.get('line')) != float(row.get('line'))
            or row.get('side') not in ('over', 'under')
            or safe_time(q.get('commence_time_utc')) != start
            or abs(float(q.get('over_price'))) < 100
            or abs(float(q.get('under_price'))) < 100
            or float(q.get(row['side'] + '_price')) != float(row.get('price'))
        ):
            return 'captured_quote_identity_mismatch'
    except (TypeError, ValueError):
        return 'missing_or_invalid_paired_price'
    if (lock - fetched).total_seconds() > config['rules']['max_quote_age_minutes'] * 60:
        return 'stale_original_quote'
    return None


def classify(row, source):
    if row.get('status') != 'final':
        return 'pending_game'
    if row.get('graded_result')=='void_nonparticipant':
        return 'void_nonparticipant'
    if row.get('actual') is not None:
        return 'settled_verified_participation'
    if row.get('result_game_id') is None:
        return 'missing_exact_player_game_result'
    participated = (row.get('actual_offense_snaps') or 0) > 0 or any(
        (row.get(k) or 0) > 0 for k in ('actual_targets','actual_carries','actual_pass_attempts'))
    if participated:
        return 'missing_stat_with_verified_participation'
    if row.get('actual_offense_snaps') == 0:
        return 'zero_offensive_snaps_requires_book_settlement_review'
    return 'missing_participation_evidence'


def build(records, cfg, sources):
    rows = []
    for r in records:
        if eligible_error(r,cfg):
            continue
        source = sources.get(int(r['season']), {})
        state = classify(r, source)
        if state == 'settled_verified_participation':
            continue
        p = r['forecast_payload']
        rows.append(dict(forecast_id=int(r['id']), game_id=r['game_id'], player_id=r['player_id'],
            player=p.get('player_name'), team=p.get('team_abbr'), state=state,
            snap_source_status='fetch_failed' if source.get('error') else
                'source_not_observed' if not source else 'game_published' if r['game_id'] in source.get('games',[]) else 'game_not_published',
            snap_source_observed_at=source.get('observed_at'),
            action='wait_for_final' if state=='pending_game' else 'refresh_official_results_and_participation_then_regrade',
            actual=None, inferred_zero=False))
    return dict(built_at=datetime.now(timezone.utc).isoformat(), counts=dict(Counter(r['state'] for r in rows)),
        unresolved_final_rows=sum(r['state'] not in ('pending_game','void_nonparticipant') for r in rows), rows=rows,
        unique_unresolved_player_games=len({(r['game_id'],r['player_id']) for r in rows if r['state'] not in ('pending_game','void_nonparticipant')}),
        limitations=['Absent source rows do not prove nonparticipation.',
            'Explicit zero offensive snaps require book-rule settlement review, never an automatic winning under.',
            'Refresh changes only verified result inputs; original locks and scoring versions are immutable.'])


def load_reconciliation_rows(day, cfg):
    """Load only the small lock contract needed to reconcile the fixed trial.

    Full forecast payloads contain large distributions and scoring traces.  The
    general replay loader needs those fields, but reconciliation only needs the
    original offer identity/timing plus result and participation columns.  This
    query keeps the same eligibility checks without repeatedly transferring and
    interning megabytes of unrelated JSON for every forecast revision.
    """
    registered_at = cfg['registered_at']
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(
        cursor_factory=psycopg2.extras.RealDictCursor
    ) as cur:
        cur.execute("SET LOCAL statement_timeout='60s'")
        cur.execute(
            """
            SELECT p.id,p.game_id,p.player_id,p.player_name,p.team_abbr,p.stat,
                   p.side,p.line,p.price,p.book,p.offer_id,p.offer_player_name_norm,
                   p.created_at_utc,p.is_current,p.model_version,
                   g.season,g.week,g.status,g.start_ts_utc,
                   o.fetched_at_utc AS offer_fetched_at,
                   o.provider AS offer_provider,o.event_id AS offer_event_id,
                   o.bookmaker_key AS offer_book,o.player_name_norm AS offer_player_norm,
                   o.market_key AS offer_market_key,o.stat AS offer_stat,o.line AS offer_line,
                   o.over_price AS offer_over_price,o.under_price AS offer_under_price,
                   o.commence_time_utc AS offer_commence_time,
                   l.game_id AS result_game_id,l.offense_snaps AS actual_offense_snaps,
                   graded.result AS graded_result,
                   CASE WHEN g.status='final' THEN l.targets END AS actual_targets,
                   CASE WHEN g.status='final' THEN l.carries END AS actual_carries,
                   CASE WHEN g.status='final' THEN l.pass_attempts END AS actual_pass_attempts,
                   CASE WHEN g.status='final' AND (
                     COALESCE(l.offense_snaps,0)>0 OR
                     COALESCE(l.pass_attempts,0)+COALESCE(l.carries,0)+COALESCE(l.targets,0)>0 OR
                     (l.raw_json #> '{receiving_result_repair,verified_game_participation}')='true'::jsonb
                   ) THEN l.receiving_yards END AS actual
            FROM bets.nfl_player_prop_predictions p
            JOIN raw.nfl_games g USING(game_id)
            LEFT JOIN raw.nfl_player_gamelogs l
              ON l.game_id=p.game_id AND l.player_id=p.player_id AND l.team_abbr=p.team_abbr
            LEFT JOIN odds.nfl_player_prop_lines o ON o.id=p.offer_id
            LEFT JOIN bets.nfl_player_prop_prediction_results graded ON graded.prediction_id=p.id
            WHERE p.integrity_version='nfl-asof-v2'
              AND p.game_date_et=%s AND p.stat=%s AND p.book=%s
              AND p.model_version=%s AND p.created_at_utc >= %s::timestamptz
              AND p.line IS NOT NULL AND p.offer_id IS NOT NULL
              AND o.market_key=ANY(%s)
            ORDER BY p.created_at_utc,p.id
            """,
            (day, cfg['stat'], cfg['book'], cfg['production_release'], registered_at,
             cfg['market_keys']),
        )
        rows = []
        for raw in cur.fetchall():
            row = dict(raw)
            scoring_offer = {
                'offer_id': row['offer_id'],
                'provider': row.pop('offer_provider', None),
                'event_id': row.pop('offer_event_id', None),
                'bookmaker_key': row.pop('offer_book', None),
                'player_name_norm': row.pop('offer_player_norm', None),
                'market_key': row.pop('offer_market_key', None),
                'stat': row.pop('offer_stat', None),
                'line': row.pop('offer_line', None),
                'over_price': row.pop('offer_over_price', None),
                'under_price': row.pop('offer_under_price', None),
                'fetched_at_utc': row['offer_fetched_at'],
                'commence_time_utc': row.pop('offer_commence_time', None),
            }
            payload = {
                'player_name': row['player_name'],
                'team_abbr': row['team_abbr'],
                'book': row['book'],
                'model_version': row['model_version'],
                'line': row['line'],
                'side': row['side'],
                'price': row['price'],
                'offer_id': row['offer_id'],
                'offer_player_name_norm': row['offer_player_name_norm'],
                'scoring_replay': {'offer': scoring_offer},
            }
            row['forecast_payload'] = payload
            rows.append(row)
        return rows


def run(day):
    cfg = load_registration_config(); records = load_reconciliation_rows(day,cfg); sources = {}
    for season in {int(r['season']) for r in records}:
        path = ROOT/'reports'/f'nfl_snap_source_health_{season}.json'
        if path.exists():
            sources[season] = json.loads(path.read_text())
    doc = build(records,cfg,sources)
    for suffix in (str(day),'latest'):
        atomic_json(ROOT/'reports'/f'nfl_receiving_result_reconciliation_{suffix}.json',doc)
        text = ['# Receiving Result Reconciliation','',json.dumps(doc['counts']),
            f"Unresolved final forecasts: {doc['unresolved_final_rows']}; player-games: {doc['unique_unresolved_player_games']}",'']
        text += [f"- {r['forecast_id']}: {r['player']} ({r['game_id']}): {r['state']}; {r['snap_source_status']}" for r in doc['rows']]
        (ROOT/'reports'/f'nfl_receiving_result_reconciliation_{suffix}.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return doc


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--date',type=date.fromisoformat,required=True)
    result=run(parser.parse_args().date); print(json.dumps(result['counts']))
