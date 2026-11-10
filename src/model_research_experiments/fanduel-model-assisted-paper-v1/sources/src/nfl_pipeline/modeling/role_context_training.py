"""Strict as-of role evidence for player-games; no hindsight roster backfill."""
from collections import Counter, defaultdict

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

from nfl_pipeline.context_contract import number, safe_time
from nfl_pipeline.db import PG_DSN

CONTRACT = 'nfl-player-game-role-asof-v1'
KINDS = ('nfl_depth_charts', 'nfl_injuries', 'nfl_rosters')
OUT = {'out', 'injured reserve', 'ir', 'inactive'}
LIMITED = ('questionable', 'doubtful', 'limited', 'did not participate')


def identity(value):
    return str(value).strip() if value is not None else ''


def phase(value):
    value = str(value or '').upper()
    return {'REGULAR': 'REG', 'REGULAR SEASON': 'REG', 'POSTSEASON': 'POST', 'PRESEASON': 'PRE'}.get(value, value)


class RoleEvidenceIndex:
    def __init__(self, observations):
        self.players = defaultdict(list)
        self.injuries = defaultdict(list)
        self.excluded = Counter()
        for r in observations:
            p = r.get('payload') or {}
            at = safe_time(r.get('observed_at'))
            season = number(p.get('season'))
            player, team = identity(p.get('player_id')), identity(p.get('team_abbr'))
            kind = r.get('kind')
            if not at or season is None or not player or not team or kind not in KINDS:
                self.excluded['unusable_identity_or_timestamp'] += 1
                continue
            row = dict(r, payload=p, observed_at=at)
            self.players[(int(season), team, player, kind)].append(row)
            if kind == 'nfl_injuries':
                self.injuries[(int(season), team)].append(row)

    @staticmethod
    def eligible(rows, cutoff, week, exact_week=False, season_type=None):
        result = []
        for r in rows:
            p = r['payload']; period = number(p.get('week'))
            effective = safe_time(p.get('snapshot_ts_utc'))
            source_phase = phase(p.get('season_type') or p.get('game_type'))
            if source_phase and source_phase != phase(season_type):
                continue
            if r['observed_at'] > cutoff or (effective and effective > cutoff):
                continue
            if period is not None and period > week:
                continue
            if exact_week and period != week:
                continue
            result.append(r)
        return sorted(result, key=lambda r: (number(r['payload'].get('week')) or 0,
            safe_time(r['payload'].get('snapshot_ts_utc')) or r['observed_at'],
            r['observed_at'], str(r.get('row_id', ''))))

    def resolve(self, request):
        cutoff = safe_time(request.get('role_lock_cutoff'))
        kickoff = safe_time(request.get('start_ts_utc'))
        result = dict(contract=CONTRACT, cutoff=cutoff.isoformat() if cutoff else None,
            lock_id=request.get('role_lock_id'), status='missing_verified_lock',
            depth_rank=None, depth_movement=None, expected_starter=None,
            injury_status=None, practice_status=None, roster_status=None,
            teammate_absences=None, teammate_limited=None, teammate_reported_players=None,
            teammate_coverage='unknown', provenance={}, exclusions={})
        if not cutoff or not kickoff or cutoff >= kickoff:
            return result
        season = int(request['season']); week = int(request['week'])
        team, player = identity(request['team_abbr']), identity(request['player_id'])
        result['status'] = 'no_matching_pregame_observations'
        own = {}
        for kind in KINDS:
            candidates = self.players[(season, team, player, kind)]
            rows = self.eligible(candidates, cutoff, week,
                                 exact_week=kind == 'nfl_injuries', season_type=request.get('season_type'))
            if not rows:
                result['exclusions'][kind] = ('no_identity_match' if not candidates else
                    'observed_only_after_lock' if all(r['observed_at'] > cutoff for r in candidates) else
                    'source_time_week_or_phase_mismatch')
                continue
            last = rows[-1]; p = last['payload']; own[kind] = p
            result['provenance'][kind] = dict(row_id=last.get('row_id'),
                observed_at=last['observed_at'].isoformat(), source=p.get('source'),
                source_snapshot=p.get('snapshot_ts_utc'), week=p.get('week'), team=team, player_id=player)
            if kind == 'nfl_depth_charts':
                rank = number(p.get('pos_rank'))
                previous = [r for r in rows[:-1] if r['observed_at'] < last['observed_at']
                    and r['payload'].get('pos_abb') == p.get('pos_abb')
                    and number(r['payload'].get('pos_rank')) is not None]
                old = number(previous[-1]['payload'].get('pos_rank')) if previous else None
                result.update(depth_rank=rank, expected_starter=(rank == 1) if rank is not None else None,
                    depth_movement=(old-rank) if old is not None and rank is not None else None)
            elif kind == 'nfl_injuries':
                result.update(injury_status=p.get('report_status'), practice_status=p.get('practice_status'))
            else:
                result['roster_status'] = p.get('roster_status')
        teammates = {}
        for r in self.eligible(self.injuries[(season, team)], cutoff, week, exact_week=True,
                               season_type=request.get('season_type')):
            p = r['payload']; pid = identity(p.get('player_id'))
            if pid != player and p.get('position') in ('QB', 'WR', 'TE', 'RB'):
                teammates[pid] = r
        if teammates:
            statuses = [str(r['payload'].get('report_status') or '').lower() for r in teammates.values()]
            limited = [str(r['payload'].get('report_status') or '').lower() + ' ' +
                str(r['payload'].get('practice_status') or '').lower() for r in teammates.values()]
            has_status = any(r['payload'].get('report_status') is not None for r in teammates.values())
            has_any = any(r['payload'].get('report_status') is not None or r['payload'].get('practice_status') is not None
                          for r in teammates.values())
            result.update(teammate_reported_players=len(teammates), teammate_coverage='partial_observed_reports',
                teammate_absences=sum(s in OUT for s in statuses) if has_status else None,
                teammate_limited=sum(any(tag in s for tag in LIMITED) for s in limited) if has_any else None)
            result['provenance']['teammates'] = [dict(player_id=pid, row_id=r.get('row_id'),
                observed_at=r['observed_at'].isoformat()) for pid, r in sorted(teammates.items())]
        if result['provenance']:
            result['status'] = 'asof_observed'
        return result


def load_sources(game_ids, before):
    """Observation timestamps, not today's raw tables, determine availability."""
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='120s'")
        cur.execute('''SELECT DISTINCT ON (p.game_id,p.player_id)
            p.game_id,p.player_id,p.id AS role_lock_id,p.created_at_utc,
            p.forecast_payload->>'prediction_context_cutoff_utc' AS source_cutoff,
            g.start_ts_utc,g.season_type,p.team_abbr
            FROM bets.nfl_player_prop_predictions p JOIN raw.nfl_games g USING(game_id)
            WHERE p.game_id=ANY(%s) AND p.created_at_utc<g.start_ts_utc
              AND p.created_at_utc<=%s
            ORDER BY p.game_id,p.player_id,p.created_at_utc,p.id''', (list(game_ids), before))
        locks = [dict(r) for r in cur.fetchall()]
        cur.execute('''SELECT kind,row_id,observed_at,payload FROM raw.nfl_context_observations
            WHERE observed_at<=%s AND kind=ANY(%s)
              AND extract(year FROM observed_at)<=(payload->>'season')::int+1
            ORDER BY observed_at,row_id''', (before, list(KINDS)))
        observations = [dict(r) for r in cur.fetchall()]
    return locks, observations


def join_role_context(frame, locks, observations):
    if frame.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Role training join requires one row per player-game')
    lookup = {}
    for r in locks:
        at = safe_time(r.get('created_at_utc')); start = safe_time(r.get('start_ts_utc'))
        source = safe_time(r.get('source_cutoff'))
        if not at or not start or at >= start or (r.get('source_cutoff') and (not source or source > at)):
            continue
        key = (r['game_id'], identity(r['player_id']))
        if key not in lookup or at < lookup[key]['created_at_utc']:
            lookup[key] = dict(r, created_at_utc=at, role_lock_cutoff=source or at)
    index = RoleEvidenceIndex(observations)
    rows = []
    for request in frame.to_dict('records'):
        lock = lookup.get((request['game_id'], identity(request['player_id'])))
        if lock and identity(lock['team_abbr']) == identity(request['team_abbr']):
            request.update({k: lock[k] for k in ('role_lock_id', 'role_lock_cutoff', 'start_ts_utc')})
            request['season_type'] = lock.get('season_type', request.get('season_type'))
        else:
            request.update(role_lock_id=None, role_lock_cutoff=None)
        rows.append(index.resolve(request))
    result = frame.copy()
    result['target_role_evidence'] = rows
    return result, coverage(result, index.excluded)


def coverage(frame, excluded=None):
    fields = ('depth_rank', 'expected_starter', 'depth_movement', 'injury_status', 'practice_status',
              'roster_status', 'teammate_absences', 'teammate_limited')
    def summarize(group):
        evidence = group.target_role_evidence.tolist()
        return dict(rows=len(group), verified_locks=sum(r.get('cutoff') is not None for r in evidence),
            status=dict(Counter(r['status'] for r in evidence)),
            exclusions=dict(Counter(f'{kind}:{reason}' for r in evidence for kind,reason in r.get('exclusions',{}).items())),
            fields={k: dict(nonmissing=sum(r.get(k) is not None for r in evidence),
                distinct=len({str(r[k]) for r in evidence if r.get(k) is not None})) for k in fields})
    return dict(contract=CONTRACT, overall=summarize(frame),
        by_season={str(k): summarize(g) for k, g in frame.groupby('season')},
        rejected_observations=dict(excluded or {}),
        limitations=['No actual lock means role evidence is unknown, not a kickoff reconstruction.',
            'A source snapshot date alone does not override its first observed timestamp.',
            'Teammate counts cover observed reports only; absent reports are not proof of health.',
            'This join does not manufacture measured routes or first-read history.'])


def build_current(through):
    """Refresh a separate training cache, not existing features, locks or artifacts."""
    import json
    from datetime import datetime, timezone
    from nfl_pipeline.forecast_store import clean
    from nfl_pipeline.integrity import MODEL_ROOT, atomic_json, atomic_joblib
    from nfl_pipeline.modeling.train_accuracy_challengers import load_training, ROOT
    from nfl_pipeline.modeling.train_workload_depth import prepare
    from nfl_pipeline.modeling.receiver_role_model import load_role_history
    players, _ = load_training()
    players = players.loc[players.position.isin(('WR','TE','RB')) & pd.to_datetime(players.game_date_et).le(pd.Timestamp(through))]
    frame = prepare(players, load_role_history())
    now = datetime.now(timezone.utc)
    locks, observations = load_sources(frame.game_id.unique(), now)
    joined, report = join_role_context(frame, locks, observations)
    path = MODEL_ROOT/'target_role_context'/now.strftime('%Y%m%dT%H%M%S%fZ')
    atomic_joblib(path/'training.joblib', joined)
    report.update(built_at=now.isoformat(), through=through, cache=str(path/'training.joblib'),
                  production_changed=False, historical_locks_changed=False)
    atomic_json(path/'report.json', clean(report))
    atomic_json(ROOT/'reports/nfl_target_role_context_latest.json', clean(report))
    text = ['# Target Role Context: Verified As-Of Coverage', '',
        '| Season | Player-games | Verified locks | Pregame context |', '|---|---:|---:|---:|']
    for year, r in report['by_season'].items():
        text.append(f"| {year} | {r['rows']} | {r['verified_locks']} | {r['status'].get('asof_observed',0)} |")
    text += ['', 'Field coverage: '+json.dumps(report['overall']['fields']), '',
             'Exclusions: '+json.dumps(report['overall']['exclusions']), '', *report['limitations']]
    (ROOT/'reports/nfl_target_role_context_latest.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    import argparse
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--through', required=True)
    result = build_current(p.parse_args().through)
    print(json.dumps(dict(cache=result['cache'], coverage=result['overall'])))
