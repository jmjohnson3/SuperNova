"""Account for missing receiving results; no inferred zeros, voids or winners."""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json

from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling.live_scoring_replay import load_rows, ROOT
from nfl_pipeline.modeling.receiving_research_trial import load_registration
from nfl_pipeline.modeling.receiving_trial_checkpoint import eligible_error


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


def run(day):
    records = load_rows(day,stat='receiving_yards'); cfg = load_registration()['config']; sources = {}
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
