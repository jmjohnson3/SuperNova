"""Same-line recent-game sensitivity using the complete captured live scorer."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import psycopg2

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, active_release, atomic_json, atomic_joblib
from nfl_pipeline.context_contract import safe_time, validate_evidence
from nfl_pipeline.modeling.challenger_models import curve_summary
from nfl_pipeline.modeling.receiver_role_model import CONTRACT, code_fingerprint, load_role_history, role_features
from nfl_pipeline.modeling.score_accuracy_components import full_path
from nfl_pipeline.modeling.train_accuracy_challengers import ROOT
from nfl_pipeline.modeling.train_receiver_role import curve


def load_artifact():
    manifest = json.loads((MODEL_ROOT / 'receiver_role' / 'latest.json').read_text())
    path = MODEL_ROOT / 'receiver_role' / manifest['run_id'] / 'models.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Receiver challenger checksum mismatch')
    artifact = joblib.load(path)
    if artifact['contract'] != CONTRACT or artifact['production_release'] != active_release()['release_id']:
        raise ValueError('Receiver challenger production cohort mismatch')
    if artifact.get('code_fingerprint') != code_fingerprint():
        raise ValueError('Receiver challenger code changed; rerun chronological validation')
    return artifact, manifest


def score_records(records, raw, artifact, scoring_at):
    selected = []; excluded = Counter()
    for forecast_id, payload, kickoff in records:
        features = payload.get('forecast_features')
        cutoff = safe_time(payload.get('prediction_context_cutoff_utc'))
        if (not features or payload.get('model_version') != artifact['production_release']
            or not cutoff or not validate_evidence(payload.get('context_evidence'), cutoff)):
            excluded['missing_or_invalid_locked_inputs'] += 1
            continue
        if date.fromisoformat(artifact['training_end']) >= cutoff.date():
            excluded['challenger_training_not_before_original_lock_date'] += 1
            continue
        if payload.get('line') is None or not payload.get('scoring_replay'):
            excluded['no_captured_offer_scoring'] += 1
            continue
        if float(features.get('n_games_prev_3') or 0) < 3:
            excluded['insufficient_player_history'] += 1
            continue
        selected.append((forecast_id, payload, kickoff, dict(features,
            game_id=payload['game_id'], player_id=payload['player_id'],
            game_date_et=date.fromisoformat(str(payload['game_date_et'])[:10]),
            context_evidence=payload['context_evidence'],
            prediction_context_cutoff_utc=payload['prediction_context_cutoff_utc'])))
    if not selected:
        return [], dict(excluded)
    frame = pd.DataFrame([r[3] for r in selected]).reset_index(drop=True)
    bundle = artifact['bundle']
    books = [r[1]['book'] for r in selected]
    offered = [[float(r[1]['line'])] for r in selected]
    variants = {}
    for label, weight in (('full', 1.), ('downweighted', .25), ('omitted', 0.)):
        rebuilt = pd.concat([frame, role_features(raw, frame, weight)], axis=1)
        v, w = curve(bundle, rebuilt, books=books, lines=offered)
        summary = curve_summary(v, w)
        comp = bundle['model'].components(rebuilt)
        variants[label] = (rebuilt, v, w, summary, comp)
    rows = []
    for i, (forecast_id, p, kickoff, _) in enumerate(selected):
        result = {'forecast_id': forecast_id, 'player': p['player_name'], 'game_id': p['game_id'],
            'player_id': p['player_id'], 'stat': p['stat'], 'book': p['book'], 'line': p['line'],
            'locked_price': p['price'], 'production_side': p['side'],
            'production_probability': p['probability'], 'production_projection': p['projection'],
            'original_lock_cutoff': p['prediction_context_cutoff_utc'],
            'production_release': p['model_version'], 'challenger_run': artifact['run_id'],
            'pregame_scored': scoring_at < safe_time(kickoff), 'betting_eligible': False,
            'variants': {}}
        try:
            for label, (f, v, w, s, comp) in variants.items():
                # Same offered line/book and frozen context/guards in every replay.
                scored = full_path(p, v[i], w[i], s['mean'][i])
                over = scored['calibrated_over_probability']
                result['variants'][label] = {
                    'mean': float(s['mean'][i]), 'median': float(s['median'][i]),
                    'p10': scored['live_p10'], 'p90': scored['live_p90'],
                    'targets': float(comp['targets'][i]), 'ypt': float(comp['yards_per_target'][i]),
                    'team_passes': float(comp['team_pass_attempts'][i]),
                    'over_probability': over, 'selected_side': scored['candidate_side'],
                    'raw_over_probability': scored.get('raw_over_probability'),
                    'probability_for_original_side': over if p['side'] == 'over' else 1 - over,
                    'scoring_stage': scored['scoring_stage'],
                    'role_inputs': f.iloc[i].filter(like='rr_').to_dict()}
        except ValueError as exc:
            excluded[str(exc)] += 1
            continue
        probabilities = [v['over_probability'] for v in result['variants'].values()]
        result['max_probability_swing_pp'] = 100 * (max(probabilities) - min(probabilities))
        result['direction_changes'] = len({v['selected_side'] for v in result['variants'].values()}) > 1
        result['configured_variant'] = 'downweighted' if bundle['config']['latest_weight'] == .25 else 'full'
        rows.append(result)
    return rows, dict(excluded)


def run(day):
    artifact, manifest = load_artifact()
    at = datetime.now(timezone.utc)
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout='60s'")
            cur.execute("""SELECT DISTINCT ON (p.game_id,p.player_id)
                p.id,p.forecast_payload,g.start_ts_utc
                FROM bets.nfl_player_prop_predictions p JOIN raw.nfl_games g USING(game_id)
                WHERE p.is_current AND p.game_date_et=%s AND p.stat='receiving_yards'
                  AND p.created_at_utc<g.start_ts_utc AND p.integrity_version='nfl-asof-v2'
                ORDER BY p.game_id,p.player_id,p.created_at_utc DESC,p.id DESC""", (day,))
            records = cur.fetchall()
    raw = load_role_history(at)
    raw = raw.loc[raw.game_date_et < day].copy()
    rows, excluded = score_records(records, raw, artifact, at)
    completed = datetime.now(timezone.utc)
    kickoffs = {forecast_id: safe_time(kickoff) for forecast_id, _, kickoff in records}
    for row in rows:
        row['pregame_scored'] = completed < kickoffs[row['forecast_id']]
    report = clean({'contract': CONTRACT, 'history_cutoff': at.isoformat(),
        'scored_at': completed.isoformat(), 'date': str(day),
        'status': 'sensitivity_diagnostic_not_bet_recommendations', 'source_manifest': manifest,
        'rows': rows, 'exclusions': excluded, 'historical_pass': artifact['historical_pass'],
        'production_changed': False, 'limitations': [
            'Latest-game omission/downweighting changes player performance and that game in team pass history; current timestamped context is held fixed.',
            'The complete captured scoring path is replayed at the same offered line and book; these are not current executable quotes.',
            'Historical statistics are reconstructed from corrected final rows available at scoring time, not certified as their original lock-time versions.',
            'Only pregame-scored rows can become prospective challenger evidence; started games are retrospective diagnostics.',
            'A stable side is not proof of an edge, and a changed side is not an instruction to reverse a bet.',
            'This artifact cannot promote a model or create micro/bankroll ledger rows.']})
    stamp = completed.strftime('%Y%m%dT%H%M%S%fZ')
    root = MODEL_ROOT / 'receiver_role' / artifact['run_id'] / 'sensitivity' / str(day)
    atomic_json(root / (stamp + '.json'), report)
    # Preserve the exact reconstructed history used by this diagnostic.
    keep = ['game_id', 'player_id', 'team_abbr', 'opponent_abbr', 'position', 'season', 'week',
        'game_date_et', 'targets', 'receiving_yards', 'receiving_air_yards', 'offense_snaps',
        'pass_attempts', 'updated_at_utc']
    atomic_joblib(root / (stamp + '-history.joblib'), raw.reindex(columns=keep))
    atomic_json(ROOT / 'reports' / f'nfl_receiver_role_sensitivity_{day}.json', report)
    text_rows = ['# Receiving Latest-Game Sensitivity', '',
        'Diagnostic only. Production forecasts and betting eligibility are unchanged.', '',
        '| Player | Locked pick | Production yards | Challenger full / downweighted / omitted yards | Probability swing | Side changes |',
        '|---|---|---:|---:|---:|---|']
    for row in sorted(rows, key=lambda r: -r['max_probability_swing_pp']):
        projections = ' / '.join(f"{row['variants'][label]['mean']:.1f}" for label in ('full', 'downweighted', 'omitted'))
        text_rows.append(f"| {row['player']} | {row['production_side']} {row['line']} ({row['book']}) | {row['production_projection']:.1f} | {projections} | {row['max_probability_swing_pp']:.1f}pp | {row['direction_changes']} |")
    text_rows += ['', *['- ' + s for s in report['limitations']], '', f'Excluded: {excluded}']
    (ROOT / 'reports' / f'nfl_receiver_role_sensitivity_{day}.md').write_text('\n'.join(text_rows) + '\n', encoding='utf-8')
    return {'rows': len(rows), 'pregame_rows': sum(r['pregame_scored'] for r in rows),
        'side_changes': sum(r['direction_changes'] for r in rows), 'exclusions': excluded,
        'production_changed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', type=date.fromisoformat, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.date), indent=2))
