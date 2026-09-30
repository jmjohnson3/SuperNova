"""Cash policy boundary for NFL cards. Historical micro tiers are research."""
from datetime import datetime, timezone
import json

import psycopg2

from nfl_pipeline import cash_trial_policy as policy
from nfl_pipeline import cash_execution
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.modeling.live_scoring_replay import ROOT


def research_rows(rows):
    return [dict(r, original_tier=r.get('tier'), tier='paper', cash_eligible=False) for r in rows]


def prepare(day, game_rows, prop_rows, *, reserve=False):
    """No file-only report can authorize a changed model/scoring cohort."""
    games, props = research_rows(game_rows), research_rows(prop_rows)
    path = ROOT/'reports'/f'nfl_cash_readiness_{day}.json'
    if not path.exists():
        return games, props, {'status':'operational_issue', 'reason':'cash_readiness_missing'}
    doc = json.loads(path.read_text())
    registration = policy.load_registration()
    now = datetime.now(timezone.utc)
    built = safe_time(doc.get('built_at'))
    if (not registration or doc.get('registration_sha256') != registration['sha256'] or not built
            or not 0 <= (now-built).total_seconds() <= 1200):
        return games, props, {'status':'operational_issue', 'reason':'cash_readiness_stale_or_mismatched'}
    status = doc['decision']['status']
    if status != 'cash_trial_eligible':
        metrics = doc['provisional']['metrics']
        reason = ('Cash trial paused; review required. Research is not a betting instruction.' if status == 'paused' else
            f"Research only: {metrics['decisions']} settled decisions / {doc['decision']['next_checkpoint'] or 200} at the next review, "
            f"{metrics['weeks']} independent weeks / 3 minimum under the registered cash policy. "
            'Counts alone do not approve cash bets.')
        return games, props, {'status':status, 'reason':reason}
    from nfl_pipeline.modeling import benchmark_offers as benchmark
    from nfl_pipeline.modeling import receiving_research_trial as trial
    from nfl_pipeline.modeling import receiving_trial_checkpoint as checkpoint
    from nfl_pipeline.modeling.predict_player_props import _break_even_american_price, _ev_per_unit
    artifact, manifest = benchmark.load_artifact()
    cfg = registration['config']
    if manifest['sha256'] != cfg['artifact_sha256']:
        raise ValueError('Cash artifact changed')
    research = trial.load_registration()
    selected = {r['forecast_id'] for d in trial.captures(research) for r in d['selected'] if r['day']==str(day)}
    records = benchmark.load_rows(day)
    documents = [json.loads(p.read_text()) for p in (benchmark.STORE/artifact['run_id']/'prospective'/str(day)).glob('*.json')]
    captured, _ = checkpoint.capture_index(records, documents, research['config'])
    candidates = []
    for row in props:
        fid = row.get('forecast_id'); shadow = captured.get(fid)
        if fid not in selected or not shadow or row.get('model_version') != cfg['release']:
            continue
        if (row.get('scoring_replay') or {}).get('scoring_fingerprint') != cfg['scoring_version']:
            continue
        probability = float(shadow['same_side_probability'])
        push = float(shadow.get('push_probability', 0))
        # This strategy is common half-lines only until a push-curve trial is registered.
        if float(row['line']).is_integer():
            continue
        p = dict(row, probability=probability, push_probability=push,
            production_probability=row['probability'], probability_source=cfg['variant'],
            cash_artifact_sha256=cfg['artifact_sha256'], market_probability=shadow.get('market_probability'),
            market_no_vig_probability=shadow.get('market_probability'),
            projection=shadow['expected_yards'], projection_p10=shadow['live_p10'], projection_p90=shadow['live_p90'],
            minimum_american_price=_break_even_american_price(probability),
            ev=_ev_per_unit(probability, row['price'], push))
        p['model_market_edge'] = probability-float(p['market_probability']) if p['market_probability'] is not None else None
        candidates.append(p)
    if not reserve:
        return games, props, {'status':'research', 'reason':'preview_only_no_cash_capacity_reserved'}
    with psycopg2.connect(PG_DSN) as conn:
        accepted, excluded = cash_execution.reserve(conn, candidates, registration, status, now)
    accepted_ids = {r['forecast_id'] for r in accepted}
    props = [r for r in props if r.get('forecast_id') not in accepted_ids]+accepted
    reason = ('$1 flat; global caps 5/day, $20/NFL week; pause at $30 net loss' if accepted else
              'No current cash plays: '+(', '.join(excluded) if excluded else 'no fresh executable fixed selections'))
    return games, props, {'status':'cash_trial_eligible', 'accepted':len(accepted), 'exclusions':excluded, 'reason':reason}
