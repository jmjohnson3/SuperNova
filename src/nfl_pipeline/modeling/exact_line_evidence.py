"""Account for every prediction before building exact-offer training evidence."""
from collections import Counter
import math

import pandas as pd

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.markets import normalize_team
from nfl_pipeline.modeling.live_scoring_replay import validate_lock


SQL_EVIDENCE = """
SELECT p.id,p.game_date_et,p.game_id,p.player_id,p.stat,p.side,p.book,
       p.line::float AS line,p.price::float AS price,p.projection::float AS projection,
       p.baseline_projection::float AS baseline_projection,p.probability::float AS model_probability,
       p.ev::float AS model_ev,p.edge::float AS model_edge,p.model_version,
       p.offer_player_name_norm,p.created_at_utc,p.integrity_version,p.forecast_payload,
       g.status,g.start_ts_utc,g.season,g.week,g.home_team_abbr,g.away_team_abbr,
       r.result,r.actual_stat::float AS actual,r.profit_per_unit::float AS profit_per_unit,
       r.updated_at_utc AS label_available_at,
       (COALESCE(l.offense_snaps,0)>0 OR
        COALESCE(l.pass_attempts,0)+COALESCE(l.carries,0)+COALESCE(l.targets,0)>0) AS participated,
       c.clv_status,c.clv_prob_delta::float AS clv_prob_delta,
       c.updated_at_utc AS clv_available_at,
       o.id AS matched_offer_id,o.bookmaker_key AS offer_book,o.stat AS offer_stat,
       o.line::float AS offer_line,o.player_name_norm AS offer_player,
       COALESCE(NULLIF(TRIM(o.home_team),''),go.home_team_abbr) AS offer_home,
       COALESCE(NULLIF(TRIM(o.away_team),''),go.away_team_abbr) AS offer_away,o.snapshot_role,
       CASE WHEN NULLIF(TRIM(o.home_team),'') IS NULL OR NULLIF(TRIM(o.away_team),'') IS NULL
            THEN go.fetched_at_utc END AS identity_fetched_at,
       o.over_price::float AS lock_over_price,o.under_price::float AS lock_under_price,
       o.fetched_at_utc AS lock_fetched_at_utc,o.fetched_at_utc AS offer_fetched_at
FROM bets.nfl_player_prop_predictions p
LEFT JOIN raw.nfl_games g USING(game_id)
LEFT JOIN raw.nfl_player_gamelogs l ON l.game_id=p.game_id AND l.player_id=p.player_id AND l.team_abbr=p.team_abbr
LEFT JOIN bets.nfl_player_prop_prediction_results r ON r.prediction_id=p.id
LEFT JOIN bets.nfl_prediction_clv c ON c.source_kind='prop' AND c.prediction_id=p.id
LEFT JOIN odds.nfl_player_prop_lines o ON o.id=p.offer_id
LEFT JOIN LATERAL (
  SELECT x.home_team_abbr,x.away_team_abbr,x.fetched_at_utc FROM odds.nfl_game_lines x
  WHERE x.provider=o.provider AND x.event_id=o.event_id AND x.as_of_date=p.game_date_et
    AND x.fetched_at_utc<=p.created_at_utc AND x.commence_time_utc=o.commence_time_utc
    AND x.home_team_abbr IS NOT NULL AND x.away_team_abbr IS NOT NULL
  ORDER BY x.fetched_at_utc DESC,x.id DESC LIMIT 1
) go ON TRUE
ORDER BY p.created_at_utc,p.id
"""


def valid_price(value):
    try:
        return math.isfinite(float(value)) and abs(float(value)) >= 100
    except (TypeError, ValueError):
        return False


def exclusion(row):
    p = row.get('forecast_payload') or {}
    if row.get('integrity_version') != 'nfl-asof-v2':
        return 'legacy_lock_unverified'
    if row.get('side') not in ('over', 'under') or row.get('line') is None:
        return 'projection_without_offer'
    if not valid_price(row.get('price')):
        return 'invalid_locked_price'
    probability = row.get('model_probability')
    if probability is None or not 0 < float(probability) < 1:
        return 'invalid_probability'
    error = validate_lock(row)
    if error:
        return error
    if not row.get('model_version') or row['model_version'] != p.get('model_version'):
        return 'model_version_unverified'
    if row.get('matched_offer_id') is None:
        return 'missing_exact_offer'
    if (row.get('offer_book') != row['book'] or row.get('offer_stat') != row['stat']
            or row.get('offer_line') != row['line'] or row.get('offer_player') != row.get('offer_player_name_norm')):
        return 'exact_offer_identity_mismatch'
    if not row.get('offer_home') or not row.get('offer_away'):
        return 'offer_game_identity_unknown'
    identity_time = safe_time(row.get('identity_fetched_at'))
    if identity_time and identity_time > safe_time(p.get('prediction_context_cutoff_utc')):
        return 'offer_game_identity_after_cutoff'
    if (normalize_team(row.get('offer_home')) != normalize_team(row.get('home_team_abbr'))
            or normalize_team(row.get('offer_away')) != normalize_team(row.get('away_team_abbr'))):
        return 'offer_game_mismatch'
    if row.get('snapshot_role') not in ('open', 'lock', 'live'):
        return 'not_lock_time_offer'
    if not all(valid_price(row.get(k)) for k in ('lock_over_price', 'lock_under_price')):
        return 'missing_true_pair'
    if float(row['lock_' + row['side'] + '_price']) != float(row['price']):
        return 'locked_price_mismatch'
    if row.get('status') != 'final':
        return 'game_not_final'
    if row.get('participated') is not True:
        return 'participation_unverified'
    if row.get('result') == 'push':
        return 'push_not_binary'
    if row.get('result') not in ('win', 'loss') or row.get('actual') is None:
        return 'missing_settled_result'
    if safe_time(row.get('label_available_at')) is None:
        return 'missing_label_timestamp'
    return None


def audit_evidence(frame):
    kept = []; excluded = []; seen = set()
    # Deduplicate only valid decisions. A broken early row must not suppress a
    # later executable lock, but multiple prices/books never create more outcomes.
    for row in frame.sort_values(['created_at_utc', 'id']).to_dict('records'):
        row = {k: None if not isinstance(v, (dict, list, tuple)) and pd.isna(v) else v for k, v in row.items()}
        reason = exclusion(row)
        key = tuple(row.get(k) for k in ('game_id', 'player_id', 'stat', 'side', 'book', 'line'))
        if reason is None and key in seen:
            reason = 'later_revision_same_decision'
        if reason:
            excluded.append(dict(prediction_id=int(row['id']), reason=reason, day=str(row.get('game_date_et'))))
        else:
            seen.add(key); kept.append(row)
    eligible = pd.DataFrame(kept, columns=frame.columns)
    audit = dict(scanned_rows=len(frame), eligible_rows=len(eligible),
                 exclusions=dict(Counter(r['reason'] for r in excluded)), excluded_rows=excluded,
                 eligible_prediction_ids=[int(r['id']) for r in kept],
                 unique_player_games=len({(r['game_id'], r['player_id']) for r in kept}),
                 independent_weeks=len({(r['season'], r['week']) for r in kept}))
    assert audit['scanned_rows'] == audit['eligible_rows'] + len(excluded)
    return eligible, audit
