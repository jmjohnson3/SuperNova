"""Recover missing receiving results from verified snaps and complete, reconciled PBP."""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json

import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.import_usage_context import DEFAULT_PBP_URL, DEFAULT_SNAP_COUNTS_URL
from nfl_pipeline.integrity import MODEL_ROOT, atomic_json
from nfl_pipeline.markets import normalize_team


def verified_inactive(row, snaps, pbp):
    """Exact-week inactive roster plus published participation, not mere absence."""
    if row.get('roster_statuses')!=['INA'] or len(row.get('pfr_ids') or [])!=1:
        return False
    game=pbp.loc[pbp.game_id.eq(row['game_id'])].sort_values('play_id')
    if game.empty or str(game.iloc[-1]['desc']).strip().upper()!='END GAME':
        return False
    team=snaps.loc[snaps.game_id.eq(row['game_id']) & snaps.team.map(normalize_team).eq(row['team_abbr'])]
    if team.empty:
        return False
    player=team.loc[team.pfr_player_id.eq(row['pfr_ids'][0])]
    if not player.empty and (player[['offense_snaps','defense_snaps','st_snaps']].isna().any().any()
                            or (player[['offense_snaps','defense_snaps','st_snaps']]>0).any().any()):
        return False
    return not game.receiver_player_id.eq(row['player_id']).any()


def verified_receiving(row, snaps, pbp, totals):
    """Zero requires positive participation AND a complete reconciled result source."""
    ids=row.get('pfr_ids') or []
    if len(ids)!=1:
        return None,'missing_or_ambiguous_player_identity'
    snap=snaps.loc[snaps.game_id.eq(row['game_id']) & snaps.pfr_player_id.eq(ids[0])
                   & snaps.team.map(normalize_team).eq(row['team_abbr'])]
    if len(snap)!=1 or pd.isna(snap.iloc[0].offense_snaps) or snap.iloc[0].offense_snaps<=0:
        return None,'no_verified_offensive_participation'
    game=pbp.loc[pbp.game_id.eq(row['game_id'])].sort_values('play_id')
    if game.empty or str(game.iloc[-1]['desc']).strip().upper()!='END GAME':
        return None,'play_by_play_not_complete'
    plays=game.loc[game.posteam.map(normalize_team).eq(row['team_abbr']) & game.pass_attempt.eq(1)
                   & ~game.play_type.eq('no_play') & ~game.sack.eq(1) & ~game.two_point_attempt.eq(1)]
    complete=plays.loc[plays.complete_pass.eq(1)]
    if (plays.empty or complete.receiver_player_id.isna().any() or complete.receiving_yards.isna().any()
            or plays.lateral_receiver_player_id.notna().any()):
        return None,'incomplete_or_lateral_receiving_accounting'
    expected=totals.get((row['game_id'],row['team_abbr']))
    yards=float(complete.receiving_yards.sum())
    if not expected or expected[0] is None or expected[1] is None or len(plays)!=float(expected[0]) or abs(yards-float(expected[1]))>1e-8:
        return None,'team_pass_totals_do_not_reconcile'
    player=plays.loc[plays.receiver_player_id.eq(row['player_id'])]
    caught=player.loc[player.complete_pass.eq(1)]
    actual=float(caught.receiving_yards.sum())
    # A disagreeing existing stat is a review, never a silent source replacement.
    if row.get('existing_yards') is not None and float(row['existing_yards'])!=actual:
        return None,'existing_stat_conflict'
    return dict(receiving_yards=actual, targets=len(player), receptions=len(caught),
        offense_snaps=float(snap.iloc[0].offense_snaps), pfr_id=ids[0],
        team_pass_attempts=len(plays), team_passing_yards=yards,
        evidence='positive_offensive_snaps_and_complete_reconciled_play_by_play',
        player_play_ids=player.play_id.astype(float).tolist()),None


def run(day, apply=False):
    now=datetime.now(timezone.utc); outputs=[]; updated=0; voided=0
    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET LOCAL statement_timeout='60s'")
            cur.execute("SET LOCAL lock_timeout='5s'")
            cur.execute("""SELECT DISTINCT p.game_id,p.player_id,p.team_abbr,p.player_name,
                g.season,g.week,g.game_date_et, g.home_team_abbr,g.away_team_abbr,
                p.forecast_payload->>'position' AS position,l.receiving_yards AS existing_yards,
                ARRAY(SELECT DISTINCT r.pfr_id FROM raw.nfl_rosters r WHERE r.season=g.season
                  AND r.player_id=p.player_id AND r.team_abbr=p.team_abbr AND r.pfr_id IS NOT NULL) AS pfr_ids,
                ARRAY(SELECT DISTINCT r.roster_status FROM raw.nfl_rosters r WHERE r.season=g.season
                  AND r.week=g.week AND r.game_type='REG' AND r.player_id=p.player_id AND r.team_abbr=p.team_abbr
                  AND r.source='nflverse_rosters' ORDER BY r.roster_status) AS roster_statuses
              FROM bets.nfl_player_prop_predictions p JOIN raw.nfl_games g USING(game_id)
              LEFT JOIN raw.nfl_player_gamelogs l ON l.game_id=p.game_id AND l.player_id=p.player_id AND l.team_abbr=p.team_abbr
              WHERE p.game_date_et=%s AND p.stat='receiving_yards' AND g.status='final'
                AND p.line IS NOT NULL AND p.side IN ('over','under')
                AND p.created_at_utc<g.start_ts_utc AND p.integrity_version='nfl-asof-v2'
                AND (l.receiving_yards IS NULL OR NOT (COALESCE(l.offense_snaps,0)>0 OR
                     COALESCE(l.pass_attempts,0)+COALESCE(l.carries,0)+COALESCE(l.targets,0)>0))""",(day,))
            rows=[dict(r) for r in cur.fetchall()]
        with conn.cursor() as cur:
            cur.execute("""SELECT game_id,team_abbr,SUM(pass_attempts),SUM(passing_yards)
                 FROM raw.nfl_player_gamelogs WHERE game_date_et=%s GROUP BY game_id,team_abbr""",(day,))
            totals={(g,t):(a,y) for g,t,a,y in cur.fetchall()}
        for season in sorted({r['season'] for r in rows}):
            snap_url=DEFAULT_SNAP_COUNTS_URL.format(season=season); pbp_url=DEFAULT_PBP_URL.format(season=season)
            snaps=pd.read_csv(snap_url,low_memory=False)
            columns={'game_id','play_id','desc','posteam','play_type','pass_attempt','complete_pass',
                     'receiver_player_id','receiving_yards','lateral_receiver_player_id','sack','two_point_attempt'}
            pbp=pd.read_csv(pbp_url,usecols=lambda c:c in columns,low_memory=False)
            if not columns.issubset(pbp.columns):
                raise ValueError('Missing required play-by-play columns')
            observed_at=datetime.now(timezone.utc).isoformat()
            seen=set()
            for row in rows:
                key=(row['game_id'],row['player_id'],row['team_abbr'])
                if row['season']!=season or key in seen:
                    continue
                seen.add(key)
                proof,error=verified_receiving(row,snaps,pbp,totals)
                entry=dict(game_id=row['game_id'],player_id=row['player_id'],player=row['player_name'],
                    team=row['team_abbr'],status=error or 'verified',proof=proof)
                outputs.append(entry)
                if error=='no_verified_offensive_participation' and verified_inactive(row,snaps,pbp):
                    entry.update(status='verified_inactive_void',proof=dict(roster_status='INA',season=row['season'],week=row['week'],
                        pfr_id=row['pfr_ids'][0],roster_source='nflverse_rosters',snap_source=snap_url,pbp_source=pbp_url,
                        rule_source='https://www.fanduel.com/fanduel-sportsbook-house-rules-co',
                        rule_effective_date='2026-07-22',observed_at=observed_at))
                    if apply:
                        with conn.cursor() as cur:
                            cur.execute("""INSERT INTO bets.nfl_player_prop_prediction_results
                              (prediction_id,game_date_et,game_id,player_id,player_name,stat,side,line,price,actual_stat,result,profit_per_unit)
                              SELECT p.id,p.game_date_et,p.game_id,p.player_id,p.player_name,p.stat,p.side,p.line,p.price,NULL,'void_nonparticipant',0
                              FROM bets.nfl_player_prop_predictions p WHERE p.game_id=%s AND p.player_id=%s AND p.team_abbr=%s
                                AND p.book='fanduel' AND p.stat='receiving_yards' AND p.line IS NOT NULL
                                AND p.forecast_payload->'scoring_replay'->'offer'->>'market_key' IN ('player_reception_yds','player_receiving_yards')
                              ON CONFLICT (prediction_id) DO UPDATE SET actual_stat=NULL,result=EXCLUDED.result,profit_per_unit=0,updated_at_utc=NOW()
                              WHERE bets.nfl_player_prop_prediction_results.actual_stat IS NULL
                                AND bets.nfl_player_prop_prediction_results.result NOT IN ('win','loss','push','void_nonparticipant')""",key)
                            entry['voided_forecasts']=cur.rowcount;voided+=cur.rowcount
                    continue
                if proof is None or not apply:
                    continue
                provenance=dict(proof,observed_at=observed_at,snap_source=snap_url,pbp_source=pbp_url)
                with conn.cursor() as cur:
                    cur.execute("""INSERT INTO raw.nfl_player_gamelogs
                       (season,week,game_id,game_date_et,player_id,player_name,team_abbr,opponent_abbr,
                        position,is_home,receiving_yards,targets,receptions,offense_snaps,source,raw_json)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'nflverse_verified_receiving_repair',%s)
                       ON CONFLICT (season,week,game_id,player_id,team_abbr) DO UPDATE SET
                         receiving_yards=EXCLUDED.receiving_yards,targets=EXCLUDED.targets,receptions=EXCLUDED.receptions,
                         offense_snaps=EXCLUDED.offense_snaps,
                         raw_json=COALESCE(raw.nfl_player_gamelogs.raw_json,'{}'::jsonb)||EXCLUDED.raw_json,
                         updated_at_utc=NOW()
                       WHERE (raw.nfl_player_gamelogs.receiving_yards IS NULL OR
                              raw.nfl_player_gamelogs.receiving_yards=EXCLUDED.receiving_yards)
                         AND (raw.nfl_player_gamelogs.receiving_yards IS NULL OR NOT
                            (COALESCE(raw.nfl_player_gamelogs.offense_snaps,0)>0 OR
                             COALESCE(raw.nfl_player_gamelogs.pass_attempts,0)+COALESCE(raw.nfl_player_gamelogs.carries,0)+
                             COALESCE(raw.nfl_player_gamelogs.targets,0)>0))""",
                        (row['season'],row['week'],row['game_id'],row['game_date_et'],row['player_id'],row['player_name'],row['team_abbr'],
                         row['away_team_abbr'] if row['team_abbr']==row['home_team_abbr'] else row['home_team_abbr'],
                         row['position'],row['team_abbr']==row['home_team_abbr'],proof['receiving_yards'],proof['targets'],
                         proof['receptions'],proof['offense_snaps'],psycopg2.extras.Json({'receiving_result_repair':provenance})))
                    entry['updated']=cur.rowcount; updated+=cur.rowcount
    doc=dict(day=str(day),built_at=now.isoformat(),applied=apply,updated_player_games=updated,voided_forecasts=voided,
        counts=dict(Counter(r['status'] for r in outputs)),rows=outputs,forecasts_changed=False)
    atomic_json(MODEL_ROOT/'receiving_result_repair'/(now.strftime('%Y%m%dT%H%M%S%fZ')+'.json'),doc)
    return doc


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--date',type=date.fromisoformat,required=True)
    parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    print(json.dumps(run(args.date,args.apply),default=str))
