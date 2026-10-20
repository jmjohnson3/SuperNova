"""Read-only timing and equivalence check for the live player-context query."""
import argparse
from datetime import date, datetime, timezone
import json
from time import perf_counter
from unittest.mock import patch

import pandas as pd
import psycopg2

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling.live_scoring_replay import ROOT
from nfl_pipeline.modeling import predict_player_props as props
from nfl_pipeline.modeling.scoring_versions import archived_candidate


def materialized_reference(loader, conn, day, cutoff, slate_only=False):
    """Keep archived relational logic; evaluate CTEs once to bound verification."""
    with patch.object(pd, 'read_sql') as read:
        loader(None, day, cutoff)
        sql = read.call_args.args[0]
        params = read.call_args.kwargs['params']
    fields = {
        'nfl_rosters': 'season week team_abbr player_id player_name player_name_norm position depth_chart_position roster_status status_description_abbr years_exp height weight updated_at_utc',
        'nfl_depth_charts': 'season week snapshot_ts_utc team_abbr player_id player_name_norm pos_grp pos_name pos_abb pos_slot pos_rank updated_at_utc',
        'nfl_injuries': 'season week team_abbr player_id player_name_norm position report_primary_injury report_status practice_status updated_at_utc',
    }
    # Expanding a composite (jsonb_populate_record(...)).* executes the same
    # conversion per field. A lateral expansion has identical values once;
    # unused raw payload columns do not participate in the reference algorithm.
    for table, names in fields.items():
        columns=','.join('r.'+name for name in names.split())
        sql=sql.replace(f'SELECT * FROM raw.{table}_at(%(context_cutoff_utc)s)',
            f"""SELECT {columns} FROM raw.nfl_context_observations o
            CROSS JOIN LATERAL jsonb_populate_record(NULL::raw.{table},o.payload) r
            WHERE o.kind='{table}' AND o.observed_at<=%(context_cutoff_utc)s""")
    for name in ('latest_roster_season','latest_depth_season','latest_injury_season',
                 'latest_rosters','latest_depth','latest_injuries','injury_scores','teammate_injury_context'):
        sql = sql.replace(name + ' AS (', name + ' AS MATERIALIZED (')
    # The old ORDER BY left ties unspecified. Apply the same deterministic tie
    # rule to both algorithms; keep the archived per-player lookup otherwise.
    sql = sql.replace('pos_rank NULLS LAST\n', """pos_rank NULLS LAST,
                CASE WHEN UPPER(pos_abb) IN ('QB','RB','WR','TE') THEN 0 ELSE 1 END,
                updated_at_utc DESC, pos_abb, pos_slot, pos_name, pos_grp
""")
    sql = sql.replace('ORDER BY prev.snapshot_ts_utc DESC LIMIT 1', """ORDER BY prev.snapshot_ts_utc DESC NULLS LAST,
                prev.pos_rank NULLS LAST,
                CASE WHEN UPPER(prev.pos_abb) IN ('QB','RB','WR','TE') THEN 0 ELSE 1 END,
                prev.updated_at_utc DESC, prev.pos_abb, prev.pos_slot, prev.pos_name, prev.pos_grp LIMIT 1""")
    if slate_only:
        sql += """ WHERE r.roster_team_abbr IN (
            SELECT home_team_abbr FROM raw.nfl_games WHERE game_date_et=%(game_date)s
            UNION SELECT away_team_abbr FROM raw.nfl_games WHERE game_date_et=%(game_date)s)"""
    return pd.read_sql(sql, conn, params=params)


def compare_context(left, right):
    columns=list(left.columns)
    left=left.sort_values(columns).reset_index(drop=True)
    right=right.sort_values(columns).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left,right,check_dtype=False)
        return True, []
    except AssertionError:
        differences=[]
        for column in columns:
            a,b=left[column],right[column]
            if len(a)!=len(b):
                return False,[dict(reason='row_count_mismatch',current=len(a),reference=len(b))]
            same=a.eq(b) | (a.isna() & b.isna())
            if not same.all():
                differences.append(dict(column=column,rows=int((~same).sum()),
                    sample=pd.DataFrame({'player_id':left.player_id,'current':a,'reference':b}).loc[~same].head(5).to_dict('records')))
        return False,differences


def run(day, cutoff, reference=None, query_memory_mb=None, reference_materialized=False, slate_only=False):
    outputs={};stages=[]
    loaders={'current':props._load_player_context}
    if reference:
        loaders['archived_reference']=archived_candidate(reference).__globals__['_load_player_context']
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute('SELECT home_team_abbr,away_team_abbr FROM raw.nfl_games WHERE game_date_et=%s',(day,))
            teams={team for row in cur.fetchall() for team in row}
        for name,loader in loaders.items():
            started=perf_counter()
            try:
                with conn.cursor() as cur:cur.execute("SET LOCAL statement_timeout='180s'")
                if query_memory_mb:
                    with conn.cursor() as cur:
                        cur.execute("SELECT set_config('work_mem',%s,true)",(f'{int(query_memory_mb)}MB',))
                outputs[name]=(materialized_reference(loader,conn,day,cutoff,slate_only)
                    if name=='archived_reference' and reference_materialized else loader(conn,day,cutoff))
                if slate_only:
                    outputs[name]=outputs[name].loc[outputs[name].roster_team_abbr.isin(teams)]
                stages.append(dict(stage=name,seconds=perf_counter()-started,rows=len(outputs[name]),status='ok'))
            except psycopg2.Error as exc:
                stages.append(dict(stage=name,seconds=perf_counter()-started,status=type(exc).__name__))
                conn.rollback()
            print(json.dumps(stages[-1]),flush=True)
    equivalent=None;differences=[]
    if len(outputs)==2:
        equivalent,differences=compare_context(outputs['current'],outputs['archived_reference'])
    report=clean(dict(date=str(day),cutoff=cutoff.isoformat(),built_at=datetime.now(timezone.utc).isoformat(),
        stages=stages,reference_fingerprint=reference,identical_context_values=equivalent,
        query_memory_mb=query_memory_mb,
        reference_materialized=reference_materialized,
        reference_tie_rule='deterministic_date_rank_role_observed' if reference_materialized else 'archived_unspecified',
        reference_execution='single_json_expansion_narrow_ctes' if reference_materialized else 'unchanged',
        context_scope='slate_teams' if slate_only else 'all_teams',
        differences=differences,
        predictions_locked=False,discord_sent=False))
    atomic_json(ROOT/'reports/nfl_prediction_runtime_latest.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--date',type=date.fromisoformat,required=True)
    p.add_argument('--cutoff',type=datetime.fromisoformat,required=True)
    p.add_argument('--reference')
    p.add_argument('--query-memory-mb',type=int,choices=(16,32,64,128))
    p.add_argument('--reference-materialized',action='store_true',
        help='Materialize archived CTEs without changing their relational logic')
    p.add_argument('--slate-only',action='store_true',help='Compare the active slate teams only')
    a=p.parse_args()
    if a.cutoff.tzinfo is None:p.error('cutoff must include timezone')
    r=run(a.date,a.cutoff,a.reference,a.query_memory_mb,a.reference_materialized,a.slate_only)
    print(json.dumps(r))
    if (a.reference and r['identical_context_values'] is not True) or r['stages'][0]['status']!='ok':raise SystemExit(1)
