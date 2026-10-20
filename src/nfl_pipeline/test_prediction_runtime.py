from datetime import date, datetime, timezone
from unittest.mock import Mock

import pandas as pd
import pytest

from nfl_pipeline.runtime_timing import StageTimer
from nfl_pipeline.modeling import predict_player_props as props
from nfl_pipeline.prediction_runtime_audit import compare_context


def test_failed_stage_is_logged_without_swallowing_error():
    timer=StageTimer(Mock())
    def fail():raise ValueError('failed')
    with pytest.raises(ValueError):timer.call('context',fail)
    assert timer.stages[0]['status']=='failed' and timer.stages[0]['seconds']>=0


def test_no_games_does_not_read_history_or_lock_anything(monkeypatch, capsys):
    conn=Mock();manager=Mock()
    manager.__enter__=Mock(return_value=conn);manager.__exit__=Mock(return_value=False)
    monkeypatch.setattr(props.psycopg2,'connect',lambda *a,**k:manager)
    monkeypatch.setattr(props,'_load_model',lambda cfg:{})
    monkeypatch.setattr(props,'ensure_schema',lambda c:None)
    monkeypatch.setattr(props,'_prediction_context_cutoff',lambda *a:datetime(2026,10,6,tzinfo=timezone.utc))
    monkeypatch.setattr(props,'_load_games',lambda *a:pd.DataFrame(columns=['game_id','start_ts_utc']))
    history=Mock(side_effect=AssertionError('no history expected'))
    save=Mock(side_effect=AssertionError('no writes expected'))
    monkeypatch.setattr(props,'_load_history',history);monkeypatch.setattr(props,'_save_predictions',save)
    rows,meta=props.build_predictions(props.PredictConfig(et_date=date(2026,10,6)))
    assert not rows and meta['status']=='no_upcoming_games'
    history.assert_not_called();save.assert_not_called()
    props.print_discord(rows,meta,props.PredictConfig())
    assert 'No NFL rows available' in capsys.readouterr().out


def test_context_comparison_is_order_independent_but_preserves_duplicates():
    frame=pd.DataFrame(dict(player_id=['a','a','b'],roster_team_abbr=['NO']*3,
        pos_rank=[1.,2.,1.],depth_observed_at=pd.to_datetime(['2026-10-01']*3,utc=True)))
    assert compare_context(frame,frame.iloc[::-1])[0]
    changed=frame.copy();changed.loc[0,'pos_rank']=3.
    same,differences=compare_context(frame,changed)
    assert not same and any(d['column']=='pos_rank' for d in differences)


def test_context_query_evaluates_season_aggregates_once_and_breaks_depth_ties(monkeypatch):
    read=Mock(return_value=pd.DataFrame())
    monkeypatch.setattr(props.pd,'read_sql',read)
    props._load_player_context(None,date(2026,10,5),datetime(2026,10,5,tzinfo=timezone.utc))
    sql=read.call_args.args[0]
    for kind in ('roster','depth','injury'):
        assert f'latest_{kind}_season AS MATERIALIZED (' in sql
    assert sql.count("CASE WHEN UPPER(pos_abb) IN ('QB','RB','WR','TE') THEN 0 ELSE 1 END")==3
    assert 'FROM depth_observations prev' not in sql
