from collections import defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import json
import re

import pytest

import prediction_ledger as ledger


def resolver(minutes=25,points=0):
    r=ledger.Results.__new__(ledger.Results)
    r.nba_by_match=defaultdict(list)
    r.nba_by_match[('2026-11-04','DET','PHI')]=[dict(game_id='g',players=[
        dict(player_id=1,player_name='Test Player',minutes=minutes,pts=points)],
        teams=[dict(home=True,pts=100),dict(home=False,pts=105)])]
    r.nfl={};r.nfl_stats=defaultdict(list);r.ncaa=[]
    return r


def offer(**kwargs):
    return dict(dict(sport='nba',event_id='e',game_id='g',player='Test Player',market='points',side='under',line=10.5,price=-110,
        home_team='DET',away_team='PHI',commence_time_utc='2026-11-05T00:00:00+00:00',decided_at='2026-11-04T23:30:00+00:00',book='fanduel'),**kwargs)


def test_side_and_spread_signs_pushes_and_missing_results():
    assert ledger.grade('points','under',10.5,0)=='win'
    assert ledger.grade('points','under',10.5,None)=='pending'
    assert ledger.grade('spread','away',3.5,3)=='win'
    assert ledger.grade('spread','away',3.5,4)=='loss'
    assert ledger.grade('spread','home',-3,3)=='push'
    assert ledger.grade('total','over',200,200)=='push'
    assert ledger.grade('points',None,None,10)=='no_line'


def test_team_abbreviations_and_full_names_match_same_verified_result():
    r=resolver()
    assert r.actual(offer())[0]==0
    assert r.actual(offer(home_team='Detroit Pistons',away_team='Philadelphia 76ers'))[0]==0
    assert r.actual(offer(game_id='other'))[1]=='pending'


@pytest.mark.parametrize('minutes,points',[(None,0),(0,0),(25,None)])
def test_missing_participation_or_stat_cannot_win_an_under(minutes,points):
    row=ledger.entry(resolver(minutes,points),'test','one',offer())
    assert row['result']=='pending' and row['paper_units'] is None


def test_late_predictions_are_excluded_even_if_they_would_win():
    row=ledger.entry(resolver(),'test','one',offer(decided_at='2026-11-05T00:01:00+00:00'))
    assert row['result']=='excluded' and row['paper_units'] is None


def test_first_pick_survives_later_opposite_side_and_book_versions_stay_separate():
    first=ledger.entry(resolver(),'test','first',offer(),version='v1')
    later=ledger.entry(resolver(),'test','later',offer(side='over',line=0.5,decided_at='2026-11-04T23:40:00+00:00'),version='v1')
    other=dict(first,id='other',book='draftkings');version=dict(first,id='new',version='v2')
    bad=dict(first,id='bad',result='excluded',forecast_at='2026-11-04T23:00:00+00:00')
    result=ledger.first_decisions([later,first,other,version,bad])
    assert len(result)==4
    kept=next(r for r in result if r['id']=='first')
    assert kept['result']=='win' and kept['side']=='under' and kept['repeated_observations']==1


def test_hit_rate_does_not_count_push_void_pending_or_no_line():
    base=ledger.entry(resolver(),'test','one',offer())
    rows=[dict(base,result=k,paper_units=None) for k in ('win','loss','push','void','pending','no_line','excluded')]
    s=ledger.summary(rows)
    assert s['accuracy']==.5 and s['accuracy_denominator']==2
    assert s['pending']==1 and s['void']==1 and s['excluded']==1
    assert ledger.paper_profit('win',150)==1.5
    assert ledger.paper_profit('win',None) is None


def test_nfl_verified_void_conflicts_and_missing_participation():
    r=resolver();r.nfl['g']=dict(status='final',start_ts_utc=datetime(2026,11,5,tzinfo=timezone.utc),home_score=21,away_score=17)
    o=offer(sport='nfl',market='receiving_yards')
    assert r.actual(o)[1]=='pending'
    key=('g','test player','receiving_yards')
    r.nfl_stats[key]=[dict(actual_stat=None,result='void_nonparticipant')]
    assert r.actual(o)[1]=='void'
    r.nfl_stats[key].append(dict(actual_stat=0,result='win'))
    assert r.actual(o)[1]=='pending'


def test_ncaa_verified_aliases_and_total_push():
    r=resolver();r.ncaa=[dict(start_date=datetime(2026,11,5,tzinfo=timezone.utc),home_team="Hawai'i",away_team='Massachusetts',completed=True,home_points=24,away_points=20)]
    o=offer(sport='ncaaf',market='total',home_team='Hawaii Rainbow Warriors',away_team='UMass Minutemen',side='over',line=44)
    row=ledger.entry(r,'test','ncaa',o)
    assert row['actual']==44 and row['result']=='push'


def test_html_escapes_data_script_and_preserves_json(tmp_path):
    doc=dict(built_at='now',since='2026-01-01',through='2026-11-05',notes=['</script><img src=x>'],records=[],summaries=[])
    ledger.write_reports(doc,tmp_path)
    page=(tmp_path/'predictions_ledger_latest.html').read_text(encoding='utf-8')
    assert '<img src=x>' not in page
    payload=re.search(r'<script id="data" type="application/json">(.*?)</script>',page,re.S).group(1)
    assert json.loads(payload)['notes']==doc['notes']
