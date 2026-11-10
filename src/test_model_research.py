from datetime import date,datetime,timedelta,timezone
import os
import pandas as pd
import pytest

import model_research as mr
import model_research_report as report
import model_research_checks as checks
from sharp_pipeline.nba_experiment import check_evidence
from sharp_pipeline.test_database_repairs import database

NOW=datetime(2026,11,1,20,tzinfo=timezone.utc)
DOC=dict(sha256='a'*64,registered_at=(NOW-timedelta(days=1)).isoformat())


def offer(**changes):
    return dict(dict(sport='nba',event_id='event',game_id='game',market='points',player='Test Player',player_norm='test player',
        home_team='Boston Celtics',away_team='New York Knicks',side='over',line=20.5,price=110,book='fanduel',
        fair_probability=.5,ev=.05,method='exact',decided_at=NOW,commence_time_utc=NOW+timedelta(minutes=30),
        parent_check_id=None),**changes)


def forecast(**changes):
    return dict(dict(run_at_utc=NOW-timedelta(minutes=10),model_version='test',identity_verified=True,available=True,
                     mu=28,disp_a=1,disp_b=0,pred_margin=6,pred_total=220),**changes)


def test_player_model_really_changes_selection_at_identical_market_price():
    assert mr.assess(offer(),forecast())['selected']
    veto=mr.assess(offer(),forecast(mu=12))
    assert not veto['selected'] and veto['reason']=='model_nonpositive_ev'
    assert mr.assess(offer(side='under'),forecast(mu=12))['selected']
    assert not mr.assess(offer(),None)['selected']


@pytest.mark.parametrize('changes,reason',[
    ({'run_at_utc':NOW+timedelta(seconds=1)},'forecast_after_offer_or_stale'),
    ({'run_at_utc':NOW-timedelta(hours=13)},'forecast_after_offer_or_stale'),
    ({'identity_verified':False},'forecast_identity_or_version_missing'),
    ({'available':False},'participation_not_supported'),
    ({'disp_a':float('nan')},'invalid_model_distribution')])
def test_forecast_guards(changes,reason):
    assert mr.assess(offer(),forecast(**changes))['reason']==reason


def test_game_predictions_apply_correct_home_and_away_handicap_signs():
    assert mr.assess(offer(market='spread',side='home',line=-3.5),forecast())['selected']
    assert not mr.assess(offer(market='spread',side='away',line=3.5),forecast())['selected']
    total=mr.assess(offer(market='total',side='under',line=225.5),forecast())
    assert total['selected'] and total['directional_margin']==5.5
    assert total['model_probability'] is None  # no invented probability from a score projection


def test_nfl_requires_independent_exact_line_probability():
    row=offer(sport='nfl',market='receiving_yards',line=50.5)
    f=forecast(line=50.5,raw_over_probability=.6)
    assert mr.assess(row,f)['selected']
    assert mr.assess(row,dict(f,line=51.5))['reason']=='model_probability_wrong_line'
    assert mr.assess(row,dict(f,raw_over_probability=None))['reason']=='independent_model_probability_missing'


@pytest.mark.parametrize('changes',[
    {'book':'draftkings'}, {'method':'shifted'}, {'line':20.0}, {'price':-99},
    {'decided_at':NOW-timedelta(minutes=6)}, {'commence_time_utc':NOW},
    {'ev':.02}, {'decided_at':NOW+timedelta(seconds=1)}])
def test_capture_does_not_backfill_or_relax_price_guards(changes):
    assert not mr.eligible(offer(**changes),DOC['registered_at'],NOW)


def test_registration_boundary():
    assert mr.eligible(offer(),DOC['registered_at'],NOW)
    assert not mr.eligible(offer(),(NOW+timedelta(seconds=1)).isoformat(),NOW)


def metric_row(i=0,**changes):
    day=date(2026,10,1)+timedelta(days=i)
    row=dict(sport='nba',family='props',event_id=str(i),day=day,selected=True,reason='model_positive_ev',
        close_method='exact',close_ev=.04,checked_price_ev=.03,opportunity_ev=.03,
        timely_check=True,exact_offer_available=True,checked_at=NOW,check_status='available',
        result='win',paper_profit=1.1,model_probability=.55,fair_probability=.5)
    return dict(row,**changes)


def test_coverage_cannot_pass_with_inferred_closes_or_missing_live_checks():
    f=pd.DataFrame([metric_row(i,close_method='shifted',close_ev=None,checked_price_ev=None,
        opportunity_ev=None,timely_check=False,exact_offer_available=False,checked_at=None) for i in range(60)])
    result=report.summarize(f)
    assert result['groups']['nba_props']['model_assisted']['status']=='collecting'
    assert result['economics']['projected_net_usd'] is None
    assert result['betting_authorized'] is False


def test_monthly_cost_once_and_unavailable_offer_is_zero_opportunity():
    f=pd.DataFrame([metric_row(i,sport='nba' if i%2 else 'nfl') for i in range(30)])
    f.loc[0,'opportunity_ev']=0.
    result=report.economics(f)
    assert result['projected_net_usd']==pytest.approx(29*.03-30)
    f.loc[1,'opportunity_ev']=None
    assert report.economics(f)['projected_net_usd'] is None


def test_checked_price_can_erase_signal_and_late_check_is_not_evidence():
    row=dict(offer(),delivery_status='sent',sent_at=NOW,checked_at=NOW+timedelta(seconds=45),
        recorded_at=NOW+timedelta(seconds=60),check_status='available',displayed_line=20.5,displayed_price=-120,
        evidence_source='manual_live',close_method='exact',close_probability=.5)
    evidence=check_evidence(row)
    assert evidence['timely_check'] and evidence['checked_price_ev']<0
    row['checked_at']=NOW+timedelta(minutes=6);row['recorded_at']=NOW+timedelta(minutes=7)
    assert not check_evidence(row)['timely_check']


def test_model_does_not_pass_incremental_review_merely_by_selecting_everything():
    f=pd.DataFrame([metric_row(i) for i in range(60)])
    group=report.summarize(f)['groups']['nba_props']
    assert group['model_assisted']['status']=='review_only'
    assert group['paired_improvement_by_game']['lower_95']==0
    assert not group['model_adds_value_review']


@pytest.mark.skipif(os.getenv('RUN_DATABASE_REPAIR_TESTS')!='1',reason='opt-in isolated PostgreSQL test')
def test_append_only_decision_first_check_and_delivery(database):
    with database.cursor() as cur:cur.execute(mr.SCHEMA)
    first=mr.save(database,DOC,offer(),mr.assess(offer(),forecast()),NOW)
    assert first
    # A later forecast cannot replace the first selected/vetoed decision.
    assert mr.save(database,DOC,offer(price=115),mr.assess(offer(),forecast(mu=12)),NOW) is None
    messages=[]
    result=checks.dispatch(database,DOC,'nba',now=NOW,poster=lambda s:messages.append(s) or 'receipt')
    assert result['sent']==1 and 'PAPER PRICE CHECK' in messages[0] and 'model EV' in messages[0]
    assert checks.dispatch(database,DOC,'nba',now=NOW,poster=lambda s:'wrong')['sent']==0
    checks.record(database,first,checked_at=NOW+timedelta(seconds=20),now=NOW+timedelta(seconds=25),
                  status='available',line=20.5,price=-130)
    with pytest.raises(ValueError,match='First check'):
        checks.record(database,first,checked_at=NOW+timedelta(seconds=40),now=NOW+timedelta(seconds=45),
                      status='available',line=20.5,price=120)
    assert mr.rows(database,'SELECT displayed_price FROM bets.model_research_checks')[0]['displayed_price']==-130
    with database.cursor() as cur:
        cur.execute('SAVEPOINT immutable')
        with pytest.raises(Exception,match='append-only'):
            cur.execute("UPDATE bets.model_research_decisions SET payload='{}'")
        cur.execute('ROLLBACK TO SAVEPOINT immutable')


def test_spread_pair_rejects_stale_mismatched_and_tampered_quotes():
    def q(team,line,price,at=NOW):
        body=dict(book='pinnacle',player_name_norm=team,line=line,
                  quote=dict(price=price,market_updated_at=at.isoformat(),book_updated_at=NOW.isoformat()))
        return dict(body,observation_id=mr.registry._digest(body))
    h,a=q('GB',-3.5,-110),q('CAR',3.5,-110)
    assert mr.spread_pair([h,a],'pinnacle','GB','CAR',-3.5,NOW)
    assert mr.spread_pair([h,q('CAR',3.5,-110,NOW-timedelta(minutes=6))],'pinnacle','GB','CAR',-3.5,NOW) is None
    assert mr.spread_pair([h,q('CAR',4.5,-110)],'pinnacle','GB','CAR',-3.5,NOW) is None
    assert mr.spread_pair([h,dict(a,line=4.5)],'pinnacle','GB','CAR',-3.5,NOW) is None


@pytest.mark.skipif(os.getenv('RUN_DATABASE_REPAIR_TESTS')!='1',reason='opt-in isolated PostgreSQL test')
def test_nba_identity_matches_bounded_tip_difference_but_rejects_ambiguity(database):
    from nba_props.live import SCHEMA_SQL,MODEL_VERSION
    with database.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        cur.execute("""INSERT INTO nbaprops.game_projections
            (run_at_utc,game_date,game_id,tip_utc,home_team,away_team,pred_margin,pred_total,model_version)
            VALUES(%s,%s,'native',%s,'BOS','NYK',6,220,%s)""",
            (NOW-timedelta(minutes=5),NOW.date(),NOW+timedelta(minutes=20),MODEL_VERSION))
    row=offer(market='spread',side='home',line=-3.5)
    found=mr.forecast_for(database,row)
    assert found['game_id']=='native' and mr.assess(row,found)['selected']
    assert mr.forecast_for(database,dict(row,away_team='Miami Heat')) is None
    with database.cursor() as cur:
        cur.execute("""INSERT INTO nbaprops.game_projections
            (run_at_utc,game_date,game_id,tip_utc,home_team,away_team,pred_margin,pred_total,model_version)
            VALUES(%s,%s,'ambiguous',%s,'BOS','NYK',6,220,%s)""",
            (NOW-timedelta(minutes=5),NOW.date(),NOW+timedelta(minutes=20),MODEL_VERSION))
    assert mr.forecast_for(database,row) is None


def test_empty_report_and_cost_projection_use_elapsed_days_not_only_alert_dates():
    assert report.summarize(pd.DataFrame())['economics']['projected_net_usd'] is None
    f=pd.DataFrame([metric_row(i) for i in range(30)])
    assert report.economics(f,collection_days=60)['projected_net_usd']==pytest.approx(.45-30)


@pytest.mark.skipif(os.getenv('RUN_DATABASE_REPAIR_TESTS')!='1',reason='opt-in isolated PostgreSQL test')
def test_ambiguous_delivery_never_retries_and_expired_does_not_post(database):
    with database.cursor() as cur:cur.execute(mr.SCHEMA)
    mr.save(database,DOC,offer(),mr.assess(offer(),forecast()),NOW)
    def fail(message):raise TimeoutError('uncertain receipt')
    assert checks.dispatch(database,DOC,'nba',now=NOW,poster=fail)['sent']==0
    assert checks.dispatch(database,DOC,'nba',now=NOW,poster=lambda s:'unexpected')['sent']==0
    mr.save(database,DOC,offer(event_id='second'),mr.assess(offer(),forecast()),NOW)
    assert checks.dispatch(database,DOC,'nba',now=NOW+timedelta(minutes=6),poster=lambda s:'unexpected')['sent']==0
    assert {r['delivery_status'] for r in mr.rows(database,'SELECT delivery_status FROM bets.model_research_delivery')}=={'uncertain','expired'}


def test_official_final_score_is_required_and_partial_score_is_not_final():
    from model_research_results import final_game
    g=dict(gameId='native',gameStatus=3,gameTimeUTC=NOW.isoformat(),
           homeTeam=dict(teamTricode='BOS',score=110),awayTeam=dict(teamTricode='NYK',score=105))
    assert final_game(g)['status']=='final'
    assert final_game(dict(g,gameStatus=2)) is None
    assert final_game(dict(g,awayTeam=dict(teamTricode='NYK'))) is None
    assert final_game(dict(g,awayTeam=dict(teamTricode='NYK',score=110))) is None


@pytest.mark.skipif(os.getenv('RUN_DATABASE_REPAIR_TESTS')!='1',reason='opt-in isolated PostgreSQL test')
def test_official_results_and_missing_player_stay_separate(database):
    from model_research_results import SCHEMA,refresh,lookup
    from nba_props.ingest import SCHEMA_SQL
    from datetime import timedelta
    at=NOW-timedelta(days=1)
    o=offer(decided_at=at,commence_time_utc=at+timedelta(minutes=30),market='spread',side='home',line=-3.5)
    with database.cursor() as cur:
        cur.execute(mr.SCHEMA);cur.execute(SCHEMA);cur.execute(SCHEMA_SQL)
    mr.save(database,DOC,o,mr.assess(o,forecast(run_at_utc=at-timedelta(minutes=5))),at)
    g=dict(gameId='native',gameStatus=3,gameTimeUTC=(at+timedelta(minutes=20)).isoformat(),
           homeTeam=dict(teamTricode='BOS',score=110),awayTeam=dict(teamTricode='NYK',score=105))
    result=refresh(database,DOC,fetcher=lambda day:[g],now=NOW)
    assert result['new_final_snapshots']==1 and lookup(database,o)['game_id']=='native'
    assert report.verified_result(database,o,None)=='win'
    assert report.verified_result(database,dict(o,market='points',side='under',line=20.5),None) is None
    assert refresh(database,DOC,fetcher=lambda day:pytest.fail('Final should not be polled again'),now=NOW)['dates_requested']==0
