from datetime import datetime, timedelta, timezone

import pytest

from nfl_slate_readiness import news_for_game, game_report
from nfl_exact_close_review import detail

NOW = datetime(2026, 11, 8, 17, 20, tzinfo=timezone.utc)
GAME = dict(game_id='g', home_team_abbr='ATL', away_team_abbr='CIN', start_ts_utc=NOW+timedelta(minutes=40))
FORECAST = dict(id=1,game_id='g',player_name='J.Chase',offer_player_name_norm='jamarr chase',team_abbr='CIN',
                stat='receiving_yards',injury_report_status='Questionable',created_at_utc=NOW-timedelta(minutes=80))


def observation(**extra):
    return dict(dict(observed_at=NOW.isoformat(),status='observed',rows=[
        dict(team='CIN',name='JaMarr Chase',name_key='jamarr chase',status='out')]),**extra)


def test_late_out_is_reported_without_mutating_the_frozen_forecast():
    news = news_for_game(GAME,[FORECAST],observation(),NOW)
    assert news['adverse_changes_after_forecast'][0]['forecast_id']==1
    assert FORECAST['injury_report_status']=='Questionable'
    assert not news['complete_official_inactives_verified']


@pytest.mark.parametrize('extra', [dict(rows=[]),dict(status='unknown'),
    dict(observed_at=(NOW-timedelta(minutes=11)).isoformat()),dict(observed_at=(NOW+timedelta(seconds=1)).isoformat())])
def test_absence_stale_failed_or_future_news_never_clears_a_player(extra):
    news=news_for_game(GAME,[FORECAST],observation(**extra),NOW)
    assert news['adverse_changes_after_forecast']==[]
    assert not news['complete_official_inactives_verified']


def test_future_scan_is_pending_while_due_missing_quotes_are_reported():
    def result(game):
        return game_report(game,[FORECAST],[],[],{},dict(checkpoints=[]),{},NOW)
    future=result(dict(GAME,start_ts_utc=NOW+timedelta(hours=4)))
    assert future['warnings']==[] and future['phase']=='waiting'
    due=result(GAME)
    assert 'fresh_two_book_spread_quotes_missing' in due['warnings']
    assert 'live_player_status_unknown' in due['warnings']


def test_news_observed_before_a_forecast_is_not_described_as_new_late_news():
    f=dict(FORECAST,created_at_utc=NOW)
    old=observation(observed_at=(NOW-timedelta(seconds=60)).isoformat())
    assert not news_for_game(GAME,[f],old,NOW)['adverse_changes_after_forecast']


def test_finished_game_does_not_lose_missed_checkpoint_status():
    result=game_report(dict(GAME,start_ts_utc=NOW-timedelta(minutes=5)),[],[],[],{},
                       dict(checkpoints=[dict(offset_minutes=5,status='missed')]),{},NOW)
    assert result['warnings']==['checkpoint_T5_unverified']


def test_estimated_close_cannot_become_observed_exact_evidence():
    row=dict(game_id='g',player_name_norm='player',stat='receiving_yards',side='over',book='fanduel',
             fd_line=40.5,fd_price=-110,commence_time_utc=NOW,sharp_close_line=42.5,
             sharp_close_method='shifted',exact_reference_close=None,exact_execution_close=None)
    result=detail(row)
    assert result['exact_reference_close'] is None
    assert result['reference_gap_reason']=='different_reference_line_observed'
    assert result['estimated_conversion_excluded']
