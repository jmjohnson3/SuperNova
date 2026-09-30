"""Exercise production boundaries without provider calls or Discord posts."""
from datetime import date, datetime, timedelta, timezone

import pytest

from nfl_pipeline import forecast_store, refresh_lock_quotes
from nfl_pipeline.clv_report import classify_prop_close, _prob_delta
from nfl_pipeline.close_capture_diagnostic import summarize
from nfl_pipeline.discord_matchups import build_bundle, preview_markdown, selected_props
from nfl_pipeline.offer_selection import CONTRACT, eligible_player_offers
from nfl_pipeline.modeling import predict_player_props as live
from nfl_pipeline.modeling import scoring_capture, scoring_versions, benchmark_offers
from nfl_pipeline.test_integrity import Connection
from nfl_pipeline.test_prospective_market_trial import sgo_event
from nfl_pipeline.crawler_oddsapi import _canonical_event_from_sgo
from nfl_pipeline.parse_oddsapi import _rows_from_event
from nfl_pipeline.lock_ledger import _insert_rows

NOW = datetime(2099, 9, 24, 16, tzinfo=timezone.utc)
START = NOW + timedelta(hours=4)


def player():
    return dict(game_id='game', game_date_et=NOW.date(), season=2099, week=3,
        player_id='player', player_name='Test Receiver', position='WR',
        team_abbr='MIA', opponent_abbr='BUF', start_ts_utc=START)


def offer(**changes):
    return dict(dict(offer_id=1, provider='sportsgameodds', event_id='e',
        fetched_at_utc=NOW-timedelta(minutes=1), commence_time_utc=START,
        home_team='Miami Dolphins', away_team='Buffalo Bills', bookmaker_key='fanduel',
        player_name='Test Receiver', player_name_norm='test receiver', stat='receiving_yards',
        market_key='player_reception_yds', line=50.5, over_price=110, under_price=-130,
        over_link='https://sportsbook.fanduel.com/addToBetslip?marketId=42.101&selectionId=11',
        under_link='https://sportsbook.fanduel.com/addToBetslip?marketId=42.101&selectionId=12'), **changes)


def score(q):
    metrics = dict(accepted_live=True, accepted=True, live_model_version='production', mae=15.)
    dist = dict(kind='empirical_oof_residual', residual_quantiles=list(range(-40, 41)))
    r = live._candidate_from_offer(player(), 'receiving_yards', 65., 60., metrics, q, dist)
    r.update(execution_contract=CONTRACT, quote_fetched_at_utc=q['fetched_at_utc'],
        prediction_context_cutoff_utc=NOW.isoformat(), start_ts_utc=START,
        forecast_features=player(), scoring_replay=scoring_capture.capture(
            player(), 'receiving_yards', 65., 60., metrics, q, dist, {}, {}, .02, {'version':'production'}))
    return r


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW + timedelta(seconds=1)


def test_fresh_fanduel_workflow_through_exact_alternate_close_and_evaluation(monkeypatch):
    qs, issues = eligible_player_offers([offer(), offer(bookmaker_key='draftkings', over_price=250),
        offer(offer_id=3, line=60.5), offer(offer_id=4, line=40.5, fetched_at_utc=NOW-timedelta(hours=8))],
        player(), 'receiving_yards', NOW, now=NOW)
    assert [q['line'] for q in qs] == [50.5, 60.5]
    assert issues == {'other_book':1, 'stale_quote_refresh_required':1}
    rows = [score(q) for q in qs]
    assert any(r['tier']=='micro_projection' for r in rows)
    live._apply_daily_micro_cap(rows, 5)
    chosen = next(r for r in rows if r['tier']=='micro_projection')
    assert sum(r['tier']=='micro_projection' for r in rows)==1

    monkeypatch.setattr(forecast_store, 'datetime', Clock)
    saved = []
    monkeypatch.setattr(forecast_store.psycopg2.extras, 'execute_values', lambda c,q,r: saved.extend(r))
    conn = Connection(['game'])
    fields = ['game_date_et','game_id','line','price','prediction_key']
    forecast_store.save_forecasts(conn, 'nfl_player_prop_predictions', rows, fields)
    original = [(*r[:-1], r[-1].getquoted()) for r in saved]
    assert len(saved)==2 and conn.commits==1
    conn.cur.fetchone = lambda: (0,)
    conn.cur.rowcount = 1
    entry = ('prop',1,NOW.date(),'micro_projection',1.,chosen['book'],None,chosen['stat'],
             chosen['side'],chosen['line'],chosen['price'],chosen['link'],'production',chosen['prediction_key'])
    assert _insert_rows(conn,[entry])==1
    assert any('INSERT INTO bets.nfl_bet_ledger' in call[0] for call in conn.cur.calls)
    from nfl_pipeline import publish_forecasts
    conn.cur.fetchall = lambda: [(dict(chosen), True)]
    monkeypatch.setattr(publish_forecasts, 'datetime', Clock)
    displayed = publish_forecasts.display_rows(conn, NOW.date(), 'prop')
    assert displayed[0]['ledger_locked']
    schedule = [dict(game_id='game', start_ts_utc=START, home_team_abbr='MIA', away_team_abbr='BUF')]
    text = preview_markdown(build_bundle(NOW.date(), schedule, [], displayed, ['production'], now=NOW))
    assert '$1 MICRO TEST' in text and 'fanduel.com' in text and 'draftkings' not in text
    assert '[Add to slip]' in text and '[Provider link]' in text
    from nfl_pipeline.fanduel_links import selections, single_betslip_url
    assert selections(single_betslip_url(chosen['link'])) == selections(chosen['link'])

    event = sgo_event(); event['startsAt'] = START.isoformat()
    normalized = _canonical_event_from_sgo(event, include_alts=True)
    diag = dict(events_seen=0, bookmaker_entries=0, events_with_zero_books=0,
        market_entries=0, outcomes_seen=0, unknown_markets={}, skipped_outcomes={})
    close_at = START-timedelta(minutes=20)
    close_rows = _rows_from_event('close', NOW.date(), close_at, normalized, diag, 'sportsgameodds')
    c = next(r for r in close_rows if r[14]==chosen['line'])
    close = close_record(chosen, fetched_at_utc=close_at, line=c[14], over_price=c[15], under_price=c[16])
    quality = classify_prop_close(close)
    assert quality['valid'] and quality['available'] is True
    assert _prob_delta(chosen['price'], quality['close_price']) is not None
    assert [(*r[:-1], r[-1].getquoted()) for r in saved]==original

    locked = dict(id=1, game_id='game', player_id='player', stat='receiving_yards', season=2099, week=3,
        created_at_utc=NOW+timedelta(seconds=1), start_ts_utc=START,
        offer_fetched_at=chosen['quote_fetched_at_utc'], actual=70, status='final', forecast_payload=forecast_store.clean(chosen))
    ledger = dict(ledger_id=9, prediction_id=1, ledger_book='fanduel', ledger_side=chosen['side'],
        ledger_line=chosen['line'], ledger_price=chosen['price'], ledger_model_version='production',
        locked_at_utc=NOW+timedelta(seconds=2))
    shadow = dict(forecast_id=1, variant='stable_ensemble_calibrated', stat='receiving_yards',
        production_release='production', source_context_cutoff=NOW.isoformat(), source_book='fanduel',
        source_side=chosen['side'], source_line=chosen['line'], source_price=chosen['price'],
        production_probability=chosen['probability'], same_side_probability=.60,
        raw_over_probability=.60, ledger_ids=[9], expected_yards=64, median_yards=60,
        distribution_mean=64, live_p10=20, live_p90=100, candidate_side=chosen['side'])
    doc = dict(contract=benchmark_offers.CONTRACT, run_id='pinned-test',
        scored_at=(NOW+timedelta(seconds=3)).isoformat(), artifact_created_at='2099-09-20T00:00:00Z',
        training_end='2099-09-17', rows=[shadow])
    evaluated, exclusions, pending = benchmark_offers.validate_documents([locked], [ledger], [doc])
    assert not exclusions and pending==0 and len(evaluated)==1
    metrics = next(iter(benchmark_offers.offered_metrics(evaluated).values()))
    for scope in ('real_offers','exact_micro'):
        comparison = metrics[scope]['matched_market']
        assert comparison['rows']==1
        assert comparison['production']['brier'] is not None
        assert comparison['challenger']['brier'] is not None
        assert comparison['market']['brier'] is not None


def close_record(row=None, **changes):
    row = row or score(offer())
    return dict(dict(prediction_id=1, player_name=row['player_name'], stat=row['stat'], book=row['book'],
        side=row['side'], locked_line=row['line'], locked_price=row['price'], created_at_utc=NOW,
        start_ts_utc=START, commence_time_utc=START, integrity_version='nfl-asof-v2', lock_offer_id=1,
        fetched_at_utc=None, line=None, over_price=None, under_price=None,
        fresh_book_rows=2, fresh_market_rows=2, observed_lines=[51.5],
        is_current=True, selected_real_tier=True), **changes)


def test_missing_exact_line_is_unknown_not_flat_and_future_window_is_pending():
    row = close_record()
    q = classify_prop_close(row)
    assert q['status']=='exact_line_unavailable_in_captured_feed'
    assert not q['valid'] and q['available'] is None and q['close_price'] is None
    _, scopes = summarize([row], NOW)
    assert scopes['fanduel']['valid_coverage'] is None
    assert scopes['fanduel']['phases']=={'waiting_for_window':1}
    _, scopes = summarize([row], START+timedelta(minutes=1))
    assert scopes['fanduel']['valid_coverage']==0 and not scopes['fanduel']['coverage_pass']
    flat = classify_prop_close(close_record(fetched_at_utc=START-timedelta(minutes=20),
        line=row['locked_line'], over_price=row['locked_price'], under_price=row['locked_price']))
    assert flat['valid'] and _prob_delta(row['locked_price'],flat['close_price'])==0


def test_expiration_before_save_leaves_old_locks_untouched(monkeypatch):
    row = score(offer(fetched_at_utc=NOW-timedelta(minutes=21)))
    monkeypatch.setattr(forecast_store, 'datetime', Clock)
    conn = Connection(['game'])
    with pytest.raises(ValueError, match='stale_quote'):
        forecast_store.save_forecasts(conn,'nfl_player_prop_predictions',[row],[])
    assert not conn.cur.calls and conn.commits==0


def test_stale_or_unverified_prices_cannot_reappear_in_discord(monkeypatch):
    from nfl_pipeline import publish_forecasts
    monkeypatch.setattr(publish_forecasts,'datetime',Clock)
    conn=Connection(['game'])
    stale=score(offer(fetched_at_utc=NOW-timedelta(hours=8)))
    legacy=dict(stale)
    legacy.pop('scoring_replay'); legacy.pop('execution_contract'); legacy.pop('quote_fetched_at_utc')
    conn.cur.fetchall=lambda:[(stale,True),(legacy,True)]
    excluded=[]
    assert publish_forecasts.display_rows(conn,NOW.date(),'prop',quote_exclusions=excluded)==[]
    assert {r['reason'] for r in excluded}=={'stale_quote_refresh_required','missing_quote_or_lock_timing'}
    schedule=[dict(game_id='game',start_ts_utc=START,home_team_abbr='MIA',away_team_abbr='BUF')]
    text=preview_markdown(build_bundle(NOW.date(),schedule,[],[],['production'],now=NOW,quote_exclusions=excluded))
    assert '2 saved quotes withheld' in text and '$1 MICRO TEST' not in text


@pytest.mark.parametrize('changes,reason', [
    ({'away_team':'New York Jets'}, 'offer_game_identity_mismatch'),
    ({'commence_time_utc':START+timedelta(hours=1)}, 'offer_game_identity_mismatch'),
    ({'fetched_at_utc':NOW+timedelta(seconds=1)}, 'quote_or_lock_timing_invalid'),
    ({'under_price':0,'over_price':float('nan')}, 'invalid_offer_line_or_price'),
])
def test_bad_quote_never_reaches_ranking(changes, reason):
    rows, errors = eligible_player_offers([offer(**changes)],player(),'receiving_yards',NOW,now=NOW)
    assert not rows and errors=={reason:1}


def test_invalid_opposite_side_stays_unknown_without_mutating_raw_quote():
    q = offer(under_price=0)
    rows, _ = eligible_player_offers([q],player(),'receiving_yards',NOW,now=NOW)
    assert rows[0]['under_price'] is None and q['under_price']==0
    assert score(rows[0])['market_no_vig_probability'] is None


def test_no_game_never_fetches_quotes_or_creates_discord_bets(monkeypatch):
    monkeypatch.setattr(refresh_lock_quotes,'upcoming_games',lambda day:0)
    monkeypatch.setattr(refresh_lock_quotes,'fetch_for_date',lambda *a:pytest.fail('API called on no-game date'))
    assert refresh_lock_quotes.refresh(date(2099,9,24))['status']=='no_upcoming_games'
    bundle=build_bundle(NOW.date(),[],[],[],['production'],now=NOW)
    assert not bundle['cards'] and bundle['games']==0


def test_paper_caps_choose_distinct_players_after_ranking():
    rows=[dict(score(offer(line=50.5+i)), tier='paper', ev=.5-i*.01) for i in range(12)]
    rows.append(dict(rows[0], player_id='second', player_name='Other', ev=.1))
    chosen=selected_props(rows,paper_limit=2)
    assert [r['player_id'] for _,r in chosen]==['player','second']


def test_rerun_cannot_lock_another_line_on_the_same_micro_player():
    conn=Connection(['game']); conn.cur.fetchone=lambda:(True,)
    row=score(offer())
    entry=('prop',2,NOW.date(),'micro_projection',1.,'fanduel',None,row['stat'],row['side'],
           55.5,row['price'],row['link'],'production','second-revision')
    assert _insert_rows(conn,[entry])==0
    query=conn.cur.calls[-1][0]
    assert "l.tier<>'paper'" in query and 'old.line=new.line' not in query
    assert 'old.stat=new.stat' not in query


def test_scoring_archive_is_verified_and_replays_original_probability(tmp_path,monkeypatch):
    captured=score(offer())['scoring_replay']
    expected=scoring_capture.replay(captured)
    fingerprint=scoring_versions.archive_current()
    assert scoring_versions.archived_candidate(fingerprint)(
        captured['row'],captured['stat'],captured['projection'],captured['baseline'],
        captured['metrics'],captured['offer'],captured['distribution'])['probability']==expected['probability']
    monkeypatch.setattr(scoring_versions,'STORE',tmp_path)
    with pytest.raises(ValueError,match='unavailable'):
        scoring_versions.archived_candidate('f'*64)
    root=tmp_path/fingerprint; root.mkdir()
    for name in scoring_versions.FILES:
        (root/name).write_text('tampered',encoding='utf-8')
    with pytest.raises(ValueError,match='checksum'):
        scoring_versions.archived_candidate(fingerprint)
