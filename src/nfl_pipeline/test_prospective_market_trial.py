from datetime import date, datetime, timezone
import json

import pandas as pd
import pytest

from nfl_pipeline import crawler_oddsapi as crawler
from nfl_pipeline.modeling import benchmark_offers as benchmark
from nfl_pipeline.modeling import receiving_research_trial as trial
from nfl_pipeline.modeling.prospective_market import locked_market, market_comparison
from nfl_pipeline.test_benchmark_offers import record, document, ledger

NOW = datetime(2026, 9, 22, 10, 10, tzinfo=timezone.utc)


def paired_record():
    r = record(); p = r['forecast_payload']
    p.update(offer_id=51, offer_player_name_norm='testplayer', drift_guard_pass=True,
             projection_p10=0., projection_p90=80.)
    p['scoring_replay']['offer'] = dict(offer_id=51, event_id='provider-game',
        commence_time_utc=r['start_ts_utc'], fetched_at_utc=r['offer_fetched_at'],
        bookmaker_key='fanduel', stat='receiving_yards', line=30.5, player_name_norm='testplayer',
        market_key='player_reception_yds', over_price=-114, under_price=-106,
        over_link='https://sportsbook.fanduel.com/addToBetslip?selection=1')
    return r


def config():
    return dict(registered_at='2026-09-22T10:00:30Z', run_id='benchmark-test', artifact_sha256='hash',
        production_release='production', stat='receiving_yards', book='fanduel',
        variant='stable_ensemble_calibrated', market_keys=['player_reception_yds'],
        rules=dict(max_per_date=3, max_per_game=2, max_quote_age_minutes=20))


def shadow():
    return dict(document()['rows'][0], variant='stable_ensemble_calibrated', same_side_probability=.60)


def test_no_vig_uses_original_pair_and_original_side():
    r = paired_record()
    p = (114/214)/((114/214)+(106/206))
    assert locked_market(r)['market_probability'] == pytest.approx(p)
    r['forecast_payload'].update(side='under', price=-106)
    assert locked_market(r)['market_probability'] == pytest.approx(1-p)


@pytest.mark.parametrize('field,value,reason', [
    ('under_price', None, 'missing_or_invalid_paired_price'),
    ('under_price', 0, 'missing_or_invalid_paired_price'),
    ('over_price', -120, 'captured_side_price_mismatch'),
    ('bookmaker_key', 'draftkings', 'captured_quote_identity_mismatch'),
    ('line', 40.5, 'captured_quote_identity_mismatch'),
    ('offer_id', 52, 'captured_quote_identity_mismatch'),
    ('player_name_norm', 'otherplayer', 'captured_quote_identity_mismatch'),
    ('fetched_at_utc', '2026-09-22T10:30:00Z', 'captured_quote_timing_invalid'),
    ('commence_time_utc', '2026-09-23T20:00:00Z', 'captured_event_mismatch'),
    ('pair_quality', 'synthetic', 'not_true_same_book_pair'),
])
def test_untrusted_pairs_stay_unknown(field, value, reason):
    r = paired_record(); r['forecast_payload']['scoring_replay']['offer'][field] = value
    assert locked_market(r) == dict(market_probability=None, market_evidence=reason)


def test_three_way_metrics_use_identical_rows_not_different_denominators():
    rows = [dict(game_id='g', player_id='p', season=2026, week=3, outcome=1.,
        market_probability=.5, production_probability=.8, same_side_probability=.6,
        market_evidence='true_same_book_at_lock'),
        dict(game_id='g2', player_id='p2', season=2026, week=3, outcome=0.,
        market_probability=None, production_probability=.99, same_side_probability=.01,
        market_evidence='missing_captured_quote')]
    result = market_comparison(pd.DataFrame(rows))
    assert result['rows'] == 1 and result['eligible_rows'] == 2
    assert result['production']['brier'] == pytest.approx(.04)
    assert result['challenger']['brier'] == pytest.approx(.16)
    assert result['market']['brier'] == pytest.approx(.25)
    assert result['exclusions'] == {'missing_captured_quote': 1}


def test_market_comparison_includes_exact_micro_without_reselecting():
    rows, _, _ = benchmark.validate_documents([paired_record()], [ledger()], [document()])
    metrics = next(iter(benchmark.offered_metrics(rows).values()))
    for scope in ('real_offers', 'exact_micro'):
        assert metrics[scope]['matched_market']['rows'] == 1
    review = benchmark.deployment_review(metrics)
    assert not review['betting_approved'] and not review['deployment_approved']
    assert review['market_evidence_blockers']


def test_no_preregistration_backfill_and_no_post_kickoff_selection():
    r = paired_record(); c = config()
    assert trial.selection_error(r, shadow(), c, NOW) is None
    c['registered_at'] = NOW.isoformat()
    assert trial.selection_error(r, shadow(), c, NOW) == 'not_a_new_post_registration_pregame_lock'
    assert trial.selection_error(r, shadow(), config(), datetime(2026, 9, 23, tzinfo=timezone.utc))


def test_trial_respects_freshness_links_and_ev():
    r = paired_record(); s = shadow()
    assert trial.selection_error(r, dict(s, same_side_probability=.4), config(), NOW) == 'nonpositive_current_locked_ev'
    r['forecast_payload']['scoring_replay']['offer']['over_link'] = 'https://fanduel.com.evil.example/link'
    assert trial.selection_error(r, s, config(), NOW) == 'missing_valid_fanduel_link'
    assert trial.selection_error(paired_record(), s, config(), datetime(2026, 9, 22, 11, tzinfo=timezone.utc)) == 'quote_too_old_for_research_selection'


def test_fixed_cap_survives_reruns_and_outcomes_do_not_affect_selection():
    records = []; rows = []
    for i in range(6):
        r = paired_record(); r.update(id=i+1, player_id='p'+str(i), game_id='g'+str(i//3), actual=500 if i%2 else 0)
        records.append(r); rows.append(dict(shadow(), forecast_id=i+1, same_side_probability=.6+i/100))
    first, _ = trial.choose(records, rows, config(), NOW, [])
    assert [r['forecast_id'] for r in first] == [6, 5, 3]
    assert trial.choose(records, rows, config(), NOW, first)[0] == []
    for r in records:
        r['actual'] = None
    assert trial.choose(records, rows, config(), NOW, [])[0] == first


def test_trial_registration_is_pinned_and_tamper_evident(tmp_path, monkeypatch):
    monkeypatch.setattr(trial, 'STORE', tmp_path)
    artifact = dict(run_id='run', production_release='production', models={
        'receiving_yards': {'prospective_variants': ['stable_ensemble_calibrated']}})
    a = trial.register(artifact, {'sha256': 'a'})
    assert trial.register(artifact, {'sha256': 'a'}) == a
    with pytest.raises(ValueError):
        trial.register(artifact, {'sha256': 'b'})
    a['config']['book'] = 'draftkings'
    (tmp_path/'registration.json').write_text(json.dumps(a))
    with pytest.raises(ValueError):
        trial.load_registration()


def sgo_event():
    event = dict(eventID='e', startsAt='2026-09-24T20:00:00Z', homeTeamName='Miami Dolphins',
        awayTeamName='Buffalo Bills', odds={})
    for side, price in [('over', -110), ('under', -115)]:
        event['odds']['receiving_yards-P-game-ou-'+side] = dict(playerName='Test Receiver',
            byBookmaker={'fanduel': dict(available=True, odds=price, overUnder=51.5,
                deeplink='https://sportsbook.fanduel.com/main', altLines=[
                    dict(available=True, odds=price-5, overUnder=50.5, deeplink='https://sportsbook.fanduel.com/alt'),
                    dict(available=False, odds=price, overUnder=49.5),
                    dict(odds=price, overUnder=48.5)])})
    return event


def outcomes(event, alts):
    result = crawler._canonical_event_from_sgo(event, include_alts=alts)
    return [o for b in result['bookmakers'] for m in b['markets'] for o in m['outcomes']] if result else []


def test_alt_close_keeps_exact_lines_and_never_inactive_quotes():
    event = sgo_event()
    assert [o['point'] for o in outcomes(event, False)] == [51.5, 51.5]
    rows = outcomes(event, True)
    assert [o['point'] for o in rows] == [51.5, 50.5, 51.5, 50.5]
    assert rows[1]['link'].endswith('/alt')
    for odd in event['odds'].values():
        odd['byBookmaker']['fanduel']['available'] = False
    assert not outcomes(event, False)
    assert [o['point'] for o in outcomes(event, True)] == [50.5, 50.5]


def test_main_quote_cannot_borrow_alternate_price_or_line():
    event = sgo_event()
    for odd in event['odds'].values():
        del odd['byBookmaker']['fanduel']['odds']
    assert not outcomes(event, False)
    assert [o['point'] for o in outcomes(event, True)] == [50.5, 50.5]


def test_alternate_quotes_parse_as_distinct_exact_paired_lines():
    from nfl_pipeline.parse_oddsapi import _rows_from_event
    event = crawler._canonical_event_from_sgo(sgo_event(), include_alts=True)
    diagnostic = dict(events_seen=0, bookmaker_entries=0, events_with_zero_books=0,
        market_entries=0, outcomes_seen=0, unknown_markets={}, skipped_outcomes={})
    rows = _rows_from_event('close', date(2026, 9, 24), NOW, event, diagnostic, 'sportsgameodds')
    paired = {r[14]: (r[15], r[16]) for r in rows}
    assert paired == {51.5: (-110, -115), 50.5: (-115, -120)}
    assert all(r[0] == 'sportsgameodds' and r[3] == 'close' for r in rows)


def test_capture_cleans_its_lock_after_failure_but_never_another_jobs_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(trial, 'STORE', tmp_path)
    registration = dict(config=config(), sha256='a')
    monkeypatch.setattr(trial, 'load_registration', lambda: registration)
    monkeypatch.setattr(trial, 'choose', lambda *a: (_ for _ in ()).throw(ValueError('test failure')))
    doc = dict(run_id='benchmark-test', source_manifest={'sha256': 'hash'}, rows=[])
    with pytest.raises(ValueError, match='test failure'):
        trial.capture([], doc)
    assert not (tmp_path/'.capture.lock').exists()
    (tmp_path/'.capture.lock').touch()
    with pytest.raises(FileExistsError):
        trial.capture([], doc)
    assert (tmp_path/'.capture.lock').exists()


def test_component_review_is_separate_from_cash_and_market_approval():
    scope = dict(rows=60, weeks=4, production={'calibration_error': .03},
        final={'calibration_error': .02}, final_brier_gain={'lower_95': .001},
        live_curve={'coverage_80': .80}, production_curve={'coverage_80': .78})
    review = benchmark.deployment_review({'real_offers': scope, 'exact_micro': {'rows': 0}})
    assert review['status'] == 'ready_for_component_review'
    assert review['market_evidence_blockers']
    assert not review['deployment_approved'] and not review['betting_approved']


def test_three_weeks_alone_never_passes_component_review():
    scope = dict(rows=60, weeks=3, production={'calibration_error': .03},
        final={'calibration_error': .04}, final_brier_gain={'lower_95': -.001},
        live_curve={'coverage_80': .65}, production_curve={'coverage_80': .78})
    review = benchmark.deployment_review({'real_offers': scope})
    assert review['status'] == 'collecting_evidence'
    assert len(review['blockers']) == 3
    assert not review['automatic_promotion']


@pytest.mark.parametrize('role,wanted', [('close', 'true'), ('lock', 'false'), ('open', 'false')])
def test_only_close_requests_alternates(monkeypatch, role, wanted):
    from contextlib import nullcontext
    requests = []
    class Response:
        ok = True
        def json(self):
            return {'data': []}
    monkeypatch.setattr(crawler.requests, 'get', lambda u, **kw: requests.append(kw) or Response())
    monkeypatch.setattr(crawler.psycopg2, 'connect', lambda d: nullcontext(None))
    monkeypatch.setattr(crawler, 'ensure_schema', lambda c: None)
    monkeypatch.setattr(crawler, '_save_payload', lambda *a, **kw: None)
    crawler._fetch_sportsgameodds_for_date(crawler.OddsCrawlerConfig(sports_game_odds_key='test', snapshot_role=role), date(2026,9,24))
    assert requests[0]['params']['includeAltLines'] == wanted
