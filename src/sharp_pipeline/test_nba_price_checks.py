"""Offline evidence and manual-input regressions; no network or Discord writes."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sharp_pipeline import nba_experiment as experiment, paper_alerts, price_checks, report

NOW = datetime(2026, 10, 28, 22, tzinfo=timezone.utc)


def evidence(**changes):
    return dict(dict(sent_at=NOW, checked_at=NOW + timedelta(seconds=60),
                     recorded_at=NOW + timedelta(seconds=75), commence_time_utc=NOW + timedelta(minutes=30),
                     delivery_status='sent', evidence_source='manual_live', check_status='available',
                     line=22.5, displayed_line=22.5, displayed_price=110,
                     close_probability=.5, close_method='exact'), **changes)


def sample(n=60):
    rows = []
    for i in range(n):
        day = i % 21
        sent = NOW + timedelta(days=day)
        rows.append(evidence(sent_at=sent, checked_at=sent + timedelta(seconds=60),
                             recorded_at=sent + timedelta(seconds=75), commence_time_utc=sent + timedelta(minutes=30),
                             event_id=f'game-{i // 2}', ev_at_close=.05, book='fanduel',
                             result='win' if i % 2 else 'loss', profit=1.1 if i % 2 else -1,
                             model_agrees=bool(i % 2)))
    return pd.DataFrame(rows)


def test_complete_evidence_only_triggers_review_and_model_does_not_select():
    df = sample()
    result = experiment.book_summary(df)
    assert result['status'] == 'execution_review_only', result['reasons']
    assert result['betting_authorized'] is False
    assert result['span_days'] == 21
    assert result['priced_checks'] == 60
    assert result['mean_checked_price_ev'] == pytest.approx(.05)
    other = experiment.book_summary(df.assign(model_agrees=False))
    assert other['gates'] == result['gates'] and other['decisions'] == 60


@pytest.mark.parametrize('change,failed_gate', [
    ({'displayed_price': -130}, 'checked_by_game'),
    ({'close_method': 'shifted'}, 'exact_close_coverage'),
    ({'delivery_status': 'uncertain'}, 'manual_check_coverage'),
    ({'evidence_source': 'historical_snapshot'}, 'manual_check_coverage'),
    ({'check_status': 'disappeared', 'displayed_line': None, 'displayed_price': None}, 'offer_availability'),
])
def test_prices_and_missing_or_estimated_evidence_block_review(change, failed_gate):
    result = experiment.book_summary(sample().assign(**change))
    assert result['status'] == 'collecting'
    assert result['gates'][failed_gate] is False
    assert result['betting_authorized'] is False


def test_same_game_is_not_sixty_independent_observations():
    result = experiment.book_summary(sample().assign(event_id='same-game'))
    assert result['intervals']['checked_by_game']['clusters'] == 1
    assert result['intervals']['checked_by_game']['lower_95'] is None
    assert not result['gates']['games'] and not result['gates']['checked_by_game']


def test_cluster_interval_preserves_shared_game_uncertainty():
    df = pd.DataFrame({'ev': [-.2] * 100 + [.2] * 100, 'game': ['a'] * 100 + ['b'] * 100})
    ci = experiment.clustered_interval(df, 'ev', 'game')
    assert ci['lower_95'] == pytest.approx(-.2)
    assert ci['upper_95'] == pytest.approx(.2)
    assert ci['clusters'] == 2


def test_short_collection_and_small_book_cannot_borrow_sample_size():
    df = sample()
    small_book = experiment.book_summary(df.iloc[:27])
    assert not small_book['gates']['decisions']
    assert not small_book['gates']['checked_prices']
    short = experiment.book_summary(df.assign(commence_time_utc=NOW + timedelta(minutes=30)))
    assert not short['gates']['collection_span'] and not short['gates']['weeks']


def test_unknown_and_unavailable_are_distinct_and_late_records_do_not_count():
    unknown = experiment.check_evidence(evidence(checked_at=None, recorded_at=None))
    assert unknown['opportunity_ev'] is None and not unknown['timely_check']
    absent = experiment.check_evidence(evidence(check_status='disappeared', displayed_line=None, displayed_price=None))
    assert absent['opportunity_ev'] == 0 and absent['checked_price_ev'] is None
    assert absent['timely_check']
    for change in ({'checked_at': NOW + timedelta(minutes=6), 'recorded_at': NOW + timedelta(minutes=6)},
                   {'recorded_at': NOW + timedelta(minutes=7)},
                   {'checked_at': NOW + timedelta(hours=1), 'recorded_at': NOW + timedelta(hours=1)},
                   {'recorded_at': NOW - timedelta(seconds=1)},
                   {'checked_at': pd.NaT, 'recorded_at': pd.NaT}):
        assert not experiment.check_evidence(evidence(**change))['timely_check']
    moved = experiment.check_evidence(evidence(check_status='changed_line', displayed_line=23.5))
    assert moved['timely_check'] and not moved['exact_offer_available']


def test_manual_check_time_requires_timezone():
    assert price_checks.checked_time('2026-10-28T16:00:00-06:00') == NOW
    with pytest.raises(ValueError, match='UTC offset'):
        price_checks.checked_time('2026-10-28T16:00:00')


@pytest.mark.parametrize('changes', [
    {'checked_at': NOW - timedelta(seconds=1)}, {'checked_at': NOW + timedelta(minutes=2)},
    {'status': 'available', 'line': 23.5}, {'status': 'changed_line'},
    {'price': -99}, {'price': 100.5}, {'price': None}, {'line': float('nan')},
    {'status': 'disappeared'}, {'status': 'suspended'}, {'status': 'snapshot'},
])
def test_manual_check_rejects_invalid_data(changes):
    args = dict(checked_at=NOW + timedelta(seconds=30), now=NOW + timedelta(seconds=60),
                status='available', line=22.5, price=-110)
    args.update(changes)
    with pytest.raises(ValueError):
        price_checks.validate(dict(sent_at=NOW, delivery_status='sent', line=22.5), **args)


@pytest.mark.parametrize('status,line,price', [('available',22.5,-115), ('changed_line',23.5,100),
                                             ('disappeared',None,None), ('suspended',None,None)])
def test_manual_check_accepts_explicit_real_observation(status, line, price):
    price_checks.validate(dict(sent_at=NOW, delivery_status='sent', line=22.5),
        checked_at=NOW, now=NOW, status=status, line=line, price=price)


def test_card_is_exact_and_cannot_use_another_sports_webhook(monkeypatch):
    row = dict(alert_id=17, decided_at=NOW, commence_time_utc=NOW + timedelta(minutes=30),
               player='Test Player', side='over', line=22.5, market='points', book='fanduel', price=110,
               away_team='Away', home_team='Home')
    text = paper_alerts.card(row)
    for value in ('PAPER PRICE CHECK', '#17', 'Test Player', 'OVER 22.5 points', 'FanDuel', '+110',
                  '2026-10-28 22:00:00 UTC', 'no wager', 'Discord replies are not imported'):
        assert value in text
    monkeypatch.delenv('NBA_DISCORD_WEBHOOK_URL', raising=False)
    monkeypatch.setenv('MLB_DISCORD_WEBHOOK_URL', 'wrong-channel')
    monkeypatch.setattr('supernovabets_config._saved_windows_env', lambda name: '')
    assert not paper_alerts.webhook() and not report._webhook('nba')


def test_registration_rejects_changed_rules_without_replacing_original(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'STORE', tmp_path)
    saved = experiment.register()
    assert experiment.active()['sha256'] == saved['sha256']
    monkeypatch.setattr(experiment, 'MAX_CHECK_DELAY_SECONDS', 999)
    with pytest.raises(ValueError, match='configuration drift'):
        experiment.active()
    with pytest.raises(ValueError, match='cannot be replaced'):
        experiment.register()
    assert experiment.load()['sha256'] == saved['sha256']
