from datetime import datetime, timedelta, timezone
from itertools import count

import pandas as pd
import pytest

from nfl_pipeline import sharp_watch as w
from nfl_pipeline.modeling import sharp_alert_report as r

KICK = datetime(2026, 10, 4, 17, tzinfo=timezone.utc)
IDS = count(1)
EXPERIMENT = {'sha256': 'frozen-test'}


def quote(over, under, observed):
    return dict(over=over, under=under, market_updated_at=observed.isoformat(),
                book_updated_at=observed.isoformat())


def alert(**kw):
    n = next(IDS)
    base = dict(alert_id=n, game_id=f'game-{n}', event_id=f'event-{n}', book='fanduel', evidence_kind='decision_observation',
                player_name_norm='receiver', season=2026, week=4, stat='receiving_yards', side='over',
                fd_line=45.5, fd_price=-110, ev=.05, tier='alert', commence_time_utc=KICK,
                alerted_at=KICK-timedelta(minutes=30), close_price=-130, sharp_book='pinnacle',
                sharp_close_line=45.5, sharp_close_over=-125, sharp_close_under=105,
                status='final', home_score=24, away_score=20, actual=60., participation_verified=True,
                fd_then_over=-110, fd_then_under=-110, fd_close_line=45.5,
                fd_close_over=-130, fd_close_under=110)
    row = dict(base, **kw)
    edge = dict(event_id=row['event_id'], book=row['book'], stat=row['stat'], side=row['side'], line=row['fd_line'],
                price=row['fd_price'], player_norm=row['player_name_norm'],
                sharp_book='pinnacle', sharp_line=row['fd_line'], ev=row['ev'],
                quote_provenance=dict(contract=w.QUOTE_CONTRACT, observed_at_utc=row['alerted_at'].isoformat(),
                    execution_quote=quote(row['fd_then_over'], row['fd_then_under'], row['alerted_at']),
                    reference_quote=quote(-140, 120, row['alerted_at']), reference_method='exact'))
    row['initial_provenance'] = w.decision_observation(edge,
        dict(game_id=row['game_id'], start=KICK), row['tier'], EXPERIMENT['sha256'])
    return row


def closes(row, *, lag=5, source_lag=None):
    observed = KICK-timedelta(minutes=lag)
    source = KICK-timedelta(minutes=source_lag if source_lag is not None else lag)
    return [dict(contract=w.QUOTE_CONTRACT, game_id=row['game_id'], event_id=row['event_id'],
                 commence_time_utc=KICK.isoformat(), observed_at_utc=observed.isoformat(),
                 book=book, stat=row['stat'], player_name_norm=row['player_name_norm'], line=line,
                 quote=quote(over, under, source))
            for book, line, over, under in (
                (row['book'], row['fd_close_line'], row['fd_close_over'], row['fd_close_under']),
                ('pinnacle', row['sharp_close_line'], row['sharp_close_over'], row['sharp_close_under']))]


def graded(rows, observations=None):
    if observations is None:
        observations = [q for row in rows for q in closes(row)]
    return r.grade(r.attach_exact_evidence(pd.DataFrame(rows), observations))


def summarize(rows, observations=None):
    return r.summarize(graded(rows, observations), experiment=EXPERIMENT)


def test_grade_exact_closes_and_result():
    rows = [alert(), alert(side='under', actual=40.),
            alert(stat='total', fd_line=44.5, actual=None, sharp_close_line=44.5)]
    over, under, total = graded(rows).to_dict('records')
    assert over['clv'] > 0 and under['clv'] < 0
    assert over['ev_at_sharp_close'] > 0 and under['ev_at_sharp_close'] < 0
    assert over['result'] == 'win' and under['result'] == 'win' and total['result'] == 'loss'
    assert over['units'] == pytest.approx(100/110)


def test_review_requires_independence_coverage_and_weekly_support():
    assert summarize([alert()])['verdict'] == 'keep collecting'
    rows = [alert(week=week) for week in (4, 5, 6) for _ in range(40)]
    summary = summarize(rows)
    assert summary['verdict'] == 'evidence_review_ready; no cash authorization'
    assert summary['alerts'] == 120 and summary['weeks'] == 3
    assert summary['checks']['weekly_clv_confirmed']
    bad = [alert(week=week, fd_close_over=110, fd_close_under=-130,
                 sharp_close_over=105, sharp_close_under=-125) for week in (4, 5, 6) for _ in range(40)]
    assert summarize(bad)['verdict'] == 'evidence requirements not met; research only'
    missing = summarize(rows, [q for row in rows[:100] for q in closes(row)])
    assert not missing['checks']['exact_close_coverage']
    assert missing['exact_close_coverage'] == pytest.approx(100/120)


def test_later_prices_do_not_inflate_or_replace_first_decision():
    early = alert(game_id='same', event_id='same', actual=0.)
    later = alert(game_id='same', event_id='same', alerted_at=KICK-timedelta(minutes=20),
                  actual=100., ev=.20)
    summary = summarize([later, early])
    assert summary['alerts'] == 1 and summary['repeated_alerts_removed'] == 1
    assert summary['roi'] == -1.


def test_draftkings_and_old_experiments_do_not_qualify_fanduel():
    rows = [alert(book='draftkings', week=week, evidence_kind='alert_ledger') for week in (4, 5, 6) for _ in range(40)]
    rows.append(alert())
    summary = summarize(rows)
    assert summary['alerts'] == 1 and summary['by_book']['draftkings']['alerts'] == 120
    assert not summary['checks']['enough_alerts']
    assert r.summarize(graded(rows), experiment={'sha256':'new-experiment'})['alerts'] == 0


def test_shifted_line_is_only_inferred_not_exact_close():
    row = alert(fd_close_line=48.5, fd_close_over=-110, fd_close_under=-110,
                sharp_close_line=47.5)
    scored = graded([row]).iloc[0]
    assert pd.isna(scored.clv) and pd.isna(scored.ev_at_sharp_close)
    assert scored.inferred_clv > .03 and scored.inferred_ev_at_sharp_close > 0
    assert summarize([row])['exact_close_coverage'] == 0


@pytest.mark.parametrize('lag,source_lag', [(21,21), (5,21), (-1,-1), (5,4)])
def test_stale_or_future_close_stays_unknown(lag, source_lag):
    row = alert()
    scored = graded([row], closes(row, lag=lag, source_lag=source_lag)).iloc[0]
    assert pd.isna(scored.clv) and pd.isna(scored.ev_at_sharp_close)


def test_close_must_match_book_game_player_and_exact_line():
    row = alert()
    observations = closes(row)
    observations[0]['book'] = 'draftkings'
    observations[1]['event_id'] = 'another-game'
    scored = graded([row], observations).iloc[0]
    assert pd.isna(scored.clv) and pd.isna(scored.ev_at_sharp_close)


def test_legacy_or_tampered_provenance_cannot_support_approval():
    legacy, tampered = alert(), alert()
    legacy.pop('initial_provenance')
    tampered['initial_provenance']['quote_provenance']['execution_quote']['over'] = 150
    frame = graded([legacy, tampered])
    assert not frame.provenance_valid.any()
    assert frame.clv.isna().all()
    assert r.summarize(frame, experiment=EXPERIMENT)['alerts'] == 0


def test_stored_alert_quote_is_used_instead_of_later_legacy_lookup():
    row = alert()
    row['fd_then_over'], row['fd_then_under'] = -300, 240
    assert graded([row]).iloc[0].clv > 0


def test_taken_price_slippage_requires_confirmation_and_positive_means_worse():
    confirmed = alert(placed_at=KICK-timedelta(minutes=29), taken_price=-120, stake=1.)
    unconfirmed = alert(taken_price=-150, stake=5.)
    frame = graded([confirmed, unconfirmed])
    assert frame.iloc[0].slippage > 0
    assert pd.isna(frame.iloc[1].slippage) and frame.iloc[1].bet_price == -110
    summary = r.summarize(frame, experiment=EXPERIMENT)
    assert summary['placed']['alerts'] == 1 and summary['placed']['staked'] == 1.


def test_logged_and_early_are_descriptive_and_report_discloses_missingness():
    rows = [alert(tier='logged'), alert(tier='alert', alerted_at=KICK-timedelta(minutes=600)), alert()]
    frame = graded(rows)
    assert list(r._tier(frame)) == ['logged','early','alert']
    summary = r.summarize(frame, experiment=EXPERIMENT)
    assert summary['alerts'] == 1 and summary['logged']['alerts'] == 1
    md = r.markdown(summary, frame)
    assert 'Simulated flat ROI' in md and 'Missing exact closes remain unknown' in md


def test_unchanged_legacy_alert_price_does_not_suppress_new_experiment_evidence():
    original = alert(evidence_kind='alert_ledger', placed_at=KICK-timedelta(minutes=29), stake=2., taken_price=-115)
    observation = original['initial_provenance']
    frame = r.append_decision_evidence(pd.DataFrame([original]), [observation])
    frame = r.grade(r.attach_exact_evidence(frame, closes(original)))
    summary = r.summarize(frame, experiment=EXPERIMENT)
    assert summary['alerts'] == 1 and summary['captured_decisions'] == 1
    assert summary['placed']['alerts'] == 1 and summary['placed']['staked'] == 2.
    captured = frame[frame.evidence_kind.eq('decision_observation')].iloc[0]
    assert pd.isna(captured.placed_at) and captured.stake == 0


def test_zero_without_participation_does_not_become_a_winning_under():
    row = alert(side='under', actual=0., participation_verified=False)
    result = graded([row]).iloc[0]
    assert result.result is None and pd.isna(result.units)
