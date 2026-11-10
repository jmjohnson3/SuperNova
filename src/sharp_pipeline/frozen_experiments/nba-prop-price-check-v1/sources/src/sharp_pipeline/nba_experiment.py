"""Frozen prospective NBA prop price checks. Evidence review never authorizes wagering."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2.extras

from nfl_pipeline import experiment_registry as registry, sharp_math
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.integrity import atomic_json

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / 'src/sharp_pipeline/frozen_experiments'
NAME = 'nba-prop-price-check-v1'
BOOKS = ('fanduel', 'draftkings', 'espnbet')
MARKETS = ('points', 'rebounds', 'assists', 'threes')
MAX_ALERT_AGE_SECONDS = 300
MAX_CHECK_DELAY_SECONDS = 300
MAX_RECORD_LAG_SECONDS = 300
REVIEW = dict(min_decisions=50, min_span_days=21, min_active_days=7, min_weeks=3,
              min_games=20, min_close_coverage=.90, min_mean_close_ev=.015,
              min_check_coverage=.90, min_available_fraction=.80, min_priced_checks=50,
              min_checked_close_coverage=.90, min_cluster_lower=0.0)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261028
MODULES = ('sharp_pipeline.scan', 'sharp_pipeline.report', 'sharp_pipeline.outcomes',
           'sharp_pipeline.nba_experiment', 'sharp_pipeline.paper_alerts', 'sharp_pipeline.price_checks',
           'nfl_pipeline.sharp_math', 'nfl_pipeline.context_contract', 'nfl_pipeline.markets',
           'nfl_pipeline.integrity', 'nfl_pipeline.experiment_registry')


def configuration():
    from sharp_pipeline import scan, report
    return dict(sport='nba', markets=list(MARKETS), books=list(BOOKS), reference_book=scan.SHARP_BOOK,
                min_ev=scan.MIN_EV, decision_window=list(scan.DECISION_WINDOW), exact_line_only=True,
                max_quote_age=scan.MAX_SOURCE_AGE, max_quote_skew=scan.MAX_SOURCE_SKEW,
                max_close_age_minutes=report.MAX_CLOSE_AGE_MINUTES, devig=sharp_math.DEVIG_METHOD,
                selection='first qualifying exact prop per book/event/player/market; highest EV at that observation',
                model_role='shadow_only; never filters or sizes a selection', research_only=True,
                max_alert_age_seconds=MAX_ALERT_AGE_SECONDS, max_check_delay_seconds=MAX_CHECK_DELAY_SECONDS,
                max_record_lag_seconds=MAX_RECORD_LAG_SECONDS, review=REVIEW,
                clusters=['event_id', 'game_date'], bootstrap_samples=BOOTSTRAP_SAMPLES,
                bootstrap_seed=BOOTSTRAP_SEED, automatic_betting_authorization=False)


def load():
    return registry.load_experiment(NAME, root=ROOT, store=STORE)


def active():
    doc = registry.check_experiment(NAME, root=ROOT, store=STORE)
    if doc['config'] != configuration():
        raise ValueError('NBA experiment configuration drift')
    return doc


def register():
    files = {m: ROOT / 'src' / Path(*m.split('.')).with_suffix('.py') for m in MODULES}
    return registry.register_experiment(NAME, configuration(), files, root=ROOT, store=STORE)


def clustered_interval(frame: pd.DataFrame, value: str, cluster: str) -> dict:
    """Resample entire games/dates; player bets in one game are never independent draws."""
    clean = frame[[value, cluster]].copy()
    clean[value] = pd.to_numeric(clean[value], errors='coerce')
    clean = clean.dropna()
    groups = clean.groupby(cluster)[value].agg(['sum', 'count'])
    out = dict(clusters=int(len(groups)), observations=int(len(clean)),
               mean=float(clean[value].mean()) if len(clean) else None, lower_95=None, upper_95=None)
    if len(groups) < 2:
        return out
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(groups), size=(BOOTSTRAP_SAMPLES, len(groups)))
    means = groups['sum'].to_numpy()[draws].sum(axis=1) / groups['count'].to_numpy()[draws].sum(axis=1)
    out.update(lower_95=float(np.quantile(means, .025)), upper_95=float(np.quantile(means, .975)))
    return out


def check_evidence(row: dict) -> dict:
    """Missing checks stay unknown; an unavailable offer contributes zero opportunity EV."""
    sent, checked, recorded, tip = (safe_time(row.get(k)) for k in
                                   ('sent_at', 'checked_at', 'recorded_at', 'commence_time_utc'))
    delay = (checked - sent).total_seconds() if sent and checked else None
    lag = (recorded - checked).total_seconds() if checked and recorded else None
    timely = bool(row.get('delivery_status') == 'sent' and row.get('evidence_source') == 'manual_live'
                  and sent and checked and recorded and tip and checked < tip and recorded < tip
                  and 0 <= delay <= MAX_CHECK_DELAY_SECONDS and 0 <= lag <= MAX_RECORD_LAG_SECONDS)
    available = bool(timely and row.get('check_status') == 'available'
                     and pd.notna(row.get('displayed_line')) and float(row['displayed_line']) == float(row['line'])
                     and sharp_math.implied(row.get('displayed_price')) is not None)
    p = row.get('close_probability')
    ev = (sharp_math.ev(float(p), float(row['displayed_price']))
          if available and row.get('close_method') == 'exact' and pd.notna(p) else None)
    opportunity = (ev if available else 0.0) if timely else None
    return dict(check_delay_seconds=delay, recording_lag_seconds=lag, timely_check=timely,
                exact_offer_available=available, checked_price_ev=ev, opportunity_ev=opportunity)


def book_summary(frame: pd.DataFrame) -> dict:
    n = len(frame)
    if not n:
        return dict(decisions=0, status='collecting', reasons=['no prospective decisions'], betting_authorized=False)
    frame = frame.copy()
    frame['day'] = pd.to_datetime(frame.commence_time_utc, utc=True).dt.tz_convert('America/New_York').dt.date
    dates = pd.to_datetime(frame.day)
    span = int((dates.max() - dates.min()).days) + 1
    weeks = int(((dates - dates.min()).dt.days // 7).nunique())
    frame = pd.concat([frame.reset_index(drop=True), pd.DataFrame([check_evidence(r) for r in frame.to_dict('records')])], axis=1)
    close = pd.to_numeric(frame.ev_at_close, errors='coerce')
    close = close.where(frame.close_method.eq('exact'))
    frame['ev_at_close'] = close
    valid = frame.timely_check
    available = frame.exact_offer_available
    priced = pd.to_numeric(frame.checked_price_ev, errors='coerce').notna()
    intervals = {label: clustered_interval(frame, value, cluster) for label, value, cluster in (
        ('close_by_game', 'ev_at_close', 'event_id'), ('close_by_day', 'ev_at_close', 'day'),
        ('checked_by_game', 'checked_price_ev', 'event_id'), ('checked_by_day', 'checked_price_ev', 'day'),
        ('opportunity_by_game', 'opportunity_ev', 'event_id'))}
    coverage = float(close.notna().mean())
    check_coverage = float(valid.mean())
    availability = float(available.sum() / valid.sum()) if valid.any() else 0.0
    checked_close_coverage = float(priced.sum() / available.sum()) if available.any() else 0.0
    gates = dict(decisions=int(close.notna().sum()) >= REVIEW['min_decisions'],
                 collection_span=span >= REVIEW['min_span_days'], active_days=frame.day.nunique() >= REVIEW['min_active_days'],
                 weeks=weeks >= REVIEW['min_weeks'], games=frame.event_id.nunique() >= REVIEW['min_games'],
                 exact_close_coverage=coverage >= REVIEW['min_close_coverage'],
                 closing_edge=bool(close.notna().any() and close.mean() >= REVIEW['min_mean_close_ev']),
                 manual_check_coverage=check_coverage >= REVIEW['min_check_coverage'],
                 offer_availability=availability >= REVIEW['min_available_fraction'],
                 checked_prices=int(priced.sum()) >= REVIEW['min_priced_checks'],
                 checked_close_coverage=checked_close_coverage >= REVIEW['min_checked_close_coverage'])
    for label in ('close_by_game', 'close_by_day', 'checked_by_game', 'checked_by_day'):
        low = intervals[label]['lower_95']
        gates[label] = low is not None and low > REVIEW['min_cluster_lower']
    from sharp_pipeline import outcomes
    return dict(decisions=n, games=int(frame.event_id.nunique()), days=int(frame.day.nunique()), span_days=span, weeks=weeks,
                with_exact_close=int(close.notna().sum()), exact_close_coverage=coverage,
                mean_close_ev=float(close.mean()) if close.notna().any() else None,
                alerts_sent=int(frame.delivery_status.eq('sent').sum()), checks_recorded=int(frame.checked_at.notna().sum()),
                timely_checks=int(valid.sum()), check_coverage=check_coverage, available=int(available.sum()),
                availability=availability, priced_checks=int(priced.sum()), checked_close_coverage=checked_close_coverage,
                mean_checked_price_ev=intervals['checked_by_game']['mean'],
                disappeared=int((valid & frame.check_status.eq('disappeared')).sum()),
                changed_line=int((valid & frame.check_status.eq('changed_line')).sum()),
                median_check_delay_seconds=float(frame.loc[valid, 'check_delay_seconds'].median()) if valid.any() else None,
                intervals=intervals, gates={k: bool(v) for k, v in gates.items()},
                status='execution_review_only' if all(gates.values()) else 'collecting',
                reasons=[k for k, passed in gates.items() if not passed], betting_authorized=False,
                paper_record=outcomes.record(frame), model_shadow=outcomes.shadow_split(frame))


def write_report(conn, raw: pd.DataFrame, closes: dict, now: datetime) -> dict:
    from sharp_pipeline import report, outcomes
    try:
        doc = active()
        source_status = 'frozen'
    except (OSError, ValueError) as exc:
        try:
            doc = load()
        except (OSError, ValueError):
            doc = None
        source_status = 'unregistered_or_drifted'
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT to_regclass('bets.nba_paper_price_alerts') AS t")
        exists = cur.fetchone()['t'] is not None
        alerts = pd.DataFrame()
        if exists and doc:
            cur.execute("""SELECT a.*, c.checked_at, c.recorded_at, c.status AS check_status,
                                  c.displayed_line, c.displayed_price, c.evidence_source
                FROM bets.nba_paper_price_alerts a LEFT JOIN bets.nba_paper_price_checks c USING (alert_id)
                WHERE a.experiment_id=%s AND a.experiment_sha256=%s""", (NAME, doc['sha256']))
            alerts = pd.DataFrame([dict(r) for r in cur.fetchall()])
    selected = pd.DataFrame()
    if not alerts.empty and not raw.empty:
        matched = raw[raw.decision_id.isin(alerts.decision_id)].copy()
        matched = matched[(matched.experiment_id == NAME) & (matched.experiment_sha256 == doc['sha256'])
                          & (pd.to_datetime(matched.decided_at, utc=True) >= pd.Timestamp(doc['registered_at']))]
        if not matched.empty:
            selected = report.decisions_frame(matched, closes, per_book=True)
            selected = outcomes.attach(conn, selected)
            extra = alerts.drop(columns=['book','event_id','market','player_norm','experiment_id','experiment_sha256'])
            selected = selected.merge(extra, on='decision_id', validate='one_to_one')
    books = {book: book_summary(selected[selected.book.eq(book)] if not selected.empty else selected) for book in BOOKS}
    if source_status != 'frozen':
        for metrics in books.values():
            metrics['status'] = 'blocked_source_drift'
            metrics['reasons'].append('registration must match current source')
    historical = int(raw.sport.eq('nba').sum()) if not raw.empty else 0
    result = dict(built_at=now.isoformat(), experiment_id=NAME, experiment_sha256=(doc or {}).get('sha256'),
                  registered_at=(doc or {}).get('registered_at'), source_status=source_status,
                  research_only=True, betting_authorized=False, review_requirements=REVIEW,
                  model_role='shadow_only', by_book=books,
                  selected_requests=int(len(alerts)), completed_game_decisions=int(len(selected)),
                  other_nba_observations=max(0, historical - len(selected)))
    atomic_json(ROOT / 'reports/nba_price_checks_latest.json', result)
    (ROOT / 'reports/nba_price_checks_latest.md').write_text(markdown(result), encoding='utf-8')
    return result


def markdown(result: dict) -> str:
    lines = ['# NBA paper price checks', '', f"Experiment `{result['experiment_id']}` — {result['source_status']}.",
             f"Registered: {result.get('registered_at')}. Prior/unregistered observations remain historical context.",
             'Research only. Passing a review never enables stakes. Model forecasts remain a shadow comparison.', '',
             'Each book needs 50 exact closes, 21 days spanning 3 weeks, 7 active days, 20 games, 90% exact-close coverage, '
             'at least +1.5% mean closing EV, 90% timely manual checks, 80% exact-offer availability, 50 checked prices '
             'with exact closes, and positive 95% lower bounds clustered by game and by day.', '',
             'Checks must be made within 5 minutes of the alert and recorded within 5 minutes of checking, before tip. '
             'Missing checks are unknown; moved lines and disappeared offers are not treated as executable.', '',
             '| Book | Decisions | Games | Days/span | Exact closes | Timely checks | Available | Checked-price EV | Status |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for book, m in result['by_book'].items():
        ev = '-' if m.get('mean_checked_price_ev') is None else f"{m['mean_checked_price_ev']:+.2%}"
        lines.append(f"| {book} | {m['decisions']} | {m.get('games',0)} | {m.get('days',0)}/{m.get('span_days',0)} | "
                     f"{m.get('with_exact_close',0)} | {m.get('timely_checks',0)} | {m.get('available',0)} | {ev} | {m['status']} |")
    lines.append('')
    for book, m in result['by_book'].items():
        if m.get('intervals'):
            for label, ci in m['intervals'].items():
                if ci['lower_95'] is not None:
                    lines.append(f"\n{book} {label}: 95% interval {ci['lower_95']:+.2%} to {ci['upper_95']:+.2%} "
                                 f"across {ci['clusters']} clusters.")
        lines.append(f"\n{book} outstanding: {', '.join(m['reasons']) or 'manual execution review; still no wagering authorization'}.")
    return '\n'.join(lines) + '\n'


def digest(result: dict) -> str:
    """Prospective book cohorts, never a pooled readiness or betting signal."""
    lines = [f"**NBA PAPER PRICE CHECK — evidence update** ({result['source_status']})",
             f"`{result['experiment_id']}` | {result['selected_requests']} requests selected since registration.",
             'Each book is evaluated separately; counts below cover games that have started.']
    for book, m in result['by_book'].items():
        ev = '-' if m.get('mean_checked_price_ev') is None else f"{m['mean_checked_price_ev']:+.2%}"
        lines.append(f"{book}: {m['decisions']} decisions / {m.get('span_days',0)} days; "
                     f"{m.get('with_exact_close',0)} exact closes; {m.get('timely_checks',0)} timely checks; "
                     f"{m.get('available',0)} available; checked-price EV {ev}. {m['status']}.")
    lines += ['Minimum 21 days plus coverage, availability and game/day-clustered uncertainty checks. '
              'Passing triggers manual review only. No wagering authorized.',
              'Prior observations remain historical context in sharp_multi_latest.md. '
              'The projection model is a shadow comparison, never a filter.']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--register', action='store_true')
    args = ap.parse_args()
    doc = register() if args.register else active()
    print(json.dumps({k: doc[k] for k in ('experiment_id', 'sha256', 'registered_at')}, indent=2))


if __name__ == '__main__':
    main()
