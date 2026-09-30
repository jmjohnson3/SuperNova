"""Separate exact-line evidence gaps from failed NFL close collection."""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import psycopg2

from nfl_pipeline.clv_report import ClvConfig, classify_prop_close, prop_close_rows
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.context_contract import safe_time


def summarize(records, now):
    rows = []
    for p in records:
        q = classify_prop_close(p)
        start = safe_time(q['start'])
        phase = ('unknown_kickoff' if start is None else 'completed_window' if now >= start
                 else 'active_window' if (start-now).total_seconds() <= 7200 else 'waiting_for_window')
        rows.append(dict(prediction_id=p['prediction_id'], player=p['player_name'], stat=p['stat'],
            side=p['side'], book=p['book'], line=p['locked_line'], status=q['status'],
            valid=q['valid'], fresh_book_rows=p['fresh_book_rows'], fresh_market_rows=p['fresh_market_rows'],
            observed_lines=p['observed_lines'], last_exact_snapshot=p['fetched_at_utc'], phase=phase,
            is_current=p.get('is_current', False), selected_real_tier=p.get('selected_real_tier', False)))

    def coverage(group):
        completed = [r for r in group if r['phase'] == 'completed_window']
        valid = sum(r['valid'] for r in completed)
        rate = valid/len(completed) if completed else None
        return dict(predictions=len(group), completed_window_rows=len(completed), valid=valid,
            valid_coverage=rate, coverage_pass=rate >= .9 if rate is not None else None,
            phases=dict(Counter(r['phase'] for r in group)),
            completed_statuses=dict(Counter(r['status'] for r in completed)))

    scopes = {'all_locked': coverage(rows), 'current_fanduel': coverage([
        r for r in rows if r['book'] == 'fanduel' and r['is_current']]),
        'selected_real_tier': coverage([r for r in rows if r['selected_real_tier']])}
    for book in sorted({r['book'] for r in rows if r['book']}):
        scopes[book] = coverage([r for r in rows if r['book'] == book])
    return rows, scopes


def report(day):
    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout='90s'")
        records = prop_close_rows(conn, ClvConfig(game_date=day))
    now = datetime.now(timezone.utc)
    rows, scopes = summarize(records, now)
    completed = scopes['all_locked']; valid = completed['valid']
    doc = clean(dict(built_at=datetime.now(timezone.utc).isoformat(), day=str(day),
        status=('evaluated' if completed['completed_window_rows'] else 'awaiting_close_window') if rows else 'no_locked_prop_offers',
        predictions=len(rows), valid=valid, valid_coverage=completed['valid_coverage'],
        required_coverage=.9, scopes=scopes, statuses=dict(Counter(r['status'] for r in rows)), rows=rows,
        note='Observed alternative lines do not establish availability or price CLV at the locked line. Unknown CLV stays null.'))
    root = Path(__file__).resolve().parents[2] / 'reports'
    atomic_json(root / f'nfl_close_capture_diagnostic_{day}.json', doc)
    text = ['# NFL Close Capture Diagnostic', '', f"Date: {day}; completed-window valid closes: {valid}/{completed['completed_window_rows']}",
            doc['note'], '', 'Future games are not counted as failed closes. Counts are locked forecast rows, not independent bets.', '',
            '| Scope | Completed rows | Valid | Coverage | Meets 90% |', '|---|---:|---:|---:|---|']
    for scope, metrics in scopes.items():
        rate = f"{metrics['valid_coverage']:.1%}" if metrics['valid_coverage'] is not None else 'pending'
        text.append(f"| {scope} | {metrics['completed_window_rows']} | {metrics['valid']} | {rate} | {metrics['coverage_pass']} |")
    text += ['', '| Prediction | Player | Book | Locked line | Status / Phase | Observed lines |',
            '|---|---|---|---:|---|---|']
    for r in rows:
        if not r['valid']:
            text.append(f"| {r['prediction_id']} | {r['player']} | {r['book']} | {r['line']} | {r['status']} / {r['phase']} | {r['observed_lines']} |")
    rendered='\n'.join(text)+'\n'
    (root / f'nfl_close_capture_diagnostic_{day}.md').write_text(rendered, encoding='utf-8')
    if rows or not (root / 'nfl_close_capture_diagnostic_latest.json').exists():
        atomic_json(root / 'nfl_close_capture_diagnostic_latest.json', doc)
        (root / 'nfl_close_capture_diagnostic_latest.md').write_text(rendered, encoding='utf-8')
    return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', type=date.fromisoformat, required=True)
    print({k: v for k, v in report(parser.parse_args().date).items() if k != 'rows'})
