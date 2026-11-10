"""Explain the frozen NFL price experiment's exact-close gaps without filling them."""
from collections import Counter
from datetime import datetime, timezone
import json

import pandas as pd
import psycopg2

from model_research import ROOT, clean
from nfl_pipeline import experiment_registry
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling import sharp_alert_report as frozen
from nfl_pipeline.modeling.sharp_evidence_report import load_indexed


def primary(frame, doc):
    if frame.empty:
        return frame
    books = frame['book'].fillna('fanduel')
    ladder = frame.get('ladder', pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    eligible = books.isin(frozen.BET_BOOKS) & frame.provenance_valid.fillna(False) & ~ladder
    eligible &= frame.evidence_kind.eq('decision_observation')
    eligible &= frame.stat.isin(frozen.strategy.STAKED_STATS)
    eligible &= frame.experiment_sha256.eq(doc['sha256']) & frame.game_id.notna()
    return frozen.cash_decisions(frame[eligible & frozen._tier(frame).eq('alert')])


def evidence(row, name):
    source = row.get(name)
    if not isinstance(source, dict):
        return None
    quote = source.get('quote') or {}
    return dict(book=source.get('book'), line=source.get('line'), observed_at=source.get('observed_at_utc'),
                source_updated_at=frozen.source_time(quote), over=quote.get('over'), under=quote.get('under'),
                observation_id=source.get('observation_id'))


def detail(row):
    ref = evidence(row, 'exact_reference_close')
    own = evidence(row, 'exact_execution_close')
    nearby = row.get('sharp_close_line')
    nearby = float(nearby) if nearby is not None and pd.notna(nearby) else None
    cause = ('observed_exact' if ref else 'different_reference_line_observed'
             if nearby is not None and nearby != float(row['fd_line']) else 'no_valid_exact_reference_observation')
    return dict(game_id=row['game_id'], player=row.get('player_name_norm'), stat=row['stat'], side=row['side'],
                book=row['book'], line=row['fd_line'], price=row['fd_price'], decision_at=row.get('decision_at'),
                kickoff=row['commence_time_utc'], exact_execution_close=own, exact_reference_close=ref,
                reference_gap_reason=cause, nearest_reference_line_diagnostic=nearby,
                estimated_conversion_excluded=row.get('sharp_close_method') in ('shifted','ladder'),
                historical_backfill_performed=False)


def run():
    doc = experiment_registry.check_experiment(frozen.EXPERIMENT_NAME)
    with psycopg2.connect(frozen.PG_DSN) as conn:
        conn.set_session(readonly=True)
        frame = frozen.grade(load_indexed(conn))
    chosen = primary(frame, doc)
    # Reporting must describe exactly the unchanged registered cohort.
    summary = frozen.summarize(frame, doc)
    if len(chosen) != summary['alerts']:
        raise ValueError('Close detail cohort differs from frozen summary')
    rows = [detail(r) for r in chosen.to_dict('records')]
    by_book = {}
    for book in sorted({r['book'] for r in rows}):
        group = [r for r in rows if r['book'] == book]
        by_book[book] = dict(decisions=len(group), exact_reference=sum(bool(r['exact_reference_close']) for r in group),
                            exact_execution=sum(bool(r['exact_execution_close']) for r in group),
                            gap_reasons=dict(Counter(r['reference_gap_reason'] for r in group)))
    result = clean(dict(built_at=datetime.now(timezone.utc), experiment_sha256=doc['sha256'],
                        by_book=by_book, decisions=rows, betting_authorized=False,
                        note='Observed exact closes only. Different lines remain diagnostics; missing closes remain missing.'))
    atomic_json(ROOT/'reports/nfl_exact_close_review_latest.json', result)
    lines = ['# NFL exact-close review', '', result['note'], '',
             '| Book | Decisions | Exact reference | Exact execution |', '|---|---:|---:|---:|']
    for book, r in by_book.items():
        lines.append(f"| {book} | {r['decisions']} | {r['exact_reference']} | {r['exact_execution']} |")
    lines += ['', '## FanDuel decisions with missing exact reference closes', '',
              '| Game | Player / market | Side / locked line | Nearby reference line (diagnostic) | Reason |',
              '|---|---|---|---|---|']
    for r in rows:
        if r['book'] == 'fanduel' and not r['exact_reference_close']:
            lines.append(f"| {r['game_id']} | {r['player']} / {r['stat']} | {r['side']} {r['line']} | "
                         f"{r['nearest_reference_line_diagnostic']} | {r['reference_gap_reason']} |")
    lines += ['', 'No alternate line, estimated conversion, later timestamp or missing quote is relabeled as an exact close.',
              'No historical data requests, strategy changes or stake authorization.']
    (ROOT/'reports/nfl_exact_close_review_latest.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return dict(status='updated', by_book=by_book, betting_authorized=False)


if __name__ == '__main__':
    print(json.dumps(run()))
