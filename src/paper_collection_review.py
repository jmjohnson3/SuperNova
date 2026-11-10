"""Refresh frozen evidence reviews before the slower legacy all-history report.

No API odds requests, selection changes, wager records, or Discord messages.
"""
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
import argparse
import json

import psycopg2

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from model_research import ROOT, clean, safe_time


def snapshot(path, now, *, max_age_hours=24):
    """Missing/corrupt/stale sources remain visible, never ready by default."""
    try:
        data=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data,dict):raise ValueError('Report must be an object')
        at=safe_time(data.get('built_at'))
        age=(now-at).total_seconds() if at else None
        status='current' if age is not None and 0<=age<=max_age_hours*3600 else 'stale'
        return dict(status=status,path=str(path.relative_to(ROOT)),built_at=data.get('built_at'),data=data)
    except (OSError,ValueError,TypeError):
        return dict(status='unavailable',path=str(path.relative_to(ROOT)),data={})


def remaining(metrics, requirements):
    return dict(days=max(0,requirements['min_span_days']-metrics.get('span_days',0)),
                games=max(0,requirements['min_games']-metrics.get('games',0)),
                checked_prices_with_exact_closes=max(0,requirements['min_priced_checks']-metrics.get('priced_checks',0)),
                other_failed_gates=[k for k,v in metrics.get('gates',{}).items() if not v],
                counts_authorize_betting=False)


def finance_week(now):
    import sports_finance as finance
    # A rolling seven-day receipt review, refreshed automatically; no completeness attestation is invented.
    end=now.date();start=end-timedelta(days=6)
    result=finance.summarize(finance.read_sources(),start_date=start,end_date=end,
                             inventory_coverage=finance.read_coverage())
    atomic_json(ROOT/'data/sports_finance/reports/weekly_latest.json',result)
    (ROOT/'data/sports_finance/reports/weekly_latest.md').write_text(finance.markdown(result),encoding='utf-8')
    return dict(start_date=str(start),end_date=str(end),receipt_wagers=result['receipt_wagers_through_end'],
                currencies=result['currencies'],warnings=result['warnings'],cash_authorized=False)


def refresh():
    from sharp_pipeline import nba_experiment, report
    from model_research import active as model_active
    from model_research_report import build_report
    now=datetime.now(timezone.utc)
    with closing(psycopg2.connect(PG_DSN,connect_timeout=10)) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute('SELECT pg_try_advisory_lock(86731493)')
            if not cur.fetchone()[0]:return dict(status='review_busy',betting_authorized=False)
        # Authoritative frozen scorers/selection and review thresholds, not copies of their rules.
        nba_experiment.active()
        raw,closes=report.load(conn)
        nba_experiment.write_report(conn,raw,closes,now)
        build_report(conn,model_active())
    return build(now=datetime.now(timezone.utc))


def build(*,now=None):
    import collection_budget
    from sharp_pipeline import nba_experiment
    from model_research import active as model_active, CONFIG
    now=now or datetime.now(timezone.utc)
    sources={name:snapshot(ROOT/'reports'/file,now) for name,file in (
        ('nba','nba_price_checks_latest.json'),('paired','model_research_latest.json'),
        ('survival','price_survival_review_latest.json'),
        ('auto_nba','automatic_price_checks_nba_latest.json'),('auto_nfl','automatic_price_checks_nfl_latest.json'))}
    # Verify identity as well as age before displaying any review status.
    for name,verify,hash_key in [('nba',nba_experiment.active,'experiment_sha256'),('paired',model_active,'sha256')]:
        try:
            doc=verify()
            if sources[name]['data'].get(hash_key)!=doc['sha256']:sources[name]['status']='identity_mismatch'
        except (OSError,ValueError):sources[name]['status']='source_drift'
    from automatic_price_checks import active as automatic_active
    try:
        doc=automatic_active()
        for key in ('auto_nba','auto_nfl'):
            if sources[key]['data'].get('sha256')!=doc['sha256']:sources[key]['status']='identity_mismatch'
    except (OSError,ValueError):
        for key in ('auto_nba','auto_nfl'):sources[key]['status']='source_drift'
    nba=sources['nba']['data'];paired=sources['paired']['data']
    books={book:dict(metrics=m,remaining=remaining(m,nba.get('review_requirements',nba_experiment.REVIEW)))
           for book,m in nba.get('by_book',{}).items()}
    cost=collection_budget.run(quiet=True)
    actual=finance_week(now)
    result=clean(dict(built_at=now,sources={k:{p:v[p] for p in v if p!='data'} for k,v in sources.items()},
        nba_by_book=books,paired_by_sport_family=paired.get('groups',{}),
        native_close_audit=sources['survival']['data'].get('market_execution_reviews',[]),
        close_coverage_caution='Frozen report counts may include closes after the earlier native schedule. '
            'Use price_survival_review_latest for the stricter audit; unaudited counts cannot establish pregame coverage.',
        nfl_paired_requirements=CONFIG['nfl_review'],automatic={k:sources[k]['data'] for k in ('auto_nba','auto_nfl')},
        economics=cost['economics'],actual_week=actual,betting_authorized=False,
        note='Automatic public-feed checks do not satisfy frozen manual/account-availability requirements. '
        'Counts trigger review only. Missing execution and actual net profit remain unknown.'))
    atomic_json(ROOT/'reports/paper_collection_review_latest.json',result)
    lines=['# Paper collection review','',f"Updated {now.isoformat()}.",result['note'],'',
           result['close_coverage_caution'],'',
           '| NBA book | Decisions | Games / 20 | Span / 21 days | Exact closes | Timely manual checks | Checked exact closes / 50 | Status |',
           '|---|---:|---:|---:|---:|---:|---:|---|']
    for book,item in books.items():
        m=item['metrics']
        status=m['status'] if sources['nba']['status']=='current' else sources['nba']['status']
        lines.append(f"| {book} | {m['decisions']} | {m.get('games',0)} | {m.get('span_days',0)} | "
                     f"{m.get('with_exact_close',0)} | {m.get('timely_checks',0)} | {m.get('priced_checks',0)} | {status} |")
    lines+=['','NBA also requires active days/weeks, coverage, availability and positive game/day-clustered uncertainty bounds.',
            'NFL is evaluated separately: its paired trial requires 100 checked exact closes, 21 days and 20 games, plus its coverage and uncertainty checks.','',
            '| Paired sport / family | Price-only decisions | Model-selected decisions | Model-selected checked prices |',
            '|---|---:|---:|---:|']
    for key,g in paired.get('groups',{}).items():
        lines.append(f"| {key} | {g['price_control']['decisions']} | {g['model_assisted']['decisions']} | {g['model_assisted'].get('checked_prices',0)} |")
    lines+=['','Both strategies reuse the same offer and first observation. Model vetoes remain in the price-only control.','',
            '| Automatic public-feed observer | Attempts | Timely exact quotes | Status |',
            '|---|---:|---:|---|']
    for key in ('auto_nba','auto_nfl'):
        s=sources[key];d=s['data']
        lines.append(f"| {key} | {d.get('attempts','unknown')} | {d.get('timely_available','unknown')} | {s['status']} |")
    e=cost['economics'];middle=e['scenarios'][1]
    lines+=['',f"Monthly data budget: ${e['monthly_cost']}; comparison stake $1; bankroll reported by user: $50.",
            f"At an assumed true 3% return: {middle['required_monthly_bets']} one-dollar bets/month to cover the data bill.",
            f"Current frozen NBA volume: {e['observed_offers']} offers over {e['calendar_days']} calendar days. "
            f"If all were available and a true 3% return existed, the monthly gross would be "
            f"${float(middle['gross_if_all_observed_opportunities_available'] or 0):.2f}; net after data "
            f"${float(middle['net_if_all_observed_opportunities_available'] or 0):.2f}.",
            'This is conditional cost planning, not an observed edge. NFL is not assumed to cover the shortfall. '
            'No stake increase or subscription change is authorized by this report.','',
            f"Actual receipt review: {actual['start_date']} through {actual['end_date']}; "
            f"{actual['receipt_wagers']} wager receipts through period end. See private weekly finance report for costs and completeness.",
            'A monthly subscription budget is not a payment receipt; unknown invoices and wager inventories are not filled automatically.','',
            '## Sources','']
    lines += [f"- {k}: {v['status']}; {v.get('built_at','unknown')}; `{v['path']}`." for k,v in sources.items()]
    (ROOT/'reports/paper_collection_review_latest.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(status='updated',nba_source=sources['nba']['status'],paired_source=sources['paired']['status'],
                report='reports/paper_collection_review_latest.md',betting_authorized=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--refresh',action='store_true')
    args=ap.parse_args()
    print(json.dumps(refresh() if args.refresh else build()))


if __name__=='__main__':main()
