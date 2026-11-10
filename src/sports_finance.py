"""Receipt-based NFL/NBA accounting, with shared costs counted once.

This tool imports user-supplied cash receipts. It never places wagers or turns
paper alerts, projected EV or legacy confirmations into actual profit.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_CEILING
import json
import os
from pathlib import Path
import tempfile

from nfl_pipeline import wager_reconciliation as journal

ROOT = Path(__file__).resolve().parents[1]
SCOPES = ('nfl', 'nba', 'shared')
JOURNALS = {scope: ROOT / 'data' / f'{scope}_wager_reconciliation' for scope in SCOPES}
REPORT = ROOT / 'data/sports_finance/reports/latest.json'
COVERAGE = ROOT / 'data/sports_finance/wager_coverage'


def validate_sources(sources):
    """Validate every journal and reject the same ticket/invoice in multiple sports."""
    combined = []
    checked = {}
    for scope in SCOPES:
        events = journal._validated_union([], sources.get(scope, []))[0]
        if scope == 'shared' and any(e['kind'] not in {'cost', 'cost_coverage'} for e in events):
            raise ValueError('Shared scope accepts operating costs and cost coverage only')
        checked[scope] = events
        for event in events:
            row = dict(event, event_id=f'{scope}:{event["event_id"]}')
            if 'wager_id' in row:
                row['wager_id'] = f'{scope}:{row["wager_id"]}'
            # cost_id and (book,ticket_id) are deliberately global, to detect duplicate receipts.
            combined.append(row)
    journal._validated_union([], combined)
    return checked, combined


def read_sources(paths=JOURNALS):
    sources = {scope: journal.read_events(paths[scope]) for scope in SCOPES}
    return validate_sources(sources)[0]


def import_document(document, scope, *, paths=JOURNALS, dry_run=False):
    if scope not in SCOPES:
        raise ValueError('Unknown scope')
    incoming = journal.parse_document(document)
    sources = read_sources(paths)
    sources[scope] = journal._validated_union(sources[scope], incoming)[0]
    validate_sources(sources)  # Fail before writing if another sport already owns the receipt.
    return dict(scope=scope, **journal.import_document(document, paths[scope], dry_run=dry_run))


def break_even(cost, *, costs_complete):
    """Scenarios, not estimates of the strategy's edge or a staking recommendation."""
    cost = Decimal(str(cost))
    return [dict(assumed_edge=str(edge), required_turnover=str((cost / edge).quantize(Decimal('.01'), rounding=ROUND_CEILING)),
                 one_dollar_bets=int((cost / edge).to_integral_value(rounding=ROUND_CEILING)),
                 basis='complete_recorded_costs' if costs_complete else 'recorded_costs_only_lower_bound')
            for edge in (Decimal('.01'), Decimal('.03'), Decimal('.05'))]


def receipt_fingerprint(events, currency, end):
    placements = {e['wager_id'] for e in events if e['kind'] == 'wager_placed'
                  and e['currency'] == currency and e['placed_at_utc'][:10] <= str(end)}
    eligible = [e for e in events if (e['kind'] == 'wager_placed' and e['wager_id'] in placements)
                or (e['kind'] == 'wager_settled' and e['wager_id'] in placements and e['settled_at_utc'][:10] <= str(end))]
    return journal._digest(sorted(eligible, key=lambda e: e['event_id']))


def read_coverage(directory=COVERAGE):
    rows = []
    for path in sorted(Path(directory).glob('*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        if (row.get('contract') != 'sports-wager-inventory-v1' or row.get('scope') not in {'nfl','nba'}
                or path.stem != journal._digest(row)):
            raise ValueError(f'Wager inventory attestation failed integrity check: {path.name}')
        rows.append(row)
    return rows


def attest_inventory(sources, scope, currency, start, end, statement, *, directory=COVERAGE):
    """Explicit user attestation, never inferred from missing receipts or predictions."""
    if scope not in {'nfl', 'nba'}:
        raise ValueError('Wager inventory scope must be nfl or nba')
    sources = validate_sources(sources)[0]
    # Reuse the journal's date/currency/nonempty-attestation validation.
    validated = journal.normalize_event(dict(kind='cost_coverage',event_id='validation-only',
        currency=currency,start_date=str(start),end_date=str(end),attestation=statement))
    if date.fromisoformat(validated['end_date']) > datetime.now(timezone.utc).date():
        raise ValueError('Cannot attest a future wager inventory')
    row = dict(contract='sports-wager-inventory-v1', scope=scope, currency=validated['currency'],
        start_date=validated['start_date'], end_date=validated['end_date'], attestation=validated['attestation'],
        includes='All placements, settlements and open wagers through period end, including opening exposure',
        receipt_fingerprint=receipt_fingerprint(sources[scope],validated['currency'],validated['end_date']))
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    destination = directory / (journal._digest(row)+'.json')
    if destination.exists():
        read_coverage(directory)
        return dict(written=False,scope=scope)
    descriptor, name = tempfile.mkstemp(prefix='.coverage-',suffix='.tmp',dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
            stream.write(json.dumps(row,indent=2)+'\n')
            stream.flush(); os.fsync(stream.fileno())
        try:
            temporary.rename(destination)
        except FileExistsError:
            read_coverage(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(written=True,scope=scope)


def summarize(sources, *, start_date=None, end_date=None, inventory_coverage=()):
    sources, combined = validate_sources(sources)
    # A cost attestation in one scope must never certify the other scopes' costs.
    actual = journal.summarize([e for e in combined if e['kind'] != 'cost_coverage'],
                               start_date=start_date, end_date=end_date)
    start, end = date.fromisoformat(actual['start_date']), date.fromisoformat(actual['end_date'])
    parts = {scope: journal.summarize(events, start_date=start, end_date=end)
             for scope, events in sources.items()}
    currencies = sorted(set(actual['currencies']) | {e['currency'] for events in sources.values()
                         for e in events if e['kind'] == 'cost_coverage'}
                         | {r['currency'] for r in inventory_coverage})
    for currency in currencies:
        if currency not in actual['currencies']:
            # Explicit inventories get a currency row even when there are no receipts.
            # Wager completeness and unknown operating costs are evaluated separately.
            actual['currencies'][currency] = dict(placed_in_period=0, settled_wagers=0, record={},
                settled_stake='0.00', cash_return='0.00', gross_profit='0.00', gross_roi=None,
                recorded_costs='0.00', net_after_recorded_costs='0.00', open_wagers=0, open_stake='0.00')
        m = actual['currencies'][currency]
        coverage = {scope: bool(parts[scope]['currencies'].get(currency, {}).get('cost_coverage_complete'))
                    for scope in SCOPES}
        complete = all(coverage.values())
        m['cost_coverage_by_scope'] = coverage
        m['cost_coverage_complete'] = complete
        m['receipt_net_after_complete_costs'] = m['net_after_recorded_costs'] if complete else None
        inventory = {}
        for scope in ('nfl','nba'):
            valid = [r for r in inventory_coverage if r['scope'] == scope and r['currency'] == currency
                     and r['receipt_fingerprint'] == receipt_fingerprint(sources[scope],currency,r['end_date'])]
            inventory[scope] = journal._covered(valid,start,end)
        m['wager_inventory_by_scope'] = inventory
        m['wager_inventory_complete'] = all(inventory.values())
        m['net_profit'] = m['net_after_recorded_costs'] if complete and all(inventory.values()) else None
        m['costs_by_scope'] = {scope: parts[scope]['currencies'].get(currency, {}).get('recorded_costs', '0.00')
                              for scope in SCOPES}
        m['break_even_scenarios'] = break_even(m['recorded_costs'], costs_complete=complete)
    by_sport_book = {}
    for sport in ('nfl', 'nba'):
        events = sources[sport]
        books = sorted({e['book'] for e in events if e['kind'] == 'wager_placed'})
        by_sport_book[sport] = {}
        for book in books:
            wagers = {e['wager_id'] for e in events if e['kind'] == 'wager_placed' and e['book'] == book}
            group = [e for e in events if (e['kind'] in {'wager_placed', 'wager_settled'} and e['wager_id'] in wagers)
                     or (e['kind'] == 'execution_attempt' and e['book'] == book)]
            detail = journal.summarize(group, start_date=start, end_date=end)
            by_sport_book[sport][book] = {
                currency: {key: value for key, value in row.items() if key in {
                    'placed_in_period', 'settled_wagers', 'record', 'settled_stake', 'cash_return',
                    'gross_profit', 'gross_roi', 'open_wagers', 'open_stake'}}
                for currency, row in detail['currencies'].items()}
    result = dict(schema_version=1, built_at=datetime.now(timezone.utc).isoformat(), start_date=str(start), end_date=str(end),
                  accounting='Receipt settlements and actual incurred costs by UTC date; return includes stake; currencies remain separate',
                  receipt_wagers_through_end=actual['receipt_wagers_through_end'], currencies=actual['currencies'],
                  by_sport_book=by_sport_book, wagers=actual['wagers'], attempts=actual['attempts'],
                  warnings=list(dict.fromkeys(w for w in actual['warnings'] if 'complete operating costs are unknown' not in w)),
                  limitations=[
                      'These totals cover imported receipts only; missing wager records remain unknown.',
                      'Overall net_profit requires explicit user attestations of the wager inventory and all costs; it is not independently verified by a sportsbook.',
                      'Shared expenses are subtracted once from combined results; no arbitrary sport/book allocation.',
                      'Cost completeness requires coverage for NFL, NBA and shared expenses, including explicit zero-cost scopes.',
                      'Break-even edge percentages are hypothetical scenarios, not measured edges or permission to increase stakes.',
                      'Paper results and legacy confirmations are excluded from receipt profit.'],
                  cash_authorized=False, wagers_placed_by_this_tool=False)
    for currency, m in result['currencies'].items():
        missing = [scope for scope, known in m['cost_coverage_by_scope'].items() if not known]
        if missing:
            result['warnings'].append(f'{currency}: cost inventory is incomplete for {", ".join(missing)}')
        missing = [scope for scope, known in m['wager_inventory_by_scope'].items() if not known]
        if missing:
            result['warnings'].append(f'{currency}: wager inventory is unverified or changed for {", ".join(missing)}')
    if (result['receipt_wagers_through_end'] == 0 and result['currencies']
            and all(m['wager_inventory_complete'] for m in result['currencies'].values())):
        result['warnings'] = [w for w in result['warnings'] if not w.startswith('No actual wager receipts imported')]
        result['warnings'].append('No actual wagers reported through period end; NFL/NBA inventories are explicitly user-attested.')
    return result


def template(scope):
    help_doc = journal.template()
    help_doc['scope'] = scope
    help_doc['field_help']['cost'] = ('Actual expense assigned to this scope. Shared data/software invoices belong only '
        'in shared, once. Use distinct documented allocation cost_ids only if an invoice was explicitly split. '
        'Deposits and wager stakes are not operating costs.')
    help_doc['field_help']['cost_coverage'] = 'Attest all expenses in this scope/currency/date range have been supplied, including explicit zero-cost periods.'
    help_doc['field_help']['wager_placed'] += (' For NBA, use source_kind external and recommendation_id '
        'nba-price-check:<ID> when a real receipt links to a paper request, or a personal tracking ID. '
        'A paper request alone is not a wager receipt.')
    if scope == 'shared':
        help_doc['required_fields'] = {k: v for k, v in help_doc['required_fields'].items() if k in {'cost','cost_coverage'}}
    return help_doc


def markdown(result):
    lines = ['# NFL / NBA receipt accounting', '', f"Period: {result['start_date']} through {result['end_date']} (UTC).",
             f"Imported wager receipts through period end: {result['receipt_wagers_through_end']}.",
             'Paper outcomes and expected EV are not actual profit. Total net requires wager and cost reconciliation.', '',
             '| Currency | Settled wagers | Gross receipt profit | Recorded costs | Net after recorded costs | Reconciled net | Open stake |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for currency, m in result['currencies'].items():
        lines.append(f"| {currency} | {m['settled_wagers']} | {m['gross_profit']} | {m['recorded_costs']} | "
                     f"{m['net_after_recorded_costs']} | {m['net_profit'] if m['net_profit'] is not None else 'unknown'} | {m['open_stake']} |")
    if not result['currencies']:
        lines += ['', 'No receipts or costs have been imported. Actual net profit is unknown.']
    lines += ['', '## Receipt results by sport and sportsbook', '',
              '| Sport | Book | Currency | Settled wagers | Gross profit | Gross ROI | Open stake |',
              '|---|---|---|---:|---:|---:|---:|']
    for sport, books in result['by_sport_book'].items():
        for book, currencies in books.items():
            for currency, m in currencies.items():
                roi = '-' if m['gross_roi'] is None else f"{m['gross_roi']:+.2%}"
                lines.append(f"| {sport} | {book} | {currency} | {m['settled_wagers']} | {m['gross_profit']} | {roi} | {m['open_stake']} |")
    if result.get('legacy_nfl_cross_check'):
        lines += ['', f"Legacy NFL reconciliation: {result['legacy_nfl_cross_check']['counts']}. These are excluded from receipt profit."]
    lines += ['', '## Limits of this report', ''] + ['- '+s for s in result['limitations']]
    lines += [''] + ['- '+s for s in result['warnings']]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scope', choices=SCOPES)
    actions = ap.add_mutually_exclusive_group(required=True)
    actions.add_argument('--template', action='store_true')
    actions.add_argument('--validate', type=Path)
    actions.add_argument('--import-json', type=Path)
    actions.add_argument('--report', action='store_true')
    actions.add_argument('--attest-wagers', action='store_true',
        help='Explicitly attest all actual wagers, settlements and opening exposure through the stated period end are supplied')
    ap.add_argument('--start-date', type=date.fromisoformat)
    ap.add_argument('--end-date', type=date.fromisoformat)
    ap.add_argument('--include-db', action='store_true')
    ap.add_argument('--currency')
    ap.add_argument('--statement', help='Your explicit complete-inventory attestation; never filled automatically')
    args = ap.parse_args(argv)
    if not args.report and not args.scope:
        ap.error('--scope is required for template/validate/import')
    if args.report and args.scope:
        ap.error('--report always includes both sports and shared costs; omit --scope')
    if not (args.report or args.attest_wagers) and (args.start_date or args.end_date or args.include_db):
        ap.error('Dates require --report or --attest-wagers')
    if args.include_db and not args.report:
        ap.error('--include-db requires --report')
    if args.attest_wagers and not all((args.start_date,args.end_date,args.currency,args.statement)):
        ap.error('--attest-wagers requires --scope, --currency, --start-date, --end-date and --statement')
    if not args.attest_wagers and (args.currency or args.statement):
        ap.error('--currency and --statement require --attest-wagers')
    try:
        if args.template:
            result = template(args.scope)
        elif args.attest_wagers:
            result = attest_inventory(read_sources(),args.scope,args.currency,args.start_date,args.end_date,args.statement)
        elif args.validate or args.import_json:
            path = args.validate or args.import_json
            result = import_document(json.loads(path.read_text(encoding='utf-8-sig')), args.scope,
                                     dry_run=bool(args.validate))
        else:
            sources = read_sources()
            result = summarize(sources, start_date=args.start_date, end_date=args.end_date, inventory_coverage=read_coverage())
            if args.include_db:
                result['legacy_nfl_cross_check'] = journal.cross_check(sources['nfl'], journal.legacy_confirmations())
            journal.write_report(result, REPORT)
            REPORT.with_suffix('.md').write_text(markdown(result), encoding='utf-8')
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        ap.exit(2, f'Accounting failed: {exc}\n')


if __name__ == '__main__':
    main()
