"""Cross-sport receipt accounting; no journal, model or betting-policy mutations."""
from decimal import Decimal

import pytest

import sports_finance as sf
from nfl_pipeline.test_wager_reconciliation import placed, settled, cost, attestation, envelope


def report(sources):
    return sf.summarize(sources, start_date='2026-10-01', end_date='2026-10-02')


def test_actual_returns_split_by_sport_and_book_shared_cost_counted_once():
    sources = dict(nfl=[placed(), settled(), cost(), attestation()],
                   nba=[placed(ticket_id='nba-ticket', book='draftkings'), settled(result='loss',cash_return='0.00'),
                        cost(cost_id='nba-cost', amount='1.00'), attestation()],
                   shared=[cost(cost_id='shared-cost',amount='3.00'), attestation()])
    result = report(sources)
    usd = result['currencies']['USD']
    assert usd['gross_profit'] == '-1.67'
    assert usd['recorded_costs'] == '6.50'
    assert usd['net_after_recorded_costs'] == '-8.17'
    assert usd['receipt_net_after_complete_costs'] == '-8.17'
    assert usd['cost_coverage_complete'] is True
    assert usd['net_profit'] is None  # no assertion that every actual wager was supplied
    assert usd['costs_by_scope'] == {'nfl':'2.50','nba':'1.00','shared':'3.00'}
    assert result['by_sport_book']['nfl']['fanduel']['USD']['gross_profit'] == '8.33'
    assert result['by_sport_book']['nba']['draftkings']['USD']['gross_profit'] == '-10.00'
    assert {r['wager_id'] for r in result['wagers']} == {'nfl:wager-1','nba:wager-1'}
    assert result['cash_authorized'] is False


def test_one_scope_cost_attestation_cannot_certify_other_expenses():
    usd = report(dict(nfl=[placed(), settled(), attestation()], nba=[], shared=[]))['currencies']['USD']
    assert usd['cost_coverage_by_scope'] == {'nfl':True,'nba':False,'shared':False}
    assert usd['receipt_net_after_complete_costs'] is None
    assert usd['break_even_scenarios'][0]['basis'] == 'recorded_costs_only_lower_bound'


def test_no_records_is_unknown_and_explicit_zero_costs_do_not_invent_wagers():
    empty = report({})
    assert empty['receipt_wagers_through_end'] == 0 and empty['currencies'] == {}
    assert 'unknown' in sf.markdown(empty)
    zero = report({scope:[attestation()] for scope in sf.SCOPES})
    assert zero['currencies']['USD']['net_profit'] is None
    assert zero['currencies']['USD']['gross_roi'] is None
    assert zero['wagers'] == []
    assert zero['currencies']['USD']['cost_coverage_complete'] is True


def test_duplicate_tickets_and_invoices_across_sports_are_rejected():
    with pytest.raises(ValueError, match='Duplicate wager or sportsbook ticket'):
        report(dict(nfl=[placed()], nba=[placed()]))
    with pytest.raises(ValueError, match='Duplicate cost_id'):
        report(dict(nfl=[cost()], shared=[cost()]))
    with pytest.raises(ValueError, match='Shared scope'):
        report(dict(shared=[placed()]))


def test_explicit_no_wagers_shows_zero_gross_without_assuming_zero_costs(tmp_path):
    for sport in ('nfl', 'nba'):
        sf.attest_inventory({}, sport, 'USD', '2026-10-01', '2026-10-02',
                            'I have not placed any bets.', directory=tmp_path)
    result = sf.summarize({}, start_date='2026-10-01', end_date='2026-10-02',
                          inventory_coverage=sf.read_coverage(tmp_path))
    usd = result['currencies']['USD']
    assert result['wagers'] == [] and result['receipt_wagers_through_end'] == 0
    assert usd['gross_profit'] == '0.00' and usd['open_stake'] == '0.00'
    assert usd['wager_inventory_complete'] is True
    assert usd['cost_coverage_complete'] is False and usd['net_profit'] is None
    assert not any('actual execution remains unknown' in w for w in result['warnings'])
    assert 'unknown' in sf.markdown(result)
    later = sf.summarize({}, start_date='2026-10-01', end_date='2026-10-03',
                         inventory_coverage=sf.read_coverage(tmp_path))
    assert later['currencies']['USD']['wager_inventory_complete'] is False


def test_open_wagers_are_exposure_not_losses_and_currencies_not_combined():
    result = report(dict(nba=[placed(currency='EUR')], shared=[cost(currency='USD')]))
    eur = result['currencies']['EUR']
    assert eur['gross_profit'] == '0.00' and eur['open_stake'] == '10.00'
    assert eur['gross_roi'] is None and eur['settled_wagers'] == 0
    assert result['currencies']['USD']['net_after_recorded_costs'] == '-2.50'


def test_break_even_is_explicitly_hypothetical_and_rounds_up():
    scenario = sf.break_even('250.00', costs_complete=False)[1]
    assert scenario == dict(assumed_edge='0.03',required_turnover='8333.34',one_dollar_bets=8334,
                            basis='recorded_costs_only_lower_bound')
    assert Decimal(scenario['required_turnover']) * Decimal(scenario['assumed_edge']) >= 250


def test_import_is_idempotent_and_checks_all_sports_before_writing(tmp_path):
    paths = {scope:tmp_path/scope for scope in sf.SCOPES}
    assert sf.import_document(envelope(placed()),'nfl',paths=paths,dry_run=True)['written'] is False
    assert not (tmp_path/'nfl').exists()
    assert sf.import_document(envelope(placed()),'nfl',paths=paths)['added'] == 1
    assert sf.import_document(envelope(placed()),'nfl',paths=paths)['written'] is False
    with pytest.raises(ValueError, match='Duplicate wager or sportsbook ticket'):
        sf.import_document(envelope(placed()),'nba',paths=paths)
    assert not (tmp_path/'nba').exists()
    assert sf.read_sources(paths)['nba'] == []
    with pytest.raises(ValueError, match='Shared scope'):
        sf.import_document(envelope(placed(ticket_id='another')),'shared',paths=paths)
    assert not (tmp_path/'shared').exists()


def test_report_never_reads_paper_results_into_receipt_profit():
    sources = dict(nfl=[placed(),settled()])
    result = report(sources)
    assert result['receipt_wagers_through_end'] == 1
    assert result['currencies']['USD']['gross_profit'] == '8.33'
    assert 'paper' not in result['by_sport_book']
    assert sf.template('nba')['import_document']['events'] == []


def test_complete_inventory_enables_net_and_new_receipts_invalidate_attestation(tmp_path):
    sources = dict(nfl=[placed(),settled(),attestation()],nba=[attestation()],shared=[cost(),attestation()])
    for sport in ('nfl','nba'):
        assert sf.attest_inventory(sources,sport,'USD','2026-10-01','2026-10-02',
                                  'All wagers, settlements and opening exposure supplied',directory=tmp_path)['written']
    assert not sf.attest_inventory(sources,'nba','USD','2026-10-01','2026-10-02',
                                  'All wagers, settlements and opening exposure supplied',directory=tmp_path)['written']
    coverage = sf.read_coverage(tmp_path)
    complete = sf.summarize(sources,start_date='2026-10-01',end_date='2026-10-02',inventory_coverage=coverage)
    assert complete['currencies']['USD']['net_profit'] == '5.83'
    sources['nba'].append(placed(ticket_id='nba-ticket'))
    changed = sf.summarize(sources,start_date='2026-10-01',end_date='2026-10-02',inventory_coverage=coverage)
    assert changed['currencies']['USD']['net_profit'] is None
    assert changed['currencies']['USD']['wager_inventory_by_scope'] == {'nfl':True,'nba':False}


def test_future_wagers_do_not_invalidate_historical_inventory_and_tampering_fails(tmp_path):
    sources = dict(nfl=[placed(),settled()])
    sf.attest_inventory(sources,'nfl','USD','2026-10-01','2026-10-02','Complete',directory=tmp_path)
    coverage = sf.read_coverage(tmp_path)
    sources['nfl'].append(placed(event_id='later',wager_id='later',ticket_id='later',placed_at_utc='2026-10-03T18:00:00Z'))
    summary = sf.summarize(sources,start_date='2026-10-01',end_date='2026-10-02',inventory_coverage=coverage)
    assert summary['currencies']['USD']['wager_inventory_by_scope']['nfl'] is True
    path = next(tmp_path.glob('*.json'))
    path.write_text(path.read_text().replace('Complete','Altered'))
    with pytest.raises(ValueError,match='integrity check'):
        sf.read_coverage(tmp_path)
