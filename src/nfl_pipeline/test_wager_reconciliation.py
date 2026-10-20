"""Receipt accounting must not turn recommendations or missing costs into profits."""
from datetime import datetime, timezone
import json
from decimal import Decimal

import pytest

from nfl_pipeline import wager_reconciliation as wr


def placed(**changes):
    row = dict(event_id="placed-1", kind="wager_placed", wager_id="wager-1", book="fanduel",
        ticket_id="receipt-1", receipt_reference="local-receipt.pdf", bet_description="Test game receiving yards over 40.5",
        source_kind="sharp", recommendation_id="41", currency="USD", stake="10.00",
        executed_price=-120, recommended_price=-110, placed_at_utc="2026-10-01T18:00:00Z")
    row.update(changes)
    return row


def settled(**changes):
    row = dict(event_id="settled-1", kind="wager_settled", wager_id="wager-1", result="win",
        cash_return="18.33", settled_at_utc="2026-10-02T01:00:00Z", receipt_reference="settlement.pdf")
    row.update(changes)
    return row


def cost(**changes):
    row = dict(event_id="cost-1", kind="cost", cost_id="invoice-1", currency="USD", amount="2.50",
        incurred_at_utc="2026-10-01T00:00:00Z", category="data", description="NFL allocated feed cost",
        receipt_reference="invoice.pdf")
    row.update(changes)
    return row


def attestation(**changes):
    row = dict(event_id="coverage-1", kind="cost_coverage", currency="USD", start_date="2026-10-01",
        end_date="2026-10-02", attestation="All NFL costs for these dates have been supplied")
    row.update(changes)
    return row


def envelope(*events):
    return dict(schema_version=1, events=list(events))


def summary(*events, **kwargs):
    return wr.summarize(list(events), start_date="2026-10-01", end_date="2026-10-02", **kwargs)


def test_gross_net_slippage_and_actual_return():
    report = summary(placed(), settled(), cost())
    usd = report["currencies"]["USD"]
    assert usd["gross_profit"] == "8.33"
    assert usd["gross_roi"] == pytest.approx(.833)
    assert usd["recorded_costs"] == "2.50"
    assert usd["net_after_recorded_costs"] == "5.83"
    assert usd["net_profit"] is None
    assert report["wagers"][0]["implied_probability_slippage"] == pytest.approx(120/220-110/210)
    assert report["wagers"][0]["profit_per_unit_change"] < 0
    complete = summary(placed(), settled(), cost(), attestation())
    assert complete["currencies"]["USD"]["net_profit"] == "5.83"
    assert complete["cash_authorized"] is False


def test_open_bets_attempts_and_missing_odds_are_not_losses():
    wager = placed(); del wager["recommended_price"]
    attempts = [dict(event_id=f"attempt-{status}", kind="execution_attempt", book="fanduel",
        source_kind="sharp", recommendation_id="42", status=status, observed_at_utc="2026-10-01T19:00:00Z",
        reason="Recorded by user", currency="USD", observed_price=-115, available_stake="1.00")
        for status in ("available", "declined", "not_placed")]
    report = summary(wager, *attempts)
    assert report["currencies"]["USD"]["open_stake"] == "10.00"
    assert report["currencies"]["USD"]["settled_wagers"] == 0
    assert report["currencies"]["USD"]["gross_roi"] is None
    assert report["wagers"][0]["profit"] is None
    assert report["wagers"][0]["implied_probability_slippage"] is None
    assert report["attempts"] == {"available": 1, "declined": 1, "not_placed": 1}


def test_import_is_idempotent_and_does_not_rewrite_batches(tmp_path):
    result = wr.import_document(envelope(placed()), tmp_path)
    assert result["added"] == 1
    path = next((tmp_path/"batches").glob("*.json"))
    original = path.read_bytes()
    assert wr.import_document(envelope(placed()), tmp_path)["written"] is False
    assert path.read_bytes() == original
    wr.import_document(envelope(settled()), tmp_path)
    assert len(wr.read_events(tmp_path)) == 2
    assert path.read_bytes() == original


def test_conflicting_events_and_duplicate_tickets_rejected_atomically(tmp_path):
    wr.import_document(envelope(placed()), tmp_path)
    original = {p.name:p.read_bytes() for p in (tmp_path/"batches").glob("*.json")}
    with pytest.raises(ValueError, match="Conflicting event_id"):
        wr.import_document(envelope(cost(), placed(stake="20.00")), tmp_path)
    with pytest.raises(ValueError, match="Duplicate wager or sportsbook ticket"):
        wr.import_document(envelope(placed(event_id="copy", wager_id="new-name")), tmp_path)
    assert {p.name:p.read_bytes() for p in (tmp_path/"batches").glob("*.json")} == original
    assert not (tmp_path/".import.lock").exists()


def test_settlement_requires_placement_and_cannot_rewrite_result(tmp_path):
    with pytest.raises(ValueError, match="no actual placement"):
        wr.import_document(envelope(settled()), tmp_path)
    # A batch may list its settlement before its receipt; validation covers the whole batch.
    wr.import_document(envelope(settled(), placed()), tmp_path)
    with pytest.raises(ValueError, match="Settlement already recorded"):
        wr.import_document(envelope(settled(event_id="settlement-2", result="loss", cash_return="0.00")), tmp_path)
    with pytest.raises(ValueError, match="precedes placement"):
        summary(placed(), settled(settled_at_utc="2026-10-01T17:00:00Z"))


@pytest.mark.parametrize("changes", [
    {"stake":"0"}, {"stake":"nan"}, {"stake":"-1"}, {"stake":"1.001"}, {"stake":True},
    {"executed_price":0}, {"executed_price":-99}, {"executed_price":110.5},
    {"placed_at_utc":"2026-10-01T18:00:00"}, {"recommendation_id":""}, {"ticket_id":""},
    {"source_kind":"paper"}, {"execution_status":"simulated"}, {"currency":"US"},
])
def test_invalid_or_paper_receipts_rejected(changes):
    with pytest.raises(ValueError):
        wr.normalize_event(placed(**changes))


def test_explicit_zero_costs_and_gaps():
    report = summary(placed(), settled(), attestation())
    assert report["currencies"]["USD"]["net_profit"] == "8.33"
    gap = summary(placed(), settled(), attestation(end_date="2026-10-01"))
    assert gap["currencies"]["USD"]["net_profit"] is None
    joined = summary(placed(), settled(), attestation(end_date="2026-10-01"),
        attestation(event_id="coverage-2", start_date="2026-10-02"))
    assert joined["currencies"]["USD"]["cost_coverage_complete"] is True


def test_report_period_is_settlement_cash_basis_and_no_fx():
    old = placed(placed_at_utc="2026-09-30T18:00:00Z")
    eur = placed(event_id="placed-2", wager_id="wager-2", book="otherbook", currency="EUR")
    report = summary(old, settled(), eur, cost(currency="EUR"))
    assert report["currencies"]["USD"]["placed_in_period"] == 0
    assert report["currencies"]["USD"]["settled_wagers"] == 1
    assert report["currencies"]["EUR"]["open_stake"] == "10.00"
    assert report["currencies"]["EUR"]["recorded_costs"] == "2.50"


def test_void_and_cashout_use_receipt_cash_not_model_result():
    void = summary(placed(), settled(result="void", cash_return="10.00"))
    assert void["currencies"]["USD"]["gross_profit"] == "0.00"
    cashout = summary(placed(), settled(result="cashout", cash_return="4.25"))
    assert cashout["currencies"]["USD"]["gross_profit"] == "-5.75"
    unusual = summary(placed(), settled(result="loss", cash_return="1.00"))
    assert unusual["currencies"]["USD"]["gross_profit"] == "-9.00"
    assert any("receipt return differs" in warning for warning in unusual["warnings"])


def test_validation_template_and_report_create_no_records(tmp_path, capsys):
    journal = tmp_path/"new-journal"
    assert wr.import_document(envelope(placed()), journal, dry_run=True)["written"] is False
    assert not journal.exists()
    assert wr.template()["import_document"]["events"] == []
    assert wr.main(["--journal", str(journal), "--report", "--start-date", "2026-10-01", "--end-date", "2026-10-02"]) == 0
    assert json.loads(capsys.readouterr().out)["wagers"] == []
    assert not journal.exists()


def test_report_output_is_derived_artifact_not_journal(tmp_path, capsys):
    journal = tmp_path/"journal"
    output = tmp_path/"reports"/"latest.json"
    assert wr.main(["--journal", str(journal), "--report", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["receipt_wagers_through_end"] == 0
    assert report["wagers"] == []
    assert report["warnings"]
    assert not journal.exists()


def test_report_output_cannot_overwrite_journal(tmp_path):
    wr.import_document(envelope(placed()), tmp_path)
    batch = next((tmp_path/"batches").glob("*.json"))
    before = batch.read_bytes()
    with pytest.raises(SystemExit):
        wr.main(["--journal", str(tmp_path), "--report", "--output", str(batch)])
    assert batch.read_bytes() == before


def test_journal_checksum_and_concurrent_import_fail_closed(tmp_path):
    wr.import_document(envelope(placed()), tmp_path)
    (tmp_path/".import.lock").write_text("other importer")
    with pytest.raises(ValueError, match="Another import"):
        wr.import_document(envelope(cost()), tmp_path)
    assert (tmp_path/".import.lock").read_text() == "other importer"
    path = next((tmp_path/"batches").glob("*.json"))
    doc = json.loads(path.read_text()); doc["events"][0]["stake"] = "99.00"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="integrity verification"):
        wr.read_events(tmp_path)


def test_legacy_confirmations_are_cross_checks_never_duplicate_money():
    legacy = {"records":[dict(source_kind="sharp", recommendation_id="41", book="fanduel",
        stake=Decimal("10"), executed_price=-115),
        dict(source_kind="cash", recommendation_id="42", stake=1, executed_price=-110)]}
    checked = wr.cross_check([placed()], legacy)
    assert checked["counts"] == {"mismatch_review":1, "missing_receipt":1}
    assert checked["records"][0]["discrepancies"] == ["executed_price"]
    assert checked["included_in_receipt_pnl"] is False
    assert summary(placed())["currencies"]["USD"]["open_wagers"] == 1


def test_money_and_times_normalized_for_retry(tmp_path):
    first = placed(stake=10, placed_at_utc="2026-10-01T12:00:00-06:00")
    wr.import_document(envelope(first), tmp_path)
    retried = wr.import_document(envelope(placed()), tmp_path)
    assert retried["duplicate_events"] == 1


def test_duplicate_cost_and_unknown_wrapper_rejected(tmp_path):
    wr.import_document(envelope(cost()), tmp_path)
    with pytest.raises(ValueError, match="Duplicate cost_id"):
        wr.import_document(envelope(cost(event_id="cost-copy")), tmp_path)
    with pytest.raises(ValueError, match="Import requires exactly"):
        wr.parse_document(wr.template())


def test_database_cross_check_is_read_only_and_excludes_unconfirmed(monkeypatch):
    import psycopg2
    class Cursor:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def execute(self, sql, parameters=None):
            self.sql = sql
            assert sql.startswith(("SELECT", "SET LOCAL"))
        def fetchone(self):
            return {"relation": "present"}
        def fetchall(self):
            if "nfl_cash_execution_events" in self.sql:
                return [dict(ledger_id=1, state="published", payload={"stake":1}),
                    dict(ledger_id=2, state="confirmed", payload={"execution_confirmed_at":"2026-10-01T18:00:00Z",
                         "stake":1, "executed_price":-110, "forecast":{"book":"fanduel"}})]
            assert "WHERE placed_at IS NOT NULL" in self.sql
            return [dict(alert_id=3, book="fanduel", stake=1, taken_price=None,
                         placed_at=datetime(2026,10,1,18,tzinfo=timezone.utc))]
    class Connection:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def set_session(self, **kwargs):
            assert kwargs == {"readonly": True}
        def cursor(self, **kwargs):
            return Cursor()
    monkeypatch.setattr(psycopg2, "connect", lambda dsn: Connection())
    result = wr.legacy_confirmations()
    assert [r["recommendation_id"] for r in result["records"]] == ["2", "3"]
    assert result["records"][1]["executed_price"] is None
