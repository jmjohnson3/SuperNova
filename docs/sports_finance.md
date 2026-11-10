# NFL / NBA actual-profit accounting

`sports_finance` combines actual receipt records for NFL and NBA and subtracts
shared operating expenses once. It reads the existing private NFL receipt journal;
it does not change scoring, experiment registrations, recommendations or stakes.

Private journals and reports are excluded from Git:

- `data/nfl_wager_reconciliation/`: existing NFL receipts and direct expenses.
- `data/nba_wager_reconciliation/`: NBA receipts and direct expenses.
- `data/shared_wager_reconciliation/`: shared data, hosting and software expenses.
- `data/sports_finance/reports/latest.json` and `.md`: combined accounting.

## Import actual records

Generate the JSON field guide for the desired scope, then fill its empty
`import_document` with actual receipt records. The guide deliberately contains no
example wagers that could accidentally become real records.

```powershell
.\.venv\Scripts\python.exe -m sports_finance --scope nba --template
.\.venv\Scripts\python.exe -m sports_finance --scope nba --validate data\my_nba_receipts.json
.\.venv\Scripts\python.exe -m sports_finance --scope nba --import-json data\my_nba_receipts.json
```

Use `--scope nfl` for NFL and `--scope shared` for common expenses. A placement needs
the actual book, ticket ID, stake, American price, placement timestamp, selection,
receipt reference and recommendation ID. A settlement needs the actual cash return
**including returned stake**. Keep unsettled bets open; do not import paper grades
as sportsbook settlements. Deposits and withdrawals are not profits or expenses.

For an NBA receipt linked to a check request, use `source_kind: "external"` and a
reference such as `recommendation_id: "nba-price-check:123"`. The request alone is
not a receipt. This format documents wagers the user actually placed; it grants
no authorization to place one.

Retries reuse the same event ID. Duplicate book/ticket IDs across sports are rejected.
Shared expenses use one globally unique cost ID and go in the shared journal once.
If an invoice is explicitly allocated, use documented, distinct allocation cost IDs
whose amounts sum to the invoice. Do not import the full invoice in every scope.

## Weekly review

```powershell
.\.venv\Scripts\python.exe -m sports_finance --report --start-date 2026-10-26 --end-date 2026-11-01 --include-db
```

The report shows settled receipt profit, costs, open stakes, and results by sport,
book and currency. No currencies are converted. Shared costs are subtracted from
the combined total; sport/book rows show gross profit because the tool does not
invent expense allocations. NFL database confirmations are cross-checked against
receipts and shown as missing or mismatched where appropriate, never added to profit.

The report can display `net_after_recorded_costs` before completeness is verified.
Full `net_profit` remains unknown until both of these are supplied for the reporting
period and currency:

1. `cost_coverage` records in NFL, NBA and shared journals, explicitly confirming all
   expenses, including a truly zero-cost scope.
2. Wager-inventory attestations for NFL and NBA, covering every actual placement,
   settlement and open wager through period end, including opening exposure.

Only after checking your sportsbook history, attest each sport:

```powershell
.\.venv\Scripts\python.exe -m sports_finance --attest-wagers --scope nba --currency USD --start-date 2026-10-26 --end-date 2026-10-28 --statement "All NBA placements, settlements and opening exposure through this period are supplied."
```

Repeat for NFL using the actual scope and dates. No command should attest facts you
have not checked. An attestation is tied to the receipt data present when it was
made. Adding a missing receipt or settlement within that history invalidates it
until the inventory is checked again. Future activity does not invalidate an older
historical reconciliation. These attestations are user-confirmed, not independent
sportsbook verification.

An explicit statement that no wagers were placed can establish zero wagers and
zero gross betting profit for that period. It does not establish zero operating
costs: reconciled net profit stays unknown until the expense inventory is complete.

The JSON also includes break-even turnover at hypothetical 1%, 3% and 5% edges.
These are arithmetic scenarios, not measured edges or staking advice. With incomplete
costs, the required turnover is explicitly a lower bound based on recorded expenses.
Continue evaluating execution, calibration and close coverage in the frozen
experiment reports alongside actual profit. No financial report enables betting.
