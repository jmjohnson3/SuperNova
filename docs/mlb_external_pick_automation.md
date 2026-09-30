# MLB External Pick Automation

The MLB pipeline can fetch/import exports from external pick platforms and
compare them against SuperNovaBets prop predictions.

The fetcher only uses direct export/feed URLs that you are allowed to access. It
does not scrape private app pages or bypass platform logins.

Default drop folder:

```text
data/external_picks/mlb
```

Each scheduled daily/pre-game run now:

1. Fetches configured external CSV/JSON/HTML-table feeds into that folder.
2. Scans the folder for CSVs.
3. Imports CSV rows idempotently.
4. Rebuilds the external comparison report.
5. Lets matching same-side external picks help a prop qualify for the `$1 MICRO TEST` lane.

External agreement never promotes a prop directly to starter or bankroll.

## File Naming

Use either a platform subfolder:

```text
data/external_picks/mlb/outlier/2026-07-27.csv
data/external_picks/mlb/edge_terminal/2026-07-27.csv
```

or include the platform in the filename:

```text
data/external_picks/mlb/outlier_2026-07-27.csv
data/external_picks/mlb/edge_terminal_2026-07-27.csv
```

## Supported CSV Columns

The importer accepts common aliases. Useful columns include:

```text
platform/source
date/game_date
player/player_name
market/stat/prop
side/selection/pick
book/sportsbook/bookmaker
line/book_line
price/odds/american_odds
prob/probability/win_prob
ev/edge/expected_value
grade/rating
```

## Optional Env Vars

```powershell
$env:MLB_EXTERNAL_PICK_SOURCE_URLS="outlier|csv|https://docs.google.com/spreadsheets/d/.../export?format=csv;edge_terminal|json|https://example.com/api/picks"
$env:MLB_EXTERNAL_PICK_FETCH_HEADERS_JSON='{"Authorization":"Bearer ${EDGE_TERMINAL_API_TOKEN}"}'
$env:MLB_EXTERNAL_PICK_SOURCES_JSON="C:\Users\josh\Git\SuperNovaBets\config\mlb_external_pick_sources.json"
$env:MLB_EXTERNAL_PICKS_DIR="C:\Users\josh\Git\SuperNovaBets\data\external_picks\mlb"
$env:MLB_EXTERNAL_PICKS_GLOB="*.csv"
$env:MLB_EXTERNAL_PICKS_CSVS="C:\exports\outlier.csv;C:\exports\edge_terminal.csv"
$env:MLB_EXTERNAL_PICKS_PLATFORM="outlier"
```

## Source Config

For multiple feeds or per-source auth, create:

```text
config/mlb_external_pick_sources.json
```

Template:

```text
config/mlb_external_pick_sources.example.json
```

Example:

```json
{
  "sources": [
    {
      "platform": "outlier",
      "format": "csv",
      "url": "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/export?format=csv&gid=0"
    },
    {
      "platform": "edge_terminal",
      "format": "json",
      "url": "https://example.com/api/mlb/value-picks",
      "json_path": "data.picks",
      "headers": {
        "Authorization": "Bearer ${EDGE_TERMINAL_API_TOKEN}"
      }
    }
  ]
}
```

Supported source formats:

```text
csv
json
html_table
auto
```

Manual smoke run:

```powershell
python -m mlb_pipeline.modeling.external_pick_fetcher --date 2026-07-27 --import-after-fetch
python -m mlb_pipeline.modeling.external_pick_ledger --date 2026-07-27 --skip-import
```

Report:

```text
reports/mlb_external_pick_fetch_latest.md
reports/mlb_external_model_comparison_latest.md
```
