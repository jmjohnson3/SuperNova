# NFL FanDuel Discord Links

Scheduled matchup cards use `nfl_pipeline.fanduel_links` for game picks,
player picks, and research parlays. This is a publication-only change: the
original provider URL, forecast, quote, and immutable lock are not rewritten.
The frozen numeric scoring modules are unchanged.

## Link Behavior

- `Add to slip` uses the account sportsbook route with indexed, URL-encoded
  `marketId[0]` / `selectionId[0]` parameters, including for a single pick.
- `Provider link` retains the original source URL as a fallback. HTML-escaped
  query separators are decoded before rendering.
- Both scalar and indexed provider URLs are accepted. IDs are preserved as
  strings, including their market prefix; no IDs or share codes are invented.
- A homepage/event URL without exact selection IDs is labeled manual selection,
  not add-to-slip. It cannot be displayed or reserved as an executable cash pick.
- Research parlays require a valid single-selection URL for every displayed leg.
  Identical legs are deduplicated. Conflicting selections in one market, missing
  IDs, or nested multi-leg links prevent construction of a misleading partial slip.
- Publication manifests include the provider and rendered betslip URLs.

## Verification and Limits

Run `python -m pytest src/nfl_pipeline/test_fanduel_links.py -q`.
Tests cover URL parsing, exact-ID preservation, rendering, fallback links,
ambiguous IDs, malformed links, parlays, and cash reservation validation.

These tests do not establish that FanDuel accepted a betslip. The URL can still
fail if the selection is suspended/removed or the app's login/location handoff
loses the destination. A parlay may contain combinations FanDuel does not permit.
The scripts do not place a wager, set stakes in the app, or bypass those checks.

After the next scheduled Discord post, test `Add to slip` on a still-available
selection and verify the exact player, side, and line in FanDuel without placing
a wager. If it only opens the app, compare the `Provider link` and provide the
device, app/browser destination, and broken selection URL for further diagnosis.
Old Discord messages retain their old URLs; this change does not edit them.

The provider's documented scalar source format is described at
https://the-odds-api.com/releases/deep-links.html and
https://sportsgameodds.com/docs/info/v1-to-v2.
