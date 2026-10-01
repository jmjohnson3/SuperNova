# NFL FanDuel Discord Links

Scheduled matchup cards use `nfl_pipeline.fanduel_links` for game picks,
player picks, and research parlays. This is a publication-only change: the
original provider URL, forecast, quote, and immutable lock are not rewritten.
The frozen numeric scoring modules are unchanged.

## State-specific IDs (October 1)

FanDuel market IDs differ by state; only selection IDs are shared. The same Total Points market was
`734.187749378` in NJ, `739.187747322` in OH and `742.187745814` in PA. Provider links carry another
state's market ID: SportsGameOdds sends Illinois IDs (`717.`), and The Odds API sends `42.`, which no
US state checked uses. FanDuel opens, verifies location, then shows "Selection not added".
Prefixes seen: AZ 704, CO 708, CT 709, DC 711, IL 717, IN 718, IA 719, KS 720, KY 721, LA 722,
MD/ME 724, MA 725, MI 726, MO 729, NJ 734, NY 736, NC 737, OH 739, PA 742, TN 747, VT 750, VA 751,
WV 754, WY 756.

With `NFL_FANDUEL_STATE` set (HKCU environment; `co` here), `fanduel_state.py` reads that state's
public FanDuel event feed (the web client's `sbapi.<state>` endpoints: one event-list call, then one
call per game per tab, cached per process, 90 s network budget). It finds the bet by teams, player
(name normalised, Jr./III ignored), stat, side and exact line, and only on an active runner. Only then
is `Add to slip` rendered, with that state's IDs. Anything else, such as a moved line, a suspended
runner, an unsupported market (TD props) or a failed lookup, renders `Open FanDuel - manual selection`.
The resolver also catches provider links whose selection IDs point at the wrong player or line. This
applies to matchup cards, research parlays and sharp-watch alerts. Without the setting, provider IDs
are used as before.

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
