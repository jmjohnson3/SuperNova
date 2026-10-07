"""Execution/display preference, separate from multi-book modeling evidence."""
from urllib.parse import urlparse
from nfl_pipeline.fanduel_links import single_betslip_url

EXECUTION_BOOK = 'fanduel'
EXECUTION_BOOK_LABEL = 'FanDuel'
FANDUEL_LINK_PATTERN = r'^https://([a-z0-9-]+\.)*fanduel\.com(/|\?|$)'
REAL_TIERS = {'micro', 'micro_projection', 'starter', 'bankroll', 'locked_micro', 'cash_trial'}
# Markets that may be locked into a staked (non-paper) ledger tier. Everything else is
# still forecast, stored, graded and shown in Discord with links, but only as paper.
BET_PROP_STATS = frozenset({'receiving_yards'})
BET_GAME_MARKETS = frozenset()
# FanDuel offers priced off a sharp book (Pinnacle/exchange) stay paper research until the sharp-edge
# strategy shows positive CLV on the scorecard; flip to True to allow $1 micro locks.
# This gates the *forecast* path in predict_player_props, whose model edge the 2025 walk-forward
# replay disproved (docs/nfl_market_calibration.md). It stays off. The sharp_watch scanner below is
# a separate strategy with its own evidence and its own stake.
SHARP_EDGE_BETS_ENABLED = False
SHARP_EDGE_MIN_EV = 0.03

# --- sharp_watch scanner (live book-vs-sharp price gaps) -------------------------------------
# Books we can actually bet at. <= 10 named bookmakers cost one region per market at The Odds API,
# so adding a book here costs no extra credits.
SHARP_WATCH_BET_BOOKS = ('fanduel', 'draftkings')
# Only markets whose edge has been backtested may carry money. Everything else in
# NFL_SHARP_WATCH_MARKETS is still scanned, alerted and graded, but at stake 0 until it proves out
# on its own evidence in reports/nfl_sharp_alerts_latest.md.
SHARP_WATCH_STAKED_STATS = frozenset({'receiving_yards', 'rushing_yards', 'receptions'})
# Flat stake per pinged alert. 0.0 keeps every alert research-only.
SHARP_WATCH_STAKE = 5.0
SHARP_WATCH_MAX_STAKE_PER_DAY = 50.0
SHARP_WATCH_MAX_STAKE_PER_WEEK = 150.0
# Sticky: once realized profit on placed bets reaches this, staking stops until it is raised by hand.
SHARP_WATCH_LOSS_PAUSE = -200.0


def execution_link(link):
    try:
        parsed = urlparse(str(link or ''))
        host = parsed.hostname or ''
        return (parsed.scheme == 'https' and not parsed.username and not parsed.password
                and parsed.port is None and (host == 'fanduel.com' or host.endswith('.fanduel.com')))
    except ValueError:
        return False


def display_rows(rows):
    """Do not relabel another book's price, recalculate a pick, or mutate its lock."""
    visible = []
    for original in rows:
        book = str(original.get('book') or '').lower()
        projection_only = not book and original.get('line') is None and original.get('side') is None
        if book != EXECUTION_BOOK and not projection_only:
            continue
        row = dict(original)
        if not execution_link(row.get('link')):
            row['link'] = None
            if row.get('tier') in REAL_TIERS:
                row['tier'] = 'paper'
                row['reasons'] = str(row.get('reasons') or '') + ';fanduel_execution_link_missing'
        elif (row.get('tier') in REAL_TIERS - {'locked_micro'}
              and not single_betslip_url(row.get('link'))):
            row['tier'] = 'paper'
            row['reasons'] = str(row.get('reasons') or '') + ';fanduel_selection_ids_missing_or_invalid'
        visible.append(row)
    return visible
