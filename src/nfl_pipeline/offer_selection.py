"""FanDuel offer identity and freshness checks, before any EV ranking."""
from collections import Counter
from datetime import datetime, timezone
import math

from nfl_pipeline.betting_preferences import EXECUTION_BOOK
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.markets import normalize_name, normalize_team, STAT_BY_MARKET

CONTRACT = 'nfl-fanduel-fresh-lock-v1'
MAX_QUOTE_AGE_MINUTES = 20


def quote_error(quote, cutoff, now, start):
    fetched = safe_time(quote.get('fetched_at_utc'))
    cutoff, now, start = safe_time(cutoff), safe_time(now), safe_time(start)
    if not all((fetched, cutoff, now, start)):
        return 'missing_quote_or_lock_timing'
    if not fetched <= cutoff <= now < start:
        return 'quote_or_lock_timing_invalid'
    if (now-fetched).total_seconds() > MAX_QUOTE_AGE_MINUTES*60:
        return 'stale_quote_refresh_required'
    return None


def valid_price(value):
    try:
        return math.isfinite(float(value)) and abs(float(value)) >= 100
    except (TypeError, ValueError):
        return False


def eligible_player_offers(offers, row, stat, cutoff, *, now=None):
    now = now or datetime.now(timezone.utc)
    names = {normalize_name(str(row.get(k) or '')) for k in
             ('player_name', 'context_player_name', 'context_player_name_norm')}
    names.discard('')
    teams = {normalize_team(row.get(k)) for k in ('team_abbr', 'opponent_abbr')}
    selected = {}; issues = Counter()
    for quote in offers:
        if quote.get('stat') != stat or normalize_name(str(quote.get('player_name_norm') or quote.get('player_name') or '')) not in names:
            continue
        if quote.get('bookmaker_key') != EXECUTION_BOOK:
            issues['other_book'] += 1; continue
        start = safe_time(row.get('start_ts_utc'))
        quote_teams = {normalize_team(quote.get(k)) for k in ('home_team', 'away_team')}
        if (None in teams or len(teams) != 2 or teams != quote_teams or not quote.get('event_id')
                or safe_time(quote.get('commence_time_utc')) != start):
            issues['offer_game_identity_mismatch'] += 1; continue
        error = quote_error(quote, cutoff, now, start)
        if error:
            issues[error] += 1; continue
        try:
            line = float(quote['line'])
        except (KeyError, TypeError, ValueError):
            line = float('nan')
        if (not math.isfinite(line) or not quote.get('offer_id')
                or STAT_BY_MARKET.get(quote.get('market_key')) != stat
                or not any(valid_price(quote.get(s+'_price')) for s in ('over', 'under'))):
            issues['invalid_offer_line_or_price'] += 1; continue
        key = (stat, line)
        prior = selected.get(key)
        if prior is None or (safe_time(quote['fetched_at_utc']), int(quote['offer_id'])) > (
                safe_time(prior['fetched_at_utc']), int(prior['offer_id'])):
            selected[key] = dict(quote)
            # An invalid opposite side is unknown, never evidence for no-vig.
            for side in ('over', 'under'):
                if not valid_price(quote.get(side+'_price')):
                    selected[key][side+'_price'] = None
    return [selected[k] for k in sorted(selected)], dict(issues)


def forecast_quote_error(row, now):
    if row.get('line') is None or row.get('side') is None:
        return None
    if row.get('book') != EXECUTION_BOOK:
        return 'execution_book_mismatch'
    q = (row.get('scoring_replay') or {}).get('offer') or {}
    if row.get('execution_contract') == CONTRACT and row.get('quote_fetched_at_utc'):
        q = dict(q, fetched_at_utc=row['quote_fetched_at_utc'])
    start = row.get('start_ts_utc') or (row.get('forecast_features') or {}).get('start_ts_utc') or q.get('commence_time_utc')
    return quote_error(q, row.get('prediction_context_cutoff_utc'), now, start)
