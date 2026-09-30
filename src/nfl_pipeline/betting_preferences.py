"""Execution/display preference, separate from multi-book modeling evidence."""
from urllib.parse import urlparse
from nfl_pipeline.fanduel_links import single_betslip_url

EXECUTION_BOOK = 'fanduel'
EXECUTION_BOOK_LABEL = 'FanDuel'
FANDUEL_LINK_PATTERN = r'^https://([a-z0-9-]+\.)*fanduel\.com(/|\?|$)'
REAL_TIERS = {'micro', 'micro_projection', 'starter', 'bankroll', 'locked_micro', 'cash_trial'}


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
