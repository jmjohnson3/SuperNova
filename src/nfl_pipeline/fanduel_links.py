"""Selection-preserving FanDuel links for publication, not forecast mutation."""
from __future__ import annotations

import html
import re
from urllib.parse import parse_qsl, urlencode, urlsplit

CONTRACT = 'fanduel-betslip-links-v1'
BETSLIP_BASE = 'https://account.sportsbook.fanduel.com/sportsbook/addToBetslip'
MAX_LEGS = 20


def provider_link(link):
    """Keep the original destination as a fallback, with safe Markdown boundaries."""
    if not isinstance(link, str):
        return None
    link = html.unescape(link.strip())
    if not link or any(c.isspace() or ord(c) < 32 or c in '<>\\' for c in link):
        return None
    try:
        url = urlsplit(link)
        host = url.hostname or ''
        if (url.scheme != 'https' or url.username or url.password or url.port is not None
                or not (host == 'fanduel.com' or host.endswith('.fanduel.com'))):
            return None
    except ValueError:
        return None
    return link


def selections(link):
    """Read scalar or indexed IDs. Never borrow a missing ID or a different leg."""
    link = provider_link(link)
    if not link:
        return ()
    url = urlsplit(link)
    if url.path.rstrip('/') not in {'/addToBetslip', '/sportsbook/addToBetslip'} or url.fragment:
        return ()
    try:
        query = parse_qsl(url.query, keep_blank_values=True, max_num_fields=100)
    except ValueError:
        return ()
    legs = {}
    styles = set()
    for key, value in query:
        if not key.startswith(('marketId', 'selectionId')):
            continue
        match = re.fullmatch(r'(marketId|selectionId)(?:\[(\d+)\])?', key)
        if not match:
            return ()
        field, index = match.groups()
        styles.add('scalar' if index is None else 'indexed')
        index = int(index or 0)
        if index >= MAX_LEGS:
            return ()
        leg = legs.setdefault(index, {})
        if field in leg:
            return ()
        pattern = r'[0-9]+(?:\.[0-9]+)?' if field == 'marketId' else r'[0-9]+'
        if not re.fullmatch(pattern, value):
            return ()
        leg[field] = value
    if len(styles) != 1 or not legs:
        return ()
    result = []
    markets = {}
    for _, leg in sorted(legs.items()):
        if set(leg) != {'marketId', 'selectionId'}:
            return ()
        market, selection = leg['marketId'], leg['selectionId']
        if market in markets and markets[market] != selection:
            return ()
        markets[market] = selection
        if (market, selection) not in result:
            result.append((market, selection))
    return tuple(result)


def _betslip_url(legs):
    query = [(f'{key}[{i}]', value) for i, (market, selection) in enumerate(legs)
             for key, value in (('marketId', market), ('selectionId', selection))]
    # Avoid raw brackets in Discord URLs; preserve IDs as strings, never floats.
    return BETSLIP_BASE + '?' + urlencode(query)


def single_betslip_url(link):
    legs = selections(link)
    return _betslip_url(legs) if len(legs) == 1 else None


def parlay_betslip_url(links, rows=None):
    legs = []
    markets = {}
    from nfl_pipeline import fanduel_state
    if rows is not None and fanduel_state.state():
        links = [betslip_for_row(r) for r in rows]
    for link in links:
        parsed = selections(link)
        # Each displayed pick must describe one selection; never omit invalid picks.
        if len(parsed) != 1:
            return None
        market, selection = parsed[0]
        if market in markets and markets[market] != selection:
            return None
        markets[market] = selection
        if parsed[0] not in legs:
            legs.append(parsed[0])
    return _betslip_url(legs) if 2 <= len(legs) <= MAX_LEGS else None


def betslip_for_row(row):
    """Add-to-slip URL valid in the bettor's state, or None.

    With NFL_FANDUEL_STATE set, IDs come only from that state's FanDuel feed (provider market IDs belong
    to other states and are rejected). Without it, the provider's IDs are used as before.
    """
    from nfl_pipeline import fanduel_state
    if fanduel_state.state():
        leg = fanduel_state.resolve_row(row)
        return _betslip_url([leg]) if leg else None
    return single_betslip_url(row.get('link'))


def row_link(row):
    if str(row.get('book') or '').lower() != 'fanduel':
        return ''
    original = provider_link(row.get('link'))
    betslip = betslip_for_row(row)
    if betslip:
        fallback = f' | [Provider link](<{original}>)' if original != betslip else ''
        return f' [Add to slip](<{betslip}>){fallback}'
    if original:
        return f' [Open FanDuel - manual selection](<{original}>)'
    return ' | Betslip link unavailable' if row.get('line') is not None else ''


def _with_model_projection(rendered, row):
    """Show the model's own number beside the line-anchored projection (display only).

    Market calibration pulls projections toward the line (fully, for stats whose model did not beat
    it), so proj= alone often just repeats the line. The unanchored number lets disagreements be followed.
    """
    trace = row.get('probability_trace') or {}
    model, projection = trace.get('model_projection'), row.get('projection')
    trust = (trace.get('market_calibration') or {}).get('projection_trust')
    if model is None or projection is None or row.get('line') is None or trust is None or trust >= 1.0:
        return rendered
    old = f"proj={float(projection):.2f}"
    new = f"model={float(model):.1f} | line-anchored={float(projection):.1f} ({float(trust):.0%} model)"
    return rendered.replace(old, new, 1)


def format_prop_row(row, *, action):
    # Reuse the frozen numeric formatter without changing its scoring fingerprint.
    from nfl_pipeline.modeling import predict_player_props as props
    rendered = props._format_prop_row(dict(row, link=None), action=action)
    rendered = _with_model_projection(rendered, row)
    from nfl_pipeline.forecast_outputs import display_suffix
    return rendered + display_suffix(row) + (row_link(row) if row.get('line') is not None else '')
