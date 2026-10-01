"""Group existing NFL selections into matchup-scoped Discord cards."""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone

from nfl_pipeline.integrity import utc
from nfl_pipeline.modeling import predict_player_props as props
from nfl_pipeline.betting_preferences import display_rows as preferred_rows
from nfl_pipeline import fanduel_links

CONTRACT = 'nfl-matchup-cards-v1'
DESCRIPTION_LIMIT = 3900
SHARP_HEADING = 'SHARP-LINE EDGES - research'
SHARP_EDGE_LIMIT = 10
from nfl_pipeline.betting_preferences import SHARP_EDGE_MIN_EV


def sharp_suffix(row):
    p = row.get('sharp_over_probability')
    side_p = p if row.get('side') == 'over' else (1 - p if p is not None else None)
    return (f" | vs {str(row.get('sharp_book') or '').title()} {float(row['sharp_line']):g}: fair {side_p:.0%},"
            f" EV {float(row['sharp_ev']):+.1%}") if side_p is not None and row.get('sharp_line') is not None else ''
PAPER_SECTIONS = (
    ('QB Passing Yards', {'QB'}, 'passing_yards'),
    ('QB Rushing Yards', {'QB'}, 'rushing_yards'),
    ('QB Passing TDs', {'QB'}, 'passing_tds'),
    ('RB Rushing Yards', {'RB'}, 'rushing_yards'),
    ('RB Receiving Yards', {'RB'}, 'receiving_yards'),
    ('RB Receptions', {'RB'}, 'receptions'),
    ('RB Rushing TDs', {'RB'}, 'rushing_tds'),
    ('WR/TE Receiving Yards', {'WR', 'TE'}, 'receiving_yards'),
    ('WR/TE Receptions', {'WR', 'TE'}, 'receptions'),
    ('WR/TE Receiving TDs', {'WR', 'TE'}, 'receiving_tds'),
)


def number(row, field, missing=-math.inf):
    value = row.get(field)
    return float(value) if value is not None and math.isfinite(float(value)) else missing


def selected_props(rows, paper_limit=10):
    """Select once across the slate, before grouping; never grant five micros per game."""
    rows = preferred_rows(rows)
    selected = []
    selected.extend(('$1 CASH TRIAL ELIGIBLE', r) for r in rows
                    if r.get('tier') == 'cash_trial' and r.get('cash_ledger_id') and r.get('ledger_locked'))
    micro = [r for r in rows if r.get('tier') == 'micro_projection' and r.get('ledger_locked')
             and r.get('line') is not None and r.get('side') in {'over', 'under'}]
    micro.sort(key=lambda r: (number(r, 'ev'), number(r, 'model_market_edge')), reverse=True)
    selected.extend(('$1 MICRO TEST', r) for r in micro[:5])
    for row in rows:
        if row.get('tier') == 'bankroll' and row.get('ledger_locked'):
            selected.append(('BANKROLL PROPS', row))
        elif row.get('tier') == 'locked_micro':
            selected.append(('PREVIOUSLY LOCKED MICRO - NOT ADDITIONAL PLAYS', row))
    taken = {(r.get('game_id'), r.get('player_id') or r.get('player_name'), r.get('stat')) for _, r in selected}
    sharp = [r for r in rows if number(r, 'sharp_ev') >= SHARP_EDGE_MIN_EV and r.get('line') is not None
             and (r.get('game_id'), r.get('player_id') or r.get('player_name'), r.get('stat')) not in taken]
    sharp.sort(key=lambda r: number(r, 'sharp_ev'), reverse=True)
    selected.extend((SHARP_HEADING, r) for r in sharp[:SHARP_EDGE_LIMIT])
    for title, positions, stat in PAPER_SECTIONS:
        actionable = {(r.get('game_id'), r.get('player_id') or r.get('player_name'), r.get('stat')) for _, r in selected}
        candidates = [r for r in rows if r.get('position') in positions and r.get('stat') == stat
                      and r.get('tier', 'paper') not in {'micro_projection', 'locked_micro', 'bankroll','cash_trial'}
                      and (r.get('game_id'), r.get('player_id') or r.get('player_name'), stat) not in actionable]
        candidates.sort(key=lambda r: (r.get('ev') is not None, number(r, 'ev'), number(r, 'projection', 0)), reverse=True)
        seen = set()
        for row in candidates:
            key = (row.get('game_id'), row.get('player_id') or row.get('player_name'), stat)
            if key in seen:
                continue
            if len(seen) >= paper_limit:
                break
            seen.add(key)
            selected.append((f'PAPER - {title}', row))
    return selected


def game_line(row):
    price = props._format_american(row.get('price'))
    if row.get('market') == 'spread':
        team = row.get('home_team_abbr') if row.get('side') == 'home' else row.get('away_team_abbr')
        pick = f"{team} {float(row['line']):+.1f}"
        forecast = f"home margin={float(row['predicted_home_margin']):+.1f}"
    else:
        pick = f"{str(row.get('side') or '').upper()} {float(row['line']):.1f}"
        forecast = f"total={float(row['predicted_total_points']):.1f}"
    details = [f"- {pick} {price}", forecast]
    if row.get('probability') is not None:
        details.append(f"P={float(row['probability']):.0%}")
    if row.get('ev') is not None:
        details.append(f"EV={float(row['ev']):+.1%}")
    return ' | '.join(details) + fanduel_links.row_link(row)


def section_parts(title, lines, limit=DESCRIPTION_LIMIT):
    """Keep row/link boundaries intact and repeat the section name on continuation."""
    heading = f'**{title}**'
    current = heading
    for line in lines:
        if len(heading) + len(line) + 1 > limit:
            raise ValueError('A matchup row exceeds the Discord card limit; refusing to truncate its link')
        if len(current) + len(line) + 1 > limit:
            yield current
            current = heading
        current += '\n' + line
    if current != heading:
        yield current


def embed_pages(title, intro, sections, footer):
    descriptions = []
    current = intro
    for heading, lines in sections:
        for part in section_parts(heading, lines):
            if len(current) + len(part) + 2 > DESCRIPTION_LIMIT and current != intro:
                descriptions.append(current)
                current = intro
            # Leave room for the game intro when an unusually large section is split.
            if len(current) + len(part) + 2 > DESCRIPTION_LIMIT:
                for smaller in section_parts(heading, part.splitlines()[1:], DESCRIPTION_LIMIT-len(intro)-2):
                    if current != intro:
                        descriptions.append(current)
                    current = intro + '\n\n' + smaller
            else:
                current += '\n\n' + part
    if current:
        descriptions.append(current)
    return [{'title': title if len(descriptions) == 1 else f'{title} ({i+1}/{len(descriptions)})',
             'description': body, 'footer': {'text': footer}, 'color': 0x237A57}
            for i, body in enumerate(descriptions)]


def build_bundle(day, schedule, game_rows, prop_rows, releases, *, now=None, paper_limit=10, quote_exclusions=None, cash_readiness=None):
    now = now or datetime.now(timezone.utc)
    by_game = defaultdict(lambda: defaultdict(list))
    upcoming = {str(g['game_id']): g for g in schedule
                if utc(g.get('start_ts_utc')) and utc(g['start_ts_utc']) > now}
    current_props = [r for r in preferred_rows(prop_rows) if str(r['game_id']) in upcoming]
    for heading, row in selected_props(current_props, paper_limit):
        by_game[str(row['game_id'])][heading].append(row)
    games_by_id = defaultdict(list)
    for row in preferred_rows(game_rows):
        if str(row['game_id']) in upcoming:
            games_by_id[str(row['game_id'])].append(row)
    cards = []
    footer = 'Release: ' + ' / '.join(sorted(set(releases))) + ' | Paper is research. Ledger rows are not confirmed wagers.'
    for gid, game in sorted(upcoming.items(), key=lambda item: (utc(item[1]['start_ts_utc']), item[0])):
        title = f"{game['away_team_abbr']} @ {game['home_team_abbr']} | NFL {day}"
        kickoff = int(utc(game['start_ts_utc']).timestamp())
        intro = f'Kickoff: <t:{kickoff}:f> (<t:{kickoff}:R>)\nFanDuel only. Quotes are from locked forecasts; verify the price at FanDuel.'
        sections = []
        if cash_readiness is not None:
            heading = 'OPERATIONAL ISSUE' if cash_readiness['status']=='operational_issue' else 'CASH TRIAL STATUS'
            sections.append((heading, ['- '+str(cash_readiness.get('reason') or cash_readiness['status'])]))
        cash = by_game[gid].get('$1 CASH TRIAL ELIGIBLE', [])
        if cash:
            lines = ['- $1 flat only. Confirm the current price; these are recommendations, not placed wagers.']
            for r in cash:
                expires = int(utc(r['expires_at']).timestamp())
                lines.append(fanduel_links.format_prop_row(r, action='Eligible $1')+
                    f" | Lock={props._format_american(r['price'])} | Expires <t:{expires}:R> | Ledger #{r['cash_ledger_id']}")
            sections.append(('$1 CASH TRIAL ELIGIBLE', lines))
        withheld = [r for r in (quote_exclusions or []) if r['game_id'] == gid]
        if withheld:
            sections.append(('QUOTE HEALTH', [f'- {len(withheld)} saved quotes withheld: stale or unverified timing/book. Refresh required; no current betting instruction for those rows.']))
        game_picks = games_by_id[gid]
        bankroll_games = [r for r in game_picks if r.get('tier') == 'bankroll' and r.get('ledger_locked')]
        bankroll_props = by_game[gid].get('BANKROLL PROPS', [])
        if bankroll_games:
            sections.append(('BANKROLL GAMES', [game_line(r) for r in bankroll_games]))
        if bankroll_props:
            sections.append(('BANKROLL PROPS', [fanduel_links.format_prop_row(r, action='Bankroll') for r in bankroll_props]))
        if not bankroll_games and not bankroll_props:
            sections.append(('BANKROLL', ['- No qualifying bankroll bets for this matchup.']))
        for heading, action in (('$1 MICRO TEST', 'BET $1'), ('PREVIOUSLY LOCKED MICRO - NOT ADDITIONAL PLAYS', 'Already recorded')):
            rows = by_game[gid].get(heading, [])
            if rows:
                lead = ['- $1 flat only; verify Min and Drift=OK. Daily cap remains 5 across all games.'] if action == 'BET $1' else ['- Historical locked quotes, not refreshed betting instructions.']
                sections.append((heading, lead + [fanduel_links.format_prop_row(r, action=action) for r in rows]))
        sharp_rows = by_game[gid].get(SHARP_HEADING, [])
        if sharp_rows:
            sections.append((SHARP_HEADING, ['- FanDuel priced off a sharp book at the same line. Research only: tracked by CLV, not a bet.']
                             + [fanduel_links.format_prop_row(r, action='Research only') + sharp_suffix(r) for r in sharp_rows]))
        paper_games = [r for r in game_picks if r.get('tier', 'paper') != 'bankroll' and r.get('ev') is not None]
        paper_games.sort(key=lambda r: number(r, 'ev'), reverse=True)
        sections.append(('PAPER GAME PICKS', [game_line(r) for r in paper_games] or ['- No current priced game picks at FanDuel.']))
        prop_count = sum(r.get('offer_id') is not None for r in current_props if str(r['game_id']) == gid)
        if not prop_count:
            sections.append(('PROP ODDS', ['- No fresh verified FanDuel prop offers in this card. Any projections below are not bettable.']))
        paper_sections = [(heading, rows) for heading, rows in by_game[gid].items() if heading.startswith('PAPER -')]
        if not paper_sections:
            sections.append(('PAPER PROPS', ['- No props from this matchup in the current slate-wide research shortlist.']))
        for heading, rows in paper_sections:
            lines = [fanduel_links.format_prop_row(r, action='Research only') for r in rows]
            parlay = fanduel_links.parlay_betslip_url([r.get('link') for r in rows], rows)
            if parlay:
                lines.append(f'- Research parlay: [FanDuel](<{parlay}>)')
            sections.append((heading, lines))
        # Matchup-level manifest is repeated across pages. Historical locked rows
        # without a current forecast ID are not advertised as refreshed decisions.
        manifest=[]
        displayed=[('game',r) for r in bankroll_games+paper_games]
        displayed += [('prop',r) for rows in by_game[gid].values() for r in rows]
        for kind,r in displayed:
            if r.get('forecast_id') is not None:
                manifest.append(dict(kind=kind,forecast_id=r['forecast_id'],book=r.get('book'),
                    side=r.get('side'),line=r.get('line'),price=r.get('price'),
                    model_version=r.get('model_version'),tier=r.get('tier'),
                    cash_ledger_id=r.get('cash_ledger_id'), cash_policy=r.get('cash_policy'),
                    probability=r.get('probability'), probability_source=r.get('probability_source'),
                    scoring_version=(r.get('scoring_replay') or {}).get('scoring_fingerprint')))
                manifest[-1]['cash_expires_at'] = r.get('expires_at')
                manifest[-1]['provider_link'] = fanduel_links.provider_link(r.get('link'))
                manifest[-1]['betslip_link'] = fanduel_links.betslip_for_row(r)
        for page, embed in enumerate(embed_pages(title, intro, sections, footer), 1):
            import re
            cash_ids = [r['cash_ledger_id'] for r in cash if
                        re.search(r'Ledger #'+str(r['cash_ledger_id'])+r'\b', embed['description'])]
            cards.append({'game_id': gid, 'page': page, 'forecast_manifest':manifest,
                'cash_ledger_ids':cash_ids,
                'payload': {'embeds': [embed], 'allowed_mentions': {'parse': []}}})
    return {'contract': CONTRACT, 'link_contract': fanduel_links.CONTRACT,
            'date': str(day), 'games': len(upcoming), 'cards': cards,
            'cash_readiness':cash_readiness,
            'quote_exclusions': quote_exclusions or [],
            'notice': None if upcoming else 'No upcoming NFL games remain on this date.'}


def preview_markdown(bundle):
    lines = [f"# NFL Matchup Cards - {bundle['date']}", '', f"{bundle['games']} games / {len(bundle['cards'])} messages", '']
    for card in bundle['cards']:
        embed = card['payload']['embeds'][0]
        lines += ['## ' + embed['title'], '', embed['description'], '', embed['footer']['text'], '']
    if bundle.get('notice'):
        lines.append(bundle['notice'])
    return '\n'.join(lines) + '\n'
