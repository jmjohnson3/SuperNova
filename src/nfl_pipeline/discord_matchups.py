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
STARTER_POSITION_LIMITS = (('QB', 1), ('RB', 1), ('WR', 2), ('TE', 1))
STARTER_ROLE_ORDER = {'QB1': 0, 'RB1': 1, 'WR1': 2, 'WR2': 3, 'TE1': 4}
STARTER_STAT_ORDER = (
    ('passing_yards', 'Pass Yds'),
    ('passing_tds', 'Pass TD'),
    ('rushing_yards', 'Rush Yds'),
    ('receiving_yards', 'Rec Yds'),
    ('receptions', 'Rec'),
    ('rushing_tds', 'Rush TD'),
    ('receiving_tds', 'Rec TD'),
)
INACTIVE_ROSTER_STATUSES = {'INA', 'IR', 'RES', 'PUP', 'NFI', 'SUS'}
INACTIVE_INJURY_STATUSES = {'OUT', 'INACTIVE', 'INJURED RESERVE'}


def number(row, field, missing=-math.inf):
    value = row.get(field)
    return float(value) if value is not None and math.isfinite(float(value)) else missing


def independent_projection(row):
    """Return the player forecast made independently of the offered line."""
    point = ((row.get('forecast_outputs') or {}).get('point_forecast') or {}).get('value')
    for value in (point, row.get('projection'), row.get('baseline_projection')):
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def current_fanduel_line(row):
    marker = row.get('starter_current_line')
    if marker is not None:
        return bool(marker)
    return (str(row.get('book') or '').lower() == 'fanduel'
            and row.get('offer_id') is not None
            and row.get('line') is not None
            and row.get('side') in {'over', 'under'}
            and row.get('price') is not None)


def starting_skill_players(rows):
    """Select each team's active QB1, RB1, WR1/WR2 and TE1 from depth context."""
    players = {}
    for row in preferred_rows(rows):
        if row.get('previously_locked'):
            continue
        position = str(row.get('position') or '').upper()
        if position not in dict(STARTER_POSITION_LIMITS):
            continue
        # Special-teams depth entries can retain an offensive roster position.
        # Requiring the matching depth abbreviation avoids calling a KR/PR a WR1.
        if str(row.get('depth_pos_abb') or '').upper() != position:
            continue
        rank = number(row, 'depth_pos_rank')
        if not math.isfinite(rank):
            continue
        if str(row.get('roster_status') or '').upper() in INACTIVE_ROSTER_STATUSES:
            continue
        if str(row.get('injury_report_status') or '').upper() in INACTIVE_INJURY_STATUSES:
            continue
        game_id = str(row.get('game_id') or '')
        team = str(row.get('team_abbr') or '').upper()
        player_key = row.get('player_id') or row.get('player_name')
        if not game_id or not team or not player_key:
            continue
        key = (game_id, team, player_key)
        player = players.setdefault(key, {
            'game_id': game_id,
            'team_abbr': team,
            'player_id': row.get('player_id'),
            'player_name': row.get('player_name') or str(player_key),
            'position': position,
            'depth_rank': rank,
            'depth_slot': number(row, 'depth_pos_slot', math.inf),
            'stat_rows': {},
        })
        stat = row.get('stat')
        if stat not in dict(STARTER_STAT_ORDER):
            continue
        prior = player['stat_rows'].get(stat)
        if prior is None or current_fanduel_line(row) > current_fanduel_line(prior):
            player['stat_rows'][stat] = row

    grouped = defaultdict(list)
    for player in players.values():
        if player['stat_rows']:
            grouped[(player['game_id'], player['team_abbr'], player['position'])].append(player)

    selected = []
    for (game_id, team, position), candidates in grouped.items():
        limit = dict(STARTER_POSITION_LIMITS)[position]
        candidates.sort(key=lambda p: (p['depth_rank'], p['depth_slot'], p['player_name']))
        for index, player in enumerate(candidates[:limit], 1):
            selected.append(dict(player, role=f'{position}{index}'))
    return selected


SHARP_POINTER = ('- No model bets: the model forecasts are research. Bets come from the sharp-gap scanner, which posts '
                 '**SHARP EDGE** alerts in this channel 15-45 minutes before kickoff with the book, price and link.')


def starter_projection_line(player):
    stats = []
    for stat, label in STARTER_STAT_ORDER:
        row = player['stat_rows'].get(stat)
        value = independent_projection(row or {})
        if value is None:
            continue
        precision = 1 if stat.endswith('_yards') or stat == 'receptions' else 2
        stats.append(f'{label} {value:.{precision}f}')
    details = ' | '.join(stats) if stats else 'No current stat forecast'
    has_line = any(current_fanduel_line(row) for row in player['stat_rows'].values())
    availability = 'FanDuel line available' if has_line else 'Projection only - no current line'
    return f"- {player['team_abbr']} {player['role']} {player['player_name']}: {details} | {availability}"


def selected_props(rows, paper_limit=10):
    """Select executable rows once across the slate; forecasts render elsewhere."""
    rows = preferred_rows(rows)
    selected = []
    selected.extend(('$1 CASH TRIAL ELIGIBLE', r) for r in rows
                    if r.get('tier') == 'cash_trial' and r.get('cash_ledger_id') and r.get('ledger_locked'))
    micro = [r for r in rows if r.get('tier') == 'micro_projection' and r.get('ledger_locked')
             and r.get('line') is not None and r.get('side') in {'over', 'under'}]
    micro.sort(key=lambda r: (number(r, 'ev'), number(r, 'model_market_edge')), reverse=True)
    selected.extend(('$1 MICRO TEST', r) for r in micro[:3])
    for row in rows:
        if row.get('tier') == 'bankroll' and row.get('ledger_locked'):
            selected.append(('BANKROLL PROPS', row))
        elif row.get('tier') == 'locked_micro':
            selected.append(('PREVIOUSLY LOCKED MICRO - NOT ADDITIONAL PLAYS', row))
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


def build_bundle(day, schedule, game_rows, prop_rows, releases, *, now=None, paper_limit=10,
                 quote_exclusions=None, cash_readiness=None, starter_prop_rows=None):
    now = now or datetime.now(timezone.utc)
    by_game = defaultdict(lambda: defaultdict(list))
    upcoming = {str(g['game_id']): g for g in schedule
                if utc(g.get('start_ts_utc')) and utc(g['start_ts_utc']) > now}
    current_props = [r for r in preferred_rows(prop_rows) if str(r['game_id']) in upcoming]
    for heading, row in selected_props(current_props, paper_limit):
        by_game[str(row['game_id'])][heading].append(row)
    starter_source = current_props if starter_prop_rows is None else [
        r for r in preferred_rows(starter_prop_rows) if str(r['game_id']) in upcoming]
    starters_by_game = defaultdict(list)
    for player in starting_skill_players(starter_source):
        starters_by_game[player['game_id']].append(player)
    games_by_id = defaultdict(list)
    for row in preferred_rows(game_rows):
        if str(row['game_id']) in upcoming:
            games_by_id[str(row['game_id'])].append(row)
    cards = []
    footer = ('Release: ' + ' / '.join(sorted(set(releases)))
              + ' | Only labeled bankroll/micro rows are betting instructions.')
    for gid, game in sorted(upcoming.items(), key=lambda item: (utc(item[1]['start_ts_utc']), item[0])):
        title = f"{game['away_team_abbr']} @ {game['home_team_abbr']} | NFL {day}"
        kickoff = int(utc(game['start_ts_utc']).timestamp())
        intro = (f'Kickoff: <t:{kickoff}:f> (<t:{kickoff}:R>)\nModel forecasts at FanDuel only (research). '
                 'Bets to place come from the SHARP EDGE alerts in this channel.')
        sections = []
        cash = by_game[gid].get('$1 CASH TRIAL ELIGIBLE', [])
        bankroll_games = [r for r in games_by_id[gid] if r.get('tier') == 'bankroll' and r.get('ledger_locked')]
        bankroll_props = by_game[gid].get('BANKROLL PROPS', [])
        # The model trial is research (docs/nfl_cash_readiness.md): its readiness status and empty bankroll /
        # actionable sections only matter on a card that carries one of its betting rows. Otherwise the card
        # points at the sharp-gap alerts, which are where bets come from.
        model_rows = bool(cash or bankroll_games or bankroll_props or by_game[gid].get('$1 MICRO TEST'))
        if cash_readiness is not None and model_rows:
            heading = 'OPERATIONAL ISSUE' if cash_readiness['status']=='operational_issue' else 'CASH TRIAL STATUS'
            sections.append((heading, ['- '+str(cash_readiness.get('reason') or cash_readiness['status'])]))
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
        if bankroll_games:
            sections.append(('BANKROLL GAMES', [game_line(r) for r in bankroll_games]))
        if bankroll_props:
            sections.append(('BANKROLL PROPS', [fanduel_links.format_prop_row(r, action='Bankroll') for r in bankroll_props]))
        if model_rows and not bankroll_games and not bankroll_props:
            sections.append(('BANKROLL', ['- No qualifying bankroll bets for this matchup.']))
        actionable_prop_count = len(cash) + len(bankroll_props)
        for heading, action in (('$1 MICRO TEST', 'BET $1'), ('PREVIOUSLY LOCKED MICRO - NOT ADDITIONAL PLAYS', 'Already recorded')):
            rows = by_game[gid].get(heading, [])
            if rows:
                if action == 'BET $1':
                    actionable_prop_count += len(rows)
                lead = ['- $1 flat only; verify Min and Drift=OK. Maximum 3 across all games; there is no quota.'] if action == 'BET $1' else ['- Historical locked quotes, not refreshed betting instructions.']
                sections.append((heading, lead + [fanduel_links.format_prop_row(r, action=action) for r in rows]))
        if not model_rows:
            sections.append(('BETS', [SHARP_POINTER]))
        elif not actionable_prop_count:
            sections.append(('ACTIONABLE PROPS', ['- No current prop meets the FanDuel residual, price, freshness, model, and context requirements.']))
        prop_count = sum(r.get('offer_id') is not None for r in current_props if str(r['game_id']) == gid)
        if not prop_count:
            sections.append(('PROP ODDS', ['- No fresh verified FanDuel prop offers in this card. Any projections below are not bettable.']))
        team_order = {str(game.get('away_team_abbr')): 0, str(game.get('home_team_abbr')): 1}
        starters = sorted(starters_by_game.get(gid, []), key=lambda p: (
            team_order.get(p['team_abbr'], 2), STARTER_ROLE_ORDER[p['role']], p['player_name']))
        starter_lines = [starter_projection_line(player) for player in starters]
        sections.append(('STARTING SKILL PLAYER PROJECTIONS',
                         ['- Forecasts only, not bets.']
                         + (starter_lines or ['- No verified starter depth projections are available.'])))
        # Matchup-level manifest is repeated across pages. Historical locked rows
        # without a current forecast ID are not advertised as refreshed decisions.
        manifest=[]
        displayed=[('game',r) for r in bankroll_games]
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
