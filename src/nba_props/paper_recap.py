"""Audit delivered NBA paper alerts against final scores and player participation.

This diagnostic does not change the frozen experiment, select wagers, or infer
manual price checks. Results stay separated by book; repeated underlying picks
are counted separately from the number of alerts.

python -m nba_props.paper_recap --date 2026-10-28 --refresh-results
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
from zoneinfo import ZoneInfo

import psycopg2

from model_research import ROOT, NBA_STATS, clean, nba_team, number, rows, safe_time
from model_research_results import fetch, final_game
from nba_props.ingest import conflicted_games
from nba_props.model import match_name
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from sharp_pipeline import nba_experiment, outcomes, report

ET = ZoneInfo('America/New_York')


def verify_result(offer, games, players, conflicts):
    """Missing, ambiguous, partial and nonparticipant records remain pending."""
    pending = lambda reason: dict(result=None, actual=None, reason=reason)
    start = safe_time(offer.get('commence_time_utc'))
    decided, sent = safe_time(offer.get('decided_at')), safe_time(offer.get('sent_at'))
    if not start or not decided or not sent or not decided <= sent < start:
        return pending('not_delivered_pregame')
    home, away = nba_team(offer['home_team']), nba_team(offer['away_team'])
    matched = [g for g in games if g['status'] == 'final' and g['home_team'] == home
               and g['away_team'] == away
               and abs((safe_time(g['tip_utc']) - start).total_seconds()) <= 900]
    if len(matched) != 1:
        return pending('official_final_missing_or_ambiguous')
    game = matched[0]
    if sent >= safe_time(game['tip_utc']):
        return pending('delivery_after_official_start')
    if game['game_id'] in conflicts:
        return pending('unresolved_ingestion_conflict')
    logs = [p for p in players if p['game_id'] == game['game_id'] and
            (p['team'], p['opponent']) in {(home, away), (away, home)}]
    for team, score in ((home, game['home_score']), (away, game['away_score'])):
        team_rows = [p for p in logs if p['team'] == team]
        points = [number(p.get('pts')) for p in team_rows]
        if not points or any(p is None for p in points) or abs(sum(points) - score) > .01:
            return dict(pending('player_logs_do_not_match_final_scores'), game_id=game['game_id'])
    target = [p for p in logs if match_name(p['player_name']) == match_name(offer['player'])]
    if len(target) != 1 or (number(target[0].get('minutes')) or 0) <= 0:
        return pending('participation_missing_or_ambiguous')
    stat = NBA_STATS.get(offer['market'])
    actual = number(target[0].get(stat)) if stat else None
    if actual is None or actual < 0 or actual != int(actual):
        return pending('stat_missing_or_invalid')
    return dict(result=outcomes.grade(offer, actual, game['home_score'], game['away_score']),
                actual=actual, game_id=game['game_id'], reason='verified_final_and_participation')


def summarize(alerts):
    """Book profits are hypothetical $1 stakes; never manufacture a cross-book strategy."""
    books = {}
    for book in sorted({r['book'] for r in alerts}):
        part = [r for r in alerts if r['book'] == book]
        settled = [r for r in part if r['result'] in {'win', 'loss', 'push'}]
        closes = [r['close_ev'] for r in part if r.get('close_ev') is not None]
        checked = [r['checked_price_ev'] for r in part if r.get('checked_price_ev') is not None]
        units = sum(outcomes.profit(r['result'], r['price']) for r in settled)
        books[book] = dict(alerts=len(part), games=len({r['event_id'] for r in part}),
            wins=sum(r['result'] == 'win' for r in part), losses=sum(r['result'] == 'loss' for r in part),
            pushes=sum(r['result'] == 'push' for r in part), pending=len(part)-len(settled),
            paper_units=units, paper_roi=units/len(settled) if settled else None,
            exact_closes=len(closes), mean_close_ev=sum(closes)/len(closes) if closes else None,
            timely_checks=sum(bool(r.get('timely_check')) for r in part),
            mean_checked_price_ev=sum(checked)/len(checked) if checked else None)
    groups = {}
    for r in alerts:
        key = (r['event_id'], r['market'], r['player_norm'], r['side'], float(r['line']))
        groups.setdefault(key, []).append(r)
    unique_results = [g[0]['result'] if len({r['result'] for r in g}) == 1 else None for g in groups.values()]
    return dict(by_book=books, alerts=len(alerts), unique_picks=len(groups),
                repeated_alerts=len(alerts)-len(groups), games=len({r['event_id'] for r in alerts}),
                unique_record={k: unique_results.count(k) for k in ('win', 'loss', 'push', None)})


def load_alerts(conn, day):
    start = datetime.combine(day, datetime.min.time(), ET)
    return rows(conn, """SELECT d.*, a.alert_id, a.delivery_status, a.sent_at,
        a.discord_message_id, c.checked_at, c.recorded_at, c.status AS check_status,
        c.displayed_line, c.displayed_price, c.evidence_source
        FROM bets.nba_paper_price_alerts a JOIN bets.sharp_multi_decisions d USING(decision_id)
        LEFT JOIN bets.nba_paper_price_checks c USING(alert_id)
        WHERE a.delivery_status='sent' AND d.commence_time_utc >= %s AND d.commence_time_utc < %s
        ORDER BY a.sent_at, a.alert_id""", (start, start+timedelta(days=1)))


def load_closes(conn, alerts):
    ids = sorted({r['event_id'] for r in alerts})
    found = rows(conn, """SELECT event_id,observed_at_utc,payload FROM bets.sharp_multi_quotes
        WHERE sport='nba' AND event_id=ANY(%s) AND observed_at_utc<commence_time_utc
        AND observed_at_utc>=commence_time_utc-interval '22 minutes'
        ORDER BY observed_at_utc""", (ids,))
    return {('nba', r['event_id'], report.payload_kind(r['payload'])):
            (r['observed_at_utc'], r['payload']) for r in found}


def scoreboard(day, refresh=False):
    path = ROOT / 'data/nba_props/paper_recaps' / f'{day}_official.json'
    if refresh:
        raw = fetch(day)
        payload = dict(day=str(day), retrieved_at=datetime.now(timezone.utc).isoformat(),
                       source='stats.nba.com/scoreboardv3', games=raw)
        atomic_json(path, payload)
    elif path.exists():
        payload = json.loads(path.read_text(encoding='utf-8'))
    else:
        payload = dict(day=str(day), retrieved_at=None, games=[])
    if payload['day'] != str(day):
        raise ValueError('Scoreboard date mismatch')
    games = [g for raw in payload['games'] if (g := final_game(raw)) and g['game_date'] == str(day)]
    return games, dict(path=str(path) if path.exists() else None, retrieved_at=payload['retrieved_at'])


def boxscore_players(payload, game):
    """Accept only a complete box score matching both official final team totals."""
    box = payload.get('boxScoreTraditional') or {}
    if str(box.get('gameId')) != game['game_id']:
        return []
    players, ids = [], set()
    for side, other in (('home', 'away'), ('away', 'home')):
        team = box.get(side+'Team') or {}
        if (team.get('teamTricode') != game[side+'_team'] or
                number((team.get('statistics') or {}).get('points')) != game[side+'_score']):
            return []
        total = 0
        for p in team.get('players') or []:
            identity = p.get('personId')
            if not identity or identity in ids:
                return []
            ids.add(identity)
            stats = p.get('statistics') or {}
            points = number(stats.get('points'))
            if points is None or points < 0 or points != int(points):
                return []
            total += points
            try:
                mm, ss = str(stats.get('minutes')).split(':')
                minutes = int(mm)+int(ss)/60 if int(mm) >= 0 and 0 <= int(ss) < 60 else None
            except (TypeError, ValueError):
                minutes = None
            players.append(dict(game_id=game['game_id'], player_id=identity,
                player_name=' '.join(filter(None, (p.get('firstName'), p.get('familyName')))),
                team=game[side+'_team'], opponent=game[other+'_team'], minutes=minutes, pts=points,
                reb=number(stats.get('reboundsTotal')), ast=number(stats.get('assists')),
                fg3m=number(stats.get('threePointersMade'))))
        if total != game[side+'_score']:
            return []
    return players


def boxscore(game, refresh=False):
    path = ROOT / 'data/nba_props/paper_recaps' / f"{game['game_id']}_boxscore.json"
    if refresh:
        from nba_api.stats.endpoints import boxscoretraditionalv3
        raw = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game['game_id'], timeout=30).get_dict()
        atomic_json(path, dict(retrieved_at=datetime.now(timezone.utc).isoformat(),
                              source='stats.nba.com/boxscoretraditionalv3', payload=raw))
    if not path.exists():
        return [], None
    saved = json.loads(path.read_text(encoding='utf-8'))
    return boxscore_players(saved['payload'], game), dict(path=str(path), retrieved_at=saved['retrieved_at'])


def markdown(doc):
    record = doc['summary']['unique_record']
    lines = [f"# NBA delivered paper alerts — {doc['day']}", '',
        f"{doc['summary']['alerts']} delivered alerts; {doc['summary']['unique_picks']} unique picks in "
        f"{doc['summary']['games']} games. Cross-book repeats: {doc['summary']['repeated_alerts']}.",
        f"Unique-pick record: {record['win']}-{record['loss']}-{record['push']}, "
        f"{record.get('pending', 0)} pending.", '',
        'Hypothetical $1 stakes at alert prices. No actual wagers or live availability are inferred.', '',
        '| Book | W-L-P | Pending | Paper profit | Exact closes | Mean closing EV | Timely checks |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for book, s in doc['summary']['by_book'].items():
        close = f"{s['mean_close_ev']:+.2%}" if s['mean_close_ev'] is not None else 'unknown'
        lines.append(f"| {book} | {s['wins']}-{s['losses']}-{s['pushes']} | {s['pending']} | "
                     f"${s['paper_units']:+.2f} | {s['exact_closes']}/{s['alerts']} | {close} | {s['timely_checks']} |")
    lines += ['', '| Check ID | Player | Offer | Actual | Result |', '|---|---|---|---:|---|']
    for r in doc['alerts']:
        lines.append(f"| NBA #{r['alert_id']} | {r['player']} | {r['side']} {float(r['line']):g} "
                     f"{r['market']}, {r['book']} {r['price']:+d} | "
                     f"{r['actual'] if r['actual'] is not None else 'unknown'} | {r['result'] or r['reason']} |")
    resolved = [r for r in doc['alerts'] if r.get('boxscore_evidence') and r['result'] is not None]
    if resolved:
        lines += ['', f"{len(resolved)} alerts required a complete official box score because season game logs "
                  'did not reconcile with the final score. The original logs were preserved; source snapshots are in the JSON.']
    lines += ['', 'Pinnacle closing probabilities are a comparison benchmark, not known true probabilities.',
              'Cross-book duplicates and bets in the same game are correlated. This recap does not pass review gates.',
              'This recap covers price-check alerts. Model-assisted selections have their own prospective report.',
              'Monthly costs belong in the economic and actual-wager ledgers, not this one-night paper result.',
              f"Official score snapshot: {doc['official_results']['path']}."]
    return '\n'.join(lines)+'\n'


def run(day, refresh=False):
    nba_experiment.active()  # A recap must not silently conceal a broken frozen parent.
    with psycopg2.connect(PG_DSN, connect_timeout=10) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout='30s'")
        alerts = load_alerts(conn, day)
        games, evidence = scoreboard(day, refresh and bool(alerts))
        closes = load_closes(conn, alerts)
        players = rows(conn, 'SELECT * FROM nbaprops.player_games WHERE game_date=%s', (day,))
        conflicts = set(conflicted_games(conn))
        boxes = {}
        for r in alerts:
            r.update(verify_result(r, games, players, conflicts))
            r['result_source'] = 'stats.nba.com/playergamelogs' if r['result'] else None
            if r['reason'] == 'player_logs_do_not_match_final_scores':
                game = next(g for g in games if g['game_id'] == r['game_id'])
                if r['game_id'] not in boxes:
                    boxes[r['game_id']] = boxscore(game, refresh)
                corrected, source = boxes[r['game_id']]
                r['game_log_issue'] = r['reason']
                if corrected:
                    original = {p['player_id']: p for p in players if p['game_id'] == game['game_id']}
                    r['boxscore_corrections'] = [dict(player_id=p['player_id'], player_name=p['player_name'],
                        fields={k: dict(season_log=number(original.get(p['player_id'], {}).get(k)), boxscore=p[k])
                                for k in ('pts', 'reb', 'ast', 'fg3m')
                                if number(original.get(p['player_id'], {}).get(k)) != p[k]})
                        for p in corrected if any(number(original.get(p['player_id'], {}).get(k)) != p[k]
                                                  for k in ('pts', 'reb', 'ast', 'fg3m'))]
                    r.update(verify_result(r, games, corrected, conflicts))
                    r['boxscore_evidence'] = source
                    r['result_source'] = 'stats.nba.com/boxscoretraditionalv3' if r['result'] else None
            close = report.close_evidence(r, closes)
            r.update(close_method=close.get('method'),
                close_probability=close.get('probability') if close.get('method') == 'exact' else None,
                close_ev=close.get('ev') if close.get('method') == 'exact' else None,
                closing_evidence=close)
            r.update(nba_experiment.check_evidence(r))
    summary = summarize(alerts)
    summary['unique_record']['pending'] = summary['unique_record'].pop(None)
    doc = clean(dict(day=str(day), built_at=datetime.now(timezone.utc), summary=summary,
                     alerts=alerts, official_results=evidence, betting_authorized=False))
    target = ROOT / 'reports' / f'nba_paper_recap_{day}'
    atomic_json(target.with_suffix('.json'), doc)
    target.with_suffix('.md').write_text(markdown(doc), encoding='utf-8')
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--date', type=date.fromisoformat,
                    default=datetime.now(ET).date()-timedelta(days=1))
    ap.add_argument('--refresh-results', action='store_true', help='Refresh official NBA scores; no Odds API calls')
    args = ap.parse_args()
    result = run(args.date, args.refresh_results)
    print(json.dumps(result['summary'], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
