"""Read-only NFL slate monitoring; no selections, forecast changes or cash approval.

Runs after scheduled collection. A separate, timestamped ESPN observation can flag
late news without modifying the frozen model comparison or treating missing news as cleared.
"""
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json

import psycopg2
import requests

import automatic_price_checks as automatic
import model_research as model
from nfl_pipeline import pregame_checkpoints as checkpoints, run_pregame, sharp_watch
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.markets import normalize_name, normalize_team

ROOT = Path(__file__).resolve().parents[1]
MT = ZoneInfo('America/Denver')
NEWS = ROOT / 'data/nfl_slate_monitor/injuries_latest.json'
LOCK = 86731508


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def injury_observation(now, due, refresh):
    previous = read(NEWS)
    stamp = safe_time(previous.get('observed_at'))
    if not refresh or not due or (stamp and 0 <= (now-stamp).total_seconds() < 240):
        return previous
    observation = dict(observed_at=now.isoformat(), status='unknown', rows=[],
                       source='ESPN public injury feed', complete_official_inactives_verified=False)
    try:
        response = requests.get('https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries', timeout=8)
        observation['http_status'] = response.status_code
        if response.status_code == 200:
            payload = response.json()
            observation['payload_sha256'] = hashlib.sha256(response.content).hexdigest()
            entries = []
            for team in payload.get('injuries', []):
                for item in team.get('injuries', []):
                    athlete = item.get('athlete') or {}
                    abbr = normalize_team((athlete.get('team') or {}).get('abbreviation') or '')
                    name = athlete.get('displayName')
                    status = str(item.get('status') or '').strip().lower()
                    if abbr and name and status:
                        entries.append(dict(team=abbr, name=name, name_key=normalize_name(name), status=status,
                                            source_reported_at=item.get('date')))
            observation.update(status='observed' if entries else 'empty_unverified', rows=entries)
            NEWS.parent.mkdir(parents=True, exist_ok=True)
            (NEWS.parent / f'injuries_{now:%Y%m%dT%H%M%S}.json').write_text(
                json.dumps(dict(observation=observation, payload=payload)), encoding='utf-8')
    except (requests.RequestException, ValueError, AttributeError, TypeError) as exc:
        observation['error_type'] = type(exc).__name__
    NEWS.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(NEWS, observation)
    return observation


def news_for_game(game, forecasts, observation, now):
    stamp = safe_time(observation.get('observed_at'))
    current = bool(stamp and 0 <= (now-stamp).total_seconds() <= 600 and observation.get('status') == 'observed')
    teams = {game['home_team_abbr'], game['away_team_abbr']}
    entries = [r for r in observation.get('rows', []) if r['team'] in teams]
    by_player = {}
    for row in entries:
        by_player.setdefault((row['team'], row['name_key']), set()).add(row['status'])
    changes = []
    if current:
        for f in forecasts:
            forecast_at = safe_time(f.get('created_at_utc'))
            if not forecast_at or not stamp or stamp < forecast_at:
                continue
            name = f.get('offer_player_name_norm') or normalize_name(f['player_name'])
            statuses = by_player.get((f['team_abbr'], name), set())
            if len(statuses) != 1:
                continue
            status = next(iter(statuses))
            if status != str(f.get('injury_report_status') or '').lower() and status in {
                    'out', 'inactive', 'doubtful', 'injured reserve', 'suspension'}:
                changes.append(dict(forecast_id=f['id'], player=f['player_name'], stat=f['stat'],
                                    forecast_status=f.get('injury_report_status'), observed_status=status))
    return dict(status='recent_feed_observation' if current else 'unknown_or_stale', observed_at=stamp,
                listed_players=len(entries), status_counts=dict(Counter(r['status'] for r in entries)),
                adverse_changes_after_forecast=changes, complete_official_inactives_verified=False,
                note='Absence from the injury feed does not verify active participation. News is diagnostic; frozen selections are unchanged.')


def game_report(game, props, game_forecasts, quote_rows, pregame, checkpoint, observation, now):
    start = safe_time(game['start_ts_utc'])
    lead = (start-now).total_seconds()/60
    own = [p for p in props if p['game_id'] == game['game_id']]
    own_games = [p for p in game_forecasts if p['game_id'] == game['game_id']]
    quotes = [q for q in quote_rows if q['game_id'] == game['game_id']]
    counts, latest = Counter(), {}
    for row in quotes:
        p = row['payload']; at = safe_time(p.get('observed_at_utc'))
        family = p.get('stat') if p.get('stat') in {'spread','total'} else 'props'
        key = f"{p.get('book')}:{family}"
        if at and at < start:
            latest[key] = max(latest.get(key, at), at)
            q = p.get('quote') or {}
            if p.get('stat') == 'spread':
                q = dict(q, over=q.get('price'), under=q.get('price'))
            if at <= now and (now-at).total_seconds() <= 300 and not sharp_watch.quote_error(q, now):
                counts[key] += 1
    warnings = []
    if 0 < lead <= 100 and pregame.get('status') != 'completed':
        warnings.append('pregame_refresh_not_verified')
    news = news_for_game(game, own, observation, now)
    if 0 < lead <= 60:
        if news['status'] != 'recent_feed_observation':
            warnings.append('live_player_status_unknown')
        if news['adverse_changes_after_forecast']:
            warnings.append('adverse_news_after_forecast_requires_review')
        for family in ('props', 'spread', 'total'):
            if not all(counts[f'{book}:{family}'] for book in ('fanduel', 'pinnacle')):
                warnings.append(f'fresh_two_book_{family}_quotes_missing')
    if lead <= 0:
        warnings += [f"checkpoint_T{c['offset_minutes']}_unverified" for c in checkpoint['checkpoints']
                     if c['status'] != 'verified']
    return dict(game_id=game['game_id'], away=game['away_team_abbr'], home=game['home_team_abbr'], kickoff=start,
                phase='closed' if lead <= 0 else 'waiting' if lead > 120 else 'preparing' if lead > 60 else 'scanning',
                refresh_at=start-timedelta(minutes=120), scan_at=start-timedelta(minutes=60),
                selection_opens=start-timedelta(minutes=45), selection_closes=start-timedelta(minutes=15),
                pregame_run=pregame, current_prop_forecasts=len(own), current_game_forecasts=len(own_games),
                latest_prop_forecast=max((p['created_at_utc'] for p in own), default=None),
                latest_game_forecast=max((p['created_at_utc'] for p in own_games), default=None),
                news=news, fresh_quote_counts=dict(counts), latest_quote_observations=latest,
                checkpoints=checkpoint['checkpoints'], warnings=warnings)


def run(day, *, refresh_news=False):
    now = datetime.now(timezone.utc)
    with psycopg2.connect(model.PG_DSN) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout='20s'")
            cur.execute('SELECT pg_try_advisory_lock(%s)', (LOCK,))
            if not cur.fetchone()[0]:
                return dict(status='busy')
        doc, observer = model.active(), automatic.active()
        games = model.rows(conn, 'SELECT * FROM raw.nfl_games WHERE game_date_et=%s ORDER BY start_ts_utc,game_id', (day,))
        ids = [g['game_id'] for g in games]
        props = model.rows(conn, """SELECT p.id,p.game_id,p.player_name,p.team_abbr,p.stat,p.offer_player_name_norm,
            p.created_at_utc,p.injury_report_status FROM bets.nfl_player_prop_predictions p
            JOIN raw.nfl_games g USING(game_id) WHERE p.game_id=ANY(%s) AND p.is_current
            AND p.created_at_utc<g.start_ts_utc AND p.created_at_utc<=%s""", (ids, now))
        forecasts = model.rows(conn, """SELECT p.game_id,p.created_at_utc FROM bets.nfl_game_predictions p
            JOIN raw.nfl_games g USING(game_id) WHERE p.game_id=ANY(%s) AND p.is_current
            AND p.created_at_utc<g.start_ts_utc AND p.created_at_utc<=%s""", (ids, now))
        quotes = model.rows(conn, """SELECT game_id,payload FROM bets.nfl_sharp_quote_observations
            WHERE game_id=ANY(%s) AND observed_at_utc>=%s AND observed_at_utc<=%s""", (ids, now-timedelta(minutes=15), now))
        pending = automatic.pending(conn, observer, 'nfl', now)
        comparisons = model.rows(conn, """SELECT id,payload FROM bets.model_research_decisions
            WHERE sport='nfl' AND experiment_id=%s AND experiment_sha256=%s""", (model.NAME, doc['sha256']))
        attempts = model.rows(conn, """SELECT a.alert_key,a.attempted_at,o.payload FROM bets.automatic_paper_attempts a
            LEFT JOIN bets.automatic_paper_observations o ON o.attempt_id=a.id
            WHERE a.sport='nfl' AND a.observer_id=%s AND a.observer_sha256=%s""", (automatic.NAME, observer['sha256']))
        due = any(0 < (g['start_ts_utc']-now).total_seconds() <= 7200 for g in games)
        observation = injury_observation(now, due, refresh_news)
        pregame = run_pregame.load_state()
        checks = checkpoints.plan(games, checkpoints.load_state(), now)['observations']
        check_by_id = {r['game_id']: r for r in checks}
        rows = [game_report(g, props, forecasts, quotes, pregame['games'].get(run_pregame.key(g), {}),
                           check_by_id[g['game_id']], observation, now) for g in games]
    live_cycle = read(ROOT/f'reports/nfl_live_cycle_{day}.json')
    result = model.clean(dict(built_at=now, day=str(day), games=rows,
        frozen_model_sha256=doc['sha256'], frozen_observer_sha256=observer['sha256'],
        price_checks=dict(attempts=len(attempts), statuses=dict(Counter(
            (r['payload'] or {}).get('status', 'unknown') for r in attempts)),
            pending=len(pending), expired=sum((now-safe_time(a['sent_at'])).total_seconds()>300 for a in pending),
            account_availability_verified=False),
        paired=dict(price_only=len(comparisons),model_selected=sum(bool(r['payload']['decision']['selected']) for r in comparisons),
                    requirements=model.CONFIG['nfl_review']),
        economics=dict(monthly_cost_budget=30,reported_bankroll=50,comparison_stake=1,
            illustrative_roi=.03,example_100_bets_gross=3,example_100_bets_net_after_monthly_cost=-27,
            monthly_bets_to_cover_cost_at_assumed_roi=1000,actual_net_profit=None,
            note='Conditional arithmetic, not an established edge; actual profit requires wager and cost receipts.'),
        latest_live_cycle=live_cycle,
        latest_price_gap_review=read(ROOT/'reports/nfl_sharp_alerts_latest.json'),
        betting_authorized=False,selection_rules_changed=False))
    folder = ROOT / 'data/nfl_slate_monitor'
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder/f'readiness_{now:%Y%m%dT%H%M%S}.json', result)
    atomic_json(ROOT/'reports/nfl_slate_readiness_latest.json', result)
    lines = ['# NFL slate readiness', '', f'Observed {now.astimezone(MT):%b %d %I:%M %p %Z}. Paper collection; cash remains paused.', '',
             '| Game | Kickoff MT | Phase | Pregame refresh | Prop / game rows | Status feed | Warnings |',
             '|---|---|---|---|---:|---|---|']
    for r in rows:
        forecast_counts = ('closed' if r['phase']=='closed' else
                           f"{r['current_prop_forecasts']} / {r['current_game_forecasts']}")
        lines.append(f"| {r['away']} @ {r['home']} | {r['kickoff'].astimezone(MT):%I:%M %p} | {r['phase']} | "
                     f"{r['pregame_run'].get('status','pending')} | {forecast_counts} | "
                     f"{r['news']['status']} | {', '.join(r['warnings']) or 'none currently due'} |")
    p = result['price_checks']
    lines += ['', f"Automatic NFL attempts: {p['attempts']}; pending: {p['pending']}; expired: {p['expired']}. "
              'Public observations do not establish account availability.',
              f"Frozen paired offers: {result['paired']['price_only']}; model selected: {result['paired']['model_selected']}.",
              'Review minimums: 21 days, 20 games, 100 checked exact closes, coverage and positive uncertainty bounds. No automatic approval.',
              '', '## Checkpoint evidence']
    for r in rows:
        lines.append(f"- {r['away']} @ {r['home']}: " + ', '.join(f"T-{c['offset_minutes']} {c['status']}" for c in r['checkpoints']))
    lines += ['', '## Completed-window audit', '',
              f"Audit generated: {live_cycle.get('built_at', 'unknown')}. This is separate from current scheduler health."]
    for game in live_cycle.get('games', []):
        if game.get('valid_close_coverage') is not None:
            lines.append(f"- {game['game_id']}: {game['valid_close_coverage']:.1%} valid exact closes; "
                         f"{game['unknown_closes']} unknown. Audit issues: {', '.join(game['blockers']) or 'none'}.")
    lines += ['', 'Injury observations preserve observation time separately from forecasts and source-reported dates where supplied. '
              'A recent request does not prove a recent provider news update. Empty/missing feeds never mean active. '
              'This feed does not certify a complete official inactive list. New adverse statuses are diagnostic flags; frozen decisions are unchanged.',
              'Fresh quote counts are individual valid records. Exact opposing prices, identity and line matching remain the frozen selector\'s responsibility.',
              '', 'At an assumed true 3% return, 100 one-dollar bets yield $3 gross and -$27 after a $30 monthly cost. '
              'Covering $30 requires 1,000 such bets. Actual net profit remains unknown without complete receipts.']
    (ROOT/'reports/nfl_slate_readiness_latest.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return dict(status='updated',games=len(rows),warnings=sum(len(r['warnings']) for r in rows),
                completed_audit_issues=live_cycle.get('blockers', []),
                checks=p,report='reports/nfl_slate_readiness_latest.md',betting_authorized=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--date', type=date.fromisoformat)
    ap.add_argument('--refresh-news', action='store_true')
    args = ap.parse_args()
    day = args.date or datetime.now(timezone.utc).astimezone(ZoneInfo('America/New_York')).date()
    print(json.dumps(run(day, refresh_news=args.refresh_news)))
