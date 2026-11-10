"""Render frozen sharp evidence without labeling paper reservations as actual money.

The frozen scanner, selection and statistical calculations remain in their pinned
modules. This reporting layer exposes monetary totals only in the explicit legacy
confirmed-wager section. Receipt P&L is owned by wager_reconciliation.
"""
import json
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.modeling import sharp_alert_report as frozen


def attach_indexed(frame, quotes):
    """Index only the equality predicates already used by the frozen close reader.

    All timing, price, integrity and exact-line decisions remain in that reader.
    Preserve quote order (including ties) and original row order. Previously every
    decision scanned every quote in its game; large slates exceeded the 120s budget.
    """
    if frame.empty:
        return frame
    def key(row, player):
        return (str(row.get('game_id')), row.get('event_id'), row.get('stat'), row.get(player))
    indexed = defaultdict(list)
    for quote in quotes:
        indexed[key(quote, 'player_name_norm')].append(quote)
    groups = defaultdict(list)
    for i, row in enumerate(frame.to_dict('records')):
        groups[key(row, 'player_name_norm')].append((i, row))
    result = [None] * len(frame)
    for identity, group in groups.items():
        scored = frozen.attach_exact_evidence(pd.DataFrame([r for _, r in group]), indexed[identity])
        for (i, _), row in zip(group, scored.to_dict('records')):
            result[i] = row
    return pd.DataFrame(result)


def load_indexed(conn):
    """Same source rows and result join as frozen.load; indexed close attachment."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='90s'")
        cur.execute("SELECT to_regclass('bets.nfl_sharp_alerts')")
        if cur.fetchone()['to_regclass'] is None:
            return pd.DataFrame()
        cur.execute("""
            SELECT a.*, g.season, g.week, g.status, g.home_score, g.away_score,
                   gl.actual, gl.participation_verified, gl.player_team
            FROM bets.nfl_sharp_alerts a
            JOIN raw.nfl_games g ON g.game_id = a.game_id
            LEFT JOIN LATERAL (
              SELECT CASE a.stat WHEN 'receiving_yards' THEN l.receiving_yards WHEN 'rushing_yards' THEN l.rushing_yards
                       WHEN 'passing_yards' THEN l.passing_yards WHEN 'receptions' THEN l.receptions
                       WHEN 'passing_tds' THEN l.passing_tds END::float AS actual,
                     (COALESCE(l.offense_snaps,0)>0
                      OR COALESCE(l.pass_attempts,0)+COALESCE(l.carries,0)+COALESCE(l.targets,0)>0
                      OR (l.raw_json #> '{receiving_result_repair,verified_game_participation}')='true'::jsonb)
                       AS participation_verified,
                     r.team_abbr AS player_team
              FROM raw.nfl_rosters r JOIN raw.nfl_player_gamelogs l
                ON l.player_id=r.player_id AND l.game_id=a.game_id AND l.team_abbr=r.team_abbr
              WHERE a.player_name_norm IS NOT NULL AND r.season=g.season AND r.player_name_norm=a.player_name_norm
                AND r.team_abbr IN (a.home_team, a.away_team)
              LIMIT 1) gl ON TRUE
            WHERE a.commence_time_utc <= now()""")
        frame = pd.DataFrame([dict(r) for r in cur.fetchall()])
        if not frame.empty:
            frame['evidence_kind'] = 'alert_ledger'
        cur.execute("SELECT to_regclass('bets.nfl_sharp_observations')")
        if next(iter(cur.fetchone().values())) is not None and not frame.empty:
            cur.execute("SELECT payload FROM bets.nfl_sharp_observations WHERE observed_at_utc < now()")
            frame = frozen.append_decision_evidence(frame, [r['payload'] for r in cur.fetchall()])
        cur.execute("SELECT to_regclass('bets.nfl_sharp_quote_observations')")
        quotes = []
        if next(iter(cur.fetchone().values())) is not None and not frame.empty:
            cur.execute("""SELECT payload FROM bets.nfl_sharp_quote_observations
                WHERE game_id = ANY(%s) ORDER BY observed_at_utc""", (list(frame.game_id.dropna().unique()),))
            quotes = [r['payload'] for r in cur.fetchall()]
    return attach_indexed(frame, quotes)


def separate_money(summary):
    """Suggested stakes on legacy alerts are not confirmed executed wagers."""
    result = dict(summary)
    def paper(metrics):
        return {key: value for key, value in metrics.items()
                if key not in {'staked', 'realized_profit', 'mean_slippage'}}
    result = paper(result)
    for key in ('logged', 'early'):
        if key in result:
            result[key] = paper(result[key])
    for key in ('by_book', 'model'):
        if key in result:
            result[key] = {name: paper(metrics) for name, metrics in result[key].items()}
    result['execution_accounting'] = 'Paper stakes excluded; placed is legacy confirmation only; receipt net profit is separate.'
    return result


def paper_only(summary):
    """Retain the frozen diagnostics without converting a repair into cash approval.

    The user's current workflow is paper collection with actual-price checks.
    The older v10 gate permits a trial before those checks; expose its verdict as
    a diagnostic and retain a separate explicit publication hold.
    """
    result = dict(summary)
    original = summary.get('readiness') or {}
    result['frozen_readiness_diagnostic'] = original
    result['betting_authorized'] = False
    result['readiness'] = dict(original, status='paused',
        pause_reasons=list(dict.fromkeys(list(original.get('pause_reasons') or []) +
            ['Paper-only collection: actual-price evidence and explicit cash review are still required'])),
        meaning='paper checks continue; no new cash recommendations')
    result['verdict'] = 'paper only: no new stakes'
    result['exact_reference_by_book'] = {
        book: dict(decisions=int(s.get('alerts') or 0),
                   observed_exact=int(s.get('with_sharp_close') or 0)-int(s.get('shifted_sharp_closes') or 0),
                   estimated=int(s.get('shifted_sharp_closes') or 0))
        for book, s in (summary.get('by_book') or {}).items()}
    return result


def markdown(summary, frame):
    lines = ['# NFL price-gap paper evidence', '',
        '**Paper only. No cash authorization.** Actual sportsbook checks remain separate from API observations.', '',
        'Observed reference closes below match the exact selected line. Estimated line conversions do not count as observed closes.', '',
        '| Book | Decisions | Observed exact reference closes | Estimated conversions |',
        '|---|---:|---:|---:|']
    for book, item in summary['exact_reference_by_book'].items():
        lines.append(f"| {book} | {item['decisions']} | {item['observed_exact']} | {item['estimated']} |")
    lines += ['', 'The legacy mixed-book diagnostic below includes estimated conversions in its sharp-close coverage and EV. '
                  'Those figures are not exact-close coverage or evidence of obtainable profit.', '', '---', '',
              frozen.markdown(summary, frame)]
    return '\n'.join(lines)


def main():
    with psycopg2.connect(frozen.PG_DSN) as conn:
        conn.set_session(readonly=True)
        frame = frozen.grade(load_indexed(conn))
    summary = paper_only(separate_money(frozen.summarize(frame)))
    document = dict(built_at=datetime.now(timezone.utc).isoformat(), pass_bar=frozen.PASS, summary=summary,
                    reporting_contract='nfl-sharp-paper-money-separation-v1')
    frozen.atomic_json(frozen.ROOT/'reports/nfl_sharp_alerts_latest.json', document)
    (frozen.ROOT/'reports/nfl_sharp_alerts_latest.md').write_text(markdown(summary, frame), encoding='utf-8')
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
