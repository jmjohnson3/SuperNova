"""Render persisted pregame forecasts after ledger writes have succeeded."""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

import psycopg2

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import FEATURE_CONTRACT, release_artifact, atomic_json
from nfl_pipeline.modeling import predict_player_props as props
from nfl_pipeline.modeling import predict_today as games
from nfl_pipeline.betting_preferences import display_rows as preferred_rows
from nfl_pipeline.offer_selection import forecast_quote_error


def display_rows(conn, day, kind, *, quote_exclusions=None):
    table = {"prop": "nfl_player_prop_predictions", "game": "nfl_game_predictions"}[kind]
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT p.forecast_payload,
                   EXISTS(SELECT 1 FROM bets.nfl_bet_ledger l
                          WHERE l.source_kind=%s AND l.prediction_id=p.id
                            AND l.tier=p.tier) AS ledger_locked, p.id
            FROM bets.{table} p
            JOIN raw.nfl_games g ON g.game_id=p.game_id
            WHERE p.game_date_et=%s AND p.is_current AND p.integrity_version=%s
              AND g.start_ts_utc>NOW()
            ORDER BY p.id
        """, (kind, day, FEATURE_CONTRACT))
        records = cur.fetchall()
    rows = []
    for record in records:
        payload, locked = record[:2]
        row = dict(payload)
        row['forecast_id'] = record[2] if len(record)>2 else None
        error = forecast_quote_error(row, datetime.now(timezone.utc))
        if error:
            if quote_exclusions is not None:
                quote_exclusions.append(dict(game_id=str(row.get('game_id')), kind=kind, reason=error))
            continue
        row["ledger_locked"] = bool(locked)
        if row.get("tier") != "paper" and not locked:
            row["tier"] = "paper"
            row["reasons"] = str(row.get("reasons") or "") + ";not_locked_in_daily_ledger"
        rows.append(row)
    return preferred_rows(rows)


def render(day, kind):
    artifact = release_artifact("players" if kind == "prop" else "games")
    if not artifact:
        raise RuntimeError("No validated NFL release; refusing unversioned Discord cards")
    with psycopg2.connect(PG_DSN) as conn:
        rows = display_rows(conn, day, kind)
        if kind == 'prop':
            rows.extend(previously_locked_props(conn, day))
    from nfl_pipeline.cash_publication import research_rows
    rows = research_rows(rows)  # Text previews never reserve cash capacity.
    accepted = [stat for stat, metric in artifact["metrics"].items() if metric.get("accepted")]
    print(f"Release: {artifact['version']}")
    print('Book: FanDuel only. Other-book prices and links are not substituted.')
    if kind == "game":
        meta = {"game_date":str(day), "current_games":len({r['game_id'] for r in rows}),
                "game_lines":len(rows), "predictions":len(rows), "accepted_game_models":accepted}
        games.print_discord(rows, meta, games.PredictGameConfig(et_date=day))
    else:
        meta = {"game_date":str(day), "games":len({r['game_id'] for r in rows}),
                "snapshot_players":len({r['player_id'] for r in rows}),
                "offers":sum(r.get('offer_id') is not None for r in rows), "predictions":len(rows),
                "accepted_projection_models":accepted,
                "prediction_context_cutoff_utc":"recorded separately in each immutable forecast",
                "exact_line_status":"awaiting clean versioned lock/close proof"}
        props.print_discord(rows, meta, props.PredictConfig(et_date=day))
    print("- Ledger records are simulated recommendations unless you confirm actual execution.")


def previously_locked_props(conn, day):
    # A new forecast revision cannot erase an earlier immutable micro decision.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT p.forecast_payload FROM bets.nfl_bet_ledger l
            JOIN bets.nfl_player_prop_predictions p ON p.id=l.prediction_id
            JOIN raw.nfl_games g ON g.game_id=p.game_id
            WHERE l.source_kind='prop' AND l.game_date_et=%s
              AND l.tier='micro_projection' AND NOT p.is_current
              AND p.integrity_version=%s AND g.start_ts_utc>NOW()
            ORDER BY l.ledger_id
        """, (day, FEATURE_CONTRACT))
        records=cur.fetchall()
    return preferred_rows([dict(payload, tier='locked_micro', previously_locked=True) for (payload,) in records])


def matchup_bundle(day, game_id=None, *, reserve_cash=False):
    from nfl_pipeline.discord_matchups import build_bundle, preview_markdown
    artifacts = [release_artifact(kind) for kind in ('games', 'players')]
    if not all(artifacts):
        raise RuntimeError('No validated NFL release; refusing unversioned matchup cards')
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True, isolation_level='REPEATABLE READ')
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout='60s'")
            cur.execute("""SELECT game_id,home_team_abbr,away_team_abbr,start_ts_utc
                FROM raw.nfl_games WHERE game_date_et=%s AND start_ts_utc>NOW()
                ORDER BY start_ts_utc,game_id""", (day,))
            schedule = [dict(zip(('game_id','home_team_abbr','away_team_abbr','start_ts_utc'), r)) for r in cur.fetchall()]
        quote_exclusions = []
        game_rows = display_rows(conn, day, 'game', quote_exclusions=quote_exclusions)
        prop_rows = display_rows(conn, day, 'prop', quote_exclusions=quote_exclusions) + previously_locked_props(conn, day)
    from nfl_pipeline.cash_publication import prepare
    # Scope before reservation, while preserving the existing global strategy selections.
    from nfl_pipeline.game_scope import game_ids
    scope = game_ids()
    allowed = set(scope) if scope is not None else {str(g['game_id']) for g in schedule}
    if game_id is not None:
        allowed &= {str(game_id)}
    game_rows = [r for r in game_rows if str(r['game_id']) in allowed]
    prop_rows = [r for r in prop_rows if str(r['game_id']) in allowed]
    game_rows, prop_rows, readiness = prepare(day, game_rows, prop_rows, reserve=reserve_cash)
    bundle = build_bundle(day, schedule, game_rows, prop_rows, [a['version'] for a in artifacts],
                          quote_exclusions=quote_exclusions, cash_readiness=readiness)
    from nfl_pipeline.game_scope import game_ids
    scope = game_ids()
    if scope is not None:
        bundle['cards'] = [c for c in bundle['cards'] if c['game_id'] in scope]
        bundle['games'] = len({c['game_id'] for c in bundle['cards']})
        bundle['scope_game_ids'] = scope
        bundle['notice'] = None if bundle['cards'] else 'No upcoming games in the requested pregame scope.'
    if game_id is not None:
        # Filter after selection so an individual-game preview cannot reset slate caps/ranks.
        bundle['cards'] = [c for c in bundle['cards'] if c['game_id'] == str(game_id)]
        bundle['games'] = len({c['game_id'] for c in bundle['cards']})
        bundle['notice'] = None if bundle['cards'] else 'No upcoming saved matchup matches that game ID.'
    suffix = f'_{game_id}' if game_id is not None else ''
    if scope is not None:
        import hashlib
        suffix += '_scope_'+hashlib.sha256(json.dumps(scope).encode()).hexdigest()[:12]
    if game_id is not None and any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in str(game_id)):
        raise ValueError('Invalid game ID')
    root = Path(__file__).resolve().parents[2] / 'reports'
    atomic_json(root / f'nfl_discord_matchups_{day}{suffix}.json', bundle)
    (root / f'nfl_discord_matchups_{day}{suffix}.md').write_text(preview_markdown(bundle), encoding='utf-8')
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True)
    parser.add_argument('--kind', choices=['game','prop','matchups'], required=True)
    parser.add_argument('--json', action='store_true', help='Structured matchup cards for the daily runner')
    parser.add_argument('--game-id', help='Preview a single matchup without changing slate-wide selection')
    parser.add_argument('--reserve-cash', action='store_true', help='Reserve eligible capacity for imminent publication')
    args = parser.parse_args()
    if args.kind == 'matchups':
        from nfl_pipeline.discord_matchups import preview_markdown
        bundle = matchup_bundle(date.fromisoformat(args.date), args.game_id, reserve_cash=args.reserve_cash)
        print(json.dumps(bundle) if args.json else preview_markdown(bundle))
    else:
        if args.json or args.game_id:
            parser.error('--json and --game-id require --kind matchups')
        render(date.fromisoformat(args.date), args.kind)


if __name__ == '__main__':
    main()
