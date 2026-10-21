"""Grade the sharp price-gap cash strategy at the sharp close, and decide whether staking continues.

Only fresh, source-timestamped immutable observations from the registered strategy enter the
primary evidence: one decision per player-game-stat, the first eligible gap inside the stake window
at the best book, whatever its stake. The headline number is EV at the sharp close - the bet's price
valued at Pinnacle's last no-vig quote before kickoff - because it is known at kickoff for every
decision, while win/loss needs hundreds of bets to separate skill from noise. Missing closes remain
unknown, and too many of them pause staking: an ungradable strategy is a broken scanner.

The readiness verdict (sharp_strategy.GATES) is what sharp_watch reads before staking: 'trial'
($1 flat within the limits), 'paused' (close capture below 90%, closing EV not positive after 50
decisions, or the $30 loss pause) or 'scale_review_ready' (150+ decisions over 3+ weeks at +1.5%
or better at the close, with no player or team carrying the result). Raising stakes stays a manual
decision; nothing here places a bet.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline import sharp_math
from nfl_pipeline import sharp_strategy as strategy
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.sharp_watch import (STAKE_WINDOW_MINUTES, SHARP_BOOKS, BET_BOOKS,
                                     QUOTE_CONTRACT, quote_error, source_time, EXPERIMENT_NAME)
from nfl_pipeline.sharp_watch import _digest, MAX_SOURCE_SKEW_SECONDS
from nfl_pipeline.context_contract import safe_time
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[3]
PASS = dict(strategy.GATES, alpha=.05)
MAX_CLOSE_AGE_MINUTES = 20
# Close capture is judged on the most recent decisions, so a fixed scanner recovers instead of
# carrying an old outage forever.
CLOSE_COVERAGE_WINDOW = 30


def load(conn) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='120s'")
        cur.execute("SELECT to_regclass('bets.nfl_sharp_alerts')")
        if cur.fetchone()["to_regclass"] is None:
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
        decisions_exist = next(iter(cur.fetchone().values())) is not None
        if decisions_exist and not frame.empty:
            cur.execute("SELECT payload FROM bets.nfl_sharp_observations WHERE observed_at_utc < now()")
            frame = append_decision_evidence(frame, [r['payload'] for r in cur.fetchall()])
        cur.execute("SELECT to_regclass('bets.nfl_sharp_quote_observations')")
        exists = next(iter(cur.fetchone().values())) is not None
        quotes = []
        if exists and not frame.empty:
            cur.execute("""SELECT payload FROM bets.nfl_sharp_quote_observations
                WHERE game_id = ANY(%s) ORDER BY observed_at_utc""", (list(frame.game_id.dropna().unique()),))
            quotes = [r['payload'] for r in cur.fetchall()]
        return attach_exact_evidence(frame, quotes)


def append_decision_evidence(frame, observations):
    """New immutable snapshots are independent of the legacy notification price key.

    An unchanged price can exist in an old alert or another experiment. It must
    still produce prospective evidence. Actual-wager fields remain only on the
    original ledger row, never on these simulated decisions.
    """
    originals = frame.to_dict('records')
    bases = {(str(r.get('game_id')), r.get('book'), r.get('stat'), r.get('player_name_norm')): r
             for r in originals}
    additions = []
    for observation in observations:
        key = (str(observation.get('game_id')), observation.get('book'), observation.get('stat'),
               observation.get('player_norm'))
        base = bases.get(key)
        if base is None:
            continue  # No verified game/result identity; never guess another player's result.
        row = dict(base, evidence_kind='decision_observation',
                   observation_id=observation.get('observation_id'),
                   alert_id='observation:'+str(observation.get('observation_id')),
                   initial_provenance=observation, alert_provenance=None,
                   initial_observation_id=observation.get('observation_id'), alert_observation_id=None,
                   fd_line=observation.get('line'), fd_price=observation.get('price'),
                   fd_link=observation.get('link'), side=observation.get('side'),
                   ev=observation.get('ev'), tier=observation.get('tier'),
                   sharp_book=observation.get('sharp_book'), sharp_line=observation.get('sharp_line'),
                   alerted_at=(observation.get('quote_provenance') or {}).get('observed_at_utc'),
                   event_id=observation.get('event_id'), model_agrees=observation.get('model_agrees'),
                   model_filter=observation.get('model_filter'),
                   model_filter_reasons=observation.get('model_filter_reasons'),
                   placed_at=None, taken_price=None, stake=0.)
        additions.append(row)
    return pd.DataFrame(originals+additions)


def _provenance(row):
    for key in ('alert_provenance', 'initial_provenance'):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return {}


def attach_exact_evidence(frame, quotes):
    """Only immutable, source-timestamped, same-book closes are evidence.

    The execution close must be the same book at the exact line. The reference close is the sharp
    book's latest fresh quote for the player, at the exact line when it still hangs one, otherwise
    its nearest line (graded by moving it to the bet's line, see grade()).
    """
    if frame.empty:
        return frame
    by_game = {}
    for q in quotes:
        by_game.setdefault(str(q.get('game_id')), []).append(q)
    rows = []
    for row in frame.to_dict('records'):
        provenance = _provenance(row)
        trace = provenance.get('quote_provenance') or {}
        observed = safe_time(trace.get('observed_at_utc'))
        kick = safe_time(row.get('commence_time_utc'))
        execution, reference = trace.get('execution_quote') or {}, trace.get('reference_quote') or {}
        body = {k: v for k, v in provenance.items() if k != 'observation_id'}
        identity_matches = all(provenance.get(k) == row.get(v) for k, v in
            (('game_id','game_id'), ('event_id','event_id'), ('book','book'), ('stat','stat'), ('player_norm','player_name_norm'),
             ('side','side')))
        try:
            identity_matches = identity_matches and float(provenance['line']) == float(row['fd_line']) \
                and float(provenance['price']) == float(row['fd_price'])
        except (KeyError, TypeError, ValueError):
            identity_matches = False
        valid = (trace.get('contract') == QUOTE_CONTRACT and observed and kick and observed < kick
                 and identity_matches and provenance.get('observation_id') == _digest(body)
                 and safe_time(provenance.get('commence_time_utc')) == kick
                 and not quote_error(execution, observed) and not quote_error(reference, observed)
                 and abs((source_time(execution)-source_time(reference)).total_seconds()) <= MAX_SOURCE_SKEW_SECONDS)
        row.update(provenance_valid=bool(valid), decision_at=observed,
                   experiment_sha256=provenance.get('experiment_sha256'),
                   reference_method=trace.get('reference_method'),
                   exact_execution_close=None, exact_reference_close=None,
                   exact_close_reason='legacy_or_invalid_observation' if not valid else 'exact_close_missing')
        if valid:
            row['fd_then_over'], row['fd_then_under'] = execution.get('over'), execution.get('under')
            # Only a fresh close found below may grade this decision, never a value already on the row.
            for prefix in ('fd_close', 'sharp_close'):
                row.update({prefix+'_line': None, prefix+'_over': None, prefix+'_under': None})
            candidates = []
            for q in by_game.get(str(row.get('game_id')), []):
                fetched = safe_time(q.get('observed_at_utc'))
                updated = source_time(q.get('quote') or {})
                if (q.get('stat') != row.get('stat') or q.get('player_name_norm') != row.get('player_name_norm')
                        or q.get('event_id') != row.get('event_id')
                        or safe_time(q.get('commence_time_utc')) != kick
                        or fetched is None or updated is None
                        or not observed <= fetched < kick or not updated <= fetched
                        or (kick-fetched).total_seconds() > MAX_CLOSE_AGE_MINUTES*60
                        or (kick-updated).total_seconds() > MAX_CLOSE_AGE_MINUTES*60
                        or quote_error(q.get('quote') or {}, fetched)):
                    continue
                candidates.append(q)
            for book, name in ((row.get('book') or 'fanduel', 'exact_execution_close'),
                               (row.get('sharp_book'), 'exact_reference_close')):
                same_book = [q for q in candidates if q.get('book') == book]
                if same_book:
                    diagnostic = max(same_book, key=lambda q: (safe_time(q['observed_at_utc']),
                                                              -abs(float(q['line'])-float(row['fd_line']))))
                    prefix = 'fd_close' if name == 'exact_execution_close' else 'sharp_close'
                    row.update({prefix+'_line':diagnostic['line'], prefix+'_over':diagnostic['quote'].get('over'),
                                prefix+'_under':diagnostic['quote'].get('under')})
                eligible = [q for q in candidates if q.get('book') == book
                            and float(q['line']) == float(row['fd_line'])]
                if eligible:
                    row[name] = max(eligible, key=lambda q: safe_time(q['observed_at_utc']))
            row['exact_close_reason'] = 'valid' if row['exact_execution_close'] else 'fresh_same_book_exact_line_missing'
        rows.append(row)
    return pd.DataFrame(rows)


def _side(p_over, side):
    return None if p_over is None else (p_over if side == "over" else 1.0 - p_over)


def fd_clv(r) -> float | None:
    """Legacy/model-inferred movement, diagnostic only; never exact-close evidence."""
    if pd.isna(r.fd_close_line) or pd.isna(r.fd_then_over):
        return None
    close_over = sharp_math.shift_probability(r.stat, float(r.fd_close_line),
                                              sharp_math.no_vig_over(r.fd_close_over, r.fd_close_under), float(r.fd_line))
    then_over = sharp_math.no_vig_over(r.fd_then_over, r.fd_then_under)
    close, then = _side(close_over, r.side), _side(then_over, r.side)
    return None if close is None or then is None else close - then


def grade(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for col in ("fd_line", "fd_price", "close_price", "sharp_close_line", "home_score", "away_score", "actual",
                "fd_close_line", "fd_close_over", "fd_close_under", "fd_then_over", "fd_then_under"):
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["inferred_clv"] = [fd_clv(r) for r in df.itertuples()]
    if 'provenance_valid' not in df:
        df = attach_exact_evidence(df, [])
    # A placed bet is graded at the price actually taken; everything else at the alerted price.
    taken = pd.to_numeric(df.get("taken_price"), errors="coerce") if "taken_price" in df else pd.Series(np.nan, index=df.index)
    if 'placed_at' in df:
        taken = taken.where(df.placed_at.notna())
    else:
        taken[:] = np.nan
    df["bet_price"] = taken.fillna(df.fd_price)
    df["slippage"] = [None if pd.isna(t) else sharp_math.implied(t) - sharp_math.implied(p)
                      for t, p in zip(taken, df.fd_price)]
    ev_close = []
    for r in df.itertuples():
        fair_over = None
        if pd.notna(r.sharp_close_line):
            fair_over = sharp_math.over_probability_at(r.stat, float(r.sharp_close_line),
                                                       sharp_math.no_vig_over(r.sharp_close_over, r.sharp_close_under), float(r.fd_line))
        fair = None if fair_over is None else (fair_over if r.side == "over" else 1 - fair_over)
        ev_close.append(None if fair is None else sharp_math.ev(fair, r.bet_price))
    df["inferred_ev_at_sharp_close"] = ev_close
    exact_clv, exact_ev, shifted_ev, method = [], [], [], []
    for row in df.to_dict('records'):
        execution, reference = row.get('exact_execution_close'), row.get('exact_reference_close')
        before = (_provenance(row).get('quote_provenance') or {}).get('execution_quote') or {}
        then = _side(sharp_math.no_vig_over(before.get('over'), before.get('under')), row['side'])
        close = None
        if isinstance(execution, dict) and row.get('provenance_valid'):
            q = execution['quote']
            close = _side(sharp_math.no_vig_over(q.get('over'), q.get('under')), row['side'])
        exact_clv.append(None if then is None or close is None else close-then)
        fair = None
        if isinstance(reference, dict) and row.get('provenance_valid'):
            q = reference['quote']
            fair = _side(sharp_math.no_vig_over(q.get('over'), q.get('under')), row['side'])
        exact_ev.append(sharp_math.ev(fair, row['bet_price']) if fair is not None else None)
        # Pinnacle moves player lines by a yard or two into kickoff. Its close at the nearest line is
        # moved to the bet's line with the same normal model that priced nearby lines at decision time
        # (counts and other exact-only stats are never moved), so a moved line is graded, not lost.
        moved = None
        if fair is None and row.get('provenance_valid') and pd.notna(row.get('sharp_close_line')):
            p_over = sharp_math.shift_probability(row['stat'], float(row['sharp_close_line']),
                sharp_math.no_vig_over(row.get('sharp_close_over'), row.get('sharp_close_under')), float(row['fd_line']))
            side_fair = _side(p_over, row['side'])
            moved = sharp_math.ev(side_fair, row['bet_price']) if side_fair is not None else None
        shifted_ev.append(moved)
        method.append('exact' if fair is not None else 'shifted' if moved is not None else None)
    df['clv'], df['ev_at_sharp_close_exact'] = exact_clv, exact_ev
    df['ev_at_sharp_close'] = [e if e is not None else s for e, s in zip(exact_ev, shifted_ev)]
    df['sharp_close_method'] = method
    actual = np.where(df.stat == "total", df.home_score + df.away_score, df.actual)
    participation = df.get('participation_verified', pd.Series(False, index=df.index)).fillna(False)
    actual = np.where(df.stat.eq('total') | participation, actual, np.nan)
    diff = (actual - df.fd_line) * np.where(df.side == "over", 1, -1)
    df["result"] = np.where(pd.isna(actual) | (df.status != "final"), None, np.where(diff > 0, "win", np.where(diff < 0, "loss", "push")))
    payout = np.where(df.bet_price > 0, df.bet_price / 100, 100 / df.bet_price.abs())
    df["units"] = np.where(df.result == "win", payout, np.where(df.result == "loss", -1.0, np.where(df.result == "push", 0.0, np.nan)))
    return df


def _stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(alerts=0, weeks=0)
    clv = pd.to_numeric(df.clv, errors="coerce").dropna()
    evc = pd.to_numeric(df.ev_at_sharp_close, errors="coerce").dropna()
    weeks = df[["season", "week"]].drop_duplicates().shape[0]
    method = df.get('sharp_close_method', pd.Series(index=df.index, dtype=object))
    return dict(alerts=int(len(df)), weeks=int(weeks), with_fd_close=int(len(clv)), beat_fd_close=float((clv > 1e-9).mean()) if len(clv) else None,
                exact_close_coverage=float(len(clv)/len(df)), sharp_close_coverage=float(len(evc)/len(df)),
                shifted_sharp_closes=int(method.eq('shifted').sum()),
                mean_clv_lower=_week_lower(df, 'clv'), sharp_close_ev_lower=_week_lower(df, 'ev_at_sharp_close'),
                player_games=int(df[['game_id','player_name_norm']].drop_duplicates().shape[0])
                    if {'game_id','player_name_norm'} <= set(df) else 0,
                mean_fd_clv=float(clv.mean()) if len(clv) else None, with_sharp_close=int(len(evc)),
                mean_ev_at_sharp_close=float(evc.mean()) if len(evc) else None,
                positive_at_sharp_close=float((evc > 0).mean()) if len(evc) else None,
                settled=int(df.result.notna().sum()), roi=float(pd.to_numeric(df.units, errors="coerce").dropna().mean()) if df.result.notna().any() else None,
                mean_alert_ev=float(pd.to_numeric(df.ev, errors="coerce").mean()),
                staked=float(_stake(df).sum()),
                mean_slippage=float(pd.to_numeric(df.get("slippage"), errors="coerce").dropna().mean())
                if "slippage" in df and pd.to_numeric(df["slippage"], errors="coerce").notna().any() else None,
                # Dollars won or lost on settled rows. sharp_watch reads this for its sticky loss pause.
                realized_profit=float((pd.to_numeric(df.units, errors="coerce") * _stake(df)).dropna().sum()))


def _week_lower(df, field):
    values = df.assign(value=pd.to_numeric(df[field], errors='coerce')).dropna(subset=['value','season','week'])
    means = values.groupby(['season','week']).value.mean().to_numpy()
    if len(means) < 3:
        return None
    return float(means.mean() - t.ppf(1-PASS['alpha'], len(means)-1)*means.std(ddof=1)/np.sqrt(len(means)))


def fixed_decisions(df, keys=('book', 'game_id', 'player_name_norm', 'stat')):
    """One first decision per key; never choose after seeing outcomes. Within one poll the best
    EV comes first, which is the book sharp_watch stakes."""
    if df.empty:
        return df
    frame = df.copy()
    for field in ('book','game_id','player_name_norm','stat'):
        if field not in frame:
            frame[field] = 'unknown'
        frame[field] = frame[field].fillna('game' if field == 'player_name_norm' else 'unknown')
    frame['_time'] = pd.to_datetime(frame.get('decision_at', frame.get('alerted_at')), utc=True, errors='coerce')
    frame['_ev'] = -pd.to_numeric(frame.get('ev'), errors='coerce').fillna(0.0)
    if 'alert_id' not in frame:
        frame['alert_id'] = np.arange(len(frame))
    frame['alert_id'] = frame['alert_id'].astype(str)
    return frame.sort_values(['_time', '_ev', 'alert_id'], kind='stable').drop_duplicates(
        list(keys), keep='first').drop(columns=['_time', '_ev'])


def cash_decisions(df):
    """The strategy's decisions: one per player-game-stat across books, as sharp_watch stakes them."""
    return fixed_decisions(df, keys=('game_id', 'player_name_norm', 'stat'))


def _stake(df: pd.DataFrame) -> pd.Series:
    if "stake" not in df:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df["stake"], errors="coerce").fillna(0.0)


def _tier(df: pd.DataFrame) -> pd.Series:
    """Tier as the strategy is now defined: an 'alert' seen outside the stake window is relabelled
    'early' or 'late' (rows from before the window existed; stored rows are not rewritten)."""
    tier = df["tier"].fillna("alert") if "tier" in df else pd.Series("alert", index=df.index)
    if {"commence_time_utc", "alerted_at"} <= set(df.columns):
        observed = pd.to_datetime(df.alerted_at, utc=True)
        if 'decision_at' in df:
            observed = pd.to_datetime(df.decision_at, utc=True).fillna(observed)
        lead = (pd.to_datetime(df.commence_time_utc, utc=True) - observed).dt.total_seconds() / 60
        closes_at, opens_at = STAKE_WINDOW_MINUTES
        tier = tier.where(~((tier == "alert") & (lead > opens_at)), "early")
        tier = tier.where(~((tier == "alert") & (lead < closes_at)), "late")
    return tier


def _shares(df, column):
    values = df[column].dropna() if column in df else pd.Series(dtype=object)
    values = values[values.astype(str).ne('')]
    return values.value_counts(normalize=True) if len(values) else pd.Series(dtype=float)


def _max_loss_drawdown(placed: pd.DataFrame) -> float:
    """Lowest cumulative realized profit on confirmed bets, in settlement order. Sticky by design:
    winning later bets cannot lift a pause that was already reached."""
    if placed.empty:
        return 0.0
    profit = pd.to_numeric(placed.units, errors="coerce") * _stake(placed)
    order = pd.to_datetime(placed.commence_time_utc, utc=True, errors="coerce")
    running = profit.fillna(0.0).groupby(order).sum().sort_index().cumsum()
    return float(min(0.0, running.min())) if len(running) else 0.0


def readiness(primary: pd.DataFrame, stats: dict, placed: pd.DataFrame) -> dict:
    """The four gates for this strategy, and what they allow."""
    g = strategy.GATES
    decisions = int(stats.get("alerts") or 0)
    recent = primary
    if not primary.empty:
        recent = (primary.assign(_t=pd.to_datetime(primary.commence_time_utc, utc=True))
                  .sort_values(['season', 'week', '_t'], kind='stable').tail(CLOSE_COVERAGE_WINDOW))
    recent_coverage = (float(pd.to_numeric(recent.ev_at_sharp_close, errors='coerce').notna().mean())
                       if len(recent) else None)
    mean_ev = stats.get("mean_ev_at_sharp_close")
    with_close = int(stats.get("with_sharp_close") or 0)
    players, teams = _shares(primary, 'player_name_norm'), _shares(primary, 'player_team')
    drawdown = _max_loss_drawdown(placed)
    gates = dict(
        close_capture=dict(recent_coverage=recent_coverage, recent_decisions=int(len(recent)),
                           all_coverage=stats.get("sharp_close_coverage"), minimum=g["min_close_coverage"],
                           judged=len(recent) >= g["close_coverage_min_decisions"]),
        closing_ev=dict(mean=mean_ev, decisions_with_close=with_close, weekly_lower=stats.get("sharp_close_ev_lower"),
                        kill_after=g["kill_min_decisions"], scale_minimum=g["scale_min_mean_close_ev"]),
        loss_pause=dict(lowest_cumulative_profit=drawdown, pause_at=strategy.LOSS_PAUSE),
        concentration=dict(max_player_share=float(players.max()) if len(players) else None,
                           max_team_share=float(teams.max()) if len(teams) else None,
                           player_limit=g["max_player_share"], team_limit=g["max_team_share"]),
    )
    pause = []
    if gates["close_capture"]["judged"] and recent_coverage < g["min_close_coverage"]:
        pause.append(f"close capture {recent_coverage:.0%} of the last {len(recent)} decisions (needs {g['min_close_coverage']:.0%})")
    if with_close >= g["kill_min_decisions"] and mean_ev is not None and mean_ev <= g["kill_max_mean_close_ev"]:
        pause.append(f"closing EV {mean_ev:+.2%} after {with_close} decisions")
    if drawdown <= strategy.LOSS_PAUSE:
        pause.append(f"loss pause: cumulative profit reached ${drawdown:+.2f}")
    concentrated = ((gates["concentration"]["max_player_share"] or 0) > g["max_player_share"]
                    or (gates["concentration"]["max_team_share"] or 0) > g["max_team_share"])
    scale = (decisions >= g["scale_min_decisions"] and (stats.get("weeks") or 0) >= g["scale_min_weeks"]
             and mean_ev is not None and mean_ev >= g["scale_min_mean_close_ev"]
             and (stats.get("sharp_close_coverage") or 0) >= g["min_close_coverage"] and not concentrated)
    status = "paused" if pause else "scale_review_ready" if scale else "trial"
    return dict(status=status, pause_reasons=pause, gates=gates, decisions=decisions,
                concentration_ok=not concentrated,
                meaning={"trial": "$1 flat within the daily, weekly and per-game limits",
                         "paused": "no new stakes; gaps are still scanned and graded",
                         "scale_review_ready": "evidence supports reviewing a larger stake; raising it is manual"}[status])


def summarize(df: pd.DataFrame, experiment=None) -> dict:
    """Primary evidence is the registered strategy's decisions inside the stake window. Earlier,
    later and logged gaps are graded beside them as context."""
    if df.empty:
        empty = dict(alerts=0, weeks=0)
        return dict(alerts=0, verdict="no graded decisions yet", logged=empty, early=empty, late=empty,
                    placed=empty, readiness=readiness(df, empty, df))
    if experiment is None:
        from nfl_pipeline.experiment_registry import load_experiment
        experiment = load_experiment(EXPERIMENT_NAME)
    tier = _tier(df)
    books = df["book"].fillna("fanduel") if "book" in df else pd.Series("fanduel", index=df.index)
    eligible = books.isin(BET_BOOKS) & df.provenance_valid.fillna(False)
    eligible &= df.get('evidence_kind', pd.Series('', index=df.index)).eq('decision_observation')
    eligible &= df.stat.isin(strategy.STAKED_STATS)
    eligible &= df.experiment_sha256.eq((experiment or {}).get('sha256', '__unregistered__'))
    eligible &= df.game_id.notna() if 'game_id' in df else False
    primary = cash_decisions(df[eligible & tier.eq('alert')])
    out = _stats(primary)
    out['experiment_id'] = EXPERIMENT_NAME
    out['experiment_sha256'] = (experiment or {}).get('sha256')
    ledger = df.get('evidence_kind', pd.Series('alert_ledger', index=df.index)).eq('alert_ledger')
    out['raw_alerts'] = int((tier.eq('alert') & ledger).sum())
    out['captured_decisions'] = int((tier.eq('alert') & ~ledger).sum())
    out['excluded_alerts'] = int((tier.eq('alert') & ~eligible).sum())
    out['repeated_alerts_removed'] = int((eligible & tier.eq('alert')).sum()) - len(primary)
    for name in ("logged", "early", "late"):
        out[name] = _stats(cash_decisions(df[eligible & tier.eq(name)]))
    out['by_close_method'] = {str(k): _stats(g) for k, g in primary.groupby(primary.sharp_close_method)} \
        if 'sharp_close_method' in primary and primary.sharp_close_method.notna().any() else {}
    # Bets actually placed, confirmed by hand with sharp_bets --confirm. This is the real money record.
    placed = df[df.placed_at.notna()] if "placed_at" in df else df.iloc[:0]
    out["placed"] = _stats(placed)
    out["by_book"] = {str(book): _stats(g) for book, g in primary.groupby(primary.book)} if not primary.empty else {}
    out["by_stat"] = {str(stat): _stats(g) for stat, g in primary.groupby(primary.stat)} if not primary.empty else {}
    diagnostic = fixed_decisions(df[tier.eq('alert') & ~ledger])
    out['inferred_close_diagnostics'] = {
        key: dict(rows=int(pd.to_numeric(diagnostic[key], errors='coerce').notna().sum()),
                  mean=float(pd.to_numeric(diagnostic[key], errors='coerce').mean())
                      if pd.to_numeric(diagnostic[key], errors='coerce').notna().any() else None)
        for key in ('inferred_clv', 'inferred_ev_at_sharp_close')}
    # Does the model add anything on top of the price edge? Shadow only (sharp_strategy.SHADOW_FILTER).
    if "model_agrees" in df:
        priced = primary[primary.model_agrees.notna()]
        out["model"] = {("agrees" if bool(k) else "disagrees"): _stats(g)
                        for k, g in priced.groupby(priced.model_agrees)}
    out["shadow_filter"] = shadow_filter_summary(primary)
    out["readiness"] = readiness(primary, out, placed)
    out["verdict"] = {"trial": "trial: $1 stakes within limits", "paused": "paused: no new stakes",
                      "scale_review_ready": "scale review ready; raising stakes is a manual decision"}[out["readiness"]["status"]]
    return out


def shadow_filter_summary(primary: pd.DataFrame) -> dict:
    """Closing EV for gaps the model would let through versus those it would skip. The filter
    earns a place in the strategy only if skipping the flagged gaps raises EV at the close."""
    if primary.empty or 'model_filter' not in primary:
        return dict(groups={}, judged=False, improves_closing_ev=None)
    groups = {str(k): _stats(g) for k, g in primary.groupby(primary.model_filter.fillna('not_recorded'))}
    flagged, passed = groups.get('flag') or {}, groups.get('pass') or {}
    need = strategy.SHADOW_FILTER['min_decisions_to_judge']
    judged = (flagged.get('with_sharp_close') or 0) >= need // 3 and (passed.get('with_sharp_close') or 0) >= need
    overall = _stats(primary).get('mean_ev_at_sharp_close')
    improves = None
    if judged and passed.get('mean_ev_at_sharp_close') is not None and overall is not None:
        # Skipping the flagged gaps helps exactly when the gaps it keeps beat the whole set at the close.
        improves = passed['mean_ev_at_sharp_close'] > overall
    return dict(groups=groups, judged=judged, improves_closing_ev=improves,
                mean_ev_all=overall, mean_ev_if_filtered=passed.get('mean_ev_at_sharp_close'))


def markdown(summary: dict, df: pd.DataFrame) -> str:
    pct = lambda v: "-" if v is None else f"{v:.1%}"
    signed = lambda v: "-" if v is None else f"{v:+.2%}"
    a = summary
    ready = summary.get("readiness") or {}
    gates = ready.get("gates") or {}
    lines = ["# NFL Sharp-Gap Strategy", "",
             f"Verdict: **{summary['verdict']}**" + (f" - {ready.get('meaning')}" if ready.get('meaning') else ""), ""]
    if ready.get("pause_reasons"):
        lines += ["Paused because: " + "; ".join(ready["pause_reasons"]), ""]
    lines += [f"Registration `{summary.get('experiment_id', EXPERIMENT_NAME)}`: {', '.join(BET_BOOKS)} at "
              f"+{strategy.MIN_EV:.0%} or better against Pinnacle no-vig, seen {STAKE_WINDOW_MINUTES[1]}-"
              f"{STAKE_WINDOW_MINUTES[0]} minutes before kickoff, {', '.join(sorted(strategy.STAKED_STATS))}. "
              "One decision per player-game-stat (the best book when first seen), whatever its stake.", "",
              f"Decisions graded: {a.get('alerts', 0)} over {a.get('weeks', 0)} week(s).", "",
              "## Gates", "",
              "| Gate | Now | Rule |", "|---|---:|---|"]
    cc, ev, lp, conc = (gates.get(k) or {} for k in ("close_capture", "closing_ev", "loss_pause", "concentration"))
    lines += [
        f"| Close capture (last {cc.get('recent_decisions', 0)}) | {pct(cc.get('recent_coverage'))} | pause below "
        f"{strategy.GATES['min_close_coverage']:.0%} once {strategy.GATES['close_coverage_min_decisions']} decisions exist |",
        f"| Mean EV at the sharp close | {signed(ev.get('mean'))} ({ev.get('decisions_with_close', 0)} graded) | pause at or below "
        f"{strategy.GATES['kill_max_mean_close_ev']:+.0%} after {strategy.GATES['kill_min_decisions']}; scale review at "
        f"{strategy.GATES['scale_min_mean_close_ev']:+.1%} over {strategy.GATES['scale_min_decisions']}+ and "
        f"{strategy.GATES['scale_min_weeks']}+ weeks |",
        f"| Loss pause | ${lp.get('lowest_cumulative_profit', 0):+.2f} lowest | sticky at ${strategy.LOSS_PAUSE:+.0f} confirmed |",
        f"| Largest player / team share | {pct(conc.get('max_player_share'))} / {pct(conc.get('max_team_share'))} | "
        f"<= {strategy.GATES['max_player_share']:.0%} / {strategy.GATES['max_team_share']:.0%} to scale |", "",
        "## Evidence", "",
        "| Measure (most reliable first) | Decisions | Earlier gaps | Later gaps | +1% to +3% gaps |",
        "|---|---:|---:|---:|---:|"]
    e, late, g = (summary.get(k) or {} for k in ("early", "late", "logged"))
    row = lambda label, key, fmt: f"| {label} | {fmt(a.get(key))} | {fmt(e.get(key))} | {fmt(late.get(key))} | {fmt(g.get(key))} |"
    lines += [row("Count", "alerts", lambda v: "-" if v is None else str(v)),
              row("Sharp close coverage", "sharp_close_coverage", pct),
              row("Mean EV at the sharp close", "mean_ev_at_sharp_close", signed),
              row("Share still +EV at the sharp close", "positive_at_sharp_close", pct),
              row("Beat the book's own exact close", "beat_fd_close", pct),
              row("EV claimed when seen", "mean_alert_ev", signed),
              row("Simulated flat ROI (descriptive)", "roi", signed),
              row("Settled", "settled", lambda v: "-" if v is None else str(v))]
    methods = summary.get("by_close_method") or {}
    if methods:
        lines += ["", "Sharp close method: " + "; ".join(
            f"{k} {v.get('alerts', 0)} at {signed(v.get('mean_ev_at_sharp_close'))}" for k, v in sorted(methods.items()))
            + ". 'shifted' moves Pinnacle's nearest closing line to the bet's line (yardage only)."]
    for title, key in (("By book", "by_book"), ("By market", "by_stat")):
        groups = summary.get(key) or {}
        if groups:
            lines += ["", f"### {title}", "", "| | Decisions | EV at sharp close | Still +EV | ROI |", "|---|---:|---:|---:|---:|"]
            for name, st in sorted(groups.items()):
                lines.append(f"| {name} | {st.get('alerts', 0)} | {signed(st.get('mean_ev_at_sharp_close'))} | "
                             f"{pct(st.get('positive_at_sharp_close'))} | {signed(st.get('roi'))} |")
    placed = summary.get("placed") or {}
    if placed.get("alerts"):
        slip = placed.get("mean_slippage")
        lines += ["", f"**Confirmed bets: {placed['alerts']} for ${placed.get('staked', 0):.2f} staked; "
                      f"realized ${placed.get('realized_profit', 0):+.2f} on {placed.get('settled', 0)} settled "
                      f"(EV at sharp close {signed(placed.get('mean_ev_at_sharp_close'))}"
                      + (f", slippage {signed(slip)} vs the alerted price" if slip is not None else "") + ").**"]
    shadow = summary.get("shadow_filter") or {}
    if shadow.get("groups"):
        lines += ["", "## Shadow model filter", "",
                  "Flags a gap when the model sees an injury tag, a limited-usage role, or strongly prefers the other "
                  "side. Recorded only; it never skips a bet until skipping flagged gaps raises EV at the close.", "",
                  "| Verdict | Decisions | EV at sharp close | Still +EV | ROI |", "|---|---:|---:|---:|---:|"]
        for name, st in sorted(shadow["groups"].items()):
            lines.append(f"| {name} | {st.get('alerts', 0)} | {signed(st.get('mean_ev_at_sharp_close'))} | "
                         f"{pct(st.get('positive_at_sharp_close'))} | {signed(st.get('roi'))} |")
        lines.append("")
        lines.append("Judged: " + ("yes; improves closing EV: " + str(shadow.get('improves_closing_ev')) if shadow.get("judged")
                     else f"not yet (needs {strategy.SHADOW_FILTER['min_decisions_to_judge']} passed decisions with a close)."))
    inferred = summary.get('inferred_close_diagnostics') or {}
    lines += ["", "Legacy closes (diagnostic only):"]
    for name, data in inferred.items():
        lines.append(f"- {name}: {data['rows']} rows; mean {signed(data['mean'])}.")
    lines += ["", f"Legacy alerts: {summary.get('raw_alerts', 0)}; captured decisions: {summary.get('captured_decisions', 0)}; "
              f"excluded from primary: {summary.get('excluded_alerts', 0)}; "
              f"repeated eligible observations removed: {summary.get('repeated_alerts_removed', 0)}.",
              "Missing exact closes remain unknown. Weekly confidence bounds use equal week weights. "
              "Slippage is positive when the taken price is worse. Nothing here places a bet.",
              "Quote observations do not establish that a bet was available or placed. Receipt-confirmed wagers, "
              "execution attempts, availability and actual-price slippage belong to the private wager_reconciliation report."]
    if not df.empty:
        tier = _tier(df)
        lines += ["", "Raw observations by week (not independent decisions):", "",
                  "| Week | Tier | Count | Beat exact close | EV at sharp close |", "|---|---|---:|---:|---:|"]
        for (season, week, t_), grp in df.assign(tier=tier).groupby(["season", "week", "tier"]):
            c = pd.to_numeric(grp.clv, errors="coerce").dropna(); ev_ = pd.to_numeric(grp.ev_at_sharp_close, errors="coerce").dropna()
            lines.append(f"| {season}-{week} | {t_} | {len(grp)} | {pct((c > 1e-9).mean() if len(c) else None)} | {signed(ev_.mean() if len(ev_) else None)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        df = grade(load(conn))
    summary = summarize(df)
    doc = dict(built_at=datetime.now(timezone.utc).isoformat(), pass_bar=PASS, summary=summary)
    atomic_json(ROOT / "reports" / "nfl_sharp_alerts_latest.json", doc)
    (ROOT / "reports" / "nfl_sharp_alerts_latest.md").write_text(markdown(summary, df), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
