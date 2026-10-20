"""Prospective FanDuel price-gap evidence, separate from inferred movement and actual wagers.

Only fresh, source-timestamped immutable observations with exact same-book closes
enter primary evidence. Missing closes remain unknown. Review readiness never
authorizes cash; a sharp no-vig quote is a reference estimate, not known truth.
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
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.sharp_watch import (PING_WINDOW_BY_STAT, PING_WINDOW_MINUTES, SHARP_BOOKS,
                                     QUOTE_CONTRACT, quote_error, source_time, EXPERIMENT_NAME)
from nfl_pipeline.sharp_watch import _digest, MAX_SOURCE_SKEW_SECONDS
from nfl_pipeline.context_contract import safe_time
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[3]
PASS = dict(min_alerts=100, min_weeks=3, min_beat_close=0.55, min_sharp_close_ev=0.01,
            min_exact_close_coverage=.90, min_sharp_close_coverage=.90, alpha=.05)
MAX_CLOSE_AGE_MINUTES = 20


def load(conn) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='120s'")
        cur.execute("SELECT to_regclass('bets.nfl_sharp_alerts')")
        if cur.fetchone()["to_regclass"] is None:
            return pd.DataFrame()
        cur.execute("""
            SELECT a.*, g.season, g.week, g.status, g.home_score, g.away_score,
                   gl.actual, gl.participation_verified
            FROM bets.nfl_sharp_alerts a
            JOIN raw.nfl_games g ON g.game_id = a.game_id
            LEFT JOIN LATERAL (
              SELECT CASE a.stat WHEN 'receiving_yards' THEN l.receiving_yards WHEN 'rushing_yards' THEN l.rushing_yards
                       WHEN 'passing_yards' THEN l.passing_yards WHEN 'receptions' THEN l.receptions
                       WHEN 'passing_tds' THEN l.passing_tds END::float AS actual,
                     (COALESCE(l.offense_snaps,0)>0
                      OR COALESCE(l.pass_attempts,0)+COALESCE(l.carries,0)+COALESCE(l.targets,0)>0
                      OR (l.raw_json #> '{receiving_result_repair,verified_game_participation}')='true'::jsonb)
                       AS participation_verified
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
    """Only immutable, source-timestamped, same-book/exact-line closes are primary evidence."""
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
    exact_clv, exact_ev = [], []
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
    df['clv'], df['ev_at_sharp_close'] = exact_clv, exact_ev
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
    return dict(alerts=int(len(df)), weeks=int(weeks), with_fd_close=int(len(clv)), beat_fd_close=float((clv > 1e-9).mean()) if len(clv) else None,
                exact_close_coverage=float(len(clv)/len(df)), sharp_close_coverage=float(len(evc)/len(df)),
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
    if len(means) < PASS['min_weeks']:
        return None
    return float(means.mean() - t.ppf(1-PASS['alpha'], len(means)-1)*means.std(ddof=1)/np.sqrt(len(means)))


def fixed_decisions(df):
    """One first decision per book/player-game/stat; never choose after seeing outcomes."""
    if df.empty:
        return df
    frame = df.copy()
    for field in ('book','game_id','player_name_norm','stat'):
        if field not in frame:
            frame[field] = 'unknown'
        frame[field] = frame[field].fillna('game' if field == 'player_name_norm' else 'unknown')
    frame['_time'] = pd.to_datetime(frame.get('decision_at', frame.get('alerted_at')), utc=True, errors='coerce')
    if 'alert_id' not in frame:
        frame['alert_id'] = np.arange(len(frame))
    frame['alert_id'] = frame['alert_id'].astype(str)
    return frame.sort_values(['_time','alert_id'], kind='stable').drop_duplicates(
        ['book','game_id','player_name_norm','stat'], keep='first').drop(columns='_time')


def _stake(df: pd.DataFrame) -> pd.Series:
    if "stake" not in df:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df["stake"], errors="coerce").fillna(0.0)


def _tier(df: pd.DataFrame) -> pd.Series:
    """Tier as the strategy is now defined: a >= +3% gap seen more than PING_WINDOW_MINUTES before kickoff is
    'early' (pings sent before the window existed are relabelled here; stored rows are not rewritten)."""
    tier = df["tier"].fillna("alert") if "tier" in df else pd.Series("alert", index=df.index)
    if {"commence_time_utc", "alerted_at"} <= set(df.columns):
        observed = pd.to_datetime(df.alerted_at, utc=True)
        if 'decision_at' in df:
            observed = pd.to_datetime(df.decision_at, utc=True).fillna(observed)
        lead = (pd.to_datetime(df.commence_time_utc, utc=True) - observed).dt.total_seconds() / 60
        window = df.stat.map(PING_WINDOW_BY_STAT).fillna(PING_WINDOW_MINUTES) if "stat" in df else PING_WINDOW_MINUTES
        tier = tier.where(~((tier == "alert") & (lead > window)), "early")
    return tier


def summarize(df: pd.DataFrame, experiment=None) -> dict:
    """Pass/fail is judged on pinged alerts (EV >= +3%). Logged gaps (+1% to +3%, no ping) are reported
    beside them: they show sooner whether FanDuel-vs-sharp gaps predict the close at all."""
    if df.empty:
        return dict(alerts=0, verdict="no graded alerts yet", logged=dict(alerts=0, weeks=0))
    if experiment is None:
        from nfl_pipeline.experiment_registry import load_experiment
        experiment = load_experiment(EXPERIMENT_NAME)
    tier = _tier(df)
    books = df["book"].fillna("fanduel") if "book" in df else pd.Series("fanduel", index=df.index)
    eligible = books.eq('fanduel') & df.provenance_valid.fillna(False)
    eligible &= df.get('evidence_kind', pd.Series('', index=df.index)).eq('decision_observation')
    eligible &= df.reference_method.eq('exact')
    eligible &= df.stat.eq('receiving_yards')
    eligible &= df.experiment_sha256.eq((experiment or {}).get('sha256', '__unregistered__'))
    eligible &= df.game_id.notna() if 'game_id' in df else False
    primary = fixed_decisions(df[eligible & tier.eq('alert')])
    out = _stats(primary)
    out['primary_book'] = 'fanduel'
    out['experiment_sha256'] = (experiment or {}).get('sha256')
    ledger = df.get('evidence_kind', pd.Series('alert_ledger', index=df.index)).eq('alert_ledger')
    out['raw_alerts'] = int((tier.eq('alert') & ledger).sum())
    out['captured_decisions'] = int((tier.eq('alert') & ~ledger).sum())
    out['excluded_alerts'] = int((tier.eq('alert') & ~eligible).sum())
    out['repeated_alerts_removed'] = int((eligible & tier.eq('alert')).sum()) - len(primary)
    out["logged"] = _stats(fixed_decisions(df[eligible & tier.eq("logged")]))
    out["early"] = _stats(fixed_decisions(df[eligible & tier.eq("early")]))
    # Bets actually placed, confirmed by hand with sharp_bets --confirm. This is the real money record.
    placed = df[df.placed_at.notna()] if "placed_at" in df else df.iloc[:0]
    out["placed"] = _stats(placed)
    out["by_book"] = {str(book): _stats(fixed_decisions(g[_tier(g) == "alert"])) for book, g in df[ledger].groupby(books[ledger])}
    diagnostic = fixed_decisions(df[tier.eq('alert') & ~ledger])
    out['inferred_close_diagnostics'] = {
        key: dict(rows=int(pd.to_numeric(diagnostic[key], errors='coerce').notna().sum()),
                  mean=float(pd.to_numeric(diagnostic[key], errors='coerce').mean())
                      if pd.to_numeric(diagnostic[key], errors='coerce').notna().any() else None)
        for key in ('inferred_clv', 'inferred_ev_at_sharp_close')}
    # Does our own projection add anything on top of the price edge? Priced alerts only.
    if "model_agrees" in df:
        priced = primary[primary.model_agrees.notna()]
        out["model"] = {("agrees" if bool(k) else "disagrees"): _stats(g)
                        for k, g in priced.groupby(priced.model_agrees)}
    weeks = out.get("weeks", 0)
    checks = dict(enough_alerts=out["alerts"] >= PASS["min_alerts"], enough_weeks=weeks >= PASS["min_weeks"],
                  beats_fd_close=(out.get("beat_fd_close") or 0) >= PASS["min_beat_close"],
                  positive_at_sharp_close=(out.get("mean_ev_at_sharp_close") or -1) >= PASS["min_sharp_close_ev"],
                  exact_close_coverage=(out.get('exact_close_coverage') or 0) >= PASS['min_exact_close_coverage'],
                  sharp_close_coverage=(out.get('sharp_close_coverage') or 0) >= PASS['min_sharp_close_coverage'],
                  weekly_clv_confirmed=(out.get('mean_clv_lower') or 0) > 0,
                  weekly_sharp_ev_confirmed=(out.get('sharp_close_ev_lower') or 0) > 0,
                  registered=bool(experiment))
    out["checks"] = checks
    out["verdict"] = ("evidence_review_ready; no cash authorization" if all(checks.values())
                      else "keep collecting" if not (checks["enough_alerts"] and checks["enough_weeks"])
                      else "evidence requirements not met; research only")
    return out


def markdown(summary: dict, df: pd.DataFrame) -> str:
    pct = lambda v: "-" if v is None else f"{v:.1%}"
    signed = lambda v: "-" if v is None else f"{v:+.2%}"
    a, e, g = summary, summary.get("early") or {}, summary.get("logged") or {}
    row = lambda label, key, fmt, bar: f"| {label} | {fmt(a.get(key))} | {fmt(e.get(key))} | {fmt(g.get(key))} | {bar} |"
    lines = ["# NFL Sharp-Edge Alerts", "", f"Verdict: **{summary['verdict']}**", "",
             "Primary: registered FanDuel receiving-yard decisions with an exact Pinnacle reference. "
             "First eligible observation per book/player-game/stat; later prices do not add independent decisions.", "",
             f"Alerts graded: {a.get('alerts', 0)} over {a.get('weeks', 0)} week(s) "
             f"(pass bar: {PASS['min_alerts']} over {PASS['min_weeks']}). Early gaps graded: {e.get('alerts', 0)}. "
             f"Logged gaps graded: {g.get('alerts', 0)}.", "",
             f"| Evidence (most reliable first) | Alerts (>= +3%, last {PING_WINDOW_MINUTES} min, pinged) | Early (>= +3%, earlier) "
             "| Logged (+1% to +3%) | Pass bar (alerts) |",
             "|---|---:|---:|---:|---:|",
             row("Exact execution close coverage", "exact_close_coverage", pct, ">= 90%"),
             row("Exact reference close coverage", "sharp_close_coverage", pct, ">= 90%"),
             row("Mean EV at exact reference close", "mean_ev_at_sharp_close", signed, f">= {PASS['min_sharp_close_ev']:+.0%}"),
             row("Share positive at the sharp close", "positive_at_sharp_close", pct, "-"),
             row("Beat exact FanDuel close", "beat_fd_close", pct, f">= {PASS['min_beat_close']:.0%}"),
             row("Mean FanDuel CLV (no-vig prob)", "mean_fd_clv", signed, "> 0"),
             row("EV claimed when seen", "mean_alert_ev", signed, "-"),
             row("Simulated flat ROI", "roi", signed, "descriptive"),
             row("Settled", "settled", lambda v: "-" if v is None else str(v), "-")]
    placed = summary.get("placed") or {}
    if placed.get("alerts"):
        slip = placed.get("mean_slippage")
        lines += ["", f"**Legacy manually confirmed alerts: {placed['alerts']} for ${placed.get('staked', 0):.2f} staked; "
                      f"realized ${placed.get('realized_profit', 0):+.2f} on {placed.get('settled', 0)} settled "
                      f"(EV at sharp close {signed(placed.get('mean_ev_at_sharp_close'))}"
                      + (f", slippage {signed(slip)} vs the alerted price" if slip is not None else "") + ").**"]
    mv = summary.get("model") or {}
    if mv:
        lines += ["", "| Our projection | Alerts | EV at sharp close | Beat close | ROI |", "|---|---:|---:|---:|---:|"]
        for name in ("agrees", "disagrees"):
            st = mv.get(name) or {}
            lines.append(f"| model {name} with the price edge | {st.get('alerts', 0)} | "
                         f"{signed(st.get('mean_ev_at_sharp_close'))} | {pct(st.get('beat_fd_close'))} | {signed(st.get('roi'))} |")
        lines.append("")
        lines.append("_Model agreement is descriptive; it does not change the frozen selection rule._")
    by_book = summary.get("by_book") or {}
    if len(by_book) > 1:
        lines += ["", "Per-book historical diagnostics, excluded from pooled approval:", "",
                  "| Book | Decisions | Exact reference-close EV | Beat exact close |", "|---|---:|---:|---:|"]
        for book, st in sorted(by_book.items()):
            lines.append(f"| {book} | {st.get('alerts', 0)} | {signed(st.get('mean_ev_at_sharp_close'))} | {pct(st.get('beat_fd_close'))} |")
    inferred = summary.get('inferred_close_diagnostics') or {}
    lines += ["", "Model-inferred movement and legacy closes (diagnostic only):"]
    for name, data in inferred.items():
        lines.append(f"- {name}: {data['rows']} rows; mean {signed(data['mean'])}.")
    lines += ["", f"Legacy alerts: {summary.get('raw_alerts', 0)}; captured candidate decisions: {summary.get('captured_decisions', 0)}; "
              f"excluded from primary: {summary.get('excluded_alerts', 0)}; "
              f"repeated eligible alerts removed: {summary.get('repeated_alerts_removed', 0)}.",
              "Missing exact closes remain unknown. Weekly confidence bounds use equal week weights. "
              "No cash authorization is granted by this report. Slippage is positive when the taken price is worse.",
              "Quote observations do not establish that a bet was available or placed. Receipt-confirmed wagers, "
              "execution attempts, availability and actual-price slippage belong to the private wager_reconciliation report."]
    if not df.empty:
        tier = _tier(df)
        lines += ["", "Historical raw observations by week (not independent decisions):", "",
                  "| Week | Tier | Count | Beat exact close | Exact reference-close EV |", "|---|---|---:|---:|---:|"]
        for (season, week, t), grp in df.assign(tier=tier).groupby(["season", "week", "tier"]):
            c = pd.to_numeric(grp.clv, errors="coerce").dropna(); e = pd.to_numeric(grp.ev_at_sharp_close, errors="coerce").dropna()
            lines.append(f"| {season}-{week} | {t} | {len(grp)} | {pct((c > 1e-9).mean() if len(c) else None)} | {signed(e.mean() if len(e) else None)} |")
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
