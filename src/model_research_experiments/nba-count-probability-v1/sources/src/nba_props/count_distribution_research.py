"""Offline integer-count challenger using saved walk-forward NBA forecasts.

Does not change any live model, forecast, offer selection, or registration.
Only genuinely pregame database observations are used; inferred historical
snapshot times are excluded. Quote availability at betting time is unknown.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

from model_research import ROOT, clean, nba_team
from nba_props import model
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline import sharp_math
from sharp_pipeline.nba_experiment import clustered_interval

SPEC = dict(name='nba-integer-count-distribution-v1',
    change='negative binomial at variance=max(mu,a*mu+b*mu^2); Poisson at variance=mu',
    forecasts='existing saved monthly walk-forward means and dispersion; no refit or tuning',
    quotes='last actual pregame database observation within 30 hours; inferred snapshots excluded',
    book='draftkings', markets=['pts', 'reb', 'ast'], lines='nonnegative half-integers only',
    evaluation_start='2025-10-01', evaluation_end_exclusive='2026-07-01',
    selection='best side by own executable-price EV, minimum 3%; one per player-game-stat',
    primary='paired Brier improvement over frozen raw model, game/day clustered 95% intervals',
    review='positive Brier gain lower bounds for game and day; not betting approval',
    caveat='historical development analysis, not independent prospective or FanDuel execution evidence')


def count_over(mu, line, a, b):
    """An integer-valued distribution with the supplied mean and overdispersion."""
    mu, line, a, b = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in (mu, line, a, b)])
    if (not all(np.isfinite(x).all() for x in (mu, line, a, b)) or (mu < 0).any()
            or (a <= 0).any() or (b < 0).any()):
        raise ValueError('Invalid count distribution parameters')
    variance = np.maximum(mu, a*mu+b*mu**2)
    p = stats.poisson.sf(np.floor(line), mu)
    over = (mu > 0) & (variance-mu > 1e-9)
    size = np.divide(mu**2, variance-mu, out=np.ones_like(mu), where=over)
    prob = np.divide(size, size+mu, out=np.ones_like(mu), where=over)
    return np.where(over, stats.nbinom.sf(np.floor(line), size, prob), p)


def load_lines(conn):
    return pd.read_sql("""SELECT id AS quote_id, event_id, home_team, away_team, player_name,
        stat,line,over_price,under_price,commence_time_utc::timestamptz AS commence,
        fetched_at_utc,updated_at_utc FROM odds.nba_player_prop_lines
        WHERE bookmaker_key='draftkings' AND line IS NOT NULL AND over_price IS NOT NULL
        AND under_price IS NOT NULL AND fetched_at_utc<commence_time_utc::timestamptz
        AND fetched_at_utc>=commence_time_utc::timestamptz-interval '30 hours'
        AND updated_at_utc<=fetched_at_utc""", conn)


def matched_rows(predictions, quotes, identities):
    p = predictions.copy()
    p['game_id'] = p.game_id.astype(str).str.zfill(10)
    p['game_date'] = pd.to_datetime(p.game_date).dt.normalize()
    p['name_key'] = p.player_name.map(model.match_name)
    p = p[~p.duplicated(['game_date', 'name_key'], keep=False)]
    p = p.merge(identities, on=['game_id', 'player_id'], how='inner', validate='one_to_one')
    p['home_team'] = np.where(p.home, p.team, p.opponent)
    p['away_team'] = np.where(p.home, p.opponent, p.team)
    q = quotes.copy()
    for key in ('commence', 'fetched_at_utc', 'updated_at_utc'):
        q[key] = pd.to_datetime(q[key], utc=True)
    q = q[(q.fetched_at_utc < q.commence) & (q.updated_at_utc <= q.fetched_at_utc)
          & ((q.commence-q.fetched_at_utc) <= pd.Timedelta(hours=30))]
    q['game_date'] = q.commence.dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    q['name_key'] = q.player_name.map(model.match_name)
    q['home_team'], q['away_team'] = q.home_team.map(nba_team), q.away_team.map(nba_team)
    q['stat'] = q.stat.map(model.LINE_STAT)
    q['line'] = q.line.astype(float)
    q = q[q.stat.notna() & q.line.ge(0) & ((q.line % 1-.5).abs() < 1e-9)
          & q.home_team.notna() & q.away_team.notna()]
    keys = ['game_date', 'name_key', 'home_team', 'away_team', 'stat']
    # Equal-time conflicting rows stay excluded, not resolved by favorable price.
    q = q.sort_values(['fetched_at_utc', 'quote_id'])
    latest = q.groupby(keys).fetched_at_utc.transform('max')
    q = q[q.fetched_at_utc.eq(latest)].drop_duplicates(keys+['event_id', 'line', 'over_price', 'under_price'])
    q = q[~q.duplicated(keys, keep=False)]
    parts = []
    for stat in SPEC['markets']:
        cols = ['game_id', 'player_id', 'game_date', 'name_key', 'home_team', 'away_team',
                stat, f'pred_{stat}', f'disp_a_{stat}', f'disp_b_{stat}']
        part = p[cols].rename(columns={stat:'actual', f'pred_{stat}':'mu', f'disp_a_{stat}':'a', f'disp_b_{stat}':'b'})
        parts.append(part.assign(stat=stat))
    m = pd.concat(parts).merge(q.drop(columns=['player_name']), on=keys, validate='one_to_one')
    m['p_market'] = [sharp_math.no_vig_over(o, u) for o,u in zip(m.over_price, m.under_price)]
    m = m[m.p_market.notna()].copy()
    for key in ('actual','mu','a','b','line','over_price','under_price','p_market'):
        m[key] = pd.to_numeric(m[key], errors='coerce')
    m = m[np.isfinite(m[['actual','mu','a','b','line','over_price','under_price','p_market']]).all(axis=1)]
    m = m[(m.actual >= 0) & (m.actual % 1 == 0) & (m.mu >= 0) & (m.a > 0) & (m.b >= 0)]
    m['won_over'] = m.actual > m.line
    m['p_frozen'] = np.nan
    for (a,b), g in m.groupby(['a','b']):
        m.loc[g.index, 'p_frozen'] = model.p_over(g.mu.to_numpy(), g.line.to_numpy(), a,b)
    m['p_count'] = count_over(m.mu, m.line, m.a, m.b)
    return m


def selection(frame, probability):
    p = frame[probability].to_numpy(float)
    payoff = lambda price: np.where(price > 0, price/100, 100/np.abs(price))
    over_pay = payoff(frame.over_price.to_numpy(float))
    under_pay = payoff(frame.under_price.to_numpy(float))
    over_ev, under_ev = p*(over_pay+1)-1, (1-p)*(under_pay+1)-1
    side_over = over_ev >= under_ev
    selected = np.maximum(over_ev, under_ev) >= .03
    won = np.where(side_over, frame.won_over.to_numpy(), ~frame.won_over.to_numpy())
    profit = np.where(won, np.where(side_over, over_pay, under_pay), -1.)
    f = frame.loc[selected, ['game_id', 'game_date']].copy()
    f['paper_profit'] = profit[selected]
    return dict(selections=int(selected.sum()), paper_roi=float(profit[selected].mean()) if selected.any() else None,
                paper_profit=float(profit[selected].sum()),
                roi_by_game=clustered_interval(f, 'paper_profit', 'game_id'),
                roi_by_day=clustered_interval(f, 'paper_profit', 'game_date'))


def evaluate(frame):
    if frame.empty:
        return dict(rows=0, decision='insufficient_data')
    f = frame.copy()
    y = f.won_over.to_numpy(float)
    metrics = {}
    for label in ('market', 'frozen', 'count'):
        p = f['p_'+label].to_numpy(float)
        error = (p-y)**2
        metrics[label] = dict(brier=float(error.mean()),
            log_loss=float(-np.mean(y*np.log(np.clip(p,1e-12,1))+(1-y)*np.log(np.clip(1-p,1e-12,1)))),
            predicted_over=float(p.mean()), actual_over=float(y.mean()),
            selection=selection(f,'p_'+label))
    f['brier_gain'] = (f.p_frozen-y)**2 - (f.p_count-y)**2
    f['market_gain'] = (f.p_market-y)**2 - (f.p_count-y)**2
    intervals = {f'{kind}_by_{label}':clustered_interval(f,kind,column)
                 for kind in ('brier_gain','market_gain')
                 for label,column in (('game','game_id'),('day','game_date'))}
    passes = all((intervals[f'brier_gain_by_{c}']['lower_95'] or -1) > 0 for c in ('game','day'))
    return dict(rows=len(f), games=int(f.game_id.nunique()), dates=int(f.game_date.nunique()),
                metrics=metrics, intervals=intervals,
                decision='consider_prospective_comparison_only' if passes else 'do_not_promote',
                betting_authorized=False)


def markdown(doc):
    lines = ['# NBA integer-count probability research', '',
        'Development analysis on saved walk-forward forecasts and actually recorded pregame DraftKings quotes.',
        'No live model or frozen strategy was changed. These are not FanDuel execution results.', '',
        '| Period / stat | Offers | Frozen Brier | Count Brier | Market Brier | Count paper ROI | Decision |',
        '|---|---:|---:|---:|---:|---:|---|']
    for period, parts in doc['evaluation'].items():
        for stat, r in parts.items():
            if not r['rows']:
                continue
            m = r['metrics']; roi=m['count']['selection']['paper_roi']
            roi = 'no selections' if roi is None else f'{roi:+.2%}'
            lines.append(f"| {period} / {stat} | {r['rows']} | {m['frozen']['brier']:.5f} | "
                         f"{m['count']['brier']:.5f} | {m['market']['brier']:.5f} | {roi} | {r['decision']} |")
    lines += ['', 'Lower Brier is better. Paired game/day uncertainty intervals are in the JSON.',
              'Paper ROI assumes the saved offer was available, which was not checked. Costs are not included.',
              'The saved forecasts and historical outcomes are development data. No prospective efficacy is claimed.',
              'No threshold search, model refit, or live registration was performed.']
    return '\n'.join(lines)+'\n'


def run():
    predictions = ROOT/'reports/nba_props_model_predictions.csv.gz'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = ROOT/'data/nba_props/count_distribution_research'/stamp
    spec = dict(SPEC, source_sha256=hashlib.sha256(__file_bytes()).hexdigest(),
                predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest())
    # Save the fixed recipe before inspecting any evaluation results.
    atomic_json(folder/'specification.json',spec)
    preds = pd.read_csv(predictions,dtype={'game_id':str})
    with psycopg2.connect(PG_DSN,connect_timeout=10) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:cur.execute("SET statement_timeout='60s'")
        quotes=load_lines(conn)
        identities=pd.read_sql('SELECT game_id,player_id,team,opponent,home FROM nbaprops.player_games',conn)
    frame=matched_rows(preds,quotes,identities)
    frame.to_pickle(folder/'matched_offers.pkl')
    evaluation={}
    for label, part in [('earlier_development',frame[frame.game_date<pd.Timestamp(SPEC['evaluation_start'])]),
                        ('2025-26_later_period',frame[(frame.game_date>=pd.Timestamp(SPEC['evaluation_start']))
                                                   &(frame.game_date<pd.Timestamp(SPEC['evaluation_end_exclusive']))])]:
        evaluation[label]={'all':evaluate(part)}
        evaluation[label].update({stat:evaluate(g) for stat,g in part.groupby('stat')})
    doc=clean(dict(built_at=datetime.now(timezone.utc),specification=spec,matched_rows=len(frame),
                   artifact_directory=str(folder),evaluation=evaluation,betting_authorized=False))
    atomic_json(folder/'results.json',doc)
    atomic_json(ROOT/'reports/nba_count_distribution_latest.json',doc)
    (ROOT/'reports/nba_count_distribution_latest.md').write_text(markdown(doc),encoding='utf-8')
    print(json.dumps({k:{s:dict(rows=r['rows'],decision=r['decision']) for s,r in v.items()}
                      for k,v in evaluation.items()},indent=2))
    return doc


def __file_bytes():
    from pathlib import Path
    return Path(__file__).read_bytes()


if __name__=='__main__':run()
