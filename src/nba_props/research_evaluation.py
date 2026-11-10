"""Identical-offer research scoring; never treats historical quotes as executions."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

from nba_props import research_data as data,research_models as models
from nba_props.model import match_name
from model_research import nba_team
from nfl_pipeline import sharp_math
from sharp_pipeline.nba_experiment import clustered_interval

STAT_MAP={'points':'pts','rebounds':'reb','assists':'ast','threes':'fg3m'}


def prop_quotes(conn):
    return pd.DataFrame(data.rows(conn,"""SELECT id quote_id,event_id,player_name,home_team,away_team,stat,line,
        over_price,under_price,bookmaker_key book,fetched_at_utc,updated_at_utc,commence_time_utc::timestamptz commence_time_utc
        FROM odds.nba_player_prop_lines WHERE fetched_at_utc<commence_time_utc::timestamptz
        AND updated_at_utc<=fetched_at_utc AND fetched_at_utc>=commence_time_utc::timestamptz-interval '30 hours'
        AND line IS NOT NULL AND over_price IS NOT NULL AND under_price IS NOT NULL"""))


def normalize_quotes(q):
    q=q.copy()
    for c in ('fetched_at_utc','updated_at_utc','commence_time_utc'):q[c]=pd.to_datetime(q[c],utc=True)
    q=q[(q.fetched_at_utc<q.commence_time_utc)&(q.updated_at_utc<=q.fetched_at_utc)
         &(q.fetched_at_utc>=q.commence_time_utc-pd.Timedelta(hours=30))]
    q['game_date']=q.commence_time_utc.dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    for c in ('home_team','away_team'):q[c]=q[c].map(nba_team)
    return q[q.home_team.notna()&q.away_team.notna()]


def match_props(predicted,quotes):
    if quotes.empty:return pd.DataFrame()
    q=normalize_quotes(quotes);q['name_key']=q.player_name.map(match_name);q['stat']=q.stat.map(STAT_MAP)
    q['line']=pd.to_numeric(q.line,errors='coerce')
    q=q[q.stat.notna()&q.line.ge(0)&q.line.mod(1).eq(.5)]
    keys=['book','game_date','name_key','home_team','away_team','stat']
    last=q.groupby(keys).fetched_at_utc.transform('max');q=q[q.fetched_at_utc.eq(last)]
    q=q.drop_duplicates(keys+['event_id','line','over_price','under_price'])
    q=q[~q.duplicated(keys,keep=False)]
    p=predicted.copy();p['name_key']=p.player_name.map(match_name)
    p['home_team']=np.where(p.home,p.team,p.opponent);p['away_team']=np.where(p.home,p.opponent,p.team)
    p=p[~p.duplicated(['game_date','name_key','home_team','away_team'],keep=False)]
    parts=[]
    for stat in models.STATS:
        f=p[['game_id','game_date','name_key','home_team','away_team',stat,'pred_'+stat]].rename(columns={stat:'actual','pred_'+stat:'mu'})
        parts.append(f.assign(stat=stat))
    m=pd.concat(parts).merge(q,on=['game_date','name_key','home_team','away_team','stat'],validate='one_to_many')
    for c in ('over_price','under_price'):m[c]=pd.to_numeric(m[c],errors='coerce')
    m=m[m.over_price.abs().ge(100)&m.under_price.abs().ge(100)&m.mu.notna()]
    m['p_market']=[sharp_math.no_vig_over(o,u) for o,u in zip(m.over_price,m.under_price)]
    m['won_over']=m.actual>m.line
    return m


def game_quotes(conn,games):
    q=pd.DataFrame(data.rows(conn,"""SELECT * FROM odds.nba_game_lines WHERE fetched_at_utc<commence_time_utc::timestamptz
        AND updated_at_utc<=fetched_at_utc AND fetched_at_utc>=commence_time_utc::timestamptz-interval '30 hours'"""))
    if q.empty:return q
    q=normalize_quotes(q).rename(columns={'bookmaker_key':'book'})
    keys=['book','game_date','home_team','away_team']
    last=q.groupby(keys).fetched_at_utc.transform('max');q=q[q.fetched_at_utc.eq(last)]
    compare=['event_id','spread_home_points','spread_home_price','spread_away_price','total_points','total_over_price','total_under_price']
    q=q.drop_duplicates(keys+compare);q=q[~q.duplicated(keys,keep=False)]
    g=games.copy();g=g[~g.duplicated(['game_date','home_team','away_team'],keep=False)]
    m=g.merge(q,on=['game_date','home_team','away_team'],validate='one_to_many')
    return m


def probability_metrics(frame):
    if frame.empty:return dict(rows=0,status='insufficient_data')
    f=frame.copy();f['day']=f.game_date.dt.date
    y=f.won_over.to_numpy(float);out={}
    for name in ('market','raw','calibrated'):
        p=np.clip(f['p_'+name].to_numpy(float),1e-6,1-1e-6)
        bins=[]
        for low in np.arange(0,1,.1):
            mask=(p>=low)&(p<low+.1)
            if mask.any():bins.append(dict(low=round(float(low),1),n=int(mask.sum()),predicted=float(p[mask].mean()),observed=float(y[mask].mean())))
        out[name]=dict(brier=float(np.mean((p-y)**2)),log_loss=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),calibration_bins=bins)
    f['gain']=(f.p_market-y)**2-(f.p_calibrated-y)**2
    out['paired_market_gain']={k:clustered_interval(f,'gain',c) for k,c in (('game','game_id'),('day','day'))}
    return dict(rows=len(f),games=int(f.game_id.nunique()),metrics=out,execution_verified=False,betting_authorized=False)


def evaluate(predicted,game_predictions,quotes,game_quotes_frame,artifact):
    out=dict(stat_forecasts=models.forecast_metrics(predicted),props={},games={},historical_only=True,betting_authorized=False)
    matched=match_props(predicted,quotes)
    for (book,stat),f in (matched.groupby(['book','stat']) if not matched.empty else []):
        f=f.copy();f['p_raw']=models.over(f.mu,f.line,artifact['dispersion'][stat])
        f['p_calibrated']=models.calibrate(f.p_raw,artifact['calibration'][stat])
        out['props'].setdefault(book,{})[stat]=probability_metrics(f)
    for stat in ('margin','total'):
        a=game_predictions[stat];mu=game_predictions['pred_'+stat]-artifact['game_error'][stat]['bias']
        error=mu-a
        out['games'][stat]=dict(rows=len(error),mae=float(error.abs().mean()),bias=float(error.mean()),by_book={})
        if game_quotes_frame.empty:continue
        for book,f in game_quotes_frame.groupby('book'):
            line=-pd.to_numeric(f.spread_home_points,errors='coerce') if stat=='margin' else pd.to_numeric(f.total_points,errors='coerce')
            good=line.notna()&f[stat].notna()
            out['games'][stat]['by_book'][book]=dict(rows=int(good.sum()),
                model_mae=float((f.loc[good,'pred_'+stat]-artifact['game_error'][stat]['bias']-f.loc[good,stat]).abs().mean()),
                market_mae=float((line[good]-f.loc[good,stat]).abs().mean()))
    return out
