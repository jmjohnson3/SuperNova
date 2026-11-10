"""Chronological workload, shooting-opportunity and probability challengers."""
from __future__ import annotations
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy import stats,optimize,special
import lightgbm as lgb

from nba_props import research_data as data

SPEC=dict(name='nba-workload-calibration-v1',result_lag_days=2,
    first_oof='2024-10-01',dispersion_end='2025-02-01',calibration_end='2025-07-01',
    evaluation_start='2025-10-01',evaluation_end='2026-07-01',
    historical_status='reconstructed development evidence; outcome revisions and historical retrieval times are not proven',
    prospective_status='untouched period starts after registration; exact artifacts and rules pinned',
    calibration='stat distribution calibration on fixed integer thresholds; no sportsbook fitting',min_calibration_rows=200,
    calibration_max_threshold={'pts':60,'reb':25,'ast':20,'fg3m':12},
    selection='same FanDuel offered side; calibrated probability EV>=3%, verified active context, no uncertain availability',
    shooting_prior_attempts=100,minimum_player_games=5,automatic_betting_authorization=False)
PARAMS=dict(n_estimators=150,num_leaves=15,learning_rate=.04,min_child_samples=80,
    reg_lambda=5.,n_jobs=2,verbosity=-1,random_state=630)
STATS=('pts','reb','ast','fg3m')
FEATURES=['home','rest_days','b2b','team_changed','games_before','prior_starter','min3','min10','min_sd',
          'pts_rate','reb_rate','ast_rate','fg3a_rate','fga_rate','fta_rate','fg3_pct',
          'team_pace','opponent_pace','team_eff','opponent_def_eff','team_margin_sd']
GAME_FEATURES=['home_pace','away_pace','home_eff','away_eff','home_def_eff','away_def_eff','home_rest','away_rest']


def attach_past_roles(players,conn,asof):
    roles=data.rows(conn,"""SELECT DISTINCT ON(game_id,player_id) game_id,player_id,starter FROM nbaprops.lineup_snapshots
        WHERE captured_at_utc<=%s AND lineup_status='Confirmed' ORDER BY game_id,player_id,captured_at_utc DESC""",(asof,))
    f=players.copy();mapping={(r['game_id'],r['player_id']):float(r['starter']) for r in roles}
    f['actual_starter']=[mapping.get((str(g),p),np.nan) for g,p in zip(f.game_id,f.player_id)]
    return f


def states(players,teams):
    """Post-game states: joined only to targets at least two dates later."""
    p=players.copy();p['game_date']=pd.to_datetime(p.game_date)
    p=p[p.minutes.gt(0)].sort_values(['player_id','game_date','game_id']).reset_index(drop=True)
    if 'actual_starter' not in p:p['actual_starter']=np.nan
    g=p.groupby('player_id',sort=False)
    s=p[['game_id','player_id','game_date','team','player_name']].copy().rename(columns={'game_date':'history_date','team':'history_team'})
    s['games_before']=g.cumcount()+1
    s['min3']=g.minutes.transform(lambda x:x.ewm(halflife=3).mean())
    s['min10']=g.minutes.transform(lambda x:x.ewm(halflife=10).mean())
    s['min_sd']=g.minutes.transform(lambda x:x.rolling(15,min_periods=3).std())
    s['prior_starter']=g.actual_starter.transform(lambda x:x.ewm(halflife=5).mean()).fillna(.5)
    for stat in ('pts','reb','ast','fg3a','fga','fta'):
        rate=p[stat]/p.minutes.clip(lower=5)
        s[stat+'_rate']=rate.groupby(p.player_id).transform(lambda x:x.ewm(halflife=10).mean())
    makes=g.fg3m.transform(lambda x:x.rolling(40,min_periods=1).sum())
    attempts=g.fg3a.transform(lambda x:x.rolling(40,min_periods=1).sum())
    # Fixed prior chosen before evaluation; do not estimate a prior on future shooting results.
    s['fg3_pct']=(makes+.36*SPEC['shooting_prior_attempts'])/(attempts+SPEC['shooting_prior_attempts'])
    t=teams.copy();t['game_date']=pd.to_datetime(t.game_date)
    t=t.merge(t[['game_id','team','pts']].rename(columns={'team':'opponent','pts':'allowed'}),on=['game_id','opponent'],validate='one_to_one')
    t=t.sort_values(['team','game_date','game_id']).reset_index(drop=True)
    poss=t.fga+.44*t.fta-t.oreb+t.tov
    t['pace_raw']=poss/t.minutes*240
    t['eff_raw']=t.pts/poss.clip(lower=1)
    t['def_eff_raw']=t.allowed/poss.clip(lower=1)
    t['margin_raw']=t.pts-t.allowed
    tg=t.groupby('team',sort=False)
    ts=t[['game_id','team','game_date']].rename(columns={'game_date':'team_history_date'}).copy()
    for key in ('pace','eff','def_eff'):ts[key]=tg[key+'_raw'].transform(lambda x:x.ewm(halflife=12).mean())
    ts['margin_sd']=tg.margin_raw.transform(lambda x:x.rolling(20,min_periods=5).std())
    return s,ts


def attach_team(frame,team_states,key,prefix):
    left=frame.copy();left['_order']=np.arange(len(left))
    right=team_states.rename(columns={'team':key,'team_history_date':prefix+'history_date',
        'pace':prefix+'pace','eff':prefix+'eff','def_eff':prefix+'def_eff','margin_sd':prefix+'margin_sd'}).drop(columns='game_id')
    out=pd.merge_asof(left.sort_values('cutoff_date'),right.sort_values(prefix+'history_date'),
        left_on='cutoff_date',right_on=prefix+'history_date',by=key,direction='backward')
    return out.sort_values('_order').drop(columns='_order').reset_index(drop=True)


def previous_game_date(frame,team_states,key):
    # Schedule dates are known before tip; the two-day lag applies to outcome-derived statistics.
    left=frame[[key,'game_date']].copy();left['_order']=np.arange(len(left))
    left['schedule_cutoff']=left.game_date-pd.Timedelta(days=1)
    right=team_states[['team','team_history_date']].rename(columns={'team':key})
    found=pd.merge_asof(left.sort_values('schedule_cutoff'),right.sort_values('team_history_date'),
        left_on='schedule_cutoff',right_on='team_history_date',by=key,direction='backward')
    return found.sort_values('_order').team_history_date.to_numpy()


def features(targets,player_states,team_states):
    f=targets.copy();f['game_date']=pd.to_datetime(f.game_date)
    f['cutoff_date']=f.game_date-pd.Timedelta(days=SPEC['result_lag_days'])
    right=player_states.drop(columns=['game_id','player_name'])
    f=pd.merge_asof(f.sort_values('cutoff_date'),right.sort_values('history_date'),
        left_on='cutoff_date',right_on='history_date',by='player_id',direction='backward')
    f['home']=f.home.astype(float)
    f['rest_days']=(f.game_date-pd.to_datetime(previous_game_date(f,team_states,'team'))).dt.days.clip(upper=14)
    f['b2b']=f.rest_days.eq(1).astype(float);f['team_changed']=f.team.ne(f.history_team).astype(float)
    f=attach_team(f,team_states,'team','team_');f=attach_team(f,team_states,'opponent','opponent_')
    return f


def game_features(teams,team_states):
    t=teams.copy();t['game_date']=pd.to_datetime(t.game_date)
    f=t[t.home.astype(bool)][['game_id','game_date','team','opponent','pts']].rename(columns={'team':'home_team','opponent':'away_team','pts':'home_points'})
    away=t[~t.home.astype(bool)][['game_id','pts']].rename(columns={'pts':'away_points'})
    f=f.merge(away,on='game_id',how='left',validate='one_to_one')
    f['cutoff_date']=f.game_date-pd.Timedelta(days=2)
    f=attach_team(f,team_states,'home_team','home_');f=attach_team(f,team_states,'away_team','away_')
    for side in ('home','away'):f[side+'_rest']=(f.game_date-pd.to_datetime(previous_game_date(f,team_states,side+'_team'))).dt.days.clip(upper=14)
    f['margin']=f.home_points-f.away_points;f['total']=f.home_points+f.away_points
    return f


def fit(frame,games):
    f=frame[frame.minutes.gt(0)&frame.games_before.ge(SPEC['minimum_player_games'])].copy()
    if len(f)<2000:raise ValueError('Insufficient verified training rows')
    x=f[FEATURES].copy();x['role']=f.actual_starter.fillna(f.prior_starter)
    model={'minutes':lgb.LGBMRegressor(**PARAMS,objective='regression').fit(x,f.minutes),'rates':{},'games':{}}
    eligible=f.minutes.ge(5)
    for stat in ('pts','reb','ast','fg3a'):
        y=(f[stat]/f.minutes.clip(lower=5))
        model['rates'][stat]=lgb.LGBMRegressor(**PARAMS,objective='poisson').fit(x[eligible],y[eligible],sample_weight=f.loc[eligible,'minutes']/30)
    for stat in ('margin','total'):
        g=games[games[stat].notna() & games[GAME_FEATURES].notna().all(axis=1)]
        if len(g)<100:raise ValueError('Insufficient verified game rows')
        model['games'][stat]=lgb.LGBMRegressor(**PARAMS,objective='regression').fit(g[GAME_FEATURES],g[stat])
    return model


def predict(model,frame):
    x=frame[FEATURES].copy();p=frame.prior_starter.fillna(.5).clip(0,1).to_numpy(float)
    if 'confirmed_starter' in frame:
        p=np.where(frame.confirmed_starter.notna(),frame.confirmed_starter.fillna(0).to_numpy(float),p)
    caps=frame.get('minutes_cap',pd.Series(48.,index=frame.index)).fillna(48).clip(0,48).to_numpy(float)
    out=frame.copy();out['role_probability']=p
    mu={s:np.zeros(len(frame)) for s in (*STATS,'fg3a','minutes')}
    for role,weight in ((0,1-p),(1,p)):
        z=x.assign(role=float(role));minutes=np.clip(model['minutes'].predict(z),0,caps)
        out['minutes_if_'+('starter' if role else 'bench')]=minutes
        mu['minutes']+=weight*minutes
        for stat in ('pts','reb','ast','fg3a'):
            value=minutes*np.maximum(model['rates'][stat].predict(z),0)
            mu[stat]+=weight*value
            if stat=='fg3a':mu['fg3m']+=weight*value*frame.fg3_pct.to_numpy()
    for stat,value in mu.items():out['pred_'+stat]=value
    return out


def fit_dispersion(actual,mu):
    good=np.isfinite(actual)&np.isfinite(mu)&(mu>0)
    y=np.asarray(actual)[good];m=np.asarray(mu)[good]
    if len(m)<200:raise ValueError('Insufficient out-of-sample dispersion rows')
    a,b=optimize.nnls(np.column_stack([m,m*m]),(y-m)**2)[0]
    return dict(a=float(max(a,.01)),b=float(max(b,0)),rows=len(m),source='chronologically out-of-sample residuals')


def over(mu,line,dispersion):
    m=np.maximum(np.asarray(mu,float),1e-9);k=np.floor(np.asarray(line,float))
    variance=np.maximum(m,dispersion['a']*m+dispersion['b']*m*m)
    r=m*m/np.maximum(variance-m,1e-9)
    return np.where(variance-m>1e-7,stats.nbinom.sf(k,r,r/(r+m)),stats.poisson.sf(k,m))


def calibrate(p,cal):
    return special.expit(cal['slope']*special.logit(np.clip(p,1e-6,1-1e-6))+cal['intercept'])


def fit_calibration(p,y,weights=None):
    p=np.asarray(p,float);y=np.asarray(y,float)
    if len(y)<SPEC['min_calibration_rows'] or len(np.unique(y))<2:
        return dict(slope=1.,intercept=0.,rows=len(y),status='insufficient_quotes_identity_mapping')
    z=special.logit(np.clip(p,1e-6,1-1e-6))
    def loss(v):
        q=np.clip(special.expit(v[0]*z+v[1]),1e-9,1-1e-9)
        return -np.average(y*np.log(q)+(1-y)*np.log1p(-q),weights=weights)+.002*((v[0]-1)**2+v[1]**2)
    fitted=optimize.minimize(loss,[1.,0.],method='L-BFGS-B',bounds=[(.1,3.),(-3.,3.)])
    if not fitted.success:raise ValueError('Probability calibration failed')
    return dict(slope=float(fitted.x[0]),intercept=float(fitted.x[1]),rows=len(y),status='fit_on_separate_chronological_quotes')


def fit_distribution_calibration(frame,stat,dispersion):
    """Recalibrate a CDF using unseen outcomes, with each player-game weighted equally.

    Thresholds are fixed in SPEC, not chosen from evaluation or favorable bets.
    These are distribution-training pairs, not historical executable offers.
    """
    f=frame[[stat,'pred_'+stat]].dropna()
    if len(f)<SPEC['min_calibration_rows']:
        return dict(slope=1.,intercept=0.,rows=len(f),status='insufficient_games_identity_mapping')
    grid=np.arange(SPEC['calibration_max_threshold'][stat]+1)+.5
    p=over(f['pred_'+stat].to_numpy()[:,None],grid[None,:],dispersion)
    y=f[stat].to_numpy()[:,None]>grid[None,:]
    useful=(p>=.01)&(p<=.99)
    weights=np.broadcast_to(1/np.maximum(useful.sum(axis=1),1)[:,None],p.shape)
    result=fit_calibration(p[useful],y[useful],weights[useful])
    return dict(result,threshold_pairs=result['rows'],rows=len(f),status='fit_on_separate_chronological_distributions',
                synthetic_thresholds_are_not_offers=True)


def probability(artifact,stat,mu,line):
    return float(calibrate(over(mu,line,artifact['dispersion'][stat]),artifact['calibration'][stat]))


def forecast_metrics(frame):
    out={}
    for stat in ('minutes',*STATS,'fg3a'):
        error=frame['pred_'+stat]-frame[stat]
        valid=error.dropna()
        baseline=(frame.min3*frame.get(stat+'_rate',pd.Series(np.nan,index=frame.index))-frame[stat]).dropna() if stat not in ('minutes','fg3m') else None
        if stat=='minutes':baseline=(frame.min3-frame.minutes).dropna()
        if stat=='fg3m':baseline=(frame.min3*frame.fg3a_rate*frame.fg3_pct-frame.fg3m).dropna()
        out[stat]=dict(rows=len(valid),mae=float(valid.abs().mean()),rmse=float(np.sqrt((valid**2).mean())),bias=float(valid.mean()),
                       rolling_baseline_mae=float(baseline.abs().mean()) if baseline is not None and len(baseline) else None)
    return out


def chronological_predictions(frame,games):
    """Each fold trains earlier than its forecast period; no evaluation-driven tuning."""
    parts=[];game_parts=[]
    bounds=['2024-10-01','2025-02-01','2025-07-01','2026-07-01']
    for start,end in zip(bounds,bounds[1:]):
        train=frame[frame.game_date<pd.Timestamp(start)-pd.Timedelta(days=2)]
        gtrain=games[games.game_date<pd.Timestamp(start)-pd.Timedelta(days=2)]
        test=frame[(frame.game_date>=start)&(frame.game_date<end)&frame.minutes.gt(0)&frame.games_before.ge(5)]
        print('Chronological fit',start,'train',len(train),'test',len(test),flush=True)
        fitted=fit(train,gtrain)
        predicted=predict(fitted,test);predicted['trained_before']=start;parts.append(predicted)
        gt=games[(games.game_date>=start)&(games.game_date<end)].copy()
        for stat in ('margin','total'):gt['pred_'+stat]=fitted['games'][stat].predict(gt[GAME_FEATURES])
        gt['trained_before']=start;game_parts.append(gt)
    return pd.concat(parts,ignore_index=True),pd.concat(game_parts,ignore_index=True)
