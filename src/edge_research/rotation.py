"""Joint team rotations and opportunity budgets, priced as a mixture of workloads."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from scipy import optimize,stats,special

from nba_props import research_models as base

SPEC=dict(name='nba-joint-rotation-v1',rotations=16,regulation_minutes=240,maximum_overtimes=2,
    history_days=28,shooting_games=40,two_point_prior=[100,.53],free_throw_prior=[50,.78],
    workload_noise_sd=6.,unverified_availability='conditional forecast only; selection blocked',
    dispersion_period=['2024-10-01','2025-02-01'],calibration_period=['2025-02-01','2025-07-01'],
    evaluation_period=['2025-10-01','2026-07-01'],historical_status='reconstructed roster, no contemporaneous injury evidence',
    future_test='after registration only',automatic_betting_authorization=False)


def allocate(weights,total,caps=None):
    """Bounded proportional allocation. Never invent capacity or exceed a player cap."""
    w=np.maximum(np.asarray(weights,float),0)
    caps=np.full(len(w),float(total)) if caps is None else np.maximum(np.asarray(caps,float),0)
    if not np.isfinite(w).all() or not np.isfinite(caps).all() or total<0:raise ValueError('Invalid allocation')
    allowed=w>0
    if caps[allowed].sum()<total-1e-8:raise ValueError('Insufficient verified rotation capacity')
    result=np.zeros(len(w));remaining=float(total)
    for _ in range(len(w)+1):
        if remaining<=1e-9:break
        free=allowed&(caps-result>1e-9)
        if not free.any():raise ValueError('Unallocated team opportunity')
        proposal=remaining*w[free]/w[free].sum()
        given=np.minimum(proposal,caps[free]-result[free])
        result[free]+=given;remaining-=given.sum()
    if abs(result.sum()-total)>1e-6:raise ValueError('Unallocated team opportunity')
    return result


def priors(players,teams,day,team,ids):
    """Only lagged player/team outcomes. No current-game participation inference."""
    cutoff=pd.Timestamp(day)-pd.Timedelta(days=2)
    h=players[(pd.to_datetime(players.game_date)<=cutoff)&players.player_id.isin(ids)].copy()
    h=h.sort_values(['game_date','game_id']).groupby('player_id',sort=False).tail(SPEC['shooting_games'])
    shooting={}
    for pid,g in h.groupby('player_id'):
        attempts=float((g.fga-g.fg3a).sum());makes=float((g.fgm-g.fg3m).sum())
        shooting[int(pid)]=dict(two_pct=(makes+53)/(attempts+100),ft_pct=(float(g.ftm.sum())+39)/(float(g.fta.sum())+50))
    t=teams[(teams.team==team)&(pd.to_datetime(teams.game_date)<=cutoff)].sort_values(['game_date','game_id']).tail(20)
    if len(t)<5:return shooting,None
    budget={s:float((t[s]*240/t.minutes).ewm(halflife=8).mean().iloc[-1]) for s in ('fga','fg3a','fta','ast','reb')}
    return shooting,budget


def overtime_probabilities(teams,before):
    t=teams[(pd.to_datetime(teams.game_date)<pd.Timestamp(before))&teams.home.astype(bool)]
    ot=np.rint((t.minutes.to_numpy(float)-240)/25).astype(int).clip(0,2)
    counts=np.bincount(ot,minlength=3).astype(float)
    if counts.sum()<100:raise ValueError('Insufficient earlier overtime labels')
    return (counts/counts.sum()).tolist()


def scenarios(fit,frame,budget,ot_probabilities,seed):
    """Joint scenarios sum to the team's minutes, FGA, 3PA, FTA and assist budgets."""
    if len(frame)<5 or budget is None:raise ValueError('Incomplete team rotation')
    f=frame.sort_values('player_id').reset_index(drop=True)
    context=f.get('availability',pd.Series('unknown',index=f.index))
    out=context.eq('out').to_numpy()
    p=f.prior_starter.fillna(.5).clip(.001,.999).to_numpy(float)
    known=f.get('confirmed_starter',pd.Series(np.nan,index=f.index))
    starters=known.eq(True).to_numpy()&~out
    bench=known.eq(False).to_numpy()
    if starters.sum()>5:raise ValueError('Conflicting confirmed starters')
    unknown=np.flatnonzero(~starters&~bench&~out)
    if starters.sum()+len(unknown)<5:raise ValueError('Incomplete confirmed starting five')
    caps=pd.to_numeric(f.get('minutes_cap',pd.Series(np.nan,index=f.index)),errors='coerce').to_numpy(float)
    rng=np.random.default_rng(int(hashlib.sha256(str(seed).encode()).hexdigest()[:16],16))
    per_role={}
    for role in (0,1):
        x=f[base.FEATURES].assign(role=float(role))
        per_role[role]=dict(minutes=np.clip(fit['minutes'].predict(x),0,48),
            **{s:np.maximum(fit['rates'][s].predict(x),0) for s in ('reb','ast','fg3a')})
    values={s:[] for s in ('minutes','fga','fg3a','fta','pts','reb','ast','fg3m')};weights=[];roles=[]
    for draw in range(SPEC['rotations']):
        role=starters.astype(float)
        need=5-int(starters.sum())
        if need:
            chance=p[unknown]/(1-p[unknown]);chance/=chance.sum()
            role[rng.choice(unknown,size=need,replace=False,p=chance)]=1
        raw=np.where(role==1,per_role[1]['minutes'],per_role[0]['minutes'])
        raw=np.maximum(raw+rng.normal(0,SPEC['workload_noise_sd'],len(f)),.05);raw[out]=0
        for ot,probability in enumerate(ot_probabilities):
            if probability<=0:continue
            cap=np.where(np.isfinite(caps),caps,48+5*ot);cap[out]=0
            minutes=allocate(raw,240+25*ot,cap)
            scale=(240+25*ot)/240
            fga=allocate(minutes*np.maximum(f.fga_rate.to_numpy(float),.001),budget['fga']*scale)
            three_rate=np.where(role==1,per_role[1]['fg3a'],per_role[0]['fg3a'])
            fg3a=allocate(minutes*np.maximum(three_rate,.001),min(budget['fg3a']*scale,fga.sum()),fga)
            fta=allocate(minutes*np.maximum(f.fta_rate.to_numpy(float),.001),budget['fta']*scale)
            fg3m=fg3a*f.fg3_pct.to_numpy(float)
            fg2m=(fga-fg3a)*f.two_pct.to_numpy(float)
            pts=2*fg2m+3*fg3m+fta*f.ft_pct.to_numpy(float)
            ast_rate=np.where(role==1,per_role[1]['ast'],per_role[0]['ast'])
            reb_rate=np.where(role==1,per_role[1]['reb'],per_role[0]['reb'])
            ast=allocate(minutes*np.maximum(ast_rate,.001),min(budget['ast']*scale,float((fg2m+fg3m).sum())))
            reb=allocate(minutes*np.maximum(reb_rate,.001),budget['reb']*scale)
            for s,v in dict(minutes=minutes,fga=fga,fg3a=fg3a,fta=fta,pts=pts,ast=ast,reb=reb,fg3m=fg3m).items():values[s].append(v)
            weights.append(probability/SPEC['rotations']);roles.append(role)
    matrix={s:np.asarray(v).T for s,v in values.items()}
    return dict(player_ids=f.player_id.astype(int).tolist(),weights=np.asarray(weights),values=matrix,
        role=np.asarray(roles).T,selection_eligible=bool(context.isin(['active','out']).all()),
        team_budgets=budget,ot_probabilities=ot_probabilities)


def mixture_over(means,weights,line,dispersion):
    means=np.asarray(means,float);w=np.asarray(weights,float)
    return np.sum(base.over(means,np.asarray(line)[...,None],dispersion)*w,axis=-1)


def fit_dispersion(means,weights,actual):
    """Fit conditional count noise after integrating workload uncertainty."""
    mu=np.maximum(np.asarray(means,float),1e-8);y=np.asarray(actual,float)[:,None]
    w=np.asarray(weights,float)
    if len(mu)<200:raise ValueError('Insufficient out-of-sample mixture rows')
    def loss(x):
        variance=x[0]*mu+x[1]*mu*mu;r=mu*mu/np.maximum(variance-mu,1e-7)
        logp=stats.nbinom.logpmf(y,r,r/(r+mu))
        return -float(np.mean(special.logsumexp(logp+np.log(w),axis=1)))
    fitted=optimize.minimize(loss,[1.3,.03],method='L-BFGS-B',bounds=[(1.01,8),(.0001,.6)],options={'maxiter':45})
    if not fitted.success:raise ValueError('Mixture dispersion did not converge: '+fitted.message)
    return dict(a=float(fitted.x[0]),b=float(fitted.x[1]),rows=len(mu),source='out-of-sample mixture likelihood')


def fit_calibration(means,weights,actual,stat,dispersion):
    grid=np.arange(base.SPEC['calibration_max_threshold'][stat]+1)+.5
    mu=np.asarray(means);y=np.asarray(actual)
    # Avoid a large player x line x scenario temporary matrix.
    p=np.column_stack([mixture_over(mu,weights,line,dispersion) for line in grid])
    observed=y[:,None]>grid[None,:];use=(p>=.01)&(p<=.99)
    sample=np.broadcast_to(1/np.maximum(use.sum(axis=1),1)[:,None],p.shape)
    result=base.fit_calibration(p[use],observed[use],sample[use])
    return dict(result,threshold_pairs=result['rows'],rows=len(y),synthetic_thresholds_are_not_offers=True,
        status='separate_chronological_mixture_calibration')
