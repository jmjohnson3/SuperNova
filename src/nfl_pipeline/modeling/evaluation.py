"""Chronological grouped folds and proper scoring for challenger evaluations."""
import numpy as np
import pandas as pd


def expanding_week_folds(frame,seasons,width=4):
    for season in seasons:
        weeks=sorted(frame.loc[frame.season==season,'week'].unique())
        for offset in range(0,len(weeks),width):
            selected=weeks[offset:offset+width]
            test=frame.loc[(frame.season==season)&frame.week.isin(selected)]
            start=test.game_date_et.min()
            train=frame.loc[frame.game_date_et<start]
            train=train.loc[~train.game_id.isin(test.game_id)]
            if len(train)>=300 and len(test):
                yield f'{season}-W{int(selected[0]):02d}',train.copy(),test.copy()


def inner_partitions(train):
    weeks=train[['season','week','game_date_et']].groupby(['season','week']).game_date_et.min().sort_values()
    if len(weeks)<14:
        raise ValueError('Insufficient chronological training weeks')
    recent=list(weeks.index[-8:])
    tags=list(zip(train.season,train.week))
    groups=[set(recent[i:i+2]) for i in (0,2,4,6)]
    masks=[np.array([tag in group for tag in tags]) for group in groups]
    early=~np.logical_or.reduce(masks)
    return (train.loc[early].copy(),*(train.loc[m].copy() for m in masks))


def probability_metrics(p,y,weights=None):
    p=np.asarray(p,dtype=float); y=np.asarray(y,dtype=float)
    mask=np.isfinite(p)&np.isfinite(y)
    if not mask.any(): return {'rows':0}
    p=p[mask]; y=y[mask]
    w=np.ones(len(p)) if weights is None else np.asarray(weights)[mask]
    ece=0.
    for bucket in range(10):
        use=np.minimum((np.clip(p,0,1)*10).astype(int),9)==bucket
        if use.any(): ece+=w[use].sum()/w.sum()*abs(np.average(p[use]-y[use],weights=w[use]))
    clipped=np.clip(p,1e-6,1-1e-6)
    return {'rows':len(p),'brier':float(np.average((p-y)**2,weights=w)),
            'log_loss':float(np.average(-y*np.log(clipped)-(1-y)*np.log1p(-clipped),weights=w)),
            'calibration_error':float(ece),'probability_bias':float(np.average(p-y,weights=w))}


def projection_metrics(actual,center,p10=None,p90=None):
    actual=np.asarray(actual,dtype=float); center=np.asarray(center,dtype=float)
    valid=np.isfinite(actual)&np.isfinite(center)
    if not valid.any(): return {'rows':0}
    actual_all=actual
    actual=actual[valid]; error=center[valid]-actual
    result={'rows':len(actual),'mae':float(np.mean(abs(error))),'rmse':float(np.sqrt(np.mean(error**2))),
            'bias':float(np.mean(error))}
    if p10 is not None:
        low=np.asarray(p10,dtype=float); high=np.asarray(p90,dtype=float)
        intervals=valid & np.isfinite(low) & np.isfinite(high) & (low<=high)
        result['interval_rows']=int(intervals.sum())
        result['coverage_80']=float(np.mean((actual_all[intervals]>=low[intervals])&(actual_all[intervals]<=high[intervals]))) if intervals.any() else None
        result['interval_width']=float(np.mean(high[intervals]-low[intervals])) if intervals.any() else None
    return result


def clustered_gain(frame,challenger='challenger_error',reference='reference_error',samples=500):
    # Resample NFL weeks, retaining all correlated games/players within each week.
    df=frame.assign(gain=frame[reference]-frame[challenger])
    clusters=df.groupby(['season','week']).gain.agg(['sum','count'])
    if len(clusters)<2: return {'clusters':len(clusters),'lower_95':None,'upper_95':None}
    rng=np.random.default_rng(42)
    idx=rng.integers(0,len(clusters),size=(samples,len(clusters)))
    means=clusters['sum'].to_numpy()[idx].sum(axis=1)/clusters['count'].to_numpy()[idx].sum(axis=1)
    low,high=np.quantile(means,[.025,.975])
    return {'clusters':len(clusters),'lower_95':float(low),'upper_95':float(high),
            'mean_gain':float(df.gain.mean())}
