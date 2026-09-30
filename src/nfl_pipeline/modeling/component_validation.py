"""Independent forecast contracts. None of these gates authorize a wager."""
import numpy as np
import pandas as pd

from nfl_pipeline.modeling.evaluation import clustered_gain, probability_metrics, projection_metrics


def calibrate_curve(calibrator, frame, stat, values, weights, *, books=None, extra_lines=None):
    """Apply held-out line calibration to a proper CDF, in training and replay alike."""
    if not calibrator.enabled:
        return values, weights
    from nfl_pipeline.modeling.challenger_models import OPPORTUNITY, numeric
    opportunity=numeric(frame,f'{OPPORTUNITY[stat]}_avg_5').to_numpy()
    supports=[]; queries=[]
    for i,(v,w) in enumerate(zip(values,weights)):
        support=np.unique(v)
        if extra_lines is not None:
            # Include exact offered/proxy lines so bucket boundaries use that line's correction.
            offered=np.asarray(extra_lines[i],dtype=float).reshape(-1)
            offered=offered[np.isfinite(offered) & (offered>support[0]) & (offered<support[-1])]
            support=np.unique(np.concatenate([support,offered]))
        over=((v[None,:]>support[:,None])*w[None,:]).sum(axis=1)/w.sum()
        queries.append(pd.DataFrame({'player_game':str(i),'line':support,'probability':over,
            'position':frame.position.iloc[i],
            'workload_bucket':'unknown' if np.isnan(opportunity[i]) else 'low' if opportunity[i]<4 else 'normal' if opportunity[i]<10 else 'high',
            'line_bucket':(support//(50 if stat=='passing_yards' else 20)).astype(int).astype(str),
            'book':books[i] if books is not None else 'unpriced_proxy'}))
        supports.append(support)
    calibrated=calibrator.predict(pd.concat(queries,ignore_index=True))
    width=max(map(len,supports)); curves=[]; masses=[]; offset=0
    for support in supports:
        survival=np.minimum.accumulate(np.clip(calibrated[offset:offset+len(support)],0,1))
        offset+=len(support)
        survival[-1]=0.  # Finite empirical support has no invented tail beyond its endpoint.
        mass=-np.diff(np.r_[1.,survival])
        curves.append(np.pad(support,(0,width-len(support)),constant_values=support[-1]))
        masses.append(np.pad(mass,(0,width-len(support))))
    return np.asarray(curves),np.asarray(masses)


def interval_score(actual, lower, upper, alpha=.2):
    y=np.asarray(actual); lo=np.asarray(lower); hi=np.asarray(upper)
    return hi-lo+2/alpha*np.maximum(lo-y,0)+2/alpha*np.maximum(y-hi,0)


def output_gates(rows, lines, probability, *, mean='challenger', median='median',
                 lower='p10', upper='p90', point_preserved=False):
    """Positive week-clustered improvement, with separate mean/median/CDF tests."""
    y=rows.actual.to_numpy(); ref=rows.reference.to_numpy()
    projection=projection_metrics(y,rows[mean],rows[lower],rows[upper])
    reference=projection_metrics(y,ref)
    center_gain=clustered_gain(rows.assign(reference_error=(ref-y)**2,challenger_error=(rows[mean]-y)**2))
    median_gain=clustered_gain(rows.assign(reference_error=abs(ref-y),challenger_error=abs(rows[median]-y)))
    candidate=probability_metrics(lines[probability],lines.outcome,lines.weight)
    baseline=probability_metrics(lines.reference_probability,lines.outcome,lines.weight)
    paired=lines.assign(reference_error=(lines.reference_probability-lines.outcome)**2,
                        challenger_error=(lines[probability]-lines.outcome)**2)
    grouped=paired.groupby(['season','week','player_game'])[['reference_error','challenger_error']].mean().reset_index()
    probability_gain=clustered_gain(grouped)
    def positive(g): return g['lower_95'] is not None and g['lower_95']>0
    coverage=projection.get('coverage_80')
    return {
        'expected_mean':{'pass':bool(not point_preserved and positive(center_gain) and
            abs(projection['bias'])<=abs(reference['bias'])),'metrics':projection,
            'reference':reference,'clustered_squared_error_gain':center_gain},
        'median':{'pass':bool(not point_preserved and positive(median_gain)),
            'metrics':projection_metrics(y,rows[median]),'clustered_absolute_error_gain':median_gain},
        'probability':{'pass':bool(positive(probability_gain) and coverage is not None and
            abs(coverage-.8)<=.03 and candidate['calibration_error']<=baseline['calibration_error']),
            'metrics':candidate,'reference':baseline,'coverage_80':coverage,
            'interval_score_80':float(np.mean(interval_score(y,rows[lower],rows[upper]))),
            'clustered_brier_gain':probability_gain},
        'point_forecast_preserved':point_preserved,'bankroll_approval':False,
        'interpretation':'Historical proxy-line screen only; new prospective outcomes are required.'}


class WorkloadTailCalibration:
    """Asymmetric interval repair fitted before a separate validation block."""
    @staticmethod
    def groups(opportunity):
        x=np.asarray(opportunity,dtype=float)
        return np.where(np.isnan(x),'unknown',np.where(x<20,'low',np.where(x<35,'normal','high')))

    def fit(self, actual, summary, opportunity):
        median=summary['median']; y=np.asarray(actual)
        low=(median-y)/np.maximum(10.,median-summary['p10'])
        high=(y-median)/np.maximum(10.,summary['p90']-median)
        # Shrink small calibration samples toward no change, not toward more width.
        def factors(mask,parent):
            n=int(mask.sum())
            raw=np.clip([np.quantile(low[mask],.9),np.quantile(high[mask],.9)],.25,4.)
            return (n*raw+100*np.asarray(parent))/(n+100)
        self.global_factors=factors(np.ones(len(y),dtype=bool),[1.,1.])
        self.by_workload={}
        labels=self.groups(opportunity)
        for label in np.unique(labels):
            use=labels==label
            if use.sum()>=30:
                self.by_workload[label]=factors(use,self.global_factors)
        self.enabled=False
        return self

    def transform(self, values, weights, opportunity, *, force=False):
        if not self.enabled and not force: return values,weights
        from nfl_pipeline.modeling.challenger_models import curve_summary
        median=curve_summary(values,weights)['median']
        factors=np.array([self.by_workload.get(label,self.global_factors) for label in self.groups(opportunity)])
        delta=values-median[:,None]
        adjusted=median[:,None]+delta*np.where(delta<0,factors[:,0,None],factors[:,1,None])
        return adjusted,weights

    def validate(self, actual, before, after, raw_brier, repaired_brier):
        y=np.asarray(actual)
        coverage=lambda s:float(np.mean((y>=s['p10'])&(y<=s['p90'])))
        old_score=float(np.mean(interval_score(y,before['p10'],before['p90'])))
        new_score=float(np.mean(interval_score(y,after['p10'],after['p90'])))
        self.enabled=bool(len(y)>=40 and abs(coverage(after)-.8)<abs(coverage(before)-.8)
                          and new_score<old_score and repaired_brier<raw_brier)
        self.validation={'rows':len(y),'coverage_before':coverage(before),'coverage_after':coverage(after),
            'interval_score_before':old_score,'interval_score_after':new_score,
            'brier_before':float(raw_brier),'brier_after':float(repaired_brier),'enabled':self.enabled}
        return self
