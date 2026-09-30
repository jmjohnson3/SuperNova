"""Offline opportunity/rate mixtures and out-of-fold conditional uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit

OPPORTUNITY = {'receiving_yards':'targets', 'rushing_yards':'carries', 'passing_yards':'pass_attempts'}
POSITIONS = {'receiving_yards':('RB','WR','TE'), 'rushing_yards':('QB','RB'), 'passing_yards':('QB',)}
FLOORS = {'receiving_yards':5., 'rushing_yards':4., 'passing_yards':20.}
BASES = {
    'receiving_yards':('receiving_yards','receiving_air_yards','receiving_yards_after_catch','targets','receptions','offense_snaps','offense_snap_share','air_yards_share','wopr','routes_run','route_participation','first_read_targets','red_zone_targets'),
    'rushing_yards':('rushing_yards','carries','offense_snaps','offense_snap_share','red_zone_carries','goal_line_carries'),
    'passing_yards':('passing_yards','pass_attempts','passing_tds','red_zone_pass_attempts'),
}


def numeric(frame, name):
    return pd.to_numeric(frame.get(name,pd.Series(np.nan,index=frame.index)),errors='coerce')


def player_features(frame, stat):
    allowed = {f'{b}_{suffix}_{w}' for b in BASES[stat] for suffix in ('avg','std') for w in (3,5,10)}
    allowed |= {f'opp_allowed_{stat}_avg_5','rest_days','is_home','n_games_prev_5'}
    out = pd.DataFrame({c:numeric(frame,c) for c in sorted(allowed)},index=frame.index)
    for pos in ('QB','RB','WR','TE'):
        out[f'position_{pos}'] = (frame.get('position',pd.Series('',index=frame.index))==pos).astype(float)
    op=OPPORTUNITY[stat]
    out['opportunity_trend'] = numeric(frame,f'{op}_avg_3')-numeric(frame,f'{op}_avg_10')
    out['role_volatility'] = numeric(frame,f'{op}_std_5')/(numeric(frame,f'{op}_avg_5')+1)
    out['prior_yards_per_opportunity'] = numeric(frame,f'{stat}_avg_5')/numeric(frame,f'{op}_avg_5').replace(0,np.nan)
    contexts=frame.get('context_evidence',pd.Series(None,index=frame.index,dtype=object))
    for key in ('depth_rank','depth_movement','expected_starter_from_depth','teammate_injury_count'):
        out[f'asof_{key}'] = contexts.map(lambda c:c.get(key) if isinstance(c,dict) else None)
    for key in ('injury_status','practice_status'):
        for label in ('out','doubtful','questionable','limited','did not participate','full'):
            out[f'asof_{key}_{label.replace(" ","_")}'] = contexts.map(
                lambda c: float(label in str(c[key]).lower()) if isinstance(c,dict) and c.get(key) is not None else np.nan)
    # The missing indicator survives imputation; no report is not a healthy report.
    out=out.apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    missing=out.isna().astype(float).add_suffix('__missing')
    return pd.concat([out,missing],axis=1)


def workload_state(frame, stat):
    op=OPPORTUNITY[stat]
    actual=numeric(frame,op).fillna(0).to_numpy()
    prior=numeric(frame,f'{op}_avg_5').fillna(0).to_numpy()
    floor={'targets':8,'carries':18,'pass_attempts':38}[op]
    return np.where(actual<np.maximum(1,prior*.5),0,np.where(actual>=np.maximum(floor,prior*1.4),2,1))


class FittedHead:
    def __init__(self, objective='regression', classifier=False, params=None):
        self.objective=objective
        self.classifier=classifier
        self.params=params or {}

    def fit(self,X,y,weights=None):
        self.columns=[c for c in X if X[c].notna().any()]
        self.fills=X[self.columns].median().fillna(0).to_dict()
        X=self.transform(X)
        y=np.asarray(y)
        if not len(y) or not np.isfinite(y).all():
            raise ValueError('A fitted head requires finite observed targets')
        self.constant=float(np.average(y,weights=weights))
        self.classes_=np.unique(y) if self.classifier else None
        self.class_mass=np.array([np.sum((y==c)*(np.ones(len(y)) if weights is None else weights)) for c in self.classes_]) if self.classifier else None
        if self.classifier:
            self.class_mass=self.class_mass/self.class_mass.sum()
        self.model=None
        if len(y)>=40 and len(np.unique(y))>1 and self.columns:
            kwargs=dict(n_estimators=100,max_depth=3,num_leaves=7,min_child_samples=60,
                        learning_rate=.04,reg_lambda=15,verbosity=-1,random_state=42,n_jobs=2)
            kwargs.update(self.params)
            self.model=lgb.LGBMClassifier(**kwargs) if self.classifier else lgb.LGBMRegressor(objective=self.objective,**kwargs)
            self.model.fit(X,y,sample_weight=weights)
            if self.classifier:
                self.classes_=self.model.classes_
        return self

    def transform(self,X):
        return X.reindex(columns=self.columns).fillna(self.fills).fillna(0.)

    def predict(self,X):
        return np.full(len(X),self.constant) if self.model is None else self.model.predict(self.transform(X))

    def probabilities(self,X,n_classes):
        out=np.zeros((len(X),n_classes))
        if self.model is None:
            out[:,self.classes_.astype(int)]=self.class_mass
        else:
            out[:,self.classes_.astype(int)]=self.model.predict_proba(self.transform(X))
        return out


class OpportunityRateModel:
    """Conditional workload states; efficiency learns only from positive exposure."""
    def fit(self,frame,stat):
        self.stat=stat
        X=player_features(frame,stat)
        states=workload_state(frame,stat)
        exposure=numeric(frame,OPPORTUNITY[stat]).fillna(0).to_numpy().clip(0)
        if not (exposure>0).any():
            raise ValueError('Opportunity training requires positive observed exposure')
        y=numeric(frame,stat).to_numpy()
        self.state_head=FittedHead(classifier=True).fit(X,states)
        self.op_heads={}
        self.rate_heads={}
        for state in range(3):
            mask=states==state
            if mask.sum()<150:
                mask=np.ones(len(X),dtype=bool)
            self.op_heads[state]=FittedHead('poisson').fit(X.loc[mask],exposure[mask])
            positive=mask & (exposure>0)
            if not positive.any():
                positive=exposure>0
            rate=y[positive]/exposure[positive]
            # Per-opportunity loss: a one-target outlier has less influence than a full game.
            self.rate_heads[state]=FittedHead().fit(X.loc[positive],rate,np.minimum(exposure[positive],40))
        self.state_prior=np.bincount(states,minlength=3)/len(states)
        return self

    def components(self,frame):
        X=player_features(frame,self.stat)
        weights=.95*self.state_head.probabilities(X,3)+.05*self.state_prior[None,:]
        opportunities=np.column_stack([np.maximum(0,self.op_heads[s].predict(X)) for s in range(3)])
        rates=np.column_stack([self.rate_heads[s].predict(X) for s in range(3)])
        return {'weights':weights,'opportunities':opportunities,'rates':rates,'centers':opportunities*rates}

    def predict(self,frame):
        comp=self.components(frame)
        return (comp['weights']*comp['centers']).sum(axis=1)


class ConditionalResidual:
    """Scale and confidence fitted on held-out errors; residual CDF uses later rows."""
    def fit_scale(self,X,errors,center,floor,confidence_errors=None):
        self.floor=floor
        self.scale_head=FittedHead().fit(X,np.abs(errors))
        within=(np.abs(errors if confidence_errors is None else confidence_errors)<=np.maximum(floor,.35*np.abs(center))).astype(int)
        self.confidence_head=FittedHead(classifier=True).fit(X,within)
        self.global_scale=max(floor,float(np.mean(np.abs(errors))))
        return self

    def scale(self,X):
        return np.maximum(self.floor,.75*self.scale_head.predict(X)+.25*self.global_scale)

    def fit_residuals(self,X,errors,states=None):
        z=np.asarray(errors)/self.scale(X)
        grid=np.linspace(.005,.995,101)
        self.quantiles={-1:np.quantile(z,grid)}
        if states is not None:
            for state in range(3):
                vals=z[np.asarray(states)==state]
                if len(vals)>=60:
                    self.quantiles[state]=np.quantile(vals,grid)
        return self

    def confidence(self,X):
        return self.confidence_head.probabilities(X,2)[:,1]

    def mixture(self,X,centers,weights):
        if centers.ndim==1:
            centers=centers[:,None]
            weights=np.ones_like(centers)
        scale=self.scale(X)
        values=[]; masses=[]
        for state in range(centers.shape[1]):
            q=self.quantiles.get(state,self.quantiles[-1])
            values.append(centers[:,state,None]+scale[:,None]*q)
            masses.append(np.broadcast_to(weights[:,state,None]/len(q),(len(X),len(q))))
        return np.concatenate(values,axis=1),np.concatenate(masses,axis=1)


def curve_summary(values,weights):
    order=np.argsort(values,axis=1)
    vals=np.take_along_axis(values,order,axis=1)
    ws=np.take_along_axis(weights,order,axis=1)
    ws=ws/ws.sum(axis=1,keepdims=True)
    cdf=np.cumsum(ws,axis=1)
    result={'mean':(vals*ws).sum(axis=1)}
    for name,q in [('p10',.1),('median',.5),('p90',.9)]:
        result[name]=vals[np.arange(len(vals)),np.minimum((cdf<q).sum(axis=1),vals.shape[1]-1)]
    return result


def curve_over(values,weights,lines):
    return ((values>np.asarray(lines)[:,None])*weights).sum(axis=1)/weights.sum(axis=1)


class HierarchicalCalibration:
    """Monotone global logit calibration plus shrunk role/line corrections."""
    @staticmethod
    def keys(frame,level=4):
        columns=['position','workload_bucket','line_bucket','book'][:level]
        return frame.reindex(columns=columns).fillna('unknown').astype(str).agg('|'.join,axis=1)

    def fit(self,frame):
        p=np.clip(frame.probability.to_numpy(),.001,.999)
        y=frame.outcome.to_numpy()
        w=frame.weight.to_numpy()
        self.slope,self.intercept=1.,0.
        if len(frame)>=50 and len(np.unique(y))==2:
            model=LogisticRegression(C=.5,max_iter=300).fit(logit(p)[:,None],y,sample_weight=w)
            self.slope=max(.05,float(model.coef_[0,0])); self.intercept=float(model.intercept_[0])
        global_p=expit(self.intercept+self.slope*logit(p))
        self.offsets_by_level={}
        parent_p=global_p
        for level in range(1,5):
            keys=self.keys(frame,level)
            tmp=frame.assign(key=keys,wp=w*parent_p,wy=w*y)
            offsets={}
            for key,group in tmp.groupby('key'):
                n=group.weight.sum()
                parent=float(group.wp.sum()/n)
                observed=float((group.wy.sum()+60*parent)/(n+60))
                offsets[key]=float(logit(np.clip(observed,.001,.999))-logit(np.clip(parent,.001,.999)))
            self.offsets_by_level[level]=offsets
            parent_p=expit(logit(np.clip(parent_p,.001,.999))+keys.map(offsets).fillna(0).to_numpy())
        self.enabled=False
        return self

    def candidate(self,frame):
        p=np.clip(frame.probability.to_numpy(),.001,.999)
        shifts=np.zeros(len(frame))
        for level,offsets in getattr(self,'offsets_by_level',{4:getattr(self,'offsets',{})}).items():
            shifts+=self.keys(frame,level).map(offsets).fillna(0).to_numpy()
        result=expit(self.intercept+self.slope*logit(p)+shifts)
        if {'player_game','line','book'}.issubset(frame):
            order=frame.assign(_row=np.arange(len(frame))).sort_values('line')
            for _,g in order.groupby(['player_game','book']):
                idx=g._row.to_numpy()
                result[idx]=np.minimum.accumulate(result[idx])
        return result

    def validate(self,later):
        raw=later.probability.to_numpy(); new=self.candidate(later); y=later.outcome.to_numpy()
        raw_brier=np.average((raw-y)**2,weights=later.weight)
        new_brier=np.average((new-y)**2,weights=later.weight)
        self.enabled=bool(later.player_game.nunique()>=40 and new_brier<raw_brier)
        self.validation={'rows':len(later),'unique_player_games':later.player_game.nunique(),
                         'raw_brier':float(raw_brier),'calibrated_brier':float(new_brier),'enabled':self.enabled}
        return self

    def predict(self,frame):
        return self.candidate(frame) if self.enabled else frame.probability.to_numpy()
