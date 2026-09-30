"""Train isolated NFL challengers; never publish production or alter betting gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine,text

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import MODEL_ROOT,freeze_production,atomic_joblib,atomic_json
from nfl_pipeline.context_contract import validate_evidence
from nfl_pipeline.modeling.challenger_models import (OPPORTUNITY,POSITIONS,FLOORS,FittedHead,OpportunityRateModel,
    ConditionalResidual,HierarchicalCalibration,player_features,workload_state,numeric,curve_summary,curve_over)
from nfl_pipeline.modeling.evaluation import expanding_week_folds,inner_partitions,projection_metrics,probability_metrics,clustered_gain

log=logging.getLogger(__name__)
ROOT=Path(__file__).resolve().parents[3]


def load_training():
    engine=create_engine(PG_DSN)
    with engine.connect() as conn:
        conn.execute(text("SET statement_timeout='180s'"))
        players=pd.read_sql(text("""
            SELECT f.* FROM features.nfl_player_game_training_features f
            JOIN raw.nfl_games g USING(game_id)
            JOIN raw.nfl_player_gamelogs p ON p.game_id=f.game_id AND p.player_id=f.player_id AND p.team_abbr=f.team_abbr
            WHERE g.status='final' AND f.n_games_prev_3>=3
              AND (COALESCE(p.offense_snaps,0)>0 OR COALESCE(p.pass_attempts,0)+COALESCE(p.carries,0)+COALESCE(p.targets,0)>0)
            ORDER BY f.season,f.week,f.game_id,f.player_id
        """),conn)
        games=pd.read_sql(text("""SELECT f.* FROM features.nfl_game_training_features f
            JOIN raw.nfl_games g USING(game_id) WHERE g.status='final' ORDER BY f.season,f.week,f.game_id"""),conn)
        contexts=pd.read_sql(text("""SELECT DISTINCT ON (p.game_id,p.player_id)
            p.game_id,p.player_id,p.created_at_utc,p.forecast_payload->'context_evidence' AS context_evidence
            FROM bets.nfl_player_prop_predictions p JOIN raw.nfl_games g USING(game_id)
            WHERE g.status='final' AND p.created_at_utc<g.start_ts_utc
              AND p.forecast_payload->'context_evidence' IS NOT NULL
            ORDER BY p.game_id,p.player_id,p.created_at_utc,p.id"""),conn)
    contexts['context_evidence']=[validate_evidence(r.context_evidence,r.created_at_utc) for r in contexts.itertuples()]
    contexts=contexts.drop(columns=['created_at_utc'])
    players=players.drop_duplicates(['game_id','player_id']).copy()
    players=players.merge(contexts,on=['game_id','player_id'],how='left',validate='one_to_one')
    return players,games.drop_duplicates('game_id').copy()


class ReferenceRecipe:
    """Re-fit the frozen architecture using only auditable lagged dependencies.

    The deployed artifact has seen later outcomes, so it cannot score past folds.
    Its true prospective comparator is the immutable forecast, not this refit.
    """
    def fit(self,frame,stat):
        self.stat=stat
        X=player_features(frame,stat)
        self.columns=[c for c in X if not c.startswith(('asof_','position_')) and not c.endswith('__missing')]
        self.head=FittedHead('regression_l1',params=dict(n_estimators=180,max_depth=4,num_leaves=15,
            min_child_samples=80,learning_rate=.035,reg_lambda=10)).fit(X[self.columns],numeric(frame,stat).to_numpy()-self.base(frame))
        return self

    def base(self,frame):
        return numeric(frame,f'{self.stat}_avg_5').fillna(0).to_numpy().clip(0)

    def predict(self,frame):
        return np.maximum(0,self.base(frame)+.5*self.head.predict(player_features(frame,self.stat)[self.columns]))


def line_frame(frame,stat,values,weights):
    prior=numeric(frame,f'{stat}_avg_5').fillna(0).to_numpy()
    op=numeric(frame,f'{OPPORTUNITY[stat]}_avg_5').fillna(0).to_numpy()
    records=[]
    for fraction in (.75,1.,1.25):
        lines=np.maximum(.5,np.floor(prior*fraction)+.5)
        records.append(pd.DataFrame({
            'game_id':frame.game_id.to_numpy(),'player_game':frame.game_id.astype(str).to_numpy()+'|'+frame.player_id.astype(str).to_numpy(),
            'season':frame.season.to_numpy(),'week':frame.week.to_numpy(),'position':frame.position.to_numpy(),
            'workload_bucket':np.where(op<4,'low',np.where(op<10,'normal','high')),
            'line_bucket':(lines//(50 if stat=='passing_yards' else 20)).astype(int).astype(str),
            'book':'unpriced_proxy','line':lines,'probability':curve_over(values,weights,lines),
            'outcome':(numeric(frame,stat).to_numpy()>lines).astype(int),'weight':1/3,
        }))
    # Small baselines can produce the same proxy line three times.
    out=pd.concat(records,ignore_index=True).drop_duplicates(['player_game','line'])
    out['weight']=1/out.groupby('player_game').line.transform('count')
    return out


def mixture_curve(model,uncertainty,frame,stat):
    comp=model.components(frame)
    X=player_features(frame,stat)
    values,weights=uncertainty.mixture(X,comp['centers'],comp['weights'])
    return values,weights,comp


def fit_bundle(train,stat):
    early,scale_data,residual_data,cal_fit,cal_gate=inner_partitions(train)
    provisional=OpportunityRateModel().fit(early,stat)
    reference_early=ReferenceRecipe().fit(early,stat)
    scale_comp=provisional.components(scale_data)
    state=workload_state(scale_data,stat)
    centers=scale_comp['centers'][np.arange(len(scale_data)),state]
    errors=numeric(scale_data,stat).to_numpy()-centers
    mixture_center=(scale_comp['weights']*scale_comp['centers']).sum(axis=1)
    uncertainty=ConditionalResidual().fit_scale(player_features(scale_data,stat),errors,mixture_center,FLOORS[stat],
        confidence_errors=numeric(scale_data,stat).to_numpy()-mixture_center)
    comp=provisional.components(residual_data)
    state=workload_state(residual_data,stat)
    errors=numeric(residual_data,stat).to_numpy()-comp['centers'][np.arange(len(residual_data)),state]
    uncertainty.fit_residuals(player_features(residual_data,stat),errors,state)
    reference_scale=ConditionalResidual().fit_scale(player_features(scale_data,stat),
        numeric(scale_data,stat).to_numpy()-reference_early.predict(scale_data),reference_early.predict(scale_data),FLOORS[stat])
    reference_errors=numeric(residual_data,stat).to_numpy()-reference_early.predict(residual_data)
    reference_scale.fit_residuals(player_features(residual_data,stat),reference_errors)
    pooled=np.quantile(reference_errors,np.linspace(.005,.995,101))
    baseline_errors=np.quantile(numeric(residual_data,stat).to_numpy()-reference_early.base(residual_data),np.linspace(.005,.995,101))
    values,weights,_=mixture_curve(provisional,uncertainty,cal_fit,stat)
    calibrator=HierarchicalCalibration().fit(line_frame(cal_fit,stat,values,weights))
    values,weights,_=mixture_curve(provisional,uncertainty,cal_gate,stat)
    calibrator.validate(line_frame(cal_gate,stat,values,weights))
    return {'model':OpportunityRateModel().fit(train,stat),'uncertainty':uncertainty,
            'calibration_model':provisional,
            'calibrator':calibrator,'reference':ReferenceRecipe().fit(train,stat),
            'reference_uncertainty':reference_scale,'pooled_reference_errors':pooled,
            'pooled_baseline_errors':baseline_errors,
            'calibration_dates':{'fit_end':str(cal_fit.game_date_et.max()),'gate_start':str(cal_gate.game_date_et.min())},
            'training_end':str(train.game_date_et.max())}


def evaluate_players(data,stat,seasons):
    frame=data.loc[data.position.isin(POSITIONS[stat]) & numeric(data,stat).notna() & numeric(data,OPPORTUNITY[stat]).notna()].copy()
    row_parts=[]; line_parts=[]; folds=[]
    for name,train,test in expanding_week_folds(frame,seasons):
        bundle=fit_bundle(train,stat)
        values,weights,comp=mixture_curve(bundle['model'],bundle['uncertainty'],test,stat)
        summary=curve_summary(values,weights)
        ref=bundle['reference'].predict(test)
        base=numeric(test,f'{stat}_avg_5').fillna(0).to_numpy()
        ref_values=ref[:,None]+bundle['pooled_reference_errors'][None,:]
        equal=np.ones_like(ref_values)/ref_values.shape[1]
        conditional_values,conditional_weights=bundle['reference_uncertainty'].mixture(player_features(test,stat),ref,np.ones((len(test),1)))
        conditional_summary=curve_summary(conditional_values,conditional_weights)
        actual=numeric(test,stat).to_numpy()
        rows=test[['game_id','player_id','season','week','position','game_date_et']].copy().reset_index(drop=True)
        rows=rows.assign(stat=stat,fold=name,actual=actual,reference=ref,baseline=base,
            challenger=summary['mean'],median=summary['median'],p10=summary['p10'],p90=summary['p90'],
            confidence=bundle['uncertainty'].confidence(player_features(test,stat)),
            predicted_opportunity=(comp['weights']*comp['opportunities']).sum(axis=1),
            actual_opportunity=numeric(test,OPPORTUNITY[stat]).to_numpy(),
            low_probability=comp['weights'][:,0],spike_probability=comp['weights'][:,2],
            reference_error=np.abs(ref-actual),challenger_error=np.abs(summary['mean']-actual),
            baseline_error=np.abs(base-actual),conditional_p10=conditional_summary['p10'],conditional_p90=conditional_summary['p90'])
        lines=line_frame(test,stat,values,weights)
        reference_lines=line_frame(test,stat,ref_values,equal)
        conditional_lines=line_frame(test,stat,conditional_values,conditional_weights)
        baseline_lines=line_frame(test,stat,base[:,None]+bundle['pooled_baseline_errors'][None,:],equal)
        lines['reference_probability']=reference_lines.probability.to_numpy()
        lines['conditional_reference_probability']=conditional_lines.probability.to_numpy()
        lines['baseline_probability']=baseline_lines.probability.to_numpy()
        lines['calibrated_probability']=bundle['calibrator'].predict(lines)
        lines['stat']=stat; lines['fold']=name
        row_parts.append(rows); line_parts.append(lines)
        folded={'fold':name,'train_rows':len(train),'test_rows':len(test),'train_end':bundle['training_end'],
                'test_start':str(test.game_date_et.min()),'calibration':bundle['calibrator'].validation,
                'model':projection_metrics(actual,summary['mean'],summary['p10'],summary['p90']),
                'reference':projection_metrics(actual,ref),'baseline':projection_metrics(actual,base)}
        folds.append(folded)
        log.info('%s %s MAE challenger %.3f reference %.3f baseline %.3f',stat,name,
                 folded['model']['mae'],folded['reference']['mae'],folded['baseline']['mae'])
    if not row_parts: raise ValueError(f'No eligible folds for {stat}')
    rows=pd.concat(row_parts,ignore_index=True); lines=pd.concat(line_parts,ignore_index=True)
    summary={
        'projection':projection_metrics(rows.actual,rows.challenger,rows.p10,rows.p90),
        'median_projection':projection_metrics(rows.actual,rows['median']),
        'reference':projection_metrics(rows.actual,rows.reference),
        'baseline':projection_metrics(rows.actual,rows.baseline),
        'conditional_reference':projection_metrics(rows.actual,rows.reference,rows.conditional_p10,rows.conditional_p90),
        'opportunity':projection_metrics(rows.actual_opportunity,rows.predicted_opportunity),
        'probabilities':{name:probability_metrics(lines[col],lines.outcome,lines.weight) for name,col in [
            ('mixture','probability'),('calibrated_mixture','calibrated_probability'),
            ('reference','reference_probability'),('baseline','baseline_probability'),('conditional_reference','conditional_reference_probability')]},
        'clustered_mae_gain':clustered_gain(rows), 'clustered_gain_vs_baseline':clustered_gain(rows,reference='baseline_error'),
        'by_position':{str(p):projection_metrics(g.actual,g.challenger,g.p10,g.p90) for p,g in rows.groupby('position')},
        'by_season':{str(s):{'challenger':projection_metrics(g.actual,g.challenger),'reference':projection_metrics(g.actual,g.reference)} for s,g in rows.groupby('season')},
        'folds':folds,'automatic_promotion':False,
    }
    summary['historical_candidate_pass']=bool(
        summary['clustered_mae_gain']['lower_95'] is not None and summary['clustered_mae_gain']['lower_95']>0
        and summary['clustered_gain_vs_baseline']['lower_95']>0
        and summary['probabilities']['calibrated_mixture']['brier']<summary['probabilities']['reference']['brier']
        and summary['probabilities']['calibrated_mixture']['brier']<summary['probabilities']['baseline']['brier'])
    from nfl_pipeline.modeling.component_validation import output_gates
    summary['output_gates']=output_gates(rows,lines,'calibrated_probability')
    summary['legacy_combined_gate_deprecated']=True
    return summary,rows,lines,fit_bundle(frame,stat)


def game_inputs(frame):
    columns=[f'{side}_{name}_avg_{w}' for side in ('home','away') for name in
             ('pf','pa','plays','pass_rate','yards_per_play','red_zone_td_rate') for w in (3,5,10)]
    columns+=['home_rest_days','away_rest_days']
    return pd.DataFrame({c:numeric(frame,c) for c in columns},index=frame.index)


def game_baseline(frame,target):
    home=(numeric(frame,'home_pf_avg_5').fillna(22)+numeric(frame,'away_pa_avg_5').fillna(22))/2
    away=(numeric(frame,'away_pf_avg_5').fillna(22)+numeric(frame,'home_pa_avg_5').fillna(22))/2
    return (home-away if target=='home_margin' else home+away).to_numpy()


class SmallGameModel:
    def fit(self,frame,target):
        self.target=target
        self.model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),Ridge(alpha=500))
        self.model.fit(game_inputs(frame),numeric(frame,target).to_numpy()-game_baseline(frame,target))
        return self

    def predict(self,frame):
        return game_baseline(frame,self.target)+.35*self.model.predict(game_inputs(frame))


def evaluate_games(frame,target,seasons):
    parts=[]
    for name,train,test in expanding_week_folds(frame,seasons):
        early,_,residual_data,_,_=inner_partitions(train)
        residual_model=SmallGameModel().fit(early,target)
        residuals=numeric(residual_data,target).to_numpy()-residual_model.predict(residual_data)
        base_residuals=numeric(residual_data,target).to_numpy()-game_baseline(residual_data,target)
        model=SmallGameModel().fit(train,target)
        pred=model.predict(test); base=game_baseline(test,target); actual=numeric(test,target).to_numpy()
        market=-numeric(test,'market_spread_home').to_numpy() if target=='home_margin' else numeric(test,'market_total').to_numpy()
        rows=test[['game_id','season','week']].copy()
        # Lines derive only from the pregame statistical baseline, not closing prices.
        line=np.floor(base)+.5
        p=((pred[:,None]+residuals[None,:])>line[:,None]).mean(axis=1)
        bp=((base[:,None]+base_residuals[None,:])>line[:,None]).mean(axis=1)
        q10,q90=np.quantile(residuals,[.1,.9])
        rows=rows.assign(fold=name,actual=actual,challenger=pred,baseline=base,market_benchmark=market,
                         line=line,probability=p,baseline_probability=bp,outcome=(actual>line).astype(int),
                         p10=pred+q10,p90=pred+q90,
                         challenger_error=abs(pred-actual),reference_error=abs(base-actual))
        parts.append(rows)
    rows=pd.concat(parts,ignore_index=True)
    market_valid=rows.market_benchmark.notna()
    summary={'projection':projection_metrics(rows.actual,rows.challenger,rows.p10,rows.p90),'baseline':projection_metrics(rows.actual,rows.baseline),
             'proxy_probability':probability_metrics(rows.probability,rows.outcome),
             'baseline_proxy_probability':probability_metrics(rows.baseline_probability,rows.outcome),
             'retrospective_market_benchmark':projection_metrics(rows.loc[market_valid,'actual'],rows.loc[market_valid,'market_benchmark']),
             'clustered_mae_gain':clustered_gain(rows),'automatic_promotion':False,
             'market_is_feature':False,'market_benchmark_is_executable_lock':False}
    summary['historical_candidate_pass']=bool(summary['clustered_mae_gain']['lower_95'] is not None and
        summary['clustered_mae_gain']['lower_95']>0 and
        summary['projection']['mae']<summary['retrospective_market_benchmark'].get('mae',float('-inf')) and
        summary['proxy_probability']['brier']<summary['baseline_proxy_probability']['brier'])
    return summary,rows,SmallGameModel().fit(frame,target)


def write_markdown(report,path):
    lines=['# NFL Accuracy Challengers','',f"Production remains frozen: {report['production_release']}",
           f"Run: {report['run_id']}",'','## Player Models','',
           '| Stat | Challenger MAE | Reference recipe MAE | Baseline MAE | Raw / calibrated Brier | Historical gate |',
           '|---|---:|---:|---:|---:|---|']
    for stat,r in report['players'].items():
        lines.append(f"| {stat} | {r['projection']['mae']:.3f} | {r['reference']['mae']:.3f} | {r['baseline']['mae']:.3f} | {r['probabilities']['mixture']['brier']:.4f} / {r['probabilities']['calibrated_mixture']['brier']:.4f} | {r['historical_candidate_pass']} |")
    lines+=['','## Game Models','','| Target | Challenger MAE | Statistical baseline | Retrospective market |','|---|---:|---:|---:|']
    for target,r in report['games'].items():
        lines.append(f"| {target} | {r['projection']['mae']:.3f} | {r['baseline']['mae']:.3f} | {r['retrospective_market_benchmark']['mae']:.3f} |")
    lines+=['','## Interpretation','',*['- '+s for s in report['limitations']],
            '',f"Historical rows with captured as-of context: {report['context_rows']}",
            'No challenger is wired into production probabilities, Discord ranking, or bankroll selection.','']
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text('\n'.join(lines),encoding='utf-8')


def run(seasons=(2024,2025,2026)):
    freeze=freeze_production('Accuracy challengers and prospective evaluation; no automatic publication')
    before=(MODEL_ROOT/'active_release.json').read_bytes()
    log.info('Loading finalized, deduplicated player-game and game features')
    players,games=load_training()
    run_id=datetime.now(timezone.utc).strftime('accuracy-%Y%m%dT%H%M%S%fZ')
    root=MODEL_ROOT/'challengers'/run_id
    report={'run_id':run_id,'production_release':freeze['release_id'],'seasons':list(seasons),
            'context_rows':int(players.context_evidence.map(lambda c:isinstance(c,dict)).sum()),'players':{},'games':{},
            'limitations':[
                'Historical probabilities use explicitly unpriced proxy lines, never invented sportsbook odds.',
                'Reference is a chronological refit of the frozen residual architecture with audited lag inputs, not predictions from a model trained on future outcomes.',
                'Some legacy derived reference features are excluded because their lock-time dependencies cannot be reconstructed.',
                'Actual frozen-production comparison uses immutable prospective forecast records; historical recipe refits are not that proof.',
                '2026 outcomes have been examined during previous research and are not an untouched final test.',
                'Inner time blocks separate model training, scale fitting, residual CDF fitting, calibration fitting, and calibration gating.',
                'Uncertainty intervals are evaluated on outer folds; confidence is learned error-within-tolerance probability, not chance a wager wins.',
                'Historical same-week context remains unknown unless captured before kickoff. True routes and first reads are not fabricated.',
                'Game market values are retrospective benchmarks only; they are never inputs to these challengers.',
                'Statistical gates use NFL-week clustered intervals. All artifacts remain challengers regardless of pass status.',
            ]}
    models={}
    for stat in OPPORTUNITY:
        report['players'][stat],rows,lines,models[stat]=evaluate_players(players,stat,seasons)
        atomic_joblib(root/f'{stat}_oof.joblib',{'player_games':rows,'proxy_lines':lines})
        atomic_json(root/'progress.json',report)
    for target in ('home_margin','total_points_actual'):
        report['games'][target],rows,models[target]=evaluate_games(games,target,seasons)
        atomic_joblib(root/f'{target}_oof.joblib',rows)
    artifact={'run_id':run_id,'production_release':freeze['release_id'],'models':models,
              'training_end':str(players.game_date_et.max()),'status':'challenger_only'}
    atomic_joblib(root/'models.joblib',artifact)
    atomic_json(root/'report.json',report)
    atomic_json(MODEL_ROOT/'challengers'/'latest.json',{'run_id':run_id,'production_release':freeze['release_id'],
        'sha256':hashlib.sha256((root/'models.joblib').read_bytes()).hexdigest()})
    atomic_json(ROOT/'reports'/'nfl_accuracy_challengers_latest.json',report)
    write_markdown(report,ROOT/'reports'/'nfl_accuracy_challengers_latest.md')
    if before!=(MODEL_ROOT/'active_release.json').read_bytes():
        raise RuntimeError('Production changed during challenger evaluation')
    return {'status':'challenger_only','run_id':run_id,'production_unchanged':True,
            'player_historical_gates':{k:v['historical_candidate_pass'] for k,v in report['players'].items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seasons',default='2024,2025,2026')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')
    warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
    # Pickle model classes under their importable module, never __main__.
    from nfl_pipeline.modeling.train_accuracy_challengers import run as train
    print(json.dumps(train(tuple(int(s) for s in args.seasons.split(','))),indent=2))


if __name__=='__main__':
    main()
