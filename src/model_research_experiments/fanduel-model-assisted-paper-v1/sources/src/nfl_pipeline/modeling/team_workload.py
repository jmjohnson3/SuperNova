"""Pregame team budgets and roster allocations, independent of offered props."""
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sqlalchemy import create_engine,text

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.modeling.challenger_models import FittedHead, numeric, OPPORTUNITY

COUNTS=('pass_attempts','carries','targets')
YARDS=('passing_yards','rushing_yards','receiving_yards')
PLAYER_USAGE=('routes_run','pass_route_opportunities','route_participation','snap_share',
              'target_share','air_yards_share','offense_snap_share','receiving_air_yards',
              'targets_per_route_run','yards_per_route_run')
TEAM_CONTEXT = ('game_total_line', 'team_spread_line', 'team_implied_points',
                'opponent_implied_points', 'is_home', 'wind_mph', 'is_indoor')


def season_recency_weights(frame, strength=1.0):
    """Moderately favor the latest season without discarding long-term evidence.

    Opportunity can move quickly, but a handful of current-season games must
    not overwhelm several years of role history. The strength is selected on a
    later chronological block; zero is exactly the former unweighted behavior.
    """
    strength=float(strength)
    if not 0 <= strength <= 1:
        raise ValueError('Season-recency strength must be in [0,1]')
    season=numeric(frame,'season')
    if not season.notna().any() or strength == 0:
        return np.ones(len(frame),dtype=float)
    age=(float(season.max())-season).clip(lower=0).fillna(float(season.max()-season.min()))
    raw=np.power(.80,age.to_numpy(dtype=float))
    raw=np.clip(raw/max(float(np.mean(raw)),1e-9),.50,1.75)
    return (1-strength)+strength*raw


def _select_season_recency(earlier,later):
    """Choose opportunity recency using only a later training block."""
    if len(earlier)<80 or len(later)<32:
        return 0.,None
    scores={}
    for strength in (0.,.35,.70,1.):
        weight=season_recency_weights(earlier,strength)
        errors=[]
        for op in ('pass_attempts','carries'):
            head=FittedHead('regression_l1').fit(
                team_features(earlier),earlier[f'actual_{op}'],weight)
            actual=numeric(later,f'actual_{op}').to_numpy()
            pred=np.maximum(0,head.predict(team_features(later)))
            scale=max(1.,float(np.nanmean(actual)))
            errors.append(float(np.mean(np.abs(pred-actual))/scale))
        scores[strength]=float(np.mean(errors))
    selected=min(scores,key=lambda value:(scores[value],value))
    # Keep the old weighting unless recency earns a measurable later-block gain.
    if scores[selected] >= scores[0.]-1e-4:
        selected=0.
    return selected,{'selected_strength':selected,'scores':scores,
                     'train_rows':len(earlier),'validation_rows':len(later)}


def load_history():
    with create_engine(PG_DSN).connect() as conn:
        conn.execute(text("SET statement_timeout='120s'"))
        raw=pd.read_sql(text("""SELECT DISTINCT ON (p.game_id,p.team_abbr,p.player_id)
            p.game_id,p.player_id,p.team_abbr,p.opponent_abbr,p.position,
            g.season,g.week,g.game_date_et,p.pass_attempts,p.carries,p.targets,
            p.passing_yards,p.rushing_yards,p.receiving_yards,
            p.routes_run,p.pass_route_opportunities,p.route_participation,p.snap_share,
            p.target_share,p.air_yards_share,p.offense_snap_share,p.receiving_air_yards,
            p.targets_per_route_run,p.yards_per_route_run
            FROM raw.nfl_player_gamelogs p JOIN raw.nfl_games g USING(game_id)
            WHERE g.status='final'
            ORDER BY p.game_id,p.team_abbr,p.player_id,p.updated_at_utc DESC"""),conn)
        context=pd.read_sql(text("""SELECT DISTINCT ON (game_id,team_abbr)
            game_id,team_abbr,game_total_line,team_spread_line,team_implied_points,
            opponent_implied_points,is_home,wind_mph,is_indoor
            FROM features.nfl_player_game_training_features
            ORDER BY game_id,team_abbr,player_id"""),conn)
    return raw.merge(context,on=['game_id','team_abbr'],how='left',validate='many_to_one')


class PregameTeamHistory:
    """Build every roster from earlier games, never that game's participants."""
    def __init__(self):
        self.teams=defaultdict(lambda:deque(maxlen=10))

    def context(self, game, team, opponent, day, season, week, game_context=None):
        past=list(self.teams.get(team,[])); defense=list(self.teams.get(opponent,[]))
        row={'game_id':game,'team_abbr':team,'opponent_abbr':opponent,'game_date_et':day,
             'season':season,'week':week,'history_games':len(past)}
        game_context = game_context or {}
        for name in TEAM_CONTEXT:
            row[name] = game_context.get(name)
        for name in COUNTS+YARDS:
            for window in (3,5,10):
                row[f'prior_{name}_{window}']=np.mean([r[name] for r in past[-window:]]) if past else np.nan
        row['rest_days']=(pd.Timestamp(day)-pd.Timestamp(past[-1]['game_date_et'])).days if past else np.nan
        row['opponent_prior_plays']=np.mean([r['pass_attempts']+r['carries'] for r in defense[-5:]]) if defense else np.nan
        pool=defaultdict(list)
        for previous in past[-5:]:
            for player in previous['players']:
                pool[str(player['player_id'])].append(player)
        roster=[]
        for player_id, history in pool.items():
            latest=history[-1]
            member={'game_id':game,'team_abbr':team,'player_id':player_id,'position':latest['position'],
                    'game_date_et':day,'season':season,'week':week,'known_games':len(history),
                    'days_since_seen':(pd.Timestamp(day)-pd.Timestamp(latest['game_date_et'])).days,
                    'injury_out':np.nan,'depth_rank':np.nan}
            for name in COUNTS+YARDS:
                member[f'prior_{name}']=float(np.mean([r[name] for r in history]))
                member[f'prior_{name}_std']=float(np.std([r[name] for r in history]))
            for name in PLAYER_USAGE:
                values=np.asarray([r.get(name,np.nan) for r in history],dtype=float)
                values=values[np.isfinite(values)]
                member[f'prior_{name}']=float(np.mean(values)) if len(values) else np.nan
                member[f'prior_{name}_std']=float(np.std(values)) if len(values) else np.nan
            snap=member.get('prior_offense_snap_share')
            if not np.isfinite(snap): snap=member.get('prior_snap_share')
            member['starter_confidence']=float(np.clip(snap,0,1)) if np.isfinite(snap) else np.nan
            roster.append(member)
        # Historical depth is not fabricated. This lagged role rank is derived
        # solely from games before the forecast and gives the share model a
        # comparable signal when archived depth snapshots are unavailable.
        for position in ('QB','RB','WR','TE'):
            members=[m for m in roster if m['position']==position]
            role='pass_attempts' if position=='QB' else 'carries' if position=='RB' else 'targets'
            def role_value(member):
                value=member.get(f'prior_{role}')
                return float(value) if value is not None and np.isfinite(value) else 0.
            ordered=sorted(members,key=lambda m:-role_value(m))
            for rank,member in enumerate(ordered,1): member['depth_rank']=float(rank)
        return row,roster

    def update(self, team, day, records):
        totals={name:float(sum(r[name] for r in records)) for name in COUNTS+YARDS}
        self.teams[team].append({**totals,'game_date_et':day,'players':records})


def prepare_history(raw):
    raw=raw.drop_duplicates(['game_id','team_abbr','player_id']).copy()
    for name in PLAYER_USAGE:
        if name not in raw:
            raw[name]=np.nan
    for name in COUNTS+YARDS+PLAYER_USAGE:
        raw[name]=pd.to_numeric(raw[name],errors='coerce')
    # Missing source totals cannot be taught as a measured zero-volume game.
    raw=raw.dropna(subset=list(COUNTS)+list(YARDS))
    state=PregameTeamHistory(); teams=[]; rosters=[]
    for day,games in raw.sort_values('game_date_et').groupby('game_date_et',sort=True):
        pending=[]
        for (game,team),group in games.groupby(['game_id','team_abbr']):
            first=group.iloc[0]
            row,pool=state.context(game,team,first.opponent_abbr,day,int(first.season),int(first.week),
                                   {name:first.get(name) for name in TEAM_CONTEXT})
            actual=group.assign(player_id=group.player_id.astype(str)).set_index('player_id')
            for name in COUNTS+YARDS: row[f'actual_{name}']=float(group[name].sum())
            known={p['player_id'] for p in pool}
            for op in COUNTS:
                total=row[f'actual_{op}']
                row[f'unallocated_{op}']=float(group.loc[~group.player_id.astype(str).isin(known),op].sum()/total) if total>0 else 0.
            for member in pool:
                pid=member['player_id']
                for name in COUNTS+YARDS:
                    member[f'actual_{name}']=float(actual.loc[pid,name]) if pid in actual.index else 0.
            teams.append(row); rosters.extend(pool)
            pending.append((team,group.to_dict('records')))
        # No results from another game on this date enter a pregame feature.
        for team,records in pending: state.update(team,day,records)
    return pd.DataFrame(teams),pd.DataFrame(rosters)


def team_features(frame):
    names=[f'prior_{n}_{w}' for n in COUNTS+YARDS for w in (3,5,10)]
    names+=['rest_days','opponent_prior_plays','history_games',*TEAM_CONTEXT]
    out=pd.DataFrame({c:numeric(frame,c) for c in names},index=frame.index)
    # An unavailable market, injury, weather, or opponent input is not its
    # numeric zero. Trees receive both median-imputed values and the missingness
    # pattern so they can learn when less context is available.
    return pd.concat([out,out.isna().astype(float).add_suffix('_missing')],axis=1)


def roster_features(frame):
    names=[f'prior_{n}{suffix}' for n in COUNTS+YARDS for suffix in ('','_std')]
    names+=[f'prior_{n}{suffix}' for n in PLAYER_USAGE for suffix in ('','_std')]
    names+=['known_games','days_since_seen','injury_out','depth_rank',
            'starter_confidence','depth_movement','teammate_absences','teammate_limited']
    out=pd.DataFrame({c:numeric(frame,c) for c in names},index=frame.index)
    for position in ('QB','RB','WR','TE'):
        out['position_'+position]=(frame.position==position).astype(float)
    return pd.concat([out,out.isna().astype(float).add_suffix('_missing')],axis=1)


def allocation_features(teams,roster,op):
    """Team context plus the quality/completeness of the known roster pool."""
    base=team_features(teams).copy()
    if roster.empty:
        for name in ('roster_count','known_games_mean','days_since_seen_mean',
                     'prior_opportunity_sum','role_history_coverage','injury_context_coverage',
                     'depth_context_coverage'):
            base[name]=np.nan
        return base
    work=roster.copy()
    work['_known_games']=numeric(work,'known_games')
    work['_days_since_seen']=numeric(work,'days_since_seen')
    work['_prior_opportunity']=numeric(work,f'prior_{op}')
    work['_role_known']=work['_prior_opportunity'].notna().astype(float)
    work['_injury_known']=numeric(work,'injury_out').notna().astype(float)
    work['_depth_known']=numeric(work,'depth_rank').notna().astype(float)
    grouped=work.groupby(['game_id','team_abbr'],dropna=False).agg(
        roster_count=('player_id','nunique'),
        known_games_mean=('_known_games','mean'),
        days_since_seen_mean=('_days_since_seen','mean'),
        prior_opportunity_sum=('_prior_opportunity','sum'),
        role_history_coverage=('_role_known','mean'),
        injury_context_coverage=('_injury_known','mean'),
        depth_context_coverage=('_depth_known','mean'),
    ).reset_index()
    aligned=teams[['game_id','team_abbr']].merge(
        grouped,on=['game_id','team_abbr'],how='left',validate='one_to_one'
    )
    for name in aligned.columns.difference(['game_id','team_abbr']):
        base[name]=pd.to_numeric(aligned[name],errors='coerce').to_numpy()
    missing=base.isna().astype(float).add_suffix('_allocation_missing')
    return pd.concat([base,missing],axis=1)


def _best_blend(reference,challenger,actual):
    reference=np.asarray(reference,dtype=float)
    challenger=np.asarray(challenger,dtype=float)
    actual=np.asarray(actual,dtype=float)
    valid=np.isfinite(reference)&np.isfinite(challenger)&np.isfinite(actual)
    if valid.sum()<32:
        return 0.,None
    best=(float(np.mean(np.abs(reference[valid]-actual[valid]))),0.)
    for weight in (.15,.25,.35,.50,.65,1.0):
        pred=(1-weight)*reference[valid]+weight*challenger[valid]
        mae=float(np.mean(np.abs(pred-actual[valid])))
        if mae < best[0]-1e-9:
            best=(mae,weight)
    baseline=float(np.mean(np.abs(reference[valid]-actual[valid])))
    return best[1],{'rows':int(valid.sum()),'baseline_mae':baseline,
                    'selected_mae':best[0],'selected_weight':best[1]}


def _normalised_group_score(values,frame):
    values=np.maximum(0,np.asarray(values,dtype=float))
    keys=[frame.game_id,frame.team_abbr]
    total=pd.Series(values,index=frame.index).groupby(keys).transform('sum').to_numpy()
    return np.divide(values,total,out=np.zeros(len(values)),where=total>0)


def _best_share_blend(prior,learned,actual,frame):
    prior=np.asarray(prior,dtype=float)
    learned=np.asarray(learned,dtype=float)
    actual=np.asarray(actual,dtype=float)
    weights=1/frame.groupby(['game_id','team_abbr']).player_id.transform('count').to_numpy()
    valid=np.isfinite(prior)&np.isfinite(learned)&np.isfinite(actual)&np.isfinite(weights)
    if valid.sum()<64:
        return 0.,None
    def loss(blend):
        score=_normalised_group_score((1-blend)*prior+blend*learned,frame)
        return float(np.average(np.abs(score[valid]-actual[valid]),weights=weights[valid]))
    best=(loss(0.),0.)
    for blend in (.15,.25,.35,.50,.65,1.0):
        value=loss(blend)
        if value<best[0]-1e-9: best=(value,blend)
    return best[1],{'rows':int(valid.sum()),'baseline_weighted_mae':loss(0.),
                    'selected_weighted_mae':best[0],'selected_weight':best[1]}


class TeamAllocationModel:
    def fit(self,teams,roster,stat):
        self.stat=stat; self.op=OPPORTUNITY[stat]
        teams=teams.loc[teams.history_games>=3].copy()
        roster=roster.merge(teams[['game_id','team_abbr',f'actual_{self.op}']],on=['game_id','team_abbr'],
            suffixes=('','_team'),validate='many_to_one')
        valid=teams.actual_pass_attempts>0
        target_ratio=(teams.loc[valid,'actual_targets']/teams.loc[valid,'actual_pass_attempts']).clip(0,1)
        self.target_fraction=float(np.clip(
            teams.loc[valid,'actual_targets'].sum()/teams.loc[valid,'actual_pass_attempts'].sum(),0,1))
        self.reserve=float(np.clip(teams[f'unallocated_{self.op}'].mean(),0,1))
        # Learn context-sensitive targetable-attempt and unknown-roster reserve
        # rates. The influence is selected on the chronologically latest 20%
        # and may remain zero when the learned head fails to beat the constant.
        ordered=teams.sort_values(['game_date_et','game_id','team_abbr'])
        split=max(40,int(len(ordered)*.80))
        split=min(split,max(0,len(ordered)-32))
        earlier=ordered.iloc[:split]
        later=ordered.iloc[split:]
        self.recency_strength,self.recency_validation=_select_season_recency(earlier,later)
        earlier_recency=season_recency_weights(earlier,self.recency_strength)
        team_recency=season_recency_weights(teams,self.recency_strength)
        self.volume_strength_by_op={op:0. for op in ('pass_attempts','carries')}
        self.volume_validation={}
        if len(earlier)>=40 and len(later)>=32:
            for op in ('pass_attempts','carries'):
                provisional=FittedHead('regression_l1').fit(
                    team_features(earlier),earlier[f'actual_{op}'],earlier_recency)
                learned=np.maximum(0,provisional.predict(team_features(later)))
                reference=numeric(later,f'prior_{op}_5').to_numpy()
                weight,review=_best_blend(reference,learned,later[f'actual_{op}'])
                self.volume_strength_by_op[op]=weight
                self.volume_validation[op]=review
        self.volume={op:FittedHead('regression_l1').fit(
            team_features(teams),teams[f'actual_{op}'],team_recency)
            for op in ('pass_attempts','carries')}
        self.volume_strength=0.
        self.share_strength=0.
        early_valid=earlier.actual_pass_attempts>0
        self.allocation_validation={}
        self.target_fraction_strength=0.
        if early_valid.sum()>=40 and len(later)>=32:
            early_target_fraction=float(np.clip(
                earlier.loc[early_valid,'actual_targets'].sum()
                / earlier.loc[early_valid,'actual_pass_attempts'].sum(),0,1))
            target_head=FittedHead('regression_l1').fit(
                team_features(earlier.loc[early_valid]),
                (earlier.loc[early_valid,'actual_targets']/earlier.loc[early_valid,'actual_pass_attempts']).clip(0,1),
                earlier_recency[early_valid.to_numpy()],
            )
            later_valid=later.actual_pass_attempts>0
            if later_valid.any():
                actual=(later.loc[later_valid,'actual_targets']/later.loc[later_valid,'actual_pass_attempts']).clip(0,1)
                learned=np.clip(target_head.predict(team_features(later.loc[later_valid])),0.45,1.0)
                self.target_fraction_strength,review=_best_blend(
                    np.full(len(actual),early_target_fraction),learned,actual)
                self.allocation_validation['target_fraction']=review
        self.target_fraction_head=FittedHead('regression_l1').fit(
            team_features(teams.loc[valid]),target_ratio,team_recency[valid.to_numpy()])

        self.reserve_strength=0.
        earlier_roster=roster.merge(
            earlier[['game_id','team_abbr']],on=['game_id','team_abbr'],how='inner',validate='many_to_one')
        later_roster=roster.merge(
            later[['game_id','team_abbr']],on=['game_id','team_abbr'],how='inner',validate='many_to_one')
        if len(earlier)>=40 and len(later)>=32:
            early_reserve=float(np.clip(earlier[f'unallocated_{self.op}'].mean(),0,1))
            reserve_head=FittedHead('regression_l1').fit(
                allocation_features(earlier,earlier_roster,self.op),
                earlier[f'unallocated_{self.op}'].clip(0,1),
                earlier_recency,
            )
            learned=np.clip(reserve_head.predict(allocation_features(later,later_roster,self.op)),0,0.65)
            self.reserve_strength,review=_best_blend(
                np.full(len(later),early_reserve),learned,later[f'unallocated_{self.op}'].clip(0,1))
            self.allocation_validation['reserve']=review
        self.reserve_head=FittedHead('regression_l1').fit(
            allocation_features(teams,roster,self.op),teams[f'unallocated_{self.op}'].clip(0,1),
            team_recency)
        denominator=roster[f'actual_{self.op}_team'].to_numpy()
        target=np.divide(roster[f'actual_{self.op}'],denominator,out=np.zeros(len(roster)),where=denominator>0)
        group_weight=1/roster.groupby(['game_id','team_abbr']).player_id.transform('count')
        roster_recency=season_recency_weights(roster,self.recency_strength)
        self.share_validation=None
        if len(earlier_roster)>=64 and len(later_roster)>=64:
            provisional_share=FittedHead('regression_l1').fit(
                roster_features(earlier_roster),
                np.divide(
                    earlier_roster[f'actual_{self.op}'],earlier_roster[f'actual_{self.op}_team'],
                    out=np.zeros(len(earlier_roster)),
                    where=earlier_roster[f'actual_{self.op}_team'].to_numpy()>0,
                ),
                (1/earlier_roster.groupby(['game_id','team_abbr']).player_id.transform('count'))
                * season_recency_weights(earlier_roster,self.recency_strength),
            )
            prior=numeric(later_roster,f'prior_{self.op}').fillna(0).clip(lower=0)
            prior_sum=prior.groupby([later_roster.game_id,later_roster.team_abbr]).transform('sum')
            prior_share=np.divide(prior,prior_sum,out=np.zeros(len(prior)),where=prior_sum>0)
            later_team=later_roster[f'actual_{self.op}_team'].to_numpy()
            actual=np.divide(later_roster[f'actual_{self.op}'],later_team,
                             out=np.zeros(len(later_roster)),where=later_team>0)
            self.share_strength,self.share_validation=_best_share_blend(
                prior_share,provisional_share.predict(roster_features(later_roster)),
                actual,later_roster)
        self.share=FittedHead('regression_l1').fit(
            roster_features(roster),target,group_weight.to_numpy()*roster_recency)
        positive=roster[f'actual_{self.op}']>0
        exposure=roster.loc[positive,f'actual_{self.op}'].to_numpy()
        rate=roster.loc[positive,f'actual_{stat}'].to_numpy()/exposure
        self.rate=FittedHead().fit(roster_features(roster.loc[positive]),rate,np.minimum(exposure,40))
        self.league_rate=float(np.sum(roster.loc[positive,f'actual_{stat}'])/exposure.sum())
        self.league_rate_by_position={}
        for position,group in roster.loc[positive].groupby('position'):
            exp=group[f'actual_{self.op}'].sum()
            if exp>0:
                self.league_rate_by_position[str(position)]=float(group[f'actual_{stat}'].sum()/exp)
        self.efficiency_prior_opportunities=100.
        self.rate_model_strength=.25
        self.efficiency_validation=None
        early_positive=earlier_roster[f'actual_{self.op}']>0
        later_positive=later_roster[f'actual_{self.op}']>0
        if early_positive.sum()>=80 and later_positive.sum()>=32:
            early_exp=earlier_roster.loc[early_positive,f'actual_{self.op}'].to_numpy()
            early_rate=earlier_roster.loc[early_positive,f'actual_{stat}'].to_numpy()/early_exp
            provisional=FittedHead().fit(
                roster_features(earlier_roster.loc[early_positive]),early_rate,
                np.minimum(early_exp,40))
            early_global=float(earlier_roster.loc[early_positive,f'actual_{stat}'].sum()/early_exp.sum())
            early_by_position={}
            for position,group in earlier_roster.loc[early_positive].groupby('position'):
                exp=group[f'actual_{self.op}'].sum()
                if exp>0: early_by_position[str(position)]=float(group[f'actual_{stat}'].sum()/exp)
            later_frame=later_roster.loc[later_positive]
            observed_exp=later_frame[f'actual_{self.op}'].to_numpy(dtype=float)
            actual_yards=later_frame[f'actual_{stat}'].to_numpy(dtype=float)
            prior_exp=(numeric(later_frame,f'prior_{self.op}').fillna(0).to_numpy()
                       * numeric(later_frame,'known_games').fillna(0).to_numpy())
            prior_yards=(numeric(later_frame,f'prior_{self.stat}').fillna(0).to_numpy()
                         * numeric(later_frame,'known_games').fillna(0).to_numpy())
            league=np.asarray([early_by_position.get(str(p),early_global)
                               for p in later_frame.position],dtype=float)
            learned=provisional.predict(roster_features(later_frame))
            best=(np.inf,100.,0.)
            baseline=None
            for prior_strength in (20.,40.,75.,100.,150.,250.):
                eb=(prior_yards+prior_strength*league)/(prior_exp+prior_strength)
                for model_strength in (0.,.10,.20,.30,.45,.60):
                    pred_rate=(1-model_strength)*eb+model_strength*learned
                    mae=float(np.mean(np.abs(observed_exp*pred_rate-actual_yards)))
                    if prior_strength==100. and model_strength==.25:
                        baseline=mae
                    if mae<best[0]: best=(mae,prior_strength,model_strength)
            # A challenger earns influence only when it beats the old recipe.
            old_eb=(prior_yards+100*league)/(prior_exp+100)
            old_mae=float(np.mean(np.abs(observed_exp*(.75*old_eb+.25*learned)-actual_yards)))
            if best[0]<old_mae:
                self.efficiency_prior_opportunities=best[1]
                self.rate_model_strength=best[2]
            self.efficiency_validation={'rows':int(len(later_frame)),'old_recipe_mae':old_mae,
                'selected_mae':best[0],'prior_opportunities':self.efficiency_prior_opportunities,
                'model_strength':self.rate_model_strength}
        return self

    def predict(self,teams,roster,volume_strength=None,share_strength=None):
        roster=roster.copy().reset_index(drop=True)
        if teams.empty or roster.empty:
            return roster.iloc[:0].assign(expected_opportunity=[],projection=[],
                expected_efficiency=[],team_budget=[],unallocated_fraction=[])
        if (numeric(teams,'history_games').lt(3).any() or
            not np.isfinite(teams[['prior_pass_attempts_5','prior_carries_5']].to_numpy(dtype=float)).all()):
            raise ValueError('Team allocation requires complete prior team volume and three games')
        volume_override=volume_strength is not None
        volume_strength = getattr(self,'volume_strength',.25) if volume_strength is None else float(volume_strength)
        share_strength = getattr(self,'share_strength',.25) if share_strength is None else float(share_strength)
        if not (0 <= volume_strength <= 1 and 0 <= share_strength <= 1):
            raise ValueError('Team allocation strengths must be in [0,1]')
        budgets=teams[['game_id','team_abbr']].copy()
        for op in ('pass_attempts','carries'):
            baseline=numeric(teams,f'prior_{op}_5').to_numpy()
            learned=self.volume[op].predict(team_features(teams))
            selected=(volume_strength if volume_override else
                      (getattr(self,'volume_strength_by_op',{}) or {}).get(op,volume_strength))
            budgets[op]=np.maximum(0,(1-selected)*baseline+selected*learned)
        target_fraction=np.full(len(teams),self.target_fraction,dtype=float)
        if hasattr(self,'target_fraction_head'):
            learned=np.clip(self.target_fraction_head.predict(team_features(teams)),0.45,1.0)
            weight=float(getattr(self,'target_fraction_strength',0.0))
            target_fraction=(1-weight)*target_fraction+weight*learned
        budgets['target_fraction']=target_fraction
        budgets['targets']=budgets.pass_attempts*target_fraction
        reserve=np.full(len(teams),self.reserve,dtype=float)
        if hasattr(self,'reserve_head'):
            learned=np.clip(self.reserve_head.predict(allocation_features(teams,roster,self.op)),0,0.65)
            weight=float(getattr(self,'reserve_strength',0.0))
            reserve=(1-weight)*reserve+weight*learned
        budgets['unallocated_fraction']=np.clip(reserve,0,0.65)
        # Live feature frames can contain stale/current-game columns with these
        # names.  They are neither the team budget nor valid pregame labels and
        # would create _x/_y columns during the merge.
        roster=roster.drop(columns=[c for c in ('pass_attempts','carries','targets') if c in roster])
        roster=roster.merge(budgets,on=['game_id','team_abbr'],validate='many_to_one')
        prior=numeric(roster,f'prior_{self.op}').fillna(0).clip(lower=0)
        prior_sum=prior.groupby([roster.game_id,roster.team_abbr]).transform('sum')
        prior_share=np.divide(prior,prior_sum,out=np.zeros(len(roster)),where=prior_sum>0)
        learned_share=self.share.predict(roster_features(roster))
        score=np.maximum(0,(1-share_strength)*prior_share+share_strength*learned_share)
        # Only a timestamp-validated explicit out designation removes a player.
        score[numeric(roster,'injury_out').eq(1).to_numpy()]=0
        total=pd.Series(score).groupby([roster.game_id,roster.team_abbr]).transform('sum').to_numpy()
        shares=np.divide(score,total,out=np.zeros(len(score)),where=total>0)
        opportunity=(roster[self.op].to_numpy()
                     *(1-roster.unallocated_fraction.to_numpy(dtype=float))*shares)
        prior_exposure=prior.to_numpy()*numeric(roster,'known_games').fillna(0).to_numpy()
        prior_yards=numeric(roster,f'prior_{self.stat}').fillna(0).to_numpy()*numeric(roster,'known_games').fillna(0).to_numpy()
        # Efficiency is deliberately stable. Opportunity is allowed to react to
        # current role; player skill is shrunk toward a position/league prior.
        league=np.asarray([
            getattr(self,'league_rate_by_position',{}).get(str(position),self.league_rate)
            for position in roster.position
        ],dtype=float)
        prior_strength=float(getattr(self,'efficiency_prior_opportunities',100.))
        model_strength=float(getattr(self,'rate_model_strength',.25))
        eb_rate=(prior_yards+prior_strength*league)/(prior_exposure+prior_strength)
        rate=(1-model_strength)*eb_rate+model_strength*self.rate.predict(roster_features(roster))
        return roster.assign(expected_opportunity=opportunity,projection=opportunity*rate,
            expected_efficiency=rate,team_budget=roster[self.op])

    @staticmethod
    def _snapshot_inputs(frame):
        """Translate the live, full-roster snapshot into the leakage-safe contract."""
        if frame.empty:
            return pd.DataFrame(), pd.DataFrame()
        roster=frame.copy().reset_index(drop=True)
        for name in COUNTS+YARDS:
            roster[f'prior_{name}']=numeric(roster,f'{name}_avg_5')
            roster[f'prior_{name}_std']=numeric(roster,f'{name}_std_5')
        for name in PLAYER_USAGE:
            roster[f'prior_{name}']=numeric(roster,f'{name}_avg_5')
            roster[f'prior_{name}_std']=numeric(roster,f'{name}_std_5')
        roster['known_games']=numeric(roster,'n_games_prev_10')
        roster['days_since_seen']=numeric(roster,'rest_days')
        status=roster.get('injury_report_status',pd.Series(None,index=roster.index)).astype('string').str.lower()
        observed=numeric(roster,'same_week_injury_context_known').eq(1)
        roster['injury_out']=np.where(observed,status.isin(('out','injured reserve')),np.nan)
        roster['depth_rank']=numeric(roster,'depth_pos_rank')
        roster['depth_movement']=numeric(roster,'previous_depth_pos_rank')-roster['depth_rank']
        roster['starter_confidence']=numeric(roster,'starter_confidence')
        roster['teammate_absences']=numeric(roster,'same_week_teammate_skill_injury_count')
        roster['teammate_limited']=numeric(roster,'same_week_teammate_skill_injury_score')
        keys=['game_id','team_abbr']
        records=[]
        for (game,team),group in roster.groupby(keys,sort=False,dropna=False):
            first=group.iloc[0]
            row={'game_id':game,'team_abbr':team,'opponent_abbr':first.get('opponent_abbr'),
                 'game_date_et':first.get('game_date_et'),'season':first.get('season'),
                 'week':first.get('week'),'history_games':float(numeric(group,'n_games_prev_5').max())}
            for name in COUNTS+YARDS:
                # A stat-specific live subframe may omit the players who produce
                # the team volume (for example, receivers contain no QB attempts).
                # The snapshot repeats full-roster five-game sums on every row.
                repeated=numeric(group,f'team_player_{name}_avg5_sum').dropna()
                full_team_5=float(repeated.iloc[0]) if len(repeated) else np.nan
                for window in (3,5,10):
                    values=numeric(group,f'{name}_avg_{window}').clip(lower=0)
                    subtotal=float(values.sum(min_count=1)) if values.notna().any() else np.nan
                    if np.isfinite(full_team_5) and (window==5 or not np.isfinite(subtotal) or subtotal<=0):
                        subtotal=full_team_5
                    row[f'prior_{name}_{window}']=subtotal
            row['rest_days']=float(numeric(group,'rest_days').median()) if numeric(group,'rest_days').notna().any() else np.nan
            row['opponent_prior_plays']=np.nan
            for name in TEAM_CONTEXT:
                values=numeric(group,name)
                row[name]=float(values.dropna().iloc[0]) if values.notna().any() else np.nan
            records.append(row)
        return pd.DataFrame(records),roster

    def predict_snapshot(self,frame,volume_strength=None,share_strength=None):
        """Allocate one coherent team budget across the current full roster."""
        teams,roster=self._snapshot_inputs(frame)
        if teams.empty or roster.empty:
            return roster.iloc[:0]
        return self.predict(teams,roster,volume_strength,share_strength)
