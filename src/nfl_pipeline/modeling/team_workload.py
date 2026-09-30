"""Pregame team budgets and roster allocations, independent of offered props."""
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sqlalchemy import create_engine,text

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.modeling.challenger_models import FittedHead, numeric, OPPORTUNITY

COUNTS=('pass_attempts','carries','targets')
YARDS=('passing_yards','rushing_yards','receiving_yards')


def load_history():
    with create_engine(PG_DSN).connect() as conn:
        conn.execute(text("SET statement_timeout='120s'"))
        return pd.read_sql(text("""SELECT DISTINCT ON (p.game_id,p.team_abbr,p.player_id)
            p.game_id,p.player_id,p.team_abbr,p.opponent_abbr,p.position,
            g.season,g.week,g.game_date_et,p.pass_attempts,p.carries,p.targets,
            p.passing_yards,p.rushing_yards,p.receiving_yards
            FROM raw.nfl_player_gamelogs p JOIN raw.nfl_games g USING(game_id)
            WHERE g.status='final'
            ORDER BY p.game_id,p.team_abbr,p.player_id,p.updated_at_utc DESC"""),conn)


class PregameTeamHistory:
    """Build every roster from earlier games, never that game's participants."""
    def __init__(self):
        self.teams=defaultdict(lambda:deque(maxlen=10))

    def context(self, game, team, opponent, day, season, week):
        past=list(self.teams.get(team,[])); defense=list(self.teams.get(opponent,[]))
        row={'game_id':game,'team_abbr':team,'opponent_abbr':opponent,'game_date_et':day,
             'season':season,'week':week,'history_games':len(past)}
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
            roster.append(member)
        return row,roster

    def update(self, team, day, records):
        totals={name:float(sum(r[name] for r in records)) for name in COUNTS+YARDS}
        self.teams[team].append({**totals,'game_date_et':day,'players':records})


def prepare_history(raw):
    raw=raw.drop_duplicates(['game_id','team_abbr','player_id']).copy()
    for name in COUNTS+YARDS:
        raw[name]=pd.to_numeric(raw[name],errors='coerce')
    # Missing source totals cannot be taught as a measured zero-volume game.
    raw=raw.dropna(subset=list(COUNTS)+list(YARDS))
    state=PregameTeamHistory(); teams=[]; rosters=[]
    for day,games in raw.sort_values('game_date_et').groupby('game_date_et',sort=True):
        pending=[]
        for (game,team),group in games.groupby(['game_id','team_abbr']):
            first=group.iloc[0]
            row,pool=state.context(game,team,first.opponent_abbr,day,int(first.season),int(first.week))
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
    names+=['rest_days','opponent_prior_plays','history_games']
    return pd.DataFrame({c:numeric(frame,c) for c in names},index=frame.index)


def roster_features(frame):
    names=[f'prior_{n}{suffix}' for n in COUNTS+YARDS for suffix in ('','_std')]
    names+=['known_games','days_since_seen','injury_out','depth_rank']
    out=pd.DataFrame({c:numeric(frame,c) for c in names},index=frame.index)
    for position in ('QB','RB','WR','TE'):
        out['position_'+position]=(frame.position==position).astype(float)
    return pd.concat([out,out.isna().astype(float).add_suffix('_missing')],axis=1)


class TeamAllocationModel:
    def fit(self,teams,roster,stat):
        self.stat=stat; self.op=OPPORTUNITY[stat]
        teams=teams.loc[teams.history_games>=3].copy()
        roster=roster.merge(teams[['game_id','team_abbr',f'actual_{self.op}']],on=['game_id','team_abbr'],
            suffixes=('','_team'),validate='many_to_one')
        self.volume={op:FittedHead('poisson').fit(team_features(teams),teams[f'actual_{op}']) for op in ('pass_attempts','carries')}
        valid=teams.actual_pass_attempts>0
        self.target_fraction=float(np.clip(teams.loc[valid,'actual_targets'].sum()/teams.loc[valid,'actual_pass_attempts'].sum(),0,1))
        self.reserve=float(np.clip(teams[f'unallocated_{self.op}'].mean(),0,1))
        denominator=roster[f'actual_{self.op}_team'].to_numpy()
        target=np.divide(roster[f'actual_{self.op}'],denominator,out=np.zeros(len(roster)),where=denominator>0)
        group_weight=1/roster.groupby(['game_id','team_abbr']).player_id.transform('count')
        self.share=FittedHead('poisson').fit(roster_features(roster),target,group_weight)
        positive=roster[f'actual_{self.op}']>0
        exposure=roster.loc[positive,f'actual_{self.op}'].to_numpy()
        rate=roster.loc[positive,f'actual_{stat}'].to_numpy()/exposure
        self.rate=FittedHead().fit(roster_features(roster.loc[positive]),rate,np.minimum(exposure,40))
        self.league_rate=float(np.sum(roster.loc[positive,f'actual_{stat}'])/exposure.sum())
        return self

    def predict(self,teams,roster):
        roster=roster.copy().reset_index(drop=True)
        if teams.empty or roster.empty:
            return roster.iloc[:0].assign(expected_opportunity=[],projection=[],
                expected_efficiency=[],team_budget=[],unallocated_fraction=[])
        if (numeric(teams,'history_games').lt(3).any() or
            not np.isfinite(teams[['prior_pass_attempts_5','prior_carries_5']].to_numpy(dtype=float)).all()):
            raise ValueError('Team allocation requires complete prior team volume and three games')
        budgets=teams[['game_id','team_abbr']].copy()
        for op in ('pass_attempts','carries'):
            baseline=numeric(teams,f'prior_{op}_5').to_numpy()
            budgets[op]=np.maximum(0,.5*baseline+.5*self.volume[op].predict(team_features(teams)))
        budgets['targets']=budgets.pass_attempts*self.target_fraction
        roster=roster.merge(budgets,on=['game_id','team_abbr'],validate='many_to_one')
        prior=numeric(roster,f'prior_{self.op}').fillna(0).clip(lower=0)
        prior_sum=prior.groupby([roster.game_id,roster.team_abbr]).transform('sum')
        prior_share=np.divide(prior,prior_sum,out=np.zeros(len(roster)),where=prior_sum>0)
        score=np.maximum(0,.5*prior_share+.5*self.share.predict(roster_features(roster)))
        # Only a timestamp-validated explicit out designation removes a player.
        score[numeric(roster,'injury_out').eq(1).to_numpy()]=0
        total=pd.Series(score).groupby([roster.game_id,roster.team_abbr]).transform('sum').to_numpy()
        shares=np.divide(score,total,out=np.zeros(len(score)),where=total>0)
        opportunity=roster[self.op].to_numpy()*(1-self.reserve)*shares
        prior_exposure=prior.to_numpy()*numeric(roster,'known_games').fillna(0).to_numpy()
        prior_yards=numeric(roster,f'prior_{self.stat}').fillna(0).to_numpy()*numeric(roster,'known_games').fillna(0).to_numpy()
        eb_rate=(prior_yards+30*self.league_rate)/(prior_exposure+30)
        rate=.5*eb_rate+.5*self.rate.predict(roster_features(roster))
        return roster.assign(expected_opportunity=opportunity,projection=opportunity*rate,
            expected_efficiency=rate,team_budget=roster[self.op],unallocated_fraction=self.reserve)
