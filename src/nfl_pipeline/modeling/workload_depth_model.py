"""Player-game workload states and receiving-depth challenger, never production."""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from nfl_pipeline.modeling.challenger_models import FittedHead, numeric
from nfl_pipeline.modeling.receiver_role_model import context_inputs, finite

STATS = {'receiving_yards': 'targets', 'rushing_yards': 'carries'}
CONTRACT = 'workload-depth-v2-independent-player'
PARAMS = dict(n_estimators=70, max_depth=3, num_leaves=7,
              min_child_samples=90, reg_lambda=30, learning_rate=.035)


def historical_features(raw, requests):
    """Batch by date: today's outcomes cannot affect any of today's features."""
    if requests.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Expected one row per player-game')
    raw = raw.drop_duplicates(['game_id', 'player_id']).copy()
    players = defaultdict(lambda: deque(maxlen=40))
    teams = defaultdict(lambda: deque(maxlen=20))
    output = {}
    histories = {d: g for d, g in raw.groupby('game_date_et')}
    queries = {d: g for d, g in requests.groupby('game_date_et')}

    def average(rows, key):
        v = np.array([finite(r.get(key)) for r in rows])
        return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan

    for day in sorted(set(histories) | set(queries)):
        for r in queries.get(day, pd.DataFrame()).to_dict('records'):
            past = list(players[str(r['player_id'])])
            f = {'wd_' + k[3:]: v for k, v in context_inputs(r).items()}
            f.update(wd_history=len(past), wd_home=finite(r.get('is_home')),
                     wd_rest=finite(r.get('rest_days')),
                     wd_team_change=float(bool(past) and past[-1]['team_abbr'] != r['team_abbr']))
            for pos in ('QB', 'RB', 'WR', 'TE'):
                f['wd_position_' + pos] = float(r.get('position') == pos)
            for n in (3, 5, 20):
                for key in ('targets', 'carries', 'offense_snaps', 'target_share', 'carry_share', 'partial'):
                    f[f'wd_{key}_{n}'] = average(past[-n:], key)
                for side, team in (('team', r['team_abbr']), ('opponent', r.get('opponent_abbr'))):
                    for key in ('pass_attempts', 'carries'):
                        f[f'wd_{side}_{key}_{n}'] = average(list(teams[team])[-n:], key)
            # Exposure-weighted, longer-term skill. A one-target game is not a full game's evidence.
            for n in (5, 20, 40):
                tail = past[-n:]
                for op, stat in (('targets', 'receiving_yards'), ('carries', 'rushing_yards')):
                    valid = [x for x in tail if finite(x.get(op)) > 0 and np.isfinite(finite(x.get(stat)))]
                    f[f'wd_{op}_exposure_{n}'] = sum(finite(x[op]) for x in valid)
                    f[f'wd_{stat}_sum_{n}'] = sum(finite(x[stat]) for x in valid)
                for label, numerator, denominator in (
                    ('depth', 'receiving_air_yards', 'targets'),
                    ('catch', 'receptions', 'targets'),
                    ('air_rate', 'completed_air_yards', 'targets'),
                    ('yac_rate', 'receiving_yards_after_catch', 'receptions')):
                    valid = [x for x in tail if finite(x.get(denominator)) > 0 and np.isfinite(finite(x.get(numerator)))]
                    f[f'wd_{label}_exposure_{n}'] = sum(finite(x[denominator]) for x in valid)
                    f[f'wd_{label}_sum_{n}'] = sum(finite(x[numerator]) for x in valid)
            output[(r['game_id'], str(r['player_id']))] = f
        for (_, team), group in histories.get(day, pd.DataFrame()).groupby(['game_id', 'team_abbr']) if day in histories else ():
            totals = {k: float(group[k].sum(min_count=1)) for k in ('pass_attempts', 'carries')}
            teams[team].append(totals)
            for r in group.to_dict('records'):
                if not any(finite(r.get(k)) > 0 for k in ('offense_snaps', 'targets', 'carries', 'pass_attempts')):
                    continue
                prior = list(players[str(r['player_id'])])[-5:]
                snaps = np.array([finite(x.get('offense_snaps')) for x in prior])
                snaps = snaps[np.isfinite(snaps)]
                normal = np.quantile(snaps, .75) if len(snaps) >= 3 else np.nan
                s = finite(r.get('offense_snaps'))
                r['partial'] = float(s < .5 * normal) if normal > 0 and np.isfinite(s) else np.nan
                for key, op, total in (('target_share', 'targets', totals['pass_attempts']), ('carry_share', 'carries', totals['carries'])):
                    r[key] = finite(r.get(op)) / total if total > 0 else np.nan
                r['completed_air_yards'] = finite(r.get('receiving_yards')) - finite(r.get('receiving_yards_after_catch'))
                players[str(r['player_id'])].append(r)
    return pd.DataFrame([output[(r.game_id, str(r.player_id))] for r in requests.itertuples()], index=requests.index)


def inputs(frame, kind, stat):
    op = STATS[stat]
    cols = [c for c in frame if c.startswith('wd_')]
    common = ('wd_position_', 'wd_team_change', 'wd_home', 'wd_rest')
    if kind == 'team':
        cols = [c for c in cols if c.startswith(('wd_team_', 'wd_opponent_')) and c != 'wd_team_change']
    elif kind == 'workload':
        cols = [c for c in cols if c.startswith(common + (
            f'wd_{op}_', 'wd_target_share_' if op == 'targets' else 'wd_carry_share_',
            'wd_offense_snaps_', 'wd_partial_', 'wd_history', 'wd_team_', 'wd_opponent_',
            'wd_depth', 'wd_starter', 'wd_injury_', 'wd_teammate_'))
            and '_sum_' not in c and '_exposure_' not in c
            and not c.startswith('wd_depth_')] + [c for c in cols if c in ('wd_depth', 'wd_depth_change')]
    else:
        prefixes = ('wd_depth_', 'wd_catch_', 'wd_air_rate_', 'wd_yac_rate_', 'wd_targets_exposure_', 'wd_receiving_yards_sum_') if op == 'targets' else ('wd_carries_exposure_', 'wd_rushing_yards_sum_')
        cols = [c for c in cols if c.startswith(common + prefixes)]
    X = frame[sorted(set(cols))].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    return pd.concat([X, X.isna().astype(float).add_suffix('__missing')], axis=1)


def states(frame, stat):
    op = STATS[stat]
    prior = numeric(frame, f'wd_{op}_5').fillna(0).to_numpy()
    actual = numeric(frame, op).to_numpy()
    high = 8 if op == 'targets' else 18
    return np.where(actual < np.maximum(1, .5 * prior), 0,
                    np.where(actual >= np.maximum(high, 1.4 * prior), 2, 1))


class WorkloadDepthModel:
    def __init__(self, rate_weight=.25):
        self.rate_weight = rate_weight

    def _prior(self, frame, name):
        exposure = numeric(frame, f'wd_{name}_exposure_40').fillna(0).to_numpy()
        sums = numeric(frame, f'wd_{name}_sum_40').fillna(0).to_numpy()
        prior = frame.position.map(self.priors[name]).fillna(self.global_priors[name]).to_numpy()
        return (sums + 60 * prior) / (exposure + 60)

    def _rushing_prior(self, frame):
        exposure = numeric(frame, 'wd_carries_exposure_40').fillna(0).to_numpy()
        sums = numeric(frame, 'wd_rushing_yards_sum_40').fillna(0).to_numpy()
        return (sums + 80 * self.rush_prior) / (exposure + 80)

    def fit(self, frame, stat):
        self.stat = stat
        op = STATS[stat]
        volume = 'team_actual_pass_attempts' if op == 'targets' else 'team_actual_carries'
        team = frame.drop_duplicates(['game_id', 'team_abbr']).dropna(subset=[volume])
        self.team = FittedHead('poisson', params=PARAMS).fit(inputs(team, 'team', stat), team[volume])
        state = states(frame, stat)
        self.state_prior = np.bincount(state, minlength=3) / len(state)
        X = inputs(frame, 'workload', stat)
        self.state = FittedHead(classifier=True, params=PARAMS).fit(X, state)
        self.share = []
        for s in range(3):
            mask = (state == s) & numeric(frame, volume).gt(0).to_numpy()
            if not mask.any():
                mask = numeric(frame, volume).gt(0).to_numpy()
            y = (numeric(frame, op) / numeric(frame, volume)).to_numpy()
            self.share.append(FittedHead(params=PARAMS).fit(X.loc[mask], y[mask]))
        Xr = inputs(frame, 'rate', stat)
        if op == 'targets':
            self.priors = {}; self.global_priors = {}; self.heads = {}
            for name, numerator, denominator in (
                ('depth', 'receiving_air_yards', 'targets'),
                ('catch', 'receptions', 'targets'),
                ('air_rate', 'completed_air_yards', 'targets'),
                ('yac_rate', 'receiving_yards_after_catch', 'receptions')):
                exposure = numeric(frame, denominator).to_numpy()
                values = numeric(frame, numerator).to_numpy()
                mask = np.isfinite(exposure) & (exposure > 0) & np.isfinite(values)
                self.global_priors[name] = float(values[mask].sum() / exposure[mask].sum())
                self.priors[name] = {}
                for pos in ('WR', 'TE', 'RB'):
                    use = mask & frame.position.eq(pos).to_numpy()
                    self.priors[name][pos] = float((values[use].sum() + 100 * self.global_priors[name]) / (exposure[use].sum() + 100))
                prior = self._prior(frame, name)
                self.heads[name] = FittedHead(params=PARAMS).fit(Xr.loc[mask], values[mask] / exposure[mask] - prior[mask], exposure[mask])
        else:
            exposure = numeric(frame, op).to_numpy()
            values = numeric(frame, stat).to_numpy()
            mask = np.isfinite(values) & (exposure > 0)
            self.rush_prior = float(values[mask].sum() / exposure[mask].sum())
            prior = self._rushing_prior(frame)
            self.rate = FittedHead(params=PARAMS).fit(Xr.loc[mask], values[mask] / exposure[mask] - prior[mask], exposure[mask])
        return self

    def components(self, frame):
        op = STATS[self.stat]
        volume = np.maximum(1, self.team.predict(inputs(frame, 'team', self.stat)))
        X = inputs(frame, 'workload', self.stat)
        weights = .95 * self.state.probabilities(X, 3) + .05 * self.state_prior
        opportunity = np.column_stack([h.predict(X).clip(0, 1) * volume for h in self.share])
        # A candidate/participant list is not a verified pregame roster. Never
        # renormalize shares over it: adding an offered player must change nothing.
        unavailable = numeric(frame, 'wd_injury_out').eq(1).to_numpy()
        opportunity[unavailable] = 0.
        weights[unavailable] = [1., 0., 0.]
        Xr = inputs(frame, 'rate', self.stat)
        details = {}
        if op == 'targets':
            for name, head in self.heads.items():
                details[name] = self._prior(frame, name) + self.rate_weight * head.predict(Xr)
            details['catch'] = details['catch'].clip(0, 1)
            details['yac_rate'] = details['yac_rate'].clip(0)
            rate = (details['air_rate'] + details['catch'] * details['yac_rate']).clip(0)
        else:
            rate = (self._rushing_prior(frame) + self.rate_weight * self.rate.predict(Xr)).clip(0)
        return dict(centers=opportunity * rate[:, None], weights=weights, opportunities=opportunity,
                    team_volume=volume, rate=rate, **details)

    def predict(self, frame):
        c = self.components(frame)
        return (c['centers'] * c['weights']).sum(axis=1)
