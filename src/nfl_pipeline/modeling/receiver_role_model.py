"""Role-aware receiving challenger. No production publication or bet selection."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from nfl_pipeline.context_contract import safe_time, validate_evidence
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.modeling.challenger_models import FittedHead, numeric

CONTRACT = 'nfl-receiver-role-v1'
ROLES = ('WR', 'TE', 'RB')


def code_fingerprint():
    here = Path(__file__)
    paths = [here] + [here.with_name(name) for name in (
        'train_receiver_role.py', 'challenger_models.py', 'component_validation.py',
        'evaluation.py', 'train_accuracy_challengers.py', 'score_accuracy_components.py')]
    paths.append(here.parent.parent / 'context_contract.py')
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()


def load_role_history(cutoff=None):
    """Corrected final historical stats; optional cutoff excludes later revisions."""
    with create_engine(PG_DSN).connect() as conn:
        conn.execute(text("SET statement_timeout='120s'"))
        return pd.read_sql(text("""
            SELECT DISTINCT ON (p.game_id,p.player_id)
                p.*,g.start_ts_utc,g.status
            FROM raw.nfl_player_gamelogs p JOIN raw.nfl_games g USING(game_id)
            WHERE g.status='final'
              AND (CAST(:cutoff AS timestamptz) IS NULL OR (p.updated_at_utc<=CAST(:cutoff AS timestamptz)
                                      AND g.start_ts_utc<CAST(:cutoff AS timestamptz)))
            ORDER BY p.game_id,p.player_id,p.updated_at_utc DESC
        """), conn, params={'cutoff': cutoff})


def finite(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (ValueError, TypeError):
        return np.nan


def weighted_mean(rows, key, weights):
    values = np.array([finite(r.get(key)) for r in rows])
    valid = np.isfinite(values) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else np.nan


def context_inputs(row):
    evidence = row.get('context_evidence')
    lock = row.get('prediction_context_cutoff_utc') or row.get('start_ts_utc')
    evidence = validate_evidence(evidence, lock) if lock else None
    out = {k: np.nan for k in ('depth', 'depth_change', 'starter', 'injury_out',
        'injury_limited', 'teammate_absences')}
    if evidence:
        if safe_time(evidence.get('depth_observed_at')):
            out.update(depth=finite(evidence.get('depth_rank')),
                depth_change=finite(evidence.get('depth_movement')),
                starter=finite(evidence.get('expected_starter_from_depth')))
        if safe_time(evidence.get('injury_observed_at')):
            status = evidence.get('injury_status')
            practice = evidence.get('practice_status')
            if status is not None:
                out['injury_out'] = float(str(status).lower() in ('out', 'injured reserve'))
            if status is not None or practice is not None:
                out['injury_limited'] = float(any(s in str(status).lower() + ' ' + str(practice).lower()
                    for s in ('questionable', 'doubtful', 'limited', 'did not participate')))
            out['teammate_absences'] = finite(evidence.get('teammate_injury_count'))
    return {'rr_' + k: v for k, v in out.items()}


class RoleHistory:
    """Read-before-update, date-batched player and team performance history."""
    def __init__(self):
        self.players = defaultdict(lambda: deque(maxlen=40))
        self.teams = defaultdict(lambda: deque(maxlen=20))

    def update(self, records):
        grouped = defaultdict(list)
        for row in records:
            grouped[(row['game_id'], row['team_abbr'])].append(dict(row))
        for (_, team), group in grouped.items():
            attempts = [finite(r.get('pass_attempts')) for r in group]
            total = float(np.nansum(attempts)) if np.isfinite(attempts).any() else np.nan
            self.teams[team].append({'game_id': group[0]['game_id'],
                'game_date_et': group[0]['game_date_et'], 'pass_attempts': total})
            for row in group:
                # Zero-target appearances still inform opportunity, never YPT skill.
                if not (finite(row.get('offense_snaps')) > 0 or finite(row.get('targets')) > 0):
                    continue
                old = list(self.players[str(row['player_id'])])
                prior_snaps = np.array([finite(r.get('offense_snaps')) for r in old[-5:]])
                prior_snaps = prior_snaps[np.isfinite(prior_snaps)]
                normal = float(np.quantile(prior_snaps, .75)) if len(prior_snaps) >= 3 else np.nan
                snaps = finite(row.get('offense_snaps'))
                row['partial'] = float(snaps < .5 * normal) if np.isfinite(normal) and normal > 0 and np.isfinite(snaps) else np.nan
                row['team_attempts'] = total
                row['target_share_observed'] = finite(row.get('targets')) / total if total > 0 else np.nan
                self.players[str(row['player_id'])].append(row)

    def features(self, request, latest_weight=1.):
        if not 0 <= latest_weight <= 1:
            raise ValueError('Latest-game weight must be between zero and one')
        history = list(self.players.get(str(request['player_id']), ()))
        last_game = history[-1]['game_id'] if history else None
        weights = np.ones(len(history))
        if len(weights):
            weights[-1] = latest_weight
        keep = weights > 0
        history = [r for r, ok in zip(history, keep) if ok]
        weights = weights[keep]
        team = request['team_abbr']
        out = {'rr_history_games': len(history), **context_inputs(request)}
        for pos in ROLES:
            out['rr_position_' + pos] = float(request.get('position') == pos)
        for window in (3, 5, 20):
            rows = history[-window:]; w = weights[-window:]
            for key in ('targets', 'target_share_observed', 'offense_snaps', 'partial'):
                out[f'rr_{key}_{window}'] = weighted_mean(rows, key, w)
        out['rr_team_changed'] = float(bool(history) and history[-1]['team_abbr'] != team)
        out['rr_recent_team_transition'] = float(any(r['team_abbr'] != team for r in history[-5:]))
        out['rr_current_team_games'] = sum(r['team_abbr'] == team for r in history)
        out['rr_season_changed'] = float(bool(history) and history[-1]['season'] != request['season'])
        out['rr_days_since_game'] = (pd.Timestamp(request['game_date_et']) - pd.Timestamp(history[-1]['game_date_et'])).days if history else np.nan
        out['rr_last_partial'] = finite(history[-1].get('partial')) if history else np.nan
        for window in (5, 20, 40):
            rows = history[-window:]; w = weights[-window:]
            t = np.array([finite(r.get('targets')) for r in rows])
            y = np.array([finite(r.get('receiving_yards')) for r in rows])
            valid = np.isfinite(y) & np.isfinite(t) & (t > 0)
            out[f'rr_exposure_{window}'] = float(np.sum(t[valid] * w[valid]))
            out[f'rr_yards_{window}'] = float(np.sum(y[valid] * w[valid]))
            a = np.array([finite(r.get('receiving_air_yards')) for r in rows])
            av = valid & np.isfinite(a)
            out[f'rr_adot_{window}'] = float(np.sum(a[av] * w[av]) / np.sum(t[av] * w[av])) if av.any() else np.nan
        for side, name in ((team, 'team'), (request.get('opponent_abbr'), 'opponent')):
            past = list(self.teams.get(side, ()))
            tw = np.array([latest_weight if r['game_id'] == last_game else 1. for r in past])
            past = [r for r, ok in zip(past, tw > 0) if ok]; tw = tw[tw > 0]
            for window in (5, 20):
                out[f'rr_{name}_passes_{window}'] = weighted_mean(past[-window:], 'pass_attempts', tw[-window:])
        return out


def role_features(raw, requests, latest_weight=1.):
    if requests.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Role features require one row per player-game')
    raw = raw.drop_duplicates(['game_id', 'player_id']).copy()
    state = RoleHistory(); output = {}
    dates = sorted(set(raw.game_date_et) | set(requests.game_date_et))
    history_days = {d: g for d, g in raw.groupby('game_date_et')}
    request_days = {d: g for d, g in requests.groupby('game_date_et')}
    for day in dates:
        if day in request_days:
            for row in request_days[day].to_dict('records'):
                output[(row['game_id'], str(row['player_id']))] = state.features(row, latest_weight)
        if day in history_days:
            state.update(history_days[day].to_dict('records'))
    return pd.DataFrame([output[(r.game_id, str(r.player_id))] for r in requests.itertuples()], index=requests.index)


def inputs(frame, kind):
    columns = [c for c in frame if c.startswith('rr_')]
    if kind == 'team':
        columns = [c for c in columns if c.startswith(('rr_team_passes', 'rr_opponent_passes', 'rr_season_changed'))]
    elif kind == 'workload':
        columns = [c for c in columns if not c.startswith(('rr_yards_', 'rr_exposure_', 'rr_adot_'))]
    elif kind == 'rate':
        columns = [c for c in columns if c.startswith(('rr_yards_', 'rr_exposure_', 'rr_adot_', 'rr_position_', 'rr_team_changed'))]
    out = frame[sorted(columns)].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    return pd.concat([out, out.isna().astype(float).add_suffix('__missing')], axis=1)


@dataclass(frozen=True)
class RoleConfig:
    latest_weight: float = 1.
    prior_targets: float = 80.
    recent_role_weight: float = .25


class RoleAwareReceiver:
    def __init__(self, config=RoleConfig()):
        self.config = config

    def league_rate(self, frame):
        return frame.position.map(self.position_rates).fillna(self.global_rate).to_numpy()

    def efficiency_prior(self, frame):
        exposure = numeric(frame, 'rr_exposure_40').fillna(0).to_numpy()
        yards = numeric(frame, 'rr_yards_40').fillna(0).to_numpy()
        return (yards + self.config.prior_targets * self.league_rate(frame)) / (exposure + self.config.prior_targets)

    def workload_prior(self, frame):
        passes = np.maximum(1., .5 * numeric(frame, 'rr_team_passes_5').fillna(self.team_mean).to_numpy()
            + .5 * self.team.predict(inputs(frame, 'team')))
        long = numeric(frame, 'rr_target_share_observed_20').fillna(self.share_mean).to_numpy()
        short = numeric(frame, 'rr_target_share_observed_3').fillna(pd.Series(long, index=frame.index)).to_numpy()
        n = numeric(frame, 'rr_history_games').fillna(0).to_numpy()
        long = (n * long + 5 * self.share_mean) / (n + 5)
        # The recent-role coefficient is chosen on earlier weeks, never outer folds.
        share = (1 - self.config.recent_role_weight) * long + self.config.recent_role_weight * short
        return passes, passes * np.clip(share, 0, 1)

    def fit(self, frame):
        exposure = numeric(frame, 'targets').to_numpy()
        yards = numeric(frame, 'receiving_yards').to_numpy()
        positive = np.isfinite(exposure) & np.isfinite(yards) & (exposure > 0)
        self.global_rate = float(yards[positive].sum() / exposure[positive].sum())
        self.position_rates = {}
        for position in ROLES:
            mask = positive & frame.position.eq(position).to_numpy()
            self.position_rates[position] = float((yards[mask].sum() + 100 * self.global_rate) / (exposure[mask].sum() + 100))
        teams = frame.drop_duplicates(['game_id', 'team_abbr'])
        teams = teams.loc[numeric(teams, 'team_actual_pass_attempts').notna()]
        self.team_mean = float(teams.team_actual_pass_attempts.mean())
        params = {'n_estimators': 60, 'min_child_samples': 100, 'num_leaves': 7, 'reg_lambda': 30}
        self.team = FittedHead('poisson', params=params).fit(inputs(teams, 'team'), teams.team_actual_pass_attempts)
        self.share_mean = float(numeric(frame, 'rr_target_share_observed_20').mean())
        _, base_targets = self.workload_prior(frame)
        self.workload = FittedHead(params=params).fit(inputs(frame, 'workload'), exposure - base_targets)
        prior = self.efficiency_prior(frame)
        # A one-target game gets one unit of skill evidence, not one full game's weight.
        self.rate = FittedHead(params=params).fit(inputs(frame.loc[positive], 'rate'),
            yards[positive] / exposure[positive] - prior[positive], exposure[positive])
        return self

    def components(self, frame):
        passes, prior = self.workload_prior(frame)
        targets = np.clip(prior + .5 * self.workload.predict(inputs(frame, 'workload')), 0, passes)
        rate = np.maximum(0., self.efficiency_prior(frame) + .25 * self.rate.predict(inputs(frame, 'rate')))
        return {'team_pass_attempts': passes, 'targets': targets, 'yards_per_target': rate,
            'center': targets * rate}

    def predict(self, frame):
        return self.components(frame)['center']
