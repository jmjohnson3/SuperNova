"""New information about tonight's minutes for the NBA prop challenger. Research only.

The pinned model knows how much of a team's rotation is out tonight, but not who, and not whether that is news:
a player's recent minutes already include the games a teammate has missed. These inputs say what changed.

    absence     which rotation teammates are out tonight against the player's recent games (newly out, back),
                weighted by how alike they are, and how his minutes and per-minute numbers moved in earlier
                games this season with and without each of them
    returning   the player's own lay-off: days off, team games missed, games since he came back and his level
                before he went out (minutes limits after an injury)
    role        his minutes in earlier games as a starter and off the bench, the one that matches tonight's
                lineup, and whether tonight's role is a change

Everything is known before the game: earlier box scores, and tonight's lineup file (starters and inactives).
Games before the lineup files (2023-24) fall back to "did not play", as the pinned model does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nba_props import model

STATS = model.STATS
ROTATION_MINUTES = model.ROTATION_MINUTES
ROTATION_DAYS = model.ROTATION_DAYS
SHRINK = 2.0                   # games on each side before a with/without difference counts for half
EXPOSURE_HALFLIFE = 3          # matches the recent-minutes average the adjustment is measured against
LONG_ABSENCE = 3               # team games missed that count as a lay-off
EPISODE_GAMES = 10             # games after a lay-off during which the return inputs stay on
PROFILE = {"reb": (0.17, 0.08), "ast": (0.10, 0.07), "fg3m": (0.045, 0.035)}   # per-minute centre and scale, fixed
GROUPS = {
    "absence": ["wo_min_adj", "wo_support", "absent_new_min", "absent_back_min", "absent_new_sim", "absent_back_sim"]
               + [f"wo_{s}_rate_adj" for s in STATS],
    "returning": ["days_off", "team_games_missed", "games_since_return", "absence_len", "min_pre_absence", "return_gap"],
    "role": ["min_as_starter", "min_as_bench", "min_in_role", "role_min_gap", "starter_flip", "games_in_role", "team_new_starters"],
}
FEATURES = [c for cols in GROUPS.values() for c in cols]


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)


# --- who is out ------------------------------------------------------------------------------------------------
def candidates(df: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game and rotation player of that team: `out` says he was declared out before the game.

    A rotation player averaged ROTATION_MINUTES at his last appearance, which was for this team. With a lineup
    file he must be listed in it and out = Inactive; without one he must have appeared within ROTATION_DAYS and
    out = did not play."""
    d = _sorted(df)
    p = d[d.minutes > 0]
    g = p.groupby("player_id")
    ewm = lambda col, hl: g[col].transform(lambda s: s.ewm(halflife=hl, min_periods=1).mean())
    last = pd.DataFrame({"player_id": p.player_id, "last_team": p.team, "last_date": p.game_date, "pm": ewm("minutes", 10)})
    long_minutes = ewm("minutes", 40)
    for s in PROFILE:
        last[f"prof_{s}"] = ewm(s, 40) / long_minutes
    tg = d[["game_id", "game_date", "team", "season"]].drop_duplicates(["game_id", "team"])
    roster = p[["team", "season", "player_id"]].drop_duplicates()
    grid = tg.merge(roster, on=["team", "season"]).sort_values("game_date")
    grid = pd.merge_asof(grid, last.sort_values("last_date"), left_on="game_date", right_on="last_date", by="player_id",
                         allow_exact_matches=False)
    grid = grid[grid.last_team.eq(grid.team) & grid.pm.ge(ROTATION_MINUTES)]
    status = pd.Series(grid.set_index(["game_id", "player_id"]).index.map(
        lineups.set_index(["game_id", "player_id"]).roster_status.to_dict().get), index=grid.index)
    has_file = grid.game_id.isin(set(lineups.game_id))
    appeared = set(zip(p.game_id, p.player_id))
    played = pd.Series([k in appeared for k in zip(grid.game_id, grid.player_id)], index=grid.index)
    recent = (grid.game_date - grid.last_date).dt.days.le(ROTATION_DAYS)
    grid = grid[np.where(has_file, status.notna(), recent)].copy()
    grid["out"] = np.where(has_file.loc[grid.index], status.loc[grid.index].eq("Inactive"), ~played.loc[grid.index]).astype(float)
    return grid[["game_id", "team", "player_id", "out", "pm"] + [f"prof_{s}" for s in PROFILE]].reset_index(drop=True)


def absence_features(df: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Per player-game he played: what tonight's outs change against his recent games, from earlier games only."""
    d = _sorted(df)
    p = d[d.minutes > 0]
    g = p.groupby("player_id")
    own = p[["game_id", "player_id", "team", "season", "game_date", "minutes", *STATS]].copy()
    long_minutes = g.minutes.transform(lambda s: s.shift(1).ewm(halflife=40, min_periods=1).mean())
    for s in PROFILE:                                    # the player's own profile, before this game
        own[f"own_{s}"] = g[s].transform(lambda x: x.shift(1).ewm(halflife=40, min_periods=1).mean()) / long_minutes
    pairs = own.merge(grid.rename(columns={"player_id": "mate"}), on=["game_id", "team"])
    pairs = pairs[pairs.player_id != pairs.mate].sort_values(["player_id", "mate", "team", "season", "game_date", "game_id"])
    key = [pairs.player_id, pairs.mate, pairs.team, pairs.season]
    before = lambda values: values.groupby(key).cumsum() - values          # the total over earlier games of the pair
    n = pairs.groupby(key).cumcount()
    n_out = before(pairs.out)
    n_in = n - n_out
    weight = (n_out / (n_out + SHRINK)) * (n_in / (n_in + SHRINK))
    min_out = before(pairs.out * pairs.minutes)
    min_in = before(pairs.minutes) - min_out
    exposure = pairs.out.groupby(key).transform(lambda s: s.shift(1).ewm(halflife=EXPOSURE_HALFLIFE, min_periods=1).mean()).fillna(0.0)
    change = pairs.out - exposure                        # +1 newly out, -1 back, 0 nothing new
    newly, back = change.clip(lower=0), (-change).clip(lower=0)
    distance = sum(((pairs[f"own_{s}"] - pairs[f"prof_{s}"]) / scale) ** 2 for s, (_, scale) in PROFILE.items())
    alike = np.exp(-0.5 * distance.fillna(distance.median() if distance.notna().any() else 0.0))
    out = pairs[["game_id", "player_id"]].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["wo_min_adj"] = ((min_out / n_out - min_in / n_in) * weight).fillna(0.0) * change
        for s in STATS:
            stat_out = before(pairs.out * pairs[s])
            stat_in = before(pairs[s]) - stat_out
            out[f"wo_{s}_rate_adj"] = ((stat_out / min_out - stat_in / min_in) * weight).replace([np.inf, -np.inf], np.nan).fillna(0.0) * change
    out["wo_support"] = change.abs() * np.minimum(n_out, n_in)
    out["absent_new_min"], out["absent_back_min"] = pairs.pm * newly, pairs.pm * back
    out["absent_new_sim"], out["absent_back_sim"] = pairs.pm * newly * alike, pairs.pm * back * alike
    return out.groupby(["game_id", "player_id"], as_index=False).sum()


# --- the player's own lay-off ----------------------------------------------------------------------------------
def return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per player-game he played: how long he was away and, for EPISODE_GAMES after a lay-off, his earlier level."""
    d = _sorted(df)
    tg = d[["game_id", "game_date", "team", "season"]].drop_duplicates(["game_id", "team"]).sort_values(["team", "season", "game_date", "game_id"])
    tg["team_game"] = tg.groupby(["team", "season"]).cumcount()
    p = d[d.minutes > 0].merge(tg[["game_id", "team", "team_game"]], on=["game_id", "team"]).sort_values(["player_id", "game_date", "game_id"])
    g = p.groupby("player_id")
    same = p.team.eq(g.team.shift(1)) & p.season.eq(g.season.shift(1))
    out = p[["game_id", "player_id"]].copy()
    out["days_off"] = (p.game_date - g.game_date.shift(1)).dt.days.clip(upper=60)
    missed = (p.team_game - g.team_game.shift(1) - 1).where(same)
    out["team_games_missed"] = missed.clip(upper=30)
    level = g.minutes.transform(lambda s: s.shift(1).ewm(halflife=10, min_periods=1).mean())     # before this game
    recent = g.minutes.transform(lambda s: s.shift(1).ewm(halflife=3, min_periods=1).mean())
    back = missed.ge(LONG_ABSENCE)
    episode = [p.player_id, p.season, back.groupby([p.player_id, p.season]).cumsum()]
    since = back.groupby(episode).cumcount()
    live = back.groupby(episode).transform("first") & (since < EPISODE_GAMES)
    out["games_since_return"] = since.where(live, EPISODE_GAMES).astype(float)
    out["absence_len"] = missed.groupby(episode).transform("first").clip(upper=30).where(live, 0.0)
    out["min_pre_absence"] = level.groupby(episode).transform("first").where(live)
    out["return_gap"] = out.min_pre_absence - recent
    return out


# --- role ------------------------------------------------------------------------------------------------------
def role_features(f: pd.DataFrame) -> pd.DataFrame:
    """Per player-game: his minutes in earlier starts and earlier bench games, and tonight's role against them.
    `f` is the pinned feature frame (it carries tonight's starter flag from the lineup file)."""
    f = f.sort_values(["player_id", "game_date", "game_id"])
    played = f.minutes > 0
    prior = lambda values, hl: values.groupby(f.player_id).transform(lambda s: s.shift(1).ewm(halflife=hl, min_periods=1, ignore_na=True).mean())
    out = f[["game_id", "player_id"]].copy()
    out["min_as_starter"] = prior(f.minutes.where(played & f.starter.eq(1.0)), 5)
    out["min_as_bench"] = prior(f.minutes.where(played & f.starter.eq(0.0)), 5)
    out["min_in_role"] = np.where(f.starter.eq(1.0), out.min_as_starter, np.where(f.starter.eq(0.0), out.min_as_bench, np.nan))
    out["role_min_gap"] = out.min_in_role - f.min_ewm3
    last_role = f.starter.where(played).groupby(f.player_id).transform(lambda s: s.shift(1).ffill())
    out["starter_flip"] = f.starter - last_role
    started = (played & f.starter.eq(1.0)).astype(float).groupby(f.player_id).cumsum() - (played & f.starter.eq(1.0))
    benched = (played & f.starter.eq(0.0)).astype(float).groupby(f.player_id).cumsum() - (played & f.starter.eq(0.0))
    out["games_in_role"] = np.where(f.starter.eq(1.0), started, np.where(f.starter.eq(0.0), benched, np.nan)).clip(max=20)
    new = (f.starter.eq(1.0) & f.starter_rate.lt(0.5)).astype(float)
    out["team_new_starters"] = new.groupby([f.game_id, f.team]).transform("sum").where(f.starter.notna())
    return out


def add(f: pd.DataFrame, df: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    """The pinned feature frame with every input in FEATURES added."""
    absence = absence_features(df, candidates(df, lineups))
    out = f.merge(absence, on=["game_id", "player_id"], how="left")
    for c in GROUPS["absence"]:
        out[c] = out[c].fillna(0.0)
    out = out.merge(return_features(df), on=["game_id", "player_id"], how="left")
    return out.merge(role_features(f), on=["game_id", "player_id"], how="left")
