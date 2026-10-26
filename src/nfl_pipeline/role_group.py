"""One pregame definition of a team's role group, used by training (features) and live scoring.

The team-relative features (team_player_<stat>_avg5_sum, <stat>_share_avg_5, <stat>_role_rank) divide a
player's recent volume by his teammates'. Training used to count the players who went on to play plus
expected-but-absent teammates; live counted every active roster member, including players who never
touch the ball and cold-start members given a position-average prior. Live shares ran 15-25% low and
ranks 20-45% worse (docs/nfl_feature_parity.md).

The shared rule, built only from what is known before kickoff on both sides:

- a member is a skill player whose most recent game (for any team) is no older than his team's
  LOOKBACK_GAMES-th most recent game before this one: he has been part of an NFL rotation lately, which
  also covers trades and week 1 (the team's last games are then last season's);
- minus players ruled out for the week (Out or Doubtful on the injury report, or not on the active roster);
- cold-start players (no game history) count zero: no invented priors.

Every row still gets a share and a rank (rank 1 = largest), computed against the group: a non-member's
value is compared with the members' but is not added to the team total.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

LOOKBACK_GAMES = 4
RULED_OUT = ("out", "doubtful")


def is_ruled_out(report_status: pd.Series) -> pd.Series:
    status = report_status.fillna("").astype(str).str.strip().str.lower()
    return status.isin(RULED_OUT) | status.str.contains("injured reserve|reserve", regex=True, na=False)


def recency_cutoffs(team_dates: dict[str, list]) -> dict[tuple[str, object], object]:
    """{(team, game_date): date of the team's LOOKBACK_GAMES-th most recent earlier game, or None}."""
    out = {}
    for team, dates in team_dates.items():
        ordered = sorted(set(dates))
        for i, day in enumerate(ordered):
            prior = ordered[max(0, i - LOOKBACK_GAMES):i]
            out[(team, day)] = prior[0] if prior else None
    return out


def is_recent(last_played: pd.Series, cutoff: pd.Series) -> pd.Series:
    """True when the player's last game before this one is on or after the team's cutoff date.
    No cutoff (the team's first game in the data) admits anyone with a previous game."""
    last = pd.to_datetime(last_played, errors="coerce")
    cut = pd.to_datetime(cutoff, errors="coerce")
    return last.notna() & (cut.isna() | (last >= cut))


def role_features(frame: pd.DataFrame, stats: Iterable[str], member: pd.Series,
                  extra: dict[tuple, dict[str, list[float]]] | None = None,
                  group_cols: tuple[str, str] = ("game_id", "team_abbr")) -> pd.DataFrame:
    """Add team sums, shares and ranks for each stat's `<stat>_avg_5`, counting only members.

    `extra` holds members who have no row (training: expected teammates who did not play), keyed by
    (game_id, team_abbr), with their values per stat.
    """
    out = frame.copy()
    member = member.reindex(out.index).fillna(False).astype(bool)
    keys = list(zip(*(out[c] for c in group_cols)))
    extra = extra or {}
    for stat in stats:
        avg_col = f"{stat}_avg_5"
        if avg_col not in out.columns:
            out[avg_col] = np.nan
        values = pd.to_numeric(out[avg_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        counted = values.where(member, 0.0)
        extra_values = [extra.get(k, {}).get(stat, []) for k in keys]
        team_sum = counted.groupby([out[c] for c in group_cols]).transform("sum") + pd.Series(
            [float(sum(v)) for v in extra_values], index=out.index)
        sum_col, share_col, rank_col = f"team_player_{stat}_avg5_sum", f"{stat}_share_avg_5", f"{stat}_role_rank"
        out[sum_col] = team_sum.replace(0.0, np.nan)
        # a non-member's own value joins the denominator for his share, as a member's already does
        own_denominator = out[sum_col].where(member, team_sum + values).replace(0.0, np.nan)
        out[share_col] = (values / own_denominator).replace([np.inf, -np.inf], np.nan)
        ahead = []
        groups = {k: [] for k in set(keys)}
        for k, v, m in zip(keys, values, member):
            if m:
                groups[k].append(v)
        for k, v, extra_v in zip(keys, values, extra_values):
            ahead.append(sum(1 for x in groups[k] if x > v) + sum(1 for x in extra_v if x > v))
        out[rank_col] = (pd.Series(ahead, index=out.index, dtype=float) + 1.0).where(team_sum + values > 0)
    return out
