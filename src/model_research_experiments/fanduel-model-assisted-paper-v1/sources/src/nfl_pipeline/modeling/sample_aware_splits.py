"""Whole-week chronological partitions with auditable sample requirements."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BlockRequirement:
    rows: int
    weeks: int
    regular_rows: int


REQUIREMENTS = {
    'passing_yards': (BlockRequirement(180, 4, 120), BlockRequirement(160, 4, 100),
                      BlockRequirement(120, 3, 80), BlockRequirement(80, 3, 50), BlockRequirement(120, 3, 80)),
    'rushing_yards': (BlockRequirement(320, 4, 200), BlockRequirement(240, 3, 150),
                      BlockRequirement(180, 3, 100), BlockRequirement(120, 3, 80), BlockRequirement(200, 3, 120)),
    'receiving_yards': (BlockRequirement(500, 4, 300), BlockRequirement(400, 3, 250),
                        BlockRequirement(300, 3, 180), BlockRequirement(180, 3, 120), BlockRequirement(300, 3, 180)),
}
NAMES = ('model_fit', 'scale', 'residual', 'cal_fit', 'cal_tune', 'selection_gate')


def regular_mask(frame):
    # Regular seasons through 2020 had 17 calendar weeks, later seasons have 18.
    return pd.to_numeric(frame.week).le(np.where(pd.to_numeric(frame.season) <= 2020, 17, 18))


def sample_aware_partitions(frame, stat, requirements=None, min_model_rows=300, min_model_weeks=14):
    if frame.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Sample counts require unique player-games')
    if frame[['game_id', 'player_id', 'season', 'week', 'game_date_et']].isna().any().any():
        raise ValueError('Incomplete chronological identity')
    groups = list(frame.groupby(['season', 'week'], sort=False))
    groups.sort(key=lambda x: x[1].game_date_et.min())
    requirement = requirements or REQUIREMENTS[stat]
    end = len(groups); blocks = []
    for need in reversed(requirement):
        begin = end; rows = regular = 0
        while begin > 0 and (rows < need.rows or regular < need.regular_rows or end-begin < need.weeks):
            begin -= 1
            week = groups[begin][1]
            rows += len(week); regular += int(regular_mask(week).sum())
        if rows < need.rows or regular < need.regular_rows or end-begin < need.weeks:
            raise ValueError('Insufficient history for sample-aware calibration blocks')
        blocks.append(pd.concat([g for _, g in groups[begin:end]]).copy())
        end = begin
    if end < min_model_weeks:
        raise ValueError('Insufficient model-fit weeks after reserving calibration')
    early = pd.concat([g for _, g in groups[:end]]).copy()
    if len(early) < min_model_rows:
        raise ValueError('Insufficient model-fit player-games')
    parts = (early, *reversed(blocks))
    for i, left in enumerate(parts):
        for right in parts[i+1:]:
            if left.game_date_et.max() >= right.game_date_et.min() or set(left.game_id) & set(right.game_id):
                raise ValueError('Chronological blocks overlap')
    return parts


def describe_blocks(parts):
    return [dict(name=name, first=str(f.game_date_et.min()), last=str(f.game_date_et.max()),
                 rows=len(f), weeks=len(f[['season', 'week']].drop_duplicates()),
                 regular_rows=int(regular_mask(f).sum()), postseason_rows=int((~regular_mask(f)).sum()))
            for name, f in zip(NAMES, parts)]
