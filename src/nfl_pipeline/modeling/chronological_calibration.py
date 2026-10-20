"""Forward-only calibration folds, with explicit result availability."""
import pandas as pd


def forward_folds(frame):
    """Keep whole season/weeks together and purge labels unknown at first lock."""
    periods = sorted(set(zip(frame.season.astype(int), frame.week.astype(int))))
    period = pd.Series(list(zip(frame.season.astype(int), frame.week.astype(int))), index=frame.index)
    for test_period in periods[1:]:
        test = frame.loc[period.map(lambda x: x == test_period)].copy()
        earlier = frame.loc[period.map(lambda x: x < test_period)].copy()
        cutoff = pd.to_datetime(test.locked_at, utc=True, errors='coerce').min()
        available = pd.to_datetime(earlier.label_available_at, utc=True, errors='coerce')
        valid = available.notna() & (available < cutoff) & ~earlier.key.isin(test.key)
        if 'actual' in earlier:
            valid &= pd.to_numeric(earlier.actual, errors='coerce').notna()
        train = earlier.loc[valid].copy()
        yield train, test, dict(test_season=test_period[0], test_week=test_period[1],
            cutoff=cutoff.isoformat() if pd.notna(cutoff) else None,
            earlier_rows=len(earlier), eligible_train_rows=len(train),
            excluded_unavailable_or_overlapping=len(earlier)-len(train),
            train_periods=sorted(set(zip(train.season.astype(int), train.week.astype(int)))))


def equal_player_game_weights(frame):
    return 1.0 / frame.groupby('key').key.transform('size')
