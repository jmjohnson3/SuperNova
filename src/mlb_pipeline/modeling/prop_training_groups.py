"""Player-game grouping safeguards for offer-level prop model training."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GroupedTemporalSplit:
    train: pd.DataFrame
    holdout: pd.DataFrame
    strategy: str
    purged_rows: int


@dataclass(frozen=True)
class GroupedTemporalFold:
    fold_index: int
    train: pd.DataFrame
    holdout: pd.DataFrame
    train_start: Any
    train_end: Any
    holdout_start: Any
    holdout_end: Any
    purged_rows: int


def player_game_group_key(df: pd.DataFrame) -> pd.Series:
    """Return one stable identity per player-game, independent of offer count."""
    if df.empty:
        return pd.Series(dtype="object", index=df.index)

    if "game_date_et" in df:
        date = pd.to_datetime(df["game_date_et"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("unknown_date")
    else:
        date = pd.Series("unknown_date", index=df.index, dtype="object")
    if "game_slug" in df:
        game = df["game_slug"].fillna("").astype(str).str.strip()
    else:
        game = pd.Series("", index=df.index, dtype="object")
    game = game.where(game.ne(""), date)

    if "player_id" in df:
        player_id = pd.to_numeric(df["player_id"], errors="coerce")
        player = player_id.map(lambda value: str(int(value)) if pd.notna(value) else "")
    else:
        player = pd.Series("", index=df.index, dtype="object")
    for fallback in ("player_name_norm", "player_name"):
        if fallback in df:
            values = df[fallback].fillna("").astype(str).str.strip().str.lower()
            player = player.where(player.ne(""), values)
    player = player.where(player.ne(""), pd.Series(df.index.astype(str), index=df.index))
    return date + "|" + game + "|" + player


def add_player_game_weights(df: pd.DataFrame, *, weight_col: str = "player_game_weight") -> pd.DataFrame:
    """Give every player-game equal total influence regardless of offer duplication."""
    out = df.copy()
    if out.empty:
        out["player_game_group"] = pd.Series(dtype="object")
        out[weight_col] = pd.Series(dtype="float64")
        return out
    out["player_game_group"] = player_game_group_key(out)
    group_size = out.groupby("player_game_group")["player_game_group"].transform("size").clip(lower=1)
    raw = 1.0 / group_size.astype(float)
    out[weight_col] = raw * (float(len(raw)) / float(raw.sum()))
    return out


def dedupe_locked_offer_rows(
    df: pd.DataFrame,
    *,
    keep: str = "first",
) -> pd.DataFrame:
    """Collapse repeated replay rows for the same executable locked offer.

    The ledger can keep every replay/run row, but model training and promotion
    evaluation should not let repeated reruns of the same book/player/line
    count as fresh evidence.
    """
    if df.empty:
        out = df.copy()
        out.attrs["raw_rows"] = 0
        out.attrs["deduped_rows"] = 0
        return out

    out = df.copy()
    raw_rows = int(len(out))
    parts: list[pd.Series] = []
    if "prop_offer_id" in out and out["prop_offer_id"].notna().any():
        offer = pd.to_numeric(out["prop_offer_id"], errors="coerce")
        offer_key = offer.map(lambda value: f"offer:{int(value)}" if pd.notna(value) else "")
    else:
        offer_key = pd.Series("", index=out.index, dtype="object")

    if "game_date_et" in out:
        date = pd.to_datetime(out["game_date_et"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("unknown_date")
    else:
        date = pd.Series("unknown_date", index=out.index, dtype="object")
    parts.append(date)
    for col in ("game_slug", "player_id", "market", "side", "bookmaker_key"):
        if col in out:
            parts.append(out[col].fillna("").astype(str).str.lower().str.strip())
        else:
            parts.append(pd.Series("", index=out.index, dtype="object"))
    for col in ("market_line", "market_price"):
        if col in out:
            numeric = pd.to_numeric(out[col], errors="coerce")
            parts.append(numeric.map(lambda value: f"{float(value):.6f}" if pd.notna(value) else ""))
        else:
            parts.append(pd.Series("", index=out.index, dtype="object"))

    fallback_key = parts[0]
    for part in parts[1:]:
        fallback_key = fallback_key + "|" + part
    out["__locked_offer_dedupe_key"] = offer_key.where(offer_key.ne(""), "fallback:" + fallback_key)

    sort_cols = [
        col for col in (
            "game_date_et",
            "source_created_at",
            "locked_at_utc",
            "run_started_at_utc",
            "id",
            "replay_id",
        )
        if col in out.columns
    ]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort")
    out = out.drop_duplicates("__locked_offer_dedupe_key", keep=keep).drop(columns=["__locked_offer_dedupe_key"])
    out.attrs["raw_rows"] = raw_rows
    out.attrs["deduped_rows"] = int(raw_rows - len(out))
    return out


def purge_player_game_overlap(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if train.empty or holdout.empty:
        return train.copy(), holdout.copy(), 0
    holdout_keys = set(player_game_group_key(holdout))
    train_keys = player_game_group_key(train)
    keep = ~train_keys.isin(holdout_keys)
    return train.loc[keep].copy(), holdout.copy(), int((~keep).sum())


def temporal_player_game_split(
    df: pd.DataFrame,
    *,
    holdout_days: int,
    min_train_rows: int = 1,
    min_holdout_rows: int = 1,
    weight_col: str = "player_game_weight",
    min_train_fraction: float = 0.50,
    max_holdout_fraction: float = 0.45,
) -> GroupedTemporalSplit:
    """Split strictly by date, purge overlapping outcomes, then add group weights."""
    if df.empty:
        empty = add_player_game_weights(df, weight_col=weight_col)
        return GroupedTemporalSplit(empty, empty.copy(), f"last_{holdout_days}_days", 0)

    dates = pd.to_datetime(df["game_date_et"], errors="coerce").dt.date
    max_date = max(d for d in dates if pd.notna(d))
    split_date = max_date - timedelta(days=max(1, int(holdout_days)))
    train = df.loc[dates < split_date].copy()
    holdout = df.loc[dates >= split_date].copy()
    strategy = f"last_{holdout_days}_days"

    train_fraction = len(train) / max(1, len(df))
    holdout_fraction = len(holdout) / max(1, len(df))
    needs_fallback = (
        len(train) < min_train_rows
        or len(holdout) < min_holdout_rows
        or train_fraction < min_train_fraction
        or holdout_fraction > max_holdout_fraction
    )

    if needs_fallback:
        unique_dates = sorted(d for d in dates.dropna().unique())
        candidates: list[tuple[float, Any, pd.DataFrame, pd.DataFrame]] = []
        for candidate_date in unique_dates[1:]:
            candidate_train = df.loc[dates < candidate_date].copy()
            candidate_holdout = df.loc[dates >= candidate_date].copy()
            if len(candidate_train) < min_train_rows or len(candidate_holdout) < min_holdout_rows:
                continue
            train_fraction = len(candidate_train) / max(1, len(df))
            candidates.append(
                (abs(train_fraction - 0.80), candidate_date, candidate_train, candidate_holdout)
            )
        if candidates:
            _, split_date, train, holdout = min(candidates, key=lambda item: (item[0], item[1]))
            strategy = "temporal_row_80_20"
        elif len(unique_dates) > 1:
            split_date = unique_dates[-1]
            train = df.loc[dates < split_date].copy()
            holdout = df.loc[dates >= split_date].copy()
            strategy = "last_available_date"

    train, holdout, purged = purge_player_game_overlap(train, holdout)
    train = add_player_game_weights(train, weight_col=weight_col)
    holdout = add_player_game_weights(holdout, weight_col=weight_col)
    return GroupedTemporalSplit(train, holdout, f"{strategy}_player_game_purged_{purged}", purged)


def expanding_player_game_folds(
    df: pd.DataFrame,
    *,
    test_window_days: int = 7,
    step_days: int | None = None,
    min_train_dates: int = 10,
    min_train_rows: int = 1,
    min_holdout_rows: int = 1,
    max_folds: int | None = None,
    weight_col: str = "player_game_weight",
) -> list[GroupedTemporalFold]:
    """Build leakage-safe expanding walk-forward folds.

    Test windows never overlap when ``step_days`` equals the test window. Each
    player-game is also purged from the training side before weights are built.
    This is intended for evaluation; production models can then be refit on all
    settled rows after the out-of-fold decision is made.
    """
    if df.empty or "game_date_et" not in df:
        return []
    window = max(1, int(test_window_days))
    step = max(1, int(step_days if step_days is not None else window))
    dates = pd.to_datetime(df["game_date_et"], errors="coerce").dt.date
    unique_dates = sorted(d for d in dates.dropna().unique())
    if len(unique_dates) <= max(1, int(min_train_dates)):
        return []

    folds: list[GroupedTemporalFold] = []
    start_index = max(1, int(min_train_dates))
    fold_index = 0
    for test_index in range(start_index, len(unique_dates), step):
        holdout_dates = unique_dates[test_index:test_index + window]
        if len(holdout_dates) < window:
            continue
        holdout_start = holdout_dates[0]
        holdout_end = holdout_dates[-1]
        train = df.loc[dates < holdout_start].copy()
        holdout = df.loc[dates.isin(holdout_dates)].copy()
        if len(train) < min_train_rows or len(holdout) < min_holdout_rows:
            continue
        train, holdout, purged = purge_player_game_overlap(train, holdout)
        if len(train) < min_train_rows or len(holdout) < min_holdout_rows:
            continue
        train = add_player_game_weights(train, weight_col=weight_col)
        holdout = add_player_game_weights(holdout, weight_col=weight_col)
        fold_index += 1
        folds.append(GroupedTemporalFold(
            fold_index=fold_index,
            train=train,
            holdout=holdout,
            train_start=min(pd.to_datetime(train["game_date_et"]).dt.date),
            train_end=max(pd.to_datetime(train["game_date_et"]).dt.date),
            holdout_start=holdout_start,
            holdout_end=holdout_end,
            purged_rows=purged,
        ))
    if max_folds is not None and max_folds > 0 and len(folds) > max_folds:
        folds = folds[-int(max_folds):]
        folds = [
            GroupedTemporalFold(
                fold_index=index,
                train=fold.train,
                holdout=fold.holdout,
                train_start=fold.train_start,
                train_end=fold.train_end,
                holdout_start=fold.holdout_start,
                holdout_end=fold.holdout_end,
                purged_rows=fold.purged_rows,
            )
            for index, fold in enumerate(folds, start=1)
        ]
    return folds


def sample_weights(df: pd.DataFrame, *, weight_col: str = "player_game_weight") -> np.ndarray | None:
    if weight_col not in df:
        return None
    values = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0).clip(lower=1e-6)
    return values.to_numpy(dtype=float)


def grouping_summary(df: pd.DataFrame, *, weight_col: str = "player_game_weight") -> dict[str, Any]:
    if df.empty:
        return {"rows": 0, "player_games": 0, "max_rows_per_player_game": 0, "effective_weight": 0.0}
    keys = df.get("player_game_group", player_game_group_key(df))
    sizes = pd.Series(keys, index=df.index).value_counts()
    weights = pd.to_numeric(df.get(weight_col, pd.Series(1.0, index=df.index)), errors="coerce").fillna(1.0)
    return {
        "rows": int(len(df)),
        "player_games": int(sizes.size),
        "max_rows_per_player_game": int(sizes.max()) if not sizes.empty else 0,
        "mean_rows_per_player_game": float(sizes.mean()) if not sizes.empty else 0.0,
        "effective_weight": float(weights.sum()),
    }
