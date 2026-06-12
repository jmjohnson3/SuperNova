"""Train hitter player-game opportunity and outcome models.

These models validate the projection layer before betting-market logic gets a
vote.  They learn from one row per hitter/game:

* PA opportunity
* per-PA singles, doubles, triples, HR, and walk rates
* HR any-game rare-event probability

The artifact is diagnostic until holdout metrics prove it beats simple lineup
slot priors and existing prop projections.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .build_hitter_player_game_training_table import (
    HitterPlayerGameTrainingConfig,
    refresh_hitter_player_game_training,
)

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"

NUMERIC_FEATURES = [
    "lineup_slot",
    "confirmed_starter_num",
    "is_home",
    "team_implied_runs",
    "opponent_implied_runs",
    "game_total_line",
    "park_run_factor",
    "park_hr_factor",
    "park_babip_factor",
    "own_lineup_xwoba_avg",
    "own_lineup_xslg_avg",
    "own_lineup_barrel_avg",
    "own_lineup_hard_hit_avg",
    "own_lineup_k_pct_cv",
    "own_lineup_pct_lhb",
    "lineup_confirmed_flag",
    "confirmed_team_lineup_slots",
    "team_lineup_confirmed_flag",
    "lineup_boxscore_proxy_flag",
    "lineup_slot_x_team_implied_runs",
    "opp_sp_hand_l",
    "opp_sp_k_pct_10",
    "opp_sp_bb_pct",
    "opp_sp_xwoba",
    "opp_sp_hard_hit_pct",
    "opp_sp_whiff_pct",
    "opp_bp_era_10",
    "opp_bp_whip_10",
    "opp_bp_k9_10",
    "opp_bp_ip_last_3",
    "opp_bp_ip_last_7",
    "opp_team_k_pct_10",
    "opp_team_avg_10",
    "opp_team_obp_10",
    "opp_team_slg_10",
    "batter_vs_hand_hits_avg_10",
    "batter_vs_hand_tb_avg_10",
    "batter_vs_hand_hr_avg_10",
    "batter_vs_hand_iso_avg_10",
    "batter_vs_hand_k_rate_10",
    "batter_vs_hand_games_10",
    "batter_vs_rp_ba_30",
    "batter_vs_rp_slg_30",
    "batter_vs_rp_hr_rate_30",
    "batter_vs_rp_k_rate_30",
    "batter_sc_barrel_rate",
    "batter_sc_hard_hit_pct",
    "batter_sc_avg_exit_velo",
    "batter_sc_avg_launch_angle",
    "batter_sc_sweet_spot_pct",
    "batter_sc_fb_pct",
    "batter_sc_gb_pct",
    "batter_sc_ld_pct",
    "batter_sc_xba",
    "batter_sc_xslg",
    "batter_sc_xwoba",
    "batter_sc_xiso",
    "batter_sc_brl_pa",
    "batter_sprint_speed",
    "batter_disc_oz_swing_pct",
    "batter_disc_iz_contact_pct",
    "batter_disc_oz_contact_pct",
    "batter_disc_whiff_pct",
    "batter_disc_out_zone_pct",
    "batter_disc_k_pct",
    "batter_disc_bb_pct",
    "opp_sp_sc_barrel_rate",
    "opp_sp_sc_hard_hit_pct",
    "opp_sp_sc_avg_exit_velo",
    "opp_sp_sc_avg_launch_angle",
    "opp_sp_sc_xba",
    "opp_sp_sc_xslg",
    "opp_sp_sc_xwoba",
    "opp_sp_sc_xiso",
    "opp_sp_fb_pct",
    "opp_sp_fb_hard_hit_pct",
    "opp_sp_fb_xwoba",
    "opp_sp_fb_run_value_per_100",
    "opp_sp_fb_whiff_pct",
    "opp_sp_fb_k_pct",
    "opp_sp_si_pct",
    "opp_sp_si_hard_hit_pct",
    "opp_sp_si_xwoba",
    "opp_sp_si_whiff_pct",
    "opp_sp_si_k_pct",
    "opp_sp_sl_pct",
    "opp_sp_sl_hard_hit_pct",
    "opp_sp_sl_xwoba",
    "opp_sp_sl_run_value_per_100",
    "opp_sp_sl_whiff_pct",
    "opp_sp_sl_k_pct",
    "opp_sp_ch_pct",
    "opp_sp_ch_hard_hit_pct",
    "opp_sp_ch_xwoba",
    "opp_sp_ch_run_value_per_100",
    "opp_sp_ch_whiff_pct",
    "opp_sp_ch_k_pct",
    "opp_sp_fastball_family_pct",
    "opp_sp_pitch_diversity",
    "projected_pa",
    "pa_games",
]

CATEGORICAL_FEATURES = [
    "team_abbr",
    "opponent_abbr",
    "lineup_source",
    "starter_status_source",
    "primary_position",
    "batter_hand",
    "opp_sp_hand",
]

RATE_TARGETS = {
    "single_rate": "actual_singles",
    "double_rate": "actual_doubles",
    "triple_rate": "actual_triples",
    "hr_rate": "actual_home_runs",
    "walk_rate": "actual_walks",
}

EVENT_CLASSES = ["out", "walk", "single", "double", "triple", "hr"]


@dataclass(frozen=True)
class HitterOutcomeModelConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    lookback_days: int = 365
    holdout_days: int = 28
    min_train_rows: int = 1000
    min_holdout_rows: int = 200
    min_prop_holdout_rows: int = 100
    rebuild_player_game_if_empty: bool = True
    report_file: str | None = None


SQL = """
SELECT *
FROM features.mlb_hitter_player_game_training
WHERE game_date_et >= %(cutoff)s
  AND actual_pa IS NOT NULL
ORDER BY game_date_et, game_slug, player_id
"""


def _table_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
              SELECT 1 FROM information_schema.tables
              WHERE table_schema = %s AND table_name = %s
            )
            """,
            (schema, table),
        )
        return bool(cur.fetchone()[0])


def _row_count(conn) -> int:
    if not _table_exists(conn, "features", "mlb_hitter_player_game_training"):
        return 0
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM features.mlb_hitter_player_game_training")
        return int(cur.fetchone()[0] or 0)


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=columns)


def _load(cfg: HitterOutcomeModelConfig) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)
    with psycopg2.connect(cfg.pg_dsn) as conn:
        if _row_count(conn) == 0:
            if not cfg.rebuild_player_game_if_empty:
                return pd.DataFrame()
            refresh_hitter_player_game_training(
                HitterPlayerGameTrainingConfig(
                    pg_dsn=cfg.pg_dsn,
                    lookback_days=cfg.lookback_days,
                )
            )
        df = _query_df(conn, SQL, {"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    numeric = set(NUMERIC_FEATURES) | {
        "actual_pa",
        "actual_hits",
        "actual_singles",
        "actual_doubles",
        "actual_triples",
        "actual_home_runs",
        "actual_total_bases",
        "actual_walks",
        "model_pred_hits",
        "model_pred_total_bases",
        "model_pred_home_runs",
        "prop_example_rows",
    }
    df["confirmed_starter_num"] = df["confirmed_starter"].fillna(False).astype(bool).astype(float)
    for col in numeric:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        if col not in df:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _coalesce_column(df: pd.DataFrame, name: str, aliases: tuple[str, ...], default: Any = np.nan) -> pd.Series:
    if name in df:
        out = df[name].copy()
    else:
        out = pd.Series([default] * len(df), index=df.index)
    for alias in aliases:
        if alias in df:
            out = out.where(out.notna(), df[alias])
    return out


def prepare_hitter_outcome_features(
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    """Return a model-ready hitter outcome feature frame.

    Player-game training rows use names like ``lineup_slot`` while offer-level
    replay rows use names like ``confirmed_batting_order``.  This adapter keeps
    the direct event model usable in both places without duplicating feature
    mapping logic.
    """
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    out = df.copy()
    out["lineup_slot"] = _coalesce_column(out, "lineup_slot", ("confirmed_batting_order",))
    out["lineup_source"] = _coalesce_column(out, "lineup_source", ("confirmed_lineup_source",), "unknown")
    alias_map: dict[str, tuple[str, ...]] = {
        "batter_sc_barrel_rate": ("sc_barrel_rate",),
        "batter_sc_hard_hit_pct": ("sc_hard_hit_pct",),
        "batter_sc_avg_exit_velo": ("sc_avg_exit_velo",),
        "batter_sc_avg_launch_angle": ("sc_avg_launch_angle",),
        "batter_sc_sweet_spot_pct": ("sc_sweet_spot_pct",),
        "batter_sc_fb_pct": ("sc_fb_pct",),
        "batter_sc_gb_pct": ("sc_gb_pct",),
        "batter_sc_ld_pct": ("sc_ld_pct",),
        "batter_sc_xba": ("sc_xba",),
        "batter_sc_xslg": ("sc_xslg",),
        "batter_sc_xwoba": ("sc_xwoba",),
        "batter_sc_xiso": ("sc_xiso",),
        "batter_sc_brl_pa": ("sc_brl_pa",),
        "batter_sprint_speed": ("sprint_speed",),
        "batter_disc_oz_swing_pct": ("sc_b_oz_swing_pct",),
        "batter_disc_iz_contact_pct": ("sc_b_iz_contact_pct",),
        "batter_disc_oz_contact_pct": ("sc_b_oz_contact_pct",),
        "batter_disc_whiff_pct": ("sc_b_disc_whiff_pct",),
        "batter_disc_out_zone_pct": ("sc_b_out_zone_pct",),
        "batter_disc_k_pct": ("sc_b_k_pct",),
        "batter_disc_bb_pct": ("sc_b_bb_pct",),
    }
    for canonical, aliases in alias_map.items():
        out[canonical] = _coalesce_column(out, canonical, aliases)
    if "confirmed_starter_num" not in out:
        if "confirmed_starter" in out:
            out["confirmed_starter_num"] = out["confirmed_starter"].fillna(False).astype(bool).astype(float)
        else:
            slot = pd.to_numeric(out.get("lineup_slot"), errors="coerce")
            out["confirmed_starter_num"] = slot.between(1, 9).astype(float)
    slot = pd.to_numeric(out.get("lineup_slot"), errors="coerce")
    if "lineup_confirmed_flag" not in out:
        source = out.get("lineup_source", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
        out["lineup_confirmed_flag"] = source.str.contains("lineup|raw_lineups", case=False, regex=True).astype(float)
    if "lineup_boxscore_proxy_flag" not in out:
        source = out.get("lineup_source", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
        out["lineup_boxscore_proxy_flag"] = source.str.contains("boxscore", case=False, regex=False).astype(float)
    if "lineup_slot_x_team_implied_runs" not in out:
        team_runs = pd.to_numeric(out.get("team_implied_runs"), errors="coerce")
        out["lineup_slot_x_team_implied_runs"] = slot * team_runs
    if "starter_status_source" not in out:
        src = out.get("lineup_source")
        out["starter_status_source"] = np.where(
            pd.to_numeric(out.get("lineup_slot"), errors="coerce").between(1, 9),
            src.fillna("confirmed_or_projected_lineup") if isinstance(src, pd.Series) else "confirmed_or_projected_lineup",
            "unknown",
        )

    data: dict[str, Any] = {}
    for col in numeric_features:
        data[col] = pd.to_numeric(out[col], errors="coerce") if col in out else pd.Series(np.nan, index=out.index)
    for col in categorical_features:
        data[col] = out[col].fillna("unknown").astype(str) if col in out else pd.Series("unknown", index=out.index)
    return pd.DataFrame(data, index=out.index)[numeric_features + categorical_features]


def _split(df: pd.DataFrame, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_date = max(df["game_date_et"])
    split_date = max_date - timedelta(days=max(1, holdout_days - 1))
    train = df[df["game_date_et"] < split_date].copy()
    holdout = df[df["game_date_et"] >= split_date].copy()
    return train, holdout


def _one_hot_encoder(*, dense: bool = False) -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore", "min_frequency": 10}
    try:
        return OneHotEncoder(**kwargs, sparse_output=not dense)
    except TypeError:  # pragma: no cover - older sklearn
        return OneHotEncoder(**kwargs, sparse=not dense)


def _preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    *,
    dense: bool = False,
) -> ColumnTransformer:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                _one_hot_encoder(dense=dense),
                categorical_features,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0 if dense else 0.3,
    )


def _regression_pipeline(
    alpha: float = 8.0,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    return Pipeline([
        ("features", _preprocessor(numeric_features, categorical_features)),
        ("model", Ridge(alpha=alpha)),
    ])


def _classifier_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    return Pipeline([
        ("features", _preprocessor(numeric_features, categorical_features)),
        ("model", LogisticRegression(max_iter=1000, C=0.35)),
    ])


def _calibrated_hgb_classifier() -> Any:
    base = HistGradientBoostingClassifier(
        max_iter=90,
        learning_rate=0.055,
        max_leaf_nodes=17,
        min_samples_leaf=35,
        l2_regularization=0.08,
        early_stopping=True,
        random_state=42,
    )
    try:
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=2, ensemble=False)
    except TypeError:  # pragma: no cover - older sklearn
        return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=2)


def _boosted_classifier_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    return Pipeline([
        ("features", _preprocessor(numeric_features, categorical_features, dense=True)),
        ("model", _calibrated_hgb_classifier()),
    ])


def _clip(values: Any, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    return np.clip(arr, lower, upper)


def _rmse(y_true: Any, y_pred: Any) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def _count_metrics(y_true: pd.Series, y_pred: Any) -> dict[str, float]:
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(pred) & pd.notna(y_true).to_numpy()
    if not mask.any():
        return {"rows": 0}
    true = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)[mask]
    pred = pred[mask]
    return {
        "rows": int(mask.sum()),
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": _rmse(true, pred),
        "bias": float(np.mean(true - pred)),
    }


def _slot_prior(train: pd.DataFrame, holdout: pd.DataFrame, target: str) -> np.ndarray:
    overall = float(train[target].mean())
    by_slot = train.groupby(train["lineup_slot"].round())[target].mean().to_dict()
    vals = []
    for value in holdout["lineup_slot"]:
        slot = round(float(value)) if pd.notna(value) else None
        vals.append(float(by_slot.get(slot, overall)))
    return np.asarray(vals, dtype=float)


def _rate_prior(train: pd.DataFrame, holdout: pd.DataFrame, count_col: str, pa_pred: np.ndarray) -> np.ndarray:
    work = train[train["actual_pa"] > 0].copy()
    work["_rate"] = (work[count_col] / work["actual_pa"]).clip(lower=0.0, upper=1.0)
    overall = float(work["_rate"].mean())
    by_slot = work.groupby(work["lineup_slot"].round())["_rate"].mean().to_dict()
    rates = []
    for value in holdout["lineup_slot"]:
        slot = round(float(value)) if pd.notna(value) else None
        rates.append(float(by_slot.get(slot, overall)))
    return np.asarray(rates, dtype=float) * pa_pred


def _event_count_matrix(df: pd.DataFrame) -> pd.DataFrame:
    pa = pd.to_numeric(df.get("actual_pa"), errors="coerce").fillna(0.0).clip(lower=0.0)
    walks = pd.to_numeric(df.get("actual_walks"), errors="coerce").fillna(0.0).clip(lower=0.0)
    singles = pd.to_numeric(df.get("actual_singles"), errors="coerce").fillna(0.0).clip(lower=0.0)
    doubles = pd.to_numeric(df.get("actual_doubles"), errors="coerce").fillna(0.0).clip(lower=0.0)
    triples = pd.to_numeric(df.get("actual_triples"), errors="coerce").fillna(0.0).clip(lower=0.0)
    hr = pd.to_numeric(df.get("actual_home_runs"), errors="coerce").fillna(0.0).clip(lower=0.0)
    non_out = walks + singles + doubles + triples + hr
    scale = pd.Series(1.0, index=df.index, dtype="float64")
    over = (non_out > pa) & (non_out > 0)
    scale.loc[over] = (pa.loc[over] / non_out.loc[over]).clip(lower=0.0, upper=1.0)
    walks *= scale
    singles *= scale
    doubles *= scale
    triples *= scale
    hr *= scale
    out = (pa - (walks + singles + doubles + triples + hr)).clip(lower=0.0)
    return pd.DataFrame(
        {
            "out": out,
            "walk": walks,
            "single": singles,
            "double": doubles,
            "triple": triples,
            "hr": hr,
        },
        index=df.index,
    )


def _build_event_training_examples(
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    features = prepare_hitter_outcome_features(df, numeric_features, categorical_features)
    counts = _event_count_matrix(df)
    parts: list[pd.DataFrame] = []
    labels: list[str] = []
    weights: list[float] = []
    for cls in EVENT_CLASSES:
        w = pd.to_numeric(counts[cls], errors="coerce").fillna(0.0)
        mask = w > 0
        if not mask.any():
            continue
        parts.append(features.loc[mask].copy())
        labels.extend([cls] * int(mask.sum()))
        weights.extend(w.loc[mask].to_numpy(dtype=float).tolist())
    if not parts:
        return pd.DataFrame(columns=features.columns), np.asarray([], dtype=object), np.asarray([], dtype=float)
    return pd.concat(parts, ignore_index=True), np.asarray(labels, dtype=object), np.asarray(weights, dtype=float)


def _predict_event_probabilities(
    model: Pipeline,
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    features = prepare_hitter_outcome_features(df, numeric_features, categorical_features)
    raw = model.predict_proba(features[numeric_features + categorical_features])
    probs = pd.DataFrame(0.0, index=df.index, columns=[f"p_{cls}" for cls in EVENT_CLASSES])
    for i, cls in enumerate(model.classes_):
        if cls in EVENT_CLASSES:
            probs[f"p_{cls}"] = raw[:, i]
    row_sum = probs.sum(axis=1).replace(0.0, np.nan)
    probs = probs.div(row_sum, axis=0).fillna(0.0)
    return probs


def _fit_boosted_event_binary_models(
    X_event: pd.DataFrame,
    y_event: np.ndarray,
    w_event: np.ndarray,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> dict[str, Pipeline]:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    models: dict[str, Pipeline] = {}
    if len(y_event) == 0:
        return models
    feature_cols = numeric_features + categorical_features
    for cls in EVENT_CLASSES:
        y_bin = (y_event == cls).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        model = _boosted_classifier_pipeline(numeric_features, categorical_features)
        model.fit(X_event[feature_cols], y_bin, model__sample_weight=w_event)
        models[cls] = model
    return models


def _predict_boosted_event_probabilities(
    models: dict[str, Pipeline],
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    features = prepare_hitter_outcome_features(df, numeric_features, categorical_features)
    X = features[numeric_features + categorical_features]
    probs = pd.DataFrame(0.0, index=df.index, columns=[f"p_{cls}" for cls in EVENT_CLASSES])
    for cls, model in models.items():
        if cls not in EVENT_CLASSES:
            continue
        try:
            probs[f"p_{cls}"] = model.predict_proba(X)[:, 1]
        except Exception:
            continue
    row_sum = probs.sum(axis=1).replace(0.0, np.nan)
    fallback = 1.0 / float(len(EVENT_CLASSES))
    probs = probs.div(row_sum, axis=0).fillna(fallback)
    return probs


def _event_log_loss_from_counts(counts: pd.DataFrame, probs: pd.DataFrame) -> float | None:
    total = float(counts[EVENT_CLASSES].sum(axis=1).sum())
    if total <= 0:
        return None
    loss = 0.0
    for cls in EVENT_CLASSES:
        p = np.clip(pd.to_numeric(probs[f"p_{cls}"], errors="coerce").fillna(1e-6).to_numpy(dtype=float), 1e-6, 1.0)
        w = pd.to_numeric(counts[cls], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        loss += float(np.sum(w * -np.log(p)))
    return loss / total


def _event_rate_table(df: pd.DataFrame, probs: pd.DataFrame) -> dict[str, Any]:
    counts = _event_count_matrix(df)
    actual_pa = float(counts[EVENT_CLASSES].sum(axis=1).sum())
    pred_pa = float(len(df)) if len(df) else 0.0
    rows: dict[str, Any] = {"rows": int(len(df)), "actual_pa": actual_pa}
    for cls in EVENT_CLASSES:
        actual = float(counts[cls].sum())
        pred_rate = float(pd.to_numeric(probs[f"p_{cls}"], errors="coerce").fillna(0.0).mean()) if len(probs) else 0.0
        rows[cls] = {
            "actual_per_pa": actual / actual_pa if actual_pa > 0 else None,
            "pred_mean_prob": pred_rate,
            "bias_per_pa": (actual / actual_pa - pred_rate) if actual_pa > 0 else None,
        }
    rows["mean_row_weight"] = actual_pa / pred_pa if pred_pa > 0 else None
    return rows


def _event_projection_metrics(
    event_holdout: pd.DataFrame,
    event_probs: pd.DataFrame,
    event_pa_pred: np.ndarray,
) -> dict[str, Any]:
    event_hit_count = event_pa_pred * (
        event_probs["p_single"].to_numpy(dtype=float)
        + event_probs["p_double"].to_numpy(dtype=float)
        + event_probs["p_triple"].to_numpy(dtype=float)
        + event_probs["p_hr"].to_numpy(dtype=float)
    )
    event_tb_count = event_pa_pred * (
        event_probs["p_single"].to_numpy(dtype=float)
        + 2.0 * event_probs["p_double"].to_numpy(dtype=float)
        + 3.0 * event_probs["p_triple"].to_numpy(dtype=float)
        + 4.0 * event_probs["p_hr"].to_numpy(dtype=float)
    )
    event_hr_count = event_pa_pred * event_probs["p_hr"].to_numpy(dtype=float)
    event_counts = _event_count_matrix(event_holdout)
    return {
        "weighted_event_log_loss": _event_log_loss_from_counts(event_counts, event_probs),
        "event_rates": _event_rate_table(event_holdout, event_probs),
        "hits": _count_metrics(event_holdout["actual_hits"], event_hit_count),
        "total_bases": _count_metrics(event_holdout["actual_total_bases"], event_tb_count),
        "home_runs": _count_metrics(event_holdout["actual_home_runs"], event_hr_count),
    }


def _safe_auc(y_true: Any, prob: Any) -> float | None:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def _safe_log_loss(y_true: Any, prob: Any) -> float | None:
    try:
        return float(log_loss(y_true, np.clip(prob, 1e-5, 1 - 1e-5), labels=[0, 1]))
    except Exception:
        return None


def _prop_projection_metrics(holdout: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    mapping = {
        "hits": ("actual_hits", "model_pred_hits"),
        "total_bases": ("actual_total_bases", "model_pred_total_bases"),
        "home_runs": ("actual_home_runs", "model_pred_home_runs"),
    }
    prop_rows = holdout[holdout["prop_example_rows"].fillna(0) > 0].copy()
    for name, (target, pred_col) in mapping.items():
        if prop_rows.empty or pred_col not in prop_rows:
            out[name] = {"rows": 0}
            continue
        mask = prop_rows[pred_col].notna() & prop_rows[target].notna()
        if not mask.any():
            out[name] = {"rows": 0}
            continue
        out[name] = _count_metrics(prop_rows.loc[mask, target], prop_rows.loc[mask, pred_col])
    return out


def train_hitter_player_game_outcomes(cfg: HitterOutcomeModelConfig) -> dict[str, Any]:
    df = _load(cfg)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "features.mlb_hitter_player_game_training",
        "rows": int(len(df)),
        "status": "ok",
    }
    if df.empty:
        payload["status"] = "no_rows"
        _write_outputs(payload, cfg, models=None)
        return payload

    train, holdout = _split(df, cfg.holdout_days)
    payload["train_rows"] = int(len(train))
    payload["holdout_rows"] = int(len(holdout))
    payload["holdout_start"] = str(min(holdout["game_date_et"])) if len(holdout) else None
    payload["holdout_end"] = str(max(holdout["game_date_et"])) if len(holdout) else None
    if len(train) < cfg.min_train_rows or len(holdout) < cfg.min_holdout_rows:
        payload["status"] = "insufficient_rows"
        _write_outputs(payload, cfg, models=None)
        return payload

    models: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    train_features = prepare_hitter_outcome_features(train)
    holdout_features = prepare_hitter_outcome_features(holdout)

    pa_model = _regression_pipeline(alpha=10.0)
    pa_model.fit(train_features[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train["actual_pa"])
    pa_pred = np.clip(pa_model.predict(holdout_features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]), 0.0, 6.5)
    slot_pa_pred = _slot_prior(train, holdout, "actual_pa")
    projected_pa_mask = holdout["projected_pa"].notna()
    metrics["pa_model"] = {
        "model": _count_metrics(holdout["actual_pa"], pa_pred),
        "slot_prior": _count_metrics(holdout["actual_pa"], slot_pa_pred),
        "existing_projected_pa": (
            _count_metrics(holdout.loc[projected_pa_mask, "actual_pa"], holdout.loc[projected_pa_mask, "projected_pa"])
            if projected_pa_mask.any()
            else {"rows": 0}
        ),
    }
    models["pa_model"] = pa_model

    rate_train = train[train["actual_pa"] > 0].copy()
    rate_holdout = holdout[holdout["actual_pa"] > 0].copy()
    holdout_rate_index = rate_holdout.index
    pa_pred_rates = pd.Series(pa_pred, index=holdout.index).loc[holdout_rate_index].to_numpy(dtype=float)
    rate_train_features = prepare_hitter_outcome_features(rate_train)
    rate_holdout_features = prepare_hitter_outcome_features(rate_holdout)
    rate_predictions: dict[str, np.ndarray] = {}
    rate_metrics: dict[str, Any] = {}
    for rate_name, count_col in RATE_TARGETS.items():
        rate_train[rate_name] = (rate_train[count_col] / rate_train["actual_pa"]).clip(lower=0.0, upper=1.0)
        model = _regression_pipeline(alpha=12.0)
        model.fit(rate_train_features[NUMERIC_FEATURES + CATEGORICAL_FEATURES], rate_train[rate_name])
        pred_rate = _clip(model.predict(rate_holdout_features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]), 0.0, 0.9)
        rate_predictions[rate_name] = pred_rate
        pred_count = pred_rate * pa_pred_rates
        prior_count = _rate_prior(train, rate_holdout, count_col, pa_pred_rates)
        rate_metrics[rate_name] = {
            "rate_rows": int(len(rate_holdout)),
            "count_model": _count_metrics(rate_holdout[count_col], pred_count),
            "slot_rate_prior": _count_metrics(rate_holdout[count_col], prior_count),
        }
        models[rate_name] = model
    metrics["rate_models"] = rate_metrics

    single = rate_predictions["single_rate"]
    double = rate_predictions["double_rate"]
    triple = rate_predictions["triple_rate"]
    hr = rate_predictions["hr_rate"]
    hit_count = pa_pred_rates * (single + double + triple + hr)
    tb_count = pa_pred_rates * (single + 2.0 * double + 3.0 * triple + 4.0 * hr)
    hr_count = pa_pred_rates * hr
    metrics["structured_counts"] = {
        "rows": int(len(rate_holdout)),
        "hits": _count_metrics(rate_holdout["actual_hits"], hit_count),
        "total_bases": _count_metrics(rate_holdout["actual_total_bases"], tb_count),
        "home_runs": _count_metrics(rate_holdout["actual_home_runs"], hr_count),
        "slot_prior_hits": _count_metrics(
            rate_holdout["actual_hits"],
            _rate_prior(train, rate_holdout, "actual_hits", pa_pred_rates),
        ),
        "slot_prior_total_bases": _count_metrics(
            rate_holdout["actual_total_bases"],
            _rate_prior(train, rate_holdout, "actual_total_bases", pa_pred_rates),
        ),
        "slot_prior_home_runs": _count_metrics(
            rate_holdout["actual_home_runs"],
            _rate_prior(train, rate_holdout, "actual_home_runs", pa_pred_rates),
        ),
    }

    event_train = train[train["actual_pa"] > 0].copy()
    event_holdout = holdout[holdout["actual_pa"] > 0].copy()
    X_event, y_event, w_event = _build_event_training_examples(event_train)
    event_metrics: dict[str, Any] = {"train_event_rows": int(len(y_event)), "holdout_rows": int(len(event_holdout))}
    if len(y_event) > 0 and len(np.unique(y_event)) >= 4 and len(event_holdout) > 0:
        event_model = _classifier_pipeline()
        event_model.fit(X_event[NUMERIC_FEATURES + CATEGORICAL_FEATURES], y_event, model__sample_weight=w_event)
        event_probs = _predict_event_probabilities(event_model, event_holdout)
        event_pa_pred = pd.Series(pa_pred, index=holdout.index).loc[event_holdout.index].to_numpy(dtype=float)
        linear_metrics = _event_projection_metrics(event_holdout, event_probs, event_pa_pred)
        event_metrics.update({
            "active_event_model": "linear_multinomial",
            "classes": list(map(str, event_model.classes_)),
            "linear_multinomial": linear_metrics,
            **linear_metrics,
        })
        models["event_outcome_model"] = event_model
        try:
            boosted_models = _fit_boosted_event_binary_models(X_event, y_event, w_event)
        except Exception as exc:
            event_metrics["boosted_binary_error"] = str(exc)
            boosted_models = {}
        if len(boosted_models) >= 4:
            boosted_probs = _predict_boosted_event_probabilities(boosted_models, event_holdout)
            boosted_metrics = _event_projection_metrics(event_holdout, boosted_probs, event_pa_pred)
            event_metrics["boosted_binary"] = {
                "classes": sorted(boosted_models.keys()),
                **boosted_metrics,
            }
            linear_loss = linear_metrics.get("weighted_event_log_loss")
            boosted_loss = boosted_metrics.get("weighted_event_log_loss")
            use_boosted = (
                boosted_loss is not None
                and (linear_loss is None or float(boosted_loss) <= float(linear_loss))
            )
            models["event_binary_models"] = boosted_models
            if use_boosted:
                event_metrics.update({
                    "active_event_model": "boosted_binary_calibrated",
                    "classes": sorted(boosted_models.keys()),
                    **boosted_metrics,
                })
    else:
        event_metrics["status"] = "insufficient_event_classes"
    metrics["direct_event_model"] = event_metrics

    hr_train = train[train["actual_pa"] > 0].copy()
    hr_holdout = holdout[holdout["actual_pa"] > 0].copy()
    hr_train["hr_any"] = (hr_train["actual_home_runs"] > 0).astype(int)
    hr_holdout["hr_any"] = (hr_holdout["actual_home_runs"] > 0).astype(int)
    hr_any_model = _classifier_pipeline()
    hr_any_model.fit(
        prepare_hitter_outcome_features(hr_train)[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        hr_train["hr_any"],
    )
    hr_prob = hr_any_model.predict_proba(
        prepare_hitter_outcome_features(hr_holdout)[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    )[:, 1]
    hr_prior = float(hr_train["hr_any"].mean())
    prior_prob = np.full(len(hr_holdout), hr_prior, dtype=float)
    metrics["hr_any_model"] = {
        "rows": int(len(hr_holdout)),
        "model_brier": float(brier_score_loss(hr_holdout["hr_any"], hr_prob)),
        "prior_brier": float(brier_score_loss(hr_holdout["hr_any"], prior_prob)),
        "model_log_loss": _safe_log_loss(hr_holdout["hr_any"], hr_prob),
        "prior_log_loss": _safe_log_loss(hr_holdout["hr_any"], prior_prob),
        "auc": _safe_auc(hr_holdout["hr_any"], hr_prob),
        "base_rate": hr_prior,
    }
    models["hr_any_model"] = hr_any_model

    metrics["existing_prop_projection_holdout"] = _prop_projection_metrics(holdout)
    payload["metrics"] = metrics
    payload["feature_coverage"] = _feature_coverage(df)
    payload["recommendation"] = _recommend(metrics, len(holdout), cfg)
    _write_outputs(payload, cfg, models=models)
    return payload


def _feature_coverage(df: pd.DataFrame) -> dict[str, float]:
    coverage: dict[str, float] = {}
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if col not in df:
            coverage[col] = 0.0
        else:
            coverage[col] = float(df[col].notna().mean())
    return coverage


def _gain(model_metric: dict[str, Any], prior_metric: dict[str, Any], key: str = "mae") -> float | None:
    try:
        model = float(model_metric[key])
        prior = float(prior_metric[key])
    except Exception:
        return None
    return prior - model


def _recommend(metrics: dict[str, Any], holdout_rows: int, cfg: HitterOutcomeModelConfig) -> dict[str, Any]:
    pa_gain = _gain(metrics.get("pa_model", {}).get("model", {}), metrics.get("pa_model", {}).get("slot_prior", {}))
    hits_gain = _gain(
        metrics.get("structured_counts", {}).get("hits", {}),
        metrics.get("structured_counts", {}).get("slot_prior_hits", {}),
    )
    tb_gain = _gain(
        metrics.get("structured_counts", {}).get("total_bases", {}),
        metrics.get("structured_counts", {}).get("slot_prior_total_bases", {}),
    )
    hr_gain = _gain(
        metrics.get("structured_counts", {}).get("home_runs", {}),
        metrics.get("structured_counts", {}).get("slot_prior_home_runs", {}),
    )
    hr_any = metrics.get("hr_any_model", {})
    direct = metrics.get("direct_event_model", {})
    event_tb_gain = _gain(
        direct.get("total_bases", {}),
        metrics.get("structured_counts", {}).get("total_bases", {}),
    )
    event_hits_gain = _gain(
        direct.get("hits", {}),
        metrics.get("structured_counts", {}).get("hits", {}),
    )
    event_hr_gain = _gain(
        direct.get("home_runs", {}),
        metrics.get("structured_counts", {}).get("home_runs", {}),
    )
    try:
        hr_any_gain = float(hr_any.get("prior_brier")) - float(hr_any.get("model_brier"))
    except Exception:
        hr_any_gain = None
    return {
        "production_status": "diagnostic_only",
        "reason": "Require repeated holdout gains before replacing prop projections.",
        "holdout_rows": holdout_rows,
        "pa_mae_gain_vs_slot_prior": pa_gain,
        "hits_mae_gain_vs_slot_rate_prior": hits_gain,
        "tb_mae_gain_vs_slot_rate_prior": tb_gain,
        "hr_mae_gain_vs_slot_rate_prior": hr_gain,
        "direct_event_hits_mae_gain_vs_independent_rates": event_hits_gain,
        "direct_event_tb_mae_gain_vs_independent_rates": event_tb_gain,
        "direct_event_hr_mae_gain_vs_independent_rates": event_hr_gain,
        "hr_any_brier_gain_vs_prior": hr_any_gain,
        "passes_basic_gate": bool(
            holdout_rows >= cfg.min_holdout_rows
            and pa_gain is not None and pa_gain > 0.01
            and hits_gain is not None and hits_gain > 0.005
            and tb_gain is not None and tb_gain > 0.005
            and event_tb_gain is not None and event_tb_gain > 0.0
            and hr_any_gain is not None and hr_any_gain > 0.0005
        ),
    }


def _write_outputs(payload: dict[str, Any], cfg: HitterOutcomeModelConfig, models: dict[str, Any] | None) -> None:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    json_path = cfg.model_dir / "hitter_player_game_outcome_models.json"
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    if models:
        joblib.dump(
            {
                "models": models,
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "event_classes": EVENT_CLASSES,
                "active_event_model": ((payload.get("metrics") or {}).get("direct_event_model") or {}).get("active_event_model"),
                "trained_at_utc": payload.get("generated_at_utc"),
                "recommendation": payload.get("recommendation"),
            },
            cfg.model_dir / "hitter_player_game_outcome_models.joblib",
        )
    _write_report(payload, cfg.report_file)


def _write_report(payload: dict[str, Any], report_file: str | None) -> None:
    path = _REPORT_DIR / (report_file or "mlb_hitter_player_game_outcome_models_latest.md")
    path.parent.mkdir(parents=True, exist_ok=True)

    def num(value: Any, digits: int = 3) -> str:
        try:
            if value is None:
                return "-"
            return f"{float(value):.{digits}f}"
        except Exception:
            return "-"

    metrics = payload.get("metrics", {})
    rec = payload.get("recommendation", {})
    pa = metrics.get("pa_model", {})
    structured = metrics.get("structured_counts", {})
    direct = metrics.get("direct_event_model", {})
    hr_any = metrics.get("hr_any_model", {})
    prop = metrics.get("existing_prop_projection_holdout", {})

    lines = [
        "# MLB Hitter Player-Game Outcome Models",
        "",
        f"Generated: {payload.get('generated_at_utc')}",
        f"Rows: {payload.get('rows', 0)} | Train: {payload.get('train_rows', 0)} | Holdout: {payload.get('holdout_rows', 0)}",
        f"Holdout: {payload.get('holdout_start')} to {payload.get('holdout_end')}",
        f"Status: {payload.get('status')}",
        "",
        "## Recommendation",
        "",
        f"- Production status: {rec.get('production_status', 'unknown')}",
        f"- Passes basic gate: {rec.get('passes_basic_gate', False)}",
        f"- PA MAE gain vs slot prior: {num(rec.get('pa_mae_gain_vs_slot_prior'))}",
        f"- Hits MAE gain vs slot-rate prior: {num(rec.get('hits_mae_gain_vs_slot_rate_prior'))}",
        f"- TB MAE gain vs slot-rate prior: {num(rec.get('tb_mae_gain_vs_slot_rate_prior'))}",
        f"- HR MAE gain vs slot-rate prior: {num(rec.get('hr_mae_gain_vs_slot_rate_prior'))}",
        f"- Direct event TB MAE gain vs independent rates: {num(rec.get('direct_event_tb_mae_gain_vs_independent_rates'))}",
        f"- HR-any Brier gain vs prior: {num(rec.get('hr_any_brier_gain_vs_prior'), 5)}",
        "",
        "## Feature Coverage",
        "",
        "| Feature | Coverage |",
        "|---|---:|",
    ]
    coverage = payload.get("feature_coverage", {})
    for col in [
        "park_run_factor",
        "park_hr_factor",
        "park_babip_factor",
        "own_lineup_xwoba_avg",
        "own_lineup_barrel_avg",
        "lineup_confirmed_flag",
        "confirmed_team_lineup_slots",
        "team_lineup_confirmed_flag",
        "batter_sc_barrel_rate",
        "batter_sc_xwoba",
        "batter_sc_xslg",
        "batter_sprint_speed",
        "batter_disc_whiff_pct",
        "opp_sp_sc_barrel_rate",
        "opp_sp_sc_xwoba",
        "opp_sp_fb_pct",
        "opp_sp_fb_xwoba",
        "opp_sp_sl_pct",
        "opp_sp_ch_pct",
        "opp_sp_fastball_family_pct",
        "opp_sp_pitch_diversity",
    ]:
        val = coverage.get(col)
        lines.append(f"| {col} | {num(None if val is None else float(val) * 100.0, 1)}% |")
    lines.extend([
        "",
        "## Opportunity",
        "",
        "| Model | Rows | MAE | RMSE | Bias |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, key in [("PA model", "model"), ("Slot prior", "slot_prior"), ("Existing projected PA", "existing_projected_pa")]:
        row = pa.get(key, {})
        lines.append(
            f"| {label} | {row.get('rows', 0)} | {num(row.get('mae'))} | "
            f"{num(row.get('rmse'))} | {num(row.get('bias'))} |"
        )
    lines.extend([
        "",
        "## Structured Counts",
        "",
        "| Target | Model Rows | Model MAE | Prior MAE | Model Bias |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, model_key, prior_key in [
        ("Hits", "hits", "slot_prior_hits"),
        ("Total bases", "total_bases", "slot_prior_total_bases"),
        ("Home runs", "home_runs", "slot_prior_home_runs"),
    ]:
        m = structured.get(model_key, {})
        p = structured.get(prior_key, {})
        lines.append(
            f"| {label} | {m.get('rows', 0)} | {num(m.get('mae'))} | "
            f"{num(p.get('mae'))} | {num(m.get('bias'))} |"
        )
    lines.extend([
        "",
        "## Direct Per-PA Event Model",
        "",
        f"- Active event curve: {direct.get('active_event_model', '-')}",
        f"- Train event rows: {direct.get('train_event_rows', 0)}",
        f"- Holdout player-games: {direct.get('holdout_rows', 0)}",
        f"- Weighted event log loss: {num(direct.get('weighted_event_log_loss'), 5)}",
        f"- Classes: {', '.join(map(str, direct.get('classes', []))) if direct.get('classes') else '-'}",
        "",
        "| Target | Rows | Direct Event MAE | Independent Rate MAE | Direct Bias |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, key in [("Hits", "hits"), ("Total bases", "total_bases"), ("Home runs", "home_runs")]:
        d = direct.get(key, {})
        s = structured.get(key, {})
        lines.append(
            f"| {label} | {d.get('rows', 0)} | {num(d.get('mae'))} | "
            f"{num(s.get('mae'))} | {num(d.get('bias'))} |"
        )
    boosted = direct.get("boosted_binary") or {}
    linear = direct.get("linear_multinomial") or {}
    if boosted or linear:
        lines.extend([
            "",
            "## Event Model Candidates",
            "",
            "| Candidate | Log Loss | Hits MAE | TB MAE | HR MAE |",
            "|---|---:|---:|---:|---:|",
        ])
        for label, row in [("linear_multinomial", linear), ("boosted_binary_calibrated", boosted)]:
            if not row:
                continue
            lines.append(
                f"| {label} | {num(row.get('weighted_event_log_loss'), 5)} | "
                f"{num((row.get('hits') or {}).get('mae'))} | "
                f"{num((row.get('total_bases') or {}).get('mae'))} | "
                f"{num((row.get('home_runs') or {}).get('mae'))} |"
            )
    event_rates = (direct.get("event_rates") or {})
    if event_rates:
        lines.extend([
            "",
            "| Event | Actual / PA | Predicted Prob | Bias / PA |",
            "|---|---:|---:|---:|",
        ])
        for cls in EVENT_CLASSES:
            rec_rate = event_rates.get(cls, {})
            lines.append(
                f"| {cls} | {num(rec_rate.get('actual_per_pa'), 4)} | "
                f"{num(rec_rate.get('pred_mean_prob'), 4)} | {num(rec_rate.get('bias_per_pa'), 4)} |"
            )
    lines.extend([
        "",
        "## HR Rare Event",
        "",
        f"- Rows: {hr_any.get('rows', 0)}",
        f"- Model Brier: {num(hr_any.get('model_brier'), 5)}",
        f"- Prior Brier: {num(hr_any.get('prior_brier'), 5)}",
        f"- AUC: {num(hr_any.get('auc'), 3)}",
        "",
        "## Existing Prop Projection Holdout",
        "",
        "| Target | Rows | MAE | RMSE | Bias |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, key in [("Hits", "hits"), ("Total bases", "total_bases"), ("Home runs", "home_runs")]:
        row = prop.get(key, {})
        lines.append(
            f"| {label} | {row.get('rows', 0)} | {num(row.get('mae'))} | "
            f"{num(row.get('rmse'))} | {num(row.get('bias'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    try:
        import decimal

        if isinstance(value, decimal.Decimal):
            return float(value)
    except Exception:
        pass
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hitter player-game outcome models.")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--holdout-days", type=int, default=28)
    parser.add_argument("--min-train-rows", type=int, default=1000)
    parser.add_argument("--min-holdout-rows", type=int, default=200)
    parser.add_argument("--report-file")
    args = parser.parse_args()

    cfg = HitterOutcomeModelConfig(
        pg_dsn=args.pg_dsn,
        lookback_days=args.lookback_days,
        holdout_days=args.holdout_days,
        min_train_rows=args.min_train_rows,
        min_holdout_rows=args.min_holdout_rows,
        report_file=args.report_file,
    )
    print(json.dumps(train_hitter_player_game_outcomes(cfg), indent=2, default=_json_default))


if __name__ == "__main__":
    main()
