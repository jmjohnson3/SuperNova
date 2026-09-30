"""Compare live hitter rate scoring against the legacy production forecasts."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN as _PG_DSN

from .daily_forecast_ledger import ensure_daily_forecast_ledger_schema, grade_daily_forecasts
from .model_release import canonical_forecast_phase

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_HITTER_STATS = ("batter_hits", "batter_home_runs", "batter_total_bases")
_HITS_LIVE_BIAS_ALPHAS = (0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95, 1.00)
_HITS_LIVE_BIAS_GROUP_SPECS = (
    ("player_id", 0.32, 24.0),
    ("lineup_pa_bucket", 0.22, 80.0),
    ("lineup_hand_bucket", 0.16, 100.0),
    ("pa_team_total_bucket", 0.12, 120.0),
    ("home_away_bucket", 0.08, 140.0),
    ("lineup_bucket", 0.06, 160.0),
    ("handedness_bucket", 0.04, 160.0),
)


@dataclass(frozen=True)
class DiffConfig:
    pg_dsn: str = _PG_DSN
    lookback_days: int = 45
    phase: str | None = None
    top_n: int = 30
    grade: bool = False
    artifact_file: str = "hitter_live_vs_legacy_forecast_diff.json"
    report_file: str = "mlb_hitter_live_vs_legacy_forecast_diff_latest.md"


def _table_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (f"{schema}.{table}",))
        return bool(cur.fetchone()[0])


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def _as_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any, digits: int = 3) -> str:
    number = _as_float(value)
    return "-" if number is None else f"{number:.{digits}f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is pd.NA:
        return None
    return value


def _poisson_over_probability(mean: Any, line: float) -> float | None:
    value = _as_float(mean)
    if value is None:
        return None
    value = max(0.0, min(12.0, value))
    threshold = int(math.floor(float(line)))
    cdf = 0.0
    term = math.exp(-value)
    cdf += term
    for k in range(1, threshold + 1):
        term *= value / k
        cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))


def _context_value(context: Any, *keys: str) -> Any:
    if not isinstance(context, dict):
        return None
    current: Any = context
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _reason_tags(row: pd.Series) -> list[str]:
    tags: list[str] = []
    slot = _as_float(row.get("lineup_slot"))
    pa = _as_float(row.get("projected_pa"))
    team_runs = _as_float(row.get("team_implied_runs"))
    hits_delta = _as_float(row.get("hits_delta")) or 0.0
    hr_delta = _as_float(row.get("hr_delta")) or 0.0
    tb_delta = _as_float(row.get("tb_delta")) or 0.0
    batter_hand = str(row.get("batter_hand") or "").upper()
    pitcher_hand = str(row.get("opp_sp_hand") or "").upper()
    park_hr = _as_float(row.get("park_hr_factor"))

    if slot is None:
        tags.append("lineup slot missing")
    elif slot <= 2:
        tags.append("top-order PA support")
    elif slot >= 7:
        tags.append("bottom-order PA drag")
    else:
        tags.append("middle-order neutral")

    if pa is None:
        tags.append("PA unknown")
    elif pa >= 4.35:
        tags.append("high projected PA")
    elif pa <= 3.55:
        tags.append("low projected PA")

    if team_runs is not None:
        if team_runs >= 4.8:
            tags.append("strong run environment")
        elif team_runs <= 3.8:
            tags.append("low run environment")

    if batter_hand and pitcher_hand and batter_hand not in {"UNKNOWN", "NAN"} and pitcher_hand not in {"UNKNOWN", "NAN"}:
        tags.append("opposite-hand matchup" if batter_hand != pitcher_hand else "same-hand matchup")
    else:
        tags.append("pitcher handedness missing")

    if park_hr is None:
        tags.append("park factor missing")
    elif park_hr >= 1.05:
        tags.append("HR-friendly park")
    elif park_hr <= 0.95:
        tags.append("HR-suppressing park")

    if abs(tb_delta) >= 0.10:
        if hr_delta > 0.02 and tb_delta > 0:
            tags.append("TB lift is HR-tail driven")
        elif hits_delta > 0.08 and tb_delta > 0:
            tags.append("TB lift is hit-rate driven")
        elif tb_delta < 0:
            tags.append("TB rebuild pulled projection down")

    return tags[:7]


def _latest_rows(conn, cutoff: date, phase: str) -> pd.DataFrame:
    return _query_df(
        conn,
        """
        WITH canonical AS (
            SELECT DISTINCT ON (
                game_date_et, game_slug, player_id, stat,
                COALESCE(model_family, 'unknown'), COALESCE(model_version, 'unknown')
            )
                id, forecast_run_id, forecast_phase, locked_at_utc, game_date_et,
                game_slug, player_id, player_name, team_abbr, opponent_abbr, stat,
                projection_value::float AS projection_value,
                model_only_projection::float AS model_only_projection,
                baseline_value::float AS baseline_value,
                baseline_source,
                actual_value::float AS actual_value,
                result_status,
                opportunity_context,
                COALESCE(model_family, 'unknown') AS model_family,
                COALESCE(model_version, 'unknown') AS model_version
            FROM bets.mlb_daily_forecast_ledger
            WHERE game_date_et >= %(cutoff)s
              AND forecast_phase = %(phase)s
              AND forecast_type = 'player'
              AND stat = ANY(%(stats)s::text[])
              AND player_id IS NOT NULL
            ORDER BY game_date_et, game_slug, player_id, stat,
                     COALESCE(model_family, 'unknown'), COALESCE(model_version, 'unknown'),
                     locked_at_utc DESC, id DESC
        )
        SELECT *
        FROM canonical
        ORDER BY game_date_et DESC, locked_at_utc DESC, player_name, stat
        """,
        {"cutoff": cutoff, "phase": phase, "stats": list(_HITTER_STATS)},
    )


def _live_row(df: pd.DataFrame, stat: str, player_key: tuple[Any, ...]) -> pd.Series | None:
    subset = df[
        (df["game_date_et"].eq(player_key[0]))
        & (df["game_slug"].eq(player_key[1]))
        & (df["player_id"].eq(player_key[2]))
        & (df["stat"].eq(stat))
    ].copy()
    if subset.empty:
        return None
    if stat == "batter_hits":
        preferred = subset[subset["model_family"].eq("hitter_hits_direct_player_game_blend")]
    elif stat == "batter_home_runs":
        preferred = subset[subset["model_family"].eq("hitter_hr_direct_player_game_blend")]
    elif stat == "batter_total_bases":
        preferred = subset[subset["model_family"].eq("hitter_tb_component_rebuild_from_rate_challengers")]
    else:
        preferred = pd.DataFrame()
    if preferred.empty:
        preferred = subset[~subset["model_family"].str.contains("shadow", case=False, na=False)]
    return preferred.iloc[0] if not preferred.empty else subset.iloc[0]


def _shadow_row(df: pd.DataFrame, stat: str, player_key: tuple[Any, ...]) -> pd.Series | None:
    subset = df[
        (df["game_date_et"].eq(player_key[0]))
        & (df["game_slug"].eq(player_key[1]))
        & (df["player_id"].eq(player_key[2]))
        & (df["stat"].eq(stat))
        & (df["model_family"].eq("hitter_rate_shadow_direct_player_game_blend"))
    ].copy()
    return None if subset.empty else subset.iloc[0]


def _build_player_diff(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    live_rows: dict[tuple[tuple[Any, Any, Any], str], dict[str, Any]] = {}
    shadow_rows: dict[tuple[tuple[Any, Any, Any], str], dict[str, Any]] = {}

    def live_rank(record: dict[str, Any]) -> int:
        stat = str(record.get("stat") or "")
        family = str(record.get("model_family") or "")
        if stat == "batter_hits" and family == "hitter_hits_direct_player_game_blend":
            return 0
        if stat == "batter_home_runs" and family == "hitter_hr_direct_player_game_blend":
            return 0
        if stat == "batter_total_bases" and family == "hitter_tb_component_rebuild_from_rate_challengers":
            return 0
        if "shadow" not in family.lower():
            return 1
        return 9

    keys: set[tuple[Any, Any, Any]] = set()
    for record in df.to_dict(orient="records"):
        key = (record.get("game_date_et"), record.get("game_slug"), record.get("player_id"))
        stat = str(record.get("stat") or "")
        if key[2] is None or stat not in _HITTER_STATS:
            continue
        keys.add(key)
        map_key = (key, stat)
        family = str(record.get("model_family") or "")
        if family == "hitter_rate_shadow_direct_player_game_blend":
            shadow_rows.setdefault(map_key, record)
            continue
        current = live_rows.get(map_key)
        if current is None or live_rank(record) < live_rank(current):
            live_rows[map_key] = record

    records: list[dict[str, Any]] = []
    for key in sorted(keys, reverse=True):
        hit_live = live_rows.get((key, "batter_hits"))
        hr_live = live_rows.get((key, "batter_home_runs"))
        tb_live = live_rows.get((key, "batter_total_bases"))
        if hit_live is None and hr_live is None and tb_live is None:
            continue
        hit_shadow = shadow_rows.get((key, "batter_hits"))
        hr_shadow = shadow_rows.get((key, "batter_home_runs"))
        context = {}
        for source in (tb_live, hit_live, hr_live, hit_shadow, hr_shadow):
            if source is not None and isinstance(source.get("opportunity_context"), dict):
                context = source.get("opportunity_context") or {}
                break
        tb_meta = _context_value(context, "tb_component_rebuild") or {}
        player_name = (
            (tb_live if tb_live is not None else hit_live if hit_live is not None else hr_live)["player_name"]
        )
        hits_live = _as_float(hit_shadow.get("projection_value")) if hit_shadow is not None else _as_float(hit_live.get("projection_value") if hit_live is not None else None)
        hits_legacy = _as_float(hit_shadow.get("baseline_value")) if hit_shadow is not None else None
        hr_live_value = _as_float(hr_shadow.get("projection_value")) if hr_shadow is not None else _as_float(hr_live.get("projection_value") if hr_live is not None else None)
        hr_legacy = _as_float(hr_shadow.get("baseline_value")) if hr_shadow is not None else None
        tb_live_value = _as_float(tb_live.get("projection_value") if tb_live is not None else None)
        tb_legacy = _as_float(tb_meta.get("legacy_tb"))
        source_row = tb_live if tb_live is not None else hit_live if hit_live is not None else hr_live
        actual_hit_source = hit_live if hit_live is not None else hit_shadow
        actual_hr_source = hr_live if hr_live is not None else hr_shadow
        record = {
            "game_date_et": key[0],
            "game_slug": key[1],
            "player_id": int(key[2]) if key[2] is not None else None,
            "player_name": player_name,
            "team_abbr": source_row.get("team_abbr"),
            "opponent_abbr": source_row.get("opponent_abbr"),
            "locked_at_utc": str(source_row.get("locked_at_utc")),
            "result_status": str(source_row.get("result_status")),
            "legacy_hits": hits_legacy,
            "live_hits": hits_live,
            "hits_delta": None if hits_live is None or hits_legacy is None else hits_live - hits_legacy,
            "actual_hits": _as_float(actual_hit_source.get("actual_value") if actual_hit_source is not None else None),
            "legacy_hr": hr_legacy,
            "live_hr": hr_live_value,
            "hr_delta": None if hr_live_value is None or hr_legacy is None else hr_live_value - hr_legacy,
            "actual_hr": _as_float(actual_hr_source.get("actual_value") if actual_hr_source is not None else None),
            "legacy_tb": tb_legacy,
            "live_tb": tb_live_value,
            "tb_delta": None if tb_live_value is None or tb_legacy is None else tb_live_value - tb_legacy,
            "actual_tb": _as_float(tb_live.get("actual_value") if tb_live is not None else None),
            "tb_component": _as_float(tb_meta.get("component_tb")),
            "tb_component_alpha": _as_float(tb_meta.get("component_blend_alpha")),
            "non_hr_extra_per_non_hr_hit": _as_float(tb_meta.get("non_hr_extra_per_non_hr_hit")),
            "lineup_slot": _as_float(context.get("effective_batting_order") or context.get("confirmed_batting_order")),
            "projected_pa": _as_float(context.get("projected_pa")),
            "pa_model_source": context.get("pa_model_source"),
            "team_implied_runs": _as_float(context.get("team_implied_runs")),
            "batter_hand": context.get("batter_hand"),
            "opp_sp_hand": context.get("opp_sp_hand"),
            "park_factor": _as_float(context.get("park_factor") or context.get("venue_run_factor")),
            "park_hr_factor": _as_float(context.get("park_hr_factor") or context.get("venue_hr_factor")),
            "hit_model_family": None if hit_live is None else hit_live.get("model_family"),
            "hr_model_family": None if hr_live is None else hr_live.get("model_family"),
            "tb_model_family": None if tb_live is None else tb_live.get("model_family"),
        }
        deltas = [
            abs(float(value)) for value in (record["hits_delta"], record["hr_delta"], record["tb_delta"])
            if value is not None and math.isfinite(float(value))
        ]
        record["max_abs_delta"] = max(deltas) if deltas else 0.0
        records.append(record)
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["context_tags"] = out.apply(_reason_tags, axis=1)
    return out.sort_values(["game_date_et", "max_abs_delta"], ascending=[False, False]).reset_index(drop=True)


def _metric_pair(df: pd.DataFrame, live_col: str, legacy_col: str, actual_col: str) -> dict[str, Any]:
    work = df.dropna(subset=[live_col, legacy_col, actual_col]).copy()
    if work.empty:
        return {"rows": 0}
    live_error = pd.to_numeric(work[live_col], errors="coerce") - pd.to_numeric(work[actual_col], errors="coerce")
    legacy_error = pd.to_numeric(work[legacy_col], errors="coerce") - pd.to_numeric(work[actual_col], errors="coerce")
    return {
        "rows": int(len(work)),
        "dates": int(work["game_date_et"].nunique()),
        "legacy_mae": float(legacy_error.abs().mean()),
        "live_mae": float(live_error.abs().mean()),
        "mae_gain": float(legacy_error.abs().mean() - live_error.abs().mean()),
        "legacy_bias": float(legacy_error.mean()),
        "live_bias": float(live_error.mean()),
    }


def _binary_brier(df: pd.DataFrame, live_col: str, legacy_col: str, actual_col: str, line: float) -> dict[str, Any]:
    rows = []
    for record in df.to_dict(orient="records"):
        actual = _as_float(record.get(actual_col))
        live_prob = _poisson_over_probability(record.get(live_col), line)
        legacy_prob = _poisson_over_probability(record.get(legacy_col), line)
        if actual is None or live_prob is None or legacy_prob is None:
            continue
        target = 1.0 if actual > line else 0.0
        rows.append((live_prob, legacy_prob, target))
    if not rows:
        return {"rows": 0}
    arr = np.asarray(rows, dtype=float)
    live = arr[:, 0]
    legacy = arr[:, 1]
    target = arr[:, 2]
    return {
        "rows": int(len(rows)),
        "line": line,
        "legacy_brier": float(np.mean(np.square(legacy - target))),
        "live_brier": float(np.mean(np.square(live - target))),
        "brier_gain": float(np.mean(np.square(legacy - target)) - np.mean(np.square(live - target))),
    }


def _count_metrics(actual: pd.Series, pred: pd.Series) -> dict[str, Any]:
    work = pd.DataFrame({"actual": actual, "pred": pred}).dropna()
    if work.empty:
        return {"rows": 0, "mae": None, "bias": None}
    error = pd.to_numeric(work["pred"], errors="coerce") - pd.to_numeric(work["actual"], errors="coerce")
    return {
        "rows": int(len(work)),
        "mae": float(error.abs().mean()),
        "bias": float(error.mean()),
    }


def _hit_any_brier(actual: pd.Series, pred: pd.Series) -> dict[str, Any]:
    work = pd.DataFrame({"actual": actual, "pred": pred}).dropna()
    if work.empty:
        return {"rows": 0, "brier": None}
    target = pd.to_numeric(work["actual"], errors="coerce").gt(0).astype(float)
    lam = pd.to_numeric(work["pred"], errors="coerce").clip(lower=0.0, upper=8.0)
    probability = (1.0 - np.exp(-lam)).clip(1e-6, 1.0 - 1e-6)
    return {
        "rows": int(len(work)),
        "brier": float(np.mean(np.square(probability - target))),
    }


def _hits_bias_bucketize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    slot = pd.to_numeric(out.get("lineup_slot"), errors="coerce")
    out["lineup_bucket"] = pd.Series(
        np.select(
            [slot.between(1, 2), slot.between(3, 5), slot.between(6, 9)],
            ["slot_1_2", "slot_3_5", "slot_6_9"],
            default="slot_unknown",
        ),
        index=out.index,
    )
    pa = pd.to_numeric(out.get("projected_pa"), errors="coerce")
    out["projected_pa_bucket"] = pd.Series(
        np.select(
            [pa >= 4.4, pa.between(3.8, 4.4, inclusive="left"), pa < 3.8],
            ["projected_pa_high_4_4_plus", "projected_pa_mid_3_8_4_4", "projected_pa_low_under_3_8"],
            default="projected_pa_missing",
        ),
        index=out.index,
    )
    batter_hand = out.get("batter_hand", pd.Series(index=out.index, dtype=object)).fillna("unknown").astype(str).str.upper()
    pitcher_hand = out.get("opp_sp_hand", pd.Series(index=out.index, dtype=object)).fillna("unknown").astype(str).str.upper()
    out["handedness_bucket"] = pd.Series(
        np.where(
            batter_hand.isin(["L", "R"]) & pitcher_hand.isin(["L", "R"]),
            np.where(batter_hand.eq(pitcher_hand), "same_hand", "opposite_hand"),
            "unknown_hand",
        ),
        index=out.index,
    )
    team_runs = pd.to_numeric(out.get("team_implied_runs"), errors="coerce")
    out["team_total_bucket"] = pd.Series(
        np.select(
            [team_runs >= 4.8, team_runs.between(4.0, 4.8, inclusive="left"), team_runs < 4.0],
            ["team_total_high", "team_total_mid", "team_total_low"],
            default="team_total_missing",
        ),
        index=out.index,
    )
    is_home = pd.to_numeric(out.get("is_home"), errors="coerce")
    if "is_home" not in out or is_home.notna().sum() == 0:
        out["home_away_bucket"] = "missing"
    else:
        out["home_away_bucket"] = np.where(is_home >= 0.5, "home", "away")
    out["lineup_pa_bucket"] = out["lineup_bucket"].astype(str) + "|" + out["projected_pa_bucket"].astype(str)
    out["lineup_hand_bucket"] = out["lineup_bucket"].astype(str) + "|" + out["handedness_bucket"].astype(str)
    out["pa_team_total_bucket"] = out["projected_pa_bucket"].astype(str) + "|" + out["team_total_bucket"].astype(str)
    return out


def _shrunk_mean_map(train: pd.DataFrame, key: str, target: str, shrink_rows: float) -> tuple[dict[str, float], float]:
    values = pd.to_numeric(train[target], errors="coerce")
    global_mean = float(values.mean()) if values.notna().any() else 0.0
    mapping: dict[str, float] = {}
    if key not in train:
        return mapping, global_mean
    for value, group in train.groupby(key, dropna=False):
        group_values = pd.to_numeric(group[target], errors="coerce").dropna()
        if group_values.empty:
            continue
        n = float(len(group_values))
        weight = n / (n + shrink_rows)
        mapping[str(value)] = float(weight * group_values.mean() + (1.0 - weight) * global_mean)
    return mapping, global_mean


def _score_hits_bias_maps(test: pd.DataFrame, maps: list[dict[str, Any]], global_mean: float) -> pd.Series:
    residual = pd.Series(0.0, index=test.index, dtype=float)
    total_weight = 0.0
    for spec in maps:
        key = str(spec.get("key") or "")
        if key not in test:
            continue
        weight = _as_float(spec.get("weight")) or 0.0
        if weight <= 0.0:
            continue
        fallback = _as_float(spec.get("fallback"))
        if fallback is None:
            fallback = global_mean
        values = spec.get("values") or {}
        mapped = test[key].astype(str).map(values).fillna(float(fallback)).astype(float)
        residual = residual + weight * mapped
        total_weight += weight
    if total_weight <= 0:
        return pd.Series(float(global_mean), index=test.index, dtype=float)
    return (residual / total_weight).clip(lower=-1.10, upper=1.10)


def _fit_hits_live_bias_maps(train: pd.DataFrame) -> tuple[list[dict[str, Any]], float]:
    residual = pd.to_numeric(train.get("live_bias_residual"), errors="coerce")
    global_mean = float(residual.mean()) if residual.notna().any() else 0.0
    maps: list[dict[str, Any]] = []
    for key, weight, shrink_rows in _HITS_LIVE_BIAS_GROUP_SPECS:
        mapping, fallback = _shrunk_mean_map(train, key, "live_bias_residual", shrink_rows)
        maps.append({
            "key": key,
            "weight": float(weight),
            "shrink_rows": float(shrink_rows),
            "fallback": float(fallback),
            "values": mapping,
        })
    return maps, global_mean


def _hits_live_bias_repair(diff: pd.DataFrame) -> dict[str, Any]:
    required = {"game_date_et", "player_id", "live_hits", "legacy_hits", "actual_hits", "result_status"}
    if diff.empty or not required.issubset(set(diff.columns)):
        return {"status": "missing_required_columns", "accepted": False, "selected_alpha": 0.0}
    work = diff.loc[diff["result_status"].eq("graded")].dropna(
        subset=["live_hits", "legacy_hits", "actual_hits"]
    ).copy()
    if work.empty:
        return {"status": "no_graded_hits_rows", "accepted": False, "selected_alpha": 0.0}
    work = _hits_bias_bucketize(work)
    work["live_hits"] = pd.to_numeric(work["live_hits"], errors="coerce").clip(lower=0.0, upper=5.0)
    work["legacy_hits"] = pd.to_numeric(work["legacy_hits"], errors="coerce").clip(lower=0.0, upper=5.0)
    work["actual_hits"] = pd.to_numeric(work["actual_hits"], errors="coerce")
    work["live_bias_residual"] = (work["actual_hits"] - work["live_hits"]).clip(lower=-2.0, upper=2.0)
    dates = sorted(pd.to_datetime(work["game_date_et"]).dt.date.unique())
    repaired = pd.Series(np.nan, index=work.index, dtype=float)
    folds: list[dict[str, Any]] = []
    for holdout_date in dates:
        train_mask = pd.to_datetime(work["game_date_et"]).dt.date < holdout_date
        test_mask = pd.to_datetime(work["game_date_et"]).dt.date == holdout_date
        if int(train_mask.sum()) < 300 or int(test_mask.sum()) == 0:
            continue
        train = work.loc[train_mask].copy()
        test = work.loc[test_mask].copy()
        maps, global_mean = _fit_hits_live_bias_maps(train)
        residual = _score_hits_bias_maps(test, maps, global_mean)
        repaired.loc[test.index] = (test["live_hits"] + residual).clip(lower=0.0, upper=5.0)
        folds.append({
            "holdout_date": str(holdout_date),
            "train_rows": int(train_mask.sum()),
            "holdout_rows": int(test_mask.sum()),
            "global_residual": float(global_mean),
        })
    valid = repaired.notna()
    if not bool(valid.any()):
        return {
            "status": "no_oof_repair_rows",
            "accepted": False,
            "selected_alpha": 0.0,
            "rows": int(len(work)),
            "dates": int(len(dates)),
            "folds": folds,
        }
    valid_work = work.loc[valid].copy()
    live = valid_work["live_hits"]
    legacy = valid_work["legacy_hits"]
    actual = valid_work["actual_hits"]
    raw_repaired = pd.to_numeric(repaired.loc[valid], errors="coerce")
    live_metrics = _count_metrics(actual, live)
    legacy_metrics = _count_metrics(actual, legacy)
    live_brier = _hit_any_brier(actual, live)
    legacy_brier = _hit_any_brier(actual, legacy)
    records: list[dict[str, Any]] = []
    for alpha in _HITS_LIVE_BIAS_ALPHAS:
        candidate = (live + alpha * (raw_repaired - live)).clip(lower=0.0, upper=5.0)
        metrics = _count_metrics(actual, candidate)
        brier = _hit_any_brier(actual, candidate)
        records.append({
            "alpha": float(alpha),
            "rows": metrics.get("rows", 0),
            "mae": metrics.get("mae"),
            "bias": metrics.get("bias"),
            "mae_gain_vs_live": (
                float(live_metrics["mae"] - metrics["mae"])
                if live_metrics.get("mae") is not None and metrics.get("mae") is not None
                else None
            ),
            "mae_gain_vs_legacy": (
                float(legacy_metrics["mae"] - metrics["mae"])
                if legacy_metrics.get("mae") is not None and metrics.get("mae") is not None
                else None
            ),
            "any_brier": brier.get("brier"),
            "any_brier_gain_vs_live": (
                float(live_brier["brier"] - brier["brier"])
                if live_brier.get("brier") is not None and brier.get("brier") is not None
                else None
            ),
            "any_brier_gain_vs_legacy": (
                float(legacy_brier["brier"] - brier["brier"])
                if legacy_brier.get("brier") is not None and brier.get("brier") is not None
                else None
            ),
            "mean_shift_vs_live": float((candidate - live).mean()),
        })
    accepted = [
        row for row in records
        if float(row.get("alpha") or 0.0) > 0.0
        and (row.get("mae_gain_vs_live") is not None and row["mae_gain_vs_live"] >= 0.010)
        and (row.get("mae_gain_vs_legacy") is not None and row["mae_gain_vs_legacy"] >= 0.000)
        and (row.get("any_brier_gain_vs_live") is not None and row["any_brier_gain_vs_live"] >= 0.0005)
        and (row.get("any_brier_gain_vs_legacy") is not None and row["any_brier_gain_vs_legacy"] >= 0.0000)
        and abs(float(row.get("bias") or 0.0)) <= min(abs(float(live_metrics.get("bias") or 0.0)), abs(float(legacy_metrics.get("bias") or 0.0))) + 0.05
    ]
    if accepted:
        selected_record = sorted(
            accepted,
            key=lambda row: (
                -float(row.get("mae_gain_vs_legacy") or 0.0),
                -float(row.get("any_brier_gain_vs_legacy") or 0.0),
                float(row.get("alpha") or 0.0),
            ),
        )[0]
        accepted_flag = True
        reason = "repaired_hits_beats_live_and_legacy"
    else:
        selected_record = records[0]
        accepted_flag = False
        missing = []
        positives = [row for row in records if float(row.get("alpha") or 0.0) > 0.0]
        if not any((row.get("mae_gain_vs_live") is not None and row["mae_gain_vs_live"] >= 0.010) for row in positives):
            missing.append("mae_not_improved_vs_live")
        if not any((row.get("mae_gain_vs_legacy") is not None and row["mae_gain_vs_legacy"] >= 0.000) for row in positives):
            missing.append("does_not_beat_legacy")
        if not any((row.get("any_brier_gain_vs_live") is not None and row["any_brier_gain_vs_live"] >= 0.0005) for row in positives):
            missing.append("any_brier_not_improved_vs_live")
        if not any((row.get("any_brier_gain_vs_legacy") is not None and row["any_brier_gain_vs_legacy"] >= 0.0000) for row in positives):
            missing.append("any_brier_does_not_beat_legacy")
        reason = ",".join(missing or ["bias_or_threshold_gate_failed"])
    production_maps, production_global = _fit_hits_live_bias_maps(work)
    return {
        "status": "ready",
        "source": "hitter_live_vs_legacy_forecast_diff",
        "accepted": bool(accepted_flag),
        "selected_alpha": float(selected_record.get("alpha") or 0.0),
        "reason": reason,
        "rows": int(len(valid_work)),
        "dates": int(pd.to_datetime(valid_work["game_date_et"]).dt.date.nunique()),
        "live": {**live_metrics, "any_brier": live_brier.get("brier")},
        "legacy": {**legacy_metrics, "any_brier": legacy_brier.get("brier")},
        "selected": selected_record,
        "candidate_records": records,
        "folds": folds,
        "production_maps": production_maps,
        "production_global_residual": float(production_global),
        "live_usage": (
            "eligible_for_shadow_and_future_live_gate"
            if accepted_flag
            else "diagnostic_only_until_repair_beats_legacy"
        ),
    }


def _graded_comparison(diff: pd.DataFrame) -> dict[str, Any]:
    if diff.empty:
        return {}
    graded = diff[diff["result_status"].eq("graded")].copy()
    out = {
        "graded_rows": int(len(graded)),
        "hits": _metric_pair(graded, "live_hits", "legacy_hits", "actual_hits"),
        "home_runs": _metric_pair(graded, "live_hr", "legacy_hr", "actual_hr"),
        "total_bases": _metric_pair(graded, "live_tb", "legacy_tb", "actual_tb"),
        "tb_line_brier": {
            "TB 1.5": _binary_brier(graded, "live_tb", "legacy_tb", "actual_tb", 1.5),
            "TB 2.5": _binary_brier(graded, "live_tb", "legacy_tb", "actual_tb", 2.5),
            "TB 3.5": _binary_brier(graded, "live_tb", "legacy_tb", "actual_tb", 3.5),
        },
    }
    hr_brier = _binary_brier(graded, "live_hr", "legacy_hr", "actual_hr", 0.5)
    out["home_runs"]["hr_0_5_brier"] = hr_brier
    return out


def _clv_roi_true_pair(conn, cutoff: date, phase: str) -> dict[str, Any]:
    if not _table_exists(conn, "bets", "mlb_prop_predictions"):
        return {"rows": 0, "status": "missing_prop_predictions"}
    df = _query_df(
        conn,
        """
        SELECT stat, bet_side, bookmaker_key,
               COUNT(*)::int AS rows,
               AVG(CASE WHEN over_hit IS TRUE THEN 1.0 WHEN over_hit IS FALSE THEN 0.0 END)::float AS win_rate,
               AVG(ev::float) AS avg_ev,
               AVG(CASE WHEN beat_clv_price IS TRUE THEN 1.0 WHEN beat_clv_price IS FALSE THEN 0.0 END)::float AS clv_beat_rate,
               AVG(clv_price::float) AS avg_clv_price
        FROM bets.mlb_prop_predictions
        WHERE game_date_et >= %(cutoff)s
          AND stat = ANY(%(stats)s::text[])
          AND is_active IS TRUE
          AND over_price IS NOT NULL
          AND under_price IS NOT NULL
          AND COALESCE(opportunity_context ->> 'pair_quality', 'same_book') = 'same_book'
          AND COALESCE(opportunity_context ->> 'market_prob_source', 'same_book_lock_pair')
              NOT IN ('raw_implied', 'raw_implied_one_sided', 'synthetic_fanduel_over_only', 'one_sided_fanduel_ladder')
          AND COALESCE(
              CASE
                  WHEN (opportunity_context ->> 'synthetic_pair_flag') ~ '^-?[0-9]+(\\.[0-9]+)?$'
                  THEN (opportunity_context ->> 'synthetic_pair_flag')::float
                  ELSE 0.0
              END,
              0.0
          ) = 0.0
        GROUP BY stat, bet_side, bookmaker_key
        ORDER BY COUNT(*) DESC
        LIMIT 40
        """,
        {"cutoff": cutoff, "phase": phase, "stats": list(_HITTER_STATS)},
    )
    rows = []
    for row in df.to_dict(orient="records"):
        rows.append({
            "stat": row.get("stat"),
            "side": row.get("bet_side"),
            "book": row.get("bookmaker_key"),
            "rows": int(row.get("rows") or 0),
            "win_rate": _as_float(row.get("win_rate")),
            "avg_ev": _as_float(row.get("avg_ev")),
            "clv_beat_rate": _as_float(row.get("clv_beat_rate")),
            "avg_clv_price": _as_float(row.get("avg_clv_price")),
        })
    return {"rows": int(df["rows"].sum()) if not df.empty else 0, "groups": rows}


def _render(payload: dict[str, Any]) -> str:
    graded = payload.get("graded_comparison") or {}
    hits_repair = payload.get("hits_live_bias_repair_v1") or {}
    lines = [
        "# MLB Hitter Live vs Legacy Forecast Diff",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Phase: {payload.get('phase')}",
        f"Lookback days: {payload.get('lookback_days')}",
        f"Rows: {payload.get('row_count', 0)}",
        f"Freeze status: {payload.get('freeze_status')}",
        "",
        "## Graded Comparison",
        "",
    ]
    if not graded or int(graded.get("graded_rows") or 0) <= 0:
        lines.append("No graded live-vs-legacy hitter rows yet. This will populate after the slate is finalized and the forecast ledger is graded.")
    else:
        lines.extend([
            "| Stat | Rows | Legacy MAE | Live MAE | Gain | Legacy Bias | Live Bias |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for label, key in (("Hits", "hits"), ("HR", "home_runs"), ("TB", "total_bases")):
            item = graded.get(key) or {}
            lines.append(
                f"| {label} | {item.get('rows', 0)} | {_fmt(item.get('legacy_mae'))} | "
                f"{_fmt(item.get('live_mae'))} | {_fmt(item.get('mae_gain'))} | "
                f"{_fmt(item.get('legacy_bias'))} | {_fmt(item.get('live_bias'))} |"
            )
        lines.extend(["", "### Line Brier", "", "| Line | Rows | Legacy Brier | Live Brier | Gain |", "|---|---:|---:|---:|---:|"])
        for label, item in (graded.get("tb_line_brier") or {}).items():
            lines.append(
                f"| {label} | {item.get('rows', 0)} | {_fmt(item.get('legacy_brier'))} | "
                f"{_fmt(item.get('live_brier'))} | {_fmt(item.get('brier_gain'))} |"
            )
        hr_brier = ((graded.get("home_runs") or {}).get("hr_0_5_brier") or {})
        lines.append(
            f"| HR 0.5 | {hr_brier.get('rows', 0)} | {_fmt(hr_brier.get('legacy_brier'))} | "
            f"{_fmt(hr_brier.get('live_brier'))} | {_fmt(hr_brier.get('brier_gain'))} |"
        )

    lines.extend([
        "",
        "## Hits Live Bias Repair V1",
        "",
        "Prospective repair trained on the actual shadow-vs-legacy ledger rows. It must beat both live shadow and legacy before the predictor can use it.",
        "",
        "| Accepted | Reason | Rows | Dates | Alpha | Live MAE | Legacy MAE | Selected MAE | Gain vs Live | Gain vs Legacy | Live Bias | Selected Bias |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    live = hits_repair.get("live") or {}
    legacy = hits_repair.get("legacy") or {}
    selected = hits_repair.get("selected") or {}
    lines.append(
        f"| {bool(hits_repair.get('accepted'))} | {hits_repair.get('reason') or hits_repair.get('status') or '-'} | "
        f"{hits_repair.get('rows', 0)} | {hits_repair.get('dates', 0)} | {_fmt(hits_repair.get('selected_alpha'), 2)} | "
        f"{_fmt(live.get('mae'))} | {_fmt(legacy.get('mae'))} | {_fmt(selected.get('mae'))} | "
        f"{_fmt(selected.get('mae_gain_vs_live'))} | {_fmt(selected.get('mae_gain_vs_legacy'))} | "
        f"{_fmt(live.get('bias'))} | {_fmt(selected.get('bias'))} |"
    )
    lines.extend([
        "",
        "| Alpha | MAE | Gain vs Live | Gain vs Legacy | Any Brier | Brier Gain vs Live | Brier Gain vs Legacy | Bias | Mean Shift |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in hits_repair.get("candidate_records") or []:
        lines.append(
            f"| {_fmt(row.get('alpha'), 2)} | {_fmt(row.get('mae'))} | "
            f"{_fmt(row.get('mae_gain_vs_live'))} | {_fmt(row.get('mae_gain_vs_legacy'))} | "
            f"{_fmt(row.get('any_brier'))} | {_fmt(row.get('any_brier_gain_vs_live'))} | "
            f"{_fmt(row.get('any_brier_gain_vs_legacy'))} | {_fmt(row.get('bias'))} | "
            f"{_fmt(row.get('mean_shift_vs_live'))} |"
        )

    lines.extend([
        "",
        "## Biggest Movers",
        "",
        "| Date | Player | Team | Opp | H Old -> Live | HR Old -> Live | TB Old -> Live | PA | Slot | Context |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in payload.get("biggest_movers", []):
        lines.append(
            f"| {row.get('game_date_et')} | {row.get('player_name')} | {row.get('team_abbr') or '-'} | "
            f"{row.get('opponent_abbr') or '-'} | {_fmt(row.get('legacy_hits'))} -> {_fmt(row.get('live_hits'))} | "
            f"{_fmt(row.get('legacy_hr'))} -> {_fmt(row.get('live_hr'))} | "
            f"{_fmt(row.get('legacy_tb'))} -> {_fmt(row.get('live_tb'))} | "
            f"{_fmt(row.get('projected_pa'))} | {_fmt(row.get('lineup_slot'), 1)} | "
            f"{', '.join(row.get('context_tags') or [])} |"
        )

    clv = payload.get("true_pair_clv_roi") or {}
    lines.extend([
        "",
        "## True-Pair Offer Check",
        "",
        f"True-paired active hitter offer rows: {clv.get('rows', 0)}",
        "",
        "| Stat | Side | Book | Rows | Win Rate | CLV Beat | Avg CLV | Avg EV |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in clv.get("groups") or []:
        lines.append(
            f"| {row.get('stat')} | {row.get('side') or '-'} | {row.get('book') or '-'} | "
            f"{row.get('rows', 0)} | {_fmt(row.get('win_rate'))} | {_fmt(row.get('clv_beat_rate'))} | "
            f"{_fmt(row.get('avg_clv_price'))} | {_fmt(row.get('avg_ev'))} |"
        )
    return "\n".join(lines) + "\n"


def build(cfg: DiffConfig = DiffConfig()) -> dict[str, Any]:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)
    phase = cfg.phase or canonical_forecast_phase()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_daily_forecast_ledger_schema(conn)
        if cfg.grade:
            grade_daily_forecasts(conn, date_to=datetime.now(timezone.utc).date())
        if not _table_exists(conn, "bets", "mlb_daily_forecast_ledger"):
            diff = pd.DataFrame()
            clv = {"rows": 0, "status": "missing_daily_forecast_ledger"}
        else:
            raw = _latest_rows(conn, cutoff, phase)
            diff = _build_player_diff(raw)
            clv = _clv_roi_true_pair(conn, cutoff, phase)

    movers = diff.head(cfg.top_n).to_dict(orient="records") if not diff.empty else []
    hits_live_bias_repair = _hits_live_bias_repair(diff)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "phase": phase,
        "lookback_days": cfg.lookback_days,
        "source": "bets.mlb_daily_forecast_ledger",
        "row_count": int(len(diff)),
        "freeze_status": "live_forecast_artifact_pinned",
        "biggest_movers": movers,
        "graded_comparison": _graded_comparison(diff),
        "hits_live_bias_repair_v1": hits_live_bias_repair,
        "true_pair_clv_roi": clv,
    }
    payload = _json_safe(payload)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(_MODEL_DIR / cfg.artifact_file, payload)
    atomic_write_json(_MODEL_DIR / "hitter_hits_live_bias_repair_v1.json", payload.get("hits_live_bias_repair_v1") or {})
    report_path = _REPORT_DIR / cfg.report_file
    atomic_write_text(report_path, _render(payload))
    payload["report_path"] = str(report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare live hitter scoring with legacy forecast values")
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument("--phase", default=None)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--grade", action="store_true")
    args = parser.parse_args()
    payload = build(DiffConfig(
        lookback_days=args.lookback_days,
        phase=args.phase,
        top_n=args.top_n,
        grade=args.grade,
    ))
    print(json.dumps({
        "rows": payload.get("row_count", 0),
        "phase": payload.get("phase"),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
