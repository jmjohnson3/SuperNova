"""Predict NFL player props from player-game stat forecasts."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from scipy.stats import norm, poisson

from nfl_pipeline.integrity import nfl_season
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.features import ROLE_SIGNAL_STATS, ROLLING_STATS, SPARSE_USAGE_STATS, TARGET_STATS, VOLATILITY_STATS, _add_context_risk_features
from nfl_pipeline.integrity import FEATURE_CONTRACT, release_artifact
from nfl_pipeline.markets import (
    SPEC_BY_STAT,
    STAT_SPECS,
    normalize_name,
    normalize_team,
    role_for_position,
    stats_for_position,
)
from nfl_pipeline.schema import ensure_schema
from nfl_pipeline.offer_selection import CONTRACT as EXECUTION_CONTRACT, eligible_player_offers
from nfl_pipeline.modeling.train_player_stat_models import (
    _baseline_from_metric_column,
    _make_features,
    _predict_binary_probability,
    _predict_model_payload,
)
from nfl_pipeline.modeling.train_prop_exact_line_models import (
    _feature_frame as _exact_line_feature_frame,
    _line_bucket as _exact_line_line_bucket,
    _price_bucket as _exact_line_price_bucket,
)

log = logging.getLogger("nfl_pipeline.modeling.predict_player_props")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable",
    category=UserWarning,
)

_ET = ZoneInfo("America/New_York")
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"


@dataclass(frozen=True)
class PredictConfig:
    pg_dsn: str = PG_DSN
    model_dir: Path = _MODEL_DIR
    model_file: str = "nfl_player_stat_models.joblib"
    distribution_file: str = "nfl_player_stat_distributions.json"
    opportunity_model_file: str = "nfl_player_opportunity_models.joblib"
    exact_line_model_file: str = "nfl_prop_exact_line_models.joblib"
    et_date: date | None = None
    top_n_per_section: int = 10
    min_ev: float = 0.02
    max_micro_props_per_day: int = 5
    max_projection_recency_days: int = 120
    save_predictions: bool = True


def _clean_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _american_to_prob(price: Any) -> float | None:
    price_f = _clean_float(price)
    if price_f is None or price_f == 0:
        return None
    if price_f > 0:
        return 100.0 / (price_f + 100.0)
    return abs(price_f) / (abs(price_f) + 100.0)


def _ev_per_unit(prob: Any, price: Any, push_probability: Any = 0.0) -> float | None:
    """Probability is conditional on a non-push; pushes return the stake."""
    p = _clean_float(prob)
    pr = _clean_float(price)
    push = _clean_float(push_probability)
    if p is None or not 0 <= p <= 1 or pr is None or abs(pr) < 100 or push is None or not 0 <= push <= 1:
        return None
    payout = pr / 100.0 if pr > 0 else 100.0 / abs(pr)
    return (1.0 - push) * (p * payout - (1.0 - p))


def _break_even_american_price(prob: Any) -> int | None:
    """Smallest executable integer price with strictly positive EV, not fair odds."""
    p = _clean_float(prob)
    if p is None or not 0 < p < 1:
        return None
    payout = (1.0 - p) / p
    price = math.ceil(100.0 * payout) if payout >= 1.0 else math.ceil(-100.0 / payout)
    if -100 <= price < 100:
        price = 100
    while not _price_has_positive_ev(p, price):
        price = 100 if price == -101 else price + 1
    return int(price)


def _format_american(price: Any) -> str:
    clean = _clean_float(price)
    if clean is None:
        return "-"
    value = int(round(clean))
    return f"+{value}" if value > 0 else str(value)


def _price_has_positive_ev(prob: Any, price: Any, push_probability: Any = 0.0) -> bool:
    ev = _ev_per_unit(prob, price, push_probability)
    return ev is not None and ev > 1e-12


def _game_has_started(row: dict[str, Any], now_utc: datetime | None = None) -> bool:
    start = row.get("start_ts_utc")
    if start is None or pd.isna(start):
        return False
    ts = pd.to_datetime(start, errors="coerce", utc=True)
    if pd.isna(ts):
        return False
    current = now_utc or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return bool(ts.to_pydatetime() <= current)


def _no_vig_side_probability(over_price: Any, under_price: Any, side: str) -> float | None:
    over_raw = _american_to_prob(over_price)
    under_raw = _american_to_prob(under_price)
    if over_raw is None or under_raw is None:
        return None
    total = over_raw + under_raw
    if total <= 0:
        return None
    p_over = over_raw / total
    return float(p_over if side == "over" else 1.0 - p_over)


YARDAGE_PROP_STATS = {"passing_yards", "rushing_yards", "receiving_yards"}


def _clip_probability(value: Any, low: float = 0.001, high: float = 0.999, default: float = 0.5) -> float:
    clean = _clean_float(value)
    if clean is None:
        clean = default
    return float(np.clip(clean, low, high))


def _row_feature_score(row: dict[str, Any], *names: str, default: float = 0.0) -> float:
    values: list[float] = []
    for name in names:
        clean = _clean_float(row.get(name))
        if clean is not None:
            values.append(float(np.clip(clean, 0.0, 1.0)))
    if not values:
        return float(default)
    return float(max(values))


def _row_max_float(row: dict[str, Any], *names: str, default: float = 0.0) -> float:
    values: list[float] = []
    for name in names:
        clean = _clean_float(row.get(name))
        if clean is not None:
            values.append(clean)
    if not values:
        return float(default)
    return float(max(values))


def _projection_gap_ratio(projection: float, baseline: float, line: float) -> float:
    scale = max(1.0, abs(float(line)), abs(float(projection)), abs(float(baseline)))
    return float(abs(float(projection) - float(line)) / scale)


def _position_adjusted_projection(
    row: dict[str, Any],
    stat: str,
    projection: float,
    baseline: float,
) -> tuple[float, str | None]:
    """Correct broad stat projections when the position-specific role is different."""
    if stat != "rushing_yards" or str(row.get("position") or "").upper() != "QB":
        return float(projection), None

    proj = max(0.0, float(projection))
    base = max(0.0, float(baseline))
    rush_avg = _row_max_float(row, "rushing_yards_avg_3", "rushing_yards_avg_5", "rushing_yards_avg_10", default=base)
    carry_avg = _row_max_float(row, "carries_avg_3", "carries_avg_5", "carries_avg_10", default=0.0)
    mobility_score = max(
        _sigmoid_score(rush_avg, 14.0, 6.0),
        _sigmoid_score(carry_avg, 2.8, 1.1),
        0.50 * _row_feature_score(row, "qb_volume_spike_signal", "pass_spike_path_score"),
    )
    low_mobility_role = rush_avg < 7.0 and carry_avg < 1.8 and mobility_score < 0.36
    if low_mobility_role:
        cap = max(0.45, min(1.35, base * 1.35 + 0.35, rush_avg + 0.45))
        adjusted = min(proj, cap)
        if adjusted + 1e-6 < proj:
            return adjusted, "qb_rushing_low_mobility_projection_sanitized"
        return adjusted, None

    moderate_cap = max(4.5, base * 1.35 + 3.0, rush_avg * 1.25 + 2.0)
    mobile_cap = max(moderate_cap, 8.0 + 32.0 * mobility_score)
    adjusted = min(proj, mobile_cap)
    if adjusted + 1e-6 < proj:
        return adjusted, "qb_rushing_position_cap_applied"
    return adjusted, None


def _offer_probability_calibration(
    *,
    stat: str,
    raw_p_over: float,
    projection: float,
    baseline: float,
    line: float,
    offer: dict[str, Any],
    row: dict[str, Any],
    distribution_summary: dict[str, Any],
    trace: dict[str, Any] | None = None,
) -> tuple[float, list[str]]:
    """Shrink live offer probabilities before EV/ranking.

    The stat projection can beat baseline while the betting probability is still
    too confident for a specific line/book. Micro props especially should be
    calibrated around the true paired market unless the forecast context is
    unusually strong.
    """
    raw = _clip_probability(raw_p_over)
    market_over = _no_vig_side_probability(offer.get("over_price"), offer.get("under_price"), "over")
    anchor = _clip_probability(market_over) if market_over is not None else 0.5
    confidence = _clip_probability(distribution_summary.get("projection_confidence"), 0.0, 1.0, default=0.0)
    reasons: list[str] = []
    if stat in YARDAGE_PROP_STATS:
        volatility = _row_feature_score(
            row,
            "yardage_projection_volatility_v3_score",
            "workload_downside_v2_score",
            "usage_volatility_score",
            "high_usage_fragility_score",
            "limited_workload_risk_score",
            "yardage_projection_volatility_v4_score",
        )
        same_week_confidence = _row_feature_score(
            row,
            "live_usage_context_quality_v4_score",
            "same_week_usage_confidence_v3_score",
            "same_week_context_confidence_score",
        )
        usage_quality = _row_feature_score(row, "live_usage_context_quality_v4_score", default=same_week_confidence)
        if stat == "receiving_yards":
            spike = _row_feature_score(
                row,
                "receiver_spike_under_correction_v4_score",
                "receiver_live_spike_v3_score",
                "receiver_target_spike_v2_score",
                "receiver_air_yards_spike_v2_score",
                "receiver_ypt_efficiency_spike_score",
                "receiver_contextual_spike_score",
                "_pred_receiver_target_spike_v2_probability",
            )
            route_quality = _row_feature_score(row, "has_true_route_history", "receiving_usage_history_quality_score")
            trust = 0.24 + 0.32 * confidence + 0.13 * same_week_confidence + 0.10 * route_quality
            trust -= 0.18 * volatility + 0.06 * max(0.0, spike - 0.68)
        elif stat == "rushing_yards":
            spike = _row_feature_score(row, "rb_carry_under_correction_v4_score", "rb_live_carry_v3_score", "rb_carry_spike_v2_score")
            starter = _row_feature_score(row, "projected_starter_score", "starter_confidence")
            rb_quality = _row_feature_score(row, "rb_usage_history_quality_score", default=usage_quality)
            trust = 0.26 + 0.32 * confidence + 0.11 * same_week_confidence + 0.06 * starter + 0.05 * rb_quality
            trust -= 0.18 * volatility + 0.05 * max(0.0, spike - 0.70)
        else:
            spike = _row_feature_score(row, "qb_volume_spike_signal", "pass_spike_path_score")
            starter = _row_feature_score(row, "projected_starter_score", "starter_confidence")
            trust = 0.30 + 0.36 * confidence + 0.08 * same_week_confidence + 0.06 * starter
            trust -= 0.16 * volatility + 0.04 * max(0.0, spike - 0.72)
        if usage_quality < 0.45:
            trust -= 0.07
            reasons.append("low_usage_data_quality_shrunk")
        gap_ratio = _projection_gap_ratio(projection, baseline, line)
        if gap_ratio >= 0.30:
            trust -= 0.08
            reasons.append("large_projection_gap_shrunk")
        if market_over is None:
            trust *= 0.62
            reasons.append("no_market_anchor_probability_shrunk")
        trust = float(np.clip(trust, 0.18, 0.78))
        calibrated = anchor + trust * (raw - anchor)
        position = str(row.get("position") or "").upper()
        book = str(offer.get("bookmaker_key") or offer.get("book") or "").lower()
        tail_cap = 0.72
        if stat == "receiving_yards":
            tail_cap = 0.69 if line >= 45.5 else 0.72
            if position == "RB":
                tail_cap = min(tail_cap, 0.70)
        elif stat == "rushing_yards":
            tail_cap = 0.70 if line >= 45.5 else 0.72
        elif stat == "passing_yards":
            trust *= 0.72
            tail_cap = 0.60 if line >= 190.5 else 0.63
            reasons.append("passing_yards_probability_baseline_guard_shrunk")
        if book == "fanduel" and market_over is None:
            tail_cap = min(tail_cap, 0.66)
        if stat == "rushing_yards" and position == "QB":
            rush_avg = _row_max_float(row, "rushing_yards_avg_3", "rushing_yards_avg_5", "rushing_yards_avg_10", default=baseline)
            carry_avg = _row_max_float(row, "carries_avg_3", "carries_avg_5", "carries_avg_10", default=0.0)
            mobility_score = max(
                _sigmoid_score(rush_avg, 14.0, 6.0),
                _sigmoid_score(carry_avg, 2.8, 1.1),
                0.50 * _row_feature_score(row, "qb_volume_spike_signal", "pass_spike_path_score"),
            )
            if mobility_score < 0.36:
                trust = min(trust, 0.16)
                tail_cap = min(tail_cap, 0.58)
                reasons.append("qb_rushing_market_anchored_without_mobility_history")
        if gap_ratio >= 0.42:
            tail_cap = min(tail_cap, 0.64 if stat in {"receiving_yards", "rushing_yards"} else 0.62)
            reasons.append("large_gap_tail_cap_applied")
        if confidence >= 0.30 and volatility <= 0.28 and same_week_confidence >= 0.70 and abs(raw - anchor) <= 0.16:
            tail_cap = max(tail_cap, 0.78)
        calibrated = anchor + trust * (raw - anchor)
        before_cap = calibrated
        if trace is not None:
            trace.update(context_blend_over=float(before_cap), market_anchor_over=float(anchor), context_trust=float(trust))
        calibrated = float(np.clip(calibrated, 1.0 - tail_cap, tail_cap))
        if abs(calibrated - raw) >= 0.012:
            reasons.append("probability_calibrated_by_market_usage_context")
        if abs(calibrated - before_cap) >= 0.001:
            reasons.append("micro_probability_cap_applied")
        if volatility >= 0.52:
            reasons.append("yardage_usage_volatility_shrunk")
        return _clip_probability(calibrated), reasons

    if stat.endswith("_tds"):
        cap = 0.66 if stat == "receiving_tds" else 0.74
        trust = 0.55 if market_over is not None else 0.38
        calibrated = anchor + trust * (raw - anchor)
        if trace is not None:
            trace.update(context_blend_over=float(calibrated), market_anchor_over=float(anchor), context_trust=float(trust))
        calibrated = float(np.clip(calibrated, 1.0 - cap, cap))
        if abs(calibrated - raw) >= 0.012:
            reasons.append("td_probability_calibrated")
        return _clip_probability(calibrated), reasons

    if trace is not None:
        trace.update(context_blend_over=float(raw), market_anchor_over=float(anchor), context_trust=1.0)
    return raw, reasons


def _load_clv_guard(conn) -> dict[tuple[str, str, str], dict[str, float]]:
    guard: dict[tuple[str, str, str], dict[str, float]] = {}
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    lower(COALESCE(book, '')) AS book,
                    stat,
                    side,
                    COUNT(*)::int AS valid_clv_rows,
                    AVG(CASE WHEN clv_prob_delta > 0 THEN 1.0 ELSE 0.0 END)::float AS clv_beat_rate,
                    AVG(clv_prob_delta)::float AS avg_clv_prob_delta
                FROM bets.nfl_prediction_clv
                WHERE source_kind = 'prop'
                  AND stat IS NOT NULL
                  AND side IS NOT NULL
                  AND clv_prob_delta IS NOT NULL
                  AND (valid_close_snapshot_captured IS TRUE OR clv_status = 'valid_close')
                GROUP BY lower(COALESCE(book, '')), stat, side
                """
            )
            for rec in cur.fetchall():
                key = (
                    str(rec.get("book") or ""),
                    str(rec.get("stat") or ""),
                    str(rec.get("side") or ""),
                )
                guard[key] = {
                    "valid_clv_rows": float(rec.get("valid_clv_rows") or 0),
                    "clv_beat_rate": float(rec.get("clv_beat_rate") or 0.0),
                    "avg_clv_prob_delta": float(rec.get("avg_clv_prob_delta") or 0.0),
                }
    except Exception as exc:
        log.warning("Could not load NFL prop CLV guard: %s", exc)
    return guard


def _probability_bucket(probability: Any) -> str:
    p = _clip_probability(probability)
    if p < 0.52:
        return "lt_52"
    if p < 0.56:
        return "52_56"
    if p < 0.60:
        return "56_60"
    if p < 0.65:
        return "60_65"
    if p < 0.70:
        return "65_70"
    return "70_plus"


def _line_range_bucket(stat: str, line: Any) -> str:
    line_f = _clean_float(line)
    if line_f is None:
        return "line_unknown"
    if stat == "passing_yards":
        if line_f < 180.5:
            return "lt_180"
        if line_f < 220.5:
            return "180_220"
        if line_f < 260.5:
            return "220_260"
        return "260_plus"
    if stat in {"rushing_yards", "receiving_yards"}:
        if line_f < 20.5:
            return "lt_20"
        if line_f < 40.5:
            return "20_40"
        if line_f < 60.5:
            return "40_60"
        if line_f < 80.5:
            return "60_80"
        return "80_plus"
    if stat.endswith("_tds"):
        if line_f <= 0.5:
            return "td_0_5"
        return "td_alt"
    return f"line_{round(float(line_f), 1):g}"


def _projection_gap_bucket(projection: Any, baseline: Any, line: Any) -> str:
    projection_f = _clean_float(projection)
    baseline_f = _clean_float(baseline)
    line_f = _clean_float(line)
    if projection_f is None or line_f is None:
        return "gap_unknown"
    ratio = _projection_gap_ratio(projection_f, baseline_f if baseline_f is not None else projection_f, line_f)
    if ratio < 0.08:
        return "gap_lt_08"
    if ratio < 0.16:
        return "gap_08_16"
    if ratio < 0.28:
        return "gap_16_28"
    if ratio < 0.42:
        return "gap_28_42"
    return "gap_42_plus"


def _calibration_context_key(
    *,
    stat: str,
    position: str,
    line: Any,
    projection: Any,
    baseline: Any,
) -> str:
    pos = str(position or "UNK").upper() or "UNK"
    return "|".join([
        stat,
        pos,
        _line_range_bucket(stat, line),
        _projection_gap_bucket(projection, baseline, line),
    ])


def _load_recent_probability_calibration(
    conn,
    et_day: date,
    *,
    lookback_days: int = 35,
) -> dict[tuple[str, str, str, str, str], dict[str, float]]:
    """Load leakage-safe live calibration from already graded prop rows.

    The model can look much too certain on small early-season NFL samples. This
    keeps micro/paper ranking honest by shrinking only against rows that were
    locked before the slate being predicted.
    """
    groups: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    lower(COALESCE(p.book, '')) AS book,
                    p.stat,
                    p.side,
                    p.line::float AS line,
                    p.projection::float AS projection,
                    p.baseline_projection::float AS baseline_projection,
                    UPPER(COALESCE(p.position, '')) AS position,
                    p.probability::float AS probability,
                    p.market_no_vig_probability::float AS market_probability,
                    r.result
                FROM bets.nfl_player_prop_predictions p
                JOIN bets.nfl_player_prop_prediction_results r
                  ON r.prediction_id = p.id
                WHERE p.game_date_et < %(game_date)s
                  AND p.game_date_et >= %(game_date)s - (%(lookback_days)s * interval '1 day')
                  AND p.side IN ('over', 'under')
                  AND p.probability IS NOT NULL
                  AND r.result IN ('win', 'loss')
                  AND p.integrity_version='nfl-asof-v2'
                  AND EXISTS (SELECT 1 FROM raw.nfl_games g WHERE g.game_id=p.game_id AND g.status='final' AND p.created_at_utc<g.start_ts_utc)
                  AND NOT EXISTS (
                    SELECT 1 FROM bets.nfl_player_prop_predictions earlier
                    WHERE earlier.game_id=p.game_id AND earlier.player_id=p.player_id
                      AND earlier.stat=p.stat AND earlier.side=p.side AND earlier.book=p.book AND earlier.line=p.line
                      AND earlier.integrity_version=p.integrity_version AND earlier.id<p.id)
                """,
                {"game_date": et_day, "lookback_days": int(lookback_days)},
            )
            rows = cur.fetchall()
    except Exception as exc:
        log.warning("Could not load NFL prop probability calibration: %s", exc)
        return {}

    def add(key: tuple[str, str, str, str, str], prob: float, market: float | None, win: float) -> None:
        rec = groups.setdefault(
            key,
            {"rows": 0.0, "wins": 0.0, "probability_sum": 0.0, "market_sum": 0.0, "market_rows": 0.0},
        )
        rec["rows"] += 1.0
        rec["wins"] += win
        rec["probability_sum"] += prob
        if market is not None:
            rec["market_sum"] += market
            rec["market_rows"] += 1.0

    for row in rows:
        prob = _clean_float(row.get("probability"))
        if prob is None:
            continue
        prob = _clip_probability(prob)
        stat = str(row.get("stat") or "")
        side = str(row.get("side") or "")
        book = str(row.get("book") or "")
        line = _clean_float(row.get("line"))
        projection = _clean_float(row.get("projection"))
        baseline = _clean_float(row.get("baseline_projection"))
        position = str(row.get("position") or "UNK").upper() or "UNK"
        bucket = _probability_bucket(prob)
        line_bucket = _line_range_bucket(stat, line)
        gap_bucket = _projection_gap_bucket(projection, baseline, line)
        context_key = _calibration_context_key(
            stat=stat,
            position=position,
            line=line,
            projection=projection,
            baseline=baseline,
        )
        market = _clean_float(row.get("market_probability"))
        win = 1.0 if row.get("result") == "win" else 0.0
        add(("exact_context", book, context_key, side, bucket), prob, market, win)
        add(("stat_position_line", "", f"{stat}|{position}|{line_bucket}", side, bucket), prob, market, win)
        add(("stat_line", "", f"{stat}|{line_bucket}", side, bucket), prob, market, win)
        add(("stat_gap", "", f"{stat}|{gap_bucket}", side, bucket), prob, market, win)
        add(("exact", book, stat, side, bucket), prob, market, win)
        add(("stat_side", "", stat, side, bucket), prob, market, win)
        add(("stat", "", stat, "", bucket), prob, market, win)
        add(("global", "", "", "", bucket), prob, market, win)

    finalized: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
    for key, rec in groups.items():
        rows_n = float(rec["rows"])
        avg_prob = float(rec["probability_sum"] / rows_n) if rows_n else 0.5
        raw_win_rate = float(rec["wins"] / rows_n) if rows_n else 0.5
        avg_market = (
            float(rec["market_sum"] / rec["market_rows"])
            if rec.get("market_rows")
            else None
        )
        prior = avg_market if avg_market is not None else 0.5
        empirical = float((rec["wins"] + 10.0 * prior) / (rows_n + 10.0))
        finalized[key] = {
            "rows": rows_n,
            "avg_probability": avg_prob,
            "win_rate": raw_win_rate,
            "empirical_win_rate": empirical,
            "avg_market_probability": avg_market if avg_market is not None else float("nan"),
        }
    return finalized


def _recent_calibration_record(
    calibration: dict[tuple[str, str, str, str, str], dict[str, float]] | None,
    *,
    book: str,
    stat: str,
    side: str,
    probability: float,
    line: Any = None,
    projection: Any = None,
    baseline: Any = None,
    position: str | None = None,
) -> tuple[str, dict[str, float]] | None:
    if not calibration:
        return None
    bucket = _probability_bucket(probability)
    context_key = _calibration_context_key(
        stat=stat,
        position=str(position or "UNK"),
        line=line,
        projection=projection,
        baseline=baseline,
    )
    line_bucket = _line_range_bucket(stat, line)
    gap_bucket = _projection_gap_bucket(projection, baseline, line)
    pos = str(position or "UNK").upper() or "UNK"
    candidates = (
        ("exact_context", book, context_key, side, bucket, 4),
        ("stat_position_line", "", f"{stat}|{pos}|{line_bucket}", side, bucket, 8),
        ("stat_line", "", f"{stat}|{line_bucket}", side, bucket, 10),
        ("stat_gap", "", f"{stat}|{gap_bucket}", side, bucket, 10),
        ("exact", book, stat, side, bucket, 6),
        ("stat_side", "", stat, side, bucket, 12),
        ("stat", "", stat, "", bucket, 18),
        ("global", "", "", "", bucket, 30),
    )
    for level, key_book, key_stat, key_side, key_bucket, min_rows in candidates:
        rec = calibration.get((level, key_book, key_stat, key_side, key_bucket))
        if rec and float(rec.get("rows") or 0.0) >= float(min_rows):
            return level, rec
    return None


def _apply_recent_probability_calibration(
    probability: float,
    *,
    stat: str,
    side: str,
    book: str,
    market_probability: float | None,
    calibration: dict[tuple[str, str, str, str, str], dict[str, float]] | None,
    line: Any = None,
    projection: Any = None,
    baseline: Any = None,
    position: str | None = None,
) -> tuple[float, list[str]]:
    """Shrink overconfident live prop probabilities using settled prior slates."""
    prob = _clip_probability(probability)
    match = _recent_calibration_record(
        calibration,
        book=book,
        stat=stat,
        side=side,
        probability=prob,
        line=line,
        projection=projection,
        baseline=baseline,
        position=position,
    )
    if not match:
        return prob, []
    level, rec = match
    avg_prob = float(rec.get("avg_probability") or prob)
    empirical = float(rec.get("empirical_win_rate") or avg_prob)
    rows = float(rec.get("rows") or 0.0)
    overconfidence = max(0.0, avg_prob - empirical)
    if overconfidence < 0.025:
        return prob, []
    anchor = _clip_probability(market_probability, default=empirical) if market_probability is not None else empirical
    target = min(prob, 0.70 * empirical + 0.30 * anchor)
    shrink = min(0.58, rows / (rows + 24.0)) * min(1.0, overconfidence / 0.14)
    adjusted = prob + shrink * (target - prob)
    if stat in YARDAGE_PROP_STATS:
        line_bucket = _line_range_bucket(stat, line)
        gap_bucket = _projection_gap_bucket(projection, baseline, line)
        cap = max(0.52, min(0.68, target + 0.035))
        if line_bucket in {"40_60", "60_80", "80_plus", "220_260", "260_plus"}:
            cap = min(cap, 0.65)
        if gap_bucket in {"gap_28_42", "gap_42_plus"}:
            cap = min(cap, 0.63)
        adjusted = min(adjusted, cap)
    if prob - adjusted < 0.004:
        return prob, []
    return _clip_probability(adjusted), [
        (
            f"recent_probability_calibration_{level}_shrunk"
            f"_n{int(rows)}_wr{float(rec.get('win_rate') or 0.0):.0%}"
        )
    ]


def _micro_projection_status(
    *,
    stat: str,
    metrics: dict[str, Any],
    offer: dict[str, Any],
    probability: float,
    ev: float | None,
    market_probability: float | None,
    link: str | None,
    min_ev: float,
    projection_confidence: float | None = None,
    clv_guard: dict[str, float] | None = None,
    exact_line_clv_probability: float | None = None,
    game_started: bool = False,
) -> tuple[str, str]:
    if game_started:
        return "paper", "nfl_v1_projection_only_no_bankroll_gates;game_already_started_bettable_closed"
    if stat == "passing_yards":
        gain = _clean_float(metrics.get("mae_gain_vs_baseline")) or 0.0
        if gain < 2.0:
            return "paper", "nfl_v1_projection_only_no_bankroll_gates;passing_yards_projection_not_proven_for_micro"
    if stat == "receiving_tds":
        return "paper", "nfl_v1_projection_only_no_bankroll_gates;receiving_td_projection_not_approved_for_micro"
    if str(metrics.get('selection_target') or '').startswith('P(any TD)') and _clean_float(offer.get('line')) != 0.5:
        return "paper", "nfl_v1_projection_only_no_bankroll_gates;td_alternate_line_distribution_unvalidated"
    if not bool(metrics.get("accepted_live")) or not _projection_layer_accepted(metrics):
        return "paper", "nfl_v1_projection_only_no_bankroll_gates;projection_model_not_accepted"
    if offer.get("over_price") is None or offer.get("under_price") is None or market_probability is None:
        return "paper", "nfl_v1_projection_only_no_bankroll_gates;not_true_paired_price"
    if not link:
        return "paper", "nfl_v1_projection_only_no_bankroll_gates;missing_bet_link"
    if ev is None or ev < min_ev:
        return "paper", "nfl_v1_projection_only_no_bankroll_gates;micro_ev_gate_failed"
    required_market_edge = 0.025 if stat in YARDAGE_PROP_STATS else 0.015
    if probability - market_probability < required_market_edge:
        return "paper", "nfl_v1_projection_only_no_bankroll_gates;micro_market_edge_gate_failed"
    if stat in YARDAGE_PROP_STATS and (projection_confidence is None or projection_confidence < 0.08):
        return "paper", "nfl_v1_projection_only_no_bankroll_gates;micro_projection_confidence_gate_failed"
    if stat in YARDAGE_PROP_STATS and probability > 0.805:
        return "paper", "nfl_v1_projection_only_no_bankroll_gates;micro_probability_too_extreme_after_calibration"
    if exact_line_clv_probability is not None:
        if exact_line_clv_probability < 0.49:
            return "paper", "nfl_v1_projection_only_no_bankroll_gates;exact_line_clv_model_guard_failed"
        clv_reason = "exact_line_clv_model_clean"
    else:
        clv_reason = ""
    if clv_guard:
        rows = int(clv_guard.get("valid_clv_rows") or 0)
        beat_rate = _clean_float(clv_guard.get("clv_beat_rate"))
        avg_delta = _clean_float(clv_guard.get("avg_clv_prob_delta"))
        if rows >= 8 and beat_rate is not None and beat_rate < 0.46:
            return "paper", "nfl_v1_projection_only_no_bankroll_gates;clv_history_guard_failed"
        if rows >= 8 and avg_delta is not None and avg_delta < -0.008:
            return "paper", "nfl_v1_projection_only_no_bankroll_gates;negative_clv_history_guard_failed"
        clv_reason = ";".join(part for part in (clv_reason, "clv_history_clean") if part)
    elif not clv_reason:
        clv_reason = "clv_history_pending"
    return "micro_projection", f"nfl_micro_projection_true_pair_positive_ev_projection_accepted;{clv_reason}"


def _usage_context_quality(row: dict[str, Any]) -> float:
    return _row_feature_score(
        row,
        "live_usage_context_quality_v4_score",
        "receiving_usage_history_quality_score",
        "rb_usage_history_quality_score",
        "td_usage_history_quality_score",
        default=0.0,
    )


def _limited_usage_risk(row: dict[str, Any]) -> float:
    return _row_feature_score(
        row,
        "limited_workload_risk_score",
        "workload_downside_v2_score",
        "rest_risk_score",
        "weird_usage_risk_score",
        "depth_movement_risk_score",
        "injury_downgrade_score",
        default=0.0,
    )


def _spike_usage_probability(stat: str, row: dict[str, Any]) -> float | None:
    if stat in YARDAGE_PROP_STATS:
        return _yardage_high_score(stat, row)
    if stat.endswith("_tds"):
        return _row_feature_score(
            row,
            "td_goal_line_env_score",
            "td_role_team_total_adjusted",
            "starter_adjusted_red_zone_role_avg_5",
            default=0.0,
        )
    return None


def _usage_context_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    quality = _clean_float(row.get("usage_context_quality"))
    limited = _clean_float(row.get("limited_usage_risk"))
    spike = _clean_float(row.get("spike_usage_probability"))
    depth_rank = _clean_float(row.get("depth_pos_rank"))
    injury_raw = row.get("injury_report_status")
    practice_raw = row.get("injury_practice_status")
    injury = "" if pd.isna(injury_raw) else str(injury_raw or "").strip()
    practice = "" if pd.isna(practice_raw) else str(practice_raw or "").strip()
    if quality is not None:
        tags.append(f"ctx={quality:.0%}")
    if spike is not None:
        tags.append(f"spike={spike:.0%}")
    if limited is not None and limited >= 0.20:
        tags.append(f"limited={limited:.0%}")
    if depth_rank is not None and depth_rank < 90:
        tags.append(f"depth={depth_rank:.0f}")
    if injury and injury.lower() not in {"", "none"}:
        tags.append(f"inj={injury}")
    if practice and practice.lower() not in {"", "none"}:
        tags.append(f"prac={practice}")
    return tags


def _build_fd_parlay_url(links: list[str | None]) -> str | None:
    legs: list[tuple[str, str]] = []
    for link in links:
        if not link or "fanduel.com" not in link:
            continue
        try:
            qs = parse_qs(urlparse(link).query)
            market = qs.get("marketId", [None])[0]
            selection = qs.get("selectionId", [None])[0]
            if market and selection:
                legs.append((market, selection))
        except Exception:
            continue
    if len(legs) < 2:
        return None
    params = "&".join(
        f"marketId[{idx}]={market}&selectionId[{idx}]={selection}"
        for idx, (market, selection) in enumerate(dict.fromkeys(legs).keys())
    )
    return f"https://sportsbook.fanduel.com/addToBetslip?{params}"


def _load_model(cfg: PredictConfig) -> dict[str, Any]:
    released = release_artifact("players")
    if released is not None:
        return released
    path = cfg.model_dir / cfg.model_file
    if not path.exists():
        return {"status": "missing", "models": {}, "metrics": {}, "feature_columns": {}, "fill_values": {}}
    try:
        artifact = joblib.load(path)
    except Exception as exc:
        log.warning("Could not load NFL stat model artifact at %s: %s", path, exc)
        return {"status": "load_failed", "models": {}, "metrics": {}, "feature_columns": {}, "fill_values": {}}
    if not isinstance(artifact, dict):
        return {"status": "bad_artifact", "models": {}}
    dist_path = cfg.model_dir / cfg.distribution_file
    if dist_path.exists():
        try:
            dist_payload = json.loads(dist_path.read_text(encoding="utf-8"))
            artifact["distributions"] = dist_payload.get("distributions") or {}
            artifact["distribution_status"] = dist_payload.get("status")
        except Exception as exc:
            log.warning("Could not load NFL stat distribution artifact at %s: %s", dist_path, exc)
            artifact["distribution_status"] = "load_failed"
    opp_path = cfg.model_dir / cfg.opportunity_model_file
    if opp_path.exists():
        try:
            opp_artifact = joblib.load(opp_path)
            if isinstance(opp_artifact, dict):
                artifact["opportunity_artifact"] = opp_artifact
                artifact["opportunity_status"] = opp_artifact.get("status")
        except Exception as exc:
            log.warning("Could not load NFL opportunity model artifact at %s: %s", opp_path, exc)
            artifact["opportunity_status"] = "load_failed"
    exact_path = cfg.model_dir / cfg.exact_line_model_file
    if exact_path.exists():
        try:
            exact_artifact = joblib.load(exact_path)
            if isinstance(exact_artifact, dict):
                artifact["exact_line_artifact"] = exact_artifact
                artifact["exact_line_status"] = exact_artifact.get("status")
        except Exception as exc:
            log.warning("Could not load NFL exact-line model artifact at %s: %s", exact_path, exc)
            artifact["exact_line_status"] = "load_failed"
    return artifact


def _prediction_context_cutoff(conn, et_day: date) -> datetime:
    """Return the latest timestamp allowed for lock-time inputs.

    Live today/future predictions should use the freshest context available for
    games that have not started yet; per-game eligibility keeps started games
    out of bettable sections. Historical replays should not accidentally use
    injury/depth rows refreshed after kickoff, so they fall back to the
    lock/open odds time capped at first kickoff.
    """
    now_utc = datetime.now(timezone.utc)
    today_et = datetime.now(_ET).date()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                (
                    SELECT MAX(fetched_at_utc)
                    FROM raw.nfl_api_responses
                    WHERE as_of_date = %(game_date)s
                      AND endpoint IN ('nfl_game_odds', 'nfl_player_props')
                      AND snapshot_role IN ('open', 'lock', 'live', 'legacy')
                ) AS raw_lock_ts,
                (
                    SELECT MAX(fetched_at_utc)
                    FROM odds.nfl_player_prop_lines
                    WHERE as_of_date = %(game_date)s
                      AND snapshot_role IN ('open', 'lock', 'live', 'legacy')
                ) AS prop_lock_ts,
                (
                    SELECT MAX(fetched_at_utc)
                    FROM odds.nfl_game_lines
                    WHERE as_of_date = %(game_date)s
                      AND snapshot_role IN ('open', 'lock', 'live', 'legacy')
                ) AS game_lock_ts,
                (
                    SELECT MIN(start_ts_utc)
                    FROM raw.nfl_games
                    WHERE game_date_et = %(game_date)s
                ) AS first_start_ts
            """,
            {"game_date": et_day},
        )
        rec = dict(cur.fetchone() or {})
    lock_candidates = [
        value for value in (rec.get("raw_lock_ts"), rec.get("prop_lock_ts"), rec.get("game_lock_ts"))
        if isinstance(value, datetime)
    ]
    first_start = rec.get("first_start_ts")
    if et_day >= today_et:
        cutoff = now_utc
    else:
        cutoff = max(lock_candidates) if lock_candidates else now_utc
    if isinstance(first_start, datetime) and et_day < today_et:
        safe_start_cutoff = first_start - timedelta(seconds=1)
        if cutoff > safe_start_cutoff:
            cutoff = safe_start_cutoff
        if not lock_candidates:
            cutoff = safe_start_cutoff
    if cutoff > now_utc:
        cutoff = now_utc
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    return cutoff


def _load_games(conn, et_day: date, context_cutoff_utc: datetime) -> pd.DataFrame:
    return pd.read_sql(
        """
        WITH latest_game_lines AS (
            SELECT DISTINCT ON (event_id, bookmaker_key)
                event_id,
                as_of_date,
                commence_time_utc,
                home_team_abbr,
                away_team_abbr,
                spread_home_points::float AS spread_home_line,
                total_points::float AS game_total_line,
                fetched_at_utc
            FROM odds.nfl_game_lines
            WHERE as_of_date = %(game_date)s
              AND snapshot_role IN ('open', 'lock', 'live', 'legacy')
              AND fetched_at_utc <= %(context_cutoff_utc)s
            ORDER BY event_id, bookmaker_key, fetched_at_utc DESC
        ),
        line_env AS (
            SELECT
                event_id,
                home_team_abbr,
                away_team_abbr,
                AVG(spread_home_line)::float AS spread_home_line,
                AVG(game_total_line)::float AS game_total_line
            FROM latest_game_lines
            GROUP BY event_id, home_team_abbr, away_team_abbr
        ),
        raw_games AS (
            SELECT
                game_id, season, week, season_type, game_date_et, start_ts_utc,
                home_team_abbr, away_team_abbr,
                -spread_line::float AS raw_spread_home_line,
                total_line::float AS raw_game_total_line
            FROM raw.nfl_games
            WHERE game_date_et = %(game_date)s
        ),
        odds_games AS (
            SELECT DISTINCT
                COALESCE(event_id, CONCAT('odds:', as_of_date::text, ':', home_team_abbr, ':', away_team_abbr)) AS game_id,
                NULL::integer AS season,
                NULL::integer AS week,
                NULL::text AS season_type,
                as_of_date AS game_date_et,
                commence_time_utc AS start_ts_utc,
                home_team_abbr,
                away_team_abbr,
                spread_home_line AS raw_spread_home_line,
                game_total_line AS raw_game_total_line
            FROM latest_game_lines
            WHERE as_of_date = %(game_date)s
        )
        SELECT DISTINCT ON (x.game_date_et, x.home_team_abbr, x.away_team_abbr)
            x.game_id,
            x.season,
            x.week,
            x.season_type,
            x.game_date_et,
            x.start_ts_utc,
            x.home_team_abbr,
            x.away_team_abbr,
            COALESCE(env.spread_home_line, x.raw_spread_home_line) AS spread_home_line,
            COALESCE(env.game_total_line, x.raw_game_total_line) AS game_total_line
        FROM (
            SELECT * FROM raw_games
            UNION ALL
            SELECT * FROM odds_games
        ) x
        LEFT JOIN line_env env
          ON (
              env.event_id = x.game_id
              OR (
                  env.home_team_abbr = x.home_team_abbr
                  AND env.away_team_abbr = x.away_team_abbr
              )
          )
        ORDER BY x.game_date_et, x.home_team_abbr, x.away_team_abbr, x.season NULLS LAST, x.game_id
        """,
        conn,
        params={"game_date": et_day, "context_cutoff_utc": context_cutoff_utc},
    )


def _load_history(conn, et_day: date) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            p.season, p.week, p.game_id, g.season_type,
            p.game_date_et, p.player_id, p.player_name,
            UPPER(p.team_abbr) AS team_abbr, UPPER(p.opponent_abbr) AS opponent_abbr,
            UPPER(p.position) AS position, p.is_home,
            p.passing_yards::float AS passing_yards,
            p.passing_tds::float AS passing_tds,
            p.rushing_yards::float AS rushing_yards,
            p.rushing_tds::float AS rushing_tds,
            p.receiving_yards::float AS receiving_yards,
            p.receiving_tds::float AS receiving_tds,
            p.carries::float AS carries,
            p.targets::float AS targets,
            p.receptions::float AS receptions,
            p.pass_attempts::float AS pass_attempts,
            p.routes_run::float AS routes_run,
            p.pass_route_opportunities::float AS pass_route_opportunities,
            p.pass_route_opportunity_share::float AS pass_route_opportunity_share,
            p.route_participation::float AS route_participation,
            p.snap_share::float AS snap_share,
            p.target_share::float AS target_share,
            p.air_yards_share::float AS air_yards_share,
            p.wopr::float AS wopr,
            p.receiving_air_yards::float AS receiving_air_yards,
            p.receiving_yards_after_catch::float AS receiving_yards_after_catch,
            p.targets_per_route_run::float AS targets_per_route_run,
            p.yards_per_route_run::float AS yards_per_route_run,
            p.first_read_targets::float AS first_read_targets,
            p.first_read_target_share::float AS first_read_target_share,
            p.end_zone_targets::float AS end_zone_targets,
            p.end_zone_target_share::float AS end_zone_target_share,
            p.offense_snaps::float AS offense_snaps,
            p.offense_snap_share::float AS offense_snap_share,
            p.red_zone_carries::float AS red_zone_carries,
            p.red_zone_targets::float AS red_zone_targets,
            p.red_zone_receptions::float AS red_zone_receptions,
            p.red_zone_pass_attempts::float AS red_zone_pass_attempts,
            p.red_zone_pass_tds::float AS red_zone_pass_tds,
            p.red_zone_rush_tds::float AS red_zone_rush_tds,
            p.red_zone_rec_tds::float AS red_zone_rec_tds,
            p.red_zone_touches::float AS red_zone_touches,
            p.goal_line_carries::float AS goal_line_carries,
            p.goal_line_targets::float AS goal_line_targets
        FROM raw.nfl_player_gamelogs p
        LEFT JOIN raw.nfl_games g ON g.game_id = p.game_id
        WHERE p.game_date_et < %(game_date)s
          AND g.status = 'final'
          AND (COALESCE(p.offense_snaps,0)>0 OR COALESCE(p.pass_attempts,0)+COALESCE(p.carries,0)+COALESCE(p.targets,0)>0)
          AND p.position IS NOT NULL
          AND UPPER(p.position) IN ('QB', 'RB', 'WR', 'TE')
        ORDER BY p.player_id, p.game_date_et, p.season, p.week, p.game_id
        """,
        conn,
        params={"game_date": et_day},
    )


def _load_prop_lines(conn, et_day: date, context_cutoff_utc: datetime) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (player_name_norm, stat, bookmaker_key, line)
                id AS offer_id, as_of_date, fetched_at_utc, event_id, commence_time_utc,
                bookmaker_key, bookmaker_title, home_team, away_team,
                player_name, player_name_norm, market_key, stat, line::float AS line,
                over_price, under_price, over_link, under_link
            FROM odds.nfl_player_prop_lines
            WHERE as_of_date = %(game_date)s
              AND snapshot_role IN ('open', 'lock', 'live', 'legacy')
              AND fetched_at_utc <= %(context_cutoff_utc)s
              AND stat IS NOT NULL
              AND line IS NOT NULL
            ORDER BY player_name_norm, stat, bookmaker_key, line, fetched_at_utc DESC
            """,
            {"game_date": et_day, "context_cutoff_utc": context_cutoff_utc},
        )
        return [dict(row) for row in cur.fetchall()]


def _load_player_context(conn, et_day: date, context_cutoff_utc: datetime) -> pd.DataFrame:
    return pd.read_sql(
        """
        WITH roster_observations AS MATERIALIZED (
            SELECT * FROM raw.nfl_rosters_at(%(context_cutoff_utc)s)
        ), depth_observations AS MATERIALIZED (
            SELECT * FROM raw.nfl_depth_charts_at(%(context_cutoff_utc)s)
        ), injury_observations AS MATERIALIZED (
            SELECT * FROM raw.nfl_injuries_at(%(context_cutoff_utc)s)
        ), latest_roster_season AS (
            SELECT MAX(season) AS season
            FROM roster_observations
            WHERE season <= %(season)s
              AND updated_at_utc <= %(context_cutoff_utc)s
        ),
        latest_rosters AS (
            SELECT DISTINCT ON (player_id)
                r.season,
                r.week,
                r.team_abbr AS roster_team_abbr,
                r.player_id,
                r.player_name,
                r.player_name_norm,
                UPPER(r.position) AS roster_position,
                r.depth_chart_position,
                r.roster_status,
                r.status_description_abbr,
                r.years_exp,
                r.height,
                r.weight
                , r.updated_at_utc AS roster_observed_at
            FROM roster_observations r
            JOIN latest_roster_season s ON s.season = r.season
            WHERE UPPER(COALESCE(r.position, '')) IN ('QB', 'RB', 'WR', 'TE')
              AND r.updated_at_utc <= %(context_cutoff_utc)s
              AND COALESCE(r.week,0) <= COALESCE((SELECT MIN(week) FROM raw.nfl_games WHERE game_date_et=%(game_date)s),99)
            ORDER BY
                player_id,
                COALESCE(week, 0) DESC,
                CASE WHEN roster_status = 'ACT' THEN 0 ELSE 1 END,
                updated_at_utc DESC
        ),
        latest_depth_season AS (
            SELECT MAX(season) AS season
            FROM depth_observations
            WHERE season <= %(season)s
              AND COALESCE(snapshot_ts_utc, updated_at_utc) <= %(context_cutoff_utc)s
        ),
        latest_depth AS (
            SELECT DISTINCT ON (COALESCE(player_id, player_name_norm), team_abbr)
                season,
                week,
                snapshot_ts_utc,
                team_abbr,
                player_id,
                player_name_norm,
                pos_grp,
                pos_name,
                pos_abb,
                pos_slot,
                pos_rank
                , d.updated_at_utc AS depth_observed_at
            FROM depth_observations d
            JOIN latest_depth_season s USING (season)
            WHERE COALESCE(d.snapshot_ts_utc, d.updated_at_utc) <= %(context_cutoff_utc)s
              AND COALESCE(d.week,0) <= COALESCE((SELECT MIN(week) FROM raw.nfl_games WHERE game_date_et=%(game_date)s),99)
            ORDER BY
                COALESCE(player_id, player_name_norm),
                team_abbr,
                COALESCE(week, 0) DESC,
                snapshot_ts_utc DESC NULLS LAST,
                pos_rank NULLS LAST
        ),
        latest_injury_season AS (
            SELECT MAX(season) AS season
            FROM injury_observations
            WHERE season <= %(season)s
              AND updated_at_utc <= %(context_cutoff_utc)s
        ),
        latest_injuries AS (
            SELECT DISTINCT ON (COALESCE(player_id, player_name_norm), team_abbr)
                season,
                week,
                team_abbr,
                player_id,
                player_name_norm,
                position,
                report_primary_injury,
                report_status,
                practice_status
                , i.updated_at_utc AS injury_observed_at
            FROM injury_observations i
            JOIN latest_injury_season s USING (season)
            WHERE i.updated_at_utc <= %(context_cutoff_utc)s
              AND i.season=%(season)s
              AND i.week=(SELECT MIN(week) FROM raw.nfl_games WHERE game_date_et=%(game_date)s)
            ORDER BY
                COALESCE(player_id, player_name_norm),
                team_abbr,
                COALESCE(week, 0) DESC,
                updated_at_utc DESC
        )
        , injury_scores AS (
            SELECT
                i.*,
                CASE
                    WHEN LOWER(COALESCE(i.report_status, '')) ~ 'injured reserve|reserve|out' THEN TRUE
                    WHEN LOWER(COALESCE(i.report_status, '')) LIKE '%%doubtful%%' THEN TRUE
                    ELSE FALSE
                END AS is_out,
                GREATEST(
                    CASE
                        WHEN LOWER(COALESCE(i.report_status, '')) ~ 'injured reserve|reserve|out' THEN 1.00
                        WHEN LOWER(COALESCE(i.report_status, '')) LIKE '%%doubtful%%' THEN 0.85
                        WHEN LOWER(COALESCE(i.report_status, '')) LIKE '%%questionable%%' THEN 0.45
                        ELSE 0.00
                    END,
                    CASE
                        WHEN LOWER(COALESCE(i.practice_status, '')) ~ 'did not participate|dnp' THEN 0.65
                        WHEN LOWER(COALESCE(i.practice_status, '')) LIKE '%%limited%%' THEN 0.25
                        ELSE 0.00
                    END
                )::float AS injury_score
            FROM latest_injuries i
        ),
        teammate_injury_context AS (
            SELECT
                r.player_id,
                r.roster_team_abbr AS team_abbr,
                COUNT(*) FILTER (
                    WHERE UPPER(COALESCE(s.position, '')) IN ('RB', 'WR', 'TE')
                      AND s.injury_score > 0
                )::float AS same_week_teammate_skill_injury_count,
                SUM(s.injury_score) FILTER (
                    WHERE UPPER(COALESCE(s.position, '')) IN ('RB', 'WR', 'TE')
                )::float AS same_week_teammate_skill_injury_score,
                SUM(s.injury_score) FILTER (
                    WHERE UPPER(COALESCE(s.position, '')) IN ('WR', 'TE')
                )::float AS same_week_teammate_receiver_injury_score,
                COUNT(*) FILTER (
                    WHERE UPPER(COALESCE(s.position, '')) IN ('WR', 'TE')
                      AND s.is_out
                )::float AS same_week_teammate_receiver_out_count
            FROM latest_rosters r
            LEFT JOIN injury_scores s
              ON s.team_abbr = r.roster_team_abbr
             AND COALESCE(s.player_id, '') <> COALESCE(r.player_id, '')
            GROUP BY r.player_id, r.roster_team_abbr
        )
        SELECT
            r.*,
            d.snapshot_ts_utc AS depth_snapshot_ts_utc,
            d.pos_grp,
            d.pos_name,
            d.pos_abb,
            d.pos_slot,
            d.pos_rank,
            d.depth_observed_at,
            (SELECT prev.pos_rank FROM depth_observations prev
             WHERE prev.player_id = r.player_id AND prev.team_abbr = r.roster_team_abbr
               AND prev.snapshot_ts_utc < d.snapshot_ts_utc - interval '6 days'
             ORDER BY prev.snapshot_ts_utc DESC LIMIT 1) AS previous_depth_pos_rank,
            i.week AS injury_week,
            i.report_primary_injury,
            i.report_status,
            i.practice_status,
            i.injury_observed_at,
            COALESCE(tic.same_week_teammate_skill_injury_count, 0)::float AS same_week_teammate_skill_injury_count,
            COALESCE(tic.same_week_teammate_skill_injury_score, 0)::float AS same_week_teammate_skill_injury_score,
            COALESCE(tic.same_week_teammate_receiver_injury_score, 0)::float AS same_week_teammate_receiver_injury_score,
            COALESCE(tic.same_week_teammate_receiver_out_count, 0)::float AS same_week_teammate_receiver_out_count
        FROM latest_rosters r
        LEFT JOIN latest_depth d
          ON (
              (d.player_id IS NOT NULL AND d.player_id = r.player_id)
              OR (d.player_id IS NULL AND d.player_name_norm = r.player_name_norm)
          )
         AND d.team_abbr = r.roster_team_abbr
        LEFT JOIN latest_injuries i
          ON (
              (i.player_id IS NOT NULL AND i.player_id = r.player_id)
              OR (i.player_id IS NULL AND i.player_name_norm = r.player_name_norm)
          )
         AND i.team_abbr = r.roster_team_abbr
        LEFT JOIN teammate_injury_context tic
          ON tic.player_id = r.player_id
         AND tic.team_abbr = r.roster_team_abbr
        """,
        conn,
        params={"season": nfl_season(et_day), "game_date": et_day, "context_cutoff_utc": context_cutoff_utc},
    )


def _today_matchups(games: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for _, row in games.iterrows():
        home = normalize_team(row.get("home_team_abbr"))
        away = normalize_team(row.get("away_team_abbr"))
        if not home or not away:
            continue
        spread_home = _clean_float(row.get("spread_home_line"))
        total_line = _clean_float(row.get("game_total_line"))
        home_implied = None
        away_implied = None
        if spread_home is not None and total_line is not None:
            home_implied = total_line / 2.0 - spread_home / 2.0
            away_implied = total_line / 2.0 + spread_home / 2.0
        base = {
            "game_id": row.get("game_id"),
            "season": row.get("season"),
            "week": row.get("week"),
            "season_type": row.get("season_type"),
            "game_date_et": row.get("game_date_et"),
            "start_ts_utc": row.get("start_ts_utc"),
            "game_total_line": total_line,
        }
        out[home] = {
            **base,
            "team_abbr": home,
            "opponent_abbr": away,
            "is_home": True,
            "team_spread_line": spread_home,
            "team_implied_points": home_implied,
            "opponent_implied_points": away_implied,
        }
        out[away] = {
            **base,
            "team_abbr": away,
            "opponent_abbr": home,
            "is_home": False,
            "team_spread_line": -spread_home if spread_home is not None else None,
            "team_implied_points": away_implied,
            "opponent_implied_points": home_implied,
        }
    return out


def _defense_allowed(history: pd.DataFrame) -> dict[str, dict[str, float]]:
    if history.empty:
        return {}
    game_allowed = (
        history.groupby(["opponent_abbr", "game_id", "season", "week"], dropna=False)[list(TARGET_STATS)]
        .sum(min_count=1)
        .reset_index()
        .sort_values(["opponent_abbr", "season", "week", "game_id"])
    )
    out: dict[str, dict[str, float]] = {}
    for team, sub in game_allowed.groupby("opponent_abbr"):
        sub = sub.sort_values(["season", "week", "game_id"]).tail(5)
        out[str(team)] = {
            f"opp_allowed_{stat}_avg_5": float(pd.to_numeric(sub[stat], errors="coerce").mean())
            for stat in TARGET_STATS
        }
    return out


def _context_indexes(context: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if context.empty:
        return {}, {}
    by_id: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for _, row in context.iterrows():
        rec = row.to_dict()
        player_id = str(rec.get("player_id") or "").strip()
        player_name_norm = normalize_name(str(rec.get("player_name") or ""))
        if player_id:
            by_id[player_id] = rec
        if player_name_norm:
            by_name[player_name_norm] = rec
    return by_id, by_name


def _context_is_active(ctx: dict[str, Any]) -> bool:
    status = str(ctx.get("roster_status") or "").upper()
    report_status = str(ctx.get("report_status") or "").lower()
    practice_status = str(ctx.get("practice_status") or "").lower()
    if status and status not in {"ACT"}:
        return False
    if report_status in {"out", "doubtful", "injured reserve"}:
        return False
    if "did not participate" in practice_status and report_status in {"out", "doubtful"}:
        return False
    return True


def _depth_rank(value: Any) -> float | None:
    rank = _clean_float(value)
    return rank if rank is not None and rank > 0 else None


def _rolling_value(row: dict[str, Any], stat: str, window: int = 5) -> float:
    return _clean_float(row.get(f"{stat}_avg_{window}")) or 0.0


def _primary_depth_abb(row: dict[str, Any]) -> str:
    return str(row.get("depth_pos_abb") or row.get("depth_chart_position") or row.get("position") or "").upper()


def _projection_only_role_allowed(row: dict[str, Any], stat: str, projection: float) -> tuple[bool, str]:
    pos = str(row.get("position") or "").upper()
    depth_abb = _primary_depth_abb(row)
    rank = _depth_rank(row.get("depth_pos_rank"))
    pass_att = _rolling_value(row, "pass_attempts")
    carries = _rolling_value(row, "carries")
    targets = _rolling_value(row, "targets")
    rec_yards = _rolling_value(row, "receiving_yards")
    rush_yards = _rolling_value(row, "rushing_yards")

    if stat in {"passing_yards", "passing_tds"}:
        if pos != "QB":
            return False, "position_not_qb"
        if depth_abb == "QB" and rank is not None:
            return (True, "qb1_depth") if rank <= 1.25 else (False, "projection_only_qb_backup")
        if depth_abb == "QB":
            return True, "qb_depth_no_rank"
        if pass_att >= 14.0 or _rolling_value(row, "passing_yards") >= 90.0:
            return True, "qb_recent_starter_usage"
        return False, "projection_only_qb_backup"

    if stat == "rushing_yards" and pos == "QB":
        if depth_abb == "QB" and rank is not None:
            return (True, "qb1_rushing_projection") if rank <= 1.25 else (False, "projection_only_qb_backup")
        if depth_abb == "QB":
            return True, "qb_depth_no_rank"
        if carries >= 3.0 or rush_yards >= 12.0:
            return True, "qb_rushing_usage"
        return False, "projection_only_qb_backup"

    if pos == "RB":
        if rank is not None and depth_abb not in {"RB", "FB"}:
            return False, "projection_only_non_rb_depth_slot"
        if depth_abb not in {"RB", "FB"} and carries < 3.0 and targets < 1.5:
            return False, "projection_only_non_rb_depth_slot"
        if stat == "rushing_yards":
            if rank is not None and depth_abb in {"RB", "FB"}:
                return (True, "rb_rush_role") if rank <= 3.0 else (False, "projection_only_low_rb_rush_role")
            if carries >= 4.0 or rush_yards >= 16.0 or projection >= 12.0:
                return True, "rb_rush_role"
            return False, "projection_only_low_rb_rush_role"
        if stat == "receiving_yards":
            if rank is not None and depth_abb in {"RB", "FB"}:
                return (True, "rb_receiving_role") if rank <= 3.0 else (False, "projection_only_low_rb_receiving_role")
            if targets >= 1.5 or rec_yards >= 9.0 or projection >= 7.0:
                return True, "rb_receiving_role"
            return False, "projection_only_low_rb_receiving_role"
        if stat == "rushing_tds":
            if rank is not None and depth_abb in {"RB", "FB"}:
                return (True, "rb_td_role") if rank <= 2.0 else (False, "projection_only_low_rb_td_role")
            if carries >= 5.0:
                return True, "rb_td_role"
            return False, "projection_only_low_rb_td_role"

    if pos in {"WR", "TE"}:
        primary_slot = "TE" if pos == "TE" else "WR"
        if depth_abb in {"KR", "PR"} and targets < 3.0 and rec_yards < 20.0:
            return False, "projection_only_special_teams_depth_slot"
        rank_cap = 2.0 if pos == "TE" else 4.0
        if depth_abb in {primary_slot, "SWR"} and rank is not None:
            return (True, "receiver_depth_role") if rank <= rank_cap else (False, "projection_only_low_receiver_role")
        if depth_abb in {primary_slot, "SWR"}:
            return True, "receiver_depth_no_rank"
        if targets >= 2.5 or rec_yards >= 18.0 or projection >= 10.0:
            return True, "receiver_recent_usage"
        return False, "projection_only_low_receiver_role"

    return False, "projection_only_role_unknown"


def _add_snapshot_role_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    group_cols = ["game_id", "team_abbr"]
    for stat in ROLE_SIGNAL_STATS:
        avg_col = f"{stat}_avg_5"
        if avg_col not in out.columns:
            out[avg_col] = np.nan
        values = pd.to_numeric(out[avg_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        sum_col = f"team_player_{stat}_avg5_sum"
        share_col = f"{stat}_share_avg_5"
        rank_col = f"{stat}_role_rank"
        team_sum = values.groupby([out[col] for col in group_cols]).transform("sum")
        out[sum_col] = team_sum.replace(0.0, np.nan)
        out[share_col] = (values / out[sum_col]).replace([np.inf, -np.inf], np.nan)
        out[rank_col] = (
            values.groupby([out[col] for col in group_cols])
            .rank(method="min", ascending=False)
            .where(team_sum > 0)
        )
    return out


def _snapshot_players(
    history: pd.DataFrame,
    games: pd.DataFrame,
    et_day: date,
    offers: list[dict[str, Any]],
    max_recency_days: int,
    context: pd.DataFrame,
) -> pd.DataFrame:
    if history.empty or games.empty:
        return pd.DataFrame()
    context_by_id, context_by_name = _context_indexes(context)
    for col in (*TARGET_STATS, *ROLLING_STATS):
        if col in history.columns:
            values = pd.to_numeric(history[col], errors="coerce")
            history[col] = values if col in SPARSE_USAGE_STATS else values.fillna(0.0)
    matchups = _today_matchups(games)
    if not matchups:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    allowed = _defense_allowed(history)
    team_game_counts = history.assign(_team=history['team_abbr'].map(normalize_team)).groupby('_team')['game_id'].nunique().to_dict()
    use_current_context = bool(context_by_id or context_by_name)
    for player_id, sub in history.groupby("player_id", dropna=False):
        player_id_key = str(player_id or "").strip()
        sub = sub.sort_values(["game_date_et", "season", "week", "game_id"])
        latest = sub.iloc[-1].to_dict()
        player_name_norm = normalize_name(str(latest.get("player_name") or ""))
        ctx = context_by_id.get(player_id_key) or context_by_name.get(player_name_norm) or {}
        context_name_norm = normalize_name(str(ctx.get("player_name") or ctx.get("player_name_norm") or ""))
        if use_current_context:
            if not ctx or not _context_is_active(ctx):
                continue
            team = normalize_team(ctx.get("roster_team_abbr"))
            if ctx.get("roster_position"):
                latest["position"] = ctx.get("roster_position")
        else:
            team = normalize_team(latest.get("team_abbr"))
        if team not in matchups:
            continue
        last_date = pd.to_datetime(latest.get("game_date_et"), errors="coerce")
        # Roles must be calculated from the roster, never the book's offered subset.
        if not use_current_context and not pd.isna(last_date) and int((pd.Timestamp(et_day) - last_date).days) > max_recency_days:
            continue
        matchup = matchups[team]
        row = {
            **latest,
            **matchup,
            "game_date_et": et_day,
            "roster_status": ctx.get("roster_status") if ctx else None,
            "roster_is_active": _context_is_active(ctx) if ctx else None,
            "context_player_name": ctx.get("player_name") if ctx else None,
            "context_player_name_norm": context_name_norm if ctx else None,
            "depth_chart_position": ctx.get("depth_chart_position") if ctx else None,
            "depth_pos_abb": ctx.get("pos_abb") if ctx else None,
            "depth_pos_rank": ctx.get("pos_rank") if ctx else None,
            "depth_pos_slot": ctx.get("pos_slot") if ctx else None,
            "previous_depth_pos_rank": ctx.get("previous_depth_pos_rank") if ctx else None,
            "roster_observed_at": ctx.get("roster_observed_at") if ctx else None,
            "depth_observed_at": ctx.get("depth_observed_at") if ctx else None,
            "injury_observed_at": ctx.get("injury_observed_at") if ctx else None,
            "injury_report_status": ctx.get("report_status") if ctx else None,
            "injury_practice_status": ctx.get("practice_status") if ctx else None,
            "same_week_teammate_skill_injury_count": ctx.get("same_week_teammate_skill_injury_count") if ctx else None,
            "same_week_teammate_skill_injury_score": ctx.get("same_week_teammate_skill_injury_score") if ctx else None,
            "same_week_teammate_receiver_injury_score": ctx.get("same_week_teammate_receiver_injury_score") if ctx else None,
            "same_week_teammate_receiver_out_count": ctx.get("same_week_teammate_receiver_out_count") if ctx else None,
        }
        games_played = len(sub)
        row["n_games_prev_3"] = min(games_played, 3)
        row["n_games_prev_5"] = min(games_played, 5)
        row["n_games_prev_10"] = min(games_played, 10)
        row["rest_days"] = None if pd.isna(last_date) else int((pd.Timestamp(et_day) - last_date).days)
        row["team_game_number"] = int(team_game_counts.get(team,0))
        row["opp_game_number"] = int(team_game_counts.get(matchup['opponent_abbr'],0))
        for stat in ROLLING_STATS:
            values = pd.to_numeric(sub.get(stat, pd.Series(dtype=float)), errors="coerce")
            if stat not in SPARSE_USAGE_STATS:
                values = values.fillna(0.0)
            for window in (3, 5, 10):
                row[f"{stat}_avg_{window}"] = float(values.tail(window).mean()) if len(values) else None
                if stat in VOLATILITY_STATS and window in (5, 10):
                    row[f"{stat}_std_{window}"] = float(values.tail(window).std()) if values.tail(window).count() >= 2 else None
        row.update(allowed.get(matchup["opponent_abbr"], {}))
        rows.append(row)
    # Current roster members without game history get an explicitly unproven
    # position prior, never an invented player record or an approved micro model.
    seen = {str(r['player_id']) for r in rows}
    recent = history.sort_values(['season','week','game_id']).groupby('player_id').tail(5)
    for player_id, ctx in context_by_id.items():
        team = normalize_team(ctx.get('roster_team_abbr'))
        position = str(ctx.get('roster_position') or '').upper()
        if str(player_id) in seen or team not in matchups or not _context_is_active(ctx) or position not in {'QB','RB','WR','TE'}:
            continue
        prior = recent.loc[recent.position == position]
        row = {**matchups[team], 'game_date_et':et_day, 'player_id':str(player_id),
               'player_name':ctx.get('player_name'), 'position':position, 'team_abbr':team,
               'depth_pos_rank':ctx.get('pos_rank'), 'depth_pos_abb':ctx.get('pos_abb'),
               'roster_status':ctx.get('roster_status'), 'cold_start':True,
               'n_games_prev_3':0,'n_games_prev_5':0,'n_games_prev_10':0}
        for stat in ROLLING_STATS:
            value = pd.to_numeric(prior.get(stat,pd.Series(dtype=float)),errors='coerce').mean()
            for window in (3,5,10):
                row[f'{stat}_avg_{window}'] = float(value) if pd.notna(value) else None
        row.update(allowed.get(matchups[team]['opponent_abbr'],{}))
        rows.append(row)
    return _add_context_risk_features(_add_snapshot_role_features(pd.DataFrame(rows)))


def _projection_layer_accepted(metrics: dict[str, Any]) -> bool:
    if bool(metrics.get("td_probability_accepted") or metrics.get("td_probability_pass")):
        return True
    if "projection_accepted" in metrics:
        return bool(metrics.get("projection_accepted"))
    if "projection_pass" in metrics:
        return bool(metrics.get("projection_pass"))
    return bool(metrics.get("accepted"))


def _predict_opportunity_layer(snapshot: pd.DataFrame, artifact: dict[str, Any], name: str, baseline_col: str, default: float) -> np.ndarray:
    X_raw = _make_features(snapshot)
    if baseline_col in snapshot.columns:
        base_series = snapshot[baseline_col]
    elif baseline_col in X_raw.columns:
        base_series = X_raw[baseline_col]
    else:
        base_series = pd.Series(default, index=snapshot.index)
    base = pd.to_numeric(base_series, errors="coerce").fillna(default).to_numpy(dtype=float)
    opp_artifact = artifact.get("opportunity_artifact") or {}
    metrics = (opp_artifact.get("metrics") or {}).get(name) or {}
    if not (metrics.get("accepted") or metrics.get("projection_pass") or metrics.get("upside_signal_pass")):
        return np.clip(base, 0.0, 1.0)
    model_obj = (opp_artifact.get("models") or {}).get(name)
    columns = (opp_artifact.get("feature_columns") or {}).get(name) or []
    fills = (opp_artifact.get("fill_values") or {}).get(name) or {}
    if model_obj is None or not columns:
        return np.clip(base, 0.0, 1.0)
    X = X_raw.reindex(columns=columns).fillna(fills).fillna(0.0)
    if isinstance(model_obj, dict):
        use_upside_signal = bool(metrics.get("upside_signal_pass") and model_obj.get("upside_model") is not None)
        raw_model = model_obj.get("upside_model") if use_upside_signal else model_obj.get("model")
        if raw_model is None:
            return np.clip(base, 0.0, 1.0)
        raw_pred = np.asarray(raw_model.predict(X), dtype=float)
        kind_key = "upside_kind" if use_upside_signal else "kind"
        if str(model_obj.get(kind_key) or "direct") == "residual":
            shrink_key = "upside_shrink" if use_upside_signal else "shrink"
            shrink = float(model_obj.get(shrink_key) or 0.0)
            raw_pred = base + shrink * raw_pred
        blend_key = "upside_probability_blend_weight" if use_upside_signal else "probability_blend_weight"
        blend_weight = model_obj.get(blend_key)
        if blend_weight is not None:
            weight = float(blend_weight)
            raw_pred = weight * np.asarray(raw_pred, dtype=float) + (1.0 - weight) * base
        return np.clip(raw_pred, 0.0, 1.0)
    return np.clip(np.asarray(model_obj.predict(X), dtype=float), 0.0, 1.0)


def _predict_opportunity_projection(snapshot: pd.DataFrame, artifact: dict[str, Any], name: str, baseline_col: str, default: float) -> np.ndarray:
    X_raw = _make_features(snapshot)
    if baseline_col in snapshot.columns:
        base_series = snapshot[baseline_col]
    elif baseline_col in X_raw.columns:
        base_series = X_raw[baseline_col]
    else:
        base_series = pd.Series(default, index=snapshot.index)
    base = pd.to_numeric(base_series, errors="coerce").fillna(default).to_numpy(dtype=float)
    opp_artifact = artifact.get("opportunity_artifact") or {}
    metrics = (opp_artifact.get("metrics") or {}).get(name) or {}
    if not (metrics.get("accepted") or metrics.get("projection_pass") or metrics.get("live_signal_accepted")):
        return np.clip(base, 0.0, None)
    model_obj = (opp_artifact.get("models") or {}).get(name)
    columns = (opp_artifact.get("feature_columns") or {}).get(name) or []
    fills = (opp_artifact.get("fill_values") or {}).get(name) or {}
    if model_obj is None or not columns:
        return np.clip(base, 0.0, None)
    X = X_raw.reindex(columns=columns).fillna(fills).fillna(0.0)
    try:
        if isinstance(model_obj, dict):
            raw_model = model_obj.get("model")
            if raw_model is None:
                return np.clip(base, 0.0, None)
            raw_pred = np.asarray(raw_model.predict(X), dtype=float)
            if str(model_obj.get("kind") or "direct") == "residual":
                raw_pred = base + float(model_obj.get("shrink") or 0.0) * raw_pred
            blend_weight = model_obj.get("probability_blend_weight")
            if blend_weight is not None:
                weight = float(blend_weight)
                raw_pred = weight * raw_pred + (1.0 - weight) * base
            return np.clip(raw_pred, 0.0, None)
        return np.clip(np.asarray(model_obj.predict(X), dtype=float), 0.0, None)
    except Exception as exc:
        log.warning("Could not score NFL opportunity projection %s: %s", name, exc)
        return np.clip(base, 0.0, None)


def _workload_adjustment_factors(snapshot: pd.DataFrame, artifact: dict[str, Any]) -> dict[str, np.ndarray]:
    if snapshot.empty:
        empty = np.asarray([], dtype=float)
        return {
            "full": empty,
            "limited": empty,
            "fragility": empty,
            "rest": empty,
            "workload_downside_v2": empty,
            "workload_upside_v2": empty,
            "qb_pass_attempts": empty,
            "high_pass": empty,
            "high_carry": empty,
            "high_target": empty,
            "receiver_spike_volume": empty,
            "receiver_target_route_spike": empty,
            "receiver_target_spike_v2": empty,
            "receiver_air_yards_share": empty,
            "receiver_air_yards": empty,
            "receiver_air_yards_spike": empty,
            "receiver_air_yards_spike_v2": empty,
            "receiver_ypt_efficiency_spike": empty,
            "rb_carry_spike_v2": empty,
            "receiver_live_spike_v3": empty,
            "rb_live_carry_v3": empty,
            "yardage_projection_volatility_v3": empty,
            "live_usage_context_quality_v4": empty,
            "receiver_spike_under_correction_v4": empty,
            "receiver_spike_yards_anchor_v4": empty,
            "rb_carry_under_correction_v4": empty,
            "rb_rush_yards_anchor_v4": empty,
            "yardage_projection_volatility_v4": empty,
            "spike_snap": empty,
        }
    def snap_array(name: str, default: float = 0.0) -> np.ndarray:
        return pd.to_numeric(
            snapshot.get(name, pd.Series(default, index=snapshot.index)),
            errors="coerce",
        ).fillna(default).clip(0.0, 1.0).to_numpy(dtype=float)

    full = _predict_opportunity_layer(snapshot, artifact, "full_workload_probability", "full_workload_score", 0.65)
    limited = _predict_opportunity_layer(snapshot, artifact, "limited_usage_risk", "limited_workload_risk_score", 0.05)
    qb_pass_attempts = _predict_opportunity_projection(snapshot, artifact, "qb_pass_attempts", "pass_attempts_avg_5", 31.5)
    high_pass = _predict_opportunity_layer(snapshot, artifact, "high_pass_attempt_probability", "high_pass_attempt_score", 0.20)
    high_carry = _predict_opportunity_layer(snapshot, artifact, "high_carry_probability", "high_carry_score", 0.18)
    high_target = _predict_opportunity_layer(snapshot, artifact, "high_target_probability", "high_target_score", 0.18)
    workload_downside_v2 = snap_array("workload_downside_v2_score", 0.0)
    workload_upside_v2 = snap_array("workload_upside_v2_score", 0.0)
    receiver_spike_volume = _predict_opportunity_layer(
        snapshot,
        artifact,
        "receiver_spike_volume_probability",
        "receiver_spike_volume_score",
        0.18,
    )
    receiver_target_route_spike = _predict_opportunity_layer(
        snapshot,
        artifact,
        "receiver_target_route_spike_probability",
        "receiver_target_route_spike_score",
        0.16,
    )
    receiver_target_spike_v2 = _predict_opportunity_layer(
        snapshot,
        artifact,
        "receiver_target_spike_v2_probability",
        "receiver_target_spike_v2_score",
        0.16,
    )
    receiver_air_yards_share = np.clip(
        _predict_opportunity_projection(snapshot, artifact, "receiver_air_yards_share", "air_yards_share_avg_5", 0.0),
        0.0,
        1.0,
    )
    receiver_air_yards = _predict_opportunity_projection(
        snapshot,
        artifact,
        "receiver_air_yards",
        "receiving_air_yards_avg_5",
        0.0,
    )
    receiver_air_yards_spike = _predict_opportunity_layer(
        snapshot,
        artifact,
        "receiver_air_yards_spike_probability",
        "receiver_air_yards_spike_v2_score",
        0.14,
    )
    receiver_air_yards_spike_v2 = snap_array("receiver_air_yards_spike_v2_score", 0.0)
    receiver_ypt_efficiency_spike = snap_array("receiver_ypt_efficiency_spike_score", 0.0)
    receiver_live_spike_v3 = snap_array("receiver_live_spike_v3_score", 0.0)
    rb_carry_spike_v2 = _predict_opportunity_layer(
        snapshot,
        artifact,
        "rb_carry_spike_v2_probability",
        "rb_carry_spike_v2_score",
        0.16,
    )
    rb_live_carry_v3 = snap_array("rb_live_carry_v3_score", 0.0)
    yardage_projection_volatility_v3 = snap_array("yardage_projection_volatility_v3_score", 0.0)
    live_usage_context_quality_v4 = snap_array("live_usage_context_quality_v4_score", 0.0)
    receiver_spike_under_correction_v4 = _predict_opportunity_layer(
        snapshot,
        artifact,
        "receiver_spike_under_correction_v4_probability",
        "receiver_spike_under_correction_v4_score",
        0.16,
    )
    receiver_spike_yards_anchor_v4 = pd.to_numeric(
        snapshot.get("receiver_spike_yards_anchor_v4", pd.Series(0.0, index=snapshot.index)),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    rb_carry_under_correction_v4 = _predict_opportunity_layer(
        snapshot,
        artifact,
        "rb_carry_under_correction_v4_probability",
        "rb_carry_under_correction_v4_score",
        0.16,
    )
    rb_rush_yards_anchor_v4 = pd.to_numeric(
        snapshot.get("rb_rush_yards_anchor_v4", pd.Series(0.0, index=snapshot.index)),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    yardage_projection_volatility_v4 = snap_array("yardage_projection_volatility_v4_score", 0.0)
    spike_snap = _predict_opportunity_layer(snapshot, artifact, "spike_snap_share_probability", "spike_snap_share_score", 0.18)
    fragility = pd.to_numeric(snapshot.get("high_usage_fragility_score", pd.Series(0.0, index=snapshot.index)), errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    rest = pd.to_numeric(snapshot.get("rest_risk_score", pd.Series(0.0, index=snapshot.index)), errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    week = pd.to_numeric(snapshot.get("week", pd.Series(0.0, index=snapshot.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    weird_week = np.where(week >= 18, 0.20, 0.0)
    weird_usage = pd.to_numeric(snapshot.get("weird_usage_risk_score", pd.Series(0.0, index=snapshot.index)), errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    return {
        "full": full,
        "limited": np.clip(np.maximum.reduce([limited, rest * 0.65 + weird_week, weird_usage, 0.88 * workload_downside_v2]), 0.0, 1.0),
        "fragility": fragility,
        "rest": rest,
        "weird": weird_usage,
        "workload_downside_v2": workload_downside_v2,
        "workload_upside_v2": workload_upside_v2,
        "qb_pass_attempts": qb_pass_attempts,
        "high_pass": high_pass,
        "high_carry": high_carry,
        "high_target": high_target,
        "receiver_spike_volume": receiver_spike_volume,
        "receiver_target_route_spike": receiver_target_route_spike,
        "receiver_target_spike_v2": receiver_target_spike_v2,
        "receiver_air_yards_share": receiver_air_yards_share,
        "receiver_air_yards": receiver_air_yards,
        "receiver_air_yards_spike": receiver_air_yards_spike,
        "receiver_air_yards_spike_v2": receiver_air_yards_spike_v2,
        "receiver_ypt_efficiency_spike": receiver_ypt_efficiency_spike,
        "rb_carry_spike_v2": rb_carry_spike_v2,
        "receiver_live_spike_v3": receiver_live_spike_v3,
        "rb_live_carry_v3": rb_live_carry_v3,
        "yardage_projection_volatility_v3": yardage_projection_volatility_v3,
        "live_usage_context_quality_v4": live_usage_context_quality_v4,
        "receiver_spike_under_correction_v4": receiver_spike_under_correction_v4,
        "receiver_spike_yards_anchor_v4": receiver_spike_yards_anchor_v4,
        "rb_carry_under_correction_v4": rb_carry_under_correction_v4,
        "rb_rush_yards_anchor_v4": rb_rush_yards_anchor_v4,
        "yardage_projection_volatility_v4": yardage_projection_volatility_v4,
        "spike_snap": spike_snap,
    }


def _apply_workload_projection_adjustment(
    snapshot: pd.DataFrame,
    stat: str,
    projection: np.ndarray,
    baseline: np.ndarray,
    workload: dict[str, np.ndarray],
) -> np.ndarray:
    pred = np.clip(np.asarray(projection, dtype=float), 0.0, None)
    if not len(pred):
        return pred
    full = np.asarray(workload.get("full", np.ones(len(pred)) * 0.65), dtype=float)
    limited = np.asarray(workload.get("limited", np.zeros(len(pred))), dtype=float)
    fragility = np.asarray(workload.get("fragility", np.zeros(len(pred))), dtype=float)
    rest = np.asarray(workload.get("rest", np.zeros(len(pred))), dtype=float)
    weird = np.asarray(workload.get("weird", np.zeros(len(pred))), dtype=float)
    workload_downside_v2 = np.asarray(workload.get("workload_downside_v2", np.zeros(len(pred))), dtype=float)
    workload_upside_v2 = np.asarray(workload.get("workload_upside_v2", np.zeros(len(pred))), dtype=float)
    high_pass = np.asarray(workload.get("high_pass", np.zeros(len(pred))), dtype=float)
    high_carry = np.asarray(workload.get("high_carry", np.zeros(len(pred))), dtype=float)
    high_target = np.asarray(workload.get("high_target", np.zeros(len(pred))), dtype=float)
    receiver_spike_volume = np.asarray(workload.get("receiver_spike_volume", np.zeros(len(pred))), dtype=float)
    receiver_target_route_spike = np.asarray(workload.get("receiver_target_route_spike", np.zeros(len(pred))), dtype=float)
    receiver_target_spike_v2 = np.asarray(workload.get("receiver_target_spike_v2", np.zeros(len(pred))), dtype=float)
    receiver_air_yards_share = np.asarray(workload.get("receiver_air_yards_share", np.zeros(len(pred))), dtype=float)
    receiver_air_yards = np.asarray(workload.get("receiver_air_yards", np.zeros(len(pred))), dtype=float)
    receiver_air_yards_spike = np.asarray(workload.get("receiver_air_yards_spike", np.zeros(len(pred))), dtype=float)
    receiver_air_yards_spike_v2 = np.asarray(workload.get("receiver_air_yards_spike_v2", np.zeros(len(pred))), dtype=float)
    receiver_ypt_efficiency_spike = np.asarray(workload.get("receiver_ypt_efficiency_spike", np.zeros(len(pred))), dtype=float)
    rb_carry_spike_v2 = np.asarray(workload.get("rb_carry_spike_v2", np.zeros(len(pred))), dtype=float)
    receiver_live_spike_v3 = np.asarray(workload.get("receiver_live_spike_v3", np.zeros(len(pred))), dtype=float)
    rb_live_carry_v3 = np.asarray(workload.get("rb_live_carry_v3", np.zeros(len(pred))), dtype=float)
    yardage_projection_volatility_v3 = np.asarray(workload.get("yardage_projection_volatility_v3", np.zeros(len(pred))), dtype=float)
    live_usage_context_quality_v4 = np.asarray(workload.get("live_usage_context_quality_v4", np.zeros(len(pred))), dtype=float)
    receiver_spike_under_correction_v4 = np.asarray(workload.get("receiver_spike_under_correction_v4", np.zeros(len(pred))), dtype=float)
    receiver_spike_yards_anchor_v4 = np.asarray(workload.get("receiver_spike_yards_anchor_v4", np.zeros(len(pred))), dtype=float)
    rb_carry_under_correction_v4 = np.asarray(workload.get("rb_carry_under_correction_v4", np.zeros(len(pred))), dtype=float)
    rb_rush_yards_anchor_v4 = np.asarray(workload.get("rb_rush_yards_anchor_v4", np.zeros(len(pred))), dtype=float)
    yardage_projection_volatility_v4 = np.asarray(workload.get("yardage_projection_volatility_v4", np.zeros(len(pred))), dtype=float)
    spike_snap = np.asarray(workload.get("spike_snap", np.zeros(len(pred))), dtype=float)
    base = np.clip(np.asarray(baseline, dtype=float), 0.0, None)

    def feature_array(name: str, default: float = 0.0) -> np.ndarray:
        return pd.to_numeric(
            snapshot.get(name, pd.Series(default, index=snapshot.index)),
            errors="coerce",
        ).fillna(default).to_numpy(dtype=float)

    heuristic_high_carry = pd.to_numeric(
        snapshot.get("high_carry_score", pd.Series(0.0, index=snapshot.index)),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    heuristic_high_target = pd.to_numeric(
        snapshot.get("high_target_score", pd.Series(0.0, index=snapshot.index)),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    heuristic_spike_snap = pd.to_numeric(
        snapshot.get("spike_snap_share_score", pd.Series(0.0, index=snapshot.index)),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    heuristic_high_pass = pd.to_numeric(
        snapshot.get("high_pass_attempt_score", pd.Series(0.0, index=snapshot.index)),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    high_carry = np.maximum(high_carry, heuristic_high_carry)
    high_target = np.maximum(high_target, heuristic_high_target)
    high_pass = np.maximum(high_pass, heuristic_high_pass)
    spike_snap = np.maximum(spike_snap, heuristic_spike_snap)
    if stat == "passing_yards":
        live_spike = pd.to_numeric(
            snapshot.get("qb_volume_spike_signal", snapshot.get("pass_spike_path_score", pd.Series(0.0, index=snapshot.index))),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        high_pass = np.maximum(high_pass, live_spike)
    if stat == "rushing_yards":
        live_spike = pd.to_numeric(
            snapshot.get("carry_spike_path_score", snapshot.get("spike_carry_opportunity_score", pd.Series(0.0, index=snapshot.index))),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        usage_spike = pd.to_numeric(
            snapshot.get("rb_usage_spike_signal", pd.Series(0.0, index=snapshot.index)),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        role_upside = pd.to_numeric(
            snapshot.get("role_change_upside_score", pd.Series(0.0, index=snapshot.index)),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        high_carry = np.maximum(high_carry, live_spike)
        high_carry = np.maximum(high_carry, usage_spike)
        high_carry = np.maximum(high_carry, rb_carry_spike_v2)
        high_carry = np.maximum(high_carry, 0.82 * workload_upside_v2)
        high_carry = np.maximum(high_carry, rb_carry_under_correction_v4)
        high_carry = np.maximum(high_carry, 0.75 * role_upside)
    if stat == "receiving_yards":
        live_spike = pd.to_numeric(
            snapshot.get("target_spike_path_score", snapshot.get("spike_target_opportunity_score", pd.Series(0.0, index=snapshot.index))),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        usage_spike = pd.to_numeric(
            snapshot.get("receiver_usage_spike_signal", pd.Series(0.0, index=snapshot.index)),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        receiver_spike_live = pd.to_numeric(
            snapshot.get("receiver_spike_volume_score", pd.Series(0.0, index=snapshot.index)),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        target_route_live = pd.to_numeric(
            snapshot.get("receiver_target_route_spike_score", pd.Series(0.0, index=snapshot.index)),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        target_eruption_live = pd.to_numeric(
            snapshot.get("receiver_target_eruption_score", pd.Series(0.0, index=snapshot.index)),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        air_path_live = pd.to_numeric(
            snapshot.get("air_yards_spike_path_score", pd.Series(0.0, index=snapshot.index)),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        air_eruption_live = pd.to_numeric(
            snapshot.get("receiver_air_yards_eruption_score", pd.Series(0.0, index=snapshot.index)),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        explosive_live = pd.to_numeric(
            snapshot.get("receiver_explosive_spike_score", pd.Series(0.0, index=snapshot.index)),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        role_upside = pd.to_numeric(
            snapshot.get("role_change_upside_score", pd.Series(0.0, index=snapshot.index)),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        high_target = np.maximum(high_target, live_spike)
        high_target = np.maximum(high_target, usage_spike)
        high_target = np.maximum(high_target, receiver_spike_volume)
        high_target = np.maximum(high_target, receiver_spike_live)
        high_target = np.maximum(high_target, receiver_target_route_spike)
        high_target = np.maximum(high_target, receiver_target_spike_v2)
        high_target = np.maximum(high_target, receiver_air_yards_spike)
        high_target = np.maximum(high_target, receiver_air_yards_spike_v2)
        high_target = np.maximum(high_target, receiver_ypt_efficiency_spike)
        high_target = np.maximum(high_target, 0.78 * np.clip((receiver_air_yards_share - 0.20) / 0.22, 0.0, 1.0))
        high_target = np.maximum(high_target, 0.76 * np.clip((receiver_air_yards - 42.0) / 58.0, 0.0, 1.0))
        high_target = np.maximum(high_target, 0.84 * workload_upside_v2)
        high_target = np.maximum(high_target, receiver_spike_under_correction_v4)
        high_target = np.maximum(high_target, target_route_live)
        high_target = np.maximum(high_target, target_eruption_live)
        high_target = np.maximum(high_target, air_path_live)
        high_target = np.maximum(high_target, air_eruption_live)
        high_target = np.maximum(high_target, explosive_live)
        high_target = np.maximum(high_target, 0.75 * role_upside)
    rank = pd.to_numeric(snapshot.get("depth_pos_rank", pd.Series(np.nan, index=snapshot.index)), errors="coerce").fillna(99).to_numpy(dtype=float)
    depth_penalty = np.where(rank > 4.25, 0.18, np.where(rank > 2.25, 0.08, 0.0))
    full_gap = np.clip(0.68 - full, 0.0, 0.68)
    severity = np.clip(
        0.38 * limited
        + 0.17 * fragility
        + 0.14 * workload_downside_v2
        + 0.10 * np.maximum(yardage_projection_volatility_v3, yardage_projection_volatility_v4)
        + 0.08 * np.clip(1.0 - live_usage_context_quality_v4, 0.0, 1.0)
        + 0.18 * full_gap
        + depth_penalty
        + 0.08 * rest
        + 0.14 * weird,
        0.0,
        1.0,
    )
    if stat == "receiving_yards":
        # The receiving-yards model now selects route/target features directly.
        # Post-processing should only handle clear role risk and true spike
        # evidence instead of broadly moving already-calibrated projections.
        route_anchor = np.maximum.reduce([
            feature_array("route_weighted_yard_expectation_avg_5"),
            feature_array("receiver_spike_yards_projection_signal"),
            feature_array("receiver_spike_volume_anchor_yards"),
            feature_array("receiver_target_route_spike_anchor_yards"),
            feature_array("receiver_spike_yards_anchor_v2"),
            feature_array("receiver_spike_yards_anchor_v3"),
            receiver_spike_yards_anchor_v4,
            0.58 * np.clip(receiver_air_yards, 0.0, 190.0),
            feature_array("receiver_target_eruption_anchor_targets") * np.clip(
                feature_array("receiving_yards_avg_5") / np.maximum(feature_array("targets_avg_5"), 1.0),
                4.0,
                18.0,
            ),
            feature_array("receiver_air_yards_spike_anchor_yards"),
            feature_array("route_env_receiving_yards_signal"),
            feature_array("receiving_yards_avg_5"),
            base,
            pred,
        ])
        route_evidence = np.maximum.reduce([
            high_target,
            feature_array("receiver_usage_spike_signal"),
            receiver_spike_volume,
            feature_array("receiver_spike_volume_score"),
            receiver_target_route_spike,
            feature_array("receiver_target_route_spike_score"),
            receiver_target_spike_v2,
            feature_array("receiver_target_spike_v2_score"),
            receiver_live_spike_v3,
            feature_array("receiver_live_spike_v3_score"),
            receiver_spike_under_correction_v4,
            feature_array("receiver_spike_under_correction_v4_score"),
            receiver_air_yards_spike_v2,
            feature_array("receiver_air_yards_spike_v2_score"),
            receiver_ypt_efficiency_spike,
            feature_array("receiver_ypt_efficiency_spike_score"),
            feature_array("receiver_contextual_spike_score"),
            feature_array("receiver_target_eruption_score"),
            feature_array("air_yards_spike_path_score"),
            feature_array("receiver_air_yards_eruption_score"),
            feature_array("receiver_explosive_spike_score"),
            receiver_air_yards_spike,
            0.78 * np.clip((receiver_air_yards_share - 0.20) / 0.22, 0.0, 1.0),
            0.76 * np.clip((receiver_air_yards - 42.0) / 58.0, 0.0, 1.0),
            feature_array("receiver_teammate_vacancy_score"),
            0.92 * feature_array("receiver_target_command_score"),
            0.88 * feature_array("receiver_route_spike_readiness_score"),
            feature_array("target_spike_path_score"),
            0.75 * feature_array("spike_snap_share_score"),
            0.70 * feature_array("role_change_upside_score"),
        ])
        workload_floor = np.clip(feature_array("workload_floor_score", 0.55), 0.0, 1.0)
        adjusted = pred.copy()
        severe_factor = np.clip(1.0 - 0.12 * severity, 0.72, 1.02)
        adjusted = np.where(severity >= 0.60, adjusted * severe_factor, adjusted)
        anchor_gate = (
            np.clip((route_evidence - 0.50) / 0.50, 0.0, 1.0)
            * np.clip(0.35 + 0.65 * workload_floor, 0.0, 1.0)
            * np.clip(0.55 + 0.45 * live_usage_context_quality_v4, 0.0, 1.0)
            * np.clip(1.0 - 0.48 * limited - 0.35 * workload_downside_v2 - 0.30 * rest - 0.25 * weird, 0.0, 1.0)
        )
        anchor_gap = np.clip(route_anchor - adjusted, 0.0, 34.0)
        target_route_gate = np.clip((np.maximum(receiver_target_route_spike, receiver_target_spike_v2) - 0.55) / 0.40, 0.0, 1.0)
        target_route_gap = np.clip(
            np.maximum.reduce([
                feature_array("receiver_target_route_spike_anchor_yards"),
                feature_array("receiver_spike_yards_anchor_v2"),
                feature_array("receiver_spike_yards_anchor_v3"),
                receiver_spike_yards_anchor_v4,
            ]) - adjusted,
            0.0,
            38.0,
        )
        explosive_gate = np.clip((np.maximum.reduce([
            feature_array("receiver_explosive_spike_score"),
            receiver_air_yards_spike,
            receiver_air_yards_spike_v2,
            receiver_ypt_efficiency_spike,
            0.78 * np.clip((receiver_air_yards_share - 0.20) / 0.22, 0.0, 1.0),
            0.76 * np.clip((receiver_air_yards - 42.0) / 58.0, 0.0, 1.0),
        ]) - 0.58) / 0.34, 0.0, 1.0)
        explosive_gap = np.clip(
            np.maximum(feature_array("receiver_air_yards_spike_anchor_yards"), 0.58 * np.clip(receiver_air_yards, 0.0, 190.0))
            - adjusted,
            0.0,
            40.0,
        )
        return np.clip(
            adjusted
            + 0.20 * anchor_gate * anchor_gap
            + 0.12 * target_route_gate * target_route_gap
            + 0.08 * explosive_gate * explosive_gap,
            0.0,
            None,
        )
    if stat in {"passing_yards", "passing_tds"}:
        factor = np.clip(1.0 - 0.30 * severity, 0.58, 1.04)
    elif stat in {"rushing_yards", "receiving_yards"}:
        factor = np.clip(1.0 - 0.26 * severity, 0.60, 1.05)
    elif stat in {"rushing_tds", "receiving_tds"}:
        factor = np.clip(1.0 - 0.22 * severity, 0.65, 1.03)
    else:
        factor = np.ones(len(pred), dtype=float)
    if stat == "passing_yards":
        upside = np.maximum(high_pass, 0.74 * workload_upside_v2)
        upside_cap = 1.12
    elif stat == "rushing_yards":
        upside = np.maximum.reduce([high_carry, 0.65 * spike_snap, rb_carry_spike_v2, rb_live_carry_v3, rb_carry_under_correction_v4, 0.82 * workload_upside_v2])
        upside_cap = 1.20
    elif stat == "receiving_yards":
        upside = np.maximum.reduce([high_target, 0.75 * spike_snap, receiver_target_spike_v2, receiver_air_yards_spike_v2, receiver_live_spike_v3, receiver_spike_under_correction_v4, 0.84 * workload_upside_v2])
        upside_cap = 1.22
    else:
        upside = np.zeros(len(pred), dtype=float)
        upside_cap = 1.04
    upside_strength = np.clip(upside - 0.50, 0.0, 0.50)
    healthy_role_gate = np.clip((0.55 + 0.45 * full) * (1.0 - severity), 0.0, 1.15)
    upside_factor = np.clip(1.0 + 0.62 * upside_strength * healthy_role_gate, 1.0, upside_cap)
    factor = factor * upside_factor
    adjusted = pred * factor
    if stat in {"rushing_yards", "receiving_yards"}:
        # Spike games are the largest current error bucket. When the upside
        # model and baseline both point above the raw projection, let a small
        # share of the proven baseline lift the projection instead of only
        # using downside guards.
        spike_lift = np.clip((upside - 0.58) / 0.34, 0.0, 1.0) * healthy_role_gate
        adjusted = np.where(base > adjusted, adjusted + 0.50 * spike_lift * (base - adjusted), adjusted)
        if stat == "rushing_yards":
            rb_anchor = np.maximum.reduce([feature_array("rb_rush_yards_anchor_v2"), feature_array("rb_rush_yards_anchor_v3"), rb_rush_yards_anchor_v4])
            adjusted = np.where(
                rb_anchor > adjusted,
                adjusted + 0.24 * spike_lift * np.clip(rb_anchor - adjusted, 0.0, 30.0),
                adjusted,
            )
        direct_spike_lift = np.where(
            upside >= 0.72,
            np.minimum(np.maximum(base, pred) * (0.04 + 0.08 * spike_lift), 8.0 if stat == "receiving_yards" else 10.0),
            0.0,
        )
        adjusted = adjusted + direct_spike_lift
    if stat == "rushing_yards":
        positions = snapshot.get("position", pd.Series("", index=snapshot.index)).astype(str).str.upper()
        is_qb = positions.eq("QB").to_numpy()
        if np.any(is_qb):
            rush_avg = np.maximum.reduce([
                feature_array("rushing_yards_avg_3"),
                feature_array("rushing_yards_avg_5"),
                feature_array("rushing_yards_avg_10"),
                base,
            ])
            carry_avg = np.maximum.reduce([
                feature_array("carries_avg_3"),
                feature_array("carries_avg_5"),
                feature_array("carries_avg_10"),
            ])
            qb_mobile_score = np.maximum.reduce([
                np.asarray([_sigmoid_score(value, 16.0, 7.0) for value in rush_avg], dtype=float),
                np.asarray([_sigmoid_score(value, 3.2, 1.2) for value in carry_avg], dtype=float),
                feature_array("qb_volume_spike_signal"),
                feature_array("pass_spike_path_score") * 0.25,
            ])
            low_rush_role = (rush_avg < 8.0) & (carry_avg < 2.0) & (qb_mobile_score < 0.35)
            low_role_cap = np.maximum(3.0, base * 1.35 + 4.0)
            mobile_cap = np.maximum.reduce([
                base * 1.35 + 8.0,
                rush_avg * 1.35 + 8.0,
                8.0 + 34.0 * qb_mobile_score,
            ])
            qb_cap = np.where(low_rush_role, low_role_cap, mobile_cap)
            adjusted = np.where(is_qb, np.minimum(adjusted, qb_cap), adjusted)
    # If the trained model is already below baseline, do not double-punish normal players.
    already_conservative = pred <= base
    mild_risk = severity < 0.20
    adjusted = np.where(already_conservative & mild_risk, pred, adjusted)
    return np.clip(adjusted, 0.0, None)


def _sigmoid_score(value: float, center: float, scale: float) -> float:
    z = max(-35.0, min(35.0, (float(value) - center) / max(scale, 1e-6)))
    return 1.0 / (1.0 + math.exp(-z))


def _yardage_high_score(stat: str, row: dict[str, Any]) -> float:
    if stat == "passing_yards":
        pred_high_pass = _clean_float(row.get("_pred_high_pass_attempt_probability"))
        current = max(
            _clean_float(row.get("high_pass_attempt_score")) or float("nan"),
            pred_high_pass if pred_high_pass is not None else float("nan"),
            _clean_float(row.get("qb_volume_spike_signal")) or float("nan"),
            _clean_float(row.get("pass_spike_path_score")) or float("nan"),
        )
        fallback = _sigmoid_score(_clean_float(row.get("pass_attempts_avg_5")) or 0.0, 31.0, 5.5)
    elif stat == "rushing_yards":
        carry = _clean_float(row.get("high_carry_score"))
        spike = _clean_float(row.get("spike_snap_share_score"))
        carry_spike = _clean_float(row.get("spike_carry_opportunity_score"))
        carry_path = _clean_float(row.get("carry_spike_path_score"))
        usage_spike = _clean_float(row.get("rb_usage_spike_signal"))
        rb_carry_spike_v2 = _clean_float(row.get("rb_carry_spike_v2_score"))
        pred_high_carry = _clean_float(row.get("_pred_high_carry_probability"))
        rb_live_carry_v3 = _clean_float(row.get("rb_live_carry_v3_score"))
        rb_carry_under_v4 = _clean_float(row.get("rb_carry_under_correction_v4_score"))
        workload_upside_v2 = _clean_float(row.get("workload_upside_v2_score"))
        role_upside = _clean_float(row.get("role_change_upside_score"))
        current = max(
            carry if carry is not None else float("nan"),
            pred_high_carry if pred_high_carry is not None else float("nan"),
            carry_spike if carry_spike is not None else float("nan"),
            carry_path if carry_path is not None else float("nan"),
            usage_spike if usage_spike is not None else float("nan"),
            rb_carry_spike_v2 if rb_carry_spike_v2 is not None else float("nan"),
            rb_live_carry_v3 if rb_live_carry_v3 is not None else float("nan"),
            rb_carry_under_v4 if rb_carry_under_v4 is not None else float("nan"),
            0.82 * workload_upside_v2 if workload_upside_v2 is not None else float("nan"),
            0.75 * role_upside if role_upside is not None else float("nan"),
            0.65 * spike if spike is not None else float("nan"),
        )
        fallback = _sigmoid_score(_clean_float(row.get("carries_avg_5")) or 0.0, 11.0, 3.8)
    elif stat == "receiving_yards":
        target = _clean_float(row.get("high_target_score"))
        spike = _clean_float(row.get("spike_snap_share_score"))
        target_spike = _clean_float(row.get("spike_target_opportunity_score"))
        target_path = _clean_float(row.get("target_spike_path_score"))
        usage_spike = _clean_float(row.get("receiver_usage_spike_signal"))
        receiver_spike = _clean_float(row.get("receiver_spike_volume_score"))
        target_route_spike = _clean_float(row.get("receiver_target_route_spike_score"))
        target_spike_v2 = _clean_float(row.get("receiver_target_spike_v2_score"))
        air_spike_v2 = _clean_float(row.get("receiver_air_yards_spike_v2_score"))
        ypt_spike_v2 = _clean_float(row.get("receiver_ypt_efficiency_spike_score"))
        receiver_live_spike_v3 = _clean_float(row.get("receiver_live_spike_v3_score"))
        receiver_spike_under_v4 = _clean_float(row.get("receiver_spike_under_correction_v4_score"))
        workload_upside_v2 = _clean_float(row.get("workload_upside_v2_score"))
        pred_receiver_spike = _clean_float(row.get("_pred_receiver_spike_volume_probability"))
        pred_target_route_spike = _clean_float(row.get("_pred_receiver_target_route_spike_probability"))
        pred_target_spike_v2 = _clean_float(row.get("_pred_receiver_target_spike_v2_probability"))
        pred_air_yards_share = _clean_float(row.get("_pred_receiver_air_yards_share"))
        pred_air_yards = _clean_float(row.get("_pred_receiver_air_yards"))
        pred_air_yards_spike = _clean_float(row.get("_pred_receiver_air_yards_spike_probability"))
        contextual_spike = _clean_float(row.get("receiver_contextual_spike_score"))
        target_eruption = _clean_float(row.get("receiver_target_eruption_score"))
        air_path = _clean_float(row.get("air_yards_spike_path_score"))
        air_eruption = _clean_float(row.get("receiver_air_yards_eruption_score"))
        explosive = _clean_float(row.get("receiver_explosive_spike_score"))
        vacancy = _clean_float(row.get("receiver_teammate_vacancy_score"))
        command = _clean_float(row.get("receiver_target_command_score"))
        route_ready = _clean_float(row.get("receiver_route_spike_readiness_score"))
        role_upside = _clean_float(row.get("role_change_upside_score"))
        current = max(
            target if target is not None else float("nan"),
            target_spike if target_spike is not None else float("nan"),
            target_path if target_path is not None else float("nan"),
            usage_spike if usage_spike is not None else float("nan"),
            receiver_spike if receiver_spike is not None else float("nan"),
            target_route_spike if target_route_spike is not None else float("nan"),
            target_spike_v2 if target_spike_v2 is not None else float("nan"),
            air_spike_v2 if air_spike_v2 is not None else float("nan"),
            ypt_spike_v2 if ypt_spike_v2 is not None else float("nan"),
            receiver_live_spike_v3 if receiver_live_spike_v3 is not None else float("nan"),
            receiver_spike_under_v4 if receiver_spike_under_v4 is not None else float("nan"),
            0.84 * workload_upside_v2 if workload_upside_v2 is not None else float("nan"),
            pred_receiver_spike if pred_receiver_spike is not None else float("nan"),
            pred_target_route_spike if pred_target_route_spike is not None else float("nan"),
            pred_target_spike_v2 if pred_target_spike_v2 is not None else float("nan"),
            pred_air_yards_spike if pred_air_yards_spike is not None else float("nan"),
            0.78 * ((pred_air_yards_share - 0.20) / 0.22) if pred_air_yards_share is not None else float("nan"),
            0.76 * ((pred_air_yards - 42.0) / 58.0) if pred_air_yards is not None else float("nan"),
            contextual_spike if contextual_spike is not None else float("nan"),
            target_eruption if target_eruption is not None else float("nan"),
            air_path if air_path is not None else float("nan"),
            air_eruption if air_eruption is not None else float("nan"),
            explosive if explosive is not None else float("nan"),
            vacancy if vacancy is not None else float("nan"),
            0.92 * command if command is not None else float("nan"),
            0.88 * route_ready if route_ready is not None else float("nan"),
            0.75 * role_upside if role_upside is not None else float("nan"),
            0.75 * spike if spike is not None else float("nan"),
        )
        fallback = _sigmoid_score(_clean_float(row.get("targets_avg_5")) or 0.0, 6.0, 2.2)
    else:
        return 0.0
    if not math.isfinite(current):
        current = fallback
    return max(0.02, min(0.92, float(current)))


def _yardage_state_weights_single(stat: str, row: dict[str, Any]) -> dict[str, float]:
    usage_quality = _clean_float(row.get("live_usage_context_quality_v4_score"))
    if usage_quality is None:
        usage_quality = 0.0
    limited = max(
        _clean_float(row.get("limited_workload_risk_score")) or 0.05,
        0.88 * (_clean_float(row.get("workload_downside_v2_score")) or 0.0),
        0.65 * (_clean_float(row.get("rest_risk_score")) or 0.0),
        0.18 * max(0.0, 1.0 - usage_quality),
    )
    full = _clean_float(row.get("full_workload_score"))
    if full is None:
        full = 0.65
    low = max(limited, max(0.0, 0.62 - full) * 0.55)
    low = max(0.02, min(0.70, low))
    raw_high = _yardage_high_score(stat, row) * (1.0 - 0.75 * low)
    spike = 0.0
    if stat == "receiving_yards":
        target_route_spike = _clean_float(row.get("receiver_target_route_spike_score"))
        target_spike_v2 = _clean_float(row.get("receiver_target_spike_v2_score"))
        air_spike_v2 = _clean_float(row.get("receiver_air_yards_spike_v2_score"))
        ypt_spike_v2 = _clean_float(row.get("receiver_ypt_efficiency_spike_score"))
        receiver_live_spike_v3 = _clean_float(row.get("receiver_live_spike_v3_score"))
        receiver_spike_under_v4 = _clean_float(row.get("receiver_spike_under_correction_v4_score"))
        workload_upside_v2 = _clean_float(row.get("workload_upside_v2_score"))
        pred_target_route_spike = _clean_float(row.get("_pred_receiver_target_route_spike_probability"))
        pred_target_spike_v2 = _clean_float(row.get("_pred_receiver_target_spike_v2_probability"))
        pred_air_yards_share = _clean_float(row.get("_pred_receiver_air_yards_share"))
        pred_air_yards = _clean_float(row.get("_pred_receiver_air_yards"))
        pred_air_yards_spike = _clean_float(row.get("_pred_receiver_air_yards_spike_probability"))
        contextual_spike = _clean_float(row.get("receiver_contextual_spike_score"))
        target_eruption = _clean_float(row.get("receiver_target_eruption_score"))
        air_path = _clean_float(row.get("air_yards_spike_path_score"))
        air_eruption = _clean_float(row.get("receiver_air_yards_eruption_score"))
        explosive = _clean_float(row.get("receiver_explosive_spike_score"))
        vacancy = _clean_float(row.get("receiver_teammate_vacancy_score"))
        command = _clean_float(row.get("receiver_target_command_score"))
        route_ready = _clean_float(row.get("receiver_route_spike_readiness_score"))
        receiver_spike = _clean_float(row.get("receiver_spike_volume_score"))
        pred_receiver_spike = _clean_float(row.get("_pred_receiver_spike_volume_probability"))
        spike_signal = max(
            target_route_spike if target_route_spike is not None else float("nan"),
            pred_target_route_spike if pred_target_route_spike is not None else float("nan"),
            pred_target_spike_v2 if pred_target_spike_v2 is not None else float("nan"),
            pred_air_yards_spike if pred_air_yards_spike is not None else float("nan"),
            0.78 * ((pred_air_yards_share - 0.20) / 0.22) if pred_air_yards_share is not None else float("nan"),
            0.76 * ((pred_air_yards - 42.0) / 58.0) if pred_air_yards is not None else float("nan"),
            contextual_spike if contextual_spike is not None else float("nan"),
            target_eruption if target_eruption is not None else float("nan"),
            air_path if air_path is not None else float("nan"),
            air_eruption if air_eruption is not None else float("nan"),
            explosive if explosive is not None else float("nan"),
            vacancy if vacancy is not None else float("nan"),
            0.92 * command if command is not None else float("nan"),
            0.88 * route_ready if route_ready is not None else float("nan"),
            receiver_spike if receiver_spike is not None else float("nan"),
            pred_receiver_spike if pred_receiver_spike is not None else float("nan"),
            target_spike_v2 if target_spike_v2 is not None else float("nan"),
            air_spike_v2 if air_spike_v2 is not None else float("nan"),
            ypt_spike_v2 if ypt_spike_v2 is not None else float("nan"),
            receiver_live_spike_v3 if receiver_live_spike_v3 is not None else float("nan"),
            receiver_spike_under_v4 if receiver_spike_under_v4 is not None else float("nan"),
            0.84 * workload_upside_v2 if workload_upside_v2 is not None else float("nan"),
        )
        if not math.isfinite(spike_signal):
            spike_signal = raw_high
        spike = max(0.02, min(0.58, spike_signal * (1.0 - 0.75 * low)))
        high = max(0.02, min(0.62, raw_high * (1.0 - 0.55 * spike)))
    else:
        high = max(0.02, min(0.70, raw_high))
    if low + high + spike > 0.92:
        scale = 0.92 / max(low + high + spike, 1e-9)
        low *= scale
        high *= scale
        spike *= scale
    normal_w = max(0.04, min(0.96, 1.0 - low - high - spike))
    denom = max(1e-9, low + normal_w + high + spike)
    out = {"low": low / denom, "normal": normal_w / denom, "high": high / denom}
    if stat == "receiving_yards":
        out["spike"] = spike / denom
    return out


def _receiver_ypt_prior_single(row: dict[str, Any]) -> float:
    targets = _clean_float(row.get("targets_avg_5"))
    yards = _clean_float(row.get("receiving_yards_avg_5"))
    if targets and targets > 0 and yards is not None:
        return max(2.0, min(18.0, yards / targets))
    route_proxy = _clean_float(row.get("yards_per_route_proxy_avg_5"))
    if route_proxy and route_proxy > 0:
        return max(2.0, min(18.0, route_proxy * 5.6))
    targets10 = _clean_float(row.get("targets_avg_10"))
    yards10 = _clean_float(row.get("receiving_yards_avg_10"))
    if targets10 and targets10 > 0 and yards10 is not None:
        return max(2.0, min(18.0, yards10 / targets10))
    return 7.2


def _receiver_spike_signal_values_single(row: dict[str, Any]) -> dict[str, float]:
    def feature(*names: str, default: float = 0.0) -> float:
        return _row_feature_score(row, *names, default=default)

    target_projection = _row_max_float(row, "receiver_projected_targets_v3", "receiver_projected_targets_v2", default=0.0)
    target_expectation = _row_max_float(row, "route_weighted_target_expectation_avg_5", "targets_avg_5", default=0.0)
    target_signal = max(
        feature("receiver_target_route_spike_score", "receiver_target_spike_v2_score"),
        feature("_pred_receiver_target_route_spike_probability", "_pred_receiver_target_spike_v2_probability"),
        feature("receiver_target_eruption_score", "receiver_target_command_score", "receiver_route_spike_readiness_score"),
        feature("receiver_teammate_vacancy_score"),
        0.80 * feature("workload_upside_v2_score"),
        (target_projection - 5.5) / 4.5,
        (target_expectation - 5.5) / 4.5,
        _sigmoid_score(_clean_float(row.get("targets_avg_5")) or 0.0, 5.8, 2.0),
    )

    pred_air_share = _clean_float(row.get("_pred_receiver_air_yards_share"))
    pred_air_yards = _clean_float(row.get("_pred_receiver_air_yards"))
    air_signal = max(
        feature("air_yards_spike_path_score", "receiver_air_yards_eruption_score", "receiver_air_yards_spike_v2_score"),
        feature("_pred_receiver_air_yards_spike_probability"),
        0.78 * ((pred_air_share - 0.20) / 0.22) if pred_air_share is not None else float("-inf"),
        0.76 * ((pred_air_yards - 42.0) / 58.0) if pred_air_yards is not None else float("-inf"),
        ((_clean_float(row.get("receiving_air_yards_avg_5")) or 0.0) - 45.0) / 55.0,
        ((_clean_float(row.get("air_yards_share_avg_5")) or 0.0) - 0.20) / 0.20,
        feature("receiver_explosive_spike_score"),
        _sigmoid_score(_clean_float(row.get("receiving_air_yards_avg_5")) or 0.0, 42.0, 24.0),
    )

    ypt_prior = _receiver_ypt_prior_single(row)
    ypt_signal = max(
        feature("receiver_ypt_efficiency_spike_score", "receiver_explosive_spike_score", "receiver_air_yards_eruption_score"),
        (ypt_prior - 8.0) / 4.0,
        ((_clean_float(row.get("yards_per_route_proxy_avg_5")) or 0.0) - 1.65) / 1.20,
        ((_clean_float(row.get("air_yards_per_target_avg_5")) or 0.0) - 8.5) / 4.0,
        _sigmoid_score(ypt_prior, 8.0, 2.5),
    )

    low_signal = max(
        feature("limited_workload_risk_score", "workload_downside_v2_score", "rest_risk_score", "weird_usage_risk_score"),
        1.0 - (_clean_float(row.get("full_workload_score")) if _clean_float(row.get("full_workload_score")) is not None else 0.65),
        0.35 * (1.0 - feature("live_usage_context_quality_v4_score")),
        0.08,
    )
    return {
        "target_spike": max(0.01, min(0.98, float(target_signal))),
        "air_yards_spike": max(0.01, min(0.98, float(air_signal))),
        "ypt_tail": max(0.01, min(0.98, float(ypt_signal))),
        "low_workload": max(0.01, min(0.98, float(low_signal))),
    }


def _signal_calibrated_probability_single(signal: float, calibrator: dict[str, Any]) -> float:
    if not calibrator:
        return max(0.001, min(0.999, float(signal)))
    global_rate = _clean_float(calibrator.get("global_rate"))
    if global_rate is None:
        global_rate = max(0.001, min(0.999, float(signal)))
    value = max(0.0, min(1.0, float(signal)))
    for rec in calibrator.get("bins") or []:
        lo = _clean_float(rec.get("lo")) or 0.0
        hi = _clean_float(rec.get("hi"))
        if hi is None:
            hi = 1.0
        if value >= lo and (value <= hi if hi >= 1.0 else value < hi):
            return max(0.001, min(0.999, _clean_float(rec.get("probability")) or global_rate))
    return max(0.001, min(0.999, float(global_rate)))


def _receiver_spike_mixture_weights_single(row: dict[str, Any], params: dict[str, Any]) -> dict[str, float]:
    signals = _receiver_spike_signal_values_single(row)
    calibrators = params.get("calibrators") or {}
    low = max(0.02, min(0.62, _signal_calibrated_probability_single(signals["low_workload"], calibrators.get("low_workload") or {})))
    target = max(0.01, min(0.55, _signal_calibrated_probability_single(signals["target_spike"], calibrators.get("target_spike") or {})))
    air = max(0.01, min(0.50, _signal_calibrated_probability_single(signals["air_yards_spike"], calibrators.get("air_yards_spike") or {})))
    ypt = max(0.01, min(0.45, _signal_calibrated_probability_single(signals["ypt_tail"], calibrators.get("ypt_tail") or {})))

    low *= _clean_float(params.get("low_weight_scale")) if _clean_float(params.get("low_weight_scale")) is not None else 1.0
    low = max(0.02, min(0.72, low))
    spike_scale = _clean_float(params.get("spike_weight_scale"))
    if spike_scale is None:
        spike_scale = 1.0
    target = target * spike_scale * (1.0 - 0.75 * low)
    air = air * spike_scale * (1.0 - 0.75 * low) * (1.0 - 0.25 * target)
    ypt = ypt * spike_scale * (1.0 - 0.75 * low) * (1.0 - 0.15 * target)
    spike_cap = _clean_float(params.get("spike_weight_cap"))
    if spike_cap is None:
        spike_cap = 0.68
    spike_sum = max(0.0, target + air + ypt)
    if spike_sum > spike_cap:
        scale = spike_cap / max(spike_sum, 1e-9)
        target *= scale
        air *= scale
        ypt *= scale
    total = low + target + air + ypt
    if total > 0.94:
        scale = 0.94 / max(total, 1e-9)
        low *= scale
        target *= scale
        air *= scale
        ypt *= scale
    normal = max(0.04, min(0.96, 1.0 - low - target - air - ypt))
    denom = max(1e-9, low + normal + target + air + ypt)
    return {
        "low": low / denom,
        "normal": normal / denom,
        "target_spike": target / denom,
        "air_yards_spike": air / denom,
        "ypt_tail": ypt / denom,
    }


def _receiver_spike_mixture_over_probability(
    projection: float,
    line: float,
    distribution: dict[str, Any],
    row: dict[str, Any],
) -> float | None:
    params = distribution.get("receiver_spike_mixture_params") or distribution.get("mixture_params") or {}
    states = params.get("states") or {}
    if not states:
        return None
    weights = _receiver_spike_mixture_weights_single(row, params)
    bias_weight = _clean_float(params.get("state_bias_weight"))
    if bias_weight is None:
        bias_weight = 0.35
    p_over = 0.0
    for state, weight in weights.items():
        rec = states.get(state) or {}
        sigma = max(1.0, _clean_float(rec.get("sigma")) or 1.0)
        bias = _clean_float(rec.get("bias")) or 0.0
        p_over += weight * (1.0 - norm.cdf(line, loc=projection + bias_weight * bias, scale=sigma))
    return float(max(0.001, min(0.999, p_over)))


def _receiver_spike_mixture_moments(
    projection: float,
    distribution: dict[str, Any],
    row: dict[str, Any],
) -> tuple[float, float] | None:
    params = distribution.get("receiver_spike_mixture_params") or distribution.get("mixture_params") or {}
    states = params.get("states") or {}
    if not states:
        return None
    weights = _receiver_spike_mixture_weights_single(row, params)
    bias_weight = _clean_float(params.get("state_bias_weight"))
    if bias_weight is None:
        bias_weight = 0.35
    means: list[float] = []
    variances: list[float] = []
    for state, weight in weights.items():
        rec = states.get(state) or {}
        state_sigma = max(1.0, _clean_float(rec.get("sigma")) or 1.0)
        state_mean = float(projection) + bias_weight * (_clean_float(rec.get("bias")) or 0.0)
        means.append(weight * state_mean)
        variances.append(weight * (state_sigma ** 2 + state_mean ** 2))
    mix_mean = float(sum(means))
    mix_var = max(1.0, float(sum(variances) - mix_mean ** 2))
    return mix_mean, math.sqrt(mix_var)


def _qb_ypa_prior_single(row: dict[str, Any]) -> float:
    att5 = _clean_float(row.get("pass_attempts_avg_5"))
    att10 = _clean_float(row.get("pass_attempts_avg_10"))
    yds5 = _clean_float(row.get("passing_yards_avg_5"))
    yds10 = _clean_float(row.get("passing_yards_avg_10"))
    ypa5 = yds5 / att5 if att5 and att5 > 0 and yds5 is not None else None
    ypa10 = yds10 / att10 if att10 and att10 > 0 and yds10 is not None else None
    if ypa5 is None and ypa10 is None:
        return 6.8
    if ypa5 is None:
        out = ypa10
    elif ypa10 is None:
        out = ypa5
    else:
        out = 0.62 * ypa5 + 0.38 * ypa10
    return max(3.0, min(11.5, float(out or 6.8)))


def _qb_pass_volume_moments(
    projection: float,
    distribution: dict[str, Any],
    row: dict[str, Any],
) -> tuple[float, float] | None:
    params = distribution.get("qb_volume_params") or {}
    attempt_states = params.get("attempt_states") or {}
    ypa_rec = params.get("ypa") or {}
    if not attempt_states or not ypa_rec:
        return None
    pred_attempts = _clean_float(row.get("_pred_qb_pass_attempts"))
    if pred_attempts is None:
        att5 = _clean_float(row.get("pass_attempts_avg_5")) or 31.5
        att10 = _clean_float(row.get("pass_attempts_avg_10")) or att5
        pred_attempts = 0.62 * att5 + 0.38 * att10
    pred_attempts = max(1.0, min(65.0, pred_attempts))
    ypa_prior = _qb_ypa_prior_single(row)
    ypa_bias = _clean_float(ypa_rec.get("bias")) or 0.0
    ypa_sigma = max(0.7, _clean_float(ypa_rec.get("sigma")) or 2.2)
    mean_blend_weight = _clean_float(params.get("yard_mean_blend_weight"))
    if mean_blend_weight is None:
        mean_blend_weight = 0.50
    attempt_bias_weight = _clean_float(params.get("attempt_bias_weight"))
    if attempt_bias_weight is None:
        attempt_bias_weight = 0.35
    ypa_bias_weight = _clean_float(params.get("ypa_bias_weight"))
    if ypa_bias_weight is None:
        ypa_bias_weight = 0.35
    weights = _yardage_state_weights_single("passing_yards", row)
    mean_terms: list[float] = []
    second_terms: list[float] = []
    for state in ("low", "normal", "high"):
        rec = attempt_states.get(state) or {}
        state_weight = float(weights.get(state, 0.0))
        if state_weight <= 0:
            continue
        attempt_mean = max(1.0, min(65.0, pred_attempts + attempt_bias_weight * (_clean_float(rec.get("bias")) or 0.0)))
        attempt_sigma = max(1.0, _clean_float(rec.get("sigma")) or 8.0)
        ypa_mean = max(3.0, min(11.5, ypa_prior + ypa_bias_weight * ypa_bias))
        volume_mean = attempt_mean * ypa_mean
        blended_mean = mean_blend_weight * projection + (1.0 - mean_blend_weight) * volume_mean
        volume_second = (attempt_sigma ** 2 + attempt_mean ** 2) * (ypa_sigma ** 2 + ypa_mean ** 2)
        volume_var = max(1.0, volume_second - volume_mean ** 2)
        blended_var = max(1.0, (1.0 - mean_blend_weight) ** 2 * volume_var + (mean_blend_weight * 38.0) ** 2)
        mean_terms.append(state_weight * blended_mean)
        second_terms.append(state_weight * (blended_var + blended_mean ** 2))
    if not mean_terms:
        return None
    mean = float(sum(mean_terms))
    variance = max(1.0, float(sum(second_terms) - mean ** 2))
    return mean, math.sqrt(variance)


def _qb_pass_volume_over_probability(
    projection: float,
    line: float,
    distribution: dict[str, Any],
    row: dict[str, Any] | None,
) -> float | None:
    if not row:
        return None
    moments = _qb_pass_volume_moments(projection, distribution, row)
    if moments is None:
        return None
    mean, sigma = moments
    return float(max(0.001, min(0.999, 1.0 - norm.cdf(line, loc=mean, scale=max(1.0, sigma)))))


def _mixture_yardage_over_probability(
    stat: str,
    projection: float,
    line: float,
    distribution: dict[str, Any],
    row: dict[str, Any] | None,
) -> float | None:
    if not row:
        return None
    params = distribution.get("mixture_params") or {}
    states = params.get("states") or {}
    if not states:
        return None
    weights = _yardage_state_weights_single(stat, row)
    p_over = 0.0
    for state in states:
        rec = states.get(state) or {}
        sigma = max(1.0, _clean_float(rec.get("sigma")) or 1.0)
        bias = _clean_float(rec.get("bias")) or 0.0
        bias_weight = _clean_float(params.get("state_bias_weight"))
        if bias_weight is None:
            bias_weight = 0.35
        p_state = 1.0 - norm.cdf(line, loc=projection + bias_weight * bias, scale=sigma)
        p_over += weights.get(state, 0.0) * p_state
    return float(max(0.001, min(0.999, p_over)))


def _projection_distribution_summary(
    stat: str,
    projection: float,
    metrics: dict[str, Any],
    distribution: dict[str, Any] | None,
    row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    distribution = distribution or {}
    if not SPEC_BY_STAT[stat].count_like and distribution.get("kind") == "empirical_oof_residual":
        residuals = np.asarray(distribution.get("residual_quantiles") or [], dtype=float)
        if len(residuals):
            quantiles = np.maximum(0, projection + np.quantile(residuals,[0.1,0.5,0.9]))
            return {"projection_p10":float(quantiles[0]),"projection_p50":float(quantiles[1]),
                    "projection_p90":float(quantiles[2]),"distribution_kind":"empirical_oof_residual",
                    "projection_confidence":0.5}
    if distribution and not distribution.get("accepted_distribution"):
        distribution = {}
    kind = str(distribution.get("kind") or ("poisson_count" if SPEC_BY_STAT[stat].count_like else "normal_residual"))
    mean = max(0.0, float(projection))
    if SPEC_BY_STAT[stat].count_like:
        if stat.endswith("_tds"):
            td_any = _clean_float((row or {}).get("_td_any_probability"))
            if td_any is not None and bool(metrics.get("td_probability_accepted") or metrics.get("td_probability_pass")):
                positive_mean = _clean_float(((metrics.get("rare_event") or {}).get("positive_mean"))) or 1.15
                expected = max(0.0, td_any * positive_mean)
                p10 = 0.0
                p50 = 1.0 if td_any >= 0.50 else 0.0
                p90 = 1.0 if td_any >= 0.10 else 0.0
                width = p90 - p10
                return {
                    "projection_p10": p10,
                    "projection_p50": p50,
                    "projection_p90": p90,
                    "distribution_kind": "rare_event_any_td",
                    "projection_confidence": float(1.0 / (1.0 + width + abs(expected - mean))),
                }
        lam = max(0.001, mean)
        p10 = float(poisson.ppf(0.10, lam))
        p50 = float(poisson.ppf(0.50, lam))
        p90 = float(poisson.ppf(0.90, lam))
        return {
            "projection_p10": p10,
            "projection_p50": p50,
            "projection_p90": p90,
            "distribution_kind": kind,
            "projection_confidence": float(1.0 / (1.0 + max(0.0, p90 - p10))),
        }
    sigma = _clean_float(distribution.get("residual_sigma"))
    bias = _clean_float(distribution.get("residual_bias")) or 0.0
    if distribution.get("kind") == "qb_attempt_ypa_mixture" and row:
        moments = _qb_pass_volume_moments(mean, distribution, row)
        if moments is not None:
            mean, sigma = moments
            kind = "qb_attempt_ypa_mixture"
            bias = 0.0
    if distribution.get("kind") == "receiver_spike_mixture_v1" and row:
        moments = _receiver_spike_mixture_moments(mean, distribution, row)
        if moments is not None:
            mean, sigma = moments
            kind = "receiver_spike_mixture_v1"
            bias = 0.0
    elif distribution.get("kind") == "mixture_yardage_residual" and row:
        params = distribution.get("mixture_params") or {}
        states = params.get("states") or {}
        if states:
            weights = _yardage_state_weights_single(stat, row)
            means = []
            variances = []
            for state in states:
                rec = states.get(state) or {}
                state_sigma = max(1.0, _clean_float(rec.get("sigma")) or 1.0)
                bias_weight = _clean_float(params.get("state_bias_weight"))
                if bias_weight is None:
                    bias_weight = 0.35
                state_mean = mean + bias_weight * (_clean_float(rec.get("bias")) or 0.0)
                state_weight = weights.get(state, 0.0)
                means.append(state_weight * state_mean)
                variances.append(state_weight * (state_sigma ** 2 + state_mean ** 2))
            mix_mean = float(sum(means))
            mix_var = max(1.0, float(sum(variances) - mix_mean ** 2))
            sigma = math.sqrt(mix_var)
            mean = mix_mean
            kind = "mixture_yardage_residual"
    if sigma is None:
        mae = _clean_float(metrics.get("mae")) or _clean_float(metrics.get("baseline_mae")) or max(1.0, mean * 0.35)
        sigma = max(1.0, float(mae) * 1.35)
    adjusted_mean = max(0.0, mean + 0.35 * bias)
    p10 = max(0.0, float(norm.ppf(0.10, loc=adjusted_mean, scale=max(1.0, float(sigma)))))
    p50 = max(0.0, float(norm.ppf(0.50, loc=adjusted_mean, scale=max(1.0, float(sigma)))))
    p90 = max(0.0, float(norm.ppf(0.90, loc=adjusted_mean, scale=max(1.0, float(sigma)))))
    width = max(0.0, p90 - p10)
    return {
        "projection_p10": p10,
        "projection_p50": p50,
        "projection_p90": p90,
        "distribution_kind": kind,
        "projection_confidence": float(1.0 / (1.0 + width / max(1.0, adjusted_mean + 1.0))),
    }


def _classifier_probability(model_obj: Any, X: pd.DataFrame) -> float | None:
    try:
        if hasattr(model_obj, "predict_proba"):
            proba = np.asarray(model_obj.predict_proba(X), dtype=float)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return _clip_probability(proba[0, 1])
            return _clip_probability(proba.ravel()[0])
        pred = np.asarray(model_obj.predict(X), dtype=float)
        return _clip_probability(pred.ravel()[0])
    except Exception as exc:
        log.warning("Could not score NFL exact-line prop model: %s", exc)
        return None


def _exact_line_overlay(
    artifact: dict[str, Any],
    *,
    stat: str,
    side: str,
    book: str | None,
    line: float,
    price: Any,
    projection: float,
    baseline: float,
    probability: float,
    ev: float | None,
    edge: float,
    market_probability: float | None,
    model_version: str,
) -> tuple[float, float | None, float | None, str | None]:
    exact_artifact = artifact.get("exact_line_artifact") or {}
    if exact_artifact.get("status") != "ready":
        return probability, None, None, None
    key = f"{stat}|{side}"
    metric = (exact_artifact.get("metrics") or {}).get(key) or {}
    model_payload = (exact_artifact.get("models") or {}).get(key) or {}
    if not bool(metric.get("accepted")) or not isinstance(model_payload, dict):
        return probability, None, None, None
    model_obj = model_payload.get("model")
    columns = model_payload.get("feature_columns") or []
    fills = model_payload.get("fill_values") or {}
    if model_obj is None or not columns:
        return probability, None, None, None
    no_vig = market_probability if market_probability is not None else 0.5
    feature_row = {
        "stat": stat,
        "side": side,
        "book": book,
        "line_bucket": _exact_line_line_bucket(stat, line),
        "price_bucket": _exact_line_price_bucket(price),
        "model_version": model_version,
        "line": line,
        "price": _clean_float(price),
        "projection": projection,
        "baseline_projection": baseline,
        "model_probability": probability,
        "model_ev": ev if ev is not None else 0.0,
        "model_edge": edge,
        "market_probability": _american_to_prob(price),
        "no_vig_probability": no_vig,
        "model_market_delta": probability - no_vig,
        "projection_edge_ratio": edge / max(1.0, abs(line)),
    }
    X_raw = _exact_line_feature_frame(pd.DataFrame([feature_row]))
    X = X_raw.reindex(columns=columns).fillna(fills).fillna(0.0)
    exact_probability = _classifier_probability(model_obj, X)
    if exact_probability is None:
        return probability, None, None, None
    model_brier = _clean_float(metric.get("model_probability_brier"))
    exact_brier = _clean_float(metric.get("brier"))
    gain = 0.0 if model_brier is None or exact_brier is None else max(0.0, model_brier - exact_brier)
    blend_weight = 0.60
    if gain >= 0.010:
        blend_weight = 0.72
    if gain >= 0.020:
        blend_weight = 0.82
    blended = float(np.clip(blend_weight * exact_probability + (1.0 - blend_weight) * probability, 0.001, 0.999))
    clv_probability: float | None = None
    clv_metric = metric.get("clv") or {}
    clv_model = model_payload.get("clv_model")
    if clv_model is not None and bool(clv_metric.get("accepted")):
        clv_probability = _classifier_probability(clv_model, X)
    return blended, exact_probability, clv_probability, f"exact_line_model_blend_{key}"


def _predict_stat(snapshot: pd.DataFrame, artifact: dict[str, Any], stat: str) -> tuple[np.ndarray, np.ndarray, bool]:
    model = (artifact.get("models") or {}).get(stat)
    metrics = (artifact.get("metrics") or {}).get(stat) or {}
    baseline = _baseline_from_metric_column(snapshot, stat, metrics)
    accepted = _projection_layer_accepted(metrics)
    if model is None:
        return baseline, baseline, False
    X_raw = _make_features(snapshot)
    columns = (artifact.get("feature_columns") or {}).get(stat) or []
    fills = (artifact.get("fill_values") or {}).get(stat) or {}
    if not columns:
        return baseline, baseline, False
    X = X_raw.reindex(columns=columns).fillna(fills).fillna(0.0)
    pred = _predict_model_payload(model, X, baseline)
    cold = snapshot.get('cold_start', pd.Series(False,index=snapshot.index)).fillna(False).astype(bool).to_numpy()
    pred[cold] = baseline[cold]
    if not accepted:
        return baseline, baseline, False
    return pred, baseline, True


SEASON_RUSH_PRIOR_GAMES = 3.0           # weight of last season's per-game average, in games
SEASON_RUSH_PRIOR_GAMES_NEW_TEAM = 1.0  # less weight when the player changed teams
SEASON_RUSH_YPC_PRIOR_CARRIES = 20.0
SEASON_RUSH_MODEL_WEIGHT = 0.5
RB_SHARE_HALFLIFE_GAMES = 2.0     # recency weighting of this season's carry share (role changes)
RB_VACATED_ABSORPTION = 0.25      # fraction of an Out/Doubtful RB teammate's carry share passed to remaining backs


def _rb_role_adjusted_share(h: pd.DataFrame, team_game: pd.DataFrame, season: int, team: str,
                            player_id: Any, out_ids: set) -> float | None:
    """Recency-weighted carry share plus a fitted part of carries vacated by ruled-out RB teammates.

    Fit on 2020-2022, held out 2023-2024: share-estimate MAE 21.72 -> 21.47; games with an RB
    teammate Out/Doubtful 22.80 -> 21.53 (bias -11.1 -> -5.1 yards). Uses only pregame reports.
    """
    team_rbs = h[(h.season == season) & (h.team_abbr == team) & (h.position == "RB")]
    if team_rbs.empty:
        return None
    totals = team_game[(team_game.season == season) & (team_game.team_abbr == team)].set_index("game_id").team_car
    shares: dict[Any, float] = {}
    season_share: dict[Any, float] = {}
    for pid, g in team_rbs.groupby("player_id"):
        g = g.sort_values(["week", "game_id"])
        per_game = (g.carries.to_numpy() / g.game_id.map(totals).clip(lower=1.0).to_numpy())
        shares[pid] = float(pd.Series(per_game).ewm(halflife=RB_SHARE_HALFLIFE_GAMES).mean().iloc[-1])
        season_share[pid] = float(g.carries.sum() / max(1.0, float(g.game_id.map(totals).sum())))
    if player_id not in shares:
        return None
    vacated = min(0.9, sum(season_share[p] for p in out_ids if p in season_share and p != player_id))
    remaining = max(0.05, sum(s for p, s in shares.items() if p not in out_ids))
    own = shares[player_id]
    return own + RB_VACATED_ABSORPTION * vacated * own / remaining


def _ruled_out_rbs(context: pd.DataFrame | None, snapshot: pd.DataFrame) -> dict[str, set]:
    """Team -> RB player_ids listed Out/Doubtful for this game week, as of the forecast cutoff."""
    if context is None or context.empty or "report_status" not in context.columns:
        return {}
    c = context[context.report_status.isin(["Out", "Doubtful"])
                & (context.get("roster_position", pd.Series("", index=context.index)).astype(str).str.upper() == "RB")]
    if "week" in snapshot.columns and "injury_week" in c.columns:
        weeks = set(pd.to_numeric(snapshot.week, errors="coerce").dropna().astype(int))
        c = c[pd.to_numeric(c.injury_week, errors="coerce").isin(weeks)]
    out: dict[str, set] = {}
    for team, g in c.groupby(c.roster_team_abbr.astype(str).str.upper()):
        out[team] = set(g.player_id)
    return out


def _season_aware_rushing_estimate(history: pd.DataFrame, snapshot: pd.DataFrame,
                                   context: pd.DataFrame | None = None) -> np.ndarray:
    """Early-season rushing yards without stale cross-season rolling windows.

    50% shrinkage of this season's per-game average toward last season's (k=3 games, k=1 after a
    team change) + 50% this-season carry share x team carries x shrunk YPC. Weeks 1-8 of 2020-2025
    (5,584 RB/QB games): MAE 18.62 vs 19.36 for the 5-game cross-season average; lead backs in
    weeks 1-3 were over-projected by +4.8 yards. 2026 holdout: 17.93 vs 19.06.
    """
    out = np.full(len(snapshot), np.nan)
    if history.empty or snapshot.empty:
        return out
    h = history[(history.get("season_type", "REG").fillna("REG") == "REG")
                & history["position"].isin(["RB", "QB"])].copy()
    if h.empty:
        return out
    for col in ("rushing_yards", "carries"):
        h[col] = pd.to_numeric(h[col], errors="coerce").fillna(0.0)
    team_game = h.groupby(["season", "game_id", "team_abbr"]).carries.sum().rename("team_car").reset_index()
    ruled_out = _ruled_out_rbs(context, snapshot)
    pos_mean = h.groupby("position").rushing_yards.mean().to_dict()
    pos_ypc = (h.groupby("position").rushing_yards.sum() / h.groupby("position").carries.sum().replace(0, np.nan)).to_dict()
    for i, row in enumerate(snapshot.itertuples(index=False)):
        pos = str(getattr(row, "position", "") or "").upper()
        season = getattr(row, "season", None)
        if pos not in ("RB", "QB") or season is None or pd.isna(season):
            continue
        season = int(season)
        team = str(getattr(row, "team_abbr", "") or "").upper()
        mine = h[h.player_id == getattr(row, "player_id", None)]
        cur, prev = mine[mine.season == season], mine[mine.season == season - 1]
        if cur.empty and prev.empty:
            continue  # no history to correct; the backtest only covered players with prior games
        n = float(len(cur))
        prior = float(prev.rushing_yards.mean()) if len(prev) else float(pos_mean.get(pos, 0.0))
        k = SEASON_RUSH_PRIOR_GAMES_NEW_TEAM if (len(prev) and str(prev.team_abbr.iloc[-1]) != team) else SEASON_RUSH_PRIOR_GAMES
        shrunk = (n * (float(cur.rushing_yards.mean()) if n else 0.0) + k * prior) / (n + k)
        if n:
            team_cur = team_game[(team_game.season == season) & (team_game.team_abbr == team)]
            mine_games = team_game[team_game.game_id.isin(cur.game_id) & (team_game.team_abbr == team)]
            share = float(cur.carries.sum()) / max(1.0, float(mine_games.team_car.sum()))
            if pos == "RB":
                role_share = _rb_role_adjusted_share(h, team_game, season, team, getattr(row, "player_id", None),
                                                     ruled_out.get(team, set()))
                if role_share is not None:
                    share = role_share
            car = float(cur.carries.sum())
            ypc = ((float(cur.rushing_yards.sum()) + float(pos_ypc.get(pos, 4.0)) * SEASON_RUSH_YPC_PRIOR_CARRIES)
                   / (car + SEASON_RUSH_YPC_PRIOR_CARRIES))
            share_est = share * float(team_cur.team_car.mean()) * ypc if len(team_cur) else shrunk
            out[i] = 0.5 * shrunk + 0.5 * share_est
        else:
            out[i] = shrunk
    return out


RECEPTIONS_CATCH_RATE_PRIOR = {"RB": 0.78, "WR": 0.64, "TE": 0.72}
RECEPTIONS_CATCH_RATE_PSEUDO_TARGETS = 8.0


def _receptions_projection(snapshot: pd.DataFrame) -> np.ndarray:
    """Receptions = 50% recent-receptions mix + 50% projected targets x shrunk catch rate.

    Backtest on 6,369 2025-26 RB/WR/TE player-games: MAE 1.255, bias -0.05, better than
    any single rolling average (1.26-1.28). No trained receptions model exists yet.
    """
    def col(name: str) -> pd.Series:
        return pd.to_numeric(snapshot.get(name, pd.Series(np.nan, index=snapshot.index)), errors="coerce")

    avg5 = col("receptions_avg_5")
    avg3 = col("receptions_avg_3").fillna(avg5)
    avg10 = col("receptions_avg_10").fillna(avg5)
    recent = 0.25 * avg3 + 0.45 * avg5 + 0.30 * avg10
    targets10 = col("targets_avg_10").fillna(col("targets_avg_5"))
    prior = snapshot.get("position", pd.Series("", index=snapshot.index)).astype(str).str.upper().map(RECEPTIONS_CATCH_RATE_PRIOR).fillna(0.66)
    k = RECEPTIONS_CATCH_RATE_PSEUDO_TARGETS
    rate = ((avg10.fillna(0.0) * 10 + prior * k) / (targets10.fillna(0.0) * 10 + k)).clip(0.3, 0.95)
    targets = col("receiver_projected_targets_v3")
    targets = targets.where(targets > 0, col("targets_avg_5"))
    blended = 0.5 * recent + 0.5 * targets * rate
    return blended.fillna(recent).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)


def _predict_td_any_probability(snapshot: pd.DataFrame, artifact: dict[str, Any], stat: str) -> np.ndarray | None:
    model = (artifact.get("models") or {}).get(stat)
    if hasattr(model, "predict_any") and (artifact.get('metrics',{}).get(stat,{}) or {}).get('td_probability_accepted'):
        X = _make_features(snapshot).reindex(columns=artifact['feature_columns'][stat]).fillna(artifact['fill_values'][stat]).fillna(0)
        return model.predict_any(X)
    if stat != "receiving_tds":
        return None
    metrics = (artifact.get("metrics") or {}).get(stat) or {}
    if not bool(metrics.get("td_probability_accepted") or metrics.get("td_probability_pass")):
        return None
    model = (artifact.get("models") or {}).get(stat)
    if not isinstance(model, dict) or str(model.get("kind") or "") != "rare_event_classifier":
        return None
    clf = model.get("classifier")
    if clf is None:
        return None
    columns = (artifact.get("feature_columns") or {}).get(stat) or []
    fills = (artifact.get("fill_values") or {}).get(stat) or {}
    if not columns:
        return None
    X_raw = _make_features(snapshot)
    X = X_raw.reindex(columns=columns).fillna(fills).fillna(0.0)
    p_any = _predict_binary_probability(clf, X)
    probability_blend_weight = model.get("probability_blend_weight")
    if probability_blend_weight is not None:
        p_weight = float(probability_blend_weight)
        baseline = _baseline_from_metric_column(
            snapshot,
            stat,
            {
                "baseline_column": model.get("baseline_column"),
                "baseline_lookup": model.get("baseline_lookup"),
            },
        )
        baseline_prob = np.clip(1.0 - np.exp(-np.clip(np.asarray(baseline, dtype=float), 0.0, None)), 0.001, 0.999)
        p_any = np.clip(p_weight * p_any + (1.0 - p_weight) * baseline_prob, 0.001, 0.999)
    return p_any


def _side_prob(
    stat: str,
    projection: float,
    line: float,
    metrics: dict[str, Any],
    distribution: dict[str, Any] | None = None,
    row: dict[str, Any] | None = None,
) -> float:
    spec = SPEC_BY_STAT[stat]
    if stat.endswith("_tds"):
        td_any = _clean_float((row or {}).get("_td_any_probability"))
        if td_any is not None and bool(metrics.get("td_probability_accepted") or metrics.get("td_probability_pass")):
            if line < 0:
                return 1.0
            positive_mass = -math.expm1(-max(0.001, projection))
            return float(np.clip(td_any, 0.001, 0.999) * poisson.sf(math.floor(line), max(0.001, projection)) / positive_mass)
        return float(1.0 - poisson.cdf(math.floor(line), max(0.001, projection)))
    if spec.count_like:
        return float(1.0 - poisson.cdf(math.floor(line), max(0.001, projection)))
    distribution = distribution or {}
    if distribution.get("kind") == "empirical_oof_residual":
        residuals = np.asarray(distribution.get("residual_quantiles") or [], dtype=float)
        if len(residuals):
            return float(np.clip(np.mean(projection + residuals > line),0.001,0.999))
    if distribution and not distribution.get("accepted_distribution"):
        distribution = {}
    if distribution.get("kind") == "qb_attempt_ypa_mixture":
        qb_volume = _qb_pass_volume_over_probability(projection, line, distribution, row)
        if qb_volume is not None:
            return qb_volume
    if distribution.get("kind") == "receiver_spike_mixture_v1" and row:
        mixture = _receiver_spike_mixture_over_probability(projection, line, distribution, row)
        if mixture is not None:
            return mixture
    if distribution.get("kind") == "mixture_yardage_residual":
        mixture = _mixture_yardage_over_probability(stat, projection, line, distribution, row)
        if mixture is not None:
            return mixture
    sigma = _clean_float(distribution.get("residual_sigma"))
    if sigma is None:
        mae = _clean_float(metrics.get("mae")) or _clean_float(metrics.get("baseline_mae")) or max(1.0, projection * 0.35)
        sigma = max(1.0, float(mae) * 1.35)
    bias = _clean_float(distribution.get("residual_bias")) or 0.0
    adjusted_mean = projection + 0.35 * bias
    return float(1.0 - norm.cdf(line, loc=adjusted_mean, scale=max(1.0, float(sigma))))


def _line_outcomes(stat, projection, line, metrics, distribution=None, row=None, range_scale=1.0, center=None):
    """Unconditional over/under/push mass; yardages are integer-valued outcomes.

    range_scale > 1 widens the distribution about `center` (its median) by evaluating the
    original distribution at center + (x - center) / range_scale.
    """
    def point(x):
        if range_scale == 1.0 or center is None:
            return x
        return center + (x - center) / range_scale
    if float(line).is_integer():
        over = _side_prob(stat, projection, point(line + 0.5), metrics, distribution, row)
        at_least = _side_prob(stat, projection, point(line - 0.5), metrics, distribution, row)
        push = max(0.0, at_least - over)
        return float(over), float(1.0 - over - push), float(push)
    over = _side_prob(stat, projection, point(line), metrics, distribution, row)
    return float(over), float(1.0 - over), 0.0


MARKET_CALIBRATION_PATH = _MODEL_DIR / "market_calibration.json"


def _load_market_calibration() -> dict[str, Any]:
    """Fitted market-anchoring parameters; absent file means identity (no change)."""
    try:
        return json.loads(MARKET_CALIBRATION_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _market_calibration_params(metrics: dict[str, Any], stat: str) -> dict[str, float]:
    """Read from the captured metrics so replays use the parameters live scoring used."""
    rec = (((metrics or {}).get("market_calibration") or {}).get("stats") or {}).get(stat) or {}

    def value(key: str, low: float, high: float) -> float:
        v = _clean_float(rec.get(key))
        return 1.0 if v is None else float(np.clip(v, low, high))

    anchorable = not stat.endswith("_tds")
    return {
        # projection' = line + projection_trust * (model projection - line)
        "projection_trust": value("projection_trust", 0.0, 1.0) if anchorable else 1.0,
        # widen the forecast distribution about its median (yardage stats only)
        "range_scale": value("range_scale", 1.0, 3.0) if not SPEC_BY_STAT[stat].count_like else 1.0,
        # logit(p') = logit(market) + probability_trust * (logit(p) - logit(market))
        "probability_trust": value("probability_trust", 0.0, 1.0),
    }


def _widen_summary(summary: dict[str, Any], range_scale: float) -> dict[str, Any]:
    if range_scale == 1.0 or summary.get("projection_p50") is None:
        return summary
    out = dict(summary)
    p50 = float(summary["projection_p50"])
    for key in ("projection_p10", "projection_p90"):
        if summary.get(key) is not None:
            out[key] = max(0.0, p50 + range_scale * (float(summary[key]) - p50))
    confidence = _clean_float(summary.get("projection_confidence"))
    if confidence and confidence > 0:
        out["projection_confidence"] = float(1.0 / (1.0 + range_scale * (1.0 / confidence - 1.0)))
    return out


def _blend_toward_market(probability: float, market_probability: float | None, trust: float) -> float:
    if trust >= 1.0 or market_probability is None or not (0.0 < probability < 1.0) or not (0.0 < market_probability < 1.0):
        return float(probability)
    logit = lambda p: math.log(p / (1.0 - p))
    z = logit(market_probability) + trust * (logit(probability) - logit(market_probability))
    return float(1.0 / (1.0 + math.exp(-z)))


def _candidate_from_offer(
    row: dict[str, Any],
    stat: str,
    projection: float,
    baseline: float,
    metrics: dict[str, Any],
    offer: dict[str, Any],
    distribution: dict[str, Any] | None = None,
    min_ev: float = 0.02,
    clv_guard_by_key: dict[tuple[str, str, str], dict[str, float]] | None = None,
    probability_calibration_by_key: dict[tuple[str, str, str, str, str], dict[str, float]] | None = None,
    model_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    line = float(offer["line"])
    market_cal = _market_calibration_params(metrics, stat)
    model_projection = float(projection)
    if market_cal["projection_trust"] < 1.0:
        # Weeks 1-3: the model's disagreement with the line carried no information.
        projection = line + market_cal["projection_trust"] * (model_projection - line)
    distribution_summary = _projection_distribution_summary(stat, projection, metrics, distribution, row=row)
    raw_over_win, raw_under_win, push_probability = _line_outcomes(
        stat, projection, line, metrics, distribution, row,
        range_scale=market_cal["range_scale"], center=distribution_summary.get("projection_p50"))
    distribution_summary = _widen_summary(distribution_summary, market_cal["range_scale"])
    nonpush = 1.0 - push_probability
    raw_p_over = raw_over_win / nonpush if nonpush > 1e-12 else 0.5
    adjustment_trace: dict[str, Any] = {}
    p_over, calibration_reasons = _offer_probability_calibration(
        stat=stat,
        raw_p_over=raw_p_over,
        projection=projection,
        baseline=baseline,
        line=line,
        offer=offer,
        row=row,
        distribution_summary=distribution_summary,
        trace=adjustment_trace,
    )
    if raw_over_win == 0.0 or raw_under_win == 0.0:
        p_over = 0.0 if raw_over_win == 0.0 else 1.0
    over_ev = _ev_per_unit(p_over, offer.get("over_price"), push_probability)
    heuristic_p_over = float(p_over)
    under_ev = _ev_per_unit(1.0 - p_over, offer.get("under_price"), push_probability)
    if over_ev is None and under_ev is None:
        side = "over" if projection > line else "under"
    elif (over_ev if over_ev is not None else -math.inf) >= (under_ev if under_ev is not None else -math.inf):
        side = "over"
    else:
        side = "under"
    prob = p_over if side == "over" else 1.0 - p_over
    price = offer.get(f"{side}_price")
    ev = _ev_per_unit(prob, price, push_probability)
    link = offer.get(f"{side}_link")
    market_prob = _no_vig_side_probability(offer.get("over_price"), offer.get("under_price"), side)
    edge = float(projection - line) if side == "over" else float(line - projection)
    model_version = str(metrics.get("live_model_version") or metrics.get("status") or "nfl_player_stat_models")
    exact_line_model_probability: float | None = None
    exact_line_clv_probability: float | None = None
    exact_line_model_version: str | None = None
    prob, exact_line_model_probability, exact_line_clv_probability, exact_reason = _exact_line_overlay(
        model_artifact or {},
        stat=stat,
        side=side,
        book=str(offer.get("bookmaker_key") or "") or None,
        line=line,
        price=price,
        projection=projection,
        baseline=baseline,
        probability=float(prob),
        ev=ev,
        edge=edge,
        market_probability=market_prob,
        model_version=model_version,
    )
    if exact_reason:
        calibration_reasons.append(exact_reason)
        exact_line_model_version = exact_reason.replace("exact_line_model_blend_", "")
        ev = _ev_per_unit(prob, price, push_probability)
    post_exact_probability = float(prob)
    prob, recent_calibration_reasons = _apply_recent_probability_calibration(
        float(prob),
        stat=stat,
        side=side,
        book=str(offer.get("bookmaker_key") or "").lower(),
        market_probability=market_prob,
        calibration=probability_calibration_by_key,
        line=line,
        projection=projection,
        baseline=baseline,
        position=str(row.get("position") or ""),
    )
    if recent_calibration_reasons:
        calibration_reasons.extend(recent_calibration_reasons)
    p_over = prob if side == "over" else 1.0 - prob
    if raw_over_win == 0.0 or raw_under_win == 0.0:
        p_over = 0.0 if raw_over_win == 0.0 else 1.0
        prob = p_over if side == 'over' else 1.0 - p_over
        calibration_reasons.append('distribution_support_boundary')
    pre_market_probability = float(prob)
    prob = _blend_toward_market(float(prob), market_prob, market_cal["probability_trust"])
    if prob != pre_market_probability:
        p_over = prob if side == "over" else 1.0 - prob
        calibration_reasons.append(f"market_blend_trust_{market_cal['probability_trust']:.2f}")
    ev = _ev_per_unit(prob, price, push_probability)
    projection_note = str(row.get("_projection_sanity_note") or "")
    if projection_note:
        calibration_reasons.append(projection_note)
    minimum_american_price = _break_even_american_price(prob) if nonpush > 1e-12 else None
    drift_guard_pass = minimum_american_price is not None and _price_has_positive_ev(prob, price, push_probability)
    clv_guard = (clv_guard_by_key or {}).get((str(offer.get("bookmaker_key") or "").lower(), stat, side))
    tier, reasons = _micro_projection_status(
        stat=stat,
        metrics=metrics,
        offer=offer,
        probability=float(prob),
        ev=ev,
        market_probability=market_prob,
        link=link,
        min_ev=min_ev,
        projection_confidence=distribution_summary.get("projection_confidence"),
        clv_guard=clv_guard,
        exact_line_clv_probability=exact_line_clv_probability,
        game_started=_game_has_started(row),
    )
    if calibration_reasons:
        reasons = ";".join([reasons, *calibration_reasons])
    if minimum_american_price is None:
        tier = 'paper'
        reasons += ';no_finite_positive_ev_minimum'
    bet_stats = metrics.get("bet_stats")
    if tier == "micro_projection" and bet_stats is not None and stat not in bet_stats:
        tier = "paper"
        reasons += ";display_only_market_not_bet"
    return {
        "game_date_et": row.get("game_date_et"),
        "season": row.get("season"),
        "week": row.get("week"),
        "game_id": row.get("game_id"),
        "player_id": row.get("player_id"),
        "player_name": row.get("player_name"),
        "team_abbr": row.get("team_abbr"),
        "opponent_abbr": row.get("opponent_abbr"),
        "position": row.get("position"),
        "roster_status": row.get("roster_status"),
        "depth_chart_position": row.get("depth_chart_position"),
        "depth_pos_abb": row.get("depth_pos_abb"),
        "depth_pos_rank": row.get("depth_pos_rank"),
        "depth_pos_slot": row.get("depth_pos_slot"),
        "injury_report_status": row.get("injury_report_status"),
        "injury_practice_status": row.get("injury_practice_status"),
        "offer_player_name": offer.get("player_name"),
        "offer_id": offer.get("offer_id"),
        "offer_player_name_norm": offer.get("player_name_norm") or normalize_name(str(offer.get("player_name") or "")),
        "stat": stat,
        "projection": float(projection),
        "baseline_projection": float(baseline),
        "model_version": model_version,
        "line": line,
        "side": side,
        "price": price,
        "book": offer.get("bookmaker_key"),
        "link": link,
        "probability": float(prob),
        "probability_basis": "win_given_no_push",
        "win_probability": float(prob * nonpush),
        "loss_probability": float((1.0 - prob) * nonpush),
        "push_probability": float(push_probability),
        "over_probability": float(p_over * nonpush),
        "under_probability": float((1.0 - p_over) * nonpush),
        "conditional_over_probability": float(p_over),
        "raw_over_probability": float(raw_p_over),
        "probability_trace": {"raw_over":float(raw_p_over), "heuristic_over":heuristic_p_over,
                              "post_exact_side":post_exact_probability, "pre_market_side":pre_market_probability,
                              "final_side":float(prob), "side":side,
                              "probability_basis":"win_given_no_push", "push_probability":float(push_probability),
                              "model_projection":model_projection, "market_calibration":market_cal,
                              "market_calibration_version":(metrics.get("market_calibration") or {}).get("version"),
                              **adjustment_trace},
        "ev": ev,
        "edge": edge,
        "minimum_american_price": minimum_american_price,
        "drift_guard_pass": drift_guard_pass,
        "projection_p10": distribution_summary.get("projection_p10"),
        "projection_p50": distribution_summary.get("projection_p50"),
        "projection_p90": distribution_summary.get("projection_p90"),
        "distribution_kind": distribution_summary.get("distribution_kind"),
        "market_no_vig_probability": market_prob,
        "model_market_edge": None if market_prob is None else float(prob - market_prob),
        "projection_confidence": distribution_summary.get("projection_confidence"),
        "exact_line_model_probability": exact_line_model_probability,
        "exact_line_clv_probability": exact_line_clv_probability,
        "exact_line_model_version": exact_line_model_version,
        "usage_context_quality": _usage_context_quality(row),
        "limited_usage_risk": _limited_usage_risk(row),
        "spike_usage_probability": _spike_usage_probability(stat, row),
        "tier": tier,
        "reasons": reasons,
        "prediction_key": "|".join([
            str(row.get("game_date_et")),
            str(row.get("player_id")),
            stat,
            str(line),
            side,
            str(offer.get("bookmaker_key") or ""),
        ]),
    }


def _projection_only_candidate(
    row: dict[str, Any],
    stat: str,
    projection: float,
    baseline: float,
    metrics: dict[str, Any],
    distribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    distribution_summary = _widen_summary(
        _projection_distribution_summary(stat, projection, metrics, distribution, row=row),
        _market_calibration_params(metrics, stat)["range_scale"])
    return {
        "game_date_et": row.get("game_date_et"),
        "season": row.get("season"),
        "week": row.get("week"),
        "game_id": row.get("game_id"),
        "player_id": row.get("player_id"),
        "player_name": row.get("player_name"),
        "team_abbr": row.get("team_abbr"),
        "opponent_abbr": row.get("opponent_abbr"),
        "position": row.get("position"),
        "roster_status": row.get("roster_status"),
        "depth_chart_position": row.get("depth_chart_position"),
        "depth_pos_abb": row.get("depth_pos_abb"),
        "depth_pos_rank": row.get("depth_pos_rank"),
        "depth_pos_slot": row.get("depth_pos_slot"),
        "injury_report_status": row.get("injury_report_status"),
        "injury_practice_status": row.get("injury_practice_status"),
        "offer_player_name": None,
        "offer_player_name_norm": None,
        "stat": stat,
        "projection": float(projection),
        "baseline_projection": float(baseline),
        "model_version": str(metrics.get("live_model_version") or metrics.get("status") or "nfl_player_stat_models"),
        "line": None,
        "side": None,
        "price": None,
        "book": None,
        "link": None,
        "probability": None,
        "over_probability": None,
        "under_probability": None,
        "raw_over_probability": None,
        "ev": None,
        "edge": None,
        "minimum_american_price": None,
        "drift_guard_pass": None,
        "projection_p10": distribution_summary.get("projection_p10"),
        "projection_p50": distribution_summary.get("projection_p50"),
        "projection_p90": distribution_summary.get("projection_p90"),
        "distribution_kind": distribution_summary.get("distribution_kind"),
        "market_no_vig_probability": None,
        "model_market_edge": None,
        "projection_confidence": distribution_summary.get("projection_confidence"),
        "exact_line_model_probability": None,
        "exact_line_clv_probability": None,
        "exact_line_model_version": None,
        "usage_context_quality": _usage_context_quality(row),
        "limited_usage_risk": _limited_usage_risk(row),
        "spike_usage_probability": _spike_usage_probability(stat, row),
        "tier": "projection",
        "reasons": ";".join(
            part for part in (
                str(row.get("projection_only_reason") or "no_prop_line_available"),
                str(row.get("_projection_sanity_note") or ""),
            )
            if part
        ),
        "prediction_key": "|".join([str(row.get("game_date_et")), str(row.get("player_id")), stat, "projection"]),
    }


def _apply_daily_micro_cap(rows: list[dict[str, Any]], max_micro: int) -> tuple[int, int]:
    from nfl_pipeline.betting_preferences import EXECUTION_BOOK, execution_link
    for row in rows:
        if row.get('tier') == 'micro_projection' and (row.get('book') != EXECUTION_BOOK or not execution_link(row.get('link'))):
            row['tier'] = 'paper'
            row['reasons'] = str(row.get('reasons') or '') + ';execution_book_or_link_ineligible'
    eligible = [
        (idx, row) for idx, row in enumerate(rows)
        if row.get("tier") == "micro_projection"
        and row.get("line") is not None
        and row.get("side") in {"over", "under"}
        and row.get("drift_guard_pass") is True
    ]
    eligible.sort(
        key=lambda item: (
            item[1].get("ev") or -999.0,
            item[1].get("model_market_edge") or -999.0,
            item[1].get("projection_confidence") or -999.0,
        ),
        reverse=True,
    )
    keep = set(); players = set()
    for idx, row in eligible:
        key = (row.get('game_id'), row.get('player_id'))
        if len(keep) < max(0, int(max_micro)) and key not in players:
            keep.add(idx); players.add(key)
    for idx, row in eligible:
        if idx in keep:
            continue
        row["tier"] = "paper"
        row["reasons"] = ";".join(
            part for part in (str(row.get("reasons") or ""), "micro_daily_cap_research_only")
            if part
        )
    return len(eligible), len(keep)


def _save_predictions(conn, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    fields = [
        "game_date_et", "season", "week", "game_id", "player_id", "player_name",
        "team_abbr", "opponent_abbr", "position",
        "roster_status", "depth_chart_position", "depth_pos_abb", "depth_pos_rank",
        "depth_pos_slot", "injury_report_status", "injury_practice_status",
        "offer_player_name", "offer_player_name_norm",
        "stat", "projection",
        "baseline_projection", "model_version", "line", "side", "price", "book",
        "link", "probability", "ev", "edge", "projection_p10", "projection_p50",
        "projection_p90", "distribution_kind", "market_no_vig_probability",
        "model_market_edge", "projection_confidence", "exact_line_model_probability",
        "exact_line_clv_probability", "exact_line_model_version", "over_probability",
        "under_probability", "raw_over_probability", "minimum_american_price",
        "drift_guard_pass", "usage_context_quality", "limited_usage_risk",
        "spike_usage_probability", "tier", "reasons", "prediction_key",
    ]
    fields.append("offer_id")
    from nfl_pipeline.forecast_store import save_forecasts
    return save_forecasts(conn, "nfl_player_prop_predictions", rows, fields)


def build_predictions(cfg: PredictConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    et_day = cfg.et_date or datetime.now(_ET).date()
    artifact = _load_model(cfg)
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_schema(conn)
        context_cutoff_utc = _prediction_context_cutoff(conn, et_day)
        games = _load_games(conn, et_day, context_cutoff_utc)
        from nfl_pipeline.game_scope import filter_frame
        games = filter_frame(games)
        games = games.loc[pd.to_datetime(games.start_ts_utc, utc=True, errors="coerce") > pd.Timestamp.now(tz="UTC")].copy()
        history = _load_history(conn, et_day)
        offers = _load_prop_lines(conn, et_day, context_cutoff_utc)
        context = _load_player_context(conn, et_day, context_cutoff_utc)
        snapshot = _snapshot_players(history, games, et_day, offers, cfg.max_projection_recency_days, context)
        clv_guard_by_key = _load_clv_guard(conn)
        probability_calibration_by_key = _load_recent_probability_calibration(conn, et_day)
        market_calibration = _load_market_calibration()
        from nfl_pipeline.betting_preferences import BET_PROP_STATS
        rows: list[dict[str, Any]] = []
        offer_exclusions = Counter()
        sanity_excluded = 0
        sanity_excluded_by_reason: dict[str, int] = {}
        if not snapshot.empty:
            for stat in [spec.stat for spec in STAT_SPECS]:
                mask = snapshot["position"].astype(str).str.upper().isin(SPEC_BY_STAT[stat].positions)
                sub = snapshot.loc[mask].copy()
                if sub.empty:
                    continue
                preds, baselines, accepted = _predict_stat(sub, artifact, stat)
                td_any_prob = _predict_td_any_probability(sub, artifact, stat)
                workload = _workload_adjustment_factors(sub, artifact) if artifact.get('apply_workload_adjustment',True) else {}
                if stat == "receptions":
                    preds = baselines = _receptions_projection(sub)
                raw_preds = preds.copy()
                if artifact.get("apply_workload_adjustment", True) and stat != "receptions":
                    preds = _apply_workload_projection_adjustment(sub, stat, preds, baselines, workload)
                season_rush = None
                model_preds = preds.copy()
                if stat == "rushing_yards":
                    season_rush = _season_aware_rushing_estimate(history, sub, context)
                    known = np.isfinite(season_rush)
                    preds = np.where(known, SEASON_RUSH_MODEL_WEIGHT * preds + (1 - SEASON_RUSH_MODEL_WEIGHT) * np.where(known, season_rush, 0.0), preds)
                metrics = dict((artifact.get("metrics") or {}).get(stat) or {})
                metrics["market_calibration"] = market_calibration
                metrics["bet_stats"] = sorted(BET_PROP_STATS)
                metrics["accepted_live"] = accepted
                metrics["live_model_version"] = str(artifact.get("version") or ("trained_workload_adjusted" if accepted else "baseline_fallback"))
                metrics["workload_adjusted_live"] = bool(accepted and np.any(np.abs(preds - raw_preds) > 1e-6))
                model_inputs = _make_features(sub).reindex(columns=(artifact.get('feature_columns') or {}).get(stat,[])).fillna((artifact.get('fill_values') or {}).get(stat,{}))
                for idx, (_, player_row) in enumerate(sub.iterrows()):
                    projection = float(preds[idx])
                    baseline = float(baselines[idx])
                    row_dict = player_row.to_dict()
                    projection_note = None
                    if artifact.get("apply_workload_adjustment", True):
                        projection, projection_note = _position_adjusted_projection(row_dict, stat, projection, baseline)
                    if projection_note:
                        row_dict["_projection_sanity_note"] = projection_note
                    if projection < SPEC_BY_STAT[stat].min_projection:
                        continue
                    if td_any_prob is not None:
                        row_dict["_td_any_probability"] = float(td_any_prob[idx])
                    if stat == "passing_yards":
                        attempt_values = workload.get("qb_pass_attempts")
                        if attempt_values is not None and len(attempt_values) == len(sub):
                            row_dict["_pred_qb_pass_attempts"] = float(max(0.0, np.asarray(attempt_values, dtype=float)[idx]))
                        values = workload.get("high_pass")
                        if values is not None and len(values) == len(sub):
                            row_dict["_pred_high_pass_attempt_probability"] = float(np.clip(np.asarray(values, dtype=float)[idx], 0.0, 1.0))
                    elif stat == "rushing_yards":
                        if season_rush is not None and np.isfinite(season_rush[idx]):
                            row_dict["_season_aware_rushing_estimate"] = float(season_rush[idx])
                            row_dict["_model_rushing_projection"] = float(model_preds[idx])
                        values = workload.get("high_carry")
                        if values is not None and len(values) == len(sub):
                            row_dict["_pred_high_carry_probability"] = float(np.clip(np.asarray(values, dtype=float)[idx], 0.0, 1.0))
                    if stat == "receiving_yards":
                        for source_key, row_key in (
                            ("receiver_target_route_spike", "_pred_receiver_target_route_spike_probability"),
                            ("receiver_target_spike_v2", "_pred_receiver_target_spike_v2_probability"),
                            ("receiver_spike_volume", "_pred_receiver_spike_volume_probability"),
                            ("receiver_air_yards_share", "_pred_receiver_air_yards_share"),
                            ("receiver_air_yards", "_pred_receiver_air_yards"),
                            ("receiver_air_yards_spike", "_pred_receiver_air_yards_spike_probability"),
                        ):
                            values = workload.get(source_key)
                            if values is not None and len(values) == len(sub):
                                value = float(np.asarray(values, dtype=float)[idx])
                                if "probability" in row_key or row_key.endswith("_share"):
                                    value = float(np.clip(value, 0.0, 1.0))
                                row_dict[row_key] = max(0.0, value)
                    player_offers, offer_issues = eligible_player_offers(offers, row_dict, stat, context_cutoff_utc)
                    offer_exclusions.update(offer_issues)
                    distribution = dict((artifact.get("distributions") or {}).get(stat) or {})
                    if not player_offers:
                        allowed, reason = _projection_only_role_allowed(row_dict, stat, projection)
                        if not allowed:
                            sanity_excluded += 1
                            sanity_excluded_by_reason[reason] = sanity_excluded_by_reason.get(reason, 0) + 1
                            continue
                        row_dict["projection_only_reason"] = (reason or 'no_prop_line_available') + ';no_fresh_fanduel_offer'
                    for offer in player_offers or [None]:
                        candidate = _candidate_from_offer(
                            row_dict,
                            stat,
                            projection,
                            baseline,
                            metrics,
                            offer,
                            distribution,
                            min_ev=cfg.min_ev,
                            clv_guard_by_key=clv_guard_by_key,
                            probability_calibration_by_key=probability_calibration_by_key,
                            model_artifact=artifact,
                        ) if offer else _projection_only_candidate(row_dict, stat, projection, baseline, metrics, distribution)
                        candidate['prediction_context_cutoff_utc'] = context_cutoff_utc.isoformat()
                        from nfl_pipeline.context_contract import context_evidence
                        candidate['context_evidence'] = context_evidence(row_dict, context_cutoff_utc)
                        candidate['forecast_features'] = row_dict
                        candidate['execution_contract'] = EXECUTION_CONTRACT
                        if offer:
                            from nfl_pipeline.modeling.scoring_capture import capture
                            candidate['scoring_replay'] = capture(row_dict, stat, projection, baseline, metrics,
                                offer, distribution, probability_calibration_by_key, clv_guard_by_key, cfg.min_ev, artifact)
                            candidate['quote_fetched_at_utc'] = str(offer['fetched_at_utc'])
                        candidate['projection_semantics'] = 'central_forecast_not_certified_mean' if stat.endswith('yards') else 'expected_count_or_any_td_head'
                        candidate['model_inputs'] = model_inputs.iloc[idx].to_dict()
                        candidate['forecast_distribution'] = distribution
                        if row_dict.get('cold_start') is True:
                            candidate['tier'] = 'paper'
                            candidate['reasons'] = 'cold_start_position_prior_unproven'
                        rows.append(candidate)
        micro_screen_passed, micro_selected = _apply_daily_micro_cap(rows, cfg.max_micro_props_per_day)
        if cfg.save_predictions:
            saved = _save_predictions(conn, rows)
        else:
            saved = 0
    meta = {
        "game_date": et_day.isoformat(),
        "prediction_context_cutoff_utc": context_cutoff_utc.isoformat(),
        "games": int(len(games)),
        "history_rows": int(len(history)),
        "snapshot_players": int(len(snapshot)),
        "offers": int(len(offers)),
        "offer_exclusions": dict(offer_exclusions),
        "execution_book": "fanduel",
        "player_context_rows": int(len(context)),
        "clv_guard_buckets": int(len(clv_guard_by_key)),
        "probability_calibration_buckets": int(len(probability_calibration_by_key)),
        "predictions": int(len(rows)),
        "micro_screen_passed": int(micro_screen_passed),
        "micro_selected": int(micro_selected),
        "micro_daily_cap": int(cfg.max_micro_props_per_day),
        "projection_sanity_excluded": int(sanity_excluded),
        "projection_sanity_excluded_by_reason": sanity_excluded_by_reason,
        "saved": saved,
        "model_status": artifact.get("status"),
        "exact_line_status": artifact.get("exact_line_status"),
        "exact_line_models": int(len(((artifact.get("exact_line_artifact") or {}).get("models") or {}))),
        "accepted_models": [
            stat for stat, rec in (artifact.get("metrics") or {}).items()
            if isinstance(rec, dict) and _projection_layer_accepted(rec)
        ],
        "accepted_projection_models": [
            stat for stat, rec in (artifact.get("metrics") or {}).items()
            if isinstance(rec, dict) and bool(rec.get("projection_accepted") or rec.get("projection_pass"))
        ],
        "accepted_probability_models": [
            stat for stat, rec in (artifact.get("metrics") or {}).items()
            if isinstance(rec, dict)
            and bool(rec.get("td_probability_accepted") or rec.get("td_probability_pass"))
            and not bool(rec.get("projection_accepted") or rec.get("projection_pass"))
        ],
    }
    return rows, meta


def _fmt_stat(stat: str) -> str:
    return SPEC_BY_STAT.get(stat).label if stat in SPEC_BY_STAT else stat


def _print_parlay(title: str, rows: list[dict[str, Any]]) -> None:
    url = _build_fd_parlay_url([row.get("link") for row in rows])
    if url:
        print(f"- {title} Parlay: [FD]({url})")


def _row_range_summary(row: dict[str, Any]) -> str:
    if row.get("projection_p10") is None or row.get("projection_p90") is None:
        return ""
    return f" range={float(row['projection_p10']):.1f}-{float(row['projection_p90']):.1f}"


def _row_probability_summary(row: dict[str, Any]) -> str:
    parts: list[str] = []
    side = str(row.get("side") or "").lower()
    if row.get("probability") is not None and side:
        label = f"{side}|no push" if row.get('push_probability', 0) > 0 else side
        parts.append(f"P({label})={float(row['probability']):.0%}")
    if row.get('push_probability', 0) > 0:
        parts.append(f"Push={float(row['push_probability']):.1%}")
    if row.get("market_no_vig_probability") is not None:
        parts.append(f"Mkt={float(row['market_no_vig_probability']):.0%}")
    if row.get("model_market_edge") is not None:
        parts.append(f"Edge={float(row['model_market_edge']):+.1%}")
    if row.get("ev") is not None:
        parts.append(f"EV={float(row['ev']):+.1%}")
    return " | ".join(parts)


def _row_price_summary(row: dict[str, Any]) -> str:
    if row.get("price") is None:
        return ""
    if row.get('previously_locked'):
        return f"Lock={_format_american(row.get('price'))} | historical quote, not a fresh play"
    drift = "OK" if row.get("drift_guard_pass") else "CHECK"
    return f"Cur={_format_american(row.get('price'))} Min={_format_american(row.get('minimum_american_price'))} Drift={drift}"


def _row_context_summary(row: dict[str, Any]) -> str:
    tags = _usage_context_tags(row)
    return f" [{' '.join(tags[:6])}]" if tags else ""


def _row_link(row: dict[str, Any]) -> str:
    link = row.get("link")
    if not link:
        return ""
    return f" [Bet {str(row.get('book') or '').upper()}](<{link}>)"


def _format_prop_row(row: dict[str, Any], *, action: str) -> str:
    line = row.get("line")
    stat = _fmt_stat(str(row.get("stat") or ""))
    range_s = _row_range_summary(row)
    context_s = _row_context_summary(row)
    if line is None:
        version_s = " baseline" if row.get("model_version") == "baseline_fallback" else ""
        return (
            f"- {action}: {row['player_name']} ({row['team_abbr']} vs {row['opponent_abbr']}) "
            f"{stat} proj={float(row['projection']):.2f}{range_s}{version_s}{context_s}"
        )
    side = str(row.get("side") or "").upper()
    prob_s = _row_probability_summary(row)
    price_s = _row_price_summary(row)
    exact_s = ""
    if row.get("exact_line_model_probability") is not None or row.get("exact_line_clv_probability") is not None:
        exact_parts = []
        if row.get("exact_line_model_probability") is not None:
            exact_parts.append(f"xLine={float(row['exact_line_model_probability']):.0%}")
        if row.get("exact_line_clv_probability") is not None:
            exact_parts.append(f"xCLV={float(row['exact_line_clv_probability']):.0%}")
        exact_s = " | " + " ".join(exact_parts)
    summary = " | ".join(part for part in (prob_s, price_s) if part)
    summary_s = f" | {summary}" if summary else ""
    return (
        f"- {action}: {row['player_name']} ({row['team_abbr']} vs {row['opponent_abbr']}) "
        f"{stat} {side}{float(line):.1f} {_format_american(row.get('price'))} "
        f"proj={float(row['projection']):.2f}{range_s}{summary_s}{exact_s}{context_s}{_row_link(row)}"
    )


def print_discord(rows: list[dict[str, Any]], meta: dict[str, Any], cfg: PredictConfig) -> None:
    print(f"🏈 {meta['game_date']} - NFL Player Props v1")
    print("")
    print("**DATA HEALTH**")
    print(f"- Games: {meta['games']} | Snapshot players: {meta['snapshot_players']} | Offers: {meta['offers']} | Predictions: {meta['predictions']}")
    print(f"- Pregame context cutoff: {meta.get('prediction_context_cutoff_utc') or 'unknown'}")
    print(f"- Projection sanity filtered: {meta.get('projection_sanity_excluded', 0)} low-role no-line rows")
    print(f"- Accepted projection models: {', '.join(meta.get('accepted_projection_models') or []) or 'none yet'}")
    print(f"- Exact-line model status: {meta.get('exact_line_status') or 'missing'} | Models: {meta.get('exact_line_models', 0)}")
    print(f"- Probability calibration memory: {meta.get('probability_calibration_buckets', 0)} settled buckets")
    if meta.get("accepted_probability_models"):
        print(f"- Accepted probability-only models: {', '.join(meta['accepted_probability_models'])}")
    print("- Bankroll Props: none - NFL props stay paper-only until live prop odds, CLV, grading, and ledger proof exist")
    micro_rows = [
        row for row in rows
        if row.get("tier") == "micro_projection"
        and row.get("line") is not None
        and row.get("side") in {"over", "under"}
    ]
    micro_rows.sort(key=lambda row: (row.get("ev") or -999.0, row.get("model_market_edge") or -999.0), reverse=True)
    if micro_rows:
        print("- Micro Props: $1 flat only; these are live evidence rows, not bankroll")
    print("- Bet only rows marked `$1 MICRO TEST`; paper sections are research, not approved real-money plays.")
    if not rows:
        print("")
        print("**NFL PLAYER PROPS**")
        print("- No NFL rows available for this date yet")
        return
    if micro_rows:
        shown_micro = micro_rows[:5]
        print("")
        print(f"**$1 MICRO TEST - NFL PROPS ({len(shown_micro)} selected / {len(micro_rows)} passed screen / ledger cap 5)**")
        print("- Bet size: $1 flat only. Bet only if current book price is still at or better than Min and Drift=OK.")
        for row in shown_micro:
            print(_format_prop_row(row, action="BET $1"))
        _print_parlay(f"Top {len(shown_micro)} NFL Micro Props", shown_micro)
    already_locked = [row for row in rows if row.get('tier') == 'locked_micro']
    if already_locked:
        print("")
        print("**PREVIOUSLY LOCKED $1 MICRO - NOT ADDITIONAL PLAYS**")
        print("- Original locked forecasts and prices. These are not refreshed betting instructions.")
        for row in already_locked:
            print(_format_prop_row(row, action='Already recorded'))
    print("")
    print("**PAPER NFL PROP PROJECTIONS**")
    if int(meta.get("offers") or 0) <= 0:
        print("- Books have not published parsed prop lines yet; these are role-sane stat forecasts, not bettable props.")
    sections = [
        ("QB Passing Yards", lambda row: row.get("position") == "QB" and row.get("stat") == "passing_yards"),
        ("QB Rushing Yards", lambda row: row.get("position") == "QB" and row.get("stat") == "rushing_yards"),
        ("QB Passing TDs", lambda row: row.get("position") == "QB" and row.get("stat") == "passing_tds"),
        ("RB Rushing Yards", lambda row: row.get("position") == "RB" and row.get("stat") == "rushing_yards"),
        ("RB Receiving Yards", lambda row: row.get("position") == "RB" and row.get("stat") == "receiving_yards"),
        ("RB Rushing TDs", lambda row: row.get("position") == "RB" and row.get("stat") == "rushing_tds"),
        ("WR/TE Receiving Yards", lambda row: row.get("position") in {"WR", "TE"} and row.get("stat") == "receiving_yards"),
        ("WR/TE Receiving TDs", lambda row: row.get("position") in {"WR", "TE"} and row.get("stat") == "receiving_tds"),
    ]
    for title, predicate in sections:
        candidates = [row for row in rows if predicate(row) and row.get("tier") not in {"micro_projection", "locked_micro"}]
        candidates.sort(key=lambda row: (row.get("ev") is not None, row.get("ev") or -999.0, row.get("projection") or 0.0), reverse=True)
        shown = candidates[: cfg.top_n_per_section]
        if not shown:
            continue
        print("")
        print(f"**Top {len(shown)} Paper {title}**")
        for row in shown:
            print(_format_prop_row(row, action="Research only"))
        _print_parlay(f"Top {len(shown)} Paper {title}", shown)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict NFL player props")
    parser.add_argument("--date", default=None, help="ET date YYYY-MM-DD")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--top-n-per-section", type=int, default=10)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = PredictConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        et_date=date.fromisoformat(args.date) if args.date else None,
        top_n_per_section=args.top_n_per_section,
        save_predictions=not args.no_save,
    )
    rows, meta = build_predictions(cfg)
    if os.getenv("DISCORD_FORMAT") == "1":
        print_discord(rows, meta, cfg)
    else:
        preview = [{k: v for k, v in row.items() if k not in {'model_inputs', 'forecast_distribution', 'scoring_replay', 'forecast_features'}} for row in rows[:50]]
        print(json.dumps({"meta": meta, "rows": preview}, indent=2, default=str))


if __name__ == "__main__":
    main()
