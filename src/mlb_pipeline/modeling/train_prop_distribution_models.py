"""Train/evaluate shadow MLB prop distribution models.

These are not bankroll models yet. They compare pricing offered sides from
count distributions against the current side probability and the no-vig market:

* strikeouts: Poisson count distribution from predicted Ks
* hits: Binomial PA x hit-rate distribution when projected PA exists
* total bases: compound per-PA out/walk/single/double/triple/HR curve
  projections when available
* home runs: rare-event per-PA HR distribution
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
try:
    import joblib
except Exception:  # pragma: no cover - optional runtime dependency
    joblib = None

from .prop_replay import ev_per_unit
from .prop_market_training import ensure_prop_market_training_schema
from .train_hitter_player_game_outcome_models import (
    apply_player_prior_state,
    CATEGORICAL_FEATURES as HITTER_OUTCOME_CATEGORICAL,
    convolve_hitter_outcomes,
    EVENT_CLASSES,
    TB_STATE_NAMES,
    NUMERIC_FEATURES as HITTER_OUTCOME_NUMERIC,
    _apply_tb_hr_tail_logit_offset,
    _pa_uncertainty_key,
    _predict_hierarchical_tb_state_models,
    _predict_hierarchical_event_probabilities,
    prepare_hitter_outcome_features,
    projected_pa_pmf,
)
from .train_prop_opportunity_models import (
    HITTER_CATEGORICAL,
    HITTER_NUMERIC,
    PITCHER_CATEGORICAL,
    PITCHER_NUMERIC,
    add_hitter_pa_v2_features,
    add_pitcher_history_features,
    _score_linear as _score_opportunity_linear,
)

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"

_SIDE_LINE_NUMERIC = [
    "model_prob_side",
    "market_prob_side",
    "p_distribution_side",
    "p_distribution_calibrated",
    "p_distribution_blend",
    "p_empirical_bucket",
    "market_line",
    "market_price",
    "pred_count",
    "pred_value",
    "projected_pa",
    "opp_model_pa",
    "same_book_pair_flag",
    "cross_book_pair_flag",
    "synthetic_pair_flag",
    "clean_market_pair_flag",
    "true_pair_flag",
]

_SIDE_LINE_CATEGORICAL = [
    "market",
    "side",
    "line_surface",
    "line_bucket",
    "price_bucket",
    "bookmaker_key",
    "pair_quality",
    "market_prob_source",
]


@dataclass(frozen=True)
class DistributionConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    out_file: str = "prop_distribution_models.json"
    report_file: str = "mlb_prop_distribution_models_latest.md"
    opportunity_file: str = "prop_opportunity_models.json"
    lookback_days: int = 365
    holdout_days: int = 28
    min_train_rows: int = 150
    min_holdout_rows: int = 40
    min_ev: float = 0.02


SQL = """
SELECT
    e.id,
    e.replay_id,
    e.run_id,
    e.game_slug,
    e.player_id,
    e.game_date_et,
    e.market,
    e.side,
    COALESCE(e.line_surface, 'unknown') AS line_surface,
    COALESCE(e.line_bucket, 'unknown') AS line_bucket,
    COALESCE(e.price_bucket, 'missing_price') AS price_bucket,
    COALESCE(e.bookmaker_key, 'unknown') AS bookmaker_key,
    COALESCE(e.pair_quality, 'unknown') AS pair_quality,
    COALESCE(e.market_prob_source, 'unknown') AS market_prob_source,
    COALESCE(e.same_book_pair_flag::float, CASE WHEN e.pair_quality = 'same_book' THEN 1.0 ELSE 0.0 END) AS same_book_pair_flag,
    COALESCE(e.cross_book_pair_flag::float, CASE WHEN e.pair_quality = 'cross_book' THEN 1.0 ELSE 0.0 END) AS cross_book_pair_flag,
    COALESCE(e.synthetic_pair_flag::float, CASE WHEN e.pair_quality = 'synthetic' THEN 1.0 ELSE 0.0 END) AS synthetic_pair_flag,
    COALESCE(
        e.clean_market_pair_flag::float,
        CASE
            WHEN e.pair_quality IN ('same_book', 'cross_book')
             AND COALESCE(e.market_prob_source, '') NOT IN ('raw_implied', 'synthetic_fanduel_over_only')
            THEN 1.0
            ELSE 0.0
        END
    ) AS clean_market_pair_flag,
    COALESCE(e.true_pair_flag::float, CASE WHEN e.pair_quality IN ('same_book', 'cross_book') THEN 1.0 ELSE 0.0 END) AS true_pair_flag,
    e.market_line::float AS market_line,
    e.market_price::float AS market_price,
    e.model_prob_side::float AS model_prob_side,
    e.market_prob_side::float AS market_prob_side,
    e.pred_count::float AS pred_count,
    e.pred_value::float AS pred_value,
    e.projected_pa::float AS projected_pa,
    e.pa_games::float AS pa_games,
    e.confirmed_batting_order::float AS confirmed_batting_order,
    COALESCE(e.confirmed_lineup_source, 'unknown') AS confirmed_lineup_source,
    e.projected_bf::float AS projected_bf,
    e.projected_ip::float AS projected_ip,
    e.projected_pitch_count::float AS projected_pitch_count,
    e.pitcher_starts::float AS pitcher_starts,
    e.is_home::float AS is_home,
    e.team_abbr,
    e.opponent_abbr,
    e.team_implied_runs::float AS team_implied_runs,
    e.opponent_implied_runs::float AS opponent_implied_runs,
    e.game_total_line::float AS game_total_line,
    h.park_run_factor::float AS park_run_factor,
    h.park_hr_factor::float AS park_hr_factor,
    h.park_babip_factor::float AS park_babip_factor,
    h.temperature_f::float AS temperature_f,
    h.wind_speed_mph::float AS wind_speed_mph,
    h.wind_sin::float AS wind_sin,
    h.wind_cos::float AS wind_cos,
    h.precip_prob_pct::float AS precip_prob_pct,
    h.is_dome::float AS is_dome,
    h.is_day_game::float AS is_day_game,
    h.weather_pregame_flag::float AS weather_pregame_flag,
    h.own_lineup_xwoba_avg::float AS own_lineup_xwoba_avg,
    h.own_lineup_xslg_avg::float AS own_lineup_xslg_avg,
    h.own_lineup_barrel_avg::float AS own_lineup_barrel_avg,
    h.own_lineup_hard_hit_avg::float AS own_lineup_hard_hit_avg,
    h.own_lineup_k_pct_cv::float AS own_lineup_k_pct_cv,
    h.own_lineup_pct_lhb::float AS own_lineup_pct_lhb,
    h.lineup_confirmed_flag::float AS lineup_confirmed_flag,
    h.confirmed_team_lineup_slots::float AS confirmed_team_lineup_slots,
    h.team_lineup_confirmed_flag::float AS team_lineup_confirmed_flag,
    h.lineup_boxscore_proxy_flag::float AS lineup_boxscore_proxy_flag,
    h.lineup_slot_x_team_implied_runs::float AS lineup_slot_x_team_implied_runs,
    e.opp_sp_hand,
    e.opp_sp_hand_l::float AS opp_sp_hand_l,
    e.opp_sp_k_pct_10::float AS opp_sp_k_pct_10,
    e.opp_sp_bb_pct::float AS opp_sp_bb_pct,
    e.opp_sp_xwoba::float AS opp_sp_xwoba,
    e.opp_sp_hard_hit_pct::float AS opp_sp_hard_hit_pct,
    e.opp_sp_whiff_pct::float AS opp_sp_whiff_pct,
    e.opp_bp_era_10::float AS opp_bp_era_10,
    e.opp_bp_whip_10::float AS opp_bp_whip_10,
    e.opp_bp_k9_10::float AS opp_bp_k9_10,
    e.opp_bp_ip_last_3::float AS opp_bp_ip_last_3,
    e.opp_bp_ip_last_7::float AS opp_bp_ip_last_7,
    e.opp_team_k_pct_10::float AS opp_team_k_pct_10,
    e.opp_team_avg_10::float AS opp_team_avg_10,
    e.opp_team_obp_10::float AS opp_team_obp_10,
    e.opp_team_slg_10::float AS opp_team_slg_10,
    e.batter_vs_hand_hits_avg_10::float AS batter_vs_hand_hits_avg_10,
    e.batter_vs_hand_tb_avg_10::float AS batter_vs_hand_tb_avg_10,
    e.batter_vs_hand_hr_avg_10::float AS batter_vs_hand_hr_avg_10,
    e.batter_vs_hand_iso_avg_10::float AS batter_vs_hand_iso_avg_10,
    e.batter_vs_hand_k_rate_10::float AS batter_vs_hand_k_rate_10,
    e.batter_vs_hand_games_10::float AS batter_vs_hand_games_10,
    e.batter_vs_rp_ba_30::float AS batter_vs_rp_ba_30,
    e.batter_vs_rp_slg_30::float AS batter_vs_rp_slg_30,
    e.batter_vs_rp_hr_rate_30::float AS batter_vs_rp_hr_rate_30,
    e.batter_vs_rp_k_rate_30::float AS batter_vs_rp_k_rate_30,
    h.batter_sc_barrel_rate::float AS batter_sc_barrel_rate,
    h.batter_sc_hard_hit_pct::float AS batter_sc_hard_hit_pct,
    h.batter_sc_avg_exit_velo::float AS batter_sc_avg_exit_velo,
    h.batter_sc_avg_launch_angle::float AS batter_sc_avg_launch_angle,
    h.batter_sc_sweet_spot_pct::float AS batter_sc_sweet_spot_pct,
    h.batter_sc_fb_pct::float AS batter_sc_fb_pct,
    h.batter_sc_gb_pct::float AS batter_sc_gb_pct,
    h.batter_sc_ld_pct::float AS batter_sc_ld_pct,
    h.batter_sc_xba::float AS batter_sc_xba,
    h.batter_sc_xslg::float AS batter_sc_xslg,
    h.batter_sc_xwoba::float AS batter_sc_xwoba,
    h.batter_sc_xiso::float AS batter_sc_xiso,
    h.batter_sc_pull_pct::float AS batter_sc_pull_pct,
    h.batter_sc_opposite_pct::float AS batter_sc_opposite_pct,
    h.batter_sc_popup_pct::float AS batter_sc_popup_pct,
    h.batter_sc_brl_pa::float AS batter_sc_brl_pa,
    h.batter_sprint_speed::float AS batter_sprint_speed,
    h.batter_disc_oz_swing_pct::float AS batter_disc_oz_swing_pct,
    h.batter_disc_iz_contact_pct::float AS batter_disc_iz_contact_pct,
    h.batter_disc_oz_contact_pct::float AS batter_disc_oz_contact_pct,
    h.batter_disc_whiff_pct::float AS batter_disc_whiff_pct,
    h.batter_disc_out_zone_pct::float AS batter_disc_out_zone_pct,
    h.batter_disc_k_pct::float AS batter_disc_k_pct,
    h.batter_disc_bb_pct::float AS batter_disc_bb_pct,
    h.opp_sp_sc_barrel_rate::float AS opp_sp_sc_barrel_rate,
    h.opp_sp_sc_hard_hit_pct::float AS opp_sp_sc_hard_hit_pct,
    h.opp_sp_sc_avg_exit_velo::float AS opp_sp_sc_avg_exit_velo,
    h.opp_sp_sc_avg_launch_angle::float AS opp_sp_sc_avg_launch_angle,
    h.opp_sp_sc_xba::float AS opp_sp_sc_xba,
    h.opp_sp_sc_xslg::float AS opp_sp_sc_xslg,
    h.opp_sp_sc_xwoba::float AS opp_sp_sc_xwoba,
    h.opp_sp_sc_xiso::float AS opp_sp_sc_xiso,
    h.opp_sp_fb_pct::float AS opp_sp_fb_pct,
    h.opp_sp_fb_hard_hit_pct::float AS opp_sp_fb_hard_hit_pct,
    h.opp_sp_fb_xwoba::float AS opp_sp_fb_xwoba,
    h.opp_sp_fb_run_value_per_100::float AS opp_sp_fb_run_value_per_100,
    h.opp_sp_fb_whiff_pct::float AS opp_sp_fb_whiff_pct,
    h.opp_sp_fb_k_pct::float AS opp_sp_fb_k_pct,
    h.opp_sp_si_pct::float AS opp_sp_si_pct,
    h.opp_sp_si_hard_hit_pct::float AS opp_sp_si_hard_hit_pct,
    h.opp_sp_si_xwoba::float AS opp_sp_si_xwoba,
    h.opp_sp_si_whiff_pct::float AS opp_sp_si_whiff_pct,
    h.opp_sp_si_k_pct::float AS opp_sp_si_k_pct,
    h.opp_sp_sl_pct::float AS opp_sp_sl_pct,
    h.opp_sp_sl_hard_hit_pct::float AS opp_sp_sl_hard_hit_pct,
    h.opp_sp_sl_xwoba::float AS opp_sp_sl_xwoba,
    h.opp_sp_sl_run_value_per_100::float AS opp_sp_sl_run_value_per_100,
    h.opp_sp_sl_whiff_pct::float AS opp_sp_sl_whiff_pct,
    h.opp_sp_sl_k_pct::float AS opp_sp_sl_k_pct,
    h.opp_sp_ch_pct::float AS opp_sp_ch_pct,
    h.opp_sp_ch_hard_hit_pct::float AS opp_sp_ch_hard_hit_pct,
    h.opp_sp_ch_xwoba::float AS opp_sp_ch_xwoba,
    h.opp_sp_ch_run_value_per_100::float AS opp_sp_ch_run_value_per_100,
    h.opp_sp_ch_whiff_pct::float AS opp_sp_ch_whiff_pct,
    h.opp_sp_ch_k_pct::float AS opp_sp_ch_k_pct,
    h.opp_sp_fastball_family_pct::float AS opp_sp_fastball_family_pct,
    h.opp_sp_pitch_diversity::float AS opp_sp_pitch_diversity,
    e.pinch_hit_risk::float AS pinch_hit_risk,
    COALESCE(h.actual_pa, e.actual_pa)::float AS actual_pa,
    e.actual_value::float AS actual_value,
    h.actual_hits::float AS component_hits,
    h.actual_singles::float AS component_singles,
    h.actual_doubles::float AS component_doubles,
    h.actual_triples::float AS component_triples,
    h.actual_home_runs::float AS component_home_runs,
    h.actual_total_bases::float AS component_total_bases,
    CASE WHEN e.won IS TRUE THEN 1 WHEN e.won IS FALSE THEN 0 ELSE NULL END AS target,
    COALESCE(e.push, false) AS push,
    e.profit_units::float AS profit_units,
    e.clv_price::float AS clv_price,
    CASE WHEN e.beat_clv_price IS TRUE THEN 1 WHEN e.beat_clv_price IS FALSE THEN 0 ELSE NULL END AS beat_clv_price
FROM features.mlb_prop_market_training_examples e
LEFT JOIN features.mlb_hitter_player_game_training h
  ON h.game_slug = e.game_slug
 AND h.player_id = e.player_id
WHERE e.game_date_et >= %(cutoff)s
  AND e.market IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
  AND e.side IN ('over','under')
  AND e.market_line IS NOT NULL
  AND e.model_prob_side IS NOT NULL
  AND e.actual_value IS NOT NULL
  AND e.won IS NOT NULL
ORDER BY e.game_date_et, e.replay_id
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


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def _load(cfg: DistributionConfig) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, cfg.lookback_days))
    with psycopg2.connect(cfg.pg_dsn) as conn:
        if not _table_exists(conn, "features", "mlb_prop_market_training_examples"):
            return pd.DataFrame()
        ensure_prop_market_training_schema(conn)
        df = _query_df(conn, SQL, {"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    numeric = [
        "market_line", "market_price", "model_prob_side", "market_prob_side",
        "pred_count", "pred_value", "projected_pa", "projected_bf",
        "projected_ip", "projected_pitch_count", "actual_pa", "actual_value", "target", "profit_units",
        "clv_price", "beat_clv_price",
        "same_book_pair_flag", "cross_book_pair_flag", "synthetic_pair_flag",
        "clean_market_pair_flag", "true_pair_flag",
        "component_hits", "component_singles", "component_doubles", "component_triples",
        "component_home_runs", "component_total_bases",
    ] + HITTER_NUMERIC + HITTER_OUTCOME_NUMERIC + PITCHER_NUMERIC
    for col in numeric:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in sorted(set(HITTER_CATEGORICAL + PITCHER_CATEGORICAL)):
        df[col] = df[col].fillna("unknown").astype(str)
    for col in ("pair_quality", "market_prob_source"):
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str)
    df = add_hitter_pa_v2_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df["push"] = df["push"].fillna(False).astype(bool)
    df = df.loc[~df["push"]].dropna(subset=["target", "model_prob_side", "market_line"])
    df["target"] = df["target"].astype(int)
    if {"run_id", "game_slug", "player_id", "market", "pred_count"}.issubset(df.columns):
        keys = ["run_id", "game_slug", "player_id"]
        sibling = (
            df.loc[df["market"].isin(["batter_hits", "batter_total_bases", "batter_home_runs"])]
            .groupby(keys + ["market"], dropna=False)["pred_count"]
            .mean()
            .unstack("market")
            .rename(columns={
                "batter_hits": "sibling_pred_hits",
                "batter_total_bases": "sibling_pred_total_bases",
                "batter_home_runs": "sibling_pred_home_runs",
            })
        )
        df = df.join(sibling, on=keys)
        for col in ["sibling_pred_hits", "sibling_pred_total_bases", "sibling_pred_home_runs"]:
            df[col] = pd.to_numeric(df.get(col), errors="coerce")
        actual_sibling = (
            df.loc[df["market"].isin(["batter_hits", "batter_total_bases", "batter_home_runs"])]
            .groupby(keys + ["market"], dropna=False)["actual_value"]
            .mean()
            .unstack("market")
            .rename(columns={
                "batter_hits": "actual_hits",
                "batter_total_bases": "actual_total_bases",
                "batter_home_runs": "actual_home_runs",
            })
        )
        df = df.join(actual_sibling, on=keys)
        for col in ["actual_hits", "actual_total_bases", "actual_home_runs"]:
            df[col] = pd.to_numeric(df.get(col), errors="coerce")
    return df


def _poisson_cdf(k: int, lam: float) -> float:
    lam = max(1e-6, min(40.0, float(lam)))
    term = math.exp(-lam)
    total = term
    for i in range(1, max(0, int(k)) + 1):
        term *= lam / i
        total += term
    return max(0.0, min(1.0, total))


def _poisson_over(line: float, lam: float) -> float:
    return 1.0 - _poisson_cdf(math.floor(float(line)), lam)


def _binom_over(line: float, n: int, p: float) -> float:
    n = max(1, min(8, int(n)))
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    threshold = math.floor(float(line))
    prob = 0.0
    for k in range(threshold + 1, n + 1):
        prob += math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
    return max(0.0, min(1.0, prob))


def _safe_float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _rate_like(value: Any, default: float) -> float:
    v = _safe_float(value)
    if v is None:
        return default
    if v > 1.0:
        v /= 100.0
    return _clamp(v, 0.0, 1.0)


def _walk_rate(row: pd.Series) -> float:
    sp_walk = _rate_like(row.get("opp_sp_bb_pct"), 0.085)
    hitter_k = _rate_like(row.get("batter_vs_hand_k_rate_10"), 0.22)
    rp_k = _rate_like(row.get("batter_vs_rp_k_rate_30"), hitter_k)
    # Use walks as opportunity drag, with mild shrinkage when the matchup is more K-heavy.
    return _clamp(0.70 * sp_walk + 0.30 * 0.085 - 0.04 * max(0.0, ((hitter_k + rp_k) / 2.0) - 0.24), 0.035, 0.160)


def _slot_bucket(row: pd.Series) -> str:
    slot = _safe_float(row.get("confirmed_batting_order"))
    if slot is None or slot < 1 or slot > 9:
        return "slot_missing"
    slot_i = int(round(slot))
    if slot_i <= 4:
        return "slot_top"
    if slot_i <= 6:
        return "slot_middle"
    return "slot_bottom"


def _power_bucket(expected_hits: float | None, expected_tb: float | None, expected_hr: float | None, pa: float | None) -> str:
    if pa is None or pa <= 0:
        return "power_unknown"
    hits = max(1e-6, float(expected_hits or 0.0))
    tb = max(hits, float(expected_tb or (hits * 1.45)))
    hr = max(0.0, float(expected_hr or 0.0))
    hr_rate = hr / max(pa, 1.0)
    tb_per_hit = tb / hits
    if hr_rate >= 0.060 or tb_per_hit >= 1.85:
        return "power_high"
    if hr_rate >= 0.035 or tb_per_hit >= 1.55:
        return "power_mid"
    return "power_low"


def _expected_hitter_counts(row: pd.Series) -> dict[str, float | None]:
    pa = _hitter_pa_for_distribution(row)
    if pa is None or float(pa) < 1.0:
        return {"pa": None, "hits": None, "tb": None, "hr": None}
    pa_f = float(pa)
    market = str(row.get("market") or "")
    current_pred = row.get("pred_count")
    if pd.isna(current_pred):
        current_pred = row.get("pred_value")

    hits = row.get("sibling_pred_hits")
    tb = row.get("sibling_pred_total_bases")
    hr = row.get("sibling_pred_home_runs")
    if market == "batter_hits" and pd.notna(current_pred):
        hits = current_pred
    elif market == "batter_total_bases" and pd.notna(current_pred):
        tb = current_pred
    elif market == "batter_home_runs" and pd.notna(current_pred):
        hr = current_pred
    if pd.isna(hits):
        hits = row.get("batter_vs_hand_hits_avg_10")
    if pd.isna(tb):
        tb = row.get("batter_vs_hand_tb_avg_10")
    if pd.isna(hr):
        hr = row.get("batter_vs_hand_hr_avg_10")

    if pd.notna(row.get("projected_pa")) and float(row["projected_pa"]) > 0:
        scale = max(0.65, min(1.35, pa_f / float(row["projected_pa"])))
        hits = None if pd.isna(hits) else float(hits) * scale
        tb = None if pd.isna(tb) else float(tb) * scale
        hr = None if pd.isna(hr) else float(hr) * scale
    hits_f = _safe_float(hits)
    tb_f = _safe_float(tb)
    hr_f = _safe_float(hr)
    if hits_f is None and tb_f is not None:
        hits_f = tb_f / 1.45
    if tb_f is None and hits_f is not None:
        tb_f = hits_f * 1.45
    if hr_f is None and tb_f is not None and hits_f is not None:
        hr_f = min(tb_f / 4.0, hits_f * 0.14)
    return {"pa": pa_f, "hits": hits_f, "tb": tb_f, "hr": hr_f}


def _shrunk_multiplier(actual_sum: float, expected_sum: float, exposure: float, shrink_exposure: float = 350.0) -> float:
    return _empirical_bayes_component_multiplier(
        actual_sum,
        expected_sum,
        exposure,
        shrink_exposure=shrink_exposure,
    )


def _empirical_bayes_component_multiplier(
    actual_sum: float,
    expected_sum: float,
    exposure: float,
    *,
    prior_events: float = 25.0,
    shrink_exposure: float = 500.0,
) -> float:
    """Shrink a component correction while allowing well-supported large bias."""
    if expected_sum <= 1e-9 or exposure <= 0:
        return 1.0
    posterior_ratio = (max(0.0, actual_sum) + prior_events) / (expected_sum + prior_events)
    exposure_weight = exposure / (exposure + shrink_exposure)
    shrunk = math.exp(exposure_weight * math.log(max(1e-6, posterior_ratio)))
    support = expected_sum / (expected_sum + 50.0)
    upper = 1.0 + 2.0 * support
    return _clamp(shrunk, 1.0 / upper, upper)


def _empty_outcome_accumulator() -> dict[str, float]:
    return {
        "rows": 0.0,
        "pa": 0.0,
        "pred_hits": 0.0,
        "actual_hits": 0.0,
        "pred_tb": 0.0,
        "actual_tb": 0.0,
        "pred_hr": 0.0,
        "actual_hr": 0.0,
        "pred_non_hr_extra": 0.0,
        "actual_non_hr_extra": 0.0,
    }


def _add_outcome_row(acc: dict[str, float], *, pa: float, pred_hits: float, actual_hits: float, pred_tb: float, actual_tb: float, pred_hr: float, actual_hr: float) -> None:
    acc["rows"] += 1.0
    acc["pa"] += max(0.0, pa)
    acc["pred_hits"] += max(0.0, pred_hits)
    acc["actual_hits"] += max(0.0, actual_hits)
    acc["pred_tb"] += max(0.0, pred_tb)
    acc["actual_tb"] += max(0.0, actual_tb)
    acc["pred_hr"] += max(0.0, pred_hr)
    acc["actual_hr"] += max(0.0, actual_hr)
    acc["pred_non_hr_extra"] += max(0.0, pred_tb - pred_hits - 3.0 * pred_hr)
    acc["actual_non_hr_extra"] += max(0.0, actual_tb - actual_hits - 3.0 * actual_hr)


def _finalize_outcome_accumulator(acc: dict[str, float], *, min_rows: int = 30) -> dict[str, Any] | None:
    rows = int(acc.get("rows") or 0)
    if rows < min_rows:
        return None
    pa = float(acc.get("pa") or 0.0)
    return {
        "rows": rows,
        "pa": pa,
        "hit_multiplier": _shrunk_multiplier(acc["actual_hits"], acc["pred_hits"], pa),
        "tb_multiplier": _shrunk_multiplier(acc["actual_tb"], acc["pred_tb"], pa),
        "hr_multiplier": _shrunk_multiplier(acc["actual_hr"], acc["pred_hr"], pa, shrink_exposure=500.0),
        "non_hr_extra_multiplier": _shrunk_multiplier(
            acc["actual_non_hr_extra"],
            acc["pred_non_hr_extra"],
            pa,
            shrink_exposure=500.0,
        ),
        "actual_hits_per_pa": acc["actual_hits"] / pa if pa > 0 else None,
        "pred_hits_per_pa": acc["pred_hits"] / pa if pa > 0 else None,
        "actual_tb_per_pa": acc["actual_tb"] / pa if pa > 0 else None,
        "pred_tb_per_pa": acc["pred_tb"] / pa if pa > 0 else None,
        "actual_hr_per_pa": acc["actual_hr"] / pa if pa > 0 else None,
        "pred_hr_per_pa": acc["pred_hr"] / pa if pa > 0 else None,
    }


def _fit_hitter_outcome_calibrators(scored_train: pd.DataFrame) -> dict[str, Any]:
    hitter = scored_train.loc[scored_train["market"].isin(["batter_hits", "batter_total_bases", "batter_home_runs"])].copy()
    if hitter.empty:
        return {}
    keys = ["run_id", "game_slug", "player_id"]
    if not set(keys).issubset(hitter.columns):
        return {}
    player_games = hitter.drop_duplicates(keys).copy()
    accs: dict[str, dict[str, float]] = {}
    for _, row in player_games.iterrows():
        counts = _expected_hitter_counts(row)
        pa = _safe_float(row.get("actual_pa"))
        pred_pa = _safe_float(counts.get("pa"))
        actual_hits = _safe_float(row.get("actual_hits"))
        actual_tb = _safe_float(row.get("actual_total_bases"))
        actual_hr = _safe_float(row.get("actual_home_runs"))
        pred_hits = _safe_float(counts.get("hits"))
        pred_tb = _safe_float(counts.get("tb"))
        pred_hr = _safe_float(counts.get("hr"))
        if (
            pa is None
            or pa <= 0
            or pred_pa is None
            or pred_hits is None
            or pred_tb is None
            or pred_hr is None
            or actual_hits is None
            or actual_tb is None
            or actual_hr is None
        ):
            continue
        pred_hits = _clamp(pred_hits, 1e-6, pred_pa * 0.70)
        pred_hr = _clamp(pred_hr, 0.0, min(pred_hits * 0.55, pred_pa * 0.20))
        pred_tb = _clamp(pred_tb, pred_hits, pred_pa * 2.25)
        actual_hits = _clamp(actual_hits, 0.0, pa)
        actual_hr = _clamp(actual_hr, 0.0, actual_hits)
        actual_tb = _clamp(actual_tb, actual_hits, pa * 4.0)
        slot = _slot_bucket(row)
        power = _power_bucket(pred_hits, pred_tb, pred_hr, pred_pa)
        for key in ("global", f"slot={slot}", f"power={power}", f"slot_power={slot}|{power}"):
            acc = accs.setdefault(key, _empty_outcome_accumulator())
            _add_outcome_row(
                acc,
                pa=pa,
                pred_hits=pred_hits,
                actual_hits=actual_hits,
                pred_tb=pred_tb,
                actual_tb=actual_tb,
                pred_hr=pred_hr,
                actual_hr=actual_hr,
            )
    groups: dict[str, Any] = {}
    for key, acc in accs.items():
        min_rows = 20 if key == "global" else (45 if key.startswith("slot_power=") else 30)
        rec = _finalize_outcome_accumulator(acc, min_rows=min_rows)
        if rec is not None:
            groups[key] = rec
    return {
        "method": "player_game_outcome_rate_shrinkage",
        "lookup_order": ["slot_power", "power", "slot", "global"],
        "groups": groups,
    }


def _lookup_hitter_outcome_calibrator(row: pd.Series, counts: dict[str, float | None], calibrators: dict[str, Any] | None) -> dict[str, Any] | None:
    groups = (calibrators or {}).get("groups") or {}
    if not groups:
        return None
    slot = _slot_bucket(row)
    power = _power_bucket(counts.get("hits"), counts.get("tb"), counts.get("hr"), counts.get("pa"))
    for key in (f"slot_power={slot}|{power}", f"power={power}", f"slot={slot}", "global"):
        rec = groups.get(key)
        if rec:
            return rec
    return None


def _apply_hitter_outcome_calibrator(
    row: pd.Series,
    counts: dict[str, float | None],
    calibrators: dict[str, Any] | None,
) -> dict[str, float | None]:
    out = dict(counts)
    rec = _lookup_hitter_outcome_calibrator(row, counts, calibrators)
    if not rec:
        return out
    pa = _safe_float(out.get("pa"))
    hits = _safe_float(out.get("hits"))
    tb = _safe_float(out.get("tb"))
    hr = _safe_float(out.get("hr"))
    if pa is None or hits is None or tb is None or hr is None:
        return out
    hit_mult = _safe_float(rec.get("hit_multiplier")) or 1.0
    hr_mult = _safe_float(rec.get("hr_multiplier")) or 1.0
    extra_mult = _safe_float(rec.get("non_hr_extra_multiplier")) or 1.0
    hits_new = _clamp(hits * hit_mult, 1e-6, pa * 0.70)
    hr_new = _clamp(hr * hr_mult, 0.0, min(hits_new * 0.55, pa * 0.20))
    non_hr_extra = max(0.0, tb - hits - 3.0 * hr) * extra_mult
    tb_new = _clamp(hits_new + 3.0 * hr_new + non_hr_extra, hits_new, pa * 2.25)
    out["hits"] = hits_new
    out["tb"] = tb_new
    out["hr"] = hr_new
    return out


def _pa_bucket(pa: float | None) -> str:
    if pa is None or pa <= 0:
        return "pa_unknown"
    if pa < 3.25:
        return "pa_low"
    if pa < 4.25:
        return "pa_mid"
    return "pa_high"


def _base_hitter_components(
    pa: float,
    *,
    expected_hits: float | None,
    expected_tb: float | None,
    expected_hr: float | None,
) -> dict[str, float]:
    n = max(1, min(7, int(round(float(pa)))))
    hits_guess = expected_hits
    tb_guess = expected_tb
    hr_guess = expected_hr
    if hits_guess is None and tb_guess is not None:
        hits_guess = tb_guess / 1.45
    if tb_guess is None and hits_guess is not None:
        tb_guess = hits_guess * 1.45
    hits = _clamp(float(hits_guess or 1e-6), 1e-6, float(n) * 0.64)
    tb = _clamp(float(tb_guess or (hits * 1.45)), hits, float(n) * 2.10)
    hr = hr_guess if hr_guess is not None else min(tb / 4.0, hits * 0.14)
    hr = _clamp(float(hr or 0.0), 0.0, min(hits * 0.50, float(n) * 0.18))

    non_hr_hits = max(0.0, hits - hr)
    non_hr_tb = _clamp(tb - 4.0 * hr, non_hr_hits, non_hr_hits * 3.0 if non_hr_hits > 0 else 0.0)
    extra_non_hr_bases = max(0.0, non_hr_tb - non_hr_hits)
    triples = _clamp(min(0.025 * n, non_hr_hits * 0.06, extra_non_hr_bases / 2.0), 0.0, non_hr_hits)
    doubles = _clamp(min(non_hr_hits - triples, extra_non_hr_bases - 2.0 * triples), 0.0, non_hr_hits - triples)
    singles = max(0.0, non_hr_hits - doubles - triples)
    return {
        "pa_events": float(n),
        "hits": hits,
        "tb": tb,
        "singles": singles,
        "doubles": doubles,
        "triples": triples,
        "hr": hr,
    }


def _component_probs_from_counts(components: dict[str, float], walk_rate: float) -> dict[str, float]:
    n = max(1, min(7, int(round(float(components.get("pa_events") or 1.0)))))

    p_walk = _clamp(walk_rate, 0.035, 0.160)
    p_single = max(0.0, float(components.get("singles") or 0.0) / n)
    p_double = max(0.0, float(components.get("doubles") or 0.0) / n)
    p_triple = max(0.0, float(components.get("triples") or 0.0) / n)
    p_hr = max(0.0, float(components.get("hr") or 0.0) / n)
    non_zero = p_walk + p_single + p_double + p_triple + p_hr
    if non_zero > 0.95:
        scale = 0.95 / non_zero
        p_walk *= scale
        p_single *= scale
        p_double *= scale
        p_triple *= scale
        p_hr *= scale
        non_zero = 0.95
    return {
        "pa_events": float(n),
        "p_out": 1.0 - non_zero,
        "p_walk": p_walk,
        "p_single": p_single,
        "p_double": p_double,
        "p_triple": p_triple,
        "p_hr": p_hr,
        "p_hit": p_single + p_double + p_triple + p_hr,
    }


def _empty_tb_structure_accumulator() -> dict[str, float]:
    return {
        "rows": 0.0,
        "pa": 0.0,
        "pred_singles": 0.0,
        "pred_doubles": 0.0,
        "pred_triples": 0.0,
        "pred_hr": 0.0,
        "pred_tb": 0.0,
        "actual_singles": 0.0,
        "actual_doubles": 0.0,
        "actual_triples": 0.0,
        "actual_hr": 0.0,
        "actual_tb": 0.0,
        "pred_zero_tb": 0.0,
        "actual_zero_tb": 0.0,
    }


def _add_tb_structure_row(
    acc: dict[str, float],
    *,
    pa: float,
    pred_components: dict[str, float],
    actual_singles: float,
    actual_doubles: float,
    actual_triples: float,
    actual_hr: float,
    actual_tb: float,
    walk_rate: float,
) -> None:
    acc["rows"] += 1.0
    acc["pa"] += max(0.0, pa)
    acc["pred_singles"] += max(0.0, float(pred_components.get("singles") or 0.0))
    acc["pred_doubles"] += max(0.0, float(pred_components.get("doubles") or 0.0))
    acc["pred_triples"] += max(0.0, float(pred_components.get("triples") or 0.0))
    acc["pred_hr"] += max(0.0, float(pred_components.get("hr") or 0.0))
    acc["pred_tb"] += max(0.0, float(pred_components.get("tb") or 0.0))
    acc["actual_singles"] += max(0.0, actual_singles)
    acc["actual_doubles"] += max(0.0, actual_doubles)
    acc["actual_triples"] += max(0.0, actual_triples)
    acc["actual_hr"] += max(0.0, actual_hr)
    acc["actual_tb"] += max(0.0, actual_tb)
    probs = _component_probs_from_counts(pred_components, walk_rate)
    zero_pa_prob = _clamp(probs["p_out"] + probs["p_walk"], 0.0, 1.0)
    acc["pred_zero_tb"] += zero_pa_prob ** int(probs["pa_events"])
    acc["actual_zero_tb"] += 1.0 if actual_tb <= 0 else 0.0


def _finalize_tb_structure_accumulator(acc: dict[str, float], *, min_rows: int = 30) -> dict[str, Any] | None:
    rows = int(acc.get("rows") or 0)
    if rows < min_rows:
        return None
    pa = float(acc.get("pa") or 0.0)
    single_mult = _empirical_bayes_component_multiplier(
        acc["actual_singles"], acc["pred_singles"], pa, shrink_exposure=550.0
    )
    double_mult = _empirical_bayes_component_multiplier(
        acc["actual_doubles"], acc["pred_doubles"], pa, shrink_exposure=700.0
    )
    triple_mult = _empirical_bayes_component_multiplier(
        acc["actual_triples"], acc["pred_triples"], pa, prior_events=35.0, shrink_exposure=1400.0
    )
    hr_mult = _empirical_bayes_component_multiplier(
        acc["actual_hr"], acc["pred_hr"], pa, shrink_exposure=900.0
    )
    return {
        "rows": rows,
        "pa": pa,
        "single_multiplier": single_mult,
        "double_multiplier": double_mult,
        "triple_multiplier": triple_mult,
        "hr_multiplier": hr_mult,
        "tb_multiplier": _empirical_bayes_component_multiplier(
            acc["actual_tb"], acc["pred_tb"], pa, shrink_exposure=550.0
        ),
        "actual_zero_tb_rate": acc["actual_zero_tb"] / rows if rows else None,
        "pred_zero_tb_rate": acc["pred_zero_tb"] / rows if rows else None,
        "actual_single_per_pa": acc["actual_singles"] / pa if pa > 0 else None,
        "pred_single_per_pa": acc["pred_singles"] / pa if pa > 0 else None,
        "actual_double_per_pa": acc["actual_doubles"] / pa if pa > 0 else None,
        "pred_double_per_pa": acc["pred_doubles"] / pa if pa > 0 else None,
        "actual_triple_per_pa": acc["actual_triples"] / pa if pa > 0 else None,
        "pred_triple_per_pa": acc["pred_triples"] / pa if pa > 0 else None,
        "actual_hr_per_pa": acc["actual_hr"] / pa if pa > 0 else None,
        "pred_hr_per_pa": acc["pred_hr"] / pa if pa > 0 else None,
    }


def _fit_tb_structure_calibrators(scored_train: pd.DataFrame) -> dict[str, Any]:
    hitter = scored_train.loc[
        scored_train["market"].isin(["batter_hits", "batter_total_bases", "batter_home_runs"])
    ].copy()
    keys = ["run_id", "game_slug", "player_id"]
    required = {
        *keys,
        "component_singles",
        "component_doubles",
        "component_triples",
        "component_home_runs",
        "component_total_bases",
        "actual_pa",
    }
    if hitter.empty or not required.issubset(hitter.columns):
        return {}
    player_games = hitter.drop_duplicates(keys).copy()
    accs: dict[str, dict[str, float]] = {}
    for _, row in player_games.iterrows():
        counts = _expected_hitter_counts(row)
        pa = _safe_float(row.get("actual_pa"))
        pred_pa = _safe_float(counts.get("pa"))
        pred_hits = _safe_float(counts.get("hits"))
        pred_tb = _safe_float(counts.get("tb"))
        pred_hr = _safe_float(counts.get("hr"))
        actual_singles = _safe_float(row.get("component_singles"))
        actual_doubles = _safe_float(row.get("component_doubles"))
        actual_triples = _safe_float(row.get("component_triples"))
        actual_hr = _safe_float(row.get("component_home_runs"))
        actual_tb = _safe_float(row.get("component_total_bases"))
        if (
            pa is None
            or pa <= 0
            or pred_pa is None
            or pred_hits is None
            or pred_tb is None
            or pred_hr is None
            or actual_singles is None
            or actual_doubles is None
            or actual_triples is None
            or actual_hr is None
            or actual_tb is None
        ):
            continue
        pred_components = _base_hitter_components(
            pred_pa,
            expected_hits=pred_hits,
            expected_tb=pred_tb,
            expected_hr=pred_hr,
        )
        slot = _slot_bucket(row)
        power = _power_bucket(pred_hits, pred_tb, pred_hr, pred_pa)
        pa_group = _pa_bucket(pred_pa)
        for key in (
            "global",
            f"slot={slot}",
            f"power={power}",
            f"pa={pa_group}",
            f"slot_power={slot}|{power}",
            f"pa_power={pa_group}|{power}",
        ):
            acc = accs.setdefault(key, _empty_tb_structure_accumulator())
            _add_tb_structure_row(
                acc,
                pa=pa,
                pred_components=pred_components,
                actual_singles=actual_singles,
                actual_doubles=actual_doubles,
                actual_triples=actual_triples,
                actual_hr=actual_hr,
                actual_tb=actual_tb,
                walk_rate=_walk_rate(row),
            )
    groups: dict[str, Any] = {}
    for key, acc in accs.items():
        if key == "global":
            min_rows = 20
        elif key.startswith("slot_power=") or key.startswith("pa_power="):
            min_rows = 55
        else:
            min_rows = 35
        rec = _finalize_tb_structure_accumulator(acc, min_rows=min_rows)
        if rec is not None:
            groups[key] = rec
    return {
        "method": "player_game_component_shrinkage",
        "mean_policy": "preserve_source_expected_tb",
        "lookup_order": ["slot_power", "pa_power", "power", "pa", "slot", "global"],
        "groups": groups,
    }


def _lookup_tb_structure_calibrator(
    row: pd.Series,
    components: dict[str, float],
    calibrators: dict[str, Any] | None,
) -> dict[str, Any] | None:
    groups = (calibrators or {}).get("groups") or {}
    if not groups:
        return None
    pa = _safe_float(components.get("pa_events"))
    hits = _safe_float(components.get("hits"))
    tb = _safe_float(components.get("tb"))
    hr = _safe_float(components.get("hr"))
    slot = _slot_bucket(row)
    power = _power_bucket(hits, tb, hr, pa)
    pa_group = _pa_bucket(pa)
    for key in (
        f"slot_power={slot}|{power}",
        f"pa_power={pa_group}|{power}",
        f"power={power}",
        f"pa={pa_group}",
        f"slot={slot}",
        "global",
    ):
        rec = groups.get(key)
        if rec:
            return rec
    return None


def _apply_tb_structure_calibrator(
    row: pd.Series,
    components: dict[str, float],
    calibrators: dict[str, Any] | None,
) -> dict[str, float]:
    rec = _lookup_tb_structure_calibrator(row, components, calibrators)
    if not rec:
        return components
    out = dict(components)
    n = max(1.0, float(out.get("pa_events") or 1.0))
    target_tb = max(1e-6, float(out.get("tb") or 1e-6))
    singles = max(0.0, float(out.get("singles") or 0.0) * (_safe_float(rec.get("single_multiplier")) or 1.0))
    doubles = max(0.0, float(out.get("doubles") or 0.0) * (_safe_float(rec.get("double_multiplier")) or 1.0))
    triples = max(0.0, float(out.get("triples") or 0.0) * (_safe_float(rec.get("triple_multiplier")) or 1.0))
    hr = max(0.0, float(out.get("hr") or 0.0) * (_safe_float(rec.get("hr_multiplier")) or 1.0))
    total_hits = singles + doubles + triples + hr
    hit_cap = n * 0.70
    if total_hits > hit_cap and total_hits > 0:
        scale = hit_cap / total_hits
        singles *= scale
        doubles *= scale
        triples *= scale
        hr *= scale
        total_hits = hit_cap
    if hr > min(total_hits * 0.55, n * 0.20):
        hr = min(total_hits * 0.55, n * 0.20)
    raw_tb = singles + 2.0 * doubles + 3.0 * triples + 4.0 * hr
    if raw_tb > 1e-9:
        # Preserve the source model's expected TB mean and only repair the
        # component shape.  Prior runs showed that broad TB inflation worsened
        # holdout Brier even when it fixed historical doubles undercounting.
        mean_scale = target_tb / raw_tb
        singles *= mean_scale
        doubles *= mean_scale
        triples *= mean_scale
        hr *= mean_scale
    out.update({
        "singles": singles,
        "doubles": doubles,
        "triples": triples,
        "hr": hr,
        "hits": singles + doubles + triples + hr,
        "tb": singles + 2.0 * doubles + 3.0 * triples + 4.0 * hr,
    })
    return out


def _per_pa_hitter_outcomes(
    pa: float,
    *,
    expected_hits: float | None,
    expected_tb: float | None,
    expected_hr: float | None,
    walk_rate: float,
) -> dict[str, float]:
    components = _base_hitter_components(
        pa,
        expected_hits=expected_hits,
        expected_tb=expected_tb,
        expected_hr=expected_hr,
    )
    return _component_probs_from_counts(components, walk_rate)


def _compound_tb_over_from_probs(line: float, probs: dict[str, float]) -> float:
    n = int(probs["pa_events"])
    per_pa = {
        0: probs["p_out"] + probs["p_walk"],
        1: probs["p_single"],
        2: probs["p_double"],
        3: probs["p_triple"],
        4: probs["p_hr"],
    }
    dist = {0: 1.0}
    for _ in range(n):
        nxt: dict[int, float] = {}
        for current, p_current in dist.items():
            for add, p_add in per_pa.items():
                nxt[current + add] = nxt.get(current + add, 0.0) + p_current * p_add
        dist = nxt
    threshold = math.floor(float(line))
    return max(0.0, min(1.0, sum(prob for total, prob in dist.items() if total > threshold)))


def _direct_event_probs(row: pd.Series, pa: float) -> dict[str, float] | None:
    vals: dict[str, float] = {}
    for cls in EVENT_CLASSES:
        value = _safe_float(row.get(f"p_event_{cls}"))
        if value is None:
            return None
        vals[cls] = max(0.0, value)
    total = sum(vals.values())
    if total <= 0:
        return None
    vals = {cls: value / total for cls, value in vals.items()}
    n = max(1, min(7, int(round(float(pa)))))
    return {
        "pa_events": float(n),
        "p_out": vals["out"],
        "p_walk": vals["walk"],
        "p_single": vals["single"],
        "p_double": vals["double"],
        "p_triple": vals["triple"],
        "p_hr": vals["hr"],
        "p_hit": vals["single"] + vals["double"] + vals["triple"] + vals["hr"],
    }


def _direct_event_components(
    pa: float,
    probs: dict[str, float],
    *,
    expected_hits: float | None,
    expected_tb: float | None,
    expected_hr: float | None,
) -> dict[str, float]:
    n = max(1, min(7, int(round(float(pa)))))
    direct_single = n * float(probs.get("p_single") or 0.0)
    direct_double = n * float(probs.get("p_double") or 0.0)
    direct_triple = n * float(probs.get("p_triple") or 0.0)
    direct_hr = n * float(probs.get("p_hr") or 0.0)
    direct_hits = max(1e-6, direct_single + direct_double + direct_triple + direct_hr)
    direct_tb = max(direct_hits, direct_single + 2.0 * direct_double + 3.0 * direct_triple + 4.0 * direct_hr)

    target_hits = _safe_float(expected_hits)
    target_tb = _safe_float(expected_tb)
    target_hr = _safe_float(expected_hr)
    if target_hits is None and target_tb is not None:
        target_hits = target_tb * min(1.0, direct_hits / direct_tb)
    if target_tb is None and target_hits is not None:
        target_tb = target_hits * max(1.0, direct_tb / direct_hits)
    if target_hits is None:
        target_hits = direct_hits
    if target_tb is None:
        target_tb = direct_tb
    if target_hr is None:
        target_hr = direct_hr * max(0.35, min(2.25, target_hits / direct_hits))

    target_hits = _clamp(target_hits, 1e-6, n * 0.70)
    target_hr = _clamp(target_hr, 0.0, min(target_hits * 0.55, n * 0.20))
    target_tb = _clamp(target_tb, target_hits, n * 2.25)

    non_hr_hits = max(0.0, target_hits - target_hr)
    non_hr_tb = _clamp(target_tb - 4.0 * target_hr, non_hr_hits, non_hr_hits * 3.0 if non_hr_hits > 0 else 0.0)
    extra_non_hr = max(0.0, non_hr_tb - non_hr_hits)

    double_extra = max(0.0, direct_double)
    triple_extra = max(0.0, 2.0 * direct_triple)
    triple_extra_share = triple_extra / (double_extra + triple_extra) if (double_extra + triple_extra) > 1e-9 else 0.08
    triples = min(non_hr_hits, extra_non_hr * triple_extra_share / 2.0)
    doubles = min(max(0.0, non_hr_hits - triples), max(0.0, extra_non_hr - 2.0 * triples))
    singles = max(0.0, non_hr_hits - doubles - triples)
    return {
        "pa_events": float(n),
        "hits": singles + doubles + triples + target_hr,
        "tb": singles + 2.0 * doubles + 3.0 * triples + 4.0 * target_hr,
        "singles": singles,
        "doubles": doubles,
        "triples": triples,
        "hr": target_hr,
    }


def _pmf_over(line: float, pmf: dict[int, float]) -> float:
    return _clamp(sum(float(probability) for count, probability in pmf.items() if float(count) > float(line)), 0.0, 1.0)


def _compound_tb_over(
    line: float,
    pa: float,
    expected_tb: float | None,
    expected_hits: float | None,
    expected_hr: float | None,
    walk_rate: float,
    tb_structure: dict[str, Any] | None = None,
    row: pd.Series | None = None,
) -> float:
    components = _base_hitter_components(
        pa,
        expected_hits=expected_hits,
        expected_tb=expected_tb,
        expected_hr=expected_hr,
    )
    if row is not None:
        components = _apply_tb_structure_calibrator(row, components, tb_structure)
    probs = _component_probs_from_counts(components, walk_rate)
    return _compound_tb_over_from_probs(line, probs)


def _hitter_hits_over(line: float, pa: float, expected_hits: float, expected_tb: float | None, expected_hr: float | None, walk_rate: float) -> float:
    probs = _per_pa_hitter_outcomes(
        pa,
        expected_hits=expected_hits,
        expected_tb=expected_tb,
        expected_hr=expected_hr,
        walk_rate=walk_rate,
    )
    return _binom_over(line, int(probs["pa_events"]), probs["p_hit"])


def _hitter_hr_over(line: float, pa: float, expected_hr: float, expected_hits: float | None, expected_tb: float | None, walk_rate: float) -> float:
    probs = _per_pa_hitter_outcomes(
        pa,
        expected_hits=expected_hits,
        expected_tb=expected_tb,
        expected_hr=expected_hr,
        walk_rate=walk_rate,
    )
    return _binom_over(line, int(probs["pa_events"]), probs["p_hr"])


def _hitter_pa_for_distribution(row: pd.Series) -> float | None:
    pa = row.get("opp_model_pa") if pd.notna(row.get("opp_model_pa")) else row.get("projected_pa")
    if pd.isna(pa):
        return None
    pa_f = max(0.5, min(7.0, float(pa)))
    risks = []
    if pd.notna(row.get("opp_model_low_pa")):
        risks.append(float(row["opp_model_low_pa"]))
    if pd.notna(row.get("pinch_hit_risk")):
        risks.append(float(row["pinch_hit_risk"]))
    if risks:
        risk = max(0.0, min(1.0, float(np.nanmax(risks))))
        pa_f *= 1.0 - min(0.30, 0.22 * risk)
    return max(0.5, min(7.0, pa_f))


def _load_opportunity_models(cfg: DistributionConfig) -> dict[str, Any]:
    path = cfg.model_dir / cfg.opportunity_file
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_opportunity_runtime(cfg: DistributionConfig) -> dict[str, Any]:
    if joblib is None:
        return {}
    path = cfg.model_dir / "prop_opportunity_models.joblib"
    if not path.exists():
        return {}
    try:
        return joblib.load(path)
    except Exception:
        return {}


def _score_opportunity(df: pd.DataFrame, cfg: DistributionConfig) -> pd.DataFrame:
    payload = _load_opportunity_models(cfg)
    models = payload.get("models") or {}
    out = df.copy()
    out["opp_model_pa"] = np.nan
    out["opp_model_low_pa"] = np.nan
    out["opp_model_bf"] = np.nan
    out["opp_model_ip"] = np.nan
    out["opp_model_pitch_count"] = np.nan
    hitter_mask = out["market"].isin(["batter_hits", "batter_total_bases", "batter_home_runs"])
    pitcher_mask = out["market"] == "pitcher_strikeouts"
    if hitter_mask.any():
        hitter_rows = out.loc[hitter_mask]
        rec = models.get("hitter_pa") or {}
        if rec.get("status") == "trained" and rec.get("use_for_distribution"):
            out.loc[hitter_mask, "opp_model_pa"] = np.clip(_score_opportunity_linear(hitter_rows, rec), 0.0, 7.0)
        rec = models.get("hitter_low_pa") or {}
        if rec.get("status") == "trained":
            out.loc[hitter_mask, "opp_model_low_pa"] = np.clip(_score_opportunity_linear(hitter_rows, rec), 1e-6, 1 - 1e-6)
    if pitcher_mask.any():
        pitcher_rows = out.loc[pitcher_mask]
        joint = models.get("pitcher_joint_opportunity") or {}
        runtime = _load_opportunity_runtime(cfg)
        if joint.get("use_for_distribution") and runtime.get("use_for_distribution"):
            try:
                keys = ["game_slug", "player_id"]
                unique = pitcher_rows.drop_duplicates(keys).copy()
                if pd.to_numeric(unique.get("actual_bf"), errors="coerce").notna().any():
                    unique = add_pitcher_history_features(unique)
                else:
                    unique = add_pitcher_history_features(
                        unique,
                        prior_state=runtime.get("player_history_state") or {},
                    )
                history_cols = list(runtime.get("numeric_features") or [])
                lookup = unique[keys + [c for c in history_cols if c in unique.columns]].drop_duplicates(keys)
                source = pitcher_rows.reset_index(names="_source_index").merge(lookup, on=keys, how="left", suffixes=("", "_history"))
                source = source.set_index("_source_index")
                for col in history_cols:
                    history_col = f"{col}_history"
                    if history_col in source:
                        source[col] = pd.to_numeric(source[history_col], errors="coerce").combine_first(
                            pd.to_numeric(source.get(col), errors="coerce")
                        )
                feature_cols = history_cols + list(runtime.get("categorical_features") or [])
                target_map = {
                    "opp_model_bf": ("bf", 40.0),
                    "opp_model_pitch_count": ("pitch_count", 130.0),
                    "opp_model_ip": ("innings", 9.0),
                }
                for out_col, (target, hi) in target_map.items():
                    target_runtime = ((runtime.get("models") or {}).get(target) or {})
                    model = target_runtime.get("mean")
                    if model is not None:
                        baseline = pd.to_numeric(
                            source.get(target_runtime.get("baseline_feature")), errors="coerce"
                        )
                        out.loc[source.index, out_col] = np.clip(
                            baseline + model.predict(source[feature_cols]), 0.0, hi
                        )
            except Exception:
                pass
        for out_col, model_key, hi in (
            ("opp_model_bf", "pitcher_bf", 40.0),
            ("opp_model_ip", "pitcher_ip", 9.0),
            ("opp_model_pitch_count", "pitcher_pitch_count_proxy", 130.0),
        ):
            rec = models.get(model_key) or {}
            if out.loc[pitcher_mask, out_col].notna().any():
                continue
            if rec.get("status") == "trained" and rec.get("use_for_distribution"):
                out.loc[pitcher_mask, out_col] = np.clip(_score_opportunity_linear(pitcher_rows, rec), 0.0, hi)
    return out


def _load_hitter_event_artifact(cfg: DistributionConfig) -> dict[str, Any]:
    if joblib is None:
        return {}
    path = cfg.model_dir / "hitter_player_game_outcome_models.joblib"
    if not path.exists():
        return {}
    try:
        artifact = joblib.load(path)
    except Exception:
        return {}
    models = artifact.get("models") or {}
    if (
        models.get("event_outcome_model") is None
        and not models.get("event_binary_models")
        and not models.get("event_hierarchical_models")
    ):
        return {}
    recommendation = artifact.get("recommendation") or {}
    artifact["production_eligible"] = bool(recommendation.get("passes_basic_gate"))
    artifact["status"] = "loaded" if artifact["production_eligible"] else "diagnostic_candidate"
    return artifact


def _event_model_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact:
        return {"status": "missing"}
    models = (artifact.get("models") or {})
    active = artifact.get("active_event_model") or "linear_multinomial"
    if active == "hierarchical_conditional_lgbm":
        model = models.get("event_hierarchical_models")
    elif active == "boosted_binary_calibrated":
        model = models.get("event_binary_models")
    else:
        model = models.get("event_outcome_model")
    return {
        "status": "loaded" if model is not None else "missing",
        "method": active,
        "trained_at_utc": artifact.get("trained_at_utc"),
        "event_classes": list(artifact.get("event_classes") or EVENT_CLASSES),
        "recommendation": artifact.get("recommendation") or {},
        "production_eligible": bool(artifact.get("production_eligible")),
        "pa_uncertainty": artifact.get("pa_uncertainty") or {},
        "player_prior_players": len(artifact.get("player_prior_state") or {}),
        "tb_state_distribution": (
            ((artifact.get("metrics") or {}).get("direct_event_model") or {}).get("tb_state_distribution") or {}
        ),
        "tb_state_residual": (
            ((artifact.get("metrics") or {}).get("direct_event_model") or {}).get("tb_state_residual") or {}
        ),
        "tb_state_model_kind": str(models.get("tb_state_model_kind") or "convolution"),
        "two_part_pa": ((artifact.get("metrics") or {}).get("pa_model") or {}).get("two_part") or {},
    }


def _score_hitter_event_model(df: pd.DataFrame, outcome_calibrators: dict[str, Any] | None) -> pd.DataFrame:
    out = df.copy()
    for cls in EVENT_CLASSES:
        out[f"p_event_{cls}"] = np.nan
    out["p_event_hr_any"] = np.nan
    out["p_event_low_pa"] = np.nan
    out["p_event_normal_pa"] = np.nan
    for state in TB_STATE_NAMES:
        out[f"p_tb_state_{state}"] = np.nan
    artifact = (outcome_calibrators or {}).get("event_model") if isinstance(outcome_calibrators, dict) else None
    models = ((artifact or {}).get("models") or {})
    active = (artifact or {}).get("active_event_model") or "linear_multinomial"
    model = models.get("event_outcome_model")
    binary_models = models.get("event_binary_models") or {}
    hierarchical_models = models.get("event_hierarchical_models") or {}
    hr_any_model = models.get("hr_any_model")
    pa_low_model = models.get("pa_low_model")
    pa_normal_model = models.get("pa_normal_model")
    use_two_part_pa = bool(models.get("pa_two_part_use") and pa_low_model is not None and pa_normal_model is not None)
    tb_state_models = models.get("tb_state_models") or {}
    tb_state_model_kind = str(models.get("tb_state_model_kind") or "one_vs_rest")
    if model is None and not binary_models and not hierarchical_models:
        return out
    hitter_mask = out["market"].isin(["batter_hits", "batter_total_bases", "batter_home_runs"])
    if not hitter_mask.any():
        return out
    numeric = list((artifact or {}).get("numeric_features") or HITTER_OUTCOME_NUMERIC)
    categorical = list((artifact or {}).get("categorical_features") or HITTER_OUTCOME_CATEGORICAL)
    source = out.loc[hitter_mask].copy()
    if "opp_model_pa" in source:
        opp_pa = pd.to_numeric(source["opp_model_pa"], errors="coerce")
        source["projected_pa"] = pd.to_numeric(source.get("projected_pa"), errors="coerce").where(opp_pa.isna(), opp_pa)
    try:
        source = apply_player_prior_state(source, (artifact or {}).get("player_prior_state") or {})
        features = prepare_hitter_outcome_features(source, numeric, categorical)
        X = features[numeric + categorical]

        if use_two_part_pa:
            low_prob = np.clip(pa_low_model.predict_proba(X)[:, 1], 1e-5, 1.0 - 1e-5)
            normal_pa = np.clip(pa_normal_model.predict(X), 3.0, 7.0)
            out.loc[source.index, "p_event_low_pa"] = low_prob
            out.loc[source.index, "p_event_normal_pa"] = normal_pa
            out.loc[source.index, "opp_model_low_pa"] = low_prob
            out.loc[source.index, "opp_model_normal_pa"] = normal_pa
            low_states = ((((artifact or {}).get("pa_uncertainty") or {}).get("global") or {}).get("low_pa_state_probs")) or {"0": 0.05, "1": 0.20, "2": 0.75}
            low_total = sum(float(low_states.get(str(n), 0.0)) for n in range(3)) or 1.0
            low_mean = sum(n * float(low_states.get(str(n), 0.0)) for n in range(3)) / low_total
            out.loc[source.index, "opp_model_pa"] = low_prob * low_mean + (1.0 - low_prob) * normal_pa

        expected_tb_heads = 4 if tb_state_model_kind == "hierarchical_hr_tail" else len(TB_STATE_NAMES)
        if len(tb_state_models) == expected_tb_heads:
            if tb_state_model_kind == "hierarchical_hr_tail":
                state_probs = _predict_hierarchical_tb_state_models(
                    tb_state_models,
                    source,
                    numeric,
                    categorical,
                )
            else:
                state_probs = pd.DataFrame(0.0, index=source.index, columns=TB_STATE_NAMES)
                for state, state_model in tb_state_models.items():
                    state_probs[state] = np.clip(state_model.predict_proba(X)[:, 1], 1e-8, 1.0 - 1e-8)
                state_probs = state_probs.div(state_probs.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(1.0 / len(TB_STATE_NAMES))
            state_probs = _apply_tb_hr_tail_logit_offset(
                state_probs,
                float(models.get("tb_state_hr_tail_logit_offset") or 0.0),
            )
            for state in TB_STATE_NAMES:
                out.loc[source.index, f"p_tb_state_{state}"] = state_probs[state]

        def score_hr_any() -> None:
            if hr_any_model is None:
                return
            try:
                out.loc[source.index, "p_event_hr_any"] = np.clip(hr_any_model.predict_proba(X)[:, 1], 1e-6, 1.0 - 1e-6)
            except Exception:
                return

        if active == "hierarchical_conditional_lgbm" and hierarchical_models:
            hierarchical = _predict_hierarchical_event_probabilities(
                hierarchical_models,
                source,
                numeric,
                categorical,
            )
            for cls in EVENT_CLASSES:
                out.loc[hierarchical.index, f"p_event_{cls}"] = hierarchical[f"p_{cls}"]
            score_hr_any()
            return out
        if active == "boosted_binary_calibrated" and binary_models:
            raw_probs = pd.DataFrame(0.0, index=source.index, columns=[f"p_event_{cls}" for cls in EVENT_CLASSES])
            for cls, cls_model in binary_models.items():
                if cls in EVENT_CLASSES:
                    raw_probs[f"p_event_{cls}"] = cls_model.predict_proba(X)[:, 1]
            row_sum = raw_probs.sum(axis=1).replace(0.0, np.nan)
            probs = raw_probs.div(row_sum, axis=0).fillna(1.0 / float(len(EVENT_CLASSES)))
            for col in probs.columns:
                out.loc[probs.index, col] = probs[col]
            score_hr_any()
            return out
        raw = model.predict_proba(features[numeric + categorical])
    except Exception:
        return out
    probs = pd.DataFrame(0.0, index=source.index, columns=[f"p_event_{cls}" for cls in EVENT_CLASSES])
    for i, cls in enumerate(model.classes_):
        if cls in EVENT_CLASSES:
            probs[f"p_event_{cls}"] = raw[:, i]
    row_sum = probs.sum(axis=1).replace(0.0, np.nan)
    probs = probs.div(row_sum, axis=0).fillna(np.nan)
    for col in probs.columns:
        out.loc[probs.index, col] = probs[col]
    if hr_any_model is not None:
        try:
            out.loc[source.index, "p_event_hr_any"] = np.clip(hr_any_model.predict_proba(features[numeric + categorical])[:, 1], 1e-6, 1.0 - 1e-6)
        except Exception:
            pass
    return out


def _opp_adjusted_mean(row: pd.Series, base_mean: float) -> float:
    market = str(row.get("market") or "")
    if market in {"batter_hits", "batter_total_bases", "batter_home_runs"}:
        old_pa = row.get("projected_pa")
        new_pa = row.get("opp_model_pa")
        risk_adj = 1.0
        if pd.notna(row.get("opp_model_low_pa")) or pd.notna(row.get("pinch_hit_risk")):
            risks = [
                float(v) for v in (row.get("opp_model_low_pa"), row.get("pinch_hit_risk"))
                if pd.notna(v)
            ]
            if risks:
                risk_adj = 1.0 - min(0.30, 0.22 * max(0.0, min(1.0, max(risks))))
        if pd.notna(old_pa) and pd.notna(new_pa) and float(old_pa) > 0:
            ratio = max(0.65, min(1.35, float(new_pa) / float(old_pa)))
            return max(1e-6, base_mean * ratio * risk_adj)
        return max(1e-6, base_mean * risk_adj)
    if market == "pitcher_strikeouts":
        ratios = []
        if pd.notna(row.get("projected_bf")) and pd.notna(row.get("opp_model_bf")) and float(row["projected_bf"]) > 0:
            ratios.append(float(row["opp_model_bf"]) / float(row["projected_bf"]))
        if pd.notna(row.get("projected_pitch_count")) and pd.notna(row.get("opp_model_pitch_count")) and float(row["projected_pitch_count"]) > 0:
            ratios.append(float(row["opp_model_pitch_count"]) / float(row["projected_pitch_count"]))
        if ratios:
            ratio = max(0.65, min(1.35, float(np.mean(ratios))))
            return max(1e-6, base_mean * ratio)
    return base_mean


def _tb_state_name(total_bases: int, had_hr: bool) -> str:
    if total_bases <= 0:
        return "tb_0"
    if total_bases == 1:
        return "tb_1"
    if total_bases <= 3:
        return "tb_2_3"
    return "tb_4_plus_hr" if had_hr else "tb_4_plus_non_hr"


def _blend_tb_state_curve(curve: dict[str, Any], row: pd.Series, alpha: float) -> dict[str, Any]:
    if alpha <= 0.0 or not curve.get("tb_joint"):
        return curve
    direct = np.asarray([_safe_float(row.get(f"p_tb_state_{state}")) for state in TB_STATE_NAMES], dtype=object)
    if any(value is None for value in direct):
        return curve
    direct_f = np.asarray(direct, dtype=float)
    direct_total = float(direct_f.sum())
    if direct_total <= 0:
        return curve
    direct_f /= direct_total
    base_states = curve.get("tb_states") or {}
    base = np.asarray([float(base_states.get(state, 0.0)) for state in TB_STATE_NAMES], dtype=float)
    base_total = float(base.sum())
    if base_total <= 0:
        return curve
    base /= base_total
    desired = (1.0 - alpha) * base + alpha * direct_f
    desired /= desired.sum()
    factors = {
        state: (float(desired[i]) / float(base[i])) if base[i] > 1e-12 else 0.0
        for i, state in enumerate(TB_STATE_NAMES)
    }
    joint = {
        (int(tb), bool(had_hr)): float(probability) * factors[_tb_state_name(int(tb), bool(had_hr))]
        for (tb, had_hr), probability in curve["tb_joint"].items()
    }
    total = sum(joint.values()) or 1.0
    joint = {key: value / total for key, value in joint.items()}
    tb_pmf: dict[int, float] = {}
    states = {state: 0.0 for state in TB_STATE_NAMES}
    for (tb, had_hr), probability in joint.items():
        tb_pmf[tb] = tb_pmf.get(tb, 0.0) + probability
        states[_tb_state_name(tb, had_hr)] += probability
    out = dict(curve)
    out.update({"tb_joint": joint, "tb_pmf": tb_pmf, "tb_states": states})
    return out


def _tb_state_over_probability(line: float, curve: dict[str, Any]) -> float:
    """Price state-aligned TB lines directly; preserve the conditional PMF within broad states."""
    states = curve.get("tb_states") or {}
    if abs(line - 0.5) <= 1e-9:
        return _clamp(1.0 - float(states.get("tb_0", 0.0)), 1e-6, 1.0 - 1e-6)
    if abs(line - 1.5) <= 1e-9:
        return _clamp(
            float(states.get("tb_2_3", 0.0))
            + float(states.get("tb_4_plus_hr", 0.0))
            + float(states.get("tb_4_plus_non_hr", 0.0)),
            1e-6,
            1.0 - 1e-6,
        )
    if abs(line - 3.5) <= 1e-9:
        return _clamp(
            float(states.get("tb_4_plus_hr", 0.0)) + float(states.get("tb_4_plus_non_hr", 0.0)),
            1e-6,
            1.0 - 1e-6,
        )
    return _pmf_over(line, curve.get("tb_pmf") or {})


def _distribution_over(
    row: pd.Series,
    distribution_calibrators: dict[str, Any] | None = None,
    event_curve_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> float | None:
    outcome_calibrators = distribution_calibrators
    tb_structure = None
    event_artifact: dict[str, Any] = {}
    pa_uncertainty: dict[str, Any] = {}
    if distribution_calibrators and "outcome" in distribution_calibrators:
        outcome_calibrators = distribution_calibrators.get("outcome")
        tb_structure = distribution_calibrators.get("tb_structure")
        event_artifact = distribution_calibrators.get("event_model") or {}
        pa_uncertainty = event_artifact.get("pa_uncertainty") or {}
    tb_state_alpha = float((((event_artifact.get("models") or {}).get("tb_state_blend_alpha")) or 0.0))
    market = str(row.get("market") or "")
    line = row.get("market_line")
    pred = row.get("pred_count")
    if pd.isna(line) or pd.isna(pred):
        pred = row.get("pred_value")
    if pd.isna(line) or pd.isna(pred):
        return None
    pred = max(1e-6, float(pred))
    pred = _opp_adjusted_mean(row, pred)
    line_f = float(line)

    def event_curve(pa_value: float) -> dict[str, Any]:
        event_probs = _direct_event_probs(row, pa_value)
        if event_probs is None:
            return {}
        cache_key = (
            row.get("game_slug"),
            row.get("player_id"),
            round(pa_value, 5),
            *[round(float(event_probs.get(f"p_{cls}") or 0.0), 7) for cls in EVENT_CLASSES],
            _pa_uncertainty_key(row),
            round(tb_state_alpha, 4),
            *[round(float(row.get(f"p_tb_state_{state}") or 0.0), 7) for state in TB_STATE_NAMES],
        )
        if event_curve_cache is not None and cache_key in event_curve_cache:
            return event_curve_cache[cache_key]
        curve = convolve_hitter_outcomes(
            event_probs,
            projected_pa_pmf(pa_value, row, pa_uncertainty),
        )
        curve = _blend_tb_state_curve(curve, row, tb_state_alpha)
        if event_curve_cache is not None:
            event_curve_cache[cache_key] = curve
        return curve

    if market == "batter_hits":
        pa = _hitter_pa_for_distribution(row)
        if pa is not None and float(pa) >= 1.0:
            counts = _expected_hitter_counts(row)
            counts["hits"] = pred
            curve = event_curve(float(pa))
            if curve:
                return _pmf_over(line_f, curve["hits_pmf"])
            counts = _apply_hitter_outcome_calibrator(row, counts, outcome_calibrators)
            return _hitter_hits_over(
                line_f,
                float(pa),
                float(counts.get("hits") or pred),
                _safe_float(counts.get("tb")),
                _safe_float(counts.get("hr")),
                _walk_rate(row),
            )
    if market == "batter_total_bases":
        pa = _hitter_pa_for_distribution(row)
        if pa is not None and float(pa) >= 1.0:
            pa_f = float(pa)
            counts = _expected_hitter_counts(row)
            counts["tb"] = pred
            curve = event_curve(pa_f)
            if curve:
                return _tb_state_over_probability(line_f, curve)
            counts = _apply_hitter_outcome_calibrator(row, counts, outcome_calibrators)
            return _compound_tb_over(
                line_f,
                pa_f,
                _safe_float(counts.get("tb")) or pred,
                _safe_float(counts.get("hits")),
                _safe_float(counts.get("hr")),
                _walk_rate(row),
                tb_structure=tb_structure,
                row=row,
            )
    if market == "batter_home_runs":
        pa = _hitter_pa_for_distribution(row)
        if pa is not None and float(pa) >= 1.0:
            hr_any = _safe_float(row.get("p_event_hr_any"))
            if line_f <= 0.5 and hr_any is not None:
                return _clamp(hr_any, 1e-6, 1.0 - 1e-6)
            counts = _expected_hitter_counts(row)
            counts["hr"] = pred
            curve = event_curve(float(pa))
            if curve:
                return _pmf_over(line_f, curve["hr_pmf"])
            counts = _apply_hitter_outcome_calibrator(row, counts, outcome_calibrators)
            return _hitter_hr_over(
                line_f,
                float(pa),
                _safe_float(counts.get("hr")) or pred,
                _safe_float(counts.get("hits")),
                _safe_float(counts.get("tb")),
                _walk_rate(row),
            )
    return _poisson_over(line_f, pred)


def _player_game_group_key(df: pd.DataFrame) -> pd.Series:
    game = df.get("game_slug", pd.Series("unknown_game", index=df.index)).fillna("unknown_game").astype(str)
    player = df.get("player_id", pd.Series("unknown_player", index=df.index)).fillna("unknown_player").astype(str)
    return game + "|" + player


def _add_offer_group_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["offer_group_weight"] = pd.Series(dtype="float64")
        out["player_game_group"] = pd.Series(dtype="object")
        return out
    out["player_game_group"] = _player_game_group_key(out)
    counts = out.groupby("player_game_group")["player_game_group"].transform("size").clip(lower=1)
    raw = 1.0 / counts.astype(float)
    out["offer_group_weight"] = raw * (float(len(raw)) / float(raw.sum()))
    return out


def _purge_player_game_overlap(train: pd.DataFrame, holdout: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if train.empty or holdout.empty:
        return train, holdout, 0
    holdout_keys = set(_player_game_group_key(holdout))
    train_keys = _player_game_group_key(train)
    keep = ~train_keys.isin(holdout_keys)
    return train.loc[keep].copy(), holdout.copy(), int((~keep).sum())


def _split(df: pd.DataFrame, cfg: DistributionConfig) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    split = max(df["game_date_et"]) - timedelta(days=cfg.holdout_days)
    train = df.loc[df["game_date_et"] < split].copy()
    holdout = df.loc[df["game_date_et"] >= split].copy()
    if len(train) >= cfg.min_train_rows and len(holdout) >= cfg.min_holdout_rows:
        train, holdout, purged = _purge_player_game_overlap(train, holdout)
        return _add_offer_group_weights(train), _add_offer_group_weights(holdout), f"last_{cfg.holdout_days}_days_purged_{purged}"
    dates = sorted(df["game_date_et"].unique())
    if len(dates) > 1:
        holdout_date = dates[-1]
        train = df.loc[df["game_date_et"] < holdout_date].copy()
        holdout = df.loc[df["game_date_et"] >= holdout_date].copy()
        train, holdout, purged = _purge_player_game_overlap(train, holdout)
        return _add_offer_group_weights(train), _add_offer_group_weights(holdout), f"last_available_date_purged_{purged}"
    train, holdout, purged = _purge_player_game_overlap(train, holdout)
    return _add_offer_group_weights(train), _add_offer_group_weights(holdout), f"last_{cfg.holdout_days}_days_purged_{purged}"


def _weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if weights is None:
        return float(numeric.mean())
    weight_values = pd.to_numeric(weights, errors="coerce").fillna(1.0)
    mask = numeric.notna() & weight_values.notna() & (weight_values > 0)
    if not mask.any():
        return float(numeric.mean())
    return float(np.average(numeric.loc[mask], weights=weight_values.loc[mask]))


def _weighted_brier(y: pd.Series, p: pd.Series, weights: pd.Series | None = None) -> float:
    target = pd.to_numeric(y, errors="coerce")
    probability = pd.to_numeric(p, errors="coerce").clip(1e-6, 1.0 - 1e-6)
    loss = (target - probability) ** 2
    return _weighted_mean(loss, weights)


def _empirical_rates(train: pd.DataFrame) -> dict[str, float]:
    rates = {}
    for cols in (
        ["market", "side", "line_surface", "line_bucket", "price_bucket", "bookmaker_key"],
        ["market", "side", "line_surface", "line_bucket", "price_bucket"],
        ["market", "side", "line_surface", "line_bucket"],
        ["market", "side"],
    ):
        for key, group in train.groupby(cols, dropna=False):
            if len(group) < 20:
                continue
            key = key if isinstance(key, tuple) else (key,)
            rates["|".join([*cols, *[str(v) for v in key]])] = _weighted_mean(
                group["target"], group.get("offer_group_weight")
            )
    return rates


def _empirical_lookup(row: pd.Series, rates: dict[str, float]) -> float | None:
    specs = (
        ["market", "side", "line_surface", "line_bucket", "price_bucket", "bookmaker_key"],
        ["market", "side", "line_surface", "line_bucket", "price_bucket"],
        ["market", "side", "line_surface", "line_bucket"],
        ["market", "side"],
    )
    for cols in specs:
        key = "|".join([*cols, *[str(row.get(col) or "unknown") for col in cols]])
        if key in rates:
            return rates[key]
    return None


def _score_distribution_base(
    df: pd.DataFrame,
    cfg: DistributionConfig,
    outcome_calibrators: dict[str, Any] | None = None,
) -> pd.DataFrame:
    out = _score_opportunity(df, cfg)
    out = _score_hitter_event_model(out, outcome_calibrators)
    event_curve_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    p_over = [_distribution_over(row, outcome_calibrators, event_curve_cache) for _, row in out.iterrows()]
    out["p_distribution_over"] = pd.Series(p_over, index=out.index, dtype="float64")
    out["p_distribution_side"] = np.where(
        out["side"].astype(str) == "over",
        out["p_distribution_over"],
        1.0 - out["p_distribution_over"],
    )
    out["p_distribution_side"] = pd.to_numeric(out["p_distribution_side"], errors="coerce").clip(1e-6, 1 - 1e-6)
    return out


def _prob_bin(prob: Any) -> str | None:
    try:
        p = float(prob)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p):
        return None
    idx = max(0, min(9, int(p * 10.0)))
    lo = idx / 10.0
    hi = 1.0 if idx == 9 else (idx + 1) / 10.0
    return f"{lo:.1f}-{hi:.1f}"


_CALIBRATION_GROUP_SPECS = (
    ["market", "side", "line_surface", "line_bucket", "price_bucket", "bookmaker_key"],
    ["market", "side", "line_surface", "line_bucket", "price_bucket"],
    ["market", "side", "line_surface", "line_bucket"],
    ["market", "side", "line_bucket"],
    ["market", "side", "line_surface"],
    ["market", "side"],
)


def _beta_features(probabilities: pd.Series | np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.column_stack([np.log(p), np.log1p(-p)])


def _fit_serialized_calibrator(frame: pd.DataFrame, method: str, probability_col: str) -> dict[str, Any] | None:
    work = frame.dropna(subset=[probability_col, "target"]).copy()
    if len(work) < 80 or work["target"].nunique() < 2:
        return None
    p = pd.to_numeric(work[probability_col], errors="coerce").clip(1e-6, 1.0 - 1e-6)
    y = work["target"].astype(int)
    weights = pd.to_numeric(
        work.get("offer_group_weight", pd.Series(1.0, index=work.index)), errors="coerce"
    ).fillna(1.0)
    if method == "beta":
        model = LogisticRegression(max_iter=1500, C=1.0)
        model.fit(_beta_features(p), y, sample_weight=weights)
        return {
            "method": "beta",
            "intercept": float(model.intercept_[0]),
            "coef": [float(value) for value in model.coef_[0]],
        }
    if method == "isotonic":
        model = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
        model.fit(p.to_numpy(dtype=float), y.to_numpy(dtype=float), sample_weight=weights.to_numpy(dtype=float))
        return {
            "method": "isotonic",
            "x_thresholds": [float(value) for value in model.X_thresholds_],
            "y_thresholds": [float(value) for value in model.y_thresholds_],
        }
    return None


def _apply_serialized_calibrator(probabilities: pd.Series | np.ndarray, model: dict[str, Any]) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    if model.get("method") == "beta":
        coef = np.asarray(model.get("coef") or [0.0, 0.0], dtype=float)
        z = float(model.get("intercept") or 0.0) + _beta_features(p).dot(coef)
        return np.clip(1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0))), 1e-6, 1.0 - 1e-6)
    if model.get("method") == "isotonic":
        x = np.asarray(model.get("x_thresholds") or [], dtype=float)
        y = np.asarray(model.get("y_thresholds") or [], dtype=float)
        if len(x) >= 2 and len(x) == len(y):
            return np.clip(np.interp(p, x, y, left=y[0], right=y[-1]), 1e-6, 1.0 - 1e-6)
    return p


def _fit_walk_forward_calibrator(
    frame: pd.DataFrame,
    probability_col: str,
    *,
    min_rows: int = 120,
    min_gain: float = 0.0001,
) -> dict[str, Any]:
    work = frame.dropna(subset=[probability_col, "target", "game_date_et"]).copy()
    dates = sorted(pd.to_datetime(work["game_date_et"]).dt.date.unique())
    if len(work) < min_rows or len(dates) < 4:
        return {"enabled": False, "reason": "insufficient_walk_forward_rows", "rows": int(len(work)), "dates": len(dates)}
    validation_start = dates[max(1, int(len(dates) * 0.75))]
    fit = work.loc[pd.to_datetime(work["game_date_et"]).dt.date < validation_start].copy()
    validation = work.loc[pd.to_datetime(work["game_date_et"]).dt.date >= validation_start].copy()
    fit, validation, purged = _purge_player_game_overlap(fit, validation)
    fit = _add_offer_group_weights(fit)
    validation = _add_offer_group_weights(validation)
    if len(fit) < 80 or len(validation) < 30 or fit["target"].nunique() < 2 or validation["target"].nunique() < 2:
        return {
            "enabled": False,
            "reason": "insufficient_temporal_validation_rows",
            "rows": int(len(work)),
            "fit_rows": int(len(fit)),
            "validation_rows": int(len(validation)),
            "purged_rows": purged,
        }
    raw_brier = _weighted_brier(validation["target"], validation[probability_col], validation["offer_group_weight"])
    candidates: list[dict[str, Any]] = []
    fitted: dict[str, dict[str, Any]] = {}
    for method in ("beta", "isotonic"):
        try:
            model = _fit_serialized_calibrator(fit, method, probability_col)
            if not model:
                continue
            calibrated = _apply_serialized_calibrator(validation[probability_col], model)
            brier = _weighted_brier(
                validation["target"], pd.Series(calibrated, index=validation.index), validation["offer_group_weight"]
            )
            candidates.append({"method": method, "validation_brier": brier, "brier_gain": raw_brier - brier})
            fitted[method] = model
        except Exception as exc:
            candidates.append({"method": method, "error": str(exc)})
    usable = [rec for rec in candidates if rec.get("validation_brier") is not None]
    if not usable:
        return {"enabled": False, "reason": "calibrator_fit_failed", "rows": int(len(work)), "candidates": candidates}
    best = min(usable, key=lambda rec: rec["validation_brier"])
    enabled = bool(float(best["brier_gain"]) >= min_gain)
    production_model = _fit_serialized_calibrator(_add_offer_group_weights(work), str(best["method"]), probability_col) if enabled else None
    return {
        "enabled": enabled,
        "reason": "temporal_brier_improved" if enabled else "no_temporal_brier_gain",
        "method": best["method"] if enabled else "raw",
        "model": production_model,
        "rows": int(len(work)),
        "player_games": int(_player_game_group_key(work).nunique()),
        "dates": len(dates),
        "fit_rows": int(len(fit)),
        "validation_rows": int(len(validation)),
        "validation_start": str(validation_start),
        "purged_rows": purged,
        "raw_validation_brier": raw_brier,
        "calibrated_validation_brier": float(best["validation_brier"]),
        "validation_brier_gain": float(best["brier_gain"]),
        "candidates": candidates,
    }


def _fit_distribution_calibrators(scored_train: pd.DataFrame) -> dict[str, Any]:
    work = scored_train.dropna(subset=["p_distribution_side", "target"]).copy()
    if work.empty:
        return {}
    hitter_mask = work["market"].isin(["batter_hits", "batter_total_bases", "batter_home_runs"])
    work = work.loc[~hitter_mask | _true_pair_hitter_mask(work)].copy()
    if work.empty:
        return {}
    calibrators: dict[str, Any] = {
        "method": "walk_forward_beta_or_isotonic",
        "auto_disable_on_holdout_regression": True,
        "groups": {},
    }
    for cols in _CALIBRATION_GROUP_SPECS:
        for key, group in work.groupby(cols, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            group_key = "|".join([*cols, *[str(v) for v in key]])
            rec = _fit_walk_forward_calibrator(group, "p_distribution_side")
            rec.update({"columns": cols, "key_values": [str(v) for v in key]})
            calibrators["groups"][group_key] = rec
    return calibrators


def _apply_distribution_calibrators(df: pd.DataFrame, calibrators: dict[str, Any]) -> pd.Series:
    if not calibrators or calibrators.get("overall_holdout_enabled") is False:
        return pd.to_numeric(df.get("p_distribution_side"), errors="coerce")
    groups = calibrators.get("groups") or {}
    values = []
    for _, row in df.iterrows():
        p = row.get("p_distribution_side")
        calibrated = None
        if pd.notna(p):
            for cols in _CALIBRATION_GROUP_SPECS:
                group_key = "|".join([*cols, *[str(row.get(col) or "unknown") for col in cols]])
                rec = groups.get(group_key) or {}
                if rec.get("enabled") and rec.get("holdout_enabled") is not False and rec.get("model"):
                    calibrated = float(_apply_serialized_calibrator([p], rec["model"])[0])
                    break
        if calibrated is None:
            calibrated = p
        values.append(calibrated)
    return pd.Series(values, index=df.index, dtype="float64").clip(1e-6, 1 - 1e-6)


def _gate_distribution_calibrators(calibrators: dict[str, Any], holdout: pd.DataFrame) -> dict[str, Any]:
    groups = calibrators.get("groups") or {}
    for rec in groups.values():
        if not rec.get("enabled") or not rec.get("model"):
            rec["holdout_enabled"] = False
            continue
        mask = pd.Series(True, index=holdout.index)
        for col, value in zip(rec.get("columns") or [], rec.get("key_values") or []):
            mask &= holdout.get(col, pd.Series("unknown", index=holdout.index)).fillna("unknown").astype(str) == str(value)
        group = holdout.loc[mask].dropna(subset=["p_distribution_side", "target"])
        if len(group) < 30:
            rec.update({"holdout_enabled": False, "holdout_reason": "insufficient_rows", "holdout_rows": int(len(group))})
            continue
        candidate = pd.Series(
            _apply_serialized_calibrator(group["p_distribution_side"], rec["model"]), index=group.index
        )
        raw_brier = _weighted_brier(group["target"], group["p_distribution_side"], group.get("offer_group_weight"))
        candidate_brier = _weighted_brier(group["target"], candidate, group.get("offer_group_weight"))
        gain = raw_brier - candidate_brier
        rec.update({
            "holdout_enabled": bool(gain > 0.0),
            "holdout_reason": "brier_improved" if gain > 0.0 else "brier_regressed",
            "holdout_rows": int(len(group)),
            "holdout_player_games": int(_player_game_group_key(group).nunique()),
            "raw_holdout_brier": raw_brier,
            "calibrated_holdout_brier": candidate_brier,
            "holdout_brier_gain": gain,
        })
    candidate_all = _apply_distribution_calibrators(holdout, calibrators)
    valid = holdout["target"].notna() & holdout["p_distribution_side"].notna() & candidate_all.notna()
    if valid.any():
        raw = _weighted_brier(
            holdout.loc[valid, "target"], holdout.loc[valid, "p_distribution_side"], holdout.loc[valid].get("offer_group_weight")
        )
        calibrated = _weighted_brier(
            holdout.loc[valid, "target"], candidate_all.loc[valid], holdout.loc[valid].get("offer_group_weight")
        )
        calibrators["overall_holdout_enabled"] = bool(calibrated < raw)
        calibrators["raw_holdout_brier"] = raw
        calibrators["calibrated_holdout_brier"] = calibrated
        calibrators["overall_holdout_brier_gain"] = raw - calibrated
    else:
        calibrators["overall_holdout_enabled"] = False
    return calibrators


def _hitter_line_calibration_key(row: pd.Series) -> str | None:
    market = str(row.get("market") or "")
    line = _safe_float(row.get("market_line"))
    if line is None:
        return None
    if market == "batter_total_bases":
        surface = f"TB {line:.1f}"
    elif market == "batter_home_runs":
        surface = f"HR {line:.1f}"
    else:
        return None
    return "|".join([market, str(row.get("side") or "unknown"), surface])


def _true_pair_hitter_mask(df: pd.DataFrame) -> pd.Series:
    true_pair = pd.to_numeric(df.get("true_pair_flag", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0) > 0.5
    synthetic = pd.to_numeric(df.get("synthetic_pair_flag", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0) > 0.5
    source = df.get("market_prob_source", pd.Series("", index=df.index)).fillna("").astype(str)
    return true_pair & ~synthetic & ~source.isin(["raw_implied", "synthetic_fanduel_over_only"])


def _fit_true_pair_hitter_line_calibrators(
    scored_train: pd.DataFrame,
    probability_col: str = "p_distribution_side",
) -> dict[str, Any]:
    mask = _true_pair_hitter_mask(scored_train)
    work = scored_train.loc[mask].dropna(subset=[probability_col, "target"]).copy()
    work = work.loc[work["market"].isin(["batter_total_bases", "batter_home_runs"])]
    work["event_line_key"] = work.apply(_hitter_line_calibration_key, axis=1)
    groups: dict[str, Any] = {}
    for key, group in work.dropna(subset=["event_line_key"]).groupby("event_line_key"):
        rec = _fit_walk_forward_calibrator(group, probability_col, min_rows=80)
        rec["event_line_key"] = str(key)
        rec["probability_col"] = probability_col
        groups[str(key)] = rec
    return {
        "status": "trained" if groups else "insufficient_true_pair_rows",
        "evidence": "temporal_train_true_pair_non_synthetic_only",
        "method": "walk_forward_beta_or_isotonic",
        "groups": groups,
    }


def _apply_true_pair_hitter_line_calibrators(
    df: pd.DataFrame,
    base: pd.Series,
    calibrators: dict[str, Any],
) -> pd.Series:
    values = pd.to_numeric(base, errors="coerce").copy()
    groups = calibrators.get("groups") or {}
    if calibrators.get("overall_holdout_enabled") is False:
        return values
    if not groups:
        return values
    true_pair_mask = _true_pair_hitter_mask(df)
    for idx, row in df.iterrows():
        if not bool(true_pair_mask.loc[idx]):
            continue
        key = _hitter_line_calibration_key(row)
        rec = groups.get(str(key)) or {}
        raw_probability = row.get(rec.get("probability_col") or "p_distribution_side")
        if rec.get("enabled") and rec.get("holdout_enabled") is not False and rec.get("model") and pd.notna(raw_probability):
            values.loc[idx] = float(_apply_serialized_calibrator([raw_probability], rec["model"])[0])
    return values.clip(1e-6, 1.0 - 1e-6)


def _gate_true_pair_hitter_line_calibrators(
    calibrators: dict[str, Any],
    holdout: pd.DataFrame,
    probability_col: str = "p_distribution_side",
    baseline_col: str | None = None,
) -> dict[str, Any]:
    true_pairs = holdout.loc[_true_pair_hitter_mask(holdout)].copy()
    true_pairs["event_line_key"] = true_pairs.apply(_hitter_line_calibration_key, axis=1)
    for key, rec in (calibrators.get("groups") or {}).items():
        if not rec.get("enabled") or not rec.get("model"):
            rec["holdout_enabled"] = False
            continue
        required = [probability_col, "target", *([baseline_col] if baseline_col else [])]
        group = true_pairs.loc[true_pairs["event_line_key"] == key].dropna(subset=required)
        if len(group) < 30:
            rec.update({"holdout_enabled": False, "holdout_reason": "insufficient_rows", "holdout_rows": int(len(group))})
            continue
        candidate = pd.Series(
            _apply_serialized_calibrator(group[probability_col], rec["model"]), index=group.index
        )
        baseline_probability = group[baseline_col] if baseline_col else group[probability_col]
        raw_brier = _weighted_brier(group["target"], baseline_probability, group.get("offer_group_weight"))
        candidate_brier = _weighted_brier(group["target"], candidate, group.get("offer_group_weight"))
        gain = raw_brier - candidate_brier
        actual_rate = _weighted_mean(group["target"], group.get("offer_group_weight"))
        raw_mean = _weighted_mean(baseline_probability, group.get("offer_group_weight"))
        calibrated_mean = _weighted_mean(candidate, group.get("offer_group_weight"))
        rec.update({
            "holdout_enabled": bool(gain > 0.0),
            "holdout_reason": "brier_improved" if gain > 0.0 else "brier_regressed",
            "holdout_rows": int(len(group)),
            "raw_holdout_brier": raw_brier,
            "calibrated_holdout_brier": candidate_brier,
            "holdout_brier_gain": gain,
            "raw_holdout_calibration_error": actual_rate - raw_mean,
            "calibrated_holdout_calibration_error": actual_rate - calibrated_mean,
        })
    calibrators["overall_holdout_enabled"] = any(
        bool(rec.get("holdout_enabled")) for rec in (calibrators.get("groups") or {}).values()
    )
    calibrators["enabled_groups"] = sorted(
        key for key, rec in (calibrators.get("groups") or {}).items() if rec.get("holdout_enabled")
    )
    return calibrators


def _prepare_side_line_matrix(df: pd.DataFrame, *, state: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    state = dict(state or {})
    parts: list[np.ndarray] = []
    numeric_means = dict(state.get("numeric_means") or {})
    numeric_scales = dict(state.get("numeric_scales") or {})
    categorical_values = {k: list(v) for k, v in (state.get("categorical_values") or {}).items()}
    for col in _SIDE_LINE_NUMERIC:
        s = pd.to_numeric(df.get(col), errors="coerce")
        if col == "market_price":
            s = s.abs()
        if col not in numeric_means:
            numeric_means[col] = float(s.mean()) if not s.dropna().empty else 0.0
        filled = s.fillna(float(numeric_means[col]))
        if col not in numeric_scales:
            std = float(filled.std(ddof=0) or 0.0)
            numeric_scales[col] = std if std > 1e-9 else 1.0
        parts.append(((filled - float(numeric_means[col])) / float(numeric_scales[col])).to_numpy(dtype=float).reshape(-1, 1))
    for col in _SIDE_LINE_CATEGORICAL:
        series = df.get(col, pd.Series(["unknown"] * len(df), index=df.index)).fillna("unknown").astype(str)
        if col not in categorical_values:
            values = series.value_counts().loc[lambda x: x >= 10].index.astype(str).tolist()
            categorical_values[col] = sorted(values) or ["unknown"]
        for value in categorical_values[col]:
            parts.append((series == str(value)).astype(float).to_numpy(dtype=float).reshape(-1, 1))
    X = np.hstack(parts) if parts else np.zeros((len(df), 0), dtype=float)
    return X, {
        "numeric_means": numeric_means,
        "numeric_scales": numeric_scales,
        "categorical_values": categorical_values,
    }


def _clean_market_side_line_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if "clean_market_pair_flag" in df.columns:
        clean = pd.to_numeric(df["clean_market_pair_flag"], errors="coerce").fillna(0.0) >= 0.5
        return df.loc[clean].copy()
    pair_quality = df.get("pair_quality", pd.Series(["unknown"] * len(df), index=df.index)).fillna("unknown").astype(str).str.lower()
    market_source = df.get("market_prob_source", pd.Series(["unknown"] * len(df), index=df.index)).fillna("unknown").astype(str).str.lower()
    clean = pair_quality.isin(["same_book", "cross_book"]) & ~market_source.isin(["raw_implied", "synthetic_fanduel_over_only"])
    return df.loc[clean].copy()


def _fit_one_side_line_model(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    target_col: str,
    baseline_col: str,
    *,
    min_train: int = 250,
    min_holdout: int = 80,
) -> dict[str, Any]:
    train_clean = _clean_market_side_line_rows(train)
    holdout_clean = _clean_market_side_line_rows(holdout)
    tr = train_clean.dropna(subset=[target_col, "p_distribution_blend"]).copy()
    ho = holdout_clean.dropna(subset=[target_col, "p_distribution_blend"]).copy()
    if len(tr) < min_train or len(ho) < min_holdout or tr[target_col].nunique() < 2 or ho[target_col].nunique() < 2:
        return {
            "status": "insufficient_rows",
            "train_rows": int(len(tr)),
            "holdout_rows": int(len(ho)),
            "clean_pair_required": True,
            "raw_train_rows": int(len(train.dropna(subset=[target_col, "p_distribution_blend"]))),
            "raw_holdout_rows": int(len(holdout.dropna(subset=[target_col, "p_distribution_blend"]))),
        }
    X_train, state = _prepare_side_line_matrix(tr)
    y_train = tr[target_col].astype(int).to_numpy()
    model = LogisticRegression(max_iter=2500, C=0.55)
    fit_weights = pd.to_numeric(
        tr.get("offer_group_weight", pd.Series(1.0, index=tr.index)), errors="coerce"
    ).fillna(1.0)
    model.fit(X_train, y_train, sample_weight=fit_weights)
    X_hold, _ = _prepare_side_line_matrix(ho, state=state)
    p = np.clip(model.predict_proba(X_hold)[:, 1], 1e-6, 1 - 1e-6)
    y = ho[target_col].astype(int)
    baseline = pd.to_numeric(ho.get(baseline_col), errors="coerce").fillna(ho["p_distribution_blend"]).astype(float).clip(1e-6, 1 - 1e-6)
    feature_names = list(_SIDE_LINE_NUMERIC)
    for col in _SIDE_LINE_CATEGORICAL:
        values = (state.get("categorical_values") or {}).get(col, [])
        feature_names.extend([f"{col}={value}" for value in values])
    coef = {
        name: float(value)
        for name, value in zip(feature_names, model.coef_[0])
        if abs(float(value)) > 1e-12
    }
    return {
        "status": "trained",
        "method": "event_curve_side_line_logistic",
        "target": target_col,
        "train_rows": int(len(tr)),
        "holdout_rows": int(len(ho)),
        "clean_pair_required": True,
        "raw_train_rows": int(len(train.dropna(subset=[target_col, "p_distribution_blend"]))),
        "raw_holdout_rows": int(len(holdout.dropna(subset=[target_col, "p_distribution_blend"]))),
        "actual_rate_holdout": float(y.mean()),
        "avg_model_holdout": float(np.mean(p)),
        "avg_baseline_holdout": float(baseline.mean()),
        "brier_model_holdout": _weighted_brier(y, pd.Series(p, index=ho.index), ho.get("offer_group_weight")),
        "brier_baseline_holdout": _weighted_brier(y, baseline, ho.get("offer_group_weight")),
        "log_loss_model_holdout": float(log_loss(y, p, labels=[0, 1], sample_weight=ho.get("offer_group_weight"))),
        "log_loss_baseline_holdout": float(log_loss(y, baseline, labels=[0, 1], sample_weight=ho.get("offer_group_weight"))),
        "intercept": float(model.intercept_[0]),
        "coef": coef,
        **state,
    }


def _score_side_line_model(df: pd.DataFrame, model_rec: dict[str, Any]) -> pd.Series:
    if model_rec.get("status") != "trained":
        return pd.Series(np.nan, index=df.index, dtype="float64")
    state = {
        "numeric_means": model_rec.get("numeric_means") or {},
        "numeric_scales": model_rec.get("numeric_scales") or {},
        "categorical_values": model_rec.get("categorical_values") or {},
    }
    X, _ = _prepare_side_line_matrix(df, state=state)
    feature_names = list(_SIDE_LINE_NUMERIC)
    for col in _SIDE_LINE_CATEGORICAL:
        values = (state.get("categorical_values") or {}).get(col, [])
        feature_names.extend([f"{col}={value}" for value in values])
    coefs = np.asarray([float((model_rec.get("coef") or {}).get(name, 0.0)) for name in feature_names], dtype=float)
    z = float(model_rec.get("intercept", 0.0)) + X.dot(coefs)
    return pd.Series(1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0))), index=df.index, dtype="float64").clip(1e-6, 1 - 1e-6)


def _fit_side_line_models(scored_train: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, Any]:
    win_model = _fit_one_side_line_model(scored_train, holdout, "target", "p_distribution_blend")
    clv_train = scored_train.dropna(subset=["beat_clv_price"]).copy()
    clv_holdout = holdout.dropna(subset=["beat_clv_price"]).copy()
    clv_model = _fit_one_side_line_model(
        clv_train,
        clv_holdout,
        "beat_clv_price",
        "market_prob_side",
        min_train=160,
        min_holdout=40,
    )
    return {
        "win_probability": win_model,
        "clv_beat_probability": clv_model,
        "usage": "shadow_only_until_exact_buckets_show_positive_roi_and_clv",
    }


def _brier_for(df: pd.DataFrame, col: str) -> float | None:
    work = df.dropna(subset=[col, "target"])
    if work.empty:
        return None
    return float(brier_score_loss(work["target"].astype(int), work[col].astype(float).clip(1e-6, 1 - 1e-6)))


def _outcome_policy(uncalibrated: pd.DataFrame, learned: pd.DataFrame, *, min_gain: float = 0.0005) -> dict[str, Any]:
    markets = ["batter_hits", "batter_total_bases", "batter_home_runs"]
    records: dict[str, Any] = {}
    for market in markets:
        base = uncalibrated.loc[uncalibrated["market"] == market]
        cand = learned.loc[learned["market"] == market]
        true_pair_index = cand.index[_true_pair_hitter_mask(cand)]
        base = base.loc[base.index.intersection(true_pair_index)]
        cand = cand.loc[true_pair_index]
        base_brier = _brier_for(base, "p_distribution_side")
        learned_brier = _brier_for(cand, "p_distribution_side")
        use_learned = (
            base_brier is not None
            and learned_brier is not None
            and (base_brier - learned_brier) >= min_gain
        )
        records[market] = {
            "rows": int(len(cand)),
            "base_brier": base_brier,
            "learned_brier": learned_brier,
            "brier_gain": (base_brier - learned_brier) if base_brier is not None and learned_brier is not None else None,
            "use_learned": bool(use_learned),
        }
    return {"min_gain": min_gain, "markets": records}


def _bucket_key(row: pd.Series, cols: list[str]) -> str:
    return "|".join(str(row.get(col) or "unknown") for col in cols)


def _outcome_bucket_policy(
    uncalibrated: pd.DataFrame,
    learned: pd.DataFrame,
    *,
    min_rows: int = 80,
    min_gain: float = 0.002,
) -> dict[str, Any]:
    cols = ["market", "side", "line_surface", "line_bucket", "price_bucket", "bookmaker_key"]
    if uncalibrated.empty or learned.empty:
        return {"min_rows": min_rows, "min_gain": min_gain, "buckets": {}}
    rows: dict[str, Any] = {}
    work = learned.loc[learned["market"] == "batter_total_bases"].copy()
    work = work.loc[_true_pair_hitter_mask(work)]
    for key, group in work.groupby(cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        if len(group) < min_rows:
            continue
        idx = group.index
        base = uncalibrated.loc[idx]
        base_brier = _brier_for(base, "p_distribution_side")
        learned_brier = _brier_for(group, "p_distribution_side")
        if base_brier is None or learned_brier is None:
            continue
        gain = base_brier - learned_brier
        key_text = "|".join(str(v) for v in key)
        rows[key_text] = {
            "columns": cols,
            "key_values": [str(v) for v in key],
            "rows": int(len(group)),
            "base_brier": base_brier,
            "learned_brier": learned_brier,
            "brier_gain": gain,
            "use_learned": bool(gain >= min_gain),
        }
    return {"min_rows": min_rows, "min_gain": min_gain, "buckets": rows}


def _apply_outcome_policy(uncalibrated: pd.DataFrame, learned: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    out = learned.copy()
    market_policy = policy.get("markets") or {}
    bucket_policy = (policy.get("bucket_policy") or {}).get("buckets") or {}
    bucket_cols = ["market", "side", "line_surface", "line_bucket", "price_bucket", "bookmaker_key"]
    for market, rec in market_policy.items():
        if rec.get("use_learned"):
            continue
        mask = out["market"] == market
        if market == "batter_total_bases" and bucket_policy:
            use_bucket = out.loc[mask].apply(
                lambda row: bool((bucket_policy.get(_bucket_key(row, bucket_cols)) or {}).get("use_learned")),
                axis=1,
            )
            keep_idx = use_bucket.index[use_bucket]
            mask = mask & ~out.index.isin(keep_idx)
        for col in ("p_distribution_over", "p_distribution_side"):
            if col in uncalibrated.columns and col in out.columns:
                out.loc[mask, col] = uncalibrated.loc[mask, col]
    return out


def _score_probabilities(train: pd.DataFrame, holdout: pd.DataFrame, cfg: DistributionConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    rates = _empirical_rates(train)
    scored_train_uncalibrated = _score_distribution_base(train, cfg)
    outcome_calibrators = _fit_hitter_outcome_calibrators(scored_train_uncalibrated)
    tb_structure = _fit_tb_structure_calibrators(scored_train_uncalibrated)
    hitter_event_artifact = _load_hitter_event_artifact(cfg)
    learned_calibrators = {
        "outcome": outcome_calibrators,
        "tb_structure": tb_structure,
        "event_model": hitter_event_artifact,
    }
    scored_train_learned = _score_distribution_base(train, cfg, learned_calibrators)
    holdout_uncalibrated = _score_distribution_base(holdout, cfg)
    holdout_learned = _score_distribution_base(holdout, cfg, learned_calibrators)
    outcome_gate = _outcome_policy(holdout_uncalibrated, holdout_learned)
    outcome_gate["bucket_policy"] = _outcome_bucket_policy(scored_train_uncalibrated, scored_train_learned)
    scored_train = _apply_outcome_policy(scored_train_uncalibrated, scored_train_learned, outcome_gate)
    out = _apply_outcome_policy(holdout_uncalibrated, holdout_learned, outcome_gate)
    probability_calibrators = _fit_distribution_calibrators(scored_train)
    probability_calibrators = _gate_distribution_calibrators(probability_calibrators, out)
    train_generic = _apply_distribution_calibrators(scored_train, probability_calibrators)
    holdout_generic = _apply_distribution_calibrators(out, probability_calibrators)
    generic_enabled = bool(probability_calibrators.get("overall_holdout_enabled"))
    train_line_input = scored_train.copy()
    holdout_line_input = out.copy()
    train_line_input["p_line_calibration_base"] = train_generic if generic_enabled else scored_train["p_distribution_side"]
    holdout_line_input["p_line_calibration_base"] = holdout_generic if generic_enabled else out["p_distribution_side"]
    line_calibrators = _fit_true_pair_hitter_line_calibrators(
        train_line_input,
        probability_col="p_distribution_side",
    )
    line_calibrators = _gate_true_pair_hitter_line_calibrators(
        line_calibrators,
        holdout_line_input,
        probability_col="p_distribution_side",
        baseline_col="p_line_calibration_base",
    )
    train_base = train_line_input["p_line_calibration_base"]
    holdout_base = holdout_line_input["p_line_calibration_base"]
    train_combined = _apply_true_pair_hitter_line_calibrators(train_line_input, train_base, line_calibrators)
    holdout_combined = _apply_true_pair_hitter_line_calibrators(holdout_line_input, holdout_base, line_calibrators)
    valid = out["target"].notna() & out["p_distribution_side"].notna()
    weights = out.loc[valid].get("offer_group_weight")
    calibration_scores = {
        "raw": _weighted_brier(out.loc[valid, "target"], out.loc[valid, "p_distribution_side"], weights),
        "generic": _weighted_brier(out.loc[valid, "target"], holdout_generic.loc[valid], weights),
        "generic_plus_line": _weighted_brier(out.loc[valid, "target"], holdout_combined.loc[valid], weights),
    } if valid.any() else {}
    line_enabled = bool(line_calibrators.get("overall_holdout_enabled"))
    selected_calibration = (
        "generic_plus_line" if line_enabled
        else "generic" if generic_enabled
        else "raw"
    )
    scored_train["p_distribution_calibrated"] = train_combined if line_enabled else train_base
    out["p_distribution_calibrated"] = holdout_combined if line_enabled else holdout_base
    probability_calibrators["system_holdout_selection"] = selected_calibration
    probability_calibrators["system_holdout_brier"] = calibration_scores
    train_empirical = [_empirical_lookup(row, rates) for _, row in scored_train.iterrows()]
    scored_train["p_empirical_bucket"] = pd.Series(train_empirical, index=scored_train.index, dtype="float64")
    scored_train["p_distribution_blend"] = (
        0.70 * scored_train["p_distribution_calibrated"].fillna(scored_train["p_distribution_side"]).astype(float)
        + 0.30 * scored_train["p_empirical_bucket"].fillna(scored_train["model_prob_side"]).astype(float)
    ).clip(1e-6, 1 - 1e-6)
    empirical = [_empirical_lookup(row, rates) for _, row in out.iterrows()]
    out["p_empirical_bucket"] = pd.Series(empirical, index=out.index, dtype="float64")
    out["p_distribution_blend"] = (
        0.70 * out["p_distribution_calibrated"].fillna(out["p_distribution_side"]).astype(float)
        + 0.30 * out["p_empirical_bucket"].fillna(out["model_prob_side"]).astype(float)
    ).clip(1e-6, 1 - 1e-6)
    side_line_models = _fit_side_line_models(scored_train, out)
    out["p_event_side_line"] = _score_side_line_model(out, side_line_models.get("win_probability") or {})
    out["p_event_clv_beat"] = _score_side_line_model(out, side_line_models.get("clv_beat_probability") or {})
    return out, {
        "outcome": outcome_calibrators,
        "tb_structure": tb_structure,
        "event_model": _event_model_metadata(hitter_event_artifact),
        "outcome_policy": outcome_gate,
        "probability": probability_calibrators,
        "true_pair_hitter_line_calibration": line_calibrators,
        "side_line_models": side_line_models,
    }


def _bucket_model_selection(df: pd.DataFrame, cfg: DistributionConfig) -> list[dict[str, Any]]:
    variants = {
        "model_only": "model_prob_side",
        "market_only": "market_prob_side",
        "distribution": "p_distribution_side",
        "distribution_calibrated": "p_distribution_calibrated",
        "distribution_blend": "p_distribution_blend",
        "event_side_line": "p_event_side_line",
    }
    rows = []
    group_cols = ["market", "side", "line_surface", "line_bucket", "price_bucket", "bookmaker_key"]
    for key, group in df.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        if str(key[0]) in {"batter_hits", "batter_total_bases", "batter_home_runs"}:
            group = group.loc[_true_pair_hitter_mask(group)]
            if group.empty:
                continue
        summaries = {}
        best_name = None
        best_brier = None
        for name, col in variants.items():
            if col not in group:
                continue
            forecast = _forecast(group, col)
            selection = _selection(group, col, cfg)
            summaries[name] = {"forecast": forecast, "selection": selection}
            brier = forecast.get("brier")
            if brier is not None and (best_brier is None or brier < best_brier):
                best_name = name
                best_brier = brier
        best_sel = (summaries.get(best_name or "") or {}).get("selection") or {}
        best_roi = best_sel.get("roi")
        if len(group) < max(20, cfg.min_train_rows // 5):
            decision = "no_bet_sample"
        elif best_roi is not None and best_roi <= 0:
            decision = "no_bet_negative_roi"
        elif best_name == "market_only":
            decision = "use_market_only"
        elif best_name == "distribution_blend":
            decision = "use_distribution_market_blend"
        elif best_name == "distribution":
            decision = "use_distribution"
        elif best_name == "event_side_line":
            decision = "use_event_curve_side_line"
        elif best_name == "model_only":
            decision = "keep_model_only"
        else:
            decision = "no_bet_no_edge"
        rows.append({
            "bucket": "|".join(str(v) for v in key),
            "rows": int(len(group)),
            "decision": decision,
            "best_variant": best_name,
            "best_brier": best_brier,
            "best_selected_roi": best_roi,
            "model_brier": (summaries.get("model_only", {}).get("forecast") or {}).get("brier"),
            "market_brier": (summaries.get("market_only", {}).get("forecast") or {}).get("brier"),
            "distribution_brier": (summaries.get("distribution", {}).get("forecast") or {}).get("brier"),
            "calibrated_distribution_brier": (summaries.get("distribution_calibrated", {}).get("forecast") or {}).get("brier"),
            "blend_brier": (summaries.get("distribution_blend", {}).get("forecast") or {}).get("brier"),
            "event_side_line_brier": (summaries.get("event_side_line", {}).get("forecast") or {}).get("brier"),
        })
    rows.sort(key=lambda rec: (rec["decision"].startswith("no_bet"), -(rec.get("rows") or 0)))
    return rows


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _forecast(df: pd.DataFrame, col: str) -> dict[str, Any]:
    work = df.dropna(subset=[col, "target"])
    if work.empty:
        return {"rows": 0, "brier": None}
    p = work[col].astype(float).clip(1e-6, 1 - 1e-6)
    y = work["target"].astype(int)
    weights = work.get("offer_group_weight")
    return {
        "rows": int(len(work)),
        "effective_player_games": int(work.get("player_game_group", _player_game_group_key(work)).nunique()),
        "actual_rate": _weighted_mean(y, weights),
        "avg_prob": _weighted_mean(p, weights),
        "calibration_error": float(_weighted_mean(y, weights) - _weighted_mean(p, weights)),
        "brier": _weighted_brier(y, p, weights),
        "log_loss": float(log_loss(y, p, labels=[0, 1], sample_weight=weights)) if y.nunique() == 2 else None,
    }


def _selection(df: pd.DataFrame, col: str, cfg: DistributionConfig) -> dict[str, Any]:
    work = df.dropna(subset=[col, "market_price"]).copy()
    work["variant_ev"] = [ev_per_unit(prob, price) for prob, price in zip(work[col], work["market_price"])]
    selected = work.loc[pd.to_numeric(work["variant_ev"], errors="coerce") >= cfg.min_ev]
    if selected.empty:
        return {"selected_rows": 0, "roi": None, "clv_beat_rate": None}
    clv = selected.dropna(subset=["beat_clv_price"])
    selected_weights = selected.get("offer_group_weight")
    clv_weights = clv.get("offer_group_weight")
    return {
        "selected_rows": int(len(selected)),
        "selected_player_games": int(selected.get("player_game_group", _player_game_group_key(selected)).nunique()),
        "roi": _weighted_mean(selected["profit_units"], selected_weights),
        "win_rate": _weighted_mean(selected["target"], selected_weights),
        "clv_beat_rate": _weighted_mean(clv["beat_clv_price"], clv_weights) if not clv.empty else None,
        "avg_clv_price": _weighted_mean(clv["clv_price"], clv_weights) if not clv.empty else None,
    }


def _tb_hr_line_production_gates(df: pd.DataFrame, cfg: DistributionConfig) -> dict[str, Any]:
    """Evaluate real-money evidence only on non-synthetic paired offers."""
    if df.empty:
        return {"status": "no_rows", "groups": {}}
    true_pair = pd.to_numeric(
        df.get("true_pair_flag", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0) >= 0.5
    synthetic = pd.to_numeric(
        df.get("synthetic_pair_flag", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0) >= 0.5
    work = df.loc[
        df["market"].isin(["batter_total_bases", "batter_home_runs"])
        & true_pair
        & ~synthetic
    ].copy()
    groups: dict[str, Any] = {}
    for values, group in work.groupby(["market", "side", "line_bucket"], dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        key = "|".join(str(v) for v in values)
        baseline = _forecast(group, "model_prob_side")
        candidate = _forecast(group, "p_distribution_calibrated")
        selected = _selection(group, "p_distribution_calibrated", cfg)
        gain = None
        if baseline.get("brier") is not None and candidate.get("brier") is not None:
            gain = float(baseline["brier"]) - float(candidate["brier"])
        reasons: list[str] = []
        if int(candidate.get("rows") or 0) < 80:
            reasons.append("rows<80")
        if gain is None or gain <= 0.001:
            reasons.append("brier_gain<=0.001")
        cal = candidate.get("calibration_error")
        if cal is None or abs(float(cal)) > 0.05:
            reasons.append("abs_calibration_error>0.05")
        if int(selected.get("selected_rows") or 0) < 30:
            reasons.append("selected_rows<30")
        clv_beat = selected.get("clv_beat_rate")
        if clv_beat is None or float(clv_beat) < 0.55:
            reasons.append("clv_beat_rate<0.55")
        avg_clv = selected.get("avg_clv_price")
        if avg_clv is None or float(avg_clv) <= 0.0:
            reasons.append("avg_clv_price<=0")
        groups[key] = {
            "rows": int(len(group)),
            "model_brier": baseline.get("brier"),
            "distribution_brier": candidate.get("brier"),
            "brier_gain": gain,
            "calibration_error": cal,
            "selected_rows": selected.get("selected_rows"),
            "clv_beat_rate": clv_beat,
            "avg_clv_price": avg_clv,
            "passes": not reasons,
            "reasons": reasons,
        }
    return {
        "status": "ready" if groups else "no_true_pair_rows",
        "evidence": "holdout_true_pair_non_synthetic_only",
        "groups": groups,
    }


def _summaries(df: pd.DataFrame, cfg: DistributionConfig) -> dict[str, Any]:
    variants = {
        "model_only": "model_prob_side",
        "market_no_vig": "market_prob_side",
        "distribution": "p_distribution_side",
        "distribution_calibrated": "p_distribution_calibrated",
        "distribution_empirical_blend": "p_distribution_blend",
        "event_side_line": "p_event_side_line",
    }
    return {
        name: {"forecast": _forecast(df, col), "selection": _selection(df, col, cfg)}
        for name, col in variants.items()
        if col in df.columns
    }


def _group_summaries(df: pd.DataFrame, cfg: DistributionConfig, cols: list[str]) -> list[dict[str, Any]]:
    rows = []
    for key, group in df.groupby(cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({"key": "|".join(str(v) for v in key), "rows": int(len(group)), "variants": _summaries(group, cfg)})
    rows.sort(key=lambda rec: rec["rows"], reverse=True)
    return rows


def _fmt_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value) * 100:.1f}%"


def _fmt_num(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def _write_report(payload: dict[str, Any], cfg: DistributionConfig) -> str:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORT_DIR / cfg.report_file
    lines = [
        "# MLB Prop Distribution Models",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        f"Rows: {payload.get('rows', 0)}",
        f"Date range: {payload.get('date_min')} to {payload.get('date_max')}",
        f"Status: {payload.get('status')}",
        "",
        "## Overall Holdout",
        "",
        "| Variant | Rows | Brier | Log Loss | Cal Err | Selected | ROI | CLV Beat |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rec in (payload.get("overall") or {}).items():
        f = rec.get("forecast") or {}
        s = rec.get("selection") or {}
        lines.append(
            f"| {name} | {f.get('rows', 0)} | {_fmt_num(f.get('brier'))} | {_fmt_num(f.get('log_loss'))} | "
            f"{_fmt_pct(f.get('calibration_error'))} | "
            f"{s.get('selected_rows', 0)} | {_fmt_pct(s.get('roi'))} | {_fmt_pct(s.get('clv_beat_rate'))} |"
        )
    lines.extend([
        "",
        "## Market Holdout",
        "",
        "| Market | Rows | Model Brier | Distribution Brier | Cal Dist Brier | Blend Brier | Side-Line Brier | Model ROI | Blend ROI | Side-Line ROI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for rec in payload.get("market", []):
        variants = rec.get("variants") or {}
        model = variants.get("model_only", {})
        dist = variants.get("distribution", {})
        cal_dist = variants.get("distribution_calibrated", {})
        blend = variants.get("distribution_empirical_blend", {})
        side_line = variants.get("event_side_line", {})
        lines.append(
            f"| {rec['key']} | {rec['rows']} | {_fmt_num((model.get('forecast') or {}).get('brier'))} | "
            f"{_fmt_num((dist.get('forecast') or {}).get('brier'))} | "
            f"{_fmt_num((cal_dist.get('forecast') or {}).get('brier'))} | "
            f"{_fmt_num((blend.get('forecast') or {}).get('brier'))} | "
            f"{_fmt_num((side_line.get('forecast') or {}).get('brier'))} | "
            f"{_fmt_pct((model.get('selection') or {}).get('roi'))} | "
            f"{_fmt_pct((blend.get('selection') or {}).get('roi'))} | "
            f"{_fmt_pct((side_line.get('selection') or {}).get('roi'))} |"
        )
    line_gates = (payload.get("tb_hr_line_production_gates") or {}).get("groups") or {}
    lines.extend([
        "",
        "## TB/HR True-Pair Production Gates",
        "",
        "These gates use holdout rows with true, non-synthetic paired prices only.",
        "",
        "| Market / Side / Line | Rows | Brier Gain | Cal Err | Selected | CLV Beat | Avg CLV | Pass | Reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for key, rec in sorted(line_gates.items()):
        lines.append(
            f"| {key.replace('|', ' / ')} | {rec.get('rows', 0)} | {_fmt_num(rec.get('brier_gain'))} | "
            f"{_fmt_pct(rec.get('calibration_error'))} | {rec.get('selected_rows', 0)} | "
            f"{_fmt_pct(rec.get('clv_beat_rate'))} | {_fmt_pct(rec.get('avg_clv_price'))} | "
            f"{bool(rec.get('passes'))} | {', '.join(rec.get('reasons') or []) or '-'} |"
        )
    outcome_groups = (((payload.get("distribution_calibrators") or {}).get("outcome") or {}).get("groups") or {})
    lines.extend([
        "",
        "## Hitter Outcome Shrinkage",
        "",
        "| Group | Rows | PA | Hit Mult | TB Mult | HR Mult | XBH Mult | Actual H/PA | Pred H/PA | Actual TB/PA | Pred TB/PA | Actual HR/PA | Pred HR/PA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key, rec in sorted(outcome_groups.items(), key=lambda item: (item[0] != "global", item[0]))[:40]:
        lines.append(
            f"| {key} | {rec.get('rows', 0)} | {_fmt_num(rec.get('pa'), 1)} | "
            f"{_fmt_num(rec.get('hit_multiplier'))} | {_fmt_num(rec.get('tb_multiplier'))} | "
            f"{_fmt_num(rec.get('hr_multiplier'))} | {_fmt_num(rec.get('non_hr_extra_multiplier'))} | "
            f"{_fmt_num(rec.get('actual_hits_per_pa'))} | {_fmt_num(rec.get('pred_hits_per_pa'))} | "
            f"{_fmt_num(rec.get('actual_tb_per_pa'))} | {_fmt_num(rec.get('pred_tb_per_pa'))} | "
            f"{_fmt_num(rec.get('actual_hr_per_pa'))} | {_fmt_num(rec.get('pred_hr_per_pa'))} |"
        )
    event_meta = ((payload.get("distribution_calibrators") or {}).get("event_model") or {})
    event_rec = event_meta.get("recommendation") or {}
    tb_state = event_meta.get("tb_state_distribution") or {}
    tb_state_repair = event_meta.get("tb_state_residual") or {}
    lines.extend([
        "",
        "## Direct Hitter Event Model",
        "",
        f"- Status: {event_meta.get('status', 'missing')}",
        f"- Method: {event_meta.get('method', '-')}",
        f"- Trained UTC: {event_meta.get('trained_at_utc', '-')}",
        f"- Classes: {', '.join(map(str, event_meta.get('event_classes') or [])) or '-'}",
        f"- Production gate: {event_rec.get('passes_basic_gate', False)}",
        f"- Production eligible artifact: {event_meta.get('production_eligible', False)}",
        f"- Leakage-safe player priors: {event_meta.get('player_prior_players', 0)} players",
        f"- PA uncertainty groups: {len((event_meta.get('pa_uncertainty') or {}).get('groups') or {})}",
        f"- Direct event TB MAE gain vs independent rates: {_fmt_num(event_rec.get('direct_event_tb_mae_gain_vs_independent_rates'))}",
        f"- Explicit TB-state rows: {tb_state.get('rows', 0)}",
        f"- Explicit TB-state Brier: {_fmt_num(tb_state.get('multiclass_brier'))}",
        f"- Explicit TB-state log loss: {_fmt_num(tb_state.get('log_loss'))}",
        f"- Direct-state selected candidate: {tb_state_repair.get('selected_candidate', 'convolution')}",
        f"- Direct-state blend alpha: {_fmt_num(tb_state_repair.get('alpha'))}",
        f"- HR-driven 4+ tail Brier gain: {_fmt_num(tb_state_repair.get('validation_hr_tail_brier_gain'), 6)}",
    ])
    line_calibration = (
        (payload.get("distribution_calibrators") or {}).get("true_pair_hitter_line_calibration") or {}
    )
    lines.extend([
        "",
        "## True-Pair Hitter Line Calibration",
        "",
        f"- Status: {line_calibration.get('status', 'missing')}",
        f"- Evidence: {line_calibration.get('evidence', '-')}",
        f"- Calibrated line/side groups: {len(line_calibration.get('groups') or {})}",
        f"- Enabled line/side groups: {len(line_calibration.get('enabled_groups') or [])}",
        "- Synthetic and one-sided FanDuel prices are display-only and cannot train these calibrators.",
    ])
    for key, rec in sorted((line_calibration.get("groups") or {}).items()):
        lines.append(
            f"- `{key}`: {rec.get('rows', 0)} rows, method={rec.get('method', 'raw')}, "
            f"internal_gain={_fmt_num(rec.get('validation_brier_gain'))}, "
            f"holdout_gain={_fmt_num(rec.get('holdout_brier_gain'))}, "
            f"cal_before={_fmt_pct(rec.get('raw_holdout_calibration_error'))}, "
            f"cal_after={_fmt_pct(rec.get('calibrated_holdout_calibration_error'))}, "
            f"enabled={rec.get('holdout_enabled', False)}"
        )
    side_line_models = ((payload.get("distribution_calibrators") or {}).get("side_line_models") or {})
    lines.extend([
        "",
        "## Event-Curve Side/Line Models",
        "",
        "| Target | Status | Train | Holdout | Model Brier | Baseline Brier | Model Avg | Baseline Avg |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for label, rec in [
        ("win_probability", side_line_models.get("win_probability") or {}),
        ("clv_beat_probability", side_line_models.get("clv_beat_probability") or {}),
    ]:
        lines.append(
            f"| {label} | {rec.get('status', 'missing')} | {rec.get('train_rows', 0)} | {rec.get('holdout_rows', 0)} | "
            f"{_fmt_num(rec.get('brier_model_holdout'))} | {_fmt_num(rec.get('brier_baseline_holdout'))} | "
            f"{_fmt_pct(rec.get('avg_model_holdout'))} | {_fmt_pct(rec.get('avg_baseline_holdout'))} |"
        )
    tb_structure_groups = (((payload.get("distribution_calibrators") or {}).get("tb_structure") or {}).get("groups") or {})
    lines.extend([
        "",
        "## TB Component Structure",
        "",
        "| Group | Rows | PA | 1B Mult | 2B Mult | 3B Mult | HR Mult | TB Mult | Actual 0 TB | Pred 0 TB | Actual 2B/PA | Pred 2B/PA | Actual HR/PA | Pred HR/PA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key, rec in sorted(tb_structure_groups.items(), key=lambda item: (item[0] != "global", item[0]))[:50]:
        lines.append(
            f"| {key} | {rec.get('rows', 0)} | {_fmt_num(rec.get('pa'), 1)} | "
            f"{_fmt_num(rec.get('single_multiplier'))} | {_fmt_num(rec.get('double_multiplier'))} | "
            f"{_fmt_num(rec.get('triple_multiplier'))} | {_fmt_num(rec.get('hr_multiplier'))} | "
            f"{_fmt_num(rec.get('tb_multiplier'))} | {_fmt_pct(rec.get('actual_zero_tb_rate'))} | "
            f"{_fmt_pct(rec.get('pred_zero_tb_rate'))} | {_fmt_num(rec.get('actual_double_per_pa'))} | "
            f"{_fmt_num(rec.get('pred_double_per_pa'))} | {_fmt_num(rec.get('actual_hr_per_pa'))} | "
            f"{_fmt_num(rec.get('pred_hr_per_pa'))} |"
        )
    outcome_policy = ((payload.get("distribution_calibrators") or {}).get("outcome_policy") or {}).get("markets") or {}
    lines.extend([
        "",
        "## Hitter Outcome Policy",
        "",
        "| Market | Rows | Base Brier | Learned Brier | Gain | Decision |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for market, rec in outcome_policy.items():
        decision = "use_learned_outcome" if rec.get("use_learned") else "use_baseline_curve"
        lines.append(
            f"| {market} | {rec.get('rows', 0)} | {_fmt_num(rec.get('base_brier'))} | "
            f"{_fmt_num(rec.get('learned_brier'))} | {_fmt_num(rec.get('brier_gain'))} | {decision} |"
        )
    tb_bucket_policy = (
        ((payload.get("distribution_calibrators") or {}).get("outcome_policy") or {})
        .get("bucket_policy", {})
        .get("buckets", {})
    )
    lines.extend([
        "",
        "## TB Event Model Bucket Policy",
        "",
        "| Bucket | Rows | Base Brier | Learned Brier | Gain | Decision |",
        "|---|---:|---:|---:|---:|---|",
    ])
    bucket_rows = sorted(
        tb_bucket_policy.items(),
        key=lambda item: (not bool(item[1].get("use_learned")), -float(item[1].get("brier_gain") or 0.0), -int(item[1].get("rows") or 0)),
    )
    for key, rec in bucket_rows[:40]:
        decision = "use_direct_event_curve" if rec.get("use_learned") else "use_baseline_curve"
        lines.append(
            f"| {str(key).replace('|', ' / ')} | {rec.get('rows', 0)} | "
            f"{_fmt_num(rec.get('base_brier'))} | {_fmt_num(rec.get('learned_brier'))} | "
            f"{_fmt_num(rec.get('brier_gain'))} | {decision} |"
        )
    prob_groups = (((payload.get("distribution_calibrators") or {}).get("probability") or {}).get("groups") or {})
    line_bucket_groups = [
        (key, rec) for key, rec in prob_groups.items()
        if "line_bucket" in (rec.get("columns") or [])
    ]
    line_bucket_groups.sort(key=lambda item: (-int(item[1].get("rows") or 0), item[0]))
    lines.extend([
        "",
        "## Line-Bucket Probability Calibration",
        "",
        "| Group | Rows | Columns | Method | Internal Gain | Holdout Gain | Enabled |",
        "|---|---:|---|---|---:|---:|---|",
    ])
    for key, rec in line_bucket_groups[:40]:
        lines.append(
            f"| {str(key).replace('|', ' / ')} | {rec.get('rows', 0)} | "
            f"{', '.join(map(str, rec.get('columns') or []))} | {rec.get('method', 'raw')} | "
            f"{_fmt_num(rec.get('validation_brier_gain'))} | {_fmt_num(rec.get('holdout_brier_gain'))} | "
            f"{rec.get('holdout_enabled', False)} |"
        )
    lines.extend([
        "",
        "## Exact Bucket Model Selection",
        "",
        "| Bucket | Rows | Decision | Best | Best Brier | ROI | Model | Market | Distribution | Cal Dist | Blend | Side-Line |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for rec in payload.get("bucket_model_selection", [])[:60]:
        lines.append(
            f"| {rec['bucket']} | {rec['rows']} | {rec['decision']} | {rec.get('best_variant') or '-'} | "
            f"{_fmt_num(rec.get('best_brier'))} | {_fmt_pct(rec.get('best_selected_roi'))} | "
            f"{_fmt_num(rec.get('model_brier'))} | {_fmt_num(rec.get('market_brier'))} | "
            f"{_fmt_num(rec.get('distribution_brier'))} | {_fmt_num(rec.get('calibrated_distribution_brier'))} | "
            f"{_fmt_num(rec.get('blend_brier'))} | {_fmt_num(rec.get('event_side_line_brier'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def train(cfg: DistributionConfig) -> dict[str, Any]:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    df = _load(cfg)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "features.mlb_prop_market_training_examples",
        "usage": "shadow_only",
        "rows": int(len(df)),
        "date_min": str(min(df["game_date_et"])) if not df.empty else None,
        "date_max": str(max(df["game_date_et"])) if not df.empty else None,
    }
    if df.empty:
        payload["status"] = "no_rows"
        payload["report_path"] = _write_report(payload, cfg)
        (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    train_df, holdout_df, split = _split(df, cfg)
    payload["split_strategy"] = split
    payload["train_rows"] = int(len(train_df))
    payload["holdout_rows"] = int(len(holdout_df))
    payload["grouped_walk_forward"] = {
        "weighting": "inverse_offer_rows_per_player_game_normalized_to_mean_one",
        "purge_key": "game_slug|player_id",
        "strict_date_split": True,
        "train_player_games": int(train_df.get("player_game_group", pd.Series(dtype=object)).nunique()),
        "holdout_player_games": int(holdout_df.get("player_game_group", pd.Series(dtype=object)).nunique()),
    }
    if len(train_df) < cfg.min_train_rows or len(holdout_df) < cfg.min_holdout_rows:
        payload["status"] = "insufficient_rows"
        payload["report_path"] = _write_report(payload, cfg)
        (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    scored, calibrators = _score_probabilities(train_df, holdout_df, cfg)
    payload["overall"] = _summaries(scored, cfg)
    payload["market"] = _group_summaries(scored, cfg, ["market"])
    payload["market_side"] = _group_summaries(scored, cfg, ["market", "side"])
    production_mask = ~scored["market"].isin(["batter_hits", "batter_total_bases", "batter_home_runs"]) | _true_pair_hitter_mask(scored)
    payload["true_pair_production_evaluation"] = {
        "evidence": "all hitter markets use true-pair non-synthetic holdout only",
        "rows": int(production_mask.sum()),
        "overall": _summaries(scored.loc[production_mask], cfg),
        "market_side": _group_summaries(scored.loc[production_mask], cfg, ["market", "side"]),
    }
    payload["tb_hr_line_production_gates"] = _tb_hr_line_production_gates(scored, cfg)
    payload["bucket_model_selection"] = _bucket_model_selection(scored, cfg)
    payload["distribution_calibrators"] = calibrators
    payload["models"] = {
        "pitcher_strikeouts": {"distribution": "poisson_from_opportunity_adjusted_k_mean"},
        "batter_hits": {"distribution": "nonlinear_event_curve_mixed_over_projected_pa_distribution"},
        "batter_total_bases": {"distribution": "explicit_0_1_2_3_4plus_hr_nonhr_states_from_nonlinear_event_curve"},
        "batter_home_runs": {"distribution": "separate_rare_event_head_with_pa_mixture"},
    }
    payload["status"] = "ready"
    payload["report_path"] = _write_report(payload, cfg)
    (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate MLB prop distribution shadow models")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--holdout-days", type=int, default=28)
    args = parser.parse_args()
    payload = train(DistributionConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        lookback_days=args.lookback_days,
        holdout_days=args.holdout_days,
    ))
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows", 0),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
