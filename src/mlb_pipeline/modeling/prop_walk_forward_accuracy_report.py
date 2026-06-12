"""Walk-forward audit for locked MLB prop predictions.

The report uses only replay rows that were locked before the result.  CLV is
resolved from immutable close snapshots, and the walk-forward blend is tuned
only on earlier game dates.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2
import psycopg2.extras

from .prop_replay import american_to_prob, ev_per_unit, refresh_prop_replay_clv
from .side_recalibration import price_bucket, prop_line_bucket, prop_line_surface

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"


@dataclass(frozen=True)
class WalkForwardConfig:
    dsn: str = _PG_DSN
    lookback_days: int = 365
    min_ev: float = 0.02
    min_blend_train_rows: int = 40
    min_live_policy_rows: int = 150
    min_live_policy_dates: int = 3
    min_live_policy_brier_gain: float = 0.001
    out: str = "reports/mlb_prop_walk_forward_accuracy_latest.md"
    json_out: str = str(_MODEL_DIR / "prop_walk_forward_accuracy_report.json")
    refresh_clv: bool = True


SQL = """
SELECT
    r.id,
    r.run_id,
    r.source_pred_id,
    r.prediction_key,
    r.prop_offer_id,
    r.lock_snapshot_id,
    r.locked_at_utc,
    r.game_date_et,
    r.game_slug,
    r.player_id,
    r.player_name,
    r.team_abbr,
    r.stat AS market,
    r.side,
    COALESCE(r.bookmaker_key, 'unknown') AS bookmaker_key,
    r.market_line::float AS market_line,
    r.market_price::float AS market_price,
    r.over_price::float AS over_price,
    r.under_price::float AS under_price,
    r.no_vig_prob_over::float AS no_vig_prob_over,
    r.no_vig_prob_under::float AS no_vig_prob_under,
    r.model_prob_over::float AS model_prob_over,
    r.model_prob_side::float AS model_prob_side,
    r.pred_value::float AS pred_value,
    r.pred_count::float AS pred_count,
    COALESCE(r.line_bucket, 'unknown') AS line_bucket,
    COALESCE(r.model_family, 'unknown') AS model_family,
    COALESCE(r.edge_type, 'unknown') AS edge_type,
    r.ev::float AS ev,
    r.actual_value::float AS actual_value,
    r.won,
    COALESCE(r.push, false) AS push,
    r.profit_units::float AS profit_units,
    r.closing_line::float AS closing_line,
    r.closing_price::float AS closing_price,
    r.clv_line::float AS clv_line,
    r.clv_price::float AS clv_price,
    COALESCE(r.clv_valid, false) AS clv_valid,
    r.clv_status,
    r.clv_unknown_reason,
    r.closing_fetched_at_utc,
    r.result_status,
    lu.batting_order AS confirmed_batting_order,
    lu.lineup_source AS confirmed_lineup_source,
    bat_opp.projected_pa,
    bat_opp.pa_games,
    pit_opp.projected_ip,
    pit_opp.projected_bf,
    pit_opp.projected_pitch_count,
    pit_opp.pitcher_starts
FROM bets.mlb_prop_prediction_replay r
LEFT JOIN raw.mlb_lineups lu
  ON lu.game_slug = r.game_slug
 AND lu.team_abbr = r.team_abbr
 AND lu.player_id = r.player_id
LEFT JOIN LATERAL (
    SELECT
        AVG(pa_est)::float AS projected_pa,
        COUNT(*)::int AS pa_games
    FROM (
        SELECT
            GREATEST(COALESCE(gl.at_bats, 0) + COALESCE(gl.walks_batter, 0), 0) AS pa_est
        FROM raw.mlb_player_gamelogs gl
        JOIN raw.mlb_games g ON g.game_slug = gl.game_slug
        WHERE gl.player_id = r.player_id
          AND gl.team_abbr = r.team_abbr
          AND g.status = 'final'
          AND g.game_date_et < r.game_date_et
          AND (COALESCE(gl.at_bats, 0) + COALESCE(gl.walks_batter, 0)) > 0
        ORDER BY g.game_date_et DESC, gl.game_slug DESC
        LIMIT 10
    ) recent_pa
) bat_opp ON r.stat IN ('batter_hits', 'batter_total_bases', 'batter_home_runs')
LEFT JOIN LATERAL (
    SELECT
        AVG(innings_pitched)::float AS projected_ip,
        AVG(bf_est)::float AS projected_bf,
        AVG(bf_est * 3.85)::float AS projected_pitch_count,
        COUNT(*)::int AS pitcher_starts
    FROM (
        SELECT
            gl.innings_pitched,
            GREATEST(
                ROUND(COALESCE(gl.innings_pitched, 0) * 3)
                + COALESCE(gl.hits_allowed, 0)
                + COALESCE(gl.walks_allowed, 0),
                1
            ) AS bf_est
        FROM raw.mlb_player_gamelogs gl
        JOIN raw.mlb_games g ON g.game_slug = gl.game_slug
        WHERE gl.player_id = r.player_id
          AND g.status = 'final'
          AND g.game_date_et < r.game_date_et
          AND gl.is_starter IS TRUE
          AND gl.innings_pitched >= 1.0
        ORDER BY g.game_date_et DESC, gl.game_slug DESC
        LIMIT 5
    ) recent_starts
) pit_opp ON r.stat = 'pitcher_strikeouts'
WHERE r.game_date_et >= %(cutoff)s
  AND r.stat IN ('pitcher_strikeouts', 'batter_hits', 'batter_total_bases', 'batter_home_runs')
  AND r.side IN ('over', 'under')
  AND r.market_line IS NOT NULL
ORDER BY r.game_date_et, r.id
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


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _poisson_over_prob(mean: Any, line: Any) -> float | None:
    lam = _clean_float(mean)
    ln = _clean_float(line)
    if lam is None or ln is None or lam < 0:
        return None
    lam = max(0.0001, min(lam, 20.0))
    threshold = max(0, int(math.floor(ln)) + 1)
    cumulative = 0.0
    prob = math.exp(-lam)
    for k in range(threshold):
        if k == 0:
            prob = math.exp(-lam)
        elif k > 0:
            prob *= lam / k
        cumulative += prob
    return max(1e-6, min(1.0 - 1e-6, 1.0 - cumulative))


def _binom_over_prob(line: Any, n: int, p: float) -> float | None:
    ln = _clean_float(line)
    if ln is None:
        return None
    n = max(1, min(8, int(n)))
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    threshold = math.floor(ln)
    prob = 0.0
    for k in range(threshold + 1, n + 1):
        prob += math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
    return max(1e-6, min(1.0 - 1e-6, prob))


def _compound_tb_over_prob(
    line: Any,
    pa: Any,
    expected_tb: Any,
    expected_hits: Any = None,
    expected_hr: Any = None,
) -> float | None:
    ln = _clean_float(line)
    pa_f = _clean_float(pa)
    tb_f = _clean_float(expected_tb)
    if ln is None or pa_f is None or tb_f is None or pa_f < 1.0 or tb_f < 0:
        return None
    n = max(1, min(7, int(round(pa_f))))
    tb = max(1e-6, tb_f)
    hits_f = _clean_float(expected_hits)
    if hits_f is None:
        hits_f = tb / 1.45
    hits = max(1e-6, min(hits_f, float(n) * 0.65))
    hr_f = _clean_float(expected_hr)
    if hr_f is None:
        hr_f = min(tb / 4.0, hits * 0.15)
    hr = max(0.0, min(hr_f, hits * 0.60, float(n) * 0.20))

    extra_bases = max(0.0, tb - hits)
    doubles = max(0.0, min(hits - hr, extra_bases - 3.0 * hr))
    triples = max(0.0, min((extra_bases - doubles - 3.0 * hr) / 2.0, hits - hr - doubles, 0.03 * n))
    singles = max(0.0, hits - doubles - triples - hr)

    p_single = max(0.0, singles / n)
    p_double = max(0.0, doubles / n)
    p_triple = max(0.0, triples / n)
    p_hr = max(0.0, hr / n)
    total_hit_prob = p_single + p_double + p_triple + p_hr
    if total_hit_prob > 0.92:
        scale = 0.92 / total_hit_prob
        p_single *= scale
        p_double *= scale
        p_triple *= scale
        p_hr *= scale
        total_hit_prob = 0.92
    per_pa = {0: 1.0 - total_hit_prob, 1: p_single, 2: p_double, 3: p_triple, 4: p_hr}
    dist = {0: 1.0}
    for _ in range(n):
        nxt: dict[int, float] = {}
        for current, p_current in dist.items():
            for add, p_add in per_pa.items():
                nxt[current + add] = nxt.get(current + add, 0.0) + p_current * p_add
        dist = nxt
    threshold = math.floor(ln)
    return max(1e-6, min(1.0 - 1e-6, sum(prob for total, prob in dist.items() if total > threshold)))


def _distribution_prob(row: pd.Series) -> float | None:
    mean = _clean_float(row.get("pred_count"))
    if mean is None:
        mean = _clean_float(row.get("pred_value"))
    market = str(row.get("market") or "")
    p_over: float | None = None
    if market == "batter_hits":
        pa = _clean_float(row.get("projected_pa"))
        if pa is not None and mean is not None and pa >= 1.0:
            n = max(1, min(7, int(round(pa))))
            p_over = _binom_over_prob(row.get("market_line"), n, mean / max(float(n), 1.0))
    elif market == "batter_total_bases":
        p_over = _compound_tb_over_prob(row.get("market_line"), row.get("projected_pa"), mean)
    if p_over is None:
        p_over = _poisson_over_prob(mean, row.get("market_line"))
    if p_over is None:
        return None
    return p_over if row.get("side") == "over" else 1.0 - p_over


def _load(cfg: WalkForwardConfig) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, cfg.lookback_days) - 1)
    with psycopg2.connect(cfg.dsn) as conn:
        if not _table_exists(conn, "bets", "mlb_prop_prediction_replay"):
            return pd.DataFrame()
        if cfg.refresh_clv:
            refresh_prop_replay_clv(conn, date_from=cutoff, include_graded=True)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(SQL, {"cutoff": cutoff})
            rows = [dict(r) for r in cur.fetchall()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    for col in (
        "market_line", "market_price", "over_price", "under_price",
        "no_vig_prob_over", "no_vig_prob_under", "model_prob_over",
        "model_prob_side", "pred_value", "pred_count", "ev", "actual_value",
        "profit_units", "clv_line", "clv_price", "projected_pa",
        "projected_ip", "projected_bf", "projected_pitch_count",
    ):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["line_surface"] = [
        prop_line_surface(market, side, line)
        for market, side, line in zip(df["market"], df["side"], df["market_line"])
    ]
    df["line_bucket"] = [
        current if current and current != "unknown" else prop_line_bucket(market, line)
        for current, market, line in zip(df["line_bucket"], df["market"], df["market_line"])
    ]
    df["price_bucket"] = df["market_price"].map(price_bucket)
    df["model_prob_side"] = [
        model if pd.notna(model) else (
            over if side == "over" else (1.0 - over if pd.notna(over) else None)
        )
        for model, over, side in zip(df["model_prob_side"], df["model_prob_over"], df["side"])
    ]
    raw_market = [
        american_to_prob(price)
        for price in df["market_price"]
    ]
    df["market_prob_side"] = [
        nv_over if side == "over" and pd.notna(nv_over)
        else nv_under if side == "under" and pd.notna(nv_under)
        else raw
        for side, nv_over, nv_under, raw in zip(
            df["side"], df["no_vig_prob_over"], df["no_vig_prob_under"], raw_market
        )
    ]
    df["target"] = [
        1.0 if won == True else 0.0 if won == False else None  # noqa: E712
        for won in df["won"]
    ]
    df["push"] = df["push"].fillna(False).astype(bool)
    df["clv_valid"] = df["clv_valid"].fillna(False).astype(bool)
    df["p_distribution"] = df.apply(_distribution_prob, axis=1)
    return df.replace([math.inf, -math.inf], pd.NA)


def _brier(y: pd.Series, p: pd.Series) -> float | None:
    mask = y.notna() & p.notna()
    if not mask.any():
        return None
    diff = p.loc[mask].astype(float).clip(1e-6, 1 - 1e-6) - y.loc[mask].astype(float)
    return float((diff ** 2).mean())


def _best_blend_weight(history: pd.DataFrame) -> float | None:
    work = history.dropna(subset=["target", "model_prob_side", "market_prob_side"])
    work = work.loc[~work["push"]]
    if work.empty:
        return None
    y = work["target"].astype(float)
    model = work["model_prob_side"].astype(float)
    market = work["market_prob_side"].astype(float)
    best_weight = None
    best_score = None
    for step in range(11):
        model_weight = step / 10.0
        p = (model_weight * model + (1.0 - model_weight) * market).clip(1e-6, 1 - 1e-6)
        score = float(((p - y) ** 2).mean())
        if best_score is None or score < best_score:
            best_score = score
            best_weight = model_weight
    return best_weight


def _apply_walk_forward_blend(df: pd.DataFrame, min_train_rows: int) -> pd.DataFrame:
    out = df.copy()
    out["p_walk_forward_blend"] = pd.NA
    out["walk_forward_model_weight"] = pd.NA
    settled = out.loc[out["target"].notna() & ~out["push"]].copy()
    if settled.empty:
        return out

    group_cols = ["game_date_et", "market", "side", "line_surface"]
    for (game_date, market, side, line_surface), group in out.sort_values(["game_date_et", "id"]).groupby(group_cols, dropna=False):
        past = settled.loc[settled["game_date_et"] < game_date]
        if past.empty:
            continue
        exact = past.loc[
            (past["market"] == market)
            & (past["side"] == side)
            & (past["line_surface"] == line_surface)
        ]
        history = exact if len(exact) >= min_train_rows else past.loc[
            (past["market"] == market) & (past["side"] == side)
        ]
        if len(history) < min_train_rows:
            continue
        weight = _best_blend_weight(history)
        if weight is None:
            continue
        idx = group.index
        model_prob = pd.to_numeric(out.loc[idx, "model_prob_side"], errors="coerce")
        market_prob = pd.to_numeric(out.loc[idx, "market_prob_side"], errors="coerce")
        valid = model_prob.notna() & market_prob.notna()
        if not valid.any():
            continue
        valid_idx = model_prob.index[valid]
        blended = (weight * model_prob.loc[valid_idx] + (1.0 - weight) * market_prob.loc[valid_idx]).clip(1e-6, 1.0 - 1e-6)
        out.loc[valid_idx, "walk_forward_model_weight"] = weight
        out.loc[valid_idx, "p_walk_forward_blend"] = blended
    return out


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _forecast_summary(df: pd.DataFrame, prob_col: str) -> dict[str, Any]:
    work = df.loc[df["target"].notna() & ~df["push"]].copy()
    p = pd.to_numeric(work.get(prob_col), errors="coerce")
    y = pd.to_numeric(work.get("target"), errors="coerce")
    mask = p.notna() & y.notna()
    if not mask.any():
        return {"rows": 0, "avg_prob": None, "win_rate": None, "calibration_error": None, "brier": None}
    avg_prob = float(p.loc[mask].mean())
    win_rate = float(y.loc[mask].mean())
    return {
        "rows": int(mask.sum()),
        "avg_prob": avg_prob,
        "win_rate": win_rate,
        "calibration_error": win_rate - avg_prob,
        "brier": _brier(y.loc[mask], p.loc[mask]),
    }


def _selection_summary(df: pd.DataFrame, prob_col: str, min_ev: float) -> dict[str, Any]:
    work = df.loc[df["target"].notna() & ~df["push"]].copy()
    if work.empty:
        return {"picks": 0, "win_rate": None, "roi": None, "avg_ev": None, "clv_beat": None, "avg_clv": None}
    work["variant_ev"] = [
        ev_per_unit(prob, price)
        for prob, price in zip(work.get(prob_col), work.get("market_price"))
    ]
    selected = work.loc[pd.to_numeric(work["variant_ev"], errors="coerce") >= min_ev].copy()
    if selected.empty:
        return {"picks": 0, "win_rate": None, "roi": None, "avg_ev": None, "clv_beat": None, "avg_clv": None}
    valid_clv = selected.loc[selected["clv_valid"] & selected["clv_price"].notna()]
    return {
        "picks": int(len(selected)),
        "win_rate": _mean(selected["target"]),
        "roi": _mean(selected["profit_units"]),
        "avg_ev": _mean(selected["variant_ev"]),
        "clv_beat": _mean((valid_clv["clv_price"] > 0).astype(float)) if not valid_clv.empty else None,
        "avg_clv": _mean(valid_clv["clv_price"]) if not valid_clv.empty else None,
    }


def _variant_summary(df: pd.DataFrame, prob_col: str, cfg: WalkForwardConfig) -> dict[str, Any]:
    return {
        "forecast": _forecast_summary(df, prob_col),
        "selection": _selection_summary(df, prob_col, cfg.min_ev),
    }


def _all_variant_summaries(df: pd.DataFrame, cfg: WalkForwardConfig) -> dict[str, Any]:
    variants = {
        "model_only": "model_prob_side",
        "market_no_vig": "market_prob_side",
        "distribution": "p_distribution",
        "walk_forward_blend": "p_walk_forward_blend",
    }
    return {name: _variant_summary(df, col, cfg) for name, col in variants.items()}


def _clv_summary(df: pd.DataFrame) -> dict[str, Any]:
    valid = df.loc[df["clv_valid"] & df["clv_price"].notna()]
    return {
        "valid_clv_rows": int(len(valid)),
        "clv_coverage": float(len(valid) / len(df)) if len(df) else None,
        "avg_clv_price": _mean(valid["clv_price"]) if not valid.empty else None,
        "clv_price_beat_rate": _mean((valid["clv_price"] > 0).astype(float)) if not valid.empty else None,
        "avg_clv_line": _mean(valid["clv_line"]) if not valid.empty else None,
        "clv_line_beat_rate": _mean((valid["clv_line"] > 0).astype(float)) if not valid.empty else None,
    }


def _group_rows(df: pd.DataFrame, cfg: WalkForwardConfig, cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for values, group in df.groupby(cols, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        key = "|".join(str(v) for v in values)
        variants = _all_variant_summaries(group, cfg)
        blend_weight = _best_blend_weight(group)
        briers = {
            name: rec["forecast"].get("brier")
            for name, rec in variants.items()
            if rec["forecast"].get("brier") is not None
        }
        rows.append({
            "key": key,
            "rows": int(len(group)),
            "graded": int(group["target"].notna().sum()),
            "unique_dates": int(pd.Series(group["game_date_et"]).nunique()),
            "best_variant": min(briers, key=briers.get) if briers else "none",
            "blend_model_weight": blend_weight,
            "variants": variants,
            "clv": _clv_summary(group),
        })
    rows.sort(key=lambda r: (-r["graded"], r["key"]))
    return rows


def _live_policy_from_rows(rows: list[dict[str, Any]], level: str, cfg: WalkForwardConfig) -> dict[str, dict[str, Any]]:
    policy: dict[str, dict[str, Any]] = {}
    for rec in rows:
        graded = int(rec.get("graded") or 0)
        dates = int(rec.get("unique_dates") or 0)
        if graded < cfg.min_live_policy_rows or dates < cfg.min_live_policy_dates:
            continue
        variants = rec.get("variants") or {}
        model_brier = ((variants.get("model_only") or {}).get("forecast") or {}).get("brier")
        best = rec.get("best_variant")
        if best not in {"walk_forward_blend", "market_no_vig", "distribution"} or model_brier is None:
            continue
        best_brier = ((variants.get(best) or {}).get("forecast") or {}).get("brier")
        if best_brier is None:
            continue
        brier_gain = float(model_brier) - float(best_brier)
        if brier_gain < cfg.min_live_policy_brier_gain:
            continue
        if best == "walk_forward_blend" and _clean_float(rec.get("blend_model_weight")) is None:
            continue
        policy[str(rec["key"])] = {
            "level": level,
            "key": rec["key"],
            "variant": best,
            "graded_rows": graded,
            "unique_dates": dates,
            "model_brier": model_brier,
            "variant_brier": best_brier,
            "brier_gain": brier_gain,
            "model_weight": rec.get("blend_model_weight"),
        }
    return policy


def _opportunity_bucket(row: pd.Series) -> str:
    if row.get("market") == "pitcher_strikeouts":
        bf = _clean_float(row.get("projected_bf"))
        if bf is None:
            return "pitcher_bf_unknown"
        if bf < 20:
            return "pitcher_bf_under_20"
        if bf < 24:
            return "pitcher_bf_20_to_23"
        return "pitcher_bf_24_plus"
    pa = _clean_float(row.get("projected_pa"))
    if pa is None:
        return "batter_pa_unknown"
    if pa < 3.8:
        return "batter_pa_under_3_8"
    if pa < 4.3:
        return "batter_pa_3_8_to_4_2"
    return "batter_pa_4_3_plus"


def _opportunity_rows(df: pd.DataFrame, cfg: WalkForwardConfig) -> list[dict[str, Any]]:
    if df.empty:
        return []
    work = df.copy()
    work["opportunity_bucket"] = work.apply(_opportunity_bucket, axis=1)
    rows: list[dict[str, Any]] = []
    for key, group in work.groupby(["market", "opportunity_bucket"], dropna=False):
        variants = _all_variant_summaries(group, cfg)
        rows.append({
            "key": "|".join(str(v) for v in key),
            "rows": int(len(group)),
            "graded": int(group["target"].notna().sum()),
            "avg_projected_pa": _mean(group["projected_pa"]),
            "avg_projected_bf": _mean(group["projected_bf"]),
            "model_brier": variants["model_only"]["forecast"].get("brier"),
            "market_brier": variants["market_no_vig"]["forecast"].get("brier"),
            "model_roi": variants["model_only"]["selection"].get("roi"),
            "valid_clv_rows": _clv_summary(group)["valid_clv_rows"],
        })
    rows.sort(key=lambda r: (-r["graded"], r["key"]))
    return rows


def _fmt_pct(value: Any, digits: int = 1, signed: bool = False) -> str:
    v = _clean_float(value)
    if v is None:
        return "-"
    return f"{v * 100:+.{digits}f}%" if signed else f"{v * 100:.{digits}f}%"


def _fmt_num(value: Any, digits: int = 3, signed: bool = False) -> str:
    v = _clean_float(value)
    if v is None:
        return "-"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def _fmt_clv(value: Any) -> str:
    return _fmt_num(value, digits=2, signed=True)


def _variant_lines(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Variant | Rows | Brier | Cal err | EV picks | ROI | CLV beat | Avg CLV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rec in summary.items():
        forecast = rec.get("forecast") or {}
        selection = rec.get("selection") or {}
        lines.append(
            f"| {name} | {forecast.get('rows', 0)} | {_fmt_num(forecast.get('brier'))} | "
            f"{_fmt_pct(forecast.get('calibration_error'), signed=True)} | "
            f"{selection.get('picks', 0)} | {_fmt_pct(selection.get('roi'), signed=True)} | "
            f"{_fmt_pct(selection.get('clv_beat'))} | {_fmt_clv(selection.get('avg_clv'))} |"
        )
    return lines


def _market_side_lines(rows: list[dict[str, Any]], top_n: int = 40) -> list[str]:
    lines = [
        "| Bucket | Rows | Graded | Dates | Best | Model Brier | Market Brier | Dist Brier | Blend Brier | CLV rows | CLV beat | Avg CLV |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in rows[:top_n]:
        variants = rec["variants"]
        clv = rec["clv"]
        lines.append(
            f"| {rec['key']} | {rec['rows']} | {rec['graded']} | {rec['unique_dates']} | {rec['best_variant']} | "
            f"{_fmt_num(variants['model_only']['forecast'].get('brier'))} | "
            f"{_fmt_num(variants['market_no_vig']['forecast'].get('brier'))} | "
            f"{_fmt_num(variants['distribution']['forecast'].get('brier'))} | "
            f"{_fmt_num(variants['walk_forward_blend']['forecast'].get('brier'))} | "
            f"{clv['valid_clv_rows']} | {_fmt_pct(clv.get('clv_price_beat_rate'))} | {_fmt_clv(clv.get('avg_clv_price'))} |"
        )
    return lines


def _opportunity_lines(rows: list[dict[str, Any]], top_n: int = 30) -> list[str]:
    lines = [
        "| Bucket | Rows | Graded | Avg PA | Avg BF | Model Brier | Market Brier | Model ROI | CLV rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in rows[:top_n]:
        lines.append(
            f"| {rec['key']} | {rec['rows']} | {rec['graded']} | {_fmt_num(rec.get('avg_projected_pa'), 2)} | "
            f"{_fmt_num(rec.get('avg_projected_bf'), 1)} | {_fmt_num(rec.get('model_brier'))} | "
            f"{_fmt_num(rec.get('market_brier'))} | {_fmt_pct(rec.get('model_roi'), signed=True)} | "
            f"{rec['valid_clv_rows']} |"
        )
    return lines


def build_payload(cfg: WalkForwardConfig) -> dict[str, Any]:
    df = _load(cfg)
    if df.empty:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "no_rows",
            "rows": 0,
        }
    scored = _apply_walk_forward_blend(df, cfg.min_blend_train_rows)
    market_side = _group_rows(scored, cfg, ["market", "side"])
    line_surface = _group_rows(scored, cfg, ["market", "side", "line_surface"])
    exact_bucket = _group_rows(
        scored,
        cfg,
        ["market", "side", "line_surface", "line_bucket", "price_bucket", "bookmaker_key"],
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "ready",
        "source": "bets.mlb_prop_prediction_replay",
        "lookback_days": cfg.lookback_days,
        "min_ev": cfg.min_ev,
        "min_blend_train_rows": cfg.min_blend_train_rows,
        "min_live_policy_rows": cfg.min_live_policy_rows,
        "min_live_policy_dates": cfg.min_live_policy_dates,
        "min_live_policy_brier_gain": cfg.min_live_policy_brier_gain,
        "rows": int(len(scored)),
        "graded_rows": int(scored["target"].notna().sum()),
        "pending_rows": int(scored["target"].isna().sum()),
        "date_min": str(scored["game_date_et"].min()),
        "date_max": str(scored["game_date_et"].max()),
        "unique_dates": int(scored["game_date_et"].nunique()),
        "clv": _clv_summary(scored),
        "overall": _all_variant_summaries(scored, cfg),
        "market_side": market_side,
        "line_surface": line_surface,
        "exact_bucket": exact_bucket,
        "opportunity": _opportunity_rows(scored, cfg),
        "clv_unknown_reasons": scored.loc[~scored["clv_valid"], "clv_unknown_reason"].fillna("unknown").value_counts().head(15).to_dict(),
    }
    payload["live_policy"] = {
        "generated_at_utc": payload["generated_at_utc"],
        "min_rows": cfg.min_live_policy_rows,
        "min_dates": cfg.min_live_policy_dates,
        "min_brier_gain": cfg.min_live_policy_brier_gain,
        "line_surface": _live_policy_from_rows(line_surface, "line_surface", cfg),
        "market_side": _live_policy_from_rows(market_side, "market_side", cfg),
        "exact_bucket": _live_policy_from_rows(exact_bucket, "exact_bucket", cfg),
    }
    return payload


def write_report(payload: dict[str, Any], cfg: WalkForwardConfig) -> str:
    out_path = Path(cfg.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parents[3] / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if payload.get("status") != "ready":
        lines = [
            "# MLB Prop Walk-Forward Accuracy",
            "",
            "No locked replay rows were available for this lookback window.",
        ]
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(out_path)

    clv = payload["clv"]
    lines = [
        "# MLB Prop Walk-Forward Accuracy",
        "",
        f"- Generated UTC: {payload['generated_at_utc']}",
        f"- Source: {payload['source']}",
        f"- Date range: {payload['date_min']} to {payload['date_max']}",
        f"- Locked rows: {payload['rows']}",
        f"- Graded rows: {payload['graded_rows']}",
        f"- Pending rows with lock context: {payload['pending_rows']}",
        f"- Unique dates: {payload['unique_dates']}",
        f"- Valid CLV rows: {clv['valid_clv_rows']} ({_fmt_pct(clv.get('clv_coverage'))})",
        f"- Avg CLV price: {_fmt_clv(clv.get('avg_clv_price'))}",
        f"- Live blend policy buckets: {len((payload.get('live_policy') or {}).get('exact_bucket') or {})} exact, {len((payload.get('live_policy') or {}).get('line_surface') or {})} line-surface, {len((payload.get('live_policy') or {}).get('market_side') or {})} market-side",
        "",
        "This audit is walk-forward: blend weights use only earlier game dates, and valid CLV requires same book/player/stat/side/line close snapshots after lock and before first pitch.",
        "",
        "## Overall Probability Variants",
        "",
        *_variant_lines(payload["overall"]),
        "",
        "## Market And Side",
        "",
        *_market_side_lines(payload["market_side"]),
        "",
        "## Line Surface",
        "",
        *_market_side_lines(payload["line_surface"]),
        "",
        "## Exact Bucket",
        "",
        *_market_side_lines(payload["exact_bucket"], top_n=60),
        "",
        "## Opportunity Diagnostics",
        "",
        *_opportunity_lines(payload["opportunity"]),
        "",
        "## CLV Unknown Reasons",
        "",
        "| Reason | Rows |",
        "|---|---:|",
    ]
    for reason, count in payload.get("clv_unknown_reasons", {}).items():
        lines.append(f"| {reason} | {count} |")
    lines.extend([
        "",
        "## Live Probability Policy",
        "",
        "| Level | Bucket | Variant | Rows | Dates | Brier Gain | Model Weight |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    live_policy = payload.get("live_policy") or {}
    policy_rows = []
    for level in ("exact_bucket", "line_surface", "market_side"):
        policy_rows.extend((live_policy.get(level) or {}).values())
    policy_rows.sort(key=lambda rec: (-float(rec.get("brier_gain") or 0.0), rec.get("key") or ""))
    if policy_rows:
        for rec in policy_rows[:40]:
            lines.append(
                f"| {rec.get('level')} | {rec.get('key')} | {rec.get('variant')} | "
                f"{rec.get('graded_rows')} | {rec.get('unique_dates')} | "
                f"{_fmt_num(rec.get('brier_gain'), signed=True)} | {_fmt_num(rec.get('model_weight'))} |"
            )
    else:
        lines.append("| - | - | model_only | - | - | - | - |")
    lines.extend([
        "",
        "## Reading This",
        "",
        "- `model_only` is the locked player-prop probability.",
        "- `market_no_vig` is the book market baseline after removing vig when both sides were available.",
        "- `distribution` prices the exact line from the locked projected count with stat-specific curves; total bases uses a compound PA/single/double/triple/HR shape.",
        "- `walk_forward_blend` picks a model/market weight from prior dates only.",
        "- A bucket is not real-money ready merely because it appears here; it still needs enough graded rows, valid CLV, ROI, calibration, and concentration checks.",
    ])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a no-leakage walk-forward MLB prop accuracy report")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--min-ev", type=float, default=0.02)
    parser.add_argument("--min-blend-train-rows", type=int, default=40)
    parser.add_argument("--min-live-policy-rows", type=int, default=150)
    parser.add_argument("--min-live-policy-dates", type=int, default=3)
    parser.add_argument("--min-live-policy-brier-gain", type=float, default=0.001)
    parser.add_argument("--out", default="reports/mlb_prop_walk_forward_accuracy_latest.md")
    parser.add_argument("--json-out", default=str(_MODEL_DIR / "prop_walk_forward_accuracy_report.json"))
    parser.add_argument("--no-refresh-clv", action="store_true")
    args = parser.parse_args()
    cfg = WalkForwardConfig(
        dsn=args.pg_dsn,
        lookback_days=args.lookback_days,
        min_ev=args.min_ev,
        min_blend_train_rows=args.min_blend_train_rows,
        min_live_policy_rows=args.min_live_policy_rows,
        min_live_policy_dates=args.min_live_policy_dates,
        min_live_policy_brier_gain=args.min_live_policy_brier_gain,
        out=args.out,
        json_out=args.json_out,
        refresh_clv=not args.no_refresh_clv,
    )
    payload = build_payload(cfg)
    report_path = write_report(payload, cfg)
    json_path = Path(cfg.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows", 0),
        "graded_rows": payload.get("graded_rows", 0),
        "valid_clv_rows": (payload.get("clv") or {}).get("valid_clv_rows", 0),
        "report_path": report_path,
        "json_path": str(json_path),
    }, indent=2))


if __name__ == "__main__":
    main()
