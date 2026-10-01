"""Predict NFL game spreads and totals."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from scipy.stats import norm

from nfl_pipeline.integrity import nfl_season
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.offer_selection import CONTRACT as EXECUTION_CONTRACT, MAX_QUOTE_AGE_MINUTES
from nfl_pipeline.game_features import ROLL_WINDOWS
from nfl_pipeline.markets import normalize_team
from nfl_pipeline.modeling.train_game_models import (
    _total_bias_bucket_frame,
    baseline_home_margin,
    baseline_total_points,
    make_game_features,
)
from nfl_pipeline.schema import ensure_schema

log = logging.getLogger("nfl_pipeline.modeling.predict_today")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable",
    category=UserWarning,
)

_ET = ZoneInfo("America/New_York")
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "game_bets"


@dataclass(frozen=True)
class PredictGameConfig:
    pg_dsn: str = PG_DSN
    model_dir: Path = _MODEL_DIR
    model_file: str = "nfl_game_models.joblib"
    et_date: date | None = None
    top_n_per_section: int = 10
    save_predictions: bool = True


def _clean_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        if value is None:
            return None
    try:
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


def _ev_per_unit(prob: Any, price: Any) -> float | None:
    p = _clean_float(prob)
    pr = _clean_float(price)
    if p is None or pr is None or pr == 0:
        return None
    payout = pr / 100.0 if pr > 0 else 100.0 / abs(pr)
    return p * payout - (1.0 - p)


def _no_vig_probability(over_price: Any, under_price: Any, side: str) -> float | None:
    over_raw = _american_to_prob(over_price)
    under_raw = _american_to_prob(under_price)
    if over_raw is None or under_raw is None:
        return None
    total = over_raw + under_raw
    if total <= 0:
        return None
    p_over = over_raw / total
    return float(p_over if side == "over" else 1.0 - p_over)


MARKET_CALIBRATION_PATH = Path(__file__).resolve().parent / "models" / "game_bets" / "market_calibration.json"


def _load_market_calibration() -> dict[str, Any]:
    """Fitted by fit_game_market_calibration; absent file means identity (model probability kept)."""
    try:
        return json.loads(MARKET_CALIBRATION_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _market_trust(calibration: dict[str, Any] | None, market: str) -> float:
    value = _clean_float((((calibration or {}).get("markets") or {}).get(market) or {}).get("probability_trust"))
    return 1.0 if value is None else float(np.clip(value, 0.0, 1.0))


def _apply_market_calibration(choices: list[dict[str, Any]], line: pd.Series, calibration: dict[str, Any] | None) -> None:
    """Pull each side's probability toward FanDuel's no-vig price (logit blend) before EV ranking."""
    for choice in choices:
        if choice["market"] == "spread":
            own, other = line.get(f"spread_{choice['side']}_price"), line.get(
                "spread_away_price" if choice["side"] == "home" else "spread_home_price")
        else:
            own, other = line.get(f"total_{choice['side']}_price"), line.get(
                "total_under_price" if choice["side"] == "over" else "total_over_price")
        market = _no_vig_probability(own, other, "over")  # "over" = the first (own) price's share
        model = float(choice["probability"])
        choice["model_probability"] = model
        choice["market_no_vig_probability"] = market
        trust = _market_trust(calibration, choice["market"])
        choice["market_trust"] = trust
        if market is not None and trust < 1.0 and 0.0 < model < 1.0 and 0.0 < market < 1.0:
            logit = lambda p: math.log(p / (1.0 - p))
            choice["probability"] = 1.0 / (1.0 + math.exp(-(logit(market) + trust * (logit(model) - logit(market)))))


def _calibrated_total_over_probability(
    *,
    raw_p_over: float,
    pred_total: float,
    base_total: float,
    total_line: float,
    over_price: Any,
    under_price: Any,
) -> float:
    raw = float(np.clip(raw_p_over, 0.001, 0.999))
    market_over = _no_vig_probability(over_price, under_price, "over")
    anchor = float(np.clip(market_over if market_over is not None else 0.5, 0.001, 0.999))
    model_edge = abs(float(pred_total) - float(total_line))
    baseline_agreement = abs(float(pred_total) - float(base_total))
    trust = float(np.clip((model_edge - 1.0) / 5.0, 0.20, 0.74))
    if baseline_agreement < 1.25:
        trust *= 0.65
    if abs(raw - anchor) >= 0.15:
        trust *= 0.78
    calibrated = anchor + trust * (raw - anchor)
    tail_cap = 0.62 if model_edge < 3.0 else 0.68
    if model_edge >= 5.5 and baseline_agreement >= 2.0:
        tail_cap = 0.73
    return float(np.clip(calibrated, 1.0 - tail_cap, tail_cap))


def _db_value(value: Any) -> Any:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        if value is None:
            return None
    if isinstance(value, np.generic):
        return value.item()
    return value


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
    deduped = list(dict.fromkeys(legs))
    if len(deduped) < 2:
        return None
    params = "&".join(
        f"marketId[{idx}]={market}&selectionId[{idx}]={selection}"
        for idx, (market, selection) in enumerate(deduped)
    )
    return f"https://sportsbook.fanduel.com/addToBetslip?{params}"


def _apply_total_bias_calibrator(pred: np.ndarray, X: pd.DataFrame, calibrator: dict[str, Any] | None) -> np.ndarray:
    base_pred = np.asarray(pred, dtype=float)
    if not calibrator:
        return base_pred
    exact = {str(k): float(v) for k, v in (calibrator.get("exact") or {}).items()}
    fallback = {str(k): float(v) for k, v in (calibrator.get("fallback") or {}).items()}
    global_correction = _clean_float(calibrator.get("global")) or 0.0
    cap = abs(_clean_float(calibrator.get("cap")) or 4.5)
    try:
        buckets = _total_bias_bucket_frame(X)
    except Exception as exc:
        log.warning("Could not apply NFL total bias calibrator: %s", exc)
        return base_pred
    corrections = np.asarray([
        exact.get(str(row["exact"]), fallback.get(str(row["fallback"]), global_correction))
        for _, row in buckets.iterrows()
    ], dtype=float)
    return base_pred + np.clip(corrections, -cap, cap)


def _load_model(cfg: PredictGameConfig) -> dict[str, Any]:
    from nfl_pipeline.integrity import release_artifact
    released = release_artifact("games")
    if released is not None:
        return released
    path = cfg.model_dir / cfg.model_file
    if not path.exists():
        return {"status": "missing", "models": {}, "metrics": {}, "feature_columns": {}, "fill_values": {}}
    try:
        artifact = joblib.load(path)
    except Exception as exc:
        log.warning("Could not load NFL game model artifact at %s: %s", path, exc)
        return {"status": "load_failed", "models": {}, "metrics": {}, "feature_columns": {}, "fill_values": {}}
    return artifact if isinstance(artifact, dict) else {"status": "bad_artifact", "models": {}}


def _load_history(conn, et_day: date) -> pd.DataFrame:
    return pd.read_sql(
        """
        WITH team_usage AS (
            SELECT
                game_id,
                UPPER(team_abbr) AS team_abbr,
                SUM(COALESCE(pass_attempts, 0))::float AS pass_attempts,
                SUM(COALESCE(carries, 0))::float AS carries,
                SUM(COALESCE(passing_yards, 0))::float AS passing_yards,
                SUM(COALESCE(rushing_yards, 0))::float AS rushing_yards,
                SUM(COALESCE(passing_tds, 0) + COALESCE(rushing_tds, 0))::float AS offensive_tds,
                SUM(COALESCE(red_zone_pass_attempts, 0) + COALESCE(red_zone_carries, 0))::float AS red_zone_plays,
                SUM(COALESCE(red_zone_pass_tds, 0) + COALESCE(red_zone_rush_tds, 0))::float AS red_zone_tds
            FROM raw.nfl_player_gamelogs
            WHERE game_id IS NOT NULL
              AND team_abbr IS NOT NULL
            GROUP BY game_id, UPPER(team_abbr)
        )
        SELECT
            g.game_id, g.season, g.week, g.season_type, g.game_date_et, g.start_ts_utc,
            UPPER(g.home_team_abbr) AS home_team_abbr,
            UPPER(g.away_team_abbr) AS away_team_abbr,
            g.home_score::float AS home_score,
            g.away_score::float AS away_score,
            -g.spread_line::float AS market_spread_home,
            g.total_line::float AS market_total,
            g.roof, g.surface, g.temp::float AS temp, g.wind::float AS wind,
            (hu.pass_attempts + hu.carries)::float AS home_plays,
            CASE WHEN (hu.pass_attempts + hu.carries) > 0 THEN hu.pass_attempts / (hu.pass_attempts + hu.carries) ELSE NULL END AS home_pass_rate,
            CASE WHEN (hu.pass_attempts + hu.carries) > 0 THEN (hu.passing_yards + hu.rushing_yards) / (hu.pass_attempts + hu.carries) ELSE NULL END AS home_yards_per_play,
            CASE WHEN hu.pass_attempts > 0 THEN hu.passing_yards / hu.pass_attempts ELSE NULL END AS home_yards_per_pass,
            CASE WHEN hu.carries > 0 THEN hu.rushing_yards / hu.carries ELSE NULL END AS home_yards_per_carry,
            CASE WHEN (hu.pass_attempts + hu.carries) > 0 THEN hu.offensive_tds / (hu.pass_attempts + hu.carries) ELSE NULL END AS home_tds_per_play,
            hu.red_zone_plays::float AS home_red_zone_plays,
            CASE WHEN hu.red_zone_plays > 0 THEN hu.red_zone_tds / hu.red_zone_plays ELSE NULL END AS home_red_zone_td_rate,
            (au.pass_attempts + au.carries)::float AS away_plays,
            CASE WHEN (au.pass_attempts + au.carries) > 0 THEN au.pass_attempts / (au.pass_attempts + au.carries) ELSE NULL END AS away_pass_rate,
            CASE WHEN (au.pass_attempts + au.carries) > 0 THEN (au.passing_yards + au.rushing_yards) / (au.pass_attempts + au.carries) ELSE NULL END AS away_yards_per_play,
            CASE WHEN au.pass_attempts > 0 THEN au.passing_yards / au.pass_attempts ELSE NULL END AS away_yards_per_pass,
            CASE WHEN au.carries > 0 THEN au.rushing_yards / au.carries ELSE NULL END AS away_yards_per_carry,
            CASE WHEN (au.pass_attempts + au.carries) > 0 THEN au.offensive_tds / (au.pass_attempts + au.carries) ELSE NULL END AS away_tds_per_play,
            au.red_zone_plays::float AS away_red_zone_plays,
            CASE WHEN au.red_zone_plays > 0 THEN au.red_zone_tds / au.red_zone_plays ELSE NULL END AS away_red_zone_td_rate
        FROM raw.nfl_games g
        LEFT JOIN team_usage hu
          ON hu.game_id = g.game_id
         AND hu.team_abbr = UPPER(g.home_team_abbr)
        LEFT JOIN team_usage au
          ON au.game_id = g.game_id
         AND au.team_abbr = UPPER(g.away_team_abbr)
        WHERE g.game_date_et < %(game_date)s
          AND g.status = 'final'
          AND g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
          AND g.home_team_abbr IS NOT NULL
          AND g.away_team_abbr IS NOT NULL
        ORDER BY game_date_et, COALESCE(start_ts_utc, game_date_et::timestamptz), game_id
        """,
        conn,
        params={"game_date": et_day},
    )


def _load_raw_games(conn, et_day: date) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            game_id, season, week, season_type, game_date_et, start_ts_utc,
            UPPER(home_team_abbr) AS home_team_abbr,
            UPPER(away_team_abbr) AS away_team_abbr,
            -spread_line::float AS market_spread_home,
            total_line::float AS market_total,
            roof, surface, temp::float AS temp, wind::float AS wind
        FROM raw.nfl_games
        WHERE game_date_et = %(game_date)s
          AND home_team_abbr IS NOT NULL
          AND away_team_abbr IS NOT NULL
        """,
        conn,
        params={"game_date": et_day},
    )


def _load_game_lines(conn, et_day: date, cutoff: datetime | None = None) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT DISTINCT ON (
            COALESCE(event_id, CONCAT(as_of_date::text, ':', home_team_abbr, ':', away_team_abbr)),
            bookmaker_key
        )
            as_of_date, fetched_at_utc, event_id, commence_time_utc,
            bookmaker_key, bookmaker_title, home_team, away_team,
            home_team_abbr, away_team_abbr,
            spread_home_points::float AS spread_home_points,
            spread_home_price,
            spread_away_points::float AS spread_away_points,
            spread_away_price,
            total_points::float AS total_points,
            total_over_price,
            total_under_price,
            spread_home_link,
            spread_away_link,
            total_over_link,
            total_under_link
        FROM odds.nfl_game_lines
        WHERE as_of_date = %(game_date)s
          AND snapshot_role IN ('open', 'lock', 'live', 'legacy')
          AND bookmaker_key = 'fanduel'
          AND fetched_at_utc <= %(cutoff)s
          AND fetched_at_utc >= %(cutoff)s - %(max_age)s * interval '1 minute'
          AND fetched_at_utc < commence_time_utc
        ORDER BY
            COALESCE(event_id, CONCAT(as_of_date::text, ':', home_team_abbr, ':', away_team_abbr)),
            bookmaker_key,
            fetched_at_utc DESC
        """,
        conn,
        params={"game_date": et_day, "cutoff": cutoff or datetime.now(timezone.utc), "max_age": MAX_QUOTE_AGE_MINUTES},
    )


def _history_by_team(history: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if history.empty:
        return out
    for _, row in history.sort_values(["game_date_et", "start_ts_utc", "game_id"], na_position="last").iterrows():
        home = normalize_team(row.get("home_team_abbr"))
        away = normalize_team(row.get("away_team_abbr"))
        home_score = _clean_float(row.get("home_score"))
        away_score = _clean_float(row.get("away_score"))
        if not home or not away or home_score is None or away_score is None:
            continue
        game_date = pd.to_datetime(row.get("game_date_et"), errors="coerce")
        out.setdefault(home, []).append({
            "game_date_et": game_date,
            "pf": home_score,
            "pa": away_score,
            "margin": home_score - away_score,
            "total": home_score + away_score,
            "plays": _clean_float(row.get("home_plays")),
            "pass_rate": _clean_float(row.get("home_pass_rate")),
            "yards_per_play": _clean_float(row.get("home_yards_per_play")),
            "yards_per_pass": _clean_float(row.get("home_yards_per_pass")),
            "yards_per_carry": _clean_float(row.get("home_yards_per_carry")),
            "tds_per_play": _clean_float(row.get("home_tds_per_play")),
            "red_zone_plays": _clean_float(row.get("home_red_zone_plays")),
            "red_zone_td_rate": _clean_float(row.get("home_red_zone_td_rate")),
        })
        out.setdefault(away, []).append({
            "game_date_et": game_date,
            "pf": away_score,
            "pa": home_score,
            "margin": away_score - home_score,
            "total": home_score + away_score,
            "plays": _clean_float(row.get("away_plays")),
            "pass_rate": _clean_float(row.get("away_pass_rate")),
            "yards_per_play": _clean_float(row.get("away_yards_per_play")),
            "yards_per_pass": _clean_float(row.get("away_yards_per_pass")),
            "yards_per_carry": _clean_float(row.get("away_yards_per_carry")),
            "tds_per_play": _clean_float(row.get("away_tds_per_play")),
            "red_zone_plays": _clean_float(row.get("away_red_zone_plays")),
            "red_zone_td_rate": _clean_float(row.get("away_red_zone_td_rate")),
        })
    return out


def _team_features(history: list[dict[str, Any]], prefix: str, game_date: date) -> dict[str, Any]:
    out: dict[str, Any] = {
        f"{prefix}_game_number": len(history),
        f"{prefix}_rest_days": None,
    }
    if history:
        last_date = pd.to_datetime(history[-1]["game_date_et"], errors="coerce")
        if not pd.isna(last_date):
            out[f"{prefix}_rest_days"] = int((pd.Timestamp(game_date) - last_date.normalize()).days)
    hist_df = pd.DataFrame(history)
    for stat in ("pf", "pa", "margin", "total", "plays", "pass_rate", "yards_per_play", "yards_per_pass", "yards_per_carry", "tds_per_play", "red_zone_td_rate", "red_zone_plays"):
        values = pd.to_numeric(hist_df.get(stat, pd.Series(dtype=float)), errors="coerce").dropna()
        for window in ROLL_WINDOWS:
            out[f"{prefix}_{stat}_avg_{window}"] = float(values.tail(window).mean()) if len(values) else None
    return out


def _current_games(raw_games: pd.DataFrame, lines: pd.DataFrame, et_day: date) -> pd.DataFrame:
    rows: dict[str, dict[str, Any]] = {}
    for _, row in raw_games.iterrows():
        home = normalize_team(row.get("home_team_abbr"))
        away = normalize_team(row.get("away_team_abbr"))
        if not home or not away:
            continue
        key = str(row.get("game_id") or f"{et_day}:{away}:{home}")
        rows[key] = {
            "game_id": key,
            "season": row.get("season"),
            "week": row.get("week"),
            "season_type": row.get("season_type"),
            "game_date_et": et_day,
            "start_ts_utc": row.get("start_ts_utc"),
            "home_team_abbr": home,
            "away_team_abbr": away,
            "market_spread_home": _clean_float(row.get("market_spread_home")),
            "market_total": _clean_float(row.get("market_total")),
            "roof": row.get("roof"),
            "surface": row.get("surface"),
            "temp": _clean_float(row.get("temp")),
            "wind": _clean_float(row.get("wind")),
        }
    for _, row in lines.iterrows():
        home = normalize_team(row.get("home_team_abbr") or row.get("home_team"))
        away = normalize_team(row.get("away_team_abbr") or row.get("away_team"))
        if not home or not away:
            continue
        existing_key = next(
            (
                key for key, rec in rows.items()
                if normalize_team(rec.get("home_team_abbr")) == home
                and normalize_team(rec.get("away_team_abbr")) == away
            ),
            None,
        )
        key = existing_key or str(row.get("event_id") or f"{et_day}:{away}:{home}")
        rec = rows.setdefault(key, {
            "game_id": key,
            "season": None,
            "week": None,
            "season_type": None,
            "game_date_et": et_day,
            "start_ts_utc": row.get("commence_time_utc"),
            "home_team_abbr": home,
            "away_team_abbr": away,
            "market_spread_home": None,
            "market_total": None,
            "roof": None,
            "surface": None,
            "temp": None,
            "wind": None,
        })
        if _clean_float(row.get("spread_home_points")) is not None:
            rec["market_spread_home"] = _clean_float(row.get("spread_home_points"))
        if _clean_float(row.get("total_points")) is not None:
            rec["market_total"] = _clean_float(row.get("total_points"))
        if rec.get("start_ts_utc") is None:
            rec["start_ts_utc"] = row.get("commence_time_utc")
    return pd.DataFrame(rows.values())


def _build_snapshot(history: pd.DataFrame, games: pd.DataFrame, et_day: date) -> pd.DataFrame:
    histories = _history_by_team(history)
    rows: list[dict[str, Any]] = []
    for _, game in games.iterrows():
        home = normalize_team(game.get("home_team_abbr"))
        away = normalize_team(game.get("away_team_abbr"))
        if not home or not away:
            continue
        start = pd.to_datetime(game.get("start_ts_utc"), utc=True, errors="coerce")
        if pd.isna(start) or start <= pd.Timestamp.now(tz="UTC"):
            continue
        row = {
            "game_id": game.get("game_id"),
            "season": game.get("season"),
            "week": game.get("week"),
            "season_type": game.get("season_type"),
            "game_date_et": et_day,
            "start_ts_utc": game.get("start_ts_utc"),
            "home_team_abbr": home,
            "away_team_abbr": away,
            "home_score": None,
            "away_score": None,
            "home_margin": None,
            "total_points_actual": None,
            "market_spread_home": _clean_float(game.get("market_spread_home")),
            "market_total": _clean_float(game.get("market_total")),
            "roof": game.get("roof"),
            "surface": game.get("surface"),
            "temp": _clean_float(game.get("temp")),
            "wind": _clean_float(game.get("wind")),
        }
        row.update(_team_features(histories.get(home, []), "home", et_day))
        row.update(_team_features(histories.get(away, []), "away", et_day))
        rows.append(row)
    return pd.DataFrame(rows)


def _predict_target(snapshot: pd.DataFrame, artifact: dict[str, Any], target: str, baseline: pd.Series) -> tuple[np.ndarray, np.ndarray, bool]:
    base = pd.to_numeric(baseline, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    model = (artifact.get("models") or {}).get(target)
    metrics = (artifact.get("metrics") or {}).get(target) or {}
    accepted = bool(metrics.get("accepted") or metrics.get("projection_pass"))
    columns = (artifact.get("feature_columns") or {}).get(target) or []
    fills = (artifact.get("fill_values") or {}).get(target) or {}
    if model is None or not columns or not accepted:
        return base, base, False
    X_raw = make_game_features(snapshot)
    X = X_raw.reindex(columns=columns).fillna(fills).fillna(0.0)
    if isinstance(model, dict):
        raw_model = model.get("model")
        kind = str(model.get("kind") or "direct")
        shrink = _clean_float(model.get("shrink")) or 0.0
        if kind == "baseline_total_bias":
            pred = _apply_total_bias_calibrator(base, X, model.get("total_bias_calibrator"))
            return pred, base, True
        if raw_model is None:
            return base, base, False
        raw_pred = np.asarray(raw_model.predict(X), dtype=float)
        if kind == "residual":
            pred = base + shrink * raw_pred
        else:
            pred = raw_pred
        if target == "total_points_actual":
            pred = _apply_total_bias_calibrator(pred, X, model.get("total_bias_calibrator"))
        return pred, base, True
    return np.asarray(model.predict(X), dtype=float), base, True


def _line_rows_for_game(lines: pd.DataFrame, game: dict[str, Any]) -> pd.DataFrame:
    if lines.empty:
        return lines
    game_id = str(game.get("game_id") or "")
    home = normalize_team(game.get("home_team_abbr"))
    away = normalize_team(game.get("away_team_abbr"))
    mask = (
        (lines["event_id"].astype(str) == game_id)
        | (
            (lines["home_team_abbr"].map(normalize_team) == home)
            & (lines["away_team_abbr"].map(normalize_team) == away)
        )
    )
    return lines.loc[mask].copy()


def _best_spread_candidate(game: dict[str, Any], pred_margin: float, base_margin: float, sigma: float, line: pd.Series,
                           calibration: dict[str, Any] | None = None) -> dict[str, Any] | None:
    home_spread = _clean_float(line.get("spread_home_points"))
    away_spread = _clean_float(line.get("spread_away_points"))
    if home_spread is None and away_spread is not None:
        home_spread = -away_spread
    if away_spread is None and home_spread is not None:
        away_spread = -home_spread
    if home_spread is None:
        return None
    p_home = float(1.0 - norm.cdf(-home_spread, loc=pred_margin, scale=max(1.0, sigma)))
    p_away = float(1.0 - p_home)
    choices = [
        {
            "market": "spread",
            "side": "home",
            "line": home_spread,
            "price": line.get("spread_home_price"),
            "link": line.get("spread_home_link"),
            "probability": p_home,
            "edge": pred_margin + home_spread,
            "label_team": game.get("home_team_abbr"),
        },
        {
            "market": "spread",
            "side": "away",
            "line": away_spread,
            "price": line.get("spread_away_price"),
            "link": line.get("spread_away_link"),
            "probability": p_away,
            "edge": -(pred_margin + home_spread),
            "label_team": game.get("away_team_abbr"),
        },
    ]
    _apply_market_calibration(choices, line, calibration)
    for choice in choices:
        choice["ev"] = _ev_per_unit(choice["probability"], choice["price"])
    return max(choices, key=lambda rec: (rec.get("ev") is not None, rec.get("ev") or -999.0, abs(rec["edge"])))


def _best_total_candidate(game: dict[str, Any], pred_total: float, base_total: float, sigma: float, line: pd.Series,
                          calibration: dict[str, Any] | None = None) -> dict[str, Any] | None:
    total_line = _clean_float(line.get("total_points"))
    if total_line is None:
        return None
    raw_p_over = float(1.0 - norm.cdf(total_line, loc=pred_total, scale=max(1.0, sigma)))
    p_over = _calibrated_total_over_probability(
        raw_p_over=raw_p_over,
        pred_total=pred_total,
        base_total=base_total,
        total_line=total_line,
        over_price=line.get("total_over_price"),
        under_price=line.get("total_under_price"),
    )
    choices = [
        {
            "market": "total",
            "side": "over",
            "line": total_line,
            "price": line.get("total_over_price"),
            "link": line.get("total_over_link"),
            "probability": p_over,
            "edge": pred_total - total_line,
            "label_team": "OVER",
        },
        {
            "market": "total",
            "side": "under",
            "line": total_line,
            "price": line.get("total_under_price"),
            "link": line.get("total_under_link"),
            "probability": 1.0 - p_over,
            "edge": total_line - pred_total,
            "label_team": "UNDER",
        },
    ]
    _apply_market_calibration(choices, line, calibration)
    for choice in choices:
        choice["ev"] = _ev_per_unit(choice["probability"], choice["price"])
    return max(choices, key=lambda rec: (rec.get("ev") is not None, rec.get("ev") or -999.0, abs(rec["edge"])))


def _prediction_row(game: dict[str, Any], pick: dict[str, Any], *, pred_margin: float, pred_total: float, base_margin: float, base_total: float, artifact: dict[str, Any]) -> dict[str, Any]:
    book = str(game.get("bookmaker_key") or pick.get("book") or "")
    return {
        'execution_contract': EXECUTION_CONTRACT,
        'quote_fetched_at_utc': game.get('quote_fetched_at_utc'),
        'prediction_context_cutoff_utc': game.get('prediction_context_cutoff_utc'),
        'start_ts_utc': game.get('start_ts_utc'),
        "game_date_et": game.get("game_date_et"),
        "season": game.get("season"),
        "week": game.get("week"),
        "game_id": game.get("game_id"),
        "home_team_abbr": game.get("home_team_abbr"),
        "away_team_abbr": game.get("away_team_abbr"),
        "market": pick.get("market"),
        "side": pick.get("side"),
        "book": book,
        "line": pick.get("line"),
        "price": pick.get("price"),
        "link": pick.get("link"),
        "predicted_home_margin": pred_margin,
        "predicted_total_points": pred_total,
        "baseline_home_margin": base_margin,
        "baseline_total_points": base_total,
        "probability": pick.get("probability"),
        "model_probability": pick.get("model_probability"),
        "market_no_vig_probability": pick.get("market_no_vig_probability"),
        "market_trust": pick.get("market_trust"),
        "ev": pick.get("ev"),
        "edge": pick.get("edge"),
        "tier": "paper",
        "model_version": str(artifact.get("version") or artifact.get("status") or "nfl_game_models_v1"),
        "reasons": "nfl_v1_game_projection_only_no_bankroll_gates",
        "prediction_key": "|".join([
            str(game.get("game_date_et")),
            str(game.get("game_id")),
            str(pick.get("market")),
            str(pick.get("side")),
            book,
        ]),
    }


def _save_predictions(conn, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    fields = [
        "game_date_et", "season", "week", "game_id", "home_team_abbr", "away_team_abbr",
        "market", "side", "book", "line", "price", "link",
        "predicted_home_margin", "predicted_total_points",
        "baseline_home_margin", "baseline_total_points",
        "probability", "ev", "edge", "tier", "model_version", "reasons", "prediction_key",
    ]
    from nfl_pipeline.forecast_store import save_forecasts
    return save_forecasts(conn, "nfl_game_predictions", rows, fields)


def build_predictions(cfg: PredictGameConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    et_day = cfg.et_date or datetime.now(_ET).date()
    artifact = _load_model(cfg)
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_schema(conn)
        history = _load_history(conn, et_day)
        raw_games = _load_raw_games(conn, et_day)
        cutoff = datetime.now(timezone.utc)
        lines = _load_game_lines(conn, et_day, cutoff)
        games = _current_games(raw_games, lines, et_day)
        from nfl_pipeline.game_scope import filter_frame
        games = filter_frame(games)
        snapshot = _build_snapshot(history, games, et_day)
        if not snapshot.empty:
            from nfl_pipeline.game_features import GameFeatureConfig, _load_team_injury_context
            injuries = _load_team_injury_context(conn, GameFeatureConfig(min_season=nfl_season(et_day)))
            for prefix in ('home','away'):
                for feature in ('qb_injury_risk','ol_injury_score','skill_injury_score','total_injury_score'):
                    snapshot[f'{prefix}_{feature}'] = [injuries.get((int(r.season),int(r.week),r[f'{prefix}_team_abbr']),{}).get(feature) for _,r in snapshot.iterrows()]
        rows: list[dict[str, Any]] = []
        market_calibration = _load_market_calibration()
        if not snapshot.empty:
            margin_pred, margin_base, margin_accepted = _predict_target(
                snapshot,
                artifact,
                "home_margin",
                baseline_home_margin(snapshot),
            )
            total_pred, total_base, total_accepted = _predict_target(
                snapshot,
                artifact,
                "total_points_actual",
                baseline_total_points(snapshot),
            )
            margin_sigma = _clean_float(((artifact.get("metrics") or {}).get("home_margin") or {}).get("residual_sigma")) or 13.5
            total_sigma = _clean_float(((artifact.get("metrics") or {}).get("total_points_actual") or {}).get("residual_sigma")) or 10.5
            for idx, (_, game_row) in enumerate(snapshot.iterrows()):
                game_dict = game_row.to_dict()
                line_rows = _line_rows_for_game(lines, game_dict)
                for _, line_row in line_rows.iterrows():
                    game_with_book = {
                        **game_dict,
                        'quote_fetched_at_utc': str(line_row.get('fetched_at_utc')),
                        'prediction_context_cutoff_utc': cutoff.isoformat(),
                        "bookmaker_key": line_row.get("bookmaker_key"),
                        "bookmaker_title": line_row.get("bookmaker_title"),
                    }
                    spread_pick = _best_spread_candidate(
                        game_with_book,
                        float(margin_pred[idx]),
                        float(margin_base[idx]),
                        margin_sigma,
                        line_row,
                        market_calibration,
                    )
                    if spread_pick:
                        rows.append(_prediction_row(
                            game_with_book,
                            spread_pick,
                            pred_margin=float(margin_pred[idx]),
                            pred_total=float(total_pred[idx]),
                            base_margin=float(margin_base[idx]),
                            base_total=float(total_base[idx]),
                            artifact=artifact,
                        ))
                    total_pick = _best_total_candidate(
                        game_with_book,
                        float(total_pred[idx]),
                        float(total_base[idx]),
                        total_sigma,
                        line_row,
                        market_calibration,
                    )
                    if total_pick:
                        rows.append(_prediction_row(
                            game_with_book,
                            total_pick,
                            pred_margin=float(margin_pred[idx]),
                            pred_total=float(total_pred[idx]),
                            base_margin=float(margin_base[idx]),
                            base_total=float(total_base[idx]),
                            artifact=artifact,
                        ))
        saved = _save_predictions(conn, rows) if cfg.save_predictions else 0
    meta = {
        "game_date": et_day.isoformat(),
        "history_games": int(len(history)),
        "current_games": int(len(games)),
        "game_lines": int(len(lines)),
        "predictions": int(len(rows)),
        "saved": saved,
        "model_status": artifact.get("status"),
        "accepted_game_models": [
            target for target, rec in (artifact.get("metrics") or {}).items()
            if isinstance(rec, dict) and (rec.get("accepted") or rec.get("projection_pass"))
        ],
    }
    return rows, meta


def _book_label(row: dict[str, Any]) -> str:
    return str(row.get("book") or "").upper() or "BOOK"


def _print_parlay(title: str, rows: list[dict[str, Any]]) -> None:
    url = _build_fd_parlay_url([row.get("link") for row in rows])
    if url:
        print(f"- {title} Parlay: [FD]({url})")


def print_discord(rows: list[dict[str, Any]], meta: dict[str, Any], cfg: PredictGameConfig) -> None:
    print(f"NFL {meta['game_date']} - Game Bets v1")
    print("")
    print("**DATA HEALTH**")
    print(f"- Games: {meta['current_games']} | Game lines: {meta['game_lines']} | Predictions: {meta['predictions']}")
    print(f"- Accepted game models: {', '.join(meta['accepted_game_models']) or 'baseline only'}")
    print("- Bankroll Games: none - NFL game bets stay paper-only until live grading, CLV, and ledger proof exist")
    if not rows:
        print("")
        print("**NFL GAME BETS**")
        print("- No NFL spread/total rows available for this date yet")
        return
    for market, title in (("spread", "Spreads"), ("total", "Totals")):
        picks = [row for row in rows if row.get("market") == market and row.get("ev") is not None]
        picks.sort(key=lambda row: (row.get("ev") or -999.0, abs(row.get("edge") or 0.0)), reverse=True)
        shown = picks[: cfg.top_n_per_section]
        if not shown:
            continue
        print("")
        print(f"**Top {len(shown)} Paper {title}**")
        for row in shown:
            away = row.get("away_team_abbr")
            home = row.get("home_team_abbr")
            side = str(row.get("side") or "").upper()
            if row.get("market") == "spread":
                team = home if row.get("side") == "home" else away
                bet_s = f"{team} {float(row['line']):+.1f}"
                projection_s = f"margin={float(row['predicted_home_margin']):+.1f}"
            else:
                bet_s = f"{side} {float(row['line']):.1f}"
                projection_s = f"total={float(row['predicted_total_points']):.1f}"
            price_s = f" {row['price']}" if row.get("price") is not None else ""
            link = row.get("link")
            link_s = f" [Bet {_book_label(row)}](<{link}>)" if link else ""
            print(
                f"- {away} @ {home}: {bet_s}{price_s} "
                f"{projection_s} p={float(row['probability']):.0%} "
                f"EV={float(row['ev']):+.2f}{link_s}"
            )
        _print_parlay(f"Top {len(shown)} {title}", shown)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict NFL game spreads and totals")
    parser.add_argument("--date", default=None, help="ET date YYYY-MM-DD")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--top-n-per-section", type=int, default=10)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = PredictGameConfig(
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
        print(json.dumps({"meta": meta, "rows": rows[:50]}, indent=2, default=str))


if __name__ == "__main__":
    main()
