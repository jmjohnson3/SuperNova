"""Train NFL exact-line prop betting models from locked, graded prop offers."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sqlalchemy import create_engine, text

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_joblib, atomic_json
from nfl_pipeline.modeling.exact_line_evidence import SQL_EVIDENCE, audit_evidence

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
DEFAULT_JOBLIB = MODEL_DIR / "exact_line_challengers" / "latest.joblib"
DEFAULT_JSON = MODEL_DIR / "nfl_prop_exact_line_models.json"
DEFAULT_MD = ROOT / "reports" / "nfl_prop_exact_line_models_latest.md"


@dataclass(frozen=True)
class ExactLineModelConfig:
    pg_dsn: str = PG_DSN
    model_file: Path = DEFAULT_JOBLIB
    report_file: Path = DEFAULT_JSON
    md_report_file: Path = DEFAULT_MD
    min_rows: int = 200
    min_holdout_rows: int = 40
    min_clv_rows: int = 80
    random_state: int = 42


SQL_TRAIN = SQL_EVIDENCE


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
    return 100.0 / (price_f + 100.0) if price_f > 0 else abs(price_f) / (abs(price_f) + 100.0)


def _price_bucket(price: Any) -> str:
    val = _clean_float(price)
    if val is None:
        return "missing_price"
    if val <= -200:
        return "lay_200_plus"
    if val <= -150:
        return "lay_150_199"
    if val < 100:
        return "fair_lay_100_149"
    if val < 150:
        return "plus_100_149"
    if val < 250:
        return "plus_150_249"
    return "plus_250_plus"


def _line_bucket(stat: str, line: Any) -> str:
    val = _clean_float(line)
    if val is None:
        return "missing_line"
    if stat.endswith("_tds"):
        return "td_0_5" if val <= 0.5 else "td_alt_1_5_plus"
    if stat == "passing_yards":
        base = int(val // 25 * 25)
        return f"pass_yards_{base}_{base + 24}"
    if stat in {"rushing_yards", "receiving_yards"}:
        base = int(val // 10 * 10)
        return f"yards_{base}_{base + 9}"
    return f"line_{val:g}"


def _no_vig_prob(row: pd.Series) -> float | None:
    over_p = _american_to_prob(row.get("lock_over_price"))
    under_p = _american_to_prob(row.get("lock_under_price"))
    if over_p is None or under_p is None or over_p + under_p <= 0:
        return None
    over_nv = over_p / (over_p + under_p)
    return over_nv if str(row.get("side") or "").lower() == "over" else 1.0 - over_nv


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_win"] = (out["result"].astype(str).str.lower() == "win").astype(int)
    out["target_clv_beat"] = (
        (out["clv_status"] == "valid_close")
        & (pd.to_numeric(out["clv_prob_delta"], errors="coerce") > 0)
    ).astype(float).where(out['clv_status'].eq('valid_close') & out.clv_prob_delta.notna())
    out["market_probability"] = out["price"].map(_american_to_prob)
    out["no_vig_probability"] = out.apply(_no_vig_prob, axis=1)
    out["line_bucket"] = out.apply(lambda row: _line_bucket(str(row.get("stat") or ""), row.get("line")), axis=1)
    out["price_bucket"] = out["price"].map(_price_bucket)
    out["model_market_delta"] = pd.to_numeric(out["model_probability"], errors="coerce") - pd.to_numeric(out["no_vig_probability"], errors="coerce")
    out["projection_edge_ratio"] = pd.to_numeric(out["model_edge"], errors="coerce") / pd.to_numeric(out["line"], errors="coerce").replace(0, np.nan)
    return out


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "stat", "side", "book", "line_bucket", "price_bucket", "model_version",
        "line", "price", "projection", "baseline_projection", "model_probability",
        "model_ev", "model_edge", "market_probability", "no_vig_probability",
        "model_market_delta", "projection_edge_ratio",
    ]
    X = df.reindex(columns=cols).copy()
    for col in ("stat", "side", "book", "line_bucket", "price_bucket", "model_version"):
        X[col] = X[col].astype(str).str.lower()
    return pd.get_dummies(X, columns=["stat", "side", "book", "line_bucket", "price_bucket", "model_version"], dummy_na=True).replace([np.inf, -np.inf], np.nan)


def _split_by_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weeks = sorted(set(zip(df.season, df.week)))
    if len(weeks) < 3:
        return df.iloc[0:0], df.iloc[0:0]
    held = set(weeks[-max(1, int(math.ceil(len(weeks) * .25))):])
    mask = pd.Series([(s, w) in held for s, w in zip(df.season, df.week)], index=df.index)
    holdout = df.loc[mask].copy()
    cutoff = pd.to_datetime(holdout.created_at_utc, utc=True).min()
    train = df.loc[~mask & ~df.game_id.isin(holdout.game_id)
                   & pd.to_datetime(df.label_available_at, utc=True, errors='coerce').lt(cutoff)].copy()
    return train, holdout


def _fit_classifier(X: pd.DataFrame, y: pd.Series, cfg: ExactLineModelConfig, weights=None) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=0.10,
        random_state=cfg.random_state,
    ).fit(X, y, sample_weight=weights)


def _weights(df):
    return 1 / df.groupby(['game_id', 'player_id']).id.transform('size')


def _predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    proba = np.asarray(model.predict_proba(X), dtype=float)
    return np.clip(proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel(), 0.001, 0.999)


def _auc(y_true: pd.Series, prob: np.ndarray) -> float | None:
    try:
        if y_true.nunique() < 2:
            return None
        return float(roc_auc_score(y_true, prob))
    except Exception:
        return None


def _train_group(df: pd.DataFrame, cfg: ExactLineModelConfig, *, stat: str, side: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sub = df.loc[(df["stat"] == stat) & (df["side"] == side)].copy()
    if len(sub) < cfg.min_rows:
        return {
            "status": "insufficient_rows",
            "stat": stat,
            "side": side,
            "rows": int(len(sub)),
            "unique_player_games": int(len(sub.drop_duplicates(['game_id', 'player_id']))),
            "min_rows": cfg.min_rows,
            "accepted": False,
        }, None
    train, holdout = _split_by_date(sub)
    train_games = len(train.drop_duplicates(['game_id', 'player_id']))
    holdout_games = len(holdout.drop_duplicates(['game_id', 'player_id']))
    if train_games < cfg.min_rows or holdout_games < cfg.min_holdout_rows:
        return {
            "status": "insufficient_split_rows",
            "stat": stat,
            "side": side,
            "train_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "train_player_games": train_games,
            "holdout_player_games": holdout_games,
            "accepted": False,
        }, None
    X_train_raw = _feature_frame(train)
    X_holdout_raw = _feature_frame(holdout)
    columns = list(X_train_raw.columns)
    fills = {
        col: float(value) if math.isfinite(float(value)) else 0.0
        for col, value in X_train_raw.median(numeric_only=True).fillna(0.0).to_dict().items()
    }
    X_train = X_train_raw.reindex(columns=columns).fillna(fills).fillna(0.0)
    X_holdout = X_holdout_raw.reindex(columns=columns).fillna(fills).fillna(0.0)
    y_train = train["target_win"].astype(int)
    y_holdout = holdout["target_win"].astype(int)
    if y_train.nunique() < 2:
        return dict(status='single_class_training', stat=stat, side=side, accepted=False), None
    win_model = _fit_classifier(X_train, y_train, cfg, _weights(train))
    win_prob = _predict_proba(win_model, X_holdout)
    model_prob = pd.to_numeric(holdout["model_probability"], errors="coerce").fillna(0.5).clip(0.001, 0.999)
    market_prob = pd.to_numeric(holdout["no_vig_probability"], errors="coerce").fillna(0.5).clip(0.001, 0.999)
    brier = float(brier_score_loss(y_holdout, win_prob, sample_weight=_weights(holdout)))
    model_brier = float(brier_score_loss(y_holdout, model_prob, sample_weight=_weights(holdout)))
    market_brier = float(brier_score_loss(y_holdout, market_prob, sample_weight=_weights(holdout)))
    model_payload: dict[str, Any] = {
        "kind": "exact_line_win_classifier",
        "stat": stat,
        "side": side,
        "model": win_model,
        "feature_columns": columns,
        "fill_values": fills,
    }
    clv_summary: dict[str, Any] = {"status": "insufficient_clv_rows", "rows": 0}
    clv_model = None
    cutoff = pd.to_datetime(holdout.created_at_utc, utc=True).min()
    valid_clv_train = train.loc[train.target_clv_beat.notna()
        & pd.to_datetime(train.clv_available_at, utc=True, errors='coerce').lt(cutoff)].copy()
    valid_clv_holdout = holdout.loc[holdout.target_clv_beat.notna()].copy()
    if (len(valid_clv_train.drop_duplicates(['game_id', 'player_id'])) >= cfg.min_clv_rows
            and len(valid_clv_holdout.drop_duplicates(['game_id', 'player_id'])) >= max(20, cfg.min_holdout_rows // 2)):
        Xc_train = _feature_frame(valid_clv_train).reindex(columns=columns).fillna(fills).fillna(0.0)
        Xc_holdout = _feature_frame(valid_clv_holdout).reindex(columns=columns).fillna(fills).fillna(0.0)
        yc_train = valid_clv_train["target_clv_beat"].astype(int)
        yc_holdout = valid_clv_holdout["target_clv_beat"].astype(int)
        if yc_train.nunique() >= 2:
            clv_model = _fit_classifier(Xc_train, yc_train, cfg, _weights(valid_clv_train))
            clv_prob = _predict_proba(clv_model, Xc_holdout)
            clv_base = np.full(len(yc_holdout), np.average(yc_train, weights=_weights(valid_clv_train)))
            clv_brier = float(brier_score_loss(yc_holdout, clv_prob, sample_weight=_weights(valid_clv_holdout)))
            clv_base_brier = float(brier_score_loss(yc_holdout, clv_base, sample_weight=_weights(valid_clv_holdout)))
            clv_summary = {
                "status": "trained",
                "train_rows": int(len(valid_clv_train)),
                "holdout_rows": int(len(valid_clv_holdout)),
                "brier": clv_brier,
                "baseline_brier": clv_base_brier,
                "auc": _auc(yc_holdout, clv_prob),
                "accepted": clv_brier <= clv_base_brier - .001,
            }
    if clv_model is not None and clv_summary.get('accepted'):
        model_payload["clv_model"] = clv_model
    metric = {
        "status": "trained",
        "stat": stat,
        "side": side,
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "brier": brier,
        "model_probability_brier": model_brier,
        "market_no_vig_brier": market_brier,
        "auc": _auc(y_holdout, win_prob),
        "accepted": bool(brier <= min(model_brier, market_brier) - 0.001),
        "clv": clv_summary,
    }
    return metric, model_payload


def _bucket_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    grouped = []
    keys = ["stat", "side", "book", "line_bucket", "price_bucket"]
    for key, sub in df.groupby(keys, dropna=False):
        y = sub["target_win"].astype(float)
        model_prob = pd.to_numeric(sub["model_probability"], errors="coerce").fillna(0.5).clip(0.001, 0.999)
        market_prob = pd.to_numeric(sub["no_vig_probability"], errors="coerce").fillna(0.5).clip(0.001, 0.999)
        clv = sub.loc[sub.target_clv_beat.notna()].copy()
        grouped.append({
            "stat": key[0],
            "side": key[1],
            "book": key[2],
            "line_bucket": key[3],
            "price_bucket": key[4],
            "rows": int(len(sub)),
            "unique_player_games": int(len(sub.drop_duplicates(['game_id', 'player_id']))),
            "win_rate": float(y.mean()) if len(y) else None,
            "roi": float(pd.to_numeric(sub["profit_per_unit"], errors="coerce").fillna(0.0).mean()) if len(sub) else None,
            "model_probability_brier": float(brier_score_loss(y, model_prob)) if len(y) else None,
            "market_no_vig_brier": float(brier_score_loss(y, market_prob)) if len(y) else None,
            "valid_clv_rows": int(len(clv)),
            "clv_beat_rate": float((pd.to_numeric(clv["clv_prob_delta"], errors="coerce") > 0).mean()) if len(clv) else None,
            "avg_clv_prob_delta": float(pd.to_numeric(clv["clv_prob_delta"], errors="coerce").mean()) if len(clv) else None,
        })
    grouped.sort(key=lambda row: (row["rows"], row.get("roi") or -99, row.get("avg_clv_prob_delta") or -99), reverse=True)
    return grouped


def _write_markdown(payload: dict[str, Any], path: Path) -> str:
    def metric(value, spec='.3f'):
        return '-' if value is None else format(float(value), spec)
    lines = [
        "# NFL Exact-Line Prop Models",
        "",
        "These models train only after real NFL player prop offers have been locked, graded, and matched to true paired prices.",
        "",
        f"- Status: {payload.get('status')}",
        f"- Training rows: {payload.get('rows')}",
        f"- True-paired rows: {payload.get('true_paired_rows')}",
        f"- Trained at: {payload.get('trained_at_utc')}",
        "- Deployment: offline challenger only; no bankroll or micro approval",
        f"- Evidence audit: {json.dumps({k: v for k, v in payload.get('evidence_audit', {}).items() if k not in ('excluded_rows', 'eligible_prediction_ids')})}",
        "",
        "## Stat-Side Models",
        "",
        "| Stat | Side | Status | Rows | Train | Holdout | Brier | Model Brier | Market Brier | AUC | Accepted | CLV Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for rec in payload.get("metrics", []):
        auc = rec.get("auc")
        auc_s = "-" if auc is None else f"{float(auc):.3f}"
        lines.append(
            f"| {rec.get('stat')} | {rec.get('side')} | {rec.get('status')} | "
            f"{rec.get('rows', int(rec.get('train_rows') or 0)+int(rec.get('holdout_rows') or 0))} | "
            f"{int(rec.get('train_rows') or 0)} | {int(rec.get('holdout_rows') or 0)} | "
            f"{metric(rec.get('brier'))} | {metric(rec.get('model_probability_brier'))} | "
            f"{metric(rec.get('market_no_vig_brier'))} | "
            f"{auc_s} | "
            f"{'yes' if rec.get('accepted') else 'no'} | {(rec.get('clv') or {}).get('status', '-')} |"
        )
    lines.extend([
        "",
        "## Exact Buckets",
        "",
        "| Stat | Side | Book | Line Bucket | Price Bucket | Rows | Win% | ROI | Model Brier | Market Brier | CLV Rows | CLV Beat | Avg CLV |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload.get("bucket_summary", [])[:80]:
        lines.append(
            f"| {row.get('stat')} | {row.get('side')} | {row.get('book')} | {row.get('line_bucket')} | "
            f"{row.get('price_bucket')} | {int(row.get('rows') or 0)} | "
            f"{float(row.get('win_rate') or 0):.1%} | {float(row.get('roi') or 0):+.3f} | "
            f"{float(row.get('model_probability_brier') or 0):.3f} | {float(row.get('market_no_vig_brier') or 0):.3f} | "
            f"{int(row.get('valid_clv_rows') or 0)} | {metric(row.get('clv_beat_rate'), '.1%')} | "
            f"{metric(row.get('avg_clv_prob_delta'), '+.3f')} |"
        )
    if not payload.get("bucket_summary"):
        lines.append("| none | - | - | - | - | 0 | 0.0% | 0.000 | 0.000 | 0.000 | 0 | 0.0% | 0.000 |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def train(cfg: ExactLineModelConfig) -> dict[str, Any]:
    if cfg.model_file.resolve() == (MODEL_DIR / 'nfl_prop_exact_line_models.joblib').resolve():
        raise ValueError('Training evidence must not overwrite the live exact-line artifact')
    cfg.model_file.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(cfg.pg_dsn)
    try:
        with engine.connect() as conn:
            conn.execute(text("SET statement_timeout='90s'"))
            raw = pd.read_sql(text(SQL_TRAIN), conn)
    finally:
        engine.dispose()
    df, audit = audit_evidence(raw)
    df = _prepare_frame(df) if not df.empty else df
    payload: dict[str, Any] = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "waiting_for_prop_history",
        "rows": int(len(df)),
        "true_paired_rows": int(len(df)),
        "evidence_audit": audit,
        "deployment_approved": False,
        "betting_approved": False,
        "min_rows": cfg.min_rows,
        "metrics": [],
        "bucket_summary": _bucket_summary(df) if not df.empty else [],
    }
    model_payload: dict[str, Any] = {
        "status": payload["status"],
        "trained_at_utc": payload["trained_at_utc"],
        "models": {},
        "metrics": {},
        "deployment_approved": False,
        "betting_approved": False,
    }
    if len(df):
        metrics: list[dict[str, Any]] = []
        for stat in sorted(df["stat"].dropna().astype(str).unique()):
            for side in ("over", "under"):
                metric, model = _train_group(df, cfg, stat=stat, side=side)
                metrics.append(metric)
                if model is not None:
                    key = f"{stat}|{side}"
                    model_payload["models"][key] = model
                    model_payload["metrics"][key] = metric
        payload["metrics"] = metrics
        payload["status"] = "challenger_evaluated" if any(rec.get("status") == "trained" for rec in metrics) else "insufficient_independent_history"
        model_payload["status"] = payload["status"]
    cfg.report_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(cfg.report_file, payload)
    atomic_joblib(cfg.model_file, model_payload)
    _write_markdown(payload, cfg.md_report_file)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NFL exact-line prop betting models")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--model-file", default=str(DEFAULT_JOBLIB))
    parser.add_argument("--report-file", default=str(DEFAULT_JSON))
    parser.add_argument("--md-report-file", default=str(DEFAULT_MD))
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--min-holdout-rows", type=int, default=40)
    parser.add_argument("--min-clv-rows", type=int, default=80)
    args = parser.parse_args()
    payload = train(ExactLineModelConfig(
        pg_dsn=args.pg_dsn,
        model_file=Path(args.model_file),
        report_file=Path(args.report_file),
        md_report_file=Path(args.md_report_file),
        min_rows=args.min_rows,
        min_holdout_rows=args.min_holdout_rows,
        min_clv_rows=args.min_clv_rows,
    ))
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
