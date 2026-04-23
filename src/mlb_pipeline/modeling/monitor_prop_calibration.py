"""
Monitor prop classifier calibration by stat/line bucket and emit bucket controls.

Output file is consumed by predict_player_props.py:
  models/player_props/clf_bucket_controls.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import psycopg2

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"


@dataclass
class MonitorConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    out_file: str = "clf_bucket_controls.json"
    lookback_days: int = 120
    min_samples: int = 40
    max_abs_cal_error: float = 0.08


def _load_rows(conn, cutoff_date: str) -> pd.DataFrame:
    q = """
    SELECT
      game_date_et,
      stat,
      book_line::float AS book_line,
      pred_value::float AS p_over,
      CASE WHEN over_hit THEN 1.0 ELSE 0.0 END AS y
    FROM bets.mlb_prop_predictions
    WHERE game_date_et >= %(cutoff)s
      AND over_hit IS NOT NULL
      AND book_line IS NOT NULL
      AND pred_value IS NOT NULL
      AND pred_value BETWEEN 0.0 AND 1.0
      AND stat IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs','batter_walks')
    """
    return pd.read_sql(q, conn, params={"cutoff": cutoff_date})


def _bucket(stat: str, line: float) -> str:
    if stat == "pitcher_strikeouts":
        if line < 4.5:
            return "K <4.5"
        if line < 6.5:
            return "K 4.5-6.0"
        if line < 8.5:
            return "K 6.5-8.0"
        return "K 8.5+"
    if stat == "batter_total_bases":
        if line < 1.0:
            return "TB 0.5"
        if line < 2.0:
            return "TB 1.5"
        return "TB 2.5+"
    if stat == "batter_hits":
        if line < 1.0:
            return "H 0.5"
        if line < 2.0:
            return "H 1.5"
        return "H 2.5+"
    if stat == "batter_walks":
        if line < 1.0:
            return "BB 0.5"
        return "BB 1.5+"
    if stat == "batter_home_runs":
        if line < 1.0:
            return "HR 0.5"
        return "HR 1.5+"
    return "other"


def monitor(cfg: MonitorConfig) -> dict:
    conn = psycopg2.connect(cfg.pg_dsn)
    cutoff = (datetime.utcnow().date() - timedelta(days=cfg.lookback_days)).isoformat()
    df = _load_rows(conn, cutoff)
    conn.close()

    payload = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "lookback_days": cfg.lookback_days,
        "min_samples": cfg.min_samples,
        "max_abs_cal_error": cfg.max_abs_cal_error,
        "disabled_buckets": {},
        "metrics": [],
    }
    if df.empty:
        payload["status"] = "no_rows"
        return payload

    df["bucket"] = [_bucket(s, float(l)) for s, l in zip(df["stat"], df["book_line"])]
    grp = (
        df.groupby(["stat", "bucket"], as_index=False)
        .agg(
            n=("y", "size"),
            p_mean=("p_over", "mean"),
            win_rate=("y", "mean"),
        )
    )
    grp["abs_cal_error"] = (grp["p_mean"] - grp["win_rate"]).abs()
    grp["disable"] = (grp["n"] >= cfg.min_samples) & (grp["abs_cal_error"] >= cfg.max_abs_cal_error)

    disabled: dict[str, list[str]] = {}
    for _, r in grp.iterrows():
        rec = {
            "stat": str(r["stat"]),
            "bucket": str(r["bucket"]),
            "n": int(r["n"]),
            "p_mean": round(float(r["p_mean"]), 4),
            "win_rate": round(float(r["win_rate"]), 4),
            "abs_cal_error": round(float(r["abs_cal_error"]), 4),
            "disable": bool(r["disable"]),
        }
        payload["metrics"].append(rec)
        if rec["disable"]:
            disabled.setdefault(rec["stat"], []).append(rec["bucket"])
    payload["disabled_buckets"] = disabled
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Monitor prop CLF calibration by stat/line bucket")
    p.add_argument("--pg-dsn", default=_PG_DSN)
    p.add_argument("--model-dir", default=str(_MODEL_DIR))
    p.add_argument("--out-file", default="clf_bucket_controls.json")
    p.add_argument("--lookback-days", type=int, default=120)
    p.add_argument("--min-samples", type=int, default=40)
    p.add_argument("--max-abs-cal-error", type=float, default=0.08)
    args = p.parse_args()

    cfg = MonitorConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        out_file=args.out_file,
        lookback_days=args.lookback_days,
        min_samples=args.min_samples,
        max_abs_cal_error=args.max_abs_cal_error,
    )
    payload = monitor(cfg)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.model_dir / cfg.out_file
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote calibration controls: {out_path}")


if __name__ == "__main__":
    main()
