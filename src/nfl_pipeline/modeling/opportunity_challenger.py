"""Compare the production opportunity (workload) models with a challenger on the same unseen weeks.

The 26 opportunity models (targets, carries, QB attempts, snap share, limited-usage and spike
probabilities ...) drive the workload adjustment that predict_player_props applies to every projection.
Production was trained on the old training table, whose team-relative, injury and depth features
differed from what live scoring computes (docs/nfl_feature_parity.md). A challenger trained on the
corrected table is judged here: both artifacts score the same holdout rows (weeks neither trained on,
from the corrected table, which now matches live) through the live scoring function, and each model's
MAE is compared with the other's and with its baseline. Promotion is a manual decision.

    python -m nfl_pipeline.modeling.opportunity_challenger --challenger <dir> [--holdout-weeks 4]
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling import train_player_opportunity_models as trainer
from nfl_pipeline.modeling.predict_player_props import _predict_opportunity_projection
from nfl_pipeline.modeling.train_player_stat_models import _temporal_split

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[3]
PRODUCTION = trainer._MODEL_DIR / "nfl_player_opportunity_models.joblib"


def holdout_rows(holdout_weeks: int) -> pd.DataFrame:
    engine = create_engine(PG_DSN.replace("postgresql://", "postgresql+psycopg2://"))
    with engine.connect() as conn:
        df = pd.read_sql(text(trainer.SQL_TRAIN), conn, params={"min_prev_games": 3})
    df = trainer._prepare_targets(df)
    _, holdout = _temporal_split(df, holdout_weeks)
    return holdout


def compare(production: dict, challenger: dict, holdout: pd.DataFrame) -> list[dict]:
    rows = []
    for spec in trainer.OPPORTUNITY_SPECS:
        sub = holdout.loc[holdout["position"].astype(str).str.upper().isin(spec.positions)]
        y = pd.to_numeric(sub.get(spec.target), errors="coerce")
        sub, y = sub.loc[y.notna()], y.dropna()
        if sub.empty:
            continue
        default = float(y.mean())
        preds = {name: _predict_opportunity_projection(sub, {"opportunity_artifact": art}, spec.name, spec.baseline_col, default)
                 for name, art in (("production", production), ("challenger", challenger))}
        base = pd.to_numeric(sub.get(spec.baseline_col), errors="coerce").fillna(default).to_numpy(float)
        mae = {k: float(np.mean(np.abs(v - y.to_numpy(float)))) for k, v in preds.items()}
        mae["baseline"] = float(np.mean(np.abs(base - y.to_numpy(float))))
        accepted = {k: bool(((art.get("metrics") or {}).get(spec.name) or {}).get("accepted"))
                    for k, art in (("production", production), ("challenger", challenger))}
        rows.append(dict(model=spec.name, rows=int(len(sub)), **{f"mae_{k}": v for k, v in mae.items()},
                         change=mae["challenger"] / mae["production"] - 1 if mae["production"] else None,
                         production_accepted=accepted["production"], challenger_accepted=accepted["challenger"]))
    return rows


def markdown(rows: list[dict], weeks: list, challenger_dir: str) -> str:
    L = ["# Opportunity Models: Production vs Parity Challenger", "",
         f"Holdout: {', '.join(weeks)} (out of sample for both). Challenger: `{challenger_dir}`. Both scored with the "
         "live scoring function on the corrected training table. Lower MAE is better.", "",
         "| Model | Rows | Production MAE | Challenger MAE | Baseline MAE | Change |", "|---|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: r["change"] if r["change"] is not None else 0):
        L.append(f"| {r['model']} | {r['rows']} | {r['mae_production']:.4f} | {r['mae_challenger']:.4f} | "
                 f"{r['mae_baseline']:.4f} | {r['change']:+.1%} |")
    better = sum(1 for r in rows if r["change"] is not None and r["change"] < -0.005)
    worse = sum(1 for r in rows if r["change"] is not None and r["change"] > 0.005)
    L += ["", f"Challenger better on {better} models, worse on {worse}, within 0.5% on {len(rows) - better - worse}."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--challenger", required=True, help="directory holding the challenger's joblib")
    ap.add_argument("--holdout-weeks", type=int, default=4)
    args = ap.parse_args()
    folder = Path(args.challenger)
    challenger = joblib.load(folder / "nfl_player_opportunity_models.joblib")
    production = joblib.load(PRODUCTION)
    holdout = holdout_rows(args.holdout_weeks)
    weeks = sorted({f"{int(s)} wk {int(w)}" for s, w in zip(holdout.season, holdout.week)})
    rows = compare(production, challenger, holdout)
    atomic_json(ROOT / "reports" / "nfl_opportunity_challenger_latest.json",
                clean(dict(challenger=str(folder), weeks=weeks, models=rows)))
    text_md = markdown(rows, weeks, folder.as_posix())
    (ROOT / "reports" / "nfl_opportunity_challenger_latest.md").write_bytes(text_md.encode())
    print(text_md)


if __name__ == "__main__":
    main()
