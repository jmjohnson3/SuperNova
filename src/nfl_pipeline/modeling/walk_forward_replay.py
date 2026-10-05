"""Honest week-by-week replay of the player stat models for a past season.

For each target week it trains the production stat-model trainer (train_player_stat_models.train) on
rows strictly before that week, in a scratch directory, and predicts the week with the same inference
path live scoring uses (predict_player_props._predict_stat). The trainer's own model selection runs
on its trailing holdout weeks inside that history, so nothing from the target week is seen.

Scope: the core stat model only. Live-only layers (season-aware rushing blend, RB role adjustment,
same-day injury context, market calibration) are not replayed.

Output: reports/nfl_walk_forward_<season>_<stats>.csv, one row per player-game and stat, with the model
projection, the baseline it falls back to, whether the layer was accepted that week, and the actual.
Per-week results are cached under reports/walk_forward_cache/, so an interrupted run resumes.

  python -m nfl_pipeline.modeling.walk_forward_replay --season 2025 --stats receiving_yards,rushing_yards
"""
from __future__ import annotations

import argparse
import logging
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import create_engine, text

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.modeling import train_player_stat_models as trainer
from nfl_pipeline.modeling.predict_player_props import _predict_stat

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "reports" / "walk_forward_cache"
log = logging.getLogger(__name__)


def load_rows(pg_dsn: str = PG_DSN) -> pd.DataFrame:
    cfg = trainer.TrainConfig()
    return pd.read_sql(text(trainer.SQL_TRAIN), create_engine(pg_dsn), params={"min_prev_games": cfg.min_prev_games})


def train_before(df: pd.DataFrame, season: int, week: int, stats: list[str], model_dir: Path) -> dict:
    """Run the production trainer on rows before (season, week) only; return its saved payload."""
    history = df[(df.season < season) | ((df.season == season) & (df.week < week))].copy()
    specs = [s for s in trainer.STAT_SPECS if s.stat in stats]
    original_read_sql, original_specs = trainer.pd.read_sql, trainer.STAT_SPECS
    trainer.pd.read_sql = lambda *a, **k: history.copy()
    trainer.STAT_SPECS = specs
    try:
        cfg = replace(trainer.TrainConfig(), model_dir=model_dir)
        trainer.train(cfg)
    finally:
        trainer.pd.read_sql, trainer.STAT_SPECS = original_read_sql, original_specs
    return joblib.load(model_dir / cfg.out_file)


def predict_week(df: pd.DataFrame, payload: dict, season: int, week: int, stats: list[str]) -> pd.DataFrame:
    out = []
    specs = {s.stat: s for s in trainer.STAT_SPECS}
    for stat in stats:
        rows = df[(df.season == season) & (df.week == week)
                  & df.position.astype(str).str.upper().isin(specs[stat].positions)].copy()
        rows = rows[pd.to_numeric(rows[stat], errors="coerce").notna()]
        if rows.empty:
            continue
        pred, baseline, accepted = _predict_stat(rows.reset_index(drop=True), payload, stat)
        out.append(pd.DataFrame(dict(
            season=season, week=week, game_id=rows.game_id.values, player_id=rows.player_id.values,
            position=rows.position.values, stat=stat, projection=pred, baseline=baseline, accepted=accepted,
            variant=((payload.get("metrics") or {}).get(stat) or {}).get("variant"),
            actual=pd.to_numeric(rows[stat], errors="coerce").values)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--stats", default="receiving_yards,rushing_yards")
    ap.add_argument("--first-week", type=int, default=3)
    ap.add_argument("--last-week", type=int, default=18)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    stats = [s.strip() for s in args.stats.split(",") if s.strip()]
    slug = "_".join(sorted(stats))
    CACHE.mkdir(parents=True, exist_ok=True)
    df = load_rows()
    parts = []
    for week in range(args.first_week, args.last_week + 1):
        cached = CACHE / f"{args.season}_w{week:02d}_{slug}.csv"
        if cached.exists():
            parts.append(pd.read_csv(cached))
            continue
        started = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            payload = train_before(df, args.season, week, stats, Path(tmp))
        part = predict_week(df, payload, args.season, week, stats)
        part.to_csv(cached, index=False)
        parts.append(part)
        print(f"week {week}: {len(part)} rows, accepted "
              f"{part.groupby('stat').accepted.first().to_dict() if not part.empty else {}}, {time.time() - started:.0f}s", flush=True)
    result = pd.concat(parts, ignore_index=True)
    out = ROOT / "reports" / f"nfl_walk_forward_{args.season}_{slug}.csv"
    result.to_csv(out, index=False)
    print(f"wrote {out} ({len(result)} rows)")


if __name__ == "__main__":
    main()
