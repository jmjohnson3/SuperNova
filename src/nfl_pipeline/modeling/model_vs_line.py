"""Phase A: does the stat model add information beyond FanDuel's line and the sharp price?

Joins honest week-by-week projections (walk_forward_replay) to real 2025 FanDuel/Pinnacle prop quotes
(sharp_gap_backtest sides) and answers:

1. Model vs line: when the model disagrees with FanDuel's line, does the result follow the model?
   (correlation of model-minus-line with actual-minus-line; how often the model's side wins, by size of
   disagreement). One row per player-game at the T-15 snapshot.
2. Confluence: among FanDuel sides priced off Pinnacle (>= +1% / >= +3% EV), do the ones the model agrees
   with hold up better (EV at the sharp close, win rate) than the ones it disagrees with?

  python -m nfl_pipeline.modeling.model_vs_line --season 2025
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json

ROOT = Path(__file__).resolve().parents[3]
MARKETS = {"receiving_yards": "player_reception_yds", "rushing_yards": "player_rush_yds"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _strip(name: str) -> str:
    return " ".join(w for w in str(name or "").split() if w not in SUFFIXES)


def load(season: int, stats: list[str], variant: str = "") -> pd.DataFrame:
    wf = pd.read_csv(ROOT / "reports" / f"nfl_walk_forward_{season}_{'_'.join(sorted(stats))}{variant}.csv")
    with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT player_id, player_name_norm FROM raw.nfl_rosters WHERE season=%s AND player_name_norm IS NOT NULL", (season,))
        names = pd.DataFrame(cur.fetchall(), columns=["player_id", "player_norm"])
    names["player_key"] = names.player_norm.map(_strip)
    wf = wf.merge(names.drop_duplicates("player_id")[["player_id", "player_key"]], on="player_id", how="left")
    sides = []
    for stat in stats:
        path = ROOT / "reports" / f"nfl_sharp_gap_backtest_{season}_{MARKETS[stat]}_sides.csv"
        if path.exists():
            sides.append(pd.read_csv(path))
    sides = pd.concat(sides, ignore_index=True)
    sides["player_key"] = sides.player_norm.map(_strip)
    joined = sides.merge(wf[["game_id", "stat", "player_key", "projection", "baseline", "accepted", "actual"]]
                         .rename(columns={"actual": "wf_actual"}),
                         on=["game_id", "stat", "player_key"], how="inner")
    joined["model_minus_line"] = joined.projection - joined.line
    joined["model_side"] = np.where(joined.model_minus_line > 0, "over", "under")
    joined["agrees"] = joined.model_side == joined.side
    return joined, sides


def model_vs_line(j: pd.DataFrame) -> list[dict]:
    rows = []
    one = j[(j.minutes == 15) & (j.side == "over") & j.actual.notna()].drop_duplicates(["game_id", "player_key", "stat"])
    for stat, g in one.groupby("stat"):
        d_actual = g.actual - g.line
        decided = d_actual != 0
        rec = dict(stat=stat, n=int(len(g)), corr=float(np.corrcoef(g.model_minus_line, d_actual)[0, 1]),
                   corr_baseline=float(np.corrcoef(g.baseline - g.line, d_actual)[0, 1]),
                   model_mae=float((g.projection - g.actual).abs().mean()), mean_bias=float(g.model_minus_line.mean()))
        for lo in (0.0, 5.0, 10.0, 15.0):
            sub = g[(g.model_minus_line.abs() >= lo) & decided]
            hit = ((sub.model_minus_line > 0) == (sub.actual > sub.line)).mean() if len(sub) else None
            rec[f"model_side_wins_ge{int(lo)}"] = (None if hit is None else float(hit), int(len(sub)))
        rows.append(rec)
    return rows


def confluence(j: pd.DataFrame) -> list[dict]:
    rows = []
    for stat, g0 in j.groupby("stat"):
        for label, g in (("all sides", g0), (">= +1%", g0[g0.ev >= 0.01]), (">= +3%", g0[g0.ev >= 0.03])):
            for agree_label, sub in (("model agrees", g[g.agrees]), ("model disagrees", g[~g.agrees]),
                                     ("agrees by 5+", g[g.agrees & (g.model_minus_line.abs() >= 5)])):
                units = pd.to_numeric(sub.units, errors="coerce").dropna()
                evc = pd.to_numeric(sub.ev_at_sharp_close, errors="coerce").dropna()
                rows.append(dict(stat=stat, bucket=label, group=agree_label, n=int(len(sub)),
                                 mean_ev=float(sub.ev.mean()) if len(sub) else None,
                                 ev_at_sharp_close=float(evc.mean()) if len(evc) else None,
                                 win_rate=float((sub.result == "win").sum() / max(1, sub.result.isin(["win", "loss"]).sum())) if len(sub) else None,
                                 roi=float(units.mean()) if len(units) else None, settled=int(len(units))))
    return rows


def markdown(season, matched, total, a, b) -> str:
    pct = lambda v: "-" if v is None else f"{v:.1%}"
    sgn = lambda v: "-" if v is None else f"{v:+.2%}"
    lines = [f"# NFL Model vs Line ({season}, honest week-by-week replay)", "",
             f"Backtest sides matched to a replayed projection: {matched} of {total}.", "",
             "## 1. When the model disagrees with FanDuel's line, does the result follow it?", "",
             "| Stat | Player-games | Corr(model-line, actual-line) | Same for baseline | Model's side wins (any gap) | 5+ | 10+ | 15+ | Model MAE | Model minus line (avg) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda t: f"{pct(t[0])} ({t[1]})"
    for r in a:
        lines.append(f"| {r['stat']} | {r['n']} | {r['corr']:+.3f} | {r['corr_baseline']:+.3f} | {fmt(r['model_side_wins_ge0'])} | "
                     f"{fmt(r['model_side_wins_ge5'])} | {fmt(r['model_side_wins_ge10'])} | {fmt(r['model_side_wins_ge15'])} | "
                     f"{r['model_mae']:.1f} | {r['mean_bias']:+.1f} |")
    lines += ["", "## 2. Do sharp gaps the model agrees with hold up better?", "",
              "All snapshots (T-90/45/15). EV at the sharp close is the most reliable column; win rate and ROI are noisy.", "",
              "| Stat | Gap | Model | Sides | EV when seen | EV at sharp close | Win rate | ROI (settled) |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in b:
        lines.append(f"| {r['stat']} | {r['bucket']} | {r['group']} | {r['n']} | {sgn(r['mean_ev'])} | {sgn(r['ev_at_sharp_close'])} | "
                     f"{pct(r['win_rate'])} | {sgn(r['roi'])} ({r['settled']}) |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--stats", default="receiving_yards,rushing_yards")
    ap.add_argument("--variant", default="", help="replay variant suffix, e.g. _archive_context")
    args = ap.parse_args()
    stats = [s.strip() for s in args.stats.split(",") if s.strip()]
    j, sides = load(args.season, stats, args.variant)
    a, b = model_vs_line(j), confluence(j)
    stem = ROOT / "reports" / f"nfl_model_vs_line_{args.season}{args.variant}"
    atomic_json(stem.with_suffix(".json"), dict(season=args.season, matched=int(len(j)), total=int(len(sides)), model_vs_line=a, confluence=b))
    stem.with_suffix(".md").write_text(markdown(args.season, len(j), len(sides), a, b), encoding="utf-8")
    print(stem.with_suffix(".md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
