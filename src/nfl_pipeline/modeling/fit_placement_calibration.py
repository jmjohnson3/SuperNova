"""Fit where the model's raw projection should sit, from settled live forecasts, chronologically.

The model forecasts a median. modeling/projection_error showed two systematic misses that held in
every 2026 week: rushing projections sit too high (only 41% of outcomes land above them; 34% for
QBs) and passing projections sit too low and too spread out (61% above; projections under 193 yards
went over 74% of the time). This fits a correction per stat and position and keeps it only where it
helps on weeks it was not fitted on.

Families, each fitted by least absolute deviation (the loss a median minimizes):
    identity   P' = P
    scale      P' = k * P              moves every projection proportionally (rushing: shift down)
    linear     P' = a + b * P, b<=1    pulls projections toward the middle, the far ends most

Validation is expanding by week: fit on all earlier weeks, score the next week, never the reverse.
A group is corrected only when its held-out MAE improves on identity and the share of outcomes
above the projection moves toward 50%; otherwise it stays identity. The installed artifact is read
by predict_player_props before market anchoring; a missing file is identity.

    python -m nfl_pipeline.modeling.fit_placement_calibration            # report only
    python -m nfl_pipeline.modeling.fit_placement_calibration --write    # install
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import minimize

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling import projection_error

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "src/nfl_pipeline/modeling/models/player_props/placement_calibration.json"
CONTRACT = "nfl-projection-placement-v1"
# The stats the error measurement found systematically misplaced, split where the miss differs.
GROUPS = {"rushing_yards": ("QB", "RB"), "passing_yards": ("QB",)}
MIN_FIT_ROWS = 25
MIN_HELDOUT_IMPROVEMENT = 0.01   # held-out MAE must fall by at least 1%
B_RANGE, K_RANGE = (0.3, 1.0), (0.5, 1.5)


def _lad(objective, start):
    return minimize(objective, start, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=4000)).x


def fit_family(family: str, p: np.ndarray, y: np.ndarray) -> dict:
    if family == "identity":
        return dict(family="identity")
    if family == "scale":
        k = float(np.clip(_lad(lambda v: np.abs(y - v[0] * p).mean(), [1.0])[0], *K_RANGE))
        return dict(family="scale", k=k)
    def loss(v):
        b = np.clip(v[1], *B_RANGE)
        return np.abs(y - v[0] - b * p).mean()
    a, b = _lad(loss, [0.0, 1.0])
    return dict(family="linear", a=float(a), b=float(np.clip(b, *B_RANGE)))


def apply(params: dict | None, projection: float) -> float:
    """The corrected projection, never negative. Missing or identity params change nothing."""
    if not params or params.get("family") in (None, "identity"):
        return float(projection)
    if params["family"] == "scale":
        return max(0.0, float(params["k"]) * float(projection))
    if params["family"] == "linear":
        return max(0.0, float(params["a"]) + float(params["b"]) * float(projection))
    raise ValueError("Unknown placement family: " + str(params.get("family")))


def _score(y, pred) -> dict:
    err = y - pred
    return dict(rows=int(len(y)), mae=float(np.abs(err).mean()), bias=float(-err.mean()),
                above_share=float((err > 0).mean()))


def evaluate_group(df: pd.DataFrame) -> dict:
    """Expanding-week held-out comparison of each family, then the choice and its final fit."""
    weeks = sorted(int(w) for w in df.week.unique())
    heldout = {family: [] for family in ("identity", "scale", "linear")}
    folds = []
    for week in weeks:
        train, test = df[df.week < week], df[df.week == week]
        if len(train) < MIN_FIT_ROWS or test.empty:
            continue
        p, y = train.model_projection.to_numpy(float), train.actual.to_numpy(float)
        fold = dict(test_week=week, train_rows=int(len(train)), test_rows=int(len(test)))
        for family in heldout:
            params = fit_family(family, p, y)
            pred = np.array([apply(params, v) for v in test.model_projection])
            heldout[family].append(pd.DataFrame(dict(y=test.actual.to_numpy(float), pred=pred)))
            fold[family] = params
        folds.append(fold)
    if not folds:
        return dict(status="insufficient_history", rows=int(len(df)), chosen=dict(family="identity"))
    scores = {family: _score(pd.concat(parts).y.to_numpy(), pd.concat(parts).pred.to_numpy())
              for family, parts in heldout.items()}
    base = scores["identity"]
    best = min(("scale", "linear"), key=lambda f: scores[f]["mae"])
    improves = scores[best]["mae"] <= base["mae"] * (1 - MIN_HELDOUT_IMPROVEMENT)
    closer = abs(scores[best]["above_share"] - 0.5) < abs(base["above_share"] - 0.5)
    family = best if improves and closer else "identity"
    chosen = fit_family(family, df.model_projection.to_numpy(float), df.actual.to_numpy(float))
    return dict(status="corrected" if family != "identity" else "kept_identity", rows=int(len(df)), weeks=weeks,
                heldout=scores, heldout_weeks=[f["test_week"] for f in folds], best_candidate=best,
                improves_heldout_mae=bool(improves), moves_toward_median=bool(closer), chosen=chosen, folds=folds,
                in_sample_before=_score(df.actual.to_numpy(float), df.model_projection.to_numpy(float)),
                in_sample_after=_score(df.actual.to_numpy(float),
                                       np.array([apply(chosen, v) for v in df.model_projection])))


def fit(frame: pd.DataFrame) -> dict:
    groups = {}
    for stat, positions in GROUPS.items():
        decomposed = projection_error.decompose(frame, stat)
        for position in positions:
            df = decomposed[decomposed.position.eq(position)][["week", "model_projection", "actual"]].dropna()
            groups[f"{stat}|{position}"] = evaluate_group(df)
    return groups


def artifact(groups: dict, season: int) -> dict:
    stats: dict[str, dict] = {}
    for key, result in groups.items():
        stat, position = key.split("|")
        stats.setdefault(stat, {})[position] = result["chosen"]
    return dict(contract=CONTRACT, version="placement-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                fitted_at=datetime.now(timezone.utc).isoformat(), season=season,
                applies_to="raw model projection, before market anchoring",
                stats=stats, evidence={k: {x: v[x] for x in ("status", "rows", "heldout", "heldout_weeks")
                                           if x in v} for k, v in groups.items()})


def markdown(groups: dict, written: bool) -> str:
    pct = lambda v: "-" if v is None else f"{v:.0%}"
    lines = ["# NFL Projection Placement", "",
             "Held-out = each week scored by a correction fitted only on earlier weeks. A group is corrected only "
             "when held-out MAE improves at least 1% and the share above the projection moves toward 50%.", "",
             "| Group | Rows | Held-out weeks | Identity MAE / above | Scale MAE / above | Linear MAE / above | Chosen | Parameters |",
             "|---|---:|---|---|---|---|---|---|"]
    for key, g in groups.items():
        h = g.get("heldout") or {}
        cell = lambda f: f"{h[f]['mae']:.2f} / {pct(h[f]['above_share'])}" if f in h else "-"
        params = {k: round(v, 4) for k, v in g["chosen"].items() if k != "family"}
        lines.append(f"| {key} | {g['rows']} | {','.join(map(str, g.get('heldout_weeks') or [])) or '-'} | {cell('identity')} | "
                     f"{cell('scale')} | {cell('linear')} | {g['chosen']['family']} | {params or '-'} |")
    lines += ["", "In-sample effect of the chosen correction (all weeks):", "",
              "| Group | MAE before | MAE after | Above before | Above after | Bias before | Bias after |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for key, g in groups.items():
        if "in_sample_before" in g:
            b, a = g["in_sample_before"], g["in_sample_after"]
            lines.append(f"| {key} | {b['mae']:.2f} | {a['mae']:.2f} | {pct(b['above_share'])} | {pct(a['above_share'])} | "
                         f"{b['bias']:+.2f} | {a['bias']:+.2f} |")
    lines += ["", ("Installed to " if written else "Not installed (report only); install with --write to ")
              + "`models/player_props/placement_calibration.json`."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--write", action="store_true", help="install the fitted artifact")
    args = ap.parse_args()
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        frame = projection_error.load(conn, args.season)
    groups = fit(frame)
    doc = artifact(groups, args.season)
    atomic_json(ROOT / "reports" / "nfl_placement_calibration_latest.json", dict(doc, groups=groups))
    if args.write:
        atomic_json(ARTIFACT, doc)
    text = markdown(groups, args.write)
    (ROOT / "reports" / "nfl_placement_calibration_latest.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
