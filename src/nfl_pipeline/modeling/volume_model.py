"""Challenger: a dedicated pre-game model for targets and carries, judged on volume error first.

modeling/projection_error showed volume is the larger share of projection error (knowing the actual
targets would cut receiving-yard MAE 29%, carries rushing-yard MAE 39%), and that the pipeline's own
volume features predict volume worse than a plain 5-game average (carries MAE 4.28 vs 3.34). An
earlier target-only challenger (docs/nfl_target_volume_repair.md) moved target RMSE only 2.282 ->
2.252 and was rejected. This model is a count model of volume alone, so it can be judged on volume
before anything downstream depends on it.

Data is the as-of feature table (features.nfl_player_game_training_features): every feature is
built from earlier games or same-week pregame context; same-game outcome columns are excluded and a
guard drops any feature that tracks the outcome too closely to be pregame. Players who played only,
which is the population a prop settles on.

Evaluation is walk-forward in 4-week blocks over 2023-2026, each block trained only on earlier games,
with fixed hyperparameters (nothing is tuned on a test block). The model must beat the best simple
baseline on MAE and RMSE overall, with a week-clustered interval on the MAE gain that excludes zero,
and not lose in any season by more than 2%. It then has to help yards: the live 2026 projection is
rescaled by the model's volume relative to the 5-game average it implicitly used, and scored on the
same settled rows as the production number.

A challenger only: it never changes a live forecast. Artifacts go to models/volume_model/<run>/.

    python -m nfl_pipeline.modeling.volume_model [--seasons 2023 2024 2025 2026] [--no-save]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import t as student_t

from nfl_pipeline import features as feature_contract
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import MODEL_ROOT, atomic_json
from nfl_pipeline.modeling import projection_error

ROOT = Path(__file__).resolve().parents[3]
STORE = MODEL_ROOT / "volume_model"
TARGETS = {
    "targets": dict(positions=("WR", "TE", "RB"), baselines=("targets_avg_3", "targets_avg_5", "targets_avg_10",
                                                             "receiver_projected_targets_v3")),
    "carries": dict(positions=("RB", "QB"), baselines=("carries_avg_3", "carries_avg_5", "carries_avg_10",
                                                       "rb_projected_carries_v3")),
}
IDENTIFIERS = {"season", "week", "season_type", "game_id", "game_date_et", "player_id", "player_name",
               "team_abbr", "opponent_abbr", "position"}
OUTCOMES = set(feature_contract.TARGET_STATS) | set(feature_contract.USAGE_STATS)
LEAK_CORRELATION = 0.95
BLOCK_WEEKS = ((1, 4), (5, 8), (9, 12), (13, 16), (17, 22))
PARAMS = dict(objective="poisson", learning_rate=0.03, num_leaves=31, min_data_in_leaf=80,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, seed=7, num_threads=4)
ROUNDS = 600
ACCEPT = dict(max_season_regression=0.02, alpha=0.05)


def candidate_features() -> list[str]:
    return [c for c in feature_contract.FEATURE_COLUMNS if c not in IDENTIFIERS and c not in OUTCOMES]


def load(conn) -> pd.DataFrame:
    columns = sorted(set(candidate_features()) | {"season", "week", "game_id", "game_date_et", "player_id",
                                                     "team_abbr", "position", "targets", "carries"})
    frame = pd.read_sql(f"SELECT {', '.join(columns)} FROM features.nfl_player_game_training_features "
                        "WHERE season >= 2019", conn)
    frame["position"] = frame.position.fillna("").str.upper()
    for column in frame.columns:
        if column not in ("game_id", "game_date_et", "player_id", "team_abbr", "position"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def feature_matrix(frame: pd.DataFrame, features: list[str], positions: tuple[str, ...]) -> pd.DataFrame:
    flags = pd.DataFrame({f"pos_{p}": frame.position.eq(p).astype(float) for p in positions}, index=frame.index)
    return pd.concat([frame[features].astype(float), flags], axis=1)


def leak_guard(frame: pd.DataFrame, target: str, features: list[str]) -> tuple[list[str], dict]:
    """Drop any feature that correlates with the same-game outcome beyond what pregame data can."""
    dropped = {}
    y = frame[target]
    for column in features:
        x = frame[column]
        ok = x.notna() & y.notna()
        if ok.sum() < 200 or x[ok].std() == 0:
            continue
        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
        if abs(r) >= LEAK_CORRELATION:
            dropped[column] = round(r, 4)
    return [c for c in features if c not in dropped], dropped


def blocks(frame: pd.DataFrame, seasons: list[int]):
    for season in seasons:
        last = int(frame.loc[frame.season.eq(season), "week"].max() or 0)
        for start, end in BLOCK_WEEKS:
            if start > last:
                break
            test = frame.season.eq(season) & frame.week.between(start, end)
            train = frame.season.lt(season) | (frame.season.eq(season) & frame.week.lt(start))
            if test.any():
                yield season, (start, min(end, last)), train, test


def fit(frame: pd.DataFrame, target: str, features: list[str], positions) -> lgb.Booster:
    X = feature_matrix(frame, features, positions)
    return lgb.train(PARAMS, lgb.Dataset(X, frame[target].astype(float)), num_boost_round=ROUNDS)


def walk_forward(frame: pd.DataFrame, target: str, features: list[str], seasons: list[int]) -> pd.DataFrame:
    spec = TARGETS[target]
    data = frame[frame.position.isin(spec["positions"]) & frame[target].notna()].copy()
    parts = []
    for season, (start, end), train, test in blocks(data, seasons):
        booster = fit(data[train], target, features, spec["positions"])
        part = data[test].copy()
        part["model"] = booster.predict(feature_matrix(part, features, spec["positions"]))
        part["block"] = f"{season}:{start}-{end}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else data.iloc[:0]


def _scores(y: pd.Series, pred: pd.Series) -> dict:
    err = pred - y
    return dict(rows=int(len(y)), mae=float(err.abs().mean()), rmse=float(np.sqrt((err ** 2).mean())),
                bias=float(err.mean()))


def _week_interval(frame: pd.DataFrame, a: str, b: str) -> dict:
    """Mean weekly MAE difference (a - b) with a two-sided interval clustered by NFL week."""
    per_week = frame.assign(d=(frame[a] - frame.y).abs() - (frame[b] - frame.y).abs()).groupby(["season", "week"]).d.mean()
    n = len(per_week)
    if n < 3:
        return dict(weeks=n, mean=None, lower=None, upper=None)
    half = student_t.ppf(1 - ACCEPT["alpha"] / 2, n - 1) * per_week.std(ddof=1) / np.sqrt(n)
    return dict(weeks=int(n), mean=float(per_week.mean()), lower=float(per_week.mean() - half),
                upper=float(per_week.mean() + half))


def evaluate(oof: pd.DataFrame, target: str) -> dict:
    """Model vs every baseline on identical rows (rows where all baselines exist)."""
    spec = TARGETS[target]
    usable = [b for b in spec["baselines"] if b in oof and oof[b].notna().mean() > 0.5]
    rows = oof.dropna(subset=["model", *usable]).copy()
    rows["y"] = rows[target].astype(float)
    scores = {name: _scores(rows.y, rows[name]) for name in ["model", *usable]}
    best = min(usable, key=lambda name: scores[name]["mae"])
    by_season = {int(s): {name: _scores(g.y, g[name]) for name in ("model", best)} for s, g in rows.groupby("season")}
    by_position = {p: {name: _scores(g.y, g[name]) for name in ("model", best)} for p, g in rows.groupby("position")}
    interval = _week_interval(rows, "model", best)
    worst_season = max((v["model"]["mae"] / v[best]["mae"] - 1 for v in by_season.values()), default=None)
    checks = dict(beats_best_mae=scores["model"]["mae"] < scores[best]["mae"],
                  beats_best_rmse=scores["model"]["rmse"] < scores[best]["rmse"],
                  weekly_interval_excludes_zero=interval["upper"] is not None and interval["upper"] < 0,
                  no_season_worse_than_2pct=worst_season is not None and worst_season <= ACCEPT["max_season_regression"])
    return dict(rows=int(len(rows)), best_baseline=best, scores=scores, by_season=by_season,
                by_position=by_position, weekly_mae_difference_vs_best=interval, worst_season_change=worst_season,
                checks=checks, volume_accepted=all(checks.values()))


YARDAGE = {"receiving_yards": ("targets", "targets_avg_5", "receiving_yards_avg_10", "targets_avg_10", 30.0),
           "receptions": ("targets", "targets_avg_5", "receptions_avg_10", "targets_avg_10", 30.0),
           "rushing_yards": ("carries", "carries_avg_5", "rushing_yards_avg_10", "carries_avg_10", 40.0)}


def volume_times_efficiency(df: pd.DataFrame, stat_avg: str, volume_avg: str, prior_volume: float) -> pd.Series:
    """Model volume times the player's per-opportunity rate over his last 10 games, shrunk toward his
    position's rate as if `prior_volume` opportunities at that rate had been added."""
    games = df.n_games_prev_10.clip(lower=0).fillna(0)
    stat_total, volume_total = df[stat_avg].fillna(0) * games, df[volume_avg].fillna(0) * games
    sums = pd.DataFrame(dict(s=stat_total, v=volume_total, position=df.position)).groupby("position")[["s", "v"]].transform("sum")
    position_rate = (sums.s / sums.v.replace(0, np.nan)).fillna(0.0)
    rate = (stat_total + prior_volume * position_rate) / (volume_total + prior_volume)
    return df.model * rate


def yardage_check(oof: dict[str, pd.DataFrame], season: int, conn) -> dict:
    """Does the better volume make live yardage projections better on the same settled rows?

    Two ways to use it, both with out-of-sample (walk-forward) volume only:
    - rescaled: the live projection P assumes roughly the 5-game-average volume (projection_error);
      P' = P * model / avg_5.
    - volume x efficiency: model volume times the player's shrunk per-opportunity rate.
    Position rates for the shrinkage come from the same rows' pregame averages, never outcomes.
    """
    live = projection_error.load(conn, season)
    # targets_avg_10 / carries_avg_10 already come with the live rows (projection_error).
    history = pd.read_sql("SELECT game_id, player_id, n_games_prev_10, receiving_yards_avg_10, receptions_avg_10, "
                          "rushing_yards_avg_10 FROM features.nfl_player_game_training_features WHERE season = %(season)s",
                          conn, params=dict(season=season))
    out = {}
    for stat, (target, average, stat_avg, volume_avg, prior) in YARDAGE.items():
        volume = oof[target][["game_id", "player_id", "model"]]
        df = projection_error.decompose(live, stat).merge(volume, on=["game_id", "player_id"], how="inner")
        df = df.merge(history, on=["game_id", "player_id"], how="left")
        df = df[df[average].gt(0) & df.model.notna()]
        if df.empty:
            continue
        rescaled = df.model_projection * df.model / df[average]
        built = volume_times_efficiency(df, stat_avg, volume_avg, prior)
        base_err, new_err, built_err = df.actual - df.model_projection, df.actual - rescaled, df.actual - built
        out[stat] = dict(rows=int(len(df)), production_mae=float(base_err.abs().mean()),
                         rescaled_mae=float(new_err.abs().mean()), volume_x_efficiency_mae=float(built_err.abs().mean()),
                         production_above=float((base_err > 0).mean()), rescaled_above=float((new_err > 0).mean()),
                         volume_x_efficiency_above=float((built_err > 0).mean()),
                         by_position={p: dict(rows=int(len(g)), production_mae=float((g.actual - g.model_projection).abs().mean()),
                                              rescaled_mae=float((g.actual - g.model_projection * g.model / g[average]).abs().mean()))
                                      for p, g in df.groupby("position")})
    return out


def markdown(doc: dict) -> str:
    f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"
    lines = ["# NFL Volume Model (challenger)", "", f"Run `{doc['run_id']}`. Walk-forward 4-week blocks over seasons "
             f"{', '.join(map(str, doc['seasons']))}; each block trained only on earlier games, fixed hyperparameters. "
             "Players who played. Never changes a live forecast.", ""]
    for target, ev in doc["volume"].items():
        lines += [f"## {target}: {'ACCEPTED on volume' if ev['volume_accepted'] else 'not accepted'}", "",
                  f"{ev['rows']} player-games where every baseline exists. Best simple baseline: `{ev['best_baseline']}`.", "",
                  "| Estimate | MAE | RMSE | Bias |", "|---|---:|---:|---:|"]
        for name, s in ev["scores"].items():
            lines.append(f"| {name} | {f(s['mae'])} | {f(s['rmse'])} | {s['bias']:+.3f} |")
        w = ev["weekly_mae_difference_vs_best"]
        lines += ["", f"Weekly MAE difference, model minus `{ev['best_baseline']}`: {f(w['mean'])} "
                      f"(95% interval {f(w['lower'])} to {f(w['upper'])}, {w['weeks']} weeks; negative = model better).", "",
                  "| Season | Model MAE | Baseline MAE | Change |", "|---|---:|---:|---:|"]
        for season, v in ev["by_season"].items():
            m, b = v["model"]["mae"], v[ev["best_baseline"]]["mae"]
            lines.append(f"| {season} | {f(m)} | {f(b)} | {m / b - 1:+.1%} |")
        lines += ["", "| Position | Model MAE | Baseline MAE | Change |", "|---|---:|---:|---:|"]
        for position, v in ev["by_position"].items():
            m, b = v["model"]["mae"], v[ev["best_baseline"]]["mae"]
            lines.append(f"| {position} | {f(m)} | {f(b)} | {m / b - 1:+.1%} |")
        lines += ["", "Checks: " + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in ev["checks"].items()) + ".", ""]
        if ev.get("leak_guard_dropped"):
            lines += [f"Leak guard dropped: {ev['leak_guard_dropped']}", ""]
    if doc.get("yardage"):
        lines += ["## Does it help yards? (live 2026 settled forecasts)", "",
                  "Rescaled: the live projection moved from the 5-game-average volume to the model's out-of-sample "
                  "volume. Volume x efficiency: model volume times the player's last-10 per-opportunity rate, shrunk "
                  "toward his position's rate. Same settled rows as the production number.", "",
                  "| Stat | Rows | Production MAE | Rescaled MAE | Volume x efficiency MAE | Above: production / rescaled / built |",
                  "|---|---:|---:|---:|---:|---|"]
        for stat, y in doc["yardage"].items():
            lines.append(f"| {stat} | {y['rows']} | {f(y['production_mae'], 2)} | {f(y['rescaled_mae'], 2)} | "
                         f"{f(y['volume_x_efficiency_mae'], 2)} | {y['production_above']:.0%} / {y['rescaled_above']:.0%} / "
                         f"{y['volume_x_efficiency_above']:.0%} |")
        lines.append("")
    lines += ["Promotion is a manual decision recorded in a doc; this report changes nothing live."]
    return "\n".join(lines) + "\n"


def run(seasons: list[int], save: bool = True) -> dict:
    run_id = "volume-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        frame = load(conn)
        doc = dict(run_id=run_id, built_at=datetime.now(timezone.utc).isoformat(), seasons=seasons,
                   params=PARAMS, rounds=ROUNDS, volume={}, research_only=True, production_changed=False)
        oof, finals = {}, {}
        for target, spec in TARGETS.items():
            population = frame[frame.position.isin(spec["positions"]) & frame[target].notna()]
            features, dropped = leak_guard(population[population.season.lt(min(seasons))], target, candidate_features())
            oof[target] = walk_forward(frame, target, features, seasons)
            evaluation = evaluate(oof[target], target)
            evaluation.update(features=len(features), leak_guard_dropped=dropped)
            doc["volume"][target] = evaluation
            if save:
                finals[target] = dict(booster=fit(population, target, features, spec["positions"]),
                                      features=features, positions=spec["positions"])
        doc["yardage"] = yardage_check(oof, max(seasons), conn) if max(seasons) >= 2026 else {}
    if save:
        folder = STORE / run_id
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(finals, folder / "models.joblib")
        for target, part in oof.items():
            part[["season", "week", "game_id", "player_id", "position", target, "model"]].to_csv(
                folder / f"oof_{target}.csv", index=False)
        atomic_json(folder / "report.json", doc)
        atomic_json(STORE / "latest.json", dict(run_id=run_id, report=f"{run_id}/report.json",
                                                accepted={k: v["volume_accepted"] for k, v in doc["volume"].items()}))
    atomic_json(ROOT / "reports" / "nfl_volume_model_latest.json", doc)
    (ROOT / "reports" / "nfl_volume_model_latest.md").write_text(markdown(doc), encoding="utf-8")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    run(args.seasons, save=not args.no_save)
    print((ROOT / "reports" / "nfl_volume_model_latest.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
