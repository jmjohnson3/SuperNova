"""Challenger: a pre-game model of yards (and catches) per opportunity, judged with the volume known.

modeling/projection_error splits each projection miss into volume and efficiency; modeling/volume_model and
volume_absence work on the first half. This is the second half: yards per target, catches per target and yards
per carry. A model is scored on the yards it gives when multiplied by the targets or carries the player
actually had, so the volume miss is out of the picture and only the per-opportunity rate is judged.

    baseline       the player's last-10 rate shrunk toward his position's rate (what volume_model uses)
    model          LightGBM on the parity-safe features plus that baseline, weighted by opportunities
    + opponent     adds what the opponent allowed per target and per carry by position (in the table, not live)
    + weather      adds wind and temperature (the table holds the game's weather; live would need a forecast)

Same walk-forward blocks as modeling/volume_model (2023-2026, each trained on earlier games only, fixed
parameters). Then the live 2026 forecasts: the production number against the production volume proxy times each
rate, and against the line where there is one. Research only; nothing is saved or changed live.

    python -m nfl_pipeline.modeling.efficiency_model [--seasons 2023 2024 2025 2026]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling import projection_error, volume_model

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports" / "nfl_efficiency_model_latest.md"
STATS = {  # stat: volume, positions, last-10 stat average, last-10 volume average, opportunities of prior
    "receiving_yards": ("targets", ("WR", "TE", "RB"), "receiving_yards_avg_10", "targets_avg_10", 30.0),
    "receptions": ("targets", ("WR", "TE", "RB"), "receptions_avg_10", "targets_avg_10", 30.0),
    "rushing_yards": ("carries", ("RB", "QB"), "rushing_yards_avg_10", "carries_avg_10", 40.0),
}
OPPONENT = [f"opp_allowed_{kind}_{pos}_avg_5" for kind in ("receiving_yards", "yards_per_target", "air_yards_per_target") for pos in ("wr", "te", "rb")] + [
    "opp_allowed_yards_per_carry_rb_avg_5"]
WEATHER = ["wind_mph", "temp_f", "is_high_wind"]
VARIANTS = {"model": [], "+ opponent": OPPONENT, "+ opponent and weather": OPPONENT + WEATHER}
PARAMS = {**volume_model.PARAMS, "objective": "regression"}
ROUNDS = 400


def load(conn) -> pd.DataFrame:
    frame = volume_model.load(conn)
    outcomes = pd.read_sql("SELECT game_id, player_id, receiving_yards::float AS receiving_yards, receptions::float AS receptions, "
                           "rushing_yards::float AS rushing_yards FROM features.nfl_player_game_training_features WHERE season >= 2019", conn)
    return frame.drop(columns=[c for c in ("receiving_yards", "receptions", "rushing_yards") if c in frame]).merge(outcomes, on=["game_id", "player_id"], how="left")


def baseline_rate(df: pd.DataFrame, stat_avg: str, volume_avg: str, prior: float) -> pd.Series:
    """The player's last-10 per-opportunity rate, shrunk toward his position's rate as if `prior` opportunities
    at that rate had been added. Pre-game averages only."""
    games = df.n_games_prev_10.clip(lower=0).fillna(0)
    stat_total, volume_total = df[stat_avg].fillna(0) * games, df[volume_avg].fillna(0) * games
    sums = pd.DataFrame(dict(s=stat_total, v=volume_total, position=df.position)).groupby("position")[["s", "v"]].transform("sum")
    position_rate = (sums.s / sums.v.replace(0, np.nan)).fillna(0.0)
    return (stat_total + prior * position_rate) / (volume_total + prior)


def walk_forward(data: pd.DataFrame, stat: str, features: list[str], seasons: list[int]) -> pd.Series:
    """Out-of-sample rate for every test row; each block is fitted on earlier games, weighted by opportunities."""
    volume, positions = STATS[stat][0], STATS[stat][1]
    cols = features + ["base_rate"]
    out = pd.Series(np.nan, index=data.index)
    for _, _, train, test in volume_model.blocks(data, seasons):
        fit = data[train]
        booster = lgb.train(PARAMS, lgb.Dataset(volume_model.feature_matrix(fit, cols, positions), fit[stat] / fit[volume], weight=fit[volume]),
                            num_boost_round=ROUNDS)
        out[test[test].index] = booster.predict(volume_model.feature_matrix(data[test], cols, positions))
    return out


def weekly(rows: pd.DataFrame, a: str, b: str) -> dict:
    return volume_model._week_interval(rows, a, b)


def evaluate(rows: pd.DataFrame, names: list[str]) -> dict:
    """Yards given the actual volume: every estimate on the same rows, against the baseline by NFL week."""
    mae = lambda g, n: float((g[n] - g.y).abs().mean())
    table = {}
    groups = {"All": rows, **{f"Season {int(s)}": g for s, g in rows.groupby("season")}, **{p: g for p, g in rows.groupby("position")}}
    for label, g in groups.items():
        table[label] = dict(rows=int(len(g)), mae={n: mae(g, n) for n in names},
                            weekly={n: weekly(g, n, "baseline") for n in names if n != "baseline"})
    return table


def live_check(conn, season: int, rates: dict[str, pd.DataFrame]) -> dict:
    """The live forecasts of one season: the production number against its own volume proxy times each rate."""
    live = projection_error.load(conn, season)
    out = {}
    for stat, rate in rates.items():
        df = projection_error.decompose(live, stat).merge(rate, on=["game_id", "player_id"], how="inner")
        df = df[df.volume_hat.notna()]
        if df.empty:
            continue
        names = [c for c in rate.columns if c not in ("game_id", "player_id")]
        for n in names:
            df[f"proj_{n}"] = df.volume_hat * df[n]
        mae = lambda g, col: float((g.actual - g[col]).abs().mean())
        lined = df[df.line.notna()]
        out[stat] = dict(rows=int(len(df)), production=mae(df, "model_projection"), **{n: mae(df, f"proj_{n}") for n in names},
                         with_line=dict(rows=int(len(lined)), line=mae(lined, "line"), production=mae(lined, "model_projection"),
                                        **{n: mae(lined, f"proj_{n}") for n in names}) if len(lined) else None)
    return out


def markdown(doc: dict) -> str:
    f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"
    names = ["baseline", *VARIANTS]
    L = ["# NFL efficiency model (challenger)", "",
         f"Built {doc['built_at']}. Walk-forward 4-week blocks over {', '.join(map(str, doc['seasons']))}, each trained on earlier games only. "
         "Each estimate is a per-opportunity rate, scored as rate x the targets or carries the player actually had, so only efficiency is "
         "judged. Research only; nothing live changes.", "",
         "Baseline = the player's last-10 rate shrunk toward his position's rate. Weekly difference is against the baseline (negative is better)."]
    for stat, table in doc["efficiency"].items():
        L += ["", f"## {stat} with the volume known", "", "| Player-games | Rows | " + " | ".join(n.capitalize() for n in names) + " | Best model, weekly difference (95%) |",
              "|---|---:|" + "---:|" * (len(names) + 1)]
        for label, r in table.items():
            best = min(VARIANTS, key=lambda n: r["mae"][n])
            w = r["weekly"][best]
            interval = "-" if w["mean"] is None else f"{best}: {w['mean']:+.3f} ({w['lower']:+.3f} to {w['upper']:+.3f})"
            L.append(f"| {label} | {r['rows']} | " + " | ".join(f(r["mae"][n]) for n in names) + f" | {interval} |")
    if doc.get("live"):
        L += ["", f"## Live {max(doc['seasons'])} forecasts: production's volume proxy times each rate (average miss)", "",
              "| Stat | Rows | Line | Production | " + " | ".join(n.capitalize() for n in names) + " |", "|---|---:|---:|---:|" + "---:|" * len(names)]
        for stat, y in doc["live"].items():
            L.append(f"| {stat}, all | {y['rows']} | - | {f(y['production'], 2)} | " + " | ".join(f(y[n], 2) for n in names) + " |")
            w = y.get("with_line")
            if w:
                L.append(f"| {stat}, with a line | {w['rows']} | {f(w['line'], 2)} | {f(w['production'], 2)} | " + " | ".join(f(w[n], 2) for n in names) + " |")
    L += ["", "The opponent columns are in the training table but not in the live snapshot, and the weather columns are the game's own weather; "
              "either would need a live source before use."]
    return "\n".join(L) + "\n"


def run(seasons: list[int], log=print) -> dict:
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        frame = load(conn)
        servable, _ = volume_model.serving_parity(conn, frame, volume_model.candidate_features())
        doc = dict(built_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), seasons=seasons, efficiency={}, research_only=True,
                   production_changed=False)
        rates = {}
        for stat, (volume, positions, stat_avg, volume_avg, prior) in STATS.items():
            data = frame[frame.position.isin(positions) & frame[stat].notna() & frame[volume].gt(0)].copy()
            data["base_rate"] = baseline_rate(data, stat_avg, volume_avg, prior)
            features, _ = volume_model.leak_guard(data[data.season.lt(min(seasons))], stat, servable)
            extras = [c for c in OPPONENT + WEATHER if c in data]
            for name, extra in VARIANTS.items():
                log(f"  {stat}: {name}")
                data[f"rate_{name}"] = walk_forward(data, stat, features + [c for c in extra if c in extras], seasons)
            rows = data.dropna(subset=[f"rate_{n}" for n in VARIANTS]).copy()
            rows["y"], rows["baseline"] = rows[stat], rows.base_rate * rows[volume]
            for name in VARIANTS:
                rows[name] = rows[f"rate_{name}"] * rows[volume]
            doc["efficiency"][stat] = evaluate(rows, ["baseline", *VARIANTS])
            rates[stat] = rows[["game_id", "player_id", "base_rate", *[f"rate_{n}" for n in VARIANTS]]].rename(
                columns={"base_rate": "baseline", **{f"rate_{n}": n for n in VARIANTS}})
        doc["live"] = live_check(conn, max(seasons), rates) if max(seasons) >= 2026 else {}
    atomic_json(REPORT.with_suffix(".json"), doc)
    REPORT.write_text(markdown(doc), encoding="utf-8")
    log(f"  wrote {REPORT}")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    run(ap.parse_args().seasons)


if __name__ == "__main__":
    main()
