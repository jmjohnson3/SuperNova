"""Measure the model's raw projection error against outcomes, and split it into volume and efficiency.

One row per settled player-game-stat: the last forecast made before kickoff, with the model's own
number (never the line-anchored one that market calibration produces), against the official box
score. Yardage and count stats are a volume times an efficiency:

    stat              volume          efficiency
    receiving_yards   targets         yards per target
    receptions        targets         catch rate
    rushing_yards     carries         yards per carry
    passing_yards     pass attempts   yards per attempt

The production model predicts the stat directly and has no volume output, so its volume belief is
taken from the pre-game 5-game average in the as-of feature table, and its implied efficiency is
the projection divided by that volume. The pipeline's own projected targets/carries (the *_v3
features) predict actual volume worse than that average (see the volume-estimate table), so they
would only add noise to the split. With projection P, actual Y, volume estimate V^ and actual V:

    Y - P = (V - V^) * P / V^   +   (Y - V * P / V^)
            volume error            efficiency error at the actual volume

The two parts add up exactly to the projection error, so their squared-error shares say where the
miss comes from. The "knew the volume" and "knew the efficiency" counterfactuals show how much
accuracy each piece would buy if it were perfect. Because V^ is a proxy for what the model believed,
this attribution is approximate; the counterfactual MAEs do not depend on it being exact for the
volume side's own accuracy, which is measured directly (volume MAE and bias).

    python -m nfl_pipeline.modeling.projection_error [--season 2026]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json

ROOT = Path(__file__).resolve().parents[3]
# Before this, `projection` is the model's own number; after it, market calibration anchors
# `projection` to the line and the raw number is kept in the payload.
CALIBRATION_INSTALLED = "2026-10-01T03:04:58Z"

DECOMPOSED = {
    "receiving_yards": dict(outcome="receiving_yards", volume="targets", proxies=("targets_avg_5",)),
    "receptions": dict(outcome="receptions", volume="targets", proxies=("targets_avg_5",)),
    "rushing_yards": dict(outcome="rushing_yards", volume="carries", proxies=("carries_avg_5",)),
    "passing_yards": dict(outcome="passing_yards", volume="pass_attempts", proxies=("pass_attempts_avg_5",)),
}
VOLUME_SOURCES = ("receiver_projected_targets_v3", "rb_projected_carries_v3", "targets_avg_5",
                  "carries_avg_5", "pass_attempts_avg_5", "targets_avg_10", "carries_avg_10")


def load(conn, season: int) -> pd.DataFrame:
    """Last pre-kickoff forecast per player-game-stat, with outcome, volume estimates and line."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='180s'")
        cur.execute(f"""
            WITH latest AS (
              SELECT DISTINCT ON (p.game_id, p.player_id, p.stat)
                     p.id, p.season, p.week, p.game_id, p.player_id, p.player_name, p.team_abbr, p.position,
                     p.stat, p.line::float AS line, p.book, p.projection::float AS anchored_projection,
                     COALESCE((p.forecast_payload #>> '{{forecast_outputs,pricing,model_projection}}')::float,
                              (p.forecast_payload #>> '{{probability_trace,model_projection}}')::float,
                              CASE WHEN p.line IS NULL OR p.created_at_utc < %(installed)s::timestamptz
                                   THEN p.projection::float END) AS model_projection,
                     p.created_at_utc, g.start_ts_utc
              FROM bets.nfl_player_prop_predictions p
              JOIN raw.nfl_games g ON g.game_id = p.game_id
              WHERE p.season = %(season)s AND p.integrity_version = 'nfl-asof-v2'
                AND p.stat = ANY(%(stats)s) AND g.status = 'final'
                AND p.created_at_utc < g.start_ts_utc
              ORDER BY p.game_id, p.player_id, p.stat, p.created_at_utc DESC, p.id DESC)
            SELECT l.*, f.targets::float AS targets, f.carries::float AS carries, f.pass_attempts::float AS pass_attempts,
                   f.receiving_yards::float AS out_receiving_yards, f.receptions::float AS out_receptions,
                   f.rushing_yards::float AS out_rushing_yards, f.passing_yards::float AS out_passing_yards,
                   {", ".join(f"f.{c}::float AS {c}" for c in VOLUME_SOURCES)}
            FROM latest l
            JOIN features.nfl_player_game_training_features f
              ON f.game_id = l.game_id AND f.player_id = l.player_id AND f.team_abbr = UPPER(l.team_abbr)""",
                    dict(season=season, stats=list(DECOMPOSED), installed=CALIBRATION_INSTALLED))
        return pd.DataFrame([dict(r) for r in cur.fetchall()])


def decompose(frame: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Per-row volume and efficiency error terms for one stat."""
    spec = DECOMPOSED[stat]
    df = frame[frame.stat.eq(stat)].copy()
    df["actual"] = df["out_" + spec["outcome"]]
    df["volume"] = df[spec["volume"]]
    df["volume_hat"] = np.nan
    df["volume_source"] = None
    for proxy in spec["proxies"]:  # first available proxy, in preference order
        take = df.volume_hat.isna() & df[proxy].notna() & df[proxy].gt(0)
        df.loc[take, "volume_hat"] = df.loc[take, proxy]
        df.loc[take, "volume_source"] = proxy
    df = df[df.model_projection.notna() & df.actual.notna() & df.volume.notna()]
    df["error"] = df.actual - df.model_projection
    implied = df.model_projection / df.volume_hat
    df["implied_efficiency"] = implied
    df["actual_efficiency"] = np.where(df.volume > 0, df.actual / df.volume.where(df.volume > 0), np.nan)
    knew_volume = df.volume * implied
    df["volume_part"] = knew_volume - df.model_projection
    df["efficiency_part"] = df.actual - knew_volume
    # Counterfactual projections. With zero actual volume the actual efficiency is undefined, so
    # knowing it adds nothing: keep the model's own number there.
    df["knew_volume_error"] = df.actual - knew_volume
    knew_efficiency = (df.volume_hat * df.actual_efficiency).where(df.volume > 0, df.model_projection)
    df["knew_efficiency_error"] = df.actual - knew_efficiency
    df["volume_error"] = df.volume - df.volume_hat
    return df


def _mae(values) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.abs().mean()) if len(values) else None


def _mean(values) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def summarize_stat(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(player_games=0)
    out = dict(player_games=int(len(df)), weeks=sorted(int(w) for w in df.week.unique()),
               actual_mean=_mean(df.actual), projection_mean=_mean(df.model_projection),
               mae=_mae(df.error), bias=_mean(-df.error), rmse=float(np.sqrt((df.error ** 2).mean())),
               # The model forecasts a median. For a right-skewed stat the mean bias is negative by design;
               # this is the share of outcomes above the projection, which an honest median keeps near 50%.
               above_share=float((df.error > 0).mean()), median_error=float(df.error.median()))
    # Calibration-in-the-large: regress the outcome on the projection. A slope below 1 means the
    # projections spread out more than outcomes support (too confident in who is high or low).
    if df.model_projection.std() > 0:
        slope, intercept = np.polyfit(df.model_projection, df.actual, 1)
        out.update(slope=float(slope), intercept=float(intercept),
                   correlation=float(np.corrcoef(df.model_projection, df.actual)[0, 1]))
    decomposable = df[df.volume_hat.notna()]
    if len(decomposable):
        d = decomposable
        mse = float((d.error ** 2).mean())
        out["decomposition"] = dict(
            player_games=int(len(d)), mae=_mae(d.error),
            knew_volume_mae=_mae(d.knew_volume_error), knew_efficiency_mae=_mae(d.knew_efficiency_error),
            volume_share=float((d.volume_part ** 2).mean() / mse) if mse else None,
            efficiency_share=float((d.efficiency_part ** 2).mean() / mse) if mse else None,
            interaction_share=float(2 * (d.volume_part * d.efficiency_part).mean() / mse) if mse else None,
            volume_mae=_mae(d.volume_error), volume_bias=_mean(-d.volume_error),
            implied_efficiency_mean=_mean(d.implied_efficiency),
            actual_efficiency_mean=float(d.actual.sum() / d.volume.sum()) if d.volume.sum() else None,
            volume_sources={str(k): int(v) for k, v in d.volume_source.value_counts().items()})
    priced = df[df.line.notna()]
    if len(priced):
        out["vs_market"] = dict(player_games=int(len(priced)), model_mae=_mae(priced.error),
                                line_mae=_mae(priced.actual - priced.line),
                                anchored_mae=_mae(priced.actual - priced.anchored_projection),
                                model_bias=_mean(-priced.error), line_bias=_mean(priced.line - priced.actual))
    # Where in the projection range the error sits.
    try:
        bins = pd.qcut(df.model_projection, 4, duplicates="drop")
        out["by_projection_quartile"] = [
            dict(range=f"{iv.left:.1f}-{iv.right:.1f}", player_games=int(len(g)), projection_mean=_mean(g.model_projection),
                 actual_mean=_mean(g.actual), mae=_mae(g.error), bias=_mean(-g.error),
                 above_share=float((g.error > 0).mean()))
            for iv, g in df.groupby(bins, observed=True)]
    except ValueError:
        out["by_projection_quartile"] = []
    return out


def volume_source_comparison(frame: pd.DataFrame) -> list[dict]:
    """How well each pre-game volume estimate predicts the actual volume, on identical rows."""
    rows = []
    for volume, sources in (("targets", ("receiver_projected_targets_v3", "targets_avg_5", "targets_avg_10")),
                            ("carries", ("rb_projected_carries_v3", "carries_avg_5", "carries_avg_10"))):
        stat = "receiving_yards" if volume == "targets" else "rushing_yards"
        d = frame[frame.stat.eq(stat)].dropna(subset=[volume, *sources])
        for source in sources:
            rows.append(dict(volume=volume, source=source, player_games=int(len(d)),
                             mae=_mae(d[volume] - d[source]), bias=_mean(d[source] - d[volume])))
    return rows


def build(frame: pd.DataFrame) -> dict:
    per_stat, by_position = {}, {}
    for stat in DECOMPOSED:
        df = decompose(frame, stat)
        per_stat[stat] = summarize_stat(df)
        by_position[stat] = {str(pos): summarize_stat(g) for pos, g in df.groupby(df.position.fillna("?"))
                             if len(g) >= 20}
    return dict(per_stat=per_stat, by_position=by_position, volume_sources=volume_source_comparison(frame))


def _f(v, digits=1, signed=False):
    if v is None:
        return "-"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def _p(v):
    return "-" if v is None else f"{v:.0%}"


def markdown(meta: dict, result: dict) -> str:
    stats = result["per_stat"]
    lines = ["# NFL Projection Error", "",
             f"Built {meta['built_at']}. Season {meta['season']}, settled games only. One row per player-game-stat: the "
             "last forecast before kickoff, scored on the model's own number (not the line-anchored projection) "
             "against the official box score.", "",
             "## Raw accuracy", "",
             "| Stat | Player-games | Weeks | Actual mean | Projection mean | MAE | Bias | Above projection | Slope | Corr |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for stat, s in stats.items():
        if not s.get("player_games"):
            continue
        lines.append(f"| {stat} | {s['player_games']} | {s['weeks'][0]}-{s['weeks'][-1]} | {_f(s['actual_mean'], 2)} | "
                     f"{_f(s['projection_mean'], 2)} | {_f(s['mae'], 2)} | {_f(s['bias'], 2, True)} | {_p(s['above_share'])} | "
                     f"{_f(s.get('slope'), 2)} | {_f(s.get('correlation'), 2)} |")
    lines += ["", "Bias is projection minus actual (positive = too high). The model forecasts a median, so on "
              "right-skewed stats a negative mean bias is expected; \"Above projection\" is the honest check (50% "
              "for a well-placed median). Slope regresses the outcome on the projection: "
              "below 1 means projections are spread too wide (too sure who will be high or low).", "",
              "## Model vs the market line (rows with a line)", "",
              "| Stat | Player-games | Model MAE | Line MAE | Line-anchored MAE | Model bias | Line bias |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for stat, s in stats.items():
        m = s.get("vs_market")
        if m:
            lines.append(f"| {stat} | {m['player_games']} | {_f(m['model_mae'], 2)} | {_f(m['line_mae'], 2)} | "
                         f"{_f(m['anchored_mae'], 2)} | {_f(m['model_bias'], 2, True)} | {_f(m['line_bias'], 2, True)} |")
    lines += ["", "## Volume vs efficiency", "",
              "Error = volume part + efficiency part, exactly. Shares are of squared error; the interaction term is "
              "what remains when both miss in the same direction (positive) or offset (negative).", "",
              "| Stat | Player-games | MAE | Knew volume | Knew efficiency | Volume share | Efficiency share | Interaction | "
              "Volume MAE | Volume bias | Implied eff. | Actual eff. |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for stat, s in stats.items():
        d = s.get("decomposition")
        if d:
            lines.append(f"| {stat} | {d['player_games']} | {_f(d['mae'], 2)} | {_f(d['knew_volume_mae'], 2)} | "
                         f"{_f(d['knew_efficiency_mae'], 2)} | {_p(d['volume_share'])} | {_p(d['efficiency_share'])} | "
                         f"{_p(d['interaction_share'])} | {_f(d['volume_mae'], 2)} | {_f(d['volume_bias'], 2, True)} | "
                         f"{_f(d['implied_efficiency_mean'], 3)} | {_f(d['actual_efficiency_mean'], 3)} |")
    lines += ["", "\"Knew volume\" keeps the model's implied efficiency and uses the actual targets/carries/attempts; "
              "\"Knew efficiency\" keeps the pre-game volume estimate and uses the actual per-opportunity rate. "
              "The volume estimate is the 5-game average; implied efficiency is the projection divided by it.", "",
              "## Pre-game volume estimates (identical rows)", "",
              "| Volume | Estimate | Player-games | MAE | Bias |", "|---|---|---:|---:|---:|"]
    for r in result["volume_sources"]:
        lines.append(f"| {r['volume']} | {r['source']} | {r['player_games']} | {_f(r['mae'], 2)} | {_f(r['bias'], 2, True)} |")
    lines += ["", "## By position", "",
              "| Stat | Position | Player-games | MAE | Bias | Slope | Volume share | Efficiency share | Volume bias |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for stat, positions in result["by_position"].items():
        for pos, s in sorted(positions.items()):
            d = s.get("decomposition") or {}
            lines.append(f"| {stat} | {pos} | {s['player_games']} | {_f(s['mae'], 2)} | {_f(s['bias'], 2, True)} | "
                         f"{_f(s.get('slope'), 2)} | {_p(d.get('volume_share'))} | {_p(d.get('efficiency_share'))} | "
                         f"{_f(d.get('volume_bias'), 2, True)} |")
    lines += ["", "## By projection quartile", "",
              "| Stat | Projection range | Player-games | Projection mean | Actual mean | MAE | Bias | Above projection |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for stat, s in stats.items():
        for q in s.get("by_projection_quartile") or []:
            lines.append(f"| {stat} | {q['range']} | {q['player_games']} | {_f(q['projection_mean'], 2)} | "
                         f"{_f(q['actual_mean'], 2)} | {_f(q['mae'], 2)} | {_f(q['bias'], 2, True)} | {_p(q['above_share'])} |")
    lines += ["", "Measurement only: nothing here changes a model, a forecast or a bet."]
    return "\n".join(lines) + "\n"


def run(season: int) -> dict:
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        frame = load(conn, season)
    result = build(frame)
    meta = dict(built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), season=season,
                rows_loaded=int(len(frame)), raw_projection_rule=(
                    "forecast_outputs.pricing.model_projection, else probability_trace.model_projection, else "
                    f"projection when unpriced or made before market calibration ({CALIBRATION_INSTALLED})"))
    atomic_json(ROOT / "reports" / "nfl_projection_error_latest.json", dict(meta=meta, **result))
    (ROOT / "reports" / "nfl_projection_error_latest.md").write_text(markdown(meta, result), encoding="utf-8")
    return dict(meta=meta, **result)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2026)
    args = ap.parse_args()
    run(args.season)
    print((ROOT / "reports" / "nfl_projection_error_latest.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
