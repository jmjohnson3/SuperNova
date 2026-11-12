"""Challenger: does knowing which teammates are ruled out improve the targets and carries forecast?

modeling/volume_model forecasts volume from each player's own history and leaves out the table's teammate
injury scores because they do not match their live values. This adds a small set built from two sources that
are the same before a game and after it: earlier box scores and the week's injury report.

    vacated     recent targets/carries of teammates listed Out or Doubtful this week (all, and the part that
                is new: they played the team's recent games)
    returning   recent volume of teammates who missed recent games and are on this week's report without an
                Out or Doubtful tag
    without     how the player's own volume moved in earlier games without each of those teammates, shrunk
    also        vacated targets at his own position, whether the team's quarterback is out, how many are out

A teammate counts if he played for the team in one of its previous four games this season. Nothing uses who
actually played in the game being forecast. Same walk-forward and parameters as modeling/volume_model, on its
parity-safe features with and without these inputs, paired by NFL week; then the same yardage check on the
live 2026 forecasts. Research only: no artifact is saved and nothing live changes.

    python -m nfl_pipeline.modeling.volume_absence [--seasons 2023 2024 2025 2026]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling import volume_model

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports" / "nfl_volume_absence_latest.md"
LOOKBACK = 4                    # team games: a teammate counts if he played in one of these
RECENT = 5                      # games in a player's recent volume
EXPOSURE = 3                    # team games over which an absence stops being new
SHRINK = 2.0                    # games with and without a teammate before the difference counts for half
OUT = ("Out", "Doubtful", "Injured Reserve")
VOLUMES = ("targets", "carries")
FEATURES = [f"{kind}_{v}" for v in VOLUMES for kind in ("vacated", "vacated_new", "returning", "without_adj")] + [
    "vacated_position_targets", "quarterback_out", "teammates_out"]
BIG = {"targets": 8.0, "carries": 8.0}       # newly vacated volume that makes a player-game "disrupted"


def load_injuries(conn) -> pd.DataFrame:
    return pd.read_sql("SELECT DISTINCT season, week, team_abbr, player_id, report_status FROM raw.nfl_injuries "
                       "WHERE season >= 2019 AND player_id IS NOT NULL", conn)


def teammates(frame: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game and recent teammate: his recent volume, how many of the team's last games he
    missed, and what this week's injury report says. Built from earlier games and the report only."""
    played = frame[["season", "week", "game_id", "game_date_et", "team_abbr", "player_id", "position", *VOLUMES]].copy()
    played[list(VOLUMES)] = played[list(VOLUMES)].fillna(0.0)
    played = played.sort_values(["player_id", "game_date_et", "game_id"])
    for v in VOLUMES:
        played[f"recent_{v}"] = played.groupby("player_id")[v].transform(lambda s: s.rolling(RECENT, min_periods=1).mean())
    games = played[["season", "week", "game_id", "game_date_et", "team_abbr"]].drop_duplicates(["game_id", "team_abbr"])
    games = games.sort_values(["team_abbr", "season", "game_date_et", "game_id"])
    games["seq"] = games.groupby(["team_abbr", "season"]).cumcount()
    played = played.merge(games[["game_id", "team_abbr", "seq"]], on=["game_id", "team_abbr"])
    earlier = played[["season", "team_abbr", "player_id", "position", "seq", *[f"recent_{v}" for v in VOLUMES]]].rename(columns={"seq": "seen"})
    grid = games.merge(earlier, on=["season", "team_abbr"])
    grid = grid[(grid.seen < grid.seq) & (grid.seen >= grid.seq - LOOKBACK)]
    window = grid[grid.seen >= grid.seq - EXPOSURE].groupby(["game_id", "team_abbr", "player_id"]).size().rename("appeared")
    grid = grid.sort_values("seen").drop_duplicates(["game_id", "team_abbr", "player_id"], keep="last")
    grid = grid.merge(window.reset_index(), on=["game_id", "team_abbr", "player_id"], how="left")
    span = np.minimum(grid.seq, EXPOSURE)
    grid["missed"] = 1.0 - grid.appeared.fillna(0.0) / span                       # share of the team's last games he missed
    report = injuries.assign(out=injuries.report_status.isin(OUT)).groupby(["season", "week", "team_abbr", "player_id"]).out.max().reset_index()
    grid = grid.merge(report, on=["season", "week", "team_abbr", "player_id"], how="left")
    grid["listed"] = grid.out.notna()
    grid["out"] = grid.out.fillna(False).astype(bool)
    grid["new"] = grid.out * (1.0 - grid.missed)
    grid["back"] = (~grid.out & grid.listed) * grid.missed
    return grid.rename(columns={"player_id": "mate", "position": "mate_position"})[
        ["game_id", "team_abbr", "mate", "mate_position", "out", "new", "back", "missed", *[f"recent_{v}" for v in VOLUMES]]]


def pair_history(frame: pd.DataFrame, mates: pd.DataFrame) -> pd.DataFrame:
    """Per player-game and recent teammate: the player's volume in earlier games with and without him, as a
    shrunk difference (without minus with). The game's own row is never in its own history."""
    played = frame[["game_id", "game_date_et", "team_abbr", "player_id", "position", *VOLUMES]].copy()
    played[list(VOLUMES)] = played[list(VOLUMES)].fillna(0.0)
    pairs = played.merge(mates, on=["game_id", "team_abbr"])
    pairs = pairs[pairs.player_id != pairs.mate]
    there = played[["game_id", "player_id"]].rename(columns={"player_id": "mate"}).assign(mate_played=1.0)
    pairs = pairs.merge(there, on=["game_id", "mate"], how="left").fillna({"mate_played": 0.0})
    pairs = pairs.sort_values(["player_id", "mate", "game_date_et", "game_id"])
    key = [pairs.player_id, pairs.mate]
    before = lambda values: values.groupby(key).cumsum() - values
    n_with, n_without = before(pairs.mate_played), before(1.0 - pairs.mate_played)
    weight = n_with / (n_with + SHRINK) * n_without / (n_without + SHRINK)
    for v in VOLUMES:
        with_mean = before(pairs[v] * pairs.mate_played) / n_with.where(n_with > 0)
        without_mean = before(pairs[v] * (1.0 - pairs.mate_played)) / n_without.where(n_without > 0)
        pairs[f"delta_{v}"] = ((without_mean - with_mean) * weight).fillna(0.0)
    return pairs.drop(columns=["mate_played"])


def absence_features(frame: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """One row per player-game with every column in FEATURES."""
    mates = teammates(frame, injuries)
    pairs = pair_history(frame, mates)
    change = pairs.new - pairs.back
    cols = {}
    for v in VOLUMES:
        recent = pairs[f"recent_{v}"]
        cols[f"vacated_{v}"] = recent * pairs.out
        cols[f"vacated_new_{v}"] = recent * pairs.new
        cols[f"returning_{v}"] = recent * pairs.back
        cols[f"without_adj_{v}"] = pairs[f"delta_{v}"] * change
    cols["vacated_position_targets"] = pairs.recent_targets * pairs.out * (pairs.mate_position == pairs.position)
    cols["quarterback_out"] = (pairs.out & (pairs.mate_position == "QB") & (pairs.missed < 0.5)).astype(float)
    cols["teammates_out"] = pairs.out.astype(float)
    sums = pd.DataFrame(cols).groupby([pairs.game_id, pairs.player_id]).sum()
    sums["quarterback_out"] = sums.quarterback_out.clip(upper=1.0)
    out = frame[["game_id", "player_id"]].merge(sums.reset_index(), on=["game_id", "player_id"], how="left")
    return out.fillna({c: 0.0 for c in FEATURES})


def compare(base: pd.DataFrame, informed: pd.DataFrame, target: str) -> dict:
    """The two walk-forward forecasts on the same player-games, paired by NFL week."""
    rows = base[["season", "week", "game_id", "player_id", "position", target, "model", f"vacated_new_{target}", f"returning_{target}"]].merge(
        informed[["game_id", "player_id", "model"]], on=["game_id", "player_id"], suffixes=("_base", "_absence")).dropna(subset=["model_base", "model_absence"])
    rows["y"] = rows[target].astype(float)
    groups = {"All": rows, f"{BIG[target]:.0f}+ {target} newly vacated": rows[rows[f"vacated_new_{target}"] >= BIG[target]],
              f"{BIG[target]:.0f}+ {target} returning": rows[rows[f"returning_{target}"] >= BIG[target]],
              **{f"Season {int(s)}": g for s, g in rows.groupby("season")}, **{p: g for p, g in rows.groupby("position")}}
    table = {}
    for label, g in groups.items():
        if g.empty:
            continue
        table[label] = dict(rows=int(len(g)), base=volume_model._scores(g.y, g.model_base), absence=volume_model._scores(g.y, g.model_absence),
                            weekly=volume_model._week_interval(g, "model_absence", "model_base"))
    return table


def markdown(doc: dict) -> str:
    f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"
    L = ["# NFL volume: teammates ruled out (challenger)", "",
         f"Built {doc['built_at']}. Walk-forward 4-week blocks over {', '.join(map(str, doc['seasons']))}, each trained on earlier games only, "
         "same parameters and parity-safe features as the volume model, with and without the absence inputs. Research only; nothing live changes.", "",
         "Absence inputs come from earlier box scores and the week's injury report (Out, Doubtful or injured reserve): vacated and newly vacated "
         "targets/carries, volume returning, the player's own volume without those teammates, vacated targets at his position, quarterback out.", ""]
    for target, table in doc["volume"].items():
        L += [f"## {target}", "", "| Player-games | Rows | Volume model MAE | With absence inputs | Change | Weekly difference (95% interval) |",
              "|---|---:|---:|---:|---:|---:|"]
        for label, r in table.items():
            w = r["weekly"]
            interval = "-" if w["mean"] is None else f"{w['mean']:+.3f} ({w['lower']:+.3f} to {w['upper']:+.3f}, {w['weeks']} weeks)"
            L.append(f"| {label} | {r['rows']} | {f(r['base']['mae'])} | {f(r['absence']['mae'])} | {r['absence']['mae'] / r['base']['mae'] - 1:+.1%} | {interval} |")
        L.append("")
    if doc.get("yardage"):
        L += ["## Does it help yards? (live 2026 settled forecasts, same rows)", "",
              "Rescaled: the live projection moved from the 5-game-average volume to the model's out-of-sample volume.", "",
              "| Stat | Rows | Production MAE | Rescaled, volume model | Rescaled, with absence inputs |", "|---|---:|---:|---:|---:|"]
        for stat, y in doc["yardage"].items():
            L.append(f"| {stat} | {y['rows']} | {f(y['production_mae'], 2)} | {f(y['base_mae'], 2)} | {f(y['absence_mae'], 2)} |")
        L.append("")
    cover = doc["coverage"]
    L += [f"Coverage: {cover['with_someone_out']:.0%} of test player-games have a recent teammate ruled out; "
          f"{cover['newly_vacated']:.0%} have {BIG['targets']:.0f}+ targets newly vacated.", "",
          "Negative is better. The weekly difference is the mean over NFL weeks of the MAE with the absence inputs minus without."]
    return "\n".join(L) + "\n"


def run(seasons: list[int], log=print) -> dict:
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        frame = volume_model.load(conn)
        injuries = load_injuries(conn)
        servable, _ = volume_model.serving_parity(conn, frame, volume_model.candidate_features())
        log(f"  {len(frame)} player-games, {len(servable)} parity-safe features")
        frame = frame.merge(absence_features(frame, injuries), on=["game_id", "player_id"], how="left")
        doc = dict(built_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), seasons=seasons, features=FEATURES, volume={},
                   research_only=True, production_changed=False)
        base_oof, informed_oof = {}, {}
        for target, spec in volume_model.TARGETS.items():
            population = frame[frame.position.isin(spec["positions"]) & frame[target].notna()]
            features, _ = volume_model.leak_guard(population[population.season.lt(min(seasons))], target, servable)
            log(f"  {target}: volume model")
            base_oof[target] = volume_model.walk_forward(frame, target, features, seasons)
            log(f"  {target}: with absence inputs")
            informed_oof[target] = volume_model.walk_forward(frame, target, features + FEATURES, seasons)
            doc["volume"][target] = compare(base_oof[target], informed_oof[target], target)
        test = base_oof["targets"]
        doc["coverage"] = dict(with_someone_out=float((test.teammates_out > 0).mean()),
                               newly_vacated=float((test.vacated_new_targets >= BIG["targets"]).mean()))
        doc["yardage"] = {}
        if max(seasons) >= 2026:
            base, informed = (volume_model.yardage_check(oof, max(seasons), conn) for oof in (base_oof, informed_oof))
            doc["yardage"] = {stat: dict(rows=base[stat]["rows"], production_mae=base[stat]["production_mae"], base_mae=base[stat]["rescaled_mae"],
                                         absence_mae=informed[stat]["rescaled_mae"]) for stat in base if stat in informed}
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
