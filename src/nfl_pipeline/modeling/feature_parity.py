"""Live-vs-training feature parity: rebuild past live snapshots and compare them with the training table.

The production model is trained on features.nfl_player_game_training_features (built after games) and
scored on the live snapshot (predict_player_props._snapshot_players, built before them). A feature that
differs between the two is an input the model never trained on. On 2026 weeks 5-6 the teammate carry
total averaged 34 live against 28 in training.

For each past slate this rebuilds the live snapshot exactly as live scoring would have, with the same
loaders and an as-of cutoff an hour before the day's first kickoff, joins it to the training rows for
the same player-games, and reports:

- per feature: coverage on both sides and the relative mean difference;
- team membership: who the live snapshot counted in each team's role group (players who went on to
  play, players who did not, cold-start roster members given a position prior) against who training
  counted (players who played, plus teammates ruled out).

It is the yardstick for the parity fix: a fix is done when the team-relative features match.

    python -m nfl_pipeline.modeling.feature_parity [--season 2026] [--weeks 3 4 5 6]
"""
from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

from nfl_pipeline import features as feature_contract
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling import predict_player_props as props

ROOT = Path(__file__).resolve().parents[3]
KEY = ["game_id", "player_id"]
IDENTIFIERS = {"season", "week", "season_type", "game_id", "game_date_et", "player_id", "player_name",
               "team_abbr", "opponent_abbr", "position"}
OUTCOMES = set(feature_contract.TARGET_STATS) | set(feature_contract.USAGE_STATS)
MATCH_TOLERANCE = 0.02


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(object).map(lambda v: float(v) if isinstance(v, (bool, np.bool_)) else v),
                         errors="coerce")


def slates(conn, season: int, weeks: list[int]) -> list[tuple]:
    days = pd.read_sql("""SELECT game_date_et, MIN(start_ts_utc) AS first_kick FROM raw.nfl_games
                          WHERE season = %(season)s AND week = ANY(%(weeks)s) AND status = 'final'
                          GROUP BY game_date_et ORDER BY game_date_et""", conn, params=dict(season=season, weeks=weeks))
    return [(r.game_date_et, pd.Timestamp(r.first_kick).to_pydatetime() - timedelta(hours=1)) for r in days.itertuples()]


def live_snapshot(conn, day, cutoff) -> pd.DataFrame:
    games = props._load_games(conn, day, cutoff)
    history = props._load_history(conn, day)
    context = props._load_player_context(conn, day, cutoff)
    return props._snapshot_players(history, games, day, [], 120, context)


def compare(live: pd.DataFrame, table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    joined = live.merge(table, on=KEY, suffixes=("_live", "_table"))
    rows = []
    for feature in features:
        a, b = feature + "_live", feature + "_table"
        if a not in joined or b not in joined:
            rows.append(dict(feature=feature, rows=int(len(joined)), status="missing_live" if a not in joined else "missing_table"))
            continue
        x, y = _numeric(joined[a]), _numeric(joined[b])
        both = x.notna() & y.notna()
        relative = float((x[both] - y[both]).abs().mean() / (y[both].abs().mean() + 1e-9)) if both.any() else None
        rows.append(dict(feature=feature, rows=int(len(joined)), live_coverage=float(x.notna().mean()),
                         table_coverage=float(y.notna().mean()), relative_difference=relative,
                         live_mean=float(x[both].mean()) if both.any() else None,
                         table_mean=float(y[both].mean()) if both.any() else None,
                         status="match" if relative is not None and relative <= MATCH_TOLERANCE else "differs"))
    return pd.DataFrame(rows)


def membership(live: pd.DataFrame, table: pd.DataFrame) -> dict:
    """Who each side counted in a team's role group, and what each group adds to the carry/target totals."""
    played = table[KEY + ["team_abbr"]].assign(played=True)
    m = live.merge(played, on=KEY + ["team_abbr"], how="left")
    m["played"] = m.played.fillna(False).astype(bool)
    m["cold_start"] = m.get("cold_start", pd.Series(False, index=m.index)).fillna(False).astype(bool)
    out = {}
    for stat in ("carries", "targets"):
        value = _numeric(m[f"{stat}_avg_5"]).fillna(0.0).clip(lower=0.0)
        groups = dict(played=m.played & ~m.cold_start, did_not_play=~m.played & ~m.cold_start, cold_start=m.cold_start)
        per_team = {name: value[mask].groupby([m.game_id[mask], m.team_abbr[mask]]).sum() for name, mask in groups.items()}
        frame = pd.DataFrame(per_team).fillna(0.0)
        frame["live_total"] = _numeric(m.groupby(["game_id", "team_abbr"])[f"team_player_{stat}_avg5_sum"].first())
        table_total = _numeric(table.groupby(["game_id", "team_abbr"])[f"team_player_{stat}_avg5_sum"].first())
        frame = frame.join(table_total.rename("table_total"), how="inner")
        out[stat] = {k: float(v) for k, v in frame.mean().items()} | dict(team_games=int(len(frame)))
    out["live_members_per_team"] = float(m.groupby(["game_id", "team_abbr"]).size().mean())
    out["members_who_did_not_play"] = float((~m.played & ~m.cold_start).groupby([m.game_id, m.team_abbr]).sum().mean())
    out["cold_start_members"] = float(m.cold_start.groupby([m.game_id, m.team_abbr]).sum().mean())
    return out


def run(season: int, weeks: list[int]) -> dict:
    features = [c for c in feature_contract.FEATURE_COLUMNS if c not in IDENTIFIERS and c not in OUTCOMES]
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        table = pd.read_sql("SELECT * FROM features.nfl_player_game_training_features WHERE season = %(s)s AND week = ANY(%(w)s)",
                            conn, params=dict(s=season, w=weeks))
        table["player_id"] = table.player_id.astype(str)
        lives = []
        for day, cutoff in slates(conn, season, weeks):
            snap = live_snapshot(conn, day, cutoff)
            if not snap.empty:
                snap = snap.assign(player_id=snap.player_id.astype(str), slate=str(day))
                lives.append(snap[snap.game_id.isin(set(table.game_id))])
    live = pd.concat(lives, ignore_index=True) if lives else pd.DataFrame()
    result = compare(live, table, features) if not live.empty else pd.DataFrame()
    doc = dict(season=season, weeks=weeks, live_rows=int(len(live)),
               joined_rows=int(result.rows.max()) if not result.empty else 0,
               features=result.to_dict("records"), membership=membership(live, table) if not live.empty else {},
               summary=dict(result.status.value_counts().to_dict()) if not result.empty else {})
    atomic_json(ROOT / "reports" / "nfl_feature_parity_latest.json", clean(doc))
    (ROOT / "reports" / "nfl_feature_parity_latest.md").write_text(markdown(doc), encoding="utf-8")
    return doc


def markdown(doc: dict) -> str:
    f = lambda v, d=2: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"
    lines = ["# NFL Live-vs-Training Feature Parity", "",
             f"Season {doc['season']}, weeks {', '.join(map(str, doc['weeks']))}: live snapshots rebuilt with the production "
             f"loaders at each slate's cutoff ({doc['live_rows']} rows), joined to the training table on "
             f"{doc['joined_rows']} player-games. A feature matches when its relative mean difference is at most "
             f"{MATCH_TOLERANCE:.0%}.", "", "Summary: " + ", ".join(f"{k} {v}" for k, v in doc["summary"].items()), ""]
    m = doc.get("membership") or {}
    if m:
        lines += ["## Team role groups", "",
                  f"Live counts {f(m['live_members_per_team'], 1)} players per team: {f(m['members_who_did_not_play'], 1)} who did not "
                  f"go on to play and {f(m['cold_start_members'], 1)} cold-start roster members with a position prior.", "",
                  "| Total | Live | Training | Live from players who played | ...who did not play | ...cold-start priors |",
                  "|---|---:|---:|---:|---:|---:|"]
        for stat in ("carries", "targets"):
            s = m[stat]
            lines.append(f"| teammate {stat} (5-game avg sum) | {f(s['live_total'])} | {f(s['table_total'])} | {f(s['played'])} | "
                         f"{f(s['did_not_play'])} | {f(s['cold_start'])} |")
        lines.append("")
    differs = sorted((r for r in doc["features"] if r.get("status") != "match"),
                     key=lambda r: -(r.get("relative_difference") or 9))
    lines += ["## Features that differ (worst first)", "",
              "| Feature | Status | Relative difference | Live mean | Training mean | Live coverage | Training coverage |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in differs[:60]:
        lines.append(f"| {r['feature']} | {r['status']} | {f(r.get('relative_difference'), 3)} | {f(r.get('live_mean'))} | "
                     f"{f(r.get('table_mean'))} | {f(r.get('live_coverage'))} | {f(r.get('table_coverage'))} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--weeks", type=int, nargs="+", default=[3, 4, 5, 6])
    args = ap.parse_args()
    doc = run(args.season, args.weeks)
    print((ROOT / "reports" / "nfl_feature_parity_latest.md").read_text(encoding="utf-8")[:6000])
    print(json.dumps(doc["summary"]))


if __name__ == "__main__":
    main()
