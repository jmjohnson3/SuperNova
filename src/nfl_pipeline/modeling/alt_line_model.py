"""Price alternate prop lines ("60+ receiving yards") from a book's main line, and test that pricing.

Alternate ladders are over-only at every US book and Pinnacle posts none, so a rung can only be
valued by carrying the sharp main line's fair probability out to the rung's threshold. sharp_math
does that with a normal curve, but only within 3 yards; ladders sit 10-50 yards away, where the
outcome distribution is skewed (receivers' spike games) and a normal tail is wrong in exactly the
place books charge the most margin.

This fits the outcome distribution relative to the main line instead. For each stat and line band,
it keeps the empirical distribution of the ratio (actual + c) / (line + c). A rung's probability is
that distribution's tail, shifted on the logit scale so that the probability at the main line itself
equals the market's no-vig probability:

    logit P(Y > L) = logit P_band(Y > L) + [logit p_over(main) - logit P_band(Y > main)]

Fitted on main lines with both prices (the latest pre-kickoff quote per player-game: 2025 historical
snapshots and this season's lines) against official results. Validation is chronological: fit on
earlier weeks, score rungs on later weeks, and compare with the normal curve the scanner uses.

    python -m nfl_pipeline.modeling.alt_line_model            # fit, validate, report
    python -m nfl_pipeline.modeling.alt_line_model --write    # also install the artifact
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from scipy.stats import norm

from nfl_pipeline import sharp_math
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import MODEL_ROOT, atomic_json
from nfl_pipeline.markets import STAT_BY_MARKET, normalize_name, normalize_team

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = MODEL_ROOT / "player_props" / "alt_line_model.json"
CONTRACT = "nfl-alt-line-model-v1"
STATS = {"receiving_yards": "receiving_yards", "rushing_yards": "rushing_yards", "receptions": "receptions"}
OFFSET = {"receiving_yards": 5.0, "rushing_yards": 5.0, "receptions": 0.5}
BANDS = {"receiving_yards": (0, 20, 35, 55, 1000), "rushing_yards": (0, 20, 40, 65, 1000),
         "receptions": (0, 2, 3.5, 5, 100)}
GRID = np.round(np.linspace(0.0, 6.0, 601), 4)   # ratio grid for the stored survival curves
MIN_BAND_ROWS = 60
TEST_OFFSETS = {"receiving_yards": (-30, -20, -10, -5, 5, 10, 20, 30, 45, 60),
                "rushing_yards": (-30, -20, -10, -5, 5, 10, 20, 30, 45, 60),
                "receptions": (-3, -2, -1, 1, 2, 3, 4)}
_logit = lambda p: math.log(p / (1 - p))
_expit = lambda z: 1 / (1 + math.exp(-z))


# --- data -----------------------------------------------------------------------------------------
def _main_lines_from_payloads(payloads: list[tuple], games: dict) -> list[dict]:
    """Latest pre-kickoff two-sided main line per (game, player, stat, book) from raw Odds API events."""
    rows = {}
    for fetched, payload in payloads:
        home, away = normalize_team(payload.get("home_team") or ""), normalize_team(payload.get("away_team") or "")
        kick = safe_time(payload.get("commence_time"))
        game = games.get((home, away))
        if not game or not kick or safe_time(fetched) >= kick:
            continue
        for book in payload.get("bookmakers", []):
            for market in book.get("markets", []):
                stat = STAT_BY_MARKET.get(market.get("key"))
                if stat not in STATS:
                    continue
                pairs = {}
                for o in market.get("outcomes", []):
                    side = str(o.get("name") or "").lower()
                    if side in ("over", "under") and o.get("description") and o.get("point") is not None:
                        pairs.setdefault((normalize_name(o["description"]), float(o["point"])), {})[side] = o.get("price")
                for (player, line), q in pairs.items():
                    if "over" not in q or "under" not in q:
                        continue
                    key = (game["game_id"], player, stat, book["key"])
                    if key not in rows or safe_time(fetched) > rows[key]["fetched"]:
                        rows[key] = dict(game_id=game["game_id"], season=game["season"], week=game["week"],
                                         player_norm=player, stat=stat, book=book["key"], line=line,
                                         over=q["over"], under=q["under"], fetched=safe_time(fetched))
    return list(rows.values())


def load(conn) -> pd.DataFrame:
    """Main lines (one per player-game-stat, Pinnacle preferred) joined to official results."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='300s'")
        # Team abbreviations repeat across seasons; key by season-specific matchup through commence date.
        cur.execute("SELECT game_id, season, week, home_team_abbr, away_team_abbr, start_ts_utc FROM raw.nfl_games WHERE season >= 2025")
        by_start = {}
        for r in cur.fetchall():
            by_start.setdefault((normalize_team(r["home_team_abbr"]), normalize_team(r["away_team_abbr"])), []).append(dict(r))
        cur.execute("""SELECT fetched_at_utc, payload FROM raw.nfl_api_responses
                       WHERE provider IN ('oddsapi_history', 'oddsapi_watch') AND endpoint IN ('nfl_player_props_history', 'nfl_player_props')""")
        payloads = []
        for r in cur.fetchall():
            payload, observed = r["payload"], r["fetched_at_utc"]
            if isinstance(payload, dict) and "data" in payload and "bookmakers" not in payload:
                # Historical snapshots wrap the event; their time is the snapshot's, not the download's.
                observed = safe_time(payload.get("timestamp")) or observed
                payload = payload["data"]
            if isinstance(payload, dict) and "bookmakers" in payload:
                payloads.append((observed, payload))
    rows = []
    for fetched, payload in payloads:
        home, away = normalize_team(payload.get("home_team") or ""), normalize_team(payload.get("away_team") or "")
        kick = safe_time(payload.get("commence_time"))
        candidates = [g for g in by_start.get((home, away), [])
                      if kick and abs((safe_time(g["start_ts_utc"]) - kick).total_seconds()) <= 6 * 3600]
        if candidates:
            rows.extend(_main_lines_from_payloads([(fetched, payload)], {(home, away): candidates[0]}))
    lines = pd.DataFrame(rows)
    if lines.empty:
        return lines
    # Keep the latest quote per player-game-stat-book, then prefer Pinnacle, else the most common book.
    lines = lines.sort_values("fetched").drop_duplicates(["game_id", "player_norm", "stat", "book"], keep="last")
    lines["rank"] = lines.book.map({"pinnacle": 0, "fanduel": 1, "draftkings": 2}).fillna(3)
    lines = lines.sort_values(["rank", "fetched"]).drop_duplicates(["game_id", "player_norm", "stat"], keep="first")
    lines["p_over"] = [sharp_math.no_vig_over(o, u) for o, u in zip(lines.over, lines.under)]
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT l.game_id, r.player_name_norm, l.receiving_yards::float AS receiving_yards,
                              l.rushing_yards::float AS rushing_yards, l.receptions::float AS receptions
                       FROM raw.nfl_player_gamelogs l JOIN raw.nfl_games g ON g.game_id = l.game_id
                       JOIN raw.nfl_rosters r ON r.player_id = l.player_id AND r.season = g.season AND r.team_abbr = l.team_abbr
                       WHERE g.season >= 2025 AND g.status = 'final'
                         AND (COALESCE(l.offense_snaps,0) > 0 OR COALESCE(l.carries,0) + COALESCE(l.targets,0) > 0)""")
        results = pd.DataFrame([dict(r) for r in cur.fetchall()]).drop_duplicates(["game_id", "player_name_norm"])
    long = results.melt(id_vars=["game_id", "player_name_norm"], value_vars=list(STATS), var_name="stat", value_name="actual")
    merged = lines.merge(long.rename(columns={"player_name_norm": "player_norm"}), on=["game_id", "player_norm", "stat"])
    return merged.dropna(subset=["actual", "p_over"]).reset_index(drop=True)


# --- model ----------------------------------------------------------------------------------------
def _band(stat: str, line: float) -> int:
    edges = BANDS[stat]
    return int(np.searchsorted(edges, line, side="right") - 1)


def fit(frame: pd.DataFrame) -> dict:
    """Empirical survival curve of the line-relative ratio, per stat and line band."""
    model = dict(contract=CONTRACT, offsets=OFFSET, bands=BANDS, grid=GRID.tolist(), curves={}, rows={})
    for stat in STATS:
        df = frame[frame.stat.eq(stat)]
        if len(df) < MIN_BAND_ROWS:
            continue  # no curve: over_probability returns None and the rung stays unpriced
        curves = {}
        for band in range(len(BANDS[stat]) - 1):
            part = df[[_band(stat, l) == band for l in df.line]]
            pool = part if len(part) >= MIN_BAND_ROWS else df  # thin band: fall back to the whole stat
            ratio = np.sort(((pool.actual + OFFSET[stat]) / (pool.line + OFFSET[stat])).to_numpy())
            survival = 1.0 - np.searchsorted(ratio, GRID, side="right") / len(ratio)
            curves[str(band)] = np.clip(survival, 0.5 / len(ratio), 1 - 0.5 / len(ratio)).round(6).tolist()
        model["curves"][stat] = curves
        model["rows"][stat] = int(len(df))
    return model


def over_probability(model: dict, stat: str, main_line: float, p_over_main: float, threshold: float) -> float | None:
    """P(stat > threshold) given the main line and its no-vig over probability."""
    if stat not in model.get("curves", {}) or not 0 < p_over_main < 1:
        return None
    c = model["offsets"][stat]
    curve = np.asarray(model["curves"][stat][str(_band(stat, main_line))])
    grid = np.asarray(model["grid"])
    tail = lambda x: float(np.interp((x + c) / (main_line + c), grid, curve))
    anchor, raw = tail(main_line), tail(threshold)
    return _expit(_logit(raw) + _logit(p_over_main) - _logit(anchor))


def normal_probability(stat: str, main_line: float, p_over_main: float, threshold: float) -> float:
    """The scanner's current conversion, without its 3-yard limit, for comparison."""
    sigma = sharp_math.LINE_MODEL[stat]["sigma"]
    mean = main_line + sigma * norm.ppf(p_over_main)
    return float(1 - norm.cdf(threshold, loc=mean, scale=sigma))


# --- validation -----------------------------------------------------------------------------------
def rungs(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in frame.itertuples():
        for offset in TEST_OFFSETS[r.stat]:
            threshold = r.line + offset
            if threshold <= 0:
                continue
            rows.append(dict(game_id=r.game_id, season=r.season, week=r.week, stat=r.stat, line=r.line,
                             p_over=r.p_over, offset=offset, threshold=threshold, hit=float(r.actual > threshold)))
    return pd.DataFrame(rows)


def validate(frame: pd.DataFrame, test_from: tuple[int, int]) -> dict:
    """Fit on games before `test_from` (season, week); score rungs on games from it onward."""
    order = frame.season * 100 + frame.week
    cut = test_from[0] * 100 + test_from[1]
    train, test = frame[order < cut], frame[order >= cut]
    model = fit(train)
    grid = rungs(test)
    grid["fitted"] = [over_probability(model, s, l, p, t) for s, l, p, t in zip(grid.stat, grid.line, grid.p_over, grid.threshold)]
    grid["normal"] = [normal_probability(s, l, p, t) for s, l, p, t in zip(grid.stat, grid.line, grid.p_over, grid.threshold)]
    grid = grid.dropna(subset=["fitted"])
    out = dict(train_rows=int(len(train)), test_rows=int(len(test)), test_from=list(test_from), stats={})
    for stat, g in grid.groupby("stat"):
        brier = {m: float(((g[m] - g.hit) ** 2).mean()) for m in ("fitted", "normal")}
        by_offset = []
        for offset, h in g.groupby("offset"):
            by_offset.append(dict(offset=float(offset), rungs=int(len(h)), hit_rate=float(h.hit.mean()),
                                  fitted=float(h.fitted.mean()), normal=float(h.normal.mean())))
        calib = []
        g = g.assign(bucket=pd.cut(g.fitted, [0, .05, .1, .2, .35, .5, .65, .8, .9, 1.0]))
        for bucket, h in g.groupby("bucket", observed=True):
            calib.append(dict(bucket=str(bucket), rungs=int(len(h)), predicted=float(h.fitted.mean()), actual=float(h.hit.mean())))
        out["stats"][stat] = dict(rungs=int(len(g)), player_games=int(g.game_id.nunique()), brier=brier,
                                  by_offset=by_offset, calibration=calib)
    return out


def markdown(doc: dict) -> str:
    v = doc["validation"]
    lines = ["# NFL Alternate-Line Model", "",
             f"Fitted {doc['fitted_at']} on {doc['model_rows']} main lines with results. Validation: fit on games before "
             f"{v['test_from'][0]} week {v['test_from'][1]} ({v['train_rows']} lines), score 'X+' rungs around each later main "
             f"line ({v['test_rows']} lines) against what happened. Lower Brier is better; the normal curve is the scanner's "
             "current conversion without its 3-yard limit.", ""]
    for stat, s in v["stats"].items():
        lines += [f"## {stat}", "", f"{s['rungs']} rungs. Brier: fitted {s['brier']['fitted']:.4f}, normal {s['brier']['normal']:.4f}.", "",
                  "| Rung vs main line | Rungs | Actually hit | Fitted said | Normal said |", "|---:|---:|---:|---:|---:|"]
        for r in s["by_offset"]:
            lines.append(f"| {r['offset']:+g} | {r['rungs']} | {r['hit_rate']:.1%} | {r['fitted']:.1%} | {r['normal']:.1%} |")
        lines += ["", "| Fitted probability | Rungs | Predicted | Actual |", "|---|---:|---:|---:|"]
        for r in s["calibration"]:
            lines.append(f"| {r['bucket']} | {r['rungs']} | {r['predicted']:.1%} | {r['actual']:.1%} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--test-from", default="2026-4", help="season-week where held-out scoring starts")
    args = ap.parse_args()
    season, week = map(int, args.test_from.split("-"))
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        frame = load(conn)
    validation = validate(frame, (season, week))
    model = fit(frame)
    doc = dict(model, fitted_at=datetime.now(timezone.utc).isoformat(), model_rows=int(len(frame)),
               seasons=sorted(int(s) for s in frame.season.unique()), validation=validation)
    atomic_json(ROOT / "reports" / "nfl_alt_line_model_latest.json", {k: v for k, v in doc.items() if k != "curves"})
    (ROOT / "reports" / "nfl_alt_line_model_latest.md").write_text(markdown(doc), encoding="utf-8")
    if args.write:
        atomic_json(ARTIFACT, {k: v for k, v in doc.items() if k != "validation"})
    print(markdown(doc))


if __name__ == "__main__":
    main()
