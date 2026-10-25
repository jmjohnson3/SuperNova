"""NBA player-prop projections and their walk-forward test against DraftKings' pre-game lines.

Features (each from games before the one projected): the player's decayed and recent minutes, per-minute
rates and shot volume, games played, rest, home; the player's team pace and the opponent's allowed
per-minute rates and pace relative to the league. A LightGBM model per stat (points, rebounds, assists,
threes) is refit at the start of every month on all earlier games and projects that month.

Probabilities: a count distribution around the projection, variance = a*mu + b*mu^2 fitted on training
residuals (negative binomial; Poisson when b = 0).

The test (reports/nba_props_model_latest.md): DraftKings player points/rebounds/assists lines from
odds.nba_player_prop_lines, last snapshot before tip only (a third of the 01:00 UTC backfill snapshots
were taken after the game started, i.e. live lines, and are excluded). Compared on the same rows:
    - MAE of the projection vs MAE of the line, and a blend regression of the result on line and model
    - model P(over) minus DraftKings no-vig P(over) >= k: record and ROI at DraftKings' actual price

    python -m nba_props.model [--test-from 2023-11-01]
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats as st

from nfl_pipeline import sharp_math
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.markets import normalize_name

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
ROOT = Path(__file__).resolve().parents[2]
STATS = ("pts", "reb", "ast", "fg3m")
LINE_STAT = {"points": "pts", "rebounds": "reb", "assists": "ast"}
EDGES = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)
LGB_PARAMS = dict(objective="poisson", learning_rate=0.03, n_estimators=400, num_leaves=31, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


def match_name(name: str) -> str:
    return re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", normalize_name(name))


# --- features --------------------------------------------------------------------------------------
def load_games(conn) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM nbaprops.player_games ORDER BY game_date, game_id, player_id", conn)
    num = ["minutes", "pts", "reb", "ast", "fg3m", "fga", "fta", "tov", "oreb", "stl", "blk", "plus_minus"]
    df[num] = df[num].astype(float)
    df["game_date"] = pd.to_datetime(df.game_date)
    return df


def _ewm_prior(g: pd.core.groupby.SeriesGroupBy, halflife: float) -> pd.Series:
    return g.transform(lambda s: s.shift(1).ewm(halflife=halflife, min_periods=1).mean())


def team_context(df: pd.DataFrame) -> pd.DataFrame:
    """Per team-game totals, then each team's prior pace and the rates its opponents have put up against it."""
    tg = df.groupby(["game_id", "game_date", "team", "opponent"], as_index=False)[
        ["minutes", "pts", "reb", "ast", "fg3m", "fga", "fta", "tov", "oreb"]].sum(min_count=1)  # NaN for a scheduled (unplayed) game
    tg["poss"] = tg.fga + 0.44 * tg.fta + tg.tov - tg.oreb
    mins = tg.minutes.clip(lower=200)
    for s in STATS:
        tg[f"{s}_pm"] = tg[s] / mins
    tg["pace"] = tg.poss / mins * 240
    # what the opponent of each row allowed: the row's own numbers, credited to the opponent's defence
    allowed = tg.rename(columns={"team": "def_team"})[["game_id", "game_date", "def_team", "pace"] + [f"{s}_pm" for s in STATS]]
    allowed["def_team"] = tg.opponent
    allowed = allowed.sort_values(["game_date", "game_id"])
    league = allowed.groupby("game_date")[[f"{s}_pm" for s in STATS] + ["pace"]].transform("mean")
    g = allowed.groupby("def_team")
    out = allowed[["game_id", "def_team"]].copy()
    for c in [f"{s}_pm" for s in STATS] + ["pace"]:
        rel = allowed[c] / league[c].replace(0, np.nan)
        out[f"opp_allow_{c}"] = rel.groupby(allowed.def_team).transform(lambda s: s.shift(1).ewm(halflife=15, min_periods=3).mean())
    own = tg.sort_values(["game_date", "game_id"])[["game_id", "team", "pace"]].copy()
    own["team_pace"] = own.groupby("team").pace.transform(lambda s: s.shift(1).ewm(halflife=15, min_periods=3).mean())
    return out.rename(columns={"def_team": "opponent"}), own[["game_id", "team", "team_pace"]]


def features(df: pd.DataFrame, lineups: pd.DataFrame | None = None, declared_out: set | None = None) -> pd.DataFrame:
    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    g = df.groupby("player_id")
    f = df[["game_id", "player_id", "player_name", "team", "opponent", "game_date", "season", "season_type", "home",
            "minutes"] + list(STATS)].copy()
    f["games_before"] = g.cumcount()
    f["season_games_before"] = df.groupby(["player_id", "season"]).cumcount()
    f["rest_days"] = g.game_date.diff().dt.days.clip(upper=10)
    f["b2b"] = f.rest_days.eq(1).astype(float)
    f["home"] = df.home.astype(float)
    f["playoffs"] = df.season_type.eq("Playoffs").astype(float)
    played = df.minutes.where(df.minutes > 0)
    for hl in (3, 10):
        f[f"min_ewm{hl}"] = _ewm_prior(played.groupby(df.player_id), hl)
    f["min_last"] = g.minutes.shift(1)
    f["min_sd10"] = g.minutes.transform(lambda s: s.shift(1).rolling(10, min_periods=3).std())
    for s in STATS + ("fga", "fta", "tov"):
        rate = (df[s] / df.minutes.where(df.minutes >= 5))
        f[f"{s}_rate_ewm10"] = _ewm_prior(rate.groupby(df.player_id), 10)
        f[f"{s}_rate_ewm40"] = _ewm_prior(rate.groupby(df.player_id), 40)
    for s in STATS:
        f[f"{s}_ewm5"] = _ewm_prior(df[s].where(df.minutes > 0).groupby(df.player_id), 5)
        f[f"{s}_sd10"] = g[s].transform(lambda x: x.shift(1).rolling(10, min_periods=3).std())
        f[f"{s}_base"] = f.min_ewm3 * f[f"{s}_rate_ewm10"]
    opp, own = team_context(df)
    f = f.merge(opp, on=["game_id", "opponent"], how="left").merge(own, on=["game_id", "team"], how="left")
    for s in STATS:
        f[f"{s}_base_adj"] = f[f"{s}_base"] * f[f"opp_allow_{s}_pm"].fillna(1.0)
    if lineups is not None:
        f = add_availability(f, df, lineups, declared_out)
    return f


# --- availability ----------------------------------------------------------------------------------
ROTATION_MINUTES = 12.0   # a player averaging this many recent minutes is part of the rotation
ROTATION_DAYS = 15        # ...if he appeared for the team within this many days
OUT_STATS = ("minutes", "usage", "pts", "reb", "ast", "fg3m")


def final_lineups(conn) -> pd.DataFrame:
    """Last capture per game and player (the backfill's files are the final pre-game state)."""
    return pd.read_sql("""SELECT DISTINCT ON (game_id, player_id) game_id, player_id, team, starter, roster_status
                          FROM nbaprops.lineup_snapshots ORDER BY game_id, player_id, captured_at_utc DESC""", conn)


def add_availability(f: pd.DataFrame, df: pd.DataFrame, lineups: pd.DataFrame,
                     declared_out: set | None = None) -> pd.DataFrame:
    """Starter status, and what each team (and its opponent) is missing from its rotation.

    Out = a rotation player (recent minutes >= ROTATION_MINUTES, on the team within ROTATION_DAYS) who is
    listed Inactive in the game's lineup file. Games without a lineup file (before 2023-24) fall back to
    "did not appear in the box score". Live (unplayed) rows use only declared outs: lineup Inactive plus
    `declared_out` {(game_id, player_id)} from the injury report, so training and live share one meaning.
    """
    d = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    played = d.minutes > 0
    post = d[["player_id", "team", "season", "game_date", "game_id"]].copy()
    usage = d.fga + 0.44 * d.fta + d.tov
    for name, values in (("minutes", d.minutes), ("usage", usage), ("pts", d.pts), ("reb", d.reb),
                         ("ast", d.ast), ("fg3m", d.fg3m)):
        post[f"post_{name}"] = values.where(played).groupby(d.player_id).transform(
            lambda s: s.ewm(halflife=10, min_periods=1).mean())
    post = post[played | d.minutes.isna()]
    # every player seen for the team that season, against every team game that season
    tg = d[["game_id", "game_date", "team", "season"]].drop_duplicates(["game_id", "team"])
    roster = d.loc[played, ["team", "season", "player_id"]].drop_duplicates()
    grid = tg.merge(roster, on=["team", "season"]).sort_values("game_date")
    last = post[post.post_minutes.notna()].sort_values("game_date").rename(columns={"team": "last_team", "game_date": "last_date"})
    grid = pd.merge_asof(grid, last.drop(columns=["season", "game_id"]), left_on="game_date", right_on="last_date",
                         by="player_id", allow_exact_matches=False)
    rotation = (grid.last_team.eq(grid.team) & (grid.game_date - grid.last_date).dt.days.le(ROTATION_DAYS)
                & grid.post_minutes.ge(ROTATION_MINUTES))
    grid = grid[rotation]
    lu = lineups.set_index(["game_id", "player_id"])
    status = grid.set_index(["game_id", "player_id"]).index.map(lu.roster_status.to_dict().get)
    has_file = grid.game_id.isin(set(lineups.game_id))
    appeared = set(zip(d.loc[played, "game_id"], d.loc[played, "player_id"]))
    key = list(zip(grid.game_id, grid.player_id))
    unplayed_game = grid.game_id.isin(set(d.loc[d.minutes.isna(), "game_id"]))
    declared = pd.Series([k in (declared_out or set()) for k in key], index=grid.index)
    out = pd.Series(np.where(has_file, pd.Series(status, index=grid.index).eq("Inactive"),
                             ~pd.Series([k in appeared for k in key], index=grid.index)), index=grid.index)
    out = np.where(unplayed_game, out.where(has_file, False) | declared, out)
    gone = grid[out.astype(bool)]
    agg = gone.groupby(["game_id", "team"]).agg(
        out_count=("player_id", "size"), **{f"out_{s}": (f"post_{s}", "sum") for s in OUT_STATS}).reset_index()
    f = f.merge(agg, on=["game_id", "team"], how="left")
    opp = agg.rename(columns={"team": "opponent", **{c: f"opp_{c}" for c in agg.columns if c.startswith("out_")}})
    f = f.merge(opp, on=["game_id", "opponent"], how="left")
    for c in [c for c in f.columns if c.startswith("out_") or c.startswith("opp_out_")]:
        f[c] = f[c].fillna(0.0)
    # starters: this game's (known from the lineup file) and the player's recent starting rate
    starters = lineups.set_index(["game_id", "player_id"]).starter.to_dict()
    f["starter"] = [starters.get(k) for k in zip(f.game_id, f.player_id)]
    f["starter"] = f.starter.map({True: 1.0, False: 0.0})
    f = f.sort_values(["player_id", "game_date", "game_id"])
    f["starter_rate"] = f.groupby("player_id").starter.transform(lambda s: s.shift(1).ewm(halflife=5, min_periods=1).mean())
    # the player's own recent usage against what his team is missing
    f["usage_ewm10"] = (f.fga_rate_ewm10 + 0.44 * f.fta_rate_ewm10 + f.tov_rate_ewm10) * f.min_ewm10
    f["out_usage_share"] = f.out_usage / (f.out_usage + f.usage_ewm10 * 5).replace(0, np.nan)
    return f.reset_index(drop=True)


def feature_columns(f: pd.DataFrame) -> list[str]:
    skip = {"game_id", "player_id", "player_name", "team", "opponent", "game_date", "season", "season_type",
            "minutes", *STATS}
    return [c for c in f.columns if c not in skip]


# --- walk-forward ----------------------------------------------------------------------------------
def dispersion(y: np.ndarray, mu: np.ndarray) -> tuple[float, float]:
    """var = a*mu + b*mu^2 by least squares on squared residuals (both clipped at zero)."""
    X = np.column_stack([mu, mu ** 2])
    coef, *_ = np.linalg.lstsq(X, (y - mu) ** 2, rcond=None)
    return float(max(coef[0], 0.3)), float(max(coef[1], 0.0))


def p_over(mu: np.ndarray, line: np.ndarray, a: float, b: float) -> np.ndarray:
    """P(X > line) for a count with mean mu and variance a*mu + b*mu^2 (lines are x.5)."""
    mu = np.clip(mu, 0.05, None)
    k = np.floor(line)
    var = a * mu + b * mu ** 2
    if b <= 1e-9:  # (quasi-)Poisson: scale counts by a
        return st.poisson.sf(k / a, mu / a) if a != 1 else st.poisson.sf(k, mu)
    r = mu ** 2 / np.maximum(var - mu, 1e-6)
    return st.nbinom.sf(k, r, r / (r + mu))


def walk_forward(f: pd.DataFrame, test_from: str) -> pd.DataFrame:
    import lightgbm as lgb
    cols = feature_columns(f)
    rows = f[(f.minutes > 0) & (f.games_before >= 3)].copy()
    months = pd.period_range(pd.Timestamp(test_from), rows.game_date.max(), freq="M")
    parts = []
    for month in months:
        start, end = month.start_time, month.end_time
        train, test = rows[rows.game_date < start], rows[(rows.game_date >= start) & (rows.game_date <= end)]
        if test.empty:
            continue
        out = test[["game_id", "player_id", "player_name", "game_date", "season", *STATS]].copy()
        for s in STATS:
            model = lgb.LGBMRegressor(**LGB_PARAMS).fit(train[cols], train[s])
            out[f"pred_{s}"] = model.predict(test[cols])
            fit_mu = model.predict(train[cols].iloc[-60000:])
            a, b = dispersion(train[s].iloc[-60000:].to_numpy(), fit_mu)
            out[f"disp_a_{s}"], out[f"disp_b_{s}"] = a, b
        parts.append(out)
        print(f"  {month}: {len(test)} player-games", flush=True)
    return pd.concat(parts, ignore_index=True)


# --- market test -----------------------------------------------------------------------------------
def load_lines(conn) -> pd.DataFrame:
    lines = pd.read_sql("""
        SELECT player_name, stat, line, over_price, under_price, commence_time_utc::timestamptz AS commence,
               CASE WHEN fetched_at_utc - as_of_date::timestamptz > interval '3 days'
                    THEN (as_of_date::timestamp + interval '1 hour') AT TIME ZONE 'UTC'
                    ELSE COALESCE(updated_at_utc, fetched_at_utc) END AS snapshot
        FROM odds.nba_player_prop_lines
        WHERE bookmaker_key = 'draftkings' AND line IS NOT NULL AND over_price IS NOT NULL AND under_price IS NOT NULL""", conn)
    lines["commence"] = pd.to_datetime(lines.commence, utc=True)
    lines["snapshot"] = pd.to_datetime(lines.snapshot, utc=True)
    lines = lines[lines.snapshot < lines.commence]                      # pre-game only
    lines["minutes_before"] = (lines.commence - lines.snapshot).dt.total_seconds() / 60
    lines = lines[lines.minutes_before <= 30 * 60]
    lines["game_date"] = lines.commence.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    lines["name_key"] = lines.player_name.map(match_name)
    lines["stat"] = lines.stat.map(LINE_STAT)
    lines = lines.sort_values("snapshot").drop_duplicates(["game_date", "name_key", "stat"], keep="last")
    return lines


def market_rows(preds: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    p = preds.assign(name_key=preds.player_name.map(match_name))
    long = []
    for s in ("pts", "reb", "ast"):
        part = p[["game_date", "name_key", "season", s, f"pred_{s}", f"disp_a_{s}", f"disp_b_{s}"]].rename(
            columns={s: "actual", f"pred_{s}": "pred", f"disp_a_{s}": "a", f"disp_b_{s}": "b"})
        long.append(part.assign(stat=s))
    long = pd.concat(long, ignore_index=True)
    m = long.merge(lines[["game_date", "name_key", "stat", "line", "over_price", "under_price", "minutes_before"]],
                   on=["game_date", "name_key", "stat"], how="inner")
    m["line"] = m.line.astype(float)
    m = m[(m.line % 1) != 0]  # whole-number lines can push; keep x.5 only
    m = pd.concat([g.assign(p_model=p_over(g.pred.to_numpy(), g.line.to_numpy(), g.a.iloc[0], g.b.iloc[0]))
                   for _, g in m.groupby(["stat", "a", "b"], sort=False)])
    m["p_market"] = [sharp_math.no_vig_over(int(o), int(u)) for o, u in zip(m.over_price, m.under_price)]
    m["over"] = m.actual > m.line
    return m


def pick_table(m: pd.DataFrame) -> list[dict]:
    rows = []
    for k in EDGES:
        over = m[m.p_model - m.p_market > k]
        under = m[m.p_market - m.p_model > k]
        won = np.concatenate([over.over.to_numpy(), ~under.over.to_numpy()])
        prices = np.concatenate([over.over_price.to_numpy(), under.under_price.to_numpy()])
        payout = np.where(prices > 0, prices / 100, 100 / np.abs(prices))
        profit = np.where(won, payout, -1.0)
        fair = np.concatenate([over.p_market.to_numpy(), 1 - under.p_market.to_numpy()])
        rows.append(dict(edge=k, picks=int(len(won)), overs=int(len(over)), unders=int(len(under)),
                         hit_rate=float(won.mean()) if len(won) else None,
                         market_expected=float(fair.mean()) if len(won) else None,
                         roi=float(profit.mean()) if len(won) else None))
    return rows


def evaluate(m: pd.DataFrame) -> dict:
    out = dict(rows=int(len(m)), seasons=sorted(m.season.unique().tolist()))
    for s, g in m.groupby("stat"):
        X = np.column_stack([np.ones(len(g)), g.line.to_numpy(float), g.pred.to_numpy(float)])
        coef, *_ = np.linalg.lstsq(X, g.actual.to_numpy(float), rcond=None)
        resid = g.actual.to_numpy(float) - X @ coef
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=3))
        logloss = lambda p: float(-np.mean(np.where(g.over, np.log(np.clip(p, 1e-6, 1)), np.log(np.clip(1 - p, 1e-6, 1)))))
        out[s] = dict(rows=int(len(g)), model_mae=float((g.pred - g.actual).abs().mean()),
                      line_mae=float((g.line - g.actual).abs().mean()),
                      blend_model_weight=float(coef[2]), blend_model_se=float(se[2]), blend_line_weight=float(coef[1]),
                      logloss_model=logloss(g.p_model.to_numpy()), logloss_market=logloss(g.p_market.to_numpy()),
                      over_rate=float(g.over.mean()), picks=pick_table(g),
                      by_season={str(k): pick_table(v)[2] for k, v in g.groupby("season")})
    out["all"] = dict(picks=pick_table(m))
    return out


def markdown(ev: dict, model_mae: dict) -> str:
    pct = lambda v, signed=False: "-" if v is None else (f"{v:+.1%}" if signed else f"{v:.1%}")
    L = ["# NBA Props Model vs DraftKings (pre-game lines)", "",
         f"Walk-forward monthly refits; {ev['rows']} player-game-stat rows with a DraftKings line taken before tip "
         f"(seasons {', '.join(ev['seasons'])}). Live-snapshot backfill rows are excluded.", "",
         "All player-games (projection MAE): " + ", ".join(f"{k} {v:.2f}" for k, v in model_mae.items()), "",
         "| Stat | Rows | Model MAE | Line MAE | Model weight in blend (se) | Log loss model | Log loss DK no-vig | Over rate |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in ("pts", "reb", "ast"):
        if s in ev:
            e = ev[s]
            L.append(f"| {s} | {e['rows']} | {e['model_mae']:.2f} | {e['line_mae']:.2f} | {e['blend_model_weight']:+.2f} "
                     f"({e['blend_model_se']:.2f}) | {e['logloss_model']:.4f} | {e['logloss_market']:.4f} | {pct(e['over_rate'])} |")
    for name, rows in [("All stats", ev["all"]["picks"])] + [(s, ev[s]["picks"]) for s in ("pts", "reb", "ast") if s in ev]:
        L += ["", f"## {name}: model P(over) vs DraftKings no-vig", "",
              "| Edge > | Picks | Over / Under | Hit rate | DK fair expected | ROI at DK price |", "|---:|---:|---:|---:|---:|---:|"]
        L += [f"| {r['edge']:.0%} | {r['picks']} | {r['overs']} / {r['unders']} | {pct(r['hit_rate'])} | "
              f"{pct(r['market_expected'])} | {pct(r['roi'], True)} |" for r in rows]
    L += ["", "By season at edge > 4%:", ""]
    for s in ("pts", "reb", "ast"):
        if s in ev:
            L.append(f"- {s}: " + "; ".join(f"{k}: {r['picks']} picks, {pct(r['hit_rate'])}, ROI {pct(r['roi'], True)}"
                                            for k, r in ev[s]["by_season"].items()))
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-from", default="2023-11-01")
    ap.add_argument("--no-availability", action="store_true", help="the v1 feature set (no lineups/absences)")
    args = ap.parse_args()
    with psycopg2.connect(PG_DSN) as conn:
        games, lines = load_games(conn), load_lines(conn)
        lineups = None if args.no_availability else final_lineups(conn)
    f = features(games, lineups)
    preds = walk_forward(f, args.test_from)
    suffix = "_v1" if args.no_availability else ""
    preds.to_csv(ROOT / "reports" / f"nba_props_model_predictions{suffix}.csv.gz", index=False)
    model_mae = {s: float((preds[f"pred_{s}"] - preds[s]).abs().mean()) for s in STATS}
    m = market_rows(preds, lines)
    ev = evaluate(m)
    ev["model_mae_all_games"] = model_mae
    ev["feature_set"] = "v1" if args.no_availability else "v2-availability"
    atomic_json(ROOT / "reports" / f"nba_props_model{suffix}_latest.json", clean(ev))
    text = markdown(ev, model_mae)
    (ROOT / "reports" / f"nba_props_model{suffix}_latest.md").write_bytes(text.encode())
    print(text)


if __name__ == "__main__":
    main()
