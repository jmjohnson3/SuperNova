"""NBA prop challenger: do the market's game line, a model of minutes, or new information about tonight's
minutes make the projections better?

Research only. It reads the pinned model's features and test (nba_props.model) and changes nothing in them;
nothing here feeds a live forecast, a flag or a paper bet.

Variants, all walk-forward (monthly refits on earlier games only) and scored on the same DraftKings lines:
    baseline        the current feature set (v2, with availability)
    game_line       + DraftKings' pre-game spread and total: the team's implied points, the opponent's, the
                    size of the spread (blowout risk)
    minutes_model   + a separate model of the player's minutes tonight (starter, who is out, rest), itself
                    walk-forward, and the per-minute rates applied to it
    both            game_line + minutes_model (the minutes model also sees the spread)
    known_minutes   both, but with the minutes actually played: not attainable, it shows how much of the
                    remaining error is minutes
    minutes_info    + nba_props.minutes_info (who is newly out or back and how the player's minutes moved without
                    them, his own lay-off, tonight's role against his earlier starts and bench games), given to
                    the statistic models and to a minutes model
    minutes_info_direct   the same inputs without the minutes model (optional)
    stint_info      minutes_info + nba_props.stint_info, from stored rotation stints (nba_props.stints): minutes
                    with garbage time taken out, when the player comes on, and whose bench time his floor time
                    goes with (his substitutes) for the teammates newly out or back

A run keeps the variants it does not refit from the stored report, when the test period is the same. With the
minutes information it also scores the minutes forecast alone, one group of inputs at a time.

Probabilities use the pinned count distribution. Its spread is fitted two ways: on the training fit (as the
pinned model does) and on earlier out-of-sample forecasts only ("honest spread").

    python -m nba_props.challenger [--test-from 2023-11-01] [--variants baseline,minutes_info,stint_info] [--threes]
                                                         (about an hour)
    python -m nba_props.challenger --minutes-only        the minutes forecast alone (about 20 minutes)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2

from nba_props import minutes_info, model, stint_info
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import atomic_json

ROOT = model.ROOT
REPORT = ROOT / "reports" / "nba_props_challenger_latest"
STATS = model.STATS
LINE_STATS = ("pts", "reb", "ast")                 # the statistics with DraftKings line history
MINUTES_FROM = "2022-01-01"                        # the minutes model's first out-of-sample month
MIN_HONEST_ROWS = 20000                            # earlier out-of-sample forecasts needed to fit the spread on them
LINE_FEATURES = ["mkt_total", "mkt_spread", "mkt_team_total", "mkt_opp_total", "mkt_abs_spread"]
MINUTES_FEATURES = ["pred_min", "pred_min_ratio"] + [f"{s}_at_pred_min" for s in STATS]
MINUTES_PARAMS = dict(objective="regression", learning_rate=0.06, n_estimators=200, num_leaves=31, min_child_samples=50,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
VARIANTS = {
    "baseline": dict(line=False, minutes=None),
    "game_line": dict(line=True, minutes=None),
    "minutes_model": dict(line=False, minutes="model"),
    "both": dict(line=True, minutes="model"),
    "minutes_info_direct": dict(line=False, minutes=None, info=True),
    "minutes_info": dict(line=False, minutes="model", info=True),
    "stint_info": dict(line=False, minutes="model", info=True, stints=True),
    "known_minutes": dict(line=True, minutes="actual"),
}
DEFAULT_VARIANTS = "baseline,minutes_info,stint_info"      # the others are kept from the stored report
LABEL = {"baseline": "Current model", "game_line": "+ game line", "minutes_model": "+ minutes model",
         "both": "+ both", "known_minutes": "If minutes were known", "minutes_info_direct": "+ minutes information",
         "minutes_info": "+ minutes information and model", "stint_info": "+ minutes information, stints and model"}
STINT_SETS = {"Minutes information": [], "+ stint patterns": stint_info.GROUPS["stints"], "+ substitutes": stint_info.GROUPS["substitutes"],
              "+ both": stint_info.FEATURES}
STUDIES = {"minutes_information": ("New minutes information",
                                   "Absence = which rotation teammates are newly out or back and how the player's minutes moved without "
                                   "them; returning = his own lay-off; role = tonight's lineup against his earlier starts and bench games."),
           "stint_information": ("Rotation stints on top of the minutes information",
                                 "Stint patterns = minutes with garbage time taken out, share of time in the fourth quarter and close "
                                 "finishes, when he first comes on, whether his last game ended early; substitutes = the newly-out and "
                                 "returning teammates weighted by how much his floor time went with their bench time.")}
INPUT_SETS = {"Current inputs": [], "+ absence": minutes_info.GROUPS["absence"], "+ returning": minutes_info.GROUPS["returning"],
              "+ role": minutes_info.GROUPS["role"], "+ all three": minutes_info.FEATURES}
DISRUPTED = 30.0                                   # rotation minutes newly out (or back) that count as a changed line-up


# --- inputs --------------------------------------------------------------------------------------------------
def add_game_line(f: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """The pre-game spread and total from each player's side. `lines` has one row per game: game_date, home_team,
    away_team, spread (the home team's line: negative = home favoured) and total. Games without one stay empty."""
    home = lines.rename(columns={"home_team": "team"})[["game_date", "team", "spread", "total"]]
    away = lines.rename(columns={"away_team": "team"})[["game_date", "team", "spread", "total"]].assign(spread=lambda d: -d.spread)
    side = pd.concat([home, away], ignore_index=True).astype({"spread": float, "total": float})
    side = side.drop_duplicates(["game_date", "team"], keep="last").rename(columns={"spread": "mkt_spread", "total": "mkt_total"})
    out = f.merge(side, on=["game_date", "team"], how="left")
    out["mkt_team_total"] = out.mkt_total / 2 - out.mkt_spread / 2
    out["mkt_opp_total"] = out.mkt_total / 2 + out.mkt_spread / 2
    out["mkt_abs_spread"] = out.mkt_spread.abs()
    return out


def eligible(f: pd.DataFrame) -> pd.Series:
    """The pinned model's rows: the player appeared and has at least three earlier games."""
    return (f.minutes > 0) & (f.games_before >= 3)


def minutes_walk_forward(f: pd.DataFrame, cols: list[str], start: str = MINUTES_FROM, fitter=None, min_rows: int = 1000) -> pd.Series:
    """Out-of-sample minutes for every eligible player-game from `start`: each month is forecast by a model fitted
    on earlier games only. Earlier rows, and rows that are not eligible, stay empty."""
    if fitter is None:
        import lightgbm as lgb
        fitter = lambda X, y: lgb.LGBMRegressor(**MINUTES_PARAMS).fit(X, y)
    rows = f[eligible(f)]
    out = pd.Series(np.nan, index=f.index)
    for month in pd.period_range(pd.Timestamp(start), rows.game_date.max(), freq="M"):
        train = rows[rows.game_date < month.start_time]
        test = rows[(rows.game_date >= month.start_time) & (rows.game_date <= month.end_time)]
        if test.empty or len(train) < min_rows:
            continue
        out.loc[test.index] = np.clip(fitter(train[cols], train.minutes).predict(test[cols]), 0.5, 48.0)
    return out


def add_minutes(f: pd.DataFrame, minutes: pd.Series) -> pd.DataFrame:
    """The forecast minutes, how they compare with the player's recent minutes, and his per-minute rates applied
    to them (with the opponent adjustment the pinned model uses)."""
    out = f.copy()
    out["pred_min"] = minutes.to_numpy()
    out["pred_min_ratio"] = out.pred_min / out.min_ewm3.where(out.min_ewm3 > 0)
    for s in STATS:
        out[f"{s}_at_pred_min"] = out.pred_min * out[f"{s}_rate_ewm10"] * out[f"opp_allow_{s}_pm"].fillna(1.0)
    return out


# --- walk-forward --------------------------------------------------------------------------------------------
def honest_dispersion(history: pd.DataFrame, s: str, fallback: tuple[float, float]) -> tuple[float, float]:
    """The count distribution's spread from earlier out-of-sample forecasts, or the training-fit values while there
    are too few of them."""
    if len(history) < MIN_HONEST_ROWS:
        return fallback
    return model.dispersion(history[s].to_numpy(float), np.clip(history[f"pred_{s}"].to_numpy(float), 0.05, None))


def walk_forward(f: pd.DataFrame, cols: list[str], test_from: str, stats=STATS, fitter=None, log=print) -> pd.DataFrame:
    """The pinned walk-forward on a chosen feature list. Adds honest_a_/honest_b_ per statistic."""
    if fitter is None:
        import lightgbm as lgb
        fitter = lambda X, y: lgb.LGBMRegressor(**model.LGB_PARAMS).fit(X, y)
    rows = f[eligible(f)]
    parts: list[pd.DataFrame] = []
    for month in pd.period_range(pd.Timestamp(test_from), rows.game_date.max(), freq="M"):
        train = rows[rows.game_date < month.start_time]
        test = rows[(rows.game_date >= month.start_time) & (rows.game_date <= month.end_time)]
        if test.empty:
            continue
        out = test[["game_id", "player_id", "player_name", "game_date", "season", "minutes", *stats]].copy()
        earlier = pd.concat(parts, ignore_index=True) if parts else out.iloc[0:0]
        for s in stats:
            fitted = fitter(train[cols], train[s])
            out[f"pred_{s}"] = fitted.predict(test[cols])
            a, b = model.dispersion(train[s].iloc[-60000:].to_numpy(), fitted.predict(train[cols].iloc[-60000:]))
            out[f"disp_a_{s}"], out[f"disp_b_{s}"] = a, b
            out[f"honest_a_{s}"], out[f"honest_b_{s}"] = honest_dispersion(earlier, s, (a, b)) if parts else (a, b)
        parts.append(out)
        log(f"    {month}: {len(test)} player-games")
    return pd.concat(parts, ignore_index=True)


def honest(preds: pd.DataFrame, stats=LINE_STATS) -> pd.DataFrame:
    """The same forecasts with the out-of-sample spread in place of the training-fit one."""
    out = preds.copy()
    for s in stats:
        out[f"disp_a_{s}"], out[f"disp_b_{s}"] = out[f"honest_a_{s}"], out[f"honest_b_{s}"]
    return out


# --- scoring -------------------------------------------------------------------------------------------------
def log_loss(p: np.ndarray, won: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.where(won, np.log(p), np.log(1 - p))


def paired(base: pd.DataFrame, other: pd.DataFrame, draws: int = 2000, seed: int = 7) -> dict:
    """Change against the baseline on the same lines, by statistic: average miss and log loss (negative = better),
    with a 95% interval from resampling game days."""
    key = ["game_date", "name_key", "stat"]
    both = base[key + ["actual", "pred", "p_model", "over"]].merge(other[key + ["pred", "p_model"]], on=key, suffixes=("", "_new"))
    both["d_mae"] = (both.pred_new - both.actual).abs() - (both.pred - both.actual).abs()
    both["d_ll"] = log_loss(both.p_model_new.to_numpy(), both.over.to_numpy()) - log_loss(both.p_model.to_numpy(), both.over.to_numpy())
    rng = np.random.default_rng(seed)
    out = {}
    for s, g in both.groupby("stat"):
        days = g.groupby("game_date").agg(n=("d_mae", "size"), mae=("d_mae", "sum"), ll=("d_ll", "sum"))
        pick = rng.integers(0, len(days), size=(draws, len(days)))
        n = days.n.to_numpy()[pick].sum(axis=1)
        row = dict(rows=int(len(g)))
        for name in ("mae", "ll"):
            sims = days[name].to_numpy()[pick].sum(axis=1) / n
            row[name] = float(days[name].sum() / days.n.sum())
            row[f"{name}_low"], row[f"{name}_high"] = (float(v) for v in np.percentile(sims, [2.5, 97.5]))
        out[s] = row
    return out


def minutes_accuracy(f: pd.DataFrame, test_from: str) -> dict:
    """Average miss in minutes on the test period: the player's recent averages against the minutes model."""
    rows = f[eligible(f) & (f.game_date >= pd.Timestamp(test_from)) & f.pred_min.notna()]
    miss = lambda guess: float((guess - rows.minutes).abs().mean())
    starters = rows[rows.starter.eq(1.0)] if "starter" in rows else rows.iloc[0:0]
    short = rows[rows.out_count >= 2] if "out_count" in rows else rows.iloc[0:0]
    return dict(rows=int(len(rows)), last_3_average=miss(rows.min_ewm3), last_10_average=miss(rows.min_ewm10), model=miss(rows.pred_min),
                starters=dict(rows=int(len(starters)), last_3_average=float((starters.min_ewm3 - starters.minutes).abs().mean()),
                              model=float((starters.pred_min - starters.minutes).abs().mean())) if len(starters) else None,
                two_or_more_out=dict(rows=int(len(short)), last_3_average=float((short.min_ewm3 - short.minutes).abs().mean()),
                                     model=float((short.pred_min - short.minutes).abs().mean())) if len(short) else None)


def situations(f: pd.DataFrame) -> dict[str, pd.Series]:
    """The player-games where minutes are hardest to call, by what was known before the game."""
    out = {"All": pd.Series(True, index=f.index), "Starters": f.starter.eq(1.0), "Team missing 2+ rotation players": f.out_count >= 2,
           f"{DISRUPTED:.0f}+ rotation minutes newly out": f.absent_new_min >= DISRUPTED,
           f"{DISRUPTED:.0f}+ rotation minutes back": f.absent_back_min >= DISRUPTED,
           "First 5 games back from a lay-off": f.games_since_return < 5,
           "Role changed tonight": f.starter_flip.fillna(0.0) != 0}
    return out


def minutes_table(f: pd.DataFrame, forecasts: dict[str, pd.Series], test_from: str, draws: int = 2000, seed: int = 7) -> dict:
    """Average miss in minutes on the test period for each forecast, by situation, on the player-games every
    forecast covers; and the change from the first forecast to the last with a 95% interval over game days."""
    names = list(forecasts)
    keep = eligible(f) & (f.game_date >= pd.Timestamp(test_from))
    for series in forecasts.values():
        keep &= series.notna()
    rows = f[keep]
    miss = {"Last-3 average": (rows.min_ewm3 - rows.minutes).abs()} | {n: (forecasts[n][keep] - rows.minutes).abs() for n in names}
    table = {}
    for label, mask in situations(rows).items():
        if mask.any():
            table[label] = dict(rows=int(mask.sum())) | {n: float(v[mask].mean()) for n, v in miss.items()}
    diff = miss[names[-1]] - miss[names[0]]
    days = pd.DataFrame(dict(day=rows.game_date, d=diff)).groupby("day").d.agg(["size", "sum"])
    pick = np.random.default_rng(seed).integers(0, len(days), size=(draws, len(days)))
    sims = days["sum"].to_numpy()[pick].sum(axis=1) / days["size"].to_numpy()[pick].sum(axis=1)
    low, high = (float(v) for v in np.percentile(sims, [2.5, 97.5]))
    return dict(table=table, first=names[0], last=names[-1], change=float(diff.mean()), change_low=low, change_high=high)


def minutes_information(f: pd.DataFrame, base_cols: list[str], test_from: str, full: pd.Series | None = None, fitter=None, log=print,
                        sets: dict | None = None, first: pd.Series | None = None) -> dict:
    """The minutes forecast alone with each group of new inputs. `first` and `full` are existing forecasts for the
    first set and the last (all of them)."""
    sets = sets or INPUT_SETS
    have = {next(iter(sets)): first, list(sets)[-1]: full}
    forecasts = {}
    for name, extra in sets.items():
        log(f"  minutes forecast: {name.lower()}")
        forecasts[name] = have[name] if have.get(name) is not None else minutes_walk_forward(f, base_cols + extra, start=test_from, fitter=fitter)
    return minutes_table(f, forecasts, test_from)


def summary(ev: dict) -> dict:
    return {s: {k: ev[s][k] for k in ("rows", "model_mae", "line_mae", "blend_model_weight", "blend_model_se", "logloss_model",
                                      "logloss_market")} | dict(picks_4pct=ev[s]["picks"][2]) for s in LINE_STATS if s in ev}


# --- report --------------------------------------------------------------------------------------------------
def information_lines(report: dict) -> list[str]:
    L = []
    for key, (title, blurb) in STUDIES.items():
        info = report.get(key)
        if not info or not info["table"]:
            continue
        sets = [n for n in next(iter(info["table"].values())) if n != "rows"]
        L += ["", f"## {title}: the minutes forecast alone (average miss in minutes)", "", blurb, "",
              "| Player-games | Rows | " + " | ".join(sets) + " |", "|---|---:|" + "---:|" * len(sets)]
        for label, row in info["table"].items():
            L.append(f"| {label} | {row['rows']} | " + " | ".join(f"{row[n]:.2f}" for n in sets) + " |")
        L += ["", f"{info['last']} against {info['first'].lower()}: {info['change']:+.3f} minutes "
                  f"({info['change_low']:+.3f} to {info['change_high']:+.3f})."]
        if info.get("coverage") is not None:
            L += ["", f"Stints: {info['games']} games stored, {info['games_dropped']} left out because their stints do not add up to the box "
                      f"score; {info['coverage']:.0%} of the test player-games have stint inputs."]
    return L


def markdown(report: dict) -> str:
    stat_name = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "Three-pointers"}
    pct = lambda v: "-" if v is None else f"{v:+.1%}"
    names = [n for n in VARIANTS if n in report["variants"]]
    if not names:
        return "\n".join(["# NBA prop challenger: minutes information", "", f"Built {report['built_at']}. Walk-forward from "
                          f"{report['test_from']}. Research only."] + information_lines(report) + [""])
    L = ["# NBA prop challenger: game line, minutes model and minutes information", "",
         f"Built {report['built_at']}. Walk-forward from {report['test_from']}, monthly refits on earlier games only, scored on "
         f"DraftKings lines taken before tip. Research only; nothing here feeds a live forecast.", "",
         "## Average miss on lines (lower is better)", "",
         "| Statistic | Lines | DraftKings line | " + " | ".join(LABEL[n] for n in names) + " |",
         "|---|---:|---:|" + "---:|" * len(names)]
    first = report["variants"][names[0]]["training_fit_spread"]
    for s in LINE_STATS:
        L.append(f"| {stat_name[s]} | {first[s]['rows']} | {first[s]['line_mae']:.3f} | "
                 + " | ".join(f"{report['variants'][n]['training_fit_spread'][s]['model_mae']:.3f}" for n in names) + " |")
    L += ["", "## Probability score (log loss, lower is better)", "",
          "| Statistic | DraftKings no-vig | " + " | ".join(LABEL[n] for n in names) + " | " + " | ".join(f"{LABEL[n]}, honest spread" for n in names) + " |",
          "|---|---:|" + "---:|" * (2 * len(names))]
    for s in LINE_STATS:
        L.append(f"| {stat_name[s]} | {first[s]['logloss_market']:.4f} | "
                 + " | ".join(f"{report['variants'][n]['training_fit_spread'][s]['logloss_model']:.4f}" for n in names) + " | "
                 + " | ".join(f"{report['variants'][n]['honest_spread'][s]['logloss_model']:.4f}" for n in names) + " |")
    L += ["", "## Change against the current model on the same lines (95% interval; negative is better)", "",
          "| Variant | Statistic | Average miss | Log loss |", "|---|---|---:|---:|"]
    for n in names[1:]:
        for s in LINE_STATS:
            d = report["variants"][n]["against_baseline"].get(s)
            if d:
                L.append(f"| {LABEL[n]} | {stat_name[s]} | {d['mae']:+.3f} ({d['mae_low']:+.3f} to {d['mae_high']:+.3f}) | "
                         f"{d['ll']:+.4f} ({d['ll_low']:+.4f} to {d['ll_high']:+.4f}) |")
    step = report["variants"].get("stint_info", {}).get("against_minutes_info")
    if step:
        L += ["", "## What the stints add to the minutes information on the same lines (95% interval; negative is better)", "",
              "| Statistic | Average miss | Log loss |", "|---|---:|---:|"]
        for s in LINE_STATS:
            if s in step:
                d = step[s]
                L.append(f"| {stat_name[s]} | {d['mae']:+.3f} ({d['mae_low']:+.3f} to {d['mae_high']:+.3f}) | "
                         f"{d['ll']:+.4f} ({d['ll_low']:+.4f} to {d['ll_high']:+.4f}) |")
    L += ["", "## Picks where the model disagrees with DraftKings by more than 4% (honest spread)", "",
          "| Variant | Statistic | Picks | Hit rate | ROI at DraftKings' price | Model's weight beside the line (se) |", "|---|---|---:|---:|---:|---:|"]
    for n in names:
        for s in LINE_STATS:
            e = report["variants"][n]["honest_spread"][s]
            p = e["picks_4pct"]
            L.append(f"| {LABEL[n]} | {stat_name[s]} | {p['picks']} | {'-' if p['hit_rate'] is None else format(p['hit_rate'], '.1%')} | "
                     f"{pct(p['roi'])} | {e['blend_model_weight']:+.2f} ({e['blend_model_se']:.2f}) |")
    L += ["", "## Average miss on every player-game (no line needed)", "",
          "| Statistic | " + " | ".join(LABEL[n] for n in names) + " |", "|---|" + "---:|" * len(names)]
    for s in report["variants"][names[0]]["all_games_mae"]:
        L.append(f"| {stat_name[s]} | " + " | ".join(f"{report['variants'][n]['all_games_mae'][s]:.3f}" for n in names) + " |")
    for name, title in (("minutes_model", "without the game line"), ("both", "with the game line")):
        m = report["variants"].get(name, {}).get("minutes")
        if m:
            L += ["", f"## Minutes model {title}: average miss in minutes", "",
                  "| Player-games | Rows | Last-3 average | Last-10 average | Minutes model |", "|---|---:|---:|---:|---:|",
                  f"| All | {m['rows']} | {m['last_3_average']:.2f} | {m['last_10_average']:.2f} | {m['model']:.2f} |"]
            for key, label in (("starters", "Starters"), ("two_or_more_out", "Team missing 2+ rotation players")):
                if m.get(key):
                    L.append(f"| {label} | {m[key]['rows']} | {m[key]['last_3_average']:.2f} | - | {m[key]['model']:.2f} |")
    L += information_lines(report)
    kept = {n: v["kept_from"] for n, v in report["variants"].items() if v.get("kept_from")}
    if kept:
        L += ["", "Kept from an earlier run (their changes are against that run's current model): "
                  + ", ".join(f"{LABEL[n]} ({built})" for n, built in kept.items()) + "."]
    L += ["", f"Game lines: {report['lines_with_game_line']:.0%} of the scored lines ({report['game_line_coverage']:.0%} of all test "
              "player-games) have a DraftKings spread and total taken before tip; the rest are left empty. \"If minutes were known\" uses the minutes actually played and cannot be had before a game.", ""]
    return "\n".join(L)


def earlier_report(test_from: str, names: list[str]) -> dict:
    """What the stored report holds that this run does not refit, when the test period is the same."""
    try:
        stored = json.loads(REPORT.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if stored.get("test_from") != test_from:
        return {}
    kept = {n: {"kept_from": stored.get("built_at"), **v} for n, v in stored.get("variants", {}).items() if n not in names and n in VARIANTS}
    return dict(variants=kept) | {k: stored[k] for k in ("lines_with_game_line", *STUDIES) if k in stored}


def run(test_from: str, names: list[str], stats=LINE_STATS, minutes_only: bool = False, log=print) -> dict:
    from nba_props import games
    with psycopg2.connect(PG_DSN) as conn:
        frame, lines, lineups = model.load_games(conn), model.load_lines(conn), model.final_lineups(conn)
        game_lines = games.load_lines(conn)
        with_stints = minutes_only or any(VARIANTS[n].get("stints") for n in names)
        stints = stint_info.load(conn) if with_stints and stint_info.stored(conn) else None
    base = model.features(frame, lineups)
    base_cols = model.feature_columns(base)
    lined = add_game_line(base, game_lines)
    minutes: dict[tuple, pd.Series] = {}
    report = dict(built_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), test_from=test_from, variants={},
                  game_line_coverage=float(lined[eligible(lined) & (lined.game_date >= pd.Timestamp(test_from))].mkt_total.notna().mean()))
    report |= earlier_report(test_from, names)
    informed = stinted = None
    if minutes_only or any(VARIANTS[n].get("info") for n in names):
        log("  minutes information")
        informed = minutes_info.add(base, frame, lineups)
        full = None
        if "minutes_info" in names:
            log("  minutes model (with the minutes information)")
            full = minutes[(False, True)] = minutes_walk_forward(informed, base_cols + minutes_info.FEATURES)
        report["minutes_information"] = study = minutes_information(informed, base_cols, test_from, full=full, log=log)
        atomic_json(REPORT.with_suffix(".json"), clean(report))
        if stints is not None and len(stints):
            log("  rotation stints")
            stinted = stint_info.add(informed, frame, lineups, stints)
            info_cols = base_cols + minutes_info.FEATURES
            full = None
            if "stint_info" in names:
                log("  minutes model (with the minutes information and the stints)")
                full = minutes[(False, True, True)] = minutes_walk_forward(stinted, info_cols + stint_info.FEATURES)
            test = stinted[eligible(stinted) & (stinted.game_date >= pd.Timestamp(test_from))]
            games_stored = stints.game_id.nunique()
            used = stint_info.verified(*stint_info.summaries(stints), frame)[0].game_id.nunique()
            report["stint_information"] = minutes_information(stinted, info_cols, test_from, full=full, log=log, sets=STINT_SETS,
                                                              first=minutes.get((False, True))) | dict(
                games=int(games_stored), games_dropped=int(games_stored - used), coverage=float(test.comp_min_ewm3.notna().mean()))
            atomic_json(REPORT.with_suffix(".json"), clean(report))
    scored: dict[str, pd.DataFrame] = {}
    for name in names:
        spec = VARIANTS[name]
        f, cols = (lined, base_cols + LINE_FEATURES) if spec["line"] else (base, list(base_cols))
        if spec.get("info"):
            f, cols = informed, base_cols + minutes_info.FEATURES
        if spec.get("stints"):
            if stinted is None:
                log(f"  {name}: no stints stored (python -m nba_props.stints); left out")
                continue
            f, cols = stinted, cols + stint_info.FEATURES
        entry: dict = {}
        if spec["minutes"] == "model":
            key = (spec["line"], bool(spec.get("info"))) + ((True,) if spec.get("stints") else ())
            if key not in minutes:
                log(f"  minutes model ({'with' if spec['line'] else 'without'} the game line)")
                minutes[key] = minutes_walk_forward(f, cols)
            f, cols = add_minutes(f, minutes[key]), cols + MINUTES_FEATURES
            entry["minutes"] = minutes_accuracy(f, test_from)
        elif spec["minutes"] == "actual":
            f, cols = add_minutes(f, f.minutes.where(eligible(f))), cols + MINUTES_FEATURES
        log(f"  {name}: {len(cols)} features")
        preds = walk_forward(f, cols, test_from, stats=stats, log=log)
        entry["all_games_mae"] = {s: float((preds[f"pred_{s}"] - preds[s]).abs().mean()) for s in stats}
        rows = model.market_rows(preds, lines)
        rows_honest = model.market_rows(honest(preds), lines)
        entry["training_fit_spread"], entry["honest_spread"] = summary(model.evaluate(rows)), summary(model.evaluate(rows_honest))
        scored[name] = rows_honest
        if name == names[0]:
            keyed = preds.assign(name_key=preds.player_name.map(model.match_name)).merge(
                lined[["game_id", "player_id", "mkt_total"]], on=["game_id", "player_id"], how="left")
            on_lines = rows[["game_date", "name_key"]].drop_duplicates().merge(keyed[["game_date", "name_key", "mkt_total"]], on=["game_date", "name_key"])
            report["lines_with_game_line"] = float(on_lines.mkt_total.notna().mean())
        if "baseline" in scored and name != "baseline":
            entry["against_baseline"] = paired(scored["baseline"], rows_honest)
        if name == "stint_info" and "minutes_info" in scored:
            entry["against_minutes_info"] = paired(scored["minutes_info"], rows_honest)
        report["variants"][name] = entry
        atomic_json(REPORT.with_suffix(".json"), clean(report))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-from", default="2023-11-01")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--threes", action="store_true", help="also fit three-pointers (no DraftKings line history; a third more time)")
    ap.add_argument("--minutes-only", action="store_true", help="score the minutes forecast with the new inputs; refit no statistic model")
    args = ap.parse_args()
    names = [] if args.minutes_only else [n for n in args.variants.split(",") if n in VARIANTS]
    report = run(args.test_from, names, stats=STATS if args.threes else LINE_STATS, minutes_only=args.minutes_only,
                 log=lambda text: print(text, flush=True))
    text = markdown(report)
    REPORT.with_suffix(".md").write_bytes(text.encode())
    print(text)


if __name__ == "__main__":
    main()
