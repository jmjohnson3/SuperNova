"""When a sportsbook and Pinnacle disagree on a player prop, who is right? Research only; stored data, no credits.

The price-gap scanners score every decision at Pinnacle's last no-vig price, which assumes Pinnacle is the
truth. This checks that assumption against results, on every stored two-sided prop price (not only the ones a
scanner flagged): NBA from bets.sharp_multi_quotes, NFL from bets.nfl_sharp_quote_observations, both with
Pinnacle at the identical line in the same look.

    weight    the result regressed on the book's own no-vig price plus w x (Pinnacle's price - the book's), in
              log-odds: w = 1 means Pinnacle is right where they differ, w = 0 means the book is
    gaps      the better side at the book valued at Pinnacle's price, by size of the gap: what was claimed
              against what the results returned
    timing    the same for +3% gaps by minutes before the start

One row per game, player, statistic and book: the first look 15 to 45 minutes before the start (the scanners'
window), except in the timing table. Intervals resample whole games.

    python -m sharp_pipeline.gap_validity          reports/price_gap_validity_latest.md and .json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import minimize_scalar

from nba_props import news_lag
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "price_gap_validity_latest.md"
REFERENCE = "pinnacle"
BOOKS = ("fanduel", "draftkings", "espnbet", "betmgm")
WINDOW = (15.0, 45.0)                    # minutes before the start
SAME_LOOK = pd.Timedelta(minutes=4)      # a book's price and Pinnacle's are one look if this close in time
GAPS = (("Under 0% (no side is good value)", -np.inf, 0.0), ("0% to 1.5%", 0.0, 0.015), ("1.5% to 3%", 0.015, 0.03),
        ("3% to 5%", 0.03, 0.05), ("5% or more", 0.05, np.inf))
TIMING = (("45 to 35 minutes", 35.0, 45.0), ("35 to 25", 25.0, 35.0), ("25 to 15", 15.0, 25.0), ("15 to 5", 5.0, 15.0))
FLAG = 0.03
NEEDED = 1500                            # bets on 3%+ gaps to tell the claimed value from zero
HISTORY = ROOT / "reports" / "price_gap_validity_history.json"
NFL_STATS = ("receptions", "receiving_yards", "rushing_yards")
KEYS = ["sport", "game", "player", "stat"]
DRAWS = 1000


# --- data ----------------------------------------------------------------------------------------------------------
def nba_quotes(conn) -> pd.DataFrame:
    proj = news_lag.load_projections(conn)
    q = news_lag.quote_table(news_lag.parse_quotes(news_lag.load_quote_rows(conn)), proj)
    actual = news_lag.load_actuals(conn, sorted(set(q.game_id)))
    q = q.merge(actual, on=["game_id", "player_id", "stat"], how="left")
    return q.assign(sport="nba", game=q.game_id, player=q.player_id.astype(str))[
        ["sport", "game", "player", "stat", "book", "line", "over", "under", "observed_at", "commence", "actual"]]


def nfl_quotes(conn) -> pd.DataFrame:
    q = pd.read_sql("""SELECT game_id AS game, payload->>'player_name_norm' AS name, payload->>'stat' AS stat, payload->>'book' AS book,
                              (payload->>'line')::float AS line, (payload->'quote'->>'over')::float AS over,
                              (payload->'quote'->>'under')::float AS under, observed_at_utc AS observed_at,
                              (payload->>'commence_time_utc')::timestamptz AS commence
                       FROM bets.nfl_sharp_quote_observations
                       WHERE payload->>'stat' = ANY(%(stats)s) AND (payload->'quote'->>'ladder') IS DISTINCT FROM 'true'
                         AND payload->'quote'->>'over' IS NOT NULL AND payload->'quote'->>'under' IS NOT NULL""", conn, params={"stats": list(NFL_STATS)})
    for c in ("observed_at", "commence"):
        q[c] = pd.to_datetime(q[c], utc=True)
    names = pd.read_sql("SELECT DISTINCT player_name_norm AS name, player_id FROM raw.nfl_rosters WHERE season >= 2025 AND player_id IS NOT NULL", conn)
    names = names.drop_duplicates("name", keep=False)                                    # a shared name is left unmatched
    logs = pd.read_sql("SELECT game_id AS game, player_id, receptions::float AS receptions, receiving_yards::float AS receiving_yards, "
                       "rushing_yards::float AS rushing_yards FROM raw.nfl_player_gamelogs WHERE game_id = ANY(%(g)s)", conn,
                       params={"g": sorted(set(q.game))})
    actual = logs.melt(["game", "player_id"], list(NFL_STATS), "stat", "actual")
    q = q.merge(names, on="name", how="left").merge(actual, on=["game", "player_id", "stat"], how="left")
    return q.assign(sport="nfl", player=q.name)[["sport", "game", "player", "stat", "book", "line", "over", "under", "observed_at", "commence", "actual"]]


# --- one row per player, statistic and book ------------------------------------------------------------------------
def no_vig(over, under) -> np.ndarray:
    a, b = news_lag.american_probability(over), news_lag.american_probability(under)
    return a / (a + b)


def paired(quotes: pd.DataFrame) -> pd.DataFrame:
    """Every book price with Pinnacle's price at the identical line in the same look."""
    q = quotes[quotes.observed_at < quotes.commence].copy()
    q["minutes"] = (q.commence - q.observed_at).dt.total_seconds() / 60.0
    q["p_book"] = no_vig(q.over, q.under)
    ref = q[q.book == REFERENCE][KEYS + ["line", "observed_at", "p_book"]].rename(columns={"p_book": "p_ref", "observed_at": "ref_at"})
    books = q[q.book.isin(BOOKS)].sort_values("observed_at")
    out = pd.merge_asof(books, ref.sort_values("ref_at"), left_on="observed_at", right_on="ref_at", by=KEYS + ["line"],
                        direction="nearest", tolerance=SAME_LOOK)
    out = out.dropna(subset=["p_ref"])
    over_ev, under_ev = out.p_ref * news_lag.decimal(out.over) - 1.0, (1.0 - out.p_ref) * news_lag.decimal(out.under) - 1.0
    out["side"] = np.where(over_ev >= under_ev, 1.0, -1.0)
    out["ev"] = np.maximum(over_ev, under_ev)
    price = np.where(out.side > 0, out.over, out.under)
    margin = (out.actual - out.line) * out.side
    out["profit"] = np.select([margin > 0, margin < 0, margin == 0], [news_lag.decimal(price) - 1.0, -1.0, 0.0], np.nan)
    out["over_hit"] = np.where(out.actual.isna() | (out.actual == out.line), np.nan, (out.actual > out.line).astype(float))
    return out


def first_in(rows: pd.DataFrame, low: float, high: float) -> pd.DataFrame:
    """The first look in a window of minutes before the start, one per game, player, statistic and book."""
    inside = rows[(rows.minutes > low) & (rows.minutes <= high)].sort_values("observed_at")
    return inside.drop_duplicates(KEYS + ["book"])


# --- measures ------------------------------------------------------------------------------------------------------
def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0.02, 0.98)
    return np.log(p / (1 - p))


def fit_weight(book: np.ndarray, ref: np.ndarray, hit: np.ndarray) -> float:
    """w in: log-odds of the over = the book's log-odds + w x (Pinnacle's - the book's)."""
    base, gap = logit(book), logit(ref) - logit(book)

    def loss(w):
        z = base + w * gap
        return float(np.mean(np.log1p(np.exp(-np.where(hit > 0, z, -z)))))
    return float(minimize_scalar(loss, bounds=(-3.0, 5.0), method="bounded").x)


def weight(rows: pd.DataFrame, draws: int = DRAWS, seed: int = 7) -> dict | None:
    d = rows.dropna(subset=["over_hit"])
    if len(d) < 200:
        return None
    w = fit_weight(d.p_book.to_numpy(), d.p_ref.to_numpy(), d.over_hit.to_numpy())
    groups = [g[["p_book", "p_ref", "over_hit"]].to_numpy() for _, g in d.groupby(["sport", "game"])]
    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(draws if len(groups) >= 10 else 0):
        sample = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        fits.append(fit_weight(sample[:, 0], sample[:, 1], sample[:, 2]))
    low, high = (float(v) for v in np.percentile(fits, [2.5, 97.5])) if fits else (None, None)
    return dict(rows=int(len(d)), games=len(groups), weight=w, low=low, high=high,
                mean_gap=float(np.abs(d.p_ref - d.p_book).mean()))


def returns(rows: pd.DataFrame, draws: int = DRAWS, seed: int = 7) -> dict:
    """The better side at the book: the value Pinnacle's price gives it against what the results returned."""
    d = rows.dropna(subset=["profit"])
    out = dict(bets=int(len(d)), claimed=None, roi=None, low=None, high=None, hit=None)
    if d.empty:
        return out
    decided = d[d.profit != 0]
    out.update(claimed=float(d.ev.mean()), roi=float(d.profit.mean()), hit=float((decided.profit > 0).mean()) if len(decided) else None)
    per = d.groupby(["sport", "game"]).profit.agg(["sum", "count"]).to_numpy(float)
    if len(per) >= 10:
        pick = np.random.default_rng(seed).integers(0, len(per), size=(draws, len(per)))
        out["low"], out["high"] = (float(v) for v in np.percentile(per[pick, 0].sum(axis=1) / per[pick, 1].sum(axis=1), [2.5, 97.5]))
    return out


def summarise(rows: pd.DataFrame) -> dict:
    main = first_in(rows, *WINDOW)
    flagged = main[main.ev >= FLAG]
    cut = lambda frame, col: {str(k): g for k, g in frame.groupby(col)}
    return dict(
        built=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        sample={s: dict(games=int(g.game.nunique()), prices=int(len(g)), settled=int(g.over_hit.notna().sum()),
                        first=str(g.observed_at.min().date()), last=str(g.observed_at.max().date())) for s, g in main.groupby("sport")},
        weight={"All": weight(main), **{f"{s.upper()}": weight(g) for s, g in cut(main, "sport").items()},
                **{f"{b}": weight(g) for b, g in cut(main, "book").items()},
                **{f"{s.upper()} {m}": weight(g) for (s, m), g in main.groupby(["sport", "stat"])},
                "Gap of 3%+ only": weight(flagged)},
        gaps={sport: {label: returns(g[(g.ev >= lo) & (g.ev < hi)]) for label, lo, hi in GAPS}
              for sport, g in {"All": main, **{s.upper(): g for s, g in cut(main, "sport").items()}}.items()},
        flagged={"All": returns(flagged), **{b: returns(g) for b, g in cut(flagged, "book").items()},
                 **{f"{s.upper()} {m}": returns(g) for (s, m), g in flagged.groupby(["sport", "stat"])},
                 "Over side": returns(flagged[flagged.side > 0]), "Under side": returns(flagged[flagged.side < 0])},
        timing={label: returns(first_in(rows, lo, hi).query("ev >= @FLAG")) for label, lo, hi in TIMING})


# --- report --------------------------------------------------------------------------------------------------------
def markdown(report: dict) -> str:
    pct = lambda v, d=1: "-" if v is None else f"{v:+.{d}%}"
    L = ["# Price gaps: is Pinnacle right when a book disagrees?", "",
         f"Built {report['built']}. Research only; stored prices, no credits. Player props with Pinnacle at the identical line in the same look; "
         "one row per game, player, statistic and book, the first look 15 to 45 minutes before the start.", ""]
    for sport, s in report["sample"].items():
        L.append(f"- {sport.upper()}: {s['games']} games, {s['prices']} prices, {s['settled']} settled, {s['first']} to {s['last']}.")
    L += ["", "## Whose price is right", "",
          "Weight 1 means that where the two differ Pinnacle's price is the right one; 0 means the book's own price is. Intervals resample games.", "",
          "| Prices | Settled | Games | Average difference | Weight on Pinnacle (95%) |", "|---|---:|---:|---:|---:|"]
    for label, w in report["weight"].items():
        if w:
            interval = f"{w['weight']:+.2f}" if w["low"] is None else f"{w['weight']:+.2f} ({w['low']:+.2f} to {w['high']:+.2f})"
            L.append(f"| {label} | {w['rows']} | {w['games']} | {w['mean_gap']:.1%} | {interval} |")

    def table(title: str, first: str, rows: dict) -> list[str]:
        out = ["", f"## {title}", "", f"| {first} | Bets | Value claimed | Hit rate | Return on results (95%) |", "|---|---:|---:|---:|---:|"]
        for label, r in rows.items():
            if r["bets"]:
                roi = pct(r["roi"]) if r["low"] is None else f"{pct(r['roi'])} ({pct(r['low'])} to {pct(r['high'])})"
                out.append(f"| {label} | {r['bets']} | {pct(r['claimed'])} | {'-' if r['hit'] is None else format(r['hit'], '.1%')} | {roi} |")
        return out
    for sport, rows in report["gaps"].items():
        L += table(f"The better side at the book, by size of the gap: {sport}", "Gap against Pinnacle", rows)
    L += table("Gaps of 3% or more", "Group", report["flagged"])
    L += table("Gaps of 3% or more by minutes before the start (each window's first look)", "Window", report["timing"])
    if report.get("history"):
        L += ["", f"## Progress, run by run (about {NEEDED:,} bets on gaps of 3%+ are needed)", "",
              "| Run | Bets on 3%+ gaps | Value claimed | Return on results (95%) | Weight on Pinnacle (95%) |", "|---|---:|---:|---:|---:|"]
        span = lambda fmt, v, lo, hi: "-" if v is None else format(v, fmt) if lo is None else f"{format(v, fmt)} ({format(lo, fmt)} to {format(hi, fmt)})"
        for h in report["history"]:
            L.append(f"| {h['date']} | {h['bets']} | {pct(h['claimed'])} | {span('+.1%', h['roi'], h['roi_low'], h['roi_high'])} | "
                     f"{span('+.2f', h['weight'], h['weight_low'], h['weight_high'])} |")
    L += ["", "Value claimed is the bet's expected return if Pinnacle's no-vig price is the true chance. Every price is counted, whether or not a "
              "scanner flagged it or the price could really be bet. NCAAF and game lines are not included (too few stored)."]
    return "\n".join(L) + "\n"


def headline(report: dict) -> dict:
    """One line per run for the progress table: the 3%+ gaps."""
    r, w = report["flagged"]["All"], report["weight"].get("Gap of 3%+ only") or {}
    return dict(date=report["built"][:10], bets=r["bets"], claimed=r["claimed"], roi=r["roi"], roi_low=r["low"], roi_high=r["high"],
                weight=w.get("weight"), weight_low=w.get("low"), weight_high=w.get("high"))


def with_history(report: dict, earlier: list[dict]) -> list[dict]:
    """Earlier runs plus this one; a second run on the same day replaces that day's line."""
    line = news_lag.clean(headline(report))
    return [h for h in earlier if h["date"] != line["date"]] + [line]


def run(log=print) -> dict:
    with psycopg2.connect(PG_DSN) as conn:
        quotes = pd.concat([nba_quotes(conn), nfl_quotes(conn)], ignore_index=True)
    log(f"  {len(quotes)} stored prices")
    report = summarise(paired(quotes))
    earlier = json.loads(HISTORY.read_text(encoding="utf-8")) if HISTORY.exists() else []
    report["history"] = with_history(report, earlier)
    atomic_json(HISTORY, report["history"])
    atomic_json(REPORT.with_suffix(".json"), news_lag.clean(report))
    REPORT.write_text(markdown(report), encoding="utf-8")
    log(f"  wrote {REPORT}")
    return report


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    run()


if __name__ == "__main__":
    main()
