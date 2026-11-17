"""A fitted fair price from every book's prop price, tested walk-forward. Research only; stored data, no credits.

The stat-forecast models lose to the books' lines, and the price-gap scanners take Pinnacle's price as the truth.
This asks a third question: given what every book is showing for a prop at one moment, which mix of those prices
predicts the result best, and does betting where a book disagrees with that mix beat betting where it disagrees
with Pinnacle alone?

    row       one per game, player and statistic: the first look 15 to 45 minutes before the start, the line most
              books share, and each book's no-vig price for the over at that line
    model     log-odds of the over = a x (average of the books) + sum of w_book x (that book - the average)
              + a term per statistic (over or under bias), shrunk toward the plain average
    test      refitted before each game day on earlier days only; scored by log loss against the plain average,
              Pinnacle and each book, then by the return of bets a book offers at 3%+ above the model's price

Every price is assumed bettable at the price stored (the user's instruction, 2026-11-16); nothing here checks that.

    python -m sharp_pipeline.fair_price_model          reports/fair_price_model_latest.md and .json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import minimize

from nba_props import news_lag
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from sharp_pipeline import gap_validity as gv

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "fair_price_model_latest.md"
KEYS = gv.KEYS
MINE = ("fanduel", "draftkings")                 # the books the user bets at
MIN_BOOKS = 3                                    # books sharing the line for a row to count
MIN_DAYS = 7                                     # game days of history before the first forecast
SHRINK = 5.0                                     # penalty pulling the fit toward the plain average
SHRINKS = (1.0, 5.0, 25.0)
FLAG = gv.FLAG
EASTERN = "America/New_York"


# --- rows ----------------------------------------------------------------------------------------------------------
def looks(quotes: pd.DataFrame, window: tuple[float, float] = gv.WINDOW) -> pd.DataFrame:
    """One row per book at the first look in the window, on the line most books share (Pinnacle's on a tie)."""
    q = quotes[quotes.observed_at < quotes.commence].copy()
    q["minutes"] = (q.commence - q.observed_at).dt.total_seconds() / 60.0
    q = q[(q.minutes > window[0]) & (q.minutes <= window[1])]
    first = q.groupby(KEYS).observed_at.transform("min")
    q = q[q.observed_at <= first + gv.SAME_LOOK].sort_values("observed_at").drop_duplicates(KEYS + ["book"])
    q["p"] = gv.no_vig(q.over, q.under)
    q = q[np.isfinite(q.p) & (q.p > 0) & (q.p < 1)]
    count = q.groupby(KEYS + ["line"]).book.transform("size") + 0.5 * q.groupby(KEYS + ["line"]).book.transform(lambda b: (b == gv.REFERENCE).any())
    q = q[count == count.groupby([q[k] for k in KEYS]).transform("max")]
    q = q[q.line == q.groupby(KEYS).line.transform("min")]                              # still tied: the lower line
    q = q[q.groupby(KEYS).book.transform("size") >= MIN_BOOKS]
    q["day"] = q.commence.dt.tz_convert(EASTERN).dt.date
    q["over_hit"] = np.where(q.actual.isna() | (q.actual == q.line), np.nan, (q.actual > q.line).astype(float))
    return q.reset_index(drop=True)


def table(long: pd.DataFrame) -> pd.DataFrame:
    """One row per game, player and statistic: the average log-odds, each book's distance from it (0 when absent)."""
    long = long.assign(z=gv.logit(long.p))
    wide = long.pivot_table(index=KEYS, columns="book", values="z")
    base = long.drop_duplicates(KEYS).set_index(KEYS)[["line", "day", "commence", "actual", "over_hit"]]
    out = base.join(wide.mean(axis=1).rename("consensus")).join(wide.sub(wide.mean(axis=1), axis=0).fillna(0.0).add_prefix("dev_"))
    out["books"] = wide.notna().sum(axis=1)
    return out.reset_index().assign(group=lambda f: f.sport + " " + f.stat)


# --- model ---------------------------------------------------------------------------------------------------------
class Fit:
    """Logistic fit shrunk toward the plain average: slope 1 on the average, 0 on every book and every statistic."""

    def __init__(self, rows: pd.DataFrame, books: list[str], shrink: float = SHRINK):
        self.books, self.groups = list(books), sorted(rows.group.unique())
        d = rows.dropna(subset=["over_hit"])
        X, y = self.matrix(d), d.over_hit.to_numpy()
        sign = np.where(y > 0, 1.0, -1.0)
        start = np.zeros(X.shape[1])
        start[0] = 1.0

        def loss(b):
            z = X @ b
            e = np.exp(-np.abs(z))
            value = np.sum(np.maximum(-sign * z, 0.0) + np.log1p(e)) + shrink * np.sum((b - start) ** 2)
            p = 1.0 / (1.0 + np.exp(-z))
            return value, X.T @ (p - (y > 0)) + 2.0 * shrink * (b - start)
        self.coef = minimize(loss, start, jac=True, method="L-BFGS-B").x if len(d) else start

    def matrix(self, rows: pd.DataFrame) -> np.ndarray:
        devs = [rows[f"dev_{b}"].to_numpy() if f"dev_{b}" in rows else np.zeros(len(rows)) for b in self.books]
        dummies = [(rows.group == g).to_numpy(float) for g in self.groups]
        return np.column_stack([rows.consensus.to_numpy(), *devs, *dummies]) if len(rows) else np.zeros((0, 1 + len(devs) + len(dummies)))

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.matrix(rows) @ self.coef)))

    def weights(self) -> dict:
        names = ["average of the books", *self.books, *[f"{g} (toward the over)" for g in self.groups]]
        return dict(zip(names, (float(v) for v in self.coef)))


def walk_forward(rows: pd.DataFrame, books: list[str], shrink: float = SHRINK, min_days: int = MIN_DAYS) -> pd.DataFrame:
    """Each game day forecast from a fit on earlier days only."""
    days = sorted(rows.day.unique())
    out = []
    for day in days[min_days:]:
        today = rows[rows.day == day]
        out.append(today.assign(p_model=Fit(rows[rows.day < day], books, shrink).predict(today)))
    return pd.concat(out, ignore_index=True) if out else rows.iloc[0:0].assign(p_model=np.nan)


# --- scoring -------------------------------------------------------------------------------------------------------
def log_loss(p, hit) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -np.where(np.asarray(hit) > 0, np.log(p), np.log(1 - p))


def against(oof: pd.DataFrame, other: pd.Series, draws: int = gv.DRAWS, seed: int = 7) -> dict | None:
    """Model log loss minus another price's on the rows where both exist; below zero means the model is better."""
    d = oof.assign(other=other)[lambda f: f.over_hit.notna() & f.other.notna()]
    if len(d) < 100:
        return None
    d = d.assign(model_loss=log_loss(d.p_model, d.over_hit), other_loss=log_loss(d.other, d.over_hit))
    per = d.assign(diff=d.model_loss - d.other_loss).groupby(["sport", "game"])["diff"].agg(["sum", "count"]).to_numpy(float)
    pick = np.random.default_rng(seed).integers(0, len(per), size=(draws, len(per)))
    low, high = (float(v) for v in np.percentile(per[pick, 0].sum(axis=1) / per[pick, 1].sum(axis=1), [2.5, 97.5]))
    return dict(rows=int(len(d)), model=float(d.model_loss.mean()), other=float(d.other_loss.mean()),
                diff=float(d.model_loss.mean() - d.other_loss.mean()), low=low, high=high)


def offers(long: pd.DataFrame, fair: pd.DataFrame, column: str) -> pd.DataFrame:
    """Every book's price valued at a fair chance of the over: the better side, its value and what it returned."""
    d = long.merge(fair[KEYS + [column]].dropna(), on=KEYS)
    over_ev, under_ev = d[column] * news_lag.decimal(d.over) - 1.0, (1.0 - d[column]) * news_lag.decimal(d.under) - 1.0
    d["side"] = np.where(over_ev >= under_ev, 1.0, -1.0)
    d["ev"] = np.maximum(over_ev, under_ev)
    margin = (d.actual - d.line) * d.side
    d["profit"] = np.select([margin > 0, margin < 0, margin == 0], [news_lag.decimal(np.where(d.side > 0, d.over, d.under)) - 1.0, -1.0, 0.0], np.nan)
    return d


def leave_one_out(long: pd.DataFrame) -> pd.DataFrame:
    """For each book's price, the average of the other books at the same line (the Sniper's idea, unweighted)."""
    z = pd.Series(gv.logit(long.p), index=long.index)
    total, n = z.groupby([long[k] for k in KEYS]).transform("sum"), z.groupby([long[k] for k in KEYS]).transform("size")
    return long.assign(p_others=1.0 / (1.0 + np.exp(-(total - z) / (n - 1))))


def bet_table(long: pd.DataFrame, oof: pd.DataFrame) -> dict:
    long = long.merge(oof[KEYS], on=KEYS)                                               # the forecast days only
    pin = long[long.book == gv.REFERENCE][KEYS + ["p"]].rename(columns={"p": "p_pinnacle"})
    others = leave_one_out(long)
    rules = {"Fitted model": offers(long, oof, "p_model"), "Pinnacle's price": offers(long[long.book != gv.REFERENCE], pin, "p_pinnacle"),
             "Average of the books": offers(long, oof.assign(p_avg=1.0 / (1.0 + np.exp(-oof.consensus))), "p_avg")}
    loo = others.assign(side=0.0)
    over_ev, under_ev = loo.p_others * news_lag.decimal(loo.over) - 1.0, (1.0 - loo.p_others) * news_lag.decimal(loo.under) - 1.0
    loo["side"], loo["ev"] = np.where(over_ev >= under_ev, 1.0, -1.0), np.maximum(over_ev, under_ev)
    margin = (loo.actual - loo.line) * loo.side
    loo["profit"] = np.select([margin > 0, margin < 0, margin == 0], [news_lag.decimal(np.where(loo.side > 0, loo.over, loo.under)) - 1.0, -1.0, 0.0], np.nan)
    rules["Average of the other books"] = loo
    out = {}
    for name, d in rules.items():
        flagged = d[(d.ev >= FLAG) & (d.book != gv.REFERENCE)]
        out[name] = {"FanDuel and DraftKings": gv.returns(flagged[flagged.book.isin(MINE)]), "All four books": gv.returns(flagged[flagged.book.isin(gv.BOOKS)]),
                     **{f"{s.upper()} {m}": gv.returns(g) for (s, m), g in flagged[flagged.book.isin(gv.BOOKS)].groupby(["sport", "stat"])},
                     "5%+ at all four": gv.returns(flagged[flagged.book.isin(gv.BOOKS) & (flagged.ev >= 0.05)])}
    both = rules["Fitted model"].merge(rules["Pinnacle's price"][KEYS + ["book", "ev", "side"]], on=KEYS + ["book"], suffixes=("", "_pin"))
    both = both[both.book.isin(gv.BOOKS)]
    agree = (both.ev >= FLAG) & (both.ev_pin >= FLAG) & (both.side == both.side_pin)
    out["Where the two rules meet"] = {"Both flag the same side": gv.returns(both[agree]),
                                       "Model only": gv.returns(both[(both.ev >= FLAG) & ~agree]),
                                       "Pinnacle only": gv.returns(both[(both.ev_pin >= FLAG) & ~agree].assign(
                                           profit=lambda f: np.where(f.side == f.side_pin, f.profit, np.nan), ev=lambda f: f.ev_pin).dropna(subset=["profit"]))}
    return out


def summarise(long: pd.DataFrame) -> dict:
    rows = table(long)
    books = sorted(b for b in long.book.unique())
    oof = walk_forward(rows, books)
    prices = long.pivot_table(index=KEYS, columns="book", values="p").reindex(pd.MultiIndex.from_frame(oof[KEYS])).reset_index(drop=True)
    scores = {"Average of the books": against(oof, 1.0 / (1.0 + np.exp(-oof.consensus))),
              **{b: against(oof, prices[b]) for b in prices.columns}}
    return dict(
        built=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        sample=dict(rows=int(len(rows)), settled=int(rows.over_hit.notna().sum()), days=int(rows.day.nunique()), forecast_rows=int(len(oof)),
                    forecast_days=int(oof.day.nunique()), first=str(rows.day.min()), last=str(rows.day.max()), books=books),
        scores={k: v for k, v in scores.items() if v},
        by_group={g: against(d, 1.0 / (1.0 + np.exp(-d.consensus))) for g, d in oof.groupby("group")},
        shrink={str(s): against(o, 1.0 / (1.0 + np.exp(-o.consensus))) for s in SHRINKS for o in [walk_forward(rows, books, s)]},
        weights=Fit(rows, books).weights(),
        bets=bet_table(long, oof))


# --- report --------------------------------------------------------------------------------------------------------
def markdown(report: dict) -> str:
    pct = lambda v, d=1: "-" if v is None else f"{v:+.{d}%}"
    s = report["sample"]
    L = ["# A fitted fair price from every book's prop price", "",
         f"Built {report['built']}. Research only; stored prices, no credits. Every price is assumed bettable as stored.", "",
         f"- {s['rows']} props ({s['settled']} settled) over {s['days']} game days, {s['first']} to {s['last']}; books: {', '.join(s['books'])}.",
         f"- Forecast walk-forward: {s['forecast_rows']} props on {s['forecast_days']} days, each from a fit on earlier days only.", "",
         "## Is the fitted price more accurate than the prices it is built from?", "",
         "Log loss on results, on the props where both prices exist; a difference below zero means the fitted price is better. Intervals resample games.", "",
         "| Compared with | Props | Fitted | Other | Difference (95%) |", "|---|---:|---:|---:|---:|"]

    def score_row(label, r):
        return f"| {label} | {r['rows']} | {r['model']:.4f} | {r['other']:.4f} | {r['diff']:+.4f} ({r['low']:+.4f} to {r['high']:+.4f}) |"
    L += [score_row(k, r) for k, r in report["scores"].items()]
    L += ["", "Against the plain average, by statistic:", "", "| Statistic | Props | Fitted | Average | Difference (95%) |", "|---|---:|---:|---:|---:|"]
    L += [score_row(k, r) for k, r in report["by_group"].items() if r]
    L += ["", "Against the plain average, by how hard the fit is pulled toward it (higher = closer to the average):", "",
          "| Pull | Props | Fitted | Average | Difference (95%) |", "|---|---:|---:|---:|---:|"]
    L += [score_row(k, r) for k, r in report["shrink"].items() if r]
    L += ["", "## What the fit learned (all days)", "", "1 on the average and 0 elsewhere is the plain average. A positive number on a book means its price "
          "deserves more than an equal share.", "", "| Term | Value |", "|---|---:|"]
    L += [f"| {k} | {v:+.2f} |" for k, v in report["weights"].items()]
    for rule, groups in report["bets"].items():
        L += ["", f"## Bets at 3%+ above: {rule}" if rule != "Where the two rules meet" else f"## {rule} (all four books)", "",
              "| Group | Bets | Value claimed | Hit rate | Return on results (95%) |", "|---|---:|---:|---:|---:|"]
        for label, r in groups.items():
            if r["bets"]:
                roi = pct(r["roi"]) if r["low"] is None else f"{pct(r['roi'])} ({pct(r['low'])} to {pct(r['high'])})"
                L.append(f"| {label} | {r['bets']} | {pct(r['claimed'])} | {'-' if r['hit'] is None else format(r['hit'], '.1%')} | {roi} |")
    L += ["", "Each bet is the better side of one book's price, valued at the rule's fair chance, on the forecast days only. In the last table "
              "the Pinnacle-only bets are graded on Pinnacle's side."]
    return "\n".join(L) + "\n"


def run(log=print) -> dict:
    with psycopg2.connect(PG_DSN) as conn:
        quotes = pd.concat([gv.nba_quotes(conn), gv.nfl_quotes(conn)], ignore_index=True)
    log(f"  {len(quotes)} stored prices")
    report = summarise(looks(quotes))
    atomic_json(REPORT.with_suffix(".json"), news_lag.clean(report))
    REPORT.write_text(markdown(report), encoding="utf-8")
    log(f"  wrote {REPORT}")
    return report


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    run()


if __name__ == "__main__":
    main()
