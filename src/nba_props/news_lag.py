"""Do NBA prop prices lag lineup and injury news? Research only: stored data, no credits, nothing scheduled.

The news watcher (nba_props.live --news, every 10 minutes) re-projects every player with the day's saved fits
when a starter, inactive or injury status changes, so a projection that moves between two runs moved because of
the news alone. Each move of 3% or more is lined up against the prop prices the sharp scanner saved in the last
50 minutes before tip (bets.sharp_multi_quotes): the last price before we saw the news, the first price after,
and the last price before tip.

    lag       how far the market moved in the news direction before our first look and after it
    bets      the news side at FanDuel and DraftKings at the first price after the news, judged at Pinnacle's
              last no-vig price before tip and by the result
    control   players with no news that day, same books, first price to last

A price is turned into the projection it implies (normal approximation with the live model's spread), so a move
of the line and a move of the price are on one scale. Approximate; exact for the price when the line is unchanged.

    python -m nba_props.news_lag          reports/nba_news_lag_latest.md and .json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats as st

from nba_props import model
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "nba_news_lag_latest.md"
MOVE = 0.03                    # a projection that moves this much between two runs is a news move
BIG_MOVE = 0.10
FRESH = 15.0                   # minutes from our seeing the news to the first price after it
PRE_WINDOW = 30.0              # the price before the news must be this recent
SAME_GAME = pd.Timedelta(hours=3)
BOOKS = ("fanduel", "draftkings")
REFERENCE = "pinnacle"
MARKET_STAT = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast", "player_threes": "fg3m"}
KEYS = ["game_id", "player_id", "stat"]
KIND = {"role": "His own role changed", "teammates": "A teammate dropped out or came back", "other": "Other news"}
AGES = (("Under 15 minutes", 0, FRESH), ("15 to 60 minutes", FRESH, 60), ("1 to 4 hours", 60, 240), ("Over 4 hours", 240, np.inf))
QUOTE_COLUMNS = ["event_id", "observed_at", "commence", "book", "stat", "name_key", "line", "over", "under"]
DRAWS = 2000


# --- prices --------------------------------------------------------------------------------------------------------
def american_probability(price) -> np.ndarray:
    price = np.asarray(price, dtype=float)
    return np.where(price < 0, -price / (100.0 - price), 100.0 / (price + 100.0))


def decimal(price) -> np.ndarray:
    price = np.asarray(price, dtype=float)
    return np.where(price < 0, 1.0 + 100.0 / -price, 1.0 + price / 100.0)


def over_probability(mu, line, a, b) -> np.ndarray:
    """P(over the line) for a mean mu with variance a*mu + b*mu^2, normal approximation."""
    mu = np.clip(np.asarray(mu, dtype=float), 0.05, None)
    return st.norm.sf((np.asarray(line, dtype=float) - mu) / np.sqrt(a * mu + b * mu ** 2))


def implied_mean(line, p, a, b) -> np.ndarray:
    """The mean that gives probability p of going over the line: the inverse of over_probability."""
    line, a, b = (np.asarray(v, dtype=float) for v in (line, a, b))
    z = st.norm.ppf(np.clip(np.asarray(p, dtype=float), 0.02, 0.98))
    mu = np.maximum(line, 0.05)
    for _ in range(6):                                     # mu - line = z * sd(mu); sd depends on mu only through b
        spread = a + b * mu
        mu = ((z * np.sqrt(spread) + np.sqrt(z ** 2 * spread + 4 * line)) / 2) ** 2
    return mu


# --- data ----------------------------------------------------------------------------------------------------------
def load_projections(conn) -> pd.DataFrame:
    p = pd.read_sql("""SELECT run_at_utc AS run_at, game_date, game_id, tip_utc, player_id, name_key, team, stat, mu::float AS mu,
                              disp_a::float AS a, disp_b::float AS b, starter, trigger
                       FROM nbaprops.projections WHERE run_at_utc < tip_utc""", conn)
    for c in ("run_at", "tip_utc"):
        p[c] = pd.to_datetime(p[c], utc=True)
    p["game_date"] = pd.to_datetime(p.game_date)
    return p


def load_quote_rows(conn) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute("SELECT event_id, observed_at_utc, commence_time_utc, payload FROM bets.sharp_multi_quotes WHERE sport = 'nba'")
        return cur.fetchall()


def load_actuals(conn, games: list[str]) -> pd.DataFrame:
    df = pd.read_sql("SELECT game_id, player_id, minutes::float AS minutes, pts, reb, ast, fg3m FROM nbaprops.player_games "
                     "WHERE game_id = ANY(%(g)s)", conn, params={"g": list(games)})
    long = df.melt(["game_id", "player_id", "minutes"], ["pts", "reb", "ast", "fg3m"], "stat", "actual")
    return long[long.minutes > 0].drop(columns="minutes").astype({"actual": float})


def parse_quotes(rows: list[tuple]) -> pd.DataFrame:
    """One row per look, book, player and statistic with both sides priced; the line nearest even when several."""
    out = []
    for event_id, observed, commence, payload in rows:
        for book in (payload or {}).get("bookmakers") or []:
            for market in book.get("markets") or []:
                stat = MARKET_STAT.get(market.get("key"))
                if not stat:
                    continue
                sides: dict = {}
                for o in market.get("outcomes") or []:
                    if o.get("point") is None or o.get("price") is None or not o.get("description"):
                        continue
                    sides.setdefault((o["description"], float(o["point"])), {})[o.get("name")] = int(o["price"])
                for (name, line), s in sides.items():
                    if "Over" in s and "Under" in s:
                        out.append((event_id, observed, commence, book.get("key"), stat, model.match_name(name), line, s["Over"], s["Under"]))
    q = pd.DataFrame(out, columns=QUOTE_COLUMNS)
    for c in ("observed_at", "commence"):
        q[c] = pd.to_datetime(q[c], utc=True)
    over, under = american_probability(q.over), american_probability(q.under)
    q["p"] = over / (over + under)
    q = q.assign(skew=(q.p - 0.5).abs()).sort_values("skew").drop_duplicates(["event_id", "observed_at", "book", "stat", "name_key"])
    return q.drop(columns="skew").sort_values("observed_at").reset_index(drop=True)


def quote_table(quotes: pd.DataFrame, proj: pd.DataFrame) -> pd.DataFrame:
    """The looks taken before tip, tied to the player's game, with the projection each price implies."""
    info = proj.sort_values("run_at").drop_duplicates(KEYS, keep="last")[KEYS + ["name_key", "tip_utc", "game_date", "a", "b"]]
    q = quotes.merge(info, on=["name_key", "stat"])
    q = q[((q.commence - q.tip_utc).abs() <= SAME_GAME) & (q.observed_at < q.commence)].copy()
    q["imp"] = implied_mean(q.line, q.p, q.a, q.b)
    return q.sort_values("observed_at").reset_index(drop=True)


# --- news moves ----------------------------------------------------------------------------------------------------
def news_moves(proj: pd.DataFrame, threshold: float = MOVE) -> pd.DataFrame:
    """Projections that moved by the threshold or more in a news run, against the run before it."""
    p = proj.sort_values(KEYS + ["run_at"]).copy()
    g = p.groupby(KEYS, sort=False)
    p["prev_mu"], p["prev_starter"], p["prev_run"] = g.mu.shift(1), g.starter.shift(1), g.run_at.shift(1)
    rosters = p.groupby(["game_id", "team", "run_at"]).player_id.agg(frozenset).reset_index(name="roster").sort_values(["game_id", "team", "run_at"])
    earlier = rosters.groupby(["game_id", "team"]).roster.shift(1)
    rosters["roster_changed"] = [isinstance(e, frozenset) and e != r for r, e in zip(rosters.roster, earlier)]
    m = p[(p.trigger == "news") & p.prev_mu.notna()].merge(rosters.drop(columns="roster"), on=["game_id", "team", "run_at"], how="left")
    m["move"] = m.mu / m.prev_mu.clip(lower=0.05) - 1.0
    m = m[m.move.abs() >= threshold].copy()
    role = m.starter.astype("boolean").fillna(False) != m.prev_starter.astype("boolean").fillna(False)
    m["kind"] = np.select([role.to_numpy(bool), m.roster_changed.fillna(False).to_numpy(bool)], ["role", "teammates"], "other")
    m["direction"] = np.sign(m.move)
    return m[KEYS + ["game_date", "tip_utc", "team", "run_at", "prev_run", "mu", "prev_mu", "move", "direction", "kind", "a", "b"]].reset_index(drop=True)


def timeline(moves: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Each news move at each book: the last price before we saw it, the first after and the last before tip.
    A row needs a price after the news; when several moves share that first price, the latest move is kept."""
    fields = ["observed_at", "line", "over", "under", "p", "imp"]
    m = moves.sort_values("run_at").reset_index(drop=True)
    parts = []
    for book, qb in quotes.groupby("book"):
        qb = qb.sort_values("observed_at")
        named = lambda prefix: qb[KEYS + fields].rename(columns={c: f"{prefix}_{c}" for c in fields})
        row = pd.merge_asof(m, named("first"), left_on="run_at", right_on="first_observed_at", by=KEYS, direction="forward")
        pre = pd.merge_asof(m[KEYS + ["run_at"]], named("pre"), left_on="run_at", right_on="pre_observed_at", by=KEYS,
                            direction="backward", allow_exact_matches=False)
        row = pd.concat([row, pre.drop(columns=KEYS + ["run_at"])], axis=1)
        row = row.merge(named("close").drop_duplicates(KEYS, keep="last"), on=KEYS, how="left").assign(book=book)
        parts.append(row[row.first_observed_at.notna()])
    if not parts:
        return pd.DataFrame()
    t = pd.concat(parts, ignore_index=True).sort_values("run_at").drop_duplicates(KEYS + ["book", "first_observed_at"], keep="last")
    minutes = lambda a, b: (a - b).dt.total_seconds() / 60.0
    t["age"] = minutes(t.first_observed_at, t.run_at)
    has_pre = minutes(t.run_at, t.pre_observed_at) <= PRE_WINDOW
    later = t.close_observed_at > t.first_observed_at
    t["before_look"] = (t.direction * (t.first_imp / t.pre_imp - 1.0)).where(has_pre)
    t["after_look"] = (t.direction * (t.close_imp / t.first_imp - 1.0)).where(later)
    t["whole"] = (t.direction * (t.close_imp / t.pre_imp - 1.0)).where(has_pre & later)
    t["size"] = t.move.abs()
    return t.reset_index(drop=True)


# --- bets and the control ------------------------------------------------------------------------------------------
def reference_close(quotes: pd.DataFrame) -> pd.DataFrame:
    ref = quotes[quotes.book == REFERENCE].sort_values("observed_at").drop_duplicates(KEYS, keep="last")
    return ref[KEYS + ["observed_at", "imp"]].rename(columns={"observed_at": "ref_at", "imp": "ref_imp"})


def graded(b: pd.DataFrame, quotes: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Adds the expected value at the reference book's last price before tip and the result to bets that have
    side (+1 over, -1 under), line, over, under, at, a and b."""
    b = b.merge(reference_close(quotes), on=KEYS, how="left").merge(actuals, on=KEYS, how="left")
    over = b.side > 0
    price = np.where(over, b.over, b["under"])
    p_over = over_probability(b.ref_imp, b.line, b.a, b.b)
    b["ev"] = (np.where(over, p_over, 1.0 - p_over) * decimal(price) - 1.0)
    b["ev"] = b.ev.where(b.ref_at > b["at"])
    margin = (b.actual - b.line) * b.side
    b["profit"] = np.select([margin > 0, margin < 0, margin == 0], [decimal(price) - 1.0, -1.0, 0.0], np.nan)
    b["won"] = np.where(margin.isna() | (margin == 0), np.nan, (margin > 0).astype(float))
    return b


def news_bets(t: pd.DataFrame, quotes: pd.DataFrame, actuals: pd.DataFrame, at: str = "first") -> pd.DataFrame:
    """The news side at each of the user's books at the first price after the news (or, with at="pre", at the
    last price before we saw it: not a bet we could have made); one per player, statistic, book and side."""
    b = t[t.book.isin(BOOKS)].sort_values("first_observed_at").assign(side=lambda f: f.direction)
    if at == "pre":
        b = b[b.before_look.notna()]
    b = b.drop_duplicates(KEYS + ["book", "side"])
    b = b.rename(columns={f"{at}_line": "line", f"{at}_over": "over", f"{at}_under": "under", f"{at}_observed_at": "at"})
    return graded(b[KEYS + ["game_date", "book", "side", "line", "over", "under", "at", "a", "b", "age", "kind", "size"]], quotes, actuals)


def control(moves: pd.DataFrame, quotes: pd.DataFrame, actuals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Players with no news move that day: their first-to-last price move, and both sides bet at the first price."""
    q = quotes[quotes.book.isin(BOOKS)].merge(moves[KEYS].drop_duplicates(), on=KEYS, how="left", indicator=True)
    q = q[q._merge == "left_only"].sort_values("observed_at")
    first, last = q.drop_duplicates(KEYS + ["book"]), q.drop_duplicates(KEYS + ["book"], keep="last")
    drift = first.merge(last[KEYS + ["book", "imp", "observed_at"]], on=KEYS + ["book"], suffixes=("", "_close"))
    drift = drift[drift.observed_at_close > drift.observed_at].assign(drift=lambda f: f.imp_close / f.imp - 1.0)
    base = first.rename(columns={"observed_at": "at"})[KEYS + ["game_date", "book", "line", "over", "under", "at", "a", "b"]]
    both = pd.concat([base.assign(side=1.0), base.assign(side=-1.0)], ignore_index=True)
    return drift[KEYS + ["game_date", "book", "drift"]], graded(both, quotes, actuals)


# --- summaries -----------------------------------------------------------------------------------------------------
def mean_interval(frame: pd.DataFrame, col: str, draws: int = DRAWS, seed: int = 7) -> dict | None:
    """The mean of a column with a 95% interval from resampling game days."""
    d = frame[["game_date", col]].dropna()
    if d.empty:
        return None
    days = d.groupby("game_date")[col].agg(["sum", "count"]).to_numpy(float)
    out = dict(n=int(len(d)), days=int(len(days)), mean=float(d[col].mean()), low=None, high=None)
    if len(days) >= 5:
        rng = np.random.default_rng(seed)
        pick = rng.integers(0, len(days), size=(draws, len(days)))
        means = days[pick, 0].sum(axis=1) / days[pick, 1].sum(axis=1)
        out["low"], out["high"] = (float(v) for v in np.percentile(means, [2.5, 97.5]))
    return out


def lag_row(t: pd.DataFrame) -> dict:
    """`after` is over every move with a later price; `before` and `after_same` are over the moves that also
    have a price before the news."""
    both = t.dropna(subset=["before_look", "after_look"])
    moved = both[both.whole != 0]
    return dict(moves=int(len(t)), model=mean_interval(t, "size"), after=mean_interval(t, "after_look"), with_price_before=int(len(both)),
                before=mean_interval(both, "before_look"), after_same=mean_interval(both, "after_look"),
                market_agrees=float((moved.whole > 0).mean()) if len(moved) else None, moved=int(len(moved)))


def bet_row(b: pd.DataFrame) -> dict:
    return dict(bets=int(len(b)), ev=mean_interval(b, "ev"), roi=mean_interval(b, "profit"),
                hit=float(b.won.mean()) if b.won.notna().any() else None)


def summarise(moves: pd.DataFrame, quotes: pd.DataFrame, t: pd.DataFrame, bets: pd.DataFrame, drift: pd.DataFrame, blind: pd.DataFrame,
              early: pd.DataFrame | None = None) -> dict:
    fresh = t[t.age <= FRESH] if len(t) else t
    books = [*BOOKS, REFERENCE]
    age_of = lambda f, lo, hi: f[(f.age > lo) & (f.age <= hi)] if lo else f[f.age <= hi]
    mine = t[t.book.isin(BOOKS)] if len(t) else t
    report = dict(
        built=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), threshold=MOVE,
        sample=dict(first_day=str(quotes.game_date.min().date()), last_day=str(quotes.game_date.max().date()),
                    days=int(quotes.game_date.nunique()), games=int(quotes.game_id.nunique()), looks=int(quotes.observed_at.nunique()),
                    news_moves=int(len(moves)), moves_on_priced_days=int(moves.game_id.isin(set(quotes.game_id)).sum()),
                    with_price_after=int(len(mine.drop_duplicates(KEYS + ["run_at"]))) if len(mine) else 0,
                    fresh=int(len(fresh[fresh.book.isin(BOOKS)].drop_duplicates(KEYS + ["run_at"]))) if len(fresh) else 0),
        fresh_by_book={b: lag_row(fresh[fresh.book == b]) for b in books} if len(t) else {},
        by_age={label: lag_row(age_of(mine, lo, hi)) for label, lo, hi in AGES} if len(t) else {},
        fresh_by_kind={KIND[k]: lag_row(fresh[(fresh.kind == k) & fresh.book.isin(BOOKS)]) for k in KIND} if len(t) else {},
        fresh_by_size={"3% to 10%": lag_row(fresh[(fresh["size"] < BIG_MOVE) & fresh.book.isin(BOOKS)]),
                       "10% or more": lag_row(fresh[(fresh["size"] >= BIG_MOVE) & fresh.book.isin(BOOKS)])} if len(t) else {},
        bets={"All news": bet_row(bets), "Fresh news (price within 15 minutes)": bet_row(bets[bets.age <= FRESH]),
              "Older news": bet_row(bets[bets.age > FRESH]), "Fresh, 10% or more": bet_row(bets[(bets.age <= FRESH) & (bets["size"] >= BIG_MOVE)]),
              "Fresh, over side": bet_row(bets[(bets.age <= FRESH) & (bets.side > 0)]),
              "Fresh, under side": bet_row(bets[(bets.age <= FRESH) & (bets.side < 0)]),
              **{f"Fresh, {b}": bet_row(bets[(bets.age <= FRESH) & (bets.book == b)]) for b in BOOKS}} if len(bets) else {},
        speed=faster(bets, early) if early is not None and len(early) else {},
        control=dict(rows=int(len(drift)), drift=mean_interval(drift, "drift"),
                     absolute=mean_interval(drift.assign(drift=drift.drift.abs()), "drift"),
                     over=bet_row(blind[blind.side > 0]), under=bet_row(blind[blind.side < 0])))
    return report


def faster(bets: pd.DataFrame, early: pd.DataFrame) -> dict:
    """What being one look (10 minutes) faster would have been worth: the same fresh bets at the price before
    we saw the news and at the first price after."""
    key = KEYS + ["book", "side"]
    early = early[early.age <= FRESH]
    same = bets.drop(columns="size").merge(early[key + ["size"]], on=key)
    big = lambda f: f[f["size"] >= BIG_MOVE]
    return {"At the price before we saw the news (not available to us)": bet_row(early), "The same bets at the first price after": bet_row(same),
            "Model moved 10% or more: at the price before": bet_row(big(early)), "Model moved 10% or more: at the first price after": bet_row(big(same))}


# --- report --------------------------------------------------------------------------------------------------------
def pct(v: dict | None, digits: int = 2) -> str:
    if not v:
        return "-"
    text = f"{v['mean']:+.{digits}%}"
    return text if v["low"] is None else f"{text} ({v['low']:+.{digits}%} to {v['high']:+.{digits}%})"


def lag_lines(title: str, first: str, table: dict) -> list[str]:
    L = ["", f"## {title}", "", f"| {first} | Moves | Model's move | Market after our first look | With a price before | Market before our first look | "
         "Market ended the model's way |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label, r in table.items():
        agrees = "-" if r["market_agrees"] is None else f"{r['market_agrees']:.0%} of {r['moved']}"
        L.append(f"| {label} | {r['moves']} | {pct(r['model'], 1).split(' ')[0]} | {pct(r['after'])} | {r['with_price_before']} | {pct(r['before'])} | {agrees} |")
    return L


def bet_table(rows: dict) -> list[str]:
    L = ["| Bets | Count | Value at Pinnacle's close | Hit rate | Return on results |", "|---|---:|---:|---:|---:|"]
    for label, r in rows.items():
        hit = "-" if r["hit"] is None else f"{r['hit']:.1%}"
        L.append(f"| {label} | {r['bets']} | {pct(r['ev'], 1)} | {hit} | {pct(r['roi'], 1)} |")
    return L


def bet_lines(table: dict, control_rows: dict, speed: dict) -> list[str]:
    L = ["", "## Betting the news side at the first price after the news (FanDuel and DraftKings)", "",
         "Expected value is at Pinnacle's last no-vig price before tip; the result is the box score. One bet per player, statistic, book and side.", ""]
    L += bet_table({**table, "No news: every over at the first price": control_rows["over"],
                    "No news: every under at the first price": control_rows["under"]})
    if speed:
        L += ["", "## What being one look faster would be worth", "",
              "Fresh news with a price saved before the watcher saw it: the news side at that earlier price (about 10 minutes sooner; we did "
              "not have the news then) against the same bets at the first price after.", ""] + bet_table(speed)
    return L


def markdown(report: dict) -> str:
    s, c = report["sample"], report["control"]
    L = ["# NBA props: do prices lag lineup and injury news?", "",
         f"Built {report['built']}. Research only; stored data, no credits. {s['first_day']} to {s['last_day']}: {s['days']} days, "
         f"{s['games']} games with saved prices ({s['looks']} looks, all in the last 50 minutes before tip).", "",
         f"A news move is a live projection that changes {report['threshold']:.0%} or more when the news watcher sees a lineup or injury change. "
         f"{s['news_moves']} in all; {s['with_price_after']} have a FanDuel or DraftKings price after the news and {s['fresh']} of those "
         "prices came within 15 minutes of our seeing it.", "",
         "Moves are in percent of the projection the price implies, signed so that positive is the direction the model moved. "
         "Intervals are 95%, from resampling days."]
    L += lag_lines("Fresh news: where the market's move happened", "Book", report["fresh_by_book"])
    L += lag_lines("By how old the news was at the first price (FanDuel and DraftKings)", "Age of the news", report["by_age"])
    L += lag_lines("Fresh news by kind (FanDuel and DraftKings)", "Kind", report["fresh_by_kind"])
    L += lag_lines("Fresh news by size of the model's move (FanDuel and DraftKings)", "Size", report["fresh_by_size"])
    L += ["", f"Control, players with no news that day ({c['rows']} player-statistic-books): first price to last, the implied projection moved "
              f"{pct(c['drift'])} on average and {pct(c['absolute']).split(' ')[0].lstrip('+')} in size."]
    L += bet_lines(report["bets"], c, report.get("speed") or {})
    L += ["", "## Reading it", "",
          "- \"After our first look\" is what was still to come when we could first have bet. A lag we could use shows up there, and in a "
          "positive value at Pinnacle's close.",
          "- \"Before our first look\" is the move between the last price before the watcher saw the news and the first price after, for the "
          "moves that have both; the book may have had the news first, so this is the part we could not have had.",
          "- Prices are saved every 10 minutes from 50 minutes before tip, so news earlier in the day is only seen at its first price near tip.",
          "- The implied projection uses a normal approximation, so moves of the line are approximate."]
    return "\n".join(L) + "\n"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def run(log=print) -> dict:
    with psycopg2.connect(PG_DSN) as conn:
        proj = load_projections(conn)
        log(f"  {len(proj)} projection rows")
        quotes = quote_table(parse_quotes(load_quote_rows(conn)), proj)
        log(f"  {len(quotes)} priced looks")
        actuals = load_actuals(conn, sorted(set(quotes.game_id)))
    moves = news_moves(proj)
    t = timeline(moves, quotes)
    bets = news_bets(t, quotes, actuals) if len(t) else pd.DataFrame()
    early = news_bets(t, quotes, actuals, at="pre") if len(t) else pd.DataFrame()
    drift, blind = control(moves, quotes, actuals)
    report = summarise(moves, quotes, t, bets, drift, blind, early)
    atomic_json(REPORT.with_suffix(".json"), clean(report))
    REPORT.write_text(markdown(report), encoding="utf-8")
    log(f"  wrote {REPORT}")
    return report


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    run()


if __name__ == "__main__":
    main()
