"""Backtest: replay the stored price snapshots in time order and measure what a rule would have done.

Everything a rule sees is what was in a snapshot at that moment: the prices, and a fair probability built from
the other books in the same snapshot. Entries are at prices that were actually stored (never a closing price
standing in for an entry price), one per game, player, market, line and side, the first time the rule is met.
Results come from the same final statistics that settle paper bets, and closing line value from the last
snapshot before the game. `entry_delay=1` enters at the same book's price one look later (about ten minutes),
and skips the bet when that book no longer offers the line: a check on whether the price was really there.

`walk_forward` picks the minimum edge on earlier days only and applies it to later days, so a threshold is
never judged on the bets that chose it. A backtest is history, not a forecast.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from sqlalchemy import select
from sqlalchemy.engine import Engine

from sports_sniper import db, odds_math, performance, settlement
from sports_sniper.config import Settings
from sports_sniper.consensus import consensus_for, fair_probability
from sports_sniper.normalize import Quote
from sports_sniper.opportunities import detect
from sports_sniper.providers import results

SELECTION = ["event_id", "market", "player_key", "line_key", "side"]
MIN_FOR_INTERVAL = 10
MAX_DELAY_MINUTES = 25          # the next look must come this soon to count as "one look later"
CACHE_VERSION = 4
ANY_BOOK = ("any",)             # Strategy.books value for "every book that is not a sharp reference"


@dataclass(frozen=True)
class Strategy:
    """A rule for which flagged prices to bet. Empty tuples mean no restriction."""
    name: str = "custom"
    min_ev: float = 0.04
    max_ev: float | None = None
    min_conservative_ev: float = 0.0
    evidence: tuple[str, ...] = ("strong", "moderate")
    books: tuple[str, ...] = ()     # empty: the configured bet books; ANY_BOOK: every non-sharp book
    sports: tuple[str, ...] = ()
    markets: tuple[str, ...] = ()
    sides: tuple[str, ...] = ()
    min_minutes: float = 0.0
    max_minutes: float | None = None
    min_price: int | None = None
    max_price: int | None = None
    entry_delay: int = 0            # 0: the price that was flagged; 1: the same book's price one look later
    stake: float = 1.0


def presets(settings: Settings) -> list[Strategy]:
    """The rules worth comparing: the two automatic rules, the books actually held, each side, and a delayed entry."""
    books = tuple(settings.bet_books)
    v2 = Strategy("Automatic v2 (4%+, bet books)", min_ev=settings.min_ev, books=books)
    return [
        Strategy("Automatic v1 (any edge, every book)", min_ev=0.0, books=ANY_BOOK),
        v2,
        replace(v2, name="v2, FanDuel and DraftKings only", books=("fanduel", "draftkings")),
        replace(v2, name="v2, unders only", sides=("under",)),
        replace(v2, name="v2, overs only", sides=("over",)),
        replace(v2, name="v2, entered one look later", entry_delay=1),
        replace(v2, name="v2, any edge (0%+)", min_ev=0.0),
    ]


# --- candidates: every flagged price in every stored snapshot ------------------------------------------------------
def _signature(settings: Settings) -> str:
    keys = ("sharp_books", "reference_weights", "devig_method", "uncertainty_floor", "soft_only_multiplier",
            "confidence_z", "max_book_skew_seconds", "max_quote_age_seconds", "max_plausible_ev", "min_reference_books",
            "closing_window_minutes", "longshot_min_american", "outlier_min_premium", "bet_books")
    body = dict(version=CACHE_VERSION, settings={k: getattr(settings, k) for k in keys})
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def load_prices(engine: Engine) -> pd.DataFrame:
    """Every stored price from real scans (never the hypothetical sample)."""
    with engine.connect() as conn:
        real = [r[0] for r in conn.execute(select(db.Scan.id).where(db.Scan.source != "sample"))]
    frame = db.load_prices(engine)
    return frame[frame.scan_id.isin(real)].reset_index(drop=True)


def _quote(p) -> Quote:
    return Quote(sport=p.sport, event_id=p.event_id, event_name=p.event_name, commence_time=p.commence_time, book=p.book,
                 market=p.market, player=p.player, player_key=p.player_key, line=None if pd.isna(p.line) else float(p.line),
                 side=p.side, price=int(p.price), decimal=odds_math.american_to_decimal(int(p.price)),
                 last_update=p.last_update, observed_at=p.observed_at, link=p.link)


def build_event(rows: pd.DataFrame, settings: Settings) -> tuple[list[dict], list[dict]]:
    """(candidates, offers) for one game's stored prices.

    candidates: every price with a positive estimated edge at any non-sharp book in any snapshot, with its closing
    value and the same book's price one look later. offers: one row per two-sided line (the first look at it), with
    a bet book's own no-vig chance and the other books' consensus, for comparing probability sources.
    """
    wide = settings.model_copy(update={"bet_books": []})        # flag every book; a strategy narrows it later
    rows = list(rows.itertuples(index=False))
    quotes = [_quote(p) for p in rows]
    looks = sorted({q.observed_at for q in quotes})
    at = {(q.observed_at, q.book, q.market, q.player_key, db.line_key(q.line), q.side): q for q in quotes}
    by_bet: dict[tuple, list] = {}
    for p in rows:
        by_bet.setdefault((p.market, p.player_key), []).append(p)
    found = []
    for o in detect(quotes, wide):
        if o.status != "priced":
            continue
        row = dict(o.record(), line_key=db.line_key(o.line))
        bet = SimpleNamespace(**row, price_observed_at=o.observed_at)
        close = settlement.closing_value(bet, by_bet[(o.market, o.player_key)], settings)
        row["closing_line_value"] = None if close is None else close["value"]
        # one look later: the same book, line and side in the next snapshot of this game, if it comes soon enough
        nxt = bisect_right(looks, o.observed_at)
        later = looks[nxt] if nxt < len(looks) else None
        row.update(delay_price=None, delay_decimal=None, delay_observed_at=None, delay_closing_line_value=None)
        if later is not None and later < o.commence_time and later - o.observed_at <= timedelta(minutes=MAX_DELAY_MINUTES):
            again = at.get((later, o.book, o.market, o.player_key, row["line_key"], o.side))
            if again is not None:
                bet.price, bet.decimal, bet.price_observed_at = again.price, again.decimal, later
                close = settlement.closing_value(bet, by_bet[(o.market, o.player_key)], settings)
                row.update(delay_price=again.price, delay_decimal=again.decimal, delay_observed_at=later,
                           delay_closing_line_value=None if close is None else close["value"])
        found.append(row)
    return found, _offers(quotes, settings)


def _frames(found: list[dict], offers: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(found)
    for column in ("closing_line_value", "delay_price", "delay_decimal", "delay_closing_line_value"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "delay_observed_at" in frame:
        frame["delay_observed_at"] = pd.to_datetime(frame.delay_observed_at, utc=True)
    return frame, pd.DataFrame(offers)


def build_candidates(prices: pd.DataFrame, settings: Settings, progress=None, saved: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(candidates, offers, per-game results to keep). A game whose stored prices have not changed since `saved`
    was written is taken from it instead of being worked out again."""
    saved = saved or {}
    kept, found, offers, fresh = {}, [], [], 0
    events = list(prices.groupby("event_id", sort=False)) if len(prices) else []
    for event_id, rows in events:
        stamp = (len(rows), int(rows.id.max()))
        if event_id in saved and saved[event_id][0] == stamp:
            kept[event_id] = saved[event_id]
        else:
            kept[event_id] = (stamp, *build_event(rows, settings))
            fresh += 1
            if progress and fresh % 25 == 0:
                progress(f"{fresh} games worked out")
        found.extend(kept[event_id][1])
        offers.extend(kept[event_id][2])
    return (*_frames(found, offers), kept)


def _offers(quotes: list[Quote], settings: Settings) -> list[dict]:
    """The first look at each two-sided line where a bet book and a reference both price it (the over side)."""
    by_line: dict[tuple, dict] = {}
    for q in quotes:
        by_line.setdefault((q.market, q.player_key, q.line), {}).setdefault(q.observed_at, {}).setdefault(q.book, {})[q.side] = q
    sharp, out = tuple(settings.sharp_books), []
    for (market, player, line), looks in by_line.items():
        if line is None:
            continue
        for seen in sorted(looks):
            books = looks[seen]
            done = False
            for book in settings.bet_books or sorted(books):
                sides = books.get(book, {})
                if book in sharp or "over" not in sides or "under" not in sides:
                    continue
                over = sides["over"]
                if over.commence_time <= seen:
                    continue
                ref = consensus_for(over, books, method=settings.devig_method, weights=settings.reference_weights,
                                    sharp_books=sharp, max_skew_seconds=settings.max_book_skew_seconds,
                                    uncertainty_floor=settings.uncertainty_floor, soft_only_multiplier=settings.soft_only_multiplier)
                if ref is None or not ref.has_sharp:
                    continue
                out.append(dict(sport=over.sport, event_id=over.event_id, event_name=over.event_name,
                                commence_time=over.commence_time, market=market, player=over.player, player_key=player,
                                line=line, line_key=db.line_key(line), book=book, observed_at=seen,
                                over_decimal=over.decimal, under_decimal=sides["under"].decimal,
                                book_probability=fair_probability(over, sides["under"], settings.devig_method),
                                consensus_probability=ref.probability))
                done = True
                break
            if done:
                break
    return out


def grade(frame: pd.DataFrame, settings: Settings, load_results=None) -> pd.DataFrame:
    """Add status (won, lost, push, void or open) and the statistic to rows that carry a side; for offers
    (no side column) the over is graded."""
    frame = frame.copy()
    if frame.empty:
        return frame.assign(status=pd.Series(dtype=str), result_value=pd.Series(dtype=float))
    load_results = load_results or results.load
    games = {}
    for event_id, name, sport, start in frame[["event_id", "event_name", "sport", "commence_time"]].drop_duplicates("event_id").itertuples(index=False):
        if " @ " in name:
            games[event_id] = results.Game(sport, event_id, *name.split(" @ ")[::-1], start.to_pydatetime())
    try:
        found = load_results(settings.results_dsn, list(games.values())) if games else {}
    except Exception:                    # the results database is optional: everything stays open
        found = {}
    status, value = [], []
    for row in frame.itertuples(index=False):
        bet = SimpleNamespace(sport=row.sport, market=row.market, side=getattr(row, "side", "over"), player_key=row.player_key,
                              line=None if pd.isna(row.line) else float(row.line))
        s, v, _ = settlement.decide(bet, found.get(row.event_id))
        status.append(s)
        value.append(v)
    return frame.assign(status=status, result_value=value)


def candidates(engine: Engine, settings: Settings, *, cache: Path | None = None, progress=None, load_results=None):
    """(candidates, offers), graded. The slow part (detection and closing values) is kept per game in `cache` and
    redone only for games with new prices, or for all of them when a pricing setting changes; grading is always fresh."""
    prices = load_prices(engine)
    signature, saved = _signature(settings), {}
    if cache is not None and cache.exists():
        try:
            stored = pickle.loads(cache.read_bytes())
            saved = stored["events"] if stored.get("signature") == signature else {}
        except Exception:
            saved = {}
    found, offers, kept = build_candidates(prices, settings, progress, saved)
    if cache is not None and {k: v[0] for k, v in kept.items()} != {k: v[0] for k, v in saved.items()}:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(dict(signature=signature, events=kept)))
    return grade(found, settings, load_results), grade(offers, settings, load_results)


def default_cache(settings: Settings) -> Path | None:
    url = settings.database_url
    return Path(url.removeprefix("sqlite:///")).with_name("backtest_cache.pkl") if url.startswith("sqlite:///") and ":memory:" not in url else None


# --- applying a rule ---------------------------------------------------------------------------------------------
def run(found: pd.DataFrame, strategy: Strategy, default_books: tuple[str, ...] = ()) -> pd.DataFrame:
    """The bets a rule would have made, in time order, shaped like paper bets (so `performance` can read them)."""
    if found.empty:
        return _empty()
    data = found
    books = () if strategy.books == ANY_BOOK else (strategy.books or default_books)
    if strategy.entry_delay:
        data = data[data.delay_price.notna()].copy()
        data["flagged_price"] = data.price
        data["ev"] = [odds_math.expected_roi(p, d) for p, d in zip(data.probability, data.delay_decimal)]
        data["price"], data["decimal"] = data.delay_price.astype(int), data.delay_decimal
        data["price_observed_at"], data["closing_line_value"] = data.delay_observed_at, data.delay_closing_line_value
    else:
        data = data.assign(price_observed_at=data.observed_at)
    keep = (data.ev >= strategy.min_ev) & (data.conservative_ev > strategy.min_conservative_ev) & data.evidence.isin(strategy.evidence)
    keep &= data.minutes_to_start >= strategy.min_minutes
    for column, wanted in (("book", books), ("sport", strategy.sports), ("market", strategy.markets), ("side", strategy.sides)):
        if wanted:
            keep &= data[column].isin(wanted)
    if strategy.max_ev is not None:
        keep &= data.ev <= strategy.max_ev
    if strategy.max_minutes is not None:
        keep &= data.minutes_to_start <= strategy.max_minutes
    if strategy.min_price is not None:
        keep &= data.price >= strategy.min_price
    if strategy.max_price is not None:
        keep &= data.price <= strategy.max_price
    bets = (data[keep].sort_values(["observed_at", "decimal"], ascending=[True, False], kind="stable")
            .drop_duplicates(SELECTION, keep="first").copy())
    bets["stake"] = strategy.stake
    bets["profit"] = [settlement.profit(s, strategy.stake, d) if s != "open" else None for s, d in zip(bets.status, bets.decimal)]
    bets["day"] = bets.commence_time.dt.tz_convert(results.EASTERN).dt.date
    return bets.sort_values(["commence_time", "observed_at"], kind="stable").reset_index(drop=True)


def drawdown(profits: pd.Series) -> float:
    """The largest fall from a running high, in the same units as the profits (settlement order)."""
    running = profits.fillna(0).cumsum()
    return float((running.cummax().clip(lower=0) - running).max()) if len(running) else 0.0


def losing_streak(statuses) -> int:
    longest = current = 0
    for status in statuses:
        if status == "lost":
            current += 1
            longest = max(longest, current)
        elif status == "won":
            current = 0
    return longest


def metrics(bets: pd.DataFrame) -> dict:
    """Record, ROI, closing value and calibration (as for paper bets), plus drawdown, streak and intervals."""
    out = performance.summarize(bets if len(bets) else _empty())
    settled = bets[bets.status.isin(performance.SETTLED)] if len(bets) else bets
    out["max_drawdown"] = drawdown(settled.profit) if len(settled) else 0.0
    out["longest_losing_streak"] = losing_streak(settled.status) if len(settled) else 0
    out["days"] = int(settled.day.nunique()) if len(settled) else 0
    out["games"] = int(settled.event_id.nunique()) if len(settled) else 0
    enough = len(settled) >= MIN_FOR_INTERVAL         # a handful of bets gives a meaningless interval
    out["roi_interval"] = (None if out["roi"] is None or out["roi_se"] is None or not enough else
                           (out["roi"] - 1.96 * out["roi_se"], out["roi"] + 1.96 * out["roi_se"]))
    out["closing_interval"] = (None if out["closing_value"] is None or out["closing_se"] is None else
                               (out["closing_value"] - 1.96 * out["closing_se"], out["closing_value"] + 1.96 * out["closing_se"]))
    return out


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["status", "stake", "profit", "closing_line_value", "event_id", "ev", "decimal", "probability",
                                 "standard_error", "sport", "market", "book", "side", "player", "player_key", "line", "price", "day",
                                 "commence_time", "observed_at", "price_observed_at", "event_name", "evidence", "conservative_ev"])


def compare(found: pd.DataFrame, strategies: list[Strategy], default_books: tuple[str, ...] = ()) -> list[tuple[Strategy, dict]]:
    return [(s, metrics(run(found, s, default_books))) for s in strategies]


def by_period(bets: pd.DataFrame, days: int = 7) -> list[tuple[str, dict]]:
    """Results in consecutive blocks of days, to show whether a result holds across sample periods."""
    if bets.empty:
        return []
    first = min(bets.day)
    block = [(d - first).days // days for d in bets.day]
    out = []
    for number, group in bets.groupby(block):
        start = first + timedelta(days=int(number) * days)
        out.append((f"{start:%b %d} to {start + timedelta(days=days - 1):%b %d}", metrics(group)))
    return out


# --- walk-forward ------------------------------------------------------------------------------------------------
def walk_forward(found: pd.DataFrame, base: Strategy, grid: list[float], *, default_books: tuple[str, ...] = (),
                 objective: str = "roi", min_train_days: int = 7, test_days: int = 3, min_train_bets: int = 40) -> dict:
    """Choose the minimum edge on earlier days, apply it to the next `test_days`, move on, and pool the later days.

    `objective` is what the choice maximises on the earlier days: "roi" (results) or "closing_value". With fewer
    than `min_train_bets` settled bets at a threshold it is not eligible; with none eligible the base rule's own
    threshold is used. Returns the folds and the pooled out-of-sample bets.
    """
    if objective not in ("roi", "closing_value"):
        raise ValueError("objective must be roi or closing_value")
    if found.empty:
        return dict(folds=[], bets=run(found, base, default_books))
    found = found.assign(day=found.commence_time.dt.tz_convert(results.EASTERN).dt.date)
    days = sorted(found.day.unique())
    folds, pooled = [], []
    for start in range(min_train_days, len(days), test_days):
        train_days, test = set(days[:start]), set(days[start:start + test_days])
        train = found[found.day.isin(train_days)]
        scores = {}
        for threshold in grid:
            m = metrics(run(train, replace(base, min_ev=threshold), default_books))
            settled = m["won"] + m["lost"] + m["push"]
            count = settled if objective == "roi" else m["closing_count"]
            if count >= min_train_bets and m[objective] is not None:
                scores[threshold] = m[objective]
        chosen = max(scores, key=scores.get) if scores else base.min_ev
        bets = run(found[found.day.isin(test)], replace(base, min_ev=chosen), default_books)
        folds.append(dict(train_from=days[0], train_to=days[start - 1], test_from=min(test), test_to=max(test), chosen=chosen,
                          tuned=bool(scores), train_score=scores.get(chosen), **{f"test_{k}": v for k, v in metrics(bets).items()}))
        pooled.append(bets)
    out = pd.concat(pooled, ignore_index=True) if pooled else run(found.iloc[0:0], base, default_books)
    return dict(folds=folds, bets=out)


def describe(strategy: Strategy) -> dict:
    return asdict(strategy)
