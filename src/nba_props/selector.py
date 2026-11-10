"""NBA prop bet selector (paper only): machine learning on top of the sportsbook's own price.

Stage 1 is the walk-forward stat projection that already exists (nba_props.model, nbaprops.projections).
Stage 2, here, is a classifier that starts from the book's own no-vig probability and learns where that
price is wrong, from the projection and the shape of the price. A pick is an offer whose modelled chance
makes it worth at least MIN_EV more than its price. Nothing here stakes money or sends alerts.

Verdict (2026-11-10): no edge, so it is not scheduled. Month by month over two seasons it only breaks even at
DraftKings' price, and its 269 picks on live quotes sat 4.5% below Pinnacle's closing price. The reports
(reports/nba_selector_backtest_latest.md, reports/nba_selector_latest.md) are the record; rerun by hand.

  python -m nba_props.selector backtest   # each month predicted from earlier months, graded at DraftKings' price
  python -m nba_props.selector train      # fit on all priced history; save the model and its manifest
  python -m nba_props.selector capture    # score the saved live quotes, record paper picks, refresh the report
  python -m nba_props.selector report     # grade the paper picks (results and Pinnacle's close)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from nba_props import model
from nfl_pipeline import sharp_math
from nfl_pipeline.db import PG_DSN

ROOT = Path(__file__).resolve().parents[2]
VERSION = "nba-ml-selector-v1"
MARKET_STAT = {"points": "pts", "rebounds": "reb", "assists": "ast"}  # the stats with priced history to learn from
BOOKS = ("fanduel", "draftkings", "espnbet")
MIN_EV = 0.06                # a margin above the book's cut; the backtest shows no threshold that profits
WINDOW = (15, 45)            # minutes before tip, the same window as the price-gap alerts
BACKTEST_FROM = "2024-10"    # the first season only trains
THRESHOLDS = (0.0, 0.02, 0.04, 0.06, 0.08)
FEATURES = ["logit_mkt", "line", "hold", "over_dec", "under_dec", "is_pts", "is_reb", "is_ast",
            "p_model", "p_gap", "pred", "pred_minus_line", "pred_ratio"]
PARAMS = dict(n_estimators=300, learning_rate=0.02, num_leaves=8, min_child_samples=200, subsample=0.7,
              subsample_freq=1, colsample_bytree=0.6, reg_lambda=20.0, verbose=-1, random_state=11)
PREDICTIONS = ROOT / "reports" / "nba_props_model_predictions.csv.gz"   # written by `python -m nba_props.model`
MODEL_PATH = ROOT / "src" / "nba_props" / "models" / "selector_v1.txt"   # local file; its hash is in the manifest
MANIFEST = ROOT / "reports" / "nba_selector_model.json"
BACKTEST_REPORT = ROOT / "reports" / "nba_selector_backtest_latest"
REPORT = ROOT / "reports" / "nba_selector_latest"
MAX_CLOSE_AGE_MINUTES = 22

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bets.nba_selector_picks (
  pick_id BIGSERIAL PRIMARY KEY, version TEXT NOT NULL, model_sha256 TEXT NOT NULL,
  observation_id TEXT NOT NULL, observed_at_utc TIMESTAMPTZ NOT NULL, event_id TEXT NOT NULL,
  commence_time_utc TIMESTAMPTZ NOT NULL, home_team TEXT, away_team TEXT,
  player TEXT NOT NULL, player_norm TEXT NOT NULL, name_key TEXT NOT NULL, market TEXT NOT NULL,
  side TEXT NOT NULL, line NUMERIC NOT NULL, book TEXT NOT NULL, price INTEGER NOT NULL, link TEXT,
  book_fair NUMERIC NOT NULL, projection NUMERIC NOT NULL, projection_probability NUMERIC NOT NULL,
  projection_run_at TIMESTAMPTZ NOT NULL, probability NUMERIC NOT NULL, ev NUMERIC NOT NULL,
  minutes_to_start NUMERIC NOT NULL, sharp_fair NUMERIC, sharp_ev NUMERIC, replayed BOOLEAN NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(), UNIQUE (version, event_id, name_key, market));
CREATE TABLE IF NOT EXISTS bets.nba_selector_seen (
  version TEXT NOT NULL, observation_id TEXT NOT NULL, PRIMARY KEY (version, observation_id));
"""


# --- the model -------------------------------------------------------------------------------------
def decimal(price) -> float:
    price = float(price)
    return 1 + price / 100 if price > 0 else 1 + 100 / abs(price)


def design(frame: pd.DataFrame) -> pd.DataFrame:
    """Model inputs from stat, line, over_price, under_price, pred (projected stat) and p_model (its P(over))."""
    out = pd.DataFrame(index=frame.index)
    out["over_dec"] = [decimal(p) for p in frame.over_price]
    out["under_dec"] = [decimal(p) for p in frame.under_price]
    out["p_market"] = [sharp_math.no_vig_over(int(o), int(u)) for o, u in zip(frame.over_price, frame.under_price)]
    out["p_market"] = out.p_market.astype(float).clip(0.02, 0.98)
    out["logit_mkt"] = np.log(out.p_market / (1 - out.p_market))
    out["hold"] = 1 / out.over_dec + 1 / out.under_dec - 1
    out["line"] = frame.line.astype(float)
    for stat in MARKET_STAT.values():
        out[f"is_{stat}"] = (frame.stat == stat).astype(int)
    out["p_model"] = frame.p_model.astype(float)
    out["p_gap"] = out.p_model - out.p_market
    out["pred"] = frame.pred.astype(float)
    out["pred_minus_line"] = out.pred - out.line
    out["pred_ratio"] = out.pred / out.line.clip(lower=0.5)
    return out


def fit(X: pd.DataFrame, over: pd.Series):
    """Boosted trees that start from the book's no-vig probability, so they only learn its errors."""
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(**PARAMS)
    clf.fit(X[FEATURES], over.astype(int), init_score=X.logit_mkt)
    return clf.booster_


def probability(booster, X: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(booster.predict(X[FEATURES], raw_score=True), dtype=float) + X.logit_mkt.to_numpy(float)
    return 1 / (1 + np.exp(-raw))


def best_side(X: pd.DataFrame, p_over: np.ndarray) -> pd.DataFrame:
    """The better side of each offer under the model: side, its probability, decimal price and EV per $1."""
    p_over = np.asarray(p_over, dtype=float)
    ev_over, ev_under = p_over * X.over_dec.to_numpy() - 1, (1 - p_over) * X.under_dec.to_numpy() - 1
    over = ev_over >= ev_under
    return pd.DataFrame(dict(side=np.where(over, "over", "under"), probability=np.where(over, p_over, 1 - p_over),
                             dec=np.where(over, X.over_dec, X.under_dec), ev=np.where(over, ev_over, ev_under)),
                        index=X.index)


# --- history and backtest --------------------------------------------------------------------------
def history(conn) -> pd.DataFrame:
    """Every pre-game DraftKings line with its result and the projection made without that game."""
    if not PREDICTIONS.exists():
        raise SystemExit("Run `python -m nba_props.model` first: it writes the walk-forward projections this needs.")
    preds = pd.read_csv(PREDICTIONS, parse_dates=["game_date"])
    m = model.market_rows(preds, model.load_lines(conn))
    m["over"] = m.over.astype(int)
    return m.sort_values(["game_date", "name_key", "stat"]).reset_index(drop=True)


def walk_forward(m: pd.DataFrame, X: pd.DataFrame, start: str = BACKTEST_FROM) -> pd.Series:
    out = pd.Series(np.nan, index=m.index)
    for month in pd.period_range(start, m.game_date.max(), freq="M"):
        train = m.game_date < month.start_time
        test = (m.game_date >= month.start_time) & (m.game_date <= month.end_time)
        if test.any() and train.any():
            out[test] = probability(fit(X[train], m.over[train]), X[test])
    return out


def settle(m: pd.DataFrame, sides: pd.DataFrame) -> pd.DataFrame:
    won = np.where(sides.side == "over", m.over == 1, m.over == 0)
    return sides.assign(day=m.game_date, season=m.season, stat=m.stat, won=won,
                        profit=np.where(won, sides.dec - 1, -1.0))


def record(bets: pd.DataFrame) -> dict:
    """Flat $1 results; the standard error treats a whole day as one draw (same-night props move together)."""
    if bets.empty:
        return dict(bets=0, win_rate=None, roi=None, se=None, unders=None)
    dev = bets.profit - bets.profit.mean()
    se = float(np.sqrt((dev.groupby(bets.day).sum() ** 2).sum()) / len(bets))
    return dict(bets=int(len(bets)), win_rate=float(bets.won.mean()), roi=float(bets.profit.mean()), se=se,
                unders=float((bets.side == "under").mean()))


def log_loss(p, over) -> float:
    p, over = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6), np.asarray(over, dtype=float)
    return float(-np.mean(over * np.log(p) + (1 - over) * np.log(1 - p)))


def backtest(m: pd.DataFrame) -> dict:
    X = design(m)
    p = walk_forward(m, X)
    keep = p.notna()
    m, X, p = m[keep], X[keep], p[keep]
    every = lambda side: settle(m, pd.DataFrame(dict(side=side, dec=X.over_dec if side == "over" else X.under_dec), index=m.index))
    sides = settle(m, best_side(X, p.to_numpy()))
    doc = dict(version=VERSION, rows=int(len(m)), first=str(m.game_date.min().date()), last=str(m.game_date.max().date()),
               log_loss=dict(book=log_loss(X.p_market, m.over), projection=log_loss(m.p_model, m.over), selector=log_loss(p, m.over)),
               log_loss_by_season={str(s): dict(book=log_loss(X.p_market[g.index], g.over), selector=log_loss(p[g.index], g.over))
                                   for s, g in m.groupby("season")},
               blind=dict(over=record(every("over")), under=record(every("under"))), thresholds=[])
    for k in THRESHOLDS:
        chosen = sides[sides.ev >= k]
        doc["thresholds"].append(dict(min_ev=k, **record(chosen),
                                      by_season={str(s): record(g) for s, g in chosen.groupby("season")}))
    chosen = sides[sides.ev >= MIN_EV]
    doc["rule"] = dict(min_ev=MIN_EV, by_stat_side={f"{stat} {side}": record(g) for (stat, side), g in chosen.groupby(["stat", "side"])})
    return doc


def _pct(v, digits: int = 1) -> str:
    return "-" if v is None else f"{v:+.{digits}%}"


def backtest_markdown(doc: dict) -> str:
    ll = doc["log_loss"]
    L = ["# NBA prop selector: backtest against DraftKings' pre-game price", "",
         f"`{doc['version']}`. {doc['rows']:,} points, rebounds and assists offers, {doc['first']} to {doc['last']}. Each month is "
         "predicted by a model trained only on earlier months, and every bet is graded at the price DraftKings offered.", "",
         f"Log loss (lower is better): DraftKings' no-vig price {ll['book']:.5f}, the stat projection alone {ll['projection']:.5f}, "
         f"the selector {ll['selector']:.5f}.", "",
         "| Bets taken | Bets | Won | Result per $1 | Standard error | Unders | " + " | ".join(doc["log_loss_by_season"]) + " |",
         "|---|---:|---:|---:|---:|---:|" + "---:|" * len(doc["log_loss_by_season"])]
    row = lambda label, r, seasons: (f"| {label} | {r['bets']:,} | {r['win_rate']:.1%} | {_pct(r['roi'])} | {r['se']:.1%} | "
                                     f"{r['unders']:.0%} | " + " | ".join(seasons) + " |") if r["bets"] else f"| {label} | 0 |"
    blank = ["" for _ in doc["log_loss_by_season"]]
    L += [row("Every over", doc["blind"]["over"], blank), row("Every under", doc["blind"]["under"], blank)]
    for t in doc["thresholds"]:
        seasons = [f"{_pct(t['by_season'][s]['roi'])} ({t['by_season'][s]['bets']:,})" if s in t["by_season"] else "-"
                   for s in doc["log_loss_by_season"]]
        L.append(row(f"Selector, EV at least {t['min_ev']:.0%}", t, seasons))
    L += ["", f"The paper rule is EV at least {doc['rule']['min_ev']:.0%}. By stat and side:", "",
          "| Stat and side | Bets | Won | Result per $1 |", "|---|---:|---:|---:|"]
    L += [f"| {k} | {r['bets']:,} | {r['win_rate']:.1%} | {_pct(r['roi'])} |" for k, r in doc["rule"]["by_stat_side"].items()]
    best = max((t for t in doc["thresholds"] if t["bets"]), key=lambda t: t["roi"], default=None)
    if best is not None:
        L += ["", f"Best threshold: EV at least {best['min_ev']:.0%}, {_pct(best['roi'])} per $1 with a standard error of "
                  f"{best['se']:.1%}. An edge needs a result well above zero in every season; anything within two standard "
                  "errors of zero is break-even at best."]
    return "\n".join(L) + "\n"


def write(path: Path, doc: dict, text: str) -> None:
    path.with_suffix(".json").write_bytes((json.dumps(doc, indent=1, default=str) + "\n").encode())
    path.with_suffix(".md").write_bytes(text.encode())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(m: pd.DataFrame, now: datetime) -> dict:
    booster = fit(design(m), m.over)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_PATH))
    manifest = dict(version=VERSION, trained_at=now.isoformat(), rows=int(len(m)), first=str(m.game_date.min().date()),
                    last=str(m.game_date.max().date()), features=FEATURES, params=PARAMS, min_ev=MIN_EV, window=list(WINDOW),
                    books=list(BOOKS), markets=list(MARKET_STAT), model_sha256=sha256(MODEL_PATH), research_only=True)
    MANIFEST.write_bytes((json.dumps(manifest, indent=1) + "\n").encode())
    return manifest


def load_model():
    """(booster, manifest), or (None, reason) when the saved model is missing or is not the one in the manifest."""
    if not MANIFEST.exists() or not MODEL_PATH.exists():
        return None, "model_missing"
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("version") != VERSION or manifest.get("model_sha256") != sha256(MODEL_PATH):
        return None, "model_changed"
    import lightgbm as lgb
    return lgb.Booster(model_file=str(MODEL_PATH)), manifest


# --- live quotes -----------------------------------------------------------------------------------
def offers(event: dict, observed: datetime) -> list[dict]:
    """Fresh half-point over/under pairs at the bet books, with Pinnacle's fair P(over) when it quotes the same line."""
    from sharp_pipeline import scan
    books = {b["key"]: scan.pairs(dict(b, _home=event.get("home_team"), _away=event.get("away_team")), observed)
             for b in event.get("bookmakers", [])}
    sharp = books.get(scan.SHARP_BOOK) or {}
    out = []
    for book in BOOKS:
        for (market, who, line), offer in (books.get(book) or {}).items():
            if market not in MARKET_STAT or abs(line % 1 - 0.5) > 1e-9 or not offer.get("player"):
                continue
            ref = sharp.get((market, who, line))
            fair = None
            if ref is not None and abs((offer["updated"] - ref["updated"]).total_seconds()) <= scan.MAX_SOURCE_SKEW:
                fair = sharp_math.no_vig_over(ref["a"], ref["b"])
            out.append(dict(book=book, market=market, stat=MARKET_STAT[market], player=offer["player"], player_norm=who,
                            name_key=model.match_name(offer["player"]), line=float(line), over_price=int(offer["a"]),
                            under_price=int(offer["b"]), over_link=offer.get("a_link"), under_link=offer.get("b_link"),
                            sharp_fair_over=fair))
    return out


def projection_before(runs: list[tuple], observed: datetime):
    """The latest (run_at, mu, a, b) made strictly before the quote was observed; never a later one."""
    earlier = [r for r in runs if r[0] < observed]
    return max(earlier, key=lambda r: r[0]) if earlier else None


def score(rows: list[dict], projections: dict, day, observed: datetime, booster) -> pd.DataFrame:
    """Offers that have an earlier projection, with the selector's probability and the better side's EV."""
    kept = []
    for r in rows:
        found = projection_before(projections.get((day, r["name_key"], r["stat"]), []), observed)
        if found is None:
            continue
        run_at, mu, a, b = found
        p_model = float(model.p_over(np.array([mu]), np.array([r["line"]]), a, b)[0])
        kept.append(dict(r, pred=mu, p_model=p_model, projection_run_at=run_at))
    frame = pd.DataFrame(kept)
    if frame.empty:
        return frame
    X = design(frame)
    sides = best_side(X, probability(booster, X))
    frame["book_fair"] = np.where(sides.side == "over", X.p_market, 1 - X.p_market)
    frame["projection_probability"] = np.where(sides.side == "over", frame.p_model, 1 - frame.p_model)
    frame[["side", "probability", "ev"]] = sides[["side", "probability", "ev"]]
    over = frame.side == "over"
    frame["price"] = np.where(over, frame.over_price, frame.under_price)
    frame["link"] = np.where(over, frame.over_link, frame.under_link)
    fair = pd.to_numeric(frame.sharp_fair_over, errors="coerce")
    frame["sharp_fair"] = np.where(over, fair, 1 - fair)
    frame["sharp_ev"] = [None if pd.isna(f) else sharp_math.ev(float(f), int(p)) for f, p in zip(frame.sharp_fair, frame.price)]
    return frame


def decide(scored: pd.DataFrame, min_ev: float = MIN_EV) -> pd.DataFrame:
    """One pick per player and market: the offer with the highest EV, if it reaches min_ev."""
    if scored.empty:
        return scored
    best = scored.sort_values("ev", ascending=False).drop_duplicates(["name_key", "market"])
    return best[best.ev >= min_ev]


def capture(conn, now: datetime) -> dict:
    booster, manifest = load_model()
    if booster is None:
        return dict(status=manifest, picks=0)
    from zoneinfo import ZoneInfo
    eastern = ZoneInfo("America/New_York")
    trained_at = datetime.fromisoformat(manifest["trained_at"])
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(SCHEMA_SQL)
        cur.execute("""SELECT q.observation_id, q.event_id, q.observed_at_utc, q.commence_time_utc, q.payload
                       FROM bets.sharp_multi_quotes q
                       WHERE q.sport = 'nba' AND q.observed_at_utc < q.commence_time_utc
                         AND NOT EXISTS (SELECT 1 FROM bets.nba_selector_seen s
                                         WHERE s.version = %s AND s.observation_id = q.observation_id)
                       ORDER BY q.observed_at_utc""", (VERSION,))
        pending = cur.fetchall()
        days = sorted({r["commence_time_utc"].astimezone(eastern).date() for r in pending})
        cur.execute("""SELECT run_at_utc, game_date, name_key, stat, mu, disp_a, disp_b FROM nbaprops.projections
                       WHERE game_date = ANY(%s) AND stat = ANY(%s)""", (days, list(MARKET_STAT.values())))
        projections: dict = {}
        for r in cur.fetchall():
            projections.setdefault((r["game_date"], r["name_key"], r["stat"]), []).append(
                (r["run_at_utc"], float(r["mu"]), float(r["disp_a"]), float(r["disp_b"])))
        scored_offers = picks = 0
        for q in pending:
            observed, kick = q["observed_at_utc"], q["commence_time_utc"]
            lead = (kick - observed).total_seconds() / 60
            if WINDOW[0] <= lead <= WINDOW[1]:
                event = q["payload"]
                scored = score(offers(event, observed), projections, kick.astimezone(eastern).date(), observed, booster)
                scored_offers += len(scored)
                for p in decide(scored).to_dict("records"):
                    cur.execute("""INSERT INTO bets.nba_selector_picks (version, model_sha256, observation_id, observed_at_utc,
                                     event_id, commence_time_utc, home_team, away_team, player, player_norm, name_key, market,
                                     side, line, book, price, link, book_fair, projection, projection_probability,
                                     projection_run_at, probability, ev, minutes_to_start, sharp_fair, sharp_ev, replayed)
                                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                                   ON CONFLICT (version, event_id, name_key, market) DO NOTHING""",
                                (VERSION, manifest["model_sha256"], q["observation_id"], observed, q["event_id"], kick,
                                 event.get("home_team"), event.get("away_team"), p["player"], p["player_norm"], p["name_key"],
                                 p["market"], p["side"], p["line"], p["book"], int(p["price"]), p["link"], float(p["book_fair"]),
                                 float(p["pred"]), float(p["projection_probability"]), p["projection_run_at"],
                                 float(p["probability"]), float(p["ev"]), lead,
                                 None if pd.isna(p["sharp_fair"]) else float(p["sharp_fair"]),
                                 None if p["sharp_ev"] is None or pd.isna(p["sharp_ev"]) else float(p["sharp_ev"]),
                                 observed < trained_at))
                    picks += cur.rowcount
            cur.execute("INSERT INTO bets.nba_selector_seen (version, observation_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        (VERSION, q["observation_id"]))
    conn.commit()
    return dict(status="ok", observations=len(pending), offers_scored=scored_offers, picks=picks)


# --- grading the paper picks -----------------------------------------------------------------------
def load_picks(conn) -> pd.DataFrame:
    """Picks with their result and the pick price's EV against Pinnacle's last quote before tip at the same line."""
    from sharp_pipeline import outcomes, scan
    from zoneinfo import ZoneInfo
    eastern = ZoneInfo("America/New_York")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(SCHEMA_SQL)
        cur.execute("SELECT * FROM bets.nba_selector_picks WHERE version = %s ORDER BY observed_at_utc", (VERSION,))
        picks = pd.DataFrame(cur.fetchall())
        if picks.empty:
            return picks
        cur.execute("""SELECT event_id, observed_at_utc, payload FROM bets.sharp_multi_quotes
                       WHERE sport = 'nba' AND event_id = ANY(%s) AND observed_at_utc < commence_time_utc
                         AND observed_at_utc >= commence_time_utc - %s * interval '1 minute'
                       ORDER BY observed_at_utc""", (sorted(set(picks.event_id)), MAX_CLOSE_AGE_MINUTES))
        closes: dict = {}
        for r in cur.fetchall():
            event = r["payload"]
            for b in event.get("bookmakers", []):
                if b.get("key") == scan.SHARP_BOOK:
                    quotes = scan.pairs(dict(b, _home=event.get("home_team"), _away=event.get("away_team")), r["observed_at_utc"])
                    for key, quote in quotes.items():
                        closes[(r["event_id"],) + key] = (r["observed_at_utc"], quote)   # later rows overwrite earlier ones
    picks["day"] = [t.astimezone(eastern).date() for t in picks.commence_time_utc]
    players, _, _ = outcomes.nba_actuals(conn, sorted(set(picks.day)))
    results, close_ev = [], []
    for r in picks.to_dict("records"):
        stats = players.get((r["day"], r["name_key"]))
        result = outcomes.grade(r, None if stats is None else stats[MARKET_STAT[r["market"]]], None, None)
        results.append(result)
        found = closes.get((r["event_id"], r["market"], r["player_norm"], float(r["line"])))
        value = None
        if found is not None and found[0] > r["observed_at_utc"]:
            fair = sharp_math.no_vig_over(found[1]["a"], found[1]["b"])
            if fair is not None:
                value = sharp_math.ev(fair if r["side"] == "over" else 1 - fair, int(r["price"]))
        close_ev.append(value)
    picks["result"], picks["close_ev"] = results, close_ev
    picks["profit"] = [outcomes.profit(res, price) for res, price in zip(results, picks.price)]
    for col in ("ev", "sharp_ev", "probability", "book_fair"):
        picks[col] = pd.to_numeric(picks[col], errors="coerce")
    return picks


def tally(frame: pd.DataFrame) -> dict:
    settled = frame[frame.result.isin(["win", "loss", "push"])]
    close = pd.to_numeric(frame.close_ev, errors="coerce").dropna()
    sharp = pd.to_numeric(frame.sharp_ev, errors="coerce").dropna()
    out = dict(picks=int(len(frame)), settled=int(len(settled)), wins=int((settled.result == "win").sum()),
               losses=int((settled.result == "loss").sum()), units=float(settled.profit.sum()) if len(settled) else 0.0,
               roi=float(settled.profit.mean()) if len(settled) else None, model_ev=float(frame.ev.mean()) if len(frame) else None,
               sharp_ev_at_pick=float(sharp.mean()) if len(sharp) else None, with_sharp=int(len(sharp)),
               close_ev=float(close.mean()) if len(close) else None, with_close=int(len(close)), close_se=None)
    if len(close) > 1:
        days = frame.loc[close.index, "day"]
        dev = close - close.mean()
        out["close_se"] = float(np.sqrt((dev.groupby(days).sum() ** 2).sum()) / len(close))
    return out


def summarize(picks: pd.DataFrame, manifest, now: datetime) -> dict:
    doc = dict(version=VERSION, generated_at=now.isoformat(), research_only=True,
               model=manifest if isinstance(manifest, dict) else dict(status=manifest))
    if picks.empty:
        return dict(doc, total=tally(pd.DataFrame(columns=["result", "close_ev", "sharp_ev", "profit", "ev", "day"])), groups={})
    groups = {"Live (after the model was fixed)": picks[~picks.replayed], "Replayed from saved quotes": picks[picks.replayed]}
    groups.update({f"Book: {b}": g for b, g in picks.groupby("book")})
    groups.update({f"Side: {s}": g for s, g in picks.groupby("side")})
    groups.update({f"Stat: {s}": g for s, g in picks.groupby("market")})
    return dict(doc, total=tally(picks), groups={k: tally(g) for k, g in groups.items()},
                first=str(picks.day.min()), last=str(picks.day.max()), days=int(picks.day.nunique()))


def markdown(doc: dict) -> str:
    L = ["# NBA prop selector: paper picks", "",
         f"`{doc['version']}`. Research only: no stakes, no alerts. A pick is a FanDuel, DraftKings or theScore Bet points, "
         f"rebounds or assists offer, seen {WINDOW[0]}-{WINDOW[1]} minutes before tip, that the model values at least "
         f"{MIN_EV:.0%} above its price; one pick per player and stat.", ""]
    if not doc.get("groups"):
        status = doc["model"].get("status")
        return "\n".join(L + [f"No picks yet{'' if not status else f' ({status})'}."]) + "\n"
    L += [f"{doc['total']['picks']} picks over {doc['days']} game days, {doc['first']} to {doc['last']}.", "",
          "The edge against Pinnacle is the fast test: a model that finds real value should show a positive number there. "
          "Results at $1 are noisy until there are far more picks.", "",
          "| Picks | Count | Settled | Record | Result ($1 each) | Per bet | Model's EV | Edge vs Pinnacle at pick | Edge vs Pinnacle at close |",
          "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    line = lambda label, t: (f"| {label} | {t['picks']} | {t['settled']} | {t['wins']}-{t['losses']} | "
                             f"{'-' if not t['settled'] else ('+' if t['units'] >= 0 else '-') + '$' + format(abs(t['units']), '.2f')} | "
                             f"{_pct(t['roi'])} | {_pct(t['model_ev'])} | {_pct(t['sharp_ev_at_pick'])} ({t['with_sharp']}) | "
                             f"{_pct(t['close_ev'])} ({t['with_close']}) |")
    L.append(line("All", doc["total"]))
    L += [line(label, t) for label, t in doc["groups"].items() if t["picks"]]
    se = doc["total"].get("close_se")
    if se is not None:
        L += ["", f"Edge at the close across all picks: {_pct(doc['total']['close_ev'])}, standard error {se:.1%} (whole days as draws)."]
    L += ["", "Replayed picks come from quotes saved before this model existed, so the model had never seen those games. "
              "Picks that sit below zero against Pinnacle are paying the book's cut, not finding value."]
    return "\n".join(L) + "\n"


def write_report(conn, now: datetime) -> dict:
    _, manifest = load_model()
    doc = summarize(load_picks(conn), manifest, now)
    write(REPORT, doc, markdown(doc))
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("backtest", "train", "capture", "report"))
    args = ap.parse_args()
    now = datetime.now(timezone.utc)
    with psycopg2.connect(PG_DSN) as conn:
        if args.command == "backtest":
            doc = backtest(history(conn))
            text = backtest_markdown(doc)
            write(BACKTEST_REPORT, doc, text)
            print(text)
        elif args.command == "train":
            print(json.dumps(train(history(conn), now), indent=1))
        else:
            if args.command == "capture":
                print(json.dumps(capture(conn, now)))
            doc = write_report(conn, now)
            print(json.dumps(doc.get("total"), default=str))


if __name__ == "__main__":
    main()
