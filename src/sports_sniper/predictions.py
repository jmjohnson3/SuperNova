"""The forecasts the other pipelines already store, read for the dashboard beside the latest line.

NCAAF spreads and totals (`ncaaf.projections`), NBA spreads and totals (`nbaprops.game_projections`) and player
props (`nbaprops.projections`), NFL spreads, totals and player props (`bets.nfl_*_predictions`). Read-only: nothing
here trains, writes to those tables, spends API credits or changes what the scanner flags.

Sign convention everywhere: margin is home minus away, spread is the home team's line (-3.5 = home favoured by
3.5), so `margin + spread > 0` means the model has the home team covering and `total gap > 0` leans over.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from sports_sniper.normalize import player_key
from sports_sniper.providers import saved_quotes
from sports_sniper.providers.models import EASTERN, NBA_STAT, _connect, _utc

BOOK_ORDER = ("fanduel", "draftkings", "espnbet", "pinnacle")       # whose line to show first
STARTED_HOURS = 4            # keep a game on the page this long after it starts
LINE_HOURS = 36              # ignore stored lines older than this
GAME_COLUMNS = ["sport", "start", "away", "home", "pred_margin", "pred_total", "spread", "total", "book", "made_at", "note", "model"]
PROP_COLUMNS = ["sport", "start", "game", "player", "team", "stat", "projection", "raw", "low", "high", "line", "book",
                "over_price", "under_price", "model_over", "market_over", "made_at", "model"]
NBA_STAT_LABEL = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "Threes"}
NBA_MARKET = {stat: market for market, stat in NBA_STAT.items()}


def _frame(dsn, table: str, sql: str, params: tuple, connect=None) -> pd.DataFrame:
    """One read-only query as a frame; empty when the database or the table is not there."""
    dsn = dsn or saved_quotes.default_dsn()
    if dsn is None:
        return pd.DataFrame()
    try:
        conn = (connect or _connect)(dsn)
    except Exception:
        return pd.DataFrame()
    try:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (table,))
            if cur.fetchone()[0] is None:
                return pd.DataFrame()
            cur.execute(sql, params)
            return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
    finally:
        conn.close()


def _number(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _nickname(team: str) -> str:
    return str(team).lower().split()[-1] if team else ""


# --- stored game lines (NBA) ----------------------------------------------------------------------------------
def lines_from_event(event: dict) -> dict | None:
    """The home spread and the total from one saved Odds API event, from the first book in BOOK_ORDER that has them."""
    books = {b.get("key"): b for b in event.get("bookmakers") or []}
    out = dict(home=event.get("home_team"), away=event.get("away_team"), spread=None, total=None, book=None)
    for key in list(BOOK_ORDER) + [k for k in books if k not in BOOK_ORDER]:
        markets = {m.get("key"): m for m in (books.get(key) or {}).get("markets") or []}
        if out["spread"] is None and "spreads" in markets:
            for o in markets["spreads"].get("outcomes") or []:
                if o.get("name") == out["home"] and o.get("point") is not None:
                    out["spread"], out["book"] = float(o["point"]), out["book"] or key
        if out["total"] is None and "totals" in markets:
            for o in markets["totals"].get("outcomes") or []:
                if o.get("name") == "Over" and o.get("point") is not None:
                    out["total"], out["book"] = float(o["point"]), out["book"] or key
    return out if out["spread"] is not None or out["total"] is not None else None


def game_lines(dsn, sport: str, now: datetime, connect=None) -> dict:
    """Latest stored spread and total for each coming game, keyed by (home nickname, away nickname)."""
    rows = _frame(dsn, "bets.sharp_multi_quotes",
                  """SELECT DISTINCT ON (event_id) observed_at_utc, payload FROM bets.sharp_multi_quotes
                     WHERE sport = %s AND commence_time_utc > %s AND observed_at_utc > %s
                       AND payload->'bookmakers' @> '[{"markets": [{"key": "spreads"}]}]'
                     ORDER BY event_id, observed_at_utc DESC""",
                  (sport, now - timedelta(hours=STARTED_HOURS), now - timedelta(hours=LINE_HOURS)), connect)
    out = {}
    for row in rows.itertuples():
        found = lines_from_event(row.payload if isinstance(row.payload, dict) else {})
        if found:
            out[(_nickname(found["home"]), _nickname(found["away"]))] = dict(found, seen=_utc(row.observed_at_utc))
    return out


def nba_team_names() -> dict:
    try:
        from nba_api.stats.static import teams
        return {t["abbreviation"]: t["full_name"] for t in teams.get_teams()}
    except Exception:                       # the NBA pipeline is optional
        return {}


# --- games ----------------------------------------------------------------------------------------------------
def ncaaf_games(dsn, now: datetime, connect=None) -> pd.DataFrame:
    rows = _frame(dsn, "ncaaf.projections",
                  """SELECT DISTINCT ON (game_id) start_date, away_team, home_team, pred_margin, pred_total, market_provider,
                            market_spread, market_total, generated_at_utc
                     FROM ncaaf.projections WHERE start_date > %s ORDER BY game_id, generated_at_utc DESC""",
                  (now - timedelta(hours=STARTED_HOURS),), connect)
    return pd.DataFrame([dict(sport="ncaaf", start=_utc(r.start_date), away=r.away_team, home=r.home_team,
                              pred_margin=_number(r.pred_margin), pred_total=_number(r.pred_total), spread=_number(r.market_spread),
                              total=_number(r.market_total), book=(r.market_provider or "").lower().replace(" ", "") or None,
                              made_at=_utc(r.generated_at_utc), note="", model="Opponent-adjusted ratings") for r in rows.itertuples()],
                        columns=GAME_COLUMNS)


def nba_games(dsn, now: datetime, connect=None, names: dict | None = None) -> pd.DataFrame:
    rows = _frame(dsn, "nbaprops.game_projections",
                  """SELECT DISTINCT ON (game_id) tip_utc, away_team, home_team, pred_margin, pred_total, home_out, away_out,
                            run_at_utc, model_version
                     FROM nbaprops.game_projections WHERE game_date >= %s AND tip_utc > %s ORDER BY game_id, run_at_utc DESC""",
                  ((now - timedelta(days=1)).date(), now - timedelta(hours=STARTED_HOURS)), connect)
    if rows.empty:
        return pd.DataFrame(columns=GAME_COLUMNS)
    lines = game_lines(dsn, "nba", now, connect)
    names = nba_team_names() if names is None else names
    out = []
    for r in rows.itertuples():
        home, away = names.get(r.home_team, r.home_team), names.get(r.away_team, r.away_team)
        line = lines.get((_nickname(home), _nickname(away)), {})
        missing = "; ".join(f"{team} out: {who}" for team, who in ((r.home_team, r.home_out), (r.away_team, r.away_out)) if who)
        out.append(dict(sport="nba", start=_utc(r.tip_utc), away=away, home=home, pred_margin=_number(r.pred_margin),
                        pred_total=_number(r.pred_total), spread=line.get("spread"), total=line.get("total"), book=line.get("book"),
                        made_at=_utc(r.run_at_utc), note=missing, model=r.model_version))
    return pd.DataFrame(out, columns=GAME_COLUMNS)


def nfl_games(dsn, now: datetime, connect=None) -> pd.DataFrame:
    rows = _frame(dsn, "bets.nfl_game_predictions",
                  """SELECT game_id, away_team_abbr, home_team_abbr, market, side, book, line, predicted_home_margin,
                            predicted_total_points, forecast_payload->>'start_ts_utc' AS start, updated_at_utc, model_version
                     FROM bets.nfl_game_predictions WHERE is_current AND game_date_et >= %s ORDER BY game_id, updated_at_utc""",
                  ((now - timedelta(days=1)).astimezone(EASTERN).date(),), connect)
    games: dict[str, dict] = {}
    for r in rows.itertuples():
        if not r.start or _utc(r.start) <= now - timedelta(hours=STARTED_HOURS):
            continue
        g = games.setdefault(r.game_id, dict(sport="nfl", start=_utc(r.start), away=r.away_team_abbr, home=r.home_team_abbr,
                                             spread=None, total=None, book=r.book, note="", model=r.model_version))
        g.update(pred_margin=_number(r.predicted_home_margin), pred_total=_number(r.predicted_total_points), made_at=_utc(r.updated_at_utc))
        if r.market == "spread" and r.line is not None:
            g["spread"] = float(r.line) if r.side == "home" else -float(r.line)
        elif r.market == "total" and r.line is not None:
            g["total"] = float(r.line)
    return pd.DataFrame(list(games.values()), columns=GAME_COLUMNS)


def with_gaps(games: pd.DataFrame) -> pd.DataFrame:
    """Adds the model's gap to the line: spread_gap > 0 has the home team covering, total_gap > 0 leans over."""
    games = games.copy()
    for column in ("pred_margin", "pred_total", "spread", "total"):
        games[column] = pd.to_numeric(games[column], errors="coerce")
    games["spread_gap"] = games.pred_margin + games.spread
    games["total_gap"] = games.pred_total - games.total
    return games


def games(dsn, now: datetime, connect=None, names: dict | None = None) -> pd.DataFrame:
    parts = [ncaaf_games(dsn, now, connect), nba_games(dsn, now, connect, names), nfl_games(dsn, now, connect)]
    parts = [p for p in parts if len(p)]
    found = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=GAME_COLUMNS)
    return with_gaps(found).sort_values(["start", "home"], ignore_index=True)


# --- player props ---------------------------------------------------------------------------------------------
def no_vig_over(over_price, under_price) -> float | None:
    from sports_sniper import odds_math
    if over_price is None or under_price is None or pd.isna(over_price) or pd.isna(under_price):
        return None
    over, under = 1 / odds_math.american_to_decimal(int(over_price)), 1 / odds_math.american_to_decimal(int(under_price))
    return over / (over + under)


def prop_lines(prices: pd.DataFrame) -> dict:
    """The newest line for each (player key, market) from stored prices, from the first book in BOOK_ORDER."""
    if prices is None or prices.empty:
        return {}
    rank = {book: n for n, book in enumerate(BOOK_ORDER)}
    prices = prices.assign(rank=prices.book.map(lambda b: rank.get(b, len(rank))))
    out = {}
    for (key, market), group in prices.groupby(["player_key", "market"]):
        book = group.sort_values(["rank", "observed_at"], ascending=[True, False]).iloc[0].book
        latest = group[group.book == book]
        latest = latest[latest.observed_at == latest.observed_at.max()]
        line = latest.line.iloc[0]
        sides = {s.side: s.price for s in latest[latest.line == line].itertuples()}
        out[(key, market)] = dict(line=float(line), book=book, over_price=sides.get("over"), under_price=sides.get("under"))
    return out


def nba_props(dsn, now: datetime, prices: pd.DataFrame | None = None, connect=None, names: dict | None = None) -> pd.DataFrame:
    rows = _frame(dsn, "nbaprops.projections",
                  """SELECT DISTINCT ON (game_id, player_id, stat) tip_utc, player_name, team, opponent, stat, mu, disp_a, disp_b,
                            run_at_utc, model_version
                     FROM nbaprops.projections WHERE game_date >= %s AND tip_utc > %s
                     ORDER BY game_id, player_id, stat, run_at_utc DESC""", ((now - timedelta(days=1)).date(), now), connect)
    if rows.empty:
        return pd.DataFrame(columns=PROP_COLUMNS)
    try:
        import numpy as np
        from nba_props.model import p_over
    except Exception:                       # the NBA pipeline is optional
        p_over = None
    lines = prop_lines(prices)
    out = []
    for r in rows.itertuples():
        mu, a, b = float(r.mu), float(r.disp_a), float(r.disp_b)
        spread = max(a * mu + b * mu * mu, 0.0) ** 0.5
        line = lines.get((player_key(r.player_name), NBA_MARKET.get(r.stat)), {})
        chance = None
        if p_over is not None and line:
            chance = float(p_over(np.array([mu]), np.array([line["line"]]), a, b)[0])
        out.append(dict(sport="nba", start=_utc(r.tip_utc), game=f"{r.team} vs {r.opponent}", player=r.player_name, team=r.team,
                        stat=NBA_STAT_LABEL.get(r.stat, r.stat), projection=mu, raw=None, low=max(mu - 1.2816 * spread, 0.0),
                        high=mu + 1.2816 * spread, line=line.get("line"), book=line.get("book"), over_price=line.get("over_price"),
                        under_price=line.get("under_price"), model_over=chance,
                        market_over=no_vig_over(line.get("over_price"), line.get("under_price")), made_at=_utc(r.run_at_utc),
                        model=r.model_version))
    return pd.DataFrame(out, columns=PROP_COLUMNS)


def nfl_props(dsn, now: datetime, connect=None) -> pd.DataFrame:
    rows = _frame(dsn, "bets.nfl_player_prop_predictions",
                  """SELECT COALESCE(offer_player_name, player_name) AS player, team_abbr, opponent_abbr, stat, projection,
                            forecast_payload->>'stat_projection_expected_mean' AS raw, projection_p10, projection_p90, line, side,
                            price, book, raw_over_probability, market_no_vig_probability,
                            forecast_payload->>'start_ts_utc' AS start, updated_at_utc, model_version
                     FROM bets.nfl_player_prop_predictions WHERE is_current AND game_date_et >= %s""",
                  ((now - timedelta(days=1)).astimezone(EASTERN).date(),), connect)
    out = []
    for r in rows.itertuples():
        if not r.start or _utc(r.start) <= now:
            continue
        over = r.side == "over" or r.side is None
        market = _number(r.market_no_vig_probability)
        out.append(dict(sport="nfl", start=_utc(r.start), game=f"{r.team_abbr} vs {r.opponent_abbr}", player=r.player, team=r.team_abbr,
                        stat=str(r.stat).replace("_", " ").capitalize().replace("tds", "TDs"), projection=_number(r.projection),
                        raw=_number(r.raw), low=_number(r.projection_p10), high=_number(r.projection_p90), line=_number(r.line),
                        book=r.book, over_price=_number(r.price) if over else None, under_price=None if over else _number(r.price),
                        model_over=_number(r.raw_over_probability),
                        market_over=market if market is None or over else 1 - market, made_at=_utc(r.updated_at_utc), model=r.model_version))
    return pd.DataFrame(out, columns=PROP_COLUMNS)


def props(dsn, now: datetime, prices: pd.DataFrame | None = None, connect=None) -> pd.DataFrame:
    parts = [nba_props(dsn, now, prices, connect), nfl_props(dsn, now, connect)]
    parts = [p for p in parts if len(p)]
    found = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PROP_COLUMNS)
    for column in ("projection", "raw", "low", "high", "line", "model_over", "market_over"):
        found[column] = pd.to_numeric(found[column], errors="coerce")
    found["gap"] = found.projection - found.line
    return found.sort_values(["start", "game", "player", "stat"], ignore_index=True)


# --- how the game forecasts have done --------------------------------------------------------------------------
def score_games(done: pd.DataFrame) -> dict | None:
    """Average miss of the model and of the line on finished games, and the record of the model's side of the line.
    `done` needs pred_margin, pred_total, spread, total, home_points, away_points."""
    done = done.dropna(subset=["pred_margin", "pred_total", "spread", "total", "home_points", "away_points"]).astype(
        {c: float for c in ("pred_margin", "pred_total", "spread", "total", "home_points", "away_points")}) if len(done) else done
    if done.empty:
        return None
    margin, total = done.home_points - done.away_points, done.home_points + done.away_points

    def record(lean, result):
        played = (lean != 0) & (result != 0)
        won = int(((lean > 0) == (result > 0))[played].sum())
        return won, int(played.sum()) - won

    return dict(games=len(done), margin_model=float((done.pred_margin - margin).abs().mean()),
                margin_line=float((-done.spread - margin).abs().mean()), total_model=float((done.pred_total - total).abs().mean()),
                total_line=float((done.total - total).abs().mean()), spread_record=record(done.pred_margin + done.spread, margin + done.spread),
                total_record=record(done.pred_total - done.total, total - done.total))


def ncaaf_record(dsn, connect=None) -> dict | None:
    done = _frame(dsn, "ncaaf.projections",
                  """SELECT p.pred_margin, p.pred_total, p.market_spread AS spread, p.market_total AS total, g.home_points, g.away_points
                     FROM (SELECT DISTINCT ON (game_id) * FROM ncaaf.projections ORDER BY game_id, generated_at_utc DESC) p
                     JOIN ncaaf.games g USING (game_id) WHERE g.completed AND g.home_points IS NOT NULL""", (), connect)
    return score_games(done)
