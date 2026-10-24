"""Results and the model shadow filter for the multi-sport research decisions.

Results (secondary to EV at the close, which stays the primary test): NBA player stats and team scores
from nbaprops.player_games (stats.nba.com), NCAAF scores from ncaaf.games (CFBD). A prop whose player
has no game log that day did not play and is void, as books void it. First-half markets are not graded
(no half scores). ROI is at the decision's own price, $1 flat.

Shadow filter (NBA props only): the model's probability for the decision's side, from the latest
nbaprops.projections run made before the decision. "Agrees" means the model also rates the side above
Pinnacle's no-vig probability. It is recorded beside each decision and never changes a decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nba_props.live import MARKET_STAT, side_probability
from nba_props.model import match_name
from nfl_pipeline.markets import normalize_name

UNGRADED_MARKETS = ("spread_h1", "total_h1")


def payout(price: int) -> float:
    return price / 100 if price > 0 else 100 / abs(price)


def grade(row: dict, actual: float | None, home_pts: float | None, away_pts: float | None) -> str | None:
    """'win' | 'loss' | 'push' | 'void', or None while the result is unknown."""
    market, side, line = row["market"], row["side"], float(row["line"])
    if market in UNGRADED_MARKETS:
        return None
    if market in MARKET_STAT:
        if actual is None:
            return None
        value = float(actual) - line
        signed = value if side == "over" else -value
    else:
        if home_pts is None or away_pts is None:
            return None
        if market == "h2h":
            signed = (home_pts - away_pts) * (1 if side == "home" else -1)
        elif market == "spread":  # line is the bet team's own line
            signed = (home_pts - away_pts if side == "home" else away_pts - home_pts) + line
        elif market == "total":
            signed = (home_pts + away_pts - line) * (1 if side == "over" else -1)
        elif market == "team_total":
            team = home_pts if normalize_name(row.get("player") or "") == normalize_name(row["home_team"]) else away_pts
            signed = (team - line) * (1 if side == "over" else -1)
        else:
            return None
    return "win" if signed > 0 else "loss" if signed < 0 else "push"


def profit(result: str | None, price: int) -> float | None:
    return {"win": payout(int(price)), "loss": -1.0, "push": 0.0}.get(result) if result else None


# --- loaders ---------------------------------------------------------------------------------------
def _table_exists(cur, name: str) -> bool:
    cur.execute("SELECT to_regclass(%s)", (name,))
    return cur.fetchone()[0] is not None


def nba_actuals(conn, days: list) -> tuple[dict, dict, set]:
    """{(day, name_key): stats}, {(day, team full name): points}, days with any game logs."""
    from nba_api.stats.static import teams
    full = {t["abbreviation"]: t["full_name"] for t in teams.get_teams()}
    with conn.cursor() as cur:
        if not days or not _table_exists(cur, "nbaprops.player_games"):
            return {}, {}, set()
        cur.execute("""SELECT game_date, player_name, team, pts, reb, ast, fg3m FROM nbaprops.player_games
                       WHERE game_date = ANY(%s)""", (list(days),))
        rows = cur.fetchall()
    players, scores, seen = {}, {}, set()
    for day, name, team, pts, reb, ast, fg3m in rows:
        seen.add(day)
        players[(day, match_name(name))] = dict(pts=pts, reb=reb, ast=ast, fg3m=fg3m)
        key = (day, full.get(team, team))
        scores[key] = scores.get(key, 0) + (pts or 0)
    return players, scores, seen


def ncaaf_scores(conn, days: list) -> list[tuple]:
    with conn.cursor() as cur:
        if not days or not _table_exists(cur, "ncaaf.games"):
            return []
        cur.execute("""SELECT (start_date AT TIME ZONE 'America/New_York')::date, home_team, away_team, home_points, away_points
                       FROM ncaaf.games WHERE completed AND (start_date AT TIME ZONE 'America/New_York')::date = ANY(%s)""",
                    (list(days),))
        return cur.fetchall()


def match_school(odds_name: str, schools: list[str]) -> str | None:
    """'Miami (OH) RedHawks' -> 'Miami (OH)': the longest CFBD school name the Odds API name starts with."""
    hits = [s for s in schools if odds_name == s or odds_name.startswith(s + " ")]
    return max(hits, key=len) if hits else None


def latest_projections(conn, days: list) -> pd.DataFrame:
    with conn.cursor() as cur:
        if not days or not _table_exists(cur, "nbaprops.projections"):
            return pd.DataFrame()
    return pd.read_sql("""SELECT run_at_utc, game_date, name_key, stat, mu, disp_a, disp_b FROM nbaprops.projections
                          WHERE game_date = ANY(%(days)s)""", conn, params=dict(days=list(days)))


def attach(conn, df: pd.DataFrame) -> pd.DataFrame:
    """Add result, profit, model_probability and model_agrees to the decisions frame."""
    if df.empty:
        return df
    df = df.copy()
    days = sorted(set(df.day))
    players, nba_scores, nba_days = nba_actuals(conn, days)
    cf = ncaaf_scores(conn, days)
    cf_by_day: dict = {}
    for day, home, away, hp, ap in cf:
        cf_by_day.setdefault(day, []).append((home, away, hp, ap))
    proj = latest_projections(conn, days)
    results, model_p = [], []
    for r in df.to_dict("records"):
        actual = home_pts = away_pts = None
        if r["sport"] == "nba":
            if r["market"] in MARKET_STAT:
                stats = players.get((r["day"], match_name(r.get("player") or "")))
                if stats is not None:
                    actual = stats[MARKET_STAT[r["market"]]]
            home_pts, away_pts = nba_scores.get((r["day"], r["home_team"])), nba_scores.get((r["day"], r["away_team"]))
            result = grade(r, actual, home_pts, away_pts)
            if result is None and r["market"] in MARKET_STAT and r["day"] in nba_days and home_pts is not None:
                result = "void"  # the game was played and logged, the player was not in it
        else:
            games = cf_by_day.get(r["day"], [])
            schools = [g[0] for g in games] + [g[1] for g in games]
            home, away = match_school(r["home_team"] or "", schools), match_school(r["away_team"] or "", schools)
            game = next((g for g in games if g[0] == home and g[1] == away), None)
            result = grade(r, None, game[2], game[3]) if game else None
        results.append(result)
        p = None
        if r["sport"] == "nba" and r["market"] in MARKET_STAT and not proj.empty:
            cand = proj[(proj.game_date == r["day"]) & (proj.name_key == match_name(r.get("player") or ""))
                        & (proj.stat == MARKET_STAT[r["market"]])
                        & (pd.to_datetime(proj.run_at_utc, utc=True) < pd.Timestamp(r["decided_at"]).tz_convert("UTC"))]
            if not cand.empty:
                p = side_probability(cand.sort_values("run_at_utc").iloc[-1].to_dict(), r["market"], r["side"], float(r["line"]))
        model_p.append(p)
    df["result"] = results
    df["profit"] = [profit(res, price) for res, price in zip(results, df.price)]
    df["model_probability"] = model_p
    df["model_agrees"] = [None if p is None else bool(p > float(f)) for p, f in zip(model_p, df.fair_probability)]
    return df


def record(df: pd.DataFrame) -> dict:
    if df.empty or "result" not in df:
        return dict(settled=0)
    settled = df[df.result.isin(["win", "loss", "push"])]
    wins, losses, pushes = (int((settled.result == k).sum()) for k in ("win", "loss", "push"))
    return dict(settled=int(len(settled)), wins=wins, losses=losses, pushes=pushes,
                voids=int((df.result == "void").sum()), pending=int(df.result.isna().sum()),
                roi=float(settled.profit.mean()) if len(settled) else None,
                units=float(settled.profit.sum()) if len(settled) else 0.0)


def shadow_split(df: pd.DataFrame) -> dict:
    """Decisions where the model agrees vs disagrees: EV at the close first, record second."""
    if df.empty or "model_agrees" not in df:
        return {}
    out = {}
    for label, part in (("agrees", df[df.model_agrees.eq(True)]), ("disagrees", df[df.model_agrees.eq(False)]),
                        ("no projection", df[df.model_agrees.isna()])):
        close = pd.to_numeric(part.ev_at_close, errors="coerce").dropna()
        out[label] = dict(decisions=int(len(part)), ev_at_close=float(close.mean()) if len(close) else None,
                          **record(part))
    return out
