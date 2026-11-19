"""What the model is shown for one slate of NFL games, using only what was known at the report time.

A slate is every game kicking off on one Eastern date. The report time is 11:00 AM Eastern that day (after the
7:30 AM Pacific daily run stores the day's prices), or 90 minutes before the first kickoff when that is earlier
(London games), so a Thursday, Sunday and Monday of the same week are three packets. For each game the packet holds:

    game        teams, kickoff, roof, surface, weather at kickoff (a forecast in a live run)
    teams       record, against-the-spread record, season and last-3 statistics, league ranks
    injuries    the latest report status of every listed player seen by the report time, with his usage
    lines       FanDuel and DraftKings spread and total: earliest stored and latest before the report time
    props       our stored FanDuel prop lines with the model's projection range (context: the model does not beat the line)

Nothing after the report time is read, except weather. Injury snapshots start 17 Sep 2026 (week 2), so earlier
packets have none.

    python -m nfl_ai_report.packets --season 2026 --weeks 1-11
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from nfl_ai_report import team_stats

EASTERN = ZoneInfo("America/New_York")
REPORT_TIME = time(11, 0)
LEAD = timedelta(minutes=90)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "nfl_ai_report" / "packets"
BOOKS = ("fanduel", "draftkings")
PROP_STATS = ("receiving_yards", "receptions", "rushing_yards", "passing_yards")
PROPS_PER_GAME = 24
QUIET = {"cleared", "active", "full", "none", ""}
ALIASES = {"LA": "LAR", "LAR": "LA", "WSH": "WAS", "WAS": "WSH", "JAC": "JAX", "JAX": "JAC"}


SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def name_key(name) -> str:
    """'A.J. Brown' and 'A.Brown' -> 'a brown'; 'Amon-Ra St. Brown' and 'A.St. Brown' -> 'a st brown'."""
    import re
    tokens = [t for t in re.split(r"[ .]+", str(name or "").lower()) if t]
    if not tokens:
        return ""
    rest = [re.sub(r"[^a-z]", "", t.replace("-", "")) for t in tokens[1:]]
    rest = [t for t in rest if len(t) > 1 and t not in SUFFIXES]
    return (tokens[0][:1] + " " + " ".join(rest)).strip()


def report_time(day: date, first_kickoff=None) -> datetime:
    at = datetime.combine(day, REPORT_TIME, EASTERN).astimezone(timezone.utc)
    return at if first_kickoff is None else min(at, (pd.Timestamp(first_kickoff) - LEAD).to_pydatetime())


def _connect():
    import psycopg2
    from nfl_pipeline.db import PG_DSN
    conn = psycopg2.connect(PG_DSN)
    conn.set_session(readonly=True)
    return conn


def _frame(conn, sql: str, params=()) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])


def season_games(conn, season: int) -> pd.DataFrame:
    g = _frame(conn, """SELECT game_id, season, week, game_date_et, start_ts_utc, home_team_abbr AS home, away_team_abbr AS away, status,
                               home_score, away_score, spread_line, total_line, roof, surface, temp, wind
                        FROM raw.nfl_games WHERE season = %s AND season_type = 'REG' ORDER BY start_ts_utc""", (season,))
    g["start_ts_utc"] = pd.to_datetime(g.start_ts_utc, utc=True)
    g["slate"] = g.start_ts_utc.dt.tz_convert(EASTERN).dt.date
    for c in ("home_score", "away_score", "spread_line", "total_line", "temp", "wind"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    return g


def record(games: pd.DataFrame, team: str, cutoff: datetime) -> dict:
    """Straight-up and against-the-spread record in finished games before the cutoff (nflverse closing spread)."""
    done = games[(games.start_ts_utc < cutoff) & games.home_score.notna() & ((games.home == team) | (games.away == team))]
    w = l = t = aw = al = ap = 0
    margins = []
    for g in done.itertuples():
        home = g.home == team
        margin = (g.home_score - g.away_score) * (1 if home else -1)
        margin = margin if margin else 0.0                 # no '-0' for a tie
        margins.append(margin)
        w, l, t = w + (margin > 0), l + (margin < 0), t + (margin == 0)
        if pd.notna(g.spread_line):                       # nflverse spread_line: points the home team is favoured by
            cover = (g.home_score - g.away_score - g.spread_line) * (1 if home else -1)
            aw, al, ap = aw + (cover > 0), al + (cover < 0), ap + (cover == 0)
    return {"record": f"{w}-{l}" + (f"-{t}" if t else ""), "ats": f"{aw}-{al}" + (f"-{ap}" if ap else ""),
            "point_diff_per_game": round(sum(margins) / len(margins), 1) if margins else None,
            "last_results": [f"{'W' if m > 0 else 'L' if m < 0 else 'T'} {m:+.0f}" for m in margins[-3:]]}


def injuries(conn, season: int, week: int, teams: list[str], cutoff: datetime) -> pd.DataFrame:
    rows = _frame(conn, """
        SELECT DISTINCT ON (payload->>'team_abbr', payload->>'player_name_norm')
               payload->>'team_abbr' AS team, payload->>'player_name' AS player, payload->>'player_name_norm' AS player_norm,
               payload->>'position' AS position, payload->>'report_status' AS status, payload->>'practice_status' AS practice,
               payload->>'report_primary_injury' AS injury, observed_at
        FROM raw.nfl_context_observations
        WHERE kind = 'nfl_injuries' AND observed_at <= %s AND (payload->>'season')::int = %s AND (payload->>'week')::int = %s
              AND payload->>'team_abbr' = ANY(%s)
        ORDER BY payload->>'team_abbr', payload->>'player_name_norm', observed_at DESC""", (cutoff, season, week, teams))
    if rows.empty:
        return rows
    return rows[~rows.status.fillna("").str.lower().isin(QUIET)].reset_index(drop=True)


def usage(conn, season: int, cutoff_day: date) -> pd.DataFrame:
    """Each player's season usage before the slate: games, snap share, targets and carries per game."""
    return _frame(conn, """
        SELECT player_name, team_abbr AS team, count(*) AS games,
               round(avg(offense_snap_share)::numeric, 2) AS snap_share, round(avg(targets)::numeric, 1) AS targets,
               round(avg(carries)::numeric, 1) AS carries, round(avg(receiving_yards + rushing_yards)::numeric, 1) AS scrimmage_yards
        FROM raw.nfl_player_gamelogs WHERE season = %s AND game_date_et < %s GROUP BY 1, 2""", (season, cutoff_day)).assign(
        player_key=lambda f: f.player_name.map(name_key)).drop(columns="player_name").drop_duplicates(["player_key", "team"])


def lines(conn, games: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    """Every stored FanDuel/DraftKings spread and total before the cutoff for the slate's games."""
    start, end = games.start_ts_utc.min() - timedelta(hours=12), games.start_ts_utc.max() + timedelta(hours=12)
    return _frame(conn, """
        SELECT fetched_at_utc, commence_time_utc, bookmaker_key AS book, home_team_abbr AS home, away_team_abbr AS away,
               spread_home_points, spread_home_price, spread_away_price, total_points, total_over_price, total_under_price
        FROM odds.nfl_game_lines
        WHERE fetched_at_utc <= %s AND commence_time_utc BETWEEN %s AND %s AND bookmaker_key = ANY(%s)
              AND spread_home_points IS NOT NULL ORDER BY fetched_at_utc""", (cutoff, start, end, list(BOOKS)))


def _line(row) -> dict:
    def num(v):
        return None if v is None or pd.isna(v) else float(v)

    def price(v):
        return None if v is None or pd.isna(v) else int(v)
    return {"seen": pd.Timestamp(row.fetched_at_utc).tz_convert(EASTERN).strftime("%a %H:%M ET"),
            "home_spread": num(row.spread_home_points), "home_price": price(row.spread_home_price), "away_price": price(row.spread_away_price),
            "total": num(row.total_points), "over_price": price(row.total_over_price), "under_price": price(row.total_under_price)}


def game_lines(all_lines: pd.DataFrame, game) -> dict:
    if all_lines.empty:
        return {}
    homes = {game.home, ALIASES.get(game.home, game.home)}
    near = (pd.to_datetime(all_lines.commence_time_utc, utc=True) - game.start_ts_utc).abs() <= timedelta(hours=6)
    mine = all_lines[near & all_lines.home.isin(homes)]
    out = {}
    for book, rows in mine.groupby("book"):
        out[book] = {"earliest": _line(rows.iloc[0]), "latest": _line(rows.iloc[-1]), "snapshots": len(rows)}
    return out


def props(conn, game_ids: list[str], cutoff: datetime) -> pd.DataFrame:
    p = _frame(conn, """
        SELECT DISTINCT ON (game_id, player_id, stat)
               game_id, player_name, team_abbr AS team, position, stat, line, side, price, baseline_projection,
               market_no_vig_probability AS book_no_vig, injury_report_status, created_at_utc
        FROM bets.nfl_player_prop_predictions
        WHERE game_id = ANY(%s) AND book = 'fanduel' AND stat = ANY(%s) AND created_at_utc <= %s AND line IS NOT NULL
        ORDER BY game_id, player_id, stat, created_at_utc DESC""", (game_ids, list(PROP_STATS), cutoff))
    for c in ("line", "baseline_projection", "book_no_vig"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    return p


def _prop_rows(p: pd.DataFrame) -> list[dict]:
    if p.empty:
        return []
    p = p.sort_values(["stat", "line"], ascending=[True, False])
    keep = p.groupby("stat").head(PROPS_PER_GAME // len(PROP_STATS))
    return [{"player": r.player_name, "team": r.team, "pos": r.position, "stat": r.stat, "line": float(r.line), "side_priced": r.side,
             "price": None if pd.isna(r.price) else int(r.price), "book_no_vig_chance": None if pd.isna(r.book_no_vig) else round(float(r.book_no_vig), 3),
             "model_projection": None if pd.isna(r.baseline_projection) else round(float(r.baseline_projection), 1)}
            for r in keep.itertuples()]


def _injury_rows(inj: pd.DataFrame, use: pd.DataFrame, team: str) -> list[dict]:
    if inj.empty:
        return []
    mine = inj[inj.team == team].copy()
    mine["player_key"] = mine.player.map(name_key)
    mine = mine.merge(use[use.team == team].drop(columns="team"), on="player_key", how="left")
    order = {"out": 0, "injured reserve": 1, "doubtful": 2, "questionable": 3}
    mine["_o"] = mine.status.fillna("").str.lower().map(order).fillna(4)
    mine = mine.sort_values(["_o", "snap_share"], ascending=[True, False])
    rows = []
    for r in mine.itertuples():
        row = {"player": r.player, "pos": r.position, "status": r.status, "injury": r.injury}
        if pd.notna(r.games) and pd.notna(r.snap_share):
            row["usage"] = {k: float(getattr(r, k)) for k in ("snap_share", "targets", "carries", "scrimmage_yards") if pd.notna(getattr(r, k))}
        rows.append(row)
    return rows


def build_slate(conn, games: pd.DataFrame, slate: date, rows: pd.DataFrame, ranks_cache: dict | None = None) -> dict:
    """The packet for every game kicking off on the slate date."""
    todays = games[games.slate == slate].sort_values("start_ts_utc")
    cutoff = report_time(slate, todays.start_ts_utc.iloc[0])
    season, week = int(todays.season.iloc[0]), int(todays.week.iloc[0])
    teams = sorted(set(todays.home) | set(todays.away))
    inj = injuries(conn, season, week, teams, cutoff)
    use = usage(conn, season, slate)
    all_lines = lines(conn, todays, cutoff)
    all_props = props(conn, todays.game_id.tolist(), cutoff)
    ranks = team_stats.league_ranks(rows, slate)
    packet = {"season": season, "week": week, "slate": slate.isoformat(), "report_time_utc": cutoff.isoformat(),
              "injury_snapshots": not inj.empty, "games": []}
    for g in todays.itertuples():
        surface = None if pd.isna(g.surface) or not g.surface else g.surface
        game = {"game_id": g.game_id, "matchup": f"{g.away} @ {g.home}",
                "kickoff_et": g.start_ts_utc.tz_convert(EASTERN).strftime("%a %d %b %I:%M %p"),
                "venue": {"roof": g.roof, "surface": surface,
                          "weather_at_kickoff": None if g.roof in ("dome", "closed") or pd.isna(g.temp) else
                          {"temp_f": float(g.temp), "wind_mph": None if pd.isna(g.wind) else float(g.wind)}},
                "lines": game_lines(all_lines, g), "teams": {}}
        for side, team in (("away", g.away), ("home", g.home)):
            stat_team = ALIASES.get(team, team) if team not in set(rows.team) else team
            game["teams"][side] = {"team": team, **record(games, team, cutoff),
                                   "season": team_stats.before(rows, stat_team, slate), "last_3": team_stats.before(rows, stat_team, slate, last=3),
                                   "league_rank": ranks.get(stat_team, {}), "injuries": _injury_rows(inj, use, team)}
        game["props_fanduel"] = _prop_rows(all_props[all_props.game_id == g.game_id])
        packet["games"].append(game)
    return packet


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1 else f"{v:g}"
    return str(v)


STAT_LABELS = [("yards_per_game", "Yards/game", "yards_allowed_per_game"), ("yards_per_play", "Yards/play", "yards_per_play_allowed"),
               ("rush_yards_per_game", "Rush yards/game", "rush_yards_allowed_per_game"),
               ("rush_yards_per_carry", "Yards/carry", "rush_yards_per_carry_allowed"),
               ("pass_yards_per_game", "Pass yards/game", "pass_yards_allowed_per_game"), ("epa_per_play", "EPA/play", "epa_per_play_allowed"),
               ("success_rate", "Success rate", "success_rate_allowed"), ("third_down_pct", "3rd-down %", "third_down_pct_allowed"),
               ("red_zone_td_pct", "Red-zone TD %", "red_zone_td_pct_allowed"), ("sacks_taken_per_game", "Sacks taken / made per game", "sacks_per_game")]


RANK_LABELS = [("epa_per_play", "offensive EPA/play"), ("epa_per_play_allowed", "defensive EPA/play"),
               ("success_rate", "offensive success rate"), ("rush_yards_per_game", "rush yards/game"),
               ("rush_yards_allowed_per_game", "rush yards allowed/game"), ("third_down_pct", "3rd-down %"),
               ("third_down_pct_allowed", "3rd-down % allowed"), ("red_zone_td_pct", "red-zone TD %"), ("turnover_margin", "turnover margin"),
               ("penalty_yards_per_game", "fewest penalty yards"), ("net_punt_avg", "net punting")]


def to_text(packet: dict) -> str:
    """The packet as compact Markdown for the model."""
    out = [f"# NFL {packet['season']} week {packet['week']}, games on {packet['slate']}",
           f"Report time: {pd.Timestamp(packet['report_time_utc']).tz_convert(EASTERN):%I:%M %p} ET. Only information available by then is included (weather is at kickoff)."
           + ("" if packet["injury_snapshots"] else " No injury report snapshot is stored for this date."), ""]
    for g in packet["games"]:
        a, h = g["teams"]["away"], g["teams"]["home"]
        v = g["venue"]
        weather = v["weather_at_kickoff"]
        out += [f"## {g['matchup']} ({g['kickoff_et']} ET)", f"Venue: {v['roof']}" + (f", {v['surface']}" if v['surface'] else "") +
                (f"; {weather['temp_f']:.0f}F, wind {_fmt(weather['wind_mph'])} mph" if weather else ""), ""]
        if g["lines"]:
            out += ["| Book | Home spread (price) | Total (over/under) | Earliest seen | Latest seen |", "|---|---|---|---|---|"]
            for book, b in g["lines"].items():
                e, l = b["earliest"], b["latest"]
                out.append(f"| {book} | {l['home_spread']:+g} ({_fmt(l['home_price'])}) was {e['home_spread']:+g} | {_fmt(l['total'])} "
                           f"({_fmt(l['over_price'])}/{_fmt(l['under_price'])}) was {_fmt(e['total'])} | {e['seen']} | {l['seen']} |")
        else:
            out.append("No stored game line before the report time.")
        out += ["", f"| | {a['team']} (away) | {h['team']} (home) |", "|---|---|---|",
                f"| Record / against the spread | {a['record']} / {a['ats']} | {h['record']} / {h['ats']} |",
                f"| Point differential per game | {_fmt(a['point_diff_per_game'])} | {_fmt(h['point_diff_per_game'])} |",
                f"| Last results | {', '.join(a['last_results']) or '-'} | {', '.join(h['last_results']) or '-'} |"]
        for key, label, allowed in STAT_LABELS:
            for scope, name in (("season", ""), ("last_3", " (last 3)")):
                sa, sh = a[scope], h[scope]
                if sa.get("games") and sh.get("games"):
                    out.append(f"| {label}{name}: offense / defense | {_fmt(sa.get(key))} / {_fmt(sa.get(allowed))} | "
                               f"{_fmt(sh.get(key))} / {_fmt(sh.get(allowed))} |")
        for key, label in (("turnover_margin", "Turnover margin (giveaways/takeaways)"), ("penalty_yards_per_game", "Penalty yards per game"),
                           ("net_punt_avg", "Net punt average"), ("time_of_possession_per_game", "Time of possession per game")):
            sa, sh = a["season"], h["season"]
            extra = (lambda s: f" ({s.get('giveaways')}/{s.get('takeaways')})") if key == "turnover_margin" else (lambda s: "")
            out.append(f"| {label} | {_fmt(sa.get(key))}{extra(sa)} | {_fmt(sh.get(key))}{extra(sh)} |")
        for key, label in RANK_LABELS:
            if key in a["league_rank"] or key in h["league_rank"]:
                out.append(f"| Rank: {label} (1 = best of 32) | {_fmt(a['league_rank'].get(key))} | {_fmt(h['league_rank'].get(key))} |")
        for side in (a, h):
            if side["injuries"]:
                out.append(f"\nInjury report, {side['team']}: " + "; ".join(
                    f"{i['player']} ({i['pos']}) {i['status']}" + (f", {i['injury']}" if i.get("injury") else "") +
                    (f" [snap share {_fmt(i['usage'].get('snap_share'))}, {_fmt(i['usage'].get('targets'))} targets, "
                     f"{_fmt(i['usage'].get('carries'))} carries per game]" if i.get("usage") else "") for i in side["injuries"]))
        if g["props_fanduel"]:
            out += ["", "FanDuel player props with our model's own projection (tested: it has not beaten these lines):", "",
                    "| Player | Stat | Line | Priced side | Book no-vig chance of that side | Model projection |", "|---|---|---|---|---|---|"]
            out += [f"| {p['player']} ({p['team']} {p['pos']}) | {p['stat']} | {p['line']:g} | {p['side_priced']} {_fmt(p['price'])} | "
                    f"{_fmt(p['book_no_vig_chance'])} | {_fmt(p['model_projection'])} |" for p in g["props_fanduel"]]
        out.append("")
    return "\n".join(out)


def build(season: int, weeks: list[int], out: Path = OUT) -> list[Path]:
    pbp_rows = team_stats.game_rows(team_stats.load_pbp(season))
    conn = _connect()
    written = []
    try:
        games = season_games(conn, season)
        for slate in sorted(games[games.week.isin(weeks)].slate.unique()):
            packet = build_slate(conn, games, slate, pbp_rows)
            out.mkdir(parents=True, exist_ok=True)
            base = out / f"{season}_w{packet['week']:02d}_{slate.isoformat()}"
            base.with_suffix(".json").write_text(json.dumps(packet, indent=1, default=str), encoding="utf-8")
            base.with_suffix(".md").write_text(to_text(packet), encoding="utf-8")
            written.append(base.with_suffix(".md"))
            print(f"{base.name}: {len(packet['games'])} games, {sum(bool(g['lines']) for g in packet['games'])} with lines, "
                  f"{sum(len(g['props_fanduel']) for g in packet['games'])} props, injuries {packet['injury_snapshots']}")
    finally:
        conn.close()
    return written


def _weeks(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--weeks", default="1-11")
    args = ap.parse_args()
    build(args.season, _weeks(args.weeks))


if __name__ == "__main__":
    main()
