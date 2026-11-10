"""Adapter: final player statistics from the existing pipelines' Postgres. Read-only.

NBA comes from nbaprops.player_games (stats.nba.com game logs), NFL from raw.nfl_player_gamelogs (official
nflverse box scores) with full names from raw.nfl_rosters. A game is only reported complete when the source
shows it finished: NBA when both teams' minutes add up to a whole game, NFL when the schedule row is final with
a score. Nothing is inferred for a game that is not complete.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from sports_sniper.normalize import player_key
from sports_sniper.providers import saved_quotes

EASTERN = ZoneInfo("America/New_York")
NBA_TEAMS = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}
NFL_TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF", "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB", "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
NBA_FULL_GAME_MINUTES = 239.5      # five players for 48 minutes is 240; overtime adds 25 a period
NBA_STATS = ("pts", "reb", "ast", "fg3m")
NFL_STATS = ("receiving_yards", "rushing_yards", "receptions", "passing_tds", "passing_yards")


@dataclass
class GameResult:
    complete: bool = False                      # the source shows the whole game finished
    reason: str = "no result for this game yet"
    stats: dict[str, dict[str, float]] = field(default_factory=dict)   # player key -> stat -> value
    ambiguous: set[str] = field(default_factory=set)                   # keys shared by two players in this game
    known: set[str] = field(default_factory=set)                       # keys of players known on either team


@dataclass(frozen=True)
class Game:
    sport: str
    event_id: str
    home: str
    away: str
    commence_time: datetime

    @property
    def day(self) -> date:
        return self.commence_time.astimezone(EASTERN).date()


def _collect(rows, stat_names) -> tuple[dict, set]:
    stats, ambiguous = {}, set()
    for name, *values in rows:
        key = player_key(name or "")
        if not key:
            continue
        if key in stats:
            ambiguous.add(key)
        stats[key] = {s: float(v or 0) for s, v in zip(stat_names, values)}
    return stats, ambiguous


def _nba(cur, game: Game) -> GameResult:
    home, away = NBA_TEAMS.get(game.home), NBA_TEAMS.get(game.away)
    if not home or not away:
        return GameResult(reason="team name not recognised")
    cur.execute("SELECT to_regclass('nbaprops.ingest_conflicts')")
    conflicted = []
    if cur.fetchone()[0] is not None:
        cur.execute("SELECT DISTINCT game_id FROM nbaprops.ingest_conflicts WHERE NOT resolved")
        conflicted = [r[0] for r in cur.fetchall()]
    cur.execute("""SELECT game_id, team, player_name, minutes, pts, reb, ast, fg3m FROM nbaprops.player_games
                   WHERE game_date = %s AND team = ANY(%s) AND opponent = ANY(%s)""", (game.day, [home, away], [home, away]))
    rows = cur.fetchall()
    if not rows:
        return GameResult()
    games = {r[0] for r in rows}
    if len(games) != 1 or games & set(conflicted):
        return GameResult(reason="the result records for this game conflict")
    minutes = {team: sum(float(r[3] or 0) for r in rows if r[1] == team) for team in (home, away)}
    if min(minutes.values()) < NBA_FULL_GAME_MINUTES:
        return GameResult(reason="the box score is not complete yet")
    stats, ambiguous = _collect([(r[2], *r[4:]) for r in rows], NBA_STATS)
    cur.execute("""SELECT DISTINCT player_name FROM nbaprops.player_games
                   WHERE team = ANY(%s) AND game_date BETWEEN %s AND %s""", ([home, away], game.day - timedelta(days=30), game.day))
    return GameResult(True, "final", stats, ambiguous, {player_key(r[0]) for r in cur.fetchall()})


def _nfl(cur, game: Game) -> GameResult:
    home, away = NFL_TEAMS.get(game.home), NFL_TEAMS.get(game.away)
    if not home or not away:
        return GameResult(reason="team name not recognised")
    cur.execute("""SELECT game_id, season, status, home_score, away_score FROM raw.nfl_games
                   WHERE game_date_et = %s AND home_team_abbr = %s AND away_team_abbr = %s""", (game.day, home, away))
    found = cur.fetchall()
    if len(found) != 1:
        return GameResult()
    game_id, season, status, home_score, away_score = found[0]
    if str(status).lower() != "final" or home_score is None or away_score is None:
        return GameResult(reason="the game is not final yet")
    cur.execute("""SELECT COALESCE(r.player_name, g.player_name), g.receiving_yards, g.rushing_yards, g.receptions,
                          g.passing_tds, g.passing_yards
                   FROM raw.nfl_player_gamelogs g
                   LEFT JOIN LATERAL (SELECT player_name FROM raw.nfl_rosters r WHERE r.player_id = g.player_id
                                      ORDER BY r.season DESC, r.week DESC NULLS LAST LIMIT 1) r ON TRUE
                   WHERE g.game_id = %s AND g.source LIKE 'nflverse%%'""", (game_id,))
    rows = cur.fetchall()
    if not rows:
        return GameResult(reason="official box score not loaded yet")
    stats, ambiguous = _collect(rows, NFL_STATS)
    cur.execute("SELECT DISTINCT player_name FROM raw.nfl_rosters WHERE season = %s AND team_abbr = ANY(%s)", (season, [home, away]))
    return GameResult(True, "final", stats, ambiguous, {player_key(r[0]) for r in cur.fetchall()})


def load(dsn: str | None, games: list[Game], connect=None) -> dict[str, GameResult]:
    """{event_id: GameResult} for the games asked about. Empty when the results database is not there."""
    dsn = dsn or saved_quotes.default_dsn()
    if dsn is None or not games:
        return {}
    if connect is None:
        import psycopg2
        connect = psycopg2.connect
    out: dict[str, GameResult] = {}
    conn = connect(dsn)
    try:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            for game in games:
                reader = {"nba": _nba, "nfl": _nfl}.get(game.sport)
                out[game.event_id] = reader(cur, game) if reader else GameResult(reason="no results source for this sport")
    finally:
        conn.close()
    return out
