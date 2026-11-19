"""Team statistics per game from nflverse play-by-play, and season and recent-form summaries before a date.

One row per team and game holds what the team did on offense and what it allowed on defense. Summaries are
per-game averages (or rates) over the games played before the cutoff date, so a packet never sees the game it
is about.
"""
from __future__ import annotations

import urllib.request
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "nfl_ai_report" / "cache"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv"
COLUMNS = ["game_id", "season_type", "week", "game_date", "home_team", "away_team", "posteam", "defteam", "play_type", "yards_gained",
           "rush_attempt", "pass_attempt", "sack", "qb_kneel", "qb_spike", "epa", "success", "third_down_converted", "third_down_failed",
           "interception", "fumble_lost", "penalty", "penalty_team", "penalty_yards", "punt_attempt", "kick_distance", "return_yards",
           "touchback", "fixed_drive", "fixed_drive_result", "drive_inside20", "drive_time_of_possession"]


def load_pbp(season: int, refresh: bool = False) -> pd.DataFrame:
    """The season's play-by-play, downloaded once into the cache (the same file the Tuesday import reads)."""
    path = CACHE / f"play_by_play_{season}.csv"
    if refresh or not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        with urllib.request.urlopen(PBP_URL.format(season=season), timeout=600) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f)
        part.replace(path)
    return pd.read_csv(path, usecols=COLUMNS, low_memory=False)


def _seconds(clock) -> float:
    if not isinstance(clock, str) or ":" not in clock:
        return 0.0
    minutes, seconds = clock.split(":")
    return int(minutes) * 60 + int(seconds)


def _side(plays: pd.DataFrame, drives: pd.DataFrame, punts: pd.DataFrame, team_col: str) -> pd.DataFrame:
    """Totals per game for the team in team_col (posteam = offense, defteam = what the defense allowed)."""
    key = ["game_id", team_col]
    g = plays.groupby(key)
    out = pd.DataFrame({
        "plays": g.size(), "yards": g.yards_gained.sum(), "epa": g.epa.sum(), "successes": g.success.sum(),
        "rush_att": g.rush_attempt.sum(), "rush_yds": g.apply(lambda x: x.yards_gained[x.rush_attempt == 1].sum(), include_groups=False),
        "pass_yds": g.apply(lambda x: x.yards_gained[x.pass_attempt == 1].sum(), include_groups=False),
        "sacks": g.sack.sum(), "third_conv": g.third_down_converted.sum(), "third_fail": g.third_down_failed.sum(),
        "giveaways": g.interception.sum() + g.fumble_lost.sum()})
    d = drives.groupby(["game_id", team_col])
    out = out.join(pd.DataFrame({"rz_trips": d.drive_inside20.sum(), "rz_tds": d.rz_td.sum(), "possession_s": d.possession_s.sum()}))
    p = punts.groupby(["game_id", team_col])
    out = out.join(pd.DataFrame({"punts": p.size(), "punt_net": p.net.sum()}))
    return out.fillna({"punts": 0, "punt_net": 0}).reset_index().rename(columns={team_col: "team"})


def game_rows(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per team and game: offense (off_*) and what the defense allowed (def_*), plus penalties and punting."""
    pbp = pbp[pbp.season_type == "REG"].copy()
    scrimmage = pbp[pbp.play_type.isin(["run", "pass"]) & (pbp.qb_kneel != 1) & (pbp.qb_spike != 1) & pbp.posteam.notna()]
    drives = pbp[pbp.posteam.notna() & pbp.fixed_drive.notna()].drop_duplicates(["game_id", "posteam", "fixed_drive"]).copy()
    drives["rz_td"] = ((drives.drive_inside20 == 1) & (drives.fixed_drive_result == "Touchdown")).astype(int)
    drives["possession_s"] = drives.drive_time_of_possession.map(_seconds)
    punts = pbp[(pbp.punt_attempt == 1) & pbp.posteam.notna()].copy()
    punts["net"] = punts.kick_distance.fillna(0) - punts.return_yards.fillna(0) - 20 * punts.touchback.fillna(0)
    offense = _side(scrimmage, drives, punts, "posteam").add_prefix("off_").rename(columns={"off_game_id": "game_id", "off_team": "team"})
    defense = _side(scrimmage, drives, punts, "defteam").add_prefix("def_").rename(columns={"def_game_id": "game_id", "def_team": "team"})
    pens = pbp[(pbp.penalty == 1) & pbp.penalty_team.notna()].groupby(["game_id", "penalty_team"]).agg(
        penalties=("penalty", "size"), penalty_yds=("penalty_yards", "sum")).reset_index().rename(columns={"penalty_team": "team"})
    games = pbp.drop_duplicates("game_id")[["game_id", "week", "game_date", "home_team", "away_team"]]
    teams = pd.concat([games.assign(team=games.home_team, opponent=games.away_team, home=True),
                       games.assign(team=games.away_team, opponent=games.home_team, home=False)])
    rows = teams.merge(offense, on=["game_id", "team"], how="left").merge(defense, on=["game_id", "team"], how="left")
    rows = rows.merge(pens, on=["game_id", "team"], how="left").fillna({"penalties": 0, "penalty_yds": 0})
    rows["game_date"] = pd.to_datetime(rows.game_date).dt.date
    return rows.drop(columns=["home_team", "away_team", "def_punts", "def_punt_net"]).reset_index(drop=True)


def _rate(num, den):
    return float(num / den) if den else None


def _r(value, digits):
    return None if value is None else round(float(value), digits)


def summarise(rows: pd.DataFrame) -> dict:
    """Per-game averages and rates over the rows given (one team's games)."""
    n = len(rows)
    if n == 0:
        return {"games": 0}
    s = rows.sum(numeric_only=True)
    takeaways = s.def_giveaways
    return {
        "games": n,
        "yards_per_game": _r(s.off_yards / n, 1), "yards_allowed_per_game": _r(s.def_yards / n, 1),
        "yards_per_play": _r(_rate(s.off_yards, s.off_plays), 2), "yards_per_play_allowed": _r(_rate(s.def_yards, s.def_plays), 2),
        "rush_yards_per_game": _r(s.off_rush_yds / n, 1), "rush_yards_per_carry": _r(_rate(s.off_rush_yds, s.off_rush_att), 2),
        "rush_yards_allowed_per_game": _r(s.def_rush_yds / n, 1), "rush_yards_per_carry_allowed": _r(_rate(s.def_rush_yds, s.def_rush_att), 2),
        "pass_yards_per_game": _r(s.off_pass_yds / n, 1), "pass_yards_allowed_per_game": _r(s.def_pass_yds / n, 1),
        "epa_per_play": _r(_rate(s.off_epa, s.off_plays), 3), "epa_per_play_allowed": _r(_rate(s.def_epa, s.def_plays), 3),
        "success_rate": _r(_rate(s.off_successes, s.off_plays), 3), "success_rate_allowed": _r(_rate(s.def_successes, s.def_plays), 3),
        "third_down_pct": _r(_rate(s.off_third_conv, s.off_third_conv + s.off_third_fail), 3),
        "third_down_pct_allowed": _r(_rate(s.def_third_conv, s.def_third_conv + s.def_third_fail), 3),
        "red_zone_td_pct": None if not s.off_rz_trips else _r(s.off_rz_tds / s.off_rz_trips, 3),
        "red_zone_td_pct_allowed": None if not s.def_rz_trips else _r(s.def_rz_tds / s.def_rz_trips, 3),
        "giveaways": int(s.off_giveaways), "takeaways": int(takeaways), "turnover_margin": int(takeaways - s.off_giveaways),
        "sacks_taken_per_game": _r(s.off_sacks / n, 2), "sacks_per_game": _r(s.def_sacks / n, 2),
        "penalties_per_game": _r(s.penalties / n, 1), "penalty_yards_per_game": _r(s.penalty_yds / n, 1),
        "net_punt_avg": None if not s.off_punts else _r(s.off_punt_net / s.off_punts, 1),
        "time_of_possession_per_game": "{:d}:{:02d}".format(*divmod(int(round(s.off_possession_s / n)), 60)),
    }


def before(rows: pd.DataFrame, team: str, cutoff, last: int | None = None) -> dict:
    """A team's summary over its games played before the cutoff date (the last `last` of them when given)."""
    mine = rows[(rows.team == team) & (rows.game_date < cutoff)].sort_values("game_date")
    return summarise(mine.tail(last) if last else mine)


def league_ranks(rows: pd.DataFrame, cutoff) -> dict[str, dict[str, int]]:
    """Each team's rank (1 = best) on a few headline numbers over games before the cutoff."""
    teams = sorted(rows.team.unique())
    table = pd.DataFrame({t: before(rows, t, cutoff) for t in teams}).T
    if table.empty or "epa_per_play" not in table:
        return {}
    better_high = ["epa_per_play", "success_rate", "third_down_pct", "red_zone_td_pct", "turnover_margin", "net_punt_avg", "rush_yards_per_game"]
    better_low = ["epa_per_play_allowed", "third_down_pct_allowed", "rush_yards_allowed_per_game", "penalty_yards_per_game"]
    ranks = {}
    for col in better_high + better_low:
        values = pd.to_numeric(table[col], errors="coerce")
        ranks[col] = values.rank(ascending=col in better_low, method="min")
    frame = pd.DataFrame(ranks)
    return {t: {c: (None if np.isnan(v) else int(v)) for c, v in row.items()} for t, row in frame.iterrows()}
