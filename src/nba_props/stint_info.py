"""What rotation stints add to the minutes inputs for the NBA prop challenger. Research only.

Box scores say how long a player was on the floor; stints (nba_props.stints) say when, and with whom.

    stints        his minutes with garbage time taken out (the fourth quarter of a game already decided), how much
                  of his time is garbage time, fourth quarter or the close finish, when he usually first comes on,
                  how his minutes are cut up, and whether his last game ended early for him
    substitutes   for each rotation teammate who is newly out or back tonight: how much the player's floor time
                  goes with that teammate's bench time in earlier games (he comes on when the other goes off)

The score during a game is rebuilt from each stint's plus-minus, so garbage time and close finishes are
approximate. Everything is from earlier games only; games without stored stints leave the inputs empty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nba_props import minutes_info

TENTHS = 600.0                 # stint times are tenths of a second: this many to a minute
FOURTH = 36 * TENTHS           # the fourth quarter (and overtime) starts here
LATE = 43 * TENTHS             # the last five minutes of regulation, and overtime
GARBAGE_MARGIN = 15.0          # a fourth-quarter lead this large: the game is decided
CLOSE_MARGIN = 8.0             # a late margin this small: the game is being closed out
MIN_COMPETITIVE = 24 * TENTHS  # a game needs this much competitive time for a competitive-minutes figure
PAIR_HALFLIFE = 20             # shared games; how fast a pair's substitution record fades
PAIR_SHRINK = 3.0              # shared games before the record counts for half
MINUTES_TOLERANCE = 0.5        # a game whose stints disagree with a box-score minute total by more is not used
GROUPS = {
    "stints": ["comp_min_ewm3", "comp_min_ewm10", "garbage_lift3", "garbage_share_ewm10", "q4_share_ewm10", "close_share_ewm10",
               "first_in_ewm5", "stints_ewm5", "longest_ewm5", "last_out_gap", "last_comp_gap"],
    "substitutes": ["sub_new", "sub_back", "sub_support"],
}
FEATURES = [c for cols in GROUPS.values() for c in cols]
PLAYER_COLUMNS = ["game_id", "player_id", "sec", "q4_sec", "garbage_sec", "close_sec", "first_in", "last_out", "n_stints", "longest",
                  "game_len", "garbage_len", "close_len"]


def stored(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('nbaprops.rotation_stints')")
        return cur.fetchone()[0] is not None


def load(conn) -> pd.DataFrame:
    return pd.read_sql("""SELECT game_id, player_id, team_id, in_time, out_time, pt_diff::float AS pt_diff
                          FROM nbaprops.rotation_stints ORDER BY game_id, team_id, player_id, in_time""", conn)


# --- one game ------------------------------------------------------------------------------------------------------
def summarise(game: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One game's stints as a row per player (floor time by kind, in tenths of a second) and a row per ordered
    pair of teammates (the time both were on the floor)."""
    game = game[game.out_time > game.in_time]
    start, end = game.in_time.to_numpy(float), game.out_time.to_numpy(float)
    bounds = np.unique(np.concatenate([start, end]))
    seg_start, seg_len = bounds[:-1], np.diff(bounds)
    on = (start[:, None] <= seg_start[None, :]) & (end[:, None] >= bounds[1:][None, :])            # stint x segment
    teams = game.team_id.to_numpy()
    known = game.pt_diff.notna().to_numpy()
    margin = np.zeros(len(seg_start))
    if known.any():                                       # each stint's plus-minus is the sum of its segments' score changes
        side = np.where(teams == teams[0], 1.0, -1.0)
        change, *_ = np.linalg.lstsq(on[known] * side[known][:, None], game.pt_diff.to_numpy(float)[known], rcond=None)
        margin = np.concatenate([[0.0], np.cumsum(change)])[:-1]                                    # at each segment's start
    garbage = (seg_start >= FOURTH) & (np.abs(margin) >= GARBAGE_MARGIN)
    close = (seg_start >= LATE) & (np.abs(margin) <= CLOSE_MARGIN)
    codes, ids = pd.factorize(game.player_id)
    floor = np.zeros((len(ids), len(seg_start)))
    np.add.at(floor, codes, on)
    floor = (floor > 0).astype(float)
    spans = game.assign(length=end - start).groupby(codes)
    players = pd.DataFrame(dict(
        game_id=game.game_id.iloc[0], player_id=ids, sec=floor @ seg_len, q4_sec=floor @ (seg_len * (seg_start >= FOURTH)),
        garbage_sec=floor @ (seg_len * garbage), close_sec=floor @ (seg_len * close), first_in=spans.in_time.min().to_numpy(float),
        last_out=spans.out_time.max().to_numpy(float), n_stints=spans.size().to_numpy(float), longest=spans.length.max().to_numpy(float),
        game_len=float(bounds[-1] - bounds[0]), garbage_len=float(seg_len[garbage].sum()), close_len=float(seg_len[close].sum())))
    team_of = pd.Series(teams).groupby(codes).first().to_numpy()
    shared = (floor * seg_len) @ floor.T
    a, b = np.nonzero((team_of[:, None] == team_of[None, :]) & ~np.eye(len(ids), dtype=bool))
    pairs = pd.DataFrame(dict(game_id=game.game_id.iloc[0], player_id=ids[a], mate=ids[b], shared=shared[a, b], sec=players.sec.to_numpy()[a],
                              mate_sec=players.sec.to_numpy()[b], game_len=players.game_len.to_numpy()[a]))
    return players[PLAYER_COLUMNS], pairs


def summaries(stints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = [summarise(game) for _, game in stints.groupby("game_id", sort=False)]
    if not parts:
        return pd.DataFrame(columns=PLAYER_COLUMNS), pd.DataFrame(columns=["game_id", "player_id", "mate", "shared", "sec", "mate_sec", "game_len"])
    return pd.concat([p for p, _ in parts], ignore_index=True), pd.concat([q for _, q in parts], ignore_index=True)


def verified(players: pd.DataFrame, pairs: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Drops games whose stints do not add up to the box-score minutes (the feed is wrong now and then), and
    says how many."""
    check = players.merge(df[["game_id", "player_id", "minutes"]], on=["game_id", "player_id"], how="left")
    wrong = (check.sec / TENTHS - check.minutes.astype(float)).abs().fillna(np.inf) > MINUTES_TOLERANCE
    bad = set(check.game_id[wrong])
    return players[~players.game_id.isin(bad)], pairs[~pairs.game_id.isin(bad)], len(bad)


# --- inputs --------------------------------------------------------------------------------------------------------
def stint_features(df: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Per player-game he played: his earlier games' floor time by kind. Empty until he has a game with stints."""
    d = df[df.minutes > 0].sort_values(["player_id", "game_date", "game_id"])[["game_id", "player_id", "minutes"]]
    d = d.merge(players, on=["game_id", "player_id"], how="left")
    g = d.player_id
    prior = lambda values, hl: values.groupby(g).transform(lambda s: s.shift(1).ewm(halflife=hl, min_periods=1, ignore_na=True).mean())
    competitive = d.game_len - d.garbage_len
    comp_min = ((d.sec - d.garbage_sec) / competitive * 48.0).where(competitive >= MIN_COMPETITIVE)
    out = d[["game_id", "player_id"]].copy()
    out["comp_min_ewm3"], out["comp_min_ewm10"] = prior(comp_min, 3), prior(comp_min, 10)
    out["garbage_lift3"] = out.comp_min_ewm3 - prior(d.minutes.where(comp_min.notna()), 3)
    floor = prior(d.sec, 10)
    out["garbage_share_ewm10"] = prior(d.garbage_sec, 10) / floor
    out["q4_share_ewm10"] = prior(d.q4_sec, 10) / floor
    out["close_share_ewm10"] = prior(d.close_sec, 10) / prior(d.close_len, 10).where(lambda v: v > 0)
    out["first_in_ewm5"] = prior(d.first_in, 5) / TENTHS
    out["stints_ewm5"] = prior(d.n_stints, 5)
    out["longest_ewm5"] = prior(d.longest, 5) / TENTHS
    last = lambda values: values.groupby(g).shift(1)
    out["last_out_gap"] = last(d.last_out / TENTHS - prior(d.last_out, 10) / TENTHS)        # negative: his last game ended early for him
    out["last_comp_gap"] = last(comp_min - prior(comp_min, 10))
    return out


def substitute_features(df: pd.DataFrame, absence: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Per player-game he played: tonight's newly-out and returning rotation minutes, weighted by how much the
    player's floor time went with each of those teammates' bench time in earlier shared games.

    `absence` is minutes_info.absence_pairs (one row per player-game and rotation teammate)."""
    dates = df[["game_id", "game_date"]].drop_duplicates("game_id")
    record = pairs.merge(dates, on="game_id").sort_values(["player_id", "mate", "game_date", "game_id"])
    key = [record.player_id, record.mate]
    fade = lambda col: record[col].groupby(key).transform(lambda s: s.ewm(halflife=PAIR_HALFLIFE, min_periods=1).mean())
    shared, sec, mate_sec, length = fade("shared"), fade("sec"), fade("mate_sec"), fade("game_len")
    games = record.groupby(key).cumcount() + 1
    with np.errstate(divide="ignore", invalid="ignore"):                # on when the teammate sits, less on when he plays
        goes_with_bench = ((sec - shared) / (length - mate_sec) - shared / mate_sec).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    record = record.assign(affinity=goes_with_bench * games / (games + PAIR_SHRINK), shared_games=games.astype(float))
    record = record[["player_id", "mate", "game_date", "affinity", "shared_games"]].sort_values("game_date")
    rows = absence[absence.change != 0].sort_values("game_date")
    rows = pd.merge_asof(rows, record, on="game_date", by=["player_id", "mate"], allow_exact_matches=False)
    rows[["affinity", "shared_games"]] = rows[["affinity", "shared_games"]].fillna(0.0)
    rows["sub_new"], rows["sub_back"] = rows.pm * rows.newly * rows.affinity, rows.pm * rows.back * rows.affinity
    rows["sub_support"] = rows.change.abs() * rows.shared_games.clip(upper=40)
    return rows.groupby(["game_id", "player_id"], as_index=False)[GROUPS["substitutes"]].sum()


def add(f: pd.DataFrame, df: pd.DataFrame, lineups: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """The feature frame with every input in FEATURES added. Substitute inputs are zero when nothing changed
    tonight and empty for games played before any stints are stored."""
    players, pairs, _ = verified(*summaries(stints), df)
    if players.empty:
        return f.assign(**{c: np.nan for c in FEATURES})
    out = f.merge(stint_features(df, players), on=["game_id", "player_id"], how="left")
    absence = minutes_info.absence_pairs(df, minutes_info.candidates(df, lineups))
    out = out.merge(substitute_features(df, absence, pairs), on=["game_id", "player_id"], how="left")
    first = df.loc[df.game_id.isin(set(players.game_id)), "game_date"].min()
    covered = out.game_date > first if pd.notna(first) else pd.Series(False, index=out.index)
    for c in GROUPS["substitutes"]:
        out[c] = out[c].fillna(0.0).where(covered)
    return out
