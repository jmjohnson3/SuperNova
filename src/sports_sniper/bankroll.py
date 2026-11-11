"""Paper-bankroll simulator. Hypothetical money only; nothing here places a wager.

It walks settled bets one game day at a time. Stakes for a day are worked out from the bankroll at the start of
that day (the bets overlap, so the day's results are not known when they are placed), limits are applied in the
order the prices were seen, and the day's results are added at the end.

Staking is flat or fractional Kelly. Kelly uses the conservative probability (the estimate lowered by its
uncertainty), so a thin edge gets a small stake or none, and it shrinks as the bankroll falls. Nothing here ever
raises a stake because of a loss.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from sports_sniper import odds_math


@dataclass(frozen=True)
class Rules:
    start: float = 100.0
    staking: str = "flat"               # flat | kelly
    flat_stake: float = 1.0
    kelly_fraction: float = 0.25        # a quarter of the full Kelly stake
    confidence_z: float = 1.0           # how far the probability is lowered before Kelly is worked out
    min_stake: float = 0.10             # a smaller stake than this is not placed
    max_stake: float | None = 5.0
    max_daily: float | None = 20.0      # total staked in one game day
    max_per_sport_daily: float | None = None
    max_per_game: float | None = 3.0    # bets on one game move together
    max_per_player: float | None = 2.0  # so do bets on one player in one game
    stop_drawdown: float | None = None  # stop placing bets once the bankroll is this fraction below its high


def kelly_fraction(probability: float, decimal: float) -> float:
    """The full Kelly share of bankroll for a bet at decimal odds; zero when there is no edge."""
    if decimal <= 1:
        return 0.0
    return max(0.0, (probability * (decimal - 1) - (1 - probability)) / (decimal - 1))


def stake_for(bet, bankroll: float, rules: Rules) -> float:
    if rules.staking == "flat":
        stake = rules.flat_stake
    elif rules.staking == "kelly":
        error = 0.0 if pd.isna(bet.standard_error) else float(bet.standard_error)
        low = odds_math.conservative_probability(float(bet.probability), error, rules.confidence_z)
        stake = bankroll * rules.kelly_fraction * kelly_fraction(low, float(bet.decimal))
    else:
        raise ValueError("staking must be flat or kelly")
    return stake if rules.max_stake is None else min(stake, rules.max_stake)


def simulate(bets: pd.DataFrame, rules: Rules = Rules()) -> dict:
    """Run settled bets (won, lost, push, void) through the rules. Needs the columns of `backtest.run` or paper bets."""
    counted = bets[bets.status.isin(["won", "lost", "push", "void"])].copy() if len(bets) else bets
    if not len(counted):
        return dict(start=rules.start, final=rules.start, profit=0.0, staked=0.0, roi=None, placed=0, skipped={},
                    max_drawdown=0.0, max_drawdown_pct=0.0, lowest=rules.start, stopped=None, curve=pd.DataFrame(), bets=counted)
    if "day" not in counted:
        counted["day"] = counted.commence_time.dt.tz_convert("America/New_York").dt.date
    seen = "price_observed_at" if "price_observed_at" in counted else "observed_at"
    bankroll = high = lowest = rules.start
    worst = worst_pct = 0.0
    skipped: Counter = Counter()
    curve, placed, stopped = [], [], None
    for day, group in counted.sort_values(["day", seen], kind="stable").groupby("day", sort=True):
        day_total, by_sport, by_game, by_player, change = 0.0, Counter(), Counter(), Counter(), 0.0
        for bet in group.itertuples():
            if stopped is not None:
                skipped["stopped after drawdown"] += 1
                continue
            stake = stake_for(bet, bankroll, rules)
            if stake < rules.min_stake:
                skipped["stake too small (no edge after the haircut)"] += 1
                continue
            room = [bankroll - day_total]
            for limit, used, why in ((rules.max_daily, day_total, "daily limit"),
                                     (rules.max_per_sport_daily, by_sport[bet.sport], "sport limit"),
                                     (rules.max_per_game, by_game[bet.event_id], "game limit"),
                                     (rules.max_per_player, by_player[(bet.event_id, bet.player_key)], "player limit")):
                if limit is not None:
                    room.append(limit - used)
            stake = min(stake, *room)
            if stake < rules.min_stake:
                limits = [why for limit, used, why in ((rules.max_daily, day_total, "daily limit"),
                                                       (rules.max_per_sport_daily, by_sport[bet.sport], "sport limit"),
                                                       (rules.max_per_game, by_game[bet.event_id], "game limit"),
                                                       (rules.max_per_player, by_player[(bet.event_id, bet.player_key)], "player limit"))
                          if limit is not None and limit - used < rules.min_stake]
                skipped[limits[0] if limits else "bankroll used up"] += 1
                continue
            day_total += stake
            by_sport[bet.sport] += stake
            by_game[bet.event_id] += stake
            by_player[(bet.event_id, bet.player_key)] += stake
            gain = {"won": stake * (float(bet.decimal) - 1), "lost": -stake}.get(bet.status, 0.0)
            change += gain
            placed.append(dict(index=bet.Index, day=day, stake=stake, profit=gain))
        bankroll += change
        high, lowest = max(high, bankroll), min(lowest, bankroll)
        worst, worst_pct = max(worst, high - bankroll), max(worst_pct, (high - bankroll) / high if high > 0 else 0.0)
        curve.append(dict(day=day, bankroll=bankroll, staked=day_total, profit=change))
        if rules.stop_drawdown is not None and stopped is None and high > 0 and (high - bankroll) / high >= rules.stop_drawdown:
            stopped = day
    log = pd.DataFrame(placed)
    staked = float(log.stake.sum()) if len(log) else 0.0
    return dict(start=rules.start, final=bankroll, profit=bankroll - rules.start, staked=staked,
                roi=(bankroll - rules.start) / staked if staked else None, placed=len(log), skipped=dict(skipped),
                max_drawdown=worst, max_drawdown_pct=worst_pct, lowest=lowest, stopped=stopped,
                curve=pd.DataFrame(curve), bets=log)
