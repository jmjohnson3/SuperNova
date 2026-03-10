# src/nba_pipeline/modeling/post_parlay.py
"""
Build a Discord parlay message from today's best-edge game predictions
and post it to the webhook.

Usage:
    python -m nba_pipeline.modeling.post_parlay          # post to Discord
    python -m nba_pipeline.modeling.post_parlay --dry-run  # print only
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import create_engine, text

log = logging.getLogger("nba_pipeline.modeling.post_parlay")

_ET = ZoneInfo("America/New_York")
PG_DSN = "postgresql://josh:password@localhost:5432/nba"
DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1461420766108319858/LenBk50YR1eS1isFMSOzE8gMWgSgBTSYmU4Ac1unf2SOo_kPSGk71afBqbBiQDuUZwD3",
)

# Minimum edge thresholds — match predict_today.py config
MIN_EDGE_SPREAD = 10.0
MIN_EDGE_TOTAL  = 7.0


def _american_to_decimal(american: int) -> float:
    if american >= 0:
        return american / 100 + 1
    return 100 / abs(american) + 1


def _decimal_to_american(decimal: float) -> str:
    if decimal >= 2.0:
        val = round((decimal - 1) * 100)
        return f"+{val}"
    else:
        val = round(-100 / (decimal - 1))
        return f"-{val}"


def _parlay_payout(legs: list[int]) -> str:
    """Compute American parlay payout given list of American odds per leg."""
    dec = 1.0
    for odds in legs:
        dec *= _american_to_decimal(odds)
    # Net payout (subtract stake)
    return _decimal_to_american(dec)


def _parlay_payout_decimal(legs: list[int]) -> float:
    dec = 1.0
    for odds in legs:
        dec *= _american_to_decimal(odds)
    return dec


def _format_spread_label(away: str, home: str, market_spread_home: float,
                          bet_side: str) -> tuple[str, str, int]:
    """
    Returns (bet_label, matchup_str, juice).
    bet_label:   e.g. 'GSW -7.5'
    matchup_str: e.g. 'GSW @ UTA'
    juice:       -110 (standard)
    """
    matchup = f"{away} @ {home}"
    if bet_side == "home":
        # market_spread_home is home's spread: positive = home is underdog, negative = favorite
        spread_val = market_spread_home
        team = home
    else:
        # Away team's spread is the opposite of home's spread
        spread_val = -market_spread_home
        team = away

    sign = "-" if spread_val < 0 else "+"
    label = f"{team} {sign}{abs(spread_val):.1f}"
    return label, matchup, -110


def _format_total_label(away: str, home: str, market_total: float,
                         bet_side: str) -> tuple[str, str, int]:
    side = "Over" if bet_side == "over" else "Under"
    label = f"{away}@{home} {side} {market_total:.1f}"
    matchup = f"{away} @ {home}"
    return label, matchup, -110


def _load_plays(engine, et_day) -> list[dict]:
    """Load today's flagged bets meeting the edge thresholds."""
    rows = engine.connect().execute(text("""
        SELECT home_team_abbr, away_team_abbr,
               edge_spread, spread_bet_side, market_spread_home,
               win_prob_spread, kelly_fraction_spread,
               edge_total, total_bet_side, market_total,
               win_prob_total, kelly_fraction_total,
               used_residual_model
        FROM bets.game_predictions
        WHERE game_date_et = :d
        ORDER BY ABS(edge_spread) DESC NULLS LAST, ABS(edge_total) DESC NULLS LAST
    """), {"d": et_day}).fetchall()

    plays = []
    for r in rows:
        # Spread bet
        if (r.edge_spread is not None
                and abs(float(r.edge_spread)) >= MIN_EDGE_SPREAD
                and r.spread_bet_side is not None):
            label, matchup, juice = _format_spread_label(
                r.away_team_abbr, r.home_team_abbr,
                float(r.market_spread_home), r.spread_bet_side,
            )
            plays.append({
                "type": "spread",
                "matchup": matchup,
                "label": label,
                "juice": juice,
                "edge": float(r.edge_spread),
                "win_prob": float(r.win_prob_spread or 0),
                "kelly": float(r.kelly_fraction_spread or 0),
            })

        # Total bet — only if residual model was used
        if (r.edge_total is not None
                and abs(float(r.edge_total)) >= MIN_EDGE_TOTAL
                and r.total_bet_side is not None
                and r.used_residual_model):
            label, matchup, juice = _format_total_label(
                r.away_team_abbr, r.home_team_abbr,
                float(r.market_total), r.total_bet_side,
            )
            plays.append({
                "type": "total",
                "matchup": matchup,
                "label": label,
                "juice": juice,
                "edge": float(r.edge_total),
                "win_prob": float(r.win_prob_total or 0),
                "kelly": float(r.kelly_fraction_total or 0),
            })

    return plays


def _build_message(plays: list[dict], et_day) -> str:
    if not plays:
        return f"**SuperNovaBets — {et_day}**\nNo high-edge plays today."

    emojis = ["1\u20e3", "2\u20e3", "3\u20e3", "4\u20e3", "5\u20e3"]
    lines = []
    lines.append(f":basketball: **SUPERNOVA BEST BETS — {et_day}** :basketball:")
    lines.append("")

    # --- Straight plays ---
    lines.append("**STRAIGHT PLAYS**")
    for i, p in enumerate(plays):
        wp = p["win_prob"]
        qk = (p["kelly"] / 4) * 1000
        edge_abs = abs(p["edge"])
        edge_str = f"+{edge_abs:.1f}" if p["edge"] > 0 else f"-{edge_abs:.1f}"
        tag = " `[resid]`" if p["type"] == "total" else ""
        lines.append(
            f"{emojis[i] if i < len(emojis) else '*'} **{p['label']}** "
            f"({p['juice']:+d}) · edge={edge_str} · p={wp:.0%} "
            f"· 1/4K=${qk:.0f}/$1k{tag}"
        )
    lines.append("")

    # --- Parlays ---
    juices = [p["juice"] for p in plays]
    probs  = [p["win_prob"] for p in plays]

    if len(plays) >= 2:
        lines.append("**PARLAYS**")

    # 2-leg: best two plays (highest win_prob)
    if len(plays) >= 2:
        top2 = sorted(plays, key=lambda x: x["win_prob"], reverse=True)[:2]
        j2 = [p["juice"] for p in top2]
        p2_combined = math.prod(p["win_prob"] for p in top2)
        payout2 = _parlay_payout(j2)
        dec2 = _parlay_payout_decimal(j2)
        ev2 = p2_combined * dec2 - 1
        legs2 = " + ".join(f"**{p['label']}**" for p in top2)
        lines.append(
            f":zap: 2-leg: {legs2}\n"
            f"   Payout: `{payout2}` · Combined p={p2_combined:.0%} · EV={ev2:+.0%}"
        )

    # 3-leg parlay (if 3+ plays)
    if len(plays) >= 3:
        top3 = sorted(plays, key=lambda x: x["win_prob"], reverse=True)[:3]
        j3 = [p["juice"] for p in top3]
        p3_combined = math.prod(p["win_prob"] for p in top3)
        payout3 = _parlay_payout(j3)
        dec3 = _parlay_payout_decimal(j3)
        ev3 = p3_combined * dec3 - 1
        legs3 = " + ".join(f"**{p['label']}**" for p in top3)
        lines.append(
            f":zap::zap: 3-leg: {legs3}\n"
            f"   Payout: `{payout3}` · Combined p={p3_combined:.0%} · EV={ev3:+.0%}"
        )

    lines.append("")
    lines.append(f"*Model: walk-forward XGBoost | Spread MAE 10.9 | Thresholds: spread>=10pt, total>=7pt+resid*")

    return "\n".join(lines)


async def _post(content: str) -> None:
    async with httpx.AsyncClient(timeout=20) as client:
        for attempt in range(4):
            try:
                r = await client.post(DISCORD_WEBHOOK_URL, json={"content": content})
                if r.status_code in (200, 204):
                    print("Posted to Discord.")
                    return
                if r.status_code == 429 and attempt < 3:
                    retry_after = float(r.json().get("retry_after", 1.5))
                    await asyncio.sleep(retry_after)
                    continue
                r.raise_for_status()
            except httpx.TimeoutException:
                if attempt >= 3:
                    raise
                await asyncio.sleep(2.0 * (attempt + 1))


def main() -> None:
    import sys
    # Allow Unicode output on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Post today's parlay to Discord")
    parser.add_argument("--dry-run", action="store_true", help="Print message, don't post")
    parser.add_argument("--date", help="ET date override YYYY-MM-DD")
    args = parser.parse_args()

    et_day = args.date or datetime.now(_ET).date().isoformat()

    engine = create_engine(PG_DSN)
    plays = _load_plays(engine, et_day)

    msg = _build_message(plays, et_day)
    # Print safely regardless of console encoding
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))
    print()

    if not args.dry_run:
        asyncio.run(_post(msg))


if __name__ == "__main__":
    main()
