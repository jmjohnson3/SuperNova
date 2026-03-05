"""Grade yesterday's predictions against actual results.

Back-fills bets.game_predictions with actuals from raw.nba_games, then
prints a Discord-ready summary of yesterday's performance.

Run daily after parse_all so scores are current.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

from sqlalchemy import create_engine, text

log = logging.getLogger("nba_pipeline.grade_predictions")

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade_all_pending(engine) -> int:
    """UPDATE bets.game_predictions for all games that now have final scores.

    Computes:
      - actual_margin_home  (home_score - away_score)
      - actual_total        (home_score + away_score)
      - spread_covered      (True/False when we had a spread bet + market line)
      - total_correct       (True/False when we had a total bet + market line)

    Returns number of rows updated.
    """
    with engine.begin() as conn:
        result = conn.execute(text("""
            UPDATE bets.game_predictions p
            SET
                actual_margin_home = g.home_score - g.away_score,
                actual_total       = g.home_score + g.away_score,
                spread_covered = CASE
                    WHEN p.spread_bet_side IS NULL OR p.market_spread_home IS NULL THEN NULL
                    -- market_spread_home is the home team's line (neg = home favored)
                    -- home covers if actual_margin_home > -market_spread_home
                    WHEN p.spread_bet_side = 'home'
                         THEN (g.home_score - g.away_score) > -p.market_spread_home
                    WHEN p.spread_bet_side = 'away'
                         THEN (g.home_score - g.away_score) < -p.market_spread_home
                    ELSE NULL
                END,
                total_correct = CASE
                    WHEN p.total_bet_side IS NULL OR p.market_total IS NULL THEN NULL
                    WHEN p.total_bet_side = 'over'
                         THEN (g.home_score + g.away_score) > p.market_total
                    WHEN p.total_bet_side = 'under'
                         THEN (g.home_score + g.away_score) < p.market_total
                    ELSE NULL
                END
            FROM raw.nba_games g
            WHERE p.game_slug = g.game_slug
              AND g.home_score IS NOT NULL
              AND g.away_score IS NOT NULL
              AND p.actual_total IS NULL
        """))
        return result.rowcount


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------

def _spread_label(team_abbr: str, line: float) -> str:
    """Format a spread as e.g. 'BOS -4.5' or 'OKL +3'."""
    sign = "-" if line < 0 else "+"
    return f"{team_abbr} {sign}{abs(line):.1f}"


def _yesterday_summary(engine, yesterday: date) -> str:
    """Build a plain-text summary of yesterday's results for Discord."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                p.away_team_abbr,
                p.home_team_abbr,
                p.actual_total,
                p.actual_margin_home,
                p.pred_total,
                p.pred_margin_home,
                p.market_total,
                p.market_spread_home,
                p.spread_bet_side,
                p.total_bet_side,
                p.spread_covered,
                p.total_correct
            FROM bets.game_predictions p
            WHERE p.game_date_et = :d
              AND p.actual_total IS NOT NULL
            ORDER BY p.game_slug
        """), {"d": yesterday}).fetchall()

    if not rows:
        return f"No graded results for {yesterday} (scores not yet available)."

    header = f"Results — {yesterday.strftime('%b %d, %Y')} ({len(rows)} games)"
    lines = [header, ""]

    total_abs_errs: list[float] = []
    spread_abs_errs: list[float] = []
    total_bets_won = total_bets = 0
    spread_bets_won = spread_bets = 0

    for r in rows:
        away, home = r[0], r[1]
        act_total    = float(r[2])
        act_margin   = float(r[3])   # home - away (positive = home won)
        pred_total   = float(r[4])
        pred_margin  = float(r[5])
        mkt_total    = float(r[6]) if r[6] is not None else None
        mkt_spread   = float(r[7]) if r[7] is not None else None
        spread_side  = r[8]   # 'home' | 'away' | None
        total_side   = r[9]   # 'over' | 'under' | None
        spread_won   = r[10]  # bool | None
        total_won    = r[11]  # bool | None

        total_err = pred_total - act_total
        spread_err = pred_margin - act_margin
        total_abs_errs.append(abs(total_err))
        spread_abs_errs.append(abs(spread_err))

        # ── Total column ──
        if mkt_total is not None and total_side is not None:
            result_sym = "W" if total_won else "L"
            direction = total_side.upper()
            total_col = f"Total {act_total:.0f} mk={mkt_total:.1f} BET {direction} {result_sym}"
            total_bets += 1
            if total_won:
                total_bets_won += 1
        else:
            err_str = f"{total_err:+.1f}"
            total_col = f"Total {act_total:.0f} pred={pred_total:.1f} ({err_str})"

        # ── Spread column ──
        # actual winner label: positive act_margin = home won
        if act_margin >= 0:
            act_spread_str = _spread_label(home, -act_margin)   # home -X
        else:
            act_spread_str = _spread_label(away, act_margin)    # away -X (act_margin negative)

        if mkt_spread is not None and spread_side is not None:
            result_sym = "W" if spread_won else "L"
            bet_team = home if spread_side == "home" else away
            spread_col = (
                f"Spread mk={_spread_label(home, mkt_spread)} "
                f"BET {bet_team} {result_sym}"
            )
            spread_bets += 1
            if spread_won:
                spread_bets_won += 1
        else:
            # Show actual result and model prediction
            if pred_margin >= 0:
                pred_spread_str = _spread_label(home, -pred_margin)
            else:
                pred_spread_str = _spread_label(away, pred_margin)
            spread_col = f"Spread act={act_spread_str} pred={pred_spread_str}"

        lines.append(f"{away}@{home}  {total_col}  |  {spread_col}")

    # ── Aggregate stats ──
    lines.append("")
    mae_t = sum(total_abs_errs) / len(total_abs_errs) if total_abs_errs else 0.0
    mae_s = sum(spread_abs_errs) / len(spread_abs_errs) if spread_abs_errs else 0.0
    lines.append(f"Total MAE: {mae_t:.1f} pts   Spread MAE: {mae_s:.1f} pts")

    if total_bets > 0 or spread_bets > 0:
        t_pct = f" ({total_bets_won/total_bets*100:.0f}%)" if total_bets > 0 else ""
        s_pct = f" ({spread_bets_won/spread_bets*100:.0f}%)" if spread_bets > 0 else ""
        lines.append(
            f"Total bets: {total_bets_won}/{total_bets}{t_pct}   "
            f"Spread bets: {spread_bets_won}/{spread_bets}{s_pct}"
        )
    else:
        lines.append("No bets (market lines not available for this date)")

    return "\n".join(lines)


def _running_record(engine, lookback_days: int = 30) -> str:
    """Return a one-liner running W/L record over recent graded predictions."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT
                COUNT(*) FILTER (WHERE total_correct IS NOT NULL)  AS total_bets,
                COUNT(*) FILTER (WHERE total_correct = TRUE)       AS total_wins,
                COUNT(*) FILTER (WHERE spread_covered IS NOT NULL) AS spread_bets,
                COUNT(*) FILTER (WHERE spread_covered = TRUE)      AS spread_wins
            FROM bets.game_predictions
            WHERE game_date_et >= CURRENT_DATE - :days
              AND actual_total IS NOT NULL
        """), {"days": lookback_days}).fetchone()

    if row is None or row[0] == 0:
        return ""

    tb, tw, sb, sw = row
    parts = []
    if tb > 0:
        pct = tw / tb * 100
        parts.append(f"Totals L{lookback_days}: {tw}/{tb} ({pct:.0f}%)")
    if sb > 0:
        pct = sw / sb * 100
        parts.append(f"Spreads L{lookback_days}: {sw}/{sb} ({pct:.0f}%)")

    return "  |  ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    engine = create_engine(_PG_DSN)

    # 1. Grade all ungraded predictions that now have final scores
    updated = grade_all_pending(engine)
    log.info("Graded %d game predictions", updated)

    # 2. Print yesterday's summary (captured by run_daily for Discord)
    yesterday = date.today() - timedelta(days=1)
    summary = _yesterday_summary(engine, yesterday)
    record = _running_record(engine)
    if record:
        summary += f"\nRunning record — {record}"

    print(summary)


if __name__ == "__main__":
    main()
