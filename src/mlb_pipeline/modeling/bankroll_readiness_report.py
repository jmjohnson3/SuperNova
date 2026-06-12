"""Summarize shadow-bankroll tiers for a prediction date."""
from __future__ import annotations

import argparse
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import create_engine, text

_ET = ZoneInfo("America/New_York")
_PG_DSN = "postgresql://josh:password@localhost:5432/nba"


GAME_SQL = """
WITH game_signals AS (
    SELECT
        'game_run_line'::text AS market,
        run_line_bet_side AS side,
        bankroll_tier_rl AS tier,
        bankroll_candidate_rl AS candidate,
        bankroll_reasons_rl AS reasons,
        stake_pct_rl AS stake_pct
    FROM bets.mlb_game_predictions
    WHERE game_date_et = :game_date
      AND run_line_bet_side IS NOT NULL
    UNION ALL
    SELECT
        'game_total'::text AS market,
        total_bet_side AS side,
        bankroll_tier_total AS tier,
        bankroll_candidate_total AS candidate,
        bankroll_reasons_total AS reasons,
        stake_pct_total AS stake_pct
    FROM bets.mlb_game_predictions
    WHERE game_date_et = :game_date
      AND total_bet_side IS NOT NULL
)
SELECT
    market,
    side,
    COALESCE(tier, 'paper') AS tier,
    COALESCE(candidate, false) AS candidate,
    COALESCE(NULLIF(reasons, ''), 'ok') AS reasons,
    COUNT(*) AS picks,
    SUM(COALESCE(stake_pct, 0)) AS exposure_pct
FROM game_signals
GROUP BY 1, 2, 3, 4, 5
ORDER BY candidate DESC, market, side, reasons
"""


PROP_SQL = """
SELECT
    stat AS market,
    bet_side AS side,
    COALESCE(bankroll_tier, 'paper') AS tier,
    COALESCE(bankroll_candidate, false) AS candidate,
    COALESCE(NULLIF(bankroll_reasons, ''), 'ok') AS reasons,
    COUNT(*) AS picks,
    SUM(COALESCE(stake_pct, 0)) AS exposure_pct
FROM bets.mlb_prop_predictions
WHERE game_date_et = :game_date
  AND bet_side IS NOT NULL
  AND edge IS NOT NULL
  AND COALESCE(bankroll_tier, 'research') <> 'research'
GROUP BY 1, 2, 3, 4, 5
ORDER BY candidate DESC, market, side, reasons
"""


def _print_section(title: str, df: pd.DataFrame) -> None:
    print(title)
    if df.empty:
        print("  No bankroll-layer signals found.")
        return
    for _, row in df.iterrows():
        exposure = float(row["exposure_pct"] or 0.0) * 100.0
        label = "BANKROLL" if bool(row["candidate"]) else str(row["tier"]).upper()
        print(
            f"  {label:9} {row['market']} {row['side']}: "
            f"{int(row['picks'])} pick(s), exposure {exposure:.2f}%"
            f" | {row['reasons']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize shadow-bankroll tiers")
    parser.add_argument("--date", type=str, default=None, help="Prediction date YYYY-MM-DD ET")
    parser.add_argument("--dsn", type=str, default=_PG_DSN, help="Postgres DSN")
    parser.add_argument(
        "--global-max-exposure-pct",
        type=float,
        default=0.02,
        help="Global daily bankroll cap as a fraction, default 0.02 = 2%%",
    )
    args = parser.parse_args()

    game_date = date.fromisoformat(args.date) if args.date else datetime.now(_ET).date()
    engine = create_engine(args.dsn)

    print(f"Bankroll readiness for {game_date}")
    print("Shadow mode: this report labels signals; it does not suppress picks.")

    try:
        games = pd.read_sql(text(GAME_SQL), engine, params={"game_date": game_date})
        props = pd.read_sql(text(PROP_SQL), engine, params={"game_date": game_date})
    except Exception as exc:
        raise SystemExit(f"Could not load bankroll tiers. Run predictions first. Detail: {exc}") from exc

    _print_section("\nGame signals", games)
    _print_section("\nProp signals", props)

    total_candidates = int(games["picks"].where(games["candidate"], 0).sum()) if not games.empty else 0
    total_candidates += int(props["picks"].where(props["candidate"], 0).sum()) if not props.empty else 0
    total_exposure = float(games["exposure_pct"].where(games["candidate"], 0).sum()) if not games.empty else 0.0
    total_exposure += float(props["exposure_pct"].where(props["candidate"], 0).sum()) if not props.empty else 0.0
    print(f"\nBankroll candidates: {total_candidates}")
    print(f"Total candidate exposure: {total_exposure * 100:.2f}%")
    if total_exposure > args.global_max_exposure_pct + 1e-12:
        overflow = total_exposure - args.global_max_exposure_pct
        print(
            f"WARNING: exceeds global cap "
            f"{args.global_max_exposure_pct * 100:.2f}% by {overflow * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
