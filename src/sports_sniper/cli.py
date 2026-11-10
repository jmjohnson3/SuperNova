"""Command line for Sports Betting Sniper. Research and paper trading only; nothing here places a wager.

  python -m sports_sniper.cli scan                 # saved real snapshots if there are any, else sample data
  python -m sports_sniper.cli scan --source live   # The Odds API now (spends credits)
  python -m sports_sniper.cli top                  # the latest scan's ranked opportunities
  python -m sports_sniper.cli paper-bet 12         # save opportunity 12 as a paper bet
  python -m sports_sniper.cli quota                # credits left (free call)
  python -m sports_sniper.cli dashboard            # open the Streamlit dashboard
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import typer
from sqlalchemy.orm import Session

from sports_sniper import db
from sports_sniper.config import get_settings
from sports_sniper.markets import book_label, market_label
from sports_sniper.opportunities import describe
from sports_sniper.providers.odds_api import OddsApiClient, OddsApiError
from sports_sniper.scanner import SOURCES, run_scan

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)


def _pct(value) -> str:
    return "-" if value is None or value != value else f"{value:+.1%}"


def _row(o) -> str:
    return (f"{o.player} {describe(o.side, o.line)} ({market_label(o.market)}) at {book_label(o.book)} {o.price:+d} | "
            f"EV {_pct(o.ev)}, conservative {_pct(o.conservative_ev)} | {o.evidence}")


@app.command("init-db")
def init_db() -> None:
    """Create the Sniper's tables."""
    settings = get_settings()
    db.init_db(db.make_engine(settings.database_url))
    typer.echo("Tables ready.")


@app.command()
def scan(source: str = typer.Option("auto", help=" | ".join(SOURCES)),
         sport: list[str] = typer.Option(None, help="nfl, nba or mlb; repeat for several. Default: the configured sports."),
         limit: int = typer.Option(10, help="How many opportunities to print.")) -> None:
    """Collect prices, store them, and rank the opportunities."""
    if source not in SOURCES:
        raise typer.BadParameter(f"source must be one of: {', '.join(SOURCES)}")
    settings = get_settings()
    result = run_scan(settings, db.make_engine(settings.database_url), source, sports=sport or None)
    label = {"sample": "HYPOTHETICAL SAMPLE DATA", "saved": "saved real snapshots", "live": "live prices"}[result.source]
    typer.echo(f"Scan {result.scan_id} ({label}): {result.events} games, {result.quotes} prices ({result.new_quotes} new), "
               f"{len(result.current)} flagged.")
    if result.credits_used is not None:
        typer.echo(f"Credits used {result.credits_used}; remaining {result.credits_remaining}.")
    if result.note:
        typer.echo(f"Note: {result.note}")
    dropped = {k: v for k, v in result.dropped.items() if v and k != "not_player_market"}
    if dropped:
        typer.echo("Left out: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(dropped.items())))
    shown = [o for o in result.current if o.status == "priced" and o.conservative_ev > settings.min_conservative_ev]
    typer.echo(f"\n{len(shown)} with a positive conservative EV" + (":" if shown else ". Nothing clears the bar."))
    for o in shown[:limit]:
        typer.echo("  " + _row(o))


@app.command()
def top(limit: int = typer.Option(15), all_evidence: bool = typer.Option(False, "--all", help="Include speculative ones.")) -> None:
    """The latest scan's ranked opportunities, with their ids."""
    settings = get_settings()
    engine = db.make_engine(settings.database_url)
    db.init_db(engine)
    with Session(engine) as session:
        latest = db.latest_scan(session)
    if latest is None:
        typer.echo("No scans yet. Run: python -m sports_sniper.cli scan")
        raise typer.Exit()
    frame = db.load_opportunities(engine)
    frame = frame[(frame.data_source == latest.source) & ~frame.superseded & (frame.status == "priced")].sort_values("conservative_ev", ascending=False)
    if not all_evidence:
        frame = frame[frame.conservative_ev > settings.min_conservative_ev]
    typer.echo(f"Scan {latest.id} ({latest.source}), {len(frame)} shown:")
    for o in frame.head(limit).itertuples():
        typer.echo(f"  [{o.id}] " + _row(o))


@app.command("paper-bet")
def paper_bet(opportunity_id: int, stake: float = typer.Option(None, help="Hypothetical stake; default from settings."),
              allow_duplicate: bool = typer.Option(False, help="Make a separate entry even if this bet is already saved.")) -> None:
    """Save an opportunity as a simulated bet at the price that was observed."""
    settings = get_settings()
    engine = db.make_engine(settings.database_url)
    db.init_db(engine)
    with Session(engine) as session, session.begin():
        row = session.get(db.OpportunityRow, opportunity_id)
        if row is None:
            raise typer.BadParameter(f"no opportunity {opportunity_id}")
        bet, created = db.place_paper_bet(session, row, stake=stake or settings.paper_stake, allow_duplicate=allow_duplicate)
        typer.echo(("Saved" if created else "Already saved as") + f" paper bet {bet.id}: {bet.player} "
                   f"{describe(bet.side, bet.line)} at {book_label(bet.book)} {bet.price:+d}, stake {bet.stake:g} (simulated).")


@app.command()
def quota() -> None:
    """Odds API credits remaining (uses a free endpoint)."""
    settings = get_settings()

    async def check() -> None:
        async with OddsApiClient(settings.api_key() or "", base_url=settings.odds_api_base_url) as client:
            await client.sports()
            typer.echo(f"Credits remaining: {client.quota.remaining}; used this period: {client.quota.used}. "
                       f"The Sniper stops at {settings.credit_floor}.")
    try:
        asyncio.run(check())
    except OddsApiError as error:
        typer.echo(f"Odds API: {error}")
        raise typer.Exit(1)


@app.command()
def dashboard(port: int = typer.Option(8517)) -> None:
    """Open the Streamlit dashboard."""
    page = Path(__file__).resolve().parent / "dashboard" / "app.py"
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(page), "--server.port", str(port)]))


if __name__ == "__main__":
    app()
