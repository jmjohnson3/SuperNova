"""Command line for Sports Betting Sniper. Research and paper trading only; nothing here places a wager.

  python -m sports_sniper.cli scan                 # saved real snapshots if there are any, else sample data
  python -m sports_sniper.cli scan --source live   # The Odds API now (spends credits)
  python -m sports_sniper.cli top                  # the latest scan's ranked opportunities
  python -m sports_sniper.cli paper-bet 12         # save opportunity 12 as a paper bet
  python -m sports_sniper.cli settle               # grade finished paper bets and add closing line value
  python -m sports_sniper.cli watch                # one scheduled pass: new saved snapshots, paper bets, settlement
  python -m sports_sniper.cli backtest             # replay stored snapshots: rules, walk-forward, bankroll, model vs market
  python -m sports_sniper.cli bankroll --staking kelly   # hypothetical bankroll with stake and exposure limits
  python -m sports_sniper.cli quota                # credits left (free call)
  python -m sports_sniper.cli dashboard            # open the Streamlit dashboard
"""
from __future__ import annotations

import asyncio
import subprocess
import time
from datetime import datetime
import sys
from pathlib import Path

import typer
from sqlalchemy.orm import Session

from sports_sniper import db, performance
from sports_sniper.config import get_settings
from sports_sniper.markets import book_label, market_label
from sports_sniper.opportunities import describe
from sports_sniper.providers.odds_api import OddsApiClient, OddsApiError
from sports_sniper.scanner import SOURCES, run_scan, run_watch
from sports_sniper.settlement import settle as settle_bets

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
         limit: int = typer.Option(10, help="How many opportunities to print."),
         lookback_hours: float = typer.Option(None, help="Saved snapshots: how far back to read (default 36).")) -> None:
    """Collect prices, store them, and rank the opportunities."""
    if source not in SOURCES:
        raise typer.BadParameter(f"source must be one of: {', '.join(SOURCES)}")
    settings = get_settings()
    result = run_scan(settings, db.make_engine(settings.database_url), source, sports=sport or None,
                      lookback_hours=lookback_hours)
    label = {"sample": "HYPOTHETICAL SAMPLE DATA", "saved": "saved real snapshots", "live": "live prices"}[result.source]
    typer.echo(f"Scan {result.scan_id} ({label}): {result.events} games, {result.quotes} prices ({result.new_quotes} new), "
               f"{len(result.current)} flagged.")
    if result.credits_used is not None:
        typer.echo(f"Credits used {result.credits_used}; remaining {result.credits_remaining}.")
    if result.note:
        typer.echo(f"Note: {result.note}")
    if result.paper_bets:
        typer.echo(f"{result.paper_bets} new paper bets entered automatically (simulated).")
    dropped = {k: v for k, v in result.dropped.items() if v and k != "not_player_market"}
    if dropped:
        typer.echo("Left out: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(dropped.items())))
    shown = [o for o in result.current if o.status == "priced" and o.conservative_ev > settings.min_conservative_ev
             and o.ev >= settings.min_ev]
    typer.echo(f"\n{len(shown)} with an estimated edge of {settings.min_ev:.0%} or more" + (":" if shown else ". Nothing clears the bar."))
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
        frame = frame[(frame.conservative_ev > settings.min_conservative_ev) & (frame.ev >= settings.min_ev)]
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
def settle(refresh: bool = typer.Option(True, help="Read the latest saved snapshots first, so closing prices are stored.")) -> None:
    """Grade paper bets whose games are final and measure each against the closing price."""
    settings = get_settings()
    engine = db.make_engine(settings.database_url)
    if refresh:
        run_scan(settings, engine, "saved")
    counts = settle_bets(engine, settings)
    done = {k: counts.get(k, 0) for k in ("won", "lost", "push", "void")}
    typer.echo(f"Checked {counts.get('checked', 0)} open paper bets: " + ", ".join(f"{v} {k}" for k, v in done.items())
               + f", {counts.get('open', 0)} still open. Closing values added: {counts.get('closing_values', 0)}.")
    if counts.get("results_unavailable"):
        typer.echo("The results database could not be read, so nothing was graded.")
    bets = db.load_paper_bets(engine)
    bets = bets[bets.data_source != "sample"]
    for name, group in bets.groupby("strategy"):
        s = performance.summarize(group)
        typer.echo(f"{name}: {s['bets']} bets, {s['won']}-{s['lost']}-{s['push']} ({s['void']} void, {s['open']} open), "
                   f"simulated profit {s['profit']:+.2f} on {s['staked']:.0f} staked ({_pct(s['roi'])}), "
                   f"closing value {_pct(s['closing_value'])} on {s['closing_count']}.")


@app.command()
def watch(lookback_hours: float = typer.Option(3.0, help="How far back to look for saved snapshots not stored yet.")) -> None:
    """One scheduled pass: store new saved snapshots, enter the bets they flag, settle finished ones. No credits."""
    settings = get_settings()
    engine = db.make_engine(settings.database_url)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    began = time.monotonic()

    def say(message: str) -> None:                  # each step is written as it happens, with seconds since the start
        typer.echo(f"{stamp} +{time.monotonic() - began:.0f}s {message}")
    say("pass started")
    try:
        result = run_watch(settings, engine, lookback_hours=lookback_hours, progress=say)
        if result is not None:
            say(f"scan {result.scan_id}: {result.events} games, {result.quotes} prices, {len(result.current)} flagged, "
                f"{result.paper_bets} new paper bets")
        counts = settle_bets(engine, settings)
    except Exception as error:                      # one line in the log; the next pass starts clean
        say(f"failed ({type(error).__name__})")
        raise typer.Exit(1)
    graded = {k: counts[k] for k in ("won", "lost", "push", "void") if counts.get(k)}
    say(f"settled {graded or 'nothing'}; closing values added {counts.get('closing_values', 0)}"
        + ("; results database unavailable" if counts.get("results_unavailable") else "") + "; done")


@app.command("backtest")
def backtest_command(min_ev: float = typer.Option(None, help="Also test a custom rule with this minimum estimated edge (0.04 = 4%)."),
                     book: list[str] = typer.Option(None, help="Custom rule: books to bet at; repeat for several."),
                     side: list[str] = typer.Option(None, help="Custom rule: over or under."),
                     sport: list[str] = typer.Option(None, help="Custom rule: nfl or nba."),
                     delay: int = typer.Option(0, help="Custom rule: 1 enters at the same book's price one look later."),
                     write: bool = typer.Option(True, help="Write reports/sniper_backtest_latest.md.")) -> None:
    """Replay the stored snapshots: what each rule would have done, walk-forward, bankroll, model against market."""
    from sports_sniper import backtest, research
    settings = get_settings()
    engine = db.make_engine(settings.database_url)
    db.init_db(engine)
    report = research.build(engine, settings, progress=lambda message: typer.echo(f"  {message}"))
    if report["found"].empty:
        typer.echo("No stored real prices to replay yet. Run: python -m sports_sniper.cli scan --source saved")
        raise typer.Exit()
    rows = list(report["strategies"])
    if any(v for v in (min_ev is not None, book, side, sport, delay)):
        custom = backtest.Strategy("Custom rule", min_ev=settings.min_ev if min_ev is None else min_ev, books=tuple(book or ()),
                                   sides=tuple(side or ()), sports=tuple(sport or ()), entry_delay=delay)
        rows.append((custom, backtest.metrics(backtest.run(report["found"], custom, tuple(settings.bet_books)))))
    start, end = report["span"]
    typer.echo(f"Snapshots {start:%b %d} to {end:%b %d}: {report['games']} games, {len(report['found'])} flagged prices. Simulated $1 bets.")
    for rule, m in rows:
        typer.echo(f"  {rule.name}: {m['won']}-{m['lost']}-{m['push']}, ROI {_pct(m['roi'])} (95% {research.interval(m['roi_interval'])}), "
                   f"closing value {_pct(m['closing_value'])} on {m['closing_count']}, worst fall {m['max_drawdown']:.1f} units")
    for objective, title in (("roi", "results"), ("closing_value", "closing value")):
        m = report["walk_forward"][objective]["metrics"]
        chosen = ", ".join(f"{f['chosen']:.0%}" for f in report["walk_forward"][objective]["folds"])
        typer.echo(f"Walk-forward (edge chosen on earlier days by {title}: {chosen or 'not enough days'}): "
                   f"{m['won']}-{m['lost']}-{m['push']}, ROI {_pct(m['roi'])} (95% {research.interval(m['roi_interval'])}), "
                   f"closing value {_pct(m['closing_value'])}")
    for name, b in report["bankroll"].items():
        typer.echo(f"Paper bankroll, {name}: ${b['start']:.0f} -> ${b['final']:.2f}, largest fall ${b['max_drawdown']:.2f}")
    for s in report["scores"].itertuples():
        typer.echo(f"Model against market on {int(s.lines)} {s.group.upper()} lines (Brier, lower is better): book {s.book_brier:.4f}, "
                   f"consensus {s.consensus_brier:.4f}, model {s.model_brier:.4f}")
    for b in report["breakout"].itertuples():
        if b.breakout:
            typer.echo(f"Breakout signals, {b.sport.upper()}: over won {b.over_rate:.1%} on {b.lines} lines, "
                       f"betting every over {_pct(b.over_roi)}")
    if write:
        typer.echo(f"Report: {research.write(report)}")
        saved = research.save_calibration(report, settings)
        if saved:
            typer.echo(f"Model calibration: {saved} ({', '.join(report['calibration'])})")


@app.command("bankroll")
def bankroll_command(staking: str = typer.Option("flat", help="flat | kelly"),
                     start: float = typer.Option(100.0, help="Hypothetical starting bankroll."),
                     stake: float = typer.Option(1.0, help="Flat stake."),
                     kelly: float = typer.Option(0.25, help="Share of the full Kelly stake."),
                     max_stake: float = typer.Option(5.0), max_daily: float = typer.Option(20.0),
                     max_per_game: float = typer.Option(3.0), stop_drawdown: float = typer.Option(None, help="Stop after this fall from the high (0.3 = 30%)."),
                     source: str = typer.Option("backtest", help="backtest (the automatic rule replayed) | paper (the paper bets on record)")) -> None:
    """Run settled bets through a hypothetical bankroll with stake and exposure limits. Simulation only."""
    from sports_sniper import backtest, bankroll
    if staking not in ("flat", "kelly") or source not in ("backtest", "paper"):
        raise typer.BadParameter("staking must be flat or kelly; source must be backtest or paper")
    settings = get_settings()
    engine = db.make_engine(settings.database_url)
    db.init_db(engine)
    if source == "paper":
        bets = db.load_paper_bets(engine)
        bets = bets[(bets.data_source != "sample") & (bets.strategy == db.STRATEGY_AUTO)]
    else:
        found, _ = backtest.candidates(engine, settings, cache=backtest.default_cache(settings))
        bets = backtest.run(found, backtest.presets(settings)[1], tuple(settings.bet_books))
    rules = bankroll.Rules(start=start, staking=staking, flat_stake=stake, kelly_fraction=kelly, max_stake=max_stake,
                           max_daily=max_daily, max_per_game=max_per_game, stop_drawdown=stop_drawdown)
    b = bankroll.simulate(bets, rules)
    typer.echo(f"Hypothetical ${b['start']:.2f} -> ${b['final']:.2f} over {b['placed']} bets (${b['staked']:.2f} staked, "
               f"{_pct(b['roi'])} on stakes). Largest fall ${b['max_drawdown']:.2f} ({b['max_drawdown_pct']:.0%}); lowest ${b['lowest']:.2f}.")
    if b["skipped"]:
        typer.echo("Left out: " + ", ".join(f"{n} {why}" for why, n in b["skipped"].items()))
    if b["stopped"]:
        typer.echo(f"Stopped placing bets after {b['stopped']} (drawdown limit).")


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
