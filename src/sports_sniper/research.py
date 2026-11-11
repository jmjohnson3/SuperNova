"""Phase 3 report: backtest of the rules, walk-forward, bankroll simulation and model against market, in one place.

`build` works everything out from the stored snapshots; `markdown` turns it into the text written to
reports/sniper_backtest_latest.md. Every number is history on paper, with its uncertainty beside it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from sports_sniper import backtest, bankroll, breakout, calibration, ladder, model_compare, parlay
from sports_sniper.config import Settings
from sports_sniper.markets import market_label
from sports_sniper.providers import models

GRID = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "sniper_backtest_latest.md"


def save_calibration(report: dict, settings: Settings) -> Path | None:
    """Keep the latest calibration fit where the scanner reads it. Nothing is written without a fit."""
    path = calibration.default_path(settings.database_url)
    if path is not None and report.get("calibration"):
        calibration.save(report["calibration"], path)
        return path
    return None


def with_models(found: pd.DataFrame, offers: pd.DataFrame, settings: Settings, adapters=None):
    """(flagged prices with the model's view, lines with every probability source, the calibration fits by day)."""
    offers, fits = model_compare.attach_models(offers, settings, adapters)
    values, versions = models.over_probabilities(settings.results_dsn, found, adapters or models.ADAPTERS) if len(found) else ([], [])
    return calibration.annotate(found, fits, values, versions), offers, fits


def ladder_check(engine: Engine, offers: pd.DataFrame, settings: Settings, load_results=None, extra=None) -> dict:
    """The longshot ladder model scored walk-forward on every stored alternate line."""
    if extra is None:
        try:
            extra = ladder.nfl_line_history(settings.results_dsn)
        except Exception:                    # the line history is optional
            extra = None
    ratios = ladder.combined_library(offers, extra)
    return ladder.summary(ladder.evaluate(backtest.load_prices(engine), ratios, settings, load_results), settings.min_ev,
                          settings.longshot_min_american)


def due(settings: Settings, now: datetime | None = None) -> bool:
    """Whether the scheduled pass should refresh the evaluation: no report yet, or the last one is old enough."""
    marker = calibration.default_path(settings.database_url)
    if marker is None:
        return False
    marker = marker.with_name("evaluation.json")
    if not marker.exists():
        return True
    age = (now or datetime.now(timezone.utc)).timestamp() - marker.stat().st_mtime
    return age >= settings.evaluate_every_hours * 3600


def evaluate(engine: Engine, settings: Settings, *, progress=None, write_report: bool = True, **sources) -> dict:
    """The automatic evaluation: rebuild the report, refit the model calibration on every settled line, and
    re-measure the parlay links. Returns a short summary and leaves a marker so `due` knows when it last ran."""
    report = build(engine, settings, progress=progress, **sources)
    summary = dict(at=report["built_at"].isoformat(), flagged=len(report["found"]), games=report["games"],
                   calibration={k: dict(b=v["b"], blend_c=v["blend_c"], lines=v["lines"]) for k, v in report["calibration"].items()})
    if not report["found"].empty and write_report:
        summary["report"] = str(write(report))
    saved = save_calibration(report, settings)
    summary["calibration_saved"] = saved is not None
    links = parlay.default_path(settings.database_url)
    if links is not None:
        try:
            table = parlay.build(settings.results_dsn)
            if table:
                parlay.save(table, links)
            summary["parlay_links"] = {sport: len(found) for sport, found in table.items()}
        except Exception as error:           # the game-log database is optional
            summary["parlay_links"] = f"unavailable ({type(error).__name__})"
        marker = links.with_name("evaluation.json")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def build(engine: Engine, settings: Settings, *, progress=None, load_results=None, adapters=None, history=None, line_history=None) -> dict:
    found, offers = backtest.candidates(engine, settings, cache=backtest.default_cache(settings), progress=progress,
                                        load_results=load_results)
    found, offers, _ = with_models(found, offers, settings, adapters)
    latest = calibration.fit_latest(offers)
    if history is None:
        try:
            history = breakout.load_history(settings.results_dsn)
        except Exception:                    # the statistics database is optional
            history = {}
    books = tuple(settings.bet_books)
    rules = backtest.presets(settings)
    main = rules[1]
    bets = backtest.run(found, main, books)
    forward = {objective: backtest.walk_forward(found, main, GRID, default_books=books, objective=objective)
               for objective in ("roi", "closing_value")}
    return dict(
        built_at=datetime.now(timezone.utc), found=found, offers=offers, main=main, bets=bets,
        span=(None, None) if found.empty else (found.observed_at.min(), found.observed_at.max()),
        games=0 if found.empty else int(found.event_id.nunique()),
        strategies=backtest.compare(found, rules, books),
        thresholds=[(t, backtest.metrics(backtest.run(found, backtest.replace(main, min_ev=t), books))) for t in GRID],
        periods=backtest.by_period(bets),
        markets=[(market_label(k), backtest.metrics(g)) for k, g in bets.groupby("market")] if len(bets) else [],
        walk_forward={k: dict(folds=v["folds"], metrics=backtest.metrics(v["bets"])) for k, v in forward.items()},
        bankroll={"Flat $1": bankroll.simulate(bets, bankroll.Rules()),
                  "Quarter Kelly": bankroll.simulate(bets, bankroll.Rules(staking="kelly"))},
        scores=model_compare.scores(offers, "sport"), scores_by_market=model_compare.scores(offers, "market"),
        scores_calibrated=model_compare.scores(offers, "sport", tuple(model_compare.SOURCES)),
        picks=model_compare.picks_table(offers), calibration=latest,
        breakout=breakout.validate(offers, history),
        ladder=ladder_check(engine, offers, settings, load_results, line_history) if len(offers) else ladder.summary(pd.DataFrame()),
        model_lines=int(offers.model_probability.notna().sum()) if len(offers) else 0)


def pct(value, signed: bool = True) -> str:
    return "-" if value is None or pd.isna(value) else (f"{value:+.1%}" if signed else f"{value:.1%}")


def interval(pair) -> str:
    return "-" if pair is None else f"{pair[0]:+.0%} to {pair[1]:+.0%}"


def metric_row(name: str, m: dict) -> str:
    return (f"| {name} | {m['won'] + m['lost'] + m['push']} | {m['won']}-{m['lost']}-{m['push']} | {pct(m['roi'])} | "
            f"{interval(m['roi_interval'])} | {pct(m['estimated_ev'])} | {pct(m['closing_value'])} ({m['closing_count']}) | "
            f"{m['max_drawdown']:.1f} | {m['longest_losing_streak']} |")


HEAD = ("| {} | Settled | W-L-P | ROI | 95% interval | Est. EV | Closing value (bets) | Max drawdown (units) | Longest losing run |\n"
        "|---|---|---|---|---|---|---|---|---|")


def markdown(r: dict) -> str:
    start, end = r["span"]
    lines = ["# Sports Betting Sniper: backtest", "",
             f"Built {r['built_at']:%Y-%m-%d %H:%M} UTC. Research on paper only: simulated $1 bets at prices that were stored, "
             "graded from final statistics. A past result is not a forecast.", ""]
    if start is None:
        return "\n".join(lines + ["No stored real prices yet."]) + "\n"
    lines += [f"Stored snapshots {start:%b %d} to {end:%b %d, %Y}, {r['games']} games, {len(r['found'])} flagged prices.", "",
              "## Rules compared", "", HEAD.format("Rule")]
    lines += [metric_row(s.name, m) for s, m in r["strategies"]]
    lines += ["", "ROI is profit over stakes; the interval treats whole games as the draws. Closing value is the edge of the "
              "price taken against the fair price at the last look before the game. \"One look later\" enters at the same "
              "book's price about ten minutes on and drops the bet when the line is gone or no longer clears the bar.", "",
              "## By minimum estimated edge (bet books)", "", HEAD.format("Minimum edge")]
    lines += [metric_row(f"{t:.0%}+", m) for t, m in r["thresholds"]]
    lines += ["", "These rows overlap and were all measured on the same bets, so the best-looking row is partly luck. "
              "The walk-forward below is the fair version.", "", "## Walk-forward", ""]
    for objective, title in (("roi", "results (ROI)"), ("closing_value", "closing value")):
        w = r["walk_forward"][objective]
        lines += [f"Minimum edge chosen on earlier days by {title}, applied to the next three game days:", "",
                  "| Later days | Edge chosen | Settled | W-L-P | ROI | Closing value |", "|---|---|---|---|---|---|"]
        for f in w["folds"]:
            lines.append(f"| {f['test_from']:%b %d} to {f['test_to']:%b %d} | {f['chosen']:.0%}{'' if f['tuned'] else ' (default)'} | "
                         f"{f['test_won'] + f['test_lost'] + f['test_push']} | {f['test_won']}-{f['test_lost']}-{f['test_push']} | "
                         f"{pct(f['test_roi'])} | {pct(f['test_closing_value'])} |")
        m = w["metrics"]
        lines += [f"| **All later days** | | {m['won'] + m['lost'] + m['push']} | {m['won']}-{m['lost']}-{m['push']} | "
                  f"{pct(m['roi'])} ({interval(m['roi_interval'])}) | {pct(m['closing_value'])} |", ""]
    main = r["main"].name
    lines += [f"## {main}: by period", "", HEAD.format("Days")] + [metric_row(n, m) for n, m in r["periods"]]
    lines += ["", f"## {main}: by market", "", HEAD.format("Market")] + [metric_row(n, m) for n, m in r["markets"]]
    lines += ["", f"## Paper bankroll ({main})", "",
              "| Staking | Start | End | Staked | Return on stakes | Bets placed | Largest fall | Lowest point | Left out |", "|---|---|---|---|---|---|---|---|---|"]
    for name, b in r["bankroll"].items():
        left = ", ".join(f"{n} {why}" for why, n in b["skipped"].items()) or "none"
        lines.append(f"| {name} | ${b['start']:.0f} | ${b['final']:.2f} | ${b['staked']:.2f} | {pct(b['roi'])} | {b['placed']} | "
                     f"${b['max_drawdown']:.2f} ({b['max_drawdown_pct']:.0%}) | ${b['lowest']:.2f} | {left} |")
    lines += ["", "Hypothetical $100. Limits: $5 a bet, $20 a day, $3 a game, $2 a player. Quarter Kelly stakes a quarter of "
              "the Kelly amount on the probability lowered by its uncertainty.", "", "## Model against market (NBA)", ""]
    if r["scores"].empty:
        lines += ["No settled lines with a model forecast made before the price was seen.", ""]
    else:
        lines += [f"{r['model_lines']} two-sided lines had a model forecast made before the price was seen. Lower is better.", "",
                  "| Lines | Settled | Bet book's own price | Market consensus | Prediction model |", "|---|---|---|---|---|"]
        for s in pd.concat([r["scores"], r["scores_by_market"]]).itertuples():
            name = s.group.upper() if s.group in ("nba", "nfl", "mlb") else market_label(s.group)
            lines.append(f"| {name} (Brier) | {s.lines} | {s.book_brier:.4f} | {s.consensus_brier:.4f} | {s.model_brier:.4f} |")
        if not r["scores_calibrated"].empty:
            lines += ["", "With the model calibrated on earlier days only (the lines after the first fit):", "",
                      "| Lines | Settled | Bet book | Market consensus | Model | Model, calibrated | Market moved by the model |", "|---|---|---|---|---|---|---|"]
            for s in r["scores_calibrated"].itertuples():
                lines.append(f"| {s.group.upper()} (Brier) | {s.lines} | {s.book_brier:.4f} | {s.consensus_brier:.4f} | {s.model_brier:.4f} | "
                             f"{s.calibrated_brier:.4f} | {s.blend_brier:.4f} |")
        for sport, c in r["calibration"].items():
            lines += ["", f"{sport.upper()} calibration on {c['lines']} settled lines: the model's log-odds are multiplied by "
                      f"{c['b']:.2f} (1 would mean its confidence is right, 0 that it says nothing); set against the market, its "
                      f"disagreement gets a weight of {c['blend_c']:.2f} (0 means the market already knows it)."]
        lines += ["", "Betting the side each source says is underpriced, at the bet book's price, on those same lines:", "",
                  "| Source | Edge needed | Settled | W-L | ROI | Its own estimate |", "|---|---|---|---|---|---|"]
        for p in r["picks"].itertuples():
            se = "" if pd.isna(p.roi_se) else f" ± {p.roi_se:.1%}"
            lines.append(f"| {model_compare.SOURCES[p.source]} | {p.threshold:.0%}+ | {p.bets} | {p.won}-{p.lost} | "
                         f"{pct(p.roi)}{se} | {pct(p.estimated_ev)} |")
        lines += ["", "The NFL model is compared on receptions and passing touchdowns only: its yardage projections are "
                  "anchored to the line, so they carry no chance of their own."]
    lines += ["", "## Breakout signals", ""]
    if r["breakout"].empty:
        lines += ["No settled lines to test the signals on."]
    else:
        lines += ["Lines on players whose recent opportunity was above their season baseline (games before that day only), "
                  "against every other line. Over ROI is betting every over at the bet book's price.", "",
                  "| Sport | Lines | Count | Players | Over won | Market's chance of the over | Over ROI |", "|---|---|---|---|---|---|---|"]
        for b in r["breakout"].itertuples():
            se = "" if b.over_roi_se is None or pd.isna(b.over_roi_se) else f" ± {b.over_roi_se:.1%}"
            lines.append(f"| {b.sport.upper()} | {'Breakout signal' if b.breakout else 'No signal'} | {b.lines} | {b.players} | "
                         f"{pct(b.over_rate, False)} | {pct(b.market_over, False)} | {pct(b.over_roi)}{se} |")
    lines += ["", "## Longshot ladder model", ""]
    check = r["ladder"]
    if not check["lines"]:
        lines += ["No stored alternate lines could be priced and settled."]
    else:
        lines += [f"One-sided alternate lines priced from the sharp main line and the spread of past results, each using only "
                  f"results known before its game: {check['lines']} settled alternates from {check['games']} games. The NFL scanner "
                  "stopped collecting alternates on Oct 26, so this is a small set of games.", "",
                  "| Model's chance | Lines | Model said | Actually won | Book's break-even |", "|---|---|---|---|---|"]
        lines += [f"| {b['low']:.0%} to {b['high']:.0%} | {b['n']} | {b['predicted']:.1%} | {b['observed']:.1%} | {b['break_even']:.1%} |"
                  for b in check["calibration"]]
        lines += ["", "| Bets (one per player, market and line, at the best price) | Settled | W-L | ROI | The model's own estimate |", "|---|---|---|---|---|"]
        for name, m in check["groups"]:
            se = "" if m["roi_se"] is None else f" ± {m['roi_se']:.1%}"
            lines.append(f"| {name} | {m['won'] + m['lost'] + m['push']} | {m['won']}-{m['lost']} | {pct(m['roi'])}{se} | {pct(m['estimated_ev'])} |")
    return "\n".join(lines) + "\n"


def write(r: dict, path: Path = REPORT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown(r), encoding="utf-8")
    return path
