"""Replay the sharp-gap cash rules over the scanner's saved payloads and grade them at the sharp close.

Retrospective diagnostic, not prospective evidence: the payloads were captured under earlier rules
(FanDuel-only staking, a 180-minute window, a different book list), so what was saved decides what
can be replayed. It answers three questions on real 2026 prices before money depends on them:

- does a +3% gap seen 15-45 minutes out still carry EV at Pinnacle's last pre-kickoff quote, and
  does it hold better than a gap seen earlier;
- which of FanDuel, ESPN Bet and DraftKings supplies the volume and the quality;
- does the shadow model filter (sharp_strategy.SHADOW_FILTER) separate good gaps from bad, using
  only the forecast that existed when the gap was seen.

    python -m nfl_pipeline.modeling.sharp_replay [--season 2026]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline import sharp_math
from nfl_pipeline import sharp_strategy as strategy
from nfl_pipeline import sharp_watch as watch
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.markets import normalize_team

ROOT = Path(__file__).resolve().parents[3]
CLOSE_MAX_AGE_MINUTES = 20
LEAD_BUCKETS = ((15, 45, "T-45..T-15 (stake window)"), (45, 90, "T-90..T-45"), (90, 240, "T-240..T-90"),
                (0, 15, "T-15..kickoff"))


def load_payloads(conn, season: int) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='120s'")
        cur.execute("""SELECT r.fetched_at_utc, r.payload FROM raw.nfl_api_responses r
            WHERE r.provider=%s AND r.endpoint='nfl_player_props'
              AND r.as_of_date >= make_date(%s, 3, 1) AND r.as_of_date < make_date(%s + 1, 3, 1)
            ORDER BY r.fetched_at_utc""", (watch.PROVIDER, season, season))
        return [dict(r) for r in cur.fetchall()]


def load_games(conn, season: int) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT game_id, season, week, home_team_abbr, away_team_abbr, start_ts_utc
            FROM raw.nfl_games WHERE season=%s""", (season,))
        return {(r["home_team_abbr"], r["away_team_abbr"], safe_time(r["start_ts_utc"])): dict(r) for r in cur.fetchall()}


def match_game(payload: dict, games: dict) -> dict | None:
    """Same matchup, kickoff within sharp_watch's tolerance (the API moves it to the actual kickoff)."""
    home, away = normalize_team(payload.get("home_team") or ""), normalize_team(payload.get("away_team") or "")
    return next((g for (h, a, start), g in games.items()
                 if h == home and a == away and watch.same_kickoff(payload.get("commence_time"), start)), None)


def decisions(payloads: list[dict], games: dict) -> tuple[list[dict], dict]:
    """First +3% gap per player-game-stat in each lead bucket, best book first, as sharp_watch stakes."""
    seen: dict[tuple, dict] = {}
    by_event: dict[str, list] = defaultdict(list)
    skipped = defaultdict(int)
    for record in payloads:
        payload, fetched = record["payload"], safe_time(record["fetched_at_utc"])
        game = match_game(payload, games)
        if game is None:
            skipped["no_matching_game"] += 1
            continue
        kick = safe_time(game["start_ts_utc"])
        by_event[game["game_id"]].append((fetched, payload))
        minutes = (kick - fetched).total_seconds() / 60
        for edge in watch.find_edges(payload, min_ev=strategy.MIN_EV, bet_books=strategy.BET_BOOKS, now=fetched):
            if edge["stat"] not in strategy.STAKED_STATS or not edge["player_norm"]:
                continue
            bucket = next((label for lo, hi, label in LEAD_BUCKETS if lo <= minutes <= hi), None)
            if bucket is None:
                continue
            key = (game["game_id"], edge["stat"], edge["player_norm"], bucket)
            if key not in seen:  # edges are EV-ranked across books within one payload
                seen[key] = dict(edge, game_id=game["game_id"], season=game["season"], week=game["week"],
                                 home=game["home_team_abbr"], away=game["away_team_abbr"], kickoff=kick,
                                 decided_at=fetched, minutes_to_kickoff=minutes, bucket=bucket)
    return list(seen.values()), dict(by_event=by_event, skipped=dict(skipped))


def sharp_close(decision: dict, snapshots: list) -> tuple[float | None, str | None]:
    """(EV of the bet price at Pinnacle's last fresh pre-kickoff quote, method)."""
    kick = decision["kickoff"]
    best = None
    for fetched, payload in snapshots:
        if not (decision["decided_at"] <= fetched < kick) or (kick - fetched).total_seconds() > CLOSE_MAX_AGE_MINUTES * 60:
            continue
        for book in payload.get("bookmakers", []):
            if book.get("key") != "pinnacle":
                continue
            for (stat, who, line), quote in watch._pairs(book).items():
                if stat != decision["stat"] or who != decision["player_norm"] or watch.quote_error(quote, fetched):
                    continue
                rank = (fetched, -abs(line - float(decision["line"])))
                if best is None or rank > best[0]:
                    best = (rank, line, quote)
    if best is None:
        return None, None
    _, line, quote = best
    p_over = sharp_math.no_vig_over(quote["over"], quote["under"])
    if abs(line - float(decision["line"])) > 1e-9:
        p_over, method = sharp_math.shift_probability(decision["stat"], line, p_over, float(decision["line"])), "shifted"
    else:
        method = "exact"
    if p_over is None:
        return None, None
    fair = p_over if decision["side"] == "over" else 1 - p_over
    return sharp_math.ev(fair, decision["price"]), method


def attach_forecasts(conn, rows: list[dict]) -> None:
    """The forecast that existed when the gap was seen, never a later one."""
    with conn.cursor() as cur:
        for row in rows:
            cur.execute("""SELECT projection, line, over_probability, under_probability, injury_report_status,
                    roster_status, limited_usage_risk FROM bets.nfl_player_prop_predictions
                WHERE game_id=%s AND stat=%s AND offer_player_name_norm=%s AND created_at_utc <= %s
                ORDER BY created_at_utc DESC, id DESC LIMIT 1""",
                        (row["game_id"], row["stat"], row["player_norm"], row["decided_at"]))
            found = cur.fetchone()
            forecast = dict(zip(watch.FORECAST_FIELDS, found)) if found else None
            row["model_filter"], row["model_filter_reasons"] = watch.shadow_filter(row, forecast)


def attach_results(conn, rows: list[dict]) -> None:
    column = {"receiving_yards": "receiving_yards", "rushing_yards": "rushing_yards", "receptions": "receptions"}
    with conn.cursor() as cur:
        for row in rows:
            cur.execute(f"""SELECT l.{column[row['stat']]}::float, r.team_abbr,
                    (COALESCE(l.offense_snaps,0)>0 OR COALESCE(l.pass_attempts,0)+COALESCE(l.carries,0)+COALESCE(l.targets,0)>0)
                FROM raw.nfl_rosters r JOIN raw.nfl_player_gamelogs l
                  ON l.player_id=r.player_id AND l.game_id=%s AND l.team_abbr=r.team_abbr
                JOIN raw.nfl_games g ON g.game_id=l.game_id
                WHERE r.season=%s AND r.player_name_norm=%s AND r.team_abbr IN (%s,%s) AND g.status='final'
                LIMIT 1""", (row["game_id"], row["season"], row["player_norm"], row["home"], row["away"]))
            found = cur.fetchone()
            actual, team, played = found if found else (None, None, False)
            row["team"] = team
            if actual is None or not played:
                row["units"] = None
                continue
            diff = (actual - float(row["line"])) * (1 if row["side"] == "over" else -1)
            price = float(row["price"])
            row["units"] = 0.0 if diff == 0 else (price / 100 if price > 0 else 100 / -price) if diff > 0 else -1.0


def summarize(frame: pd.DataFrame) -> dict:
    def stats(g):
        ev = pd.to_numeric(g.ev_at_sharp_close, errors="coerce").dropna()
        units = pd.to_numeric(g.units, errors="coerce").dropna()
        return dict(decisions=int(len(g)), with_close=int(len(ev)),
                    close_coverage=float(len(ev) / len(g)) if len(g) else None,
                    ev_when_seen=float(g.ev.mean()) if len(g) else None,
                    ev_at_sharp_close=float(ev.mean()) if len(ev) else None,
                    still_positive=float((ev > 0).mean()) if len(ev) else None,
                    settled=int(len(units)), roi=float(units.mean()) if len(units) else None)
    if frame.empty:
        return dict(by_bucket={}, stake_window={})
    window = frame[frame.bucket.eq(LEAD_BUCKETS[0][2])]
    out = dict(by_bucket={b: stats(g) for b, g in frame.groupby("bucket")}, stake_window=stats(window),
               by_book={b: stats(g) for b, g in window.groupby("book")},
               by_stat={s: stats(g) for s, g in window.groupby("stat")},
               by_week={f"{s}-{w}": stats(g) for (s, w), g in window.groupby(["season", "week"])},
               by_close_method={m: stats(g) for m, g in window.groupby(window.close_method.fillna("none"))},
               shadow_filter={f: stats(g) for f, g in window.groupby(window.model_filter.fillna("none"))})
    flagged = window[window.model_filter.eq("flag")]
    reasons = defaultdict(int)
    for values in flagged.model_filter_reasons:
        for reason in values or []:
            reasons[reason] += 1
    out["flag_reasons"] = dict(reasons)
    # What the strategy would actually have staked: caps applied in decision order.
    staked, per_day, per_game, keys = [], defaultdict(int), defaultdict(int), set()
    for row in window.sort_values(["decided_at", "ev"], ascending=[True, False]).to_dict("records"):
        day = row["decided_at"].astimezone(watch._ET).date()
        key = (row["game_id"], row["stat"], row["player_norm"])
        if key in keys or per_day[day] >= strategy.MAX_BETS_PER_DAY or per_game[row["game_id"]] >= strategy.MAX_BETS_PER_GAME:
            continue
        keys.add(key); per_day[day] += 1; per_game[row["game_id"]] += 1
        staked.append(row)
    out["capped_stakes"] = stats(pd.DataFrame(staked)) if staked else stats(window.iloc[:0])
    return out


def pct(v):
    return "-" if v is None else f"{v:.1%}"


def signed(v):
    return "-" if v is None else f"{v:+.2%}"


def markdown(meta: dict, summary: dict) -> str:
    header = "| | Decisions | Close coverage | EV when seen | EV at sharp close | Still +EV | Settled | Flat ROI |"
    rule = "|---|---:|---:|---:|---:|---:|---:|---:|"
    row = lambda name, s: (f"| {name} | {s['decisions']} | {pct(s['close_coverage'])} | {signed(s['ev_when_seen'])} | "
                           f"{signed(s['ev_at_sharp_close'])} | {pct(s['still_positive'])} | {s['settled']} | {signed(s['roi'])} |")
    lines = ["# NFL Sharp-Gap Replay", "",
             f"Built {meta['built_at']}. Retrospective: {meta['payloads']} saved scanner payloads "
             f"({meta['first']} to {meta['last']}), replayed with the {strategy.EXPERIMENT_NAME} rules "
             f"(+{strategy.MIN_EV:.0%} vs Pinnacle no-vig, {', '.join(strategy.BET_BOOKS)}; "
             f"{', '.join(sorted(strategy.STAKED_STATS))}). Graded at Pinnacle's last fresh quote within "
             f"{CLOSE_MAX_AGE_MINUTES} minutes of kickoff. Not prospective evidence: the payloads were captured under "
             "earlier rules and book lists, so absent books or polls cannot be replayed.", "",
             "## When the gap was seen", "", header, rule]
    for _, _, label in LEAD_BUCKETS:
        if label in summary["by_bucket"]:
            lines.append(row(label, summary["by_bucket"][label]))
    for title, key in (("By book (stake window)", "by_book"), ("By market (stake window)", "by_stat"),
                       ("By week (stake window)", "by_week"), ("Sharp close method (stake window)", "by_close_method"),
                       ("Shadow model filter (stake window)", "shadow_filter")):
        if summary.get(key):
            lines += ["", f"## {title}", "", header, rule]
            lines += [row(name, s) for name, s in sorted(summary[key].items())]
    if summary.get("flag_reasons"):
        lines += ["", "Flag reasons: " + ", ".join(f"{k} {v}" for k, v in sorted(summary["flag_reasons"].items()))]
    lines += ["", "## What the limits would have staked", "", header, rule,
              row(f"{strategy.MAX_BETS_PER_DAY}/day, {strategy.MAX_BETS_PER_GAME}/game, one per player-game-stat",
                  summary["capped_stakes"]), "",
              "Flat ROI is descriptive: on this many bets a coin flip at -110 swings roughly +/-15 points."]
    return "\n".join(lines) + "\n"


def run(season: int) -> dict:
    with psycopg2.connect(PG_DSN) as conn:
        conn.set_session(readonly=True)
        payloads = load_payloads(conn, season)
        games = load_games(conn, season)
        rows, context = decisions(payloads, games)
        for row in rows:
            row["ev_at_sharp_close"], row["close_method"] = sharp_close(row, context["by_event"].get(row["game_id"], []))
        attach_forecasts(conn, rows)
        attach_results(conn, rows)
    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    meta = dict(built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), season=season,
                payloads=len(payloads), skipped=context["skipped"],
                first=str(min((p["fetched_at_utc"] for p in payloads), default="")),
                last=str(max((p["fetched_at_utc"] for p in payloads), default="")),
                experiment_id=strategy.EXPERIMENT_NAME, retrospective=True, prospective_evidence=False)
    atomic_json(ROOT / "reports" / f"nfl_sharp_replay_{season}.json", dict(meta=meta, summary=summary))
    (ROOT / "reports" / f"nfl_sharp_replay_{season}.md").write_text(markdown(meta, summary), encoding="utf-8")
    return dict(meta=meta, summary=summary)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2026)
    args = ap.parse_args()
    result = run(args.season)
    print((ROOT / "reports" / f"nfl_sharp_replay_{args.season}.md").read_text(encoding="utf-8"))
    print(json.dumps(result["meta"], indent=2, default=str))


if __name__ == "__main__":
    main()
