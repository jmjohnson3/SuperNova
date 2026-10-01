"""Run the NFL prediction path and optionally post to Discord."""
from __future__ import annotations

import argparse
import asyncio
import logging
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import psycopg2

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.subprocess_utils import run_subprocess
from nfl_pipeline.integrity import active_release, atomic_json, nfl_season, production_freeze
from supernovabets_config import _saved_windows_env
from nfl_pipeline.game_scope import ENV as SCOPE_ENV, validate_ids

log = logging.getLogger("nfl_pipeline.run_daily_and_notify")
_ET = ZoneInfo("America/New_York")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class Step:
    label: str
    module: str
    args: tuple[str, ...] = ()
    critical: bool = True
    post_output: bool = False
    timeout_s: int = 900


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_run_date(explicit_date: str | None) -> date:
    if explicit_date:
        return date.fromisoformat(explicit_date)
    today = datetime.now(_ET).date()
    lookahead_days = int(os.getenv("NFL_DEFAULT_DATE_LOOKAHEAD_DAYS", "14"))
    try:
        with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT MIN(game_date_et)
                FROM raw.nfl_games
                WHERE game_date_et >= %s
                  AND game_date_et <= %s
                  AND start_ts_utc > NOW()
                """,
                (today, today + timedelta(days=lookahead_days)),
            )
            value = cur.fetchone()[0]
            if value:
                if value != today:
                    log.info("No --date supplied; selected next scheduled NFL slate: %s", value)
                return value
    except Exception as exc:
        log.warning("Could not resolve next NFL slate date; using today %s: %s", today, exc)
    return today


def _webhook_url() -> str:
    return (
        os.getenv("NFL_DISCORD_WEBHOOK_URL")
        or _saved_windows_env("NFL_DISCORD_WEBHOOK_URL")
        or os.getenv("MLB_DISCORD_WEBHOOK_URL")
        or os.getenv("DISCORD_WEBHOOK_URL")
        or ""
    )


async def _post(content: str) -> None:
    await _post_payload({"content": content[:1950], "allowed_mentions": {"parse": []}})


async def _post_payload(payload: dict) -> None:
    webhook = _webhook_url()
    if not webhook:
        raise RuntimeError("NFL Discord webhook is not configured")
    async with httpx.AsyncClient(timeout=20) as client:
        for attempt in range(4):
            try:
                response = await client.post(webhook, params={"wait": "true"}, json=payload)
            except httpx.RequestError:
                raise RuntimeError('Discord delivery was not confirmed; not retrying an uncertain send') from None
            if response.status_code in {200, 204}:
                return
            if response.status_code != 429 or attempt == 3:
                raise RuntimeError(f"Discord post failed (HTTP {response.status_code})")
            retry_after = float(response.json().get('retry_after', 1))
            if not 0 <= retry_after <= 60:
                raise RuntimeError('Discord rate limit exceeds bounded retry window')
            await asyncio.sleep(max(.1, retry_after))


async def _post_matchups(stdout: str, publications: list) -> None:
    from nfl_pipeline.discord_matchups import CONTRACT
    bundle = json.loads(stdout)
    if bundle.get('contract') != CONTRACT or not isinstance(bundle.get('cards'), list):
        raise ValueError('Invalid NFL matchup publication bundle')
    if bundle['cards'] and (bundle.get('cash_readiness') or {}).get('status') == 'operational_issue':
        raise ValueError('Cash publication readiness failed; no fresh betting instructions will be published')
    for card in bundle['cards']:
        payload = card['payload']
        embeds = payload.get('embeds', [])
        if len(embeds) != 1 or len(embeds[0].get('description', '')) > 4096 or len(embeds[0].get('title', '')) > 256:
            raise ValueError('Invalid NFL matchup embed')
        embed = embeds[0]
        footer = embed.get('footer', {}).get('text', '')
        if len(footer) > 2048 or sum(len(s) for s in (embed.get('title', ''), embed.get('description', ''), footer)) > 6000:
            raise ValueError('NFL matchup embed exceeds the total character limit')
        if payload.get('allowed_mentions') != {'parse': []}:
            raise ValueError('NFL matchup cards must disable automatic mentions')
    for card in bundle['cards']:
        from nfl_pipeline.context_contract import safe_time
        now = datetime.now(timezone.utc)
        for row in card.get('forecast_manifest', []):
            if row.get('cash_ledger_id') in card.get('cash_ledger_ids', []):
                expiry = safe_time(row.get('cash_expires_at'))
                if not expiry or now >= expiry:
                    raise ValueError('Cash quote expired before publication; reservation requires reconciliation')
        await _post_payload(card['payload'])
        publications.append({'section': 'NFL Matchup', 'game_id': card['game_id'], 'page': card['page'],
            'payload': card['payload'], 'forecast_manifest':card.get('forecast_manifest',[]),
            'sent_at': datetime.now(timezone.utc).isoformat()})
        from nfl_pipeline.cash_execution import publication
        publication(card.get('cash_ledger_ids', []))
    if not bundle['cards']:
        await _post_section('**NFL Matchups**', bundle.get('notice') or 'No upcoming matchup cards.')


async def _post_section(header: str, body: str) -> None:
    lines = body.strip().splitlines() or ["(no output)"]
    current = header
    for line in lines:
        if len(current) + len(line) + 1 > 1900:
            await _post(current)
            current = line
        else:
            current = f"{current}\n{line}"
    if current:
        await _post(current)


def _run(step: Step) -> tuple[int, str, str]:
    env = os.environ.copy()
    src_dir = str(_repo_root() / "src")
    env["PYTHONPATH"] = src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONIOENCODING"] = "utf-8"
    if step.post_output:
        env["DISCORD_FORMAT"] = "1"
    return run_subprocess(
        [sys.executable, "-m", step.module, *step.args],
        cwd=str(_repo_root()),
        env=env,
        timeout_s=step.timeout_s,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run NFL daily prediction pipeline with Discord")
    parser.add_argument("--date", default=None, help="ET date YYYY-MM-DD")
    parser.add_argument("--skip-crawl", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument('--pregame', action='store_true', help='Fast game-scoped context/odds/prediction/publication run')
    parser.add_argument('--game-id', action='append', default=[])
    parser.add_argument('--run-id', help='Unique operational attempt ID for the parent scheduler')
    args = parser.parse_args()
    if args.pregame and (not args.date or not args.game_id or args.skip_crawl or args.skip_predict):
        parser.error('--pregame requires --date and --game-id, with odds refresh and prediction enabled')
    if args.run_id and not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}',args.run_id):
        parser.error('Invalid run ID')
    if args.game_id:
        os.environ[SCOPE_ENV] = json.dumps(validate_ids(args.game_id))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    run_date = _resolve_run_date(args.date)
    date_args = ("--date", run_date.isoformat())
    context_year = str(nfl_season(run_date))
    context_args = ("--seasons", context_year)
    steps = [
        Step("NFL Schema", "nfl_pipeline.schema", timeout_s=180),
        Step("NFL Results Refresh", "nfl_pipeline.import_nflverse", args=context_args, timeout_s=600),
        Step("NFL Context Import", "nfl_pipeline.import_context", args=context_args, critical=False, timeout_s=600),
        Step("NFL Usage Context Import", "nfl_pipeline.import_usage_context", args=context_args, critical=False, timeout_s=1200),
        # Same-day ESPN statuses (incl. game-day inactives at T-90); nflverse injury files lag days.
        Step("NFL Live Injury Report", "nfl_pipeline.live_injuries", args=date_args, critical=False, timeout_s=120),
        Step("NFL Validated Release", "nfl_pipeline.run_training", args=("--season",context_year,"--skip-context"), timeout_s=3600),
        # Missing/stale quotes only block locking (scoring and forecast_store refuse stale quotes);
        # the morning run still publishes projections. Pregame keeps it critical so it retries.
        Step("NFL Fresh Lock Quotes", "nfl_pipeline.refresh_lock_quotes", args=date_args, critical=False, timeout_s=600),
        Step("NFL Snapshot Health", "nfl_pipeline.snapshot_health_report", args=date_args, critical=False, timeout_s=120),
        Step("NFL Game Predictions", "nfl_pipeline.modeling.predict_today", args=date_args, timeout_s=600),
        Step("NFL Team Workload Snapshot", "nfl_pipeline.modeling.score_accuracy_components", args=(*date_args,"--capture-team-context"), critical=False, timeout_s=120),
        Step("NFL Player Predictions", "nfl_pipeline.modeling.predict_player_props", args=date_args, timeout_s=600),
        Step("NFL Lock Paper Ledger", "nfl_pipeline.lock_ledger", args=(*date_args,"--tier","paper","--max-rows","250","--stake","0"), timeout_s=120),
        Step("NFL Lock Micro Ledger", "nfl_pipeline.lock_ledger", args=(*date_args,"--tier","micro_projection","--max-rows","5","--stake","1"), timeout_s=120),
        Step("NFL Benchmark Offer Shadows", "nfl_pipeline.modeling.benchmark_offers", args=date_args, timeout_s=300),
        Step("NFL Receiving Trial Checkpoint", "nfl_pipeline.modeling.receiving_trial_checkpoint", args=date_args, timeout_s=180),
        Step("NFL Locked Micro Uncertainty", "nfl_pipeline.modeling.score_locked_micro_uncertainty", args=date_args, critical=False, timeout_s=120),
        Step("NFL Accepted Accuracy Components", "nfl_pipeline.modeling.score_accuracy_components", args=date_args, critical=False, timeout_s=180),
        Step("NFL Readiness Report", "nfl_pipeline.readiness_report", args=date_args, critical=False, timeout_s=180),
        Step("NFL Active Release Report", "nfl_pipeline.modeling.model_holdout_report", timeout_s=120),
        Step("NFL Cash Readiness", "nfl_pipeline.cash_readiness", args=date_args, timeout_s=180),
        Step("NFL Matchup Cards", "nfl_pipeline.publish_forecasts", args=(*date_args,"--kind","matchups","--json","--reserve-cash"), post_output=True, timeout_s=120),
    ]
    if args.pregame:
        omitted = {'nfl_pipeline.schema','nfl_pipeline.import_nflverse','nfl_pipeline.import_usage_context',
            'nfl_pipeline.run_training','nfl_pipeline.snapshot_health_report','nfl_pipeline.readiness_report',
            'nfl_pipeline.modeling.model_holdout_report','nfl_pipeline.modeling.score_locked_micro_uncertainty'}
        steps = [s for s in steps if s.module not in omitted and not (
            s.module=='nfl_pipeline.modeling.score_accuracy_components' and '--capture-team-context' not in s.args)]
        steps = [Step(s.label,s.module,s.args,True,s.post_output,s.timeout_s)
                 if s.module in ('nfl_pipeline.import_context','nfl_pipeline.refresh_lock_quotes') else s for s in steps]
    results, publications = [], []
    try:
        for step in steps:
            if args.skip_crawl and step.module == "nfl_pipeline.refresh_lock_quotes":
                continue
            if step.module == "nfl_pipeline.run_training" and (args.skip_train or production_freeze()):
                log.info("Using existing release; offline training is not part of this frozen live run")
                continue
            if args.skip_predict and (step.post_output or step.module in {
                "nfl_pipeline.modeling.predict_today", "nfl_pipeline.modeling.predict_player_props", "nfl_pipeline.modeling.score_accuracy_components", "nfl_pipeline.modeling.score_locked_micro_uncertainty", "nfl_pipeline.modeling.benchmark_offers", "nfl_pipeline.modeling.receiving_trial_checkpoint", "nfl_pipeline.lock_ledger"}):
                continue
            log.info("Starting %s",step.label)
            if step.module == 'nfl_pipeline.modeling.predict_today':
                release = active_release()
                if not release:
                    raise RuntimeError('A validated NFL release is required')
                os.environ['NFL_MODEL_RELEASE_ID'] = release['release_id']
            rc, stdout, stderr = _run(step)
            results.append({"step":step.label,"returncode":rc,"critical":step.critical,
                            "stdout_tail":stdout[-1500:],"stderr_tail":stderr[-1500:]})
            if rc:
                # Command output may contain authenticated provider URLs; keep
                # Discord failures concise and leave detail in local logs.
                await _post_section(f"FAILED **{step.label}**",f"Exit {rc}. No new bet card published from this failed step.")
                if step.critical:
                    break
                continue
            if step.post_output:
                await _post_matchups(stdout, publications)
            log.info("%s complete",step.label)
    except Exception as exc:
        results.append({"step":"Discord or runner","returncode":1,"critical":True,"error_type":type(exc).__name__})
        raise
    finally:
        stamp=args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        result={"date":str(run_date),"status":"failed" if any(r['returncode'] for r in results) else "ok",
                "steps":results,"publications":publications,"run_id":args.run_id,
                "pregame":args.pregame,"game_ids":args.game_id}
        atomic_json(_repo_root()/"reports"/f"nfl_daily_run_{stamp}.json",result)
        atomic_json(_repo_root()/"reports"/"nfl_daily_run_latest.json",result)
    if args.pregame and not any(r['returncode'] for r in results):
        # Publication is already persisted so the audit can verify its exact IDs.
        check=Step('NFL Live Cycle Audit','nfl_pipeline.live_cycle_audit',args=date_args,timeout_s=120)
        rc,stdout,stderr=_run(check)
        results.append(dict(step=check.label,returncode=rc,critical=True,
            stdout_tail=stdout[-1500:],stderr_tail=stderr[-1500:]))
        result['status']='failed' if rc else 'ok'
        atomic_json(_repo_root()/"reports"/f"nfl_daily_run_{stamp}.json",result)
        atomic_json(_repo_root()/"reports"/"nfl_daily_run_latest.json",result)
    if any(r["returncode"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
