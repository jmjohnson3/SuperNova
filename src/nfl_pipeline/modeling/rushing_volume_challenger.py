"""Prospective challenger: rushing-yard projections rescaled by the carries model.

The carries model (modeling/volume_model) beat every simple carries estimate walk-forward, and
moving the live rushing projection from the 5-game-average carries it implicitly assumes to the
model's carries cut 2026 rushing-yard MAE 19.02 -> 18.16 in hindsight. This captures that claim
before the games are played and grades it afterwards; nothing here changes a live forecast.

For every current pregame rushing-yards forecast (RB and QB, the model's population):

    reference = the production model's own projection (never the line-anchored one)
    carries   = carries model on the forecast's archived pregame features
    candidate = reference * carries / carries_avg_5

Registration pins the volume artifact by checksum and starts the cohort; captures are append-only
files, one per run. Grading keeps the last capture made before kickoff per player-game, settles it
against the official box score (players who played), and judges carries first, then yards: the
candidate is ready for a deployment review only when its week-clustered MAE gain over the reference
has a 95% lower bound above zero across at least three weeks. Deployment stays a manual decision.

    python -m nfl_pipeline.modeling.rushing_volume_challenger --register
    python -m nfl_pipeline.modeling.rushing_volume_challenger capture --date 2026-10-25
    python -m nfl_pipeline.modeling.rushing_volume_challenger evaluate
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from scipy.stats import t as student_t

from nfl_pipeline.context_contract import number, safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, active_release, atomic_json
from nfl_pipeline.modeling import volume_model

ROOT = Path(__file__).resolve().parents[3]
STORE = MODEL_ROOT / "rushing_volume_challenger"
REGISTRATION = STORE / "registration.json"
CONTRACT = "nfl-rushing-volume-challenger-v1"
POSITIONS = ("RB", "QB")
MINIMUM_WEEKS = 3


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register(run_id: str | None = None, *, store: Path = STORE, now: datetime | None = None) -> dict:
    """Pin a volume run as the challenger. Create-once: a different pin needs a new store."""
    run_id = run_id or json.loads((volume_model.STORE / "latest.json").read_text(encoding="utf-8"))["run_id"]
    folder = volume_model.STORE / run_id
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if not report["volume"]["carries"]["volume_accepted"]:
        raise ValueError("The carries model did not pass its volume checks")
    artifact = folder / "models.joblib"
    now = now or datetime.now(timezone.utc)
    doc = dict(contract=CONTRACT, volume_run_id=run_id, artifact=str(artifact.relative_to(ROOT).as_posix()),
               artifact_sha256=_sha(artifact), production_release=active_release()["release_id"],
               method="reference * carries_model / carries_avg_5", positions=list(POSITIONS),
               registered_at=now.isoformat(), acceptance_start=now.date().isoformat(), minimum_weeks=MINIMUM_WEEKS,
               status="prospective_only", deployment_approved=False, betting_approved=False)
    target = store / "registration.json"
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if any(old[k] != doc[k] for k in ("volume_run_id", "artifact_sha256", "production_release", "method")):
            raise ValueError("A different rushing challenger is already registered; use a new store")
        return old
    atomic_json(target, doc)
    return doc


def load_registered(store: Path = STORE) -> tuple[dict, dict] | tuple[None, None]:
    path = store / "registration.json"
    if not path.exists():
        return None, None
    registration = json.loads(path.read_text(encoding="utf-8"))
    artifact_path = ROOT / registration["artifact"]
    if _sha(artifact_path) != registration["artifact_sha256"]:
        raise ValueError("Registered carries artifact checksum mismatch")
    if registration["production_release"] != active_release()["release_id"]:
        raise ValueError("Production release changed since the challenger was registered")
    return registration, joblib.load(artifact_path)["carries"]


def raw_projection(payload: dict) -> float | None:
    """The production model's own number, as projection_error defines it."""
    outputs = ((payload.get("forecast_outputs") or {}).get("pricing") or {}).get("model_projection")
    trace = (payload.get("probability_trace") or {}).get("model_projection")
    for value in (outputs, trace):
        if number(value) is not None:
            return float(value)
    return float(payload["projection"]) if payload.get("line") is None and number(payload.get("projection")) is not None else None


def predict_carries(model: dict, features: list[dict], positions: list[str]) -> np.ndarray:
    frame = pd.DataFrame(features).reindex(columns=model["features"])
    frame = frame.apply(lambda s: volume_model._numeric(s)).assign(position=positions)
    return model["booster"].predict(volume_model.feature_matrix(frame, model["features"], tuple(model["positions"])))


def score_records(records: list[tuple[int, dict, dict]], registration: dict, model: dict, now: datetime) -> tuple[list, Counter]:
    """records: (forecast_id, payload, info with start_ts_utc and created_at_utc)."""
    excluded, usable = Counter(), []
    registered = safe_time(registration["registered_at"])
    for forecast_id, payload, info in records:
        kickoff, created = safe_time(info["start_ts_utc"]), safe_time(info["created_at_utc"])
        position = str(payload.get("position") or "").upper()
        features = payload.get("forecast_features") or {}
        reference = raw_projection(payload)
        average = number(features.get("carries_avg_5"))
        if position not in POSITIONS:
            excluded["position_outside_model"] += 1
        elif not kickoff or not created or not registered <= created < kickoff or now >= kickoff:
            excluded["not_post_registration_pregame_forecast"] += 1
        elif payload.get("model_version") != registration["production_release"]:
            excluded["different_production_release"] += 1
        elif reference is None:
            excluded["missing_raw_model_projection"] += 1
        elif not features:
            excluded["missing_archived_features"] += 1
        elif average is None or average <= 0:
            excluded["no_carry_history"] += 1
        else:
            usable.append((forecast_id, payload, info, position, features, reference, average))
    if not usable:
        return [], excluded
    carries = predict_carries(model, [u[4] for u in usable], [u[3] for u in usable])
    rows = []
    for (forecast_id, payload, info, position, features, reference, average), predicted in zip(usable, carries):
        rows.append(dict(forecast_id=int(forecast_id), game_id=payload["game_id"], player_id=str(payload["player_id"]),
                         player_name=payload.get("player_name"), position=position, season=payload.get("season"),
                         week=payload.get("week"), model_version=payload["model_version"],
                         scoring_version=(payload.get("scoring_replay") or {}).get("scoring_fingerprint"),
                         forecast_created_at=safe_time(info["created_at_utc"]).isoformat(),
                         kickoff=safe_time(info["start_ts_utc"]).isoformat(), captured_at=now.isoformat(),
                         reference=float(reference), carries_model=float(predicted), carries_avg_5=float(average),
                         candidate=float(max(0.0, reference * predicted / average)), line=number(payload.get("line"))))
    return rows, excluded


def capture(day: date, *, store: Path = STORE, now: datetime | None = None) -> dict:
    registration, model = load_registered(store)
    if registration is None:
        return dict(status="not_registered", captured=0)
    now = now or datetime.now(timezone.utc)
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='60s'")
        cur.execute("""SELECT p.id, p.forecast_payload, p.created_at_utc, g.start_ts_utc
            FROM bets.nfl_player_prop_predictions p JOIN raw.nfl_games g USING (game_id)
            WHERE p.is_current AND p.game_date_et = %s AND p.stat = 'rushing_yards'
              AND p.integrity_version = 'nfl-asof-v2' AND g.start_ts_utc > %s
            ORDER BY p.created_at_utc, p.id""", (day, now))
        records = [(r["id"], r["forecast_payload"] or {}, r) for r in cur.fetchall()]
    rows, excluded = score_records(records, registration, model, now)
    if rows:
        doc = clean(dict(status="prospective_only", registration=registration, rows=rows,
                         exclusions=dict(excluded), captured_at=now.isoformat(), production_changed=False))
        atomic_json(store / "prospective" / str(day) / (now.strftime("%Y%m%dT%H%M%S%fZ") + ".json"), doc)
    return dict(status="prospective_only", captured=len(rows), exclusions=dict(excluded))


def _clustered_gain(frame: pd.DataFrame, reference: str, candidate: str, actual: str) -> dict:
    """Weekly mean of |reference error| - |candidate error| (positive = candidate better)."""
    gain = ((frame[reference] - frame[actual]).abs() - (frame[candidate] - frame[actual]).abs())
    weekly = gain.groupby([frame.season, frame.week]).mean()
    n = len(weekly)
    if n < 2:
        return dict(weeks=n, mean=float(weekly.mean()) if n else None, lower_95=None)
    half = student_t.ppf(0.975, n - 1) * weekly.std(ddof=1) / np.sqrt(n)
    return dict(weeks=int(n), mean=float(weekly.mean()), lower_95=float(weekly.mean() - half),
                upper_95=float(weekly.mean() + half))


def load_captures(store: Path, registration: dict) -> tuple[pd.DataFrame, Counter]:
    excluded, rows = Counter(), []
    for path in sorted((store / "prospective").glob("*/*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("registration") != registration:
            excluded["different_registration"] += len(doc.get("rows") or [])
            continue
        rows.extend(doc.get("rows") or [])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, excluded
    frame["_captured"] = pd.to_datetime(frame.captured_at, utc=True)
    frame = frame[frame._captured < pd.to_datetime(frame.kickoff, utc=True)]
    before = len(frame)
    frame = frame.sort_values("_captured").drop_duplicates(["game_id", "player_id"], keep="last").drop(columns="_captured")
    excluded["earlier_capture_of_same_player_game"] += before - len(frame)
    return frame, excluded


def settle(conn, frame: pd.DataFrame) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT l.game_id, l.player_id, l.rushing_yards::float AS actual, l.carries::float AS actual_carries
            FROM raw.nfl_player_gamelogs l JOIN raw.nfl_games g USING (game_id)
            WHERE g.status = 'final' AND l.game_id = ANY(%s)
              AND (COALESCE(l.offense_snaps,0) > 0 OR COALESCE(l.carries,0) + COALESCE(l.targets,0) + COALESCE(l.pass_attempts,0) > 0)""",
                    (list(frame.game_id.unique()),))
        results = pd.DataFrame([dict(r) for r in cur.fetchall()])
    if results.empty:
        return frame.assign(actual=np.nan, actual_carries=np.nan)
    return frame.merge(results.astype({"player_id": str}), on=["game_id", "player_id"], how="left")


def _metrics(df: pd.DataFrame, name: str) -> dict:
    err = df[name] - df.actual
    return dict(mae=float(err.abs().mean()), bias=float(err.mean()), above_share=float((df.actual > df[name]).mean()))


def evaluate(*, store: Path = STORE) -> dict:
    registration, _ = load_registered(store)
    if registration is None:
        return dict(status="not_registered")
    frame, excluded = load_captures(store, registration)
    result = dict(status="waiting_for_prospective_results", registration=registration, player_games=0,
                  exclusions=dict(excluded), deployment_approved=False, betting_approved=False)
    if not frame.empty:
        with psycopg2.connect(PG_DSN) as conn:
            conn.set_session(readonly=True)
            frame = settle(conn, frame)
        settled = frame.dropna(subset=["actual"])
        result.update(captured_player_games=int(len(frame)), player_games=int(len(settled)),
                      pending_or_did_not_play=int(len(frame) - len(settled)))
        if len(settled):
            gain = _clustered_gain(settled, "reference", "candidate", "actual")
            carries = settled.dropna(subset=["actual_carries"])
            ready = gain["weeks"] >= registration["minimum_weeks"] and (gain.get("lower_95") or 0) > 0
            result.update(
                weeks=sorted({f"{s}-{w}" for s, w in zip(settled.season, settled.week)}),
                carries=dict(model_mae=float((carries.carries_model - carries.actual_carries).abs().mean()),
                             average_mae=float((carries.carries_avg_5 - carries.actual_carries).abs().mean()),
                             player_games=int(len(carries))),
                reference=_metrics(settled, "reference"), candidate=_metrics(settled, "candidate"),
                clustered_mae_gain=gain,
                by_position={p: dict(player_games=int(len(g)), reference=_metrics(g, "reference"),
                                     candidate=_metrics(g, "candidate")) for p, g in settled.groupby("position")},
                status="ready_for_deployment_review" if ready else "collecting_evidence")
    atomic_json(ROOT / "reports" / "nfl_rushing_volume_challenger_latest.json", clean(result))
    (ROOT / "reports" / "nfl_rushing_volume_challenger_latest.md").write_text(markdown(result), encoding="utf-8")
    return result


def markdown(result: dict) -> str:
    f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"
    reg = result.get("registration") or {}
    lines = ["# NFL Rushing Volume Challenger (prospective)", "",
             f"Status: **{result['status']}**. Registered {reg.get('registered_at', '-')} with volume run "
             f"`{reg.get('volume_run_id', '-')}`; candidate = production model projection x model carries / 5-game "
             f"average carries. Ready for a deployment review only after {reg.get('minimum_weeks', MINIMUM_WEEKS)}+ weeks "
             "with a 95% lower bound on the weekly MAE gain above zero. Nothing here changes a live forecast.", "",
             f"Settled player-games: {result.get('player_games', 0)} (captured {result.get('captured_player_games', 0)}; "
             f"pending or did not play {result.get('pending_or_did_not_play', 0)}).", ""]
    if result.get("carries"):
        c, g = result["carries"], result["clustered_mae_gain"]
        lines += ["| Measure | Production | Challenger |", "|---|---:|---:|",
                  f"| Carries MAE (5-game average vs model) | {f(c['average_mae'])} | {f(c['model_mae'])} |",
                  f"| Rushing-yard MAE | {f(result['reference']['mae'])} | {f(result['candidate']['mae'])} |",
                  f"| Bias (projection minus actual) | {result['reference']['bias']:+.2f} | {result['candidate']['bias']:+.2f} |",
                  f"| Share above projection | {result['reference']['above_share']:.0%} | {result['candidate']['above_share']:.0%} |", "",
                  f"Weekly MAE gain {f(g.get('mean'))} yards (95% lower bound {f(g.get('lower_95'))}) over {g['weeks']} week(s).", ""]
        lines += ["| Position | Player-games | Production MAE | Challenger MAE |", "|---|---:|---:|---:|"]
        for p, v in result.get("by_position", {}).items():
            lines.append(f"| {p} | {v['player_games']} | {f(v['reference']['mae'])} | {f(v['candidate']['mae'])} |")
    if result.get("exclusions"):
        lines += ["", "Exclusions: " + ", ".join(f"{k} {v}" for k, v in result["exclusions"].items()) + "."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=("capture", "evaluate"))
    ap.add_argument("--date", type=date.fromisoformat)
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--run-id")
    args = ap.parse_args()
    if args.register:
        print(json.dumps(register(args.run_id), indent=2))
    elif args.command == "capture":
        print(json.dumps(capture(args.date or datetime.now().date()), indent=2))
    else:
        print(json.dumps({k: v for k, v in evaluate().items() if k != "registration"}, indent=2, default=str))


if __name__ == "__main__":
    main()
