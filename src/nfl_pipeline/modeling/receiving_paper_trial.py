"""Frozen receiving OE experiment: independently priced PAPER decisions only.

The original full-roster challenger CDF is the final research probability path.
Production side, EV, drift, ranking and market-blend weight never select a row.
This module cannot create wagers, alter production or authorize betting.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from nfl_pipeline import experiment_registry as experiments
from nfl_pipeline.context_contract import number, safe_time, validate_evidence
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, atomic_json
from nfl_pipeline.markets import normalize_name, normalize_team
from nfl_pipeline.modeling.distribution_policy import curve, outcomes, summary
from nfl_pipeline.modeling.evaluation import clustered_gain, probability_metrics, projection_metrics
from nfl_pipeline.modeling.prospective_market import implied, locked_market

# v5 (22 October 2026) succeeds v4, which drifted when its pinned sources changed: the paper-cohort
# repair in this module, then features.py (batched context-risk columns, proven identical), and
# predict_player_props (projection placement and the targets model). Rules are unchanged; v4's
# captures stay on disk as their own cohort and are never pooled (captures()).
NAME = "fanduel-receiving-oe-paper-v5"
CONTRACT = "nfl-receiving-independent-paper-v2"
STORE = MODEL_ROOT / "receiving_paper_trial"
ROOT = Path(__file__).resolve().parents[3]


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def registration_config(artifact, sha):
    from nfl_pipeline.modeling.scoring_versions import SOURCE, source_hash
    return dict(contract=CONTRACT, component_id=artifact["run_id"], artifact_sha256=sha,
        production_release=artifact["production_release"], training_end=artifact["training_end"],
        scoring_fingerprint=source_hash(SOURCE),
        stat="receiving_yards", book="fanduel", research_only=True,
        probability_path="immutable_full_roster_oe_cdf_no_production_adjustments",
        rules=dict(market_keys=["player_reception_yds", "player_receiving_yards"],
            max_quote_age_minutes=20, max_per_date=3, max_per_game=2,
            max_per_player_game=1, min_ev=0.0, min_market_edge=0.0,
            min_context_quality=.68, max_limited_risk=.42, min_role_stability=.55,
            rank="own_ev_desc_then_forecast_id_then_side", no_quota=True,
            capture="first_eligible_offer_per_player_game_no_replacement",
            selection="first_eligible_batch_fills_remaining_daily_slots_no_replacements"),
        review=dict(min_player_games=100, min_selected_player_games=50,
            min_independent_weeks=3, brier_gain_lower_95_above_zero=True,
            min_valid_exact_close_coverage=.90, max_close_age_minutes=20),
        automatic_deployment=False, automatic_betting_approval=False)


def register(artifact_path=None, *, extra_files=None):
    """Explicit one-time registration; daily calls never register or roll over."""
    import joblib
    from nfl_pipeline.modeling.stat_model_registry import load_registry
    registry = load_registry()
    candidates = (registry.get("candidates") or {}).get("receiving_yards", {})
    if artifact_path is None:
        matches = [e for e in candidates.values() if "probability" in e.get("outputs", ())
                   and str(e.get("component_id", "")).startswith("receiving-oe-v2-")]
        if len(matches) != 1:
            raise ValueError("Exactly one existing OE probability candidate must be specified")
        artifact_path = matches[0]["artifact"]
    path = Path(artifact_path).resolve()
    artifact = joblib.load(path)
    if (artifact.get("contract") != "nfl-receiving-opportunity-efficiency-v2"
            or artifact.get("stat") != "receiving_yards"):
        raise ValueError("The paper trial requires the existing receiving OE v2 artifact")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = candidates.get(artifact["run_id"], {})
    if entry.get("sha256") != sha or "probability" not in entry.get("outputs", ()):
        raise ValueError("Pinned OE probability artifact is not the registered shadow scorer")
    pointer=json.loads((MODEL_ROOT/"active_release.json").read_text(encoding="utf-8"))
    if pointer.get("release_id") != artifact["production_release"]:
        raise ValueError("Active production pointer does not match the OE artifact base release")
    files = experiments.source_dependencies(
        "nfl_pipeline.modeling.receiving_paper_trial", "nfl_pipeline.modeling.predict_player_props",
        "nfl_pipeline.modeling.stat_runtime", "nfl_pipeline.modeling.team_workload")
    files.update(extra_files or {})
    files["oe_probability_artifact"] = path
    files["production_release_pointer"] = MODEL_ROOT/"active_release.json"
    release_folder = MODEL_ROOT/"releases"/artifact["production_release"]
    for file in release_folder.iterdir():
        if file.is_file():
            files["production_release_"+file.name] = file
    absent = {}
    for label, file in {
        "market_calibration": MODEL_ROOT/"player_props/market_calibration.json",
        "interval_calibration": MODEL_ROOT/"player_props/interval_calibration.json",
        # Both move production numbers the scorer receives, so a change to either is drift.
        "placement_calibration": MODEL_ROOT/"player_props/placement_calibration.json",
        "targets_model_pointer": MODEL_ROOT/"player_props/targets_model.json",
        "production_distribution_policy": MODEL_ROOT/"distribution_policy/active.json",
    }.items():
        if file.exists():
            files[label] = file
        else:
            absent[label] = file
    config=registration_config(artifact, sha)
    config["production_stat_active"] = registry.get("active") or {}
    for stat, outputs in config["production_stat_active"].items():
        for output, active_entry in outputs.items():
            files["active_stat_"+stat+"_"+output] = Path(active_entry["artifact"])
    return experiments.register_experiment(NAME, config, files, absent_paths=absent)


def _check_candidate(manifest):
    from nfl_pipeline.modeling.stat_model_registry import load_registry
    cfg = manifest["config"]
    registry = load_registry()
    entry = ((registry.get("candidates") or {}).get(cfg["stat"]) or {}).get(cfg["component_id"], {})
    if (entry.get("sha256") != cfg["artifact_sha256"] or "probability" not in entry.get("outputs", ())
            or entry.get("base_release") != cfg["production_release"]):
        raise ValueError("Frozen paper shadow component changed or is no longer captured")
    if (registry.get("active") or {}) != cfg.get("production_stat_active", {}):
        raise ValueError("Frozen production stat components changed during the paper experiment")


def generation_attestation(component_id, artifact_sha256):
    """Bind a full-roster shadow generation to the frozen files while scoring."""
    manifest=experiments.load_experiment(NAME)
    if manifest is None or component_id != manifest["config"]["component_id"]:
        return None
    manifest=experiments.check_experiment(NAME); _check_candidate(manifest)
    cfg=manifest["config"]
    if artifact_sha256 != cfg["artifact_sha256"]:
        raise ValueError("Shadow generation artifact does not match the frozen experiment")
    from nfl_pipeline.modeling.scoring_versions import SOURCE, source_hash
    fingerprint=source_hash(SOURCE)
    if fingerprint != cfg["scoring_fingerprint"]:
        raise ValueError("Shadow generation scoring fingerprint changed")
    return dict(registration_sha256=manifest["sha256"], artifact_sha256=artifact_sha256,
                scoring_fingerprint=fingerprint, verified_at=datetime.now(timezone.utc).isoformat())


def _context_error(payload, cutoff, rules):
    evidence = validate_evidence(payload.get("context_evidence"), cutoff)
    if evidence is None:
        return "missing_or_future_context_evidence"
    features = payload.get("forecast_features") or {}
    status = str(payload.get("roster_status") or features.get("roster_status") or "").upper()
    injury = str(evidence.get("injury_status") or payload.get("injury_report_status") or "").lower()
    if status != "ACT" or injury in {"out", "doubtful", "injured reserve", "inactive"}:
        return "inactive_or_unknown_roster_status"
    # These are information-quality checks, independent of approved-model/trust gates.
    if number(features.get("same_week_injury_context_known")) != 1.0:
        return "same_week_injury_context_unknown"
    if not evidence.get("roster_observed_at") or not evidence.get("depth_observed_at"):
        return "missing_roster_or_depth_observation"
    if any((cutoff-safe_time(evidence[key])).total_seconds() > 7*86400
           for key in ("roster_observed_at", "depth_observed_at")):
        return "stale_roster_or_depth_observation"
    quality = number(payload.get("usage_context_quality"))
    limited = number(payload.get("limited_usage_risk"))
    stability = number(features.get("starter_role_stability_score"))
    if quality is None or quality < rules["min_context_quality"]:
        return "usage_context_quality_insufficient"
    if limited is None or limited > rules["max_limited_risk"]:
        return "limited_role_risk_unknown_or_high"
    if stability is None or stability < rules["min_role_stability"]:
        return "role_stability_unknown_or_low"
    if str(features.get("route_data_source") or "unknown") in {"unknown", "missing"}:
        return "route_context_missing"
    return None


def price_record(record, manifest, now):
    """Price both sides of one exact quote; no outcome or production-side veto."""
    cfg = manifest["config"]; rules = cfg["rules"]
    p = record.get("forecast_payload") or {}; q = (p.get("scoring_replay") or {}).get("offer") or {}
    if record.get("stat") != cfg["stat"] or p.get("book") != cfg["book"]:
        return None, "outside_trial"
    if any(str(p.get(k)) != str(record.get(k)) for k in ("game_id", "player_id", "stat")):
        return None, "forecast_identity_mismatch"
    if p.get("model_version") != cfg["production_release"]:
        return None, "production_release_mismatch"
    lock = safe_time(record.get("created_at_utc")); start = safe_time(record.get("start_ts_utc"))
    cutoff = safe_time(p.get("prediction_context_cutoff_utc")); registered = safe_time(manifest["registered_at"])
    if not all((lock, start, cutoff, registered)) or not registered <= cutoff <= lock <= now < start:
        return None, "not_a_new_post_registration_pregame_lock"
    day = str(p.get("game_date_et") or "")[:10]
    if not day or day <= cfg["training_end"]:
        return None, "training_includes_forecast_day"
    if q.get("market_key") not in rules["market_keys"] or q.get("is_alt_line") is True:
        return None, "not_common_full_game_line"
    if not q.get("provider"):
        return None, "missing_quote_provider"
    if number(q.get("line")) is None:
        return None, "invalid_offer_line"
    market = locked_market(record)
    if market["market_probability"] is None:
        return None, market["market_evidence"]
    features = p.get("forecast_features") or {}
    names = {normalize_name(str(features.get(k) or p.get(k) or ""))
             for k in ("player_name", "context_player_name", "context_player_name_norm")}
    names.discard("")
    if normalize_name(str(q.get("player_name_norm") or "")) not in names:
        return None, "quote_player_identity_mismatch"
    teams = {normalize_team(features.get(k) or p.get(k)) for k in ("team_abbr", "opponent_abbr")}
    if None in teams or len(teams) != 2 or teams != {normalize_team(q.get(k)) for k in ("home_team", "away_team")}:
        return None, "quote_team_identity_mismatch"
    fetched = safe_time(q.get("fetched_at_utc"))
    if not fetched or not 0 <= (now-fetched).total_seconds() <= 60*rules["max_quote_age_minutes"]:
        return None, "stale_or_future_quote"
    issue = _context_error(p, cutoff, rules)
    if issue:
        return None, issue
    shadows = p.get("stat_model_challengers") or {}
    shadow = shadows.get(cfg["component_id"]) or {}
    if (shadow.get("component_id") != cfg["component_id"] or "probability" not in shadow.get("outputs", ())
            or shadow.get("same_offer") is not True or number(shadow.get("line")) != number(p.get("line"))):
        return None, "missing_pinned_same_offer_curve"
    generation=shadow.get("experiment_generation") or {}
    generation_at=safe_time(generation.get("verified_at")); generation_start=safe_time(generation.get("started_at"))
    if (shadow.get("artifact_sha256") != cfg["artifact_sha256"]
            or generation.get("artifact_sha256") != cfg["artifact_sha256"]
            or generation.get("registration_sha256") != manifest["sha256"]
            or generation.get("scoring_fingerprint") != cfg["scoring_fingerprint"]
            or (p.get("scoring_replay") or {}).get("scoring_fingerprint") != cfg["scoring_fingerprint"]
            or not generation_start or not generation_at
            or not cutoff <= generation_start <= generation_at <= lock):
        return None, "unverified_frozen_shadow_generation"
    c = shadow.get("curve") or {}
    try:
        # Reject malformed mass instead of silently normalizing a damaged capture.
        weights = np.asarray(c["weights"], float)
        if not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1., atol=1e-8):
            raise ValueError("invalid_curve_mass")
        c = curve(c["values"], weights)
        distribution = outcomes(c, float(q["line"]))
        curve_summary = summary(c)
    except (ValueError, KeyError, TypeError):
        return None, "invalid_pinned_distribution"
    if distribution["push"] >= 1-1e-12:
        return None, "distribution_all_push"
    production = number(p.get("probability"))
    if production is None or not 0 <= production <= 1 or p.get("side") not in {"over", "under"}:
        return None, "missing_production_probability_comparator"
    prod_over = production if p["side"] == "over" else 1-production
    market_over = implied(q["over_price"])/(implied(q["over_price"])+implied(q["under_price"]))
    sides = []
    for side in ("over", "under"):
        link = q.get(side+"_link"); parsed = urlparse(str(link or ""))
        if parsed.scheme != "https" or not (parsed.hostname == "fanduel.com" or (parsed.hostname or "").endswith(".fanduel.com")):
            continue
        price = float(q[side+"_price"]); payout = price/100 if price > 0 else 100/-price
        probability = distribution["conditional_over"] if side == "over" else 1-distribution["conditional_over"]
        market_p = market_over if side == "over" else 1-market_over
        ev = distribution[side]*payout - distribution["under" if side == "over" else "over"]
        sides.append(dict(side=side, price=price, probability=probability, push_probability=distribution["push"],
            market_probability=market_p, production_probability=prod_over if side == "over" else 1-prod_over,
            ev=ev, market_edge=probability-market_p, link=link))
    if not sides:
        return None, "missing_valid_fanduel_link"
    best = sorted(sides, key=lambda s: (-s["ev"], s["side"]))[0]
    decision = dict(forecast_id=int(record["id"]), component_id=cfg["component_id"],
        game_id=record["game_id"], player_id=record["player_id"], player_name=p.get("player_name"),
        season=int(record["season"]), week=int(record["week"]), day=day, team=p.get("team_abbr"),
        stat=cfg["stat"], book=cfg["book"], line=float(q["line"]), offer_id=q["offer_id"],
        source_lock_at=lock.isoformat(), start_ts_utc=start.isoformat(), captured_at=now.isoformat(),
        source_context_cutoff=cutoff.isoformat(), quote=clean(q), context_evidence=clean(p["context_evidence"]),
        curve=c, distribution_summary=curve_summary, production_side=p["side"],
        production_drift_guard_pass=p.get("drift_guard_pass"), source_payload_sha256=digest(p),
        sides=sides, **best, research_only=True, betting_approved=False)
    decision["decision_id"] = digest(dict(experiment=manifest["sha256"], forecast_id=record["id"],
        component_id=cfg["component_id"], line=decision["line"], side=best["side"], price=best["price"]))
    return decision, None


def choose(records, manifest, now, previous=()):
    """First safe offer per player-game; retain all rows even when EV is negative."""
    used = {(r["game_id"], r["player_id"]) for r in previous}
    selected_previous = [r for r in previous if r["selected"]]
    days = Counter(r["day"] for r in selected_previous); games = Counter(r["game_id"] for r in selected_previous)
    rules = manifest["config"]["rules"]; rows = []; errors = Counter()
    for record in sorted(records, key=lambda r: (str(r["created_at_utc"]), int(r["id"]))):
        key = (record["game_id"], record["player_id"])
        if key in used:
            continue
        row, issue = price_record(record, manifest, now)
        if issue:
            errors[issue] += 1; continue
        used.add(key); rows.append(row)
    for row in sorted(rows, key=lambda r: (-r["ev"], r["forecast_id"], r["side"])):
        eligible = row["ev"] > rules["min_ev"] and row["market_edge"] > rules["min_market_edge"]
        cap = days[row["day"]] < rules["max_per_date"] and games[row["game_id"]] < rules["max_per_game"]
        row["selected"] = bool(eligible and cap)
        row["selection_reason"] = "positive_own_ev_and_market_edge" if row["selected"] else "daily_or_game_cap" if eligible else "no_positive_own_edge"
        if row["selected"]:
            days[row["day"]] += 1; games[row["game_id"]] += 1
    return rows, dict(errors)


def captures(manifest):
    result = []
    for path in sorted((STORE/"captures").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        body = {k:v for k,v in doc.items() if k != "sha256"}
        if doc.get("sha256") != digest(body):
            raise ValueError("Paper capture checksum mismatch")
        # Registrations are immutable cohorts. Preserve older captures on disk, but never
        # pool their selections or probabilities into the currently registered trial.
        if doc.get("registration_sha256") != manifest["sha256"]:
            continue
        result.append(doc)
    return result


def load_records(day=None):
    # Read-only, with participation-aware actual_stat; never import current results into selection.
    from nfl_pipeline.modeling.live_scoring_replay import load_rows
    return [r for r in load_rows(day, "receiving_yards") if r.get("is_current") is True]


def load_results(ids):
    """Use verified settlement evidence, never zero-fill missing participation."""
    if not ids:
        return []
    import psycopg2
    import psycopg2.extras
    from nfl_pipeline.db import PG_DSN
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        conn.set_session(readonly=True)
        cur.execute("SET LOCAL statement_timeout='30s'")
        cur.execute("""SELECT p.id,p.game_id,p.player_id,p.stat,g.status,
                   r.actual_stat::float AS actual,r.result AS graded_result
            FROM bets.nfl_player_prop_predictions p JOIN raw.nfl_games g USING(game_id)
            LEFT JOIN bets.nfl_player_prop_prediction_results r ON r.prediction_id=p.id
            WHERE p.id=ANY(%s)""", (list(ids),))
        return [dict(r) for r in cur.fetchall()]


def capture(day=None, *, records=None, now=None):
    if experiments.load_experiment(NAME) is None:
        return dict(status="not_registered", betting_approved=False)
    manifest = experiments.check_experiment(NAME); _check_candidate(manifest)
    source = load_records(day) if records is None else records
    STORE.mkdir(parents=True, exist_ok=True); lock = STORE/".capture.lock"; owned = False
    try:
        with lock.open("x"):
            owned = True
            at = now or datetime.now(timezone.utc)
            previous = [r for d in captures(manifest) for r in d["rows"]]
            rows, exclusions = choose(source, manifest, at, previous)
            # Verify no dependency changed during this capture.
            experiments.check_experiment(NAME); _check_candidate(manifest)
            body = clean(dict(contract=CONTRACT, registration_sha256=manifest["sha256"],
                captured_at=at.isoformat(), rows=rows, exclusions=exclusions,
                research_only=True, betting_approved=False))
            target = STORE/"captures"/(at.strftime("%Y%m%dT%H%M%S%fZ")+".json")
            if target.exists():
                raise ValueError("Refusing to replace an immutable paper capture")
            atomic_json(target, dict(body, sha256=digest(body)))
    finally:
        if owned:
            lock.unlink()
    return dict(status="paper_only", captured=len(rows), selected=sum(r["selected"] for r in rows),
                exclusions=exclusions, betting_approved=False)


def _settled(row, record):
    if record is None:
        return "missing_original_forecast", None
    if any(str(row[k]) != str(record.get(k)) for k in ("game_id", "player_id", "stat")):
        return "result_identity_mismatch", None
    if record.get("status") != "final":
        return "pending_game", None
    grade = record.get("graded_result")
    if grade == "void_nonparticipant":
        return "verified_nonparticipant_void", None
    actual = number(record.get("actual"))
    if grade not in {"win", "loss", "push"} or actual is None:
        return "pending_participation_or_result", None
    if (grade == "push") != (actual == row["line"]):
        return "inconsistent_verified_result", None
    if actual == row["line"]:
        return "push", actual
    win = (actual > row["line"]) == (row["side"] == "over")
    return "win" if win else "loss", actual


def _metrics(rows, lookup):
    accounting = Counter(); binary = []; points = []; profits = []
    for row in rows:
        state, actual = _settled(row, lookup.get(row["forecast_id"])); accounting[state] += 1
        if actual is not None:
            points.append(dict(actual=actual, **row["distribution_summary"]))
        if state not in {"win", "loss", "push"}:
            continue
        price = row["price"]
        profits.append((price/100 if price > 0 else 100/-price) if state == "win" else -1. if state == "loss" else 0.)
        if state != "push":
            binary.append(dict(row, outcome=float(state == "win")))
    out = dict(rows=len(rows), accounting=dict(accounting), settled_binary=len(binary),
        hypothetical_profit_units=sum(profits), hypothetical_roi=sum(profits)/len(profits) if profits else None,
        weeks=len({(r["season"],r["week"]) for r in binary}),
        player_games=len({(r["game_id"],r["player_id"]) for r in binary}))
    if points:
        out["distribution"] = projection_metrics([r["actual"] for r in points], [r["mean"] for r in points],
            [r["p10"] for r in points], [r["p90"] for r in points])
    if binary:
        frame = pd.DataFrame(binary)
        for label, column in (("challenger", "probability"), ("production", "production_probability"), ("market", "market_probability")):
            out[label] = probability_metrics(frame[column], frame.outcome)
        for label, column in (("production", "production_probability"), ("market", "market_probability")):
            out["brier_gain_vs_"+label] = clustered_gain(frame.assign(
                reference_error=(frame[column]-frame.outcome)**2,
                challenger_error=(frame.probability-frame.outcome)**2))
    return out


def exact_close(row, quotes, now, max_age_minutes=20):
    """Own side, same provider/event/player/stat/line, real paired pregame price."""
    start = safe_time(row["start_ts_utc"]); at = safe_time(row["captured_at"]); original = row["quote"]
    unknown = dict(decision_id=row["decision_id"], valid=False, reason="missing_exact_pregame_close")
    if now < start:
        return dict(unknown, reason="awaiting_kickoff")
    valid = []
    for q in quotes:
        try:
            if any(not original.get(k) or str(q.get(k)) != str(original[k]) for k in (
                    "provider", "event_id", "bookmaker_key", "player_name_norm", "stat", "market_key")):
                continue
            if number(q.get("line")) != row["line"] or safe_time(q.get("commence_time_utc")) != start:
                continue
            if q.get("snapshot_role") != "close" or q.get("pair_quality") in {"synthetic", "cross_book", "one_sided"}:
                continue
            fetched = safe_time(q.get("fetched_at_utc"))
            if not fetched or not at <= fetched < start or (start-fetched).total_seconds() > max_age_minutes*60:
                continue
            over, under = implied(q.get("over_price")), implied(q.get("under_price"))
            probability = (over if row["side"] == "over" else under)/(over+under)
            # CLV is expected return at the locked executable price, using the
            # closing no-vig conditional probability. No closing outcome label.
            price = row["price"]; payout = price/100 if price > 0 else 100/-price
            valid.append(dict(decision_id=row["decision_id"], valid=True, reason="valid_exact_close",
                side=row["side"], line=row["line"], book=row["book"], quote=clean(q),
                close_probability=probability, probability_delta=probability-row["market_probability"],
                closing_fair_conditional_ev=probability*(1+payout)-1, close_price=q[row["side"]+"_price"]))
        except (ValueError, TypeError, KeyError):
            continue
    return max(valid, key=lambda c: (str(c["quote"]["fetched_at_utc"]), int(c["quote"].get("id") or 0))) if valid else unknown


def load_close_quotes(rows):
    """Only exact captured identities; another provider's event is not substituted."""
    if not rows:
        return {}
    import psycopg2
    import psycopg2.extras
    from nfl_pipeline.db import PG_DSN
    result = {}
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        conn.set_session(readonly=True)
        cur.execute("SET LOCAL statement_timeout='30s'")
        for row in rows:
            q = row["quote"]
            cur.execute("""SELECT id,provider,event_id,bookmaker_key,player_name_norm,stat,market_key,
                       line::float,commence_time_utc,fetched_at_utc,snapshot_role,over_price,under_price
                FROM odds.nfl_player_prop_lines WHERE provider=%s AND event_id=%s AND bookmaker_key=%s
                  AND player_name_norm=%s AND stat=%s AND market_key=%s AND line=%s
                  AND commence_time_utc=%s AND snapshot_role='close'
                  AND fetched_at_utc >= %s AND fetched_at_utc < %s
                ORDER BY fetched_at_utc DESC,id DESC""",
                (q.get("provider"),q.get("event_id"),row["book"],q.get("player_name_norm"),row["stat"],
                 q.get("market_key"),row["line"],row["start_ts_utc"],row["captured_at"],row["start_ts_utc"]))
            result[row["decision_id"]] = [dict(q) for q in cur.fetchall()]
    return result


def capture_closes(manifest, rows, now, *, quote_lookup=None):
    eligible = [r for r in rows if r["selected"] and safe_time(r["start_ts_utc"]) <= now]
    folder = STORE/"closes"; folder.mkdir(parents=True, exist_ok=True)
    saved = {}; pending = []
    for row in eligible:
        path = folder/(row["decision_id"]+".json")
        if not path.exists():
            pending.append(row); continue
        doc=json.loads(path.read_text(encoding="utf-8")); body={k:v for k,v in doc.items() if k!="sha256"}
        if doc.get("registration_sha256") != manifest["sha256"] or doc.get("sha256") != digest(body):
            raise ValueError("Paper close evidence checksum mismatch")
        saved[row["decision_id"]] = doc["close"]
    lookup = load_close_quotes(pending) if quote_lookup is None else quote_lookup
    for row in pending:
        close=exact_close(row, lookup.get(row["decision_id"], []), now,
                          manifest["config"]["review"]["max_close_age_minutes"])
        saved[row["decision_id"]]=close
        if close["valid"]:
            body=clean(dict(registration_sha256=manifest["sha256"], captured_at=now.isoformat(), close=close))
            # Exclusive create preserves the original exact-close evidence on reruns.
            path=folder/(row["decision_id"]+".json")
            try:
                with path.open("x", encoding="utf-8") as f:
                    json.dump(dict(body,sha256=digest(body)), f, indent=2, allow_nan=False)
            except FileExistsError:
                pass
    return saved


def build_report(manifest, documents, records, *, closes=None, now=None):
    rows = [r for d in documents for r in d["rows"]]
    if len({r["decision_id"] for r in rows}) != len(rows) or len({(r["game_id"],r["player_id"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate paper experiment decisions")
    lookup = {int(r["id"]):r for r in records}
    all_metrics = _metrics(rows, lookup); selected = [r for r in rows if r["selected"]]
    selected_metrics = _metrics(selected, lookup); rules = manifest["config"]["review"]
    at=now or datetime.now(timezone.utc); closes=closes or {}
    matured=[r for r in selected if safe_time(r["start_ts_utc"]) <= at]
    valid=[closes[r["decision_id"]] for r in matured if closes.get(r["decision_id"],{}).get("valid")]
    close_coverage=len(valid)/len(matured) if matured else None
    close_metrics=dict(matured=len(matured),valid=len(valid),coverage=close_coverage,
        unknown_reasons=dict(Counter(closes.get(r["decision_id"],{}).get("reason","missing_exact_pregame_close")
            for r in matured if not closes.get(r["decision_id"],{}).get("valid"))),
        average_probability_delta=float(np.mean([c["probability_delta"] for c in valid])) if valid else None,
        average_closing_fair_conditional_ev=float(np.mean([c["closing_fair_conditional_ev"] for c in valid])) if valid else None,
        observations=list(closes.values()))
    blockers = ["paper_only_no_automatic_cash_authorization"]
    if close_coverage is None or close_coverage < rules["min_valid_exact_close_coverage"]:
        blockers.append("independent_exact_close_coverage_below_floor")
    if not valid or close_metrics["average_closing_fair_conditional_ev"] <= 0:
        blockers.append("positive_selected_closing_fair_conditional_ev_unconfirmed")
    if all_metrics["player_games"] < rules["min_player_games"]:
        blockers.append("all_offer_player_game_sample_below_floor")
    if selected_metrics["player_games"] < rules["min_selected_player_games"]:
        blockers.append("selected_player_game_sample_below_floor")
    for label, scope in (("all_offers",all_metrics), ("frozen_selected",selected_metrics)):
        if scope["weeks"] < rules["min_independent_weeks"]:
            blockers.append(label+"_independent_weeks_below_floor")
        for comparator in ("production", "market"):
            if (scope.get("brier_gain_vs_"+comparator, {}).get("lower_95") or 0) <= 0:
                blockers.append(label+"_brier_gain_vs_"+comparator+"_unconfirmed")
    return clean(dict(contract=CONTRACT, registration_sha256=manifest["sha256"], status="paper_evidence_collection",
        all_offers=all_metrics, frozen_selected=selected_metrics,
        selected_decision_ids=[r["decision_id"] for r in selected], blockers=blockers,
        exact_closes=close_metrics, valid_exact_close_coverage=close_coverage,
        research_only=True, production_changed=False, betting_approved=False,
        limitations=["Prospective captures only; neither reviewed history nor later forecasts replace decisions.",
            "One player-game per cohort; NFL weeks are the uncertainty clusters.",
            "Probabilities are conditional on no push. Actual pushes return zero profit and are excluded from binary scores.",
            "Closing fair EV is conditional on no push; a closing push probability is not fabricated.",
            "Paper returns use captured prices, not confirmed fills. Source production W/L is never used for an independent side.",
            "An empty selection or unconfirmed interval never counts as an improvement."]))


def report(*, records=None):
    manifest = experiments.load_experiment(NAME)
    if manifest is None:
        return dict(status="not_registered", betting_approved=False)
    # Reports may account for old captures after code drift, but disclose the blocked experiment.
    drift = None
    try:
        experiments.check_experiment(NAME)
        _check_candidate(manifest)
    except ValueError as exc:
        drift = str(exc)
    documents=captures(manifest); rows=[r for d in documents for r in d["rows"]]
    now=datetime.now(timezone.utc)
    closes=capture_closes(manifest,rows,now)
    result = build_report(manifest, documents,
        load_results([r["forecast_id"] for r in rows]) if records is None else records, closes=closes, now=now)
    result["built_at"] = datetime.now(timezone.utc).isoformat(); result["dependency_drift"] = drift
    if drift:
        result["blockers"].append("experiment_dependency_drift")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    atomic_json(STORE/"reports"/(stamp+".json"), result)
    atomic_json(ROOT/"reports/nfl_receiving_paper_v2_latest.json", result)
    lines = ["# Receiving OE v2 paper experiment", "", "Paper only. No wager or production authorization.", "",
        "| Cohort | Player-games | Weeks | Challenger Brier | Production Brier | Market Brier | Paper ROI |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("all_offers", "frozen_selected"):
        s=result[name]
        fmt=lambda v: "unknown" if v is None else f"{v:.4f}"
        lines.append(f"| {name} | {s['player_games']} | {s['weeks']} | " + " | ".join(
            [fmt(s.get(k,{}).get('brier')) for k in ('challenger','production','market')]+[fmt(s['hypothetical_roi'])])+" |")
    lines += ["", "Blockers: "+", ".join(result["blockers"]), "", *result["limitations"]]
    (ROOT/"reports/nfl_receiving_paper_v2_latest.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command", required=True)
    reg=sub.add_parser("register"); reg.add_argument("--artifact", type=Path)
    cap=sub.add_parser("capture"); cap.add_argument("--date", type=date.fromisoformat)
    sub.add_parser("report"); args=parser.parse_args()
    result=register(args.artifact) if args.command=="register" else capture(args.date) if args.command=="capture" else report()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
