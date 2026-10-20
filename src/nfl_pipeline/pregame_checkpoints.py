"""Stateful FanDuel pregame checkpoints for the scheduled NFL close runner.

Each checkpoint is evidence from one real provider refresh. A later snapshot
cannot be relabeled as an earlier one, and a missed checkpoint stays missed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json


CONTRACT = "nfl-fanduel-pregame-checkpoints-v1"
OFFSETS = (120, 60, 20, 5)
EARLY_TOLERANCE_MINUTES = 0.0
LATE_TOLERANCE_MINUTES = 10.0
ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "reports" / "nfl_pregame_checkpoint_state.json"


def _game_key(game: dict[str, Any]) -> str:
    kickoff = safe_time(game.get("start_ts_utc"))
    return f"{game['game_id']}@{kickoff.isoformat() if kickoff else 'unknown'}"


def load_state(path: Path | None = None) -> dict[str, Any]:
    path = path or STATE
    if not path.exists():
        return {"contract": CONTRACT, "games": {}}
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("contract") != CONTRACT or not isinstance(doc.get("games"), dict):
        raise ValueError("invalid_nfl_pregame_checkpoint_state")
    return doc


def save_state(state: dict[str, Any], path: Path | None = None) -> None:
    state["contract"] = CONTRACT
    atomic_json(path or STATE, state)


def plan(
    games: list[dict[str, Any]],
    state: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return at most one due checkpoint per game.

    The ten-minute late window tolerates scheduler jitter without admitting a
    quote before the two-hour CLV window or allowing one observation to satisfy
    two target times.
    """
    now = now or datetime.now(timezone.utc)
    due: dict[str, int] = {}
    observations: list[dict[str, Any]] = []
    for game in games:
        kickoff = safe_time(game.get("start_ts_utc"))
        remaining = game.get("minutes_to_start")
        if remaining is None and kickoff is not None:
            remaining = (kickoff - now).total_seconds() / 60.0
        remaining = float(remaining) if remaining is not None else None
        game_state = state.get("games", {}).get(_game_key(game), {})
        checkpoints = game_state.get("checkpoints") or {}
        selected: int | None = None
        rows = []
        for offset in OFFSETS:
            previous = checkpoints.get(str(offset)) or {}
            if previous.get("status") == "verified":
                status = "verified"
            elif remaining is None:
                status = "schedule_unknown"
            elif remaining <= 0:
                status = "missed"
            elif remaining > offset + EARLY_TOLERANCE_MINUTES:
                status = "waiting"
            elif remaining >= max(0.0, offset - LATE_TOLERANCE_MINUTES):
                status = "due" if selected is None else "waiting_for_prior_checkpoint"
                if selected is None:
                    selected = offset
            else:
                status = "missed"
            rows.append(
                {
                    "offset_minutes": offset,
                    "target_at": (
                        datetime.fromtimestamp(
                            kickoff.timestamp() - offset * 60,
                            tz=timezone.utc,
                        ).isoformat()
                        if kickoff is not None
                        else None
                    ),
                    "status": status,
                    "attempts": int(previous.get("attempts") or 0),
                    "captured_at": previous.get("captured_at"),
                    "verification": previous.get("verification"),
                }
            )
        if selected is not None:
            due[str(game["game_id"])] = selected
        observations.append(
            {
                "game_id": str(game["game_id"]),
                "kickoff": kickoff.isoformat() if kickoff else None,
                "minutes_to_start": round(remaining, 2) if remaining is not None else None,
                "due_offset": selected,
                "checkpoints": rows,
            }
        )
    return {"due": due, "observations": observations}


def lock_verification_passed(row: dict[str, Any]) -> bool:
    """Return whether current FanDuel locks satisfy the lock-time contract."""
    return bool(
        row.get("current_fanduel_locks", 0) > 0
        and row.get("immutable_lock_violations", 0) == 0
        and row.get("explicit_injury_context_locks") == row.get("current_fanduel_locks")
        and row.get("receiving_same_line_triplets") == row.get("receiving_locks")
        and row.get("all_locks_missing_links", 0) == 0
        and row.get("actionable_missing_links", 0) == 0
    )


def verification_passed(row: dict[str, Any]) -> bool:
    """Return whether exact paired close prices were captured for every locked line.

    Bet links are required at lock/publication time, where a user may act on the
    offer.  A closing snapshot is evidence only: providers can omit a deep link
    while still returning valid same-book over/under prices.  Keep the close-link
    count in diagnostics, but do not discard otherwise valid CLV evidence for it.
    """
    return bool(
        row.get("fresh_normalized_rows", 0) > 0
        and row.get("fresh_normalized_rows") == row.get("fresh_true_pair_rows")
        and lock_verification_passed(row)
    )


def record_attempt(
    state: dict[str, Any],
    games: list[dict[str, Any]],
    assignments: dict[str, int],
    verification: dict[str, dict[str, Any]],
    captured_at: datetime | None = None,
) -> None:
    captured_at = captured_at or datetime.now(timezone.utc)
    state.setdefault("games", {})
    for game in games:
        game_id = str(game["game_id"])
        if game_id not in assignments:
            continue
        game_state = state["games"].setdefault(
            _game_key(game),
            {
                "game_id": game_id,
                "kickoff": safe_time(game.get("start_ts_utc")).isoformat()
                if safe_time(game.get("start_ts_utc"))
                else None,
                "checkpoints": {},
            },
        )
        offset = str(assignments[game_id])
        previous = game_state["checkpoints"].get(offset) or {}
        check = verification.get(game_id) or {
            "game_id": game_id,
            "error": "checkpoint_verification_missing",
        }
        game_state["checkpoints"][offset] = {
            "status": "verified" if verification_passed(check) else "captured_unverified",
            "attempts": int(previous.get("attempts") or 0) + 1,
            "captured_at": captured_at.isoformat(),
            "verification": check,
        }
    state["updated_at"] = captured_at.isoformat()


def _empty_lock_verification(game_id: str) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "current_fanduel_locks": 0,
        "receiving_locks": 0,
        "receiving_same_line_triplets": 0,
        "explicit_injury_context_locks": 0,
        "known_player_injury_context_locks": 0,
        "immutable_lock_violations": 0,
        "all_locks_missing_links": 0,
        "actionable_missing_links": 0,
    }


def verify_locks(et_date, game_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Audit current immutable FanDuel locks on their exact offered lines."""
    game_ids = [str(game_id) for game_id in game_ids]
    if not game_ids:
        return {}
    rows = {game_id: _empty_lock_verification(game_id) for game_id in game_ids}
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(
        cursor_factory=psycopg2.extras.RealDictCursor
    ) as cur:
        cur.execute("SET LOCAL statement_timeout='45s'")
        cur.execute(
            """
            SELECT p.game_id,
                   count(*)::int AS current_fanduel_locks,
                   count(*) FILTER (WHERE p.stat='receiving_yards')::int AS receiving_locks,
                   count(*) FILTER (
                     WHERE p.stat='receiving_yards'
                       AND p.market_no_vig_probability IS NOT NULL
                       AND p.probability IS NOT NULL
                       AND (
                         EXISTS (
                           SELECT 1
                           FROM jsonb_each(COALESCE(p.forecast_payload->'stat_model_challengers','{}'::jsonb)) x
                           WHERE COALESCE((x.value->>'same_offer')::boolean,false)
                             AND (x.value->>'line')::numeric=p.line
                             AND x.value->>'side'=p.side
                             AND (x.value->>'price')::integer=p.price
                         ) OR EXISTS (
                           SELECT 1
                           FROM jsonb_array_elements(COALESCE(
                             p.forecast_payload#>'{scoring_replay,offer,distribution_policy_variants}',
                             '[]'::jsonb)) x
                           WHERE x->'challenger' IS NOT NULL
                         )
                       )
                   )::int AS receiving_same_line_triplets,
                   count(*) FILTER (
                     WHERE p.forecast_payload ? 'injury_report_status'
                       AND p.forecast_payload ? 'injury_practice_status'
                       AND p.forecast_payload ? 'prediction_context_cutoff_utc'
                   )::int AS explicit_injury_context_locks,
                   count(*) FILTER (
                     WHERE NULLIF(p.forecast_payload->>'injury_report_status','') IS NOT NULL
                        OR NULLIF(p.forecast_payload->>'injury_practice_status','') IS NOT NULL
                   )::int AS known_player_injury_context_locks,
                   count(*) FILTER (
                     WHERE p.integrity_version<>'nfl-asof-v2'
                        OR p.created_at_utc>=g.start_ts_utc
                        OR NULLIF(p.forecast_payload->>'quote_fetched_at_utc','')::timestamptz>p.created_at_utc
                        OR NULLIF(p.forecast_payload->>'prediction_context_cutoff_utc','')::timestamptz>p.created_at_utc
                   )::int AS immutable_lock_violations,
                   count(*) FILTER (
                     WHERE p.link IS NULL OR p.link !~* '^https?://' OR lower(p.link) NOT LIKE '%%fanduel%%'
                   )::int AS all_locks_missing_links,
                   count(*) FILTER (
                     WHERE p.tier IN ('micro_projection','starter','bankroll')
                       AND (p.link IS NULL OR p.link !~* '^https?://' OR lower(p.link) NOT LIKE '%%fanduel%%')
                   )::int AS actionable_missing_links
            FROM bets.nfl_player_prop_predictions p
            JOIN raw.nfl_games g USING (game_id)
            WHERE p.game_date_et=%s AND p.game_id=ANY(%s)
              AND p.book='fanduel' AND p.is_current
            GROUP BY p.game_id
            """,
            (et_date, game_ids),
        )
        for row in cur.fetchall():
            rows[str(row["game_id"])].update(dict(row))
    for row in rows.values():
        row["lock_verification_passed"] = lock_verification_passed(row)
        row["injury_context_note"] = (
            "known_player_status"
            if row["known_player_injury_context_locks"]
            else "explicit_unknown_not_assumed_healthy"
        )
    return rows


def verify_capture(
    et_date,
    games: list[dict[str, Any]],
    since: datetime,
) -> dict[str, dict[str, Any]]:
    """Audit the provider refresh and the immutable locks it is meant to serve."""
    game_ids = [str(g["game_id"]) for g in games]
    if not game_ids:
        return {}
    rows = verify_locks(et_date, game_ids)
    for row in rows.values():
        row.update(
            fresh_normalized_rows=0,
            fresh_true_pair_rows=0,
            fresh_true_pair_rows_with_links=0,
        )
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(
        cursor_factory=psycopg2.extras.RealDictCursor
    ) as cur:
        cur.execute("SET LOCAL statement_timeout='45s'")
        cur.execute(
            """
            WITH locks AS (
              SELECT p.*, g.start_ts_utc,
                     NULLIF(p.forecast_payload->>'offer_id', '')::bigint AS payload_offer_id
              FROM bets.nfl_player_prop_predictions p
              JOIN raw.nfl_games g USING (game_id)
              WHERE p.game_date_et=%s AND p.game_id=ANY(%s)
                AND p.book='fanduel' AND p.is_current
            ), expected AS (
              SELECT DISTINCT l.game_id,o.provider,o.event_id,o.bookmaker_key,l.start_ts_utc,
                     o.player_name_norm,o.stat,o.line
              FROM locks l JOIN odds.nfl_player_prop_lines o ON o.id=l.payload_offer_id
            ), fresh AS (
              SELECT DISTINCT e.game_id,c.*
              FROM expected e
              JOIN odds.nfl_player_prop_lines c
                ON c.provider=e.provider AND c.event_id=e.event_id
               AND c.bookmaker_key=e.bookmaker_key
               AND c.player_name_norm=e.player_name_norm
               AND c.stat=e.stat AND c.line=e.line
              WHERE c.snapshot_role='close' AND c.fetched_at_utc >= %s
                AND c.fetched_at_utc >= e.start_ts_utc-interval '120 minutes'
                AND c.fetched_at_utc < e.start_ts_utc
            )
            SELECT game_id,
                   count(*)::int AS fresh_normalized_rows,
                   count(*) FILTER (WHERE abs(over_price)>=100 AND abs(under_price)>=100)::int
                     AS fresh_true_pair_rows,
                   count(*) FILTER (
                     WHERE abs(over_price)>=100 AND abs(under_price)>=100
                       AND over_link ~* '^https?://' AND under_link ~* '^https?://'
                       AND lower(over_link) LIKE '%%fanduel%%'
                       AND lower(under_link) LIKE '%%fanduel%%'
                   )::int AS fresh_true_pair_rows_with_links
            FROM fresh GROUP BY game_id
            """,
            (et_date, game_ids, since),
        )
        for row in cur.fetchall():
            rows[str(row["game_id"])].update(dict(row))
    for row in rows.values():
        row["verification_passed"] = verification_passed(row)
    return rows


def write_report(
    et_date,
    state: dict[str, Any],
    observations: list[dict[str, Any]],
    path_root: Path | None = None,
) -> dict[str, Any]:
    root = path_root or (ROOT / "reports")
    doc = {
        "contract": CONTRACT,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "date": str(et_date),
        "offsets_minutes": list(OFFSETS),
        "games": observations,
        "state": state,
    }
    atomic_json(root / f"nfl_pregame_checkpoints_{et_date}.json", doc)
    atomic_json(root / "nfl_pregame_checkpoints_latest.json", doc)
    lines = [
        "# NFL FanDuel Pregame Checkpoints",
        "",
        f"Date: {et_date}. Required captures: T-120, T-60, T-20 and T-5.",
        "",
        "| Game | Kickoff | Offset | Status | Attempts | True pairs / linked | Locks | Receiving triplets | Injury context |",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for game in observations:
        for checkpoint in game["checkpoints"]:
            v = checkpoint.get("verification") or {}
            lines.append(
                f"| {game['game_id']} | {game.get('kickoff') or '-'} | T-{checkpoint['offset_minutes']} | "
                f"{checkpoint['status']} | {checkpoint['attempts']} | "
                f"{v.get('fresh_true_pair_rows', 0)} / {v.get('fresh_true_pair_rows_with_links', 0)} | "
                f"{v.get('current_fanduel_locks', 0)} | {v.get('receiving_same_line_triplets', 0)} / "
                f"{v.get('receiving_locks', 0)} | {v.get('injury_context_note', '-')} |"
            )
    rendered = "\n".join(lines) + "\n"
    (root / f"nfl_pregame_checkpoints_{et_date}.md").write_text(rendered, encoding="utf-8")
    (root / "nfl_pregame_checkpoints_latest.md").write_text(rendered, encoding="utf-8")
    return doc
