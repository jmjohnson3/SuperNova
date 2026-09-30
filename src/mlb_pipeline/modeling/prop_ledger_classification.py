"""Canonical ledger classification for MLB player-prop records."""
from __future__ import annotations

import json
import math
from typing import Any, Mapping


PROP_LEDGER_LABELS: tuple[tuple[str, str], ...] = (
    ("bankroll", "Bankroll props"),
    ("micro", "Micro props"),
    ("micro_external_agreement", "Micro external-agreement props"),
    ("micro_projection", "Micro projection props"),
    ("watch", "Watch props"),
    ("paper_common", "Paper common props"),
    ("lottery", "Lottery props"),
    ("one_sided_fanduel", "One-sided FanDuel props"),
)

_HITTER_MARKETS = {"batter_hits", "batter_total_bases", "batter_home_runs"}


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _meta(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("model_meta")
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def classify_prop_ledger(row: Mapping[str, Any]) -> str:
    """Classify one prop record using the same priority as Discord reporting."""
    meta = _meta(row)
    book = _text(row.get("bookmaker_key"))
    market = _text(row.get("market") or row.get("stat"))
    side = _text(row.get("side"))
    tier = _text(row.get("model_tier"))
    selector_tier = _text(meta.get("selector_tier") or row.get("selector_tier"))
    pair_quality = _text(meta.get("pair_quality") or row.get("pair_quality"))
    market_source = _text(meta.get("market_prob_source") or row.get("market_prob_source"))
    reasons = " ".join(
        filter(
            None,
            (
                _text(row.get("warning_reasons")),
                _text(meta.get("selector_reasons")),
                _text(row.get("selector_reasons")),
            ),
        )
    )
    line = _float(row.get("market_line") if row.get("market_line") is not None else row.get("bet_line"))

    explicit_one_sided = (
        any(token in reasons for token in ("fanduel_synthetic", "one_sided", "one-sided", "synthetic"))
        or pair_quality in {"synthetic", "one_sided"}
        or market_source
        in {
            "raw_implied_one_sided",
            "one_sided",
            "one_sided_fanduel_ladder",
            "synthetic_fanduel_over_only",
        }
    )
    historical_one_sided = side == "over" and pair_quality not in {"same_book", "cross_book"}
    if book == "fanduel" and market in _HITTER_MARKETS and (explicit_one_sided or historical_one_sided):
        return "one_sided_fanduel"

    is_tail = bool(
        side == "over"
        and line is not None
        and (
            (market == "batter_hits" and line >= 1.5)
            or (market == "batter_total_bases" and line >= 2.5)
            or (market == "batter_home_runs" and line >= 1.5)
        )
    )
    if selector_tier == "lottery" or "alt_line_lottery" in reasons or "lottery" in reasons or is_tail:
        return "lottery"
    if (
        selector_tier == "micro_projection"
        and _text(meta.get("micro_external_agreement") or row.get("micro_external_agreement")) in {"1", "true", "yes", "y", "on"}
    ):
        return "micro_external_agreement"
    if tier == "micro_projection" and _text(meta.get("micro_external_agreement") or row.get("micro_external_agreement")) in {"1", "true", "yes", "y", "on"}:
        return "micro_external_agreement"
    if selector_tier == "micro_projection" or tier == "micro_projection":
        return "micro_projection"
    if tier == "micro":
        return "micro"
    if tier in {"bankroll", "starter"}:
        return "bankroll"
    if tier == "watch":
        return "watch"
    return "paper_common"
