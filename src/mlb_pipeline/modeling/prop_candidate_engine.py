"""Shared MLB prop candidate helpers.

This keeps presentation-facing candidate construction out of the large
prediction script.  Inputs are persisted prediction rows plus the normalized
prop line map; output is a small dict that Discord/report code can render.
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

import pandas as pd

from .bankroll_layers import BankrollAssessment

STAT_DISPLAY: dict[str, tuple[str, str]] = {
    "pitcher_strikeouts": ("K", "{:.1f}"),
    "batter_hits": ("H", "{:.3f}"),
    "batter_total_bases": ("TB", "{:.3f}"),
    "batter_home_runs": ("HR", "{:.3f}"),
}

STAT_SECTIONS: list[tuple[str, str]] = [
    ("pitcher_strikeouts", "Strikeouts"),
    ("batter_hits", "Hits"),
    ("batter_total_bases", "Total Bases"),
    ("batter_home_runs", "Home Runs"),
]


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "")
    ascii_str = s.encode("ascii", "ignore").decode("ascii")
    ascii_str = re.sub(r"[^a-z0-9\s]", "", ascii_str.lower())
    return re.sub(r"\s+", " ", ascii_str).strip()


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def book_label(link: str | None, fallback: str | None = None) -> str:
    if link:
        low = link.lower()
        if "fanduel.com" in low:
            return "FD"
        if "draftkings.com" in low:
            return "DK"
    fb = (fallback or "").lower()
    if fb == "fanduel":
        return "FD"
    if fb == "draftkings":
        return "DK"
    return "Bet"


def pick_score(item: dict[str, Any]) -> tuple[float, float]:
    ev = clean_float(item.get("ev"))
    edge = clean_float(item.get("edge"))
    return (
        ev if ev is not None else -999.0,
        abs(edge) if edge is not None else 0.0,
    )


def _same_line(a: Any, b: Any) -> bool:
    av = clean_float(a)
    bv = clean_float(b)
    return av is not None and bv is not None and abs(av - bv) <= 1e-9


def _current_exact_offer(
    line_data: dict[str, Any],
    *,
    prop_offer_id: Any,
    side: str,
    line: Any,
    bookmaker_key: Any,
) -> dict[str, Any] | None:
    offers = [dict(offer) for offer in (line_data.get("offers") or [])]
    exact = [
        offer for offer in offers
        if str(offer.get("side") or "").lower() == side
        and str(offer.get("bookmaker_key") or "").lower() == str(bookmaker_key or "").lower()
        and _same_line(offer.get("line"), line)
    ]
    if not exact:
        return None
    for offer in exact:
        if prop_offer_id is not None and str(offer.get("id")) == str(prop_offer_id):
            return offer
    return max(
        exact,
        key=lambda offer: (
            clean_float(offer.get("price")) if clean_float(offer.get("price")) is not None else -99999.0,
            1 if offer.get("link") else 0,
        ),
    )


def _append_reason(existing: str, reason: str) -> str:
    parts = [part.strip() for part in str(existing or "").split(";") if part.strip()]
    if reason not in parts:
        parts.append(reason)
    return "; ".join(parts)


def candidate_from_prediction_row(
    row: dict[str, Any],
    prop_lines: dict[tuple[str, str], dict[str, Any]],
    *,
    opponent: str | None = None,
) -> dict[str, Any] | None:
    stat = row.get("stat")
    if stat not in STAT_DISPLAY:
        return None
    side_raw = (row.get("bet_side") or "").lower()
    if side_raw not in {"over", "under"}:
        return None

    market, pred_fmt = STAT_DISPLAY[stat]
    name = row.get("player_name", f"id={row.get('player_id')}")
    line_data = prop_lines.get((normalize_name(name), stat), {})
    locked_line = clean_float(row.get("book_line"))
    locked_book = row.get("bookmaker_key")
    current_offer = _current_exact_offer(
        line_data,
        prop_offer_id=row.get("prop_offer_id"),
        side=side_raw,
        line=locked_line,
        bookmaker_key=locked_book,
    )
    link = (current_offer or {}).get("link") or row.get("bet_link")
    book = (current_offer or {}).get("bookmaker_key") or locked_book

    p_over = clean_float(row.get("pred_prob_over"))
    p_side = (1.0 - p_over) if side_raw == "under" and p_over is not None else p_over
    pred_val = clean_float(row.get("pred_count"))
    if pred_val is None:
        pred_val = clean_float(row.get("pred_value"))
    assessment = BankrollAssessment(
        tier=row.get("bankroll_tier") or "paper",
        candidate=bool(row.get("bankroll_candidate")),
        reasons=row.get("bankroll_reasons") or "",
        stake_pct=float(row.get("stake_pct") or 0.0),
        stake_usd=float(row.get("stake_usd") or 0.0),
    )
    entry_price = clean_float(row.get("bet_price"))
    current_price = clean_float((current_offer or {}).get("price"))
    minimum_price = clean_float(row.get("minimum_acceptable_price"))
    drift_reason = ""
    if current_offer is None:
        drift_reason = "offer_no_longer_available"
    elif current_price is None:
        drift_reason = "current_price_missing"
    elif minimum_price is None:
        drift_reason = "minimum_acceptable_price_missing"
    elif current_price < minimum_price:
        drift_reason = "price_drift_below_minimum"
    price_drift_ok = not drift_reason
    if assessment.candidate and not price_drift_ok:
        assessment = BankrollAssessment(
            tier="watch",
            candidate=False,
            reasons=_append_reason(assessment.reasons, drift_reason),
            stake_pct=0.0,
            stake_usd=0.0,
        )
    return {
        "market": market,
        "stat_key": stat,
        "side": "O" if side_raw == "over" else "U",
        "name": name,
        "team": row.get("team_abbr", "?"),
        "opp": opponent or row.get("opponent_abbr") or "?",
        "pred_val": pred_val if pred_val is not None else 0.0,
        "pred_fmt": pred_fmt,
        "line": locked_line or clean_float(line_data.get("line")) or 0.0,
        "p_over": p_side,
        "ev": clean_float(row.get("ev")),
        "edge": clean_float(row.get("edge")),
        "link": link,
        "prop_offer_id": row.get("prop_offer_id"),
        "entry_price": entry_price,
        "current_price": current_price,
        "minimum_acceptable_price": minimum_price,
        "price_drift_ok": price_drift_ok,
        "book": book_label(link, book),
        "bankroll": assessment,
    }


def sort_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=pick_score, reverse=True)


def candidates_by_stat(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {stat: [] for stat, _label in STAT_SECTIONS}
    for candidate in candidates:
        stat = candidate.get("stat_key")
        if stat in grouped:
            grouped[stat].append(candidate)
    for stat in grouped:
        grouped[stat] = sort_candidates(grouped[stat])
    return grouped
