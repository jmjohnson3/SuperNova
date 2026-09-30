"""Timestamped, nullable same-week evidence; proxies are never measured routes."""
from __future__ import annotations

import math
from nfl_pipeline.integrity import utc


def present(value):
    return value is not None and str(value).strip().lower() not in {'', 'nan', 'nat', 'none', '<na>'}


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def safe_time(value):
    try:
        return utc(value) if present(value) else None
    except (ValueError, TypeError, OverflowError):
        return None


def validate_evidence(evidence, lock):
    if not isinstance(evidence, dict):
        return None
    cutoff=safe_time(evidence.get('cutoff')); lock=safe_time(lock)
    if not cutoff or not lock or cutoff>lock:
        return None
    for key in ('injury_observed_at','depth_observed_at','roster_observed_at'):
        value=evidence.get(key)
        when=safe_time(value)
        if present(value) and (not when or when>cutoff):
            return None
    return evidence


def context_evidence(row, cutoff):
    cutoff = utc(cutoff)
    def observed(name):
        value = row.get(name)
        when = safe_time(value)
        return when if when and when <= cutoff else None

    injury_at = observed('injury_observed_at')
    depth_at = observed('depth_observed_at')
    roster_at = observed('roster_observed_at')
    rank = number(row.get('depth_pos_rank')) if depth_at else None
    previous = number(row.get('previous_depth_pos_rank')) if depth_at else None
    routes = number(row.get('routes_run_avg_5'))
    proxy = number(row.get('route_participation_proxy_avg_5'))
    return {
        'contract': 'nfl-context-evidence-v1', 'cutoff': cutoff.isoformat(),
        'injury_status': row.get('injury_report_status') if injury_at else None,
        'practice_status': row.get('injury_practice_status') if injury_at else None,
        'injury_observed_at': injury_at.isoformat() if injury_at else None,
        'injury_missing': injury_at is None,
        'depth_rank': rank, 'depth_movement': previous-rank if rank is not None and previous is not None else None,
        'expected_starter_from_depth': rank == 1 if rank is not None else None,
        'depth_observed_at': depth_at.isoformat() if depth_at else None,
        'roster_observed_at': roster_at.isoformat() if roster_at else None,
        'teammate_injury_count': number(row.get('same_week_teammate_skill_injury_count')) if injury_at else None,
        'teammate_coverage': 'partial_observed_reports' if injury_at else 'unknown',
        'routes_run_history': routes,
        'routes_source': 'observed_prior_games' if routes is not None else 'proxy_only' if proxy is not None else 'missing',
        'route_participation_proxy': proxy,
        'first_read_history': number(row.get('first_read_targets_avg_5')),
        'missing_is_not_zero': True,
    }
