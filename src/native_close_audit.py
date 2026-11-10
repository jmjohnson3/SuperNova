"""Separate close reader using only previously saved, strictly pregame quotes.

Does not replace frozen reports or rewrite decisions. No historical API calls.
"""
from datetime import timedelta

from automatic_price_checks import rows
from nfl_pipeline.context_contract import safe_time
from model_research_report import exact_close
from sharp_pipeline import report


METHOD='latest_saved_family_snapshot_before_earliest_schedule_v1'


def select_saved(offer,native_tip,snapshots):
    native,provider=safe_time(native_tip),safe_time(offer.get('commence_time_utc'))
    decided=safe_time(offer.get('decided_at'))
    if not native or not provider or not decided:return {}
    cutoff=min(native,provider)
    candidates=[r for r in snapshots if safe_time(r.get('observed_at_utc'))
        and max(decided,cutoff-timedelta(minutes=report.MAX_CLOSE_AGE_MINUTES))<=safe_time(r['observed_at_utc'])<cutoff
        and report.payload_kind(r['payload'])==report.row_kind(offer)]
    if not candidates:return {}
    at=max(safe_time(r['observed_at_utc']) for r in candidates)
    last=[r for r in candidates if safe_time(r['observed_at_utc'])==at]
    # A tied last timestamp with different responses is ambiguous, never pick a favorable close.
    if any(r['payload']!=last[0]['payload'] for r in last[1:]):return {}
    event=last[0]['payload']
    if str(event.get('id'))!=str(offer['event_id']):return {}
    if event.get('home_team')!=offer.get('home_team') or event.get('away_team')!=offer.get('away_team'):return {}
    evidence=report.close_evidence(offer,{(offer['sport'],offer['event_id'],report.row_kind(offer)):(at,event)})
    if evidence.get('method')!='exact':return {}
    updated=safe_time(evidence.get('source_updated_at'))
    if not updated or updated>=cutoff:return {}
    return dict(evidence,cutoff=cutoff.isoformat(),audit_reader=METHOD)


def load(conn,offer,native_tip):
    native,provider=safe_time(native_tip),safe_time(offer.get('commence_time_utc'))
    if not native or not provider:return {}
    if offer['sport']=='nfl':
        # The pinned NFL reader already enforces this cutoff; mismatches remain unknown.
        return exact_close(conn,offer) if native==provider else {}
    cutoff=min(native,provider)
    found=rows(conn,"""SELECT observed_at_utc,payload FROM bets.sharp_multi_quotes
        WHERE sport='nba' AND event_id=%s AND observed_at_utc>=%s AND observed_at_utc<%s
        ORDER BY observed_at_utc DESC""",(offer['event_id'],max(safe_time(offer['decided_at']),
                        cutoff-timedelta(minutes=report.MAX_CLOSE_AGE_MINUTES)),cutoff))
    return select_saved(offer,native,found)
