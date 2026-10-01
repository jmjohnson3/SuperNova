"""Capture the actual live scoring arguments without changing scoring behavior."""
from __future__ import annotations

import hashlib
import inspect
from functools import lru_cache
from pathlib import Path
from nfl_pipeline.forecast_store import clean


@lru_cache(maxsize=1)
def scoring_fingerprint():
    from nfl_pipeline.modeling import predict_player_props as live
    # Include helper implementations, not just the public candidate function.
    paths = [Path(inspect.getfile(live)), Path(__file__)]
    fingerprint = hashlib.sha256(b''.join(path.read_bytes() for path in paths)).hexdigest()
    from nfl_pipeline.modeling.scoring_versions import archive_current
    if archive_current() != fingerprint:
        raise ValueError('Scoring source changed during capture')
    return fingerprint


def capture(row, stat, projection, baseline, metrics, offer, distribution, calibration, guards, min_ev, artifact):
    exact = (artifact or {}).get('exact_line_artifact')
    return clean({
        'version': 'nfl-live-scoring-v1', 'scoring_fingerprint': scoring_fingerprint(),
        'row': row, 'stat': stat, 'projection': projection, 'baseline': baseline,
        'metrics': metrics, 'offer': offer, 'distribution': distribution,
        'calibration': [{'key': list(k), 'value': v} for k,v in (calibration or {}).items()],
        'clv_guards': [{'key': list(k), 'value': v} for k,v in (guards or {}).items()],
        'min_ev': min_ev, 'release_id': (artifact or {}).get('version'),
        'exact_overlay_present': bool(exact),
    })


def replay(captured):
    from nfl_pipeline.modeling.predict_player_props import _candidate_from_offer
    if captured.get('scoring_fingerprint') != scoring_fingerprint():
        from nfl_pipeline.modeling.scoring_versions import archived_candidate
        _candidate_from_offer = archived_candidate(captured.get('scoring_fingerprint'))
    if captured.get('exact_overlay_present'):
        raise ValueError('exact_overlay_artifact_not_captured')
    return _candidate_from_offer(captured['row'], captured['stat'], captured['projection'],
        captured['baseline'], captured['metrics'], captured['offer'], captured['distribution'],
        min_ev=captured['min_ev'], probability_calibration_by_key={tuple(r['key']):r['value'] for r in captured['calibration']},
        clv_guard_by_key={tuple(r['key']):r['value'] for r in captured['clv_guards']}, model_artifact={})
