"""Dependency provenance for the legacy cash cohort; never authorizes a wager.

Only an explicit registration command creates a freeze. Forecast generation can
stamp verification without turning a blocked cash experiment into a forecast failure.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path

from nfl_pipeline import experiment_registry as experiments

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / 'src/nfl_pipeline/modeling/models'
CASH_REGISTRATION = MODEL_ROOT / 'cash_trial/registration.json'
FIELD = 'frozen_provenance'
CONTRACT = 'nfl-legacy-cash-dependency-freeze-v1'
METADATA = {'registered_at', 'policy', 'supersedes_registration_sha256', 'rollover_reason', FIELD}


def basis(config):
    return {key: value for key, value in config.items() if key not in METADATA}


def _registry_selection(model_root):
    path = model_root / 'stat_model_registry.json'
    registry = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    return dict(base_release=registry.get('base_release'), active=registry.get('active') or {})


def dependency_inputs(config, *, root=ROOT):
    """Pin runtime dependencies, never the mutable cash registration/report itself."""
    root = Path(root)
    model_root = root / 'src/nfl_pipeline/modeling/models'
    files = experiments.source_dependencies(
        'nfl_pipeline.cash_freeze', 'nfl_pipeline.cash_readiness', 'nfl_pipeline.cash_trial_policy',
        'nfl_pipeline.cash_publication', 'nfl_pipeline.cash_execution',
        'nfl_pipeline.modeling.predict_player_props', 'nfl_pipeline.modeling.scoring_capture',
        'nfl_pipeline.modeling.benchmark_offers', 'nfl_pipeline.modeling.receiving_research_trial', root=root)
    absent = {}
    files['active_release'] = model_root / 'active_release.json'
    folder = model_root / 'releases' / config['release']
    for file in folder.iterdir():
        if file.is_file():
            files['release_' + file.name] = file
    files['benchmark_pointer'] = model_root / 'model_benchmark/active_trial.json'
    files['research_registration'] = model_root / 'receiving_research_trial/registration.json'
    research = config['research_registration']['config']
    files['benchmark_artifact'] = model_root / 'model_benchmark' / research['run_id'] / 'models.joblib'
    for relative in ('player_props/market_calibration.json', 'player_props/interval_calibration.json',
                     'distribution_policy/active.json'):
        path = model_root / relative
        (files if path.exists() else absent)['optional_' + relative] = path
    # Active policy descriptors may load a separate runtime model binary.
    active_path = model_root / 'distribution_policy/active.json'
    if active_path.exists():
        active = json.loads(active_path.read_text(encoding='utf-8'))
        for key, policy in (active.get('policies') or {}).items():
            descriptor = policy.get('source_artifact') or {}
            if descriptor.get('path'):
                files['distribution_source_' + key] = Path(descriptor['path'])
    selection = _registry_selection(model_root)
    components = []
    for outputs in selection['active'].values():
        components.extend(outputs.values())
    for entry in components:
        if entry.get('artifact'):
            files['stat_artifact_' + entry['component_id']] = Path(entry['artifact'])
    return files, absent, selection


def create_freeze(name, config, *, root=ROOT, store=experiments.STORE):
    """Explicit only: create an immutable manifest before attaching a new cohort."""
    files, absent, selection = dependency_inputs(config, root=root)
    frozen = experiments.register_experiment(name, dict(contract=CONTRACT, cash_basis=basis(config),
        stat_registry_selection=selection), files, absent_paths=absent, root=root, store=store)
    return dict(experiment_id=frozen['experiment_id'], sha256=frozen['sha256'])


def check_registration(registration, *, root=ROOT, store=experiments.STORE):
    cfg = registration.get('config') or {}
    reference = cfg.get(FIELD)
    if not isinstance(reference, dict) or set(reference) != {'experiment_id', 'sha256'}:
        raise ValueError('Cash dependency freeze missing; explicit frozen cohort registration required')
    frozen = experiments.check_experiment(reference['experiment_id'], root=root, store=store)
    if frozen['sha256'] != reference['sha256'] or frozen['config'].get('contract') != CONTRACT:
        raise ValueError('Cash dependency freeze identity mismatch')
    if frozen['config'].get('cash_basis') != basis(cfg):
        raise ValueError('Cash strategy differs from its dependency freeze')
    if frozen['config'].get('stat_registry_selection') != _registry_selection(Path(root)/'src/nfl_pipeline/modeling/models'):
        raise ValueError('Cash active stat components changed')
    return frozen


def provenance_error(provenance, registration):
    reference = (registration.get('config') or {}).get(FIELD)
    if not reference:
        return 'missing_cash_dependency_freeze'
    if (not isinstance(provenance, dict) or provenance.get('status') != 'verified'
            or any(provenance.get(key) != reference.get(key) for key in ('experiment_id', 'sha256'))):
        return 'missing_or_mismatched_cash_generation_provenance'
    return None


def eligible_documents(documents, registration):
    return [doc for doc in documents if provenance_error(doc.get('cash_freeze'), registration) is None]


def _capture_stamp(raw_registration):
    """Cheap metadata invalidation avoids hashing model binaries per player."""
    registration = json.loads(raw_registration)
    reference = registration['config'][FIELD]
    frozen = experiments.load_experiment(reference['experiment_id'])
    if frozen is None:
        raise ValueError('Cash dependency freeze is not registered')
    stamps = []
    for label, entry in sorted(frozen['files'].items()):
        path = ROOT / entry['path']
        try:
            info = path.stat()
            stamps.append((label, info.st_mtime_ns, info.st_size))
        except FileNotFoundError:
            stamps.append((label, None, None))
    return frozen['sha256'], tuple(stamps), experiments._digest(_registry_selection(MODEL_ROOT))


@lru_cache(maxsize=8)
def _verified_capture(raw_registration, dependency_stamp):
    registration = json.loads(raw_registration)
    from nfl_pipeline.cash_trial_policy import digest
    if registration.get('sha256') != digest(registration.get('config')):
        raise ValueError('Cash registration checksum mismatch')
    frozen = check_registration(registration)
    return dict(status='verified', experiment_id=frozen['experiment_id'], sha256=frozen['sha256'],
                verified_at=datetime.now(timezone.utc).isoformat())


def capture_provenance():
    """Cache unchanged file metadata per process; drifted cash stays paper.

    Readiness and publication independently recheck files without this cache. A
    stored forecast and its benchmark capture must both carry the matching stamp.
    """
    try:
        if not CASH_REGISTRATION.exists():
            return None
        raw = CASH_REGISTRATION.read_text(encoding='utf-8')
        if not (json.loads(raw).get('config') or {}).get(FIELD):
            return None
        return dict(_verified_capture(raw, _capture_stamp(raw)))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return dict(status='blocked', reason=str(exc))
