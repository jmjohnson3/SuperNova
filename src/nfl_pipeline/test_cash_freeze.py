"""Legacy cash cannot reuse evidence or authority across dependency drift."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from nfl_pipeline import cash_freeze as freeze
from nfl_pipeline import cash_trial_policy as policy
from nfl_pipeline import cash_readiness as readiness
from nfl_pipeline import cash_publication as publication
from nfl_pipeline import cash_execution as execution

NOW = datetime(2026, 10, 18, 12, tzinfo=timezone.utc)


def config():
    return dict(trial_sha256='trial', release='release', scoring_version='two-file-score', variant='ensemble',
        artifact_sha256='artifact', stat='receiving_yards', source_kind='prop', book='fanduel',
        research_registration=dict(config={}))


def reference():
    return dict(experiment_id='cash-test', sha256='a'*64)


def registration():
    return dict(sha256='registration', config=dict(config(), frozen_provenance=reference(),
        registered_at=(NOW-timedelta(days=1)).isoformat()))


def stamp():
    return dict(reference(), status='verified')


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    source = tmp_path/'scorer.py'; source.write_text('score = 1\n')
    optional = tmp_path/'calibration.json'
    models = tmp_path/'src/nfl_pipeline/modeling/models'; models.mkdir(parents=True)
    registry = models/'stat_model_registry.json'
    registry.write_text(json.dumps(dict(base_release='release', active={}, candidates={})))
    monkeypatch.setattr(freeze, 'dependency_inputs', lambda config, root: (
        {'source':source}, {'calibration':optional}, freeze._registry_selection(models)))
    store = tmp_path/'manifests'
    cfg = config()
    cfg[freeze.FIELD] = freeze.create_freeze('cash-test', cfg, root=tmp_path, store=store)
    reg = policy.register(cfg, store=tmp_path/'cash')
    return reg, tmp_path, store, source, optional, registry


def test_dependency_source_and_absent_calibration_are_frozen(frozen):
    reg, root, store, source, optional, _ = frozen
    assert freeze.check_registration(reg, root=root, store=store)['sha256'] == reg['config'][freeze.FIELD]['sha256']
    source.write_text('score = 2\n')
    with pytest.raises(ValueError, match='changed'):
        freeze.check_registration(reg, root=root, store=store)
    source.write_text('score = 1\n'); optional.write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        freeze.check_registration(reg, root=root, store=store)


def test_unrelated_candidates_can_develop_but_active_changes_block(frozen):
    reg, root, store, _, _, registry = frozen
    doc = json.loads(registry.read_text())
    doc['candidates'] = {'receiving_yards': {'another-shadow': {'artifact':'new-research'}}}
    registry.write_text(json.dumps(doc))
    freeze.check_registration(reg, root=root, store=store)
    doc['active'] = {'receiving_yards': {'median': {'component_id':'changed'}}}
    registry.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='active stat'):
        freeze.check_registration(reg, root=root, store=store)


def test_attach_freeze_requires_explicit_rollover_and_archives_old(tmp_path):
    old = policy.register(config(), store=tmp_path)
    cfg = dict(config(), frozen_provenance=reference())
    with pytest.raises(ValueError, match='changed'):
        policy.register(cfg, store=tmp_path)
    new = policy.rollover_scoring(cfg, 'Explicit dependency freeze', store=tmp_path, now=NOW)
    assert new['sha256'] != old['sha256']
    assert new['config']['scoring_version'] == old['config']['scoring_version']
    assert new['config']['registered_at'] == NOW.isoformat()
    assert json.loads((tmp_path/'registrations'/f"{old['sha256']}.json").read_text()) == old
    assert policy.rollover_scoring(cfg, 'Idempotent retry', store=tmp_path) == new
    with pytest.raises(ValueError, match='cannot be removed'):
        policy.rollover_scoring(config(), 'Remove freeze', store=tmp_path)
    with pytest.raises(ValueError, match='outside scoring'):
        policy.rollover_scoring(dict(cfg, variant='different'), 'Changed strategy', store=tmp_path)


def test_generation_stamps_required_for_original_and_benchmark(monkeypatch):
    reg = registration()
    assert freeze.provenance_error(stamp(), reg) is None
    assert freeze.provenance_error(None, reg)
    assert freeze.provenance_error(dict(stamp(), sha256='b'*64), reg)
    assert freeze.eligible_documents([{'cash_freeze':stamp()}, {}, {'cash_freeze':{'status':'blocked'}}], reg) == [{'cash_freeze':stamp()}]
    rec = dict(id=1, created_at_utc=NOW, forecast_payload=dict(model_version='release',
        scoring_replay={'scoring_fingerprint':'two-file-score'}))
    monkeypatch.setattr(readiness.checkpoint, 'capture_index', lambda records, documents, config: ({}, {}))
    selected, rows, excluded, unresolved, pending = readiness.evidence([rec], [], reg, [{'forecast_id':1}], {}, NOW)
    assert not selected and not rows and unresolved == 1 and pending == 0
    assert excluded['missing_or_mismatched_cash_generation_provenance'] == 1


def test_capture_stamp_cache_invalidates_on_source_change(tmp_path, monkeypatch):
    path = tmp_path/'registration.json'
    cfg = dict(config(), frozen_provenance=reference())
    reg = dict(config=cfg, sha256=policy.digest(cfg)); path.write_text(json.dumps(reg))
    monkeypatch.setattr(freeze, 'CASH_REGISTRATION', path)
    signature = [1]; calls = []
    monkeypatch.setattr(freeze, '_capture_stamp', lambda raw: signature[0])
    def check(registration):
        calls.append(signature[0])
        if signature[0] == 2:
            raise ValueError('source drift')
        return reference()
    monkeypatch.setattr(freeze, 'check_registration', check)
    freeze._verified_capture.cache_clear()
    assert freeze.capture_provenance()['status'] == 'verified'
    assert freeze.capture_provenance()['status'] == 'verified'
    assert calls == [1]
    signature[0] = 2
    assert freeze.capture_provenance()['status'] == 'blocked'
    freeze._verified_capture.cache_clear()


def test_generation_without_registered_freeze_remains_forecast_only(tmp_path, monkeypatch):
    path = tmp_path/'registration.json'
    monkeypatch.setattr(freeze, 'CASH_REGISTRATION', path)
    assert freeze.capture_provenance() is None
    path.write_text(json.dumps({'config':config()}))
    assert freeze.capture_provenance() is None


def test_report_checks_freeze_before_database(monkeypatch):
    monkeypatch.setattr(readiness.policy, 'load_registration', lambda: registration())
    monkeypatch.setattr(readiness.cash_freeze, 'check_registration', lambda reg: (_ for _ in ()).throw(ValueError('freeze drift')))
    monkeypatch.setattr(readiness, 'load_review_inputs', lambda *args: pytest.fail('Evidence must not load on drift'))
    with pytest.raises(ValueError, match='freeze drift'):
        readiness.report(NOW.date())


def test_publication_downgrades_eligible_report_on_freeze_drift(tmp_path, monkeypatch):
    folder=tmp_path/'reports';folder.mkdir()
    doc=dict(registration_sha256='registration',built_at=datetime.now(timezone.utc).isoformat(),
        decision={'status':'cash_trial_eligible'}, frozen_provenance=reference(), dependency_freeze_status='verified')
    (folder/f'nfl_cash_readiness_{NOW.date()}.json').write_text(json.dumps(doc))
    monkeypatch.setattr(publication, 'ROOT', tmp_path)
    monkeypatch.setattr(publication.policy, 'load_registration', lambda: registration())
    monkeypatch.setattr(publication.cash_freeze, 'check_registration', lambda reg: (_ for _ in ()).throw(ValueError('freeze drift')))
    _, rows, status = publication.prepare(NOW.date(), [], [{'tier':'cash_trial'}], reserve=True)
    assert rows[0]['tier'] == 'paper'
    assert status['status'] == 'operational_issue' and 'freeze drift' in status['reason']


def test_reservation_cannot_bypass_freeze_or_generation_provenance(monkeypatch, tmp_path):
    monkeypatch.setattr(execution.policy, 'STORE', tmp_path)
    monkeypatch.setattr(execution.cash_freeze, 'check_registration', lambda reg: (_ for _ in ()).throw(ValueError('freeze drift')))
    monkeypatch.setattr(execution, 'lock', lambda conn: pytest.fail('No DB work on invalid freeze'))
    with pytest.raises(ValueError, match='freeze drift'):
        execution.reserve(None, [{}], registration(), 'cash_trial_eligible', NOW)
    monkeypatch.setattr(execution.cash_freeze, 'check_registration', lambda reg: {})
    monkeypatch.setattr(execution, 'lock', lambda conn: None)
    monkeypatch.setattr(execution, 'states', lambda conn: [])
    row = dict(model_version='release', stat='receiving_yards', cash_artifact_sha256='artifact',
        probability_source='ensemble', prediction_context_cutoff_utc=NOW.isoformat(),
        scoring_replay={'scoring_fingerprint':'two-file-score'})
    accepted, excluded = execution.reserve(None, [row], registration(), 'cash_trial_eligible', NOW)
    assert not accepted and excluded['missing_or_mismatched_cash_generation_provenance'] == 1
    row['scoring_replay']['cash_freeze'] = stamp(); row['cash_artifact_sha256']='wrong'
    accepted, excluded = execution.reserve(None, [row], registration(), 'cash_trial_eligible', NOW)
    assert not accepted and excluded['different_cash_execution_cohort'] == 1


def test_registration_wrapper_never_silently_creates_freeze(monkeypatch):
    monkeypatch.setattr(readiness, 'registration_config', config)
    monkeypatch.setattr(readiness.cash_freeze, 'create_freeze', lambda *args: pytest.fail('No automatic registration'))
    with pytest.raises(ValueError, match='freeze-name'):
        readiness.registration_with_freeze(None)
