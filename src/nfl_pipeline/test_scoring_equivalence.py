from copy import deepcopy
import pytest

from nfl_pipeline.modeling import scoring_equivalence as eq


SOURCE = '''
def _load_player_context():
    return "input-v1"
def build_predictions():
    return _load_player_context()
def _candidate_from_offer(row, stat, projection, baseline, metrics, offer, distribution,
                          min_ev=0, **kwargs):
    return dict(side="over", probability=.6)
'''


def archives(tmp_path, monkeypatch, texts):
    monkeypatch.setattr(eq, 'STORE', tmp_path)
    monkeypatch.setattr(eq, 'dependency_digest', lambda: 'dependencies')
    functions = {}
    versions = []
    for text in texts:
        stage = tmp_path / 'staging'
        stage.mkdir()
        (stage / eq.FILES[0]).write_text(text)
        (stage / eq.FILES[1]).write_text('CONTRACT = "capture-v1"\n')
        version = eq.source_hash(stage)
        stage.rename(tmp_path / version)
        module = {}
        exec(compile(text, '<test scoring>', 'exec'), module)
        functions[version] = module['_candidate_from_offer']
        versions.append(version)
    monkeypatch.setattr(eq, 'archived_candidate', functions.__getitem__)
    return versions


def record(ident, version):
    return dict(id=ident, season=2026, week=ident, forecast_payload=dict(side='over', probability=.6,
        scoring_replay=dict(scoring_fingerprint=version, row={}, stat='receiving_yards', projection=30,
            baseline=30, metrics={}, offer={}, distribution={}, min_ev=.02, calibration=[], clv_guards=[])))


def test_comments_can_pool_only_after_full_replay(tmp_path, monkeypatch):
    a, b = archives(tmp_path, monkeypatch, [SOURCE, SOURCE+'\n# Operational comment\n'])
    records = [record(1, a), record(2, b)]
    before = deepcopy(records)
    result = eq.audit(records)
    assert result['groups'][0]['pooled']
    assert result['groups'][0]['replayed_rows'] == 2
    aliases = eq.verified_aliases(result)
    assert eq.cohort_for(a, 1, aliases) == eq.cohort_for(b, 2, aliases)
    assert eq.cohort_for(a, 999, aliases) == a
    assert eq.cohort_for('another-version', 1, aliases) == 'another-version'
    assert records == before


def test_true_probability_change_does_not_pool(tmp_path, monkeypatch):
    a, b = archives(tmp_path, monkeypatch, [SOURCE, SOURCE.replace('probability=.6', 'probability=.7')])
    second = record(2, b); second['forecast_payload']['probability'] = .7
    result = eq.audit([record(1, a), second])
    assert not result['aliases']
    assert len(result['groups']) == 2
    assert not result['comparisons'][0]['same_scoring_program']


def test_same_pricing_with_changed_inputs_remains_separate(tmp_path, monkeypatch):
    a, b = archives(tmp_path, monkeypatch, [SOURCE, SOURCE.replace('input-v1', 'input-v2')])
    result = eq.audit([record(1, a), record(2, b)])
    assert not result['aliases']
    assert result['comparisons'][0]['same_scoring_program']
    assert not result['comparisons'][0]['same_input_program']


def test_original_replay_mismatch_rejects_whole_alias(tmp_path, monkeypatch):
    a, b = archives(tmp_path, monkeypatch, [SOURCE, SOURCE+'\n# comment\n'])
    second = record(2, b); second['forecast_payload']['probability'] = .9
    result = eq.audit([record(1, a), second])
    assert not result['aliases']
    assert result['exclusions']['original_probability_mismatch'] == 1


def test_changed_dependencies_or_archive_invalidate_proof(tmp_path, monkeypatch):
    a, b = archives(tmp_path, monkeypatch, [SOURCE, SOURCE+'\n# comment\n'])
    result = eq.audit([record(1, a), record(2, b)])
    monkeypatch.setattr(eq, 'dependency_digest', lambda: 'changed')
    assert not eq.verified_aliases(result)
    monkeypatch.setattr(eq, 'dependency_digest', lambda: 'dependencies')
    (tmp_path / a / eq.FILES[0]).write_text(SOURCE+'\n# tampered\n')
    assert not eq.verified_aliases(result)


def test_timing_wrapper_does_not_change_input_signature(tmp_path):
    reference = tmp_path/'reference'; timed = tmp_path/'timed'
    reference.mkdir(); timed.mkdir()
    (reference/eq.FILES[0]).write_text(SOURCE)
    (timed/eq.FILES[0]).write_text(SOURCE.replace('return _load_player_context()',
        'from nfl_pipeline.runtime_timing import StageTimer\n    timer = StageTimer(log)\n'
        '    return timer.call("context", _load_player_context)'))
    for root in (reference, timed):
        (root/eq.FILES[1]).write_text('CONTRACT = "same"')
    a, b = eq.program_signatures(reference), eq.program_signatures(timed)
    assert a['scoring'] == b['scoring']
    assert a['inputs'] == b['inputs']


def test_archive_checksum_cache_invalidates_on_file_change(tmp_path, monkeypatch):
    from nfl_pipeline.modeling import scoring_versions as versions
    stage = tmp_path/'staging'; stage.mkdir()
    (stage/versions.FILES[0]).write_text('def _candidate_from_offer():\n    return .5\n')
    (stage/versions.FILES[1]).write_text('# captured')
    fingerprint = versions.source_hash(stage)
    root = tmp_path/fingerprint; stage.rename(root)
    monkeypatch.setattr(versions, 'STORE', tmp_path)
    monkeypatch.setattr(versions, '_loaded', {})
    monkeypatch.setattr(versions, '_verified_sources', {})
    original = versions.source_hash; calls = []
    def counted(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(versions, 'source_hash', counted)
    assert versions.archived_candidate(fingerprint)() == .5
    assert versions.archived_candidate(fingerprint)() == .5
    assert len(calls) == 1
    (root/versions.FILES[0]).write_text('def _candidate_from_offer():\n    return .7 # changed\n')
    with pytest.raises(ValueError, match='checksum_mismatch'):
        versions.archived_candidate(fingerprint)
