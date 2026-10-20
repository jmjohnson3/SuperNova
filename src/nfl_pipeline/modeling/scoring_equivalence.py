"""Conservative, replay-proven evaluation aliases; never rewrite forecast versions."""
import argparse
import ast
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling.scoring_versions import STORE, SOURCE, FILES, source_hash, archived_candidate

REPORT = SOURCE.parents[2] / 'reports/nfl_scoring_equivalence_latest.json'
# Reviewed boundaries: these create inputs, not conditional offer probabilities.
# They still participate in the separate input-program hash below.
INPUT_FUNCTIONS = {'_load_player_context', 'build_predictions'}
PROBES = (None, {'projection_trust': .35, 'range_scale': 1.4, 'probability_trust': .65})


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class TimingOnly(ast.NodeTransformer):
    """Remove only the reviewed StageTimer wrappers, never forecast operations."""
    def visit_ImportFrom(self, node):
        if node.module == 'nfl_pipeline.runtime_timing' and [n.name for n in node.names] == ['StageTimer']:
            return None
        return node

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'timer' and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == 'StageTimer'):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'timer' and node.func.attr == 'call' and len(node.args) >= 2):
            return ast.Call(func=node.args[1], args=node.args[2:], keywords=node.keywords)
        return node

    def visit_Dict(self, node):
        pairs = [(k, v) for k, v in zip(node.keys, node.values)
                 if not (isinstance(k, ast.Constant) and k.value == 'stage_timings'
                         and isinstance(v, ast.Attribute) and ast.unparse(v) == 'timer.stages')]
        node.keys, node.values = [k for k, _ in pairs], [v for _, v in pairs]
        return self.generic_visit(node)


def program_signatures(root):
    """No heuristic function similarity: AST equality, with input changes separate."""
    tree = ast.parse((root / FILES[0]).read_text(encoding='utf-8'))
    scoring, inputs = [], []
    functions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = digest(ast.dump(node, include_attributes=False))
        if isinstance(node, ast.FunctionDef) and node.name in INPUT_FUNCTIONS:
            inputs.append(ast.dump(TimingOnly().visit(deepcopy(node)), include_attributes=False))
        else:
            scoring.append(ast.dump(node, include_attributes=False))
    capture = ast.dump(ast.parse((root / FILES[1]).read_text(encoding='utf-8')), include_attributes=False)
    return dict(scoring=digest([scoring, capture]), inputs=digest(inputs), functions=functions)


def dependency_digest():
    # Old archives did not freeze imported local helpers. Bind proof to the exact
    # current helper tree, and still demand original-probability reproduction.
    root = SOURCE.parent
    files = sorted(p for p in root.rglob('*.py') if 'models' not in p.relative_to(root).parts
                   and not p.name.startswith('test_') and p.name != 'scoring_equivalence.py')
    return digest([(str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])


def invoke(fingerprint, captured, params=None):
    cap = deepcopy(captured)
    if params:
        cap['metrics']['market_calibration'] = {'version': 'equivalence-probe', 'stats': {cap['stat']: params}}
    return clean(archived_candidate(fingerprint)(cap['row'], cap['stat'], cap['projection'], cap['baseline'],
        cap['metrics'], cap['offer'], cap['distribution'], min_ev=cap['min_ev'],
        probability_calibration_by_key={tuple(r['key']): r['value'] for r in cap['calibration']},
        clv_guard_by_key={tuple(r['key']): r['value'] for r in cap['clv_guards']}, model_artifact={}))


def audit(records):
    sources, invalid = {}, {}
    by_version = defaultdict(list)
    for record in records:
        cap = (record.get('forecast_payload') or {}).get('scoring_replay') or {}
        version = cap.get('scoring_fingerprint')
        if version:
            by_version[version].append(record)
    for version in by_version:
        try:
            archived_candidate(version)
            sources[version] = program_signatures(STORE / version)
        except (ValueError, OSError) as exc:
            invalid[version] = str(exc)
    groups = defaultdict(list)
    for version, signature in sources.items():
        groups[(signature['scoring'], signature['inputs'])].append(version)
    proofs, aliases, excluded = [], {}, Counter()
    for key, versions in sorted(groups.items()):
        versions.sort()
        pool = [r for version in versions for r in by_version[version]]
        checked, failures, output_hashes = [], Counter(), []
        for record in pool:
            p = record['forecast_payload']; cap = p['scoring_replay']
            if cap.get('exact_overlay_present'):
                failures['uncaptured_exact_overlay'] += 1
                continue
            try:
                original = invoke(cap['scoring_fingerprint'], cap)
                if (original['side'] != p.get('side') or
                        abs(original['probability'] - float(p['probability'])) > 1e-10):
                    failures['original_probability_mismatch'] += 1
                    continue
                for params in PROBES:
                    reference = invoke(versions[0], cap, params)
                    if any(invoke(v, cap, params) != reference for v in versions[1:]):
                        raise ValueError('cross_version_replay_mismatch')
                    output_hashes.append(digest(reference))
                checked.append(int(record['id']))
            except (ValueError, KeyError, TypeError) as exc:
                failures[str(exc)] += 1
        # Any mismatch rejects the whole alias. Unchecked evidence cannot ride on
        # successful examples; original IDs remain the scope of every proof.
        accepted = bool(checked) and not failures
        cohort = 'equivalent:' + digest(key) if accepted and len(versions) > 1 else None
        if cohort:
            aliases.update({str(i): dict(cohort=cohort, versions=versions) for i in checked})
        excluded.update(failures)
        proofs.append(dict(versions=versions, scoring_program=key[0], input_program=key[1],
            rows=len(pool), replayed_rows=len(checked), probe_count=len(PROBES),
            replay_output_sha256=digest(output_hashes), failures=dict(failures),
            compatibility_verified=accepted, pooled=len(versions) > 1 and accepted,
            evaluation_cohort=cohort, season_weeks=sorted({(int(r['season']), int(r['week'])) for r in pool})))
    comparisons = []
    ordered = sorted(sources)
    for i, left in enumerate(ordered):
        for right in ordered[i+1:]:
            a, b = sources[left], sources[right]
            comparisons.append(dict(left=left, right=right,
                same_scoring_program=a['scoring'] == b['scoring'], same_input_program=a['inputs'] == b['inputs'],
                changed_functions=sorted(k for k in set(a['functions']) | set(b['functions'])
                                         if a['functions'].get(k) != b['functions'].get(k))))
    return dict(contract='nfl-scoring-equivalence-v1', built_at=datetime.now(timezone.utc).isoformat(),
        input_rows=len(records), versioned_rows=sum(map(len, by_version.values())),
        missing_scoring_version_rows=len(records)-sum(map(len, by_version.values())),
        dependency_sha256=dependency_digest(), sources=sources, groups=proofs, comparisons=comparisons,
        aliases=aliases, exclusions=dict(excluded), invalid_archives=invalid,
        forecasts_changed=False, production_changed=False,
        limitations=['Only identical scoring AND input programs may share evaluation evidence.',
            'Changed SQL/input semantics remain separate, even if conditional pricing code matches.',
            'Original fingerprints, model releases and selected forecast IDs are preserved.',
            'Historical imported dependencies were not archived; exact original replay is required.',
            'Replay equivalence is not predictive improvement or cash approval.'])


def save(report):
    atomic_json(REPORT, report)
    lines = ['# NFL Scoring Version Equivalence', '',
        '| Versions | Weeks | Rows replayed | Combined for evaluation |', '|---|---|---:|---|']
    for g in report['groups']:
        lines.append(f"| {', '.join(v[:12] for v in g['versions'])} | {g['season_weeks']} | {g['replayed_rows']} | {g['pooled']} |")
    lines += ['', *report['limitations']]
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def verified_aliases(report=None):
    if report is None:
        if not REPORT.exists():
            return {}
        report = json.loads(REPORT.read_text())
    if report.get('contract') != 'nfl-scoring-equivalence-v1' or report.get('dependency_sha256') != dependency_digest():
        return {}
    for version, signature in report.get('sources', {}).items():
        try:
            if source_hash(STORE / version) != version or program_signatures(STORE / version) != signature:
                return {}
        except (ValueError, OSError):
            return {}
    return report.get('aliases', {})


def cohort_for(version, forecast_id, aliases):
    proof = aliases.get(str(forecast_id)) or {}
    return proof['cohort'] if version in proof.get('versions', []) else version


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    from nfl_pipeline.modeling.fit_market_calibration import load_calibration_records
    report = audit(load_calibration_records('receiving_yards'))
    save(report)
    print(json.dumps(dict(groups=len(report['groups']), combined=sum(g['pooled'] for g in report['groups']), exclusions=report['exclusions'])))
