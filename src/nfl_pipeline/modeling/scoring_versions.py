"""Verified local scoring source snapshots for immutable forecast replay."""
import hashlib
import importlib.util
from pathlib import Path
import re
import shutil
import sys
import tempfile

from nfl_pipeline.integrity import MODEL_ROOT

STORE = MODEL_ROOT/'scoring_code_versions'
SOURCE = Path(__file__).resolve().parent
FILES = ('predict_player_props.py', 'scoring_capture.py')
_loaded = {}


def source_hash(root):
    return hashlib.sha256(b''.join((root/name).read_bytes() for name in FILES)).hexdigest()


def archive_current():
    fingerprint = source_hash(SOURCE)
    STORE.mkdir(parents=True, exist_ok=True)
    target = STORE/fingerprint
    if target.exists():
        if source_hash(target) != fingerprint:
            raise ValueError('Archived scoring source checksum mismatch')
        return fingerprint
    with tempfile.TemporaryDirectory(dir=STORE) as temp:
        staging = Path(temp)/'source'
        staging.mkdir()
        for name in FILES:
            shutil.copyfile(SOURCE/name, staging/name)
        if source_hash(staging) != fingerprint:
            raise ValueError('Scoring source changed during archive')
        staging.rename(target)
    return fingerprint


def archived_candidate(fingerprint):
    if not re.fullmatch(r'[0-9a-f]{64}', str(fingerprint)):
        raise ValueError('scoring_code_version_unavailable')
    root = STORE/fingerprint
    try:
        if source_hash(root) != fingerprint:
            raise ValueError('archived_scoring_checksum_mismatch')
    except FileNotFoundError as exc:
        raise ValueError('scoring_code_version_unavailable') from exc
    if fingerprint not in _loaded:
        name = 'nfl_pipeline.modeling._archived_scoring_'+fingerprint
        spec = importlib.util.spec_from_file_location(name, root/FILES[0])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
        _loaded[fingerprint] = module
    return _loaded[fingerprint]._candidate_from_offer


if __name__ == '__main__':
    print(archive_current())
