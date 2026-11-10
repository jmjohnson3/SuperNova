"""Immutable, locally verifiable registrations for prospective NFL experiments.

A registration pins executable source, rules and artifacts. It never grants cash
authority and never silently rolls over when development changes a dependency.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "src/nfl_pipeline/modeling/models/frozen_experiments"
CONTRACT = "nfl-frozen-paper-experiment-v1"
RUNTIME_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "xgboost", "lightgbm")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _name(name):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,100}", str(name)):
        raise ValueError("Invalid experiment name")
    return str(name)


def _inside(path, root):
    resolved = Path(path).resolve()
    resolved.relative_to(root.resolve())
    return resolved


def runtime_versions():
    result = {"python": platform.python_version(), "python_implementation": platform.python_implementation()}
    for package in RUNTIME_PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def source_dependencies(*modules, root=ROOT):
    """Discover local static imports, including imports within scorer functions.

    This does not import/execute any application module. Dynamic artifact paths,
    SQL definitions and other runtime inputs must be supplied explicitly too.
    """
    root = Path(root)
    pending = list(modules)
    found = {}
    while pending:
        module = pending.pop()
        if module in found or not (module == "supernovabets_config" or module.startswith("nfl_pipeline")):
            continue
        source = root / "src" / Path(*module.split("."))
        path = source.with_suffix(".py")
        if not path.exists():
            path = source / "__init__.py"
        if not path.is_file():
            continue
        found[module] = path
        # Importing a module also executes each parent package initializer.
        parts = module.split(".")
        pending.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
                    parts = package.split(".")
                    base = ".".join(parts[:len(parts) - node.level + 1])
                    imported = ".".join(filter(None, (base, node.module)))
                else:
                    imported = node.module or ""
                pending.append(imported)
                pending.extend(imported + "." + alias.name for alias in node.names)
    return dict(sorted(found.items()))


def load_experiment(name, *, root=ROOT, store=STORE):
    path = Path(store) / _name(name) / "registration.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in doc.items() if key != "sha256"}
    if (doc.get("sha256") != _digest(body) or doc.get("contract") != CONTRACT
            or doc.get("experiment_id") != name or doc.get("betting_approved") is not False):
        raise ValueError("Frozen experiment registration checksum/contract mismatch")
    for entry in doc["files"].values():
        _inside(Path(root) / entry["path"], Path(root))
    return doc


def check_experiment(name, *, root=ROOT, store=STORE):
    doc = load_experiment(name, root=root, store=store)
    if doc is None:
        raise ValueError("Experiment is not registered: " + name)
    changed = []
    for label, entry in doc["files"].items():
        path = _inside(Path(root) / entry["path"], Path(root))
        if entry.get("absent"):
            if path.exists():
                changed.append(label)
        elif not path.is_file() or _file_hash(path) != entry["sha256"]:
            changed.append(label)
    if doc["runtime_versions"] != runtime_versions():
        changed.append("runtime_versions")
    if changed:
        raise ValueError("Frozen experiment changed; restore the pinned version or explicitly register a new experiment: "
                         + ", ".join(sorted(changed)))
    return doc


def register_experiment(name, config, files, *, root=ROOT, store=STORE, now=None, absent_paths=None):
    """Create once; identical reruns are safe, material changes require a new ID."""
    name = _name(name)
    root, store = Path(root).resolve(), Path(store)
    if not files:
        raise ValueError("A frozen experiment must pin executable files and artifacts")
    config = json.loads(json.dumps(config, sort_keys=True, allow_nan=False))
    entries = {}
    for label, file in sorted(files.items()):
        path = _inside(file, root)
        if not path.is_file():
            raise ValueError("Missing experiment input: " + str(label))
        entries[str(label)] = {"path": path.relative_to(root).as_posix(), "sha256": _file_hash(path)}
    for label, file in sorted((absent_paths or {}).items()):
        path = _inside(file, root)
        if str(label) in entries or path.exists():
            raise ValueError("Expected an absent experiment input: " + str(label))
        entries[str(label)] = {"path": path.relative_to(root).as_posix(), "sha256": None, "absent": True}
    existing = load_experiment(name, root=root, store=store)
    if existing is not None:
        if existing["config"] != config or existing["files"] != entries:
            raise ValueError("Experiment already registered; rules and files cannot be replaced")
        return check_experiment(name, root=root, store=store)
    at = now or datetime.now(timezone.utc)
    if at.tzinfo is None:
        raise ValueError("Registration requires a timezone-aware timestamp")
    body = dict(contract=CONTRACT, experiment_id=name, registered_at=at.astimezone(timezone.utc).isoformat(),
                config=config, files=entries, runtime_versions=runtime_versions(),
                research_only=True, betting_approved=False, automatic_rollover=False)
    doc = dict(body, sha256=_digest(body))
    store.mkdir(parents=True, exist_ok=True)
    # Directory rename publishes source snapshots and registration together. There
    # is no mutable active pointer and no operation that replaces an old trial.
    with tempfile.TemporaryDirectory(dir=store) as staging:
        folder = Path(staging) / "experiment"
        folder.mkdir()
        for entry in entries.values():
            if entry.get("absent"):
                continue
            path = root / entry["path"]
            if path.suffix in (".py", ".json", ".sql", ".toml") and path.stat().st_size <= 2_000_000:
                target = folder / "sources" / entry["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                if _file_hash(target) != entry["sha256"]:
                    raise ValueError("Experiment input changed while registering")
        (folder / "registration.json").write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Recheck binaries as well before committing the immutable registration.
        if any(((root / e["path"]).exists() if e.get("absent") else
                _file_hash(root / e["path"]) != e["sha256"]) for e in entries.values()):
            raise ValueError("Experiment input changed while registering")
        try:
            folder.rename(store / name)
        except FileExistsError:
            return register_experiment(name, config, files, root=root, store=store, now=at, absent_paths=absent_paths)
    return check_experiment(name, root=root, store=store)


def status(*, root=ROOT, store=STORE):
    rows = []
    for path in sorted(Path(store).glob("*/registration.json")):
        try:
            doc = check_experiment(path.parent.name, root=root, store=store)
            rows.append(dict(experiment_id=path.parent.name, status="frozen", sha256=doc["sha256"],
                             registered_at=doc["registered_at"], betting_approved=False))
        except (ValueError, OSError) as exc:
            rows.append(dict(experiment_id=path.parent.name, status="blocked", reason=str(exc), betting_approved=False))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", help="Verify one experiment; exit nonzero on drift")
    args = parser.parse_args()
    if args.check:
        doc = check_experiment(args.check)
        print(json.dumps(dict(experiment_id=doc["experiment_id"], status="frozen", sha256=doc["sha256"])))
    else:
        print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    main()
