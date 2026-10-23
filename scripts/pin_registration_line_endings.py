"""Regenerate the .gitattributes block that keeps active registrations' pinned files byte-identical.

Frozen registrations hash raw bytes, and core.autocrlf can rewrite line endings on checkout, which
blocks a registration with no edit at all. Run after registering an experiment:

    .venv\\Scripts\\python.exe scripts\\pin_registration_line_endings.py nfl-sharp-gap-cash-v7 fanduel-receiving-oe-paper-v5

then prove it: back up the tracked pinned files, delete them, `git checkout --` them, and
`python -m nfl_pipeline.experiment_registry --check <name>` for each registration.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "src/nfl_pipeline/modeling/models/frozen_experiments"
BEGIN, END = "# --- BEGIN registered line endings (generated) ---", "# --- END registered line endings ---"


def forms(active: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in active:
        doc = json.loads((STORE / name / "registration.json").read_text(encoding="utf-8"))
        for entry in doc["files"].values():
            if entry.get("absent"):
                continue
            path = entry["path"]
            if subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=ROOT, capture_output=True).returncode:
                continue  # untracked (e.g. gitignored binaries): git never rewrites them
            raw = (ROOT / path).read_bytes()
            lf = raw.replace(b"\r\n", b"\n")
            digest = lambda b: hashlib.sha256(b).hexdigest()
            form = "lf" if digest(lf) == entry["sha256"] else "crlf" if digest(lf.replace(b"\n", b"\r\n")) == entry["sha256"] else None
            if form is None:
                raise SystemExit(f"{path}: on-disk bytes no longer match {name}; restore it before pinning")
            if result.get(path, form) != form:
                raise SystemExit(f"{path}: registrations disagree on line endings")
            result[path] = form
    return result


def main() -> None:
    active = sys.argv[1:]
    if not active:
        raise SystemExit(__doc__)
    pinned = forms(active)
    path = ROOT / ".gitattributes"
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    head = text.split(BEGIN)[0].rstrip("\n") + "\n"
    block = [BEGIN, "# Active frozen registrations hash raw bytes. With core.autocrlf a checkout can rewrite a file's",
             "# line endings and block the registration, so each pinned file keeps exactly the endings it was",
             f"# registered with. Registrations: {', '.join(active)}.",
             *(f"{p} text eol={f}" for p, f in sorted(pinned.items())), END]
    path.write_text(head + "\n" + "\n".join(block) + "\n", encoding="utf-8")
    print(f"{len(pinned)} files pinned for {', '.join(active)}")


if __name__ == "__main__":
    main()
