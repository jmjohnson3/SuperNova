"""Shared runtime configuration for SuperNovaBets scripts.

Keep operational secrets and machine-specific connection strings outside the
source tree. Local developer defaults can still live here for non-secret values.
"""
from __future__ import annotations

import os


DEFAULT_PG_DSN = "postgresql://josh:password@localhost:5432/nba"


def _saved_windows_env(*names: str) -> str:
    """Read saved User/Machine environment values from Windows registry.

    Newly written Windows env vars are not always visible to already-open
    terminals. This fallback lets scheduled scripts and direct CLI runs see
    saved secrets without requiring a reboot or new shell.
    """
    if os.name != "nt":
        return ""
    try:
        import winreg
    except Exception:
        return ""

    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(root, "Environment") as key:
                for name in names:
                    try:
                        value, _ = winreg.QueryValueEx(key, name)
                    except OSError:
                        continue
                    if value:
                        return str(value)
        except OSError:
            continue
    return ""


def pg_dsn() -> str:
    return (
        os.getenv("SUPERNOVABETS_PG_DSN")
        or os.getenv("PG_DSN")
        or _saved_windows_env("SUPERNOVABETS_PG_DSN", "PG_DSN")
        or DEFAULT_PG_DSN
    )


def mysportsfeeds_api_key() -> str:
    return (
        os.getenv("MYSPORTSFEEDS_API_KEY")
        or os.getenv("MSF_API_KEY")
        or _saved_windows_env("MYSPORTSFEEDS_API_KEY", "MSF_API_KEY")
        or ""
    )
