"""Lightweight startup timing helper.

Set CONVEX_HULL_STARTUP_PROFILE=1 before launching to enable logging.
Each mark writes one line to startup_profile.log in the working directory.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

_T0 = time.perf_counter()
_ENABLED = os.environ.get("CONVEX_HULL_STARTUP_PROFILE") == "1"


def mark(name: str) -> None:
    """Record a named checkpoint if profiling is enabled."""
    if not _ENABLED:
        return

    elapsed = time.perf_counter() - _T0
    path = Path.cwd() / "startup_profile.log"

    with path.open("a", encoding="utf-8") as f:
        f.write(f"{elapsed:8.3f}s {name}\n")
