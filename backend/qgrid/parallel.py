"""Worker counts for the benchmark scripts.

Time-limited MILPs (the engine's local models have a 0.5 s branch-and-bound budget) find worse incumbents when the
machine is oversubscribed, which makes results depend on load. Benchmarks therefore default to one worker per physical
core and clamp larger requests, unless QGRID_ALLOW_OVERSUBSCRIBE=1 is set.
"""
from __future__ import annotations

import os


def physical_cores() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except ImportError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)          # no psutil: assume two hardware threads per core


def workers(requested: int | None = None, default: int | None = None) -> int:
    """requested (e.g. --workers=N) or default, capped at the physical core count."""
    cap = physical_cores()
    w = requested if requested else (default if default else cap)
    if w > cap and os.environ.get("QGRID_ALLOW_OVERSUBSCRIBE") != "1":
        print(f"[workers] {w} requested, capped at {cap} physical cores (set QGRID_ALLOW_OVERSUBSCRIBE=1 to override)",
              flush=True)
        w = cap
    return max(1, int(w))
