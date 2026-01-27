#!/usr/bin/env python3
"""
BGEMM example (new workflow)

This example runs the **new** PTO-AS flow through the **runtime**:
  Python kernel → PTO-AS text → `ptoas` → CCE → runtime graph → NPU run + timing.

For the actual implementation, see:
  `kernels/python/bgemm_performance/`
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    # Allow running from the examples/ directory.
    repo_root = _repo_root()
    sys.path.insert(0, os.fspath(repo_root))

    from kernels.python.bgemm_performance.run_runtime import main as bgemm_main

    return int(bgemm_main())


if __name__ == "__main__":
    raise SystemExit(main())
