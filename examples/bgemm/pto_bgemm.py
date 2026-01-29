#!/usr/bin/env python3
"""
BGEMM example (new workflow)

This example runs the **new** PTO-AS flow through the **runtime**:
  Python kernel → PTO-AS text → `ptoas` → CCE → runtime graph → NPU run + timing.
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

    from examples.bgemm.run_ascend_a2a3 import main as bgemm_main

    return int(bgemm_main())


if __name__ == "__main__":
    raise SystemExit(main())
