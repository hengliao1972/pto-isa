#!/usr/bin/env python3
"""
BGEMM example (new workflow, "adaptive" runner)

This is a thin wrapper around `kernels/python/bgemm_performance/run_runtime.py`.
Use `--allow-unaligned` to accept (m,n,k) that are not exact tile multiples.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    repo_root = _repo_root()
    sys.path.insert(0, os.fspath(repo_root))

    from kernels.python.bgemm_performance.run_runtime import main as bgemm_main

    return int(bgemm_main())


if __name__ == "__main__":
    raise SystemExit(main())
