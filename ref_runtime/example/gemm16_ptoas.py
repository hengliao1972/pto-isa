#!/usr/bin/env python3
"""
GEMM16 via PTO-AS + runtime (Python)

Flow:
  PTO-AS (.pto) → ptoas (CCE C++) → runtime compile_and_load_kernel → graph run

This is a minimal end-to-end sanity check for the MPMD ABI path:
  kernel(args: __gm__ int64_t*)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import tempfile

import numpy as np


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "include" / "pto").exists():
            return p
    raise RuntimeError("failed to locate repo root (missing include/pto)")


def main() -> int:
    # Locate repo root and put it on PYTHONPATH to import `pto.runtime`.
    repo_root = _find_repo_root(Path(__file__).resolve())
    sys.path.insert(0, os.fspath(repo_root))

    try:
        import pto_runtime
    except ImportError as exc:
        print(f"Error: Cannot import pto_runtime: {exc}")
        print("Expected repo root to contain `pto_runtime.py`.")
        return 1

    from pto.runtime import PtoasConfig, compile_and_load_kernel_from_pto

    ascend_home = (
        Path(os.environ.get("ASCEND_HOME_PATH", ""))
        if os.environ.get("ASCEND_HOME_PATH")
        else Path.home() / "Ascend" / "ascend-toolkit" / "latest"
    )
    os.environ["ASCEND_HOME_PATH"] = os.fspath(ascend_home.resolve())

    device_id = int(os.environ.get("PTO_DEVICE", "0"))

    runner = pto_runtime.DeviceRunner.get()
    rc = int(runner.init(device_id, 1))
    if rc != 0:
        print(f"Error: DeviceRunner.init failed: {rc}")
        return rc

    # Compile a small PTO-AS program to CCE and load it as func_id=0.
    pto_path = repo_root / "ptoas" / "examples" / "gemm16_e2e.pto"
    if not pto_path.exists():
        print(f"Error: missing PTO-AS example: {pto_path}")
        runner.finalize()
        return 1

    out_dir = Path(
        os.environ.get("PTO_OUT_DIR", "").strip() or tempfile.mkdtemp(prefix="pto_runtime_gemm16_")
    ).resolve()
    print(f"out_dir={out_dir}", flush=True)

    timeout_s = None
    if os.environ.get("PTO_PTOAS_TIMEOUT_S"):
        timeout_s = float(os.environ["PTO_PTOAS_TIMEOUT_S"])

    cfg = PtoasConfig(
        ptoas=repo_root / "bin" / "ptoas",
        arch="dav-c220-cube",
        memory_model="MEMORY_BASE",
        kernel_abi="mpmd",
        insert_events=True,
        assign_tile_addrs=True,
        ascend_home=ascend_home,
        repo_root=repo_root,
        timeout_s=timeout_s,
        log_path=out_dir / "ptoas.log",
        print_cmd=True,
    )

    print(f"Compiling via ptoas: {pto_path}", flush=True)
    compile_and_load_kernel_from_pto(
        runner=runner,
        func_id=0,
        pto=pto_path,
        out_dir=out_dir,
        pto_isa_root=repo_root,
        ptoas_cfg=cfg,
    )
    print("Kernel loaded (func_id=0)", flush=True)

    # Allocate and initialize inputs.
    print("Allocating device buffers...", flush=True)
    a = (np.random.default_rng(0).standard_normal((16, 16)).astype(np.float16))
    b = (np.random.default_rng(1).standard_normal((16, 16)).astype(np.float16))
    c = np.zeros((16, 16), dtype=np.float32)

    dev_a = int(runner.allocate_tensor(int(a.nbytes)))
    dev_b = int(runner.allocate_tensor(int(b.nbytes)))
    dev_c = int(runner.allocate_tensor(int(c.nbytes)))
    if not (dev_a and dev_b and dev_c):
        print("Error: allocate_tensor failed")
        runner.finalize()
        return 1
    print(f"dev_a=0x{dev_a:x} dev_b=0x{dev_b:x} dev_c=0x{dev_c:x}", flush=True)

    print("Copying inputs to device...", flush=True)
    rc = int(runner.copy_to_device(dev_a, a))
    if rc != 0:
        print(f"Error: copy_to_device(A) failed: {rc}")
        return rc
    rc = int(runner.copy_to_device(dev_b, b))
    if rc != 0:
        print(f"Error: copy_to_device(B) failed: {rc}")
        return rc
    print("H2D done", flush=True)

    # Build and run a 1-task graph: C = A @ B
    graph = pto_runtime.Graph()
    graph.add_task([dev_a, dev_b, dev_c], func_id=0)
    print("Running graph...", flush=True)
    rc = int(runner.run(graph, 1))
    if rc != 0:
        print(f"Error: graph run failed: {rc}")
        return rc
    print("Graph done", flush=True)

    print("Copying outputs from device...", flush=True)
    rc = int(runner.copy_from_device(c, dev_c))
    if rc != 0:
        print(f"Error: copy_from_device(C) failed: {rc}")
        return rc
    print("D2H done", flush=True)

    expected = (a.astype(np.float32) @ b.astype(np.float32))
    max_abs = float(np.max(np.abs(c - expected)))
    print(f"max_abs_error={max_abs}")
    if not np.allclose(c, expected, rtol=1e-2, atol=1e-2):
        print("FAILED: GEMM16 mismatch")
        return 1

    print("OK: GEMM16 via ptoas + runtime")
    runner.free_tensor(dev_a)
    runner.free_tensor(dev_b)
    runner.free_tensor(dev_c)
    runner.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
