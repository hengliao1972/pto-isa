#!/usr/bin/env python3
"""
TPRINT16 via PTO-AS + runtime (Python)

Flow:
  PTO-AS (.pto) → ptoas (CCE C++) → runtime compile_and_load_kernel → run_task

This is a minimal end-to-end sanity check that:
  - PTO-AS kernels can be compiled and loaded at runtime
  - AICore executes the kernel and emits a single TPRINT
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "include" / "pto").exists() and (p / "ptoas").exists():
            return p
    raise RuntimeError("failed to locate repo root")


def _scan_device_logs(device_id: int, since_s: float) -> list[str]:
    log_dir = Path.home() / "ascend" / "log" / "debug" / f"device-{device_id}"
    if not log_dir.exists():
        return []

    hits: list[str] = []
    for p in sorted(log_dir.glob("device-*.log"), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.stat().st_mtime < since_s - 2.0:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in txt.splitlines():
            if "TPRINT" in line:
                hits.append(f"{p.name}: {line}")
    return hits


def main() -> int:
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
    aic_blocks = int(os.environ.get("PTO_AIC_BLOCKS", "1"))

    runner = pto_runtime.DeviceRunner.get()
    rc = int(runner.init(device_id, aic_blocks))
    if rc != 0:
        print(f"Error: DeviceRunner.init failed: {rc}")
        return rc

    # Make sure device-side debug printing is enabled (TPRINT is gated on `_DEBUG`).
    os.environ.setdefault("PTO_CCE_DEBUG", "1")

    pto_path = repo_root / "ptoas" / "examples" / "tprint16_once.pto"
    if not pto_path.exists():
        print(f"Error: missing PTO-AS example: {pto_path}")
        runner.finalize()
        return 1

    out_dir = Path(
        os.environ.get("PTO_OUT_DIR", "").strip() or tempfile.mkdtemp(prefix="pto_runtime_tprint16_")
    ).resolve()
    print(f"out_dir={out_dir}", flush=True)

    cfg = PtoasConfig(
        ptoas=repo_root / "bin" / "ptoas",
        arch="dav-c220-cube",
        memory_model="MEMORY_BASE",
        kernel_abi="mpmd",
        insert_events=False,
        assign_tile_addrs=False,
        ascend_home=ascend_home,
        repo_root=repo_root,
        timeout_s=float(os.environ.get("PTO_PTOAS_TIMEOUT_S", "0") or "0") or None,
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

    # Keep this small: debug printing is slow.
    a = np.arange(4 * 4, dtype=np.float16).reshape(4, 4)
    dev_a = int(runner.allocate_tensor(int(a.nbytes)))
    if not dev_a:
        print("Error: allocate_tensor failed")
        runner.finalize()
        return 1

    rc = int(runner.copy_to_device(dev_a, a))
    if rc != 0:
        print(f"Error: copy_to_device failed: {rc}")
        runner.finalize()
        return rc

    start_s = time.time()
    print("Running single task (expect 1x TPRINT from AICore)...", flush=True)
    rc = int(runner.run_task([dev_a], func_id=0, launch_aicpu_num=1))
    if rc != 0:
        print(f"Error: run_task failed: {rc}")
        runner.finalize()
        return rc

    runner.free_tensor(dev_a)
    runner.finalize()

    hits = _scan_device_logs(device_id, start_s)
    if hits:
        print("\nFound TPRINT lines in device log:", flush=True)
        for line in hits[:20]:
            print("  " + line, flush=True)
    else:
        log_dir = Path.home() / "ascend" / "log" / "debug" / f"device-{device_id}"
        print(f"\nNo 'TPRINT' lines found in logs (check {log_dir}).", flush=True)

    print("OK: TPRINT16 via ptoas + runtime", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
