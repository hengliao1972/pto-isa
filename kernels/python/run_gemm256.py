#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptoas.python import binding, pipeline  # noqa: E402
from ptoas.python.host_spec import prepend_host_spec_to_pto  # noqa: E402


def _default_ptoas(repo: Path) -> Path:
    return repo / "bin/ptoas"


def _soc_from_alias(alias: str) -> str:
    if alias == "a3":
        return "Ascend910B1"
    if alias == "a5":
        return "Ascend910_9599"
    return alias


def main() -> int:
    repo = pipeline.repo_root()
    ap = argparse.ArgumentParser(description="Run GEMM 256x256x256 (tiled 16x16) end-to-end.")
    ap.add_argument("--run-mode", choices=["npu", "sim"], default="npu")
    ap.add_argument("--soc", default="a3", help="Simulator SoC alias when --run-mode=sim (a3|a5|other)")
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas(repo))
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/pto_gemm256"))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--block-dim", type=int, default=1)
    ap.add_argument("--memory-model", default="MEMORY_BASE")
    ap.add_argument("--no-insert-events", dest="insert_events", action="store_false", default=True)
    ap.add_argument("--verbose-build", action="store_true", help="Print compiler commands/warnings")
    args = ap.parse_args()

    # Build logs are useful for simulator debugging; default to verbose in sim mode.
    if not args.verbose_build and args.run_mode != "sim":
        os.environ.setdefault("PTOAS_QUIET", "1")

    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2
    if not args.ascend_home or not args.ascend_home.exists():
        print("error: set --ascend-home or ASCEND_HOME_PATH to your Ascend toolkit root", file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.run_mode == "sim":
        # Ensure simulator emits logs to a predictable location.
        camodel = args.outdir / "camodel_logs"
        camodel.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("CAMODEL_LOG_PATH", str(camodel))
        print(f"CAMODEL_LOG_PATH={os.environ['CAMODEL_LOG_PATH']}", flush=True)

        soc = _soc_from_alias(args.soc)
        print(f"Stage SIM: ensure env (soc={soc})...", flush=True)
        pipeline.ensure_ascend_sim_env(ascend_home=args.ascend_home, soc=soc)

    py = Path(__file__).resolve().with_name("gemm256.py")
    spec = binding.compile_file(py, kernel="gemm256")
    pto_text = prepend_host_spec_to_pto(pto=spec.pto, spec=binding.default_host_spec(spec))

    pto_path = args.outdir / f"{spec.name}.pto"
    pto_path.write_text(pto_text, encoding="utf-8")

    host_spec = pipeline.parse_or_default_host_spec(pto_text=pto_text)
    host_spec = type(host_spec)(
        args=host_spec.args, seed=host_spec.seed, block_dim=args.block_dim, kernel_name=host_spec.kernel_name
    )
    base_arrays = pipeline.make_host_arrays(host_spec)

    t0 = time.perf_counter()

    # CPU reference.
    print("Stage CPU: compile + run reference...", flush=True)
    cpu_cpp = pipeline.compile_pto_to_cpu_cpp(pto_path=pto_path, outdir=args.outdir, ptoas=args.ptoas)
    cpu_so = args.outdir / f"lib{spec.name}_cpu.so"
    pipeline.build_cpu_so_from_cpp(cpp_path=cpu_cpp, out_so=cpu_so)
    cpu_arrays = [a.copy() for a in base_arrays]
    cpu_out = pipeline.run_cpu_kernel_from_so(so_path=cpu_so, host_spec=host_spec, host_arrays=cpu_arrays)
    print(f"Stage CPU: OK ({time.perf_counter() - t0:.1f}s total)", flush=True)

    # NPU run (sim or real).
    if args.run_mode == "sim":
        soc = _soc_from_alias(args.soc)
        runtime_lib = "runtime_camodel"
    else:
        soc = None
        runtime_lib = "runtime"

    print(f"Stage NPU({args.run_mode}): compile + build .so...", flush=True)
    cfg = pipeline.CompileConfig(
        ptoas=args.ptoas,
        ascend_home=args.ascend_home,
        arch="dav-c220-cube",
        memory_model=args.memory_model,
        insert_events=args.insert_events,
    )
    cce_cpp, _bin = pipeline.compile_pto_to_cce_and_bin(pto_path=pto_path, outdir=args.outdir, cfg=cfg)
    try:
        summary = pipeline.summarize_cce_events(cce_path=cce_cpp)
        (args.outdir / "event_summary.txt").write_text(str(summary) + "\n", encoding="utf-8")
        print(f"Event summary: set={summary['set_total']} wait={summary['wait_total']}", flush=True)
    except Exception:
        pass
    npu_so = args.outdir / f"lib{spec.name}_{args.run_mode}.so"
    pipeline.build_fatobj_so_from_cce(
        cce_path=cce_cpp,
        out_so=npu_so,
        arch=cfg.arch,
        ascend_home=cfg.ascend_home,
        runtime_lib=runtime_lib,
        soc=(soc if args.run_mode == "sim" else None),
    )

    print(f"Stage NPU({args.run_mode}): run kernel (device={args.device}, block_dim={args.block_dim})...", flush=True)
    npu_arrays = [a.copy() for a in base_arrays]
    npu_res = pipeline.run_npu_kernel_from_so(
        so_path=npu_so, host_spec=host_spec, host_arrays=npu_arrays, device_id=args.device, block_dim=args.block_dim
    )
    npu_out = npu_res.outputs

    out_dtypes = [host_spec.args[i].dtype for i in host_spec.output_indices()]
    pipeline.compare_cpu_and_npu_outputs(cpu_out=cpu_out, npu_out=npu_out, out_dtypes=out_dtypes)
    for a in npu_out:
        if a.dtype in (np.float16, np.float32):
            print("OK (max abs):", float(np.max(np.abs(a))), flush=True)
            break
    print(f"ALL OK ({time.perf_counter() - t0:.1f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
