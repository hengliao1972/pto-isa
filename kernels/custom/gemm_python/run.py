#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptoas.python import pipeline  # noqa: E402
from gemm_pto import make_gemm16_pto  # noqa: E402
from ptoas.python.host_spec import prepend_host_spec_to_pto  # noqa: E402


def _repo_root() -> Path:
    return _REPO_ROOT


def _default_ptoas() -> Path:
    return _repo_root() / "bin/ptoas"


def main() -> int:
    ap = argparse.ArgumentParser(description="kernels/custom/gemm_python: Python -> PTO-AS -> ptoas -> run (CPU/NPU).")
    ap.add_argument("--target", choices=["cpu", "npu", "both"], default="cpu")
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas())
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/pto_kernel_gemm_python"))

    # NPU options
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    ap.add_argument("--run-mode", choices=["npu", "sim"], default="npu")
    ap.add_argument("--soc", default="a3", help="Simulator SoC (a3|a5|Ascend910B1|...) when --run-mode=sim")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--block-dim", type=int, default=1)
    ap.add_argument("--memory-model", default="MEMORY_BASE")
    args = ap.parse_args()

    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)

    pto_text = make_gemm16_pto(target="npu")
    pto_text = prepend_host_spec_to_pto(pto=pto_text, spec=pipeline.parse_or_default_host_spec(pto_text=pto_text))
    pto_path = args.outdir / "gemm16.pto"
    pto_path.write_text(pto_text, encoding="utf-8")

    cpu_cpp = pipeline.compile_pto_to_cpu_cpp(pto_path=pto_path, outdir=args.outdir, ptoas=args.ptoas)
    cpu_so = args.outdir / "libgemm16_cpu.so"
    pipeline.build_cpu_so_from_cpp(cpp_path=cpu_cpp, out_so=cpu_so)

    host_spec = pipeline.parse_or_default_host_spec(pto_text=pto_text)
    host_spec = type(host_spec)(
        args=host_spec.args, seed=host_spec.seed, block_dim=args.block_dim, kernel_name=host_spec.kernel_name
    )
    base = pipeline.make_host_arrays(host_spec)
    cpu_arrays = [a.copy() for a in base]
    cpu_out = pipeline.run_cpu_kernel_from_so(so_path=cpu_so, host_spec=host_spec, host_arrays=cpu_arrays)

    if args.target in ("npu", "both"):
        if not args.ascend_home or not args.ascend_home.exists():
            print("error: set --ascend-home or ASCEND_HOME_PATH to your Ascend toolkit root", file=sys.stderr)
            return 2
        if args.run_mode == "sim":
            soc_full = "Ascend910B1" if args.soc == "a3" else ("Ascend910_9599" if args.soc == "a5" else args.soc)
            pipeline.ensure_ascend_sim_env(ascend_home=args.ascend_home, soc=soc_full)
            runtime_lib = "runtime_camodel"
        else:
            runtime_lib = "runtime"
            soc_full = None

        cfg = pipeline.CompileConfig(
            ptoas=args.ptoas,
            ascend_home=args.ascend_home,
            arch="dav-c220-cube",
            memory_model=args.memory_model,
            insert_events=True,
        )
        cce_path, bin_path = pipeline.compile_pto_to_cce_and_bin(pto_path=pto_path, outdir=args.outdir, cfg=cfg)
        npu_so = args.outdir / "libgemm16_npu.so"
        pipeline.build_fatobj_so_from_cce(
            cce_path=cce_path,
            out_so=npu_so,
            arch=cfg.arch,
            ascend_home=cfg.ascend_home,
            runtime_lib=runtime_lib,
            soc=soc_full,
        )

        npu_arrays = [a.copy() for a in base]
        npu_res = pipeline.run_npu_kernel_from_so(
            so_path=npu_so, host_spec=host_spec, host_arrays=npu_arrays, device_id=args.device, block_dim=args.block_dim
        )
        npu_out = npu_res.outputs
        out_dtypes = [host_spec.args[i].dtype for i in host_spec.output_indices()]
        pipeline.compare_cpu_and_npu_outputs(cpu_out=cpu_out, npu_out=npu_out, out_dtypes=out_dtypes)
        print(f"OK: gemm16 matched CPU reference (bin: {bin_path.name})")

    print(f"OK: kernels/custom/gemm_python (target={args.target}) outdir={args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
