#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate BGEMM intermediate artifacts into examples/ for demo.")
    ap.add_argument("--outdir", type=Path, default=_REPO_ROOT / "examples" / "bgemm" / "generated")
    ap.add_argument("--ptoas", type=Path, default=_REPO_ROOT / "bin" / "ptoas")
    ap.add_argument("--memory-model", type=str, default="MEMORY_BASE")

    # Keep defaults small so generated C++ stays readable.
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--grid-m", type=int, default=1)
    ap.add_argument("--grid-n", type=int, default=1)
    ap.add_argument("--base-m", type=int, default=128)
    ap.add_argument("--base-n", type=int, default=256)
    ap.add_argument("--base-k", type=int, default=64)
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    from examples.bgemm.bgemm_kernel import make_bgemm_kernel

    spec = make_bgemm_kernel(
        batch=int(args.batch),
        m=int(args.m),
        n=int(args.n),
        k=int(args.k),
        grid_m=int(args.grid_m),
        grid_n=int(args.grid_n),
        base_m=int(args.base_m),
        base_n=int(args.base_n),
        base_k=int(args.base_k),
    )

    pto_path = outdir / "kernel_0.pto"
    cpp_path = outdir / "kernel_0.cpp"
    log_path = outdir / "ptoas.log"
    orch_path = outdir / "orchestration.py"

    pto_path.write_text(spec.pto, encoding="utf-8")

    orch_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "# Demo orchestration function: builds a runtime task graph.",
                "#",
                "# Execution is performed by launching the device-side scheduler on AICPU, which",
                "# dispatches tasks to AICore workers (see pto_runtime.DeviceRunner.run).",
                "",
                "import pto_runtime",
                "",
                "",
                "def build_graph(*, runner: pto_runtime.DeviceRunner, func_id: int, dev_a: int, dev_b: int, dev_c: int, block_dim: int) -> pto_runtime.OrchestrationRuntime:",
                "    orch = pto_runtime.OrchestrationRuntime(runner=runner)",
                "    for tile_id in range(int(block_dim)):",
                "        orch.add_task([int(tile_id), int(dev_a), int(dev_b), int(dev_c)], func_id=int(func_id), core_type=1)",
                "    return orch",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    from pto.runtime import PtoasConfig, compile_pto_to_cce_cpp

    cfg = PtoasConfig(
        ptoas=Path(args.ptoas),
        memory_model=str(args.memory_model),
        repo_root=_REPO_ROOT,
        log_path=log_path,
        print_cmd=True,
    )
    compile_pto_to_cce_cpp(pto_path=pto_path, out_cpp=cpp_path, cfg=cfg)

    print(f"wrote: {pto_path}")
    print(f"wrote: {cpp_path}")
    print(f"wrote: {log_path}")
    print(f"wrote: {orch_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
