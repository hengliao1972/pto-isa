#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptoas.python import binding, pipeline  # noqa: E402
from ptoas.python.host_spec import prepend_host_spec_to_pto  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    py: Path
    kernel: str
    arch: str
    input_scale: float | None = None
    block_dim: int | None = None
    split_kernels: bool = False


def _default_ptoas(repo: Path) -> Path:
    return repo / "bin/ptoas"


def _soc_from_alias(alias: str) -> str:
    if alias == "a3":
        return "Ascend910B1"
    if alias == "a5":
        return "Ascend910_9599"
    return alias


@dataclass
class CaseReport:
    name: str
    arch: str
    block_dim: int
    status: str  # OK|FAIL|TIMEOUT
    wall_s: float
    outdir: str
    # NPU-only (may be None even in NPU mode if timing not collected).
    device_id: int | None = None
    soc: str | None = None
    device_count: int | None = None
    avg_us: float | None = None
    p50_us: float | None = None
    min_us: float | None = None
    max_us: float | None = None
    iters: int | None = None
    # Short error message for table display.
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True) + "\n"

    @staticmethod
    def from_json(text: str) -> "CaseReport":
        d = json.loads(text)
        return CaseReport(**d)


def _write_report(path: Path, r: CaseReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(r.to_json(), encoding="utf-8")


def _fmt_float(x: float | None, *, fmt: str = "{:.2f}") -> str:
    if x is None:
        return "-"
    try:
        return fmt.format(float(x))
    except Exception:
        return "-"


def _format_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [0] * len(rows[0])
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def _is_numberish(s: str) -> bool:
        if s in ("-", "inf", "+inf", "-inf"):
            return True
        try:
            float(s)
            return True
        except Exception:
            return False

    def _join(r: list[str]) -> str:
        out: list[str] = []
        for i, cell in enumerate(r):
            if _is_numberish(cell):
                out.append(cell.rjust(widths[i]))
            else:
                out.append(cell.ljust(widths[i]))
        return "  ".join(out).rstrip()

    header = _join(rows[0])
    sep = "  ".join("-" * w for w in widths).rstrip()
    body = "\n".join(_join(r) for r in rows[1:])
    return header + "\n" + sep + ("\n" + body if body else "")


def _auto_npu_devices(*, fallback_device: int) -> list[int]:
    try:
        import acl

        try:
            acl.init()
        except Exception:
            pass
        try:
            c, r = acl.rt.get_device_count()
            if int(r) == 0 and int(c) > 0:
                return list(range(int(c)))
        finally:
            try:
                acl.finalize()
            except Exception:
                pass
    except Exception:
        pass
    return [int(fallback_device)]


def _parse_devices(arg: str | None, *, fallback_device: int) -> list[int]:
    if not arg or arg in ("auto", "all"):
        return _auto_npu_devices(fallback_device=fallback_device)

    out: list[int] = []
    for part in str(arg).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = a.strip()
            b = b.strip()
            if a.isdigit() and b.isdigit():
                lo = int(a)
                hi = int(b)
                if hi < lo:
                    lo, hi = hi, lo
                out.extend(list(range(lo, hi + 1)))
                continue
        out.append(int(part))

    # De-dup while preserving order.
    seen: set[int] = set()
    uniq: list[int] = []
    for d in out:
        if d in seen:
            continue
        seen.add(d)
        uniq.append(d)
    return uniq if uniq else [int(fallback_device)]


def _build_child_cmd(
    *,
    script: Path,
    case: Case,
    args: argparse.Namespace,
    case_block_dim: int,
    device_id: int,
    result_json: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(script),
        "--run-mode",
        str(args.run_mode),
        "--soc",
        str(args.soc),
        "--ascend-home",
        str(args.ascend_home),
        "--ptoas",
        str(args.ptoas),
        "--outdir",
        str(args.outdir),
        "--device",
        str(int(device_id)),
        "--block-dim",
        str(int(case_block_dim)),
        "--memory-model",
        str(args.memory_model),
        "--retries",
        "0",
        "--bench-iters",
        str(int(args.bench_iters)),
        "--bench-warmup",
        str(int(args.bench_warmup)),
        "--bench-max-bytes",
        str(int(args.bench_max_bytes)),
        "--_result-json",
        str(result_json),
        "--cases",
        case.name,
    ]
    if args.verbose_build:
        cmd.append("--verbose-build")
    if args.keep_going:
        cmd.append("--keep-going")
    if not args.insert_events:
        cmd.append("--no-insert-events")
    if args.show_events:
        cmd.append("--show-events")
    return cmd


def _cases() -> list[Case]:
    base = Path(__file__).resolve().parent
    return [
        # Core kernels.
        Case(name="add16", py=base / "add16.py", kernel="add16", arch="dav-c220-vec"),
        Case(name="mul16_f16", py=base / "mul16_f16.py", kernel="mul16_f16", arch="dav-c220-vec"),
        Case(name="mul16", py=base / "mul16.py", kernel="mul16", arch="dav-c220-vec"),
        Case(name="sub16", py=base / "sub16.py", kernel="sub16", arch="dav-c220-vec"),
        Case(name="neg16", py=base / "neg16.py", kernel="neg16", arch="dav-c220-vec"),
        Case(name="scale16", py=base / "scale16.py", kernel="scale16", arch="dav-c220-vec"),
        Case(name="bias16", py=base / "bias16.py", kernel="bias16", arch="dav-c220-vec"),
        Case(name="transpose16", py=base / "transpose16.py", kernel="transpose16", arch="dav-c220-vec"),
        Case(name="tiled_transpose64", py=base / "tiled_transpose64.py", kernel="tiled_transpose64", arch="dav-c220-vec"),
        Case(name="abs16", py=base / "abs16.py", kernel="abs16", arch="dav-c220-vec"),
        Case(name="abs_add16", py=base / "abs_add16.py", kernel="abs_add16", arch="dav-c220-vec"),
        Case(name="axpy16", py=base / "axpy16.py", kernel="axpy16", arch="dav-c220-vec"),
        Case(name="fma16", py=base / "fma16.py", kernel="fma16", arch="dav-c220-vec"),
        Case(name="rowmax16", py=base / "rowmax16.py", kernel="rowmax16", arch="dav-c220-vec"),
        Case(name="rowsum16", py=base / "rowsum16.py", kernel="rowsum16", arch="dav-c220-vec"),
        Case(name="tiled_rowsum64", py=base / "tiled_rowsum64.py", kernel="tiled_rowsum64", arch="dav-c220-vec"),
        Case(name="tiled_add128", py=base / "tiled_add128.py", kernel="tiled_add128", arch="dav-c220-vec"),
        # SPMD / multi-block stress tests (per-case block_dim).
        Case(
            name="spmd_tiled_add256",
            py=base / "spmd_tiled_add256.py",
            kernel="spmd_tiled_add256",
            arch="dav-c220-vec",
            block_dim=8,
        ),
        Case(
            name="spmd_tiled_transpose256",
            py=base / "spmd_tiled_transpose256.py",
            kernel="spmd_tiled_transpose256",
            arch="dav-c220-vec",
            block_dim=8,
        ),
        Case(
            name="spmd_tiled_rowsum256",
            py=base / "spmd_tiled_rowsum256.py",
            kernel="spmd_tiled_rowsum256",
            arch="dav-c220-vec",
            block_dim=8,
        ),
        # NOTE: TEXP is approx on NPU; scale inputs down to keep CPU-vs-NPU drift small.
        Case(name="sinh16", py=base / "sinh16.py", kernel="sinh16", arch="dav-c220-vec", input_scale=0.05),
        Case(name="softmax16", py=base / "softmax16.py", kernel="softmax16", arch="dav-c220-vec"),
        Case(name="softmax32x16", py=base / "softmax32x16.py", kernel="softmax32x16", arch="dav-c220-vec"),
        Case(name="gemm16", py=base / "gemm16.py", kernel="gemm16", arch="dav-c220-cube"),

        # Ported from `~/github/pto-isa/examples/*.py` (kept runnable end-to-end here).
        Case(name="pto_isa_sinh", py=base / "pto_isa_sinh.py", kernel="pto_isa_sinh", arch="dav-c220-vec", input_scale=0.05),
        Case(name="pto_fused_softmax", py=base / "pto_fused_softmax.py", kernel="pto_fused_softmax", arch="dav-c220-vec"),
        Case(name="pto_aten_ir_primitives", py=base / "pto_aten_ir_primitives.py", kernel="pto_aten_ir_primitives", arch="dav-c220-vec"),
        Case(name="pto_torch_tensor", py=base / "pto_torch_tensor.py", kernel="pto_torch_tensor", arch="dav-c220-vec"),
        Case(name="pto_torch_functional", py=base / "pto_torch_functional.py", kernel="pto_torch_functional", arch="dav-c220-vec"),
        Case(name="pto_torch_nn_operators", py=base / "pto_torch_nn_operators.py", kernel="pto_torch_nn_operators", arch="dav-c220-cube"),
        Case(name="pto_torch_flexattention", py=base / "pto_torch_flexattention.py", kernel="pto_torch_flexattention", arch="dav-c220-vec"),
        Case(name="pto_llama7B_dynamic", py=base / "pto_llama7B_dynamic.py", kernel="pto_llama7B_dynamic", arch="dav-c220-vec"),
        # FlashAttention-like multi-stage demo (cube + vec), split into multiple kernels.
        Case(
            name="flash_attention64_split",
            py=base / "flash_attention64_split.py",
            kernel="flash_attention64_split",
            arch="dav-c220",
            block_dim=8,
            # FlashAttention-style softmax is sensitive to exp approximation drift on NPU/sim.
            # Keep inputs small so CPU-vs-NPU remains comparable under default tolerances.
            input_scale=0.1,
            split_kernels=True,
        ),

        # pyPTO API coverage (one kernel covers multiple shapes with <=32KB tiles).
        Case(name="api_memory_ops", py=base / "pypto_api_suite.py", kernel="api_memory_ops", arch="dav-c220-vec"),
        Case(name="api_push_pop_ops", py=base / "pypto_api_suite.py", kernel="api_push_pop_ops", arch="dav-c220-vec"),
        Case(name="api_vec_binary_ops", py=base / "pypto_api_suite.py", kernel="api_vec_binary_ops", arch="dav-c220-vec"),
        Case(name="api_vec_unary_ops", py=base / "pypto_api_suite.py", kernel="api_vec_unary_ops", arch="dav-c220-vec"),
        Case(name="api_vec_scalar_ops", py=base / "pypto_api_suite.py", kernel="api_vec_scalar_ops", arch="dav-c220-vec"),
        Case(name="api_row_reduce_ops", py=base / "pypto_api_suite.py", kernel="api_row_reduce_ops", arch="dav-c220-vec"),
        Case(name="api_row_expand_ops", py=base / "pypto_api_suite.py", kernel="api_row_expand_ops", arch="dav-c220-vec"),
        Case(name="api_transpose_ops", py=base / "pypto_api_suite.py", kernel="api_transpose_ops", arch="dav-c220-vec"),
        # Matmul uses cube; keep inputs small to reduce CPU-vs-NPU drift on large K.
        Case(
            name="api_matmul_ops",
            py=base / "pypto_api_suite.py",
            kernel="api_matmul_ops",
            arch="dav-c220-cube",
            input_scale=0.1,
        ),
        Case(
            name="api_matmul_bias_ops",
            py=base / "pypto_api_suite.py",
            kernel="api_matmul_bias_ops",
            arch="dav-c220-cube",
            input_scale=0.1,
        ),
        Case(
            name="api_matmul_mx_ops",
            py=base / "pypto_api_suite.py",
            kernel="api_matmul_mx_ops",
            arch="dav-c220-cube",
            input_scale=0.1,
        ),
        Case(name="api_memory_extra_ops", py=base / "pypto_api_suite.py", kernel="api_memory_extra_ops", arch="dav-c220-vec"),
        Case(name="api_addc_ops", py=base / "pypto_api_suite.py", kernel="api_addc_ops", arch="dav-c220-vec"),
        Case(name="api_bitwise_shift_ops", py=base / "pypto_api_suite.py", kernel="api_bitwise_shift_ops", arch="dav-c220-vec"),
        Case(name="api_part_ops", py=base / "pypto_api_suite.py", kernel="api_part_ops", arch="dav-c220-vec"),
        Case(name="api_rem_ops", py=base / "pypto_api_suite.py", kernel="api_rem_ops", arch="dav-c220-vec"),
        Case(name="api_cmp_select_ops", py=base / "pypto_api_suite.py", kernel="api_cmp_select_ops", arch="dav-c220-vec"),
        Case(name="api_col_expand_ops", py=base / "pypto_api_suite.py", kernel="api_col_expand_ops", arch="dav-c220-vec"),
        Case(name="api_fillpad_ops", py=base / "pypto_api_suite.py", kernel="api_fillpad_ops", arch="dav-c220-vec"),
        Case(
            name="api_extract_insert_reshape_ops",
            py=base / "pypto_api_suite.py",
            kernel="api_extract_insert_reshape_ops",
            arch="dav-c220-cube",
            input_scale=0.1,
        ),
        Case(name="api_sort_ops", py=base / "pypto_api_suite.py", kernel="api_sort_ops", arch="dav-c220-vec"),
        Case(name="api_gather_scatter_ops", py=base / "pypto_api_suite.py", kernel="api_gather_scatter_ops", arch="dav-c220-vec"),
    ]


def _run_kernel_file_e2e(
    *,
    case: Case,
    outdir: Path,
    ptoas: Path,
    ascend_home: Path,
    run_mode: str,
    soc: str,
    device: int,
    block_dim: int,
    memory_model: str,
    insert_events: bool,
    bench_iters: int,
    bench_warmup: int,
    bench_max_bytes: int,
    show_events: bool,
) -> pipeline.NpuRunResult:
    # Compile Python -> PTO-AS.
    spec = binding.compile_file(case.py, kernel=case.kernel)
    pto_text = prepend_host_spec_to_pto(pto=spec.pto, spec=binding.default_host_spec(spec))

    outdir.mkdir(parents=True, exist_ok=True)
    pto_path = outdir / f"{spec.name}.pto"
    pto_path.write_text(pto_text, encoding="utf-8")

    # CPU reference.
    host_spec = pipeline.parse_or_default_host_spec(pto_text=pto_text)
    host_spec = type(host_spec)(
        args=host_spec.args, seed=host_spec.seed, block_dim=block_dim, kernel_name=host_spec.kernel_name
    )
    base_arrays = pipeline.make_host_arrays(host_spec)
    if case.input_scale is not None:
        for i, a in enumerate(host_spec.args):
            if a.role == "out":
                continue
            if base_arrays[i].dtype in (np.float16, np.float32):
                base_arrays[i] = (base_arrays[i].astype(np.float32) * float(case.input_scale)).astype(base_arrays[i].dtype)

    cpu_cpp = pipeline.compile_pto_to_cpu_cpp(pto_path=pto_path, outdir=outdir, ptoas=ptoas)
    cpu_so = outdir / f"lib{spec.name}_cpu.so"
    pipeline.build_cpu_so_from_cpp(cpp_path=cpu_cpp, out_so=cpu_so)
    cpu_arrays = [a.copy() for a in base_arrays]
    cpu_out = pipeline.run_cpu_kernel_from_so(so_path=cpu_so, host_spec=host_spec, host_arrays=cpu_arrays)

    # NPU run (sim or real).
    if run_mode == "sim":
        camodel = outdir / "camodel_logs"
        camodel.mkdir(parents=True, exist_ok=True)
        os.environ["CAMODEL_LOG_PATH"] = str(camodel)
        soc_full = _soc_from_alias(soc)
        runtime_lib = "runtime_camodel"
    else:
        soc_full = None
        runtime_lib = "runtime"

    cfg = pipeline.CompileConfig(
        ptoas=ptoas,
        ascend_home=ascend_home,
        arch=case.arch,
        memory_model=memory_model,
        insert_events=insert_events,
        split_kernels=case.split_kernels,
    )
    cce_cpp, _bin = pipeline.compile_pto_to_cce_and_bin(pto_path=pto_path, outdir=outdir, cfg=cfg)
    # Emit a small summary of inserted events for debugging deadlocks.
    try:
        summary = pipeline.summarize_cce_events(cce_path=cce_cpp)
        (outdir / "event_summary.txt").write_text(str(summary) + "\n", encoding="utf-8")
        snippet = pipeline.extract_cce_set_wait_lines(cce_path=cce_cpp, limit=200)
        (outdir / "set_wait_snippet.txt").write_text("\n".join(snippet) + ("\n" if snippet else ""), encoding="utf-8")
        if show_events:
            sys.stdout.write(
                f"  events: set={summary.get('set_total')} wait={summary.get('wait_total')} (see {outdir / 'event_summary.txt'})\n"
            )
            sys.stdout.write(f"  set/wait snippet: {outdir / 'set_wait_snippet.txt'}\n")
    except Exception:
        pass
    npu_so = outdir / f"lib{spec.name}_{run_mode}.so"
    pipeline.build_fatobj_so_from_cce(
        cce_path=cce_cpp,
        out_so=npu_so,
        arch=cfg.arch,
        ascend_home=cfg.ascend_home,
        runtime_lib=runtime_lib,
        soc=soc_full,
    )

    npu_arrays = [a.copy() for a in base_arrays]
    iters = int(bench_iters) if run_mode == "npu" else 0
    warmup = int(bench_warmup) if run_mode == "npu" else 0
    if iters > 0 and int(bench_max_bytes) > 0:
        total_bytes = sum(int(a.nbytes) for a in npu_arrays)
        if total_bytes > int(bench_max_bytes):
            iters = 1
            warmup = 0

    npu_res = pipeline.run_npu_kernel_from_so(
        so_path=npu_so,
        host_spec=host_spec,
        host_arrays=npu_arrays,
        device_id=device,
        block_dim=block_dim,
        bench_iters=iters,
        bench_warmup=warmup,
    )
    npu_out = npu_res.outputs
    out_dtypes = [host_spec.args[i].dtype for i in host_spec.output_indices()]
    if case.name in ("sinh16", "pto_isa_sinh"):
        # `sinh(x) = (exp(x) - exp(-x)) / 2` is numerically unstable for small x, and
        # NPU vector subtraction may introduce larger relative error when subtracting
        # near-equal values. Keep this tolerance slightly looser than the default f32
        # settings so the regression is robust.
        for i, (c, n, dt) in enumerate(zip(cpu_out, npu_out, out_dtypes)):
            if dt != "f32":
                raise AssertionError(f"unexpected output dtype for {case.name}: output {i} is {dt}")
            np.testing.assert_allclose(n, c, rtol=2e-2, atol=3e-2, err_msg=f"output {i} ({dt}) mismatch")
    else:
        pipeline.compare_cpu_and_npu_outputs(cpu_out=cpu_out, npu_out=npu_out, out_dtypes=out_dtypes)
    return npu_res


def main() -> int:
    repo = pipeline.repo_root()
    ap = argparse.ArgumentParser(description="Run kernels/python examples end-to-end with progress output.")
    ap.add_argument("--verbose-build", action="store_true", help="Print compiler commands/warnings")
    ap.add_argument("--run-mode", choices=["npu", "sim"], default="npu")
    ap.add_argument("--soc", default="a3", help="Simulator SoC alias when --run-mode=sim (a3|a5|...)")
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas(repo))
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/pto_kernels_python_regression"))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--block-dim", type=int, default=1)
    ap.add_argument("--memory-model", default="MEMORY_BASE")
    ap.add_argument("--no-insert-events", dest="insert_events", action="store_false", default=True)
    ap.add_argument("--filter", default=None, help="Substring filter for case names")
    ap.add_argument("--cases", default=None, help="Comma-separated case names (overrides --filter)")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--timeout-sec", type=float, default=None, help="Per-case wall-time timeout (kills hung NPU runs).")
    ap.add_argument("--sim-on-timeout", action="store_true", help="If NPU run times out, rerun the same case in sim mode.")
    ap.add_argument("--retries", type=int, default=1, help="Retry a failing case up to N times (helps with flaky NPU runs).")
    ap.add_argument(
        "--devices",
        default="auto",
        help="NPU device ids for parallel runs: auto|all|0,1,2|0-7 (default: auto).",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Max concurrent cases (default: 0=auto, uses number of selected devices).",
    )
    ap.add_argument(
        "--parallel",
        action="store_true",
        help="Run cases concurrently across selected NPU devices (NPU only).",
    )
    ap.add_argument(
        "--no-parallel",
        action="store_true",
        help="Force sequential execution even when multiple NPU devices are available.",
    )
    ap.add_argument("--bench-iters", type=int, default=50, help="NPU-only: measure kernel time with ACL events.")
    ap.add_argument("--bench-warmup", type=int, default=10, help="NPU-only: warmup iterations before timing.")
    ap.add_argument(
        "--bench-max-bytes",
        type=int,
        default=1 << 20,
        help="If total H2D bytes exceed this, reduce benchmark to 1 iteration (avoids slow large-kernel benches).",
    )
    ap.add_argument("--show-events", action="store_true", help="Print set/wait summary paths (for deadlock debug).")
    ap.add_argument(
        "--show-perf-apis",
        action="store_true",
        help="Print which timing/profiling APIs are present in the Ascend toolkit headers.",
    )
    ap.add_argument("--_result-json", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not args.verbose_build:
        os.environ.setdefault("PTOAS_QUIET", "1")

    is_child = os.environ.get("PTOAS_REG_CHILD", "") in ("1", "true", "True", "yes", "YES")

    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2
    if not args.ascend_home or not args.ascend_home.exists():
        print("error: set --ascend-home or ASCEND_HOME_PATH to your Ascend toolkit root", file=sys.stderr)
        return 2

    if args.show_perf_apis and not is_child:
        print(json.dumps(pipeline.discover_ascend_perf_apis(ascend_home=args.ascend_home), indent=2, sort_keys=True))

    # In timeout mode we spawn child processes; do not re-exec the parent via ensure_ascend_sim_env().
    if args.run_mode == "sim" and not args.timeout_sec:
        pipeline.ensure_ascend_sim_env(ascend_home=args.ascend_home, soc=_soc_from_alias(args.soc))

    cases = _cases()
    if args.cases:
        want = [s.strip() for s in args.cases.split(",") if s.strip()]
        want_set = set(want)
        cases = [c for c in cases if c.name in want_set]
        missing = [n for n in want if n not in {c.name for c in cases}]
        if missing:
            print("error: unknown cases: " + ", ".join(missing), file=sys.stderr)
            print("available: " + ", ".join(c.name for c in _cases()), file=sys.stderr)
            return 2
    elif args.filter:
        cases = [c for c in cases if args.filter in c.name]
    if not cases:
        print("error: no cases selected", file=sys.stderr)
        return 2

    if not is_child:
        print(f"ptoas: {args.ptoas}")
        print(f"ascend_home: {args.ascend_home}")
        print(f"run_mode: {args.run_mode}  device: {args.device}  block_dim: {args.block_dim}")
        print(f"outdir: {args.outdir}")
        print(f"cases: {', '.join(c.name for c in cases)}")
        print()

    reports: dict[str, CaseReport] = {}
    report_order: list[str] = []
    timed_out: list[str] = []

    def _record(r: CaseReport) -> None:
        if r.name not in reports:
            report_order.append(r.name)
        reports[r.name] = r

    def _make_ok_report(*, case: Case, block_dim: int, wall_s: float, outdir: Path, npu_res: pipeline.NpuRunResult) -> CaseReport:
        r = CaseReport(
            name=case.name,
            arch=case.arch,
            block_dim=int(block_dim),
            status="OK",
            wall_s=float(wall_s),
            outdir=str(outdir),
        )
        if args.run_mode == "npu":
            r.device_id = int(npu_res.device.device_id)
            r.soc = npu_res.device.soc
            r.device_count = npu_res.device.device_count
            if npu_res.bench is not None:
                b = npu_res.bench
                r.avg_us = float(b.avg_us)
                r.p50_us = float(b.p50_us)
                r.min_us = float(b.min_us)
                r.max_us = float(b.max_us)
                r.iters = int(b.iters)
        return r

    t0 = time.perf_counter()

    # Parallel NPU mode: schedule cases across devices to improve regression speed.
    devices = _parse_devices(args.devices, fallback_device=int(args.device))
    use_parallel = (
        (not is_child)
        and (not args.no_parallel)
        and (args.run_mode == "npu")
        and (len(cases) > 1)
        and (len(devices) > 1)
        and (bool(args.parallel) or float(args.timeout_sec or 0.0) > 0.0)
    )
    if use_parallel:
        jobs = int(args.jobs) if int(args.jobs) > 0 else len(devices)
        jobs = max(1, min(jobs, len(devices)))
        script = Path(__file__).resolve()

        if not is_child:
            timeout_s = float(args.timeout_sec) if args.timeout_sec is not None else None
            timeout_txt = f"{timeout_s}s" if timeout_s is not None else "-"
            note = " (no timeout: a deadlock may hang a worker)" if timeout_s is None else ""
            sys.stdout.write(
                f"parallel: jobs={jobs} devices={','.join(str(d) for d in devices)} timeout={timeout_txt}{note}\n"
            )
            sys.stdout.flush()

        queue: list[tuple[Case, int, int]] = [
            (c, int(c.block_dim) if c.block_dim is not None else int(args.block_dim), 0) for c in cases
        ]
        free_devices: list[int] = list(devices)

        @dataclass
        class _Job:
            case: Case
            block_dim: int
            device_id: int
            attempt: int
            start_t: float
            proc: subprocess.Popen[bytes]
            result_json: Path
            log_path: Path
            log_f: object

        running: dict[int, _Job] = {}
        done = 0
        total = len(cases)
        retries = max(0, int(args.retries))

        def _start_one(case: Case, block_dim: int, device_id: int, attempt: int) -> _Job:
            case_outdir = args.outdir / case.name
            case_outdir.mkdir(parents=True, exist_ok=True)
            result_json = case_outdir / "run_result.json"
            log_path = case_outdir / f"child_dev{device_id}_attempt{attempt + 1}.log"

            cmd = _build_child_cmd(
                script=script,
                case=case,
                args=args,
                case_block_dim=block_dim,
                device_id=device_id,
                result_json=result_json,
            )

            env = dict(os.environ)
            env["PTOAS_REG_CHILD"] = "1"

            log_f = log_path.open("wb")
            try:
                proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)
            except Exception:
                log_f.close()
                raise

            return _Job(
                case=case,
                block_dim=block_dim,
                device_id=device_id,
                attempt=attempt,
                start_t=time.perf_counter(),
                proc=proc,
                result_json=result_json,
                log_path=log_path,
                log_f=log_f,
            )

        def _finish_job(job: _Job, *, status_override: str | None = None, error_override: str | None = None) -> None:
            nonlocal done
            wall_s = float(time.perf_counter() - job.start_t)
            r: CaseReport | None = None
            if job.result_json.exists():
                try:
                    r = CaseReport.from_json(job.result_json.read_text(encoding="utf-8"))
                except Exception:
                    r = None
            if r is None:
                r = CaseReport(
                    name=job.case.name,
                    arch=job.case.arch,
                    block_dim=int(job.block_dim),
                    status="OK" if int(job.proc.returncode or 0) == 0 else "FAIL",
                    wall_s=wall_s,
                    outdir=str(args.outdir / job.case.name),
                )
            r.wall_s = wall_s
            if status_override is not None:
                r.status = status_override
            if error_override is not None:
                r.error = error_override
            _record(r)
            done += 1
            try:
                job.log_f.close()
            except Exception:
                pass
            if not is_child:
                sys.stdout.write(f"[{done:>2}/{total}] DONE {r.name} {r.status} dev={job.device_id} wall={wall_s:.2f}s\n")
                sys.stdout.flush()

        # Main scheduler loop.
        while queue or running:
            # Launch while we have capacity.
            while queue and free_devices and len(running) < jobs:
                case, block_dim, attempt = queue.pop(0)
                dev = free_devices.pop(0)
                job = _start_one(case, block_dim, dev, attempt)
                running[job.proc.pid] = job

            # Poll running.
            for pid, job in list(running.items()):
                ret = job.proc.poll()
                if ret is None:
                    continue

                if int(ret) == 0:
                    _finish_job(job)
                    free_devices.append(job.device_id)
                    del running[pid]
                    continue

                # Non-zero: retry if allowed, otherwise fail.
                if job.attempt < retries:
                    try:
                        job.log_f.close()
                    except Exception:
                        pass
                    # Requeue the same case for another attempt.
                    free_devices.append(job.device_id)
                    del running[pid]
                    queue.append((job.case, job.block_dim, job.attempt + 1))
                    continue

                _finish_job(job, status_override="FAIL", error_override=f"child returned {int(ret)} (see {job.log_path})")
                free_devices.append(job.device_id)
                del running[pid]

            # Timeouts.
            if float(args.timeout_sec or 0.0) > 0.0:
                now = time.perf_counter()
                for pid, job in list(running.items()):
                    if now - job.start_t <= float(args.timeout_sec):
                        continue
                    try:
                        job.proc.kill()
                    except Exception:
                        pass
                    try:
                        job.proc.wait(timeout=5.0)
                    except Exception:
                        pass

                    timed_out.append(job.case.name)
                    _finish_job(job, status_override="TIMEOUT", error_override=f"timeout>{float(args.timeout_sec)}s (see {job.log_path})")

                    # Optional simulator fallback for debugging deadlocks.
                    if args.run_mode == "npu" and args.sim_on_timeout:
                        sim_outdir = args.outdir.with_name(args.outdir.name + "_timeout_sim")
                        sim_result = sim_outdir / job.case.name / "run_result.json"
                        sim_cmd = _build_child_cmd(
                            script=script,
                            case=job.case,
                            args=args,
                            case_block_dim=job.block_dim,
                            device_id=int(args.device),
                            result_json=sim_result,
                        )
                        sim_cmd[sim_cmd.index("--run-mode") + 1] = "sim"
                        sim_cmd[sim_cmd.index("--outdir") + 1] = str(sim_outdir)
                        env = dict(os.environ)
                        env.setdefault("PTOAS_VERBOSE_RUN", "1")
                        env.setdefault("PTOAS_DISABLE_RPATH", "1")
                        env["PTOAS_REG_CHILD"] = "1"
                        try:
                            subprocess.run(sim_cmd, check=False, timeout=120.0, env=env)
                        except subprocess.TimeoutExpired:
                            pass

                    free_devices.append(job.device_id)
                    del running[pid]

            time.sleep(0.05)

        total_dt = time.perf_counter() - t0
        # Fall through to the common summary/table printing below.

    else:
        script = Path(__file__).resolve()
        retries = max(0, int(args.retries))

        for i, c in enumerate(cases, start=1):
            case_outdir = args.outdir / c.name
            case_block_dim = int(c.block_dim) if c.block_dim is not None else int(args.block_dim)
            case_t0 = time.perf_counter()
            try:
                if not is_child:
                    sys.stdout.write(f"[{i:>2}/{len(cases)}] RUN {c.name} (block_dim={case_block_dim})\n")
                    sys.stdout.flush()

                if args.timeout_sec and float(args.timeout_sec) > 0:
                    result_json = case_outdir / "run_result.json"
                    cmd = _build_child_cmd(
                        script=script,
                        case=c,
                        args=args,
                        case_block_dim=case_block_dim,
                        device_id=int(args.device),
                        result_json=result_json,
                    )

                    attempt = 0
                    while True:
                        case_outdir.mkdir(parents=True, exist_ok=True)
                        log_path = case_outdir / f"child_dev{int(args.device)}_attempt{attempt + 1}.log"
                        env = dict(os.environ)
                        env["PTOAS_REG_CHILD"] = "1"
                        try:
                            with log_path.open("wb") as log_f:
                                proc = subprocess.run(
                                    cmd,
                                    check=False,
                                    timeout=float(args.timeout_sec),
                                    env=env,
                                    stdout=log_f,
                                    stderr=log_f,
                                )
                        except subprocess.TimeoutExpired:
                            timed_out.append(c.name)
                            wall_s = float(time.perf_counter() - case_t0)
                            _record(
                                CaseReport(
                                    name=c.name,
                                    arch=c.arch,
                                    block_dim=int(case_block_dim),
                                    status="TIMEOUT",
                                    wall_s=wall_s,
                                    outdir=str(case_outdir),
                                    error=f"timeout>{float(args.timeout_sec)}s (see {log_path})",
                                )
                            )

                            if args.run_mode == "npu" and args.sim_on_timeout:
                                sim_outdir = args.outdir.with_name(args.outdir.name + "_timeout_sim")
                                sim_result = sim_outdir / c.name / "run_result.json"
                                sim_cmd = _build_child_cmd(
                                    script=script,
                                    case=c,
                                    args=args,
                                    case_block_dim=case_block_dim,
                                    device_id=int(args.device),
                                    result_json=sim_result,
                                )
                                sim_cmd[sim_cmd.index("--run-mode") + 1] = "sim"
                                sim_cmd[sim_cmd.index("--outdir") + 1] = str(sim_outdir)
                                env = dict(os.environ)
                                env.setdefault("PTOAS_VERBOSE_RUN", "1")
                                env.setdefault("PTOAS_DISABLE_RPATH", "1")
                                env["PTOAS_REG_CHILD"] = "1"
                                try:
                                    subprocess.run(sim_cmd, check=False, timeout=120.0, env=env)
                                except subprocess.TimeoutExpired:
                                    pass
                            break

                        if int(proc.returncode) == 0:
                            wall_s = float(time.perf_counter() - case_t0)
                            if result_json.exists():
                                try:
                                    rr = CaseReport.from_json(result_json.read_text(encoding="utf-8"))
                                    rr.wall_s = wall_s
                                    _record(rr)
                                except Exception:
                                    _record(
                                        CaseReport(
                                            name=c.name,
                                            arch=c.arch,
                                            block_dim=int(case_block_dim),
                                            status="OK",
                                            wall_s=wall_s,
                                            outdir=str(case_outdir),
                                        )
                                    )
                            else:
                                _record(
                                    CaseReport(
                                        name=c.name,
                                        arch=c.arch,
                                        block_dim=int(case_block_dim),
                                        status="OK",
                                        wall_s=wall_s,
                                        outdir=str(case_outdir),
                                    )
                                )
                            break

                        attempt += 1
                        if attempt > retries:
                            wall_s = float(time.perf_counter() - case_t0)
                            _record(
                                CaseReport(
                                    name=c.name,
                                    arch=c.arch,
                                    block_dim=int(case_block_dim),
                                    status="FAIL",
                                    wall_s=wall_s,
                                    outdir=str(case_outdir),
                                    error=f"child returned {int(proc.returncode)} (see {log_path})",
                                )
                            )
                            raise RuntimeError(f"child returned {proc.returncode}")
                        time.sleep(0.5)

                    # Timeout mode always runs via child; do not run again in-process.
                    continue

                attempt = 0
                while True:
                    try:
                        npu_res = _run_kernel_file_e2e(
                            case=c,
                            outdir=case_outdir,
                            ptoas=args.ptoas,
                            ascend_home=args.ascend_home,
                            run_mode=args.run_mode,
                            soc=args.soc,
                            device=args.device,
                            block_dim=case_block_dim,
                            memory_model=args.memory_model,
                            insert_events=args.insert_events,
                            bench_iters=int(args.bench_iters),
                            bench_warmup=int(args.bench_warmup),
                            bench_max_bytes=int(args.bench_max_bytes),
                            show_events=bool(args.show_events),
                        )
                        wall_s = float(time.perf_counter() - case_t0)
                        r = _make_ok_report(
                            case=c,
                            block_dim=case_block_dim,
                            wall_s=wall_s,
                            outdir=case_outdir,
                            npu_res=npu_res,
                        )
                        _record(r)
                        if args._result_json is not None:
                            _write_report(args._result_json, r)
                        break
                    except Exception:
                        attempt += 1
                        if attempt > retries:
                            raise
                        time.sleep(0.5)
            except Exception:
                wall_s = float(time.perf_counter() - case_t0)
                tb = traceback.format_exc()
                case_outdir.mkdir(parents=True, exist_ok=True)
                (case_outdir / "error.txt").write_text(tb, encoding="utf-8")
                err = tb.strip().splitlines()[-1] if tb.strip().splitlines() else "error"
                r = CaseReport(
                    name=c.name,
                    arch=c.arch,
                    block_dim=int(case_block_dim),
                    status="FAIL",
                    wall_s=wall_s,
                    outdir=str(case_outdir),
                    error=err,
                )
                _record(r)
                if args._result_json is not None:
                    _write_report(args._result_json, r)
                if not args.keep_going:
                    break

    total_dt = time.perf_counter() - t0
    if not is_child:
        all_reports = [reports[n] for n in report_order if n in reports]

        # Sort to highlight performance: OK rows by avg_us (when present), then others.
        def _key(r: CaseReport) -> tuple[int, float, str]:
            if r.status != "OK":
                return (1, float("inf"), r.name)
            return (0, float(r.avg_us) if r.avg_us is not None else float("inf"), r.name)

        ordered = sorted(all_reports, key=_key)

        header = [
            "case",
            "arch",
            "blk",
            "status",
            "avg_us",
            "p50_us",
            "min_us",
            "max_us",
            "iters",
            "wall_s",
            "npu",
            "err",
        ]
        table_rows: list[list[str]] = [header]
        for r in ordered:
            npu_s = "-"
            if r.device_id is not None:
                soc = r.soc or "unknown"
                cnt = f"/{r.device_count}" if r.device_count is not None else ""
                npu_s = f"{soc} dev{r.device_id}{cnt}"
            err_s = "-"
            if r.status == "FAIL" and r.error:
                err_s = r.error.strip()
                if len(err_s) > 60:
                    err_s = err_s[:57] + "..."
            table_rows.append(
                [
                    r.name,
                    r.arch,
                    str(r.block_dim),
                    r.status,
                    _fmt_float(r.avg_us),
                    _fmt_float(r.p50_us),
                    _fmt_float(r.min_us),
                    _fmt_float(r.max_us),
                    str(r.iters) if r.iters is not None else "-",
                    _fmt_float(r.wall_s, fmt="{:.2f}"),
                    npu_s,
                    err_s,
                ]
            )

        print(_format_table(table_rows))
        print()

        ok = sum(1 for r in all_reports if r.status == "OK")
        fail = sum(1 for r in all_reports if r.status == "FAIL")
        tout = sum(1 for r in all_reports if r.status == "TIMEOUT")
        print(f"Summary: {ok} OK, {fail} FAIL, {tout} TIMEOUT, total {total_dt:.1f}s")

    if any(r.status == "FAIL" for r in reports.values()):
        return 1
    if any(r.status == "TIMEOUT" for r in reports.values()):
        return 124
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
