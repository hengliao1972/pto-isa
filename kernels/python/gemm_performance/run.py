#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptoas.python import pipeline  # noqa: E402
from ptoas.python import binding  # noqa: E402
from ptoas.python.host_spec import prepend_host_spec_to_pto  # noqa: E402
from ptoas.python.host_codegen import TensorSpec, emit_acl_host_cpp  # noqa: E402

from kernels.python.gemm_performance.kernel import make_gemm_performance_kernel  # noqa: E402


def _default_ptoas() -> Path:
    return _REPO_ROOT / "bin/ptoas"


def _soc_from_alias(alias: str) -> str:
    if alias == "a3":
        return "Ascend910B1"
    if alias == "a5":
        return "Ascend910_9599"
    return alias


def _summarize_camodel_set_wait_flags(*, log_dir: Path) -> tuple[int, int, list[str]]:
    """
    Best-effort scan of Ascend simulator dumps for SET_FLAG / WAIT_FLAG.
    """
    set_count = 0
    wait_count = 0
    samples: list[str] = []
    if not log_dir.exists():
        return (0, 0, [])
    for f in sorted(log_dir.rglob("*.instr_log.dump")):
        try:
            with f.open("r", errors="ignore") as fp:
                for line in fp:
                    if "SET_FLAG" in line:
                        set_count += 1
                        if len(samples) < 6:
                            samples.append(f"{f.name}: {line.strip()}")
                    if "WAIT_FLAG" in line:
                        wait_count += 1
                        if len(samples) < 6:
                            samples.append(f"{f.name}: {line.strip()}")
        except OSError:
            continue
    return (set_count, wait_count, samples)


def _build_and_run_sim_dump_instr(
    *,
    outdir: Path,
    so_path: Path,
    host_specs: list[TensorSpec],
    ascend_home: Path,
    soc: str,
    device: int,
    block_dim: int,
) -> Path:
    import subprocess

    host_cpp = outdir / "host.cpp"
    host_cpp.write_text(emit_acl_host_cpp(so_basename=str(so_path), args=host_specs), encoding="utf-8")

    sim_lib = pipeline.resolve_ascend_simulator_lib_dir(ascend_home=ascend_home, soc=soc)
    host_exe = outdir / "host_sim_gemm_performance"

    cmd = [
        "g++",
        str(host_cpp),
        "-o",
        str(host_exe),
        "-O2",
        "-std=c++17",
        f"-I{ascend_home / 'include'}",
        f"-I{ascend_home / 'pkg_inc'}",
        f"-I{ascend_home / 'pkg_inc' / 'runtime' / 'runtime'}",
        f"-I{ascend_home / 'pkg_inc' / 'profiling'}",
        f"-L{ascend_home / 'lib64'}",
        f"-L{sim_lib}",
        f"-Wl,-rpath,{ascend_home / 'lib64'}",
        f"-Wl,-rpath,{sim_lib}",
        "-lruntime_camodel",
        "-lnpu_drv_camodel",
        "-lascendcl",
        "-ltiling_api",
        "-lplatform",
        "-lc_sec",
        "-ldl",
        "-lm",
        "-lstdc++",
        "-lpthread",
    ]
    subprocess.run(cmd, check=True)

    dump_dir = outdir / "host_camodel_logs"
    dump_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CAMODEL_LOG_PATH"] = str(dump_dir)
    env["ASCEND_PROCESS_LOG_PATH"] = str(dump_dir)
    env["LD_LIBRARY_PATH"] = f"{sim_lib}:{ascend_home / 'lib64'}:{env.get('LD_LIBRARY_PATH', '')}"
    subprocess.run(
        [str(host_exe), "--so", str(so_path), "--device", str(int(device)), "--block-dim", str(int(block_dim))],
        cwd=str(outdir),
        env=env,
        check=True,
    )
    return dump_dir


def _read_device_f32(acl, *, dev_ptr: int, offset_bytes: int) -> float:
    import ctypes
    import struct

    buf = (ctypes.c_ubyte * 4)()
    host_ptr = ctypes.addressof(buf)
    src_ptr = int(dev_ptr) + int(offset_bytes)
    ret = acl.rt.memcpy(int(host_ptr), 4, int(src_ptr), 4, 2)  # D2H
    if int(ret) != 0:
        raise RuntimeError(f"acl.rt.memcpy(D2H, 4B) failed (ret={ret})")
    return float(struct.unpack("<f", bytes(buf))[0])


def _check_samples_device(
    *,
    acl,
    c_dev: int,
    a: np.ndarray,
    b_t: np.ndarray,
    samples: int,
    seed: int,
    rtol: float,
    atol: float,
    m_limit: int | None = None,
    n_limit: int | None = None,
) -> None:
    rng = np.random.default_rng(int(seed))
    m, k = a.shape
    n = b_t.shape[0]
    m_lim = int(m_limit) if m_limit is not None else int(m)
    n_lim = int(n_limit) if n_limit is not None else int(n)
    if m_lim <= 0 or m_lim > m:
        raise ValueError(f"invalid m_limit: {m_limit} (m={m})")
    if n_lim <= 0 or n_lim > n:
        raise ValueError(f"invalid n_limit: {n_limit} (n={n})")
    rs = rng.integers(0, m, size=(samples,), dtype=np.int64)
    cs = rng.integers(0, n, size=(samples,), dtype=np.int64)
    rs = rs % m_lim
    cs = cs % n_lim
    a32 = a.astype(np.float32, copy=False)
    b32 = b_t.astype(np.float32, copy=False)
    for r, col in zip(rs, cs):
        r_i = int(r)
        c_i = int(col)
        expected = float(np.dot(a32[r_i, :], b32[c_i, :]))
        offset = (r_i * n + c_i) * 4
        got = _read_device_f32(acl, dev_ptr=int(c_dev), offset_bytes=int(offset))
        if not np.isfinite(got):
            raise AssertionError(f"non-finite output at ({r_i},{c_i}): {got}")
        if not np.isclose(got, expected, rtol=float(rtol), atol=float(atol)):
            raise AssertionError(f"mismatch at ({r_i},{c_i}): got={got} expected={expected}")


def _benchmark_so(
    *,
    so_path: Path,
    device_id: int,
    block_dim: int,
    a: np.ndarray,
    b_t: np.ndarray,
    iters: int,
    warmup: int,
) -> tuple[float, float, float]:
    import ctypes
    import acl

    def _recent() -> str:
        try:
            return str(acl.get_recent_err_msg())
        except Exception:
            return ""

    def _check(ret: int, what: str) -> None:
        if int(ret) == 0:
            return
        msg = _recent()
        raise RuntimeError(f"{what} failed (ret={ret})" + (f": {msg}" if msg else ""))

    m, k = a.shape
    n = b_t.shape[0]
    c_nbytes = int(m) * int(n) * 4

    acl.init()
    acl.rt.set_device(int(device_id))
    stream, ret = acl.rt.create_stream()
    _check(ret, "acl.rt.create_stream")

    a_dev, ret = acl.rt.malloc(int(a.nbytes), 0)
    _check(ret, "acl.rt.malloc(a)")
    b_dev, ret = acl.rt.malloc(int(b_t.nbytes), 0)
    _check(ret, "acl.rt.malloc(b)")
    c_dev, ret = acl.rt.malloc(int(c_nbytes), 0)
    _check(ret, "acl.rt.malloc(c)")

    _check(acl.rt.memcpy(int(a_dev), int(a.nbytes), int(a.ctypes.data), int(a.nbytes), 1), "acl.rt.memcpy(a H2D)")
    _check(acl.rt.memcpy(int(b_dev), int(b_t.nbytes), int(b_t.ctypes.data), int(b_t.nbytes), 1), "acl.rt.memcpy(b H2D)")

    lib = ctypes.CDLL(str(so_path))
    launch = lib.ptoas_launch
    launch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    launch.restype = None

    def _launch():
        launch(
            ctypes.c_void_p(int(stream)),
            ctypes.c_uint32(int(block_dim)),
            ctypes.c_void_p(int(a_dev)),
            ctypes.c_void_p(int(b_dev)),
            ctypes.c_void_p(int(c_dev)),
        )

    for _ in range(int(warmup)):
        _launch()
    _check(acl.rt.synchronize_stream(stream), "acl.rt.synchronize_stream(warmup)")

    start, ret = acl.rt.create_event()
    _check(ret, "acl.rt.create_event(start)")
    end, ret = acl.rt.create_event()
    _check(ret, "acl.rt.create_event(end)")

    _check(acl.rt.record_event(start, stream), "acl.rt.record_event(start)")
    for _ in range(int(iters)):
        _launch()
    _check(acl.rt.record_event(end, stream), "acl.rt.record_event(end)")
    _check(acl.rt.synchronize_event(end), "acl.rt.synchronize_event(end)")
    elapsed_ms, ret = acl.rt.event_elapsed_time(start, end)
    _check(ret, "acl.rt.event_elapsed_time")

    avg_ms = float(elapsed_ms) / float(max(1, int(iters)))

    # Keep output on device for sampled validation.
    acl.rt.destroy_event(start)
    acl.rt.destroy_event(end)

    return avg_ms, int(c_dev), (lambda: (acl, stream, a_dev, b_dev, c_dev))()


def _cleanup_npu(acl, stream, a_dev, b_dev, c_dev, device_id: int):
    acl.rt.free(int(a_dev))
    acl.rt.free(int(b_dev))
    acl.rt.free(int(c_dev))
    acl.rt.destroy_stream(stream)
    acl.rt.reset_device(int(device_id))
    acl.finalize()


def main() -> int:
    ap = argparse.ArgumentParser(description="kernels/python/gemm_performance: PTO-AS + ptoas --insert-events + NPU TFLOPS.")
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas())
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/pto_kernel_python_gemm_performance"))

    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    ap.add_argument("--run-mode", choices=["npu", "sim"], default="npu")
    ap.add_argument("--soc", default="a3")
    ap.add_argument("--device", type=int, default=7)

    ap.add_argument("--m", type=int, default=6144)
    ap.add_argument("--n", type=int, default=6144)
    ap.add_argument("--k", type=int, default=6144)
    ap.add_argument("--grid-m", type=int, default=4)
    ap.add_argument("--grid-n", type=int, default=6)
    ap.add_argument("--block-dim", type=int, default=None, help="Launch blockDim (default: grid_m*grid_n)")
    ap.add_argument(
        "--allow-unaligned",
        action="store_true",
        help="Allow unaligned (m,n,k) by padding inputs to tiling multiples and validating only the original region.",
    )

    ap.add_argument("--emit-bin", action="store_true", help="Also emit *.bin via ptoas (slower; not needed to benchmark).")
    ap.add_argument(
        "--emit-exe",
        action="store_true",
        help="Also build a standalone executable (links fatobj; no dlopen .so) under the case outdir.",
    )
    ap.add_argument("--skip-build", action="store_true", help="Reuse existing built .so if present in outdir.")
    ap.add_argument("--compile-only", action="store_true", help="Only build artifacts, do not run the kernel.")
    ap.add_argument(
        "--dump-instr",
        action="store_true",
        help="(sim mode) Also run a generated host.cpp to produce *.instr_log.dump and summarize SET_FLAG/WAIT_FLAG.",
    )
    ap.add_argument("--dump-instr-only", action="store_true", help="(sim mode) Only run the host.cpp dump flow, skip Python launch.")

    ap.add_argument(
        "--camodel-log-path",
        type=Path,
        default=None,
        help="Simulator CAMODEL_LOG_PATH (enables simulator dumps when --run-mode=sim).",
    )
    ap.add_argument(
        "--enable-camodel-logs",
        action="store_true",
        help="(sim mode) Enable simulator dumps under <outdir>/<case>/camodel_logs (can be very slow).",
    )

    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--check", dest="check", action="store_true", help="Enable sampled output checks (default).")
    ap.add_argument("--no-check", dest="check", action="store_false", help="Disable sampled output checks.")
    ap.set_defaults(check=True)
    ap.add_argument("--check-samples", type=int, default=16)
    ap.add_argument("--check-rtol", type=float, default=2e-2)
    ap.add_argument("--check-atol", type=float, default=5e-2)
    args = ap.parse_args()

    if not args.ptoas.exists():
        print(f"error: ptoas not found: {args.ptoas}", file=sys.stderr)
        return 2
    if not args.ascend_home or not args.ascend_home.exists():
        print("error: set --ascend-home or ASCEND_HOME_PATH to your Ascend toolkit root", file=sys.stderr)
        return 2

    block_dim = int(args.block_dim) if args.block_dim is not None else int(args.grid_m) * int(args.grid_n)
    if block_dim <= 0:
        print("error: block_dim must be > 0", file=sys.stderr)
        return 2

    def _ceil_div(a: int, b: int) -> int:
        return (int(a) + int(b) - 1) // int(b)

    m_req = int(args.m)
    n_req = int(args.n)
    k_req = int(args.k)
    base_m = 128
    base_n = 256
    base_k = 64
    tile_m = int(args.grid_m) * int(base_m)
    tile_n = int(args.grid_n) * int(base_n)
    m_pad = _ceil_div(m_req, tile_m) * tile_m if args.allow_unaligned else m_req
    n_pad = _ceil_div(n_req, tile_n) * tile_n if args.allow_unaligned else n_req
    k_pad = _ceil_div(k_req, base_k) * base_k if args.allow_unaligned else k_req

    pad_tag = ""
    if args.allow_unaligned and (m_pad != m_req or n_pad != n_req or k_pad != k_req):
        pad_tag = f"_mp{m_pad}_np{n_pad}_kp{k_pad}"

    case_dir = args.outdir / (
        f"m{m_req}_n{n_req}_k{k_req}{pad_tag}_gm{int(args.grid_m)}_gn{int(args.grid_n)}_bd{block_dim}_{args.run_mode}"
    )
    case_dir.mkdir(parents=True, exist_ok=True)

    if args.run_mode == "sim" and (args.enable_camodel_logs or args.camodel_log_path is not None):
        # Must be set *before* ensure_ascend_sim_env re-execs.
        log_dir = args.camodel_log_path or (case_dir / "camodel_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CAMODEL_LOG_PATH"] = str(log_dir)
        os.environ["ASCEND_PROCESS_LOG_PATH"] = str(log_dir)

    if args.run_mode == "sim":
        soc_full = _soc_from_alias(str(args.soc))
        pipeline.ensure_ascend_sim_env(ascend_home=args.ascend_home, soc=soc_full)
        runtime_lib = "runtime_camodel"
    else:
        runtime_lib = "runtime"
        soc_full = None

    # Build PTO-AS from Python.
    spec = make_gemm_performance_kernel(
        m=int(m_pad),
        k=int(k_pad),
        n=int(n_pad),
        grid_m=int(args.grid_m),
        grid_n=int(args.grid_n),
    )
    pto_path = case_dir / "gemm_performance.pto"
    pto_text = prepend_host_spec_to_pto(pto=spec.pto, spec=binding.default_host_spec(spec))
    pto_path.write_text(pto_text, encoding="utf-8")

    # Compile via ptoas with --insert-events (set/wait flags insertion).
    cfg = pipeline.CompileConfig(
        ptoas=args.ptoas,
        ascend_home=args.ascend_home,
        arch="dav-c220-cube",
        memory_model="MEMORY_BASE",
        insert_events=True,
    )

    cce_path = case_dir / "gemm_performance.cpp"
    so_path = case_dir / f"libgemm_performance_{args.run_mode}.so"
    exe_path = case_dir / f"gemm_performance_{args.run_mode}"
    bin_path = case_dir / "gemm_performance.bin"

    if not (args.skip_build and so_path.exists()):
        # Emit CCE source (and optionally *.bin) first.
        pipeline.compile_pto_to_device_cpp(
            pto_path=pto_path,
            out_cpp=cce_path,
            ptoas=cfg.ptoas,
            arch=cfg.arch,
            memory_model=cfg.memory_model,
            insert_events=cfg.insert_events,
            assign_tile_addrs=True,
        )
        if args.emit_bin:
            pipeline.compile_pto_to_cce_and_bin(pto_path=pto_path, outdir=case_dir, cfg=cfg, out_cpp=cce_path, out_bin=bin_path)

        # Build the fatobj shared library used for launch.
        pipeline.build_fatobj_so_from_cce(
            cce_path=cce_path,
            out_so=so_path,
            arch=cfg.arch,
            ascend_home=cfg.ascend_home,
            fixed_block_dim=int(block_dim),
            runtime_lib=runtime_lib,
            soc=soc_full,
            cce_extra_flags=[
                "-mllvm",
                "-cce-aicore-stack-size=0x8000",
                "-mllvm",
                "-cce-aicore-function-stack-size=0x8000",
                "-mllvm",
                "-cce-aicore-record-overflow=true",
                "-mllvm",
                "-cce-aicore-addr-transform",
                "-mllvm",
                "-cce-aicore-dcci-insert-for-scalar=false",
            ],
        )

        if args.emit_exe:
            host_spec = pipeline.parse_or_default_host_spec(pto_text=pto_text)
            host_specs = [TensorSpec(dtype=a.dtype, shape=(int(a.shape[0]), int(a.shape[1]))) for a in host_spec.args]
            pipeline.build_fatobj_exe_from_cce(
                cce_path=cce_path,
                out_exe=exe_path,
                arch=cfg.arch,
                ascend_home=cfg.ascend_home,
                host_specs=host_specs,
                fixed_block_dim=int(block_dim),
                runtime_lib=runtime_lib,
                soc=soc_full,
                add_rpath=True,
                cce_extra_flags=[
                    "-mllvm",
                    "-cce-aicore-stack-size=0x8000",
                    "-mllvm",
                    "-cce-aicore-function-stack-size=0x8000",
                    "-mllvm",
                    "-cce-aicore-record-overflow=true",
                    "-mllvm",
                    "-cce-aicore-addr-transform",
                    "-mllvm",
                    "-cce-aicore-dcci-insert-for-scalar=false",
                ],
            )

    # Keep a quick summary of inserted set/wait flags for debugging.
    try:
        summary = pipeline.summarize_cce_events(cce_path=cce_path)
        import json

        (case_dir / "event_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        snippet = pipeline.extract_cce_set_wait_lines(cce_path=cce_path, limit=200)
        (case_dir / "set_wait_snippet.txt").write_text("\n".join(snippet) + ("\n" if snippet else ""), encoding="utf-8")
    except Exception:
        pass

    if args.compile_only:
        extra = f" exe={exe_path.name}" if args.emit_exe else ""
        print(f"OK: built so={so_path}{extra} outdir={case_dir}")
        return 0

    if args.run_mode == "sim" and args.dump_instr:
        host_spec = pipeline.parse_or_default_host_spec(pto_text=pto_text)
        host_specs = [TensorSpec(dtype=a.dtype, shape=(int(a.shape[0]), int(a.shape[1]))) for a in host_spec.args]
        dump_dir = _build_and_run_sim_dump_instr(
            outdir=case_dir,
            so_path=so_path,
            host_specs=host_specs,
            ascend_home=args.ascend_home,
            soc=_soc_from_alias(str(args.soc)),
            device=int(args.device),
            block_dim=int(block_dim),
        )
        s, w, samples = _summarize_camodel_set_wait_flags(log_dir=dump_dir)
        (case_dir / "host_camodel_set_wait_summary.txt").write_text(
            f"CAMODEL_LOG_PATH={dump_dir}\nSET_FLAG={s}\nWAIT_FLAG={w}\n"
            + ("\n".join(samples) + ("\n" if samples else "")),
            encoding="utf-8",
        )
        print(f"sim_dump: dir={dump_dir}  SET_FLAG={s}  WAIT_FLAG={w}")
        if args.dump_instr_only:
            return 0

    # Host inputs.
    rng = np.random.default_rng(19)
    a = np.zeros((int(m_pad), int(k_pad)), dtype=np.float16)
    a_i16 = rng.integers(-1000, 1000, size=(int(m_req), int(k_req)), dtype=np.int16)
    a[: int(m_req), : int(k_req)] = (a_i16.astype(np.float16) / np.float16(256.0)).astype(np.float16, copy=False)
    del a_i16

    # DN tensor is backed by a physical [n, k] row-major buffer (host passes B^T contiguous).
    b_t = np.zeros((int(n_pad), int(k_pad)), dtype=np.float16)
    b_t_i16 = rng.integers(-1000, 1000, size=(int(n_req), int(k_req)), dtype=np.int16)
    b_t[: int(n_req), : int(k_req)] = (b_t_i16.astype(np.float16) / np.float16(256.0)).astype(np.float16, copy=False)
    del b_t_i16

    # Benchmark + optional sampled validation.
    t0 = time.time()
    avg_ms, c_dev, state = _benchmark_so(
        so_path=so_path,
        device_id=int(args.device),
        block_dim=int(block_dim),
        a=a,
        b_t=b_t,
        iters=int(args.iters),
        warmup=int(args.warmup),
    )
    acl, stream, a_dev, b_dev, c_dev = state
    try:
        flops = 2.0 * float(args.m) * float(args.n) * float(args.k)
        tflops = flops / ((avg_ms / 1e3) * 1.0e12)
        print(f"avg_time_ms: {avg_ms:.4f}  tflops: {tflops:.2f}  (m={args.m} n={args.n} k={args.k})")

        if args.check:
            _check_samples_device(
                acl=acl,
                c_dev=int(c_dev),
                a=a,
                b_t=b_t,
                samples=int(args.check_samples),
                seed=20,
                rtol=float(args.check_rtol),
                atol=float(args.check_atol),
                m_limit=int(m_req),
                n_limit=int(n_req),
            )
            print(f"check: OK (samples={int(args.check_samples)})")
    finally:
        _cleanup_npu(acl, stream, a_dev, b_dev, c_dev, device_id=int(args.device))

    extra = f" bin={bin_path.name}" if args.emit_bin else ""
    print(f"OK: outdir={case_dir}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
