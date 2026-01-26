#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptoas.python import pipeline  # noqa: E402
from ptoas.python import binding  # noqa: E402
from ptoas.python.host_spec import HostSpec, prepend_host_spec_to_pto  # noqa: E402
from kernels.python.gemm_big.kernel import GemmConfig, make_gemm_f16f16f32_kernel  # noqa: E402


def _default_ptoas() -> Path:
    return _REPO_ROOT / "bin/ptoas"


def _soc_from_alias(alias: str) -> str:
    if alias == "a3":
        return "Ascend910B1"
    if alias == "a5":
        return "Ascend910_9599"
    return alias


def _check_samples(*, a: np.ndarray, b: np.ndarray, c: np.ndarray, samples: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    m, k = a.shape
    k2, n = b.shape
    assert k2 == k
    fixed: list[tuple[int, int]] = [
        (0, 0),
        (0, n - 1),
        (m - 1, 0),
        (m - 1, n - 1),
        (m // 4, n // 4),
        (3 * m // 4, 3 * n // 4),
    ]
    points = fixed[:]
    for _ in range(samples):
        points.append((int(rng.integers(0, m)), int(rng.integers(0, n))))

    for i, j in points:
        ref = (a[i, :].astype(np.float32) * b[:, j].astype(np.float32)).sum(dtype=np.float32)
        got = np.float32(c[i, j])
        np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-2, err_msg=f"mismatch at ({i},{j})")


def _benchmark_so(
    *,
    so_path: Path,
    host_spec: HostSpec,
    a: np.ndarray,
    b: np.ndarray,
    b_dev: np.ndarray,
    device_id: int,
    block_dim: int,
    iters: int,
    warmup: int,
    copy_back: bool,
) -> tuple[float, np.ndarray | None]:
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
    k2, n = b.shape
    assert k2 == k
    if b_dev.dtype != b.dtype or int(b_dev.nbytes) != int(b.nbytes):
        raise ValueError(f"b_dev must match b dtype/bytes (b={b.dtype}/{b.nbytes}, b_dev={b_dev.dtype}/{b_dev.nbytes})")

    c_host = np.empty((m, n), dtype=np.float32)

    acl.init()
    acl.rt.set_device(device_id)
    stream, ret = acl.rt.create_stream()
    _check(ret, "acl.rt.create_stream")

    # Host->device once.
    a_dev, ret = acl.rt.malloc(int(a.nbytes), 0)
    _check(ret, "acl.rt.malloc(a)")
    b_dev_ptr, ret = acl.rt.malloc(int(b_dev.nbytes), 0)
    _check(ret, "acl.rt.malloc(b)")
    c_dev, ret = acl.rt.malloc(int(c_host.nbytes), 0)
    _check(ret, "acl.rt.malloc(c)")

    _check(acl.rt.memcpy(int(a_dev), int(a.nbytes), int(a.ctypes.data), int(a.nbytes), 1), "acl.rt.memcpy(a H2D)")
    _check(
        acl.rt.memcpy(int(b_dev_ptr), int(b_dev.nbytes), int(b_dev.ctypes.data), int(b_dev.nbytes), 1),
        "acl.rt.memcpy(b H2D)",
    )

    lib = ctypes.CDLL(str(so_path))
    launch = lib.ptoas_launch
    launch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    launch.restype = None

    # Warmup.
    for _ in range(warmup):
        launch(
            ctypes.c_void_p(stream),
            int(block_dim),
            ctypes.c_void_p(int(a_dev)),
            ctypes.c_void_p(int(b_dev_ptr)),
            ctypes.c_void_p(int(c_dev)),
        )
    _check(acl.rt.synchronize_stream(stream), "acl.rt.synchronize_stream (warmup)")

    # Timing with events.
    start, ret = acl.rt.create_event()
    _check(ret, "acl.rt.create_event(start)")
    end, ret = acl.rt.create_event()
    _check(ret, "acl.rt.create_event(end)")

    _check(acl.rt.record_event(start, stream), "acl.rt.record_event(start)")
    for _ in range(iters):
        launch(
            ctypes.c_void_p(stream),
            int(block_dim),
            ctypes.c_void_p(int(a_dev)),
            ctypes.c_void_p(int(b_dev_ptr)),
            ctypes.c_void_p(int(c_dev)),
        )
    _check(acl.rt.record_event(end, stream), "acl.rt.record_event(end)")
    _check(acl.rt.synchronize_event(end), "acl.rt.synchronize_event(end)")
    elapsed_ms, ret = acl.rt.event_elapsed_time(start, end)
    _check(ret, "acl.rt.event_elapsed_time")
    avg_ms = float(elapsed_ms) / float(iters)

    out = None
    if copy_back:
        _check(acl.rt.memcpy(int(c_host.ctypes.data), int(c_host.nbytes), int(c_dev), int(c_host.nbytes), 2), "acl.rt.memcpy(c D2H)")
        out = c_host

    acl.rt.destroy_event(start)
    acl.rt.destroy_event(end)
    acl.rt.free(a_dev)
    acl.rt.free(b_dev_ptr)
    acl.rt.free(c_dev)
    acl.rt.destroy_stream(stream)
    acl.rt.reset_device(device_id)
    acl.finalize()
    return avg_ms, out


def main() -> int:
    ap = argparse.ArgumentParser(description="Big GEMM (Python AST -> PTO-AS -> ptoas -> NPU A3 run + TFLOPS).")
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas())
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    ap.add_argument("--run-mode", choices=["npu", "sim"], default="npu")
    ap.add_argument("--soc", default="a3", help="Simulator SoC (a3|a5|Ascend910B1|...) when --run-mode=sim")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/pto_kernel_python_gemm_big"))
    ap.add_argument("--no-insert-events", dest="insert_events", action="store_false", default=True)

    ap.add_argument("--m", type=int, default=8192)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--bm", type=int, default=128)
    ap.add_argument("--bn", type=int, default=128)
    ap.add_argument("--bk", type=int, default=64)
    ap.add_argument("--block-dim", type=int, default=0, help="Launch blockDim (default: (m/bm)*(n/bn))")

    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--no-check", dest="check", action="store_false", default=True)
    ap.add_argument("--samples", type=int, default=8, help="Sampled output elements to verify against CPU dot")
    ap.add_argument("--no-d2h", dest="d2h", action="store_false", default=True, help="Skip copying C back to host")
    args = ap.parse_args()

    if not args.ptoas.exists():
        raise SystemExit(f"error: ptoas not found: {args.ptoas}")
    if args.run_mode == "sim":
        if not args.ascend_home or not args.ascend_home.exists():
            raise SystemExit("error: set --ascend-home or ASCEND_HOME_PATH for simulator mode")
        soc_full = _soc_from_alias(args.soc)
        pipeline.ensure_ascend_sim_env(ascend_home=args.ascend_home, soc=soc_full)
        runtime_lib = "runtime_camodel"
    else:
        runtime_lib = "runtime"
        soc_full = None

    cfg = GemmConfig(m=args.m, n=args.n, k=args.k, bm=args.bm, bn=args.bn, bk=args.bk)
    spec = make_gemm_f16f16f32_kernel(cfg=cfg)

    args.outdir.mkdir(parents=True, exist_ok=True)
    pto_path = args.outdir / "gemm_big.pto"
    pto_text = prepend_host_spec_to_pto(pto=spec.pto, spec=binding.default_host_spec(spec))
    pto_path.write_text(pto_text, encoding="utf-8")

    # Compile NPU.
    if not args.ascend_home or not args.ascend_home.exists():
        raise SystemExit("error: set --ascend-home or ASCEND_HOME_PATH for NPU compilation")
    cfg_compile = pipeline.CompileConfig(
        ptoas=args.ptoas,
        ascend_home=args.ascend_home,
        arch="dav-c220-cube",
        memory_model="MEMORY_BASE",
        insert_events=args.insert_events,
    )
    cce_path, bin_path = pipeline.compile_pto_to_cce_and_bin(pto_path=pto_path, outdir=args.outdir, cfg=cfg_compile)
    so_path = args.outdir / "libgemm_big_npu.so"
    block_dim = args.block_dim or ((args.m // args.bm) * (args.n // args.bn))
    pipeline.build_fatobj_so_from_cce(
        cce_path=cce_path,
        out_so=so_path,
        arch=cfg_compile.arch,
        ascend_home=args.ascend_home,
        fixed_block_dim=int(block_dim),
        runtime_lib=runtime_lib,
        soc=soc_full,
    )
    print(f"built: {so_path}", file=sys.stderr, flush=True)

    # Initialize inputs (host).
    rng = np.random.default_rng(0)
    a = (rng.random((args.m, args.k), dtype=np.float32) - 0.5).astype(np.float16)
    b = (rng.random((args.k, args.n), dtype=np.float32) - 0.5).astype(np.float16)
    # Kernel expects B as DN: physical [n, k] row-major buffer.
    b_dev = np.ascontiguousarray(b.T)

    host_spec = pipeline.parse_or_default_host_spec(pto_text=pto_text)
    host_spec = type(host_spec)(
        args=host_spec.args, seed=host_spec.seed, block_dim=int(block_dim), kernel_name=host_spec.kernel_name
    )

    # Benchmark.
    print(
        f"running: device={args.device} blockDim={block_dim} iters={args.iters} warmup={args.warmup}",
        file=sys.stderr,
        flush=True,
    )
    t0 = time.time()
    avg_ms, c = _benchmark_so(
        so_path=so_path,
        host_spec=host_spec,
        a=a,
        b=b,
        b_dev=b_dev,
        device_id=args.device,
        block_dim=block_dim,
        iters=args.iters,
        warmup=args.warmup,
        copy_back=args.d2h,
    )
    wall_s = time.time() - t0

    flops = 2.0 * float(args.m) * float(args.n) * float(args.k)
    tflops = (flops / (avg_ms / 1000.0)) / 1e12

    print(f"OK: built {bin_path.name} and ran blockDim={block_dim} avg_kernel={avg_ms:.3f} ms ({tflops:.2f} TFLOPS)")
    if c is not None:
        print(f"OK: copied back C (dtype={c.dtype}, bytes={c.nbytes})")
        print(
            f"C stats: min={float(c.min()):.6g} max={float(c.max()):.6g} nnz={int(np.count_nonzero(c))}",
            file=sys.stderr,
            flush=True,
        )
        if args.check:
            _check_samples(a=a, b=b, c=c, samples=args.samples, seed=0)
    print(f"note: total script wall time {wall_s:.1f} s (includes H2D and optional D2H)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
