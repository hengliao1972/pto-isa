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
from ptoas.python.host_spec import HostSpec, HostTensorArg, prepend_host_spec_to_pto  # noqa: E402

from kernels.python.bgemm_performance.kernel import make_bgemm_performance_kernel  # noqa: E402


def _default_ptoas() -> Path:
    return _REPO_ROOT / "bin/ptoas"


def _soc_from_alias(alias: str) -> str:
    if alias == "a3":
        return "Ascend910B1"
    if alias == "a5":
        return "Ascend910_9599"
    return alias


def _benchmark_so(
    *,
    so_path: Path,
    device_id: int,
    block_dim: int,
    a: np.ndarray,
    b_t: np.ndarray,
    iters: int,
    warmup: int,
) -> tuple[float, int, tuple]:
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

    m_total, k = a.shape
    n = b_t.shape[0]
    c_nbytes = int(m_total) * int(n) * 4

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

    _check(acl.rt.memcpy(int(a_dev), int(a.nbytes), int(a.ctypes.data), int(a.nbytes), 1), "acl.rt.memcpy(H2D, a)")
    _check(
        acl.rt.memcpy(int(b_dev), int(b_t.nbytes), int(b_t.ctypes.data), int(b_t.nbytes), 1),
        "acl.rt.memcpy(H2D, b)",
    )

    lib = ctypes.CDLL(str(so_path))
    launch = lib.ptoas_launch
    launch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    launch.restype = None

    def _launch() -> None:
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

    acl.rt.destroy_event(start)
    acl.rt.destroy_event(end)

    return avg_ms, int(c_dev), (acl, stream, a_dev, b_dev, c_dev)


def _cleanup_npu(acl, stream, a_dev, b_dev, c_dev, device_id: int) -> None:
    acl.rt.free(int(a_dev))
    acl.rt.free(int(b_dev))
    acl.rt.free(int(c_dev))
    acl.rt.destroy_stream(stream)
    acl.rt.reset_device(int(device_id))
    acl.finalize()


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
    m_limit: int,
    n_limit: int,
    m_per_batch: int,
    k_stride: int,
) -> None:
    rng = np.random.default_rng(int(seed))
    m, k = a.shape
    n = b_t.shape[0]
    if m_limit <= 0 or m_limit > m:
        raise ValueError(f"invalid m_limit: {m_limit} (m={m})")
    if n_limit <= 0 or n_limit > n:
        raise ValueError(f"invalid n_limit: {n_limit} (n={n})")
    rs = rng.integers(0, m_limit, size=(samples,), dtype=np.int64)
    cs = rng.integers(0, n_limit, size=(samples,), dtype=np.int64)
    a32 = a.astype(np.float32, copy=False)
    b32 = b_t.astype(np.float32, copy=False)
    for r, col in zip(rs, cs):
        r_i = int(r)
        c_i = int(col)
        b_id = r_i // int(m_per_batch)
        b_off = int(b_id) * int(k_stride)
        expected = float(np.dot(a32[r_i, :], b32[c_i, b_off : b_off + int(k)]))
        offset = (r_i * n + c_i) * 4
        got = _read_device_f32(acl, dev_ptr=int(c_dev), offset_bytes=int(offset))
        if not np.isfinite(got):
            raise AssertionError(f"non-finite output at ({r_i},{c_i}): {got}")
        if not np.isclose(got, expected, rtol=float(rtol), atol=float(atol)):
            raise AssertionError(f"mismatch at ({r_i},{c_i}): got={got} expected={expected}")


def main() -> int:
    ap = argparse.ArgumentParser(description="kernels/python/bgemm_performance: PTO-AS + ptoas + NPU TFLOPS (BGEMM: per-batch A and B).")
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas())
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/pto_kernel_python_bgemm_performance"))

    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    ap.add_argument("--run-mode", choices=["npu", "sim"], default="npu")
    ap.add_argument("--soc", default="a3")
    ap.add_argument("--device", type=int, default=0)

    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--m", type=int, default=6144)
    ap.add_argument("--n", type=int, default=6144)
    ap.add_argument("--k", type=int, default=6144)
    ap.add_argument("--grid-m", type=int, default=4)
    ap.add_argument("--grid-n", type=int, default=6)
    ap.add_argument("--block-dim", type=int, default=None, help="Launch blockDim (default: batch*grid_m*grid_n)")
    ap.add_argument(
        "--allow-unaligned",
        action="store_true",
        help="Allow unaligned (m,n,k) by padding inputs to tiling multiples and validating only the original region.",
    )

    ap.add_argument("--skip-build", action="store_true", help="Reuse existing built .so if present in outdir.")
    ap.add_argument("--compile-only", action="store_true", help="Only build artifacts, do not run the kernel.")

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

    batch = int(args.batch)
    if batch <= 0:
        print("error: batch must be > 0", file=sys.stderr)
        return 2

    grid_m = int(args.grid_m)
    grid_n = int(args.grid_n)
    block_dim = int(args.block_dim) if args.block_dim is not None else batch * grid_m * grid_n
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

    tile_m = grid_m * base_m
    tile_n = grid_n * base_n

    m_pad = _ceil_div(m_req, tile_m) * tile_m if args.allow_unaligned else m_req
    n_pad = _ceil_div(n_req, tile_n) * tile_n if args.allow_unaligned else n_req
    k_pad = _ceil_div(k_req, base_k) * base_k if args.allow_unaligned else k_req

    pad_tag = ""
    if args.allow_unaligned and (m_pad != m_req or n_pad != n_req or k_pad != k_req):
        pad_tag = f"_mp{m_pad}_np{n_pad}_kp{k_pad}"

    case_dir = args.outdir / (
        f"b{batch}_m{m_req}_n{n_req}_k{k_req}{pad_tag}_gm{grid_m}_gn{grid_n}_bd{block_dim}_{args.run_mode}"
    )
    case_dir.mkdir(parents=True, exist_ok=True)

    if args.run_mode == "sim":
        soc_full = _soc_from_alias(str(args.soc))
        pipeline.ensure_ascend_sim_env(ascend_home=args.ascend_home, soc=soc_full)
        runtime_lib = "runtime_camodel"
    else:
        runtime_lib = "runtime"
        soc_full = None

    # Build PTO-AS from Python.
    spec = make_bgemm_performance_kernel(
        batch=int(batch),
        m=int(m_pad),
        k=int(k_pad),
        n=int(n_pad),
        grid_m=int(grid_m),
        grid_n=int(grid_n),
    )
    pto_path = case_dir / "bgemm_performance.pto"
    # Embed a host spec with the padded shapes (used mainly for metadata/debugging).
    hs = binding.default_host_spec(spec)
    host_spec = HostSpec(
        args=(
            HostTensorArg(dtype="f16", shape=(int(batch) * int(m_pad), int(k_pad)), role="in"),
            HostTensorArg(dtype="f16", shape=(int(k_pad), int(n_pad)), role="in", layout="DN", stride=(1, int(k_pad))),
            HostTensorArg(dtype="f32", shape=(int(batch) * int(m_pad), int(n_pad)), role="out"),
        ),
        seed=0,
        block_dim=1,
        kernel_name=hs.kernel_name,
    )
    pto_text = prepend_host_spec_to_pto(pto=spec.pto, spec=host_spec)
    pto_path.write_text(pto_text, encoding="utf-8")

    # Compile via ptoas with --insert-events.
    cce_path = case_dir / "bgemm_performance.cpp"
    so_path = case_dir / f"libbgemm_performance_{args.run_mode}.so"
    if not (args.skip_build and so_path.exists()):
        pipeline.compile_pto_to_device_cpp(
            pto_path=pto_path,
            out_cpp=cce_path,
            ptoas=args.ptoas,
            arch="dav-c220-cube",
            memory_model="MEMORY_BASE",
            insert_events=True,
            assign_tile_addrs=True,
        )
        pipeline.build_fatobj_so_from_cce(
            cce_path=cce_path,
            out_so=so_path,
            arch="dav-c220-cube",
            ascend_home=args.ascend_home,
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

    if args.compile_only:
        print(f"OK: built so={so_path} outdir={case_dir}")
        return 0

    # Host inputs.
    rng = np.random.default_rng(19)
    a = np.zeros((int(batch) * int(m_pad), int(k_pad)), dtype=np.float16)
    a_i16 = rng.integers(-1000, 1000, size=(int(batch) * int(m_req), int(k_req)), dtype=np.int16)
    a[: int(batch) * int(m_req), : int(k_req)] = (a_i16.astype(np.float16) / np.float16(256.0)).astype(np.float16, copy=False)
    del a_i16

    # DN tensor is backed by a physical [n, batch*k] row-major buffer (host passes B^T contiguous per batch).
    k_total_pad = int(batch) * int(k_pad)
    b_t = np.zeros((int(n_pad), k_total_pad), dtype=np.float16)
    for b in range(int(batch)):
        b_t_i16 = rng.integers(-1000, 1000, size=(int(n_req), int(k_req)), dtype=np.int16)
        b_off = int(b) * int(k_pad)
        b_t[: int(n_req), b_off : b_off + int(k_req)] = (b_t_i16.astype(np.float16) / np.float16(256.0)).astype(
            np.float16, copy=False
        )
        del b_t_i16

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
        flops = 2.0 * float(batch) * float(m_req) * float(n_req) * float(k_req)
        tflops = flops / ((avg_ms / 1e3) * 1.0e12)
        wall = time.time() - t0
        print(f"avg_time_ms: {avg_ms:.4f}  tflops: {tflops:.2f}  (batch={batch} m={m_req} n={n_req} k={k_req})")
        print(f"wall_s: {wall:.2f}  block_dim: {block_dim} (grid={grid_m}x{grid_n})")

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
                m_limit=int(batch) * int(m_req),
                n_limit=int(n_req),
                m_per_batch=int(m_req),
                k_stride=int(k_pad),
            )
            print(f"check: OK (samples={int(args.check_samples)})")
    finally:
        _cleanup_npu(acl, stream, a_dev, b_dev, c_dev, device_id=int(args.device))

    print(f"OK: outdir={case_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
