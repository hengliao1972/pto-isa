#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kernels.python.bgemm_performance.kernel import make_bgemm_performance_kernel  # noqa: E402


def _stage(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def _kv(key: str, value: object, *, indent: int = 2) -> None:
    pad = " " * int(indent)
    print(f"{pad}{key}: {value}", flush=True)


def _fmt_bytes(n: int) -> str:
    v = float(int(n))
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if v < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(v)} {unit}"
            return f"{v:.2f} {unit}"
        v /= 1024.0
    return f"{int(n)} B"


def _ensure_ascend_home_env(p: Path) -> Path:
    p = Path(p).resolve()
    os.environ["ASCEND_HOME_PATH"] = os.fspath(p)
    return p


def _read_device_f32(runner, *, dev_ptr: int, offset_bytes: int) -> float:
    buf = np.zeros((1,), dtype=np.float32)
    rc = int(runner.copy_from_device(buf, int(dev_ptr) + int(offset_bytes)))
    if rc != 0:
        raise RuntimeError(f"copy_from_device(4B) failed (rc={rc})")
    return float(buf[0])


def _check_samples(
    *,
    runner,
    c_dev: int,
    a: np.ndarray,
    b_t: np.ndarray,
    samples: int,
    seed: int,
    rtol: float,
    atol: float,
    batch: int,
    m_req: int,
    n_req: int,
    k_req: int,
    m_pad: int,
    k_pad: int,
) -> None:
    rng = np.random.default_rng(int(seed))
    rs = rng.integers(0, int(batch) * int(m_req), size=(samples,), dtype=np.int64)
    cs = rng.integers(0, int(n_req), size=(samples,), dtype=np.int64)
    a32 = a.astype(np.float32, copy=False)
    b32 = b_t.astype(np.float32, copy=False)
    for r, col in zip(rs, cs):
        r_i = int(r)
        c_i = int(col)
        b_id = r_i // int(m_pad)
        b_off = int(b_id) * int(k_pad)
        expected = float(np.dot(a32[r_i, : int(k_req)], b32[c_i, b_off : b_off + int(k_req)]))
        offset = (r_i * int(b_t.shape[0]) + c_i) * 4  # C is [batch*m_pad, n_pad] f32
        got = _read_device_f32(runner, dev_ptr=int(c_dev), offset_bytes=int(offset))
        if not np.isfinite(got):
            raise AssertionError(f"non-finite output at ({r_i},{c_i}): {got}")
        if not np.isclose(got, expected, rtol=float(rtol), atol=float(atol)):
            raise AssertionError(f"mismatch at ({r_i},{c_i}): got={got} expected={expected}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="BGEMM via runtime (PTO-AS → ptoas → runtime compile/load → task graph on Ascend NPU)."
    )
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/pto_bgemm_runtime"))
    ap.add_argument("--ptoas", type=Path, default=_REPO_ROOT / "bin" / "ptoas")
    ap.add_argument(
        "--ascend-home",
        type=Path,
        default=Path(os.environ.get("ASCEND_HOME_PATH", "")) if os.environ.get("ASCEND_HOME_PATH") else Path.home() / "Ascend" / "ascend-toolkit" / "latest",
    )
    ap.add_argument("--device", type=int, default=int(os.environ.get("PTO_DEVICE", "0")))
    ap.add_argument("--aic-blocks", type=int, default=int(os.environ.get("PTO_AIC_BLOCKS", "24")))
    ap.add_argument(
        "--aicpu-so",
        type=Path,
        default=None,
        help="Optional prebuilt AICPU scheduler .so. If omitted, pto_runtime builds it via ref_runtime/python/binary_compiler.py.",
    )
    ap.add_argument(
        "--aicore-kernel",
        type=Path,
        default=None,
        help="Optional prebuilt AICore worker kernel .o. If omitted, pto_runtime builds it via ref_runtime/python/binary_compiler.py.",
    )

    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--m", type=int, default=6144)
    ap.add_argument("--n", type=int, default=6144)
    ap.add_argument("--k", type=int, default=6144)
    ap.add_argument("--grid-m", type=int, default=4)
    ap.add_argument("--grid-n", type=int, default=6)
    ap.add_argument("--allow-unaligned", action="store_true")

    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--check", dest="check", action="store_true")
    ap.add_argument("--no-check", dest="check", action="store_false")
    ap.set_defaults(check=True)
    ap.add_argument("--check-samples", type=int, default=16)
    ap.add_argument("--check-rtol", type=float, default=2e-2)
    ap.add_argument("--check-atol", type=float, default=5e-2)

    ap.add_argument("--trace-json", type=Path, default=None, help="Write Chrome trace JSON from one profiled run.")
    ap.add_argument("--trace-svg", type=Path, default=None, help="Write a simple swimlane SVG from one profiled run.")
    default_tick_hz = float(os.environ.get("PTO_TRACE_TICK_HZ", "20000000"))  # sys_cnt is often 20MHz on Ascend
    ap.add_argument(
        "--trace-tick-hz",
        type=float,
        default=default_tick_hz,
        help="Clock frequency to scale device ticks to microseconds in trace output (default: $PTO_TRACE_TICK_HZ or 20_000_000).",
    )
    ap.add_argument(
        "--cpu-sim-verify",
        action="store_true",
        help="Compile the generated PTO-AS to CPU via ptoas and compare CPU vs NPU outputs (uses a small config if the current one is too large).",
    )
    args = ap.parse_args()

    try:
        import pto_runtime  # type: ignore
    except Exception as exc:
        print(f"error: cannot import pto_runtime: {exc}", file=sys.stderr)
        print("  expected repo root to contain `pto_runtime.py`", file=sys.stderr)
        return 2

    sys.path.insert(0, os.fspath(_REPO_ROOT))
    from pto.runtime import PtoasConfig, compile_and_load_kernel_from_pto

    _stage("Config")
    _kv("repo_root", _REPO_ROOT)
    _kv("outdir", Path(args.outdir).resolve())
    _kv("device", int(args.device))
    _kv("aic_blocks", int(args.aic_blocks))
    _kv("ascend_home", _ensure_ascend_home_env(Path(args.ascend_home)))
    _kv("ptoas", Path(args.ptoas).resolve())

    batch = int(args.batch)
    grid_m = int(args.grid_m)
    grid_n = int(args.grid_n)
    if batch <= 0 or grid_m <= 0 or grid_n <= 0:
        raise ValueError("batch/grid must be > 0")

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

    block_dim = int(batch) * int(grid_m) * int(grid_n)
    blocks_per_batch = int(grid_m) * int(grid_n)
    tile_m_per_task = int(m_pad) // int(grid_m)
    tile_n_per_task = int(n_pad) // int(grid_n)
    waves = (int(block_dim) + int(args.aic_blocks) - 1) // int(args.aic_blocks) if int(args.aic_blocks) > 0 else 0

    _stage("Problem Size")
    _kv("BGEMM", f"batch={batch}  M={m_req}  N={n_req}  K={k_req}")
    if (m_pad, n_pad, k_pad) != (m_req, n_req, k_req):
        _kv("padded", f"M={m_pad}  N={n_pad}  K={k_pad}  (use --allow-unaligned)")
    _kv("grid (MN)", f"{grid_m} x {grid_n}  (blocks_per_batch={blocks_per_batch})")
    _kv("task tile", f"{tile_m_per_task} x {tile_n_per_task} output tile per task (full K)")
    _kv("tasks", f"{block_dim} (= batch * grid_m * grid_n)")
    _kv("scheduler", f"aic_blocks={int(args.aic_blocks)} → ~{waves} wave(s) for {block_dim} task(s)")

    # Build PTO-AS kernel spec from Python.
    _stage("Build PTO-AS (Python frontend)")
    spec = make_bgemm_performance_kernel(
        batch=int(batch),
        m=int(m_pad),
        k=int(k_pad),
        n=int(n_pad),
        grid_m=int(grid_m),
        grid_n=int(grid_n),
        base_m=int(base_m),
        base_k=int(base_k),
        base_n=int(base_n),
    )

    out_dir = Path(args.outdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Runtime build artifacts.
    _stage("Init runtime")
    aicpu_so = Path(args.aicpu_so).resolve() if args.aicpu_so else None
    aicore_kernel = Path(args.aicore_kernel).resolve() if args.aicore_kernel else None
    _kv("ASCEND_HOME_PATH", os.environ.get("ASCEND_HOME_PATH", ""))
    _kv("aicpu_so", aicpu_so if aicpu_so else "(build via BinaryCompiler)")
    _kv("aicore_kernel", aicore_kernel if aicore_kernel else "(build via BinaryCompiler)")

    runner = pto_runtime.DeviceRunner.get()
    try:
        rc = int(
            runner.init(
                int(args.device),
                int(args.aic_blocks),
                os.fspath(aicpu_so) if aicpu_so else None,
                os.fspath(aicore_kernel) if aicore_kernel else None,
            )
        )
    except Exception as exc:
        print(f"error: DeviceRunner.init failed: {exc}", file=sys.stderr)
        return 2
    if rc != 0:
        print(f"error: DeviceRunner.init failed: {rc}", file=sys.stderr)
        return rc

    # Compile PTO-AS → CCE C++ via ptoas, then compile+load via runtime.
    _stage("Compile & load (ptoas → CCE C++ → runtime)")
    cfg = PtoasConfig(
        ptoas=Path(args.ptoas),
        arch="dav-c220-cube",
        memory_model="MEMORY_BASE",
        kernel_abi="mpmd",
        insert_events=True,
        assign_tile_addrs=True,
        ascend_home=Path(args.ascend_home),
        repo_root=_REPO_ROOT,
        log_path=out_dir / "ptoas.log",
        print_cmd=True,
    )
    out_cpp = compile_and_load_kernel_from_pto(
        runner=runner,
        func_id=0,
        pto=spec,
        out_dir=out_dir,
        pto_isa_root=_REPO_ROOT,
        ptoas_cfg=cfg,
    )
    _kv("pto_path", out_dir / "kernel_0.pto")
    _kv("cce_cpp", out_cpp)
    _kv("ptoas_log", out_dir / "ptoas.log")

    # Host inputs.
    _stage("Allocate & upload tensors")
    rng = np.random.default_rng(19)
    a = np.zeros((int(batch) * int(m_pad), int(k_pad)), dtype=np.float16)
    a_i16 = rng.integers(-1000, 1000, size=(int(batch) * int(m_req), int(k_req)), dtype=np.int16)
    a[: int(batch) * int(m_req), : int(k_req)] = (a_i16.astype(np.float16) / np.float16(256.0)).astype(
        np.float16, copy=False
    )
    del a_i16

    # DN tensor is backed by a physical [n, batch*k] row-major buffer (host passes B^T contiguous per batch).
    b_t = np.zeros((int(n_pad), int(batch) * int(k_pad)), dtype=np.float16)
    for b in range(int(batch)):
        b_t_i16 = rng.integers(-1000, 1000, size=(int(n_req), int(k_req)), dtype=np.int16)
        b_off = int(b) * int(k_pad)
        b_t[: int(n_req), b_off : b_off + int(k_req)] = (b_t_i16.astype(np.float16) / np.float16(256.0)).astype(
            np.float16, copy=False
        )
        del b_t_i16

    c = np.zeros((int(batch) * int(m_pad), int(n_pad)), dtype=np.float32)

    _kv("A", f"shape={tuple(a.shape)} dtype={a.dtype} bytes={_fmt_bytes(int(a.nbytes))}")
    _kv("B^T", f"shape={tuple(b_t.shape)} dtype={b_t.dtype} bytes={_fmt_bytes(int(b_t.nbytes))} (DN physical)")
    _kv("C", f"shape={tuple(c.shape)} dtype={c.dtype} bytes={_fmt_bytes(int(c.nbytes))}")

    dev_a = int(runner.allocate_tensor(int(a.nbytes)))
    dev_b = int(runner.allocate_tensor(int(b_t.nbytes)))
    dev_c = int(runner.allocate_tensor(int(c.nbytes)))
    if not (dev_a and dev_b and dev_c):
        print("error: allocate_tensor failed", file=sys.stderr)
        runner.finalize()
        return 1

    try:
        runner.set_profile_enabled(False)
        rc = int(runner.copy_to_device(dev_a, a))
        if rc != 0:
            raise RuntimeError(f"copy_to_device(A) failed: rc={rc}")
        rc = int(runner.copy_to_device(dev_b, b_t))
        if rc != 0:
            raise RuntimeError(f"copy_to_device(B) failed: rc={rc}")

        # Graph: one task per tile (batch * grid_m * grid_n).
        _stage("Build task graph")
        _kv("graph", f"Graph() + add_task() for tile_id in [0,{block_dim})")
        _kv("task args", "[task_id, dev_A, dev_B, dev_C]  (task_id == tile_id)")
        _kv("mapping", "batch = task_id // (grid_m*grid_n), m_core = (task_id%blocks_per_batch)%grid_m, n_core = (task_id%blocks_per_batch)//grid_m")
        _kv("note", f"this partitions M,N across cores; each task reduces full K ({k_pad} / {base_k} tiles)")
        graph = pto_runtime.Graph()
        for tile_id in range(block_dim):
            graph.add_task([int(tile_id), dev_a, dev_b, dev_c], func_id=0, core_type=1)

        if int(args.warmup) > 0:
            _stage(f"Warmup ({int(args.warmup)} run(s))")
            for _ in range(int(args.warmup)):
                rc = int(runner.run(graph, 1))
                if rc != 0:
                    raise RuntimeError(f"warmup run failed: rc={rc}")

        _stage("Benchmark")
        t0 = time.perf_counter()
        for _ in range(int(args.iters)):
            rc = int(runner.run(graph, 1))
            if rc != 0:
                raise RuntimeError(f"run failed: rc={rc}")
        t1 = time.perf_counter()

        avg_s = (t1 - t0) / float(max(1, int(args.iters)))
        avg_ms = avg_s * 1e3
        flops_req = 2.0 * float(batch) * float(m_req) * float(n_req) * float(k_req)
        flops_exec = 2.0 * float(batch) * float(m_pad) * float(n_pad) * float(k_pad)
        tflops_req = flops_req / (avg_s * 1.0e12)
        tflops_exec = flops_exec / (avg_s * 1.0e12)
        _kv("avg_time_ms", f"{avg_ms:.4f}  (iters={int(args.iters)})")
        _kv("throughput", f"{tflops_req:.2f} TFLOPS (logical)  |  {tflops_exec:.2f} TFLOPS (executed)")
        _kv("shape", f"batch={batch} m={m_req} n={n_req} k={k_req}  (padded m={m_pad} n={n_pad} k={k_pad})")
        _kv("grid", f"{grid_m}x{grid_n}  tasks={block_dim}  base={base_m}x{base_n}x{base_k}  block_dim={block_dim}")

        def _write_trace(*, profile, out_json: Path | None, out_svg: Path | None, tick_hz: float, title: str) -> None:
            if not profile:
                raise RuntimeError("empty profile; did you enable profiling?")

            # Normalize to the first task's time for nicer traces.
            t0 = min(int(p.start_time) for p in profile if int(p.start_time) != 0)

            def _to_us(ticks: int) -> int:
                if tick_hz and tick_hz > 0:
                    return int((float(ticks) * 1.0e6) / float(tick_hz))
                return int(ticks)

            # Task table.
            if out_json is not None:
                out_json.parent.mkdir(parents=True, exist_ok=True)
                rows = []
                for p in profile:
                    start = int(p.start_time)
                    end = int(p.end_time)
                    dur = max(0, end - start)
                    rows.append(
                        {
                            "task_id": int(p.task_id),
                            "func_id": int(p.func_id),
                            "core_type": int(p.core_type),
                            "exec_core_id": int(p.exec_core_id),
                            "exec_core_type": int(p.exec_core_type),
                            "exec_phys_core_id": int(p.exec_phys_core_id),
                            "start_time": start,
                            "end_time": end,
                            "duration": dur,
                            "pmu_cnt": [int(x) for x in p.pmu_cnt],
                        }
                    )
                (out_json.parent / "task_profile.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

            # Chrome trace.
            if out_json is not None:
                events = []
                events.append(
                    {
                        "ph": "M",
                        "name": "pto_runtime_meta",
                        "pid": 0,
                        "args": {
                            "tick_hz": float(tick_hz),
                            "time_unit": "us" if tick_hz and tick_hz > 0 else "ticks_as_us",
                        },
                    }
                )
                for p in profile:
                    start = int(p.start_time) - t0
                    end = int(p.end_time) - t0
                    dur = max(0, end - start)
                    events.append(
                        {
                            "name": f"task_{int(p.task_id)}",
                            "cat": "pto",
                            "ph": "X",
                            "pid": 0,
                            "tid": int(p.exec_core_id),
                            "ts": _to_us(start),
                            "dur": _to_us(dur),
                            "args": {
                                "func_id": int(p.func_id),
                                "core_type": int(p.core_type),
                                "exec_core_type": int(p.exec_core_type),
                                "exec_phys_core_id": int(p.exec_phys_core_id),
                                "pmu_cnt0": int(p.pmu_cnt[0]),
                            },
                        }
                    )
                trace = {
                    "displayTimeUnit": "us",
                    "traceEvents": events,
                }
                out_json.write_text(json.dumps(trace), encoding="utf-8")

            # Simple swimlane SVG.
            if out_svg is not None:
                out_svg.parent.mkdir(parents=True, exist_ok=True)
                # Lanes only for cores that actually executed tasks.
                lanes = sorted({int(p.exec_core_id) for p in profile})
                lane_index = {cid: i for i, cid in enumerate(lanes)}
                lane_h = 18
                pad_x = 160
                pad_top = 52
                pad_bottom = 46
                pad_right = 20
                width = 1400
                height = pad_top + lane_h * len(lanes) + pad_bottom

                starts = [int(p.start_time) for p in profile]
                ends = [int(p.end_time) for p in profile]
                t_min = min(starts)
                t_max = max(ends)
                span_ticks = max(1, t_max - t_min)
                span_us = max(1, _to_us(span_ticks))

                def _x_from_us(us: int) -> float:
                    return pad_x + (float(us) / float(span_us)) * float(width - pad_x - pad_right)

                def _color(task_id: int) -> str:
                    # Colorful but deterministic (golden-angle hue spacing).
                    hue = (float(task_id) * 137.508) % 360.0
                    return f"hsl({hue:.3f}, 75%, 55%)"

                parts = [
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
                    '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
                    f'<text x="10" y="18" font-family="monospace" font-size="13">{title}</text>',
                    f'<text x="10" y="36" font-family="monospace" font-size="11">time unit: us (tick_hz={tick_hz:.3f})</text>',
                ]

                # X-axis (microseconds) + grid.
                axis_y = pad_top + lane_h * len(lanes) + 10
                parts.append(f'<line x1="{pad_x}" y1="{axis_y}" x2="{width - pad_right}" y2="{axis_y}" stroke="#222222"/>')
                parts.append(f'<text x="{pad_x}" y="{axis_y + 28}" font-family="monospace" font-size="11">time (us)</text>')

                def _nice_step(span: int, target: int) -> int:
                    # Pick a "nice" tick step: 1/2/5 * 10^k.
                    if span <= 0:
                        return 1
                    raw = float(span) / float(max(1, target))
                    base = 10 ** int(np.floor(np.log10(raw))) if raw > 0 else 1
                    for m in (1, 2, 5, 10):
                        step = int(base * m)
                        if step >= raw:
                            return max(1, step)
                    return max(1, int(base * 10))

                tick_step = _nice_step(span_us, target=max(4, int((width - pad_x - pad_right) / 120)))
                tick = 0
                while tick <= span_us:
                    x = _x_from_us(tick)
                    # vertical grid line across lanes
                    parts.append(
                        f'<line x1="{x:.2f}" y1="{pad_top - 6}" x2="{x:.2f}" y2="{axis_y}" stroke="#eeeeee"/>'
                    )
                    # tick mark
                    parts.append(f'<line x1="{x:.2f}" y1="{axis_y}" x2="{x:.2f}" y2="{axis_y + 4}" stroke="#222222"/>')
                    parts.append(
                        f'<text x="{x:.2f}" y="{axis_y + 18}" text-anchor="middle" font-family="monospace" font-size="10">{tick}</text>'
                    )
                    tick += tick_step

                for cid in lanes:
                    y = pad_top + lane_index[cid] * lane_h + 12
                    parts.append(
                        f'<text x="10" y="{y}" font-family="monospace" font-size="10">core {cid}</text>'
                    )
                    parts.append(
                        f'<line x1="{pad_x}" y1="{pad_top + lane_index[cid] * lane_h + 8}" '
                        f'x2="{width - pad_right}" y2="{pad_top + lane_index[cid] * lane_h + 8}" stroke="#f3f3f3"/>'
                    )

                for p in profile:
                    start_us = _to_us(int(p.start_time) - t_min)
                    end_us = _to_us(int(p.end_time) - t_min)
                    x0 = _x_from_us(start_us)
                    x1 = _x_from_us(end_us)
                    w = max(1.0, x1 - x0)
                    y = pad_top + lane_index[int(p.exec_core_id)] * lane_h
                    tid = int(p.task_id)
                    fill = _color(tid)
                    parts.append(
                        f'<rect x="{x0:.2f}" y="{y:.2f}" width="{w:.2f}" height="{lane_h - 2}" '
                        f'fill="{fill}" opacity="0.88">'
                        f'<title>task {tid} on core {int(p.exec_core_id)}: start={start_us}us dur={max(0, end_us-start_us)}us</title>'
                        f'</rect>'
                    )
                    # Label the bar with task id if there is room.
                    if w >= 18:
                        parts.append(
                            f'<text x="{x0 + w/2:.2f}" y="{y + lane_h - 5:.2f}" text-anchor="middle" '
                            f'font-family="monospace" font-size="9" fill="#111111" stroke="#ffffff" stroke-width="0.8" '
                            f'paint-order="stroke">t{tid}</text>'
                        )
                parts.append("</svg>")
                out_svg.write_text("\n".join(parts), encoding="utf-8")

        if args.trace_json or args.trace_svg:
            # Profile one representative run (keep timing runs clean).
            _stage("Profile & trace export")
            runner.set_profile_enabled(True)
            rc = int(runner.run(graph, 1))
            if rc != 0:
                raise RuntimeError(f"profile run failed: rc={rc}")
            profile = runner.get_last_profile()
            _write_trace(
                profile=profile,
                out_json=args.trace_json.resolve() if args.trace_json else None,
                out_svg=args.trace_svg.resolve() if args.trace_svg else None,
                tick_hz=float(args.trace_tick_hz),
                title=f"BGEMM trace (batch={batch} m={m_req} n={n_req} k={k_req}, grid={grid_m}x{grid_n}, block_dim={block_dim})",
            )
            # Core-level summary (PMU-like cycles from end-start ticks).
            per_core = {}
            for p in profile:
                dur = max(0, int(p.end_time) - int(p.start_time))
                per_core.setdefault(int(p.exec_core_id), 0)
                per_core[int(p.exec_core_id)] += dur
            top = sorted(per_core.items(), key=lambda kv: kv[1], reverse=True)[:10]
            print("profile: top cores by total ticks:", ", ".join(f"{cid}:{ticks}" for cid, ticks in top))
            if args.trace_json:
                print(f"profile: wrote {args.trace_json.resolve()} and {args.trace_json.resolve().parent / 'task_profile.json'}")
            if args.trace_svg:
                print(f"profile: wrote {args.trace_svg.resolve()}")
            runner.set_profile_enabled(False)

        if args.check:
            _stage("Correctness check (random samples)")
            _check_samples(
                runner=runner,
                c_dev=dev_c,
                a=a,
                b_t=b_t,
                samples=int(args.check_samples),
                seed=20,
                rtol=float(args.check_rtol),
                atol=float(args.check_atol),
                batch=int(batch),
                m_req=int(m_req),
                n_req=int(n_req),
                k_req=int(k_req),
                m_pad=int(m_pad),
                k_pad=int(k_pad),
            )
            print(f"check: OK (samples={int(args.check_samples)})")

        if args.cpu_sim_verify:
            _stage("CPU simulator verify (ptoas --target=cpu)")
            # CPU simulation is intended for functional verification; keep it small by default.
            too_big = flops_req > 2.0e8
            vb = int(batch) if not too_big else 1
            vgm = int(grid_m) if not too_big else 1
            vgn = int(grid_n) if not too_big else 1
            vm = int(m_pad) if not too_big else 128
            vn = int(n_pad) if not too_big else 256
            vk = int(k_pad) if not too_big else 64
            if too_big:
                print(
                    f"cpu-sim: current config is too large for CPU simulation (flops≈{flops_req:.2e}); "
                    f"verifying on a small config instead: batch={vb} m={vm} n={vn} k={vk} grid={vgm}x{vgn}"
                )

            # Build a separate kernel for verification (func_id=1) to avoid clobbering the perf kernel.
            verify_dir = out_dir / "cpu_sim_verify"
            verify_dir.mkdir(parents=True, exist_ok=True)
            spec_v = make_bgemm_performance_kernel(
                batch=vb, m=vm, k=vk, n=vn, grid_m=vgm, grid_n=vgn, base_m=base_m, base_k=base_k, base_n=base_n
            )
            cfg_v = PtoasConfig(
                ptoas=Path(args.ptoas),
                arch="dav-c220-cube",
                memory_model="MEMORY_BASE",
                kernel_abi="mpmd",
                insert_events=True,
                assign_tile_addrs=True,
                ascend_home=Path(args.ascend_home),
                repo_root=_REPO_ROOT,
                log_path=verify_dir / "ptoas_verify.log",
                print_cmd=True,
            )
            compile_and_load_kernel_from_pto(
                runner=runner,
                func_id=1,
                pto=spec_v,
                out_dir=verify_dir,
                pto_isa_root=_REPO_ROOT,
                ptoas_cfg=cfg_v,
            )

            # CPU: compile PTO-AS -> C++ -> .so and execute.
            from ptoas.python import pipeline as _pl  # noqa: E402

            pto_path_v = verify_dir / "kernel_1.pto"
            pto_text_v = pto_path_v.read_text(encoding="utf-8")
            host_spec = _pl.parse_or_default_host_spec(pto_text=pto_text_v)
            block_dim_v = int(vb) * int(vgm) * int(vgn)
            host_spec = type(host_spec)(
                args=host_spec.args, seed=getattr(host_spec, "seed", 0), block_dim=block_dim_v, kernel_name=host_spec.kernel_name
            )
            base_arrays = _pl.make_host_arrays(host_spec)
            cpu_cpp = _pl.compile_pto_to_cpu_cpp(pto_path=pto_path_v, outdir=verify_dir, ptoas=Path(args.ptoas))
            cpu_so = verify_dir / "libbgemm_cpu.so"
            _pl.build_cpu_so_from_cpp(cpp_path=cpu_cpp, out_so=cpu_so)
            cpu_out = _pl.run_cpu_kernel_from_so(
                so_path=cpu_so, host_spec=host_spec, host_arrays=[a.copy() for a in base_arrays]
            )

            # NPU runtime: run the same kernel and compare outputs.
            dev_a_v = int(runner.allocate_tensor(int(base_arrays[0].nbytes)))
            dev_b_v = int(runner.allocate_tensor(int(base_arrays[1].nbytes)))
            dev_c_v = int(runner.allocate_tensor(int(base_arrays[2].nbytes)))
            if not (dev_a_v and dev_b_v and dev_c_v):
                raise RuntimeError("cpu-sim: allocate_tensor failed for verify run")
            try:
                rc = int(runner.copy_to_device(dev_a_v, base_arrays[0]))
                if rc != 0:
                    raise RuntimeError(f"cpu-sim: copy_to_device(A) failed: rc={rc}")
                rc = int(runner.copy_to_device(dev_b_v, base_arrays[1]))
                if rc != 0:
                    raise RuntimeError(f"cpu-sim: copy_to_device(B) failed: rc={rc}")

                graph_v = pto_runtime.Graph()
                for tile_id in range(int(block_dim_v)):
                    graph_v.add_task([int(tile_id), dev_a_v, dev_b_v, dev_c_v], func_id=1, core_type=1)
                rc = int(runner.run(graph_v, 1))
                if rc != 0:
                    raise RuntimeError(f"cpu-sim: NPU run failed: rc={rc}")

                c_npu = np.zeros_like(base_arrays[2])
                rc = int(runner.copy_from_device(c_npu, dev_c_v))
                if rc != 0:
                    raise RuntimeError(f"cpu-sim: copy_from_device(C) failed: rc={rc}")
            finally:
                runner.free_tensor(dev_a_v)
                runner.free_tensor(dev_b_v)
                runner.free_tensor(dev_c_v)

            c_cpu = cpu_out[0]
            if not np.allclose(c_npu, c_cpu, rtol=float(args.check_rtol), atol=float(args.check_atol)):
                diff = float(np.max(np.abs(c_npu.astype(np.float32) - c_cpu.astype(np.float32))))
                raise AssertionError(f"cpu-sim: mismatch (max_abs_diff={diff})")
            print(f"cpu-sim: OK (matched CPU simulator output)  outdir={verify_dir}")
    finally:
        runner.free_tensor(dev_a)
        runner.free_tensor(dev_b)
        runner.free_tensor(dev_c)
        runner.finalize()

    _stage("Done")
    _kv("outdir", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
