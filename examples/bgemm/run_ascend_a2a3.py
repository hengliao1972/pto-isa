#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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


@dataclass(frozen=True)
class _SvgTask:
    task_id: int
    exec_core_id: int
    start: int
    end: int


def _write_task_trace_svg(
    *,
    svg_path: Path,
    tasks: list[_SvgTask],
    num_cores: int,
    task_names: dict[int, str] | None = None,
) -> None:
    svg_path = Path(svg_path).resolve()
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    if not tasks:
        svg_path.write_text(
            "\n".join(
                [
                    '<?xml version="1.0" encoding="UTF-8"?>',
                    '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="120">',
                    '<rect x="0" y="0" width="640" height="120" fill="#fff"/>',
                    '<text x="20" y="60" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="14">',
                    "no tasks",
                    "</text>",
                    "</svg>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return

    min_start = min(t.start for t in tasks)
    max_end = max(t.end for t in tasks)
    span = max(1, int(max_end - min_start))

    width = 1800
    margin_left = 120
    margin_right = 30
    margin_top = 50
    lane_h = 18
    lane_gap = 6
    axis_h = 24
    footer_h = 50
    height = margin_top + axis_h + int(num_cores) * (lane_h + lane_gap) + footer_h

    x0 = margin_left
    x1 = width - margin_right
    usable_w = max(1, x1 - x0)
    scale = float(usable_w) / float(span)

    def _x(tick: int) -> float:
        return float(x0) + float(int(tick) - int(min_start)) * scale

    def _y(core_id: int) -> int:
        return int(margin_top + axis_h + int(core_id) * (lane_h + lane_gap))

    def _esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    lines.append(
        "<style>"
        "text{font-family:ui-monospace, SFMono-Regular, Menlo, monospace; fill:#111}"
        ".lane{fill:#fafafa}"
        ".laneAlt{fill:#f3f6fb}"
        ".axis{stroke:#222; stroke-width:1}"
        ".tick{stroke:#888; stroke-width:1}"
        ".bar{fill:#4C78A8; stroke:#2b4c6f; stroke-width:1}"
        ".label{font-size:11px; fill:#fff}"
        ".coreLabel{font-size:11px; fill:#222}"
        ".title{font-size:16px; font-weight:600}"
        ".sub{font-size:12px; fill:#333}"
        "</style>"
    )
    lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>')
    lines.append(f'<text class="title" x="{margin_left}" y="26">Task trace (AICore)</text>')
    lines.append(
        f'<text class="sub" x="{margin_left}" y="44">time unit: device sys_cnt ticks (relative), cores={int(num_cores)}, tasks={len(tasks)}, span={span}</text>'
    )

    # Axis
    axis_y = margin_top + 10
    lines.append(f'<line class="axis" x1="{x0}" y1="{axis_y}" x2="{x1}" y2="{axis_y}"/>')
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tick = int(min_start + int(frac * span))
        x = _x(tick)
        lines.append(f'<line class="tick" x1="{x:.2f}" y1="{axis_y - 6}" x2="{x:.2f}" y2="{axis_y + 6}"/>')
        rel = int(tick - min_start)
        lines.append(f'<text x="{x:.2f}" y="{axis_y - 10}" font-size="10" text-anchor="middle">{rel}</text>')

    # Lanes
    for core in range(int(num_cores)):
        y = _y(core)
        klass = "laneAlt" if (core % 2 == 1) else "lane"
        lines.append(f'<rect class="{klass}" x="0" y="{y}" width="{width}" height="{lane_h + lane_gap}"/>')
        lines.append(
            f'<text class="coreLabel" x="{margin_left - 10}" y="{y + lane_h - 4}" text-anchor="end">core {core:02d}</text>'
        )

    # Bars
    name_map = task_names or {}
    for t in sorted(tasks, key=lambda v: (v.exec_core_id, v.start, v.task_id)):
        if t.end <= t.start:
            continue
        if int(t.exec_core_id) < 0 or int(t.exec_core_id) >= int(num_cores):
            continue
        y = _y(int(t.exec_core_id))
        x = _x(int(t.start))
        w = max(1.0, _x(int(t.end)) - x)
        bar_y = y + 2
        bar_h = lane_h - 4
        lines.append(f'<g>')
        lines.append(f'  <rect class="bar" x="{x:.2f}" y="{bar_y}" width="{w:.2f}" height="{bar_h}" rx="2" ry="2"/>')
        label = name_map.get(int(t.task_id), f"T{t.task_id}")
        lines.append(f'  <text class="label" x="{x + 4:.2f}" y="{bar_y + bar_h - 4}">{_esc(label)}</text>')
        lines.append(
            f'  <title>{_esc(f"task {t.task_id} ({label}) | core {t.exec_core_id} | start {t.start - min_start} | end {t.end - min_start} | dur {t.end - t.start}")}</title>'
        )
        lines.append(f"</g>")

    lines.append(
        f'<text class="sub" x="{margin_left}" y="{height - 18}">Tip: hover a bar to see exact start/end ticks + duration.</text>'
    )
    lines.append("</svg>")
    lines.append("")

    svg_path.write_text("\n".join(lines), encoding="utf-8")


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
    b: np.ndarray,
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
    # B is passed as a physical row-major matrix of shape [n_pad, batch*k_pad],
    # corresponding to a logical DN tensor of shape [batch*k_pad, n_pad].
    b32 = b.astype(np.float32, copy=False)
    n_pad = int(b32.shape[0])
    for r, col in zip(rs, cs):
        r_i = int(r)
        c_i = int(col)
        b_id = r_i // int(m_pad)
        b_off = int(b_id) * int(k_pad)
        expected = float(np.dot(a32[r_i, : int(k_req)], b32[c_i, b_off : b_off + int(k_req)]))
        offset = (r_i * int(n_pad) + c_i) * 4  # C is [batch*m_pad, n_pad] f32
        got = _read_device_f32(runner, dev_ptr=int(c_dev), offset_bytes=int(offset))
        if not np.isfinite(got):
            raise AssertionError(f"non-finite output at ({r_i},{c_i}): {got}")
        if not np.isclose(got, expected, rtol=float(rtol), atol=float(atol)):
            raise AssertionError(f"mismatch at ({r_i},{c_i}): got={got} expected={expected}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="BGEMM on Ascend A2/A3 via runtime (Python frontend -> .pto -> ptoas -> CCE -> AICPU graph -> AICore)."
    )
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/pto_bgemm_a2a3"))
    ap.add_argument("--ptoas", type=Path, default=_REPO_ROOT / "bin" / "ptoas")
    ap.add_argument(
        "--ascend-home",
        type=Path,
        default=Path(os.environ.get("ASCEND_HOME_PATH", "")) if os.environ.get("ASCEND_HOME_PATH") else Path.home() / "Ascend" / "ascend-toolkit" / "latest",
    )
    ap.add_argument("--device", type=int, default=int(os.environ.get("PTO_DEVICE", "0")))
    ap.add_argument("--aic-blocks", type=int, default=int(os.environ.get("PTO_AIC_BLOCKS", "24")))
    ap.add_argument("--memory-model", type=str, default="MEMORY_BASE", choices=["MEMORY_BASE", "REGISTER_BASE"])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--m", type=int, default=6144)
    ap.add_argument("--n", type=int, default=6144)
    ap.add_argument("--k", type=int, default=6144)
    ap.add_argument("--grid-m", type=int, default=4)
    ap.add_argument("--grid-n", type=int, default=6)
    ap.add_argument("--allow-unaligned", action="store_true")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--profile", action="store_true", help="Enable per-task profiling (start/end timestamps + core id).")
    ap.add_argument(
        "--trace-svg",
        type=Path,
        default=None,
        help="Write a per-core task trace SVG (requires --profile; default uses the last run's profile).",
    )
    ap.add_argument("--check", dest="check", action="store_true")
    ap.add_argument("--no-check", dest="check", action="store_false")
    ap.set_defaults(check=True)
    ap.add_argument("--check-samples", type=int, default=16)
    ap.add_argument("--check-rtol", type=float, default=2e-2)
    ap.add_argument("--check-atol", type=float, default=5e-2)
    args = ap.parse_args()

    try:
        import pto_runtime  # type: ignore
    except Exception as exc:
        print(f"error: cannot import pto_runtime: {exc}", file=sys.stderr)
        return 2

    from examples.bgemm.bgemm_kernel import make_bgemm_kernel
    from pto.runtime import PtoasConfig, compile_and_load_kernel_from_pto

    _stage("Config")
    _kv("repo_root", _REPO_ROOT)
    out_dir = Path(args.outdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _kv("outdir", out_dir)
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

    _stage("Build PTO-AS (Python frontend)")
    spec = make_bgemm_kernel(
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
    (out_dir / "kernel_0.pto").write_text(spec.pto, encoding="utf-8")
    _kv("pto_path", out_dir / "kernel_0.pto")

    _stage("Init runtime")
    runner = pto_runtime.DeviceRunner.get()
    try:
        rc = int(runner.init(int(args.device), int(args.aic_blocks), None, None))
    except Exception as exc:
        print(f"error: DeviceRunner.init failed: {exc}", file=sys.stderr)
        return 2
    if rc != 0:
        print(f"error: DeviceRunner.init failed: {rc}", file=sys.stderr)
        return rc
    runner.set_profile_enabled(bool(args.profile or args.trace_svg))

    _stage("Compile & load (ptoas → CCE C++ → runtime)")
    cfg = PtoasConfig(
        ptoas=Path(args.ptoas),
        memory_model=str(args.memory_model),
        repo_root=_REPO_ROOT,
        log_path=out_dir / "ptoas.log",
        print_cmd=True,
    )
    out_cpp = compile_and_load_kernel_from_pto(
        runner=runner,
        func_id=0,
        pto=(out_dir / "kernel_0.pto"),
        out_dir=out_dir,
        pto_isa_root=_REPO_ROOT,
        ptoas_cfg=cfg,
    )
    _kv("cce_cpp", out_cpp)
    _kv("ptoas_log", out_dir / "ptoas.log")

    _stage("Allocate & upload tensors")
    rng = np.random.default_rng(19)
    a = np.zeros((int(batch) * int(m_pad), int(k_pad)), dtype=np.float16)
    a_i16 = rng.integers(-1000, 1000, size=(int(batch) * int(m_req), int(k_req)), dtype=np.int16)
    a[: int(batch) * int(m_req), : int(k_req)] = (a_i16.astype(np.float16) / np.float16(256.0)).astype(np.float16, copy=False)
    del a_i16

    # B physical (row-major): [n_pad, batch*k_pad], representing a logical DN tensor [batch*k_pad, n_pad].
    # The i-th batch occupies columns [i*k_pad, (i+1)*k_pad) in the physical buffer.
    b = np.zeros((int(n_pad), int(batch) * int(k_pad)), dtype=np.float16)
    for b_id in range(int(batch)):
        b_i16 = rng.integers(-1000, 1000, size=(int(k_req), int(n_req)), dtype=np.int16)
        b_off = int(b_id) * int(k_pad)
        b[: int(n_req), b_off : b_off + int(k_req)] = (
            (b_i16.astype(np.float16) / np.float16(256.0)).astype(np.float16, copy=False).T
        )
        del b_i16

    c = np.zeros((int(batch) * int(m_pad), int(n_pad)), dtype=np.float32)

    _kv("A", f"shape={tuple(a.shape)} dtype={a.dtype} bytes={_fmt_bytes(int(a.nbytes))}")
    _kv("B", f"shape={tuple(b.shape)} dtype={b.dtype} bytes={_fmt_bytes(int(b.nbytes))} (DN physical: B^T)")
    _kv("C", f"shape={tuple(c.shape)} dtype={c.dtype} bytes={_fmt_bytes(int(c.nbytes))}")

    dev_a = int(runner.allocate_tensor(int(a.nbytes)))
    dev_b = int(runner.allocate_tensor(int(b.nbytes)))
    dev_c = int(runner.allocate_tensor(int(c.nbytes)))
    if not (dev_a and dev_b and dev_c):
        print("error: allocate_tensor failed", file=sys.stderr)
        runner.finalize()
        return 1

    try:
        rc = int(runner.copy_to_device(dev_a, a))
        if rc != 0:
            raise RuntimeError(f"copy_to_device(A) failed: rc={rc}")
        rc = int(runner.copy_to_device(dev_b, b))
        if rc != 0:
            raise RuntimeError(f"copy_to_device(B) failed: rc={rc}")

        _stage("Build task graph (orchestration)")
        orch = pto_runtime.OrchestrationRuntime(runner=runner)
        for tile_id in range(block_dim):
            # BGEMM tasks are independent (disjoint output tiles), so we do not add coarse
            # read/write dependency tracking here.
            orch.add_task([int(tile_id), dev_a, dev_b, dev_c], func_id=0, core_type=1)

        if int(args.warmup) > 0:
            _stage(f"Warmup ({int(args.warmup)} run(s))")
            for _ in range(int(args.warmup)):
                rc = int(orch.run())
                if rc != 0:
                    raise RuntimeError(f"warmup run failed: rc={rc}")

        _stage("Benchmark")
        t0 = time.perf_counter()
        for _ in range(int(args.iters)):
            rc = int(orch.run())
            if rc != 0:
                raise RuntimeError(f"run failed: rc={rc}")
        t1 = time.perf_counter()

        if args.trace_svg:
            _stage("Trace (SVG)")
            prof = runner.get_last_profile()
            tasks: list[_SvgTask] = []
            for r in prof:
                if int(r.start_time) == 0 or int(r.end_time) == 0:
                    continue
                tasks.append(
                    _SvgTask(
                        task_id=int(r.task_id),
                        exec_core_id=int(r.exec_core_id),
                        start=int(r.start_time),
                        end=int(r.end_time),
                    )
                )
            svg_path = Path(args.trace_svg)
            _write_task_trace_svg(svg_path=svg_path, tasks=tasks, num_cores=int(args.aic_blocks), task_names=orch.get_task_name_map())
            _kv("trace_svg", svg_path.resolve())

        avg_s = (t1 - t0) / float(max(1, int(args.iters)))
        avg_ms = avg_s * 1e3
        flops_req = 2.0 * float(batch) * float(m_req) * float(n_req) * float(k_req)
        flops_exec = 2.0 * float(batch) * float(m_pad) * float(n_pad) * float(k_pad)
        tflops_req = flops_req / (avg_s * 1.0e12)
        tflops_exec = flops_exec / (avg_s * 1.0e12)
        _kv("avg_time_ms", f"{avg_ms:.4f}  (iters={int(args.iters)})")
        _kv("throughput", f"{tflops_req:.2f} TFLOPS (logical)  |  {tflops_exec:.2f} TFLOPS (executed)")

        if args.check:
            _stage("Correctness check (random samples)")
            _check_samples(
                runner=runner,
                c_dev=dev_c,
                a=a,
                b=b,
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
    finally:
        try:
            runner.free_tensor(dev_a)
            runner.free_tensor(dev_b)
            runner.free_tensor(dev_c)
        except Exception:
            pass
        runner.finalize()

    _stage("Done")
    _kv("outdir", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
