# Getting started

This repo provides an end-to-end flow for PTO kernels:

`Python → PTO-AS (.pto) → ptoas → (Ascend CCE C++ or CPU simulator) → run`

## Prerequisites

### Python

```bash
python3 -m pip install -r requirements.txt
```

### `ptoas`

`bin/ptoas` is a wrapper that dispatches to `bin/<os-arch>/ptoas`.

- Linux aarch64: `bin/linux-aarch64/ptoas` is included in this repo.
- Linux x86_64: build from source (see `bin/linux-x86_64/README.md`).
- macOS aarch64: build from source (see `bin/macos-aarch64/README.md`).

Quick check:

```bash
./bin/ptoas --help
```

## CPU-only quickstart (Ubuntu/macOS)

Compile a small GEMM kernel to a CPU `.so` and run a numpy check:

```bash
python3 kernels/python/gemm/run.py --target cpu --ptoas ./bin/ptoas --outdir /tmp/pto_kernel_python_gemm
```

## Ascend A2/A3 quickstart (Ubuntu aarch64)

### 1) Point to your CANN installation

```bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
```

### 2) Runtime (auto-built)

No manual build step is required: `pto_runtime.py` builds/loads the host runtime and AICPU/AICore helper binaries on first use via `ref_runtime/python/binary_compiler.py` (requires `cmake`/`make` and the CANN toolchain).

### 3) Run BGEMM and export a task trace

This produces:

- `kernel_0.pto`: generated PTO-AS
- `kernel_0.cpp`: `ptoas` output (Ascend CCE C++)
- `ptoas.log`: `ptoas` logs
- `trace.svg`: per-core swimlane trace (microseconds)
- `trace.json`: Chrome/Perfetto trace
- `task_profile.json`: raw per-task profiling records

```bash
python3 kernels/python/bgemm_performance/run_runtime.py \
  --ptoas ./bin/ptoas --ascend-home $ASCEND_HOME_PATH \
  --device 0 --aic-blocks 24 \
  --batch 2 --m 6144 --n 6144 --k 6144 --grid-m 4 --grid-n 6 \
  --iters 10 --warmup 2 --no-check \
  --outdir /tmp/pto_bgemm_runtime_profile \
  --trace-json /tmp/pto_bgemm_runtime_profile/trace.json \
  --trace-svg  /tmp/pto_bgemm_runtime_profile/trace.svg
```

## Troubleshooting

- `error: ptoas binary not found or not executable`: see the `bin/<os-arch>/README.md` for your platform.
- `error: cannot import pto_runtime`: run from a repo checkout (repo root must be on `PYTHONPATH`).
- `ptoas failed ... (log: ...)`: open the printed `ptoas.log` and fix the compiler/toolchain issue.
- Trace axis looks wrong: set `PTO_TRACE_TICK_HZ=<hz>` to match your device clock.
