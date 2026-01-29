# Generated BGEMM intermediates

This folder is meant to *demonstrate* the full pipeline artifacts:

- `orchestration.py`: orchestration function (graph builder) that runs the scheduler on **AICPU**
- `kernel_0.pto`: PTO-AS kernel emitted by the Python frontend
- `kernel_0.cpp`: `ptoas` output (Ascend CCE C++)
- `ptoas.log`: `ptoas` compilation log
- `swimlane.svg`: per-core task trace (24 lanes, random task names assigned at submission + start/end ticks)
- `perf_2048.txt`: one captured BGEMM run (command + throughput)

To regenerate:

```bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
python3 examples/bgemm/generate_intermediates.py --ptoas ./bin/ptoas
```

To (re)capture a task trace on NPU:

```bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
timeout 900s PYTHONDONTWRITEBYTECODE=1 python3 examples/bgemm/run_ascend_a2a3.py \
  --outdir /tmp/pto_bgemm_trace_2048 --ptoas ./bin/ptoas --ascend-home "$ASCEND_HOME_PATH" \
  --device 0 --aic-blocks 24 \
  --batch 2 --m 2048 --n 2048 --k 2048 --grid-m 4 --grid-n 4 \
  --iters 1 --warmup 0 --no-check --profile --trace-svg examples/bgemm/generated/swimlane.svg
```
