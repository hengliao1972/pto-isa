# GEMM (Python → PTO-AS → ptoas) Example

This is a minimal GEMM example written in Python that generates PTO-AS text, then uses the `ptoas` toolchain to build and run on:

- **CPU simulator** (`ptoas --target cpu` → C++ → `clang++` → `.so`)
- **Real NPU** (`ptoas --target npu` → CCE → `bisheng` → fatobj `.so`)

It is intentionally small (16×16) and focuses on the end-to-end flow.

## Prereqs

- Ensure `ptoas` is available for your platform:
  - `./bin/ptoas` should work (see `bin/README.md`).

- For NPU: source Ascend env and ensure `ASCEND_HOME_PATH` is set:

```bash
source $HOME/Ascend/ascend-toolkit/latest/bin/setenv.bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
```

## Run

CPU:

```bash
python3 kernels/custom/gemm_python/run.py --target cpu
```

Or:

```bash
cd kernels/custom/gemm_python
./run.sh cpu
```

NPU:

```bash
python3 kernels/custom/gemm_python/run.py --target npu --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
```

Both:

```bash
python3 kernels/custom/gemm_python/run.py --target both --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
```
