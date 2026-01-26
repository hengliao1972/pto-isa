# BGEMM examples (runtime workflow)

These examples run the **PTO-AS** flow through the **runtime**:

`Python kernel → PTO-AS → ptoas → CCE → runtime graph → Ascend NPU`

## Requirements (Ascend NPU)

```bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
```

Build the runtime Python bindings once:

```bash
cd runtime
mkdir -p build
cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON
make -j
export PYTHONPATH=$PWD/python:$PYTHONPATH
```

## Performance BGEMM (recommended)

Even `k/64` gives the fastest ping-pong schedule.

```bash
python3 examples/bgemm/pto_bgemm.py \
  --ptoas ./bin/ptoas --ascend-home $ASCEND_HOME_PATH \
  --device 0 --aic-blocks 24 \
  --batch 2 --m 6144 --n 6144 --k 6144 --grid-m 4 --grid-n 6
```

## “Adaptive” (allows unaligned shapes)

This pads inputs to tile multiples (and validates only the original region if `--check` is enabled).

```bash
python3 examples/bgemm/pto_bgemm_adaptive.py \
  --ptoas ./bin/ptoas --ascend-home $ASCEND_HOME_PATH \
  --device 0 --aic-blocks 24 \
  --allow-unaligned --batch 2 --m 4097 --n 6150 --k 4101 --grid-m 4 --grid-n 6
```
