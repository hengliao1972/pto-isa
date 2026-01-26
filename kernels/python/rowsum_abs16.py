from __future__ import annotations

from pto import PTO


def rowsum_abs16():
    # NOTE: This currently mismatches CPU-vs-NPU (seen as garbage values in a couple rows).
    # Kept around as a reduction+unary repro; it is intentionally NOT in `run_regression.py`.
    # y[r,0] = sum_c abs(x[r,c])
    pto = PTO("rowsum_abs16")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 1), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    ax = pto.vec(dtype="f32", shape=(16, 16))
    tmp = pto.vec(dtype="f32", shape=(16, 16))
    row_sum = pto.vec(dtype="f32", shape=(16, 1), blayout="ColMajor")

    tx = pto.load(x)
    ax = pto.abs(tx)
    row_sum = pto.rowsum(ax, tmp)
    pto.store(y, row_sum)

    pto.epilogue()
    return pto.program()
