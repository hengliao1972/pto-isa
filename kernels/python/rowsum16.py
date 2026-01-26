from __future__ import annotations

from pto import PTO


def rowsum16():
    # y[r,0] = sum_c x[r,c]
    pto = PTO("rowsum16")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 1), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    tmp = pto.vec(dtype="f32", shape=(16, 16))
    row_sum = pto.vec(dtype="f32", shape=(16, 1), blayout="ColMajor")

    tx = pto.load(x)
    row_sum = pto.rowsum(tx, tmp)
    pto.store(y, row_sum)

    pto.epilogue()
    return pto.program()
