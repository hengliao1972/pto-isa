from __future__ import annotations

from pto import PTO


def rowmax16():
    # y[r,0] = max_c x[r,c]
    pto = PTO("rowmax16")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 1), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    tmp = pto.vec(dtype="f32", shape=(16, 16))
    row_max = pto.vec(dtype="f32", shape=(16, 1), blayout="ColMajor")

    tx = pto.load(x)
    row_max = pto.rowmax(tx, tmp)
    pto.store(y, row_max)

    pto.epilogue()
    return pto.program()
