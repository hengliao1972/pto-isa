from __future__ import annotations

from pto import PTO


def pto_torch_flexattention():
    # Upstream ref: `~/github/pto-isa/examples/pto_torch_flexattention.py`
    pto = PTO("pto_torch_flexattention")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    row_max = pto.vec(dtype="f32", shape=(16, 1), blayout="ColMajor")
    tmp = pto.vec(dtype="f32", shape=(16, 16))
    centered = pto.vec(dtype="f32", shape=(16, 16))
    exp_x = pto.vec(dtype="f32", shape=(16, 16))
    row_sum = pto.vec(dtype="f32", shape=(16, 1), blayout="ColMajor")
    out = pto.vec(dtype="f32", shape=(16, 16))

    tx = pto.load(x)
    row_max = pto.rowmax(tx, tmp)
    centered = pto.rowexpandsub(tx, row_max)
    exp_x = pto.exp(centered)
    row_sum = pto.rowsum(exp_x, tmp)
    out = pto.rowexpanddiv(exp_x, row_sum)
    pto.store(y, out)

    pto.epilogue()
    return pto.program()
