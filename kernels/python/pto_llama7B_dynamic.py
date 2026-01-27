from __future__ import annotations

from pto import PTO


def pto_llama7B_dynamic():
    # Upstream ref: `~/github/pto-isa/examples/pto_llama7B_dynamic.py`
    pto = PTO("pto_llama7B_dynamic")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(32, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(32, 16), role="out")

    # Use two GM->Vec load buffers and two Vec->GM store buffers to avoid
    # cross-pipe buffer reuse hazards between tiles.
    tx0 = pto.vec(dtype="f32", shape=(16, 16))
    tx1 = pto.vec(dtype="f32", shape=(16, 16))
    row_max = pto.vec(dtype="f32", shape=(16, 1), blayout="ColMajor")
    tmp = pto.vec(dtype="f32", shape=(16, 16))
    centered = pto.vec(dtype="f32", shape=(16, 16))
    exp_x = pto.vec(dtype="f32", shape=(16, 16))
    row_sum = pto.vec(dtype="f32", shape=(16, 1), blayout="ColMajor")
    out0 = pto.vec(dtype="f32", shape=(16, 16))
    out1 = pto.vec(dtype="f32", shape=(16, 16))

    # Tile 0 (rows 0..15).
    tx0 = pto.load(x, 0, 0)
    row_max = pto.rowmax(tx0, tmp)
    centered = pto.rowexpandsub(tx0, row_max)
    exp_x = pto.exp(centered)
    row_sum = pto.rowsum(exp_x, tmp)
    out0 = pto.rowexpanddiv(exp_x, row_sum)
    pto.store(y, 0, 0, out0)

    # Tile 1 (rows 16..31).
    tx1 = pto.load(x, 16, 0)
    row_max = pto.rowmax(tx1, tmp)
    centered = pto.rowexpandsub(tx1, row_max)
    exp_x = pto.exp(centered)
    row_sum = pto.rowsum(exp_x, tmp)
    out1 = pto.rowexpanddiv(exp_x, row_sum)
    pto.store(y, 16, 0, out1)

    pto.epilogue()
    return pto.program()
