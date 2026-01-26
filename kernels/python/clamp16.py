from __future__ import annotations

from pto import PTO, scalar


def clamp16():
    # y = clamp(x, lo, hi)
    pto = PTO("clamp16")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    lo_tile = pto.vec(dtype="f32", shape=(16, 16))
    hi_tile = pto.vec(dtype="f32", shape=(16, 16))
    tmp = pto.vec(dtype="f32", shape=(16, 16))
    out = pto.vec(dtype="f32", shape=(16, 16))

    lo = pto.const("lo", -0.25, scalar("f32"))
    hi = pto.const("hi", 0.25, scalar("f32"))

    tx = pto.load(x)
    lo_tile = pto.expands(lo)
    tmp = pto.max(tx, lo_tile)
    hi_tile = pto.expands(hi)
    out = pto.min(tmp, hi_tile)
    pto.store(y, out)

    pto.epilogue()
    return pto.program()
