from __future__ import annotations

from pto import PTO


def abs_add16():
    # z = abs(x) + y
    pto = PTO("abs_add16")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    z = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    ty = pto.vec(dtype="f32", shape=(16, 16))
    ax = pto.vec(dtype="f32", shape=(16, 16))
    out = pto.vec(dtype="f32", shape=(16, 16))

    tx = pto.load(x)
    ty = pto.load(y)
    ax = pto.abs(tx)
    out = pto.add(ax, ty)
    pto.store(z, out)

    pto.epilogue()
    return pto.program()
