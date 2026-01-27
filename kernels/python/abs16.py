from __future__ import annotations

from pto import PTO


def abs16():
    pto = PTO("abs16")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    out = pto.vec(dtype="f32", shape=(16, 16))

    tx = pto.load(x)
    out = pto.abs(tx)
    pto.store(y, out)

    pto.epilogue()
    return pto.program()
