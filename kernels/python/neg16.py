from __future__ import annotations

from pto import PTO


def neg16():
    # y = -x
    pto = PTO("neg16")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    out = pto.vec(dtype="f32", shape=(16, 16))

    tx = pto.load(x)
    out = pto.neg(tx)
    pto.store(y, out)

    pto.epilogue()
    return pto.program()
