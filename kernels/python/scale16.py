from __future__ import annotations

from pto import PTO, scalar


def scale16():
    # y = beta * x
    pto = PTO("scale16")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    out = pto.vec(dtype="f32", shape=(16, 16))

    beta = pto.const("beta", 3.0, scalar("f32"))

    tx = pto.load(x)
    out = pto.muls(tx, beta)
    pto.store(y, out)

    pto.epilogue()
    return pto.program()
