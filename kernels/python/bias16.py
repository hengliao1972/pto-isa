from __future__ import annotations

from pto import PTO, scalar


def bias16():
    # y = x + b
    pto = PTO("bias16")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    out = pto.vec(dtype="f32", shape=(16, 16))

    b = pto.const("b", 1.5, scalar("f32"))

    tx = pto.load(x)
    out = pto.adds(tx, b)
    pto.store(y, out)

    pto.epilogue()
    return pto.program()
