from __future__ import annotations

from pto import PTO, scalar


def axpy16():
    # z = alpha * x + y
    pto = PTO("axpy16")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    z = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    ty = pto.vec(dtype="f32", shape=(16, 16))
    tmp = pto.vec(dtype="f32", shape=(16, 16))
    out = pto.vec(dtype="f32", shape=(16, 16))

    alpha = pto.const("alpha", 0.25, scalar("f32"))

    tx = pto.load(x)
    ty = pto.load(y)
    tmp = pto.muls(tx, alpha)
    out = pto.add(tmp, ty)
    pto.store(z, out)

    pto.epilogue()
    return pto.program()
