from __future__ import annotations

from pto import PTO


def fma16():
    # z = x * y + b
    pto = PTO("fma16")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    b = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    z = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    ty = pto.vec(dtype="f32", shape=(16, 16))
    tb = pto.vec(dtype="f32", shape=(16, 16))
    prod = pto.vec(dtype="f32", shape=(16, 16))
    out = pto.vec(dtype="f32", shape=(16, 16))

    tx = pto.load(x)
    ty = pto.load(y)
    tb = pto.load(b)
    prod = pto.mul(tx, ty)
    out = pto.add(prod, tb)
    pto.store(z, out)

    pto.epilogue()
    return pto.program()
