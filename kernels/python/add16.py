from __future__ import annotations

from pto import PTO


def add16():
    pto = PTO("add16")
    pto.prologue()

    x = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    z = pto.tensor(dtype="f16", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f16", shape=(16, 16))
    ty = pto.vec(dtype="f16", shape=(16, 16))
    tz = pto.vec(dtype="f16", shape=(16, 16))

    tx = pto.load(x)
    ty = pto.load(y)
    tz = pto.add(tx, ty)
    pto.store(z, tz)

    pto.epilogue()
    return pto.program()
