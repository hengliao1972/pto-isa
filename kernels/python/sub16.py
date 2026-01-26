from __future__ import annotations

from pto import PTO


def sub16():
    # z = x - y
    pto = PTO("sub16")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    z = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    ty = pto.vec(dtype="f32", shape=(16, 16))
    out = pto.vec(dtype="f32", shape=(16, 16))

    tx = pto.load(x)
    ty = pto.load(y)
    out = pto.sub(tx, ty)
    pto.store(z, out)

    pto.epilogue()
    return pto.program()
