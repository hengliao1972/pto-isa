from __future__ import annotations

from pto import PTO


def transpose16():
    pto = PTO("transpose16")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    ty = pto.vec(dtype="f32", shape=(16, 16))
    tmp = pto.vec(dtype="f32", shape=(16, 16))

    tx = pto.load(x)
    ty = pto.trans(tx, tmp)
    pto.store(y, ty)

    pto.epilogue()
    return pto.program()
