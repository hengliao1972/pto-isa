from __future__ import annotations

from pto import PTO, scalar


def rsqrt16():
    # y = rsqrt(abs(x) + eps)
    pto = PTO("rsqrt16")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    ax = pto.vec(dtype="f32", shape=(16, 16))
    tmp = pto.vec(dtype="f32", shape=(16, 16))
    out = pto.vec(dtype="f32", shape=(16, 16))

    eps = pto.const("eps", 1e-3, scalar("f32"))

    tx = pto.load(x)
    ax = pto.abs(tx)
    tmp = pto.adds(ax, eps)
    out = pto.rsqrt(tmp)
    pto.store(y, out)

    pto.epilogue()
    return pto.program()
