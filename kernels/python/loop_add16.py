from __future__ import annotations

from pto import PTO


def loop_add16():
    # Simple SCF loop + vector pipe stress: z = x * (iters + 1)
    pto = PTO("loop_add16")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    z = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    acc0 = pto.vec(dtype="f32", shape=(16, 16))
    acc1 = pto.vec(dtype="f32", shape=(16, 16))

    tx = pto.load(x)
    acc0 = pto.mov(tx)

    iters = 3
    for _ in range(iters):
        # Avoid in-place `tadd(dst=src0, ...)` (not guaranteed safe across backends).
        acc1 = pto.add(acc0, tx)
        acc0 = pto.mov(acc1)

    pto.store(z, acc0)
    pto.epilogue()
    return pto.program()
