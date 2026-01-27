from __future__ import annotations

from pto import PTO


def pto_aten_ir_primitives():
    # Upstream ref: `~/github/pto-isa/examples/pto_aten_ir_primitives.py`
    pto = PTO("pto_aten_ir_primitives")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    z = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    ty = pto.vec(dtype="f32", shape=(16, 16))
    tz = pto.vec(dtype="f32", shape=(16, 16))

    tx = pto.load(x)
    ty = pto.load(y)
    tz = pto.mul(tx, ty)
    pto.store(z, tz)

    pto.epilogue()
    return pto.program()
