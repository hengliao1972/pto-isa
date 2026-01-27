from __future__ import annotations

from pto import PTO


def tiled_transpose64():
    # Transpose a (64,64) tensor using (16,16) tiles:
    #   y[c:c+16, r:r+16] = transpose(x[r:r+16, c:c+16])
    pto = PTO("tiled_transpose64")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(64, 64), role="in")
    y = pto.tensor(dtype="f32", shape=(64, 64), role="out")

    tx0 = pto.vec(dtype="f32", shape=(16, 16))
    tx1 = pto.vec(dtype="f32", shape=(16, 16))
    tmp = pto.vec(dtype="f32", shape=(16, 16))
    out0 = pto.vec(dtype="f32", shape=(16, 16))
    out1 = pto.vec(dtype="f32", shape=(16, 16))

    for r in range(0, 64, 16):
        for c in range(0, 64, 32):
            # Tile (r, c) -> (c, r)
            tx0 = pto.load(x, r, c)
            out0 = pto.trans(tx0, tmp)
            pto.store(y, c, r, out0)

            # Tile (r, c+16) -> (c+16, r)
            c1 = c + 16
            tx1 = pto.load(x, r, c1)
            out1 = pto.trans(tx1, tmp)
            pto.store(y, c1, r, out1)

    pto.epilogue()
    return pto.program()
