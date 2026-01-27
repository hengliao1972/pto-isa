from __future__ import annotations

from pto import PTO


def tiled_add128():
    # Add on a (128,128) tensor processed as (16,16) tiles.
    pto = PTO("tiled_add128")
    pto.prologue()

    x = pto.tensor(dtype="f16", shape=(128, 128), role="in")
    y = pto.tensor(dtype="f16", shape=(128, 128), role="in")
    z = pto.tensor(dtype="f16", shape=(128, 128), role="out")

    # Ping-pong buffers to avoid cross-pipe reuse hazards across tiles.
    tx0 = pto.vec(dtype="f16", shape=(16, 16))
    tx1 = pto.vec(dtype="f16", shape=(16, 16))
    ty0 = pto.vec(dtype="f16", shape=(16, 16))
    ty1 = pto.vec(dtype="f16", shape=(16, 16))
    out0 = pto.vec(dtype="f16", shape=(16, 16))
    out1 = pto.vec(dtype="f16", shape=(16, 16))

    for r in range(0, 128, 16):
        for c in range(0, 128, 32):
            # Tile (r, c)
            tx0 = pto.load(x, r, c)
            ty0 = pto.load(y, r, c)
            out0 = pto.add(tx0, ty0)
            pto.store(z, r, c, out0)

            # Tile (r, c+16)
            c1 = c + 16
            tx1 = pto.load(x, r, c1)
            ty1 = pto.load(y, r, c1)
            out1 = pto.add(tx1, ty1)
            pto.store(z, r, c1, out1)

    pto.epilogue()
    return pto.program()
