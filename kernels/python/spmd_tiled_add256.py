from __future__ import annotations

from pto import PTO


def spmd_tiled_add256():
    # Block-parallel add/sub + mul on a (256,256) tensor, tiled as (16,16).
    #
    # Work partition:
    #   rows_per_blk = 256 // get_block_num()
    #   each block handles a contiguous row stripe.
    #
    # Control flow:
    #   alternate between add and sub based on the current column tile.
    pto = PTO("spmd_tiled_add256")
    pto.prologue()

    x = pto.tensor(dtype="f16", shape=(256, 256), role="in")
    y = pto.tensor(dtype="f16", shape=(256, 256), role="in")
    z = pto.tensor(dtype="f16", shape=(256, 256), role="out")

    tx = pto.vec(dtype="f16", shape=(16, 16))
    ty = pto.vec(dtype="f16", shape=(16, 16))
    tmp = pto.vec(dtype="f16", shape=(16, 16))
    out = pto.vec(dtype="f16", shape=(16, 16))

    bn = pto.get_block_num()
    bid = pto.get_block_idx()
    rows_per_blk = 256 // bn
    r_base = bid * rows_per_blk
    r_end = r_base + rows_per_blk

    for r in range(r_base, r_end, 16):
        for c in range(0, 256, 16):
            tx = pto.load(x, r, c)
            ty = pto.load(y, r, c)

            c_mod = c % 32
            if c_mod == 0:
                tmp = pto.add(tx, ty)
            else:
                tmp = pto.sub(tx, ty)

            # Back-to-back vector ops to stress PIPE_V barriers.
            out = pto.mul(tmp, tx)
            pto.store(z, r, c, out)

    pto.epilogue()
    return pto.program()
