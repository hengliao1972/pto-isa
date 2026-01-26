from __future__ import annotations

from pto import PTO


def spmd_tiled_transpose256():
    # Block-parallel transpose of a (256,256) tensor, tiled as (16,16).
    #
    # Each block owns a contiguous row stripe in `x`, which maps to a contiguous
    # column stripe in the transposed output `y`. This avoids output races for
    # block_dim > 1.
    #
    # Control flow:
    #   for alternating column tiles, add a no-op "normalize" path to exercise
    #   scf.if + extra vector ops without changing numerics.
    pto = PTO("spmd_tiled_transpose256")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(256, 256), role="in")
    y = pto.tensor(dtype="f32", shape=(256, 256), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    tmp = pto.vec(dtype="f32", shape=(16, 16))
    t0 = pto.vec(dtype="f32", shape=(16, 16))
    zeros = pto.vec(dtype="f32", shape=(16, 16))
    out = pto.vec(dtype="f32", shape=(16, 16))

    bn = pto.get_block_num()
    bid = pto.get_block_idx()
    rows_per_blk = 256 // bn
    r_base = bid * rows_per_blk
    r_end = r_base + rows_per_blk

    for r in range(r_base, r_end, 16):
        for c in range(0, 256, 16):
            tx = pto.load(x, r, c)
            t0 = pto.trans(tx, tmp)

            c_mod = c % 32
            if c_mod == 0:
                # out = t0 + 0
                zeros = pto.sub(t0, t0)
                out = pto.add(t0, zeros)
                pto.store(y, c, r, out)
            else:
                pto.store(y, c, r, t0)

    pto.epilogue()
    return pto.program()
