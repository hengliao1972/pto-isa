from __future__ import annotations

from pto import PTO


def spmd_tiled_rowsum256():
    # Block-parallel row reduction on a (256,256) tensor using (16,16) tiles.
    #
    # Output: y has shape (256,16). Each row stores a "signed" reduction result
    # broadcast across 16 columns:
    #   y[r, :] = sum_{c tiles} (+/-) sum(x[r, c:c+16])
    #
    # Control flow:
    #   alternate add/sub accumulation based on the current column tile.
    pto = PTO("spmd_tiled_rowsum256")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(256, 256), role="in")
    y = pto.tensor(dtype="f32", shape=(256, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    tmp = pto.vec(dtype="f32", shape=(16, 16))
    row_sum = pto.vec(dtype="f32", shape=(16, 1), blayout="ColMajor")
    bcast = pto.vec(dtype="f32", shape=(16, 16))
    acc = pto.vec(dtype="f32", shape=(16, 16))

    bn = pto.get_block_num()
    bid = pto.get_block_idx()
    rows_per_blk = 256 // bn
    r_base = bid * rows_per_blk
    r_end = r_base + rows_per_blk

    for r in range(r_base, r_end, 16):
        # Init accumulator to 0 using the first tile in this row block.
        tx = pto.load(x, r, 0)
        acc = pto.sub(tx, tx)

        for c in range(0, 256, 16):
            tx = pto.load(x, r, c)
            row_sum = pto.rowsum(tx, tmp)
            bcast = pto.rowexpand(row_sum)

            c_mod = c % 32
            if c_mod == 0:
                acc = pto.add(acc, bcast)
            else:
                acc = pto.sub(acc, bcast)

        pto.store(y, r, 0, acc)

    pto.epilogue()
    return pto.program()
