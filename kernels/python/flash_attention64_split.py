from __future__ import annotations

from math import sqrt

from pto import PTO, scalar


def flash_attention64_split():
    """
    FlashAttention-like demo split into multiple kernels:
      1) QK^T (cube)  -> scores (GM f32)
      2) softmax (vec)-> probs  (GM f16)
      3) PV (cube)    -> out    (GM f32)

    Notes:
    - This kernel is parsed by the AST frontend (not executed).
    - The split is driven by `pto.stage_*()` marker ops and `ptoas --split-kernels`.
    - Workspace tensors (`scores_gm`, `probs_gm`) are explicit args so stages can communicate via GM.
    """
    S = 64
    D = 32
    TS = 16

    pto = PTO("flash_attention64_split")
    pto.prologue()

    # Inputs.
    q = pto.tensor(dtype="f16", shape=(S, D), role="in")      # [S, D]
    kt = pto.tensor(dtype="f16", shape=(D, S), role="in")     # [D, S]  (transposed K)
    v = pto.tensor(dtype="f16", shape=(S, D), role="in")      # [S, D]

    # Workspaces (role="in" so regression compares only the final `out`).
    scores_gm = pto.tensor(dtype="f32", shape=(S, S), role="in")  # [S, S]
    probs_gm = pto.tensor(dtype="f16", shape=(S, S), role="in")   # [S, S]

    # Output.
    out = pto.tensor(dtype="f32", shape=(S, D), role="out")   # [S, D]

    bn = pto.get_block_num()
    bid = pto.get_block_idx()

    # --- Tiles: QK stage (cube) ---
    q_mat = pto.mat(dtype="f16", shape=(TS, TS))
    kt_mat = pto.mat(dtype="f16", shape=(TS, TS))
    q_left = pto.left(dtype="f16", shape=(TS, TS), blayout="ColMajor", slayout="RowMajor")
    kt_right = pto.right(dtype="f16", shape=(TS, TS))
    scores_acc = pto.acc(dtype="f32", shape=(TS, TS))

    # --- Tiles: softmax stage (vec) ---
    scores_tile = pto.vec(dtype="f32", shape=(TS, TS))
    tmp = pto.vec(dtype="f32", shape=(TS, TS))
    # NOTE: For A5 vector codegen, keep per-row reduction outputs in a row-major tile
    # (stride=TS) so subsequent `rowexpand()` can use aligned vector loads.
    row_max_part = pto.vec(dtype="f32", shape=(TS, TS))
    max_bcast = pto.vec(dtype="f32", shape=(TS, TS))
    max_acc = pto.vec(dtype="f32", shape=(TS, TS))
    centered = pto.vec(dtype="f32", shape=(TS, TS))
    exp_scores = pto.vec(dtype="f32", shape=(TS, TS))
    row_sum_part = pto.vec(dtype="f32", shape=(TS, TS))
    sum_bcast = pto.vec(dtype="f32", shape=(TS, TS))
    sum_acc = pto.vec(dtype="f32", shape=(TS, TS))
    probs_f32 = pto.vec(dtype="f32", shape=(TS, TS))
    probs_f16 = pto.vec(dtype="f16", shape=(TS, TS))

    scale = pto.const("scale", 1.0 / sqrt(D), scalar("f32"))

    # --- Tiles: PV stage (cube) ---
    p_mat = pto.mat(dtype="f16", shape=(TS, TS))
    v_mat = pto.mat(dtype="f16", shape=(TS, TS))
    p_left = pto.left(dtype="f16", shape=(TS, TS), blayout="ColMajor", slayout="RowMajor")
    v_right = pto.right(dtype="f16", shape=(TS, TS))
    out_acc = pto.acc(dtype="f32", shape=(TS, TS))

    # -------------------------------------------------------------------------
    # Stage 1: scores = Q @ K^T (cube)
    # -------------------------------------------------------------------------
    pto.stage_qk_cube()
    for mi in range(0, S, TS):
        tile_row = mi // TS
        blk_lane = tile_row % bn
        if blk_lane == bid:
            for nj in range(0, S, TS):
                for kk in range(0, D, TS):
                    q_mat = pto.load(q, mi, kk)
                    kt_mat = pto.load(kt, kk, nj)
                    q_left = pto.mov(q_mat)
                    kt_right = pto.mov(kt_mat)
                    if kk == 0:
                        scores_acc = pto.matmul(q_left, kt_right)
                    else:
                        scores_acc = pto.matmul_acc(scores_acc, q_left, kt_right)

                pto.store(scores_gm, mi, nj, scores_acc)
        else:
            pto.comment("qk: skip tile row")

    # -------------------------------------------------------------------------
    # Stage 2: probs = softmax(scores * scale) (vec)
    # -------------------------------------------------------------------------
    pto.stage_softmax_vec()
    for mi in range(0, S, TS):
        tile_row = mi // TS
        blk_lane = tile_row % bn
        if blk_lane == bid:
            # Pass 1: row_max across all column tiles.
            for nj in range(0, S, TS):
                scores_tile = pto.load(scores_gm, mi, nj)
                scores_tile = pto.muls(scores_tile, scale)
                row_max_part = pto.rowmax(scores_tile, tmp)
                max_bcast = pto.rowexpand(row_max_part)
                if nj == 0:
                    max_acc = pto.mov(max_bcast)
                else:
                    max_acc = pto.max(max_acc, max_bcast)

            # Pass 2: row_sum across all column tiles.
            for nj in range(0, S, TS):
                scores_tile = pto.load(scores_gm, mi, nj)
                scores_tile = pto.muls(scores_tile, scale)
                centered = pto.sub(scores_tile, max_acc)
                exp_scores = pto.exp(centered)
                row_sum_part = pto.rowsum(exp_scores, tmp)
                sum_bcast = pto.rowexpand(row_sum_part)
                if nj == 0:
                    sum_acc = pto.mov(sum_bcast)
                else:
                    sum_acc = pto.add(sum_acc, sum_bcast)

            # Pass 3: probs tile-by-tile.
            for nj in range(0, S, TS):
                scores_tile = pto.load(scores_gm, mi, nj)
                scores_tile = pto.muls(scores_tile, scale)
                centered = pto.sub(scores_tile, max_acc)
                exp_scores = pto.exp(centered)
                probs_f32 = pto.div(exp_scores, sum_acc)
                probs_f16 = pto.cvt(probs_f32, RoundMode.CAST_ROUND)
                pto.store(probs_gm, mi, nj, probs_f16)
        else:
            pto.comment("softmax: skip tile row")

    # -------------------------------------------------------------------------
    # Stage 3: out = probs @ V (cube)
    # -------------------------------------------------------------------------
    pto.stage_pv_cube()
    for mi in range(0, S, TS):
        tile_row = mi // TS
        blk_lane = tile_row % bn
        if blk_lane == bid:
            for dj in range(0, D, TS):
                for kk in range(0, S, TS):
                    p_mat = pto.load(probs_gm, mi, kk)
                    v_mat = pto.load(v, kk, dj)
                    p_left = pto.mov(p_mat)
                    v_right = pto.mov(v_mat)
                    if kk == 0:
                        out_acc = pto.matmul(p_left, v_right)
                    else:
                        out_acc = pto.matmul_acc(out_acc, p_left, v_right)

                pto.store(out, mi, dj, out_acc)
        else:
            pto.comment("pv: skip tile row")

    pto.epilogue()
    return pto.program()
