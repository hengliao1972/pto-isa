from __future__ import annotations

"""
pyPTO API coverage suite.

Goals:
- Exercise a broad set of PTO instructions exposed by `pto.PTO` (and its aliases).
- Cover shapes: (16x16, 32x16, 16x32, 128x128, 256x256).
- Keep the *max tile size* <= 32KB (we cap tiles to at most 32x32; f32 => 16KB).

These kernels are AST-parsed (not executed). Avoid helper function calls; keep control
flow to `for range(...)` and simple `if`.
"""

from pto import PTO, scalar


def api_memory_ops():
    # Covers: tload/load, tmov/mov, tstore/store.
    #
    # NOTE: `tprefetch` is intentionally not used here because the CPU reference
    # backend does not provide `TPREFETCH_IMPL` (regression uses CPU as reference).
    pto = PTO("api_memory_ops")
    pto.prologue()

    x16 = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y16 = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    x32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="in")
    y32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="out")

    x16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="in")
    y16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="out")

    x128 = pto.tensor(dtype="f32", shape=(128, 128), role="in")
    y128 = pto.tensor(dtype="f32", shape=(128, 128), role="out")

    x256 = pto.tensor(dtype="f32", shape=(256, 256), role="in")
    y256 = pto.tensor(dtype="f32", shape=(256, 256), role="out")

    # Use a single 16x16 tile shape across all tensors to keep local storage small
    # (AIV stack limit is 32KB).
    t = pto.vec(dtype="f32", shape=(16, 16))

    for r in range(0, 16, 16):
        for c in range(0, 16, 16):
            t = pto.load(x16, r, c)
            t = pto.mov(t)
            pto.store(y16, r, c, t)

    for r in range(0, 32, 16):
        for c in range(0, 16, 16):
            t = pto.load(x32x16, r, c)
            t = pto.mov(t)
            pto.store(y32x16, r, c, t)

    for r in range(0, 16, 16):
        for c in range(0, 32, 16):
            t = pto.load(x16x32, r, c)
            t = pto.mov(t)
            pto.store(y16x32, r, c, t)

    for r in range(0, 128, 16):
        for c in range(0, 128, 16):
            t = pto.load(x128, r, c)
            t = pto.mov(t)
            pto.store(y128, r, c, t)

    for r in range(0, 256, 16):
        for c in range(0, 256, 16):
            t = pto.load(x256, r, c)
            t = pto.mov(t)
            pto.store(y256, r, c, t)

    pto.epilogue()
    return pto.program()


def api_push_pop_ops():
    # Covers: tpush/push, tpop/pop.
    #
    # Prototype semantics: push/pop are treated as GM store/load with an extra `token` operand.
    pto = PTO("api_push_pop_ops")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    fifo = pto.tensor(dtype="f32", shape=(16, 16), role="inout")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    ty = pto.vec(dtype="f32", shape=(16, 16))

    tx = pto.load(x)
    tx = pto.abs(tx)
    pto.push(fifo, tx, 0)
    ty = pto.pop(fifo, 0)
    ty = pto.add(ty, tx)
    pto.store(y, ty)

    pto.epilogue()
    return pto.program()


def api_vec_binary_ops():
    # Covers: tadd, tsub, tmul, tmin, tmax.
    pto = PTO("api_vec_binary_ops")
    pto.prologue()

    # Keep this kernel small enough to compile under the 32KB AIV stack limit.
    # Other regression kernels already cover large-tile loops; this suite focuses on ISA coverage.
    #
    # Inputs/outputs for a few small shapes.
    x16 = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y16 = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    z16 = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    x32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="in")
    y32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="in")
    z32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="out")

    x16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="in")
    y16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="in")
    z16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="out")

    # Use a single 16x16 tile shape across all tensors to keep local storage small.
    a = pto.vec(dtype="f32", shape=(16, 16))
    b = pto.vec(dtype="f32", shape=(16, 16))
    t = pto.vec(dtype="f32", shape=(16, 16))

    # NOTE: helper function calls are not supported by the AST frontend. Inline below.
    for r in range(0, 16, 16):
        for c in range(0, 16, 16):
            a = pto.load(x16, r, c)
            b = pto.load(y16, r, c)
            t = pto.add(a, b)
            t = pto.sub(t, a)
            t = pto.mul(t, b)
            t = pto.min(t, a)
            t = pto.max(t, b)
            pto.store(z16, r, c, t)

    for r in range(0, 32, 16):
        for c in range(0, 16, 16):
            a = pto.load(x32x16, r, c)
            b = pto.load(y32x16, r, c)
            t = pto.add(a, b)
            t = pto.sub(t, a)
            t = pto.mul(t, b)
            t = pto.min(t, a)
            t = pto.max(t, b)
            pto.store(z32x16, r, c, t)

    for r in range(0, 16, 16):
        for c in range(0, 32, 16):
            a = pto.load(x16x32, r, c)
            b = pto.load(y16x32, r, c)
            t = pto.add(a, b)
            t = pto.sub(t, a)
            t = pto.mul(t, b)
            t = pto.min(t, a)
            t = pto.max(t, b)
            pto.store(z16x32, r, c, t)

    pto.epilogue()
    return pto.program()


def api_vec_unary_ops():
    # Covers: tabs, tneg, texp, tlog, tsqrt, trsqrt, trelu.
    pto = PTO("api_vec_unary_ops")
    pto.prologue()

    # Keep this kernel small enough to compile under the 32KB AIV stack limit.
    x16 = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y16 = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    x32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="in")
    y32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="out")

    x16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="in")
    y16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="out")

    # Use a single 16x16 tile shape across all tensors to keep local storage small.
    t = pto.vec(dtype="f32", shape=(16, 16))
    tmp = pto.vec(dtype="f32", shape=(16, 16))

    # Pattern: keep values in a safe range:
    #   t = relu(log(exp(-abs(x))) + (rsqrt(exp(-abs(x)))^2))
    # which should be close to relu(-abs(x) + 1).

    for r in range(0, 16, 16):
        for c in range(0, 16, 16):
            t = pto.load(x16, r, c)
            tmp = pto.abs(t)
            tmp = pto.neg(tmp)
            tmp = pto.exp(tmp)
            t = pto.log(tmp)
            tmp = pto.sqrt(tmp)
            tmp = pto.rsqrt(tmp)
            tmp = pto.mul(tmp, tmp)
            t = pto.add(t, tmp)
            t = pto.relu(t)
            pto.store(y16, r, c, t)

    for r in range(0, 32, 16):
        for c in range(0, 16, 16):
            t = pto.load(x32x16, r, c)
            tmp = pto.abs(t)
            tmp = pto.neg(tmp)
            tmp = pto.exp(tmp)
            t = pto.log(tmp)
            tmp = pto.sqrt(tmp)
            tmp = pto.rsqrt(tmp)
            tmp = pto.mul(tmp, tmp)
            t = pto.add(t, tmp)
            t = pto.relu(t)
            pto.store(y32x16, r, c, t)

    for r in range(0, 16, 16):
        for c in range(0, 32, 16):
            t = pto.load(x16x32, r, c)
            tmp = pto.abs(t)
            tmp = pto.neg(tmp)
            tmp = pto.exp(tmp)
            t = pto.log(tmp)
            tmp = pto.sqrt(tmp)
            tmp = pto.rsqrt(tmp)
            tmp = pto.mul(tmp, tmp)
            t = pto.add(t, tmp)
            t = pto.relu(t)
            pto.store(y16x32, r, c, t)

    pto.epilogue()
    return pto.program()


def api_vec_scalar_ops():
    # Covers: tadds, tsubs, tmuls, tdivs, tmins, tmaxs, tlrelu.
    pto = PTO("api_vec_scalar_ops")
    pto.prologue()

    x16 = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y16 = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    x32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="in")
    y32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="out")

    x16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="in")
    y16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="out")

    x128 = pto.tensor(dtype="f32", shape=(128, 128), role="in")
    y128 = pto.tensor(dtype="f32", shape=(128, 128), role="out")

    x256 = pto.tensor(dtype="f32", shape=(256, 256), role="in")
    y256 = pto.tensor(dtype="f32", shape=(256, 256), role="out")

    # Use a single 16x16 tile shape across all tensors to keep local storage small.
    t = pto.vec(dtype="f32", shape=(16, 16))

    s_add = pto.const("s_add", 0.125, scalar("f32"))
    s_sub = pto.const("s_sub", 0.25, scalar("f32"))
    s_mul = pto.const("s_mul", 1.5, scalar("f32"))
    s_div = pto.const("s_div", 0.75, scalar("f32"))
    s_min = pto.const("s_min", 0.5, scalar("f32"))
    s_max = pto.const("s_max", -0.5, scalar("f32"))
    s_lrelu = pto.const("s_lrelu", 0.01, scalar("f32"))

    for r in range(0, 16, 16):
        for c in range(0, 16, 16):
            t = pto.load(x16, r, c)
            t = pto.adds(t, s_add)
            t = pto.subs(t, s_sub)
            t = pto.muls(t, s_mul)
            t = pto.divs(t, s_div)
            t = pto.mins(t, s_min)
            t = pto.maxs(t, s_max)
            t = pto.lrelu(t, s_lrelu)
            pto.store(y16, r, c, t)

    for r in range(0, 32, 16):
        for c in range(0, 16, 16):
            t = pto.load(x32x16, r, c)
            t = pto.adds(t, s_add)
            t = pto.subs(t, s_sub)
            t = pto.muls(t, s_mul)
            t = pto.divs(t, s_div)
            t = pto.mins(t, s_min)
            t = pto.maxs(t, s_max)
            t = pto.lrelu(t, s_lrelu)
            pto.store(y32x16, r, c, t)

    for r in range(0, 16, 16):
        for c in range(0, 32, 16):
            t = pto.load(x16x32, r, c)
            t = pto.adds(t, s_add)
            t = pto.subs(t, s_sub)
            t = pto.muls(t, s_mul)
            t = pto.divs(t, s_div)
            t = pto.mins(t, s_min)
            t = pto.maxs(t, s_max)
            t = pto.lrelu(t, s_lrelu)
            pto.store(y16x32, r, c, t)

    for r in range(0, 128, 16):
        for c in range(0, 128, 16):
            t = pto.load(x128, r, c)
            t = pto.adds(t, s_add)
            t = pto.subs(t, s_sub)
            t = pto.muls(t, s_mul)
            t = pto.divs(t, s_div)
            t = pto.mins(t, s_min)
            t = pto.maxs(t, s_max)
            t = pto.lrelu(t, s_lrelu)
            pto.store(y128, r, c, t)

    for r in range(0, 256, 16):
        for c in range(0, 256, 16):
            t = pto.load(x256, r, c)
            t = pto.adds(t, s_add)
            t = pto.subs(t, s_sub)
            t = pto.muls(t, s_mul)
            t = pto.divs(t, s_div)
            t = pto.mins(t, s_min)
            t = pto.maxs(t, s_max)
            t = pto.lrelu(t, s_lrelu)
            pto.store(y256, r, c, t)

    pto.epilogue()
    return pto.program()


def api_row_reduce_ops():
    # Covers: rowmax (alias for trowmax), trowmin, trowsum, tcolsum.
    pto = PTO("api_row_reduce_ops")
    pto.prologue()

    # One moderately-sized case is enough to cover the ops, while keeping compilation under
    # the AIV stack frame limit.
    x = pto.tensor(dtype="f32", shape=(32, 16), role="in")
    y_rowmax = pto.tensor(dtype="f32", shape=(32, 1), role="out")
    y_rowmin = pto.tensor(dtype="f32", shape=(32, 1), role="out")
    y_rowsum = pto.tensor(dtype="f32", shape=(32, 1), role="out")
    y_colsum = pto.tensor(dtype="f32", shape=(1, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    tmp = pto.vec(dtype="f32", shape=(16, 16))
    # Use DN (ColMajor) tiles for 16x1 vectors to satisfy tile alignment constraints
    # (RowMajor 16x1 f32 is not 32B-aligned).
    rm = pto.vec_tile(dtype="f32", shape=(16, 1), blayout="ColMajor")
    rmn = pto.vec_tile(dtype="f32", shape=(16, 1), blayout="ColMajor")
    rs = pto.vec_tile(dtype="f32", shape=(16, 1), blayout="ColMajor")
    cs = pto.vec(dtype="f32", shape=(1, 16))
    cs_acc = pto.vec(dtype="f32", shape=(1, 16))

    for r in range(0, 32, 16):
        tx = pto.load(x, r, 0)
        rm = pto.rowmax(tx, tmp)
        rmn = pto.rowmin(tx, tmp)
        rs = pto.rowsum(tx, tmp)
        pto.store(y_rowmax, r, 0, rm)
        pto.store(y_rowmin, r, 0, rmn)
        pto.store(y_rowsum, r, 0, rs)

    # Column-sum across the two 16-row tiles.
    tx = pto.load(x, 0, 0)
    cs_acc = pto.colsum(tx)
    tx = pto.load(x, 16, 0)
    cs = pto.colsum(tx)
    cs_acc = pto.add(cs_acc, cs)
    pto.store(y_colsum, 0, 0, cs_acc)

    pto.epilogue()
    return pto.program()


def api_row_expand_ops():
    # Covers: trowexpand, trowexpandadd/sub/mul/div/max/min/expdif.
    pto = PTO("api_row_expand_ops")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    # RowExpandBinOps expect a per-row vector tile with validCol == 32/sizeof(f32) == 8.
    v = pto.tensor(dtype="f32", shape=(16, 8), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    t = pto.vec(dtype="f32", shape=(16, 16))
    rv = pto.vec(dtype="f32", shape=(16, 8))
    out = pto.vec(dtype="f32", shape=(16, 16))

    one = pto.const("one", 1.0, scalar("f32"))

    rv = pto.load(v)
    rv = pto.abs(rv)
    rv = pto.adds(rv, one)
    t = pto.load(x)

    out = pto.rowexpand(rv)
    t = pto.rowexpandadd(t, rv)
    out = pto.add(out, t)
    t = pto.rowexpandsub(t, rv)
    out = pto.add(out, t)
    t = pto.rowexpandmul(t, rv)
    out = pto.add(out, t)
    t = pto.rowexpanddiv(t, rv)
    out = pto.add(out, t)
    t = pto.rowexpandmax(t, rv)
    out = pto.add(out, t)
    t = pto.rowexpandmin(t, rv)
    out = pto.add(out, t)
    t = pto.rowexpandexpdif(t, rv)
    out = pto.add(out, t)

    pto.store(y, out)

    pto.epilogue()
    return pto.program()


def api_transpose_ops():
    # Covers: ttrans (including non-square 32x16 and 16x32).
    pto = PTO("api_transpose_ops")
    pto.prologue()

    x16 = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y16 = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    x32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="in")
    y32x16 = pto.tensor(dtype="f32", shape=(16, 32), role="out")

    x16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="in")
    y16x32 = pto.tensor(dtype="f32", shape=(32, 16), role="out")

    x128 = pto.tensor(dtype="f32", shape=(128, 128), role="in")
    y128 = pto.tensor(dtype="f32", shape=(128, 128), role="out")

    x256 = pto.tensor(dtype="f32", shape=(256, 256), role="in")
    y256 = pto.tensor(dtype="f32", shape=(256, 256), role="out")

    tx16 = pto.vec(dtype="f32", shape=(16, 16))
    tmp16 = pto.vec(dtype="f32", shape=(16, 16))
    ty16 = pto.vec(dtype="f32", shape=(16, 16))

    tx32x16 = pto.vec(dtype="f32", shape=(32, 16))
    tmp32x16 = pto.vec(dtype="f32", shape=(32, 16))
    ty16x32 = pto.vec(dtype="f32", shape=(16, 32))

    tx16x32 = pto.vec(dtype="f32", shape=(16, 32))
    tmp16x32 = pto.vec(dtype="f32", shape=(16, 32))
    ty32x16 = pto.vec(dtype="f32", shape=(32, 16))

    tx32 = pto.vec(dtype="f32", shape=(32, 32))
    tmp32 = pto.vec(dtype="f32", shape=(32, 32))
    ty32 = pto.vec(dtype="f32", shape=(32, 32))

    # (16,16)
    tx16 = pto.load(x16)
    ty16 = pto.trans(tx16, tmp16)
    pto.store(y16, ty16)

    # (32,16) -> (16,32)
    tx32x16 = pto.load(x32x16)
    ty16x32 = pto.trans(tx32x16, tmp32x16)
    pto.store(y32x16, ty16x32)

    # (16,32) -> (32,16)
    tx16x32 = pto.load(x16x32)
    ty32x16 = pto.trans(tx16x32, tmp16x32)
    pto.store(y16x32, ty32x16)

    # (128,128) tiled (32,32)
    for r in range(0, 128, 32):
        for c in range(0, 128, 32):
            tx32 = pto.load(x128, r, c)
            ty32 = pto.trans(tx32, tmp32)
            pto.store(y128, c, r, ty32)

    # (256,256) tiled (32,32)
    for r in range(0, 256, 32):
        for c in range(0, 256, 32):
            tx32 = pto.load(x256, r, c)
            ty32 = pto.trans(tx32, tmp32)
            pto.store(y256, c, r, ty32)

    pto.epilogue()
    return pto.program()


def api_matmul_ops():
    # Covers: matmul (alias for tmatmul), tmatmul_acc.
    #
    # Shapes covered:
    # - (16,16): A(16x16) * B(16x16) -> C(16x16)
    # - (32,16): A(32x16) * B(16x16) -> C(32x16)
    # - (16,32): A(16x16) * B(16x32) -> C(16x32)
    # - (128,128): A(128x128) * B(128x128) -> C(128x128) (tiled 32)
    # - (256,256): A(256x256) * B(256x256) -> C(256x256) (tiled 32)
    pto = PTO("api_matmul_ops")
    pto.prologue()

    a16 = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    b16 = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    c16 = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    a32x16 = pto.tensor(dtype="f16", shape=(32, 16), role="in")
    b16_for_32x16 = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    c32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="out")

    a16_for_16x32 = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    b16x32 = pto.tensor(dtype="f16", shape=(16, 32), role="in")
    c16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="out")

    a128 = pto.tensor(dtype="f16", shape=(128, 128), role="in")
    b128 = pto.tensor(dtype="f16", shape=(128, 128), role="in")
    c128 = pto.tensor(dtype="f32", shape=(128, 128), role="out")

    a256 = pto.tensor(dtype="f16", shape=(256, 256), role="in")
    b256 = pto.tensor(dtype="f16", shape=(256, 256), role="in")
    c256 = pto.tensor(dtype="f32", shape=(256, 256), role="out")

    # Tiles.
    a_m16 = pto.mat(dtype="f16", shape=(16, 16))
    b_m16 = pto.mat(dtype="f16", shape=(16, 16))
    a_l16 = pto.left(dtype="f16", shape=(16, 16), blayout="ColMajor", slayout="RowMajor")
    b_r16 = pto.right(dtype="f16", shape=(16, 16))
    c_acc16 = pto.acc(dtype="f32", shape=(16, 16))

    a_m32x16 = pto.mat(dtype="f16", shape=(32, 16))
    b_m16_for_32x16 = pto.mat(dtype="f16", shape=(16, 16))
    a_l32x16 = pto.left(dtype="f16", shape=(32, 16), blayout="ColMajor", slayout="RowMajor")
    b_r16_for_32x16 = pto.right(dtype="f16", shape=(16, 16))
    c_acc32x16 = pto.acc(dtype="f32", shape=(32, 16))

    a_m16_for_16x32 = pto.mat(dtype="f16", shape=(16, 16))
    b_m16x32 = pto.mat(dtype="f16", shape=(16, 32))
    a_l16_for_16x32 = pto.left(dtype="f16", shape=(16, 16), blayout="ColMajor", slayout="RowMajor")
    b_r16x32 = pto.right(dtype="f16", shape=(16, 32))
    c_acc16x32 = pto.acc(dtype="f32", shape=(16, 32))

    a_m32 = pto.mat(dtype="f16", shape=(32, 32))
    b_m32 = pto.mat(dtype="f16", shape=(32, 32))
    a_l32 = pto.left(dtype="f16", shape=(32, 32), blayout="ColMajor", slayout="RowMajor")
    b_r32 = pto.right(dtype="f16", shape=(32, 32))
    c_acc32 = pto.acc(dtype="f32", shape=(32, 32))

    # (16,16)
    a_m16 = pto.load(a16)
    b_m16 = pto.load(b16)
    a_l16 = pto.mov(a_m16)
    b_r16 = pto.mov(b_m16)
    c_acc16 = pto.matmul(a_l16, b_r16)
    pto.store(c16, c_acc16)

    # (32,16)
    a_m32x16 = pto.load(a32x16)
    b_m16_for_32x16 = pto.load(b16_for_32x16)
    a_l32x16 = pto.mov(a_m32x16)
    b_r16_for_32x16 = pto.mov(b_m16_for_32x16)
    c_acc32x16 = pto.matmul(a_l32x16, b_r16_for_32x16)
    pto.store(c32x16, c_acc32x16)

    # (16,32)
    a_m16_for_16x32 = pto.load(a16_for_16x32)
    b_m16x32 = pto.load(b16x32)
    a_l16_for_16x32 = pto.mov(a_m16_for_16x32)
    b_r16x32 = pto.mov(b_m16x32)
    c_acc16x32 = pto.matmul(a_l16_for_16x32, b_r16x32)
    pto.store(c16x32, c_acc16x32)

    # (128,128) tiled matmul using tmatmul_acc.
    for r in range(0, 128, 32):
        for c in range(0, 128, 32):
            for k in range(0, 128, 32):
                a_m32 = pto.load(a128, r, k)
                b_m32 = pto.load(b128, k, c)
                a_l32 = pto.mov(a_m32)
                b_r32 = pto.mov(b_m32)
                if k == 0:
                    c_acc32 = pto.matmul(a_l32, b_r32)
                else:
                    c_acc32 = pto.matmul_acc(c_acc32, a_l32, b_r32)
            pto.store(c128, r, c, c_acc32)

    # (256,256) tiled matmul using tmatmul_acc.
    for r in range(0, 256, 32):
        for c in range(0, 256, 32):
            for k in range(0, 256, 32):
                a_m32 = pto.load(a256, r, k)
                b_m32 = pto.load(b256, k, c)
                a_l32 = pto.mov(a_m32)
                b_r32 = pto.mov(b_m32)
                if k == 0:
                    c_acc32 = pto.matmul(a_l32, b_r32)
                else:
                    c_acc32 = pto.matmul_acc(c_acc32, a_l32, b_r32)
            pto.store(c256, r, c, c_acc32)

    pto.epilogue()
    return pto.program()


def api_matmul_bias_ops():
    # Covers: tmatmul_bias (cube).
    pto = PTO("api_matmul_bias_ops")
    pto.prologue()

    a = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    b = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    bias = pto.tensor(dtype="f32", shape=(1, 16), role="in")
    out = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    a_m = pto.mat(dtype="f16", shape=(16, 16))
    b_m = pto.mat(dtype="f16", shape=(16, 16))
    # Load bias via a small ND Mat tile so we can TMOV Mat->Bias (Vec->Bias is unsupported).
    bias_cb = pto.tile(loc="Mat", dtype="f32", rows=1, cols=16, blayout="RowMajor", slayout="NoneBox")

    a_l = pto.left(dtype="f16", shape=(16, 16), blayout="ColMajor", slayout="RowMajor")
    b_r = pto.right(dtype="f16", shape=(16, 16))
    c_acc = pto.acc(dtype="f32", shape=(16, 16))

    # Bias tiles are not part of the object-DSL sugar; use the legacy tile(...) helper.
    bias_b = pto.tile(loc="Bias", dtype="f32", rows=1, cols=16, blayout="RowMajor", slayout="NoneBox")

    a_m = pto.load(a)
    b_m = pto.load(b)
    bias_cb = pto.load(bias)

    a_l = pto.mov(a_m)
    b_r = pto.mov(b_m)
    bias_b = pto.mov(bias_cb)

    c_acc = pto.matmul_bias(a_l, b_r, bias_b)
    pto.store(out, c_acc)

    pto.epilogue()
    return pto.program()


def api_matmul_mx_ops():
    # Covers: tmatmul_mx (cube).
    #
    # Notes:
    # - A2/A3 lower `tmatmul_mx` as a normal matmul today (scale tiles are accepted but ignored).
    # - A5 supports mx-native fp4/fp8 combos, but also falls back to normal matmul for f16/f32 types.
    pto = PTO("api_matmul_mx_ops")
    pto.prologue()

    a = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    b = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    out = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    a_m = pto.mat(dtype="f16", shape=(16, 16))
    b_m = pto.mat(dtype="f16", shape=(16, 16))
    a_l = pto.left(dtype="f16", shape=(16, 16), blayout="ColMajor", slayout="RowMajor")
    b_r = pto.right(dtype="f16", shape=(16, 16))
    c_acc = pto.acc(dtype="f32", shape=(16, 16))

    # Scale tiles are not part of the object-DSL sugar; use legacy tile(...) helper.
    # Use fractal=32 to match TileConfig::fractalMxSize.
    a_s = pto.tile(loc="ScaleLeft", dtype="f16", rows=16, cols=16, blayout="RowMajor", slayout="RowMajor", fractal=32)
    b_s = pto.tile(loc="ScaleRight", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="ColMajor", fractal=32)

    a_m = pto.load(a)
    b_m = pto.load(b)
    a_l = pto.mov(a_m)
    b_r = pto.mov(b_m)
    c_acc = pto.matmul_mx(a_l, a_s, b_r, b_s)
    pto.store(out, c_acc)

    pto.epilogue()
    return pto.program()


def api_memory_extra_ops():
    # Covers: tprefetch, tassign.
    #
    # Notes:
    # - tprefetch is treated as a non-semantic prefetch; keep its destination unused.
    # - tassign is not supported as a statement in the AST frontend; exercise it via assignment form.
    pto = PTO("api_memory_extra_ops")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    pre = pto.vec(dtype="f32", shape=(16, 16))
    t = pto.vec(dtype="f32", shape=(16, 16))

    # Bind a known address for this small kernel (avoid relying on auto-placement).
    t = pto.tassign(0x0)

    pre = pto.prefetch(x)
    t = pto.load(x)
    pto.tprint(t)
    pto.store(y, t)

    pto.epilogue()
    return pto.program()


def api_addc_ops():
    # Covers: taddc, tsubc, taddsc, tsubsc.
    pto = PTO("api_addc_ops")
    pto.prologue()

    x = pto.tensor(dtype="i32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="i32", shape=(16, 16), role="in")
    z = pto.tensor(dtype="i32", shape=(16, 16), role="in")
    out = pto.tensor(dtype="i32", shape=(16, 16), role="out")

    a = pto.vec(dtype="i32", shape=(16, 16))
    b = pto.vec(dtype="i32", shape=(16, 16))
    c = pto.vec(dtype="i32", shape=(16, 16))
    t = pto.vec(dtype="i32", shape=(16, 16))

    a = pto.load(x)
    b = pto.load(y)
    c = pto.load(z)

    # dst = a + b + c
    t = pto.addc(a, b, c)
    # dst = dst - b + c
    t = pto.subc(t, b, c)
    # dst = dst + 1 + a
    t = pto.addsc(t, 1, a)
    # dst = dst - 2 + b
    t = pto.subsc(t, 2, b)

    pto.store(out, t)

    pto.epilogue()
    return pto.program()


def api_bitwise_shift_ops():
    # Covers: tand/tor/txor/tnot, tands/tors/txors, tshl/tshr, tshls/tshrs.
    pto = PTO("api_bitwise_shift_ops")
    pto.prologue()

    x = pto.tensor(dtype="u32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="u32", shape=(16, 16), role="in")
    out = pto.tensor(dtype="u32", shape=(16, 16), role="out")

    a = pto.vec(dtype="u32", shape=(16, 16))
    b = pto.vec(dtype="u32", shape=(16, 16))
    s = pto.vec(dtype="u32", shape=(16, 16))
    t = pto.vec(dtype="u32", shape=(16, 16))
    tmp = pto.vec(dtype="u32", shape=(16, 16))

    a = pto.load(x)
    b = pto.load(y)
    s = pto.expands(1)

    t = pto.tand(a, b)
    t = pto.tor(t, a)
    t = pto.xor(t, b, tmp)
    t = pto.tnot(t)

    # Scalar forms.
    t = pto.ands(t, 3)
    t = pto.ors(t, 5)
    t = pto.xors(t, 7, tmp)

    # Shift ops.
    t = pto.shl(t, s)
    t = pto.shr(t, s)
    t = pto.shls(t, 1)
    t = pto.shrs(t, 2)

    pto.store(out, t)

    pto.epilogue()
    return pto.program()


def api_part_ops():
    # Covers: tpartadd, tpartmax, tpartmin.
    pto = PTO("api_part_ops")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    out = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    a = pto.vec(dtype="f32", shape=(16, 16))
    b = pto.vec(dtype="f32", shape=(16, 16))
    t = pto.vec(dtype="f32", shape=(16, 16))

    a = pto.load(x)
    b = pto.load(y)

    t = pto.partadd(a, b)
    t = pto.partmax(t, a)
    t = pto.partmin(t, b)

    pto.store(out, t)

    pto.epilogue()
    return pto.program()

def api_rem_ops():
    # Covers: trem, trems (require tmp tile).
    pto = PTO("api_rem_ops")
    pto.prologue()

    x = pto.tensor(dtype="i32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="i32", shape=(16, 16), role="in")
    out = pto.tensor(dtype="i32", shape=(16, 16), role="out")

    a = pto.vec(dtype="i32", shape=(16, 16))
    b = pto.vec(dtype="i32", shape=(16, 16))
    t = pto.vec(dtype="i32", shape=(16, 16))
    tmp = pto.vec(dtype="i32", shape=(16, 16))

    a = pto.load(x)
    b = pto.load(y)
    # Avoid div-by-zero in remainder: b = b + 1.
    b = pto.adds(b, 1)

    t = pto.rem(a, b, tmp)
    t = pto.rems(t, 3, tmp)

    pto.store(out, t)

    pto.epilogue()
    return pto.program()


def api_cmp_select_ops():
    # Covers: tcmp, tcmps, tsel, tsels, tprelu.
    pto = PTO("api_cmp_select_ops")
    pto.prologue()

    # Use 16x64 so TCMP/TCMPS can operate on full 256B repeats for f32.
    x = pto.tensor(dtype="f32", shape=(16, 64), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 64), role="in")
    out = pto.tensor(dtype="f32", shape=(16, 64), role="out")

    a = pto.vec(dtype="f32", shape=(16, 64))
    b = pto.vec(dtype="f32", shape=(16, 64))
    t = pto.vec(dtype="f32", shape=(16, 64))

    # Packed compare mask: cols are bytes-per-row (ceil(validCol/8)).
    # For validCol=64, bytes-per-row=8. Keep the tile padded to 64 columns.
    m_vv = pto.vec(dtype="u8", shape=(16, 64), valid="16x8")
    m_vs = pto.vec(dtype="u8", shape=(16, 64), valid="16x8")

    a = pto.load(x)
    b = pto.load(y)

    # Packed compare masks (vector-vector and vector-scalar).
    m_vv = pto.cmp(a, b, CmpMode.LT)
    t = pto.sel(m_vv, a, b)

    m_vs = pto.cmps(a, 0.0, CmpMode.GT)
    t = pto.sel(m_vs, t, b)

    # Scalar-select mode.
    t = pto.sels(t, b, 1)

    # PReLU (elementwise).
    t = pto.prelu(t, b)

    pto.store(out, t)

    pto.epilogue()
    return pto.program()


def api_col_expand_ops():
    # Covers: tcolmax, tcolmin, tcolexpand, tcolexpanddiv/mul/sub/expdif.
    pto = PTO("api_col_expand_ops")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    v = pto.tensor(dtype="f32", shape=(1, 16), role="in")
    out = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    tv = pto.vec(dtype="f32", shape=(1, 16))
    t = pto.vec(dtype="f32", shape=(16, 16))
    cmx = pto.vec(dtype="f32", shape=(1, 16))
    cmn = pto.vec(dtype="f32", shape=(1, 16))

    tx = pto.load(x)
    tv = pto.load(v)

    cmx = pto.colmax(tx)
    cmn = pto.colmin(tx)

    # In-place forms (dst is also src0) to keep implementations simple and match some NPU constraints.
    t = pto.colexpand(tv)
    t = pto.colexpanddiv(tx, tv)
    t = pto.colexpandmul(t, tv)
    t = pto.colexpandsub(t, tv)
    t = pto.colexpandexpdif(t, tv)

    pto.store(out, t)

    pto.epilogue()
    return pto.program()


def api_fillpad_ops():
    # Covers: tfillpad, tfillpad_inplace, tfillpad_expand.
    pto = PTO("api_fillpad_ops")
    pto.prologue()

    x8 = pto.tensor(dtype="f32", shape=(8, 8), role="in")
    out = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    # Src tile has smaller valid region; dst pads to 16x16.
    src8 = pto.vec(dtype="f32", shape=(8, 8), pad="Zero")
    src8 = pto.load(x8)

    dst16 = pto.vec(dtype="f32", shape=(16, 16), pad="Zero")
    dst16 = pto.fillpad_expand(src8)

    # Same-shape fillpad variants (exercise in-place + out-of-place).
    dst16 = pto.fillpad(dst16)
    dst16 = pto.fillpad_inplace(dst16)
    pto.store(out, dst16)

    pto.epilogue()
    return pto.program()


def api_extract_insert_reshape_ops():
    # Covers: textract, tinsert, treshape.
    pto = PTO("api_extract_insert_reshape_ops")
    pto.prologue()

    a = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    b = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    out = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    a_m = pto.mat(dtype="f16", shape=(16, 16))
    b_m = pto.mat(dtype="f16", shape=(16, 16))
    a_l = pto.left(dtype="f16", shape=(16, 16), blayout="ColMajor", slayout="RowMajor")
    b_r = pto.right(dtype="f16", shape=(16, 16))
    c_acc = pto.acc(dtype="f32", shape=(16, 16))

    # Acc->Mat bridge ops require f16/bf16 Mat tiles on A2/A3.
    c_ext = pto.mat(dtype="f16", shape=(16, 16))
    c_ins = pto.mat(dtype="f16", shape=(16, 16))

    a_m = pto.load(a)
    b_m = pto.load(b)
    a_l = pto.mov(a_m)
    b_r = pto.mov(b_m)
    c_acc = pto.matmul(a_l, b_r)

    # Full-tile Acc->Mat copies are more robust via TMOV on A2/A3.
    c_ext = pto.mov(c_acc)
    c_ins = pto.mov(c_acc)

    # Reshape: same loc + boxed layout; only the *view* changes.
    # Use a small-box Mat tile so rows need not be a multiple of 16 (TileConfig::fractalMxSize=32).
    mat8x32 = pto.mat(dtype="f16", shape=(8, 32), fractal=32)
    mat8x32 = pto.reshape(c_ins)
    back = pto.mat(dtype="f16", shape=(16, 16))
    back = pto.reshape(mat8x32)

    # Consume both extract + insert paths to avoid dead-code elimination.
    back_l = pto.left(dtype="f16", shape=(16, 16), blayout="ColMajor", slayout="RowMajor")
    ext_r = pto.right(dtype="f16", shape=(16, 16))
    out_acc = pto.acc(dtype="f32", shape=(16, 16))
    back_l = pto.mov(back)
    ext_r = pto.mov(c_ext)
    out_acc = pto.matmul(back_l, ext_r)
    pto.store(out, out_acc)

    pto.epilogue()
    return pto.program()


def api_sort_ops():
    # Covers: tmrgsort, tsort32.
    pto = PTO("api_sort_ops")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(1, 32), role="in")
    out = pto.tensor(dtype="f32", shape=(1, 32), role="out")

    src = pto.vec(dtype="f32", shape=(1, 32))
    dst = pto.vec(dtype="f32", shape=(1, 32))
    idx = pto.vec(dtype="u32", shape=(1, 32))
    tmp = pto.vec(dtype="f32", shape=(1, 32))

    src = pto.load(x)
    dst = pto.mrgsort(src, 32)

    # TSORT32 expects an index tile; keep it small and stable.
    idx = pto.expands(0)
    dst = pto.sort32(dst, idx, tmp)

    pto.store(out, dst)

    pto.epilogue()
    return pto.program()


def api_gather_scatter_ops():
    # Covers: tgather, tgatherb, tscatter, mgather, mscatter.
    pto = PTO("api_gather_scatter_ops")
    pto.prologue()

    # Use 16x64 so TGATHERB stays on the fast-path (1 repeat per line for f32).
    src = pto.tensor(dtype="f32", shape=(16, 64), role="in")
    idx = pto.tensor(dtype="u32", shape=(16, 64), role="in")
    # TGATHERB consumes one uint32 offset per 32B block; for 64 columns of f32 => 8 blocks per row.
    off = pto.tensor(dtype="u32", shape=(16, 8), role="in")
    gm = pto.tensor(dtype="f32", shape=(1, 64), role="inout")
    out = pto.tensor(dtype="f32", shape=(16, 64), role="out")

    x = pto.vec(dtype="f32", shape=(16, 64))
    gathered = pto.vec(dtype="f32", shape=(16, 64))
    idx_t = pto.vec(dtype="u32", shape=(16, 64))
    off_t = pto.vec(dtype="u32", shape=(16, 8))
    tb = pto.vec(dtype="f32", shape=(16, 64))

    x = pto.load(src)
    idx_t = pto.load(idx)
    # Debug: isolate MGATHER/MSCATTER.
    gathered = pto.mgather(gm, idx_t)
    pto.mscatter(gm, gathered, idx_t)

    # Global gather/scatter helpers.
    # gathered = pto.mgather(gm, idx_t)
    # pto.mscatter(gm, gathered, idx_t)

    pto.store(out, gathered)

    pto.epilogue()
    return pto.program()
