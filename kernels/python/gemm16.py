from __future__ import annotations

from pto import PTO


def gemm16():
    pto = PTO("gemm16")
    pto.prologue()

    a = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    b = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    c = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    a_mat = pto.mat(dtype="f16", shape=(16, 16))
    b_mat = pto.mat(dtype="f16", shape=(16, 16))

    # Use a Left layout that matches both CPU simulator and NPU cube core.
    a_left = pto.left(dtype="f16", shape=(16, 16), blayout="ColMajor", slayout="RowMajor")
    b_right = pto.right(dtype="f16", shape=(16, 16))
    c_acc = pto.acc(dtype="f32", shape=(16, 16))

    a_mat = pto.load(a)
    b_mat = pto.load(b)
    a_left = pto.mov(a_mat)
    b_right = pto.mov(b_mat)
    c_acc = pto.matmul(a_left, b_right)
    pto.store(c, c_acc)

    pto.epilogue()
    return pto.program()
