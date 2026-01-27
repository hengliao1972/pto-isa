from __future__ import annotations

from pto import PTO


def gemm256():
    # C[256,256] = A[256,256] @ B[256,256], using (16,16,16) tiles.
    pto = PTO("gemm256")
    pto.prologue()

    a = pto.tensor(dtype="f16", shape=(256, 256), role="in")
    b = pto.tensor(dtype="f16", shape=(256, 256), role="in")
    c = pto.tensor(dtype="f32", shape=(256, 256), role="out")

    a_mat = pto.mat(dtype="f16", shape=(16, 16))
    b_mat = pto.mat(dtype="f16", shape=(16, 16))

    # Use a Left layout that matches both CPU simulator and NPU cube core.
    a_left_0 = pto.left(dtype="f16", shape=(16, 16), blayout="ColMajor", slayout="RowMajor")
    a_left_1 = pto.left(dtype="f16", shape=(16, 16), blayout="ColMajor", slayout="RowMajor")
    b_right_0 = pto.right(dtype="f16", shape=(16, 16))
    b_right_1 = pto.right(dtype="f16", shape=(16, 16))
    c_acc = pto.acc(dtype="f32", shape=(16, 16))

    for mi in range(0, 256, 16):
        for nj in range(0, 256, 16):
            for kk in range(0, 256, 16):
                a_mat = pto.load(a, mi, kk)
                b_mat = pto.load(b, kk, nj)

                it0 = kk // 16
                lane = it0 % 2
                if lane == 0:
                    a_left_0 = pto.mov(a_mat)
                    b_right_0 = pto.mov(b_mat)
                    if kk == 0:
                        c_acc = pto.matmul(a_left_0, b_right_0)
                    else:
                        c_acc = pto.matmul_acc(c_acc, a_left_0, b_right_0)
                else:
                    a_left_1 = pto.mov(a_mat)
                    b_right_1 = pto.mov(b_mat)
                    c_acc = pto.matmul_acc(c_acc, a_left_1, b_right_1)

            pto.store(c, mi, nj, c_acc)

    pto.epilogue()
    return pto.program()
