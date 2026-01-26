from __future__ import annotations

from dataclasses import dataclass

from pto import PTO, KernelSpec, compile_kernel_spec


@dataclass(frozen=True)
class GemmConfig:
    m: int
    n: int
    k: int
    bm: int = 128
    bn: int = 128
    bk: int = 64


def gemm_f16f16f32():
    pto = PTO("gemm_f16f16f32")
    pto.prologue()
    bid = pto.get_block_idx()

    A = pto.tensor(dtype="f16", shape=(m, k), role="in")
    # Represent B as a DN tensor backed by a physical [n, k] row-major buffer on host.
    # This matches the manual A2/A3 GEMM kernels which typically load B in DN/ZN form.
    B = pto.tensor(dtype="f16", shape=(k, n), stride=(1, k), layout="DN", role="in")
    C = pto.tensor(dtype="f32", shape=(m, n), role="out")

    # Mat tiles (loaded from GM). Shape matches Left/Right so we can use TMOV (no TEXTRACT here).
    a_mat = pto.mat(dtype="f16", shape=(bm, bk))
    # For DN global tensors, use a ZN Mat tile (DN->ZN is supported by TLOAD(MatTile,...)).
    b_mat = pto.mat(dtype="f16", shape=(bk, bn), blayout="RowMajor", slayout="ColMajor")

    # Matmul operands / accumulator.
    a_left_0 = pto.left(dtype="f16", shape=(bm, bk))
    a_left_1 = pto.left(dtype="f16", shape=(bm, bk))
    b_right_0 = pto.right(dtype="f16", shape=(bk, bn))
    b_right_1 = pto.right(dtype="f16", shape=(bk, bn))
    c_acc = pto.acc(dtype="f32", shape=(bm, bn))

    tiles_m = m // bm
    tiles_n = n // bn

    m_idx = bid % tiles_m
    n_idx = bid // tiles_m

    if n_idx < tiles_n:
        m0 = m_idx * bm
        n0 = n_idx * bn

        for k0 in range(0, k, bk):
            a_mat = pto.load(A, m0, k0)
            b_mat = pto.load(B, k0, n0)
            it0 = k0 // bk
            lane = it0 % 2
            if lane == 0:
                a_left_0 = pto.mov(a_mat)
                b_right_0 = pto.mov(b_mat)
                if k0 == 0:
                    c_acc = pto.matmul(a_left_0, b_right_0)
                else:
                    c_acc = pto.matmul_acc(c_acc, a_left_0, b_right_0)
            else:
                a_left_1 = pto.mov(a_mat)
                b_right_1 = pto.mov(b_mat)
                c_acc = pto.matmul_acc(c_acc, a_left_1, b_right_1)
        pto.store(C, m0, n0, c_acc)

    pto.epilogue()
    return pto.program()


def make_gemm_f16f16f32_kernel(*, cfg: GemmConfig) -> KernelSpec:
    if cfg.m % cfg.bm != 0 or cfg.n % cfg.bn != 0 or cfg.k % cfg.bk != 0:
        raise ValueError("m,n,k must be multiples of bm,bn,bk")
    return compile_kernel_spec(
        gemm_f16f16f32,
        consts={"m": int(cfg.m), "n": int(cfg.n), "k": int(cfg.k), "bm": int(cfg.bm), "bn": int(cfg.bn), "bk": int(cfg.bk)},
    )
