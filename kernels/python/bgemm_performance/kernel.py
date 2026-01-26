from __future__ import annotations

from pto import PTO, KernelSpec, compile_kernel_spec


def make_bgemm_performance_kernel(
    *,
    batch: int = 2,
    m: int = 6144,
    k: int = 6144,
    n: int = 6144,
    grid_m: int = 4,
    grid_n: int = 6,
    base_m: int = 128,
    base_k: int = 64,
    base_n: int = 256,
) -> KernelSpec:
    """
    Batched GEMM (BGEMM) built on the `gemm_performance` kernel structure.

    Notes:
    - A and C are packed by batch along the row dimension:
        A: [batch*m, k] (ND)
        C: [batch*m, n] (ND)
      The i-th batch owns rows [i*m, (i+1)*m).
    - B is also batched, but packed by batch along the row dimension:
        B: [batch*k, n] (DN)
      The i-th batch owns rows [i*k, (i+1)*k).
      For DN layout, host passes B^T as a physical [n, batch*k] row-major buffer.
    - Blocks are partitioned as: batch * (grid_m * grid_n).
    """
    if batch <= 0:
        raise ValueError("batch must be > 0")
    if grid_m <= 0 or grid_n <= 0:
        raise ValueError("grid_m/grid_n must be > 0")
    if base_m <= 0 or base_k <= 0 or base_n <= 0:
        raise ValueError("base_m/base_k/base_n must be > 0")
    if (m % (grid_m * base_m)) != 0 or (n % (grid_n * base_n)) != 0 or (k % base_k) != 0:
        raise ValueError(
            f"shape must be divisible by grid/base tiles "
            f"(m%{grid_m*base_m}==0, n%{grid_n*base_n}==0, k%{base_k}==0)"
        )

    return compile_kernel_spec(
        bgemm_performance,
        consts={
            "batch": int(batch),
            "m": int(m),
            "k": int(k),
            "n": int(n),
            "grid_m": int(grid_m),
            "grid_n": int(grid_n),
            "base_m": int(base_m),
            "base_k": int(base_k),
            "base_n": int(base_n),
        },
    )


def bgemm_performance():
    pto = PTO("bgemm_performance")
    pto.prologue()

    # Task identifier:
    # - Runtime/MPMD scheduler path: provided by the orchestration layer as args[0]
    # - Direct block launch path: lowered to get_block_idx()
    bid = pto.get_task_id()

    # Reserve %arg0 for the orchestration-provided task id; tensors start at %arg1.
    A = pto.tensor(dtype="f16", shape=(batch * m, k), role="in", arg=1)
    # DN tensor backed by a physical [n, batch*k] row-major buffer (host passes per-batch B^T contiguous).
    B = pto.tensor(dtype="f16", shape=(batch * k, n), stride=(1, batch * k), layout="DN", role="in", arg=2)
    C = pto.tensor(dtype="f32", shape=(batch * m, n), role="out", arg=3)

    # Double-buffer GM->L1 (Mat) and L1->L0 (Left/Right) to enable overlap across pipes.
    a_mat0 = pto.mat(dtype="f16", shape=(base_m, base_k))
    a_mat1 = pto.mat(dtype="f16", shape=(base_m, base_k))
    b_mat0 = pto.mat(dtype="f16", shape=(base_k, base_n), blayout="RowMajor", slayout="ColMajor")
    b_mat1 = pto.mat(dtype="f16", shape=(base_k, base_n), blayout="RowMajor", slayout="ColMajor")

    a0 = pto.left(dtype="f16", shape=(base_m, base_k))
    a1 = pto.left(dtype="f16", shape=(base_m, base_k))
    b0 = pto.right(dtype="f16", shape=(base_k, base_n))
    b1 = pto.right(dtype="f16", shape=(base_k, base_n))

    c = pto.acc(dtype="f32", shape=(base_m, base_n))

    blocks_per_batch = grid_m * grid_n
    b_id = bid // blocks_per_batch
    bid0 = bid % blocks_per_batch

    # Per-batch geometry.
    single_core_m = m // grid_m
    single_core_n = n // grid_n
    m_loop = single_core_m // base_m
    n_loop = single_core_n // base_n
    k_tiles = k // base_k

    m_core = bid0 % grid_m
    n_core = bid0 // grid_m

    if b_id < batch:
        if n_core < grid_n:
            b_off = b_id * m
            bk_off = b_id * k
            m_off = m_core * single_core_m
            base_m0 = b_off + m_off
            base_n0 = n_core * single_core_n

            for mi in range(0, m_loop):
                mi_off = mi * base_m
                m0 = base_m0 + mi_off
                for nj in range(0, n_loop):
                    nj_off = nj * base_n
                    n0 = base_n0 + nj_off

                    # Prime buffer 0 (k0=0).
                    a_mat0 = pto.load(A, m0, 0)
                    b_mat0 = pto.load(B, bk_off, n0)
                    a0 = pto.mov(a_mat0)
                    b0 = pto.mov(b_mat0)

                    k_rem2 = k_tiles % 2
                    if k_rem2 == 0:
                        # Prime buffer 1 (k0=base_k) to simplify conservative event insertion.
                        if k_tiles > 1:
                            a_mat1 = pto.load(A, m0, base_k)
                            bk1 = bk_off + base_k
                            b_mat1 = pto.load(B, bk1, n0)
                            a1 = pto.mov(a_mat1)
                            b1 = pto.mov(b_mat1)

                        for kt in range(0, k_tiles):
                            k_next_t = kt + 1
                            if k_next_t < k_tiles:
                                k1 = k_next_t * base_k
                                lane = kt % 2
                                if lane == 0:
                                    a_mat1 = pto.load(A, m0, k1)
                                    b_k1 = bk_off + k1
                                    b_mat1 = pto.load(B, b_k1, n0)
                                    a1 = pto.mov(a_mat1)
                                    b1 = pto.mov(b_mat1)
                                else:
                                    a_mat0 = pto.load(A, m0, k1)
                                    b_k1 = bk_off + k1
                                    b_mat0 = pto.load(B, b_k1, n0)
                                    a0 = pto.mov(a_mat0)
                                    b0 = pto.mov(b_mat0)

                            lane = kt % 2
                            if lane == 0:
                                if kt == 0:
                                    c = pto.matmul(a0, b0)
                                else:
                                    c = pto.matmul_acc(c, a0, b0)
                            else:
                                c = pto.matmul_acc(c, a1, b1)
                    else:
                        # Odd K-tiles: use a conservative sequential schedule (no ping-pong prefetch) to avoid
                        # L0A/L0B hazards at tile boundaries.
                        for kt in range(0, k_tiles):
                            k0 = kt * base_k
                            a_mat0 = pto.load(A, m0, k0)
                            b_k0 = bk_off + k0
                            b_mat0 = pto.load(B, b_k0, n0)
                            a0 = pto.mov(a_mat0)
                            b0 = pto.mov(b_mat0)
                            if kt == 0:
                                c = pto.matmul(a0, b0)
                            else:
                                c = pto.matmul_acc(c, a0, b0)

                    pto.store(C, m0, n0, c)

    pto.epilogue()
    return pto.program()
