from __future__ import annotations

from pto import PTO, KernelSpec, compile_kernel_spec


def make_gemm_performance_kernel(
    *,
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
    High-performance-ish GEMM for A3 cube (fp16 inputs -> fp32 output).

    Design goals:
    - Match the manual GEMM performance example's launch geometry: 24 blocks (4x6 split).
    - Keep the kernel simple and let `ptoas --insert-events` insert the necessary set/wait flags.
    - Use device-side timing + sampled numpy validation in the runner.

    Layout:
    - A: ND [m, k] fp16
    - B: DN [k, n] fp16 backed by a physical [n, k] row-major buffer on host (i.e. B^T contiguous)
    - C: ND [m, n] fp32
    """
    if grid_m <= 0 or grid_n <= 0:
        raise ValueError("grid_m/grid_n must be > 0")
    if base_m <= 0 or base_k <= 0 or base_n <= 0:
        raise ValueError("base_m/base_k/base_n must be > 0")
    block_dim = int(grid_m) * int(grid_n)
    if (m % (grid_m * base_m)) != 0 or (n % (grid_n * base_n)) != 0 or (k % base_k) != 0:
        raise ValueError(
            f"shape must be divisible by grid/base tiles "
            f"(m%{grid_m*base_m}==0, n%{grid_n*base_n}==0, k%{base_k}==0)"
        )

    single_core_m = m // grid_m
    single_core_n = n // grid_n

    return compile_kernel_spec(
        gemm_performance,
        consts={
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


def gemm_performance():
    pto = PTO("gemm_performance")
    pto.prologue()
    bid = pto.get_block_idx()

    A = pto.tensor(dtype="f16", shape=(m, k), role="in")
    # DN tensor backed by a physical [n, k] row-major buffer (host passes B^T contiguous).
    B = pto.tensor(dtype="f16", shape=(k, n), stride=(1, k), layout="DN", role="in")
    C = pto.tensor(dtype="f32", shape=(m, n), role="out")

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

    # Derived compile-time geometry.
    single_core_m = m // grid_m
    single_core_n = n // grid_n
    m_loop = single_core_m // base_m
    n_loop = single_core_n // base_n
    k_tiles = k // base_k

    m_core = bid % grid_m
    n_core = bid // grid_m

    if n_core < grid_n:
        base_m0 = m_core * single_core_m
        base_n0 = n_core * single_core_n

        for mi in range(0, m_loop):
            mi_off = mi * base_m
            m0 = base_m0 + mi_off
            for nj in range(0, n_loop):
                nj_off = nj * base_n
                n0 = base_n0 + nj_off

                # Prime buffer 0 (k0=0).
                a_mat0 = pto.load(A, m0, 0)
                b_mat0 = pto.load(B, 0, n0)
                a0 = pto.mov(a_mat0)
                b0 = pto.mov(b_mat0)

                # Also prime buffer 1 (k0=base_k) so event insertion can conservatively
                # synchronize both ping-pong buffers through control-flow merges.
                if k_tiles > 1:
                    a_mat1 = pto.load(A, m0, base_k)
                    b_mat1 = pto.load(B, base_k, n0)
                    a1 = pto.mov(a_mat1)
                    b1 = pto.mov(b_mat1)

                # Main K loop with software pipelining:
                # - Prefetch (k+1) into the other buffer
                # - Compute current buffer
                for kt in range(0, k_tiles):
                    k_next_t = kt + 1
                    if k_next_t < k_tiles:
                        k1 = k_next_t * base_k
                        lane = kt % 2
                        if lane == 0:
                            a_mat1 = pto.load(A, m0, k1)
                            b_mat1 = pto.load(B, k1, n0)
                            a1 = pto.mov(a_mat1)
                            b1 = pto.mov(b_mat1)
                        else:
                            a_mat0 = pto.load(A, m0, k1)
                            b_mat0 = pto.load(B, k1, n0)
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

                pto.store(C, m0, n0, c)

    pto.epilogue()
    return pto.program()
