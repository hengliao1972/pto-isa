from __future__ import annotations

from pto import PTO, KernelSpec, compile_kernel_spec


def make_bgemm_kernel(
    *,
    batch: int,
    m: int,
    k: int,
    n: int,
    grid_m: int,
    grid_n: int,
    base_m: int = 128,
    base_k: int = 64,
    base_n: int = 256,
) -> KernelSpec:
    """
    Batched GEMM (BGEMM) built on the `bgemm_performance` kernel structure from the original repo.

    Conventions:
    - A and C are packed by batch along the row dimension:
        A: [batch*m, k] (ND)
        C: [batch*m, n] (ND)
      The i-th batch owns rows [i*m, (i+1)*m).
    - B is also batched and packed by batch along the row dimension:
        B: [batch*k, n] (ND)
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
        bgemm_kernel,
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


def bgemm_kernel():
    pto = PTO("bgemm_kernel")
    pto.prologue()

    # Task identifier:
    # - runtime/MPMD scheduler path: provided by orchestration as args[0]
    # - direct block launch path: lowered to get_block_idx()
    bid = pto.get_task_id()

    # Reserve %arg0 for orchestration-provided task id; tensors start at %arg1.
    A = pto.tensor(dtype="f16", shape=(batch * m, k), role="in", arg=1)
    # DN tensor backed by a physical [n, batch*k] row-major buffer (host passes per-batch B^T contiguous).
    B = pto.tensor(dtype="f16", shape=(batch * k, n), stride=(1, batch * k), layout="DN", role="in", arg=2)
    C = pto.tensor(dtype="f32", shape=(batch * m, n), role="out", arg=3)

    # Conservative single-buffer kernel:
    # - Avoids complex control-flow in the K loop.
    # - Relies on `ptoas --insert-events` for cross-pipe dependencies.
    a_mat = pto.mat(dtype="f16", shape=(base_m, base_k))
    # B staging tile.
    #
    # B is passed as DN (host provides B^T physical row-major), so tload uses the DN->ZN path.
    # Keep this as Mat(ZN) to match the Right tile layout and avoid transpose TMOV.
    b_mat = pto.mat(dtype="f16", shape=(base_k, base_n), blayout="RowMajor", slayout="ColMajor")

    # NOTE: On Ascend A2/A3 (MEMORY_BASE), the cube core expects L0A (Left) tiles
    # in RowMajor/RowMajor (see `include/pto/common/pto_tile.hpp` TileLeft alias).
    # Do not force ColMajor here (that matches REGISTER_BASE / CPU sim).
    a_left = pto.left(dtype="f16", shape=(base_m, base_k))
    b_right = pto.right(dtype="f16", shape=(base_k, base_n))

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

                    # First K tile (k0 = 0): matmul initializes the accumulator.
                    a_mat = pto.load(A, m0, 0)
                    b_mat = pto.load(B, bk_off, n0)
                    a_left = pto.mov(a_mat)
                    b_right = pto.mov(b_mat)
                    c = pto.matmul(a_left, b_right)

                    # Remaining K tiles: accumulate.
                    for kt in range(1, k_tiles):
                        k0 = kt * base_k
                        a_mat = pto.load(A, m0, k0)
                        b_k0 = bk_off + k0
                        b_mat = pto.load(B, b_k0, n0)
                        a_left = pto.mov(a_mat)
                        b_right = pto.mov(b_mat)
                        c = pto.matmul_acc(c, a_left, b_right)

                    pto.store(C, m0, n0, c)

    pto.epilogue()
    return pto.program()
