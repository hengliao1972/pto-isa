// PTO Program: rmsnorm_tile
// Function Type: InCore (single-core tile computation)
// Execution Mode: Single-Core (SPSD) - NOT SPMD kernel
// This function is scheduled as a task by PTO Runtime
// ======================================================================
// TILE BUFFER ANALYSIS: rmsnorm_tile
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 82,304 bytes (80.4 KB)
//   Total capacity (w/ reuse): 49,408 bytes (48.2 KB)
//   Reuse savings:            32,896 bytes (40.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   gamma                32x128     f32     16384   [  1,  10]           -
//   result               32x128     f32     16384   [ 10,  11]           <- x
//   row_mean             32x1       f32       128   [  5,   8]           -
//   row_rsqrt            32x1       f32       128   [  8,   9]           <- row_sum
//   row_sum              32x1       f32       128   [  3,   5]           -
//   x                    32x128     f32     16384   [  0,   9]           -
//   x_norm               32x128     f32     16384   [  9,  10]           <- x_sq
//   x_sq                 32x128     f32     16384   [  2,   3]           -
//
// BUFFER REUSE MAP:
//   row_rsqrt reuses buffer of row_sum
//   x_norm reuses buffer of x_sq
//   result reuses buffer of x
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

/**
 * InCore Function: rmsnorm_tile
 * Single-core tile computation function.
 * Called by PTO Runtime as a scheduled task.
 * NOT a kernel entry - use rmsnorm_tile_kernel_wrapper() to launch as kernel.
 */
class rmsnorm_tileInCore {
public:
    // Single-core constructor - no block coordination
    __aicore__ inline rmsnorm_tileInCore() {}

    // Initialize with global memory pointers
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
        inputGm.SetGlobalBuffer((__gm__ float*)input);
        outputGm.SetGlobalBuffer((__gm__ float*)output);
        pipe.InitBuffer(inQueueX, 1, 82304);
        pipe.InitBuffer(outQueueY, 1, 82304);
    }

    // Main processing - single tile, single core
    __aicore__ inline void Process() {
        CopyIn(); Compute(); CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, inputGm, 20576);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // Loop fusion: 5 loop overheads saved

        // FUSED (3 ops): TLOAD; TLOAD; TMUL
        // TLOAD: Operation
        // TLOAD: Operation
        Mul(x_sq, x, x, 64);

        // TROWSUM: reduction operation
        ReduceSum(row_sum, x_sq, 32);

        int inv_cols = 0.0078125;

        // FUSED (1 ops): TMULS
        Muls(row_mean, row_sum, inv_colsf, 64);

        int eps = 1e-05;

        // FUSED (2 ops): TADDS; TRSQRT
        Adds(row_mean, row_mean, epsf, 64);
        Rsqrt(row_rsqrt, row_mean, 64);

        // FUSED (3 ops): TROWEXPANDMUL; TMUL; TSTORE
        BroadcastMul(x_norm, x, row_rsqrt, 64, 8);  // row-wise broadcast multiply
        Mul(result, x_norm, gamma, 64);
        // TSTORE: Operation

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(outputGm, yLocal, 20576);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    GlobalTensor<float> inputGm;
    GlobalTensor<float> outputGm;
};

/**
 * Callable InCore function for PTO Runtime task scheduling.
 * This function is invoked by the runtime when this task is dispatched.
 * Execution: Single AI Core, single tile at specified offset.
 */
__aicore__ inline void rmsnorm_tile(
    GM_ADDR input, int32_t in_row_off, int32_t in_col_off,
    GM_ADDR output, int32_t out_row_off, int32_t out_col_off,
    int32_t tile_rows, int32_t tile_cols)
{
    // Calculate byte offsets for this tile
    int32_t in_offset = (in_row_off * tile_cols + in_col_off) * sizeof(float);
    int32_t out_offset = (out_row_off * tile_cols + out_col_off) * sizeof(float);
    
    rmsnorm_tileInCore op;
    op.Init((GM_ADDR)((uint8_t*)input + in_offset), 
            (GM_ADDR)((uint8_t*)output + out_offset));
    op.Process();
}

#ifdef PTO_GENERATE_SPMD_KERNEL
/**
 * SPMD Kernel Wrapper (for standalone testing only)
 * This launches the InCore function as a multi-core kernel.
 * In production, use PTO Runtime to schedule tasks instead.
 */
extern "C" __global__ __aicore__ void rmsnorm_tile_kernel(GM_ADDR input, GM_ADDR output) {
    rmsnorm_tileInCore op;
    op.Init(input, output);
    op.Process();
}
#endif  // PTO_GENERATE_SPMD_KERNEL