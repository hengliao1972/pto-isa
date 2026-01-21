// PTO Program: softmax_tile
// Function Type: InCore (single-core tile computation)
// Execution Mode: Single-Core (SPSD) - NOT SPMD kernel
// This function is scheduled as a task by PTO Runtime
// ======================================================================
// TILE BUFFER ANALYSIS: softmax_tile
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 65,792 bytes (64.2 KB)
//   Total capacity (w/ reuse): 32,896 bytes (32.1 KB)
//   Reuse savings:            32,896 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_x                32x128     f32     16384   [  3,   5]           <- x
//   result               32x128     f32     16384   [  5,   6]           <- x_shifted
//   row_max              32x1       f32       128   [  1,   2]           -
//   row_sum              32x1       f32       128   [  4,   5]           <- row_max
//   x                    32x128     f32     16384   [  0,   2]           -
//   x_shifted            32x128     f32     16384   [  2,   3]           -
//
// BUFFER REUSE MAP:
//   exp_x reuses buffer of x
//   row_sum reuses buffer of row_max
//   result reuses buffer of x_shifted
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

/**
 * InCore Function: softmax_tile
 * Single-core tile computation function.
 * Called by PTO Runtime as a scheduled task.
 * NOT a kernel entry - use softmax_tile_kernel_wrapper() to launch as kernel.
 */
class softmax_tileInCore {
public:
    // Single-core constructor - no block coordination
    __aicore__ inline softmax_tileInCore() {}

    // Initialize with global memory pointers
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
        inputGm.SetGlobalBuffer((__gm__ float*)input);
        outputGm.SetGlobalBuffer((__gm__ float*)output);
        pipe.InitBuffer(inQueueX, 1, 65792);
        pipe.InitBuffer(outQueueY, 1, 65792);
    }

    // Main processing - single tile, single core
    __aicore__ inline void Process() {
        CopyIn(); Compute(); CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, inputGm, 16448);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // Loop fusion: 2 loop overheads saved

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // TROWMAX: reduction max operation
        ReduceMax(row_max, x, 32);

        // FUSED (2 ops): TROWEXPANDSUB; TEXP
        BroadcastSub(x_shifted, x, row_max, 64, 8);  // row-wise broadcast subtract
        Exp(exp_x, x_shifted, 64);

        // TROWSUM: reduction operation
        ReduceSum(row_sum, exp_x, 32);

        // FUSED (2 ops): TROWEXPANDDIV; TSTORE
        BroadcastDiv(result, exp_x, row_sum, 64, 8);  // row-wise broadcast divide
        // TSTORE: Operation

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(outputGm, yLocal, 16448);
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
__aicore__ inline void softmax_tile(
    GM_ADDR input, int32_t in_row_off, int32_t in_col_off,
    GM_ADDR output, int32_t out_row_off, int32_t out_col_off,
    int32_t tile_rows, int32_t tile_cols)
{
    // Calculate byte offsets for this tile
    int32_t in_offset = (in_row_off * tile_cols + in_col_off) * sizeof(float);
    int32_t out_offset = (out_row_off * tile_cols + out_col_off) * sizeof(float);
    
    softmax_tileInCore op;
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
extern "C" __global__ __aicore__ void softmax_tile_kernel(GM_ADDR input, GM_ADDR output) {
    softmax_tileInCore op;
    op.Init(input, output);
    op.Process();
}
#endif  // PTO_GENERATE_SPMD_KERNEL