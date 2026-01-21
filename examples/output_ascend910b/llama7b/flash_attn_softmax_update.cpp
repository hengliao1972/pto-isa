// PTO Program: flash_attn_softmax_update
// Function Type: InCore (single-core tile computation)
// Execution Mode: Single-Core (SPSD) - NOT SPMD kernel
// This function is scheduled as a task by PTO Runtime
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_softmax_update
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     12
//   Total capacity (no reuse): 51,456 bytes (50.2 KB)
//   Total capacity (w/ reuse): 34,048 bytes (33.2 KB)
//   Reuse savings:            17,408 bytes (33.8%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   l_new                64x1       f32       256   [ 11,  13]           -
//   l_prev               64x1       f32       256   [  2,   9]           -
//   l_scaled             64x1       f32       256   [  9,  11]           <- m_diff
//   m_cur                64x1       f32       256   [  3,   4]           -
//   m_diff               64x1       f32       256   [  7,   8]           <- m_cur
//   m_new                64x1       f32       256   [  4,  12]           -
//   m_prev               64x1       f32       256   [  1,   7]           -
//   p_block              64x64      f32     16384   [  6,  14]           <- s_block
//   p_rowsum             64x1       f32       256   [ 10,  11]           <- l_prev
//   s_block              64x64      f32     16384   [  0,   5]           -
//   s_shifted            64x64      f32     16384   [  5,   6]           -
//   scale_old            64x1       f32       256   [  8,  15]           <- m_prev
//
// BUFFER REUSE MAP:
//   p_block reuses buffer of s_block
//   scale_old reuses buffer of m_prev
//   m_diff reuses buffer of m_cur
//   l_scaled reuses buffer of m_diff
//   p_rowsum reuses buffer of l_prev
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

/**
 * InCore Function: flash_attn_softmax_update
 * Single-core tile computation function.
 * Called by PTO Runtime as a scheduled task.
 * NOT a kernel entry - use flash_attn_softmax_update_kernel_wrapper() to launch as kernel.
 */
class flash_attn_softmax_updateInCore {
public:
    // Single-core constructor - no block coordination
    __aicore__ inline flash_attn_softmax_updateInCore() {}

    // Initialize with global memory pointers
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
        inputGm.SetGlobalBuffer((__gm__ float*)input);
        outputGm.SetGlobalBuffer((__gm__ float*)output);
        pipe.InitBuffer(inQueueX, 1, 51456);
        pipe.InitBuffer(outQueueY, 1, 51456);
    }

    // Main processing - single tile, single core
    __aicore__ inline void Process() {
        CopyIn(); Compute(); CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, inputGm, 12864);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // Loop fusion: 6 loop overheads saved

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // FUSED (2 ops): TLOAD; TLOAD
        // TLOAD: Operation
        // TLOAD: Operation

        // TROWMAX: reduction max operation
        ReduceMax(m_cur, s_block, 64);

        // FUSED (1 ops): TMAX
        Max(m_new, m_prev, m_cur, 64);

        // FUSED (2 ops): TROWEXPANDSUB; TEXP
        BroadcastSub(s_shifted, s_block, m_new, 64, 8);  // row-wise broadcast subtract
        Exp(p_block, s_shifted, 64);

        // FUSED (3 ops): TSUB; TEXP; TMUL
        Sub(m_diff, m_prev, m_new, 64);
        Exp(scale_old, m_diff, 64);
        Mul(l_scaled, scale_old, l_prev, 64);

        // TROWSUM: reduction operation
        ReduceSum(p_rowsum, p_block, 64);

        // FUSED (3 ops): TADD; TSTORE; TSTORE
        Add(l_new, l_scaled, p_rowsum, 64);
        // TSTORE: Operation
        // TSTORE: Operation

        // FUSED (1 ops): TSTORE
        // TSTORE: Operation

        // FUSED (1 ops): TSTORE
        // TSTORE: Operation

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(outputGm, yLocal, 12864);
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
__aicore__ inline void flash_attn_softmax_update(
    GM_ADDR input, int32_t in_row_off, int32_t in_col_off,
    GM_ADDR output, int32_t out_row_off, int32_t out_col_off,
    int32_t tile_rows, int32_t tile_cols)
{
    // Calculate byte offsets for this tile
    int32_t in_offset = (in_row_off * tile_cols + in_col_off) * sizeof(float);
    int32_t out_offset = (out_row_off * tile_cols + out_col_off) * sizeof(float);
    
    flash_attn_softmax_updateInCore op;
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
extern "C" __global__ __aicore__ void flash_attn_softmax_update_kernel(GM_ADDR input, GM_ADDR output) {
    flash_attn_softmax_updateInCore op;
    op.Init(input, output);
    op.Process();
}
#endif  // PTO_GENERATE_SPMD_KERNEL