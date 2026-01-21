// PTO Program: flash_attn_output_update
// Function Type: InCore (single-core tile computation)
// Execution Mode: Single-Core (SPSD) - NOT SPMD kernel
// This function is scheduled as a task by PTO Runtime
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_output_update
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 180,480 bytes (176.2 KB)
//   Total capacity (w/ reuse): 147,712 bytes (144.2 KB)
//   Reuse savings:            32,768 bytes (18.2%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   o_new                64x128     f32     32768   [  6,   7]           -
//   o_prev               64x128     f32     32768   [  0,   4]           -
//   o_scaled             64x128     f32     32768   [  4,   6]           -
//   p_block              64x64      f32     16384   [  1,  -1]           -
//   pv                   64x128     f32     32768   [  5,   6]           <- o_prev
//   scale_old            64x1       f32       256   [  3,   4]           -
//   v_block              64x128     f32     32768   [  2,  -1]           -
//
// BUFFER REUSE MAP:
//   pv reuses buffer of o_prev
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

/**
 * InCore Function: flash_attn_output_update
 * Single-core tile computation function.
 * Called by PTO Runtime as a scheduled task.
 * NOT a kernel entry - use flash_attn_output_update_kernel_wrapper() to launch as kernel.
 */
class flash_attn_output_updateInCore {
public:
    // Single-core constructor - no block coordination
    __aicore__ inline flash_attn_output_updateInCore() {}

    // Initialize with global memory pointers
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
        inputGm.SetGlobalBuffer((__gm__ float*)input);
        outputGm.SetGlobalBuffer((__gm__ float*)output);
        pipe.InitBuffer(inQueueX, 1, 180480);
        pipe.InitBuffer(outQueueY, 1, 180480);
    }

    // Main processing - single tile, single core
    __aicore__ inline void Process() {
        CopyIn(); Compute(); CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, inputGm, 45120);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // Loop fusion: 1 loop overheads saved

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // FUSED (1 ops): TROWEXPANDMUL
        BroadcastMul(o_scaled, o_prev, scale_old, 64, 8);  // row-wise broadcast multiply

        // TMATMUL: pv = p_block @ v_block
        Matmul(pv, p_block, v_block, 64, 128);

        // FUSED (2 ops): TADD; TSTORE
        Add(o_new, o_scaled, pv, 64);
        // TSTORE: Operation

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(outputGm, yLocal, 45120);
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
__aicore__ inline void flash_attn_output_update(
    GM_ADDR input, int32_t in_row_off, int32_t in_col_off,
    GM_ADDR output, int32_t out_row_off, int32_t out_col_off,
    int32_t tile_rows, int32_t tile_cols)
{
    // Calculate byte offsets for this tile
    int32_t in_offset = (in_row_off * tile_cols + in_col_off) * sizeof(float);
    int32_t out_offset = (out_row_off * tile_cols + out_col_off) * sizeof(float);
    
    flash_attn_output_updateInCore op;
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
extern "C" __global__ __aicore__ void flash_attn_output_update_kernel(GM_ADDR input, GM_ADDR output) {
    flash_attn_output_updateInCore op;
    op.Init(input, output);
    op.Process();
}
#endif  // PTO_GENERATE_SPMD_KERNEL