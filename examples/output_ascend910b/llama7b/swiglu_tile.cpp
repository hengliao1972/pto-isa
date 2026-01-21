// PTO Program: swiglu_tile
// Function Type: InCore (single-core tile computation)
// Execution Mode: Single-Core (SPSD) - NOT SPMD kernel
// This function is scheduled as a task by PTO Runtime
// ======================================================================
// TILE BUFFER ANALYSIS: swiglu_tile
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 131,072 bytes (128.0 KB)
//   Total capacity (w/ reuse): 65,536 bytes (64.0 KB)
//   Reuse savings:            65,536 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_neg_gate         32x128     f32     16384   [  3,   4]           -
//   gate                 32x128     f32     16384   [  0,   6]           -
//   gate_silu            32x128     f32     16384   [  6,   7]           <- one_plus_exp
//   neg_gate             32x128     f32     16384   [  2,   3]           -
//   one_plus_exp         32x128     f32     16384   [  4,   5]           <- neg_gate
//   result               32x128     f32     16384   [  7,   8]           <- gate
//   sigmoid_gate         32x128     f32     16384   [  5,   6]           <- exp_neg_gate
//   up                   32x128     f32     16384   [  1,   7]           -
//
// BUFFER REUSE MAP:
//   one_plus_exp reuses buffer of neg_gate
//   sigmoid_gate reuses buffer of exp_neg_gate
//   gate_silu reuses buffer of one_plus_exp
//   result reuses buffer of gate
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

/**
 * InCore Function: swiglu_tile
 * Single-core tile computation function.
 * Called by PTO Runtime as a scheduled task.
 * NOT a kernel entry - use swiglu_tile_kernel_wrapper() to launch as kernel.
 */
class swiglu_tileInCore {
public:
    // Single-core constructor - no block coordination
    __aicore__ inline swiglu_tileInCore() {}

    // Initialize with global memory pointers
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
        inputGm.SetGlobalBuffer((__gm__ float*)input);
        outputGm.SetGlobalBuffer((__gm__ float*)output);
        pipe.InitBuffer(inQueueX, 1, 131072);
        pipe.InitBuffer(outQueueY, 1, 131072);
    }

    // Main processing - single tile, single core
    __aicore__ inline void Process() {
        CopyIn(); Compute(); CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, inputGm, 32768);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // Loop fusion: 8 loop overheads saved

        // FUSED (9 ops): TLOAD; TLOAD; TNEG; TEXP; TADDS; TRECIP; TMUL; TMUL; TSTORE
        // TLOAD: Operation
        // TLOAD: Operation
        Neg(neg_gate, gate, 64);
        Exp(exp_neg_gate, neg_gate, 64);
        Adds(one_plus_exp, exp_neg_gate, 1.0f, 64);
        Reciprocal(sigmoid_gate, one_plus_exp, 64);
        Mul(gate_silu, gate, sigmoid_gate, 64);
        Mul(result, gate_silu, up, 64);
        // TSTORE: Operation

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(outputGm, yLocal, 32768);
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
__aicore__ inline void swiglu_tile(
    GM_ADDR input, int32_t in_row_off, int32_t in_col_off,
    GM_ADDR output, int32_t out_row_off, int32_t out_col_off,
    int32_t tile_rows, int32_t tile_cols)
{
    // Calculate byte offsets for this tile
    int32_t in_offset = (in_row_off * tile_cols + in_col_off) * sizeof(float);
    int32_t out_offset = (out_row_off * tile_cols + out_col_off) * sizeof(float);
    
    swiglu_tileInCore op;
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
extern "C" __global__ __aicore__ void swiglu_tile_kernel(GM_ADDR input, GM_ADDR output) {
    swiglu_tileInCore op;
    op.Init(input, output);
    op.Process();
}
#endif  // PTO_GENERATE_SPMD_KERNEL