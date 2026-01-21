// PTO Program: swiglu_tile_64
// Function Type: InCore (single-core tile computation)
// Execution Mode: Single-Core (SPSD) - NOT SPMD kernel
// This function is scheduled as a task by PTO Runtime
// ======================================================================
// TILE BUFFER ANALYSIS: swiglu_tile_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 262,144 bytes (256.0 KB)
//   Total capacity (w/ reuse): 131,072 bytes (128.0 KB)
//   Reuse savings:            131,072 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_neg_gate         64x128     f32     32768   [  3,   4]           -
//   gate                 64x128     f32     32768   [  0,   6]           -
//   gate_silu            64x128     f32     32768   [  6,   7]           <- one_plus_exp
//   neg_gate             64x128     f32     32768   [  2,   3]           -
//   one_plus_exp         64x128     f32     32768   [  4,   5]           <- neg_gate
//   result               64x128     f32     32768   [  7,   8]           <- gate
//   sigmoid_gate         64x128     f32     32768   [  5,   6]           <- exp_neg_gate
//   up                   64x128     f32     32768   [  1,   7]           -
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
 * InCore Function: swiglu_tile_64
 * Single-core tile computation function.
 * Called by PTO Runtime as a scheduled task.
 * NOT a kernel entry - use swiglu_tile_64_kernel_wrapper() to launch as kernel.
 */
class swiglu_tile_64InCore {
public:
    // Single-core constructor - no block coordination
    __aicore__ inline swiglu_tile_64InCore() {}

    // Initialize with global memory pointers
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
        inputGm.SetGlobalBuffer((__gm__ float*)input);
        outputGm.SetGlobalBuffer((__gm__ float*)output);
        pipe.InitBuffer(inQueueX, 1, 262144);
        pipe.InitBuffer(outQueueY, 1, 262144);
    }

    // Main processing - single tile, single core
    __aicore__ inline void Process() {
        CopyIn(); Compute(); CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, inputGm, 65536);
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
        DataCopy(outputGm, yLocal, 65536);
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
__aicore__ inline void swiglu_tile_64(
    GM_ADDR input, int32_t in_row_off, int32_t in_col_off,
    GM_ADDR output, int32_t out_row_off, int32_t out_col_off,
    int32_t tile_rows, int32_t tile_cols)
{
    // Calculate byte offsets for this tile
    int32_t in_offset = (in_row_off * tile_cols + in_col_off) * sizeof(float);
    int32_t out_offset = (out_row_off * tile_cols + out_col_off) * sizeof(float);
    
    swiglu_tile_64InCore op;
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
extern "C" __global__ __aicore__ void swiglu_tile_64_kernel(GM_ADDR input, GM_ADDR output) {
    swiglu_tile_64InCore op;
    op.Init(input, output);
    op.Process();
}
#endif  // PTO_GENERATE_SPMD_KERNEL