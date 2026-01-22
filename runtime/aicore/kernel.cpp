/**
 * Minimal AICore Kernel with PTO Support
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "../graph/handshake.h"
#include "../graph/graph.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#ifndef __in__
#define __in__
#endif

#ifndef __out__
#define __out__
#endif

#ifdef __AIV__
#define KERNEL_ENTRY(x) x##_0_mix_aiv   // 动态生成函数名 KERNEL_ENTRY(my_kernel) -> my_kernel_0_mix_aiv
#define blockIdx blockIdx_aiv
#else
#define KERNEL_ENTRY(x) x##_0_mix_aic
#define blockIdx blockIdx_aic
#endif

[[block_local]] int blockIdx;

using namespace pto;

// TADD implementation (float path)
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__aicore__ __attribute__((always_inline)) void runTAdd(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(dstTile, src0Tile, src1Tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

/**
 * Simple kernel: Add two tensors element-wise
 * Computes: out[i] = src0[i] + src1[i]
 */
__aicore__ __attribute__((always_inline)) static void kernel_add(__gm__ float* out, __gm__ float* src0, __gm__ float* src1, int size)
{
    for (int i = 0; i < size; i++) {
        out[i] = src0[i] + src1[i];
    }
}

/**
 * Simple kernel: Add scalar to tensor
 * Computes: out[i] = src[i] + scalar
 */
__aicore__ __attribute__((always_inline)) static void kernel_add_scalar(__gm__ float* out, __gm__ float* src, float scalar, int size)
{
    for (int i = 0; i < size; i++) {
        out[i] = src[i] + scalar;
    }
}

/**
 * Simple kernel: Multiply two tensors element-wise
 * Computes: out[i] = src0[i] * src1[i]
 */
__aicore__ __attribute__((always_inline)) static void kernel_mul(__gm__ float* out, __gm__ float* src0, __gm__ float* src1, int size)
{
    for (int i = 0; i < size; i++) {
        out[i] = src0[i] * src1[i];
    }
}



/**
 * Task execution wrapper - dispatches tasks based on function ID
 *
 * This function executes a task on the AICore. The task pointer is provided
 * by AICPU through the handshake buffer and contains:
 * - func_id: identifies which function to execute (0 = TADD, etc.)
 * - args: function-specific arguments
 * - dependency metadata (fanin/fanout)
 *
 * @param task Pointer to task in global memory (null during initialization)
 */
__aicore__ __attribute__((always_inline)) static void execute_task(__gm__ Task* task)
{
    // Null task pointer indicates no work assigned (initialization state)
    if (task == nullptr) {
        return;
    }

    // Helper union to convert uint64_t to float for scalar arguments
    union {
        uint64_t u64;
        float f32;
    } converter;

    // Dispatch to specific function based on task->func_id
    switch(task->func_id) {
        case 0:  // kernel_add: out = src0 + src1
            {
                __gm__ float* src0 = reinterpret_cast<__gm__ float*>(task->args[0]);
                __gm__ float* src1 = reinterpret_cast<__gm__ float*>(task->args[1]);
                __gm__ float* out = reinterpret_cast<__gm__ float*>(task->args[2]);
                int size = static_cast<int>(task->args[3]);
                kernel_add(out, src0, src1, size);
            }
            break;
        case 1:  // kernel_add_scalar: out = src + scalar
            {
                __gm__ float* src = reinterpret_cast<__gm__ float*>(task->args[0]);
                converter.u64 = task->args[1];
                float scalar = converter.f32;
                __gm__ float* out = reinterpret_cast<__gm__ float*>(task->args[2]);
                int size = static_cast<int>(task->args[3]);
                kernel_add_scalar(out, src, scalar, size);
            }
            break;
        case 2:  // kernel_mul: out = src0 * src1
            {
                __gm__ float* src0 = reinterpret_cast<__gm__ float*>(task->args[0]);
                __gm__ float* src1 = reinterpret_cast<__gm__ float*>(task->args[1]);
                __gm__ float* out = reinterpret_cast<__gm__ float*>(task->args[2]);
                int size = static_cast<int>(task->args[3]);
                kernel_mul(out, src0, src1, size);
            }
            break;
        default:
            // Unknown function ID - skip execution
            break;
    }
}

/**
 * Kernel entry point with control loop
 *
 * This function implements the AICore-side task execution protocol:
 * 1. Wait for AICPU ready signal (handshake initialization)
 * 2. Signal AICore is ready (aicore_done = core_id + 1)
 * 3. Enter polling loop:
 *    - Check control flag (1 = quit, 0 = continue)
 *    - If task pointer is non-zero, execute task and mark as complete
 *    - Use DCCI to ensure cache coherency with AICPU
 *
 * Each core (AIC or AIV) gets its own handshake buffer indexed by blockIdx.
 *
 * @param hank Array of handshake buffers (one per core)
 */
extern "C" __global__ __aicore__ void KERNEL_ENTRY(aicore_kernel)(__gm__ struct Handshake* hank) {
    // Calculate blockIdx for this core
#ifdef __AIV__
    blockIdx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
#else
    blockIdx = get_block_idx();
#endif

    // Get this core's handshake buffer
    __gm__ Handshake* my_hank = &hank[blockIdx];

    // Phase 1: Wait for AICPU initialization signal
    while (my_hank->aicpu_ready == 0) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    }

    // Phase 2: Signal AICore is ready (use core_id + 1 to avoid 0)
    my_hank->aicore_done = blockIdx + 1;

    // Phase 3: Main execution loop - poll for tasks until quit signal
    while (true) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);

        // Check for quit command from AICPU
        if (my_hank->control == 1) {
            break;  // Exit kernel
        }

        // Execute task if assigned (task != 0 means valid Task* pointer)
        if (my_hank->task != 0) {
            __gm__ Task* task_ptr = reinterpret_cast<__gm__ Task*>(my_hank->task);
            execute_task(task_ptr);
            // Mark task as complete (task_status: 0=idle, 1=busy)
            my_hank->task_status = 0;
        }
    }
}
