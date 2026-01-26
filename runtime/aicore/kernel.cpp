/**
 * Minimal AICore Kernel with PTO Support
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "../graph/handshake.h"
#include "../graph/graph.h"

#if defined(__CCE__) && defined(__clang__)
#include <cce_aicore_intrinsics.h>
#endif

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

__aicore__ __attribute__((always_inline)) static inline void Barrier() {
#if defined(__CCE_KT_TEST__) && __CCE_KT_TEST__ == 1
    __asm__ __volatile__("" ::: "memory");
#else
    __asm__ __volatile__("");
#endif
}

__aicore__ __attribute__((always_inline)) static inline void RefreshHandshake(__gm__ volatile Handshake *hank) {
    Barrier();
    dcci((__gm__ void *)hank, SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    Barrier();
}

__aicore__ __attribute__((always_inline)) static inline void FlushHandshake(__gm__ volatile Handshake *hank) {
    Barrier();
    dcci((__gm__ void *)hank, SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    Barrier();
}

__aicore__ __attribute__((always_inline)) static inline uint64_t ReadDeviceClock() {
#if defined(__CCE__) && defined(__clang__)
    // Raw system counter ticks exposed by the CCE toolchain.
    return static_cast<uint64_t>(get_sys_cnt());
#else
    return 0;
#endif
}

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
 * Unified function pointer type for kernel dispatch
 *
 * All kernels follow the same signature: void kernel(__gm__ int64_t* args)
 * This enables simple, switch-free dispatch.
 */
typedef void (*UnifiedKernelFunc)(__gm__ int64_t*);

/**
 * Task execution wrapper - dispatches tasks using function pointers
 *
 * This function demonstrates the runtime function pointer dispatch pattern.
 * Following the production system flow:
 * - functionBinAddr points to compiled kernel code in device GM memory
 * - The address is cast to a function pointer: UnifiedKernelFunc kernel = (UnifiedKernelFunc)functionBinAddr
 * - The kernel is invoked: kernel(task->args)
 *
 * This is the KEY difference from compile-time linking:
 * - OLD: extern "C" declarations, resolved at link time
 * - NEW: functionBinAddr from GM memory, cast at runtime
 *
 * With unified kernel signature, no switch statement is needed.
 * All kernels unpack their own arguments from the args array.
 *
 * @param task Pointer to task in global memory (null during initialization)
 */
__aicore__ __attribute__((always_inline)) static void execute_task(__gm__ Task* task)
{
    // Null task pointer indicates no work assigned (initialization state)
    if (task == nullptr) {
        return;
    }

    // Check for valid functionBinAddr
    if (task->functionBinAddr == 0) {
        // Invalid address - skip execution
        return;
    }

    // Cast functionBinAddr to unified function pointer and invoke
    // All kernels have signature: void kernel(__gm__ int64_t* args)
    UnifiedKernelFunc kernel = (UnifiedKernelFunc)task->functionBinAddr;
    kernel(reinterpret_cast<__gm__ int64_t*>(task->args));
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
    __gm__ volatile Handshake* my_hank = &hank[blockIdx];

    // Phase 1: Wait for AICPU initialization signal
    while (my_hank->aicpu_ready == 0) {
        RefreshHandshake(my_hank);
    }

    // Phase 2: Signal AICore is ready (use core_id + 1 to avoid 0)
    my_hank->aicore_done = blockIdx + 1;
    FlushHandshake(my_hank);

    // Phase 3: Main execution loop - poll for tasks until quit signal
    while (true) {
        RefreshHandshake(my_hank);

        // Check for quit command from AICPU
        if (my_hank->control == 1) {
            break;  // Exit kernel
        }

        // Execute task if assigned (task != 0 means valid Task* pointer)
        if (my_hank->task != 0 && my_hank->task_status != 0) {
            __gm__ Task* task_ptr = reinterpret_cast<__gm__ Task*>(my_hank->task);
            if (my_hank->profile_enable != 0) {
                // Record per-task profiling (best-effort; visible to host via Graph copy-back).
                // NOTE: We flush exactly one cache line (TaskProfile is 64B-aligned and 64B-sized).
                task_ptr->profile.exec_core_id = static_cast<uint32_t>(blockIdx);
    #ifdef __AIV__
                task_ptr->profile.exec_core_type = 2;
    #else
                task_ptr->profile.exec_core_type = 1;
    #endif
                task_ptr->profile.exec_phys_core_id = static_cast<uint32_t>(get_coreid());
                const uint64_t t0 = ReadDeviceClock();
                task_ptr->profile.start_time = t0;
                task_ptr->profile.end_time = 0;
                execute_task(task_ptr);
                const uint64_t t1 = ReadDeviceClock();
                task_ptr->profile.end_time = t1;
                // For convenience, store the duration ticks in pmu_cnt[0] (clipped to 32-bit).
                task_ptr->profile.pmu_cnt[0] = static_cast<uint32_t>(t1 - t0);
                Barrier();
                dcci((__gm__ void *)&task_ptr->profile, SINGLE_CACHE_LINE);
                dsb(DSB_DDR);
                Barrier();
            } else {
                execute_task(task_ptr);
            }
            // Mark task as complete (task_status: 0=idle, 1=busy)
            my_hank->task_status = 0;
            FlushHandshake(my_hank);
        }
    }
}
