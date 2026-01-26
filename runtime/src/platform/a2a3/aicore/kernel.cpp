/**
 * Minimal AICore Kernel
 */

 #include <cstdint>
 #include "graph.h"

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
