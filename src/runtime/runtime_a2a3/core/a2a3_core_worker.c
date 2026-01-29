/**
 * PTO Runtime - Ascend A2/A3 Hardware Core Worker Implementation
 * 
 * This file implements the worker functions for real Ascend A2/A3 hardware.
 * Requires CANN SDK for actual NPU kernel execution.
 */

#define _POSIX_C_SOURCE 199309L
#include "a2a3_core_worker.h"
#include "../orchestration/a2a3_orchestration.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// =============================================================================
// CANN SDK Requirement Check
// =============================================================================

#if defined(A2A3_TARGET_HARDWARE) && !defined(CANN_SDK_AVAILABLE) && !defined(A2A3_SKIP_CANN_CHECK)
#error "=================================================================="
#error "Ascend A2/A3 Hardware Core Worker requires CANN SDK."
#error ""
#error "To compile for real hardware, you need to:"
#error "  1. Install Huawei CANN SDK (version 6.0 or later)"
#error "  2. Set environment: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
#error "  3. Define CANN_SDK_AVAILABLE when compiling"
#error ""
#error "For simulation/testing without hardware, use:"
#error "  - Platform: ascend_a2a3_sim"
#error "  - Or define A2A3_SKIP_CANN_CHECK for stub-only compilation"
#error "=================================================================="
#endif

// =============================================================================
// Task Execution (Hardware Implementation)
// =============================================================================

// Forward declarations for binary loader (to avoid include dependency)
// These are defined in a2a3_binary_loader.c and will be linked at runtime
typedef struct A2A3InCoreBinaryEntry A2A3InCoreBinaryEntry;
extern uint64_t a2a3_get_incore_device_addr(const char* func_name) __attribute__((weak));
extern A2A3InCoreBinaryEntry* a2a3_lookup_incore_binary(const char* func_name) __attribute__((weak));

#ifdef CANN_SDK_AVAILABLE
#include <acl/acl.h>

// Static flag to track if we've warned about NPU execution
static int s_npu_warning_shown = 0;

void a2a3_core_execute_task(PTORuntime* rt, int32_t task_id, int32_t worker_id) {
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    
    DEBUG_PRINT("[A2A3 Core HW] Worker %d executing task %d: %s\n", 
                worker_id, task_id, task->func_name);
    
    // Build argument array
    void* args[PTO_MAX_ARGS * 2];
    int arg_idx = 0;
    
    for (int i = 0; i < task->num_args; i++) {
        TaskArg* arg = &task->args[i];
        float* base_ptr = (float*)arg->region.raw_tensor;
        int64_t offset = arg->region.row_offset * arg->region.cols + arg->region.col_offset;
        args[arg_idx++] = (void*)(base_ptr + offset);
    }
    
    // Priority 1: Use function pointer if available (from .so loading)
    if (task->func_ptr) {
        PTOInCoreFunc func = (PTOInCoreFunc)task->func_ptr;
        func(args, task->num_args);
        return;
    }
    
    // Priority 2: Check if binary is loaded (from .o file)
    // For real NPU execution, we need to use the NPU launcher
    // Note: a2a3_get_incore_device_addr is a weak symbol - check if it's available
    uint64_t device_addr = 0;
    if (a2a3_get_incore_device_addr) {
        device_addr = a2a3_get_incore_device_addr(task->func_name);
    }
    if (device_addr != 0) {
        // Binary is loaded but we can't execute it in CPU worker threads
        // Real execution requires launching AICore kernel via CANN API
        if (!s_npu_warning_shown) {
            printf("\n[A2A3 Core HW] ================================================\n");
            printf("[A2A3 Core HW] WARNING: InCore function '%s' is loaded as .o binary\n", task->func_name);
            printf("[A2A3 Core HW] .o files are AICore binaries and cannot be executed by CPU workers.\n");
            printf("[A2A3 Core HW] \n");
            printf("[A2A3 Core HW] For real NPU execution, the task graph should be:\n");
            printf("[A2A3 Core HW]   1. Built by orchestration function\n");
            printf("[A2A3 Core HW]   2. Copied to device memory\n");
            printf("[A2A3 Core HW]   3. Executed by AICPU + AICore kernels\n");
            printf("[A2A3 Core HW] \n");
            printf("[A2A3 Core HW] Current architecture uses CPU workers for testing.\n");
            printf("[A2A3 Core HW] Task execution is SKIPPED. Output will be zeros.\n");
            printf("[A2A3 Core HW] ================================================\n\n");
            s_npu_warning_shown = 1;
        }
        // Skip execution - the actual computation would be done on NPU
        return;
    }
    
    // No function available
    DEBUG_PRINT("[A2A3 Core HW] WARNING: No executable for task %s\n", task->func_name);
}

#else /* No CANN SDK - stub implementation for compilation testing */

// Static flag to track if we've warned about missing functions
static int s_stub_warning_shown = 0;

void a2a3_core_execute_task(PTORuntime* rt, int32_t task_id, int32_t worker_id) {
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    
    DEBUG_PRINT("[A2A3 Core STUB] Worker %d executing task %d: %s\n", 
                worker_id, task_id, task->func_name);
    
    // Build argument array
    void* args[PTO_MAX_ARGS * 2];
    int arg_idx = 0;
    
    for (int i = 0; i < task->num_args; i++) {
        TaskArg* arg = &task->args[i];
        float* base_ptr = (float*)arg->region.raw_tensor;
        int64_t offset = arg->region.row_offset * arg->region.cols + arg->region.col_offset;
        args[arg_idx++] = (void*)(base_ptr + offset);
    }
    
    // Priority 1: Use function pointer if available (from .so loading)
    if (task->func_ptr) {
        PTOInCoreFunc func = (PTOInCoreFunc)task->func_ptr;
        func(args, task->num_args);
        return;
    }
    
    // Priority 2: Check if binary is loaded
    // In stub mode without CANN SDK, we cannot execute .o files
    // Note: a2a3_lookup_incore_binary is a weak symbol - check if it's available
    int binary_loaded = 0;
    if (a2a3_lookup_incore_binary) {
        A2A3InCoreBinaryEntry* entry = a2a3_lookup_incore_binary(task->func_name);
        binary_loaded = (entry && entry->is_loaded);
    }
    if (binary_loaded) {
        if (!s_stub_warning_shown) {
            printf("\n[A2A3 Core STUB] ================================================\n");
            printf("[A2A3 Core STUB] InCore functions loaded as .o binaries (AICore code).\n");
            printf("[A2A3 Core STUB] These cannot be executed on CPU.\n");
            printf("[A2A3 Core STUB] \n");
            printf("[A2A3 Core STUB] Options to get actual execution:\n");
            printf("[A2A3 Core STUB]   1. Run on real Ascend hardware with NPU launcher\n");
            printf("[A2A3 Core STUB]   2. Use ascend_a2a3_sim platform (compiles to .so)\n");
            printf("[A2A3 Core STUB] \n");
            printf("[A2A3 Core STUB] Task execution is SKIPPED. Output will be zeros.\n");
            printf("[A2A3 Core STUB] ================================================\n\n");
            s_stub_warning_shown = 1;
        }
        return;
    }
    
    // No function available
    DEBUG_PRINT("[A2A3 Core STUB] WARNING: No executable for task %s\n", task->func_name);
}

#endif /* CANN_SDK_AVAILABLE */

// =============================================================================
// Task Completion (Hardware Implementation)
// =============================================================================

#ifdef CANN_SDK_AVAILABLE

void a2a3_core_complete_task(PTORuntime* rt, int32_t task_id) {
    // Delegate completion and dependency propagation to orchestration layer.
    a2a3_orch_complete_task_threadsafe(rt, task_id);
}

#else /* No CANN SDK - stub implementation */

void a2a3_core_complete_task(PTORuntime* rt, int32_t task_id) {
    // Delegate completion and dependency propagation to orchestration layer.
    a2a3_orch_complete_task_threadsafe(rt, task_id);
}

#endif /* CANN_SDK_AVAILABLE */

// =============================================================================
// Worker Thread Functions (Hardware Implementation)
// =============================================================================

void* a2a3_vector_worker_func(void* arg) {
    A2A3WorkerContext* ctx = (A2A3WorkerContext*)arg;
    PTORuntime* rt = ctx->rt;
    int worker_id = ctx->worker_id;
    
    printf("[A2A3 Worker] Vector worker %d started\n", worker_id);
    fflush(stdout);
    
#ifdef CANN_SDK_AVAILABLE
    // Set device context for this worker thread
    printf("[A2A3 Worker] Vector worker %d: Setting device...\n", worker_id);
    fflush(stdout);
    aclrtSetDevice(0);
    printf("[A2A3 Worker] Vector worker %d: Device set OK\n", worker_id);
    fflush(stdout);
#endif
    
    while (!rt->shutdown_requested) {
        int32_t task_id = a2a3_orch_get_vector_task_blocking(rt);
        
        if (task_id < 0) {
            if (rt->shutdown_requested) break;
            if (rt->execution_started && 
                rt->total_tasks_completed >= rt->total_tasks_scheduled) break;
            continue;
        }
        
        a2a3_core_execute_task(rt, task_id, worker_id);
        a2a3_core_complete_task(rt, task_id);
    }
    
    DEBUG_PRINT("[A2A3 Core HW] Vector worker %d exiting\n", worker_id);
    free(ctx);
    return NULL;
}

void* a2a3_cube_worker_func(void* arg) {
    A2A3WorkerContext* ctx = (A2A3WorkerContext*)arg;
    PTORuntime* rt = ctx->rt;
    int worker_id = ctx->worker_id;
    
    printf("[A2A3 Worker] Cube worker %d started\n", worker_id);
    fflush(stdout);
    
#ifdef CANN_SDK_AVAILABLE
    // Set device context for this worker thread
    printf("[A2A3 Worker] Cube worker %d: Setting device...\n", worker_id);
    fflush(stdout);
    aclrtSetDevice(0);
    printf("[A2A3 Worker] Cube worker %d: Device set OK\n", worker_id);
    fflush(stdout);
#endif
    
    while (!rt->shutdown_requested) {
        int32_t task_id = a2a3_orch_get_cube_task_blocking(rt);
        
        if (task_id < 0) {
            if (rt->shutdown_requested) break;
            if (rt->execution_started && 
                rt->total_tasks_completed >= rt->total_tasks_scheduled) break;
            continue;
        }
        
        a2a3_core_execute_task(rt, task_id, worker_id);
        a2a3_core_complete_task(rt, task_id);
    }
    
    DEBUG_PRINT("[A2A3 Core HW] Cube worker %d exiting\n", worker_id);
    free(ctx);
    return NULL;
}

// =============================================================================
// Dependency Resolution Thread Functions
// =============================================================================

/**
 * Queue of completed task IDs waiting for dependency processing.
 * This separates task completion notification from actual dependency update.
 */
#define MAX_COMPLETION_QUEUE 1024

static int32_t g_completion_queue[MAX_COMPLETION_QUEUE];
static int g_completion_head = 0;
static int g_completion_tail = 0;
static pthread_mutex_t g_completion_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_completion_not_empty = PTHREAD_COND_INITIALIZER;

/**
 * Push a completed task ID to the completion queue.
 * Called by workers after executing a task.
 */
static void push_completion(int32_t task_id) {
    pthread_mutex_lock(&g_completion_mutex);
    
    int next_tail = (g_completion_tail + 1) % MAX_COMPLETION_QUEUE;
    if (next_tail == g_completion_head) {
        // Queue is full - this shouldn't happen with proper sizing
        fprintf(stderr, "[A2A3 DepResolver] WARNING: Completion queue full!\n");
    } else {
        g_completion_queue[g_completion_tail] = task_id;
        g_completion_tail = next_tail;
    }
    
    pthread_cond_signal(&g_completion_not_empty);
    pthread_mutex_unlock(&g_completion_mutex);
}

/**
 * Pop a completed task ID from the completion queue.
 * Returns -1 if queue is empty (non-blocking).
 */
static int32_t pop_completion_nonblocking(void) {
    pthread_mutex_lock(&g_completion_mutex);
    
    int32_t task_id = -1;
    if (g_completion_head != g_completion_tail) {
        task_id = g_completion_queue[g_completion_head];
        g_completion_head = (g_completion_head + 1) % MAX_COMPLETION_QUEUE;
    }
    
    pthread_mutex_unlock(&g_completion_mutex);
    return task_id;
}

/**
 * Pop a completed task ID from the completion queue (blocking).
 * Returns -1 on shutdown.
 */
static int32_t pop_completion_blocking(PTORuntime* rt) {
    pthread_mutex_lock(&g_completion_mutex);
    
    while (g_completion_head == g_completion_tail && !rt->shutdown_requested) {
        // Check if all tasks are done
        if (rt->execution_started && 
            rt->total_tasks_completed >= rt->total_tasks_scheduled) {
            pthread_mutex_unlock(&g_completion_mutex);
            return -1;
        }
        
        // Wait with timeout to check shutdown periodically
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 10000000;  // 10ms timeout
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&g_completion_not_empty, &g_completion_mutex, &ts);
    }
    
    int32_t task_id = -1;
    if (g_completion_head != g_completion_tail) {
        task_id = g_completion_queue[g_completion_head];
        g_completion_head = (g_completion_head + 1) % MAX_COMPLETION_QUEUE;
    }
    
    pthread_mutex_unlock(&g_completion_mutex);
    return task_id;
}

void a2a3_process_task_completion(PTORuntime* rt, int32_t task_id) {
    // Delegate completion and dependency propagation to orchestration layer.
    a2a3_orch_complete_task_threadsafe(rt, task_id);

    // Wake up other dep resolvers on completion to allow clean shutdown.
    if (rt->execution_started && rt->total_tasks_completed >= rt->total_tasks_scheduled) {
        pthread_mutex_lock(&g_completion_mutex);
        pthread_cond_broadcast(&g_completion_not_empty);
        pthread_mutex_unlock(&g_completion_mutex);
    }
}

bool a2a3_has_pending_completions(PTORuntime* rt) {
    pthread_mutex_lock(&g_completion_mutex);
    bool has_pending = (g_completion_head != g_completion_tail);
    pthread_mutex_unlock(&g_completion_mutex);
    return has_pending;
}

void* a2a3_dep_resolver_func(void* arg) {
    A2A3DepResolverContext* ctx = (A2A3DepResolverContext*)arg;
    PTORuntime* rt = ctx->rt;
    int thread_id = ctx->thread_id;
    
    DEBUG_PRINT("[A2A3 DepResolver] Thread %d started\n", thread_id);
    
    while (!rt->shutdown_requested) {
        // Get a completed task from the queue
        int32_t task_id = pop_completion_blocking(rt);
        
        if (task_id < 0) {
            // Check termination conditions
            if (rt->shutdown_requested) break;
            if (rt->execution_started && 
                rt->total_tasks_completed >= rt->total_tasks_scheduled) break;
            continue;
        }
        
        // Process the completion (update dependencies, route ready tasks)
        a2a3_process_task_completion(rt, task_id);
    }
    
    DEBUG_PRINT("[A2A3 DepResolver] Thread %d exiting\n", thread_id);
    free(ctx);
    return NULL;
}

// =============================================================================
// Orchestration Thread Function
// =============================================================================

/**
 * Forward declaration of orchestration function type from so_loader
 */
typedef void (*A2A3OrchFuncPtr)(void* runtime, void* user_data);

void* a2a3_orch_thread_func(void* arg) {
    A2A3OrchContext* ctx = (A2A3OrchContext*)arg;
    PTORuntime* rt = ctx->rt;
    A2A3OrchFuncPtr orch_func = (A2A3OrchFuncPtr)ctx->orch_func;
    void* user_data = ctx->user_data;
    
    printf("[A2A3 Orch] Orchestration thread started\n");
    fflush(stdout);
    
    if (!orch_func) {
        fprintf(stderr, "[A2A3 Orch] ERROR: No orchestration function provided!\n");
        fflush(stderr);
        free(ctx);
        return NULL;
    }
    
    printf("[A2A3 Orch] Orchestration function ptr: %p\n", (void*)orch_func);
    printf("[A2A3 Orch] User data ptr: %p\n", user_data);
    fflush(stdout);
    
    // Execute the orchestration function
    // This will call into the runtime to build the task graph
    printf("[A2A3 Orch] Calling orchestration function...\n");
    fflush(stdout);
    orch_func(rt, user_data);
    printf("[A2A3 Orch] Orchestration function returned.\n");
    fflush(stdout);
    
    // Mark orchestration as complete
    DEBUG_PRINT("[A2A3 Orch] Orchestration function complete, scheduled %d tasks\n",
               rt->total_tasks_scheduled);
    
    // Signal that orchestration is done (no more tasks will be added)
    pthread_mutex_lock(&rt->task_mutex);
    rt->orchestration_complete = true;
    pthread_cond_broadcast(&rt->window_not_full);
    pthread_mutex_unlock(&rt->task_mutex);
    
    // Signal worker queues in case they're waiting
    pthread_mutex_lock(&rt->queue_mutex);
    pthread_cond_broadcast(&rt->vector_queue_not_empty);
    pthread_cond_broadcast(&rt->cube_queue_not_empty);
    pthread_mutex_unlock(&rt->queue_mutex);
    
    DEBUG_PRINT("[A2A3 Orch] Orchestration thread exiting\n");
    free(ctx);
    return NULL;
}

// =============================================================================
// Helper for Workers: Notify Completion to Dep Resolver
// =============================================================================

/**
 * Notify the dependency resolution system that a task has been executed.
 * This is called by workers instead of directly calling a2a3_core_complete_task
 * when using the dep resolver threads.
 */
void a2a3_notify_task_executed(int32_t task_id) {
    push_completion(task_id);
}
