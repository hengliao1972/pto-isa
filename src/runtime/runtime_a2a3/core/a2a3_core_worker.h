/**
 * PTO Runtime - Ascend A2/A3 Core Worker Interface
 * 
 * This header defines the worker functions that execute InCore functions
 * on Cube and Vector cores. These are compiled as part of the core layer.
 * 
 * The implementation differs between hardware (CANN SDK) and simulator.
 */

#ifndef A2A3_CORE_WORKER_H
#define A2A3_CORE_WORKER_H

#include "../../pto_runtime_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Worker Context
// =============================================================================

/**
 * Context passed to each worker thread.
 */
typedef struct {
    PTORuntime* rt;
    int worker_id;
    bool is_cube_worker;
} A2A3WorkerContext;

// =============================================================================
// Task Execution (called by workers)
// =============================================================================

/**
 * Execute an InCore function task.
 * This is called by workers after dequeuing a task.
 * 
 * @param rt        Runtime context
 * @param task_id   Task to execute
 * @param worker_id Worker executing the task
 */
void a2a3_core_execute_task(PTORuntime* rt, int32_t task_id, int32_t worker_id);

// =============================================================================
// Task Completion (called by workers after execution)
// =============================================================================

/**
 * Mark a task as complete and propagate to dependents (thread-safe).
 * This is called by workers after executing an InCore function.
 * 
 * Implementation differs between hardware and simulator:
 * - Hardware: Uses CANN SDK synchronization primitives
 * - Simulator: Uses pthread mutexes with cycle-accurate tracking
 * 
 * @param rt        Runtime context
 * @param task_id   Completed task ID
 */
void a2a3_core_complete_task(PTORuntime* rt, int32_t task_id);

// =============================================================================
// Worker Thread Functions
// =============================================================================

/**
 * Vector core worker thread main function.
 * Loops: get task from vector queue -> execute -> complete -> repeat
 * 
 * @param arg   Pointer to A2A3WorkerContext
 * @return      NULL (thread exit)
 */
void* a2a3_vector_worker_func(void* arg);

/**
 * Cube core worker thread main function.
 * Loops: get task from cube queue -> execute -> complete -> repeat
 * 
 * @param arg   Pointer to A2A3WorkerContext
 * @return      NULL (thread exit)
 */
void* a2a3_cube_worker_func(void* arg);

// =============================================================================
// Dependency Resolution Thread (AICPU)
// =============================================================================

/**
 * Context for dependency resolution thread.
 */
typedef struct {
    PTORuntime* rt;
    int thread_id;
} A2A3DepResolverContext;

/**
 * Dependency resolution thread main function.
 * 
 * These threads run on AICPU and handle:
 * - Monitoring completed tasks from AICore workers
 * - Updating fanin counts of dependent tasks
 * - Routing newly ready tasks to appropriate queues (AIV/AIC)
 * 
 * Based on ref_runtime/src/runtime/aicpu/graphexecutor.cpp logic.
 * 
 * @param arg   Pointer to A2A3DepResolverContext
 * @return      NULL (thread exit)
 */
void* a2a3_dep_resolver_func(void* arg);

/**
 * Process completion of a task and update dependencies.
 * 
 * This function:
 * 1. Marks the task as complete
 * 2. Increments successors' fanin_refcount until it reaches fanin_count
 * 3. Routes newly ready tasks to appropriate queues
 * 
 * Thread-safe: can be called from multiple dep resolver threads.
 * 
 * @param rt        Runtime context
 * @param task_id   Completed task ID
 */
void a2a3_process_task_completion(PTORuntime* rt, int32_t task_id);

/**
 * Check if there are pending completions to process.
 * 
 * @param rt  Runtime context
 * @return    true if there are completions pending
 */
bool a2a3_has_pending_completions(PTORuntime* rt);

// =============================================================================
// Orchestration Thread (AICPU)
// =============================================================================

/**
 * Context for orchestration thread.
 */
typedef struct {
    PTORuntime* rt;
    void* orch_func;    // A2A3OrchFunc pointer
    void* user_data;
} A2A3OrchContext;

/**
 * Orchestration thread main function.
 * 
 * Executes the orchestration function which builds the task graph.
 * Runs on a dedicated AICPU thread.
 * 
 * @param arg   Pointer to A2A3OrchContext
 * @return      NULL (thread exit)
 */
void* a2a3_orch_thread_func(void* arg);

// =============================================================================
// Worker-to-DepResolver Communication
// =============================================================================

/**
 * Notify the dependency resolution system that a task has been executed.
 * 
 * When using dependency resolver threads, workers call this instead of
 * a2a3_core_complete_task() to hand off completion processing to the
 * dep resolver threads.
 * 
 * @param task_id  Task that has been executed
 */
void a2a3_notify_task_executed(int32_t task_id);

#ifdef __cplusplus
}
#endif

#endif /* A2A3_CORE_WORKER_H */
