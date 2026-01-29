/**
 * PTO Runtime - Ascend A2/A3 Simulator Core Worker Implementation
 * 
 * This file implements the worker functions for the simulator platform.
 * Uses pthread synchronization and cycle-accurate simulation.
 */

#include "../../runtime_a2a3/core/a2a3_core_worker.h"
#include "../../runtime_a2a3/orchestration/a2a3_orchestration.h"
#include <stdlib.h>
#include <stdio.h>

// =============================================================================
// Task Execution (Simulator Implementation)
// =============================================================================

void a2a3_core_execute_task(PTORuntime* rt, int32_t task_id, int32_t worker_id) {
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    
    DEBUG_PRINT("[A2A3 Core] Worker %d executing task %d: %s\n", 
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
    
    // Simulation mode: use cycle cost function for timing
    if (rt->simulation_mode && task->cycle_func) {
        int64_t cycle_cost = task->cycle_func(args, task->num_args);
        
        int64_t worker_current = pto_trace_get_cycle(worker_id);
        int64_t actual_start = (worker_current > task->earliest_start_cycle) ? 
            worker_current : task->earliest_start_cycle;
        int64_t actual_end = actual_start + cycle_cost;
        
        task->end_cycle = actual_end;
        
        pto_trace_record_with_time(worker_id, task->func_name, actual_start, actual_end);
        DEBUG_PRINT("[A2A3 Core] Task %d simulated: %lld cycles\n", 
                    task_id, (long long)cycle_cost);
    }
    // Also execute actual function if provided (for correctness verification)
    if (task->func_ptr) {
        PTOInCoreFunc func = (PTOInCoreFunc)task->func_ptr;
        func(args, task->num_args);
    }
}

// =============================================================================
// Task Completion (Simulator Implementation - pthread based)
// =============================================================================

void a2a3_core_complete_task(PTORuntime* rt, int32_t task_id) {
    // Delegate completion and dependency propagation to orchestration layer.
    a2a3_orch_complete_task_threadsafe(rt, task_id);
}

// =============================================================================
// Worker Thread Functions (Simulator Implementation)
// =============================================================================

void* a2a3_vector_worker_func(void* arg) {
    A2A3WorkerContext* ctx = (A2A3WorkerContext*)arg;
    PTORuntime* rt = ctx->rt;
    int worker_id = ctx->worker_id;
    
    DEBUG_PRINT("[A2A3 Core] Vector worker %d started\n", worker_id);
    
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
    
    DEBUG_PRINT("[A2A3 Core] Vector worker %d exiting\n", worker_id);
    free(ctx);
    return NULL;
}

void* a2a3_cube_worker_func(void* arg) {
    A2A3WorkerContext* ctx = (A2A3WorkerContext*)arg;
    PTORuntime* rt = ctx->rt;
    int worker_id = ctx->worker_id;
    
    DEBUG_PRINT("[A2A3 Core] Cube worker %d started\n", worker_id);
    
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
    
    DEBUG_PRINT("[A2A3 Core] Cube worker %d exiting\n", worker_id);
    free(ctx);
    return NULL;
}
